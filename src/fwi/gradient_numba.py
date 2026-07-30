# ==============================================================================
# src/fwi/gradient_numba.py
#
# Checkpointed discrete model gradient for the production Numba elastic-DAS
# solver.
#
# Parameterisation
# ----------------
# Forward propagation uses vp, vs, rho.  For the first FWI implementation,
# rho is fixed and the model gradient is returned for:
#
#     lambda = rho (vp^2 - 2 vs^2)
#     mu     = rho vs^2
#
# and then transformed exactly to:
#
#     g_vp = 2 rho vp g_lambda
#     g_vs = -4 rho vs g_lambda + 2 rho vs g_mu
#
# plus log-velocity gradients:
#
#     g_log_vp = vp * g_vp
#     g_log_vs = vs * g_vs
#
# Exactness details
# -----------------
# - The gradient is for the DISCRETE forward operator, not a continuum formula.
# - The shear modulus used by sxz is the harmonic staggered average mu_xz.
#   Its transpose is explicitly applied back to cell-centred mu.
# - The multiplicative sponge is included through the exact reverse ordering.
# - free_surface=True is intentionally unsupported until its exact transpose is
#   implemented.
# - TS-SFD is intentionally unsupported here because its FD coefficients depend
#   on the model-dependent representative Courant number in the current solver.
#
# Memory/performance
# ------------------
# Full wavefield storage would be large.  Instead we:
#   1) save the five forward state fields every checkpoint_interval steps;
#   2) during reverse time, replay one block from its checkpoint;
#   3) keep only vx/vz after the velocity update for that block;
#   4) correlate with the adjoint stress fields and discard the block.
#
# This is exact recomputation (up to floating-point reproducibility) and keeps
# memory bounded while retaining the production Numba kernels.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numba import njit, prange

from src.das import DASResult, compute_axial_strain_rate, das_adjoint
from src.sampling import ReceiverSampling2D
from src.solver_numpy import (
    fd_coefficients,
    make_sponge_mask,
    prepare_staggered_materials,
)
from src.solver_numba_fused import (
    update_velocity_fused_numba,
    update_stress_fused_numba,
    inject_stress_source_numba,
    apply_sponge_numba,
    sample_receivers_numba_bilinear,
)
from src.source_injection import StressSourceInjection

from src.fwi.adjoint_numba import (
    scatter_receivers_adjoint_numba,
    stress_update_transpose_add_numba,
    velocity_update_transpose_add_numba,
)
from src.fwi.two_layer import (
    ElasticMedium2D,
    PreparedMomentTensorSource,
    SolverSpec,
)


# ==============================================================================
# 1. RESULT / CHECKPOINT CONTAINERS
# ==============================================================================

@dataclass(frozen=True)
class CheckpointedForwardResult:
    """Forward data plus in-memory block-start checkpoints."""

    receiver_vx: np.ndarray
    receiver_vz: np.ndarray
    das: DASResult

    block_starts: np.ndarray
    checkpoint_vx: np.ndarray
    checkpoint_vz: np.ndarray
    checkpoint_sxx: np.ndarray
    checkpoint_szz: np.ndarray
    checkpoint_sxz: np.ndarray

    forward_elapsed_s: float

    @property
    def n_blocks(self) -> int:
        return int(self.block_starts.size)


@dataclass(frozen=True)
class ElasticDASGradientResult:
    """Objective, residual, and exact discrete model gradient."""

    objective: float
    residual: np.ndarray

    g_lambda: np.ndarray
    g_mu: np.ndarray

    g_vp: np.ndarray
    g_vs: np.ndarray

    g_log_vp: np.ndarray
    g_log_vs: np.ndarray

    forward_elapsed_s: float
    reverse_elapsed_s: float
    replay_elapsed_s: float
    checkpoint_memory_mb: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.objective):
            raise ValueError("objective must be finite.")

        reference_shape = None
        for name in (
            "g_lambda",
            "g_mu",
            "g_vp",
            "g_vs",
            "g_log_vp",
            "g_log_vs",
        ):
            arr = np.array(
                getattr(self, name),
                dtype=np.float64,
                copy=True,
                order="C",
            )

            if arr.ndim != 2:
                raise ValueError(
                    f"{name} must be 2D; got {arr.shape}."
                )

            if reference_shape is None:
                reference_shape = arr.shape
            elif arr.shape != reference_shape:
                raise ValueError(
                    "All model gradients must have identical shapes."
                )

            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"{name} contains NaN or Inf."
                )

            arr.flags.writeable = False
            object.__setattr__(
                self,
                name,
                arr,
            )

        residual = np.array(
            self.residual,
            dtype=np.float64,
            copy=True,
            order="C",
        )
        if residual.ndim != 2:
            raise ValueError(
                f"residual must be 2D; got {residual.shape}."
            )
        if not np.all(np.isfinite(residual)):
            raise ValueError(
                "residual contains NaN or Inf."
            )
        residual.flags.writeable = False
        object.__setattr__(
            self,
            "residual",
            residual,
        )


# ==============================================================================
# 2. DISCRETE MODEL-GRADIENT KERNELS
# ==============================================================================

@njit(parallel=True, fastmath=True, cache=True)
def accumulate_lambda_mu_xz_gradient_numba(
    vx: np.ndarray,
    vz: np.ndarray,
    adj_sxx: np.ndarray,
    adj_szz: np.ndarray,
    adj_sxz: np.ndarray,
    g_lambda: np.ndarray,
    g_mu_normal: np.ndarray,
    g_mu_xz: np.ndarray,
    dx: float,
    dz: float,
    dt: float,
    a: np.ndarray,
) -> None:
    """
    Accumulate the exact derivative of one discrete stress update.

    Forward stress update:
        sxx += dt * [(lambda+2mu) dvx_dx + lambda dvz_dz]
        szz += dt * [lambda dvx_dx + (lambda+2mu) dvz_dz]
        sxz += dt * mu_xz * (dvx_dz + dvz_dx)

    Therefore:
        dJ/dlambda += dt (qxx+qzz) (dvx_dx+dvz_dz)

        direct normal-grid dJ/dmu +=
            2 dt [qxx dvx_dx + qzz dvz_dz]

        dJ/dmu_xz +=
            dt qxz (dvx_dz + dvz_dx)

    The mu_xz gradient is mapped back through the harmonic averaging operator
    after all time steps have been accumulated.
    """
    nx, nz = vx.shape
    M = a.size

    for i in prange(M, nx - M):
        for j in range(M, nz - M):
            dvx_dx = 0.0
            dvz_dz = 0.0
            dvx_dz = 0.0
            dvz_dx = 0.0

            for m in range(1, M + 1):
                am = a[m - 1]

                dvx_dx += am * (
                    vx[i + m - 1, j]
                    - vx[i - m, j]
                )

                dvz_dz += am * (
                    vz[i, j + m - 1]
                    - vz[i, j - m]
                )

                dvx_dz += am * (
                    vx[i, j + m]
                    - vx[i, j - m + 1]
                )

                dvz_dx += am * (
                    vz[i + m, j]
                    - vz[i - m + 1, j]
                )

            dvx_dx /= dx
            dvz_dz /= dz
            dvx_dz /= dz
            dvz_dx /= dx

            qxx = adj_sxx[i, j]
            qzz = adj_szz[i, j]
            qxz = adj_sxz[i, j]

            g_lambda[i, j] += (
                dt
                * (qxx + qzz)
                * (dvx_dx + dvz_dz)
            )

            g_mu_normal[i, j] += (
                2.0
                * dt
                * (
                    qxx * dvx_dx
                    + qzz * dvz_dz
                )
            )

            g_mu_xz[i, j] += (
                dt
                * qxz
                * (dvx_dz + dvz_dx)
            )


@njit(parallel=True, fastmath=True, cache=True)
def harmonic_mu_transpose_numba(
    mu: np.ndarray,
    mu_xz: np.ndarray,
    g_mu_xz: np.ndarray,
    g_mu: np.ndarray,
) -> None:
    """
    Add the transpose of prepare_staggered_materials()' harmonic mu mapping.

    For i<nx-1 and j<nz-1:

        H = 4 / (1/m00 + 1/m10 + 1/m01 + 1/m11)

    and:

        dH/dmk = H^2 / (4 mk^2)

    The last x row or last z column of mu_xz is copied directly from mu in the
    forward material preparation, so its transpose is an identity contribution.

    Gather-form implementation: each base mu[i,j] reads the at-most-four
    neighbouring harmonic cells that depend on it.  No atomics are needed.
    """
    nx, nz = mu.shape

    for i in prange(nx):
        for j in range(nz):
            value = 0.0
            mu_ij = mu[i, j]
            inv_factor = 1.0 / (
                4.0
                * mu_ij
                * mu_ij
            )

            # Harmonic cell (i, j)
            if i < nx - 1 and j < nz - 1:
                H = mu_xz[i, j]
                value += (
                    g_mu_xz[i, j]
                    * H
                    * H
                    * inv_factor
                )

            # Harmonic cell (i-1, j)
            if i > 0 and j < nz - 1:
                H = mu_xz[i - 1, j]
                value += (
                    g_mu_xz[i - 1, j]
                    * H
                    * H
                    * inv_factor
                )

            # Harmonic cell (i, j-1)
            if i < nx - 1 and j > 0:
                H = mu_xz[i, j - 1]
                value += (
                    g_mu_xz[i, j - 1]
                    * H
                    * H
                    * inv_factor
                )

            # Harmonic cell (i-1, j-1)
            if i > 0 and j > 0:
                H = mu_xz[i - 1, j - 1]
                value += (
                    g_mu_xz[i - 1, j - 1]
                    * H
                    * H
                    * inv_factor
                )

            # mu_xz is copied directly from mu on the final x row and/or
            # final z column.
            if i == nx - 1 or j == nz - 1:
                value += g_mu_xz[i, j]

            g_mu[i, j] += value


# ==============================================================================
# 3. CHECKPOINTED FORWARD / REPLAY ENGINE
# ==============================================================================

class NumbaElasticDASGradient:
    """
    Checkpointed exact discrete gradient engine for fixed-density vp/vs FWI.

    This object reuses the forward engine's grid, receivers, sampling metadata,
    solver controls, and finite-gauge configuration.
    """

    def __init__(
        self,
        *,
        grid,
        receivers,
        receiver_sampling: ReceiverSampling2D,
        gauge_length_m: float,
        channel_spacing_m: float,
        solver: SolverSpec,
        checkpoint_interval: int = 96,
    ) -> None:
        self.grid = grid
        self.receivers = receivers
        self.receiver_sampling = receiver_sampling
        self.gauge_length_m = float(
            gauge_length_m
        )
        self.channel_spacing_m = float(
            channel_spacing_m
        )
        self.solver = solver
        self.checkpoint_interval = int(
            checkpoint_interval
        )

        if self.gauge_length_m <= 0.0:
            raise ValueError(
                "gauge_length_m must be positive."
            )

        if self.channel_spacing_m <= 0.0:
            raise ValueError(
                "channel_spacing_m must be positive."
            )

        if self.checkpoint_interval < 1:
            raise ValueError(
                "checkpoint_interval must be >= 1."
            )

        if solver.free_surface:
            raise NotImplementedError(
                "Gradient engine currently supports free_surface=False only."
            )

        if solver.use_ts_sfd:
            raise NotImplementedError(
                "The first exact model gradient requires use_ts_sfd=False. "
                "Current TS-SFD coefficients depend on a model-dependent "
                "representative Courant number, whose derivative would also "
                "have to be included."
            )

    @classmethod
    def from_forward_engine(
        cls,
        forward_engine,
        *,
        channel_spacing_m: float,
        checkpoint_interval: int = 96,
    ) -> "NumbaElasticDASGradient":
        return cls(
            grid=forward_engine.grid,
            receivers=forward_engine.receivers,
            receiver_sampling=forward_engine.receiver_sampling,
            gauge_length_m=forward_engine.gauge_length_m,
            channel_spacing_m=channel_spacing_m,
            solver=forward_engine.solver,
            checkpoint_interval=checkpoint_interval,
        )

    def _prepare_model_coefficients(
        self,
        medium: ElasticMedium2D,
    ):
        if medium.grid is not self.grid:
            raise ValueError(
                "medium.grid must be the same Grid2D instance used to build "
                "the gradient engine."
            )

        rho = np.asarray(
            medium.rho,
            dtype=np.float64,
        )
        vp = np.asarray(
            medium.vp,
            dtype=np.float64,
        )
        vs = np.asarray(
            medium.vs,
            dtype=np.float64,
        )

        mu = np.ascontiguousarray(
            rho * vs**2
        )
        lam = np.ascontiguousarray(
            rho * (
                vp**2
                - 2.0 * vs**2
            )
        )
        l2m = np.ascontiguousarray(
            lam
            + 2.0 * mu
        )

        bx, bz, mu_xz = prepare_staggered_materials(
            rho,
            mu,
        )

        a = fd_coefficients(
            self.solver.half_order,
            use_ts_sfd=False,
        )

        sponge = make_sponge_mask(
            int(self.grid.nx),
            int(self.grid.nz),
            int(self.solver.n_boundary),
            float(self.solver.gamma_s),
            float(self.grid.dt),
            free_surface=False,
        )

        return (
            lam,
            mu,
            l2m,
            np.ascontiguousarray(mu_xz),
            np.ascontiguousarray(bx),
            np.ascontiguousarray(bz),
            np.ascontiguousarray(a),
            np.ascontiguousarray(sponge),
        )

    def _allocate_checkpoints(
        self,
    ):
        nt = int(
            self.grid.nt
        )
        K = self.checkpoint_interval

        block_starts = np.arange(
            0,
            nt,
            K,
            dtype=np.int64,
        )

        shape = (
            block_starts.size,
            int(self.grid.nx),
            int(self.grid.nz),
        )

        arrays = tuple(
            np.empty(
                shape,
                dtype=np.float64,
            )
            for _ in range(5)
        )

        return (
            block_starts,
            *arrays,
        )

    def forward_with_checkpoints(
        self,
        *,
        medium: ElasticMedium2D,
        source: PreparedMomentTensorSource,
    ) -> CheckpointedForwardResult:
        """
        Run one production-equivalent forward pass while saving block starts.

        The generated receiver traces should match NumbaElasticDASForward.run()
        to roundoff because the same update/sampling kernels are used.
        """
        (
            lam,
            mu,
            l2m,
            mu_xz,
            bx,
            bz,
            a,
            sponge,
        ) = self._prepare_model_coefficients(
            medium
        )

        (
            block_starts,
            cp_vx,
            cp_vz,
            cp_sxx,
            cp_szz,
            cp_sxz,
        ) = self._allocate_checkpoints()

        nx = int(self.grid.nx)
        nz = int(self.grid.nz)
        nt = int(self.grid.nt)

        vx = np.zeros(
            (nx, nz),
            dtype=np.float64,
        )
        vz = np.zeros(
            (nx, nz),
            dtype=np.float64,
        )
        sxx = np.zeros(
            (nx, nz),
            dtype=np.float64,
        )
        szz = np.zeros(
            (nx, nz),
            dtype=np.float64,
        )
        sxz = np.zeros(
            (nx, nz),
            dtype=np.float64,
        )

        nrec = int(
            self.receiver_sampling.nrec
        )

        rec_vx = np.zeros(
            (nrec, nt),
            dtype=np.float64,
        )
        rec_vz = np.zeros(
            (nrec, nt),
            dtype=np.float64,
        )

        rs = self.receiver_sampling
        inj: StressSourceInjection = (
            source.injection
        )

        src_scale = (
            float(self.grid.dt)
            / (
                float(self.grid.dx)
                * float(self.grid.dz)
            )
        )

        K = self.checkpoint_interval
        iblock = 0

        start_clock = perf_counter()

        for it in range(nt):
            if it % K == 0:
                cp_vx[iblock] = vx
                cp_vz[iblock] = vz
                cp_sxx[iblock] = sxx
                cp_szz[iblock] = szz
                cp_sxz[iblock] = sxz
                iblock += 1

            update_velocity_fused_numba(
                vx,
                vz,
                sxx,
                szz,
                sxz,
                bx,
                bz,
                float(self.grid.dx),
                float(self.grid.dz),
                float(self.grid.dt),
                a,
            )

            update_stress_fused_numba(
                vx,
                vz,
                sxx,
                szz,
                sxz,
                lam,
                l2m,
                mu_xz,
                float(self.grid.dx),
                float(self.grid.dz),
                float(self.grid.dt),
                a,
            )

            inject_stress_source_numba(
                sxx,
                szz,
                sxz,
                inj.normal_ix,
                inj.normal_iz,
                inj.normal_w,
                inj.shear_ix,
                inj.shear_iz,
                inj.shear_w,
                source.stf_xx[it] * src_scale,
                source.stf_zz[it] * src_scale,
                source.stf_xz[it] * src_scale,
            )

            apply_sponge_numba(
                vx,
                vz,
                sxx,
                szz,
                sxz,
                sponge,
            )

            sample_receivers_numba_bilinear(
                vx,
                vz,
                rs.vx.ix,
                rs.vx.iz,
                rs.vx.w00,
                rs.vx.w10,
                rs.vx.w01,
                rs.vx.w11,
                rs.vz.ix,
                rs.vz.iz,
                rs.vz.w00,
                rs.vz.w10,
                rs.vz.w01,
                rs.vz.w11,
                rec_vx,
                rec_vz,
                it,
            )

        elapsed = (
            perf_counter()
            - start_clock
        )

        das = compute_axial_strain_rate(
            vx=rec_vx,
            vz=rec_vz,
            receivers=self.receivers,
            gauge_length_m=self.gauge_length_m,
        )

        return CheckpointedForwardResult(
            receiver_vx=rec_vx,
            receiver_vz=rec_vz,
            das=das,
            block_starts=block_starts,
            checkpoint_vx=cp_vx,
            checkpoint_vz=cp_vz,
            checkpoint_sxx=cp_sxx,
            checkpoint_szz=cp_szz,
            checkpoint_sxz=cp_sxz,
            forward_elapsed_s=float(elapsed),
        )

    @staticmethod
    def _checkpoint_memory_mb(
        forward: CheckpointedForwardResult,
    ) -> float:
        total_bytes = 0

        for arr in (
            forward.checkpoint_vx,
            forward.checkpoint_vz,
            forward.checkpoint_sxx,
            forward.checkpoint_szz,
            forward.checkpoint_sxz,
        ):
            total_bytes += arr.nbytes

        return float(
            total_bytes
            / 1024.0**2
        )

    def _replay_block_velocity_history(
        self,
        *,
        medium: ElasticMedium2D,
        source: PreparedMomentTensorSource,
        forward: CheckpointedForwardResult,
        iblock: int,
        lam: np.ndarray,
        l2m: np.ndarray,
        mu_xz: np.ndarray,
        bx: np.ndarray,
        bz: np.ndarray,
        a: np.ndarray,
        sponge: np.ndarray,
        hist_vx: np.ndarray,
        hist_vz: np.ndarray,
    ) -> int:
        """
        Restore one checkpoint and exactly replay its forward block.

        hist_vx/hist_vz store the velocity state immediately after the velocity
        update and BEFORE the stress update/sponge.  That is precisely the state
        entering the model-dependent stress update.
        """
        block_start = int(
            forward.block_starts[iblock]
        )

        block_stop = min(
            block_start
            + self.checkpoint_interval,
            int(self.grid.nt),
        )

        n_local = (
            block_stop
            - block_start
        )

        vx = np.array(
            forward.checkpoint_vx[iblock],
            copy=True,
            order="C",
        )
        vz = np.array(
            forward.checkpoint_vz[iblock],
            copy=True,
            order="C",
        )
        sxx = np.array(
            forward.checkpoint_sxx[iblock],
            copy=True,
            order="C",
        )
        szz = np.array(
            forward.checkpoint_szz[iblock],
            copy=True,
            order="C",
        )
        sxz = np.array(
            forward.checkpoint_sxz[iblock],
            copy=True,
            order="C",
        )

        inj = source.injection
        src_scale = (
            float(self.grid.dt)
            / (
                float(self.grid.dx)
                * float(self.grid.dz)
            )
        )

        for local_it in range(
            n_local
        ):
            it = (
                block_start
                + local_it
            )

            update_velocity_fused_numba(
                vx,
                vz,
                sxx,
                szz,
                sxz,
                bx,
                bz,
                float(self.grid.dx),
                float(self.grid.dz),
                float(self.grid.dt),
                a,
            )

            hist_vx[local_it] = vx
            hist_vz[local_it] = vz

            update_stress_fused_numba(
                vx,
                vz,
                sxx,
                szz,
                sxz,
                lam,
                l2m,
                mu_xz,
                float(self.grid.dx),
                float(self.grid.dz),
                float(self.grid.dt),
                a,
            )

            inject_stress_source_numba(
                sxx,
                szz,
                sxz,
                inj.normal_ix,
                inj.normal_iz,
                inj.normal_w,
                inj.shear_ix,
                inj.shear_iz,
                inj.shear_w,
                source.stf_xx[it] * src_scale,
                source.stf_zz[it] * src_scale,
                source.stf_xz[it] * src_scale,
            )

            apply_sponge_numba(
                vx,
                vz,
                sxx,
                szz,
                sxz,
                sponge,
            )

        return n_local

    def objective_only(
        self,
        *,
        medium: ElasticMedium2D,
        source: PreparedMomentTensorSource,
        observed_das: np.ndarray,
    ) -> float:
        """
        Evaluate the same weighted L2 objective without checkpoint allocation.

        J = 0.5 * dt * dCh * ||d_syn - d_obs||^2
        """
        # Import locally to avoid forcing the higher-level forward module into
        # the gradient hot path.
        from src.fwi.two_layer import NumbaElasticDASForward

        engine = NumbaElasticDASForward(
            grid=self.grid,
            receivers=self.receivers,
            gauge_length_m=self.gauge_length_m,
            solver=self.solver,
        )

        synthetic = engine.run(
            medium,
            source,
        ).das.data

        observed = np.asarray(
            observed_das,
            dtype=np.float64,
        )

        if synthetic.shape != observed.shape:
            raise ValueError(
                f"synthetic/observed DAS shapes differ: "
                f"{synthetic.shape} vs {observed.shape}."
            )

        residual = (
            synthetic
            - observed
        )

        weight = (
            float(self.grid.dt)
            * self.channel_spacing_m
        )

        return float(
            0.5
            * weight
            * np.sum(
                residual**2,
                dtype=np.float64,
            )
        )

    def compute_gradient(
        self,
        *,
        medium: ElasticMedium2D,
        source: PreparedMomentTensorSource,
        observed_das: np.ndarray,
    ) -> ElasticDASGradientResult:
        """
        Evaluate weighted L2 DAS objective and its exact fixed-density vp/vs
        discrete gradient.
        """
        (
            lam,
            mu,
            l2m,
            mu_xz,
            bx,
            bz,
            a,
            sponge,
        ) = self._prepare_model_coefficients(
            medium
        )

        forward = self.forward_with_checkpoints(
            medium=medium,
            source=source,
        )

        observed = np.asarray(
            observed_das,
            dtype=np.float64,
        )

        if observed.shape != forward.das.data.shape:
            raise ValueError(
                "observed_das shape must match synthetic DAS: "
                f"{observed.shape} != {forward.das.data.shape}."
            )

        if not np.all(np.isfinite(observed)):
            raise ValueError(
                "observed_das contains NaN or Inf."
            )

        residual = (
            forward.das.data
            - observed
        )

        quadrature_weight = (
            float(self.grid.dt)
            * self.channel_spacing_m
        )

        objective = float(
            0.5
            * quadrature_weight
            * np.sum(
                residual**2,
                dtype=np.float64,
            )
        )

        q_das = (
            quadrature_weight
            * residual
        )

        receiver_qx, receiver_qz = das_adjoint(
            q_das,
            self.receivers,
            forward.das,
        )

        nx = int(self.grid.nx)
        nz = int(self.grid.nz)

        adj_vx = np.zeros(
            (nx, nz),
            dtype=np.float64,
        )
        adj_vz = np.zeros(
            (nx, nz),
            dtype=np.float64,
        )
        adj_sxx = np.zeros(
            (nx, nz),
            dtype=np.float64,
        )
        adj_szz = np.zeros(
            (nx, nz),
            dtype=np.float64,
        )
        adj_sxz = np.zeros(
            (nx, nz),
            dtype=np.float64,
        )

        g_lambda = np.zeros(
            (nx, nz),
            dtype=np.float64,
        )
        g_mu_normal = np.zeros(
            (nx, nz),
            dtype=np.float64,
        )
        g_mu_xz = np.zeros(
            (nx, nz),
            dtype=np.float64,
        )

        K = self.checkpoint_interval

        hist_vx = np.empty(
            (K, nx, nz),
            dtype=np.float64,
        )
        hist_vz = np.empty(
            (K, nx, nz),
            dtype=np.float64,
        )

        rs = self.receiver_sampling

        reverse_elapsed = 0.0
        replay_elapsed = 0.0

        for iblock in range(
            forward.n_blocks - 1,
            -1,
            -1,
        ):
            replay_start = perf_counter()

            n_local = (
                self._replay_block_velocity_history(
                    medium=medium,
                    source=source,
                    forward=forward,
                    iblock=iblock,
                    lam=lam,
                    l2m=l2m,
                    mu_xz=mu_xz,
                    bx=bx,
                    bz=bz,
                    a=a,
                    sponge=sponge,
                    hist_vx=hist_vx,
                    hist_vz=hist_vz,
                )
            )

            replay_elapsed += (
                perf_counter()
                - replay_start
            )

            block_start = int(
                forward.block_starts[iblock]
            )

            reverse_start = perf_counter()

            for local_it in range(
                n_local - 1,
                -1,
                -1,
            ):
                it = (
                    block_start
                    + local_it
                )

                # 5^T: receiver sample transpose
                scatter_receivers_adjoint_numba(
                    adj_vx,
                    adj_vz,
                    receiver_qx[:, it],
                    receiver_qz[:, it],
                    rs.vx.ix,
                    rs.vx.iz,
                    rs.vx.w00,
                    rs.vx.w10,
                    rs.vx.w01,
                    rs.vx.w11,
                    rs.vz.ix,
                    rs.vz.iz,
                    rs.vz.w00,
                    rs.vz.w10,
                    rs.vz.w01,
                    rs.vz.w11,
                )

                # 4^T: sponge
                apply_sponge_numba(
                    adj_vx,
                    adj_vz,
                    adj_sxx,
                    adj_szz,
                    adj_sxz,
                    sponge,
                )

                # 3^T: source-add state transpose is the identity.  Source
                # parameter gradients are not needed for this fixed-source
                # vp/vs inversion, so no source gather is required here.

                # Model gradient belongs here: adjoint stress is now the
                # adjoint of sigma after the stress update and before sponge.
                accumulate_lambda_mu_xz_gradient_numba(
                    hist_vx[local_it],
                    hist_vz[local_it],
                    adj_sxx,
                    adj_szz,
                    adj_sxz,
                    g_lambda,
                    g_mu_normal,
                    g_mu_xz,
                    float(self.grid.dx),
                    float(self.grid.dz),
                    float(self.grid.dt),
                    a,
                )

                # 2^T: stress update
                stress_update_transpose_add_numba(
                    adj_vx,
                    adj_vz,
                    adj_sxx,
                    adj_szz,
                    adj_sxz,
                    lam,
                    l2m,
                    mu_xz,
                    float(self.grid.dx),
                    float(self.grid.dz),
                    float(self.grid.dt),
                    a,
                )

                # 1^T: velocity update
                velocity_update_transpose_add_numba(
                    adj_sxx,
                    adj_szz,
                    adj_sxz,
                    adj_vx,
                    adj_vz,
                    bx,
                    bz,
                    float(self.grid.dx),
                    float(self.grid.dz),
                    float(self.grid.dt),
                    a,
                )

            reverse_elapsed += (
                perf_counter()
                - reverse_start
            )

        g_mu = np.array(
            g_mu_normal,
            copy=True,
            order="C",
        )

        harmonic_mu_transpose_numba(
            mu,
            mu_xz,
            g_mu_xz,
            g_mu,
        )

        rho = np.asarray(
            medium.rho,
            dtype=np.float64,
        )
        vp = np.asarray(
            medium.vp,
            dtype=np.float64,
        )
        vs = np.asarray(
            medium.vs,
            dtype=np.float64,
        )

        g_vp = (
            2.0
            * rho
            * vp
            * g_lambda
        )

        g_vs = (
            -4.0
            * rho
            * vs
            * g_lambda
            + 2.0
            * rho
            * vs
            * g_mu
        )

        g_log_vp = (
            vp
            * g_vp
        )

        g_log_vs = (
            vs
            * g_vs
        )

        block_memory_bytes = (
            hist_vx.nbytes
            + hist_vz.nbytes
        )

        checkpoint_memory_mb = (
            self._checkpoint_memory_mb(
                forward
            )
            + block_memory_bytes
            / 1024.0**2
        )

        return ElasticDASGradientResult(
            objective=objective,
            residual=residual,
            g_lambda=g_lambda,
            g_mu=g_mu,
            g_vp=g_vp,
            g_vs=g_vs,
            g_log_vp=g_log_vp,
            g_log_vs=g_log_vs,
            forward_elapsed_s=forward.forward_elapsed_s,
            reverse_elapsed_s=reverse_elapsed,
            replay_elapsed_s=replay_elapsed,
            checkpoint_memory_mb=checkpoint_memory_mb,
        )
