# ==============================================================================
# src/fwi/adjoint_numba.py
#
# Exact discrete adjoint of the production Numba velocity-stress solver for the
# current full-space/sponge configuration.
#
# Forward step (fixed model, fixed source geometry)
# -------------------------------------------------
#   [sigma^n, v^(n-1/2)]
#       1) velocity update       -> v^(n+1/2)
#       2) stress update         -> sigma^(n+1), before source
#       3) stress-source add     -> sigma^(n+1), before sponge
#       4) sponge               -> state carried to next step
#       5) receiver sampling     -> data at t_v[n]
#
# Reverse step
# ------------
#   5^T) receiver scatter
#   4^T) sponge (self-adjoint diagonal multiplication)
#   3^T) source gather
#   2^T) stress-update transpose
#   1^T) velocity-update transpose
#
# Scope
# -----
# - exact transpose of solver_numba_fused.py for free_surface=False;
# - C-SFD and TS-SFD coefficients are both supported;
# - multiplicative sponge is included exactly;
# - bilinear receiver sampling and bilinear/nearest source spreading supported;
# - DAS transpose is composed outside the hot loop using src.das.das_adjoint;
# - no model gradient yet: this file first closes the wave-equation adjoint.
#
# Performance design
# ------------------
# OOP/configuration stays in Python.  Hot spatial transposes are gather-form
# @njit(parallel=True) kernels, so there are no write races and no np.add.at
# inside the reverse-time grid loop.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numba import njit, prange

from src.das import DASResult, das_adjoint
from src.sampling import ReceiverSampling2D
from src.solver_numpy import (
    fd_coefficients,
    make_sponge_mask,
    prepare_staggered_materials,
)
from src.solver_numba_fused import apply_sponge_numba
from src.source_injection import StressSourceInjection

from src.fwi.two_layer import (
    ElasticMedium2D,
    PreparedMomentTensorSource,
    SolverSpec,
)


# ==============================================================================
# 1. RESULT CONTAINERS
# ==============================================================================

@dataclass(frozen=True)
class SourceAdjointResult:
    """
    Adjoint gradient with respect to the three solver input source sequences.

    These arrays are derivatives with respect to stf_xx, stf_zz, stf_xz
    BEFORE the solver multiplies them by dt/(dx*dz).
    """

    grad_stf_xx: np.ndarray
    grad_stf_zz: np.ndarray
    grad_stf_xz: np.ndarray
    elapsed_s: float

    def __post_init__(self) -> None:
        shapes = set()

        for name in (
            "grad_stf_xx",
            "grad_stf_zz",
            "grad_stf_xz",
        ):
            arr = np.array(
                getattr(self, name),
                dtype=np.float64,
                copy=True,
                order="C",
            )

            if arr.ndim != 1:
                raise ValueError(
                    f"{name} must be 1D; got {arr.shape}."
                )

            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"{name} contains NaN or Inf."
                )

            shapes.add(arr.shape)
            arr.flags.writeable = False
            object.__setattr__(
                self,
                name,
                arr,
            )

        if len(shapes) != 1:
            raise ValueError(
                "All source-adjoint arrays must have the same shape."
            )

        if not np.isfinite(self.elapsed_s) or self.elapsed_s < 0.0:
            raise ValueError(
                f"elapsed_s must be finite and non-negative; got {self.elapsed_s}."
            )

        object.__setattr__(
            self,
            "elapsed_s",
            float(self.elapsed_s),
        )


# ==============================================================================
# 2. RECEIVER SAMPLING TRANSPOSE — HOT-LOOP NUMBA VERSION
# ==============================================================================

@njit(fastmath=True, cache=True)
def scatter_receivers_adjoint_numba(
    adj_vx: np.ndarray,
    adj_vz: np.ndarray,
    qx: np.ndarray,
    qz: np.ndarray,
    ix_vx: np.ndarray,
    iz_vx: np.ndarray,
    w00_vx: np.ndarray,
    w10_vx: np.ndarray,
    w01_vx: np.ndarray,
    w11_vx: np.ndarray,
    ix_vz: np.ndarray,
    iz_vz: np.ndarray,
    w00_vz: np.ndarray,
    w10_vz: np.ndarray,
    w01_vz: np.ndarray,
    w11_vz: np.ndarray,
) -> None:
    """
    Add S^T(qx,qz) directly into the current adjoint velocity fields.

    Serial over receivers by design: only O(nrec) work and repeated receivers
    may share grid nodes, so a parallel scatter would introduce write races.
    """
    nrec = qx.size

    for k in range(nrec):
        value_x = qx[k]
        i = ix_vx[k]
        j = iz_vx[k]

        adj_vx[i,     j    ] += value_x * w00_vx[k]
        adj_vx[i + 1, j    ] += value_x * w10_vx[k]
        adj_vx[i,     j + 1] += value_x * w01_vx[k]
        adj_vx[i + 1, j + 1] += value_x * w11_vx[k]

        value_z = qz[k]
        i = ix_vz[k]
        j = iz_vz[k]

        adj_vz[i,     j    ] += value_z * w00_vz[k]
        adj_vz[i + 1, j    ] += value_z * w10_vz[k]
        adj_vz[i,     j + 1] += value_z * w01_vz[k]
        adj_vz[i + 1, j + 1] += value_z * w11_vz[k]


# ==============================================================================
# 3. TRANSPOSE OF STRESS UPDATE
# ==============================================================================

@njit(parallel=True, fastmath=True, cache=True)
def stress_update_transpose_add_numba(
    adj_vx: np.ndarray,
    adj_vz: np.ndarray,
    adj_sxx: np.ndarray,
    adj_szz: np.ndarray,
    adj_sxz: np.ndarray,
    lam: np.ndarray,
    l2m: np.ndarray,
    mu_xz: np.ndarray,
    dx: float,
    dz: float,
    dt: float,
    a: np.ndarray,
) -> None:
    """
    Add T^T * adj_sigma to adj_v.

    This is the exact algebraic transpose of update_stress_fused_numba().

    Forward:
        sigma_new = sigma_old + T v

    Reverse:
        adj_v       += T^T adj_sigma_new
        adj_sigma_old += adj_sigma_new

    The identity contribution to adj_sigma_old needs no operation because the
    adjoint stress arrays are carried in place through the reverse step.

    The implementation is gather-form: each (i,j) writes only adj_vx[i,j] and
    adj_vz[i,j].  This permits safe prange parallelism without atomics.
    """
    nx, nz = adj_vx.shape
    M = a.size

    for i in prange(nx):
        for j in range(nz):
            add_vx = 0.0
            add_vz = 0.0

            for m in range(1, M + 1):
                am = a[m - 1]

                # --------------------------------------------------------------
                # vx <- D_minus_x(vx) contribution to sxx/szz
                #
                # Forward target ti contains:
                #   +vx[ti+m-1] - vx[ti-m]
                #
                # For source i:
                #   positive target ti = i-m+1
                #   negative target ti = i+m
                # --------------------------------------------------------------
                ti = i - m + 1
                if M <= ti < nx - M and M <= j < nz - M:
                    q_normal = dt * (
                        l2m[ti, j] * adj_sxx[ti, j]
                        + lam[ti, j] * adj_szz[ti, j]
                    )
                    add_vx += am * q_normal / dx

                ti = i + m
                if M <= ti < nx - M and M <= j < nz - M:
                    q_normal = dt * (
                        l2m[ti, j] * adj_sxx[ti, j]
                        + lam[ti, j] * adj_szz[ti, j]
                    )
                    add_vx -= am * q_normal / dx

                # --------------------------------------------------------------
                # vx <- D_plus_z(vx) contribution to sxz
                #
                # Forward target tj:
                #   +vx[tj+m] - vx[tj-m+1]
                #
                # For source j:
                #   positive target tj = j-m
                #   negative target tj = j+m-1
                # --------------------------------------------------------------
                tj = j - m
                if M <= i < nx - M and M <= tj < nz - M:
                    q_shear = (
                        dt
                        * mu_xz[i, tj]
                        * adj_sxz[i, tj]
                    )
                    add_vx += am * q_shear / dz

                tj = j + m - 1
                if M <= i < nx - M and M <= tj < nz - M:
                    q_shear = (
                        dt
                        * mu_xz[i, tj]
                        * adj_sxz[i, tj]
                    )
                    add_vx -= am * q_shear / dz

                # --------------------------------------------------------------
                # vz <- D_minus_z(vz) contribution to sxx/szz
                # --------------------------------------------------------------
                tj = j - m + 1
                if M <= i < nx - M and M <= tj < nz - M:
                    q_normal = dt * (
                        lam[i, tj] * adj_sxx[i, tj]
                        + l2m[i, tj] * adj_szz[i, tj]
                    )
                    add_vz += am * q_normal / dz

                tj = j + m
                if M <= i < nx - M and M <= tj < nz - M:
                    q_normal = dt * (
                        lam[i, tj] * adj_sxx[i, tj]
                        + l2m[i, tj] * adj_szz[i, tj]
                    )
                    add_vz -= am * q_normal / dz

                # --------------------------------------------------------------
                # vz <- D_plus_x(vz) contribution to sxz
                # --------------------------------------------------------------
                ti = i - m
                if M <= ti < nx - M and M <= j < nz - M:
                    q_shear = (
                        dt
                        * mu_xz[ti, j]
                        * adj_sxz[ti, j]
                    )
                    add_vz += am * q_shear / dx

                ti = i + m - 1
                if M <= ti < nx - M and M <= j < nz - M:
                    q_shear = (
                        dt
                        * mu_xz[ti, j]
                        * adj_sxz[ti, j]
                    )
                    add_vz -= am * q_shear / dx

            adj_vx[i, j] += add_vx
            adj_vz[i, j] += add_vz


# ==============================================================================
# 4. TRANSPOSE OF VELOCITY UPDATE
# ==============================================================================

@njit(parallel=True, fastmath=True, cache=True)
def velocity_update_transpose_add_numba(
    adj_sxx: np.ndarray,
    adj_szz: np.ndarray,
    adj_sxz: np.ndarray,
    adj_vx: np.ndarray,
    adj_vz: np.ndarray,
    bx: np.ndarray,
    bz: np.ndarray,
    dx: float,
    dz: float,
    dt: float,
    a: np.ndarray,
) -> None:
    """
    Add V^T * adj_v to adj_sigma.

    Exact algebraic transpose of update_velocity_fused_numba().

    Forward:
        v_new = v_old + V sigma

    Reverse:
        adj_sigma_old += V^T adj_v_new
        adj_v_old      = adj_v_new

    The velocity identity requires no copy because adj_v is carried in place.

    Gather-form implementation avoids parallel write conflicts.
    """
    nx, nz = adj_sxx.shape
    M = a.size

    for i in prange(nx):
        for j in range(nz):
            add_sxx = 0.0
            add_szz = 0.0
            add_sxz = 0.0

            for m in range(1, M + 1):
                am = a[m - 1]

                # --------------------------------------------------------------
                # sxx -> D_plus_x(sxx) -> vx
                # For source i:
                #   positive target ti = i-m
                #   negative target ti = i+m-1
                # --------------------------------------------------------------
                ti = i - m
                if M <= ti < nx - M and M <= j < nz - M:
                    q_vx = (
                        dt
                        * bx[ti, j]
                        * adj_vx[ti, j]
                    )
                    add_sxx += am * q_vx / dx

                ti = i + m - 1
                if M <= ti < nx - M and M <= j < nz - M:
                    q_vx = (
                        dt
                        * bx[ti, j]
                        * adj_vx[ti, j]
                    )
                    add_sxx -= am * q_vx / dx

                # --------------------------------------------------------------
                # szz -> D_plus_z(szz) -> vz
                # --------------------------------------------------------------
                tj = j - m
                if M <= i < nx - M and M <= tj < nz - M:
                    q_vz = (
                        dt
                        * bz[i, tj]
                        * adj_vz[i, tj]
                    )
                    add_szz += am * q_vz / dz

                tj = j + m - 1
                if M <= i < nx - M and M <= tj < nz - M:
                    q_vz = (
                        dt
                        * bz[i, tj]
                        * adj_vz[i, tj]
                    )
                    add_szz -= am * q_vz / dz

                # --------------------------------------------------------------
                # sxz -> D_minus_z(sxz) -> vx
                # For source j:
                #   positive target tj = j-m+1
                #   negative target tj = j+m
                # --------------------------------------------------------------
                tj = j - m + 1
                if M <= i < nx - M and M <= tj < nz - M:
                    q_vx = (
                        dt
                        * bx[i, tj]
                        * adj_vx[i, tj]
                    )
                    add_sxz += am * q_vx / dz

                tj = j + m
                if M <= i < nx - M and M <= tj < nz - M:
                    q_vx = (
                        dt
                        * bx[i, tj]
                        * adj_vx[i, tj]
                    )
                    add_sxz -= am * q_vx / dz

                # --------------------------------------------------------------
                # sxz -> D_minus_x(sxz) -> vz
                # --------------------------------------------------------------
                ti = i - m + 1
                if M <= ti < nx - M and M <= j < nz - M:
                    q_vz = (
                        dt
                        * bz[ti, j]
                        * adj_vz[ti, j]
                    )
                    add_sxz += am * q_vz / dx

                ti = i + m
                if M <= ti < nx - M and M <= j < nz - M:
                    q_vz = (
                        dt
                        * bz[ti, j]
                        * adj_vz[ti, j]
                    )
                    add_sxz -= am * q_vz / dx

            adj_sxx[i, j] += add_sxx
            adj_szz[i, j] += add_szz
            adj_sxz[i, j] += add_sxz


# ==============================================================================
# 5. SOURCE-INJECTION TRANSPOSE
# ==============================================================================

@njit(fastmath=True, cache=True)
def gather_source_adjoint_numba(
    adj_sxx: np.ndarray,
    adj_szz: np.ndarray,
    adj_sxz: np.ndarray,
    normal_ix: np.ndarray,
    normal_iz: np.ndarray,
    normal_w: np.ndarray,
    shear_ix: np.ndarray,
    shear_iz: np.ndarray,
    shear_w: np.ndarray,
    src_scale: float,
) -> tuple[float, float, float]:
    """
    Apply G^T to the current pre-sponge stress adjoint.

    Forward source add:
        sigma += stf_component * src_scale * weights

    Returned values are gradients with respect to stf_xx, stf_zz, stf_xz.
    """
    grad_xx = 0.0
    grad_zz = 0.0
    grad_xz = 0.0

    for k in range(4):
        i = normal_ix[k]
        j = normal_iz[k]
        w = normal_w[k]

        grad_xx += (
            src_scale
            * w
            * adj_sxx[i, j]
        )
        grad_zz += (
            src_scale
            * w
            * adj_szz[i, j]
        )

        i = shear_ix[k]
        j = shear_iz[k]
        w = shear_w[k]

        grad_xz += (
            src_scale
            * w
            * adj_sxz[i, j]
        )

    return grad_xx, grad_zz, grad_xz


# ==============================================================================
# 6. HIGH-LEVEL NUMBA ADJOINT ENGINE
# ==============================================================================

class NumbaElasticDASAdjoint:
    """
    Reverse-time discrete adjoint for the current production full-space solver.

    Geometry/sampling metadata is reused from the forward engine.  Model
    coefficients are prepared once per adjoint call; the expensive reverse
    propagation stays in compiled kernels.

    Notes
    -----
    free_surface=True is intentionally rejected for now.  The Robertsson/Graves
    free-surface operations have their own discrete transpose and must be added
    explicitly rather than silently approximated.
    """

    def __init__(
        self,
        *,
        grid,
        receivers,
        receiver_sampling: ReceiverSampling2D,
        gauge_length_m: float,
        solver: SolverSpec,
    ) -> None:
        self.grid = grid
        self.receivers = receivers
        self.receiver_sampling = receiver_sampling
        self.gauge_length_m = float(gauge_length_m)
        self.solver = solver

        if solver.free_surface:
            raise NotImplementedError(
                "The first exact Numba adjoint supports free_surface=False only. "
                "Do not approximate the free-surface transpose."
            )

        if self.gauge_length_m <= 0.0:
            raise ValueError(
                "gauge_length_m must be positive."
            )

    @classmethod
    def from_forward_engine(
        cls,
        forward_engine,
    ) -> "NumbaElasticDASAdjoint":
        """Build an adjoint engine reusing the forward engine's cached geometry."""
        return cls(
            grid=forward_engine.grid,
            receivers=forward_engine.receivers,
            receiver_sampling=forward_engine.receiver_sampling,
            gauge_length_m=forward_engine.gauge_length_m,
            solver=forward_engine.solver,
        )

    def _prepare_model_coefficients(
        self,
        medium: ElasticMedium2D,
    ):
        if medium.grid is not self.grid:
            raise ValueError(
                "medium.grid must be the same Grid2D instance used by the "
                "adjoint engine."
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

        mu = rho * vs**2
        lam = rho * (
            vp**2
            - 2.0 * vs**2
        )
        l2m = lam + 2.0 * mu

        bx, bz, mu_xz = prepare_staggered_materials(
            rho,
            mu,
        )

        courant_rep = (
            float(vp.mean())
            * float(self.grid.dt)
            / float(self.grid.dx)
        )

        a = fd_coefficients(
            self.solver.half_order,
            use_ts_sfd=self.solver.use_ts_sfd,
            courant=courant_rep,
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
            np.ascontiguousarray(lam),
            np.ascontiguousarray(l2m),
            np.ascontiguousarray(mu_xz),
            np.ascontiguousarray(bx),
            np.ascontiguousarray(bz),
            np.ascontiguousarray(a),
            np.ascontiguousarray(sponge),
        )

    def run_receiver_adjoint_to_source(
        self,
        *,
        medium: ElasticMedium2D,
        source: PreparedMomentTensorSource,
        receiver_qx: np.ndarray,
        receiver_qz: np.ndarray,
    ) -> SourceAdjointResult:
        """
        Apply the transpose of:
            source sequences -> elastic propagation -> receiver Vx/Vz.

        receiver_qx/qz must have shape (nrec, nt).
        """
        qx = np.asarray(
            receiver_qx,
            dtype=np.float64,
        )
        qz = np.asarray(
            receiver_qz,
            dtype=np.float64,
        )

        expected_shape = (
            int(self.receivers.nrec),
            int(self.grid.nt),
        )

        if qx.shape != expected_shape or qz.shape != expected_shape:
            raise ValueError(
                "receiver_qx and receiver_qz must both have shape "
                f"{expected_shape}; got {qx.shape} and {qz.shape}."
            )

        if not np.all(np.isfinite(qx)) or not np.all(np.isfinite(qz)):
            raise ValueError(
                "receiver adjoint traces contain NaN or Inf."
            )

        (
            lam,
            l2m,
            mu_xz,
            bx,
            bz,
            a,
            sponge,
        ) = self._prepare_model_coefficients(
            medium
        )

        nx = int(self.grid.nx)
        nz = int(self.grid.nz)
        nt = int(self.grid.nt)

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

        grad_stf_xx = np.zeros(
            nt,
            dtype=np.float64,
        )
        grad_stf_zz = np.zeros(
            nt,
            dtype=np.float64,
        )
        grad_stf_xz = np.zeros(
            nt,
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

        start = perf_counter()

        for it in range(
            nt - 1,
            -1,
            -1,
        ):
            # --------------------------------------------------------------
            # 5^T. Receiver sampling transpose.
            #
            # Forward samples the already-damped velocity state.
            # Therefore receiver adjoint values are first added to the
            # adjoint of that post-sponge state.
            # --------------------------------------------------------------
            scatter_receivers_adjoint_numba(
                adj_vx,
                adj_vz,
                qx[:, it],
                qz[:, it],
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

            # --------------------------------------------------------------
            # 4^T. Sponge transpose.
            #
            # The forward sponge is diagonal multiplication, hence its
            # Euclidean transpose is exactly the same multiplication.
            # --------------------------------------------------------------
            apply_sponge_numba(
                adj_vx,
                adj_vz,
                adj_sxx,
                adj_szz,
                adj_sxz,
                sponge,
            )

            # --------------------------------------------------------------
            # 3^T. Source injection transpose.
            #
            # Must occur after sponge^T because the forward source was added
            # immediately before the sponge.
            # --------------------------------------------------------------
            (
                grad_stf_xx[it],
                grad_stf_zz[it],
                grad_stf_xz[it],
            ) = gather_source_adjoint_numba(
                adj_sxx,
                adj_szz,
                adj_sxz,
                inj.normal_ix,
                inj.normal_iz,
                inj.normal_w,
                inj.shear_ix,
                inj.shear_iz,
                inj.shear_w,
                src_scale,
            )

            # --------------------------------------------------------------
            # 2^T. Stress-update transpose.
            # --------------------------------------------------------------
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

            # --------------------------------------------------------------
            # 1^T. Velocity-update transpose.
            # --------------------------------------------------------------
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

        elapsed_s = (
            perf_counter()
            - start
        )

        return SourceAdjointResult(
            grad_stf_xx=grad_stf_xx,
            grad_stf_zz=grad_stf_zz,
            grad_stf_xz=grad_stf_xz,
            elapsed_s=elapsed_s,
        )

    def run_das_adjoint_to_source(
        self,
        *,
        medium: ElasticMedium2D,
        source: PreparedMomentTensorSource,
        das_adjoint_data: np.ndarray,
        das_template: DASResult,
    ) -> SourceAdjointResult:
        """
        Apply the complete observation transpose:
            DAS residual
              -> D^T receiver traces
              -> S^T
              -> elastic reverse propagation
              -> source-sequence adjoint.

        das_template must be the DASResult produced by the matching forward
        geometry/gauge configuration.  Its interpolation metadata defines D^T.
        """
        q_das = np.asarray(
            das_adjoint_data,
            dtype=np.float64,
        )

        if q_das.shape != das_template.data.shape:
            raise ValueError(
                "das_adjoint_data shape must match das_template.data: "
                f"{q_das.shape} != {das_template.data.shape}."
            )

        qx, qz = das_adjoint(
            q_das,
            self.receivers,
            das_template,
        )

        return self.run_receiver_adjoint_to_source(
            medium=medium,
            source=source,
            receiver_qx=qx,
            receiver_qz=qz,
        )
