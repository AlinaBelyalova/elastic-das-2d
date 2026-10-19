"""
2-D isotropic elastic wave equation on a staggered SBP-SAT grid.

This is an intentionally separate, correctness-first solver branch.

Grid layout
-----------
x is periodic in the first controlled implementation, so the N and M x-grids
contain the same number of points and are shifted by dx/2.

z has physical boundaries:
    N_z : boundary aligned, z_j = j dz, j=0,...,Nz
    M_z : staggered,        z_j = (j+1/2) dz, j=0,...,Nz-1

Field placement:
    sxx, szz : (N_x, N_z)
    vx       : (M_x, N_z)
    vz       : (N_x, M_z)
    sxz      : (M_x, M_z)

The top boundary is z=0.  Since this project uses depth-positive-down
coordinates, the top is the LEFT endpoint of the one-dimensional z SBP pair.

Free surface
------------
Traction-free conditions are imposed weakly with SAT penalties in the velocity
equations, following Gao & Keyes (2019):

    szz = 0,
    sxz = 0

at top and bottom.  With z positive downward the energy-conserving penalty
signs are:
    top (left endpoint)    eta = +1
    bottom (right endpoint) eta = -1.

No ghost nodes are used.

Scope of this first branch
--------------------------
* fourth-order staggered interior derivatives;
* second-order SBP boundary closures;
* periodic x;
* free-surface SAT at top and bottom;
* second-order leapfrog time integration;
* diagonal normal-stress point source;
* native-grid point receivers;
* exact Euclidean discrete adjoint;
* exact model gradient for log(Vp), log(Vs), rho fixed.

Absorbing boundaries, arbitrary source interpolation, DAS observation physics,
and Numba optimization are deliberately deferred until the SBP-SAT core passes
its own forward/adjoint/Taylor validations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np

from .operators import (
    StaggeredSBP1D,
    periodic_D_M_to_N,
    periodic_D_N_to_M,
)


@dataclass(frozen=True)
class StressPointSource:
    """
    Scalar source time function injected into sxx/szz on the N-N stress grid.

    At each integer-stress update:
        sxx[ix, iz] += dt * mxx * stf[it]
        szz[ix, iz] += dt * mzz * stf[it]
    """
    ix: int
    iz: int
    mxx: float = 1.0
    mzz: float = 1.0


@dataclass(frozen=True)
class PointReceiver:
    """Point sample on one native velocity subgrid."""
    component: str
    ix: int
    iz: int

    def __post_init__(self) -> None:
        if self.component not in ("vx", "vz"):
            raise ValueError("component must be 'vx' or 'vz'.")


@dataclass(frozen=True)
class SBPSATModel2D:
    """Material coefficients on all native staggered subgrids."""
    vp: np.ndarray
    vs: np.ndarray
    rho: np.ndarray

    lam_NN: np.ndarray
    l2m_NN: np.ndarray
    mu_NN: np.ndarray
    mu_MM: np.ndarray
    rho_vx: np.ndarray
    rho_vz: np.ndarray

    @property
    def nx(self) -> int:
        return int(self.vp.shape[0])

    @property
    def nz_N(self) -> int:
        return int(self.vp.shape[1])

    @property
    def nz_M(self) -> int:
        return self.nz_N - 1


@dataclass
class SBPSATState:
    vx: np.ndarray
    vz: np.ndarray
    sxx: np.ndarray
    szz: np.ndarray
    sxz: np.ndarray

    def copy(self) -> "SBPSATState":
        return SBPSATState(
            vx=self.vx.copy(),
            vz=self.vz.copy(),
            sxx=self.sxx.copy(),
            szz=self.szz.copy(),
            sxz=self.sxz.copy(),
        )


@dataclass(frozen=True)
class ForwardResult:
    data: np.ndarray
    final_state: SBPSATState
    velocity_history: tuple[tuple[np.ndarray, np.ndarray], ...] | None


def _validate_base_model(vp: np.ndarray, vs: np.ndarray, rho: np.ndarray) -> None:
    if vp.ndim != 2:
        raise ValueError("vp, vs, rho must be 2-D arrays.")
    if vs.shape != vp.shape or rho.shape != vp.shape:
        raise ValueError("vp, vs, rho must have identical shapes.")
    if vp.shape[1] < 9:
        raise ValueError("z N-grid needs at least 9 points for the 4-2 SBP pair.")
    if not (
        np.all(np.isfinite(vp))
        and np.all(np.isfinite(vs))
        and np.all(np.isfinite(rho))
    ):
        raise ValueError("vp, vs, rho must be finite.")
    if np.any(vp <= 0.0) or np.any(vs <= 0.0) or np.any(rho <= 0.0):
        raise ValueError("vp, vs, rho must be positive.")
    if np.any(vp <= vs):
        raise ValueError("vp must exceed vs.")
    if np.any(vs > vp / np.sqrt(2.0)):
        raise ValueError("This branch requires lambda >= 0: vs <= vp/sqrt(2).")


def _average_NN_to_MM(a: np.ndarray) -> np.ndarray:
    """
    Arithmetic average from the N-N material grid to the M-M shear grid.

    x is periodic. z is bounded.
    """
    a_x = 0.5 * (a + np.roll(a, -1, axis=0))
    return 0.5 * (a_x[:, :-1] + a_x[:, 1:])


def _average_NN_to_MM_transpose(g_MM: np.ndarray) -> np.ndarray:
    """Exact Euclidean transpose of _average_NN_to_MM."""
    nx, nz_M = g_MM.shape
    g_NN = np.zeros((nx, nz_M + 1), dtype=np.float64)

    # Contributions from x index i.
    g_NN[:, :-1] += 0.25 * g_MM
    g_NN[:, 1:] += 0.25 * g_MM

    # Contributions from x index i+1 in the forward average.
    g_shift = np.roll(g_MM, 1, axis=0)
    g_NN[:, :-1] += 0.25 * g_shift
    g_NN[:, 1:] += 0.25 * g_shift

    return g_NN


def prepare_model(
    vp: np.ndarray,
    vs: np.ndarray,
    rho: np.ndarray,
) -> SBPSATModel2D:
    """
    Prepare native-grid coefficients.

    The inversion model lives on the N-N grid. rho is fixed in the first
    gradient implementation.

    Native coefficients:
        lambda_NN = rho (vp^2 - 2 vs^2)
        (lambda+2mu)_NN = rho vp^2
        mu_NN = rho vs^2

    mu_MM is an arithmetic four-cell average of mu_NN. This choice is positive,
    simple, and—crucially for the first adjoint branch—has an explicit exact
    transpose used by the gradient.
    """
    vp = np.ascontiguousarray(np.asarray(vp, dtype=np.float64))
    vs = np.ascontiguousarray(np.asarray(vs, dtype=np.float64))
    rho = np.ascontiguousarray(np.asarray(rho, dtype=np.float64))
    _validate_base_model(vp, vs, rho)

    mu_NN = rho * vs**2
    lam_NN = rho * (vp**2 - 2.0 * vs**2)
    l2m_NN = rho * vp**2

    # Density at native velocity locations.
    rho_vx = 0.5 * (rho + np.roll(rho, -1, axis=0))
    rho_vz = 0.5 * (rho[:, :-1] + rho[:, 1:])

    mu_MM = _average_NN_to_MM(mu_NN)

    return SBPSATModel2D(
        vp=vp,
        vs=vs,
        rho=rho,
        lam_NN=lam_NN,
        l2m_NN=l2m_NN,
        mu_NN=mu_NN,
        mu_MM=mu_MM,
        rho_vx=rho_vx,
        rho_vz=rho_vz,
    )


def zero_state(model: SBPSATModel2D) -> SBPSATState:
    return SBPSATState(
        vx=np.zeros((model.nx, model.nz_N), dtype=np.float64),
        vz=np.zeros((model.nx, model.nz_M), dtype=np.float64),
        sxx=np.zeros((model.nx, model.nz_N), dtype=np.float64),
        szz=np.zeros((model.nx, model.nz_N), dtype=np.float64),
        sxz=np.zeros((model.nx, model.nz_M), dtype=np.float64),
    )


def _check_ops(model: SBPSATModel2D, op_z: StaggeredSBP1D) -> None:
    if op_z.n_N != model.nz_N or op_z.n_M != model.nz_M:
        raise ValueError(
            "z SBP operator dimensions do not match the model: "
            f"operator N/M=({op_z.n_N},{op_z.n_M}), "
            f"model N/M=({model.nz_N},{model.nz_M})."
        )


def _Dz_M_to_N(f: np.ndarray, op_z: StaggeredSBP1D) -> np.ndarray:
    return f @ op_z.D_M_to_N.T


def _Dz_N_to_M(f: np.ndarray, op_z: StaggeredSBP1D) -> np.ndarray:
    return f @ op_z.D_N_to_M.T


def velocity_rhs(
    state: SBPSATState,
    model: SBPSATModel2D,
    op_z: StaggeredSBP1D,
    dx: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Semi-discrete velocity RHS including top/bottom traction-free SATs.

    Before division by density, the interior elastic divergence is

        fx = Dx(sxx) + Dz(sxz)
        fz = Dx(sxz) + Dz(szz).

    The staggered SBP identity leaves boundary energy fluxes. The SAT terms
    cancel those fluxes for homogeneous traction conditions. Because z grows
    downward here, the top/left penalty has +1 and the bottom/right penalty -1.
    """
    _check_ops(model, op_z)

    fx = (
        periodic_D_N_to_M(state.sxx, dx)
        + _Dz_M_to_N(state.sxz, op_z)
    )
    fz = (
        periodic_D_M_to_N(state.sxz, dx)
        + _Dz_N_to_M(state.szz, op_z)
    )

    # Tangential traction sxz lives on the staggered M_z grid. Project it to
    # each boundary, then inject the SAT into boundary-aligned vx rows.
    t_x_top = state.sxz @ op_z.p_left
    t_x_bottom = state.sxz @ op_z.p_right

    fx[:, 0] += t_x_top / op_z.H_N[0]
    fx[:, -1] -= t_x_bottom / op_z.H_N[-1]

    # Normal traction szz already lives on boundary-aligned N_z points. The
    # SAT is injected into the staggered vz grid through H_M^{-1} p.
    fz += (
        state.szz[:, 0, None]
        * (op_z.p_left / op_z.H_M)[None, :]
    )
    fz -= (
        state.szz[:, -1, None]
        * (op_z.p_right / op_z.H_M)[None, :]
    )

    return fx / model.rho_vx, fz / model.rho_vz


def stress_rhs(
    vx: np.ndarray,
    vz: np.ndarray,
    model: SBPSATModel2D,
    op_z: StaggeredSBP1D,
    dx: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Semi-discrete isotropic stress RHS on native SBP subgrids."""
    exx = periodic_D_M_to_N(vx, dx)
    ezz = _Dz_M_to_N(vz, op_z)
    gamma_xz = (
        _Dz_N_to_M(vx, op_z)
        + periodic_D_N_to_M(vz, dx)
    )

    dsxx = model.l2m_NN * exx + model.lam_NN * ezz
    dszz = model.lam_NN * exx + model.l2m_NN * ezz
    dsxz = model.mu_MM * gamma_xz
    return dsxx, dszz, dsxz


def _validate_source(source: StressPointSource, model: SBPSATModel2D) -> None:
    if not (0 <= source.ix < model.nx and 0 <= source.iz < model.nz_N):
        raise ValueError("source index is outside the N-N stress grid.")


def _validate_receivers(
    receivers: Sequence[PointReceiver],
    model: SBPSATModel2D,
) -> None:
    if len(receivers) == 0:
        raise ValueError("At least one receiver is required.")
    for rec in receivers:
        nz = model.nz_N if rec.component == "vx" else model.nz_M
        if not (0 <= rec.ix < model.nx and 0 <= rec.iz < nz):
            raise ValueError(f"receiver {rec} is outside its native grid.")


def sample_receivers(
    vx: np.ndarray,
    vz: np.ndarray,
    receivers: Sequence[PointReceiver],
) -> np.ndarray:
    out = np.empty(len(receivers), dtype=np.float64)
    for ir, rec in enumerate(receivers):
        if rec.component == "vx":
            out[ir] = vx[rec.ix, rec.iz]
        else:
            out[ir] = vz[rec.ix, rec.iz]
    return out


def scatter_receivers_adjoint(
    q: np.ndarray,
    receivers: Sequence[PointReceiver],
    adj_vx: np.ndarray,
    adj_vz: np.ndarray,
) -> None:
    for ir, rec in enumerate(receivers):
        if rec.component == "vx":
            adj_vx[rec.ix, rec.iz] += q[ir]
        else:
            adj_vz[rec.ix, rec.iz] += q[ir]


def advance_one_step(
    state: SBPSATState,
    model: SBPSATModel2D,
    op_z: StaggeredSBP1D,
    dx: float,
    dt: float,
    *,
    source: StressPointSource | None = None,
    source_amplitude: float = 0.0,
) -> SBPSATState:
    """
    One leapfrog step:

        [sigma^n, v^(n-1/2)]
              -> v^(n+1/2)    (SBP derivative + SAT)
              -> sigma^(n+1)
              -> source injection.

    SATs are already part of the velocity RHS. There are no ghost fills.
    """
    if dt <= 0.0 or dx <= 0.0:
        raise ValueError("dx and dt must be positive.")

    vx = state.vx.copy()
    vz = state.vz.copy()
    sxx = state.sxx.copy()
    szz = state.szz.copy()
    sxz = state.sxz.copy()

    dvx, dvz = velocity_rhs(state, model, op_z, dx)
    vx += dt * dvx
    vz += dt * dvz

    dsxx, dszz, dsxz = stress_rhs(vx, vz, model, op_z, dx)
    sxx += dt * dsxx
    szz += dt * dszz
    sxz += dt * dsxz

    if source is not None:
        _validate_source(source, model)
        sxx[source.ix, source.iz] += (
            dt * float(source_amplitude) * float(source.mxx)
        )
        szz[source.ix, source.iz] += (
            dt * float(source_amplitude) * float(source.mzz)
        )

    return SBPSATState(vx=vx, vz=vz, sxx=sxx, szz=szz, sxz=sxz)


def run_forward(
    stf: np.ndarray,
    model: SBPSATModel2D,
    op_z: StaggeredSBP1D,
    dx: float,
    dt: float,
    source: StressPointSource,
    receivers: Sequence[PointReceiver],
    *,
    store_velocity_history: bool = False,
    initial_state: SBPSATState | None = None,
) -> ForwardResult:
    """Run the controlled SBP-SAT forward source-to-receiver operator."""
    stf = np.asarray(stf, dtype=np.float64)
    if stf.ndim != 1 or stf.size < 1:
        raise ValueError("stf must be a nonempty 1-D array.")
    _check_ops(model, op_z)
    _validate_source(source, model)
    _validate_receivers(receivers, model)

    state = zero_state(model) if initial_state is None else initial_state.copy()
    data = np.zeros((len(receivers), stf.size), dtype=np.float64)
    history: list[tuple[np.ndarray, np.ndarray]] | None
    history = [] if store_velocity_history else None

    for it in range(stf.size):
        # Velocity half step.
        dvx, dvz = velocity_rhs(state, model, op_z, dx)
        vx_new = state.vx + dt * dvx
        vz_new = state.vz + dt * dvz

        data[:, it] = sample_receivers(vx_new, vz_new, receivers)

        if history is not None:
            history.append((vx_new.copy(), vz_new.copy()))

        # Stress integer step.
        dsxx, dszz, dsxz = stress_rhs(
            vx_new, vz_new, model, op_z, dx
        )
        sxx_new = state.sxx + dt * dsxx
        szz_new = state.szz + dt * dszz
        sxz_new = state.sxz + dt * dsxz

        sxx_new[source.ix, source.iz] += (
            dt * stf[it] * source.mxx
        )
        szz_new[source.ix, source.iz] += (
            dt * stf[it] * source.mzz
        )

        state = SBPSATState(
            vx=vx_new,
            vz=vz_new,
            sxx=sxx_new,
            szz=szz_new,
            sxz=sxz_new,
        )

    return ForwardResult(
        data=data,
        final_state=state,
        velocity_history=None if history is None else tuple(history),
    )


# =============================================================================
# Exact Euclidean discrete transpose
# =============================================================================

def _stress_update_transpose(
    adj_sxx: np.ndarray,
    adj_szz: np.ndarray,
    adj_sxz: np.ndarray,
    model: SBPSATModel2D,
    op_z: StaggeredSBP1D,
    dx: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transpose contribution from

        sigma_new = sigma_old + dt K_sigma(v_new).

    This returns the contribution to adjoint v_new. The identity contribution
    sigma_new -> sigma_old is handled by keeping the stress adjoint arrays.
    """
    qxx = dt * adj_sxx
    qzz = dt * adj_szz
    qxz = dt * adj_sxz

    adj_exx = model.l2m_NN * qxx + model.lam_NN * qzz
    adj_ezz = model.lam_NN * qxx + model.l2m_NN * qzz
    adj_gamma = model.mu_MM * qxz

    # Periodic x derivatives have exact Euclidean transpose relations:
    #   (D_M->N)^T = -D_N->M
    #   (D_N->M)^T = -D_M->N
    adj_vx = (
        -periodic_D_N_to_M(adj_exx, dx)
        + adj_gamma @ op_z.D_N_to_M
    )
    adj_vz = (
        adj_ezz @ op_z.D_M_to_N
        - periodic_D_M_to_N(adj_gamma, dx)
    )
    return adj_vx, adj_vz


def _velocity_update_transpose(
    adj_vx: np.ndarray,
    adj_vz: np.ndarray,
    model: SBPSATModel2D,
    op_z: StaggeredSBP1D,
    dx: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transpose contribution from

        v_new = v_old + dt rho^{-1} [div(sigma) + SAT(sigma)].

    SAT transpose terms are written explicitly; this is what makes the adjoint
    the transpose of the ACTUAL weak boundary implementation rather than a
    continuous-adjoint approximation.
    """
    qx = dt * adj_vx / model.rho_vx
    qz = dt * adj_vz / model.rho_vz

    adj_sxx = -periodic_D_M_to_N(qx, dx)

    # sxz -> vx via Dz(M->N), plus its top/bottom traction SATs.
    adj_sxz = qx @ op_z.D_M_to_N
    adj_sxz += np.outer(
        qx[:, 0] / op_z.H_N[0],
        op_z.p_left,
    )
    adj_sxz -= np.outer(
        qx[:, -1] / op_z.H_N[-1],
        op_z.p_right,
    )

    # sxz -> vz via periodic Dx(M->N).
    adj_sxz += -periodic_D_N_to_M(qz, dx)

    # szz -> vz via Dz(N->M), plus normal-traction SATs.
    adj_szz = qz @ op_z.D_N_to_M
    adj_szz[:, 0] += qz @ (op_z.p_left / op_z.H_M)
    adj_szz[:, -1] -= qz @ (op_z.p_right / op_z.H_M)

    return adj_sxx, adj_szz, adj_sxz


def run_adjoint_to_source(
    q_data: np.ndarray,
    model: SBPSATModel2D,
    op_z: StaggeredSBP1D,
    dx: float,
    dt: float,
    source: StressPointSource,
    receivers: Sequence[PointReceiver],
) -> np.ndarray:
    """
    Apply the exact Euclidean transpose of run_forward to receiver-space data.

    If F maps source sequence -> receiver data, this returns F^T q_data.
    """
    q_data = np.asarray(q_data, dtype=np.float64)
    if q_data.ndim != 2:
        raise ValueError("q_data must have shape (n_receivers, nt).")
    if q_data.shape[0] != len(receivers):
        raise ValueError("q_data receiver dimension does not match receivers.")

    nt = q_data.shape[1]

    adj_vx = np.zeros((model.nx, model.nz_N), dtype=np.float64)
    adj_vz = np.zeros((model.nx, model.nz_M), dtype=np.float64)
    adj_sxx = np.zeros((model.nx, model.nz_N), dtype=np.float64)
    adj_szz = np.zeros((model.nx, model.nz_N), dtype=np.float64)
    adj_sxz = np.zeros((model.nx, model.nz_M), dtype=np.float64)

    grad_stf = np.zeros(nt, dtype=np.float64)

    for it in range(nt - 1, -1, -1):
        # Forward source injection is the last stress-state operation.
        grad_stf[it] = dt * (
            source.mxx * adj_sxx[source.ix, source.iz]
            + source.mzz * adj_szz[source.ix, source.iz]
        )

        # Reverse stress update. sigma identity means adj_s* remain carried.
        add_vx, add_vz = _stress_update_transpose(
            adj_sxx, adj_szz, adj_sxz,
            model, op_z, dx, dt,
        )
        adj_vx += add_vx
        adj_vz += add_vz

        # Receiver sampling transpose.
        scatter_receivers_adjoint(
            q_data[:, it], receivers, adj_vx, adj_vz
        )

        # Reverse velocity update. v identity means adj_v* remain carried.
        add_sxx, add_szz, add_sxz = _velocity_update_transpose(
            adj_vx, adj_vz,
            model, op_z, dx, dt,
        )
        adj_sxx += add_sxx
        adj_szz += add_szz
        adj_sxz += add_sxz

    return grad_stf


# =============================================================================
# Exact rho-fixed model gradient
# =============================================================================

def objective_and_gradient_log_vp_vs(
    stf: np.ndarray,
    observed: np.ndarray,
    model: SBPSATModel2D,
    op_z: StaggeredSBP1D,
    dx: float,
    dt: float,
    source: StressPointSource,
    receivers: Sequence[PointReceiver],
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute J and exact discrete gradients for log(Vp), log(Vs), rho fixed.

    Objective:
        J = 0.5 * dt * ||d_syn - d_obs||_2^2.

    Model dependence in this first branch:
        lambda_NN = rho (Vp^2 - 2 Vs^2)
        l2m_NN    = rho Vp^2
        mu_NN     = rho Vs^2
        mu_MM     = A mu_NN

    where A is the four-cell arithmetic staggered average.  The gradient uses
    A^T exactly.
    """
    observed = np.asarray(observed, dtype=np.float64)

    fwd = run_forward(
        stf,
        model,
        op_z,
        dx,
        dt,
        source,
        receivers,
        store_velocity_history=True,
    )
    if observed.shape != fwd.data.shape:
        raise ValueError(
            f"observed shape {observed.shape} != synthetic {fwd.data.shape}."
        )
    if fwd.velocity_history is None:
        raise RuntimeError("Internal error: velocity history was not stored.")

    residual = fwd.data - observed
    J = 0.5 * dt * float(np.sum(residual**2))

    adj_vx = np.zeros((model.nx, model.nz_N), dtype=np.float64)
    adj_vz = np.zeros((model.nx, model.nz_M), dtype=np.float64)
    adj_sxx = np.zeros((model.nx, model.nz_N), dtype=np.float64)
    adj_szz = np.zeros((model.nx, model.nz_N), dtype=np.float64)
    adj_sxz = np.zeros((model.nx, model.nz_M), dtype=np.float64)

    g_lam = np.zeros_like(model.lam_NN)
    g_l2m = np.zeros_like(model.l2m_NN)
    g_mu_MM = np.zeros_like(model.mu_MM)

    q_data = dt * residual

    for it in range(stf.size - 1, -1, -1):
        # Gradient of the stress-update material multiplications.  It must use
        # the adjoint of sigma_new BEFORE that adjoint is propagated backward.
        vx_new, vz_new = fwd.velocity_history[it]

        exx = periodic_D_M_to_N(vx_new, dx)
        ezz = _Dz_M_to_N(vz_new, op_z)
        gamma_xz = (
            _Dz_N_to_M(vx_new, op_z)
            + periodic_D_N_to_M(vz_new, dx)
        )

        g_lam += dt * (
            adj_sxx * ezz
            + adj_szz * exx
        )
        g_l2m += dt * (
            adj_sxx * exx
            + adj_szz * ezz
        )
        g_mu_MM += dt * (
            adj_sxz * gamma_xz
        )

        # Reverse stress update.
        add_vx, add_vz = _stress_update_transpose(
            adj_sxx, adj_szz, adj_sxz,
            model, op_z, dx, dt,
        )
        adj_vx += add_vx
        adj_vz += add_vz

        # Objective / receiver transpose.
        scatter_receivers_adjoint(
            q_data[:, it], receivers, adj_vx, adj_vz
        )

        # Reverse velocity update, including SAT^T.
        add_sxx, add_szz, add_sxz = _velocity_update_transpose(
            adj_vx, adj_vz,
            model, op_z, dx, dt,
        )
        adj_sxx += add_sxx
        adj_szz += add_szz
        adj_sxz += add_sxz

    g_mu_NN = _average_NN_to_MM_transpose(g_mu_MM)

    # Chain rule from (lambda, l2m, mu) to Vp, Vs with rho fixed.
    #
    # lambda = rho(Vp^2 - 2 Vs^2)
    # l2m    = rho Vp^2
    # mu     = rho Vs^2
    g_vp = (
        2.0 * model.rho * model.vp
        * (g_lam + g_l2m)
    )
    g_vs = (
        -4.0 * model.rho * model.vs * g_lam
        + 2.0 * model.rho * model.vs * g_mu_NN
    )

    g_log_vp = model.vp * g_vp
    g_log_vs = model.vs * g_vs

    return J, fwd.data, g_log_vp, g_log_vs


# =============================================================================
# Energy diagnostic
# =============================================================================

def semidiscrete_energy_rate(
    state: SBPSATState,
    model: SBPSATModel2D,
    op_z: StaggeredSBP1D,
    dx: float,
) -> tuple[float, float]:
    """
    Return (dE/dt, scale) for the semi-discrete source-free SBP-SAT system.

    For exact traction-free SATs, dE/dt should be roundoff-level. The returned
    scale is the sum of absolute kinetic/potential rate contributions and is
    useful for a relative residual.
    """
    dvx, dvz = velocity_rhs(state, model, op_z, dx)
    dsxx, dszz, dsxz = stress_rhs(
        state.vx, state.vz, model, op_z, dx
    )

    w_N = dx * op_z.H_N[None, :]
    w_M = dx * op_z.H_M[None, :]

    dE_kin = float(
        np.sum(w_N * model.rho_vx * state.vx * dvx)
        + np.sum(w_M * model.rho_vz * state.vz * dvz)
    )

    # Inverse of the 2x2 normal-stress stiffness block:
    # C = [[l2m, lambda], [lambda, l2m]].
    det = model.l2m_NN**2 - model.lam_NN**2
    S11 = model.l2m_NN / det
    S12 = -model.lam_NN / det

    dE_normal = float(
        np.sum(
            w_N
            * (
                state.sxx * (S11 * dsxx + S12 * dszz)
                + state.szz * (S12 * dsxx + S11 * dszz)
            )
        )
    )

    # Shear stress energy density is sxz^2/(2 mu).
    dE_shear = float(
        np.sum(
            w_M
            * (state.sxz / model.mu_MM)
            * dsxz
        )
    )

    total = dE_kin + dE_normal + dE_shear
    scale = abs(dE_kin) + abs(dE_normal) + abs(dE_shear)
    return total, scale
