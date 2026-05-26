# ==============================================================================
# src/solver_numpy.py — 2D staggered-grid elastic FD solver (NumPy)
#
# Formulation
#   Virieux (1986) first-order velocity-stress staggered-grid leapfrog.
#   Spatial order 2M (M=1..4), second-order in time.
#   Assumes dx == dz (uniform grid).
#
# Time staggering (leapfrog)
#
#   Integer times (stress grid):  t_sigma[n] = n * dt,       n = 0, 1, ..., nt
#   Half times   (velocity grid): t_v[n]     = (n+1/2) * dt, n = 0, 1, ..., nt-1
#
#   Each time step n advances:
#       sigma^n, v^(n-1/2)  →  v^(n+1/2)  →  sigma^(n+1)
#
#   At the start of iteration n:
#       state.{sxx, szz, sxz}  live at t_sigma[n]
#       state.{vx, vz}         live at t_v[n-1] = (n-1/2)*dt
#                              (t_v[-1] = -dt/2 initially, zeroed by IC)
#
#   After step 1 (velocity update):
#       state.{vx, vz}  advance to t_v[n] = (n+1/2)*dt
#
#   After step 2 (stress update + source injection):
#       state.{sxx, szz, sxz}  advance to t_sigma[n+1] = (n+1)*dt
#
# Source convention
#   stf_xx[n], stf_zz[n], stf_xz[n] are indexed n = 0, ..., nt-1
#   and injected into the updated stress field at t_sigma[n+1] = (n+1)*dt.
#   Scaling: src_scale = dt / (dx*dz).
#   sxz source is split equally over four surrounding sxz nodes so that
#   its centroid coincides with (source_ix, source_iz).
#
# Staggered-grid node placement (h = dx = dz)
#   sxx[i,j], szz[i,j]  at  (i*h,       j*h)
#   sxz[i,j]            at  ((i+1/2)*h, (j+1/2)*h)
#   vx[i,j]             at  ((i+1/2)*h, j*h)
#   vz[i,j]             at  (i*h,       (j+1/2)*h)
#
# Free surface (free_surface=True)
#   Physical surface at z = M*h (array index iz=M). The first M z-nodes
#   (iz=0..M-1) are ghost nodes (air). Top sponge is disabled.
#   Implementation: Robertsson (1996) image method + Graves (1996) correction.
#     - velocity ghost nodes extrapolated before each velocity update
#     - dvz_dz at surface overwritten to enforce σ_zz = 0 analytically
#     - stress ghost nodes set by mirror symmetry after each stress update
#   Moment-tensor sources must satisfy source_iz >= M+1: a source at iz=M
#   would inject into szz at the free surface, violating σ_zz = 0.
#
# Physical constraints
#   mu > 0,  lambda >= 0,  lambda + 2*mu > 0
#   (lambda >= 0 ↔ Vs <= Vp/sqrt(2))
#
# Sponge note
#   The multiplicative sponge is applied to both velocity and stress fields
#   at the end of each step. Because these fields live at different times,
#   this is an O(dt) approximation; it is standard practice and negligible
#   relative to the spatial discretisation error of the damping layer.
#
# Known limitation
#   FD operators leave M edge cells un-updated. Sponge layer must satisfy
#   n_boundary > M.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.sampling import ReceiverSampling2D, sample_receivers


# ==============================================================================
# 1. FD COEFFICIENTS
# ==============================================================================

_C_SFD: dict[int, tuple[float, ...]] = {
    1: (1.0,),
    2: (9 / 8, -1 / 24),
    3: (75 / 64, -25 / 384, 3 / 640),
    4: (1225 / 1024, -245 / 3072, 49 / 5120, -5 / 7168),
}


def c_sfd_coefficients(half_order: int) -> np.ndarray:
    """Classical space-domain Taylor FD coefficients for spatial order 2M."""
    if half_order not in _C_SFD:
        raise ValueError(f"half_order must be 1–4, got {half_order}.")
    return np.array(_C_SFD[half_order], dtype=np.float64)


def ts_sfd_coefficients(
    r: float,
    half_order: int,
    n_sample: int = 400,
) -> np.ndarray:
    """
    TS-SFD coefficients via constrained least squares.

    Minimises the L2 norm of the dispersion error over wavenumber space
    subject to the consistency constraint sum_m a_m*(2m-1) = 1.
    """
    if r <= 0.0:
        raise ValueError(f"r must be positive, got {r}.")
    if half_order <= 0:
        raise ValueError(f"half_order must be positive, got {half_order}.")
    if n_sample < 8:
        raise ValueError(f"n_sample must be >= 8, got {n_sample}.")

    M = half_order
    xi = np.linspace(1 / (2 * n_sample), 1 - 1 / (2 * n_sample), n_sample)

    lhs = np.sin(r * np.pi * xi / 2.0) / r
    A = np.column_stack(
        [np.sin((2 * m - 1) * np.pi * xi / 2.0) for m in range(1, M + 1)]
    )
    c = np.array([2 * m - 1 for m in range(1, M + 1)], dtype=np.float64)

    K = np.zeros((M + 1, M + 1), dtype=np.float64)
    K[:M, :M] = A.T @ A
    K[:M, M] = c
    K[M, :M] = c

    rhs = np.zeros(M + 1, dtype=np.float64)
    rhs[:M] = A.T @ lhs
    rhs[M] = 1.0

    a = np.linalg.solve(K, rhs)[:M]

    consist = float(np.dot(a, c))
    if abs(consist - 1.0) > 1e-9:
        raise RuntimeError(
            f"TS-SFD consistency violated: Σ a_m*(2m-1) = {consist:.8f} ≠ 1."
        )
    return a


def fd_coefficients(
    half_order: int,
    *,
    use_ts_sfd: bool = False,
    courant: float = 0.3,
) -> np.ndarray:
    """Return C-SFD (default) or TS-SFD staggered FD coefficients."""
    if use_ts_sfd:
        return ts_sfd_coefficients(courant, half_order)
    return c_sfd_coefficients(half_order)


# ==============================================================================
# 2. CFL STABILITY
# ==============================================================================

# Conservative Courant limits for the coupled 2D elastic P-SV system on the
# staggered grid. These are stricter than the scalar-wave bound 1/sum|a_m|.
_COURANT_LIMITS: dict[int, float] = {
    1: 0.707,
    2: 0.495,
    3: 0.409,
    4: 0.351,
}


def max_stable_dt(
    vp_max: float,
    dx: float,
    dz: float,
    half_order: int,
    safety: float = 0.90,
    use_ts_sfd: bool = False,
) -> float:
    """
    CFL upper bound on dt for the 2D elastic staggered-grid scheme.

    For dx = dz = h:
        dt <= safety * C_M / (vp_max * sqrt(2) / h)

    C_M is taken from conservative von Neumann limits for the coupled elastic
    system. When use_ts_sfd=True, an extra safety reduction is applied because
    TS-SFD stability depends on the Courant number used during optimisation.
    """
    if vp_max <= 0.0:
        raise ValueError(f"vp_max must be positive, got {vp_max}.")
    if dx <= 0.0 or dz <= 0.0:
        raise ValueError(f"dx, dz must be positive, got dx={dx}, dz={dz}.")
    if not (0.0 < safety <= 1.0):
        raise ValueError(f"safety must be in (0, 1], got {safety}.")

    C = _COURANT_LIMITS.get(half_order, 0.35)

    if use_ts_sfd:
        safety = min(safety, 0.80)

    return safety * C / (vp_max * np.sqrt(1.0 / dx**2 + 1.0 / dz**2))


# ==============================================================================
# 3. MATERIAL AVERAGING
# ==============================================================================

def prepare_staggered_materials(
    rho: np.ndarray,
    mu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate material parameters to staggered node positions.

    bx  = 1/rho at vx nodes — arithmetic average in x
    bz  = 1/rho at vz nodes — arithmetic average in z
    mu_xz = mu at sxz nodes — harmonic average of four neighbours
    """
    if rho.shape != mu.shape:
        raise ValueError(
            f"rho and mu must share shape; got {rho.shape} vs {mu.shape}."
        )
    if np.any(rho <= 0.0):
        raise ValueError("rho must be positive everywhere.")
    if np.any(mu <= 0.0):
        raise ValueError("mu must be positive everywhere.")

    bx = (1.0 / rho).copy()
    bx[:-1, :] = 2.0 / (rho[:-1, :] + rho[1:, :])

    bz = (1.0 / rho).copy()
    bz[:, :-1] = 2.0 / (rho[:, :-1] + rho[:, 1:])

    inv_mu = 1.0 / mu
    mu_xz = mu.copy()
    mu_xz[:-1, :-1] = 4.0 / (
        inv_mu[:-1, :-1] + inv_mu[1:, :-1]
        + inv_mu[:-1, 1:] + inv_mu[1:, 1:]
    )

    return bx, bz, mu_xz


# ==============================================================================
# 4. SPONGE BOUNDARY
# ==============================================================================

def make_sponge_mask(
    nx: int,
    nz: int,
    n_boundary: int,
    gamma_s: float,
    dt: float,
    free_surface: bool = False,
) -> np.ndarray:
    """
    Multiplicative Gaussian sponge mask.

    When free_surface=True, the top boundary is left unattenuated so the
    physical free surface is not damped.
    """
    if nx <= 1 or nz <= 1:
        raise ValueError(f"nx, nz must be > 1, got nx={nx}, nz={nz}.")
    if gamma_s < 0.0:
        raise ValueError(f"gamma_s must be non-negative, got {gamma_s}.")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}.")

    if n_boundary <= 0:
        return np.ones((nx, nz), dtype=np.float64)

    mask = np.ones((nx, nz), dtype=np.float64)
    for i in range(n_boundary):
        w = (n_boundary - i) / n_boundary
        f = np.exp(-gamma_s * dt * w**2)

        mask[i, :] *= f
        mask[nx - 1 - i, :] *= f

        if not free_surface:
            mask[:, i] *= f

        mask[:, nz - 1 - i] *= f

    return mask


# ==============================================================================
# 5. STAGGERED DERIVATIVE OPERATORS
# ==============================================================================

def D_plus_x(f: np.ndarray, a: np.ndarray, dx: float) -> np.ndarray:
    """Forward staggered x-derivative: output at half-x nodes."""
    M = len(a)
    out = np.zeros_like(f)
    i0, i1 = M, f.shape[0] - M
    for m, am in enumerate(a, start=1):
        out[i0:i1] += am * (f[i0 + m : i1 + m] - f[i0 - m + 1 : i1 - m + 1])
    return out / dx


def D_minus_x(f: np.ndarray, a: np.ndarray, dx: float) -> np.ndarray:
    """Backward staggered x-derivative: output at integer-x nodes."""
    M = len(a)
    out = np.zeros_like(f)
    i0, i1 = M, f.shape[0] - M
    for m, am in enumerate(a, start=1):
        out[i0:i1] += am * (f[i0 + m - 1 : i1 + m - 1] - f[i0 - m : i1 - m])
    return out / dx


def D_plus_z(f: np.ndarray, a: np.ndarray, dz: float) -> np.ndarray:
    """Forward staggered z-derivative: output at half-z nodes."""
    M = len(a)
    out = np.zeros_like(f)
    j0, j1 = M, f.shape[1] - M
    for m, am in enumerate(a, start=1):
        out[:, j0:j1] += am * (
            f[:, j0 + m : j1 + m] - f[:, j0 - m + 1 : j1 - m + 1]
        )
    return out / dz


def D_minus_z(f: np.ndarray, a: np.ndarray, dz: float) -> np.ndarray:
    """Backward staggered z-derivative: output at integer-z nodes."""
    M = len(a)
    out = np.zeros_like(f)
    j0, j1 = M, f.shape[1] - M
    for m, am in enumerate(a, start=1):
        out[:, j0:j1] += am * (
            f[:, j0 + m - 1 : j1 + m - 1] - f[:, j0 - m : j1 - m]
        )
    return out / dz


# ==============================================================================
# 6. STATE / RESULT
# ==============================================================================

@dataclass
class ElasticState2D:
    """
    Wavefield state at a single leapfrog step.

    vx, vz live at the current velocity half-step time level.
    sxx, szz, sxz live at the current integer stress time level.
    """
    vx: np.ndarray
    vz: np.ndarray
    sxx: np.ndarray
    szz: np.ndarray
    sxz: np.ndarray


@dataclass
class ElasticRunResult:
    """
    Simulation output with explicit staggered time axes.
    """
    t_sigma: np.ndarray
    t_v: np.ndarray
    receiver_vx: np.ndarray
    receiver_vz: np.ndarray
    snapshots_vz: np.ndarray | None
    snapshot_times_v: np.ndarray | None

    @property
    def t(self) -> np.ndarray:
        """Backward-compatible alias for velocity time axis."""
        return self.t_v

    @property
    def snapshot_times_s(self) -> np.ndarray | None:
        """Backward-compatible alias for velocity snapshot times."""
        return self.snapshot_times_v


# ==============================================================================
# 7. MAIN SOLVER
# ==============================================================================

def run_elastic_solver_numpy(
    *,
    vp: np.ndarray,
    vs: np.ndarray,
    rho: np.ndarray,
    dx: float,
    dz: float,
    dt: float,
    nt: int,
    source_ix: int,
    source_iz: int,
    stf_xx: np.ndarray,
    stf_zz: np.ndarray,
    stf_xz: np.ndarray,
    receiver_sampling: ReceiverSampling2D,
    half_order: int = 2,
    use_ts_sfd: bool = False,
    n_boundary: int = 40,
    gamma_s: float = 300.0,
    snapshot_stride: int | None = None,
    free_surface: bool = False,
) -> ElasticRunResult:
    """
    2D isotropic elastic wave simulation on a staggered grid.

    The leapfrog scheme advances each step n = 0, ..., nt-1:

        [sigma^n, v^(n-1/2)]  →  v^(n+1/2)  →  sigma^(n+1)
    """
    vp = np.asarray(vp, dtype=np.float64)
    vs = np.asarray(vs, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    nx, nz = vp.shape

    if nt <= 1:
        raise ValueError(f"nt must be > 1, got {nt}.")
    if dx <= 0.0 or dz <= 0.0 or dt <= 0.0:
        raise ValueError(
            f"dx, dz, dt must be positive; got dx={dx}, dz={dz}, dt={dt}."
        )
    if vs.shape != (nx, nz) or rho.shape != (nx, nz):
        raise ValueError(
            f"vp, vs, rho must all be ({nx},{nz}); "
            f"got vs={vs.shape}, rho={rho.shape}."
        )
    if not np.isclose(dx, dz):
        raise ValueError(f"Solver requires dx == dz; got dx={dx}, dz={dz}.")

    stf_xx = np.asarray(stf_xx, dtype=np.float64)
    stf_zz = np.asarray(stf_zz, dtype=np.float64)
    stf_xz = np.asarray(stf_xz, dtype=np.float64)
    if stf_xx.shape != (nt,) or stf_zz.shape != (nt,) or stf_xz.shape != (nt,):
        raise ValueError(
            f"Source arrays must have shape ({nt},); "
            f"got {stf_xx.shape}, {stf_zz.shape}, {stf_xz.shape}."
        )

    if receiver_sampling.nrec <= 0:
        raise ValueError("receiver_sampling must contain at least one receiver.")
    if not (0 <= source_ix < nx and 0 <= source_iz < nz):
        raise ValueError(
            f"Source ({source_ix},{source_iz}) out of bounds "
            f"for grid ({nx},{nz})."
        )

    courant_rep = float(vp.mean()) * dt / dx
    a = fd_coefficients(half_order, use_ts_sfd=use_ts_sfd, courant=courant_rep)
    M = len(a)

    dt_max = max_stable_dt(
        float(vp.max()), dx, dz, half_order, safety=1.0, use_ts_sfd=use_ts_sfd
    )
    if dt > dt_max:
        raise ValueError(
            f"dt={dt:.4e} s exceeds CFL limit {dt_max:.4e} s "
            f"for spatial order {2*M}."
        )
    if n_boundary <= M:
        raise ValueError(
            f"n_boundary={n_boundary} must exceed half_order M={M}."
        )

    ix_min = n_boundary
    ix_max = nx - n_boundary
    iz_min = M + 1 if free_surface else n_boundary
    iz_max = nz - n_boundary

    if not (ix_min <= source_ix < ix_max and iz_min <= source_iz < iz_max):
        fs_note = (
            " When free_surface=True, moment-tensor sources must be at "
            f"iz >= M+1 = {M+1} to avoid violating σ_zz=0 at the free surface."
            if free_surface else ""
        )
        raise ValueError(
            f"Source ({source_ix},{source_iz}) is outside the valid interior. "
            f"Required: ix in [{ix_min}, {ix_max}), "
            f"iz in [{iz_min}, {iz_max})."
            + fs_note
        )

    mu = rho * vs**2
    lam = rho * (vp**2 - 2.0 * vs**2)

    if np.any(mu <= 0.0):
        raise ValueError(f"mu must be positive; min(mu)={mu.min():.3e}.")
    if np.any(lam < 0.0):
        raise ValueError(
            f"lambda < 0 (requires Vs <= Vp/sqrt(2)); "
            f"min(lambda)={lam.min():.3e}."
        )
    if np.any(lam + 2.0 * mu <= 0.0):
        raise ValueError(
            f"lambda+2mu must be positive; "
            f"min(lambda+2mu)={(lam + 2.0*mu).min():.3e}."
        )

    l2m = lam + 2.0 * mu

    bx, bz, mu_xz = prepare_staggered_materials(rho, mu)
    sponge = make_sponge_mask(
        nx, nz, n_boundary, gamma_s, dt, free_surface=free_surface
    )
    src_scale = dt / (dx * dz)

    t_sigma = np.arange(nt + 1, dtype=np.float64) * dt
    t_v = (np.arange(nt, dtype=np.float64) + 0.5) * dt

    state = ElasticState2D(
        vx=np.zeros((nx, nz), dtype=np.float64),
        vz=np.zeros((nx, nz), dtype=np.float64),
        sxx=np.zeros((nx, nz), dtype=np.float64),
        szz=np.zeros((nx, nz), dtype=np.float64),
        sxz=np.zeros((nx, nz), dtype=np.float64),
    )

    nrec = receiver_sampling.nrec
    rec_vx = np.zeros((nrec, nt), dtype=np.float64)
    rec_vz = np.zeros((nrec, nt), dtype=np.float64)

    if snapshot_stride is not None and snapshot_stride > 0:
        snap_indices = np.arange(0, nt, snapshot_stride, dtype=int)
        snaps_vz = np.zeros((snap_indices.size, nx, nz), dtype=np.float64)
        snaps_t_v = t_v[snap_indices]
        isnap = 0
    else:
        snap_indices = None
        snaps_vz = None
        snaps_t_v = None
        isnap = 0

    ix, iz = source_ix, source_iz

    for it in range(nt):
        # 0. Free-surface ghost-node velocity fill
        if free_surface:
            for m in range(1, M + 1):
                state.vx[:, M - m] = state.vx[:, M + m]
                state.vz[:, M - m] = state.vz[:, M + m - 1]

        # 1. Velocity update: sigma^n -> v^(n+1/2)
        state.vx += dt * bx * (
            D_plus_x(state.sxx, a, dx) + D_minus_z(state.sxz, a, dz)
        )
        state.vz += dt * bz * (
            D_minus_x(state.sxz, a, dx) + D_plus_z(state.szz, a, dz)
        )

        # 2. Stress update: v^(n+1/2) -> sigma^(n+1)
        dvx_dx = D_minus_x(state.vx, a, dx)
        dvz_dz = D_minus_z(state.vz, a, dz)

        if free_surface:
            dvz_dz[:, M] = -(lam[:, M] / l2m[:, M]) * dvx_dx[:, M]

        state.sxx += dt * (l2m * dvx_dx + lam * dvz_dz)
        state.szz += dt * (lam * dvx_dx + l2m * dvz_dz)
        state.sxz += dt * mu_xz * (
            D_plus_z(state.vx, a, dz) + D_plus_x(state.vz, a, dx)
        )

        if free_surface:
            state.szz[:, M] = 0.0
            for m in range(1, M + 1):
                state.szz[:, M - m] = -state.szz[:, M + m]
                state.sxx[:, M - m] = state.sxx[:, M + m]
                state.sxz[:, M - m] = -state.sxz[:, M + m - 1]

        # 3. Source injection into updated stress at t_sigma[it+1]
        state.sxx[ix, iz] += stf_xx[it] * src_scale
        state.szz[ix, iz] += stf_zz[it] * src_scale

        amp_xz = stf_xz[it] * src_scale * 0.25
        state.sxz[ix, iz] += amp_xz
        state.sxz[ix - 1, iz] += amp_xz
        state.sxz[ix, iz - 1] += amp_xz
        state.sxz[ix - 1, iz - 1] += amp_xz

        # 4. Sponge damping
        state.vx *= sponge
        state.vz *= sponge
        state.sxx *= sponge
        state.szz *= sponge
        state.sxz *= sponge

        # 5. Sample receivers at t_v[it]
        rec_vx[:, it], rec_vz[:, it] = sample_receivers(
            state.vx, state.vz, receiver_sampling
        )

        # 6. Optional vz snapshot at t_v[it]
        if (
            snap_indices is not None
            and isnap < len(snap_indices)
            and it == snap_indices[isnap]
        ):
            snaps_vz[isnap] = state.vz
            isnap += 1

    return ElasticRunResult(
        t_sigma=t_sigma,
        t_v=t_v,
        receiver_vx=rec_vx,
        receiver_vz=rec_vz,
        snapshots_vz=snaps_vz,
        snapshot_times_v=snaps_t_v,
    )