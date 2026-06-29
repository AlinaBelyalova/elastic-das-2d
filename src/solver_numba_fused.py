# ==============================================================================
# src/solver_numba_fused.py — 2D staggered-grid elastic FD solver (Numba Fused)
#
# Purpose
#   Numba-accelerated backend mirroring the NumPy baseline solver.
#
# Time staggering
#   Integer times (stress grid):  t_sigma[n] = n * dt,       n = 0, ..., nt
#   Half times   (velocity grid): t_v[n]     = (n+1/2) * dt, n = 0, ..., nt-1
#
#   Each time step n advances:
#       sigma^n, v^(n-1/2)  →  v^(n+1/2)  →  sigma^(n+1)
#
# Current scope
#   - Explicit fused Numba loops for velocity/stress updates
#   - Source injection on the updated stress field at t_sigma[n+1]
#   - Sponge damping
#   - Receiver sampling with precomputed bilinear weights
#   - API aligned with src/solver_numpy.py
#   - free_surface=True supported
#
# Important
#   This file should remain mathematically consistent with src/solver_numpy.py.
# ==============================================================================

from __future__ import annotations

import numpy as np
from numba import njit, prange

from src.solver_numpy import (
    fd_coefficients,
    max_stable_dt,
    prepare_staggered_materials,
    make_sponge_mask,
    ElasticRunResult,
)
from src.sampling import ReceiverSampling2D
from src.source_injection import StressSourceInjection


# ==============================================================================
# 1. LOW-LEVEL NUMBA KERNELS
# ==============================================================================

@njit(parallel=True, fastmath=True, cache=True)
def fill_velocity_ghosts_free_surface_numba(
    vx: np.ndarray,
    vz: np.ndarray,
    M: int,
) -> None:
    """
    Free-surface ghost-node velocity fill (Robertsson 1996).

    vx at integer z-nodes → even symmetry about iz=M:
        vx[:, M-m] = +vx[:, M+m]

    vz at half-integer z-nodes → even symmetry with half-node shift:
        vz[:, M-m] = +vz[:, M+m-1]
    """
    nx, _ = vx.shape
    for i in prange(nx):
        for m in range(1, M + 1):
            vx[i, M - m] = vx[i, M + m]
            vz[i, M - m] = vz[i, M + m - 1]


@njit(parallel=True, fastmath=True, cache=True)
def update_velocity_fused_numba(
    vx: np.ndarray,
    vz: np.ndarray,
    sxx: np.ndarray,
    szz: np.ndarray,
    sxz: np.ndarray,
    bx: np.ndarray,
    bz: np.ndarray,
    dx: float,
    dz: float,
    dt: float,
    a: np.ndarray,
) -> None:
    """
    Fused staggered-grid velocity update:
        vx <- vx + dt * bx * (D_plus_x(sxx) + D_minus_z(sxz))
        vz <- vz + dt * bz * (D_minus_x(sxz) + D_plus_z(szz))

    Updates are done in-place.
    """
    nx, nz = vx.shape
    M = a.size

    for i in prange(M, nx - M):
        for j in range(M, nz - M):
            dsxx_dx = 0.0
            dsxz_dz = 0.0
            dsxz_dx = 0.0
            dszz_dz = 0.0

            for m in range(1, M + 1):
                am = a[m - 1]

                # D_plus_x(sxx)
                dsxx_dx += am * (sxx[i + m, j] - sxx[i - m + 1, j])

                # D_minus_z(sxz)
                dsxz_dz += am * (sxz[i, j + m - 1] - sxz[i, j - m])

                # D_minus_x(sxz)
                dsxz_dx += am * (sxz[i + m - 1, j] - sxz[i - m, j])

                # D_plus_z(szz)
                dszz_dz += am * (szz[i, j + m] - szz[i, j - m + 1])

            dsxx_dx /= dx
            dsxz_dz /= dz
            dsxz_dx /= dx
            dszz_dz /= dz

            vx[i, j] += dt * bx[i, j] * (dsxx_dx + dsxz_dz)
            vz[i, j] += dt * bz[i, j] * (dsxz_dx + dszz_dz)


@njit(parallel=True, fastmath=True, cache=True)
def update_stress_fused_numba(
    vx: np.ndarray,
    vz: np.ndarray,
    sxx: np.ndarray,
    szz: np.ndarray,
    sxz: np.ndarray,
    lam: np.ndarray,
    l2m: np.ndarray,
    mu_xz: np.ndarray,
    dx: float,
    dz: float,
    dt: float,
    a: np.ndarray,
) -> None:
    """
    Fused staggered-grid stress update (standard case):
        sxx <- sxx + dt * (l2m * D_minus_x(vx) + lam * D_minus_z(vz))
        szz <- szz + dt * (lam * D_minus_x(vx) + l2m * D_minus_z(vz))
        sxz <- sxz + dt * mu_xz * (D_plus_z(vx) + D_plus_x(vz))
    """
    nx, nz = sxx.shape
    M = a.size

    for i in prange(M, nx - M):
        for j in range(M, nz - M):
            dvx_dx = 0.0
            dvz_dz = 0.0
            dvx_dz = 0.0
            dvz_dx = 0.0

            for m in range(1, M + 1):
                am = a[m - 1]

                # D_minus_x(vx)
                dvx_dx += am * (vx[i + m - 1, j] - vx[i - m, j])

                # D_minus_z(vz)
                dvz_dz += am * (vz[i, j + m - 1] - vz[i, j - m])

                # D_plus_z(vx)
                dvx_dz += am * (vx[i, j + m] - vx[i, j - m + 1])

                # D_plus_x(vz)
                dvz_dx += am * (vz[i + m, j] - vz[i - m + 1, j])

            dvx_dx /= dx
            dvz_dz /= dz
            dvx_dz /= dz
            dvz_dx /= dx

            sxx[i, j] += dt * (l2m[i, j] * dvx_dx + lam[i, j] * dvz_dz)
            szz[i, j] += dt * (lam[i, j] * dvx_dx + l2m[i, j] * dvz_dz)
            sxz[i, j] += dt * mu_xz[i, j] * (dvx_dz + dvz_dx)


@njit(parallel=True, fastmath=True, cache=True)
def update_stress_fused_free_surface_numba(
    vx: np.ndarray,
    vz: np.ndarray,
    sxx: np.ndarray,
    szz: np.ndarray,
    sxz: np.ndarray,
    lam: np.ndarray,
    l2m: np.ndarray,
    mu_xz: np.ndarray,
    dx: float,
    dz: float,
    dt: float,
    a: np.ndarray,
) -> None:
    """
    Fused staggered-grid stress update with free-surface correction.

    At j == M, enforce analytically:
        sigma_zz = (lambda + 2mu) dvz_dz + lambda dvx_dx = 0
    i.e.
        dvz_dz = -(lambda / (lambda + 2mu)) * dvx_dx
    """
    nx, nz = sxx.shape
    M = a.size

    for i in prange(M, nx - M):
        for j in range(M, nz - M):
            dvx_dx = 0.0
            dvz_dz = 0.0
            dvx_dz = 0.0
            dvz_dx = 0.0

            for m in range(1, M + 1):
                am = a[m - 1]

                # D_minus_x(vx)
                dvx_dx += am * (vx[i + m - 1, j] - vx[i - m, j])

                # D_minus_z(vz)
                dvz_dz += am * (vz[i, j + m - 1] - vz[i, j - m])

                # D_plus_z(vx)
                dvx_dz += am * (vx[i, j + m] - vx[i, j - m + 1])

                # D_plus_x(vz)
                dvz_dx += am * (vz[i + m, j] - vz[i - m + 1, j])

            dvx_dx /= dx
            dvz_dz /= dz
            dvx_dz /= dz
            dvz_dx /= dx

            if j == M:
                dvz_dz = -(lam[i, j] / l2m[i, j]) * dvx_dx

            sxx[i, j] += dt * (l2m[i, j] * dvx_dx + lam[i, j] * dvz_dz)
            szz[i, j] += dt * (lam[i, j] * dvx_dx + l2m[i, j] * dvz_dz)
            sxz[i, j] += dt * mu_xz[i, j] * (dvx_dz + dvz_dx)


@njit(parallel=True, fastmath=True, cache=True)
def mirror_stress_ghosts_free_surface_numba(
    sxx: np.ndarray,
    szz: np.ndarray,
    sxz: np.ndarray,
    M: int,
) -> None:
    """
    Free-surface stress ghost-node mirroring (Robertsson 1996).

    szz at integer z → odd symmetry:
        szz[:, M-m] = -szz[:, M+m]

    sxx at integer z → even symmetry:
        sxx[:, M-m] = +sxx[:, M+m]

    sxz at half-integer z → odd symmetry with half-node shift:
        sxz[:, M-m] = -sxz[:, M+m-1]
    """
    nx, _ = sxx.shape
    for i in prange(nx):
        szz[i, M] = 0.0
        for m in range(1, M + 1):
            szz[i, M - m] = -szz[i, M + m]
            sxx[i, M - m] =  sxx[i, M + m]
            sxz[i, M - m] = -sxz[i, M + m - 1]


@njit(fastmath=True, cache=True)
def inject_source_numba(
    sxx: np.ndarray,
    szz: np.ndarray,
    sxz: np.ndarray,
    ix: int,
    iz: int,
    amp_xx: float,
    amp_zz: float,
    amp_xz: float,
) -> None:
    """
    Legacy nearest-node source injection.

    Kept for reference only. The production solver uses inject_stress_source_numba,
    which handles both "nearest" and "bilinear" spreading via StressSourceInjection
    arrays. Do not call this function from the main time loop.
    """
    sxx[ix, iz] += amp_xx
    szz[ix, iz] += amp_zz

    q = 0.25 * amp_xz
    sxz[ix,     iz    ] += q
    sxz[ix - 1, iz    ] += q
    sxz[ix,     iz - 1] += q
    sxz[ix - 1, iz - 1] += q


@njit(fastmath=True, cache=True)
def inject_stress_source_numba(
    sxx: np.ndarray,
    szz: np.ndarray,
    sxz: np.ndarray,
    normal_ix: np.ndarray,
    normal_iz: np.ndarray,
    normal_w:  np.ndarray,
    shear_ix:  np.ndarray,
    shear_iz:  np.ndarray,
    shear_w:   np.ndarray,
    amp_xx: float,
    amp_zz: float,
    amp_xz: float,
) -> None:
    """
    Unified stress source injection kernel.

    sxx and szz use the normal-stress stencil (integer grid).
    sxz uses the shear-stress stencil (half-integer grid).
    Each stencil has 4 nodes with weights summing to 1.0.

    Works for both spreading modes without branching:
      spreading="nearest"  → normal_w=[1,0,0,0], shear_w=[0.25,...] (equal centroid)
      spreading="bilinear" → arbitrary bilinear weights from physical position

    Weights are encoded in StressSourceInjection; no division by 4 here.
    """
    for k in range(4):
        sxx[normal_ix[k], normal_iz[k]] += amp_xx * normal_w[k]
        szz[normal_ix[k], normal_iz[k]] += amp_zz * normal_w[k]
        sxz[shear_ix[k],  shear_iz[k] ] += amp_xz * shear_w[k]


@njit(parallel=True, fastmath=True, cache=True)
def apply_sponge_numba(
    vx: np.ndarray,
    vz: np.ndarray,
    sxx: np.ndarray,
    szz: np.ndarray,
    sxz: np.ndarray,
    sponge: np.ndarray,
) -> None:
    """Apply multiplicative sponge damping in-place."""
    nx, nz = vx.shape
    for i in prange(nx):
        for j in range(nz):
            sp = sponge[i, j]
            vx[i, j]  *= sp
            vz[i, j]  *= sp
            sxx[i, j] *= sp
            szz[i, j] *= sp
            sxz[i, j] *= sp


@njit(fastmath=True, cache=True)
def sample_receivers_numba_bilinear(
    vx: np.ndarray,
    vz: np.ndarray,
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
    rec_vx: np.ndarray,
    rec_vz: np.ndarray,
    it: int,
) -> None:
    """
    Extract receiver velocities using precomputed bilinear interpolation
    weights for the staggered vx and vz grids separately.
    """
    nrec = ix_vx.size

    for k in range(nrec):
        i = ix_vx[k]
        j = iz_vx[k]
        rec_vx[k, it] = (
            w00_vx[k] * vx[i,     j    ] +
            w10_vx[k] * vx[i + 1, j    ] +
            w01_vx[k] * vx[i,     j + 1] +
            w11_vx[k] * vx[i + 1, j + 1]
        )

        i = ix_vz[k]
        j = iz_vz[k]
        rec_vz[k, it] = (
            w00_vz[k] * vz[i,     j    ] +
            w10_vz[k] * vz[i + 1, j    ] +
            w01_vz[k] * vz[i,     j + 1] +
            w11_vz[k] * vz[i + 1, j + 1]
        )


# ==============================================================================
# 2. MAIN SOLVER
# ==============================================================================

def run_elastic_solver_numba_fused(
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
    source_injection: StressSourceInjection | None = None,
) -> ElasticRunResult:
    """
    Numba-accelerated elastic solver with fused explicit loops.

    Time staggering
    ---------------
    Each step advances:
        [sigma^n, v^(n-1/2)] -> v^(n+1/2) -> sigma^(n+1)

    Source timing
    -------------
    stf_xx[it], stf_zz[it], stf_xz[it] are injected into the updated stress
    field at t_sigma[it+1].

    free_surface
    ------------
    Mirrors the NumPy baseline logic:
      - top sponge disabled
      - velocity ghost fill before velocity update
      - Graves correction at j=M
      - stress ghost mirroring after stress update

    source_injection
    ----------------
    Pre-built StressSourceInjection containing plain numpy arrays (shape (4,))
    for each stress component. Both "nearest" and "bilinear" spreading modes
    produce the same array format; the same injection kernel handles both.
    If None: legacy fallback arrays are built from source_ix/source_iz.
    """
    vp  = np.asarray(vp,  dtype=np.float64)
    vs  = np.asarray(vs,  dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    nx, nz = vp.shape

    if nt <= 1:
        raise ValueError(f"nt must be > 1, got {nt}.")
    if dx <= 0.0 or dz <= 0.0 or dt <= 0.0:
        raise ValueError(
            f"dx, dz, dt must be positive, got dx={dx}, dz={dz}, dt={dt}."
        )
    if vs.shape != (nx, nz) or rho.shape != (nx, nz):
        raise ValueError(
            f"vp, vs, rho must all have shape ({nx}, {nz}); "
            f"got vs={vs.shape}, rho={rho.shape}."
        )
    if not np.isclose(dx, dz):
        raise ValueError(f"Solver requires dx == dz; got dx={dx}, dz={dz}.")

    stf_xx = np.asarray(stf_xx, dtype=np.float64)
    stf_zz = np.asarray(stf_zz, dtype=np.float64)
    stf_xz = np.asarray(stf_xz, dtype=np.float64)
    if stf_xx.shape != (nt,) or stf_zz.shape != (nt,) or stf_xz.shape != (nt,):
        raise ValueError(
            f"stf_xx, stf_zz, stf_xz must all have shape ({nt},); "
            f"got {stf_xx.shape}, {stf_zz.shape}, {stf_xz.shape}."
        )

    if receiver_sampling.nrec <= 0:
        raise ValueError("receiver_sampling must contain at least one receiver.")
    if not (0 <= source_ix < nx and 0 <= source_iz < nz):
        raise ValueError(
            f"Source ({source_ix},{source_iz}) out of bounds for grid ({nx},{nz})."
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
        raise ValueError(f"n_boundary={n_boundary} must exceed half_order M={M}.")

    ix_min = n_boundary
    ix_max = nx - n_boundary
    iz_min = M + 1 if free_surface else n_boundary
    iz_max = nz - n_boundary

    if not (ix_min <= source_ix < ix_max and iz_min <= source_iz < iz_max):
        fs_note = (
            f" When free_surface=True, moment-tensor sources must satisfy "
            f"iz >= M+1 = {M+1} to avoid violating σ_zz=0 at the free surface."
            if free_surface else ""
        )
        raise ValueError(
            f"Source ({source_ix},{source_iz}) is outside the valid interior. "
            f"Required: ix in [{ix_min},{ix_max}), iz in [{iz_min},{iz_max})."
            + fs_note
        )

    mu  = rho * vs**2
    lam = rho * (vp**2 - 2.0 * vs**2)
    l2m = lam + 2.0 * mu

    if np.any(mu <= 0.0):
        raise ValueError(f"mu must be positive; min(mu)={mu.min():.3e}.")
    if np.any(lam < 0.0):
        raise ValueError(
            f"lambda < 0 (requires Vs <= Vp/sqrt(2)); min(lambda)={lam.min():.3e}."
        )
    if np.any(l2m <= 0.0):
        raise ValueError(
            f"lambda+2mu must be positive; min(lambda+2mu)={l2m.min():.3e}."
        )

    bx, bz, mu_xz = prepare_staggered_materials(rho, mu)
    sponge = make_sponge_mask(
        nx, nz, n_boundary, gamma_s, dt, free_surface=free_surface
    )
    src_scale = dt / (dx * dz)

    # ── time axes ──────────────────────────────────────────────────────────────
    t_sigma = np.arange(nt + 1, dtype=np.float64) * dt
    t_v     = (np.arange(nt,    dtype=np.float64) + 0.5) * dt

    # ── wavefield arrays ───────────────────────────────────────────────────────
    vx  = np.zeros((nx, nz), dtype=np.float64)
    vz  = np.zeros((nx, nz), dtype=np.float64)
    sxx = np.zeros((nx, nz), dtype=np.float64)
    szz = np.zeros((nx, nz), dtype=np.float64)
    sxz = np.zeros((nx, nz), dtype=np.float64)

    # ── receiver gathers ───────────────────────────────────────────────────────
    nrec   = receiver_sampling.nrec
    rec_vx = np.zeros((nrec, nt), dtype=np.float64)
    rec_vz = np.zeros((nrec, nt), dtype=np.float64)

    # Guard against silent out-of-bounds in bilinear kernel
    rs = receiver_sampling
    if rs.vx.ix.max() + 1 >= nx or rs.vx.iz.max() + 1 >= nz:
        raise ValueError("vx receiver indices out of bounds for this grid.")
    if rs.vz.ix.max() + 1 >= nx or rs.vz.iz.max() + 1 >= nz:
        raise ValueError("vz receiver indices out of bounds for this grid.")

    # ── snapshots ──────────────────────────────────────────────────────────────
    if snapshot_stride is not None and snapshot_stride > 0:
        snap_indices = np.arange(0, nt, snapshot_stride, dtype=int)
        snaps_vz     = np.zeros((snap_indices.size, nx, nz), dtype=np.float64)
        snaps_t_v    = t_v[snap_indices]
        isnap        = 0
    else:
        snap_indices = None
        snaps_vz     = None
        snaps_t_v    = None
        isnap        = 0

    ix = int(source_ix)
    iz = int(source_iz)

    # ── source injection arrays ────────────────────────────────────────────────
    # source_injection holds plain numpy arrays (int64/float64, shape (4,)).
    # If not provided, build the nearest-node equivalent on the fly:
    #   normal stencil → w=[1,0,0,0] (point injection at ix, iz)
    #   shear  stencil → w=[0.25,...] (centroid correction)
    # Both cases use the same inject_stress_source_numba kernel.
    if source_injection is not None:
        _normal_ix = source_injection.normal_ix
        _normal_iz = source_injection.normal_iz
        _normal_w  = source_injection.normal_w
        _shear_ix  = source_injection.shear_ix
        _shear_iz  = source_injection.shear_iz
        _shear_w   = source_injection.shear_w
    else:
        # Legacy fallback: nearest-node equivalent via hard-coded arrays
        _normal_ix = np.array([ix,     ix + 1, ix,     ix + 1], dtype=np.int64)
        _normal_iz = np.array([iz,     iz,     iz + 1, iz + 1], dtype=np.int64)
        _normal_w  = np.array([1.0,    0.0,    0.0,    0.0   ], dtype=np.float64)
        _shear_ix  = np.array([ix - 1, ix,     ix - 1, ix    ], dtype=np.int64)
        _shear_iz  = np.array([iz - 1, iz - 1, iz,     iz    ], dtype=np.int64)
        _shear_w   = np.array([0.25,   0.25,   0.25,   0.25  ], dtype=np.float64)

    # ── leapfrog time loop ─────────────────────────────────────────────────────
    for it in range(nt):
        # 0. Free-surface ghost-node velocity fill
        if free_surface:
            fill_velocity_ghosts_free_surface_numba(vx, vz, M)

        # 1. Velocity update: sigma^n -> v^(n+1/2)
        update_velocity_fused_numba(
            vx, vz, sxx, szz, sxz, bx, bz, dx, dz, dt, a
        )

        # 2. Stress update: v^(n+1/2) -> sigma^(n+1)
        if free_surface:
            update_stress_fused_free_surface_numba(
                vx, vz, sxx, szz, sxz, lam, l2m, mu_xz, dx, dz, dt, a
            )
            mirror_stress_ghosts_free_surface_numba(sxx, szz, sxz, M)
        else:
            update_stress_fused_numba(
                vx, vz, sxx, szz, sxz, lam, l2m, mu_xz, dx, dz, dt, a
            )

        # 3. Source injection into updated stress at t_sigma[it+1]
        inject_stress_source_numba(
            sxx, szz, sxz,
            _normal_ix, _normal_iz, _normal_w,
            _shear_ix,  _shear_iz,  _shear_w,
            stf_xx[it] * src_scale,
            stf_zz[it] * src_scale,
            stf_xz[it] * src_scale,
        )

        # 4. Sponge damping
        apply_sponge_numba(vx, vz, sxx, szz, sxz, sponge)

        # 5. Sample receivers at t_v[it]
        sample_receivers_numba_bilinear(
            vx, vz,
            rs.vx.ix, rs.vx.iz,
            rs.vx.w00, rs.vx.w10, rs.vx.w01, rs.vx.w11,
            rs.vz.ix, rs.vz.iz,
            rs.vz.w00, rs.vz.w10, rs.vz.w01, rs.vz.w11,
            rec_vx, rec_vz, it,
        )

        # 6. Optional vz snapshot at t_v[it]
        if (
            snap_indices is not None
            and isnap < len(snap_indices)
            and it == snap_indices[isnap]
        ):
            snaps_vz[isnap] = vz
            isnap += 1

    return ElasticRunResult(
        t_sigma=t_sigma,
        t_v=t_v,
        receiver_vx=rec_vx,
        receiver_vz=rec_vz,
        snapshots_vz=snaps_vz,
        snapshot_times_v=snaps_t_v,
    )