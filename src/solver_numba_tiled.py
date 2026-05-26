# ==============================================================================
# src/solver_numba_tiled.py — 2D staggered-grid elastic FD solver (Numba Tiled)
#
# Purpose
#   Cache-blocked Numba backend with tiled loop structure for improved
#   L1/L2 cache reuse compared to solver_numba_fused.py.
#
# Tiling
#   The spatial domain is partitioned into (tile_i × tile_j) blocks.
#   prange parallelises over tile indices in x. This avoids Numba's restriction
#   that prange must have unit step in the loop variable.
#
# Time staggering
#   Integer times (stress grid):  t_sigma[n] = n * dt,       n = 0, ..., nt
#   Half times   (velocity grid): t_v[n]     = (n+1/2) * dt, n = 0, ..., nt-1
#
#   Each time step n advances:
#       sigma^n, v^(n-1/2)  →  v^(n+1/2)  →  sigma^(n+1)
#
# Free surface
#   Mirrors solver_numpy.py: Robertsson (1996) + Graves (1996).
#   Moment-tensor sources must satisfy source_iz >= M+1.
#
# Important
#   This file must remain mathematically consistent with src/solver_numpy.py.
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


# ==============================================================================
# 1. LOW-LEVEL NUMBA KERNELS (TILED)
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
def update_velocity_tiled_numba(
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
    tile_i: int,
    tile_j: int,
) -> None:
    """
    Tiled fused velocity update:
        vx <- vx + dt * bx * (D_plus_x(sxx) + D_minus_z(sxz))
        vz <- vz + dt * bz * (D_minus_x(sxz) + D_plus_z(szz))

    prange parallelises over x-tiles. No shared writes between tiles.
    """
    nx, nz = vx.shape
    M = a.size

    i_start = M
    i_stop = nx - M
    j_start = M
    j_stop = nz - M

    n_tiles_i = (i_stop - i_start + tile_i - 1) // tile_i

    for ti in prange(n_tiles_i):
        ii = i_start + ti * tile_i
        i_end = min(ii + tile_i, i_stop)

        for jj in range(j_start, j_stop, tile_j):
            j_end = min(jj + tile_j, j_stop)

            for i in range(ii, i_end):
                for j in range(jj, j_end):
                    dsxx_dx = 0.0
                    dsxz_dz = 0.0
                    dsxz_dx = 0.0
                    dszz_dz = 0.0

                    for m in range(1, M + 1):
                        am = a[m - 1]
                        dsxx_dx += am * (sxx[i + m, j] - sxx[i - m + 1, j])
                        dsxz_dz += am * (sxz[i, j + m - 1] - sxz[i, j - m])
                        dsxz_dx += am * (sxz[i + m - 1, j] - sxz[i - m, j])
                        dszz_dz += am * (szz[i, j + m] - szz[i, j - m + 1])

                    dsxx_dx /= dx
                    dsxz_dz /= dz
                    dsxz_dx /= dx
                    dszz_dz /= dz

                    vx[i, j] += dt * bx[i, j] * (dsxx_dx + dsxz_dz)
                    vz[i, j] += dt * bz[i, j] * (dsxz_dx + dszz_dz)


@njit(parallel=True, fastmath=True, cache=True)
def update_stress_tiled_numba(
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
    tile_i: int,
    tile_j: int,
) -> None:
    """
    Tiled fused stress update (standard case, no free surface):
        sxx <- sxx + dt * (l2m * D_minus_x(vx) + lam * D_minus_z(vz))
        szz <- szz + dt * (lam * D_minus_x(vx) + l2m * D_minus_z(vz))
        sxz <- sxz + dt * mu_xz * (D_plus_z(vx) + D_plus_x(vz))
    """
    nx, nz = sxx.shape
    M = a.size

    i_start = M
    i_stop = nx - M
    j_start = M
    j_stop = nz - M

    n_tiles_i = (i_stop - i_start + tile_i - 1) // tile_i

    for ti in prange(n_tiles_i):
        ii = i_start + ti * tile_i
        i_end = min(ii + tile_i, i_stop)

        for jj in range(j_start, j_stop, tile_j):
            j_end = min(jj + tile_j, j_stop)

            for i in range(ii, i_end):
                for j in range(jj, j_end):
                    dvx_dx = 0.0
                    dvz_dz = 0.0
                    dvx_dz = 0.0
                    dvz_dx = 0.0

                    for m in range(1, M + 1):
                        am = a[m - 1]
                        dvx_dx += am * (vx[i + m - 1, j] - vx[i - m, j])
                        dvz_dz += am * (vz[i, j + m - 1] - vz[i, j - m])
                        dvx_dz += am * (vx[i, j + m] - vx[i, j - m + 1])
                        dvz_dx += am * (vz[i + m, j] - vz[i - m + 1, j])

                    dvx_dx /= dx
                    dvz_dz /= dz
                    dvx_dz /= dz
                    dvz_dx /= dx

                    sxx[i, j] += dt * (l2m[i, j] * dvx_dx + lam[i, j] * dvz_dz)
                    szz[i, j] += dt * (lam[i, j] * dvx_dx + l2m[i, j] * dvz_dz)
                    sxz[i, j] += dt * mu_xz[i, j] * (dvx_dz + dvz_dx)


@njit(parallel=True, fastmath=True, cache=True)
def update_stress_tiled_free_surface_numba(
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
    tile_i: int,
    tile_j: int,
) -> None:
    """
    Tiled fused stress update with Graves (1996) free-surface correction.

    At j == M, dvz_dz is overwritten before the stress accumulation:
        σ_zz = (λ+2μ) dvz_dz + λ dvx_dx = 0
        →  dvz_dz = -(λ/(λ+2μ)) dvx_dx
    """
    nx, nz = sxx.shape
    M = a.size

    i_start = M
    i_stop = nx - M
    j_start = M
    j_stop = nz - M

    n_tiles_i = (i_stop - i_start + tile_i - 1) // tile_i

    for ti in prange(n_tiles_i):
        ii = i_start + ti * tile_i
        i_end = min(ii + tile_i, i_stop)

        for jj in range(j_start, j_stop, tile_j):
            j_end = min(jj + tile_j, j_stop)

            for i in range(ii, i_end):
                for j in range(jj, j_end):
                    dvx_dx = 0.0
                    dvz_dz = 0.0
                    dvx_dz = 0.0
                    dvz_dx = 0.0

                    for m in range(1, M + 1):
                        am = a[m - 1]
                        dvx_dx += am * (vx[i + m - 1, j] - vx[i - m, j])
                        dvz_dz += am * (vz[i, j + m - 1] - vz[i, j - m])
                        dvx_dz += am * (vx[i, j + m] - vx[i, j - m + 1])
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

    szz at integer z → odd:  szz[:, M-m] = -szz[:, M+m]
    sxx at integer z → even: sxx[:, M-m] = +sxx[:, M+m]
    sxz at half-z   → odd with shift: sxz[:, M-m] = -sxz[:, M+m-1]
    """
    nx, _ = sxx.shape
    for i in prange(nx):
        szz[i, M] = 0.0
        for m in range(1, M + 1):
            szz[i, M - m] = -szz[i, M + m]
            sxx[i, M - m] = sxx[i, M + m]
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
    Source injection consistent with solver_numpy.py:
      - sxx, szz injected at (ix, iz)
      - sxz distributed equally over four surrounding nodes
    """
    sxx[ix, iz] += amp_xx
    szz[ix, iz] += amp_zz

    q = 0.25 * amp_xz
    sxz[ix, iz] += q
    sxz[ix - 1, iz] += q
    sxz[ix, iz - 1] += q
    sxz[ix - 1, iz - 1] += q


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
            vx[i, j] *= sp
            vz[i, j] *= sp
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
    Receiver extraction via precomputed bilinear interpolation weights
    on the separate vx and vz staggered grids.
    """
    nrec = ix_vx.size
    for k in range(nrec):
        i = ix_vx[k]
        j = iz_vx[k]
        rec_vx[k, it] = (
            w00_vx[k] * vx[i, j]
            + w10_vx[k] * vx[i + 1, j]
            + w01_vx[k] * vx[i, j + 1]
            + w11_vx[k] * vx[i + 1, j + 1]
        )

        i = ix_vz[k]
        j = iz_vz[k]
        rec_vz[k, it] = (
            w00_vz[k] * vz[i, j]
            + w10_vz[k] * vz[i + 1, j]
            + w01_vz[k] * vz[i, j + 1]
            + w11_vz[k] * vz[i + 1, j + 1]
        )


# ==============================================================================
# 2. MAIN SOLVER
# ==============================================================================

def run_elastic_solver_numba_tiled(
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
    tile_i: int = 32,
    tile_j: int = 32,
) -> ElasticRunResult:
    """
    Cache-blocked Numba elastic solver with tiled spatial loops.

    Mathematically identical to solver_numpy.py and solver_numba_fused.py.
    Tiling improves L1/L2 cache reuse for large grids.

    Time staggering
    ---------------
    Each step advances:
        [sigma^n, v^(n-1/2)] -> v^(n+1/2) -> sigma^(n+1)

    Source timing
    -------------
    stf_xx[it], stf_zz[it], stf_xz[it] injected at t_sigma[it+1].

    Free surface
    ------------
    When free_surface=True: Robertsson (1996) + Graves (1996), consistent
    with solver_numpy.py. Moment-tensor sources must satisfy
    source_iz >= M+1.

    Parameters
    ----------
    tile_i, tile_j : int
        Tile dimensions in x and z. Default 32×32 is a practical starting
        point; optimal values depend on cache size and half_order M.
    """
    # ── cast ───────────────────────────────────────────────────────────────────
    vp = np.asarray(vp, dtype=np.float64)
    vs = np.asarray(vs, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    nx, nz = vp.shape

    # ── validation ─────────────────────────────────────────────────────────────
    if nt <= 1:
        raise ValueError(f"nt must be > 1, got {nt}.")
    if dx <= 0.0 or dz <= 0.0 or dt <= 0.0:
        raise ValueError(
            f"dx, dz, dt must be positive; got dx={dx}, dz={dz}, dt={dt}."
        )
    if vs.shape != (nx, nz) or rho.shape != (nx, nz):
        raise ValueError(
            f"vp, vs, rho must all have shape ({nx},{nz}); "
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
            f"Source ({source_ix},{source_iz}) out of bounds for grid ({nx},{nz})."
        )

    if tile_i <= 0 or tile_j <= 0:
        raise ValueError(f"tile_i and tile_j must be positive, got {tile_i}, {tile_j}.")

    # ── FD coefficients + CFL ─────────────────────────────────────────────────
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

    # ── source position ────────────────────────────────────────────────────────
    ix_min = n_boundary
    ix_max = nx - n_boundary
    iz_min = M + 1 if free_surface else n_boundary
    iz_max = nz - n_boundary

    if not (ix_min <= source_ix < ix_max and iz_min <= source_iz < iz_max):
        fs_note = (
            f" When free_surface=True, source must satisfy iz >= M+1 = {M+1}."
            if free_surface else ""
        )
        raise ValueError(
            f"Source ({source_ix},{source_iz}) outside valid interior. "
            f"Required: ix in [{ix_min},{ix_max}), iz in [{iz_min},{iz_max})."
            + fs_note
        )

    # ── elastic moduli ─────────────────────────────────────────────────────────
    mu = rho * vs**2
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

    # ── precomputed fields ─────────────────────────────────────────────────────
    bx, bz, mu_xz = prepare_staggered_materials(rho, mu)
    sponge = make_sponge_mask(
        nx, nz, n_boundary, gamma_s, dt, free_surface=free_surface
    )
    src_scale = dt / (dx * dz)

    # ── time axes ──────────────────────────────────────────────────────────────
    t_sigma = np.arange(nt + 1, dtype=np.float64) * dt
    t_v = (np.arange(nt, dtype=np.float64) + 0.5) * dt

    # ── wavefield arrays ───────────────────────────────────────────────────────
    vx = np.zeros((nx, nz), dtype=np.float64)
    vz = np.zeros((nx, nz), dtype=np.float64)
    sxx = np.zeros((nx, nz), dtype=np.float64)
    szz = np.zeros((nx, nz), dtype=np.float64)
    sxz = np.zeros((nx, nz), dtype=np.float64)

    # ── receiver gathers ───────────────────────────────────────────────────────
    nrec = receiver_sampling.nrec
    rec_vx = np.zeros((nrec, nt), dtype=np.float64)
    rec_vz = np.zeros((nrec, nt), dtype=np.float64)

    # guard against silent OOB in bilinear kernel
    rs = receiver_sampling
    if rs.vx.ix.max() + 1 >= nx or rs.vx.iz.max() + 1 >= nz:
        raise ValueError("vx receiver indices out of bounds for this grid.")
    if rs.vz.ix.max() + 1 >= nx or rs.vz.iz.max() + 1 >= nz:
        raise ValueError("vz receiver indices out of bounds for this grid.")

    # ── snapshots ──────────────────────────────────────────────────────────────
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

    ix, iz = int(source_ix), int(source_iz)

    # ── leapfrog time loop ─────────────────────────────────────────────────────
    for it in range(nt):
        # 0. Free surface: velocity ghost fill
        if free_surface:
            fill_velocity_ghosts_free_surface_numba(vx, vz, M)

        # 1. Velocity update: sigma^n -> v^(n+1/2)
        update_velocity_tiled_numba(
            vx, vz, sxx, szz, sxz, bx, bz, dx, dz, dt, a, tile_i, tile_j
        )

        # 2. Stress update: v^(n+1/2) -> sigma^(n+1)
        if free_surface:
            update_stress_tiled_free_surface_numba(
                vx, vz, sxx, szz, sxz, lam, l2m, mu_xz,
                dx, dz, dt, a, tile_i, tile_j
            )
            mirror_stress_ghosts_free_surface_numba(sxx, szz, sxz, M)
        else:
            update_stress_tiled_numba(
                vx, vz, sxx, szz, sxz, lam, l2m, mu_xz,
                dx, dz, dt, a, tile_i, tile_j
            )

        # 3. Source injection at t_sigma[it+1]
        inject_source_numba(
            sxx, szz, sxz, ix, iz,
            stf_xx[it] * src_scale,
            stf_zz[it] * src_scale,
            stf_xz[it] * src_scale,
        )

        # 4. Sponge damping
        apply_sponge_numba(vx, vz, sxx, szz, sxz, sponge)

        # 5. Sample receivers at t_v[it]
        sample_receivers_numba_bilinear(
            vx, vz,
            rs.vx.ix, rs.vx.iz, rs.vx.w00, rs.vx.w10, rs.vx.w01, rs.vx.w11,
            rs.vz.ix, rs.vz.iz, rs.vz.w00, rs.vz.w10, rs.vz.w01, rs.vz.w11,
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