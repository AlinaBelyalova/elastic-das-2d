# ==============================================================================
# src/solver_numpy_pointforce.py — 2D elastic FD solver with vertical point force
#
# Purpose
#   Dedicated validation solver for analytical comparison in a homogeneous 2D
#   elastic medium using a vertical point-force source.
#
# Important
#   This is a separate validation path.
#   It does NOT replace the main double-couple / stress-injection workflow.
#
# Time staggering
#   Integer times (stress grid):  t_sigma[n] = n * dt,       n = 0, 1, ..., nt
#   Half times   (velocity grid): t_v[n]     = (n+1/2) * dt, n = 0, 1, ..., nt-1
#
#   Each time step n advances:
#       sigma^n, v^(n-1/2)  →  v^(n+1/2)  →  sigma^(n+1)
#
# Source convention
#   The source is a vertical body force injected into the vz equation:
#
#       rho dvz/dt = ... + fz
#
#   In discrete form, a 2D point force F_z [N/m] is distributed over a single
#   cell of area dx*dz to give a body force density [N/m^3]:
#
#       vz[ix, iz] += dt * bz[ix, iz] * force_stf_half[n] / (dx * dz)
#
#   force_stf_half[n] is the force amplitude at the velocity half-step:
#
#       t_v[n] = (n + 1/2) * dt
#
# Staggered-grid geometry
#   vz[i,j] lives at (x0 + i*dx,  z0 + (j + 0.5)*dz).
#   For analytical comparison, source and receiver coordinates must be
#   interpreted on the vz staggered grid.
#
# Supported spatial orders
#   half_order = 1, 2, 3, 4  →  spatial FD orders 2, 4, 6, 8.
# ==============================================================================

from __future__ import annotations

import numpy as np

from src.sampling import ReceiverSampling2D, sample_receivers
from src.solver_numpy import (
    fd_coefficients,
    max_stable_dt,
    prepare_staggered_materials,
    make_sponge_mask,
    D_plus_x,
    D_minus_x,
    D_plus_z,
    D_minus_z,
    ElasticState2D,
    ElasticRunResult,
)


def run_elastic_solver_numpy_pointforce(
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
    force_stf_half: np.ndarray,
    receiver_sampling: ReceiverSampling2D,
    half_order: int = 2,
    use_ts_sfd: bool = False,
    n_boundary: int = 40,
    gamma_s: float = 300.0,
    snapshot_stride: int | None = None,
) -> ElasticRunResult:
    """
    2D isotropic elastic wave simulation with a vertical point-force source.

    The source is injected directly into the vz velocity equation at each
    half-step, consistent with the leapfrog time staggering. This is the
    correct placement for a body-force source and enables clean comparison
    against the 2D analytical Green's function.

    Parameters
    ----------
    vp, vs, rho : shape (nx, nz)
        Medium properties on the integer grid [m/s, m/s, kg/m^3].
    dx, dz : float
        Grid spacing [m]. Must satisfy dx == dz.
    dt : float
        Time step [s]. Must satisfy the CFL condition.
    nt : int
        Number of time steps.
    source_ix, source_iz : int
        Source indices on the vz staggered grid.
        vz[i,j] is located at (x0 + i*dx, z0 + (j+0.5)*dz).
    force_stf_half : shape (nt,)
        Vertical force amplitude sampled at t_v[n] = (n+1/2)*dt [N/m].
    receiver_sampling : ReceiverSampling2D
        Precomputed staggered bilinear interpolation metadata.
    half_order : int
        Spatial half-order M in {1,2,3,4} (spatial order = 2M).
    use_ts_sfd : bool
        Use TS-SFD dispersion-optimised coefficients instead of Taylor.
    n_boundary : int
        Sponge layer width in grid cells. Must exceed half_order M.
    gamma_s : float
        Sponge damping coefficient [1/s].
    snapshot_stride : int or None
        Save a vz snapshot every `snapshot_stride` steps.

    Returns
    -------
    ElasticRunResult
        Explicit staggered time axes, receiver gathers at t_v, optional
        vz snapshots.
    """
    # ── cast inputs ────────────────────────────────────────────────────────────
    vp             = np.asarray(vp,             dtype=np.float64)
    vs             = np.asarray(vs,             dtype=np.float64)
    rho            = np.asarray(rho,            dtype=np.float64)
    force_stf_half = np.asarray(force_stf_half, dtype=np.float64)

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
    if force_stf_half.shape != (nt,):
        raise ValueError(
            f"force_stf_half must have shape ({nt},), got {force_stf_half.shape}."
        )
    if receiver_sampling.nrec <= 0:
        raise ValueError("receiver_sampling must contain at least one receiver.")
    if not (0 <= source_ix < nx and 0 <= source_iz < nz):
        raise ValueError(
            f"Source ({source_ix},{source_iz}) out of bounds for grid ({nx},{nz})."
        )

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

    # ── source position validation ─────────────────────────────────────────────
    ix_min = n_boundary
    ix_max = nx - n_boundary
    iz_min = n_boundary
    iz_max = nz - n_boundary

    if not (ix_min <= source_ix < ix_max and iz_min <= source_iz < iz_max):
        raise ValueError(
            f"Point-force source ({source_ix},{source_iz}) outside valid interior. "
            f"Required: ix in [{ix_min},{ix_max}), iz in [{iz_min},{iz_max}). "
            f"(n_boundary={n_boundary}, M={M})"
        )

    # ── elastic moduli ─────────────────────────────────────────────────────────
    mu  = rho * vs**2
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

    # ── precomputed fields ─────────────────────────────────────────────────────
    bx, bz, mu_xz = prepare_staggered_materials(rho, mu)
    sponge = make_sponge_mask(nx, nz, n_boundary, gamma_s, dt, free_surface=False)

    # ── time axes ──────────────────────────────────────────────────────────────
    t_sigma = np.arange(nt + 1, dtype=np.float64) * dt
    t_v     = (np.arange(nt,    dtype=np.float64) + 0.5) * dt

    # ── initial state: sigma^0 = 0,  v^(-1/2) = 0 ─────────────────────────────
    state = ElasticState2D(
        vx  = np.zeros((nx, nz), dtype=np.float64),
        vz  = np.zeros((nx, nz), dtype=np.float64),
        sxx = np.zeros((nx, nz), dtype=np.float64),
        szz = np.zeros((nx, nz), dtype=np.float64),
        sxz = np.zeros((nx, nz), dtype=np.float64),
    )

    # ── output arrays ─────────────────────────────────────────────────────────
    nrec   = receiver_sampling.nrec
    rec_vx = np.zeros((nrec, nt), dtype=np.float64)
    rec_vz = np.zeros((nrec, nt), dtype=np.float64)

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

    ix, iz = int(source_ix), int(source_iz)

    # ── leapfrog time loop ─────────────────────────────────────────────────────
    for it in range(nt):

        # 1. Velocity update: sigma^n → v^(n+1/2)
        state.vx += dt * bx * (
            D_plus_x(state.sxx, a, dx) + D_minus_z(state.sxz, a, dz)
        )
        state.vz += dt * bz * (
            D_minus_x(state.sxz, a, dx) + D_plus_z(state.szz, a, dz)
        )

        # 2. Vertical point-force injection at t_v[it]
        #    F_z [N/m] distributed over cell area dx*dz → body force [N/m^3]
        #    dvz = dt * bz * F_z / (dx*dz),  with bz = 1/rho at vz nodes
        state.vz[ix, iz] += dt * bz[ix, iz] * force_stf_half[it] / (dx * dz)

        # 3. Stress update: v^(n+1/2) → sigma^(n+1)
        dvx_dx = D_minus_x(state.vx, a, dx)
        dvz_dz = D_minus_z(state.vz, a, dz)

        state.sxx += dt * (l2m * dvx_dx + lam * dvz_dz)
        state.szz += dt * (lam * dvx_dx + l2m * dvz_dz)
        state.sxz += dt * mu_xz * (
            D_plus_z(state.vx, a, dz) + D_plus_x(state.vz, a, dx)
        )

        # 4. Sponge damping
        state.vx  *= sponge
        state.vz  *= sponge
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
        t_sigma          = t_sigma,
        t_v              = t_v,
        receiver_vx      = rec_vx,
        receiver_vz      = rec_vz,
        snapshots_vz     = snaps_vz,
        snapshot_times_v = snaps_t_v,
    )