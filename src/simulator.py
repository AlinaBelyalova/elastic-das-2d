# ==============================================================================
# src/simulator.py — Forward modeling orchestrator
# ==============================================================================

from __future__ import annotations

import warnings

import numpy as np

from src.model import ElasticModel2D
from src.source import EmbeddedSource2D
from src.receivers import Receivers2D
from src.das import compute_axial_strain_rate, DASResult
from src.sampling import build_receiver_sampling
from src.source_injection import build_stress_source_injection
from src.solver_numpy import run_elastic_solver_numpy, ElasticRunResult
from src.solver_numba_fused import run_elastic_solver_numba_fused


def run_forward_simulation(
    model: ElasticModel2D,
    source: EmbeddedSource2D,
    receivers: Receivers2D,
    gauge_length_m: float = 10.0,
    half_order: int = 2,
    use_ts_sfd: bool = False,
    n_boundary: int = 40,
    gamma_s: float = 300.0,
    snapshot_stride: int | None = None,
    backend: str = "numba_fused",
    free_surface: bool = False,
) -> tuple[ElasticRunResult, DASResult]:
    """
    Run the full elastic forward simulation and post-process into DAS strain-rate.

    Parameters
    ----------
    model : ElasticModel2D
        Elastic medium model and computational grid.
    source : EmbeddedSource2D
        Embedded 2D moment-tensor source.
        source.spreading controls injection mode:
          "nearest"  — snap to nearest stress-grid node (fast, default)
          "bilinear" — distribute onto four surrounding nodes (for off-grid sources)
    receivers : Receivers2D
        Receiver geometry.
    gauge_length_m : float
        DAS gauge length [m].
    half_order : int
        Spatial half-order M (spatial FD order = 2M).
    use_ts_sfd : bool
        Use TS-SFD coefficients instead of Taylor coefficients.
    n_boundary : int
        Sponge layer width in grid cells.
    gamma_s : float
        Sponge damping coefficient.
    snapshot_stride : int or None
        Save vz snapshots every snapshot_stride steps.
    backend : str
        Solver backend: "numba_fused" (default) or "numpy" (reference/debug).
        Bilinear spreading is only supported for "numba_fused".
    free_surface : bool
        Enable stress-free top boundary condition.
    """
    grid = model.grid

    if source.stf.nt != grid.nt:
        raise ValueError(
            f"Source STF nt={source.stf.nt} does not match grid.nt={grid.nt}."
        )
    if not np.isclose(source.stf.dt, grid.dt):
        raise ValueError(
            f"Source STF dt={source.stf.dt:.6e} does not match grid.dt={grid.dt:.6e}."
        )
    if not (0 <= source.ix < grid.nx and 0 <= source.iz < grid.nz):
        raise ValueError(
            f"Source indices ({source.ix}, {source.iz}) are outside the grid."
        )

    sampling = build_receiver_sampling(grid, receivers)

    stf_vals = source.stf.values
    stf_xx   = stf_vals * source.m2d.Mxx
    stf_zz   = stf_vals * source.m2d.Mzz
    stf_xz   = stf_vals * source.m2d.Mxz

    if backend == "numba_fused":
        # Build solver-ready injection arrays (nearest or bilinear, same API).
        # Only built for numba_fused; numpy backend is frozen and uses ix/iz directly.
        injection = build_stress_source_injection(source, grid)

        run_result = run_elastic_solver_numba_fused(
            vp=model.vp,
            vs=model.vs,
            rho=model.rho,
            dx=grid.dx,
            dz=grid.dz,
            dt=grid.dt,
            nt=grid.nt,
            source_ix=source.ix,
            source_iz=source.iz,
            stf_xx=stf_xx,
            stf_zz=stf_zz,
            stf_xz=stf_xz,
            receiver_sampling=sampling,
            half_order=half_order,
            use_ts_sfd=use_ts_sfd,
            n_boundary=n_boundary,
            gamma_s=gamma_s,
            snapshot_stride=snapshot_stride,
            free_surface=free_surface,
            source_injection=injection,
        )

    elif backend == "numpy":
        if source.spreading == "bilinear":
            warnings.warn(
                "source.spreading='bilinear' is not supported by the numpy backend. "
                "Falling back to nearest-node injection. "
                "Use backend='numba_fused' for bilinear spreading.",
                UserWarning,
                stacklevel=2,
            )
        run_result = run_elastic_solver_numpy(
            vp=model.vp,
            vs=model.vs,
            rho=model.rho,
            dx=grid.dx,
            dz=grid.dz,
            dt=grid.dt,
            nt=grid.nt,
            source_ix=source.ix,
            source_iz=source.iz,
            stf_xx=stf_xx,
            stf_zz=stf_zz,
            stf_xz=stf_xz,
            receiver_sampling=sampling,
            half_order=half_order,
            use_ts_sfd=use_ts_sfd,
            n_boundary=n_boundary,
            gamma_s=gamma_s,
            snapshot_stride=snapshot_stride,
            free_surface=free_surface,
        )

    else:
        raise ValueError(
            f"Unknown backend='{backend}'. Valid options: 'numba_fused', 'numpy'."
        )

    das_result = compute_axial_strain_rate(
        vx=run_result.receiver_vx,
        vz=run_result.receiver_vz,
        receivers=receivers,
        gauge_length_m=gauge_length_m,
    )

    return run_result, das_result