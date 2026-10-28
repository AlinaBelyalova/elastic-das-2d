"""SBP-SAT adapter for the existing model/source/receiver/DAS pipeline.

No model resampling, sponge, moment rescaling or change of DAS observable.
Grid2D points are the normal-stress grid, including both physical endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from contextlib import nullcontext
from numbers import Integral
from time import perf_counter
import numpy as np

from src.common.model import ElasticModel2D
from src.das.strain import compute_axial_strain_rate
from src.elastic.solver_numpy import ElasticRunResult
from .operators import staggered_sbp_4_2
from .rectangular import (
    BoundaryConditions,
    RectangularSBPSAT,
    prepare_rectangular_model,
    rectangular_timestep_bound,
)
from .rectangular_observations import build_rectangular_das_observation
from .sources import build_moment_source


def configure_time_grid(model, *, duration_s=None, safety=0.8):
    """Select a certified dt; preserve requested physical duration, not step count.

    If duration_s is omitted, retain at least (old_nt-1)*old_dt. Rebuild the
    source AFTER this function. Spatial nodes and materials are unchanged.
    """
    if not np.isfinite(safety) or not 0 < safety < 1:
        raise ValueError("SBP timestep safety must be strictly between zero and one.")
    g = model.grid
    duration = (g.nt - 1) * g.dt if duration_s is None else float(duration_s)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Requested duration must be finite and positive.")
    m = prepare_rectangular_model(model.vp, model.vs, model.rho)
    limit = rectangular_timestep_bound(
        m, staggered_sbp_4_2(g.nx - 1, g.dx), staggered_sbp_4_2(g.nz - 1, g.dz)
    )
    dt = min(g.dt, safety * limit)
    nt = max(2, int(np.ceil(duration / dt)) + 1)
    grid = replace(g, dt=dt, nt=nt)
    adjusted = ElasticModel2D(grid, model.vp, model.vs, model.rho)
    return adjusted, dict(
        requested_duration_s=duration,
        input_nt=g.nt,
        input_dt_s=g.dt,
        dt_s=dt,
        nt=nt,
        dt_sufficient_limit_s=limit,
        timestep_safety=safety,
    )


@dataclass(frozen=True)
class PreparedSimulation:
    solver: RectangularSBPSAT
    source: object
    observation: object


def prepare_simulation(
    model,
    source,
    receivers,
    gauge_length_m,
    *,
    free_surface=True,
    sparse_derivatives=True,
    execution=None,
    numba_threads=None,
):
    """Shared setup for production forward, source adjoint and model gradient."""
    g = model.grid
    execution = (
        ("csr" if sparse_derivatives else "numpy") if execution is None else execution
    )
    if execution not in ("numba", "csr", "numpy"):
        raise ValueError("SBP execution must be numba, csr or numpy.")
    coefficients = prepare_rectangular_model(model.vp, model.vs, model.rho)
    bc = BoundaryConditions(top="free" if free_surface else "absorbing")
    if execution == "numba":
        from .numba_solver import NumbaSBPSAT

        f = NumbaSBPSAT(coefficients, g.dx, g.dz, g.dt, bc, numba_threads=numba_threads)
    else:
        f = RectangularSBPSAT(
            coefficients, g.dx, g.dz, g.dt, bc, sparse_derivatives=execution == "csr"
        )
    src = build_moment_source(f, source, g)
    obs = build_rectangular_das_observation(
        f, receivers, gauge_length_m, x0=g.x0, z0=g.z0
    )
    return PreparedSimulation(f, src, obs)


def estimate_storage(grid, nrec, snapshot_stride=None):
    """Conservative allocation estimate, not a measured process peak.

    Includes material/step temporaries, source/DAS setup, outputs, and dense
    1-D operator certificates. Excludes plotting, Python and external data.
    Gradient history is separate and is NOT allocated by the forward loop.
    """
    from .checkpointing import plan_checkpoints

    plan = plan_checkpoints(grid.nx, grid.nz, grid.nt)
    frames = 0 if snapshot_stride is None else len(range(0, grid.nt, snapshot_stride))
    cells = grid.nx * grid.nz
    return dict(
        forward_estimated_array_gib=8
        * (
            100 * cells
            + 3 * nrec * grid.nt
            + frames * grid.nx * (grid.nz - 1)
            + 4 * (grid.nx**2 + grid.nz**2)
        )
        / 2**30,
        gradient_velocity_history_gib=8
        * grid.nt
        * ((grid.nx - 1) * grid.nz + grid.nx * (grid.nz - 1))
        / 2**30,
        gradient_checkpointed_history_gib=plan.history_bytes / 2**30,
        gradient_checkpoint_interval=plan.interval,
        gradient_checkpoint_count=plan.checkpoint_count,
        snapshot_frames=frames,
    )


def run_forward_simulation(
    model,
    source,
    receivers,
    gauge_length_m,
    *,
    snapshot_stride=None,
    free_surface=True,
    execution="numba",
    numba_threads=None,
):
    """Return the existing ElasticRunResult/DASResult interfaces.

    Only the small receiver histories and requested native vz snapshots are
    stored; no full wavefield history is retained for a forward calculation.
    """
    if snapshot_stride is not None and (
        isinstance(snapshot_stride, (bool, np.bool_))
        or not isinstance(snapshot_stride, Integral)
        or snapshot_stride < 1
    ):
        raise ValueError("snapshot_stride must be a positive integer or None.")
    prepared = prepare_simulation(
        model,
        source,
        receivers,
        gauge_length_m,
        free_surface=free_surface,
        execution=execution,
        numba_threads=numba_threads,
    )
    f, src, obs = prepared.solver, prepared.source, prepared.observation
    g = model.grid
    tv = (np.arange(g.nt) + 0.5) * g.dt
    ts = np.arange(g.nt + 1) * g.dt
    vx, vz = np.empty((receivers.nrec, g.nt)), np.empty((receivers.nrec, g.nt))
    indices = None if snapshot_stride is None else np.arange(0, g.nt, snapshot_stride)
    snapshots = None if indices is None else np.empty((len(indices), *f.model.vz_shape))
    state = f.zero_state()
    iframe = 0
    progress_stride = min(200, max(1, g.nt // 10))
    compiled = execution == "numba"
    warmup_s = 0.0
    context = f.thread_context() if compiled else nullcontext()
    with context:
        if compiled:
            print(
                f"SBP-SAT Numba: {f.numba_threads} threads; compiling/warming kernels...",
                flush=True,
            )
            started = perf_counter()
            # Zero state + zero source amplitude stays exactly zero; compilation
            # is measured separately and the numerical time grid is unchanged.
            f.advance_inplace(state, src, 0.0)
            f.sample_receivers(state, obs, vx, vz, 0)
            warmup_s = perf_counter() - started
            print(f"SBP-SAT Numba warmup: {warmup_s:.2f} s", flush=True)
        started = perf_counter()
        for it, amplitude in enumerate(source.stf.values):
            state = (
                f.advance_inplace(state, src, amplitude)
                if compiled
                else f._advance(state, src, amplitude)
            )
            if compiled:
                f.sample_receivers(state, obs, vx, vz, it)
            else:
                vx[:, it], vz[:, it] = (
                    obs._vx.sample(state.vx),
                    obs._vz.sample(state.vz),
                )
            if indices is not None and iframe < len(indices) and it == indices[iframe]:
                snapshots[iframe] = state.vz
                iframe += 1
            if it == 0 or (it + 1) % progress_stride == 0 or it + 1 == g.nt:
                if not all(
                    np.isfinite(a).all()
                    for a in (state.vx, state.vz, state.sxx, state.szz, state.sxz)
                ):
                    raise FloatingPointError(
                        f"Nonfinite SBP wavefield at step {it + 1}."
                    )
                elapsed = perf_counter() - started
                print(
                    f"SBP-SAT {execution}: {it + 1}/{g.nt} steps, {elapsed:.1f} s elapsed",
                    flush=True,
                )
        elapsed = perf_counter() - started
    metadata = dict(
        solver_backend="sbp_sat",
        solver_spatial_order="4 interior / 2 closure",
        solver_derivatives=(
            "numba_sbp_4_2_float64_no_fastmath"
            if compiled
            else "scipy_csr_same_sbp_coefficients"
            if execution == "csr"
            else "numpy_dense_sbp"
        ),
        solver_execution=execution,
        numba_threads=f.numba_threads if compiled else 0,
        kernel_warmup_s=warmup_s,
        propagation_elapsed_s=elapsed,
        boundary_top=f.boundaries.top,
        boundary_bottom=f.boundaries.bottom,
        boundary_left=f.boundaries.left,
        boundary_right=f.boundaries.right,
        boundary_method="traction SAT + first-order impedance; no sponge or PML",
        sbp_dt_sufficient_limit_s=f.dt_limit,
        source_spatial_normalization="bilinear_transpose_divided_by_native_SBP_quadrature",
        source_amplitude_convention="existing_2d_tensor_scale; no_3d_to_2d_conversion",
        source_stf_kind=source.stf.kind,
        source_stf_sampling_offset_steps=source.stf.time_offset_steps,
        velocity_time_offset_steps=0.5,
        native_shape_normal_stress=np.array(f.model.vp.shape),
        native_shape_vx=np.array(f.model.vx_shape),
        native_shape_vz=np.array(f.model.vz_shape),
        native_shape_shear_stress=np.array(f.model.mu_MM.shape),
    )
    result = ElasticRunResult(
        ts,
        tv,
        vx,
        vz,
        snapshots,
        None if indices is None else tv[indices],
        solver_metadata=metadata,
    )
    das = compute_axial_strain_rate(vx, vz, receivers, gauge_length_m)
    return result, das


def objective_and_gradient_log_vp_vs(
    model,
    source,
    receivers,
    observed,
    gauge_length_m,
    *,
    free_surface=True,
    execution="numba",
    checkpoint_interval="auto",
):
    """Same physical source/DAS chain as forward; fixed rho/dt/source/geometry.

    Auto checkpoint/replay is the default; an integer fixes block length.
    None retains the independent full-history reference. Only history storage
    changes; the reverse/VJP is still the matrix reference, not compiled.
    """
    p = prepare_simulation(
        model,
        source,
        receivers,
        gauge_length_m,
        free_surface=free_surface,
        execution=execution,
    )
    return p.solver.objective_and_gradient_log_vp_vs(
        source.stf.values,
        observed,
        p.source,
        p.observation,
        checkpoint_interval=checkpoint_interval,
    )
