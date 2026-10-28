"""Warm SBP-SAT Numba/CSR benchmark and parity; optional event-sized grid.

No external data. Reports compilation separately and projects ONLY wave-loop
cost to the requested event step count. No timing threshold is a math test.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter
import numpy as np
import numba

from src.sbp_sat import prepare_rectangular_model, StressPointSource
from src.sbp_sat.rectangular import RectangularSBPSAT
from src.sbp_sat.numba_solver import NumbaSBPSAT
from src.sbp_sat.rectangular_observations import build_rectangular_das_observation
from src.das.receivers import Receivers2D

FIELDS = ("vx", "vz", "sxx", "szz", "sxz")


def benchmark(nx, nz, steps, nrec, threads, project_steps, repeats=3):
    x, z = np.indices((nx, nz))
    model = prepare_rectangular_model(
        5500 + 500 * np.sin(0.07 * x + 0.02 * z),
        3000 + 180 * np.cos(0.03 * z) + np.zeros_like(x),
        2500 + 30 * np.sin(0.03 * x - 0.04 * z),
    )
    ref = RectangularSBPSAT(model, 5.0, 5.0, 0.0001, sparse_derivatives=True)
    del x, z
    src = StressPointSource(nx // 2, nz // 2, 1.0, -0.3, 0.7)
    theta = np.linspace(-0.7, 0.7, nrec)
    radius = min(nx - 1, nz - 1) * 5 * 0.2
    rec = Receivers2D(
        x=(nx - 1) * 2.5 + radius * np.sin(theta),
        z=(nz - 1) * 2 + radius * np.cos(theta),
        tx=np.cos(theta),
        tz=-np.sin(theta),
        s=radius * (theta - theta[0]),
    )
    obs = build_rectangular_das_observation(ref, rec, radius * 0.1)
    initial = ref.zero_state()
    rng = np.random.default_rng(771)
    for name in FIELDS:
        getattr(initial, name)[:] = rng.normal(size=getattr(initial, name).shape)
    stf = rng.normal(size=steps)

    def run(f, compiled):
        state = initial.copy()
        data = np.empty((2, nrec, steps))
        start = perf_counter()
        for it, amplitude in enumerate(stf):
            state = (
                f.advance_inplace(state, src, amplitude)
                if compiled
                else f._advance(state, src, amplitude)
            )
            if compiled:
                f.sample_receivers(state, obs, data[0], data[1], it)
            else:
                data[0, :, it] = obs._vx.sample(state.vx)
                data[1, :, it] = obs._vz.sample(state.vz)
        return perf_counter() - start, state, data

    # Component timings are diagnostic, not a decomposition of a whole-step
    # wall time: each call has its own arrays/cache and may overlap allocations.
    profile = {}
    for name, fn in (
        ("force_and_free_SAT", lambda: ref.force(initial)),
        ("damping_force", lambda: ref.damping_force(initial.vx, initial.vz)),
        ("implicit_velocity_solve", lambda: ref.solve_velocity(initial.vx, initial.vz)),
        ("stress_rhs", lambda: ref.stress_rhs(initial.vx, initial.vz)),
    ):
        start = perf_counter()
        for _ in range(3):
            fn()
        profile[name] = (perf_counter() - start) / 3
    print(f"CSR reference: {nx}x{nz}, {steps} steps...", flush=True)
    csr_times=[]
    for _ in range(repeats):
        seconds,reference,data_ref=run(ref,False)
        csr_times.append(seconds)
    seconds=float(np.median(csr_times))
    results = []
    for count in threads:
        f = NumbaSBPSAT(model, 5.0, 5.0, 0.0001, numba_threads=count)
        with f.thread_context():
            print(f"Numba warmup: {f.numba_threads} threads...", flush=True)
            state = initial.copy()
            start = perf_counter()
            f.advance_inplace(state, src, 0.0)
            temp = np.empty((nrec, 1))
            f.sample_receivers(state, obs, temp, temp.copy(), 0)
            warmup = perf_counter() - start
            del state
            times=[]
            for _ in range(repeats):
                elapsed,state,data=run(f,True)
                times.append(elapsed)
            elapsed=float(np.median(times))
        errors = {
            name: float(
                np.linalg.norm(getattr(state, name) - getattr(reference, name))
                / max(np.linalg.norm(getattr(reference, name)), np.finfo(float).tiny)
            )
            for name in FIELDS
        }
        data_error = float(
            np.linalg.norm(data - data_ref)
            / max(np.linalg.norm(data_ref), np.finfo(float).tiny)
        )
        results.append(
            dict(
                threads=f.numba_threads,
                warmup_s=warmup,
                propagation_s=elapsed,
                propagation_repeats_s=times,
                seconds_per_step=elapsed / steps,
                speedup=seconds / elapsed,
                projected_wave_loop_s=project_steps * elapsed / steps,
                field_relative_errors=errors,
                receiver_relative_error=data_error,
                passed=bool(max(errors.values()) < 1e-11 and data_error < 1e-11),
            )
        )
        del f, state, data
    return dict(
        passed=all(r["passed"] for r in results),
        problem="manufactured heterogeneous medium; not actual SAFOD velocity inputs",
        python=platform.python_version(),
        numpy=np.__version__,
        numba=numba.__version__,
        native_shape_normal_stress=[nx, nz],
        dx_m=5.0,
        dz_m=5.0,
        dt_s=0.0001,
        steps=steps,
        receivers=nrec,
        projected_event_steps=project_steps,
        csr_propagation_s=seconds,
        csr_propagation_repeats_s=csr_times,
        repeats=repeats,
        csr_seconds_per_step=seconds / steps,
        csr_projected_wave_loop_s=project_steps * seconds / steps,
        csr_component_seconds_per_call=profile,
        numba_results=results,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nx", type=int, default=401)
    p.add_argument("--nz", type=int, default=361)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--receivers", type=int, default=1200)
    p.add_argument("--threads", type=int, nargs="+", default=None)
    p.add_argument("--project-steps", type=int, default=19062)
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    if min(a.nx, a.nz) < 9 or min(a.steps, a.receivers, a.project_steps) < 2 or a.repeats<1:
        p.error("Require nx,nz>=9 and steps/receivers/project-steps>=2.")
    if (
        a.output_dir
        and a.output_dir.exists()
        and any(a.output_dir.iterdir())
        and not a.overwrite
    ):
        p.error("Output directory is not empty; choose another or use --overwrite.")
    report = benchmark(
        a.nx, a.nz, a.steps, a.receivers, a.threads or [None], a.project_steps,a.repeats
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    if a.output_dir:
        a.output_dir.mkdir(parents=True, exist_ok=True)
        (a.output_dir / "summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    if not report["passed"]:
        raise SystemExit("SBP-SAT Numba parity FAILED")


if __name__ == "__main__":
    main()
