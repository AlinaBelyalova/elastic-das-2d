"""Full-history parity, checkpointed Taylor QC and allocation-free event estimate.

Manufactured SI medium only. No real data inversion or event-size gradient
is launched. Reverse/VJP still uses the independent matrix implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src.common.grid import Grid2D
from src.common.model import ElasticModel2D
from src.das.receivers import Receivers2D
from src.elastic.source import build_dc_source
from src.sbp_sat.checkpointing import plan_checkpoints
from src.sbp_sat.simulator import prepare_simulation


def relative_error(value, reference):
    norm = max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
    return float(np.linalg.norm(value - reference)) / norm


def run_checks():
    grid = Grid2D(15, 17, 50, 40, 500, 0.0008, x0=-150)
    x, z = np.indices(grid.shape)
    model = ElasticModel2D(
        grid,
        3200 + 70 * np.sin(0.4 * x),
        1700 + 30 * np.cos(0.5 * z),
        2200 + 20 * np.sin(x + z),
    )
    source = build_dc_source(
        grid=grid,
        x_m=123.7,
        z_m=241.3,
        theta_deg=35,
        scalar_moment=1e11,
        nt=grid.nt,
        dt=grid.dt,
        f0_hz=10,
        source_time_mode="ricker_moment",
        spreading="bilinear",
    )
    theta = np.linspace(-0.7, 0.7, 31)
    rec = Receivers2D(
        x=160 + 120 * np.sin(theta),
        z=200 + 120 * np.cos(theta),
        tx=np.cos(theta),
        tz=-np.sin(theta),
        s=120 * (theta - theta[0]),
    )
    p = prepare_simulation(model, source, rec, 17.3, execution="numba")
    truth = ElasticModel2D(grid, model.vp * 1.01, model.vs * 0.99, model.rho)
    pt = prepare_simulation(truth, source, rec, 17.3, execution="numba")
    observed = pt.solver.run_forward(source.stf.values, pt.source, pt.observation).data
    args = source.stf.values, observed, p.source, p.observation
    full = p.solver.objective_and_gradient_log_vp_vs(*args)
    started = perf_counter()
    checkpointed = p.solver.objective_and_gradient_log_vp_vs(
        *args, checkpoint_interval="auto"
    )
    elapsed = perf_counter() - started
    names = ("objective", "data", "gradient_log_vp", "gradient_log_vs")
    parity = {
        name: relative_error(value, ref)
        for name, value, ref in zip(names, checkpointed, full)
    }
    bitwise = all(np.array_equal(a, b) for a, b in zip(checkpointed, full))
    J, data, gp, gs = checkpointed
    rng = np.random.default_rng(20261027)
    u, v = rng.normal(size=grid.shape), rng.normal(size=grid.shape)
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    derivative = float(np.sum(gp * u + gs * v))

    def objective(eps):
        mm = ElasticModel2D(
            grid, model.vp * np.exp(eps * u), model.vs * np.exp(eps * v), model.rho
        )
        pp = prepare_simulation(mm, source, rec, 17.3, execution="numba")
        d = pp.solver.run_forward(source.stf.values, pp.source, pp.observation).data
        return 0.5 * grid.dt * float(np.sum((d - observed) ** 2))

    steps = np.array([0.01, 0.005, 0.0025, 0.00125])
    remainders = [abs(objective(e) - J - e * derivative) for e in steps]
    slopes = np.log2(np.array(remainders[:-1]) / remainders[1:])
    fd = (objective(1e-5) - objective(-1e-5)) / 2e-5
    fd_error = abs(fd - derivative) / max(abs(derivative), np.finfo(float).tiny)
    plan = plan_checkpoints(grid.nx, grid.nz, grid.nt)
    event = plan_checkpoints(1241, 1121, 19062)
    report = {
        "passed": bool(
            max(parity.values()) < 1e-12
            and np.all((slopes > 1.9) & (slopes < 2.1))
            and fd_error < 2e-6
            and abs(derivative) > 1e-12
        ),
        "problem": "manufactured SI model; no real-event FWI",
        "solver_execution": "numba_forward; independent_matrix_reverse_and_VJP",
        "checkpoint_scheme": "uniform_full_state; forward_replay; float64",
        "parity_relative_errors": parity,
        "parity_bitwise_equal": bitwise,
        "objective": J,
        "gradient_directional_derivative": derivative,
        "finite_difference_relative_error": fd_error,
        "taylor_steps": steps.tolist(),
        "taylor_remainders": remainders,
        "taylor_slopes": slopes.tolist(),
        "model_dependent_boundary_impedances": True,
        "plan": plan.summary(),
        "example_event_history_only": event.summary(),
        "example_event_allocated": False,
        "checkpointed_gradient_elapsed_s": elapsed,
        "limitations": [
            "Uniform in-memory checkpoints; not disk/Revolve.",
            "History estimates exclude other process allocations.",
            "Matrix reverse/VJP is retained; full event gradient runtime has not been measured.",
            "Grid, dt, rho, source and receiver geometry are fixed in this derivative.",
        ],
    }
    if not report["passed"]:
        raise AssertionError(json.dumps(report, indent=2))
    return report, {"synthetic": data, "gradient_log_vp": gp, "gradient_log_vs": gs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if (
        args.output_dir
        and args.output_dir.exists()
        and any(args.output_dir.iterdir())
        and not args.overwrite
    ):
        raise FileExistsError(
            "Output directory is not empty; use --overwrite intentionally."
        )
    report, arrays = run_checks()
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        np.savez_compressed(args.output_dir / "checkpoint_qc.npz", **arrays)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
