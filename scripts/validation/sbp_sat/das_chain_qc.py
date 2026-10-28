"""End-to-end SBP-SAT -> curved off-grid DAS transpose and gradient QC.

Uses a manufactured small model, with no SAFOD files or external packages.
This is a closed periodic-x / free-z reference, not an absorbing-boundary run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from src.das.receivers import Receivers2D
from src.sbp_sat import (
    StressPointSource,
    build_das_observation,
    objective_and_gradient_log_vp_vs,
    prepare_model,
    run_adjoint_to_source,
    run_forward,
    stable_timestep_bound,
    staggered_sbp_4_2,
)


def run_checks():
    nx, n, dx, dz = 20, 20, 1.1, 1.0
    x, z = np.indices((nx, n + 1))
    model = prepare_model(
        3.4 + 0.12 * np.sin(x + z), 1.8 + 0.08 * np.cos(z), 2.2 + 0.1 * np.cos(x - z)
    )
    op = staggered_sbp_4_2(n, dz)
    theta = np.linspace(-0.8, 0.8, 25)
    rec = Receivers2D(
        x=9.2 + 4 * np.sin(theta),
        z=4.1 + 4 * np.cos(theta),
        tx=np.cos(theta),
        tz=-np.sin(theta),
        s=4 * (theta - theta[0]),
    )
    observation = build_das_observation(model, op, dx, rec, 1.37)
    source = StressPointSource(8, 6, 1.0, -0.4, 0.7)
    dt_limit = stable_timestep_bound(model, op, dx)
    dt = 0.6 * dt_limit
    rng = np.random.default_rng(20261026)
    nt = 160
    sequence = rng.normal(size=nt)
    q = rng.normal(size=(len(observation), nt))
    forward = run_forward(sequence, model, op, dx, dt, source, observation)
    transpose = run_adjoint_to_source(q, model, op, dx, dt, source, observation)
    lhs, rhs = float(np.vdot(forward.data, q)), float(np.vdot(sequence, transpose))
    dot_error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), np.finfo(float).tiny)

    times = (np.arange(nt) + 0.5) * dt  # actual receiver velocity half-times
    stf = 100 * np.exp(-(((np.arange(nt) * dt - 0.4) / 0.15) ** 2))
    truth = prepare_model(model.vp * 1.015, model.vs * 0.985, model.rho)
    observed = run_forward(stf, truth, op, dx, dt, source, observation).data
    J, data, gp, gs = objective_and_gradient_log_vp_vs(
        stf, observed, model, op, dx, dt, source, observation
    )
    p, q = rng.normal(size=model.vp.shape), rng.normal(size=model.vs.shape)
    p /= np.linalg.norm(p)
    q /= np.linalg.norm(q)
    directional = float(np.sum(gp * p + gs * q))
    steps = np.array([0.02, 0.01, 0.005, 0.0025])
    remainders = []
    for eps in steps:
        perturbed = prepare_model(
            model.vp * np.exp(eps * p), model.vs * np.exp(eps * q), model.rho
        )
        d = run_forward(stf, perturbed, op, dx, dt, source, observation).data
        j = 0.5 * dt * float(np.sum((d - observed) ** 2))
        remainders.append(abs(j - J - eps * directional))
    slopes = np.log2(np.array(remainders[:-1]) / remainders[1:])
    passed = bool(
        dot_error < 1e-11
        and np.all((slopes > 1.9) & (slopes < 2.1))
        and np.linalg.norm(data) > 0
        and abs(directional) > 1e-12
    )
    report = {
        "passed": passed,
        "boundary_conditions": {
            "x": "periodic",
            "top": "traction_free_sat",
            "bottom": "traction_free_sat",
        },
        "native_shapes": {"sxx_szz": [nx, n + 1], "vx": [nx, n + 1], "vz_sxz": [nx, n]},
        "physical_bounds_m": {"x_period": [0.0, nx * dx], "z": [0.0, n * dz]},
        "dx_m": dx,
        "dz_m": dz,
        "dt_s": dt,
        "dt_sufficient_limit_s": dt_limit,
        "receiver_channels": rec.nrec,
        "valid_gauge_centres": len(observation),
        "gauge_length_m": observation.gauge_length_m,
        "observable": "axial_strain_rate_1_per_s_centre_tangent_convention",
        "source": "three_component_grid_stress_rate; no earthquake_moment_normalization",
        "dot_product_relative_error": dot_error,
        "objective": J,
        "gradient_directional_derivative": directional,
        "taylor_steps": steps.tolist(),
        "taylor_remainders": remainders,
        "taylor_slopes": slopes.tolist(),
    }
    traces = {
        "time_s": times,
        "synthetic": data,
        "observed": observed,
        "fibre_distance_m": rec.s[observation.channel_indices],
        "channel_indices": observation.channel_indices,
    }
    return report, traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if (
        args.output_dir
        and args.output_dir.exists()
        and any(args.output_dir.iterdir())
        and not args.overwrite
    ):
        raise SystemExit(
            "Output directory is not empty; choose a new one or explicitly use --overwrite."
        )
    report, traces = run_checks()
    print(json.dumps(report, indent=2, allow_nan=False))
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        np.savez(args.output_dir / "traces.npz", **traces)
    if not report["passed"]:
        raise SystemExit("SBP-SAT/DAS chain QC FAILED")


if __name__ == "__main__":
    main()
