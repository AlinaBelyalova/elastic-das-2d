"""Bounded SBP-SAT: dissipation, DAS transpose, full gradient and outgoing P."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from src.das.receivers import Receivers2D
from src.sbp_sat import StressPointSource, staggered_sbp_4_2
from src.sbp_sat.rectangular import (
    BoundaryConditions,
    RectangularSBPSAT,
    prepare_rectangular_model,
    rectangular_timestep_bound,
)
from src.sbp_sat.rectangular_observations import build_rectangular_das_observation


def normal_incidence_packet(h, bottom="absorbing"):
    """Actual 2-D engine in the near-lambda=0 decoupled normal-P limit."""
    speed, density = np.sqrt(2.0), 2.0
    dt = 0.12 * h / speed
    shape = (int(8 / h) + 1, int(80 / h) + 1)
    model = prepare_rectangular_model(
        np.full(shape, speed), np.ones(shape), np.full(shape, density)
    )
    solver = RectangularSBPSAT(
        model,
        h,
        h,
        dt,
        BoundaryConditions(top="free", bottom=bottom, left="free", right="free"),
    )
    state = solver.zero_state()
    zn = np.arange(shape[1]) * h
    zm = (np.arange(shape[1] - 1) + 0.5) * h
    # Stresses at t=0; velocity at -dt/2 for a packet moving toward +z.
    state.vz[:] = np.exp(-0.5 * ((zm + speed * dt / 2 - 30) / 5) ** 2)
    state.szz[:] = -density * speed * np.exp(-0.5 * ((zn - 30) / 5) ** 2)
    initial = solver.energy(state)
    for _ in range(int(np.ceil((100 / speed) / dt))):
        state = solver._advance(state)
    return float(solver.energy(state) / initial)


def run_checks():
    nx, nz, dx, dz = 13, 14, 1.2, 0.9
    x, z = np.indices((nx, nz))
    model = prepare_rectangular_model(
        3.2 + 0.1 * np.sin(x + z), 1.5 + 0.06 * np.cos(z), 2.2 + 0.1 * np.sin(x - z)
    )
    limit = rectangular_timestep_bound(
        model, staggered_sbp_4_2(nx - 1, dx), staggered_sbp_4_2(nz - 1, dz)
    )
    f = RectangularSBPSAT(model, dx, dz, 0.4 * limit)
    rng = np.random.default_rng(20261027)
    state = f.zero_state()
    for name in ("vx", "vz", "sxx", "szz", "sxz"):
        a = getattr(state, name)
        a[:] = rng.normal(size=a.shape)
    initial = f.energy(state)
    energies = [initial]
    balance = []
    for _ in range(100):
        new = f.step(state)
        loss = f.dt * f.dissipation((state.vx + new.vx) / 2, (state.vz + new.vz) / 2)
        balance.append((f.energy(new) - energies[-1] + loss) / initial)
        energies.append(f.energy(new))
        state = new
    energy_error = float(np.max(np.abs(balance)))

    theta = np.linspace(-0.8, 0.8, 25)
    R = 2.4
    rec = Receivers2D(
        x=6 + R * np.sin(theta),
        z=5 + R * np.cos(theta),
        tx=np.cos(theta),
        tz=-np.sin(theta),
        s=R * (theta - theta[0]),
    )
    o = build_rectangular_das_observation(f, rec, 0.79)
    source = StressPointSource(5, 5, 1.0, -0.3, 0.5)
    nt = 220
    sequence = rng.normal(size=nt)
    q = rng.normal(size=(len(o), nt))
    d = f.run_forward(sequence, source, o).data
    adj = f.run_adjoint_to_source(q, source, o)
    lhs, rhs = float(np.vdot(d, q)), float(np.vdot(sequence, adj))
    dot_error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), np.finfo(float).tiny)

    times = (np.arange(nt) + 0.5) * f.dt
    stf = 100 * np.exp(-(((times - 0.4) / 0.15) ** 2))
    truth = prepare_rectangular_model(model.vp * 1.015, model.vs * 0.985, model.rho)
    ft = RectangularSBPSAT(truth, dx, dz, f.dt, f.boundaries)
    observed = ft.run_forward(stf, source, o).data
    J, synthetic, gp, gs = f.objective_and_gradient_log_vp_vs(stf, observed, source, o)
    p, s = rng.normal(size=model.vp.shape), rng.normal(size=model.vs.shape)
    p /= np.linalg.norm(p)
    s /= np.linalg.norm(s)
    derivative = float(np.sum(gp * p + gs * s))
    steps = np.array([0.01, 0.005, 0.0025, 0.00125])
    remainders = []
    for eps in steps:
        mm = prepare_rectangular_model(
            model.vp * np.exp(eps * p), model.vs * np.exp(eps * s), model.rho
        )
        ff = RectangularSBPSAT(mm, dx, dz, f.dt, f.boundaries)
        data = ff.run_forward(stf, source, o).data
        j = 0.5 * f.dt * float(np.sum((data - observed) ** 2))
        remainders.append(abs(j - J - eps * derivative))
    slopes = np.log2(np.array(remainders[:-1]) / remainders[1:])
    packet_coarse = normal_incidence_packet(1.0)
    packet_fine = normal_incidence_packet(0.5)
    packet_free = normal_incidence_packet(1.0, "free")
    passed = bool(
        energy_error < 1e-12
        and np.all(np.array(energies) > 0)
        and np.max(np.diff(energies)) / initial < 1e-13
        and dot_error < 1e-11
        and np.all((slopes > 1.9) & (slopes < 2.1))
        and abs(derivative) > 1e-10
        and packet_fine < packet_coarse < 1e-4
        and abs(packet_free - 1) < 1e-12
    )
    report = {
        "passed": passed,
        "problem": "manufactured SI-scale model, not a SAFOD material model",
        "boundary_conditions": {
            "top": "free",
            "bottom": "absorbing_impedance",
            "left": "absorbing_impedance",
            "right": "absorbing_impedance",
        },
        "boundary_model": "first-order impedance; not PML and not exact at oblique incidence",
        "native_shapes": {
            "normal_stress": list(model.vp.shape),
            "vx": list(model.vx_shape),
            "vz": list(model.vz_shape),
            "shear_stress": list(model.mu_MM.shape),
        },
        "physical_bounds_m": {"x": [0.0, (nx - 1) * dx], "z": [0.0, (nz - 1) * dz]},
        "dt_s": f.dt,
        "dt_sufficient_limit_s": limit,
        "maximum_relative_energy_balance_error": energy_error,
        "final_energy_ratio": float(energies[-1] / initial),
        "dot_product_relative_error": dot_error,
        "objective": J,
        "gradient_directional_derivative": derivative,
        "gradient_includes_model_dependent_boundary_impedances": True,
        "taylor_steps": steps.tolist(),
        "taylor_remainders": remainders,
        "taylor_slopes": slopes.tolist(),
        "normal_P_packet_energy_ratio": {
            "h1_absorbing": packet_coarse,
            "h0p5_absorbing": packet_fine,
            "h1_free_control": packet_free,
        },
    }
    traces = {
        "time_s": times,
        "synthetic": synthetic,
        "observed": observed,
        "fibre_distance_m": rec.s[o.channel_indices],
        "energy": np.array(energies),
        "relative_energy_balance": np.array(balance),
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
            "Output directory is not empty; choose another or use --overwrite explicitly."
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
        raise SystemExit("Bounded absorbing SBP-SAT QC FAILED")


if __name__ == "__main__":
    main()
