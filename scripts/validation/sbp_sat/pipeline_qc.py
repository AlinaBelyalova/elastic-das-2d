"""Existing physical-source API -> sparse bounded SBP -> DAS/adjoint/gradient.

Manufactured medium, no Sherlock/Oak/velocity-model files required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from src.common.grid import Grid2D
from src.common.model import ElasticModel2D
from src.elastic.source import build_dc_source
from src.elastic.simulator import run_forward_simulation
from src.das.receivers import Receivers2D
from src.sbp_sat.simulator import prepare_simulation, estimate_storage


def run_checks():
    grid = Grid2D(15, 17, 50.0, 40.0, 500, 0.0008, x0=-150.0, z0=0.0)
    x, z = np.indices(grid.shape)
    model = ElasticModel2D(
        grid,
        3200.0 + 70 * np.sin(0.4 * x),
        1700.0 + 30 * np.cos(0.5 * z),
        2200.0 + 20 * np.sin(x + z),
    )
    source = build_dc_source(
        grid=grid,
        x_m=123.7,
        z_m=241.3,
        theta_deg=35.0,
        scalar_moment=1e11,
        nt=grid.nt,
        dt=grid.dt,
        f0_hz=10.0,
        source_time_mode="ricker_moment",
        spreading="bilinear",
    )
    theta = np.linspace(-0.7, 0.7, 31)
    R = 120.0
    rec = Receivers2D(
        x=160.0 + R * np.sin(theta),
        z=200.0 + R * np.cos(theta),
        tx=np.cos(theta),
        tz=-np.sin(theta),
        s=R * (theta - theta[0]),
    )
    L = 17.3
    run, das = run_forward_simulation(
        model, source, rec, L, backend="sbp_sat", free_surface=True, snapshot_stride=100
    )
    p = prepare_simulation(model, source, rec, L, execution="numba")
    dense = prepare_simulation(model, source, rec, L, sparse_derivatives=False)
    reference = dense.solver.run_forward(
        source.stf.values, dense.source, dense.observation
    )
    amplitude = max(float(np.max(np.abs(reference.data))), np.finfo(float).tiny)
    parity_error = float(np.max(np.abs(das.data - reference.data)) / amplitude)
    csr = prepare_simulation(model, source, rec, L)
    csr_data = csr.solver.run_forward(
        source.stf.values, csr.source, csr.observation
    ).data
    csr_error = float(np.max(np.abs(csr_data - reference.data)) / amplitude)
    rng = np.random.default_rng(20261027)
    q = rng.normal(size=das.data.shape)
    adj = p.solver.run_adjoint_to_source(q, p.source, p.observation)
    lhs, rhs = float(np.vdot(das.data, q)), float(np.vdot(source.stf.values, adj))
    dot_error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), np.finfo(float).tiny)
    state = p.solver.zero_state()
    p.source.inject(state.sxx, state.szz, state.sxz, 1.0)
    integrals = [
        float(np.sum(p.solver.wn * state.sxx)),
        float(np.sum(p.solver.wn * state.szz)),
        float(np.sum(p.solver.ws * state.sxz)),
    ]
    mt = np.array([source.m2d.Mxx, source.m2d.Mzz, source.m2d.Mxz])
    source_error = float(np.max(np.abs(np.array(integrals) - mt)) / np.max(np.abs(mt)))
    truth = ElasticModel2D(grid, model.vp * 1.01, model.vs * 0.99, model.rho)
    pt = prepare_simulation(truth, source, rec, L)
    observed = pt.solver.run_forward(source.stf.values, pt.source, pt.observation).data
    J, data, gp, gs = p.solver.objective_and_gradient_log_vp_vs(
        source.stf.values, observed, p.source, p.observation
    )
    u, v = rng.normal(size=grid.shape), rng.normal(size=grid.shape)
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    derivative = float(np.sum(gp * u + gs * v))
    steps = np.array([0.01, 0.005, 0.0025, 0.00125])
    remainders = []
    for eps in steps:
        mm = ElasticModel2D(
            grid, model.vp * np.exp(eps * u), model.vs * np.exp(eps * v), model.rho
        )
        pp = prepare_simulation(mm, source, rec, L, execution="numba")
        d = pp.solver.run_forward(source.stf.values, pp.source, pp.observation).data
        j = 0.5 * grid.dt * float(np.sum((d - observed) ** 2))
        remainders.append(abs(j - J - eps * derivative))
    slopes = np.log2(np.array(remainders[:-1]) / remainders[1:])
    time_ok = bool(
        np.array_equal(run.t, (np.arange(grid.nt) + 0.5) * grid.dt)
        and source.stf.time_offset_steps == 0.5
    )
    report = dict(
        passed=bool(
            parity_error < 1e-11
            and csr_error < 1e-11
            and dot_error < 1e-11
            and source_error < 1e-14
            and np.all((slopes > 1.9) & (slopes < 2.1))
            and abs(derivative) > 1e-12
            and time_ok
        ),
        problem="manufactured SI medium; inherited numerical 2-D moment amplitude",
        solver_backend="sbp_sat",
        sparse_dense_relative_error=csr_error,
        numba_dense_relative_error=parity_error,
        solver_execution="numba",
        numba_threads=run.solver_metadata["numba_threads"],
        source_quadrature_relative_error=source_error,
        dot_product_relative_error=dot_error,
        source_position_m=[source.x_embedded_m, source.z_embedded_m],
        source_stf_kind=source.stf.kind,
        source_stf_time_offset_steps=0.5,
        source_t0_s=source.stf.t0,
        time_axes_passed=time_ok,
        vp_range_m_per_s=[float(model.vp.min()), float(model.vp.max())],
        vs_range_m_per_s=[float(model.vs.min()), float(model.vs.max())],
        rho_range_kg_per_m3=[float(model.rho.min()), float(model.rho.max())],
        physical_bounds_m=dict(
            x=[float(grid.x[0]), float(grid.x[-1])],
            z=[float(grid.z[0]), float(grid.z[-1])],
        ),
        normal_stress_shape=list(grid.shape),
        dx_m=grid.dx,
        dz_m=grid.dz,
        dt_s=grid.dt,
        nt=grid.nt,
        dt_sufficient_limit_s=p.solver.dt_limit,
        receiver_channels=rec.nrec,
        valid_gauge_centres=das.nchan_out,
        gauge_length_m=L,
        objective=J,
        gradient_directional_derivative=derivative,
        gradient_includes_model_dependent_boundary_impedances=True,
        taylor_steps=steps.tolist(),
        taylor_remainders=remainders,
        taylor_slopes=slopes.tolist(),
        storage=estimate_storage(grid, rec.nrec, 100),
        limitations=[
            "first-order impedance, not PML",
            "no absolute 3-D earthquake-moment amplitude calibration",
            "This QC exercises full history; checkpoint_qc checks exact checkpoint/replay.",
        ],
    )
    package = dict(
        **run.solver_metadata,
        t=run.t,
        t_sigma=run.t_sigma,
        dt=grid.dt,
        receiver_vx=run.receiver_vx,
        receiver_vz=run.receiver_vz,
        das_data=das.data,
        das_channel_indices=das.channel_indices,
        das_gauge_length_m=L,
        receiver_x=rec.x,
        receiver_z=rec.z,
        receiver_s=rec.s,
        grid_x=grid.x,
        grid_z=grid.z,
        vp=model.vp,
        vs=model.vs,
        rho=model.rho,
        source_time_function_t0_s=source.stf.t0,
        observed=observed,
        snapshots_vz=run.snapshots_vz,
        snapshot_times_v=run.snapshot_times_v,
        snapshot_grid_x=grid.x,
        snapshot_grid_z=grid.z[:-1] + grid.dz / 2,
    )
    return report, package


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
            "Output directory is not empty; choose another or use --overwrite."
        )
    report, package = run_checks()
    print(json.dumps(report, indent=2, allow_nan=False))
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        np.savez_compressed(args.output_dir / "outputs_sbp_pipeline_qc.npz", **package)
    if not report["passed"]:
        raise SystemExit("SBP-SAT pipeline QC FAILED")


if __name__ == "__main__":
    main()
