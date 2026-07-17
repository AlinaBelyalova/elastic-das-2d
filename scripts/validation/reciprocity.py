# ==============================================================================
# Elastic wave reciprocity test: G_zz(x_A, x_B) == G_zz(x_B, x_A)
#
# For a vertical point force F_z, the reciprocity theorem states:
#
#   v_z at x_B due to F_z at x_A  ==  v_z at x_A due to F_z at x_B
#
# This test uses solver_numpy_pointforce.py, because this validation path already
# injects a body force directly into the vz equation at the velocity half step.
#
# Source and receiver are placed exactly on vz staggered nodes so that receiver
# interpolation degenerates to a single-node lookup.
# ==============================================================================

from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from src.grid import Grid2D
from src.model import ElasticModel2D
from src.sampling import build_receiver_sampling
from src.solver_numpy import max_stable_dt
from src.solver_numpy_pointforce import run_elastic_solver_numpy_pointforce


def relative_l2(a: np.ndarray, b: np.ndarray, eps: float = 1.0e-30) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + eps))


def vz_node_coords(grid: Grid2D, ix: int, iz: int) -> tuple[float, float]:
    """
    Physical coordinates of vz[ix, iz].

    vz lives at:
        x = x0 + ix*dx
        z = z0 + (iz + 0.5)*dz
    """
    return float(grid.x0 + ix * grid.dx), float(grid.z0 + (iz + 0.5) * grid.dz)


def ricker_half_step(*, nt: int, dt: float, f0_hz: float) -> np.ndarray:
    """
    Ricker wavelet sampled on the velocity half-step axis:
        t_v[n] = (n + 0.5) dt
    """
    if nt <= 1:
        raise ValueError(f"nt must be > 1, got {nt}.")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}.")
    if f0_hz <= 0.0:
        raise ValueError(f"f0_hz must be positive, got {f0_hz}.")

    t0 = 1.2 / f0_hz
    t = (np.arange(nt, dtype=np.float64) + 0.5) * dt
    x = np.pi * f0_hz * (t - t0)

    return (1.0 - 2.0 * x**2) * np.exp(-x**2)


def build_model(
    *,
    nx: int = 401,
    nz: int = 401,
    nt: int = 5000,
    dx: float = 10.0,
    dz: float = 10.0,
    vp: float = 3000.0,
    vs: float = 1700.0,
    rho: float = 2500.0,
    half_order: int = 2,
    cfl_safety: float = 0.20,
) -> ElasticModel2D:
    dt = max_stable_dt(
        vp,
        dx,
        dz,
        half_order,
        safety=cfl_safety,
        use_ts_sfd=False,
    )

    grid = Grid2D(
        nx=nx,
        nz=nz,
        dx=dx,
        dz=dz,
        nt=nt,
        dt=dt,
        x0=0.0,
        z0=0.0,
    )

    return ElasticModel2D(
        grid=grid,
        vp=np.full(grid.shape, vp, dtype=np.float64),
        vs=np.full(grid.shape, vs, dtype=np.float64),
        rho=np.full(grid.shape, rho, dtype=np.float64),
    )


class SingleReceiver:
    """
    Minimal receiver object compatible with build_receiver_sampling().
    """
    def __init__(self, x: float, z: float) -> None:
        self.x = np.array([x], dtype=np.float64)
        self.z = np.array([z], dtype=np.float64)
        self.nrec = 1


def run_reciprocity_pair(
    *,
    model: ElasticModel2D,
    ix_A: int,
    iz_A: int,
    ix_B: int,
    iz_B: int,
    half_order: int,
    n_boundary: int,
    gamma_s: float,
    f0_hz: float,
) -> dict:
    """
    Run two simulations:

    A -> B:
        vertical force at A, record vz at B

    B -> A:
        vertical force at B, record vz at A

    For reciprocal elastic media:
        G_zz(B,A) == G_zz(A,B)
    """
    grid = model.grid

    force_stf_half = ricker_half_step(
        nt=grid.nt,
        dt=grid.dt,
        f0_hz=f0_hz,
    )

    x_A, z_A = vz_node_coords(grid, ix_A, iz_A)
    x_B, z_B = vz_node_coords(grid, ix_B, iz_B)

    rec_A = SingleReceiver(x_A, z_A)
    rec_B = SingleReceiver(x_B, z_B)

    sampling_A = build_receiver_sampling(grid, rec_A)
    sampling_B = build_receiver_sampling(grid, rec_B)

    shared = dict(
        vp=model.vp,
        vs=model.vs,
        rho=model.rho,
        dx=grid.dx,
        dz=grid.dz,
        dt=grid.dt,
        nt=grid.nt,
        force_stf_half=force_stf_half,
        half_order=half_order,
        use_ts_sfd=False,
        n_boundary=n_boundary,
        gamma_s=gamma_s,
        snapshot_stride=None,
    )

    # Source at A, receiver at B.
    res_AB = run_elastic_solver_numpy_pointforce(
        **shared,
        source_ix=ix_A,
        source_iz=iz_A,
        receiver_sampling=sampling_B,
    )
    vz_AB = res_AB.receiver_vz[0].copy()

    # Source at B, receiver at A.
    res_BA = run_elastic_solver_numpy_pointforce(
        **shared,
        source_ix=ix_B,
        source_iz=iz_B,
        receiver_sampling=sampling_A,
    )
    vz_BA = res_BA.receiver_vz[0].copy()

    rel_l2 = relative_l2(vz_AB, vz_BA)
    max_abs = float(np.max(np.abs(vz_AB - vz_BA)))
    peak = float(np.max(np.abs(vz_AB)))

    return {
        "ix_A": int(ix_A),
        "iz_A": int(iz_A),
        "ix_B": int(ix_B),
        "iz_B": int(iz_B),
        "x_A_m": x_A,
        "z_A_m": z_A,
        "x_B_m": x_B,
        "z_B_m": z_B,
        "offset_m": float(np.sqrt((x_B - x_A) ** 2 + (z_B - z_A) ** 2)),
        "rel_l2": rel_l2,
        "max_abs": max_abs,
        "peak_amplitude": peak,
        "t_v": res_AB.t_v,
        "vz_AB": vz_AB,
        "vz_BA": vz_BA,
    }


def main() -> None:
    outdir = Path("results/validation_reciprocity")
    outdir.mkdir(parents=True, exist_ok=True)

    half_order = 2
    n_boundary = 40
    gamma_s = 300.0
    f0_hz = 10.0

    model = build_model(
        half_order=half_order,
        cfl_safety=0.20,
    )
    grid = model.grid

    print("Reciprocity test: G_zz(A->B) == G_zz(B->A)")
    print(f"Grid       : {grid.nx} x {grid.nz}")
    print(f"nt         : {grid.nt}")
    print(f"dt         : {grid.dt:.6e} s")
    print(f"f0         : {f0_hz:.2f} Hz")
    print(f"half_order : {half_order}")
    print(f"n_boundary : {n_boundary}")

    cx = grid.nx // 2
    cz = grid.nz // 2

    pairs = [
        (cx - 50, cz,      cx + 50, cz,      "horizontal offset 100 cells"),
        (cx,      cz - 50, cx,      cz + 50, "vertical offset 100 cells"),
        (cx - 40, cz - 40, cx + 40, cz + 40, "diagonal offset"),
        (cx - 80, cz,      cx + 80, cz,      "horizontal offset 160 cells"),
        (cx - 30, cz + 20, cx + 50, cz - 30, "asymmetric offset"),
    ]

    all_results = []
    worst_rel_l2 = -np.inf
    worst_label = None

    for ix_A, iz_A, ix_B, iz_B, label in pairs:
        print("\n" + "-" * 80)
        print(f"Pair: {label}")
        print(f"A = ({ix_A}, {iz_A}), B = ({ix_B}, {iz_B})")

        result = run_reciprocity_pair(
            model=model,
            ix_A=ix_A,
            iz_A=iz_A,
            ix_B=ix_B,
            iz_B=iz_B,
            half_order=half_order,
            n_boundary=n_boundary,
            gamma_s=gamma_s,
            f0_hz=f0_hz,
        )
        result["label"] = label

        print(f"offset        : {result['offset_m']:.1f} m")
        print(f"peak amplitude: {result['peak_amplitude']:.6e}")
        print(f"rel_L2        : {result['rel_l2']:.6e}")
        print(f"max_abs       : {result['max_abs']:.6e}")

        all_results.append(result)

        if result["rel_l2"] > worst_rel_l2:
            worst_rel_l2 = result["rel_l2"]
            worst_label = label

    # Save compact JSON metrics without full traces.
    summary = {
        "grid": {
            "nx": grid.nx,
            "nz": grid.nz,
            "nt": grid.nt,
            "dx": grid.dx,
            "dz": grid.dz,
            "dt": grid.dt,
        },
        "model": {
            "vp": float(model.vp[0, 0]),
            "vs": float(model.vs[0, 0]),
            "rho": float(model.rho[0, 0]),
        },
        "settings": {
            "f0_hz": f0_hz,
            "half_order": half_order,
            "n_boundary": n_boundary,
            "gamma_s": gamma_s,
        },
        "worst_rel_l2": float(worst_rel_l2),
        "worst_label": worst_label,
        "pairs": [
            {
                "label": r["label"],
                "ix_A": r["ix_A"],
                "iz_A": r["iz_A"],
                "ix_B": r["ix_B"],
                "iz_B": r["iz_B"],
                "x_A_m": r["x_A_m"],
                "z_A_m": r["z_A_m"],
                "x_B_m": r["x_B_m"],
                "z_B_m": r["z_B_m"],
                "offset_m": r["offset_m"],
                "rel_l2": r["rel_l2"],
                "max_abs": r["max_abs"],
                "peak_amplitude": r["peak_amplitude"],
            }
            for r in all_results
        ],
    }

    json_path = outdir / "reciprocity_metrics.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved metrics to {json_path}")

    # Save full traces to npz.
    trace_payload = {
        "t_v": all_results[0]["t_v"],
    }
    for i, r in enumerate(all_results):
        trace_payload[f"vz_AB_{i}"] = r["vz_AB"]
        trace_payload[f"vz_BA_{i}"] = r["vz_BA"]

    npz_path = outdir / "reciprocity_traces.npz"
    np.savez_compressed(npz_path, **trace_payload)
    print(f"Saved traces to {npz_path}")

    threshold = 1.0e-8

    print("\n" + "=" * 80)
    print(f"Worst rel_L2: {worst_rel_l2:.6e} ({worst_label})")
    print(f"Threshold   : {threshold:.1e}")

    if worst_rel_l2 > threshold:
        raise RuntimeError(
            f"Reciprocity test FAILED: worst rel_L2={worst_rel_l2:.3e} > {threshold:.1e}. "
            f"Pair: {worst_label}"
        )

    print("Reciprocity test PASSED.")


if __name__ == "__main__":
    main()