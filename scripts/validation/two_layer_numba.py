# ==============================================================================
# scripts/validation/two_layer_numba.py
#
# Controlled heterogeneous benchmark for the production Numba elastic-DAS
# forward engine. This bridges the homogeneous analytical validation and the
# forthcoming discrete adjoint/Taylor tests.
# ==============================================================================

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from src.fwi.two_layer import TwoLayerFWIProblem


def _robust_limit(data: np.ndarray, percentile: float = 99.0) -> float:
    value = float(np.percentile(np.abs(data), percentile))
    return value if np.isfinite(value) and value > 0.0 else 1.0


def _relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(
        np.linalg.norm(candidate - reference)
        / max(np.linalg.norm(reference), 1.0e-30)
    )


def _plot_model(problem: TwoLayerFWIProblem, out_path: Path) -> None:
    g = problem.grid
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    im = ax.imshow(
        problem.true_medium.vp.T,
        origin="upper",
        extent=[float(g.x[0]), float(g.x[-1]), float(g.z[-1]), float(g.z[0])],
        aspect="equal",
    )
    fig.colorbar(im, ax=ax, label="Vp [m/s]")
    ax.axhline(problem.model_spec.interface_depth_m, ls="--", lw=1.2, label="interface")
    ax.plot(problem.receivers.x, problem.receivers.z, lw=1.8, label="DAS cable")
    ax.scatter(
        [problem.source_spec.x_m],
        [problem.source_spec.z_m],
        marker="*",
        s=120,
        label="moment-tensor source",
        zorder=5,
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("depth z [m]")
    ax.set_title("Two-layer controlled elastic-DAS benchmark")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_gather(
    *,
    time_s: np.ndarray,
    data: np.ndarray,
    s_m: np.ndarray,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    vmax = _robust_limit(data)
    fig, ax = plt.subplots(figsize=(10.0, 6.5))
    ax.imshow(
        data,
        origin="upper",
        aspect="auto",
        extent=[float(time_s[0]), float(time_s[-1]), float(s_m[-1]), float(s_m[0])],
        vmin=-vmax,
        vmax=vmax,
        cmap="seismic",
        interpolation="none",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/validation/two_layer_numba"),
    )
    parser.add_argument(
        "--snapshot-stride",
        type=int,
        default=0,
        help="0 disables snapshots; positive values save vz every N time steps.",
    )
    parser.add_argument(
        "--benchmark-warm",
        action="store_true",
        help="Repeat the true model after JIT compilation and report warm timing.",
    )
    parser.add_argument(
        "--initial-transition-m",
        type=float,
        default=100.0,
        help="Half-width of tanh smoothing in the controlled starting model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    problem = TwoLayerFWIProblem.default()
    print(problem.summary())

    min_s_wavelength = min(
        problem.model_spec.top.vs_m_s,
        problem.model_spec.bottom.vs_m_s,
    ) / problem.source_spec.f0_hz
    print(f"\nminimum S wavelength : {min_s_wavelength:.2f} m")
    print(f"S points/wavelength  : {min_s_wavelength / problem.grid.dx:.2f}")

    _plot_model(problem, args.out_dir / "model.png")

    snapshot_stride = args.snapshot_stride if args.snapshot_stride > 0 else None

    print("\nRunning TRUE two-layer model with Numba...")
    cold_start = perf_counter()
    true_result = problem.run_true(snapshot_stride=snapshot_stride)
    cold_wall = perf_counter() - cold_start
    print(f"forward elapsed reported : {true_result.elapsed_s:.3f} s")
    print(f"outer wall time          : {cold_wall:.3f} s")

    if args.benchmark_warm:
        print("\nRepeating TRUE model for warm-kernel timing...")
        warm_start = perf_counter()
        warm_result = problem.run_true(snapshot_stride=None)
        warm_wall = perf_counter() - warm_start
        warm_error = max(
            _relative_l2(true_result.wavefield.receiver_vx, warm_result.wavefield.receiver_vx),
            _relative_l2(true_result.wavefield.receiver_vz, warm_result.wavefield.receiver_vz),
        )
        print(f"warm forward elapsed     : {warm_result.elapsed_s:.3f} s")
        print(f"warm outer wall          : {warm_wall:.3f} s")
        print(f"cold/warm speed ratio    : {cold_wall / max(warm_wall, 1.0e-12):.2f}")
        print(f"repeatability rel. error : {warm_error:.3e}")
        if warm_error > 1.0e-12:
            raise RuntimeError("Repeated Numba forward run is not reproducible.")
    else:
        warm_wall = np.nan

    print("\nRunning SMOOTH initial model with identical acquisition...")
    initial_medium = problem.smooth_initial_medium(
        transition_half_width_m=args.initial_transition_m,
    )
    initial_result = problem.engine.run(initial_medium, problem.source)

    vx_mismatch = _relative_l2(
        true_result.wavefield.receiver_vx,
        initial_result.wavefield.receiver_vx,
    )
    vz_mismatch = _relative_l2(
        true_result.wavefield.receiver_vz,
        initial_result.wavefield.receiver_vz,
    )
    das_mismatch = _relative_l2(true_result.das.data, initial_result.das.data)

    print("\nControlled model mismatch")
    print("-------------------------")
    print(f"receiver Vx relative L2 : {vx_mismatch:.6f}")
    print(f"receiver Vz relative L2 : {vz_mismatch:.6f}")
    print(f"DAS relative L2         : {das_mismatch:.6f}")

    for name, array in (
        ("true receiver_vx", true_result.wavefield.receiver_vx),
        ("true receiver_vz", true_result.wavefield.receiver_vz),
        ("true DAS", true_result.das.data),
        ("initial receiver_vx", initial_result.wavefield.receiver_vx),
        ("initial receiver_vz", initial_result.wavefield.receiver_vz),
        ("initial DAS", initial_result.das.data),
    ):
        if not np.all(np.isfinite(array)):
            raise RuntimeError(f"{name} contains NaN or Inf.")

    _plot_gather(
        time_s=true_result.wavefield.t_v,
        data=true_result.wavefield.receiver_vx,
        s_m=problem.receivers.s,
        title="Two-layer Numba forward: receiver Vx",
        ylabel="Cable coordinate s [m]",
        out_path=args.out_dir / "receiver_vx.png",
    )
    _plot_gather(
        time_s=true_result.wavefield.t_v,
        data=true_result.wavefield.receiver_vz,
        s_m=problem.receivers.s,
        title="Two-layer Numba forward: receiver Vz",
        ylabel="Cable coordinate s [m]",
        out_path=args.out_dir / "receiver_vz.png",
    )

    das_s = problem.receivers.s[true_result.das.channel_indices]
    _plot_gather(
        time_s=true_result.wavefield.t_v,
        data=true_result.das.data,
        s_m=das_s,
        title="Two-layer Numba forward: DAS axial strain-rate",
        ylabel="DAS gauge-centre s [m]",
        out_path=args.out_dir / "das.png",
    )

    np.savez_compressed(
        args.out_dir / "forward_true.npz",
        t_v=true_result.wavefield.t_v,
        receiver_vx=true_result.wavefield.receiver_vx,
        receiver_vz=true_result.wavefield.receiver_vz,
        das_data=true_result.das.data,
        das_channel_indices=true_result.das.channel_indices,
        receiver_x=problem.receivers.x,
        receiver_z=problem.receivers.z,
        receiver_s=problem.receivers.s,
        grid_x=problem.grid.x,
        grid_z=problem.grid.z,
        vp=problem.true_medium.vp,
        vs=problem.true_medium.vs,
        rho=problem.true_medium.rho,
        interface_depth_m=np.array(problem.model_spec.interface_depth_m),
        source_x_m=np.array(problem.source_spec.x_m),
        source_z_m=np.array(problem.source_spec.z_m),
        source_theta_deg=np.array(problem.source_spec.theta_deg),
        source_f0_hz=np.array(problem.source_spec.f0_hz),
        elapsed_s=np.array(true_result.elapsed_s),
    )

    np.savez_compressed(
        args.out_dir / "forward_initial.npz",
        t_v=initial_result.wavefield.t_v,
        receiver_vx=initial_result.wavefield.receiver_vx,
        receiver_vz=initial_result.wavefield.receiver_vz,
        das_data=initial_result.das.data,
        vp=initial_medium.vp,
        vs=initial_medium.vs,
        rho=initial_medium.rho,
        transition_half_width_m=np.array(args.initial_transition_m),
        elapsed_s=np.array(initial_result.elapsed_s),
    )

    summary = (
        problem.summary()
        + "\n\nResults\n"
        + f"  true elapsed        : {true_result.elapsed_s:.6f} s\n"
        + f"  warm wall           : {warm_wall:.6f} s\n"
        + f"  initial elapsed     : {initial_result.elapsed_s:.6f} s\n"
        + f"  Vx mismatch         : {vx_mismatch:.8e}\n"
        + f"  Vz mismatch         : {vz_mismatch:.8e}\n"
        + f"  DAS mismatch        : {das_mismatch:.8e}\n"
    )
    (args.out_dir / "summary.txt").write_text(summary)

    print(f"\nSaved results to: {args.out_dir.resolve()}")
    print("Two-layer Numba benchmark PASSED.")


if __name__ == "__main__":
    main()
