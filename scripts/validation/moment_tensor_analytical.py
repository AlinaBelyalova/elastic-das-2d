# ==============================================================================
# scripts/validation/moment_tensor_analytical.py
#
# Independent homogeneous-medium validation of the production 2D elastic
# moment-tensor forward solver against an analytical line-source reference.
#
# This test is deliberately separate from SAFOD and FWI. It checks:
#   1. P/S kinematics in a homogeneous full space;
#   2. moment-tensor radiation pattern and relative polarity;
#   3. whether solver receiver velocity corresponds to analytical U, V, or A;
#   4. one global amplitude factor (reported, not silently normalised per trace);
#   5. convergence of the numerical y integration used for the 2D reference.
#
# Absolute 2D source amplitude is reported but not used as a hard pass/fail
# because line-source moment units and the solver's discrete cell-area source
# normalisation must be audited separately.
# ==============================================================================

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.analytical_moment_tensor import (
    kinematics_2d,
    ricker_moment_derivative,
)
from src.grid import Grid2D
from src.model import ElasticModel2D
from src.receivers import build_receivers_from_channel_centres
from src.simulator import run_forward_simulation
from src.solver_numpy import max_stable_dt
from src.source import build_dc_source


# Physical and numerical defaults. These deliberately give many grid points per
# minimum S wavelength and leave the direct-wave comparison window before the
# first physical-boundary reflection.
VP_M_S = 4000.0
VS_M_S = 2300.0
RHO_KG_M3 = 2500.0
F0_HZ = 10.0
SCALAR_MOMENT = 1.0e12
THETA_DEG = 35.0

DX_M = 5.0
DZ_M = 5.0
DOMAIN_WIDTH_M = 2500.0
DURATION_S = 0.56
HALF_ORDER = 2
CFL_SAFETY = 0.80

# The production solver requires n_boundary > half_order even when damping is
# disabled, because this width also protects the finite-difference stencil at
# the outer grid edge.  gamma_s=0 keeps the mask identically one, so this is
# still a full-space comparison before the first physical-boundary return.
VALIDATION_N_BOUNDARY = HALF_ORDER + 1
VALIDATION_GAMMA_S = 0.0

RECEIVER_RADIUS_M = 500.0
N_RECEIVERS = 12
GAUGE_LENGTH_M = 100.0

POINTS_PER_S_WAVELENGTH = 20.0
LINE_EXTENT_FACTOR = 1.25

OUT_DIR = Path("results/validation/moment_tensor_analytical")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the 2D moment-tensor solver against an analytical line source."
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--dx", type=float, default=DX_M)
    parser.add_argument("--duration", type=float, default=DURATION_S)
    parser.add_argument("--radius", type=float, default=RECEIVER_RADIUS_M)
    parser.add_argument("--n-receivers", type=int, default=N_RECEIVERS)
    parser.add_argument("--theta-deg", type=float, default=THETA_DEG)
    parser.add_argument("--f0", type=float, default=F0_HZ)
    parser.add_argument(
        "--skip-analytical-convergence",
        action="store_true",
        help="Skip the refined y-integration comparison.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail unless analytical velocity is the best candidate and QC thresholds pass.",
    )
    return parser.parse_args()


def build_homogeneous_model(grid: Grid2D) -> ElasticModel2D:
    shape = grid.shape
    return ElasticModel2D(
        grid=grid,
        vp=np.full(shape, VP_M_S, dtype=np.float64),
        vs=np.full(shape, VS_M_S, dtype=np.float64),
        rho=np.full(shape, RHO_KG_M3, dtype=np.float64),
    )


def build_receiver_ring(
    *,
    x_source_m: float,
    z_source_m: float,
    radius_m: float,
    n_receivers: int,
):
    if n_receivers < 8:
        raise ValueError("n_receivers must be >= 8 to sample the radiation pattern.")
    phi = np.arange(n_receivers, dtype=np.float64) * 2.0 * np.pi / n_receivers
    x = x_source_m + radius_m * np.cos(phi)
    z = z_source_m + radius_m * np.sin(phi)
    s = radius_m * phi
    receivers = build_receivers_from_channel_centres(x=x, z=z, s=s)
    return receivers, phi


def physical_moment_tensor_from_source(source) -> np.ndarray:
    """Extract the exact in-plane tensor actually passed to the solver."""
    tensor = np.array(
        [
            [float(source.m2d.Mxx), 0.0, float(source.m2d.Mxz)],
            [0.0, 0.0, 0.0],
            [float(source.m2d.Mxz), 0.0, float(source.m2d.Mzz)],
        ],
        dtype=np.float64,
    )
    if not np.allclose(tensor, tensor.T, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("Extracted source moment tensor is not symmetric.")
    return tensor


def compute_analytical_ring(
    *,
    receivers,
    x_source_m: float,
    z_source_m: float,
    time_s: np.ndarray,
    f0_hz: float,
    t0_s: float,
    moment_tensor_nm: np.ndarray,
    points_per_s_wavelength: float,
    line_extent_factor: float,
) -> tuple[dict[str, np.ndarray], list]:
    nrec = receivers.nrec
    nt = time_s.size
    result = {
        "displacement": np.empty((nrec, 2, nt), dtype=np.float64),
        "velocity": np.empty((nrec, 2, nt), dtype=np.float64),
        "acceleration": np.empty((nrec, 2, nt), dtype=np.float64),
    }
    integration_info = []

    for irec in range(nrec):
        print(f"  analytical receiver {irec + 1:02d}/{nrec}", flush=True)
        kinematics, info = kinematics_2d(
            vp_m_s=VP_M_S,
            vs_m_s=VS_M_S,
            rho_kg_m3=RHO_KG_M3,
            x_m=float(receivers.x[irec] - x_source_m),
            z_m=float(receivers.z[irec] - z_source_m),
            time_s=time_s,
            f0_hz=f0_hz,
            moment_tensor_nm=moment_tensor_nm,
            t0_s=t0_s,
            points_per_s_wavelength=points_per_s_wavelength,
            line_extent_factor=line_extent_factor,
            chunk_size=128,
        )
        for name in result:
            result[name][irec] = kinematics[name]
        integration_info.append(info)

    return result, integration_info


def comparison_mask(
    *,
    time_s: np.ndarray,
    radius_m: float,
    t0_s: float,
    f0_hz: float,
) -> np.ndarray:
    # Include P and S, but stop before the expected first physical-boundary
    # return for the default geometry.
    start = max(0.0, t0_s + radius_m / VP_M_S - 1.5 / f0_hz)
    stop = min(float(time_s[-1]), t0_s + radius_m / VS_M_S + 2.0 / f0_hz)
    mask = (time_s >= start) & (time_s <= stop)
    if np.count_nonzero(mask) < 20:
        raise RuntimeError("Comparison window is too short.")
    print(f"comparison window : {start:.4f} to {stop:.4f} s")
    return mask


def metric_for_lag(
    fd: np.ndarray,
    analytical: np.ndarray,
    time_s: np.ndarray,
    mask: np.ndarray,
    lag_samples: float,
) -> dict:
    """
    Fit one signed global scale after a possibly fractional-sample lag.

    Positive lag means the analytical trace is evaluated at a later time than
    FD. Interpolation avoids hiding a half-time-step staggering error inside an
    integer-lag metric.
    """
    shifted_analytical = shifted_and_scaled(
        analytical,
        time_s=time_s,
        lag_samples=float(lag_samples),
        scale=1.0,
    )

    fd_vector = fd[:, :, mask].reshape(-1)
    an_vector = shifted_analytical[:, :, mask].reshape(-1)

    denominator = float(np.vdot(an_vector, an_vector))
    if denominator <= 0.0:
        raise RuntimeError("Analytical candidate has zero norm in comparison window.")

    scale = float(np.vdot(an_vector, fd_vector) / denominator)
    fitted = scale * an_vector

    fd_norm = float(np.linalg.norm(fd_vector))
    fitted_norm = float(np.linalg.norm(fitted))
    relative_error = float(np.linalg.norm(fd_vector - fitted) / max(fd_norm, 1.0e-30))
    correlation = float(
        np.vdot(fd_vector, fitted)
        / max(fd_norm * fitted_norm, 1.0e-30)
    )
    raw_correlation = float(
        np.vdot(fd_vector, an_vector)
        / max(fd_norm * float(np.linalg.norm(an_vector)), 1.0e-30)
    )

    component_correlation = []
    for icomp in range(2):
        f = fd[:, icomp, :][:, mask].reshape(-1)
        a = shifted_analytical[:, icomp, :][:, mask].reshape(-1) * scale
        component_correlation.append(
            float(
                np.vdot(f, a)
                / max(np.linalg.norm(f) * np.linalg.norm(a), 1.0e-30)
            )
        )

    return {
        "lag_samples": float(lag_samples),
        "scale_analytical_to_fd": scale,
        "relative_error": relative_error,
        "correlation": correlation,
        "raw_correlation": raw_correlation,
        "vx_correlation": component_correlation[0],
        "vz_correlation": component_correlation[1],
    }


def evaluate_candidate(
    *,
    fd: np.ndarray,
    analytical: np.ndarray,
    time_s: np.ndarray,
    mask: np.ndarray,
    max_lag_samples: float = 4.0,
    lag_step_samples: float = 0.25,
) -> dict:
    lags = np.arange(
        -float(max_lag_samples),
        float(max_lag_samples) + 0.5 * float(lag_step_samples),
        float(lag_step_samples),
    )
    trials = [
        metric_for_lag(fd, analytical, time_s, mask, float(lag))
        for lag in lags
    ]
    return min(trials, key=lambda item: item["relative_error"])


def shifted_and_scaled(
    analytical: np.ndarray,
    *,
    time_s: np.ndarray,
    lag_samples: float,
    scale: float,
) -> np.ndarray:
    shift_s = float(lag_samples) * float(np.median(np.diff(time_s)))
    output = np.empty_like(analytical)
    for irec in range(analytical.shape[0]):
        for icomp in range(analytical.shape[1]):
            output[irec, icomp] = scale * np.interp(
                time_s + shift_s,
                time_s,
                analytical[irec, icomp],
                left=0.0,
                right=0.0,
            )
    return output


def peak_time(
    energy: np.ndarray,
    time_s: np.ndarray,
    *,
    centre_s: float,
    half_width_s: float,
) -> float:
    mask = np.abs(time_s - centre_s) <= half_width_s
    if not np.any(mask):
        return float("nan")
    local_indices = np.flatnonzero(mask)
    return float(time_s[local_indices[np.argmax(energy[mask])]])


def save_overlay(
    *,
    time_s: np.ndarray,
    fd: np.ndarray,
    analytical_fitted: np.ndarray,
    receiver_phi: np.ndarray,
    candidate_name: str,
    out_path: Path,
) -> None:
    receiver_ids = np.linspace(0, fd.shape[0] - 1, 4, dtype=int)
    fig, axes = plt.subplots(4, 2, figsize=(12, 11), sharex=True)
    for row, irec in enumerate(receiver_ids):
        for component in range(2):
            ax = axes[row, component]
            ax.plot(time_s, fd[irec, component], label="FD", lw=1.1)
            ax.plot(
                time_s,
                analytical_fitted[irec, component],
                "--",
                label=f"analytical {candidate_name}",
                lw=1.1,
            )
            ax.set_ylabel("Vx" if component == 0 else "Vz")
            ax.set_title(
                f"receiver {irec}, azimuth={np.degrees(receiver_phi[irec]):.0f} deg"
            )
            ax.grid(alpha=0.25)
            if row == 0 and component == 0:
                ax.legend(fontsize=8)
    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Time [s]")
    fig.suptitle("Moment-tensor analytical validation: one global scale")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.dx <= 0.0 or args.duration <= 0.0 or args.radius <= 0.0:
        raise ValueError("dx, duration, and radius must be positive.")

    nx = int(round(DOMAIN_WIDTH_M / args.dx)) + 1
    nz = nx
    if nx % 2 == 0:
        nx += 1
        nz += 1

    dt = max_stable_dt(
        vp_max=VP_M_S,
        dx=args.dx,
        dz=args.dx,
        half_order=HALF_ORDER,
        safety=CFL_SAFETY,
    )
    nt = int(np.ceil(args.duration / dt)) + 1

    grid = Grid2D(nx=nx, nz=nz, dx=args.dx, dz=args.dx, nt=nt, dt=dt)
    model = build_homogeneous_model(grid)

    ix_centre = grid.nx // 2
    iz_centre = grid.nz // 2
    x_source = float(grid.x[ix_centre] + 0.37 * grid.dx)
    z_source = float(grid.z[iz_centre] + 0.61 * grid.dz)

    source = build_dc_source(
        grid=grid,
        x_m=x_source,
        z_m=z_source,
        theta_deg=args.theta_deg,
        scalar_moment=SCALAR_MOMENT,
        nt=grid.nt,
        dt=grid.dt,
        f0_hz=args.f0,
        source_time_mode="ricker_moment",
        spreading="bilinear",
    )
    t0_s = float(getattr(source.stf, "t0", 1.2 / args.f0))
    moment_tensor = physical_moment_tensor_from_source(source)

    receivers, receiver_phi = build_receiver_ring(
        x_source_m=x_source,
        z_source_m=z_source,
        radius_m=args.radius,
        n_receivers=args.n_receivers,
    )

    source_to_boundary = min(
        x_source - float(grid.x[0]),
        float(grid.x[-1]) - x_source,
        z_source - float(grid.z[0]),
        float(grid.z[-1]) - z_source,
    )
    earliest_boundary_return = (
        t0_s
        + (source_to_boundary + max(source_to_boundary - args.radius, 0.0)) / VP_M_S
    )

    print("\nConfiguration")
    print("-------------")
    print(f"grid                 : {grid.nx} x {grid.nz}")
    print(f"dx, dt, nt           : {grid.dx:.3f} m, {grid.dt:.6e} s, {grid.nt}")
    print(f"duration             : {(grid.nt - 1) * grid.dt:.4f} s")
    print(f"minimum S wavelength : {VS_M_S / args.f0:.2f} m")
    print(f"S points/wavelength  : {VS_M_S / args.f0 / grid.dx:.2f}")
    print(f"source requested     : {x_source:.3f}, {z_source:.3f} m")
    print(f"source embedded      : {source.x_embedded_m:.3f}, {source.z_embedded_m:.3f} m")
    print(f"receiver radius      : {args.radius:.2f} m")
    print(f"receivers            : {receivers.nrec}")
    print(f"analytical t0        : {t0_s:.6f} s")
    print(f"source STF kind      : {source.stf.kind}")
    print(
        "source STF sampling  : "
        f"t=(n+{source.stf.time_offset_steps:.1f})dt"
    )
    print(f"boundary width       : {VALIDATION_N_BOUNDARY} cells")
    print(f"boundary damping     : gamma={VALIDATION_GAMMA_S:.1f} (disabled)")
    print(f"boundary-return QC   : about {earliest_boundary_return:.4f} s")
    print("moment tensor [N m]:")
    print(moment_tensor)

    if float((grid.nt - 1) * grid.dt) >= earliest_boundary_return:
        raise RuntimeError(
            "Record reaches the estimated first physical-boundary return. "
            "Increase the domain or shorten --duration."
        )

    print("\nRunning production FD solver...")
    run_result, _ = run_forward_simulation(
        model=model,
        source=source,
        receivers=receivers,
        gauge_length_m=GAUGE_LENGTH_M,
        half_order=HALF_ORDER,
        use_ts_sfd=False,
        n_boundary=VALIDATION_N_BOUNDARY,
        gamma_s=VALIDATION_GAMMA_S,
        snapshot_stride=None,
        backend="numba_fused",
        free_surface=False,
    )

    time_s = np.asarray(run_result.t, dtype=np.float64)
    fd = np.stack(
        (
            np.asarray(run_result.receiver_vx, dtype=np.float64),
            np.asarray(run_result.receiver_vz, dtype=np.float64),
        ),
        axis=1,
    )
    if fd.shape != (receivers.nrec, 2, grid.nt):
        raise RuntimeError(f"Unexpected FD receiver shape: {fd.shape}.")
    if not np.all(np.isfinite(fd)):
        raise RuntimeError("FD output contains NaN or Inf.")

    # Verify the discrete physical source convention exactly:
    # the stress equation receives -dW/dt at the stress-update midpoint.
    source_time = np.asarray(source.stf.t, dtype=np.float64)
    expected_stf = -ricker_moment_derivative(
        source_time,
        f0_hz=args.f0,
        t0_s=t0_s,
    )
    source_values = np.asarray(source.stf.values, dtype=np.float64)
    stf_relative_error = float(
        np.linalg.norm(source_values - expected_stf)
        / max(np.linalg.norm(expected_stf), 1.0e-30)
    )
    print(f"source moment-rate STF error: {stf_relative_error:.3e}")

    print("\nComputing analytical line-source reference...")
    analytical, integration_info = compute_analytical_ring(
        receivers=receivers,
        x_source_m=source.x_embedded_m,
        z_source_m=source.z_embedded_m,
        time_s=time_s,
        f0_hz=args.f0,
        t0_s=t0_s,
        moment_tensor_nm=moment_tensor,
        points_per_s_wavelength=POINTS_PER_S_WAVELENGTH,
        line_extent_factor=LINE_EXTENT_FACTOR,
    )
    print(
        "y integration        : "
        f"n_y={integration_info[0].n_y}, "
        f"dy={integration_info[0].dy_m:.3f} m, "
        f"ymax={integration_info[0].y_max_m:.1f} m"
    )

    mask = comparison_mask(
        time_s=time_s,
        radius_m=args.radius,
        t0_s=t0_s,
        f0_hz=args.f0,
    )

    metrics = []
    for candidate_name in ("displacement", "velocity", "acceleration"):
        metric = evaluate_candidate(
            fd=fd,
            analytical=analytical[candidate_name],
            time_s=time_s,
            mask=mask,
            max_lag_samples=4.0,
            lag_step_samples=0.25,
        )
        metric["candidate"] = candidate_name
        metrics.append(metric)

    metrics.sort(key=lambda item: item["relative_error"])
    best = metrics[0]

    print("\nSource-convention candidates")
    print("----------------------------")
    for item in metrics:
        print(
            f"{item['candidate']:12s}: "
            f"corr={item['correlation']:.5f}, "
            f"relerr={item['relative_error']:.5f}, "
            f"lag={item['lag_samples']:+.2f} samples, "
            f"scale={item['scale_analytical_to_fd']:.6e}, "
            f"rawcorr={item['raw_correlation']:.5f}"
        )

    analytical_convergence_error = float("nan")
    if not args.skip_analytical_convergence:
        print("\nComputing refined analytical reference...")
        refined, _ = compute_analytical_ring(
            receivers=receivers,
            x_source_m=source.x_embedded_m,
            z_source_m=source.z_embedded_m,
            time_s=time_s,
            f0_hz=args.f0,
            t0_s=t0_s,
            moment_tensor_nm=moment_tensor,
            points_per_s_wavelength=1.5 * POINTS_PER_S_WAVELENGTH,
            line_extent_factor=1.15 * LINE_EXTENT_FACTOR,
        )
        coarse = analytical["velocity"][:, :, mask]
        fine = refined["velocity"][:, :, mask]
        analytical_convergence_error = float(
            np.linalg.norm(fine - coarse) / max(np.linalg.norm(fine), 1.0e-30)
        )
        print(
            "analytical convergence error: "
            f"{analytical_convergence_error:.5e}"
        )

    # Aggregate-energy P/S timing is used only as a kinematic diagnostic. A
    # double-couple may suppress individual receivers near nodal directions.
    fd_energy = np.sum(fd**2, axis=(0, 1))
    best_fitted = shifted_and_scaled(
        analytical[best["candidate"]],
        time_s=time_s,
        lag_samples=best["lag_samples"],
        scale=best["scale_analytical_to_fd"],
    )
    analytical_energy = np.sum(best_fitted**2, axis=(0, 1))

    predicted_p = t0_s + args.radius / VP_M_S
    predicted_s = t0_s + args.radius / VS_M_S
    half_window = 0.45 / args.f0
    fd_p_peak = peak_time(fd_energy, time_s, centre_s=predicted_p, half_width_s=half_window)
    an_p_peak = peak_time(
        analytical_energy, time_s, centre_s=predicted_p, half_width_s=half_window
    )
    fd_s_peak = peak_time(fd_energy, time_s, centre_s=predicted_s, half_width_s=half_window)
    an_s_peak = peak_time(
        analytical_energy, time_s, centre_s=predicted_s, half_width_s=half_window
    )

    print("\nBest comparison")
    print("---------------")
    print(f"best candidate       : {best['candidate']}")
    print(f"global correlation   : {best['correlation']:.6f}")
    print(f"relative L2 error    : {best['relative_error']:.6f}")
    print(f"Vx / Vz correlation : {best['vx_correlation']:.6f} / {best['vz_correlation']:.6f}")
    print(f"lag                  : {best['lag_samples']:+.2f} samples")
    print(f"signed global scale  : {best['scale_analytical_to_fd']:.6e}")
    print(f"P peak FD / analytic : {fd_p_peak:.6f} / {an_p_peak:.6f} s")
    print(f"S peak FD / analytic : {fd_s_peak:.6f} / {an_s_peak:.6f} s")
    if best["scale_analytical_to_fd"] < 0.0:
        print("WARNING: one global polarity reversal is required.")

    with open(args.out_dir / "candidate_metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)

    np.savez_compressed(
        args.out_dir / "validation_arrays.npz",
        time_s=time_s,
        receiver_x=receivers.x,
        receiver_z=receivers.z,
        receiver_phi_rad=receiver_phi,
        fd_vx=fd[:, 0],
        fd_vz=fd[:, 1],
        analytical_displacement=analytical["displacement"],
        analytical_velocity=analytical["velocity"],
        analytical_acceleration=analytical["acceleration"],
        analytical_best_fitted=best_fitted,
        comparison_mask=mask,
        moment_tensor_nm=moment_tensor,
        best_candidate=np.array(best["candidate"]),
        best_lag_samples=np.array(best["lag_samples"], dtype=np.float64),
        best_scale=np.array(best["scale_analytical_to_fd"]),
        best_correlation=np.array(best["correlation"]),
        best_relative_error=np.array(best["relative_error"]),
        source_stf_relative_error=np.array(stf_relative_error),
        analytical_convergence_error=np.array(analytical_convergence_error),
    )

    save_overlay(
        time_s=time_s,
        fd=fd,
        analytical_fitted=best_fitted,
        receiver_phi=receiver_phi,
        candidate_name=best["candidate"],
        out_path=args.out_dir / "velocity_overlay.png",
    )

    strict_failures = []
    if best["candidate"] != "velocity":
        strict_failures.append(
            f"best analytical candidate is {best['candidate']!r}, not 'velocity'"
        )
    if best["raw_correlation"] <= 0.0:
        strict_failures.append(
            "raw correlation is non-positive, indicating a global polarity error"
        )
    if best["scale_analytical_to_fd"] <= 0.0:
        strict_failures.append(
            "signed global scale is non-positive"
        )
    if abs(best["scale_analytical_to_fd"] - 1.0) > 0.03:
        strict_failures.append(
            "global amplitude scale differs from one by more than 3%: "
            f"{best['scale_analytical_to_fd']:.6f}"
        )
    if best["correlation"] < 0.999:
        strict_failures.append(
            f"correlation {best['correlation']:.6f} < 0.999"
        )
    if best["relative_error"] > 0.02:
        strict_failures.append(
            f"relative error {best['relative_error']:.6f} > 0.02"
        )
    if abs(best["lag_samples"]) > 1.0:
        strict_failures.append(
            f"|lag|={abs(best['lag_samples']):.2f} > 1 sample"
        )
    if stf_relative_error > 1.0e-12:
        strict_failures.append(
            f"source STF mismatch {stf_relative_error:.3e} > 1e-12"
        )
    if (
        np.isfinite(analytical_convergence_error)
        and analytical_convergence_error > 0.02
    ):
        strict_failures.append(
            "analytical convergence error "
            f"{analytical_convergence_error:.4f} > 0.02"
        )

    print(f"\nSaved validation results to: {args.out_dir.resolve()}")
    if strict_failures:
        print("QC warnings:")
        for failure in strict_failures:
            print(f"  - {failure}")
        if args.strict:
            raise RuntimeError("Strict analytical validation failed.")
    else:
        print("Moment-tensor analytical validation PASSED.")


if __name__ == "__main__":
    main()