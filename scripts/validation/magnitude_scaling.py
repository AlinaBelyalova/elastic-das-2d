from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

from src.grid import Grid2D
from src.model import ElasticModel2D
from src.source import build_dc_source
from src.receivers import build_das_cable
from src.simulator import run_forward_simulation
from src.solver_numpy import max_stable_dt


def mw_to_m0(mw: float) -> float:
    """
    Convert moment magnitude Mw to scalar seismic moment M0 [N m].

    Standard relation:
        log10(M0) = 1.5 Mw + 9.1
    """
    return float(10.0 ** (1.5 * mw + 9.1))


def peak_envelope_amplitude(trace: np.ndarray) -> float:
    trace = np.asarray(trace, dtype=np.float64)
    env = np.abs(hilbert(trace))
    return float(np.max(env))


def build_model(
    *,
    nx: int = 301,
    nz: int = 251,
    nt: int = 1000,
    dx: float = 10.0,
    dz: float = 10.0,
    vp: float = 3000.0,
    vs: float = 1700.0,
    rho: float = 2500.0,
    half_order: int = 2,
    cfl_safety: float = 0.85,
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


def build_geometry(model: ElasticModel2D, *, n_boundary: int):
    grid = model.grid

    x_src = grid.x[grid.nx // 3]
    z_src = grid.z[grid.nz // 2]

    ix_cable = grid.nx - n_boundary - 25
    iz_top = n_boundary + 10
    iz_bot = grid.nz - n_boundary - 10

    receivers = build_das_cable(
        grid=grid,
        waypoints_x=[grid.x[ix_cable], grid.x[ix_cable]],
        waypoints_z=[grid.z[iz_top], grid.z[iz_bot]],
        channel_spacing_m=10.0,
        n_pml=n_boundary,
    )

    return x_src, z_src, receivers


def run_for_magnitude(
    *,
    model: ElasticModel2D,
    x_src: float,
    z_src: float,
    receivers,
    mw: float,
    backend: str,
    free_surface: bool,
    half_order: int,
    n_boundary: int,
):
    m0 = mw_to_m0(mw)

    source = build_dc_source(
        grid=model.grid,
        x_m=x_src,
        z_m=z_src,
        theta_deg=0.0,
        scalar_moment=m0,
        nt=model.grid.nt,
        dt=model.grid.dt,
        f0_hz=8.0,
        derivative_order=0,
    )

    run_result, das_result = run_forward_simulation(
        model=model,
        source=source,
        receivers=receivers,
        gauge_length_m=20.0,
        half_order=half_order,
        use_ts_sfd=False,
        n_boundary=n_boundary,
        gamma_s=50.0,
        snapshot_stride=None,
        backend=backend,
        free_surface=free_surface,
    )

    mid = receivers.nrec // 2

    amp_vx = peak_envelope_amplitude(run_result.receiver_vx[mid])
    amp_vz = peak_envelope_amplitude(run_result.receiver_vz[mid])

    # Use a DAS channel near the middle of the valid gauge-output array.
    mid_das = das_result.nchan_out // 2
    amp_das = peak_envelope_amplitude(das_result.data[mid_das])

    return {
        "mw": float(mw),
        "m0_nm": float(m0),
        "amp_vx": amp_vx,
        "amp_vz": amp_vz,
        "amp_das": amp_das,
        "trace_vx": run_result.receiver_vx[mid].copy(),
        "trace_vz": run_result.receiver_vz[mid].copy(),
        "trace_das": das_result.data[mid_das].copy(),
        "t_v": run_result.t_v.copy(),
    }


def fit_slope(mw_values: np.ndarray, amplitudes: np.ndarray) -> tuple[float, float]:
    log_amp = np.log10(amplitudes)
    slope, intercept = np.polyfit(mw_values, log_amp, deg=1)
    return float(slope), float(intercept)


def main() -> None:
    outdir = Path("results/validation_magnitude_scaling")
    outdir.mkdir(parents=True, exist_ok=True)

    backend = "numba_fused"
    free_surface = True
    half_order = 2
    n_boundary = 50

    # Use integer Mw values so the expected amplitude ratio is exactly 10^1.5.
    mw_values = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)

    model = build_model(half_order=half_order)
    x_src, z_src, receivers = build_geometry(model, n_boundary=n_boundary)

    print("Magnitude scaling validation")
    print(f"Backend       : {backend}")
    print(f"free_surface  : {free_surface}")
    print(f"Source        : x={x_src:.1f} m, z={z_src:.1f} m")
    print(f"Receivers     : {receivers.nrec}")
    print(f"dt            : {model.grid.dt:.6e} s")
    print(f"Mw values     : {mw_values}")

    outputs = []
    for mw in mw_values:
        print(f"\nRunning Mw={mw:.1f} ...")
        out = run_for_magnitude(
            model=model,
            x_src=x_src,
            z_src=z_src,
            receivers=receivers,
            mw=float(mw),
            backend=backend,
            free_surface=free_surface,
            half_order=half_order,
            n_boundary=n_boundary,
        )
        outputs.append(out)
        print(f"  M0      = {out['m0_nm']:.6e} N m")
        print(f"  amp_vx  = {out['amp_vx']:.6e}")
        print(f"  amp_vz  = {out['amp_vz']:.6e}")
        print(f"  amp_DAS = {out['amp_das']:.6e}")

    amp_vx = np.array([o["amp_vx"] for o in outputs], dtype=np.float64)
    amp_vz = np.array([o["amp_vz"] for o in outputs], dtype=np.float64)
    amp_das = np.array([o["amp_das"] for o in outputs], dtype=np.float64)
    m0_values = np.array([o["m0_nm"] for o in outputs], dtype=np.float64)

    slope_vx, intercept_vx = fit_slope(mw_values, amp_vx)
    slope_vz, intercept_vz = fit_slope(mw_values, amp_vz)
    slope_das, intercept_das = fit_slope(mw_values, amp_das)

    expected_slope = 1.5
    expected_ratio = 10.0 ** 1.5

    ratio_vx = amp_vx[1:] / amp_vx[:-1]
    ratio_vz = amp_vz[1:] / amp_vz[:-1]
    ratio_das = amp_das[1:] / amp_das[:-1]

    metrics = {
        "backend": backend,
        "free_surface": bool(free_surface),
        "mw_values": mw_values.tolist(),
        "m0_values_nm": m0_values.tolist(),
        "amp_vx": amp_vx.tolist(),
        "amp_vz": amp_vz.tolist(),
        "amp_das": amp_das.tolist(),
        "slope_vx": slope_vx,
        "slope_vz": slope_vz,
        "slope_das": slope_das,
        "intercept_vx": intercept_vx,
        "intercept_vz": intercept_vz,
        "intercept_das": intercept_das,
        "expected_slope": expected_slope,
        "expected_ratio_per_magnitude": expected_ratio,
        "ratio_vx": ratio_vx.tolist(),
        "ratio_vz": ratio_vz.tolist(),
        "ratio_das": ratio_das.tolist(),
    }

    out_json = outdir / "magnitude_scaling_metrics.json"
    with out_json.open("w") as f:
        json.dump(metrics, f, indent=2)

    print("\nFitted slopes log10(amplitude) vs Mw:")
    print(f"  vx  slope = {slope_vx:.6f}")
    print(f"  vz  slope = {slope_vz:.6f}")
    print(f"  DAS slope = {slope_das:.6f}")
    print(f"  expected  = {expected_slope:.6f}")

    print("\nAmplitude ratios for ΔMw=1:")
    print(f"  vx  ratios = {ratio_vx}")
    print(f"  vz  ratios = {ratio_vz}")
    print(f"  DAS ratios = {ratio_das}")
    print(f"  expected   = {expected_ratio:.6f}")

    # Save traces.
    np.savez_compressed(
        outdir / "magnitude_scaling_traces.npz",
        t_v=outputs[0]["t_v"],
        mw_values=mw_values,
        m0_values=m0_values,
        amp_vx=amp_vx,
        amp_vz=amp_vz,
        amp_das=amp_das,
        traces_vx=np.stack([o["trace_vx"] for o in outputs], axis=0),
        traces_vz=np.stack([o["trace_vz"] for o in outputs], axis=0),
        traces_das=np.stack([o["trace_das"] for o in outputs], axis=0),
    )

    # Plot amplitude scaling.
    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    ax.plot(mw_values, np.log10(amp_vx), "o-", label=f"receiver vx, slope={slope_vx:.3f}")
    ax.plot(mw_values, np.log10(amp_vz), "s-", label=f"receiver vz, slope={slope_vz:.3f}")
    ax.plot(mw_values, np.log10(amp_das), "^-", label=f"DAS, slope={slope_das:.3f}")

    # Reference slope 1.5 through first DAS point.
    ref = np.log10(amp_das[0]) + expected_slope * (mw_values - mw_values[0])
    ax.plot(mw_values, ref, "k--", lw=1.5, label="expected slope 1.5")

    ax.set_xlabel(r"Moment magnitude $M_w$")
    ax.set_ylabel(r"$\log_{10}$ peak envelope amplitude")
    ax.set_title("Magnitude scaling check")
    ax.grid(True, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "magnitude_scaling.png", dpi=250)
    plt.close(fig)

    print(f"\nSaved metrics to {out_json}")
    print(f"Saved figure to {outdir / 'magnitude_scaling.png'}")

    slope_errors = np.array(
        [
            abs(slope_vx - expected_slope),
            abs(slope_vz - expected_slope),
            abs(slope_das - expected_slope),
        ],
        dtype=np.float64,
    )

    # This should be very small because the solver is linear in source moment.
    threshold = 1.0e-3

    if np.max(slope_errors) > threshold:
        raise RuntimeError(
            f"Magnitude scaling failed: max slope error={np.max(slope_errors):.3e} > {threshold:.1e}"
        )

    print("Magnitude scaling check PASSED.")


if __name__ == "__main__":
    main()