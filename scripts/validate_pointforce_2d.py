from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.grid import Grid2D
from src.model import ElasticModel2D
from src.sampling import build_receiver_sampling
from src.solver_numpy import max_stable_dt
from src.solver_numpy_pointforce import run_elastic_solver_numpy_pointforce
from src.analytical_2d import analytical_vz_trace_from_pointforce


# ==============================================================================
# 1. HELPERS
# ==============================================================================

def relative_l2(a: np.ndarray, b: np.ndarray, eps: float = 1e-30) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + eps))


def ricker_wavelet(
    *,
    nt: int,
    dt: float,
    f0_hz: float,
    t0: float | None = None,
    amplitude: float = 1.0,
    half_step: bool = False,
) -> np.ndarray:
    """
    Ricker wavelet on integer (half_step=False) or half-step (half_step=True)
    time grid.

    Use half_step=False for analytical_vz_trace_from_pointforce (expects n*dt).
    Use half_step=True  for run_elastic_solver_numpy_pointforce (expects t_v).
    """
    if nt <= 1:
        raise ValueError(f"nt must be > 1, got {nt}.")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}.")
    if f0_hz <= 0.0:
        raise ValueError(f"f0_hz must be positive, got {f0_hz}.")
    if amplitude <= 0.0:
        raise ValueError(f"amplitude must be positive, got {amplitude}.")

    if t0 is None:
        t0 = 1.2 / f0_hz

    offset = 0.5 if half_step else 0.0
    t = (np.arange(nt, dtype=np.float64) + offset) * dt
    x = np.pi * f0_hz * (t - t0)
    return amplitude * (1.0 - 2.0 * x**2) * np.exp(-x**2)


def vz_node_coordinates(grid: Grid2D, ix: int, iz: int) -> tuple[float, float]:
    """Physical coordinates of vz[ix, iz]: x = x0+ix*dx, z = z0+(iz+0.5)*dz."""
    return float(grid.x0 + ix * grid.dx), float(grid.z0 + (iz + 0.5) * grid.dz)


def build_homogeneous_model(
    *,
    nx: int,
    nz: int,
    nt: int,
    dx: float,
    dz: float,
    vp: float,
    vs: float,
    rho: float,
    half_order: int,
    cfl_safety: float,
) -> ElasticModel2D:
    dt = max_stable_dt(
        vp, dx, dz, half_order, safety=cfl_safety, use_ts_sfd=False
    )

    grid = Grid2D(
        nx=nx,
        nz=nz,
        nt=nt,
        dx=dx,
        dz=dz,
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


# ==============================================================================
# 2. MAIN DRIVER
# ==============================================================================

def main() -> None:
    outdir = Path("results/pointforce_validation")
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Validation setup
    # ------------------------------------------------------------------
    nx, nz = 601, 601
    nt = 12000
    dx = dz = 15.0

    vp, vs, rho = 3000.0, 1700.0, 2500.0
    f0_hz = 15.0
    cfl_safety = 0.20
    n_boundary = 40

    half_orders = [1, 2, 3, 4]
    spatial_orders = [2 * ho for ho in half_orders]

    # Build model using highest order only to define a conservative dt
    model = build_homogeneous_model(
        nx=nx,
        nz=nz,
        nt=nt,
        dx=dx,
        dz=dz,
        vp=vp,
        vs=vs,
        rho=rho,
        half_order=max(half_orders),
        cfl_safety=cfl_safety,
    )
    grid = model.grid

    # ------------------------------------------------------------------
    # Source / receiver exactly on vz staggered nodes
    # ------------------------------------------------------------------
    ix_src, iz_src = nx // 2, nz // 2
    ix_rec = ix_src + 150
    iz_rec = iz_src + 150

    x_src, z_src = vz_node_coordinates(grid, ix_src, iz_src)
    x_rec, z_rec = vz_node_coordinates(grid, ix_rec, iz_rec)

    r = float(np.sqrt((x_rec - x_src) ** 2 + (z_rec - z_src) ** 2))
    lambda_p = vp / f0_hz
    t_p = r / vp
    t_s = r / vs

    print("Validation geometry")
    print(f"  Source   (ix,iz) = ({ix_src},{iz_src})  (x,z) = ({x_src:.1f},{z_src:.1f}) m")
    print(f"  Receiver (ix,iz) = ({ix_rec},{iz_rec})  (x,z) = ({x_rec:.1f},{z_rec:.1f}) m")
    print(f"  r = {r:.1f} m = {r / lambda_p:.1f} λ_P")
    print(f"  t_P ≈ {t_p:.3f} s")
    print(f"  t_S ≈ {t_s:.3f} s")
    print(f"  dt = {grid.dt:.4e} s,  nt = {nt},  T_total = {nt * grid.dt:.3f} s")

    if r < 10.0 * lambda_p:
        raise ValueError(
            f"Source-receiver distance {r:.0f} m < 10 λ_P = {10.0 * lambda_p:.0f} m. "
            "Increase the offset for a cleaner asymptotic comparison."
        )

    if nt * grid.dt <= 1.1 * t_p:
        raise ValueError(
            "Recording time is too short: direct P-wave has not reached the receiver."
        )

    # ------------------------------------------------------------------
    # Receiver sampling object for a single receiver
    # ------------------------------------------------------------------
    receivers = type(
        "TmpReceivers",
        (),
        {
            "x": np.array([x_rec], dtype=np.float64),
            "z": np.array([z_rec], dtype=np.float64),
            "nrec": 1,
        },
    )()
    sampling = build_receiver_sampling(grid, receivers)

    # ------------------------------------------------------------------
    # Source wavelets
    #
    # Analytical module expects STF on integer grid t = n*dt.
    # Point-force solver expects force STF on half-step grid t_v = (n+1/2)dt.
    # ------------------------------------------------------------------
    stf_integer = ricker_wavelet(
        nt=nt,
        dt=grid.dt,
        f0_hz=f0_hz,
        amplitude=1.0,
        half_step=False,
    )

    force_stf_half = ricker_wavelet(
        nt=nt,
        dt=grid.dt,
        f0_hz=f0_hz,
        amplitude=1.0,
        half_step=True,
    )

    t_v = (np.arange(nt, dtype=np.float64) + 0.5) * grid.dt

    # ------------------------------------------------------------------
    # Analytical reference on half-step axis
    # ------------------------------------------------------------------
    t_analytic, vz_analytic = analytical_vz_trace_from_pointforce(
        stf=stf_integer,
        dt=grid.dt,
        x_src=x_src,
        z_src=z_src,
        x_rec=x_rec,
        z_rec=z_rec,
        rho=rho,
        vp=vp,
        vs=vs,
        pad_factor=2,
        return_half_step_times=True,
    )

    if not np.allclose(t_analytic, t_v):
        raise RuntimeError("Analytical and numerical half-step time axes do not match.")

    # ------------------------------------------------------------------
    # Numerical runs
    # ------------------------------------------------------------------
    numerical_traces = []
    rel_l2_errors = []
    max_abs_errors = []

    for ho, so in zip(half_orders, spatial_orders):
        print(f"\nRunning spatial order {so} (half_order={ho}) ...")

        result = run_elastic_solver_numpy_pointforce(
            vp=model.vp,
            vs=model.vs,
            rho=model.rho,
            dx=grid.dx,
            dz=grid.dz,
            dt=grid.dt,
            nt=grid.nt,
            source_ix=ix_src,
            source_iz=iz_src,
            force_stf_half=force_stf_half,
            receiver_sampling=sampling,
            half_order=ho,
            use_ts_sfd=False,
            n_boundary=n_boundary,
            gamma_s=300.0,
            snapshot_stride=None,
        )

        if not np.allclose(result.t_v, t_v):
            raise RuntimeError(f"Unexpected t_v returned for half_order={ho}.")

        vz_num = result.receiver_vz[0].copy()

        numerical_traces.append(vz_num)
        rel_l2_errors.append(relative_l2(vz_analytic, vz_num))
        max_abs_errors.append(float(np.max(np.abs(vz_analytic - vz_num))))

    numerical_traces = np.asarray(numerical_traces, dtype=np.float64)
    rel_l2_errors = np.asarray(rel_l2_errors, dtype=np.float64)
    max_abs_errors = np.asarray(max_abs_errors, dtype=np.float64)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    np.savez(
        outdir / "pointforce_validation_traces.npz",
        t_v=t_v,
        t_analytic=t_analytic,
        vz_analytic=vz_analytic,
        stf_integer=stf_integer,
        force_stf_half=force_stf_half,
        half_orders=np.asarray(half_orders, dtype=np.int64),
        spatial_orders=np.asarray(spatial_orders, dtype=np.int64),
        numerical_traces=numerical_traces,
        rel_l2_errors=rel_l2_errors,
        max_abs_errors=max_abs_errors,
        x_src=x_src,
        z_src=z_src,
        x_rec=x_rec,
        z_rec=z_rec,
        ix_src=ix_src,
        iz_src=iz_src,
        ix_rec=ix_rec,
        iz_rec=iz_rec,
        dt=grid.dt,
        dx=grid.dx,
        dz=grid.dz,
        vp=vp,
        vs=vs,
        rho=rho,
        f0_hz=f0_hz,
        t_p=t_p,
        t_s=t_s,
    )

    # ------------------------------------------------------------------
    # Plot 1: analytical vs numerical traces
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_v, vz_analytic, lw=2.2, label="Analytical")
    for so, trace in zip(spatial_orders, numerical_traces):
        ax.plot(t_v, trace, alpha=0.85, label=f"FD order {so}")
    ax.set_xlim(max(0.0, t_p - 0.4), min(t_v[-1], t_s + 1.0))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$v_z$ [arb. units]")
    ax.set_title("2D point-force validation: analytical vs numerical")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "pointforce_validation_traces.png", dpi=200)
    plt.close()

    # ------------------------------------------------------------------
    # Plot 1b: zoom around arrivals
    # ------------------------------------------------------------------
    t_zoom_min = max(0.0, t_p - 0.15)
    t_zoom_max = min(t_v[-1], t_s + 0.25)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_v, vz_analytic, lw=2.2, label="Analytical")
    for so, trace in zip(spatial_orders, numerical_traces):
        ax.plot(t_v, trace, alpha=0.85, label=f"FD order {so}")
    ax.set_xlim(t_zoom_min, t_zoom_max)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$v_z$ [arb. units]")
    ax.set_title("2D point-force validation: zoom around arrivals")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "pointforce_validation_traces_zoom.png", dpi=200)
    plt.close()

    # ------------------------------------------------------------------
    # Plot 2: convergence
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(spatial_orders, rel_l2_errors, "o-")
    ax.set_yscale("log")
    ax.set_xlabel("Spatial FD order")
    ax.set_ylabel(r"Relative $L_2$ error (receiver $v_z$ only)")
    ax.set_title(r"Convergence: $\|v^{num} - v^{analytic}\|_2 / \|v^{analytic}\|_2$")
    ax.grid(True, which="both", alpha=0.3)

    for x, y in zip(spatial_orders, rel_l2_errors):
        ax.annotate(
            f"{y:.2e}",
            xy=(x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(outdir / "pointforce_validation_error_vs_order.png", dpi=200)
    plt.close()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n=== Relative L2 errors (receiver_vz only) ===")
    for so, err, emax in zip(spatial_orders, rel_l2_errors, max_abs_errors):
        print(f"  order {so:>2d}: rel_L2 = {err:.4e},  max_abs = {emax:.4e}")

    print("\n=== Error reduction factors ===")
    for i in range(1, len(rel_l2_errors)):
        reduction = rel_l2_errors[i - 1] / rel_l2_errors[i]
        print(
            f"  {spatial_orders[i-1]} -> {spatial_orders[i]} : {reduction:.2f}x"
        )


if __name__ == "__main__":
    main()