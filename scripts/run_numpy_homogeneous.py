# ==============================================================================
# scripts/run_numpy_homogeneous.py
#
# Final homogeneous validation script for the 2D elastic + DAS pipeline.
#
# Includes:
#   1. Homogeneous elastic model
#   2. Physical double-couple source run
#   3. Permanent explosive validation run
#   4. Straight DAS cable
#   5. Staggered-aware receiver extraction
#   6. DAS strain-rate output
#   7. Raw and trace-normalized gathers
#   8. Rough travel-time validation
#   9. Wavefield animation saved as MP4 or GIF
# ==============================================================================

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from scipy.signal import hilbert

from src.grid import Grid2D
from src.model import ElasticModel2D
from src.source import build_dc_source, DoubleCoupleSource2D
from src.receivers import build_das_cable, Receivers2D
from src.simulator import run_forward_simulation
from src.sampling import build_receiver_sampling
from src.solver_numpy import run_elastic_solver_numpy, max_stable_dt


# ==============================================================================
# 1. MODEL / GEOMETRY BUILDERS
# ==============================================================================

def build_homogeneous_model(
    *,
    nx: int = 301,
    nz: int = 201,
    dx: float = 10.0,
    dz: float = 10.0,
    nt: int = 2000,
    vp: float = 3000.0,
    vs: float = 1700.0,
    rho: float = 2500.0,
    cfl_safety: float = 0.90,
    half_order: int = 2,
) -> ElasticModel2D:
    """
    Build a homogeneous elastic model with a CFL-safe timestep.
    """
    dt = cfl_safety * max_stable_dt(vp, dx, dz, half_order)

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

    vp_arr = np.full(grid.shape, vp, dtype=np.float64)
    vs_arr = np.full(grid.shape, vs, dtype=np.float64)
    rho_arr = np.full(grid.shape, rho, dtype=np.float64)

    return ElasticModel2D(grid=grid, vp=vp_arr, vs=vs_arr, rho=rho_arr)


def build_geometry(
    model: ElasticModel2D,
    n_pml: int,
) -> tuple[DoubleCoupleSource2D, Receivers2D]:
    """
    Build the physical 2D double-couple source and a straight vertical DAS cable.
    """
    grid = model.grid

    # Source roughly left-of-centre
    x_src = grid.x[grid.nx // 3]
    z_src = grid.z[grid.nz // 2]

    source = build_dc_source(
        grid=grid,
        x_m=x_src,
        z_m=z_src,
        theta_deg=0.0,
        scalar_moment=1.0e10,
        nt=grid.nt,
        dt=grid.dt,
        f0_hz=8.0,
        derivative_order=0,
    )

    # Keep cable safely inside the non-PML region
    ix_cable = grid.nx - n_pml - 25
    iz_top = n_pml + 10
    iz_bot = grid.nz - n_pml - 10

    x_cable = grid.x[ix_cable]
    z_top = grid.z[iz_top]
    z_bot = grid.z[iz_bot]

    receivers = build_das_cable(
        grid=grid,
        waypoints_x=[x_cable, x_cable],
        waypoints_z=[z_top, z_bot],
        channel_spacing_m=10.0,
        n_pml=n_pml,
    )

    return source, receivers


def build_explosive_validation_source(
    model: ElasticModel2D,
    *,
    f0_hz: float = 8.0,
    amplitude: float = 1.0e10,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a simple isotropic / explosive-like stress source for permanent validation.
    """
    grid = model.grid

    x_src = grid.x[grid.nx // 3]
    z_src = grid.z[grid.nz // 2]

    source_ix = int(np.argmin(np.abs(grid.x - x_src)))
    source_iz = int(np.argmin(np.abs(grid.z - z_src)))

    t = np.arange(grid.nt, dtype=np.float64) * grid.dt
    t0 = 1.2 / f0_hz
    arg = (np.pi * f0_hz * (t - t0)) ** 2
    stf = (1.0 - 2.0 * arg) * np.exp(-arg)

    stf_xx = amplitude * stf
    stf_zz = amplitude * stf
    stf_xz = np.zeros_like(stf)

    return source_ix, source_iz, stf_xx, stf_zz, stf_xz


# ==============================================================================
# 2. PLOTTING HELPERS
# ==============================================================================

def normalize_traces(data: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize each trace independently by its maximum absolute amplitude.
    """
    data = np.asarray(data, dtype=np.float64)
    scale = np.max(np.abs(data), axis=1, keepdims=True)
    scale = np.maximum(scale, eps)
    return data / scale


def plot_model_geometry(
    model: ElasticModel2D,
    source,
    receivers: Receivers2D,
    out_dir: Path,
    filename: str = "01_model_geometry.png",
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    im = ax.pcolormesh(
        model.grid.x,
        model.grid.z,
        model.vp.T,
        shading="auto",
        cmap="turbo",
    )
    fig.colorbar(im, ax=ax, label="Vp [m/s]")

    ax.plot(receivers.x, receivers.z, "w-", lw=2, label="DAS cable")
    ax.scatter(source.x_embedded_m, source.z_embedded_m, c="red", s=60, label="Source")

    ax.invert_yaxis()
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Depth [m]")
    ax.set_title("Homogeneous model with source and DAS cable")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_receiver_gather(
    t: np.ndarray,
    data: np.ndarray,
    receivers: Receivers2D,
    title: str,
    cbar_label: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    extent = [t[0], t[-1], receivers.s[-1], receivers.s[0]]

    vmax = np.percentile(np.abs(data), 99.0)
    if vmax == 0.0:
        vmax = 1.0

    im = ax.imshow(
        data,
        aspect="auto",
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
        extent=extent,
        origin="upper",
        interpolation="none",
    )
    fig.colorbar(im, ax=ax, label=cbar_label)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Arc length along cable [m]")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_receiver_gather_normalized(
    t: np.ndarray,
    data: np.ndarray,
    receivers: Receivers2D,
    title: str,
    out_path: Path,
) -> None:
    data_n = normalize_traces(data)

    fig, ax = plt.subplots(figsize=(10, 6))
    extent = [t[0], t[-1], receivers.s[-1], receivers.s[0]]

    im = ax.imshow(
        data_n,
        aspect="auto",
        cmap="seismic",
        vmin=-1.0,
        vmax=1.0,
        extent=extent,
        origin="upper",
        interpolation="none",
    )
    fig.colorbar(im, ax=ax, label="Trace-normalized amplitude")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Arc length along cable [m]")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_das_gather(
    t: np.ndarray,
    das_result,
    receivers: Receivers2D,
    out_path: Path,
) -> None:
    s_valid = receivers.s[das_result.channel_indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    extent = [t[0], t[-1], s_valid[-1], s_valid[0]]

    vmax = np.percentile(np.abs(das_result.data), 99.0)
    if vmax == 0.0:
        vmax = 1.0

    im = ax.imshow(
        das_result.data,
        aspect="auto",
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
        extent=extent,
        origin="upper",
        interpolation="none",
    )
    fig.colorbar(im, ax=ax, label="Axial strain-rate [1/s]")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Arc length along cable [m]")
    ax.set_title(f"DAS axial strain-rate (gauge = {das_result.gauge_length_m:.1f} m)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_das_gather_normalized(
    t: np.ndarray,
    das_result,
    receivers: Receivers2D,
    out_path: Path,
) -> None:
    s_valid = receivers.s[das_result.channel_indices]
    data_n = normalize_traces(das_result.data)

    fig, ax = plt.subplots(figsize=(10, 6))
    extent = [t[0], t[-1], s_valid[-1], s_valid[0]]

    im = ax.imshow(
        data_n,
        aspect="auto",
        cmap="seismic",
        vmin=-1.0,
        vmax=1.0,
        extent=extent,
        origin="upper",
        interpolation="none",
    )
    fig.colorbar(im, ax=ax, label="Trace-normalized amplitude")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Arc length along cable [m]")
    ax.set_title("DAS axial strain-rate (trace-normalized)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# 3. ARRIVAL-TIME VALIDATION
# ==============================================================================

def predict_first_arrivals(
    source,
    receivers: Receivers2D,
    vp: float,
    vs: float,
    channel_ids: list[int],
) -> list[dict]:
    """
    Compute rough straight-ray P and S arrival times.
    """
    xs = source.x_embedded_m
    zs = source.z_embedded_m

    out = []
    for ich in channel_ids:
        xr = receivers.x[ich]
        zr = receivers.z[ich]
        dist = float(np.sqrt((xr - xs) ** 2 + (zr - zs) ** 2))

        out.append(
            {
                "channel": ich,
                "arc_m": float(receivers.s[ich]),
                "distance_m": dist,
                "tP_s": dist / vp,
                "tS_s": dist / vs,
            }
        )
    return out


def estimate_observed_arrivals(
    t: np.ndarray,
    data: np.ndarray,
    channel_ids: list[int],
) -> list[dict]:
    """
    Rough observed-arrival estimate using the maximum of the analytic-signal envelope.
    """
    out = []
    for ich in channel_ids:
        tr = np.asarray(data[ich], dtype=np.float64)
        env = np.abs(hilbert(tr))
        it_max = int(np.argmax(env))
        out.append(
            {
                "channel": ich,
                "t_obs_s": float(t[it_max]),
                "amp_env_max": float(env[it_max]),
            }
        )
    return out


def compare_arrivals(
    source,
    receivers: Receivers2D,
    t: np.ndarray,
    data: np.ndarray,
    vp: float,
    vs: float,
    channel_ids: list[int],
) -> None:
    pred = predict_first_arrivals(source, receivers, vp, vs, channel_ids)
    obs = estimate_observed_arrivals(t, data, channel_ids)
    obs_map = {d["channel"]: d for d in obs}

    print("\nApproximate first-arrival comparison")
    print("channel | arc[m] | dist[m] | tP[s] | tS[s] | t_obs[s]")
    print("-" * 58)
    for p in pred:
        o = obs_map[p["channel"]]
        print(
            f"{p['channel']:7d} | "
            f"{p['arc_m']:6.1f} | "
            f"{p['distance_m']:7.1f} | "
            f"{p['tP_s']:.3f} | "
            f"{p['tS_s']:.3f} | "
            f"{o['t_obs_s']:.3f}"
        )


# ==============================================================================
# 4. ANIMATION
# ==============================================================================

def save_wavefield_animation(
    model: ElasticModel2D,
    source,
    receivers: Receivers2D,
    run_result,
    out_stem: Path,
    title_prefix: str,
    fps: int = 6,
    pclip: float = 99.0,
) -> None:
    """
    Save wavefield animation as MP4 if ffmpeg is available, otherwise as GIF.
    """
    if run_result.snapshots_vz is None or run_result.snapshot_times_s is None:
        print(f"[{title_prefix}] No snapshots available for animation.")
        return

    snaps = run_result.snapshots_vz
    ts = run_result.snapshot_times_s
    grid = model.grid

    vmax = np.percentile(np.abs(snaps), pclip)
    if vmax == 0.0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(9, 6))
    mesh = ax.pcolormesh(
        grid.x,
        grid.z,
        snaps[0].T,
        shading="auto",
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
    )
    fig.colorbar(mesh, ax=ax, label="Vz [m/s]")

    ax.plot(receivers.x, receivers.z, "k-", lw=1.5, label="DAS cable")
    ax.scatter(
        source.x_embedded_m,
        source.z_embedded_m,
        c="yellow",
        edgecolors="k",
        s=60,
        label="Source",
    )

    ax.invert_yaxis()
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Depth [m]")
    title = ax.set_title(f"{title_prefix}: Vz snapshot at t = {ts[0]:.3f} s")
    ax.legend(loc="upper right")
    fig.tight_layout()

    def update(i):
        mesh.set_array(snaps[i].T.ravel())
        title.set_text(f"{title_prefix}: Vz snapshot at t = {ts[i]:.3f} s")
        return (mesh, title)

    anim = FuncAnimation(fig, update, frames=len(snaps), interval=1000 / fps, blit=False)

    mp4_path = out_stem.with_suffix(".mp4")
    gif_path = out_stem.with_suffix(".gif")

    try:
        writer = FFMpegWriter(fps=fps, bitrate=1800)
        anim.save(mp4_path, writer=writer, dpi=160)
        print(f"Saved animation: {mp4_path}")
    except Exception as e:
        print(f"MP4 save failed ({e}). Falling back to GIF...")
        writer = PillowWriter(fps=fps)
        anim.save(gif_path, writer=writer, dpi=120)
        print(f"Saved animation: {gif_path}")

    plt.close(fig)


# ==============================================================================
# 5. EXPLOSIVE VALIDATION RUN
# ==============================================================================

def run_explosive_validation(
    model: ElasticModel2D,
    receivers: Receivers2D,
    *,
    half_order: int = 2,
    n_boundary: int = 50,
    gamma_s: float = 50.0,
    snapshot_stride: int = 100,
):
    """
    Permanent sanity-check run with a simple isotropic stress source.
    """
    grid = model.grid
    sampling = build_receiver_sampling(grid, receivers)

    source_ix, source_iz, stf_xx, stf_zz, stf_xz = build_explosive_validation_source(model)

    run_result = run_elastic_solver_numpy(
        vp=model.vp,
        vs=model.vs,
        rho=model.rho,
        dx=grid.dx,
        dz=grid.dz,
        dt=grid.dt,
        nt=grid.nt,
        source_ix=source_ix,
        source_iz=source_iz,
        stf_xx=stf_xx,
        stf_zz=stf_zz,
        stf_xz=stf_xz,
        receiver_sampling=sampling,
        half_order=half_order,
        use_ts_sfd=False,
        n_boundary=n_boundary,
        gamma_s=gamma_s,
        snapshot_stride=snapshot_stride,
    )

    class DummySource:
        def __init__(self, x_embedded_m: float, z_embedded_m: float) -> None:
            self.x_embedded_m = x_embedded_m
            self.z_embedded_m = z_embedded_m

    dummy_source = DummySource(
        x_embedded_m=grid.x[source_ix],
        z_embedded_m=grid.z[source_iz],
    )

    return run_result, dummy_source


# ==============================================================================
# 6. MAIN
# ==============================================================================

def main() -> None:
    # Fixed baseline parameters
    half_order = 2
    use_ts_sfd = False
    vp0 = 3000.0
    vs0 = 1700.0
    rho0 = 2500.0
    n_boundary = 50
    gamma_s = 50.0
    snapshot_stride = 100
    gauge_length_m = 20.0

    out_dir = Path("results/run_numpy_homogeneous_baseline")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building model and geometry...")
    model = build_homogeneous_model(
        vp=vp0,
        vs=vs0,
        rho=rho0,
        half_order=half_order,
    )
    source, receivers = build_geometry(model, n_pml=n_boundary)

    print(f"Grid: {model.grid.shape}, dt = {model.grid.dt:.6e} s")
    print(f"Source placed at X={source.x_embedded_m:.1f} m, Z={source.z_embedded_m:.1f} m")
    print(f"Receivers: {receivers.nrec} channels")
    print(f"half_order = {half_order}, use_ts_sfd = {use_ts_sfd}")
    print(f"f0 = 8.0 Hz, n_boundary = {n_boundary}, gamma_s = {gamma_s}")
    print(f"Output directory: {out_dir}")

    # ------------------------------------------------------------------
    # 1. Main physical run: double-couple source
    # ------------------------------------------------------------------
    print("\nRunning forward simulation (double-couple source)...")
    run_result, das_result = run_forward_simulation(
        model=model,
        source=source,
        receivers=receivers,
        gauge_length_m=gauge_length_m,
        half_order=half_order,
        use_ts_sfd=use_ts_sfd,
        n_boundary=n_boundary,
        gamma_s=gamma_s,
        snapshot_stride=snapshot_stride,
    )
    print("Double-couple simulation finished.")

    print("Saving double-couple outputs...")
    plot_model_geometry(model, source, receivers, out_dir)

    plot_receiver_gather(
        run_result.t,
        run_result.receiver_vx,
        receivers,
        title="Receiver gather: Vx",
        cbar_label="Vx [m/s]",
        out_path=out_dir / "02_receiver_vx.png",
    )
    plot_receiver_gather(
        run_result.t,
        run_result.receiver_vz,
        receivers,
        title="Receiver gather: Vz",
        cbar_label="Vz [m/s]",
        out_path=out_dir / "03_receiver_vz.png",
    )
    plot_das_gather(
        run_result.t,
        das_result,
        receivers,
        out_path=out_dir / "04_das_strain_rate.png",
    )

    plot_receiver_gather_normalized(
        run_result.t,
        run_result.receiver_vx,
        receivers,
        title="Receiver gather: Vx (trace-normalized)",
        out_path=out_dir / "02b_receiver_vx_normalized.png",
    )
    plot_receiver_gather_normalized(
        run_result.t,
        run_result.receiver_vz,
        receivers,
        title="Receiver gather: Vz (trace-normalized)",
        out_path=out_dir / "03b_receiver_vz_normalized.png",
    )
    plot_das_gather_normalized(
        run_result.t,
        das_result,
        receivers,
        out_path=out_dir / "04b_das_strain_rate_normalized.png",
    )

    save_wavefield_animation(
        model=model,
        source=source,
        receivers=receivers,
        run_result=run_result,
        out_stem=out_dir / "double_couple_wavefield",
        title_prefix="double_couple",
        fps=6,
        pclip=99.0,
    )

    np.savez_compressed(
        out_dir / "outputs_double_couple.npz",
        t=run_result.t,
        receiver_vx=run_result.receiver_vx,
        receiver_vz=run_result.receiver_vz,
        das_data=das_result.data,
        das_channel_indices=das_result.channel_indices,
        das_gauge_samples=das_result.gauge_samples,
        das_gauge_length_m=das_result.gauge_length_m,
        receiver_x=receivers.x,
        receiver_z=receivers.z,
        receiver_s=receivers.s,
        source_x=source.x_embedded_m,
        source_z=source.z_embedded_m,
    )

    channel_ids = [0, receivers.nrec // 2, receivers.nrec - 1]
    compare_arrivals(
        source=source,
        receivers=receivers,
        t=run_result.t,
        data=run_result.receiver_vz,
        vp=vp0,
        vs=vs0,
        channel_ids=channel_ids,
    )

    # ------------------------------------------------------------------
    # 2. Permanent validation run: simple explosive source
    # ------------------------------------------------------------------
    validation_dir = out_dir / "explosive_validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    print("\nRunning validation simulation (explosive-like source)...")
    val_run_result, val_source = run_explosive_validation(
        model=model,
        receivers=receivers,
        half_order=half_order,
        n_boundary=n_boundary,
        gamma_s=gamma_s,
        snapshot_stride=snapshot_stride,
    )
    print("Explosive validation simulation finished.")

    plot_receiver_gather(
        val_run_result.t,
        val_run_result.receiver_vx,
        receivers,
        title="Explosive validation: Receiver gather Vx",
        cbar_label="Vx [m/s]",
        out_path=validation_dir / "receiver_vx_validation.png",
    )
    plot_receiver_gather(
        val_run_result.t,
        val_run_result.receiver_vz,
        receivers,
        title="Explosive validation: Receiver gather Vz",
        cbar_label="Vz [m/s]",
        out_path=validation_dir / "receiver_vz_validation.png",
    )
    plot_receiver_gather_normalized(
        val_run_result.t,
        val_run_result.receiver_vx,
        receivers,
        title="Explosive validation: Receiver gather Vx (trace-normalized)",
        out_path=validation_dir / "receiver_vx_validation_normalized.png",
    )
    plot_receiver_gather_normalized(
        val_run_result.t,
        val_run_result.receiver_vz,
        receivers,
        title="Explosive validation: Receiver gather Vz (trace-normalized)",
        out_path=validation_dir / "receiver_vz_validation_normalized.png",
    )

    save_wavefield_animation(
        model=model,
        source=val_source,
        receivers=receivers,
        run_result=val_run_result,
        out_stem=validation_dir / "explosive_wavefield",
        title_prefix="explosive",
        fps=6,
        pclip=99.0,
    )

    np.savez_compressed(
        validation_dir / "outputs_explosive_validation.npz",
        t=val_run_result.t,
        receiver_vx=val_run_result.receiver_vx,
        receiver_vz=val_run_result.receiver_vz,
        receiver_x=receivers.x,
        receiver_z=receivers.z,
        receiver_s=receivers.s,
        source_x=val_source.x_embedded_m,
        source_z=val_source.z_embedded_m,
    )

    print(f"\nPipeline complete. All results saved to: {out_dir.absolute()}")


if __name__ == "__main__":
    main()