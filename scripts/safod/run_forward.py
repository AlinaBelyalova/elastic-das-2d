# ==============================================================================
# scripts/safod/run_forward.py
#
# SAFOD initial-model forward simulation.
#
# This script is a QC forward run, not FWI yet.
#
# Current default mode:
#   catalog_event:
#       projected source and corrected down-going SAFOD DAS geometry for
#       NC75336802. The alternative deep_saf mode remains available for
#       controlled synthetic QC.
#
# Requirements:
#   - src.safod_builder.build_safod_model
#   - src.safod_builder.fault_x_at_z
#   - src.das supports continuous physical gauge lengths, e.g. GL=16.6213 m
#   - src.plotting.place_safod_legend for figure-fraction legend placement
#     (do NOT call fig.tight_layout() after plot_safod_model() — it would
#     undo the reserved right margin and re-overlap the colorbar)
# ==============================================================================

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.safod_builder import build_safod_model, fault_x_at_z
from src.source import build_dc_source
from src.receivers import build_das_cable
from src.simulator import run_forward_simulation
from src.plotting import plot_safod_model, place_safod_legend
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.interpolate import RegularGridInterpolator

from scripts.safod.settings import (
    DEFAULT_THETA_DEG,
    REAL_EVENT_PACKAGE,
    forward_dir_for_theta,
    forward_run_tag,
)

# ==============================================================================
# HELPERS
# ==============================================================================

def normalize_traces(data: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Trace-normalize a gather for display only.
    """
    data = np.asarray(data, dtype=np.float64)
    scale = np.max(np.abs(data), axis=1, keepdims=True)
    scale = np.maximum(scale, eps)
    return data / scale

def load_real_event_package(path: Path) -> dict:
    """
    Load prepared real-event metadata for synthetic modelling.

    The package is created by scripts.safod.prepare_event and contains:
        - event_x_model_m
        - event_z_model_m
        - gauge_length_m
        - channel_spacing_m
        - geometry_csv or fallback geometry path
        - event metadata
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Real-event package not found: {path}")

    pkg = np.load(path, allow_pickle=True)

    def get_scalar(name: str, default=None):
        if name not in pkg.files:
            if default is None:
                raise KeyError(f"Missing {name!r} in real-event package: {path}")
            return default

        val = pkg[name]

        if val.shape == ():
            return val.item()

        if val.size == 1:
            return val.reshape(-1)[0].item()

        return val

    event_dir = path.parent

    if "geometry_csv" in pkg.files:
        geom_file = str(get_scalar("geometry_csv"))
    else:
        geom_file = str(event_dir / "SAFOD_Phase2_projected_from_georef.csv")

    cfg = {
        "package_path": str(path),
        "event_dir": str(event_dir),
        "geom_file": geom_file,

        "event_id": str(get_scalar("ev_id", "unknown")),
        "origin_time": str(get_scalar("ev_origin_time", "unknown")),
        "magnitude": float(get_scalar("ev_mag", np.nan)),
        "depth_km": float(get_scalar("ev_depth_km", np.nan)),

        "x_src": float(get_scalar("event_x_model_m")),
        "z_src": float(get_scalar("event_z_model_m")),
        "event_along_profile_m": float(get_scalar("event_along_profile_m", np.nan)),
        "event_crossline_m": float(get_scalar("event_crossline_m", np.nan)),

        "gauge_length_m": float(get_scalar("gauge_length_m")),
        "real_channel_spacing_m": float(get_scalar("channel_spacing_m")),
    }

    print("\nLoaded real-event package")
    print("-------------------------")
    print(f"package          : {cfg['package_path']}")
    print(f"event id         : {cfg['event_id']}")
    print(f"origin           : {cfg['origin_time']}")
    print(f"magnitude        : {cfg['magnitude']:.2f}")
    print(f"depth            : {cfg['depth_km']:.2f} km")
    print(f"source x,z       : {cfg['x_src']:.3f}, {cfg['z_src']:.3f} m")
    print(f"crossline        : {cfg['event_crossline_m']:.3f} m")
    print(f"gauge length     : {cfg['gauge_length_m']:.6f} m")
    print(f"real dCh         : {cfg['real_channel_spacing_m']:.6f} m")
    print(f"geometry csv     : {cfg['geom_file']}")

    return cfg

def compute_straight_ray_arrivals(
    *,
    grid,
    model,
    receivers,
    x_src: float,
    z_src: float,
    n_samples_per_ray: int = 256,
    time_shift_s: float = 0.0,
) -> dict:
    """
    Approximate P/S arrivals by integrating slowness along straight rays.

    This is a model-based straight-ray QC calculation, not full ray tracing.
    The velocity fields are sampled for all source-receiver rays in one
    vectorized RegularGridInterpolator call.

    Parameters
    ----------
    time_shift_s
        Optional constant added to the travel times. Use 0.0 for physical
        travel-time / first-arrival overlays.
    """
    if n_samples_per_ray < 2:
        raise ValueError("n_samples_per_ray must be >= 2.")

    rx = np.asarray(receivers.x, dtype=np.float64)
    rz = np.asarray(receivers.z, dtype=np.float64)
    s = np.asarray(receivers.s, dtype=np.float64)

    if rx.ndim != 1 or rz.ndim != 1 or s.ndim != 1:
        raise ValueError("receivers.x, receivers.z, and receivers.s must be 1D.")

    if not (rx.size == rz.size == s.size == int(receivers.nrec)):
        raise ValueError(
            "Receiver coordinate lengths do not match receivers.nrec: "
            f"x={rx.size}, z={rz.size}, s={s.size}, nrec={receivers.nrec}."
        )

    vp = np.asarray(model.vp, dtype=np.float64)
    vs = np.asarray(model.vs, dtype=np.float64)
    expected_shape = (int(grid.nx), int(grid.nz))

    if vp.shape != expected_shape or vs.shape != expected_shape:
        raise ValueError(
            "Vp/Vs shape must match (grid.nx, grid.nz): "
            f"Vp={vp.shape}, Vs={vs.shape}, expected={expected_shape}."
        )

    if not np.all(np.isfinite(vp)) or not np.all(np.isfinite(vs)):
        raise ValueError("Vp/Vs contain NaN or Inf.")

    if np.any(vp <= 0.0) or np.any(vs <= 0.0):
        raise ValueError("Vp/Vs must be strictly positive.")

    interp_vp = RegularGridInterpolator(
        (np.asarray(grid.x, dtype=np.float64),
         np.asarray(grid.z, dtype=np.float64)),
        vp,
        method="linear",
        bounds_error=True,
    )
    interp_vs = RegularGridInterpolator(
        (np.asarray(grid.x, dtype=np.float64),
         np.asarray(grid.z, dtype=np.float64)),
        vs,
        method="linear",
        bounds_error=True,
    )

    dx_ray = rx - float(x_src)
    dz_ray = rz - float(z_src)
    length = np.hypot(dx_ray, dz_ray)

    q = np.linspace(0.0, 1.0, n_samples_per_ray, dtype=np.float64)

    # Shape: (n_receivers, n_samples_per_ray)
    x_lines = float(x_src) + np.outer(dx_ray, q)
    z_lines = float(z_src) + np.outer(dz_ray, q)

    points = np.column_stack(
        (x_lines.ravel(), z_lines.ravel())
    )

    vp_lines = interp_vp(points).reshape(rx.size, n_samples_per_ray)
    vs_lines = interp_vs(points).reshape(rx.size, n_samples_per_ray)

    if (
        not np.all(np.isfinite(vp_lines))
        or not np.all(np.isfinite(vs_lines))
        or np.any(vp_lines <= 0.0)
        or np.any(vs_lines <= 0.0)
    ):
        raise ValueError(
            "Interpolated Vp/Vs along one or more straight rays are invalid."
        )

    t_p = (
        length
        * np.trapezoid(1.0 / vp_lines, q, axis=1)
        + float(time_shift_s)
    )
    t_s = (
        length
        * np.trapezoid(1.0 / vs_lines, q, axis=1)
        + float(time_shift_s)
    )

    zero_length = length == 0.0
    t_p[zero_length] = float(time_shift_s)
    t_s[zero_length] = float(time_shift_s)

    return {
        "s": s,
        "P": t_p,
        "S": t_s,
        "time_shift_s": float(time_shift_s),
        "method": "straight_ray_slowness_integral_vectorized",
    }


def subset_arrivals(arrivals: dict, channel_indices: np.ndarray) -> dict:
    """
    Subset receiver arrival curves to DAS gauge-centre channels.
    """
    idx = np.asarray(channel_indices, dtype=np.int64)
    return {
        "s": np.asarray(arrivals["s"])[idx],
        "P": np.asarray(arrivals["P"])[idx],
        "S": np.asarray(arrivals["S"])[idx],
        "time_shift_s": float(arrivals.get("time_shift_s", 0.0)),
        "method": arrivals.get("method", "unknown"),
    }


def _add_arrival_overlays(ax, arrival_times: dict | None) -> None:
    """
    Add approximate P/S arrival curves to a gather plot.

    x-axis: time [s]
    y-axis: cable arc length [m]
    """
    if arrival_times is None:
        return

    s = np.asarray(arrival_times["s"], dtype=np.float64)
    t_p = np.asarray(arrival_times["P"], dtype=np.float64)
    t_s = np.asarray(arrival_times["S"], dtype=np.float64)

    ax.plot(t_p, s, color="black", lw=1.4, ls="--", label="Approx. P")
    ax.plot(t_s, s, color="black", lw=1.4, ls=":", label="Approx. S")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)


def trim_cable_for_solver_domain(
    *,
    grid,
    x_cable: np.ndarray,
    z_cable: np.ndarray,
    n_boundary: int,
    half_order: int,
    free_surface: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep receiver cable points inside the solver-valid domain.

    For free_surface=True:
      - top sponge is disabled, so shallow receivers are allowed;
      - but receivers should stay below the ghost/stencil region.

    For side and bottom boundaries:
      - receivers must stay outside the sponge region.
    """
    x_cable = np.asarray(x_cable, dtype=np.float64)
    z_cable = np.asarray(z_cable, dtype=np.float64)

    if x_cable.shape != z_cable.shape:
        raise ValueError(
            f"x_cable and z_cable must have same shape; "
            f"got {x_cable.shape} and {z_cable.shape}."
        )

    x_min = float(grid.x[0] + n_boundary * grid.dx)
    x_max = float(grid.x[-1] - n_boundary * grid.dx)
    z_bottom_max = float(grid.z[-1] - n_boundary * grid.dz)

    if free_surface:
        z_top_min = float(grid.z[half_order + 1])
    else:
        z_top_min = float(grid.z[0] + n_boundary * grid.dz)

    keep = (
        (x_cable >= x_min)
        & (x_cable <= x_max)
        & (z_cable >= z_top_min)
        & (z_cable <= z_bottom_max)
    )

    n_keep = int(np.count_nonzero(keep))
    n_drop = int(x_cable.size - n_keep)

    if n_keep < 2:
        raise ValueError(
            "Too few cable points remain after trimming. "
            f"n_keep={n_keep}, n_drop={n_drop}. "
            f"x allowed [{x_min:.1f}, {x_max:.1f}], "
            f"z allowed [{z_top_min:.1f}, {z_bottom_max:.1f}]."
        )

    print("\nReceiver cable trimming")
    print("-----------------------")
    print(f"raw cable points     : {x_cable.size}")
    print(f"kept cable points    : {n_keep}")
    print(f"dropped cable points : {n_drop}")
    print(f"x allowed            : {x_min:.1f} to {x_max:.1f} m")
    print(f"z allowed            : {z_top_min:.1f} to {z_bottom_max:.1f} m")
    print(f"kept x range         : {x_cable[keep].min():.1f} to {x_cable[keep].max():.1f} m")
    print(f"kept z range         : {z_cable[keep].min():.1f} to {z_cable[keep].max():.1f} m")

    return x_cable[keep], z_cable[keep]


def check_source_inside_solver_domain(
    *,
    grid,
    x_src: float,
    z_src: float,
    n_boundary: int,
    half_order: int,
) -> None:
    """
    Cheap source-position pre-check before running the expensive solver.
    """
    ix_check, iz_check, _, _ = grid.get_closest_node(x_src, z_src)

    margin = n_boundary + half_order + 5

    ok = (
        margin <= ix_check < grid.nx - margin
        and margin <= iz_check < grid.nz - margin
    )

    if not ok:
        raise ValueError(
            f"Source position ({x_src:.1f}, {z_src:.1f}) m is too close to "
            f"sponge/stencil boundary: ix={ix_check}, iz={iz_check}, "
            f"required margin={margin} cells. "
            "Adjust source position or enlarge model padding/domain."
        )

    print("\nSource boundary pre-check")
    print("-------------------------")
    print(f"source x,z  : {x_src:.1f}, {z_src:.1f} m")
    print(f"source ix,iz: {ix_check}, {iz_check}")
    print(f"required margin: {margin} cells")
    print("source domain check: OK")


def check_record_duration(
    *,
    grid,
    max_s_arrival_s: float,
    min_tail_after_s_s: float = 0.50,
) -> None:
    """
    Record-length QC based on the maximum model-based straight-ray S arrival.

    The arrival remains an approximation because the paths are constrained to
    be straight. It is nevertheless more consistent than using one global
    percentile velocity unrelated to the actual source-receiver paths.
    """
    max_s_arrival_s = float(max_s_arrival_s)
    min_tail_after_s_s = float(min_tail_after_s_s)

    if not np.isfinite(max_s_arrival_s) or max_s_arrival_s < 0.0:
        raise ValueError(
            f"max_s_arrival_s must be finite and non-negative; "
            f"got {max_s_arrival_s}."
        )

    if min_tail_after_s_s < 0.0:
        raise ValueError(
            f"min_tail_after_s_s must be non-negative; "
            f"got {min_tail_after_s_s}."
        )

    duration = float((grid.nt - 1) * grid.dt)
    tail_after_s = duration - max_s_arrival_s

    print("\nRecord-duration QC")
    print("------------------")
    print(f"duration                           : {duration:.3f} s")
    print(f"max straight-ray S arrival (approx): {max_s_arrival_s:.3f} s")
    print(f"tail after far S                   : {tail_after_s:.3f} s")

    if tail_after_s < min_tail_after_s_s:
        required_duration = max_s_arrival_s + min_tail_after_s_s
        suggested_nt = int(np.ceil(required_duration / grid.dt)) + 1

        raise ValueError(
            "Record is probably too short for useful QC after the latest "
            "straight-ray S arrival. "
            f"tail_after_s={tail_after_s:.3f} s, "
            f"required >= {min_tail_after_s_s:.3f} s. "
            f"Increase nt from {grid.nt} to about {suggested_nt}."
        )

    print("record duration check: OK")


def plot_receiver_gather(
    *,
    t: np.ndarray,
    data: np.ndarray,
    receivers,
    title: str,
    cbar_label: str,
    out_path: Path,
    normalized: bool = False,
    arrival_times=None,
) -> None:
    data = np.asarray(data, dtype=np.float64)
    arr = normalize_traces(data) if normalized else data

    vmax = 1.0 if normalized else float(np.percentile(np.abs(arr), 99.0))
    if vmax == 0.0 or not np.isfinite(vmax):
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(10, 6))

    extent = [
        float(t[0]),
        float(t[-1]),
        float(receivers.s[-1]),
        float(receivers.s[0]),
    ]

    im = ax.imshow(
        arr,
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
    ax.set_ylabel("Arc length along DAS cable [m]")
    ax.set_title(title)

    _add_arrival_overlays(ax, arrival_times)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_das_gather(
    *,
    t: np.ndarray,
    das_result,
    receivers,
    title: str,
    out_path: Path,
    normalized: bool = False,
    arrival_times=None,
) -> None:
    data = np.asarray(das_result.data, dtype=np.float64)
    arr = normalize_traces(data) if normalized else data

    s_valid = receivers.s[das_result.channel_indices]

    vmax = 1.0 if normalized else float(np.percentile(np.abs(arr), 99.0))
    if vmax == 0.0 or not np.isfinite(vmax):
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(10, 6))

    extent = [
        float(t[0]),
        float(t[-1]),
        float(s_valid[-1]),
        float(s_valid[0]),
    ]

    im = ax.imshow(
        arr,
        aspect="auto",
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
        extent=extent,
        origin="upper",
        interpolation="none",
    )

    label = "Trace-normalized amplitude" if normalized else "Axial strain-rate [1/s]"
    fig.colorbar(im, ax=ax, label=label)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Arc length along DAS cable [m]")
    ax.set_title(title)

    _add_arrival_overlays(ax, arrival_times)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _snapshot_frame_2d(
    snapshots: np.ndarray,
    iframe: int,
    *,
    nx: int,
    nz: int,
) -> np.ndarray:
    """
    Return one snapshot as image array with shape (nz, nx).

    Solver/storage conventions can differ:
    - (nsnap, nx, nz)
    - (nsnap, nz, nx)
    - (nx, nz, nsnap)
    - (nz, nx, nsnap)

    This helper makes the GIF writer robust to all of these.
    """
    snapshots = np.asarray(snapshots)

    if snapshots.ndim != 3:
        raise ValueError(
            f"snapshots must be 3D, got shape {snapshots.shape}."
        )

    # Case 1: snapshots[iframe] is one 2D frame.
    if snapshots.shape[0] > iframe:
        frame = snapshots[iframe]

        if frame.shape == (nx, nz):
            return frame.T

        if frame.shape == (nz, nx):
            return frame

    # Case 2: last axis is frame index.
    if snapshots.shape[-1] > iframe:
        frame = snapshots[:, :, iframe]

        if frame.shape == (nx, nz):
            return frame.T

        if frame.shape == (nz, nx):
            return frame

    raise ValueError(
        "Could not infer snapshot layout. "
        f"snapshots.shape={snapshots.shape}, expected nx={nx}, nz={nz}."
    )


def make_wavefield_gif(
    *,
    grid,
    snapshots_vz: np.ndarray,
    snapshot_times: np.ndarray,
    x_cable: np.ndarray,
    z_cable: np.ndarray,
    metadata,
    source,
    out_path: Path,
    title: str,
    fps: int = 6,
    max_frames: int = 80,
    percentile_clip: float = 99.5,
) -> None:
    """
    Make GIF of Vz wavefield propagation.

    This is mainly for QC:
    - radiation pattern of the double-couple moment tensor source
    - free-surface behaviour
    - side/bottom sponge absorption
    - scattering / bending near the SAF low-velocity zone
    """
    snapshots_vz = np.asarray(snapshots_vz, dtype=np.float64)
    snapshot_times = np.asarray(snapshot_times, dtype=np.float64)

    if snapshots_vz.ndim != 3:
        raise ValueError(
            f"snapshots_vz must be 3D, got shape {snapshots_vz.shape}."
        )

    # Infer number of frames.
    if snapshots_vz.shape[0] == snapshot_times.size:
        nframes_total = snapshots_vz.shape[0]
        frame_axis = 0
    elif snapshots_vz.shape[-1] == snapshot_times.size:
        nframes_total = snapshots_vz.shape[-1]
        frame_axis = -1
    else:
        raise ValueError(
            "snapshot_times length does not match first or last snapshot axis: "
            f"snapshots_vz.shape={snapshots_vz.shape}, "
            f"snapshot_times.size={snapshot_times.size}."
        )

    if nframes_total < 1:
        raise ValueError("No snapshots available for GIF.")

    # Limit GIF length if many snapshots are present.
    if nframes_total > max_frames:
        frame_ids = np.linspace(0, nframes_total - 1, max_frames).astype(int)
    else:
        frame_ids = np.arange(nframes_total, dtype=int)

    # Robust symmetric colour scale from all selected frames.
    sample_vals = []
    for iframe in frame_ids:
        frame = _snapshot_frame_2d(
            snapshots_vz,
            int(iframe),
            nx=grid.nx,
            nz=grid.nz,
        )
        sample_vals.append(np.ravel(frame))

    sample_vals = np.concatenate(sample_vals)
    vmax = float(np.percentile(np.abs(sample_vals), percentile_clip))

    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0

    extent = [
        float(grid.x[0]),
        float(grid.x[-1]),
        float(grid.z[-1]),
        float(grid.z[0]),
    ]

    x_cable = np.asarray(x_cable, dtype=np.float64)
    z_cable = np.asarray(z_cable, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.5, 9.0))
    fig.subplots_adjust(left=0.12, right=0.86, top=0.92, bottom=0.08)

    frame0 = _snapshot_frame_2d(
        snapshots_vz,
        int(frame_ids[0]),
        nx=grid.nx,
        nz=grid.nz,
    )

    im = ax.imshow(
        frame0,
        extent=extent,
        origin="upper",
        aspect="equal",
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Vz [m/s]")

    # Static overlays.
    ax.plot(
        x_cable,
        z_cable,
        color="white",
        lw=2.0,
        label="DAS cable",
        zorder=5,
    )

    if hasattr(metadata, "x_fault_line") and hasattr(metadata, "z_fault_line"):
        ax.plot(
            metadata.x_fault_line,
            metadata.z_fault_line,
            "k--",
            lw=1.8,
            label="SAF prior",
            zorder=6,
        )

    ax.scatter(
        [source.x_embedded_m],
        [source.z_embedded_m],
        marker="*",
        s=140,
        c="yellow",
        edgecolors="black",
        zorder=10,
        label="Double-couple source",
    )

    time_text = ax.text(
        0.02,
        0.96,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
    )

    ax.set_xlim(float(grid.x[0]), float(grid.x[-1]))
    ax.set_ylim(float(grid.z[-1]), float(grid.z[0]))
    ax.set_xlabel("Projected 2D section coordinate X [m]")
    ax.set_ylabel("Depth [m]")
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=8)

    def update(k: int):
        iframe = int(frame_ids[k])

        frame = _snapshot_frame_2d(
            snapshots_vz,
            iframe,
            nx=grid.nx,
            nz=grid.nz,
        )

        im.set_data(frame)
        time_text.set_text(f"t = {snapshot_times[iframe]:.3f} s")
        return im, time_text

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_ids),
        interval=1000.0 / fps,
        blit=False,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(
        out_path,
        writer=PillowWriter(fps=fps),
        dpi=120,
    )

    plt.close(fig)

    print(f"Saved wavefield GIF to: {out_path}")


# ==============================================================================
# COMMAND-LINE INTERFACE
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the SAFOD initial-model elastic forward simulation for one "
            "effective 2D double-couple orientation."
        )
    )

    parser.add_argument(
        "--theta-deg",
        type=float,
        default=DEFAULT_THETA_DEG,
        help=(
            "Effective 2D double-couple orientation in degrees. "
            "The current parameterisation requires 0 <= theta < 90. "
            f"Default: {DEFAULT_THETA_DEG:.1f}."
        ),
    )

    gif_group = parser.add_mutually_exclusive_group()

    gif_group.add_argument(
        "--save-gif",
        dest="save_gif",
        action="store_true",
        help=(
            "Store wavefield snapshots and create the Vz GIF. "
            "This is the default."
        ),
    )

    gif_group.add_argument(
        "--no-gif",
        dest="save_gif",
        action="store_false",
        help=(
            "Skip wavefield snapshots and GIF creation. "
            "Use this only for large parameter sweeps."
        ),
    )

    parser.set_defaults(save_gif=True)

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing forward NPZ package for the selected angle. "
            "Without this flag, an existing package aborts the run."
        ),
    )

    return parser.parse_args()


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    args = parse_args()

    if not 0.0 <= args.theta_deg < 90.0:
        raise ValueError(
            "--theta-deg must satisfy 0 <= theta < 90 for the current "
            "2D double-couple parameterisation."
        )

    run_tag = forward_run_tag(args.theta_deg)
    # --------------------------------------------------------------------------
    # Run mode
    # --------------------------------------------------------------------------
    # "deep_saf"       : old synthetic deep source near the SAF prior
    # "catalog_event" : source from prepared real-event package
    source_mode = "catalog_event"

    real_event_package = REAL_EVENT_PACKAGE

    event_cfg = None

    if source_mode == "catalog_event":
        event_cfg = load_real_event_package(real_event_package)
        geom_file = event_cfg["geom_file"]
        # geom_file = "/home/groups/ettore88/alina/imaging/SAFOD_downleg_Projected_2D.csv"
        out_dir = forward_dir_for_theta(args.theta_deg)
    elif source_mode == "deep_saf":
        geom_file = "/home/groups/ettore88/alina/imaging/SAFOD_downleg_Projected_2D.csv"
        out_dir = Path("results/safod_initial_forward")
    else:
        raise ValueError(f"Unknown source_mode: {source_mode!r}")

    out_dir.mkdir(parents=True, exist_ok=True)

    output_package = (
        out_dir / "outputs_safod_initial_forward.npz"
    )

    if output_package.exists() and not args.overwrite:
        raise FileExistsError(
            f"Forward package already exists: {output_package}\n"
            "Use --overwrite only when replacement is intentional."
        )

    print("\nRun identity")
    print("------------")
    print(f"run tag       : {run_tag}")
    print(f"theta         : {args.theta_deg:.1f} deg")
    print(f"save GIF      : {args.save_gif}")
    print(f"output dir    : {out_dir}")

    # --------------------------------------------------------------------------
    # Numerical settings
    # --------------------------------------------------------------------------
    dx = 5.0
    dz = 5.0

    # The exact duration is printed after grid construction because dt is
    # selected from the CFL condition.

    nt = 12000
    half_order = 2

    # Boundary configuration selected by the controlled homogeneous sweep.
    n_boundary = 120
    gamma_s = 80.0
    free_surface = True

    # Preserve the physical entrance of the absorbing boundary used by the
    # original n40 model. Increasing n_boundary must enlarge the grid outward,
    # not move the sponge into the scientific model domain.
    baseline_n_boundary = 40
    baseline_x_padding_m = 800.0
    baseline_z_max_m = 9000.0

    # Physical undamped margins in the original n40 model.
    baseline_undamped_side_margin_m = (
        baseline_x_padding_m
        - baseline_n_boundary * dx
    )
    baseline_bottom_sponge_entry_m = (
        baseline_z_max_m
        - baseline_n_boundary * dz
    )

    # Add real physical model space, not just more absorbing cells.
    extra_scientific_x_margin_m = 500.0

    x_padding_m = (
        baseline_undamped_side_margin_m
        + extra_scientific_x_margin_m
        + n_boundary * dx
    )

    z_max_m = (
        baseline_bottom_sponge_entry_m
        + n_boundary * dz
    )

    print("\nAbsorbing-boundary configuration")
    print("--------------------------------")
    print(f"n_boundary                 : {n_boundary}")
    print(f"gamma_s                    : {gamma_s:.1f}")
    print(f"sponge width               : {n_boundary * dx:.1f} m")
    print(
        "undamped side margin      : "
        f"{baseline_undamped_side_margin_m + extra_scientific_x_margin_m:.1f} m"
    )
    print(f"extra scientific x margin  : {extra_scientific_x_margin_m:.1f} m")
    print(f"total x padding            : {x_padding_m:.1f} m")
    print(f"model bottom               : {z_max_m:.1f} m")

    if source_mode == "catalog_event":
        # Use the actual acquisition parameters of the selected real event.
        gauge_length_m = float(
            event_cfg["gauge_length_m"]
        )

        channel_spacing_m = float(
            event_cfg["real_channel_spacing_m"]
        )

        event_id_for_title = event_cfg["event_id"]
    else:
        gauge_length_m = 10.209524
        channel_spacing_m = 5.0
        event_id_for_title = "deep_saf"

    # --------------------------------------------------------------------------
    # 1. Build extended SAFOD initial model
    # --------------------------------------------------------------------------

    if source_mode == "catalog_event":
        # New real-event geometry file:
        #   X_2D_m = along-profile x
        #   Z_2D_m = TVD depth
        x_column = "X_2D_m"
        z_column = "Z_2D_m"
    else:
        # Old projected geometry file:
        #   model x <- Z_2D_m
        #   model z <- X_2D_m
        x_column = "Z_2D_m"
        z_column = "X_2D_m"

    grid, model, x_cable_raw, z_cable_raw, metadata = build_safod_model(
        geom_file=geom_file,
        x_column=x_column,
        z_column=z_column,
        build_initial_model=True,
        dx=dx,
        dz=dz,
        dt=None,
        nt=nt,
        half_order=half_order,
        cfl_safety=0.80,

        # Enlarged outward so that the wider sponge does not consume the
        # original physical model interior.
        x_padding_m=x_padding_m,
        z_max_m=z_max_m,
        z_padding_bottom_m=2500.0,

        z_tie_m=None,
        anchor_fault_to_cable_end=True,
        fault_offset_from_cable_m=105.0,
        fault_dip_deg=82.0,
        fault_dip_sign=-1.0,
        left_block_name="salinian",
        right_block_name="franciscan",
        initial_cross_fault_contrast=-0.08,
        initial_cross_fault_transition_m=350.0,
        initial_fault_zone_width_m=160.0,
        initial_fault_zone_velocity_reduction=0.14,
        include_pilot_hole_lvz_in_initial=True,
        initial_pilot_hole_lvz_strength=0.035,
        smooth_initial_sigma_m=80.0,
    )

    duration = float((grid.nt - 1) * grid.dt)

    print("\nSAFOD initial forward run")
    print("--------------------------------------")
    print(f"grid       : nx={grid.nx}, nz={grid.nz}, dx={grid.dx:.1f}, dz={grid.dz:.1f} m")
    print(f"dt, nt     : {grid.dt:.6e} s, {grid.nt}")
    print(f"duration   : {duration:.3f} s")
    print(f"Vp range   : {model.vp.min():.1f} to {model.vp.max():.1f} m/s")
    print(f"Vs range   : {model.vs.min():.1f} to {model.vs.max():.1f} m/s")
    print(f"rho range  : {model.rho.min():.1f} to {model.rho.max():.1f} kg/m^3")
    print(f"cable end  : x={metadata.x_cable_end_m:.1f} m, z={metadata.z_cable_end_m:.1f} m")
    print(f"SAF tie    : x={metadata.x_tie_m:.1f} m, z={metadata.z_tie_m:.1f} m")
    print(f"GL         : {gauge_length_m:.4f} m")
    print(f"receiver ds: {channel_spacing_m:.2f} m")

    # --------------------------------------------------------------------------
    # 2. Trim cable and build DAS receivers
    # --------------------------------------------------------------------------
    x_cable_use, z_cable_use = trim_cable_for_solver_domain(
        grid=grid,
        x_cable=x_cable_raw,
        z_cable=z_cable_raw,
        n_boundary=n_boundary,
        half_order=half_order,
        free_surface=free_surface,
    )

    receivers = build_das_cable(
        grid=grid,
        waypoints_x=x_cable_use.tolist(),
        waypoints_z=z_cable_use.tolist(),
        channel_spacing_m=channel_spacing_m,
        n_pml=0,
    )

    print("\nReceivers")
    print("---------")
    print(f"receivers  : {receivers.nrec} DAS channels")
    print(f"cable s    : {receivers.s[0]:.1f} to {receivers.s[-1]:.1f} m")
    print(f"receiver x : {receivers.x.min():.1f} to {receivers.x.max():.1f} m")
    print(f"receiver z : {receivers.z.min():.1f} to {receivers.z.max():.1f} m")

    # --------------------------------------------------------------------------
    # 3. Source
    # --------------------------------------------------------------------------
    if source_mode == "catalog_event":
        # Prepared real earthquake source projected into the 2D model.
        #
        # For NC75336802:
        #   origin    = 2026-04-01T04:57:57.470000Z
        #   M         = 0.77 Md
        #   depth     = 1.57 km
        #   x_model   = 1687.279 m
        #   crossline = 116.379 m
        #
        # This is a good 2D-compatible event geometrically, but small magnitude.
        x_src = float(event_cfg["x_src"])
        z_src = float(event_cfg["z_src"])

        source_theta_deg = float(args.theta_deg)
        source_scalar_moment = 1.0e12
        source_f0_hz = 10.0

        print("\nCatalog-event source")
        print("--------------------")
        print(f"event id      : {event_cfg['event_id']}")
        print(f"origin        : {event_cfg['origin_time']}")
        print(f"magnitude     : {event_cfg['magnitude']:.2f}")
        print(f"crossline     : {event_cfg['event_crossline_m']:.1f} m")
        print(f"x_src, z_src  : {x_src:.3f}, {z_src:.3f} m")
        print(f"theta         : {source_theta_deg:.1f} deg")
        print(f"f0            : {source_f0_hz:.2f} Hz")

    elif source_mode == "deep_saf":
        # Old synthetic source near the SAF prior.
        z_src_target_m = 5200.0

        z_src = float(
            np.clip(
                z_src_target_m,
                grid.z[0] + (n_boundary + half_order + 10) * grid.dz,
                grid.z[-1] - (n_boundary + half_order + 10) * grid.dz,
            )
        )

        x_fault_src = float(
            fault_x_at_z(
                z_src,
                x_tie_m=metadata.x_tie_m,
                z_tie_m=metadata.z_tie_m,
                fault_dip_deg=metadata.fault_dip_deg,
                fault_dip_sign=metadata.fault_dip_sign,
            )
        )

        x_src = float(x_fault_src - 80.0 + 0.37 * grid.dx)
        z_src = float(z_src + 0.61 * grid.dz)

        source_theta_deg = float(args.theta_deg)
        source_scalar_moment = 1.0e12
        source_f0_hz = 6.0

        print("\nDeep SAF synthetic source")
        print("-------------------------")
        print(f"x_src, z_src  : {x_src:.3f}, {z_src:.3f} m")
        print(f"theta         : {source_theta_deg:.1f} deg")
        print(f"f0            : {source_f0_hz:.2f} Hz")

    else:
        raise ValueError(f"Unknown source_mode: {source_mode!r}")

    check_source_inside_solver_domain(
        grid=grid,
        x_src=x_src,
        z_src=z_src,
        n_boundary=n_boundary,
        half_order=half_order,
    )

    source = build_dc_source(
        grid=grid,
        x_m=x_src,
        z_m=z_src,
        theta_deg=source_theta_deg,
        scalar_moment=source_scalar_moment,
        nt=grid.nt,
        dt=grid.dt,
        f0_hz=source_f0_hz,
        derivative_order=0,
        spreading="bilinear",
    )

    print("\nSource")
    print("------")
    print(source.summary())

    # --------------------------------------------------------------------------
    # 4. Straight-ray arrival QC and record-duration pre-check
    # --------------------------------------------------------------------------
    # Always compute physical travel times with zero source-time shift.
    # These arrays are reused later for the gather overlays and saved output.
    arrivals_receiver = compute_straight_ray_arrivals(
        grid=grid,
        model=model,
        receivers=receivers,
        x_src=source.x_embedded_m,
        z_src=source.z_embedded_m,
        n_samples_per_ray=256,
        time_shift_s=0.0,
    )

    max_s_arrival_s = float(np.max(arrivals_receiver["S"]))

    check_record_duration(
        grid=grid,
        max_s_arrival_s=max_s_arrival_s,
        min_tail_after_s_s=0.50,
    )

    print("\nApproximate P/S arrival overlays")
    print("--------------------------------")
    print(f"method      : {arrivals_receiver['method']}")
    print(f"time shift  : {arrivals_receiver['time_shift_s']:.3f} s")
    print(
        f"P range     : {arrivals_receiver['P'].min():.3f} to "
        f"{arrivals_receiver['P'].max():.3f} s"
    )
    print(
        f"S range     : {arrivals_receiver['S'].min():.3f} to "
        f"{arrivals_receiver['S'].max():.3f} s"
    )

    # --------------------------------------------------------------------------
    # 5. Save model plot with source overlay
    # --------------------------------------------------------------------------
    fig, ax = plot_safod_model(
        grid=grid,
        model=model,
        x_cable=x_cable_use,
        z_cable=z_cable_use,
        metadata=metadata,
        field="vp",
        show_fault=True,
        show_tie_point=True,
        show_offset_segment=True,
    )

    ax.scatter(
        [source.x_embedded_m],
        [source.z_embedded_m],
        marker="*",
        s=150,
        c="yellow",
        edgecolors="black",
        zorder=20,
        label="Synthetic source",
    )

    # Re-place the legend using the same robust figure-fraction coordinates
    # as plot_safod_model. Do NOT call fig.tight_layout() after this — it
    # would undo the reserved right margin and re-overlap the colorbar.
    place_safod_legend(fig, ax, fontsize=8)

    fig.savefig(out_dir / "01_model_vp_with_source.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # 6. Run forward simulation
    # --------------------------------------------------------------------------
    snapshot_stride = 300 if args.save_gif else None

    print("\nRunning forward simulation...")
    print(f"snapshot stride: {snapshot_stride}")
    run_result, das_result = run_forward_simulation(
        model=model,
        source=source,
        receivers=receivers,
        gauge_length_m=gauge_length_m,
        half_order=half_order,
        use_ts_sfd=False,
        n_boundary=n_boundary,
        gamma_s=gamma_s,
        snapshot_stride=snapshot_stride,
        backend="numba_fused",
        free_surface=free_surface,
    )
    print("Forward simulation finished.")

    # --------------------------------------------------------------------------
    # 6b. Wavefield propagation GIF
    # --------------------------------------------------------------------------
    if run_result.snapshots_vz is not None and run_result.snapshot_times_v is not None:
        make_wavefield_gif(
            grid=grid,
            snapshots_vz=run_result.snapshots_vz,
            snapshot_times=run_result.snapshot_times_v,
            x_cable=x_cable_use,
            z_cable=z_cable_use,
            metadata=metadata,
            source=source,
            out_path=out_dir / "05_wavefield_vz_moment_tensor.gif",
            title=f"SAFOD Vz wavefield: {event_id_for_title}, double-couple source",
            fps=6,
            max_frames=80,
            percentile_clip=99.5,
        )
    else:
        print("No Vz snapshots available; skipping wavefield GIF.")

    # --------------------------------------------------------------------------
    # 7. Sanity checks
    # --------------------------------------------------------------------------
    for name, arr in [
        ("receiver_vx", run_result.receiver_vx),
        ("receiver_vz", run_result.receiver_vz),
        ("das_data", das_result.data),
    ]:
        arr = np.asarray(arr)

        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"{name} contains NaN/Inf.")

        print(
            f"{name:12s}: shape={arr.shape}, "
            f"max_abs={np.max(np.abs(arr)):.6e}, "
            f"p99_abs={np.percentile(np.abs(arr), 99.0):.6e}"
        )

    print("\nDAS operator")
    print("------------")
    print(f"gauge_length_m : {das_result.gauge_length_m:.6f}")
    print(f"gauge_samples  : {das_result.gauge_samples:.6f}")
    print(f"nchan_out      : {das_result.nchan_out}")

    # --------------------------------------------------------------------------
    # 7b. DAS-channel subset of the precomputed first-arrival curves
    # --------------------------------------------------------------------------
    source_time_shift_s = float(arrivals_receiver["time_shift_s"])

    arrivals_das = subset_arrivals(
        arrivals_receiver,
        das_result.channel_indices,
    )

    # --------------------------------------------------------------------------
    # 8. Save figures
    # --------------------------------------------------------------------------
    plot_receiver_gather(
        t=run_result.t,
        data=run_result.receiver_vx,
        receivers=receivers,
        title="SAFOD forward: receiver Vx",
        cbar_label="Vx [m/s]",
        out_path=out_dir / "02_receiver_vx.png",
        normalized=False,
        arrival_times=arrivals_receiver,
    )

    plot_receiver_gather(
        t=run_result.t,
        data=run_result.receiver_vz,
        receivers=receivers,
        title="SAFOD forward: receiver Vz",
        cbar_label="Vz [m/s]",
        out_path=out_dir / "03_receiver_vz.png",
        normalized=False,
        arrival_times=arrivals_receiver,
    )

    plot_das_gather(
        t=run_result.t,
        das_result=das_result,
        receivers=receivers,
        title=f"SAFOD forward {event_id_for_title}: DAS strain-rate, GL={gauge_length_m:.4f} m",
        out_path=out_dir / "04_das_strain_rate.png",
        normalized=False,
        arrival_times=arrivals_das,
    )

    plot_receiver_gather(
        t=run_result.t,
        data=run_result.receiver_vx,
        receivers=receivers,
        title="SAFOD forward: receiver Vx trace-normalized",
        cbar_label="Trace-normalized amplitude",
        out_path=out_dir / "02b_receiver_vx_normalized.png",
        normalized=True,
        arrival_times=arrivals_receiver,
    )

    plot_receiver_gather(
        t=run_result.t,
        data=run_result.receiver_vz,
        receivers=receivers,
        title="SAFOD forward: receiver Vz trace-normalized",
        cbar_label="Trace-normalized amplitude",
        out_path=out_dir / "03b_receiver_vz_normalized.png",
        normalized=True,
        arrival_times=arrivals_receiver,
    )

    plot_das_gather(
        t=run_result.t,
        das_result=das_result,
        receivers=receivers,
        title=f"SAFOD forward {event_id_for_title}: DAS trace-normalized, GL={gauge_length_m:.4f} m",
        out_path=out_dir / "04b_das_strain_rate_normalized.png",
        normalized=True,
        arrival_times=arrivals_das,
    )

    # --------------------------------------------------------------------------
    # 9. Save arrays
    # --------------------------------------------------------------------------
    np.savez_compressed(
        output_package,

        t=run_result.t,
        t_sigma=run_result.t_sigma,

        receiver_vx=run_result.receiver_vx,
        receiver_vz=run_result.receiver_vz,

        das_data=das_result.data,
        das_channel_indices=das_result.channel_indices,
        das_gauge_samples=np.array(das_result.gauge_samples),
        das_gauge_length_m=np.array(das_result.gauge_length_m),

        receiver_x=receivers.x,
        receiver_z=receivers.z,
        receiver_s=receivers.s,

        arrival_s_receiver=arrivals_receiver["s"],
        arrival_p_receiver=arrivals_receiver["P"],
        arrival_swave_receiver=arrivals_receiver["S"],

        arrival_s_das=arrivals_das["s"],
        arrival_p_das=arrivals_das["P"],
        arrival_swave_das=arrivals_das["S"],
        arrival_time_shift_s=np.array(source_time_shift_s),
        arrival_method=np.array(arrivals_receiver["method"]),

        x_cable_raw=x_cable_raw,
        z_cable_raw=z_cable_raw,
        x_cable_used=x_cable_use,
        z_cable_used=z_cable_use,

        source_x=np.array(source.x_embedded_m),
        source_z=np.array(source.z_embedded_m),
        source_ix=np.array(source.ix),
        source_iz=np.array(source.iz),
        source_spreading=np.array(source.spreading),
        source_mode=np.array(source_mode),
        event_id=np.array(event_cfg["event_id"] if event_cfg is not None else "deep_saf"),
        event_origin_time=np.array(event_cfg["origin_time"] if event_cfg is not None else ""),
        event_crossline_m=np.array(event_cfg["event_crossline_m"] if event_cfg is not None else np.nan),

        source_theta_deg=np.array(source_theta_deg),
        source_f0_hz=np.array(source_f0_hz),
        source_scalar_moment=np.array(source_scalar_moment),

        run_tag=np.array(run_tag),
        n_boundary=np.array(n_boundary),
        gamma_s=np.array(gamma_s),
        free_surface=np.array(free_surface),
        extra_scientific_x_margin_m=np.array(
            extra_scientific_x_margin_m
        ),
        x_padding_m=np.array(x_padding_m),
        z_max_m=np.array(z_max_m),

        grid_x=grid.x,
        grid_z=grid.z,
        dx=np.array(grid.dx),
        dz=np.array(grid.dz),
        dt=np.array(grid.dt),
        nt=np.array(grid.nt),

        vp=model.vp,
        vs=model.vs,
        rho=model.rho,

        x_fault_line=metadata.x_fault_line,
        z_fault_line=metadata.z_fault_line,
        x_tie_m=np.array(metadata.x_tie_m),
        z_tie_m=np.array(metadata.z_tie_m),
        fault_offset_from_cable_m=np.array(metadata.fault_offset_from_cable_m),
        fault_dip_deg=np.array(metadata.fault_dip_deg),
        fault_dip_sign=np.array(metadata.fault_dip_sign),
        model_type=np.array(metadata.model_type),
    )

    if run_result.snapshots_vz is not None:
        np.savez_compressed(
            out_dir / "snapshots_vz.npz",
            snapshots_vz=run_result.snapshots_vz,
            snapshot_times_v=run_result.snapshot_times_v,
        )

    print(f"\nSaved results to: {out_dir.absolute()}")
    print("SAFOD forward run PASSED.")


if __name__ == "__main__":
    main()