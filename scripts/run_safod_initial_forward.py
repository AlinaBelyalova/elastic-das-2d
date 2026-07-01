# ==============================================================================
# scripts/run_safod_initial_forward.py
#
# SAFOD initial-model forward simulation.
#
# This script is a QC forward run, not FWI yet.
#
# Current mode:
#   deep_saf source:
#       synthetic local-earthquake-like source below the DAS cable,
#       near the SAF prior line, inside an extended model domain.
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

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.safod_builder import build_safod_model, fault_x_at_z
from src.source import build_dc_source
from src.receivers import build_das_cable
from src.simulator import run_forward_simulation
from src.plotting import plot_safod_model, place_safod_legend
from matplotlib.animation import FuncAnimation, PillowWriter


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
    model,
    receivers,
    x_src: float,
    z_src: float,
    min_tail_after_s_s: float = 0.50,
) -> None:
    """
    Approximate timing QC.

    Uses straight-line distance and median Vp/Vs only. This is not ray tracing,
    but catches obviously too-short simulations before the run.
    """
    rx = np.asarray(receivers.x, dtype=np.float64)
    rz = np.asarray(receivers.z, dtype=np.float64)

    dist = np.sqrt((rx - x_src) ** 2 + (rz - z_src) ** 2)
    dmax = float(np.max(dist))

    vp_ref = float(np.percentile(model.vp, 50.0))
    vs_ref = float(np.percentile(model.vs, 50.0))

    t_p_far = dmax / vp_ref
    t_s_far = dmax / vs_ref

    duration = float((grid.nt - 1) * grid.dt)
    tail_after_s = duration - t_s_far

    print("\nRecord-duration QC")
    print("------------------")
    print(f"duration                    : {duration:.3f} s")
    print(f"max source-receiver distance: {dmax:.1f} m")
    print(f"median Vp / Vs              : {vp_ref:.1f} / {vs_ref:.1f} m/s")
    print(f"estimated far P             : {t_p_far:.3f} s")
    print(f"estimated far S             : {t_s_far:.3f} s")
    print(f"tail after far S            : {tail_after_s:.3f} s")

    if tail_after_s < min_tail_after_s_s:
        required_duration = t_s_far + min_tail_after_s_s
        suggested_nt = int(np.ceil(required_duration / grid.dt)) + 1

        raise ValueError(
            "Record is probably too short for useful QC after far S arrivals. "
            f"tail_after_s={tail_after_s:.3f} s, required >= {min_tail_after_s_s:.3f} s. "
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
# MAIN
# ==============================================================================

def main() -> None:
    geom_file = "/home/groups/ettore88/alina/imaging/SAFOD_downleg_Projected_2D.csv"

    out_dir = Path("results/safod_initial_forward")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    # Numerical settings for deep-source QC
    # --------------------------------------------------------------------------
    # Use 10 m for the first deep-source run. This keeps the model affordable.
    # Once the wavefield looks physically reasonable, repeat with dx=5 m.
    dx = 10.0
    dz = 10.0

    # With dt ~4.8e-4 s, nt=8500 gives about 4.1 s.
    nt = 8500

    half_order = 2
    n_boundary = 40
    gamma_s = 80.0
    free_surface = True

    # Realistic-ish Nano/Sintela gauge length. This does not need to be
    # a multiple of channel spacing after the src.das continuous-GL fix.
    gauge_length_m = 16.6213
    channel_spacing_m = 10.0

    # --------------------------------------------------------------------------
    # 1. Build extended SAFOD initial model
    # --------------------------------------------------------------------------
    grid, model, x_cable_raw, z_cable_raw, metadata = build_safod_model(
        geom_file=geom_file,

        # Axis fix:
        # model x <- CSV Z_2D_m
        # model z <- CSV X_2D_m
        x_column="Z_2D_m",
        z_column="X_2D_m",

        build_initial_model=True,

        dx=dx,
        dz=dz,
        dt=None,
        nt=nt,
        half_order=half_order,
        cfl_safety=0.80,

        # Important for deep source:
        # extend the model well below the DAS cable.
        z_max_m=6500.0,
        z_padding_bottom_m=900.0,

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
    # 3. Deep local-earthquake-like source near SAF
    # --------------------------------------------------------------------------
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

    # Put source near the SAF, slightly on the cable side.
    # Fractions make it deliberately off-grid for bilinear spreading.
    x_src = float(x_fault_src - 80.0 + 0.37 * grid.dx)
    z_src = float(z_src + 0.61 * grid.dz)

    check_source_inside_solver_domain(
        grid=grid,
        x_src=x_src,
        z_src=z_src,
        n_boundary=n_boundary,
        half_order=half_order,
    )

    # Longer path -> slightly lower dominant frequency than near-cable QC.
    source = build_dc_source(
        grid=grid,
        x_m=x_src,
        z_m=z_src,
        theta_deg=35.0,
        scalar_moment=1.0e12,
        nt=grid.nt,
        dt=grid.dt,
        f0_hz=6.0,
        derivative_order=0,
        spreading="bilinear",
    )

    print("\nSource")
    print("------")
    print(source.summary())

    # --------------------------------------------------------------------------
    # 4. Timing pre-check
    # --------------------------------------------------------------------------
    check_record_duration(
        grid=grid,
        model=model,
        receivers=receivers,
        x_src=source.x_embedded_m,
        z_src=source.z_embedded_m,
        min_tail_after_s_s=0.50,
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
    print("\nRunning forward simulation...")
    run_result, das_result = run_forward_simulation(
        model=model,
        source=source,
        receivers=receivers,
        gauge_length_m=gauge_length_m,
        half_order=half_order,
        use_ts_sfd=False,
        n_boundary=n_boundary,
        gamma_s=gamma_s,
        snapshot_stride=300,
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
            title="SAFOD Vz wavefield propagation: double-couple moment tensor",
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
    )

    plot_receiver_gather(
        t=run_result.t,
        data=run_result.receiver_vz,
        receivers=receivers,
        title="SAFOD forward: receiver Vz",
        cbar_label="Vz [m/s]",
        out_path=out_dir / "03_receiver_vz.png",
        normalized=False,
    )

    plot_das_gather(
        t=run_result.t,
        das_result=das_result,
        receivers=receivers,
        title=f"SAFOD forward: DAS strain-rate, GL={gauge_length_m:.4f} m",
        out_path=out_dir / "04_das_strain_rate.png",
        normalized=False,
    )

    plot_receiver_gather(
        t=run_result.t,
        data=run_result.receiver_vx,
        receivers=receivers,
        title="SAFOD forward: receiver Vx trace-normalized",
        cbar_label="Trace-normalized amplitude",
        out_path=out_dir / "02b_receiver_vx_normalized.png",
        normalized=True,
    )

    plot_receiver_gather(
        t=run_result.t,
        data=run_result.receiver_vz,
        receivers=receivers,
        title="SAFOD forward: receiver Vz trace-normalized",
        cbar_label="Trace-normalized amplitude",
        out_path=out_dir / "03b_receiver_vz_normalized.png",
        normalized=True,
    )

    plot_das_gather(
        t=run_result.t,
        das_result=das_result,
        receivers=receivers,
        title=f"SAFOD forward: DAS trace-normalized, GL={gauge_length_m:.4f} m",
        out_path=out_dir / "04b_das_strain_rate_normalized.png",
        normalized=True,
    )

    # --------------------------------------------------------------------------
    # 9. Save arrays
    # --------------------------------------------------------------------------
    np.savez_compressed(
        out_dir / "outputs_safod_initial_forward.npz",

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

        x_cable_raw=x_cable_raw,
        z_cable_raw=z_cable_raw,
        x_cable_used=x_cable_use,
        z_cable_used=z_cable_use,

        source_x=np.array(source.x_embedded_m),
        source_z=np.array(source.z_embedded_m),
        source_ix=np.array(source.ix),
        source_iz=np.array(source.iz),
        source_spreading=np.array(source.spreading),
        source_theta_deg=np.array(35.0),
        source_f0_hz=np.array(6.0),
        source_scalar_moment=np.array(1.0e12),

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