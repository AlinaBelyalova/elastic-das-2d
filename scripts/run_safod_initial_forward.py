# ==============================================================================
# scripts/run_safod_initial_forward.py
#
# First forward simulation on the SAFOD geologically informed initial model.
#
# Purpose
# -------
#   - build SAFOD initial model
#   - build projected DAS receiver cable
#   - place one synthetic double-couple source near the SAF/cable end
#   - run backend="numba_fused"
#   - save Vx, Vz, DAS gathers and metadata
#
# This is forward modelling
# ==============================================================================

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.safod_builder import build_safod_model, fault_x_at_z
from src.source import build_dc_source
from src.receivers import build_das_cable
from src.simulator import run_forward_simulation
from src.plotting import plot_safod_model


# ==============================================================================
# SMALL HELPERS
# ==============================================================================

def normalize_traces(data: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Trace-normalize a gather for display.
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
      - but the first half_order z-nodes are ghost/stencil nodes, so receivers
        should stay below them.

    For side and bottom boundaries:
      - keep cable points outside the sponge region.
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
    Cheap pre-check before building source / running solver.
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
            "Adjust source offset, fault_offset_from_cable_m, or enlarge model padding."
        )

    print("\nSource boundary pre-check")
    print("-------------------------")
    print(f"source x,z : {x_src:.1f}, {z_src:.1f} m")
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
    min_tail_after_s_s: float = 0.30,
) -> None:
    """
    Estimate whether the record is long enough to see arrivals at far channels.

    This is an approximate QC check, not ray tracing.
    Uses median Vp/Vs as representative propagation speeds.
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
    print(f"duration             : {duration:.3f} s")
    print(f"max source-receiver distance: {dmax:.1f} m")
    print(f"median Vp / Vs       : {vp_ref:.1f} / {vs_ref:.1f} m/s")
    print(f"estimated far P      : {t_p_far:.3f} s")
    print(f"estimated far S      : {t_s_far:.3f} s")
    print(f"tail after far S     : {tail_after_s:.3f} s")

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
    """
    Plot receiver gather as cable arc length vs time.
    """
    data = np.asarray(data, dtype=np.float64)
    arr = normalize_traces(data) if normalized else data

    vmax = 1.0 if normalized else float(np.percentile(np.abs(arr), 99.0))
    if vmax == 0.0 or not np.isfinite(vmax):
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(10, 6))

    extent = [float(t[0]), float(t[-1]), float(receivers.s[-1]), float(receivers.s[0])]

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
    """
    Plot DAS strain-rate gather.
    """
    data = np.asarray(das_result.data, dtype=np.float64)
    arr = normalize_traces(data) if normalized else data

    s_valid = receivers.s[das_result.channel_indices]

    vmax = 1.0 if normalized else float(np.percentile(np.abs(arr), 99.0))
    if vmax == 0.0 or not np.isfinite(vmax):
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(10, 6))

    extent = [float(t[0]), float(t[-1]), float(s_valid[-1]), float(s_valid[0])]

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


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    geom_file = "/home/groups/ettore88/alina/imaging/SAFOD_downleg_Projected_2D.csv"

    out_dir = Path("results/safod_initial_forward")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    # Numerical settings
    # --------------------------------------------------------------------------
    dx = 10.0
    dz = 10.0

    # nt=3500 gave only a very small tail after estimated far S arrival.
    # Use 5000 for a safer QC record.
    nt = 5000

    half_order = 2
    n_boundary = 40
    gamma_s = 80.0
    free_surface = True

    gauge_length_m = 20.0
    channel_spacing_m = 10.0

    # --------------------------------------------------------------------------
    # 1. Build SAFOD initial model
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
    print("-------------------------")
    print(f"grid       : nx={grid.nx}, nz={grid.nz}, dx={grid.dx:.1f}, dz={grid.dz:.1f} m")
    print(f"dt, nt     : {grid.dt:.6e} s, {grid.nt}")
    print(f"duration   : {duration:.3f} s")
    print(f"Vp range   : {model.vp.min():.1f} to {model.vp.max():.1f} m/s")
    print(f"Vs range   : {model.vs.min():.1f} to {model.vs.max():.1f} m/s")
    print(f"rho range  : {model.rho.min():.1f} to {model.rho.max():.1f} kg/m^3")
    print(f"cable end  : x={metadata.x_cable_end_m:.1f} m, z={metadata.z_cable_end_m:.1f} m")
    print(f"SAF tie    : x={metadata.x_tie_m:.1f} m, z={metadata.z_tie_m:.1f} m")

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
    # 3. Source placement near SAF / cable end
    # --------------------------------------------------------------------------
    z_src = float(metadata.z_cable_end_m - 350.0)
    z_src = float(
        np.clip(
            z_src,
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

    # Source near the fault, slightly on cable side, deliberately off-grid.
    x_src = float(x_fault_src - 60.0 + 0.37 * grid.dx)
    z_src = float(z_src + 0.61 * grid.dz)

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
        theta_deg=35.0,
        scalar_moment=1.0e12,
        nt=grid.nt,
        dt=grid.dt,
        f0_hz=8.0,
        derivative_order=0,
        spreading="bilinear",
    )

    print("\nSource")
    print("------")
    print(source.summary())

    # --------------------------------------------------------------------------
    # 4. Timing pre-check after receivers and source are known
    # --------------------------------------------------------------------------
    check_record_duration(
        grid=grid,
        model=model,
        receivers=receivers,
        x_src=source.x_embedded_m,
        z_src=source.z_embedded_m,
        min_tail_after_s_s=0.30,
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
        s=140,
        c="yellow",
        edgecolors="black",
        zorder=20,
        label="Synthetic source",
    )
    
    ax.legend(loc="upper left", bbox_to_anchor=(1.22, 1.0), fontsize=8)
    fig.tight_layout()
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
        snapshot_stride=250,
        backend="numba_fused",
        free_surface=free_surface,
    )
    print("Forward simulation finished.")

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

    # --------------------------------------------------------------------------
    # 8. Save figures
    # --------------------------------------------------------------------------
    plot_receiver_gather(
        t=run_result.t,
        data=run_result.receiver_vx,
        receivers=receivers,
        title="SAFOD initial forward: receiver Vx",
        cbar_label="Vx [m/s]",
        out_path=out_dir / "02_receiver_vx.png",
        normalized=False,
    )

    plot_receiver_gather(
        t=run_result.t,
        data=run_result.receiver_vz,
        receivers=receivers,
        title="SAFOD initial forward: receiver Vz",
        cbar_label="Vz [m/s]",
        out_path=out_dir / "03_receiver_vz.png",
        normalized=False,
    )

    plot_das_gather(
        t=run_result.t,
        das_result=das_result,
        receivers=receivers,
        title=f"SAFOD initial forward: DAS strain-rate, gauge={gauge_length_m:.1f} m",
        out_path=out_dir / "04_das_strain_rate.png",
        normalized=False,
    )

    plot_receiver_gather(
        t=run_result.t,
        data=run_result.receiver_vx,
        receivers=receivers,
        title="SAFOD initial forward: receiver Vx trace-normalized",
        cbar_label="Trace-normalized amplitude",
        out_path=out_dir / "02b_receiver_vx_normalized.png",
        normalized=True,
    )

    plot_receiver_gather(
        t=run_result.t,
        data=run_result.receiver_vz,
        receivers=receivers,
        title="SAFOD initial forward: receiver Vz trace-normalized",
        cbar_label="Trace-normalized amplitude",
        out_path=out_dir / "03b_receiver_vz_normalized.png",
        normalized=True,
    )

    plot_das_gather(
        t=run_result.t,
        das_result=das_result,
        receivers=receivers,
        title="SAFOD initial forward: DAS trace-normalized",
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
        das_gauge_samples=das_result.gauge_samples,
        das_gauge_length_m=das_result.gauge_length_m,

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
    print("SAFOD initial forward run PASSED.")


if __name__ == "__main__":
    main()