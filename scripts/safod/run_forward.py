# ==============================================================================
# scripts/safod/run_forward.py
#
# SAFOD initial-model forward simulation.
#
# This script is a QC forward run, not FWI yet.
#
# Current default mode:
#   catalog_event:
#       projected source and event-specific registered down-going SAFOD DAS
#       geometry. The alternative deep_saf mode remains available for
#       controlled synthetic QC.
#
# Requirements:
#   - src.safod.models.factory.build_initial_model
#   - src.safod.models.smooth_prior.fault_x_at_z
#   - src.das supports continuous physical gauge lengths, e.g. GL=16.6213 m
#   - src.plotting.place_safod_legend for figure-fraction legend placement
#     (do NOT call fig.tight_layout() after plot_safod_model() — it would
#     undo the reserved right margin and re-overlap the colorbar)
# ==============================================================================

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.safod.models import (
    AVAILABLE_INITIAL_MODELS,
    build_initial_model,
    fault_x_at_z,
)
from src.source import build_dc_source
from src.receivers import (
    build_das_cable,
    build_receivers_from_channel_centres,
)
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
# PLOT STYLE
# ==============================================================================

AXIS_LABEL_FONTSIZE = 14
TICK_LABEL_FONTSIZE = 11
TITLE_FONTSIZE = 13
COLORBAR_LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 9


def style_axis_text_bold(
    ax,
    *,
    is_colorbar: bool = False,
) -> None:
    """
    Make every visible text element on one Matplotlib axis bold.

    This covers:
      - x/y axis labels;
      - x/y tick labels;
      - titles;
      - annotations created with ax.text();
      - legend entries and legend title;
      - colorbar label/ticks, because a colorbar is also a Matplotlib axis.
    """
    label_size = (
        COLORBAR_LABEL_FONTSIZE
        if is_colorbar
        else AXIS_LABEL_FONTSIZE
    )

    ax.xaxis.label.set_fontsize(label_size)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontsize(label_size)
    ax.yaxis.label.set_fontweight("bold")

    ax.title.set_fontsize(TITLE_FONTSIZE)
    ax.title.set_fontweight("bold")

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_LABEL_FONTSIZE,
        width=1.2,
        length=5,
    )

    for label in ax.get_xticklabels():
        label.set_fontweight("bold")

    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    for annotation in ax.texts:
        annotation.set_fontweight("bold")

    legend = ax.get_legend()
    if legend is not None:
        for legend_text in legend.get_texts():
            legend_text.set_fontweight("bold")
            legend_text.set_fontsize(LEGEND_FONTSIZE)

        legend_title = legend.get_title()
        if legend_title is not None:
            legend_title.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def style_figure_text_bold(
    fig,
    *,
    main_axis=None,
) -> None:
    """
    Apply bold styling to every axis in a figure, including colorbars.

    main_axis can be supplied to distinguish the main plotting axis from
    auxiliary colorbar axes.
    """
    for axis in fig.axes:
        style_axis_text_bold(
            axis,
            is_colorbar=(
                main_axis is not None
                and axis is not main_axis
            ),
        )

    figure_title = getattr(
        fig,
        "_suptitle",
        None,
    )
    if figure_title is not None:
        figure_title.set_fontweight("bold")
        figure_title.set_fontsize(TITLE_FONTSIZE)


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
    Load prepared real-event metadata and channel registration.

    The package is created by scripts.safod.prepare_event.  For catalogue-event
    modelling, ``raw_channels`` is the authoritative real-data channel axis and
    must match one channel-identification column in the registered geometry CSV.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Real-event package not found: {path}"
        )

    with np.load(
        path,
        allow_pickle=True,
    ) as pkg:

        def get_scalar(name: str, default=None):
            if name not in pkg.files:
                if default is None:
                    raise KeyError(
                        f"Missing {name!r} in real-event package: {path}"
                    )
                return default

            value = np.asarray(
                pkg[name]
            )

            if value.shape == ():
                return value.item()

            if value.size == 1:
                return value.reshape(-1)[0].item()

            return value.copy()

        event_dir = path.parent

        if "geometry_csv" in pkg.files:
            geom_file = Path(
                str(
                    get_scalar(
                        "geometry_csv"
                    )
                )
            )
            if not geom_file.exists():
                relocated_geom_file = event_dir / geom_file.name
                if relocated_geom_file.exists():
                    geom_file = relocated_geom_file

        else:
            geom_file = (
                event_dir
                / "SAFOD_Phase2_projected_from_georef.csv"
            )

        # Saved paths are normally relative to the project root.  If a package
        # contains only a filename, also allow resolution relative to event_dir.
        if not geom_file.exists() and not geom_file.is_absolute():
            event_relative = (
                event_dir
                / geom_file
            )
            if event_relative.exists():
                geom_file = event_relative

        if "raw_channels" not in pkg.files:
            raise KeyError(
                "Prepared real-event package does not contain 'raw_channels'. "
                "Rerun scripts.safod.prepare_event."
            )

        real_raw_channels = np.asarray(
            pkg["raw_channels"],
            dtype=np.float64,
        ).copy()

        cfg = {
            "package_path": str(path),
            "event_dir": str(event_dir),
            "geom_file": str(geom_file),

            "event_id": str(
                get_scalar(
                    "ev_id",
                    "unknown",
                )
            ),
            "origin_time": str(
                get_scalar(
                    "ev_origin_time",
                    "unknown",
                )
            ),
            "magnitude": float(
                get_scalar(
                    "ev_mag",
                    np.nan,
                )
            ),
            "depth_km": float(
                get_scalar(
                    "ev_depth_km",
                    np.nan,
                )
            ),

            "x_src": float(
                get_scalar(
                    "event_x_model_m"
                )
            ),
            "z_src": float(
                get_scalar(
                    "event_z_model_m"
                )
            ),
            "event_along_profile_m": float(
                get_scalar(
                    "event_along_profile_m",
                    np.nan,
                )
            ),
            "event_crossline_m": float(
                get_scalar(
                    "event_crossline_m",
                    np.nan,
                )
            ),

            "gauge_length_m": float(
                get_scalar(
                    "gauge_length_m"
                )
            ),
            "real_channel_spacing_m": float(
                get_scalar(
                    "channel_spacing_m"
                )
            ),
            "real_raw_channels": real_raw_channels,
        }

    if real_raw_channels.ndim != 1:
        raise ValueError(
            "real-event raw_channels must be one-dimensional; "
            f"got shape {real_raw_channels.shape}."
        )

    if real_raw_channels.size < 2:
        raise ValueError(
            "real-event package contains fewer than two raw channels."
        )

    if not np.all(
        np.isfinite(real_raw_channels)
    ):
        raise ValueError(
            "real-event raw_channels contains NaN or Inf."
        )

    if (
        not np.isfinite(
            cfg["gauge_length_m"]
        )
        or cfg["gauge_length_m"] <= 0.0
    ):
        raise ValueError(
            "Prepared gauge length must be finite and positive; "
            f"got {cfg['gauge_length_m']}."
        )

    if (
        not np.isfinite(
            cfg["real_channel_spacing_m"]
        )
        or cfg["real_channel_spacing_m"] <= 0.0
    ):
        raise ValueError(
            "Prepared channel spacing must be finite and positive; "
            f"got {cfg['real_channel_spacing_m']}."
        )

    print("\nLoaded real-event package")
    print("-------------------------")
    print(f"package          : {cfg['package_path']}")
    print(f"event id         : {cfg['event_id']}")
    print(f"origin           : {cfg['origin_time']}")
    print(f"magnitude        : {cfg['magnitude']:.2f}")
    print(f"depth            : {cfg['depth_km']:.2f} km")
    print(
        "source x,z       : "
        f"{cfg['x_src']:.3f}, {cfg['z_src']:.3f} m"
    )
    print(
        "crossline        : "
        f"{cfg['event_crossline_m']:.3f} m"
    )
    print(
        "gauge length     : "
        f"{cfg['gauge_length_m']:.6f} m"
    )
    print(
        "real dCh         : "
        f"{cfg['real_channel_spacing_m']:.6f} m"
    )
    print(
        "real channels    : "
        f"{real_raw_channels.size} "
        f"({real_raw_channels.min():.1f} to "
        f"{real_raw_channels.max():.1f})"
    )
    print(f"geometry csv     : {cfg['geom_file']}")

    return cfg


def load_registered_channel_geometry(
    *,
    path: str | Path,
    x_column: str,
    z_column: str,
    expected_raw_channels: np.ndarray,
) -> dict:
    """
    Load exact registered field channel centres from the event geometry CSV.

    The geometry row order must match ``raw_channels`` stored in the prepared
    real-event package.  A matching channel-identification column is selected
    explicitly rather than inferred from row position.
    """
    path = Path(
        path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Registered geometry CSV not found: {path}"
        )

    table = pd.read_csv(
        path
    )

    for column in (
        x_column,
        z_column,
    ):
        if column not in table.columns:
            raise ValueError(
                f"Missing required geometry column {column!r} in {path}."
            )

    x = pd.to_numeric(
        table[x_column],
        errors="coerce",
    ).to_numpy(
        dtype=np.float64
    )

    z = pd.to_numeric(
        table[z_column],
        errors="coerce",
    ).to_numpy(
        dtype=np.float64
    )

    expected = np.asarray(
        expected_raw_channels,
        dtype=np.float64,
    )

    if expected.ndim != 1:
        raise ValueError(
            "expected_raw_channels must be one-dimensional."
        )

    if not (
        x.size
        == z.size
        == expected.size
    ):
        raise ValueError(
            "Registered geometry and prepared real-event data have different "
            "row counts: "
            f"geometry={x.size}, real={expected.size}."
        )

    if not (
        np.all(
            np.isfinite(x)
        )
        and np.all(
            np.isfinite(z)
        )
    ):
        raise ValueError(
            "Registered geometry x/z contains NaN or Inf."
        )

    candidate_columns = (
        "Channel",
        "DataRow",
        "AcquisitionChannel",
        "ReferenceChannel",
    )

    matched_column = None
    raw_channels = None
    candidate_diagnostics = []

    for column in candidate_columns:
        if column not in table.columns:
            continue

        values = pd.to_numeric(
            table[column],
            errors="coerce",
        ).to_numpy(
            dtype=np.float64
        )

        candidate_diagnostics.append(
            (
                column,
                values,
            )
        )

        if (
            values.size == expected.size
            and np.all(
                np.isfinite(values)
            )
            and np.allclose(
                values,
                expected,
                rtol=0.0,
                atol=1.0e-6,
            )
        ):
            matched_column = column
            raw_channels = values
            break

    if matched_column is None:
        available = [
            name
            for name, _
            in candidate_diagnostics
        ]

        raise ValueError(
            "No channel-identification column in the registered geometry "
            "matches real_event_package['raw_channels']. "
            f"Tried columns: {available}. "
            f"Expected range: {expected.min():.1f} to {expected.max():.1f}."
        )

    print("\nRegistered field geometry")
    print("-------------------------")
    print(f"rows                : {x.size}")
    print(f"x column            : {x_column}")
    print(f"z column            : {z_column}")
    print(f"channel column      : {matched_column}")
    print(
        "raw-channel range   : "
        f"{raw_channels.min():.1f} to {raw_channels.max():.1f}"
    )
    print(
        "x range             : "
        f"{x.min():.3f} to {x.max():.3f} m"
    )
    print(
        "z range             : "
        f"{z.min():.3f} to {z.max():.3f} m"
    )

    return {
        "table": table,
        "x": x,
        "z": z,
        "raw_channels": raw_channels,
        "channel_column": matched_column,
        "row_indices": np.arange(
            x.size,
            dtype=np.int64,
        ),
    }


def compute_gauge_curvature_qc(
    *,
    receivers,
    gauge_length_m: float,
) -> dict:
    """
    Quantify projected tangent rotation across one physical gauge length.

    The current DAS forward operator uses the centre tangent for each gauge.
    Small endpoint-to-endpoint tangent rotation confirms that this locally
    straight-gauge approximation is appropriate for the registered cable.
    """
    s = np.asarray(
        receivers.s,
        dtype=np.float64,
    )
    tx = np.asarray(
        receivers.tx,
        dtype=np.float64,
    )
    tz = np.asarray(
        receivers.tz,
        dtype=np.float64,
    )

    if receivers.nrec < 2:
        raise ValueError(
            "At least two receivers are required for curvature QC."
        )

    ds = float(
        receivers.channel_spacing
    )

    if not np.isfinite(ds) or ds <= 0.0:
        raise ValueError(
            f"Invalid receiver spacing for curvature QC: {ds}."
        )

    gauge_length_m = float(
        gauge_length_m
    )

    if (
        not np.isfinite(
            gauge_length_m
        )
        or gauge_length_m <= 0.0
    ):
        raise ValueError(
            "gauge_length_m must be finite and positive."
        )

    half_gauge = (
        gauge_length_m / 2.0
    )
    tolerance = (
        1.0e-6 * ds
    )

    valid = (
        (s - half_gauge >= s[0] - tolerance)
        & (s + half_gauge <= s[-1] + tolerance)
    )

    centre_indices = np.flatnonzero(
        valid
    ).astype(
        np.int64
    )

    if centre_indices.size == 0:
        raise ValueError(
            "No valid gauge centres remain for curvature QC."
        )

    s_centre = s[
        centre_indices
    ]
    s_left = (
        s_centre
        - half_gauge
    )
    s_right = (
        s_centre
        + half_gauge
    )

    tx_left = np.interp(
        s_left,
        s,
        tx,
    )
    tz_left = np.interp(
        s_left,
        s,
        tz,
    )
    tx_right = np.interp(
        s_right,
        s,
        tx,
    )
    tz_right = np.interp(
        s_right,
        s,
        tz,
    )

    norm_left = np.hypot(
        tx_left,
        tz_left,
    )
    norm_right = np.hypot(
        tx_right,
        tz_right,
    )

    if (
        np.any(
            norm_left <= 1.0e-12
        )
        or np.any(
            norm_right <= 1.0e-12
        )
    ):
        raise ValueError(
            "Interpolated gauge-end tangent is degenerate."
        )

    tx_left /= norm_left
    tz_left /= norm_left
    tx_right /= norm_right
    tz_right /= norm_right

    dot = np.clip(
        tx_left * tx_right
        + tz_left * tz_right,
        -1.0,
        1.0,
    )

    rotation_deg = np.degrees(
        np.arccos(
            dot
        )
    )

    result = {
        "channel_indices": centre_indices,
        "rotation_deg": rotation_deg,
        "median_deg": float(
            np.median(
                rotation_deg
            )
        ),
        "p95_deg": float(
            np.percentile(
                rotation_deg,
                95.0,
            )
        ),
        "max_deg": float(
            np.max(
                rotation_deg
            )
        ),
    }

    print("\nGauge-curvature QC")
    print("------------------")
    print(
        "valid gauge centres : "
        f"{centre_indices.size}"
    )
    print(
        "median rotation     : "
        f"{result['median_deg']:.6f} deg"
    )
    print(
        "95th percentile     : "
        f"{result['p95_deg']:.6f} deg"
    )
    print(
        "maximum rotation    : "
        f"{result['max_deg']:.6f} deg"
    )

    return result


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Keep receiver cable points inside the solver-valid domain.

    Returns the retained x/z arrays and the Boolean mask into the original
    registered geometry so channel identity is preserved exactly.

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

    kept_indices = np.flatnonzero(
        keep
    )

    if (
        kept_indices.size > 1
        and not np.all(
            np.diff(
                kept_indices
            )
            == 1
        )
    ):
        raise ValueError(
            "Solver-domain trimming produced a non-contiguous receiver block. "
            "A finite-gauge DAS operator must not bridge gaps in the cable."
        )

    print(
        "kept input rows       : "
        f"{kept_indices[0]} to {kept_indices[-1]}"
    )

    return (
        x_cable[keep],
        z_cable[keep],
        keep,
    )


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

    style_figure_text_bold(
        fig,
        main_axis=ax,
    )

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

    style_figure_text_bold(
        fig,
        main_axis=ax,
    )

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
    plot_x_min_m: float,
    plot_x_max_m: float,
    plot_z_max_m: float,
    fps: int = 6,
    max_frames: int = 80,
    percentile_clip: float = 99.5,
) -> None:
    """
    Make a Vz wavefield GIF over the scientific model domain only.

    The solver snapshots contain the complete computational grid, including
    side and bottom absorbing sponge cells. Those cells are intentionally
    excluded from both the displayed image and the colour-scale calculation.
    Boundary-performance QC belongs in a separate validation figure.
    """
    snapshots_vz = np.asarray(snapshots_vz, dtype=np.float64)
    snapshot_times = np.asarray(snapshot_times, dtype=np.float64)

    if snapshots_vz.ndim != 3:
        raise ValueError(
            f"snapshots_vz must be 3D, got shape {snapshots_vz.shape}."
        )

    if snapshots_vz.shape[0] == snapshot_times.size:
        nframes_total = snapshots_vz.shape[0]
    elif snapshots_vz.shape[-1] == snapshot_times.size:
        nframes_total = snapshots_vz.shape[-1]
    else:
        raise ValueError(
            "snapshot_times length does not match first or last snapshot axis: "
            f"snapshots_vz.shape={snapshots_vz.shape}, "
            f"snapshot_times.size={snapshot_times.size}."
        )

    if nframes_total < 1:
        raise ValueError("No snapshots available for GIF.")

    if not (
        np.isfinite(plot_x_min_m)
        and np.isfinite(plot_x_max_m)
        and np.isfinite(plot_z_max_m)
    ):
        raise ValueError("GIF plot limits must be finite.")

    if plot_x_min_m >= plot_x_max_m:
        raise ValueError(
            f"Invalid GIF x limits: {plot_x_min_m} >= {plot_x_max_m}."
        )

    x_grid = np.asarray(grid.x, dtype=np.float64)
    z_grid = np.asarray(grid.z, dtype=np.float64)

    x_keep = (
        (x_grid >= float(plot_x_min_m) - 0.5 * grid.dx)
        & (x_grid <= float(plot_x_max_m) + 0.5 * grid.dx)
    )
    z_keep = (
        (z_grid >= float(z_grid[0]) - 0.5 * grid.dz)
        & (z_grid <= float(plot_z_max_m) + 0.5 * grid.dz)
    )

    ix = np.flatnonzero(x_keep)
    iz = np.flatnonzero(z_keep)

    if ix.size < 2 or iz.size < 2:
        raise ValueError(
            "Scientific GIF crop contains too few grid points: "
            f"nx={ix.size}, nz={iz.size}."
        )

    def crop_frame(frame: np.ndarray) -> np.ndarray:
        return frame[np.ix_(iz, ix)]

    if nframes_total > max_frames:
        frame_ids = np.linspace(
            0,
            nframes_total - 1,
            max_frames,
        ).astype(np.int64)
    else:
        frame_ids = np.arange(
            nframes_total,
            dtype=np.int64,
        )

    # Scale from the scientific domain only. Hidden sponge amplitudes must not
    # determine the displayed dynamic range.
    sample_vals = []
    for iframe in frame_ids:
        frame = _snapshot_frame_2d(
            snapshots_vz,
            int(iframe),
            nx=grid.nx,
            nz=grid.nz,
        )
        sample_vals.append(
            crop_frame(frame).ravel()
        )

    sample_vals = np.concatenate(sample_vals)
    vmax = float(
        np.percentile(
            np.abs(sample_vals),
            percentile_clip,
        )
    )

    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0

    extent = [
        float(x_grid[ix[0]]),
        float(x_grid[ix[-1]]),
        float(z_grid[iz[-1]]),
        float(z_grid[iz[0]]),
    ]

    x_cable = np.asarray(x_cable, dtype=np.float64)
    z_cable = np.asarray(z_cable, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.5, 9.0))
    fig.subplots_adjust(
        left=0.12,
        right=0.86,
        top=0.98,
        bottom=0.08,
    )

    frame0 = crop_frame(
        _snapshot_frame_2d(
            snapshots_vz,
            int(frame_ids[0]),
            nx=grid.nx,
            nz=grid.nz,
        )
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

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )
    cbar.set_label(
        "Vz [m/s]",
        fontsize=12,
        fontweight="bold",
        labelpad=8,
    )
    cbar.ax.tick_params(
        labelsize=10,
        width=1.1,
        length=4,
    )
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.plot(
        x_cable,
        z_cable,
        color="gray",
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
        fontweight="bold",
        bbox=dict(
            facecolor="white",
            alpha=0.75,
            edgecolor="none",
        ),
    )

    ax.set_xlim(
        float(plot_x_min_m),
        float(plot_x_max_m),
    )
    ax.set_ylim(
        float(plot_z_max_m),
        float(z_grid[0]),
    )
    ax.set_xlabel(
        "Projected 2D section coordinate X [m]",
        fontsize=13,
        fontweight="bold",
        labelpad=8,
    )
    ax.set_ylabel(
        "Depth [m]",
        fontsize=13,
        fontweight="bold",
        labelpad=8,
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=11,
        width=1.2,
        length=5,
    )
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    ax.legend(
        loc="lower left",
        fontsize=LEGEND_FONTSIZE,
    )

    style_figure_text_bold(
        fig,
        main_axis=ax,
    )

    def update(k: int):
        iframe = int(frame_ids[k])

        frame = crop_frame(
            _snapshot_frame_2d(
                snapshots_vz,
                iframe,
                nx=grid.nx,
                nz=grid.nz,
            )
        )

        im.set_data(frame)
        time_text.set_text(
            f"t = {snapshot_times[iframe]:.3f} s"
        )
        return im, time_text

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_ids),
        interval=1000.0 / fps,
        blit=False,
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
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

    parser.add_argument(
        "--initial-model",
        choices=AVAILABLE_INITIAL_MODELS,
        default="bill_logs",
        help=(
            "SAFOD initial velocity model. Available: "
            + ", ".join(AVAILABLE_INITIAL_MODELS)
            + ". Default: bill_logs."
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
        "--nt",
        type=int,
        default=12000,
        help=(
            "Number of solver time steps. "
            "Default: 12000."
        ),
    )

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
        out_dir = (
            forward_dir_for_theta(args.theta_deg)
            / args.initial_model
        )
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
    print(f"initial model : {args.initial_model}")
    print(f"save GIF      : {args.save_gif}")
    print(f"output dir    : {out_dir}")

    # --------------------------------------------------------------------------
    # Numerical settings
    # --------------------------------------------------------------------------
    dx = 5.0
    dz = 5.0

    # The exact duration is printed after grid construction because dt is
    # selected from the CFL condition.

    nt = int(args.nt)

    if nt < 2:
        raise ValueError(
            f"--nt must be >= 2; got {nt}."
        )

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

    # --------------------------------------------------------------------------
    # Scientific and computational domains
    # --------------------------------------------------------------------------
    # Geological/interpreted model ends at 5 km.
    # The bottom absorbing sponge is appended outside this domain.
    scientific_z_max_m = 5000.0

    sponge_width_x_m = n_boundary * dx
    sponge_width_z_m = n_boundary * dz

    computational_z_max_m = (
        scientific_z_max_m
        + sponge_width_z_m
    )

    # Preserve the existing undamped lateral margin.
    baseline_undamped_side_margin_m = (
        baseline_x_padding_m
        - baseline_n_boundary * dx
    )

    extra_scientific_x_margin_m = 500.0

    # Includes the side sponge outside the scientific domain.
    x_padding_m = (
        baseline_undamped_side_margin_m
        + extra_scientific_x_margin_m
        + sponge_width_x_m
    )

    print(f"side sponge width          : {sponge_width_x_m:.1f} m")
    print(f"bottom sponge width        : {sponge_width_z_m:.1f} m")
    print(
        "undamped side margin      : "
        f"{baseline_undamped_side_margin_m + extra_scientific_x_margin_m:.1f} m"
    )
    print(f"extra scientific x margin  : {extra_scientific_x_margin_m:.1f} m")
    print(f"total x padding            : {x_padding_m:.1f} m")
    print(f"scientific model bottom    : {scientific_z_max_m:.1f} m")
    print(f"computational grid bottom  : {computational_z_max_m:.1f} m")

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


    bill_logs_csv = Path(
        "data/safod/velocity_models/"
        "ellsworth_malin_2011/"
        "fig3a_digitized.csv"
    )

    zhang_section_npz = Path(
        "data/safod/velocity_models/"
        "zhang_thurber_bedrosian_2009/"
        "processed/"
        "zhang2009_safod_section_2d.npz"
    )

    grid, model, x_cable_raw, z_cable_raw, metadata = (
        build_initial_model(
            model_name=args.initial_model,

            geom_file=geom_file,

            bill_logs_csv=bill_logs_csv,
            zhang_section_npz=zhang_section_npz,

            x_column=x_column,
            z_column=z_column,

            dx=dx,
            dz=dz,
            dt=None,
            nt=nt,
            half_order=half_order,
            cfl_safety=0.80,

            x_padding_m=x_padding_m,
            z_max_m=computational_z_max_m,
            z_padding_bottom_m=0.0,

            z_tie_m=None,
            anchor_fault_to_cable_end=True,
            fault_offset_from_cable_m=105.0,

            fault_dip_deg=82.0,
            fault_dip_sign=-1.0,

            left_block_name="salinian",
            right_block_name="franciscan",

            # Model-A parameters.
            # bill_logs disables/replaces the old lateral SAF structure
            # internally, so these remain relevant only to smooth_prior.
            initial_cross_fault_contrast=-0.08,
            initial_cross_fault_transition_m=350.0,
            initial_fault_zone_width_m=160.0,
            initial_fault_zone_velocity_reduction=0.14,

            include_pilot_hole_lvz_in_initial=True,
            initial_pilot_hole_lvz_strength=0.035,

            smooth_initial_sigma_m=80.0,
        )
    )

    expected_bottom_tolerance_m = 0.51 * dz

    if (
        abs(float(grid.z[-1]) - computational_z_max_m)
        > expected_bottom_tolerance_m
    ):
        raise RuntimeError(
            "Unexpected computational grid bottom: "
            f"grid.z[-1]={grid.z[-1]:.1f} m, "
            f"expected {computational_z_max_m:.1f} m. "
            "Check build_safod_model z_max_m/z_padding_bottom_m semantics."
        )

    # Scientific plotting/inversion domain. The side and bottom sponge remain
    # in the computational arrays but lie outside these limits.
    scientific_x_min_m = float(
        grid.x[0] + sponge_width_x_m
    )
    scientific_x_max_m = float(
        grid.x[-1] - sponge_width_x_m
    )

    if scientific_x_min_m >= scientific_x_max_m:
        raise RuntimeError(
            "Scientific x domain is empty after removing side sponge cells."
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
    print(
        "scientific x: "
        f"{scientific_x_min_m:.1f} to {scientific_x_max_m:.1f} m"
    )

    # --------------------------------------------------------------------------
    # 2. Trim cable and build DAS receivers
    # --------------------------------------------------------------------------
    x_cable_use, z_cable_use, cable_keep_mask = (
        trim_cable_for_solver_domain(
            grid=grid,
            x_cable=x_cable_raw,
            z_cable=z_cable_raw,
            n_boundary=n_boundary,
            half_order=half_order,
            free_surface=free_surface,
        )
    )

    if source_mode == "catalog_event":
        registered_geometry = load_registered_channel_geometry(
            path=geom_file,
            x_column=x_column,
            z_column=z_column,
            expected_raw_channels=event_cfg["real_raw_channels"],
        )

        registered_x = np.asarray(
            registered_geometry["x"],
            dtype=np.float64,
        )
        registered_z = np.asarray(
            registered_geometry["z"],
            dtype=np.float64,
        )

        if (
            registered_x.shape
            != np.asarray(
                x_cable_raw
            ).shape
            or registered_z.shape
            != np.asarray(
                z_cable_raw
            ).shape
        ):
            raise ValueError(
                "build_safod_model returned a cable with a different row count "
                "from the registered geometry CSV."
            )

        builder_mismatch = np.hypot(
            np.asarray(
                x_cable_raw,
                dtype=np.float64,
            )
            - registered_x,
            np.asarray(
                z_cable_raw,
                dtype=np.float64,
            )
            - registered_z,
        )

        builder_mismatch_max_m = float(
            np.max(
                builder_mismatch
            )
        )

        if builder_mismatch_max_m > 1.0e-6:
            raise ValueError(
                "build_safod_model cable coordinates do not reproduce the "
                "registered geometry CSV exactly. "
                f"Maximum mismatch={builder_mismatch_max_m:.6e} m."
            )

        receiver_raw_channels = np.asarray(
            registered_geometry[
                "raw_channels"
            ][
                cable_keep_mask
            ],
            dtype=np.float64,
        )

        receiver_geometry_row_indices = np.asarray(
            registered_geometry[
                "row_indices"
            ][
                cable_keep_mask
            ],
            dtype=np.int64,
        )

        # The DAS operator is parameterised by the interrogator's physical
        # uniform channel spacing.  x/z remain the exact registered channel
        # centres; only the cable-coordinate origin is reset after trimming.
        receiver_s = (
            np.arange(
                x_cable_use.size,
                dtype=np.float64,
            )
            * channel_spacing_m
        )

        receivers = build_receivers_from_channel_centres(
            x=x_cable_use,
            z=z_cable_use,
            s=receiver_s,
            grid=grid,
            n_pml=0,
        )

        centre_mismatch = np.hypot(
            receivers.x
            - x_cable_use,
            receivers.z
            - z_cable_use,
        )

        receiver_centre_mismatch_max_m = float(
            np.max(
                centre_mismatch
            )
        )

        if receiver_centre_mismatch_max_m > 1.0e-12:
            raise RuntimeError(
                "Exact-channel-centre constructor moved one or more registered "
                "receivers. "
                f"Maximum mismatch={receiver_centre_mismatch_max_m:.6e} m."
            )

        geometry_channel_column = str(
            registered_geometry[
                "channel_column"
            ]
        )
        receiver_geometry_mode = (
            "exact_registered_channel_centres"
        )

    else:
        # Controlled synthetic geometry remains generated from a continuous
        # waypoint polyline.
        receivers = build_das_cable(
            grid=grid,
            waypoints_x=x_cable_use.tolist(),
            waypoints_z=z_cable_use.tolist(),
            channel_spacing_m=channel_spacing_m,
            n_pml=0,
        )

        receiver_raw_channels = np.arange(
            receivers.nrec,
            dtype=np.float64,
        )

        receiver_geometry_row_indices = np.arange(
            receivers.nrec,
            dtype=np.int64,
        )

        receiver_centre_mismatch_max_m = np.nan
        builder_mismatch_max_m = np.nan
        geometry_channel_column = ""
        receiver_geometry_mode = (
            "uniform_waypoint_resampling"
        )

    if not np.isclose(
        receivers.channel_spacing,
        channel_spacing_m,
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise RuntimeError(
            "Constructed receiver spacing does not match the selected "
            "acquisition spacing: "
            f"receivers={receivers.channel_spacing:.12f} m, "
            f"requested={channel_spacing_m:.12f} m."
        )

    gauge_curvature_qc = compute_gauge_curvature_qc(
        receivers=receivers,
        gauge_length_m=gauge_length_m,
    )

    print("\nReceivers")
    print("---------")
    print(
        f"geometry mode       : "
        f"{receiver_geometry_mode}"
    )
    print(
        f"receivers           : "
        f"{receivers.nrec} channel centres"
    )
    print(
        f"receiver spacing    : "
        f"{receivers.channel_spacing:.6f} m"
    )
    print(
        f"gauge / spacing     : "
        f"{gauge_length_m / receivers.channel_spacing:.6f}"
    )
    print(
        f"cable s             : "
        f"{receivers.s[0]:.3f} to {receivers.s[-1]:.3f} m"
    )
    print(
        f"receiver x          : "
        f"{receivers.x.min():.3f} to {receivers.x.max():.3f} m"
    )
    print(
        f"receiver z          : "
        f"{receivers.z.min():.3f} to {receivers.z.max():.3f} m"
    )
    print(
        f"raw-channel range   : "
        f"{receiver_raw_channels.min():.1f} to "
        f"{receiver_raw_channels.max():.1f}"
    )
    print(
        "centre mismatch max : "
        f"{receiver_centre_mismatch_max_m:.6e} m"
    )

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
        z_src_target_m = 4500.0

        z_src = float(
            np.clip(
                z_src_target_m,
                grid.z[0] + (half_order + 10) * grid.dz,
                scientific_z_max_m - (half_order + 10) * grid.dz,
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

    if not (
        scientific_x_min_m <= x_src <= scientific_x_max_m
        and float(grid.z[0]) <= z_src <= scientific_z_max_m
    ):
        raise ValueError(
            "Source lies outside the scientific model domain: "
            f"source=({x_src:.1f}, {z_src:.1f}) m, "
            f"x=[{scientific_x_min_m:.1f}, {scientific_x_max_m:.1f}] m, "
            f"z=[{grid.z[0]:.1f}, {scientific_z_max_m:.1f}] m. "
            "Sources must not be placed inside the absorbing sponge."
        )

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
        source_time_mode="ricker_moment",
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

    # Show only the scientific domain. The numerical side and bottom sponge
    # remain in the solver arrays but are not presented as geology.
    ax.set_xlim(
        scientific_x_min_m,
        scientific_x_max_m,
    )
    ax.set_ylim(
        scientific_z_max_m,
        float(grid.z[0]),
    )

    style_figure_text_bold(
        fig,
        main_axis=ax,
    )

    fig.savefig(
        out_dir / "01_model_vp_with_source.png",
        dpi=220,
        bbox_inches="tight",
    )
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
            plot_x_min_m=scientific_x_min_m,
            plot_x_max_m=scientific_x_max_m,
            plot_z_max_m=scientific_z_max_m,
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

    if not np.array_equal(
        das_result.channel_indices,
        gauge_curvature_qc["channel_indices"],
    ):
        raise RuntimeError(
            "Gauge-curvature QC and DAS operator selected different gauge "
            "centres. Their physical gauge-validity logic must remain identical."
        )

    das_raw_channels = receiver_raw_channels[
        das_result.channel_indices
    ]

    das_geometry_row_indices = receiver_geometry_row_indices[
        das_result.channel_indices
    ]

    print(
        "DAS raw channels: "
        f"{das_raw_channels.min():.1f} to "
        f"{das_raw_channels.max():.1f}"
    )

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
        receiver_raw_channels=receiver_raw_channels,
        das_raw_channels=das_raw_channels,
        receiver_geometry_row_indices=receiver_geometry_row_indices,
        das_geometry_row_indices=das_geometry_row_indices,
        receiver_channel_spacing_m=np.array(
            receivers.channel_spacing
        ),
        receiver_geometry_mode=np.array(
            receiver_geometry_mode
        ),
        receiver_s_definition=np.array(
            "uniform acquisition coordinate from event-specific dCh"
            if source_mode == "catalog_event"
            else "uniform synthetic waypoint-resampled coordinate"
        ),
        geometry_channel_column=np.array(
            geometry_channel_column
        ),
        receiver_centre_mismatch_max_m=np.array(
            receiver_centre_mismatch_max_m
        ),
        builder_geometry_mismatch_max_m=np.array(
            builder_mismatch_max_m
        ),
        gauge_tangent_rotation_deg=(
            gauge_curvature_qc[
                "rotation_deg"
            ]
        ),
        gauge_tangent_rotation_median_deg=np.array(
            gauge_curvature_qc[
                "median_deg"
            ]
        ),
        gauge_tangent_rotation_p95_deg=np.array(
            gauge_curvature_qc[
                "p95_deg"
            ]
        ),
        gauge_tangent_rotation_max_deg=np.array(
            gauge_curvature_qc[
                "max_deg"
            ]
        ),

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
        source_time_mode=np.array("ricker_moment"),

        run_tag=np.array(run_tag),
        initial_model_name=np.array(
            args.initial_model
        ),
        n_boundary=np.array(n_boundary),
        gamma_s=np.array(gamma_s),
        free_surface=np.array(free_surface),
        extra_scientific_x_margin_m=np.array(
            extra_scientific_x_margin_m
        ),
        x_padding_m=np.array(x_padding_m),
        scientific_x_min_m=np.array(scientific_x_min_m),
        scientific_x_max_m=np.array(scientific_x_max_m),
        scientific_z_max_m=np.array(scientific_z_max_m),
        computational_z_max_m=np.array(computational_z_max_m),
        sponge_width_x_m=np.array(sponge_width_x_m),
        sponge_width_z_m=np.array(sponge_width_z_m),

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