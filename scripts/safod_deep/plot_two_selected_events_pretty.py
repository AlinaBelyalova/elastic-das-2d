# ==============================================================================
# scripts/safod_deep/plot_two_selected_events_pretty.py
#
# Publication-style QC plots for two selected SAFOD deep-DAS earthquakes:
#
#   75343317 -> display to 7.5 s
#   75371066 -> display to 3.5 s
#
# Both figures use the same canvas size and the same 5-30 Hz causal filter.
# The horizontal coordinate is the physical reference channel from GEO_XLSX.
# ==============================================================================

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.safod_deep.plot_catalog_events import (
    DEFAULT_CATALOG,
    DEFAULT_MANIFEST,
    GEO_XLSX,
    DisplayRecipe,
    coverage_summary,
    decimate_for_display,
    filter_event_window,
    infer_reference_locus_offset,
    load_catalog,
    load_manifest,
    load_reference_channel_limits,
    parse_beg_time_from_info,
    physical_reference_axis_and_crop,
    read_event_files,
    robust_clip,
    select_event_files,
)


# ==============================================================================
# SELECTED EVENTS AND DISPLAY WINDOWS
# ==============================================================================

TARGET_EVENT_IDS = (
    "75343317",
    "75371066",
)

DISPLAY_TMAX_BY_EVENT = {
    "75343317": 7.0,
    "75371066": 2.5,
}

DISPLAY_TMIN_S = -0.5

FMIN_HZ = 5.0
FMAX_HZ = 30.0
FILTER_PAD_S = 3.0
PCLIP = 96.0

FIGSIZE = (18.0, 8.0)
DEFAULT_DPI = 220


# ==============================================================================
# PLOTTING
# ==============================================================================

def plot_event_pretty(
    *,
    data: np.ndarray,
    time_s: np.ndarray,
    x_axis: np.ndarray,
    event_row: pd.Series,
    out_path: Path,
    pclip: float,
    dpi: int,
) -> float:
    """Save one clean, title-free DAS event figure."""
    clip_value = robust_clip(
        data,
        pclip,
    )

    event_id = str(
        event_row["event_id"]
    )
    magnitude = float(
        event_row["magnitude"]
    )
    magnitude_type = str(
        event_row.get(
            "magnitude_type",
            "",
        )
    ).strip()
    depth_km = float(
        event_row["depth_km"]
    )
    origin_time = pd.Timestamp(
        event_row["origin_time_utc"]
    )

    fig, ax = plt.subplots(
        figsize=FIGSIZE,
        constrained_layout=True,
    )

    image = ax.imshow(
        data.T,
        extent=[
            float(x_axis[0]),
            float(x_axis[-1] + 1.0),
            float(time_s[-1]),
            float(time_s[0]),
        ],
        aspect="auto",
        cmap="seismic",
        vmin=-clip_value,
        vmax=clip_value,
        interpolation="none",
    )

    ax.axhline(
        0.0,
        color="black",
        linewidth=1.4,
        linestyle="--",
        label="Catalog origin",
    )

    # No title by design.
    ax.set_xlabel(
        "Channel",
        fontsize=20,
        fontweight="bold",
        labelpad=10,
    )
    ax.set_ylabel(
        "Time from origin [s]",
        fontsize=20,
        fontweight="bold",
        labelpad=10,
    )

    ax.set_xlim(
        float(x_axis[0]),
        float(x_axis[-1] + 1.0),
    )
    ax.set_ylim(
        float(time_s[-1]),
        float(time_s[0]),
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=16,
        width=1.4,
        length=6,
    )

    for label in ax.get_xticklabels():
        label.set_fontweight("bold")

    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_linewidth(1.3)

    ax.legend(
        loc="upper right",
        fontsize=12,
        framealpha=0.95,
        edgecolor="0.3",
    )

    # Keep only the first three requested annotation lines.
    magnitude_label = (
        f"M{magnitude:.1f} {magnitude_type}"
        if magnitude_type
        else f"M{magnitude:.1f}"
    )

    annotation = (
        f"Event {event_id}   {magnitude_label}\n"
        f"Origin: {origin_time.isoformat()}\n"
        f"Depth: {depth_km:.2f} km"
    )

    ax.text(
        0.012,
        0.985,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        linespacing=1.25,
        # bbox={
        #     "facecolor": "white",
        #     "alpha": 0.90,
        #     "edgecolor": "none",
        #     "boxstyle": "round,pad=0.35",
        # },

        bbox={
            "facecolor": "none",
            "edgecolor": "none",
            "boxstyle": "round,pad=0.35",
        },
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        pad=0.015,
    )
    colorbar.set_label(
        "Strain rate [nm/m/s]",
        fontsize=17,
        fontweight="bold",
        labelpad=10,
    )
    colorbar.ax.tick_params(
        labelsize=14,
        width=1.2,
        length=5,
    )

    for label in colorbar.ax.get_yticklabels():
        label.set_fontweight("bold")

    fig.savefig(
        out_path,
        dpi=int(dpi),
        bbox_inches="tight",
    )
    plt.close(fig)

    return clip_value


# ==============================================================================
# EVENT PROCESSING
# ==============================================================================

def process_one_event(
    *,
    event_row: pd.Series,
    manifest: pd.DataFrame,
    output_dir: Path,
    reference_locus_offset: float,
    physical_channel_min: float,
    physical_channel_max: float,
    zero_phase: bool,
    dpi: int,
) -> Path:
    """Read, filter, register, crop, and plot one selected event."""
    event_id = str(
        event_row["event_id"]
    )

    if event_id not in DISPLAY_TMAX_BY_EVENT:
        raise ValueError(
            f"No display window configured for event {event_id}."
        )

    display_tmax_s = float(
        DISPLAY_TMAX_BY_EVENT[event_id]
    )

    recipe = DisplayRecipe(
        name=f"pretty_5_30Hz_tmax_{display_tmax_s:g}s",
        fmin_hz=FMIN_HZ,
        fmax_hz=FMAX_HZ,
        display_tmin_s=DISPLAY_TMIN_S,
        display_tmax_s=display_tmax_s,
        filter_pad_s=FILTER_PAD_S,
        pclip=PCLIP,
    )

    origin = pd.Timestamp(
        event_row["origin_time_utc"]
    )

    filter_tmin_s = (
        recipe.display_tmin_s
        - recipe.filter_pad_s
    )
    filter_tmax_s = (
        recipe.display_tmax_s
        + recipe.filter_pad_s
    )

    read_start_utc = (
        origin
        + pd.to_timedelta(
            filter_tmin_s,
            unit="s",
        )
    )
    read_end_utc = (
        origin
        + pd.to_timedelta(
            filter_tmax_s,
            unit="s",
        )
    )

    selected = select_event_files(
        manifest=manifest,
        event_row=event_row,
        read_start_utc=read_start_utc,
        read_end_utc=read_end_utc,
    )

    coverage_ok, coverage_message = coverage_summary(
        selected,
        required_start=read_start_utc,
        required_end=read_end_utc,
    )

    if not coverage_ok:
        raise RuntimeError(
            "Requested padded window is not continuously covered: "
            f"{coverage_message}"
        )

    files = (
        selected["file_path"]
        .astype(str)
        .tolist()
    )

    data_full, info = read_event_files(
        files
    )

    fs_hz = float(
        info["fs"]
    )

    if (
        not np.isfinite(fs_hz)
        or fs_hz <= 0.0
    ):
        raise ValueError(
            f"Invalid sampling rate: {fs_hz}"
        )

    beg_time = parse_beg_time_from_info(
        info
    )
    origin_datetime = origin.to_pydatetime()

    relative_start_s = (
        beg_time
        - origin_datetime
    ).total_seconds()

    time_full_s = (
        relative_start_s
        + np.arange(
            data_full.shape[1],
            dtype=np.float64,
        )
        / fs_hz
    )

    filter_mask = (
        (time_full_s >= filter_tmin_s)
        & (time_full_s <= filter_tmax_s)
    )

    if np.count_nonzero(filter_mask) < 16:
        raise RuntimeError(
            "Too few samples remain in the padded filter window."
        )

    data_filter = np.ascontiguousarray(
        data_full[:, filter_mask]
    )
    time_filter_s = time_full_s[
        filter_mask
    ]

    del data_full

    (
        data_display,
        time_display_s,
        effective_fmax_hz,
    ) = filter_event_window(
        data=data_filter,
        time_s=time_filter_s,
        fs_hz=fs_hz,
        recipe=recipe,
        zero_phase=zero_phase,
    )

    del data_filter

    (
        data_display,
        time_display_s,
        _,
    ) = decimate_for_display(
        data=data_display,
        time_s=time_display_s,
        fs_hz=fs_hz,
        fmax_hz=effective_fmax_hz,
    )

    (
        x_axis,
        physical_row_mask,
        _,
        _,
    ) = physical_reference_axis_and_crop(
        selected_manifest=selected,
        n_channels=data_display.shape[0],
        reference_locus_offset=reference_locus_offset,
        physical_channel_min=physical_channel_min,
        physical_channel_max=physical_channel_max,
    )

    data_display = data_display[
        physical_row_mask,
        :,
    ]

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path = (
        output_dir
        / (
            f"event_{event_id}_pretty_"
            f"5_30Hz_tmax_{display_tmax_s:g}s.png"
        )
    )

    clip_value = plot_event_pretty(
        data=data_display,
        time_s=time_display_s,
        x_axis=x_axis,
        event_row=event_row,
        out_path=out_path,
        pclip=recipe.pclip,
        dpi=dpi,
    )

    print(
        f"Saved {event_id}: {out_path}"
    )
    print(
        f"  window : {recipe.display_tmin_s:g} to "
        f"{recipe.display_tmax_s:g} s"
    )
    print(
        f"  band   : {recipe.fmin_hz:g}-"
        f"{effective_fmax_hz:g} Hz"
    )
    print(
        f"  clip   : ±{clip_value:.3g} nm/m/s"
    )

    return out_path


# ==============================================================================
# CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create two clean publication-style SAFOD deep-DAS event plots."
        )
    )

    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/safod_deep/event_qc_pretty_two"
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
    )
    parser.add_argument(
        "--zero-phase",
        action="store_true",
        help=(
            "Use zerophase=True. Default is the causal filter used in "
            "Ettore's QC example."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    catalog = load_catalog(
        args.catalog
    )
    manifest = load_manifest(
        args.manifest
    )

    reference_locus_offset = infer_reference_locus_offset(
        manifest
    )
    (
        physical_channel_min,
        physical_channel_max,
    ) = load_reference_channel_limits(
        GEO_XLSX
    )

    selected_catalog = catalog[
        catalog["event_id"]
        .astype(str)
        .isin(TARGET_EVENT_IDS)
    ].copy()

    found_ids = set(
        selected_catalog["event_id"]
        .astype(str)
    )
    missing_ids = sorted(
        set(TARGET_EVENT_IDS)
        .difference(found_ids)
    )

    if missing_ids:
        raise ValueError(
            f"Events not found in catalog: {missing_ids}"
        )

    # Keep the requested order rather than chronological order.
    selected_catalog["_event_order"] = (
        selected_catalog["event_id"]
        .astype(str)
        .map(
            {
                event_id: index
                for index, event_id
                in enumerate(TARGET_EVENT_IDS)
            }
        )
    )
    selected_catalog = (
        selected_catalog
        .sort_values("_event_order")
        .reset_index(drop=True)
    )

    for _, event_row in selected_catalog.iterrows():
        process_one_event(
            event_row=event_row,
            manifest=manifest,
            output_dir=args.output_dir,
            reference_locus_offset=reference_locus_offset,
            physical_channel_min=physical_channel_min,
            physical_channel_max=physical_channel_max,
            zero_phase=bool(args.zero_phase),
            dpi=int(args.dpi),
        )


if __name__ == "__main__":
    main()