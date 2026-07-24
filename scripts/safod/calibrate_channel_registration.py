#!/usr/bin/env python3
"""
Register a new SAFOD acquisition to the unchanged full fibre geometry.

The reference Excel workbook contains the complete physical template:

    Surface Spool -> Down-leg -> turnaround -> Up-leg -> surface

The interrogator may change channel spacing, gauge length, channel numbering,
and lead-fibre registration without changing the borehole trajectory.  This
script therefore does not copy the old channel numbers.  It:

1. reads the full reference Excel geometry;
2. finds the turnaround in the new DAS recording from mirrored down/up
   waveform coherence;
3. uses the new acquisition channel spacing and the known measured depth of
   the borehole leg to map new data rows onto the old physical trajectory;
4. saves both full dual-pass and down-leg-only geometry CSV files.

The HDF5 data and the reference workbook are never modified.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, hilbert, sosfiltfilt


DEFAULT_UTILITIES_ROOT = Path(
    "/home/groups/ettore88/alina/packages/DAS-utilities"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Register a new SAFOD DAS acquisition to the unchanged full "
            "surface/down/up fibre geometry."
        )
    )

    parser.add_argument(
        "--h5",
        type=Path,
        required=True,
        help="Converted DAS HDF5 file.",
    )

    parser.add_argument(
        "--origin-time",
        required=True,
        help="Catalogue origin time in UTC.",
    )

    parser.add_argument(
        "--reference-xlsx",
        type=Path,
        required=True,
        help="Full SAFOD georeferenced channel workbook.",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for registration QC and mapped geometry.",
    )

    parser.add_argument(
        "--utilities-root",
        type=Path,
        default=DEFAULT_UTILITIES_ROOT,
        help="DAS-utilities checkout root.",
    )

    parser.add_argument(
        "--fmin",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--fmax",
        type=float,
        default=25.0,
    )

    parser.add_argument(
        "--read-start",
        type=float,
        default=-0.30,
    )

    parser.add_argument(
        "--read-end",
        type=float,
        default=1.80,
    )

    parser.add_argument(
        "--match-start",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--match-end",
        type=float,
        default=1.55,
    )

    parser.add_argument(
        "--target-fs",
        type=float,
        default=250.0,
    )

    parser.add_argument(
        "--offset-step",
        type=int,
        default=3,
        help="Use every Nth mirrored offset while scoring turnaround centres.",
    )

    parser.add_argument(
        "--minimum-offset",
        type=int,
        default=20,
        help="Ignore this many rows nearest the turnaround.",
    )

    parser.add_argument(
        "--minimum-model-tvd",
        type=float,
        default=10.0,
        help=(
            "Minimum TVD retained in the down-leg-only model geometry. "
            "This removes the surface/spool transition."
        ),
    )

    return parser.parse_args()


def add_utilities_to_path(root: Path) -> None:
    root = root.expanduser().resolve()
    build = root / "build"
    python_dir = root / "python"

    for path in (build, python_dir):
        if not path.exists():
            raise FileNotFoundError(path)
        sys.path.insert(0, str(path))

    old_ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = (
        f"{build}:{old_ld}"
        if old_ld
        else str(build)
    )


def read_h5_metadata(
    path: Path,
    utilities_root: Path,
) -> dict:
    add_utilities_to_path(utilities_root)

    import DASutils  # type: ignore

    with h5py.File(path, "r") as handle:
        (
            data_kind,
            system,
            fs_hz,
            nt,
            n_channels,
            channel_spacing_m,
            _,
        ) = DASutils.parse_sampling_das_hdf5(handle)

        if "Data" not in handle:
            raise KeyError("Expected a root /Data dataset.")

        data = handle["Data"]

        start_time = data.attrs.get("startTime")

        if start_time is None:
            raise KeyError("/Data is missing startTime.")

        if isinstance(start_time, bytes):
            start_time = start_time.decode("utf-8")

        acquisition = handle.get("Acquisition_origin")

        start_channel = 0
        channel_step = 1
        gauge_length_m = float(
            data.attrs.get("GaugeLength", np.nan)
        )

        if acquisition is not None:
            start_channel = int(
                acquisition.attrs.get(
                    "acquisition.start_channel",
                    acquisition.attrs.get(
                        "packet.common_header.start_channel",
                        0,
                    ),
                )
            )

            channel_step = int(
                acquisition.attrs.get(
                    "packet.channel_step",
                    1,
                )
            )

            gauge_length_m = float(
                acquisition.attrs.get(
                    "acquisition.gauge_length",
                    gauge_length_m,
                )
            )

        shape = tuple(int(value) for value in data.shape)

    return {
        "data_kind": str(data_kind),
        "system": str(system),
        "fs_hz": float(fs_hz),
        "nt": int(nt),
        "n_channels": int(n_channels),
        "channel_spacing_m": float(channel_spacing_m),
        "gauge_length_m": gauge_length_m,
        "start_time": str(start_time),
        "start_channel": start_channel,
        "channel_step": channel_step,
        "data_shape": shape,
    }


def read_reference_geometry(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full = pd.read_excel(path)

    required = {
        "Channel",
        "Section",
        "MD_m",
        "TVD_m",
        "UTM_E_m",
        "UTM_N_m",
    }

    missing = sorted(required.difference(full.columns))

    if missing:
        raise KeyError(
            f"Reference workbook is missing columns: {missing}"
        )

    full = (
        full.sort_values("Channel")
        .drop_duplicates("Channel")
        .reset_index(drop=True)
    )

    down = (
        full.loc[full["Section"] == "Down-leg"]
        .sort_values("MD_m")
        .reset_index(drop=True)
    )

    up = (
        full.loc[full["Section"] == "Up-leg"]
        .sort_values("MD_m")
        .reset_index(drop=True)
    )

    if down.empty or up.empty:
        raise RuntimeError(
            "Reference workbook must contain both Down-leg and Up-leg sections."
        )

    down_md = pd.to_numeric(
        down["MD_m"],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    up_md = pd.to_numeric(
        up["MD_m"],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    if np.any(np.diff(down_md) <= 0.0):
        raise ValueError("Down-leg MD must increase strictly.")

    if np.any(np.diff(up_md) < 0.0):
        raise ValueError("Sorted Up-leg MD must not decrease.")

    return full, down, up


def read_event_window(
    *,
    h5_path: Path,
    metadata: dict,
    origin_time: str,
    t_start_s: float,
    t_end_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    fs_hz = float(metadata["fs_hz"])
    n_channels = int(metadata["n_channels"])

    file_start = pd.Timestamp(metadata["start_time"])

    if file_start.tzinfo is None:
        file_start = file_start.tz_localize("UTC")
    else:
        file_start = file_start.tz_convert("UTC")

    origin = pd.Timestamp(origin_time)

    if origin.tzinfo is None:
        origin = origin.tz_localize("UTC")
    else:
        origin = origin.tz_convert("UTC")

    origin_sample = (
        (origin - file_start).total_seconds()
        * fs_hz
    )

    i0 = int(
        math.floor(
            origin_sample + t_start_s * fs_hz
        )
    )
    i1 = int(
        math.ceil(
            origin_sample + t_end_s * fs_hz
        )
    )

    with h5py.File(h5_path, "r") as handle:
        dataset = handle["Data"]

        if dataset.shape[0] == n_channels:
            available_nt = dataset.shape[1]

            if i0 < 0 or i1 > available_nt:
                raise ValueError(
                    f"Requested samples {i0}:{i1}; available 0:{available_nt}."
                )

            data = np.asarray(
                dataset[:, i0:i1],
                dtype=np.float64,
            )

        elif dataset.shape[1] == n_channels:
            available_nt = dataset.shape[0]

            if i0 < 0 or i1 > available_nt:
                raise ValueError(
                    f"Requested samples {i0}:{i1}; available 0:{available_nt}."
                )

            data = np.asarray(
                dataset[i0:i1, :],
                dtype=np.float64,
            ).T

        else:
            raise ValueError(
                f"Cannot identify channel axis in /Data shape {dataset.shape}."
            )

    time_s = (
        np.arange(i0, i1, dtype=np.float64)
        - origin_sample
    ) / fs_hz

    return data, time_s


def filter_and_decimate(
    data: np.ndarray,
    time_s: np.ndarray,
    *,
    fs_hz: float,
    fmin_hz: float,
    fmax_hz: float,
    target_fs_hz: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    data = np.asarray(data, dtype=np.float64)
    data = data - np.mean(
        data,
        axis=1,
        keepdims=True,
    )

    sos = butter(
        4,
        [fmin_hz, fmax_hz],
        btype="bandpass",
        fs=fs_hz,
        output="sos",
    )

    filtered = sosfiltfilt(
        sos,
        data,
        axis=1,
    )

    factor = max(
        1,
        int(
            math.floor(
                fs_hz / target_fs_hz
            )
        ),
    )

    return (
        filtered[:, ::factor],
        time_s[::factor],
        fs_hz / factor,
    )


def normalize_rows(data: np.ndarray) -> np.ndarray:
    centred = data - np.mean(
        data,
        axis=1,
        keepdims=True,
    )

    norm = np.linalg.norm(
        centred,
        axis=1,
        keepdims=True,
    )

    return centred / np.maximum(
        norm,
        1.0e-12,
    )


def score_turnaround_centres(
    *,
    signed_features: np.ndarray,
    envelope_features: np.ndarray,
    snr: np.ndarray,
    expected_leg_rows: float,
    minimum_offset: int,
    offset_step: int,
) -> pd.DataFrame:
    n_channels = signed_features.shape[0]
    nominal = int(round(expected_leg_rows))

    required_margin = int(
        math.floor(
            0.86 * nominal
        )
    )

    centre2_min = 2 * required_margin
    centre2_max = 2 * (
        n_channels - 1 - required_margin
    )

    if centre2_max <= centre2_min:
        raise RuntimeError(
            "Not enough channels to contain both borehole legs."
        )

    max_offset = int(
        math.floor(
            0.94 * nominal
        )
    )

    rows: list[dict] = []

    for centre2 in range(
        centre2_min,
        centre2_max + 1,
    ):
        left_centre = centre2 // 2
        right_centre = (centre2 + 1) // 2

        maximum_available = min(
            left_centre,
            n_channels - 1 - right_centre,
            max_offset,
        )

        if maximum_available <= minimum_offset:
            continue

        offsets = np.arange(
            minimum_offset,
            maximum_available + 1,
            offset_step,
            dtype=np.int64,
        )

        left = left_centre - offsets
        right = right_centre + offsets

        pair_snr = np.minimum(
            snr[left],
            snr[right],
        )

        finite = np.isfinite(pair_snr)

        if np.count_nonzero(finite) < 60:
            continue

        threshold = max(
            1.5,
            float(
                np.nanpercentile(
                    pair_snr[finite],
                    35.0,
                )
            ),
        )

        valid = finite & (
            pair_snr >= threshold
        )

        if np.count_nonzero(valid) < 60:
            valid = finite

        left = left[valid]
        right = right[valid]
        pair_snr = pair_snr[valid]

        signed_corr = np.einsum(
            "ij,ij->i",
            signed_features[left],
            signed_features[right],
        )

        envelope_corr = np.einsum(
            "ij,ij->i",
            envelope_features[left],
            envelope_features[right],
        )

        abs_signed = np.abs(
            signed_corr
        )

        combined = (
            0.65 * abs_signed
            + 0.35 * np.maximum(
                envelope_corr,
                0.0,
            )
        )

        weights = np.log1p(
            np.maximum(
                pair_snr,
                0.0,
            )
        )

        weights = weights / max(
            float(np.nanmedian(weights)),
            1.0e-12,
        )

        score = (
            0.70 * float(
                np.nanmedian(combined)
            )
            + 0.30 * float(
                np.average(
                    combined,
                    weights=weights,
                )
            )
        )

        rows.append(
            {
                "centre_row": 0.5 * centre2,
                "score": score,
                "n_pairs": int(left.size),
                "median_abs_signed_corr": float(
                    np.nanmedian(abs_signed)
                ),
                "median_envelope_corr": float(
                    np.nanmedian(envelope_corr)
                ),
                "median_signed_corr": float(
                    np.nanmedian(signed_corr)
                ),
                "median_pair_snr": float(
                    np.nanmedian(pair_snr)
                ),
            }
        )

    table = pd.DataFrame(rows)

    if table.empty:
        raise RuntimeError(
            "No valid turnaround candidates were scored."
        )

    return table


def interpolate_section(
    reference: pd.DataFrame,
    target_md_m: np.ndarray,
) -> pd.DataFrame:
    md = pd.to_numeric(
        reference["MD_m"],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    output = pd.DataFrame(
        index=np.arange(target_md_m.size)
    )

    for column in reference.columns:
        if column == "Section":
            continue

        values = pd.to_numeric(
            reference[column],
            errors="coerce",
        ).to_numpy(dtype=np.float64)

        if not np.all(np.isfinite(values)):
            continue

        output_column = (
            "ReferenceChannel"
            if column == "Channel"
            else column
        )

        output[output_column] = np.interp(
            target_md_m,
            md,
            values,
        )

    return output


def build_dual_pass_mapping(
    *,
    down_reference: pd.DataFrame,
    up_reference: pd.DataFrame,
    n_channels: int,
    channel_spacing_m: float,
    turn_row: float,
    acquisition_start_channel: int,
    acquisition_channel_step: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bottom_md_m = float(
        pd.to_numeric(
            down_reference["MD_m"],
            errors="raise",
        ).max()
    )

    rows = np.arange(
        n_channels,
        dtype=np.int64,
    )

    signed_distance_from_turn_m = (
        rows.astype(np.float64)
        - turn_row
    ) * channel_spacing_m

    md_m = (
        bottom_md_m
        - np.abs(
            signed_distance_from_turn_m
        )
    )

    in_borehole = (
        (md_m >= 0.0)
        & (md_m <= bottom_md_m)
    )

    rows = rows[in_borehole]
    md_m = md_m[in_borehole]
    signed_distance_from_turn_m = (
        signed_distance_from_turn_m[
            in_borehole
        ]
    )

    down_mask = (
        signed_distance_from_turn_m <= 0.0
    )
    up_mask = ~down_mask

    blocks = []

    if np.any(down_mask):
        down_interp = interpolate_section(
            down_reference,
            md_m[down_mask],
        )

        down_interp.insert(
            0,
            "Section",
            "Down-leg",
        )

        down_interp.insert(
            0,
            "MD_target_m",
            md_m[down_mask],
        )

        down_interp.insert(
            0,
            "DataRow",
            rows[down_mask],
        )

        blocks.append(
            down_interp
        )

    if np.any(up_mask):
        up_interp = interpolate_section(
            up_reference,
            md_m[up_mask],
        )

        up_interp.insert(
            0,
            "Section",
            "Up-leg",
        )

        up_interp.insert(
            0,
            "MD_target_m",
            md_m[up_mask],
        )

        up_interp.insert(
            0,
            "DataRow",
            rows[up_mask],
        )

        blocks.append(
            up_interp
        )

    mapped = (
        pd.concat(
            blocks,
            ignore_index=True,
        )
        .sort_values("DataRow")
        .reset_index(drop=True)
    )

    mapped.insert(
        1,
        "Channel",
        mapped["DataRow"].astype(
            np.float64
        ),
    )

    mapped.insert(
        2,
        "AcquisitionChannel",
        (
            acquisition_start_channel
            + acquisition_channel_step
            * mapped["DataRow"].to_numpy(
                dtype=np.int64
            )
        ),
    )

    mapped.insert(
        3,
        "distance_from_turnaround_m",
        (
            mapped["DataRow"].to_numpy(
                dtype=np.float64
            )
            - turn_row
        ) * channel_spacing_m,
    )

    mapped["registration_turn_row"] = (
        turn_row
    )

    mapped["registration_channel_spacing_m"] = (
        channel_spacing_m
    )

    down_only = (
        mapped.loc[
            mapped["Section"] == "Down-leg"
        ]
        .copy()
        .reset_index(drop=True)
    )

    return mapped, down_only


def make_qc_figure(
    *,
    output: Path,
    filtered: np.ndarray,
    time_s: np.ndarray,
    scores: pd.DataFrame,
    turn_row: float,
    first_down_row: float,
    last_up_row: float,
) -> None:
    display = filtered / np.maximum(
        np.max(
            np.abs(filtered),
            axis=1,
            keepdims=True,
        ),
        1.0e-12,
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 10),
        gridspec_kw={
            "height_ratios": [2.4, 1.0],
        },
    )

    ax = axes[0]

    ax.imshow(
        display.T,
        aspect="auto",
        cmap="seismic",
        vmin=-1.0,
        vmax=1.0,
        extent=[
            -0.5,
            display.shape[0] - 0.5,
            float(time_s[-1]),
            float(time_s[0]),
        ],
        interpolation="nearest",
    )

    ax.axvline(
        first_down_row,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=(
            f"Down-leg surface crossing ≈ {first_down_row:.1f}"
        ),
    )

    ax.axvline(
        turn_row,
        color="magenta",
        linewidth=2.0,
        label=(
            f"Turnaround ≈ {turn_row:.1f}"
        ),
    )

    ax.axvline(
        last_up_row,
        color="black",
        linestyle=":",
        linewidth=1.5,
        label=(
            f"Up-leg surface crossing ≈ {last_up_row:.1f}"
        ),
    )

    ax.set_title(
        "June DAS gather with inferred unchanged-cable registration"
    )

    ax.set_xlabel(
        "HDF5 data row"
    )

    ax.set_ylabel(
        "Time from catalogue origin [s]"
    )

    ax.legend(
        loc="lower right",
    )

    score_ax = axes[1]

    score_ax.plot(
        scores["centre_row"],
        scores["score"],
        linewidth=1.2,
    )

    score_ax.axvline(
        turn_row,
        color="magenta",
        linewidth=1.8,
    )

    score_ax.set_xlabel(
        "Candidate turnaround data row"
    )

    score_ax.set_ylabel(
        "Mirrored waveform coherence"
    )

    score_ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    args = parse_args()

    h5_path = args.h5.expanduser().resolve()
    xlsx_path = (
        args.reference_xlsx
        .expanduser()
        .resolve()
    )
    out_dir = (
        args.out_dir
        .expanduser()
        .resolve()
    )

    for path in (h5_path, xlsx_path):
        if not path.exists():
            raise FileNotFoundError(path)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = read_h5_metadata(
        h5_path,
        args.utilities_root,
    )

    full_reference, down_reference, up_reference = (
        read_reference_geometry(
            xlsx_path
        )
    )

    section_summary = (
        full_reference.groupby("Section")["Channel"]
        .agg(["min", "max", "count"])
    )

    print("Reference geometry sections")
    print("---------------------------")
    print(section_summary.to_string())

    bottom_md_m = float(
        down_reference["MD_m"].max()
    )

    dch_m = float(
        metadata["channel_spacing_m"]
    )

    expected_leg_rows = (
        bottom_md_m / dch_m
    )

    print()
    print("New acquisition")
    print("---------------")
    print(
        f"channels          : {metadata['n_channels']}"
    )
    print(
        f"channel spacing   : {dch_m:.9f} m"
    )
    print(
        f"gauge length      : {metadata['gauge_length_m']:.9f} m"
    )
    print(
        f"start channel     : {metadata['start_channel']}"
    )
    print(
        f"physical leg MD   : {bottom_md_m:.4f} m"
    )
    print(
        f"expected leg rows : {expected_leg_rows:.3f}"
    )

    raw, time_s = read_event_window(
        h5_path=h5_path,
        metadata=metadata,
        origin_time=args.origin_time,
        t_start_s=args.read_start,
        t_end_s=args.read_end,
    )

    filtered, filtered_time, effective_fs = (
        filter_and_decimate(
            raw,
            time_s,
            fs_hz=float(
                metadata["fs_hz"]
            ),
            fmin_hz=args.fmin,
            fmax_hz=args.fmax,
            target_fs_hz=args.target_fs,
        )
    )

    match_mask = (
        (filtered_time >= args.match_start)
        & (filtered_time <= args.match_end)
    )

    noise_mask = (
        (filtered_time >= args.read_start)
        & (filtered_time <= 0.0)
    )

    match = filtered[
        :,
        match_mask,
    ]

    noise = filtered[
        :,
        noise_mask,
    ]

    event_rms = np.sqrt(
        np.mean(
            match**2,
            axis=1,
        )
    )

    noise_rms = np.sqrt(
        np.mean(
            noise**2,
            axis=1,
        )
    )

    snr = event_rms / np.maximum(
        noise_rms,
        1.0e-12,
    )

    signed_features = normalize_rows(
        match
    )

    envelope = np.abs(
        hilbert(
            match,
            axis=1,
        )
    )

    envelope = gaussian_filter1d(
        envelope,
        sigma=max(
            1.0,
            0.012 * effective_fs,
        ),
        axis=1,
        mode="nearest",
    )

    envelope_features = normalize_rows(
        envelope
    )

    scores = score_turnaround_centres(
        signed_features=signed_features,
        envelope_features=envelope_features,
        snr=snr,
        expected_leg_rows=expected_leg_rows,
        minimum_offset=args.minimum_offset,
        offset_step=args.offset_step,
    )

    ranked = (
        scores.sort_values(
            "score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    best = ranked.iloc[0]
    turn_row = float(
        best["centre_row"]
    )

    first_down_row = (
        turn_row - expected_leg_rows
    )

    last_up_row = (
        turn_row + expected_leg_rows
    )

    mapped_full, mapped_down = build_dual_pass_mapping(
        down_reference=down_reference,
        up_reference=up_reference,
        n_channels=int(
            metadata["n_channels"]
        ),
        channel_spacing_m=dch_m,
        turn_row=turn_row,
        acquisition_start_channel=int(
            metadata["start_channel"]
        ),
        acquisition_channel_step=int(
            metadata["channel_step"]
        ),
    )

    mapped_down_model = (
        mapped_down.loc[
            pd.to_numeric(
                mapped_down["TVD_m"],
                errors="coerce",
            )
            >= args.minimum_model_tvd
        ]
        .copy()
        .reset_index(drop=True)
    )

    full_path = (
        out_dir
        / "mapped_full_dual_pass_geometry.csv"
    )

    down_path = (
        out_dir
        / "mapped_downleg_geometry.csv"
    )

    down_model_path = (
        out_dir
        / "mapped_downleg_model_geometry.csv"
    )

    score_path = (
        out_dir
        / "turnaround_score.csv"
    )

    figure_path = (
        out_dir
        / "channel_registration_qc.png"
    )

    json_path = (
        out_dir
        / "channel_registration.json"
    )

    mapped_full.to_csv(
        full_path,
        index=False,
    )

    mapped_down.to_csv(
        down_path,
        index=False,
    )

    mapped_down_model.to_csv(
        down_model_path,
        index=False,
    )

    scores.to_csv(
        score_path,
        index=False,
    )

    make_qc_figure(
        output=figure_path,
        filtered=filtered,
        time_s=filtered_time,
        scores=scores,
        turn_row=turn_row,
        first_down_row=first_down_row,
        last_up_row=last_up_row,
    )

    result = {
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "h5": str(h5_path),
        "reference_xlsx": str(xlsx_path),
        "origin_time": args.origin_time,
        "method": (
            "mirrored_down_up_waveform_coherence_with_full_reference_geometry"
        ),
        "new_channel_spacing_m": dch_m,
        "new_gauge_length_m": float(
            metadata["gauge_length_m"]
        ),
        "physical_bottom_md_m": bottom_md_m,
        "expected_leg_rows": expected_leg_rows,
        "turnaround_data_row": turn_row,
        "downleg_surface_row": first_down_row,
        "upleg_surface_row": last_up_row,
        "best_score": float(
            best["score"]
        ),
        "median_abs_signed_corr": float(
            best["median_abs_signed_corr"]
        ),
        "median_envelope_corr": float(
            best["median_envelope_corr"]
        ),
        "n_pairs": int(
            best["n_pairs"]
        ),
        "mapped_full_geometry": str(
            full_path
        ),
        "mapped_downleg_geometry": str(
            down_path
        ),
        "mapped_downleg_model_geometry": str(
            down_model_path
        ),
    }

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(
            result,
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")

    print()
    print("Inferred registration")
    print("---------------------")
    print(
        f"turnaround data row : {turn_row:.2f}"
    )
    print(
        f"down-leg surface row: {first_down_row:.2f}"
    )
    print(
        f"up-leg surface row  : {last_up_row:.2f}"
    )
    print(
        f"coherence score     : {float(best['score']):.4f}"
    )
    print(
        "median |corr|      : "
        f"{float(best['median_abs_signed_corr']):.4f}"
    )
    print(
        "median env corr    : "
        f"{float(best['median_envelope_corr']):.4f}"
    )
    print(
        f"mirrored pairs      : {int(best['n_pairs'])}"
    )
    print(
        "mapped down-leg rows: "
        f"{int(mapped_down['DataRow'].min())} to "
        f"{int(mapped_down['DataRow'].max())}"
    )
    print(
        "model down-leg rows : "
        f"{int(mapped_down_model['DataRow'].min())} to "
        f"{int(mapped_down_model['DataRow'].max())}"
    )

    print()
    print("Saved")
    print("-----")
    print(figure_path)
    print(json_path)
    print(score_path)
    print(full_path)
    print(down_path)
    print(down_model_path)

    if float(best["score"]) < 0.25:
        print()
        print(
            "WARNING: waveform coherence is weak. Inspect the QC figure before "
            "using the inferred mapping."
        )


if __name__ == "__main__":
    main()