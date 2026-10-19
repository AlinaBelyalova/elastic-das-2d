#!/usr/bin/env python3
"""
Focused Oct-03-2026 SAFOD comparison around the CURRENT Sintela turnaround.

This version fixes two issues discovered in the diagnostics:

1) Current Sintela channel numbers are not the old OptaSense/Excel channel
   numbers.  The Oct-03 gather shows the turnaround near current ch ~1278-1280,
   so this script works directly with CURRENT Sintela channels and does not use
   SAFOD_Phase2_GeoReferenced_Channels.xlsx for channel indexing.

2) Ettore's notebook used the rounded reference time 04:52:31.000Z, while the
   exact NCEDC origin is 04:52:31.320Z.  Therefore a feature plotted at
   0.8-1.0 s in Ettore's notebook corresponds to ~0.48-0.68 s after the exact
   catalog origin.  The secondary-phase shading is corrected accordingly.

Processing:
- DAS: Sintela protobuf, exactly as Ettore reads it (diff=False).
- Common filter: DASutils.bandpass2D_c(..., zerophase=False).
- Timing plot: current MH030 GP1 RAW COUNTS vs DAS strain rate.  No response
  metadata are needed for this plot.
- Physical plots: DAS strain vs MH030 GP1 velocity and DAS strain rate vs
  MH030 GP1 acceleration.  MH030 response metadata are tried first; if they
  are not yet available, historical MH029 Pod-1 metadata are used provisionally.
- No relative time shift is ever applied.
"""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path
import sys

import dateutil.parser
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException
from scipy.integrate import cumulative_trapezoid
from scipy.signal import detrend


# =============================================================================
# DASutils
# =============================================================================

DAS_UTILITIES_ROOT = Path(
    "/home/groups/ettore88/alina/packages/DAS-utilities"
)
DAS_UTILITIES_BUILD = DAS_UTILITIES_ROOT / "build"
DAS_UTILITIES_PYTHON = DAS_UTILITIES_ROOT / "python"

existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = (
    f"{existing_ld}:{DAS_UTILITIES_BUILD}"
    if existing_ld
    else str(DAS_UTILITIES_BUILD)
)

sys.path.insert(0, str(DAS_UTILITIES_BUILD))
sys.path.insert(0, str(DAS_UTILITIES_PYTHON))

import DASutils  # noqa: E402


# =============================================================================
# Configuration
# =============================================================================

EVENT_ID_NUMERIC = 75443276
FALLBACK_ORIGIN = UTCDateTime("2026-10-03T04:52:31.320000Z")

DAS_FILES = [
    Path(
        "/oak/stanford/groups/ettore88/data/SAFOD/SAFOD_events/"
        "timeseriesScaler_2026-10-03_04.51.40_UTC_073863.pb"
    ),
    Path(
        "/oak/stanford/groups/ettore88/data/SAFOD/SAFOD_events/"
        "timeseriesScaler_2026-10-03_04.52.40_UTC_073864.pb"
    ),
]

NETWORK = "SF"
CURRENT_STATION = "MH030"
METADATA_FALLBACK_STATION = "MH029"
LOCATION = "01"
GP1 = "GP1"

# Current Sintela turnaround from the Oct-03 gather.
DEFAULT_CHANNEL = 1279
TURNAROUND_CHANNELS = (1275, 1276, 1277, 1278, 1279, 1280, 1281, 1282)

BANDS_HZ = (
    (5.0, 40.0),
    (5.0, 200.0),
)

# Exact-origin times.
#
# Ettore notebook reference: 04:52:31.000
# Exact NCEDC origin:       04:52:31.320
#
# Thus 0.8-1.0 s in the notebook -> 0.48-0.68 s after exact origin.
SECONDARY_TMIN_S = 0.48
SECONDARY_TMAX_S = 0.68

TIMING_TMIN_S = 0.10
TIMING_TMAX_S = 0.35

PHYSICAL_TMIN_S = 0.10
PHYSICAL_TMAX_S = 0.90

RESPONSE_MARGIN_S = 20.0
DPI = 300

DEFAULT_OUTPUT_DIR = Path(
    "results/events/20261003_75443276/compare/"
    "current_turnaround_comparison"
)


# =============================================================================
# Helpers
# =============================================================================

def parse_beg_time(info) -> UTCDateTime:
    value = info["begTime"]

    if isinstance(value, datetime.datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return UTCDateTime(dt)

    dt = dateutil.parser.parse(str(value))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)

    return UTCDateTime(dt)


def get_origin(client: Client) -> UTCDateTime:
    try:
        catalog = client.get_events(
            eventid=EVENT_ID_NUMERIC
        )

        event = catalog[0]
        origin_obj = (
            event.preferred_origin()
            or event.origins[0]
        )

        return origin_obj.time

    except Exception as exc:
        print(
            f"WARNING: could not retrieve exact origin ({exc}); "
            f"using {FALLBACK_ORIGIN}"
        )
        return FALLBACK_ORIGIN


def common_causal_filter(
    x: np.ndarray,
    fs_hz: float,
    fmin_hz: float,
    fmax_hz: float,
) -> np.ndarray:
    """
    Same causal/non-zero-phase DASutils filter for DAS and geophone.
    DASutils expects channel x time.
    """
    y = np.asarray(
        x,
        dtype=np.float64,
    )

    was_1d = (
        y.ndim == 1
    )

    if was_1d:
        y = y[
            None,
            :
        ]

    y = y.copy()

    DASutils.bandpass2D_c(
        y,
        fs_hz,
        fmin_hz,
        fmax_hz,
        zerophase=False,
    )

    if was_1d:
        return y[0]

    return y


def rms_normalize(
    x: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    y = np.asarray(
        x,
        dtype=np.float64,
    )

    scale = float(
        np.sqrt(
            np.mean(
                y[mask] ** 2
            )
        )
    )

    if (
        not np.isfinite(scale)
        or scale <= 0.0
    ):
        return y.copy()

    return (
        y
        / scale
    )


def zero_lag_corr(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    x = np.asarray(
        a,
        dtype=np.float64,
    )
    y = np.asarray(
        b,
        dtype=np.float64,
    )

    x = x - np.mean(x)
    y = y - np.mean(y)

    denom = (
        np.linalg.norm(x)
        * np.linalg.norm(y)
    )

    if denom <= 0.0:
        return np.nan

    return float(
        np.dot(x, y)
        / denom
    )


# =============================================================================
# DAS
# =============================================================================

def load_das(
    origin: UTCDateTime,
) -> dict:

    das_data, info = DASutils.readFile_protobuf(
        [
            str(path)
            for path in DAS_FILES
        ],
        0.01,
        500.0,
        verbose=1,
        diff=False,
        detrend=False,
        tapering=False,
        filter=False,
        median=False,
        desampling=False,
        nChbuffer=3000,
        system="Sintela",
    )

    das_data = np.asarray(
        das_data,
        dtype=np.float64,
    )

    fs_hz = float(
        info["fs"]
    )

    beg_time = parse_beg_time(
        info
    )

    time_s = (
        float(
            beg_time
            - origin
        )
        + np.arange(
            das_data.shape[1],
            dtype=np.float64,
        )
        / fs_hz
    )

    return {
        "data": das_data,
        "fs_hz": fs_hz,
        "dx_m": float(
            info["dx"]
        ),
        "time_s": time_s,
        "beg_time": beg_time,
    }


def prepare_das_channel(
    *,
    raw_data: np.ndarray,
    channel: int,
    fs_hz: float,
    fmin_hz: float,
    fmax_hz: float,
) -> dict:

    rate = np.asarray(
        raw_data[
            channel,
            :
        ],
        dtype=np.float64,
    )

    rate = detrend(
        rate,
        type="linear",
    )

    rate = (
        rate
        - np.mean(rate)
    )

    rate_filtered = common_causal_filter(
        rate,
        fs_hz,
        fmin_hz,
        fmax_hz,
    )

    # Integrate the prepared raw strain-rate trace first, then apply
    # the same final causal bandpass.
    strain = cumulative_trapezoid(
        rate,
        dx=1.0 / fs_hz,
        initial=0.0,
    )

    strain = detrend(
        strain,
        type="linear",
    )

    strain = (
        strain
        - np.mean(strain)
    )

    strain_filtered = common_causal_filter(
        strain,
        fs_hz,
        fmin_hz,
        fmax_hz,
    )

    return {
        "strain_rate": (
            rate_filtered
        ),
        "strain": (
            strain_filtered
        ),
    }


# =============================================================================
# MH030 GP1
# =============================================================================

def load_gp1(
    *,
    client: Client,
    origin: UTCDateTime,
    das_time_s: np.ndarray,
    das_fs_hz: float,
    max_target_fmax_hz: float,
) -> dict:

    absolute_start = (
        origin
        + float(
            das_time_s[0]
        )
        - RESPONSE_MARGIN_S
    )

    absolute_end = (
        origin
        + float(
            das_time_s[-1]
        )
        + RESPONSE_MARGIN_S
    )

    waveform_seed = (
        f"{NETWORK}."
        f"{CURRENT_STATION}."
        f"{LOCATION}."
        f"{GP1}"
    )

    stream = client.get_waveforms(
        network=NETWORK,
        station=CURRENT_STATION,
        location=LOCATION,
        channel=GP1,
        starttime=absolute_start,
        endtime=absolute_end,
        attach_response=False,
    )

    stream.merge(
        method=1,
        fill_value="interpolate",
    )

    tr_raw = stream[0].copy()

    raw_counts = detrend(
        np.asarray(
            tr_raw.data,
            dtype=np.float64,
        ),
        type="linear",
    )

    raw_counts = (
        raw_counts
        - np.mean(raw_counts)
    )

    native_fs_hz = float(
        tr_raw.stats.sampling_rate
    )

    geo_time_s = (
        float(
            tr_raw.stats.starttime
            - origin
        )
        + np.arange(
            tr_raw.stats.npts,
            dtype=np.float64,
        )
        / native_fs_hz
    )

    raw_counts_on_das_grid = np.interp(
        das_time_s,
        geo_time_s,
        raw_counts,
    )

    # ------------------------------------------------------------------
    # Provisional response metadata.
    # ------------------------------------------------------------------

    metadata_station = (
        CURRENT_STATION
    )

    metadata_fallback = False

    try:
        inventory = client.get_stations(
            network=NETWORK,
            station=CURRENT_STATION,
            location=LOCATION,
            channel=GP1,
            starttime=absolute_start,
            endtime=absolute_end,
            level="response",
        )

    except FDSNNoDataException:

        metadata_station = (
            METADATA_FALLBACK_STATION
        )

        metadata_fallback = True

        inventory = client.get_stations(
            network=NETWORK,
            station=metadata_station,
            location=LOCATION,
            channel=GP1,
            starttime=origin,
            endtime=origin + 1.0,
            level="response",
        )

    metadata_seed = (
        f"{NETWORK}."
        f"{metadata_station}."
        f"{LOCATION}."
        f"{GP1}"
    )

    tr_velocity = tr_raw.copy()

    if metadata_fallback:
        tr_velocity.stats.station = (
            metadata_station
        )

    # Wide response-removal prefilter.  Final science band is applied below
    # with the same DASutils causal filter as DAS.
    nyquist = (
        0.5
        * native_fs_hz
    )

    high1 = min(
        1.15
        * max_target_fmax_hz,
        0.82
        * nyquist,
    )

    high2 = min(
        1.30
        * max_target_fmax_hz,
        0.92
        * nyquist,
    )

    pre_filt = (
        0.5,
        1.0,
        high1,
        high2,
    )

    tr_velocity.data = (
        raw_counts.copy()
    )

    tr_velocity.remove_response(
        inventory=inventory,
        output="VEL",
        pre_filt=pre_filt,
        water_level=60.0,
        taper=True,
        zero_mean=False,
    )

    velocity_native = np.asarray(
        tr_velocity.data,
        dtype=np.float64,
    )

    velocity_on_das_grid = np.interp(
        das_time_s,
        geo_time_s,
        velocity_native,
    )

    return {
        "raw_counts": (
            raw_counts_on_das_grid
        ),
        "velocity_unfiltered": (
            velocity_on_das_grid
        ),
        "metadata_seed": (
            metadata_seed
        ),
        "metadata_fallback": (
            metadata_fallback
        ),
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_timing_stack(
    *,
    time_s: np.ndarray,
    gp1_raw_filtered: np.ndarray,
    das_by_channel: dict,
    output_path: Path,
    band_label: str,
) -> None:

    mask = (
        (time_s >= TIMING_TMIN_S)
        & (time_s <= TIMING_TMAX_S)
    )

    channels = list(
        das_by_channel.keys()
    )

    n_traces = (
        1
        + len(channels)
    )

    offsets = np.arange(
        n_traces,
        dtype=np.float64,
    )[::-1]

    fig, ax = plt.subplots(
        figsize=(12.5, 7.0)
    )

    gp = rms_normalize(
        gp1_raw_filtered,
        mask,
    )

    ax.plot(
        time_s[mask],
        0.32
        * gp[mask]
        + offsets[0],
        color="#940025",
        linewidth=1.6,
    )

    labels = [
        "MH030 GP1 raw counts"
    ]

    for i, channel in enumerate(
        channels,
        start=1,
    ):

        trace = rms_normalize(
            das_by_channel[
                channel
            ],
            mask,
        )

        ax.plot(
            time_s[mask],
            0.32
            * trace[mask]
            + offsets[i],
            color="black",
            linewidth=1.0,
        )

        labels.append(
            f"DAS ch {channel}"
        )

    ax.set_yticks(
        offsets
    )

    ax.set_yticklabels(
        labels,
        fontweight="bold",
    )

    ax.set_xlim(
        TIMING_TMIN_S,
        TIMING_TMAX_S,
    )

    ax.set_xlabel(
        "Time from exact NCEDC origin [s]",
        fontweight="bold",
    )

    ax.set_title(
        f"MH030 vs current Sintela turnaround | {band_label}",
        fontweight="bold",
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def plot_physical_overlay(
    *,
    time_s: np.ndarray,
    das_trace: np.ndarray,
    geo_trace: np.ndarray,
    das_label: str,
    geo_label: str,
    title: str,
    output_path: Path,
) -> tuple[float, float]:

    mask = (
        (time_s >= PHYSICAL_TMIN_S)
        & (time_s <= PHYSICAL_TMAX_S)
    )

    secondary_mask = (
        (time_s >= SECONDARY_TMIN_S)
        & (time_s <= SECONDARY_TMAX_S)
    )

    das_n = rms_normalize(
        das_trace,
        mask,
    )

    geo_n = rms_normalize(
        geo_trace,
        mask,
    )

    # Global sign convention only.
    corr_native = zero_lag_corr(
        das_n[mask],
        geo_n[mask],
    )

    sign = (
        -1.0
        if (
            np.isfinite(
                corr_native
            )
            and corr_native < 0.0
        )
        else 1.0
    )

    geo_n = (
        sign
        * geo_n
    )

    corr_full = zero_lag_corr(
        das_n[mask],
        geo_n[mask],
    )

    corr_secondary = zero_lag_corr(
        das_n[
            secondary_mask
        ],
        geo_n[
            secondary_mask
        ],
    )

    fig, ax = plt.subplots(
        figsize=(11.5, 4.2)
    )

    ax.axvspan(
        SECONDARY_TMIN_S,
        SECONDARY_TMAX_S,
        color="0.92",
        zorder=0,
    )

    ax.plot(
        time_s[mask],
        das_n[mask],
        color="black",
        linewidth=1.6,
        label=das_label,
    )

    ax.plot(
        time_s[mask],
        geo_n[mask],
        color="#940025",
        linewidth=1.5,
        label=geo_label,
    )

    ax.axhline(
        0.0,
        color="0.82",
        linewidth=0.7,
    )

    ax.set_xlim(
        PHYSICAL_TMIN_S,
        PHYSICAL_TMAX_S,
    )

    ax.set_xlabel(
        "Time from exact NCEDC origin [s]",
        fontweight="bold",
    )

    ax.set_ylabel(
        "Normalized amplitude",
        fontweight="bold",
    )

    ax.set_title(
        title,
        fontweight="bold",
    )

    ax.legend(
        frameon=False,
        loc="upper right",
    )

    ax.text(
        0.01,
        0.03,
        "gray: 0.48-0.68 s exact-origin window "
        "(= 0.80-1.00 s in Ettore's rounded-origin notebook)",
        transform=ax.transAxes,
        fontsize=8.5,
        color="0.35",
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return (
        corr_full,
        corr_secondary,
    )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--channel",
        type=int,
        default=DEFAULT_CHANNEL,
        help=(
            "Current Sintela channel used for "
            "the two physical overlays."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = Client(
        "https://service.ncedc.org"
    )

    origin = get_origin(
        client
    )

    das = load_das(
        origin
    )

    print()
    print(
        "CURRENT TURNAROUND COMPARISON"
    )
    print(
        "============================="
    )
    print(
        f"exact origin               : "
        f"{origin}"
    )
    print(
        f"DAS shape                  : "
        f"{das['data'].shape}"
    )
    print(
        f"DAS fs                     : "
        f"{das['fs_hz']:.1f} Hz"
    )
    print(
        f"DAS dx                     : "
        f"{das['dx_m']:.4f} m"
    )
    print(
        f"current comparison channel : "
        f"{args.channel}"
    )
    print(
        f"turnaround channels shown  : "
        f"{TURNAROUND_CHANNELS}"
    )
    print(
        "secondary phase window     : "
        f"{SECONDARY_TMIN_S:.2f}-"
        f"{SECONDARY_TMAX_S:.2f} s "
        "after exact origin"
    )

    gp1 = load_gp1(
        client=client,
        origin=origin,
        das_time_s=(
            das[
                "time_s"
            ]
        ),
        das_fs_hz=(
            das[
                "fs_hz"
            ]
        ),
        max_target_fmax_hz=max(
            fmax
            for _, fmax in BANDS_HZ
        ),
    )

    print(
        f"GP1 metadata source        : "
        f"{gp1['metadata_seed']}"
    )
    print(
        f"GP1 metadata fallback      : "
        f"{gp1['metadata_fallback']}"
    )

    for (
        fmin_hz,
        fmax_hz,
    ) in BANDS_HZ:

        band_label = (
            f"{fmin_hz:g}-"
            f"{fmax_hz:g} Hz"
        )

        band_tag = (
            f"{int(fmin_hz)}_"
            f"{int(fmax_hz)}Hz"
        )

        gp1_raw_filtered = common_causal_filter(
            gp1[
                "raw_counts"
            ],
            das[
                "fs_hz"
            ],
            fmin_hz,
            fmax_hz,
        )

        gp1_velocity = common_causal_filter(
            gp1[
                "velocity_unfiltered"
            ],
            das[
                "fs_hz"
            ],
            fmin_hz,
            fmax_hz,
        )

        gp1_acceleration = np.gradient(
            gp1_velocity,
            1.0
            / das[
                "fs_hz"
            ],
        )

        das_rate_by_channel = {}

        for channel in TURNAROUND_CHANNELS:

            prepared = prepare_das_channel(
                raw_data=das[
                    "data"
                ],
                channel=channel,
                fs_hz=das[
                    "fs_hz"
                ],
                fmin_hz=fmin_hz,
                fmax_hz=fmax_hz,
            )

            das_rate_by_channel[
                channel
            ] = prepared[
                "strain_rate"
            ]

        plot_timing_stack(
            time_s=das[
                "time_s"
            ],
            gp1_raw_filtered=(
                gp1_raw_filtered
            ),
            das_by_channel=(
                das_rate_by_channel
            ),
            output_path=(
                args.output_dir
                / (
                    "01_timing_turnaround_"
                    f"{band_tag}.png"
                )
            ),
            band_label=band_label,
        )

        selected = prepare_das_channel(
            raw_data=das[
                "data"
            ],
            channel=args.channel,
            fs_hz=das[
                "fs_hz"
            ],
            fmin_hz=fmin_hz,
            fmax_hz=fmax_hz,
        )

        corr_sv = plot_physical_overlay(
            time_s=das[
                "time_s"
            ],
            das_trace=selected[
                "strain"
            ],
            geo_trace=gp1_velocity,
            das_label=(
                f"DAS current ch "
                f"{args.channel}: strain"
            ),
            geo_label=(
                "MH030 GP1: particle velocity"
            ),
            title=(
                "DAS strain vs MH030 GP1 velocity | "
                f"{band_label}"
            ),
            output_path=(
                args.output_dir
                / (
                    "02_strain_vs_velocity_"
                    f"{band_tag}.png"
                )
            ),
        )

        corr_ra = plot_physical_overlay(
            time_s=das[
                "time_s"
            ],
            das_trace=selected[
                "strain_rate"
            ],
            geo_trace=gp1_acceleration,
            das_label=(
                f"DAS current ch "
                f"{args.channel}: strain rate"
            ),
            geo_label=(
                "MH030 GP1: particle acceleration"
            ),
            title=(
                "DAS strain rate vs MH030 GP1 acceleration | "
                f"{band_label}"
            ),
            output_path=(
                args.output_dir
                / (
                    "03_strainrate_vs_acceleration_"
                    f"{band_tag}.png"
                )
            ),
        )

        print()
        print(
            f"--- {band_label} ---"
        )
        print(
            "filter                     : "
            "DASutils.bandpass2D_c, zerophase=False"
        )
        print(
            "relative time shift        : NONE"
        )
        print(
            f"corr(strain, velocity)     : "
            f"full={corr_sv[0]:+.4f}, "
            f"secondary={corr_sv[1]:+.4f}"
        )
        print(
            f"corr(rate, acceleration)   : "
            f"full={corr_ra[0]:+.4f}, "
            f"secondary={corr_ra[1]:+.4f}"
        )

    print()
    print(
        f"Saved to: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
