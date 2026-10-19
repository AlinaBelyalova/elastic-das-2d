#!/usr/bin/env python3
"""
SAFOD Oct-03-2026 diagnostic:
compare MH030 GP1 with CURRENT Sintela DAS channels 1276-1281.

Main figure:
    6 panels (3x2), one DAS channel per panel.
    Gray  = DAS strain rate [um/m/s]           (left Y axis)
    Red   = MH030 GP1 ground acceleration [cm/s^2] (right Y axis)

Processing:
    DAS:
        readFile_protobuf(..., diff=False, system="Sintela")
        detrend + demean + 5% taper
        DASutils.bandpass2D_c(data, fmin, fmax, dt, zerophase=False)

    MH030:
        current waveform = SF.MH030.01.GP1
        response removal to VEL
        current MH030 StationXML tried first
        if unavailable, provisional SF.MH029.01.GP1 metadata fallback
        interpolate velocity to the exact DAS 500-Hz time grid
        same DASutils causal bandpass
        differentiate velocity -> acceleration [cm/s^2]

No relative time shift is applied to the plotted data.

For each channel, the script calculates a diagnostic best lag only in the
P-wave window.  The lag is NOT applied.

Lag convention:
    positive lag = MH030 would need to move LATER to best match DAS
    negative lag = MH030 would need to move EARLIER to best match DAS

Default P window:
    0.17-0.34 s after the exact NCEDC origin

Example:
    python scripts/safod/experiments/oct03_2026/diagnose_oct03_mh030_channels.py

Optional:
    python scripts/safod/experiments/oct03_2026/diagnose_oct03_mh030_channels.py \
        --fmin 5 --fmax 40 \
        --pmin 0.17 --pmax 0.34
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

from scipy.signal import detrend
from scipy.signal.windows import tukey


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

EVENT_ID = 75443276

FALLBACK_ORIGIN = UTCDateTime(
    "2026-10-03T04:52:31.320000Z"
)

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

DAS_CHANNELS = (
    1276,
    1277,
    1278,
    1279,
    1280,
    1281,
)

NETWORK = "SF"
CURRENT_STATION = "MH030"
METADATA_FALLBACK_STATION = "MH029"
LOCATION = "01"
GEOPHONE_CHANNEL = "GP1"

DEFAULT_FMIN_HZ = 5.0
DEFAULT_FMAX_HZ = 40.0

DEFAULT_TMIN_S = 0.10
DEFAULT_TMAX_S = 0.45

DEFAULT_PMIN_S = 0.17
DEFAULT_PMAX_S = 0.34

MAX_LAG_S = 0.10

EDGE_TAPER_FRACTION = 0.05
RESPONSE_MARGIN_S = 20.0

DAS_COLOR = "0.52"
GEO_COLOR = "#940025"

DAS_LINEWIDTH = 1.7
GEO_LINEWIDTH = 1.7

DPI = 300

DEFAULT_OUTPUT_DIR = Path(
    "results/events/20261003_75443276/compare/"
    "mh030_channel_diagnostic"
)


# =============================================================================
# Helpers
# =============================================================================

def parse_beg_time(info) -> UTCDateTime:
    value = info["begTime"]

    if isinstance(value, datetime.datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(
                tzinfo=datetime.timezone.utc
            )
        return UTCDateTime(dt)

    dt = dateutil.parser.parse(
        str(value)
    )

    if dt.tzinfo is None:
        dt = dt.replace(
            tzinfo=datetime.timezone.utc
        )

    return UTCDateTime(dt)


def get_exact_origin(
    client: Client,
) -> UTCDateTime:

    try:
        catalog = client.get_events(
            eventid=EVENT_ID
        )

        event = catalog[0]

        origin_obj = (
            event.preferred_origin()
            or event.origins[0]
        )

        return origin_obj.time

    except Exception as exc:

        print(
            "WARNING: exact NCEDC origin query failed: "
            f"{exc}"
        )

        print(
            f"Using fallback origin: "
            f"{FALLBACK_ORIGIN}"
        )

        return FALLBACK_ORIGIN


def detrend_demean(
    trace: np.ndarray,
) -> np.ndarray:

    x = detrend(
        np.asarray(
            trace,
            dtype=np.float64,
        ),
        type="linear",
    )

    return (
        x
        - np.mean(x)
    )


def five_percent_taper(
    trace: np.ndarray,
) -> np.ndarray:

    x = np.asarray(
        trace,
        dtype=np.float64,
    )

    window = tukey(
        x.size,
        alpha=(
            2.0
            * EDGE_TAPER_FRACTION
        ),
    )

    return (
        x
        * window
    )


def common_causal_bandpass(
    trace: np.ndarray,
    *,
    fs_hz: float,
    fmin_hz: float,
    fmax_hz: float,
) -> np.ndarray:
    """
    Exact DASutils calling convention used by Ettore:

        bandpass2D_c(data, fmin, fmax, dt, zerophase=False)
    """

    if not (
        0.0
        < fmin_hz
        < fmax_hz
        < 0.5 * fs_hz
    ):
        raise ValueError(
            f"Invalid band "
            f"{fmin_hz:g}-{fmax_hz:g} Hz "
            f"for fs={fs_hz:g} Hz."
        )

    x = np.asarray(
        trace,
        dtype=np.float64,
    )[None, :]

    filtered = DASutils.bandpass2D_c(
        x,
        fmin_hz,
        fmax_hz,
        1.0 / fs_hz,
        zerophase=False,
    )

    filtered = np.asarray(
        filtered,
        dtype=np.float64,
    )

    if (
        filtered.ndim != 2
        or filtered.shape[0] != 1
    ):
        raise RuntimeError(
            "Unexpected DASutils.bandpass2D_c "
            f"output shape: {filtered.shape}"
        )

    return filtered[0]


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

    x = (
        x
        - np.mean(x)
    )

    y = (
        y
        - np.mean(y)
    )

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


def best_lag_in_window(
    *,
    das_trace: np.ndarray,
    geo_trace: np.ndarray,
    time_s: np.ndarray,
    pmin_s: float,
    pmax_s: float,
    fs_hz: float,
) -> tuple[float, float, float]:
    """
    Diagnostic only.

    Returns:
        lag_s
        corr_at_best_lag
        zero_lag_corr

    Convention:
        lag > 0:
            MH030 would need to move LATER to best align with DAS.

        lag < 0:
            MH030 would need to move EARLIER to best align with DAS.
    """

    mask = (
        (time_s >= pmin_s)
        & (time_s <= pmax_s)
    )

    x = np.asarray(
        das_trace[mask],
        dtype=np.float64,
    )

    y = np.asarray(
        geo_trace[mask],
        dtype=np.float64,
    )

    x = (
        x
        - np.mean(x)
    )

    y = (
        y
        - np.mean(y)
    )

    corr_zero = zero_lag_corr(
        x,
        y,
    )

    max_lag_samples = int(
        round(
            MAX_LAG_S
            * fs_hz
        )
    )

    best_abs_corr = -np.inf
    best_corr = np.nan
    best_lag_samples = 0

    for lag in range(
        -max_lag_samples,
        max_lag_samples + 1,
    ):

        if lag > 0:
            # Geo sample y[i] would be moved to x[i + lag]:
            # geophone moves later.
            xx = x[
                lag:
            ]

            yy = y[
                :-lag
            ]

        elif lag < 0:

            xx = x[
                :lag
            ]

            yy = y[
                -lag:
            ]

        else:

            xx = x
            yy = y

        if xx.size < 20:
            continue

        corr = zero_lag_corr(
            xx,
            yy,
        )

        if (
            np.isfinite(corr)
            and abs(corr) > best_abs_corr
        ):

            best_abs_corr = abs(
                corr
            )

            best_corr = corr
            best_lag_samples = lag

    lag_s = (
        best_lag_samples
        / fs_hz
    )

    return (
        float(lag_s),
        float(best_corr),
        float(corr_zero),
    )


def symmetric_limit(
    data: np.ndarray,
) -> float:

    value = (
        1.05
        * float(
            np.nanmax(
                np.abs(data)
            )
        )
    )

    if (
        not np.isfinite(value)
        or value <= 0.0
    ):
        return 1.0

    return value


# =============================================================================
# DAS
# =============================================================================

def load_das_channels(
    *,
    origin: UTCDateTime,
    fmin_hz: float,
    fmax_hz: float,
) -> dict:

    for path in DAS_FILES:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

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

    traces = {}

    for channel in DAS_CHANNELS:

        if (
            channel < 0
            or channel >= das_data.shape[0]
        ):
            raise RuntimeError(
                f"DAS channel {channel} outside "
                f"shape {das_data.shape}."
            )

        trace = detrend_demean(
            das_data[
                channel,
                :
            ]
        )

        trace = five_percent_taper(
            trace
        )

        trace = common_causal_bandpass(
            trace,
            fs_hz=fs_hz,
            fmin_hz=fmin_hz,
            fmax_hz=fmax_hz,
        )

        traces[
            channel
        ] = trace

    return {
        "time_s": time_s,
        "fs_hz": fs_hz,
        "dx_m": float(
            info["dx"]
        ),
        "traces": traces,
    }


# =============================================================================
# MH030
# =============================================================================

def response_prefilter(
    *,
    native_fs_hz: float,
    science_fmax_hz: float,
) -> tuple[float, float, float, float]:

    nyquist = (
        0.5
        * native_fs_hz
    )

    high1 = min(
        max(
            1.20 * science_fmax_hz,
            science_fmax_hz + 10.0,
        ),
        0.82 * nyquist,
    )

    high2 = min(
        max(
            1.35 * science_fmax_hz,
            science_fmax_hz + 20.0,
        ),
        0.92 * nyquist,
    )

    return (
        0.5,
        1.0,
        high1,
        high2,
    )


def load_mh030_acceleration(
    *,
    client: Client,
    origin: UTCDateTime,
    das_time_s: np.ndarray,
    das_fs_hz: float,
    fmin_hz: float,
    fmax_hz: float,
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

    waveform_seed_id = (
        f"{NETWORK}."
        f"{CURRENT_STATION}."
        f"{LOCATION}."
        f"{GEOPHONE_CHANNEL}"
    )

    stream = client.get_waveforms(
        network=NETWORK,
        station=CURRENT_STATION,
        location=LOCATION,
        channel=GEOPHONE_CHANNEL,
        starttime=absolute_start,
        endtime=absolute_end,
        attach_response=False,
    )

    stream.merge(
        method=1,
        fill_value="interpolate",
    )

    if len(stream) != 1:
        raise RuntimeError(
            f"Expected one merged "
            f"{waveform_seed_id} trace; "
            f"found {len(stream)}."
        )

    tr = stream[0].copy()

    tr.data = detrend_demean(
        np.asarray(
            tr.data,
            dtype=np.float64,
        )
    )

    tr.data = five_percent_taper(
        tr.data
    )

    native_fs_hz = float(
        tr.stats.sampling_rate
    )

    metadata_station = (
        CURRENT_STATION
    )

    metadata_fallback = False

    try:

        inventory = client.get_stations(
            network=NETWORK,
            station=CURRENT_STATION,
            location=LOCATION,
            channel=GEOPHONE_CHANNEL,
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
            channel=GEOPHONE_CHANNEL,
            starttime=origin,
            endtime=origin + 1.0,
            level="response",
        )

    metadata_seed_id = (
        f"{NETWORK}."
        f"{metadata_station}."
        f"{LOCATION}."
        f"{GEOPHONE_CHANNEL}"
    )

    pre_filt = response_prefilter(
        native_fs_hz=native_fs_hz,
        science_fmax_hz=fmax_hz,
    )

    tr_velocity = tr.copy()

    if metadata_fallback:
        tr_velocity.stats.station = (
            metadata_station
        )

    tr_velocity.remove_response(
        inventory=inventory,
        output="VEL",
        pre_filt=pre_filt,
        water_level=60.0,
        taper=False,
        zero_mean=False,
    )

    geo_time_s = (
        float(
            tr.stats.starttime
            - origin
        )
        + np.arange(
            tr.stats.npts,
            dtype=np.float64,
        )
        / native_fs_hz
    )

    velocity_on_das_grid = np.interp(
        das_time_s,
        geo_time_s,
        np.asarray(
            tr_velocity.data,
            dtype=np.float64,
        ),
    )

    velocity_on_das_grid = detrend_demean(
        velocity_on_das_grid
    )

    velocity_filtered = common_causal_bandpass(
        velocity_on_das_grid,
        fs_hz=das_fs_hz,
        fmin_hz=fmin_hz,
        fmax_hz=fmax_hz,
    )

    acceleration_m_per_s2 = np.gradient(
        velocity_filtered,
        1.0 / das_fs_hz,
    )

    acceleration_cm_per_s2 = (
        acceleration_m_per_s2
        * 100.0
    )

    return {
        "acceleration_cm_per_s2": (
            acceleration_cm_per_s2
        ),
        "waveform_seed_id": (
            waveform_seed_id
        ),
        "metadata_seed_id": (
            metadata_seed_id
        ),
        "metadata_fallback": (
            metadata_fallback
        ),
        "response_prefilter_hz": (
            pre_filt
        ),
    }


# =============================================================================
# Plot
# =============================================================================

def make_diagnostic_figure(
    *,
    time_s: np.ndarray,
    das_traces: dict,
    acceleration_cm_per_s2: np.ndarray,
    fs_hz: float,
    tmin_s: float,
    tmax_s: float,
    pmin_s: float,
    pmax_s: float,
    fmin_hz: float,
    fmax_hz: float,
    output_path: Path,
) -> list[dict]:

    plot_mask = (
        (time_s >= tmin_s)
        & (time_s <= tmax_s)
    )

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(15.0, 10.0),
        sharex=True,
    )

    axes = axes.ravel()

    results = []

    geo_plot = (
        acceleration_cm_per_s2[
            plot_mask
        ]
    )

    geo_limit = symmetric_limit(
        geo_plot
    )

    for ax, channel in zip(
        axes,
        DAS_CHANNELS,
    ):

        das_trace = das_traces[
            channel
        ]

        das_plot = (
            das_trace[
                plot_mask
            ]
        )

        lag_s, best_corr, corr_zero = (
            best_lag_in_window(
                das_trace=das_trace,
                geo_trace=(
                    acceleration_cm_per_s2
                ),
                time_s=time_s,
                pmin_s=pmin_s,
                pmax_s=pmax_s,
                fs_hz=fs_hz,
            )
        )

        results.append(
            {
                "channel": channel,
                "lag_ms": (
                    lag_s
                    * 1000.0
                ),
                "best_corr": (
                    best_corr
                ),
                "zero_lag_corr": (
                    corr_zero
                ),
            }
        )

        ax2 = ax.twinx()

        line_das, = ax.plot(
            time_s[
                plot_mask
            ],
            das_plot,
            color=DAS_COLOR,
            linewidth=DAS_LINEWIDTH,
            label=(
                f"DAS ch {channel}"
            ),
        )

        line_geo, = ax2.plot(
            time_s[
                plot_mask
            ],
            geo_plot,
            color=GEO_COLOR,
            linewidth=GEO_LINEWIDTH,
            label="MH030 GP1",
        )

        ax.axvspan(
            pmin_s,
            pmax_s,
            color="0.94",
            zorder=0,
        )

        ax.axhline(
            0.0,
            color="0.83",
            linewidth=0.6,
        )

        ax.set_xlim(
            tmin_s,
            tmax_s,
        )

        das_limit = symmetric_limit(
            das_plot
        )

        ax.set_ylim(
            -das_limit,
            das_limit,
        )

        ax2.set_ylim(
            -geo_limit,
            geo_limit,
        )

        ax.set_title(
            (
                f"Current DAS ch {channel} | "
                f"lag = {lag_s*1000.0:+.0f} ms, "
                f"r = {best_corr:+.2f}"
            ),
            fontsize=11,
            fontweight="bold",
        )

        ax.tick_params(
            axis="both",
            labelsize=9,
        )

        ax2.tick_params(
            axis="y",
            labelsize=9,
        )

        # Y-axis labels only on outer columns.
        if ax in (
            axes[0],
            axes[2],
            axes[4],
        ):
            ax.set_ylabel(
                "DAS strain rate\n"
                + r"[$\mu$m/m/s]",
                fontsize=10,
                fontweight="bold",
            )

        if ax in (
            axes[1],
            axes[3],
            axes[5],
        ):
            ax2.set_ylabel(
                "MH030 acceleration\n"
                + r"[cm/s$^2$]",
                fontsize=10,
                fontweight="bold",
            )

        # Legend only once.
        if ax is axes[1]:

            ax.legend(
                [
                    line_das,
                    line_geo,
                ],
                [
                    "DAS strain rate",
                    "MH030 GP1 acceleration",
                ],
                loc="upper right",
                frameon=False,
                fontsize=9,
            )

    for ax in (
        axes[4],
        axes[5],
    ):
        ax.set_xlabel(
            "Time from exact NCEDC origin [s]",
            fontsize=11,
            fontweight="bold",
        )

    fig.suptitle(
        (
            "DAS / MH030 P-wave timing diagnostic | "
            f"{fmin_hz:g}-{fmax_hz:g} Hz | "
            "no time shift applied"
        ),
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )

    fig.text(
        0.5,
        0.012,
        (
            f"Gray window: P-wave lag diagnostic "
            f"{pmin_s:.2f}-{pmax_s:.2f} s. "
            "Positive lag means MH030 would move later."
        ),
        ha="center",
        fontsize=9,
        color="0.35",
    )

    fig.tight_layout(
        rect=[
            0.02,
            0.035,
            0.98,
            0.97,
        ]
    )

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
    )

    fig.savefig(
        output_path.with_suffix(
            ".pdf"
        ),
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return results


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fmin",
        type=float,
        default=DEFAULT_FMIN_HZ,
    )

    parser.add_argument(
        "--fmax",
        type=float,
        default=DEFAULT_FMAX_HZ,
    )

    parser.add_argument(
        "--tmin",
        type=float,
        default=DEFAULT_TMIN_S,
    )

    parser.add_argument(
        "--tmax",
        type=float,
        default=DEFAULT_TMAX_S,
    )

    parser.add_argument(
        "--pmin",
        type=float,
        default=DEFAULT_PMIN_S,
    )

    parser.add_argument(
        "--pmax",
        type=float,
        default=DEFAULT_PMAX_S,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = Client(
        "https://service.ncedc.org"
    )

    origin = get_exact_origin(
        client
    )

    das = load_das_channels(
        origin=origin,
        fmin_hz=args.fmin,
        fmax_hz=args.fmax,
    )

    geo = load_mh030_acceleration(
        client=client,
        origin=origin,
        das_time_s=(
            das["time_s"]
        ),
        das_fs_hz=(
            das["fs_hz"]
        ),
        fmin_hz=args.fmin,
        fmax_hz=args.fmax,
    )

    band_tag = (
        f"{args.fmin:g}-"
        f"{args.fmax:g}Hz"
    )

    figure_path = (
        args.output_dir
        / (
            "mh030_vs_das_channels_"
            f"{band_tag}.png"
        )
    )

    results = make_diagnostic_figure(
        time_s=(
            das["time_s"]
        ),
        das_traces=(
            das["traces"]
        ),
        acceleration_cm_per_s2=(
            geo[
                "acceleration_cm_per_s2"
            ]
        ),
        fs_hz=(
            das["fs_hz"]
        ),
        tmin_s=args.tmin,
        tmax_s=args.tmax,
        pmin_s=args.pmin,
        pmax_s=args.pmax,
        fmin_hz=args.fmin,
        fmax_hz=args.fmax,
        output_path=figure_path,
    )

    # Sort by absolute best correlation first.
    ranking = sorted(
        results,
        key=lambda row: abs(
            row["best_corr"]
        ),
        reverse=True,
    )

    print()
    print(
        "DAS / MH030 P-WAVE TIMING DIAGNOSTIC"
    )
    print(
        "===================================="
    )

    print(
        f"exact catalog origin       : "
        f"{origin}"
    )

    print(
        f"DAS channels               : "
        f"{DAS_CHANNELS}"
    )

    print(
        f"band                       : "
        f"{args.fmin:g}-"
        f"{args.fmax:g} Hz"
    )

    print(
        f"P-wave window              : "
        f"{args.pmin:.3f}-"
        f"{args.pmax:.3f} s"
    )

    print(
        "filter                     : "
        "DASutils.bandpass2D_c("
        "data, fmin, fmax, dt, "
        "zerophase=False)"
    )

    print(
        f"waveform                   : "
        f"{geo['waveform_seed_id']}"
    )

    print(
        f"response metadata          : "
        f"{geo['metadata_seed_id']}"
    )

    print(
        f"metadata fallback          : "
        f"{geo['metadata_fallback']}"
    )

    print(
        "relative shift applied     : NONE"
    )

    print()
    print(
        "channel    lag_ms    r_best    r_zero"
    )
    print(
        "--------------------------------------"
    )

    for row in results:

        print(
            f"{row['channel']:7d}  "
            f"{row['lag_ms']:+8.1f}  "
            f"{row['best_corr']:+8.4f}  "
            f"{row['zero_lag_corr']:+8.4f}"
        )

    print()
    print(
        "Ranking by |r_best|"
    )
    print(
        "-------------------"
    )

    for i, row in enumerate(
        ranking,
        start=1,
    ):

        print(
            f"{i}. ch {row['channel']} : "
            f"lag={row['lag_ms']:+.1f} ms, "
            f"r={row['best_corr']:+.4f}"
        )

    print()
    print(
        "Lag convention:"
    )
    print(
        "  + lag -> MH030 would move LATER"
    )
    print(
        "  - lag -> MH030 would move EARLIER"
    )
    print(
        "No lag is applied to the plotted traces."
    )

    print()
    print("Saved")
    print("=====")
    print(
        figure_path
    )
    print(
        figure_path.with_suffix(
            ".pdf"
        )
    )


if __name__ == "__main__":
    main()
