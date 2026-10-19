#!/usr/bin/env python3
"""
SAFOD Oct-03-2026 M0.8: simple DAS / MH030 comparison in the SAME style
as the earlier SCEC figure.

Outputs
-------
1) DAS strain rate [um/m/s]  vs  MH030 GP1 ground acceleration [cm/s^2]
2) DAS strain [microstrain]  vs  MH030 GP1 ground velocity [m/s]

Both figures use TWO Y AXES, one physical scale for each instrument.
No normalization. No amplitude fitting. No relative time shift.

October DAS:
    Sintela protobuf
    readFile_protobuf(..., diff=False, system="Sintela")

Common final filter:
    DASutils.bandpass2D_c(data, fmin, fmax, dt, zerophase=False)

The same final causal filter is applied to DAS and MH030 after both are on the
same 500-Hz time grid.

Current MH030 waveforms are available from NCEDC, but current StationXML
response metadata are not yet available.  The script therefore uses the
historical MH029 Pod-1 response metadata provisionally for response removal.

Default current Sintela channel: 1279.
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

NETWORK = "SF"
CURRENT_STATION = "MH030"
METADATA_FALLBACK_STATION = "MH029"
LOCATION = "01"
GEOPHONE_CHANNEL = "GP1"

DEFAULT_DAS_CHANNEL = 1279

DEFAULT_FMIN_HZ = 5.0
DEFAULT_FMAX_HZ = 40.0

DEFAULT_PLOT_TMIN_S = 0.0
DEFAULT_PLOT_TMAX_S = 1.0

EDGE_TAPER_FRACTION = 0.05
RESPONSE_MARGIN_S = 20.0

DEFAULT_OUTPUT_DIR = Path(
    "results/events/20261003_75443276/compare/"
    "das_mh030_two_physical_scales"
)

# Plot style copied from the earlier comparison.
PLOT_FIGSIZE = (11.2, 3.1)
DAS_COLOR = "0.52"
GEOPHONE_COLOR = "#940025"
DAS_LINEWIDTH = 2.0
GEOPHONE_LINEWIDTH = 1.90
DPI = 300


# =============================================================================
# Generic helpers
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
            f"WARNING: exact NCEDC origin query failed: {exc}"
        )
        print(
            f"Using fallback origin: {FALLBACK_ORIGIN}"
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
    IMPORTANT:
    exact calling convention used in the DAS notebooks:

        DASutils.bandpass2D_c(
            data,
            fmin,
            fmax,
            dt,
            zerophase=False,
        )

    One trace is wrapped as [1, nt].
    """

    if not (
        0.0
        < fmin_hz
        < fmax_hz
        < 0.5 * fs_hz
    ):
        raise ValueError(
            f"Invalid band {fmin_hz:g}-{fmax_hz:g} Hz "
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
            "Unexpected bandpass2D_c output shape: "
            f"{filtered.shape}"
        )

    return filtered[0]


def symmetric_ylim(
    ax,
    data: np.ndarray,
) -> None:

    limit = (
        1.05
        * float(
            np.nanmax(
                np.abs(data)
            )
        )
    )

    if (
        np.isfinite(limit)
        and limit > 0.0
    ):
        ax.set_ylim(
            -limit,
            limit,
        )


def format_axes(
    *,
    ax_left,
    ax_right,
) -> None:

    ax_left.tick_params(
        axis="both",
        labelsize=11,
        width=1.0,
        length=3.5,
    )

    ax_right.tick_params(
        axis="y",
        labelsize=11,
        width=1.0,
        length=3.5,
    )

    for label in (
        ax_left.get_xticklabels()
        + ax_left.get_yticklabels()
        + ax_right.get_yticklabels()
    ):
        label.set_fontweight(
            "bold"
        )

    for spine in (
        ax_left.spines.values()
    ):
        spine.set_linewidth(
            1.0
        )

    for spine in (
        ax_right.spines.values()
    ):
        spine.set_linewidth(
            1.0
        )


# =============================================================================
# DAS
# =============================================================================

def load_das_observables(
    *,
    origin: UTCDateTime,
    das_channel: int,
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

    if (
        das_channel < 0
        or das_channel >= das_data.shape[0]
    ):
        raise RuntimeError(
            f"DAS channel {das_channel} outside "
            f"shape {das_data.shape}."
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

    # Ettore's Oct-03 notebook plots the protobuf reader output directly
    # as strain rate [um/m/s].
    strain_rate_prepared = detrend_demean(
        das_data[
            das_channel,
            :
        ]
    )

    strain_rate_prepared = five_percent_taper(
        strain_rate_prepared
    )

    strain_rate_um_per_m_per_s = (
        common_causal_bandpass(
            strain_rate_prepared,
            fs_hz=fs_hz,
            fmin_hz=fmin_hz,
            fmax_hz=fmax_hz,
        )
    )

    # Integrate [um/m/s] in time -> [um/m], numerically equal to microstrain.
    strain_microstrain = cumulative_trapezoid(
        strain_rate_prepared,
        dx=(
            1.0
            / fs_hz
        ),
        initial=0.0,
    )

    strain_microstrain = detrend_demean(
        strain_microstrain
    )

    strain_microstrain = (
        common_causal_bandpass(
            strain_microstrain,
            fs_hz=fs_hz,
            fmin_hz=fmin_hz,
            fmax_hz=fmax_hz,
        )
    )

    return {
        "time_s": time_s,
        "fs_hz": fs_hz,
        "dx_m": float(
            info["dx"]
        ),
        "beg_time": beg_time,
        "end_time": (
            beg_time
            + (
                das_data.shape[1] - 1
            )
            / fs_hz
        ),
        "strain_rate_um_per_m_per_s": (
            strain_rate_um_per_m_per_s
        ),
        "strain_microstrain": (
            strain_microstrain
        ),
    }


# =============================================================================
# MH030 GP1
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

    if not (
        0.5
        < 1.0
        < science_fmax_hz
        < high1
        < high2
        < nyquist
    ):
        raise RuntimeError(
            "Could not construct a safe response-removal "
            f"prefilter for fmax={science_fmax_hz:g} Hz "
            f"and native fs={native_fs_hz:g} Hz."
        )

    return (
        0.5,
        1.0,
        high1,
        high2,
    )


def load_geophone_observables(
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

    # Current waveform.
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
            f"Expected one merged {waveform_seed_id} trace; "
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

    # Current MH030 StationXML is not yet available; try it first,
    # then fall back to historical MH029 Pod-1 metadata.
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
        # Make the trace ID match the fallback StationXML for ObsPy.
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

    if (
        das_time_s[0]
        < geo_time_s[0]
        or das_time_s[-1]
        > geo_time_s[-1]
    ):
        raise RuntimeError(
            f"{waveform_seed_id} does not cover "
            "the complete DAS interval."
        )

    # Put the response-corrected velocity onto the DAS 500-Hz time grid.
    velocity = np.interp(
        das_time_s,
        geo_time_s,
        np.asarray(
            tr_velocity.data,
            dtype=np.float64,
        ),
    )

    velocity = detrend_demean(
        velocity
    )

    velocity_filtered = (
        common_causal_bandpass(
            velocity,
            fs_hz=das_fs_hz,
            fmin_hz=fmin_hz,
            fmax_hz=fmax_hz,
        )
    )

    # Velocity -> acceleration AFTER final common filtering,
    # same logic as the earlier comparison script.
    acceleration_m_per_s2 = np.gradient(
        velocity_filtered,
        1.0 / das_fs_hz,
    )

    acceleration_cm_per_s2 = (
        acceleration_m_per_s2
        * 100.0
    )

    return {
        "velocity_m_per_s": (
            velocity_filtered
        ),
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
        "native_fs_hz": (
            native_fs_hz
        ),
        "response_prefilter_hz": (
            pre_filt
        ),
    }


# =============================================================================
# Plot 1: strain rate vs acceleration
# =============================================================================

def plot_strain_rate_vs_acceleration(
    *,
    time_s: np.ndarray,
    strain_rate_um_per_m_per_s: np.ndarray,
    acceleration_cm_per_s2: np.ndarray,
    das_channel: int,
    tmin_s: float,
    tmax_s: float,
    output_path: Path,
) -> None:

    mask = (
        (time_s >= tmin_s)
        & (time_s <= tmax_s)
    )

    t = time_s[mask]

    das = (
        strain_rate_um_per_m_per_s[
            mask
        ]
    )

    geo = (
        acceleration_cm_per_s2[
            mask
        ]
    )

    fig, ax1 = plt.subplots(
        figsize=PLOT_FIGSIZE,
    )

    line_das, = ax1.plot(
        t,
        das,
        color=DAS_COLOR,
        linewidth=DAS_LINEWIDTH,
        label=(
            f"DAS channel "
            f"{das_channel}"
        ),
    )

    ax1.set_xlabel(
        "Time from catalog origin [s]",
        fontsize=13,
        fontweight="bold",
    )

    ax1.set_ylabel(
        "Strain rate\n"
        + r"[$\mu$m/m/s]",
        fontsize=13,
        fontweight="bold",
    )

    ax1.axvline(
        0.0,
        color="0.50",
        linestyle=":",
        linewidth=0.9,
    )

    ax1.axhline(
        0.0,
        color="0.82",
        linewidth=0.7,
    )

    ax1.set_xlim(
        tmin_s,
        tmax_s,
    )

    ax2 = ax1.twinx()

    line_geo, = ax2.plot(
        t,
        geo,
        color=GEOPHONE_COLOR,
        linewidth=GEOPHONE_LINEWIDTH,
        label="MH030 GP1",
    )

    ax2.set_ylabel(
        "GP1 ground acceleration\n"
        + r"[cm/s$^2$]",
        fontsize=13,
        fontweight="bold",
    )

    symmetric_ylim(
        ax1,
        das,
    )

    symmetric_ylim(
        ax2,
        geo,
    )

    format_axes(
        ax_left=ax1,
        ax_right=ax2,
    )

    ax1.legend(
        [
            line_das,
            line_geo,
        ],
        [
            line_das.get_label(),
            line_geo.get_label(),
        ],
        loc="upper right",
        frameon=False,
        fontsize=10,
        handlelength=2.2,
    )

    fig.tight_layout(
        pad=0.5
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

    plt.close(fig)


# =============================================================================
# Plot 2: strain vs velocity
# =============================================================================

def plot_strain_vs_velocity(
    *,
    time_s: np.ndarray,
    strain_microstrain: np.ndarray,
    velocity_m_per_s: np.ndarray,
    das_channel: int,
    tmin_s: float,
    tmax_s: float,
    output_path: Path,
) -> None:

    mask = (
        (time_s >= tmin_s)
        & (time_s <= tmax_s)
    )

    t = time_s[mask]

    das = (
        strain_microstrain[
            mask
        ]
    )

    geo = (
        velocity_m_per_s[
            mask
        ]
    )

    fig, ax1 = plt.subplots(
        figsize=PLOT_FIGSIZE,
    )

    line_das, = ax1.plot(
        t,
        das,
        color=DAS_COLOR,
        linewidth=DAS_LINEWIDTH,
        label=(
            f"DAS channel "
            f"{das_channel}"
        ),
    )

    ax1.set_xlabel(
        "Time from catalog origin [s]",
        fontsize=13,
        fontweight="bold",
    )

    ax1.set_ylabel(
        "Axial strain\n"
        + r"[$\mu$m/m]",
        fontsize=13,
        fontweight="bold",
    )

    ax1.axvline(
        0.0,
        color="0.50",
        linestyle=":",
        linewidth=0.9,
    )

    ax1.axhline(
        0.0,
        color="0.82",
        linewidth=0.7,
    )

    ax1.set_xlim(
        tmin_s,
        tmax_s,
    )

    ax2 = ax1.twinx()

    line_geo, = ax2.plot(
        t,
        geo,
        color=GEOPHONE_COLOR,
        linewidth=GEOPHONE_LINEWIDTH,
        label="MH030 GP1",
    )

    ax2.set_ylabel(
        "GP1 ground velocity\n"
        "[m/s]",
        fontsize=13,
        fontweight="bold",
    )

    symmetric_ylim(
        ax1,
        das,
    )

    symmetric_ylim(
        ax2,
        geo,
    )

    format_axes(
        ax_left=ax1,
        ax_right=ax2,
    )

    ax1.legend(
        [
            line_das,
            line_geo,
        ],
        [
            line_das.get_label(),
            line_geo.get_label(),
        ],
        loc="upper right",
        frameon=False,
        fontsize=10,
        handlelength=2.2,
    )

    fig.tight_layout(
        pad=0.5
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

    plt.close(fig)


# =============================================================================
# CLI / main
# =============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Oct-03 SAFOD current Sintela DAS / "
            "MH030 GP1 physical comparison."
        )
    )

    parser.add_argument(
        "--channel",
        type=int,
        default=DEFAULT_DAS_CHANNEL,
    )

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
        default=DEFAULT_PLOT_TMIN_S,
    )

    parser.add_argument(
        "--tmax",
        type=float,
        default=DEFAULT_PLOT_TMAX_S,
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

    das = load_das_observables(
        origin=origin,
        das_channel=args.channel,
        fmin_hz=args.fmin,
        fmax_hz=args.fmax,
    )

    geo = load_geophone_observables(
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

    fig_rate = (
        args.output_dir
        / (
            "01_das_strain_rate_vs_"
            "mh030_gp1_acceleration_"
            f"{band_tag}.png"
        )
    )

    fig_strain = (
        args.output_dir
        / (
            "02_das_strain_vs_"
            "mh030_gp1_velocity_"
            f"{band_tag}.png"
        )
    )

    plot_strain_rate_vs_acceleration(
        time_s=(
            das["time_s"]
        ),
        strain_rate_um_per_m_per_s=(
            das[
                "strain_rate_um_per_m_per_s"
            ]
        ),
        acceleration_cm_per_s2=(
            geo[
                "acceleration_cm_per_s2"
            ]
        ),
        das_channel=args.channel,
        tmin_s=args.tmin,
        tmax_s=args.tmax,
        output_path=fig_rate,
    )

    plot_strain_vs_velocity(
        time_s=(
            das["time_s"]
        ),
        strain_microstrain=(
            das[
                "strain_microstrain"
            ]
        ),
        velocity_m_per_s=(
            geo[
                "velocity_m_per_s"
            ]
        ),
        das_channel=args.channel,
        tmin_s=args.tmin,
        tmax_s=args.tmax,
        output_path=fig_strain,
    )

    print()
    print(
        "OCT-03 DAS / MH030 PHYSICAL COMPARISON"
    )
    print(
        "======================================"
    )
    print(
        f"catalog origin             : "
        f"{origin}"
    )
    print(
        f"DAS channel                : "
        f"{args.channel}"
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
        f"filter                     : "
        f"{args.fmin:g}-{args.fmax:g} Hz, "
        "DASutils.bandpass2D_c, "
        "zerophase=False"
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
        f"response prefilter         : "
        f"{geo['response_prefilter_hz']}"
    )
    print(
        "relative DAS/MH030 shift  : NONE"
    )
    print(
        "normalization              : NONE"
    )
    print(
        "amplitude fitting          : NONE"
    )
    print()
    print("Saved")
    print("=====")
    print(fig_rate)
    print(fig_rate.with_suffix(".pdf"))
    print(fig_strain)
    print(fig_strain.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
