#!/usr/bin/env python3
# scripts/safod/experiments/oct03_2026/plot_oct03_das_geophone_like_scec.py
#
# Oct-03-2026 SAFOD DAS / borehole-geophone comparison.
#
# This is intentionally kept as close as possible to the earlier
# plot_das_geophone_two_physical_comparisons.py used for the SCEC figure.
#
# What is kept the same:
#   * linear detrend + demean
#   * 5% Tukey/Hann edge taper
#   * causal one-pass linear-phase Hamming FIR bandpass
#   * SAME final FIR applied to DAS and geophone
#   * DAS strain rate <-> geophone acceleration
#   * DAS strain      <-> geophone velocity
#   * geophone velocity differentiated AFTER the common final filter
#   * two independent physical Y axes
#   * no normalization
#   * no amplitude fitting
#   * no relative time shift
#
# What has to change for the Oct-03 data:
#   * DAS input is Sintela protobuf, not OptaSense HDF5
#   * readFile_protobuf(..., diff=False, system="Sintela")
#   * protobuf reader output is treated as strain rate [um/m/s],
#     following Ettore's Oct-03 notebook
#   * current waveform station is SF.MH030.01.GP1
#   * current MH030 StationXML is not yet available, so MH029 Pod-1
#     response/orientation metadata are used provisionally if needed
#   * default current Sintela channel is 1279
#
# IMPORTANT FILTER NOTE:
# The old April script used 401 taps at fs=1000 Hz:
#       group delay = (401-1)/(2*1000) = 0.200 s
#
# Oct-03 DAS is fs=500 Hz. To preserve the SAME 0.200-s FIR group delay
# and approximately the same temporal filter length, the default here is
# 201 taps:
#       group delay = (201-1)/(2*500) = 0.200 s
#
# The FIR delay is common to BOTH traces, so it does not create a relative
# DAS/MH030 time shift. It does shift both filtered traces later relative
# to the raw catalog-origin axis, exactly as the old causal-FIR workflow did.

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
from scipy.signal import detrend, firwin, lfilter
from scipy.signal.windows import tukey


# ==============================================================================
# DASutils
# ==============================================================================

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


# ==============================================================================
# Event / data configuration
# ==============================================================================

EVENT_ID_NUMERIC = 75443276
EVENT_ID_LABEL = "NC75443276"

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

DEFAULT_DAS_CHANNEL = 1279
DAS_MEDIAN_HALF_WINDOW = 0

NETWORK = "SF"
STATION = "MH030"
METADATA_FALLBACK_STATION = "MH029"
LOCATION = "01"
GEOPHONE_CHANNEL = "GP1"


# ==============================================================================
# Processing
# ==============================================================================

DEFAULT_FMIN_HZ = 5.0
DEFAULT_FMAX_HZ = 40.0

# Old April script:
#   401 taps at 1000 Hz -> 0.200 s group delay.
#
# New Oct-03 data:
#   201 taps at 500 Hz -> 0.200 s group delay.
DEFAULT_FIR_NUMTAPS = 201

EDGE_TAPER_FRACTION = 0.05

DEFAULT_PLOT_TMIN_S = 0.0
DEFAULT_PLOT_TMAX_S = 1.0

DEFAULT_CORR_TMIN_S = 0.0
DEFAULT_CORR_TMAX_S = 1.0

RESPONSE_MARGIN_S = 20.0

DEFAULT_OUTPUT_DIR = Path(
    "results/events/20261003_75443276/compare/"
    "das_geophone_like_scec"
)

DPI = 300

# Same compact poster-like style as the final comparison.
PLOT_FIGSIZE = (11.2, 3.1)
DAS_COLOR = "gray"
DAS_LINEWIDTH = 2.0
GEOPHONE_COLOR = "#940025"
GEOPHONE_LINEWIDTH = 1.90


# ==============================================================================
# Helpers
# ==============================================================================

def parse_beg_time(info) -> UTCDateTime:
    value = info["begTime"]

    if isinstance(value, datetime.datetime):
        dt = value

        if dt.tzinfo is None:
            dt = dt.replace(
                tzinfo=datetime.timezone.utc
            )

        return UTCDateTime(dt)

    dt = dateutil.parser.parse(str(value))

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
            eventid=EVENT_ID_NUMERIC
        )

        if len(catalog) < 1:
            raise RuntimeError(
                "NCEDC returned no event."
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
    x: np.ndarray,
    axis: int = -1,
) -> np.ndarray:

    y = detrend(
        np.asarray(
            x,
            dtype=np.float64,
        ),
        axis=axis,
        type="linear",
    )

    return (
        y
        - np.mean(
            y,
            axis=axis,
            keepdims=True,
        )
    )


def edge_taper(
    x: np.ndarray,
    axis: int = -1,
) -> np.ndarray:

    y = np.asarray(
        x,
        dtype=np.float64,
    )

    n = y.shape[axis]

    window = tukey(
        n,
        alpha=(
            2.0
            * EDGE_TAPER_FRACTION
        ),
    )

    shape = [1] * y.ndim
    shape[axis] = n

    return (
        y
        * window.reshape(shape)
    )


def causal_linear_phase_bandpass(
    x: np.ndarray,
    fs_hz: float,
    fmin_hz: float,
    fmax_hz: float,
    fir_numtaps: int,
    axis: int = -1,
) -> np.ndarray:
    """
    SAME filter construction as the old SCEC comparison:
      scipy.signal.firwin + one forward scipy.signal.lfilter pass.
    """

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    nyquist = (
        0.5
        * float(fs_hz)
    )

    if not (
        0.0
        < fmin_hz
        < fmax_hz
        < nyquist
    ):
        raise ValueError(
            f"Invalid band "
            f"{fmin_hz:g}-{fmax_hz:g} Hz "
            f"for fs={fs_hz:g} Hz."
        )

    if (
        fir_numtaps < 3
        or fir_numtaps % 2 == 0
    ):
        raise ValueError(
            "fir_numtaps must be an odd integer >= 3."
        )

    taps = firwin(
        fir_numtaps,
        [
            fmin_hz,
            fmax_hz,
        ],
        pass_zero=False,
        window="hamming",
        fs=float(fs_hz),
        scale=True,
    )

    return lfilter(
        taps,
        [1.0],
        x,
        axis=axis,
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


def response_prefilter(
    *,
    native_fs_hz: float,
    science_fmax_hz: float,
) -> tuple[float, float, float, float]:
    """
    Response-removal prefilter only.

    Old code used (0.2, 0.4, 30, 40) for a 1-20 Hz science band.
    Here the upper corners are widened automatically for 5-40 or 5-200 Hz.
    """

    nyquist = (
        0.5
        * native_fs_hz
    )

    low1 = 0.5
    low2 = 1.0

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
        low1
        < low2
        < science_fmax_hz
        < high1
        < high2
        < nyquist
    ):
        raise RuntimeError(
            f"Could not construct response prefilter "
            f"for fmax={science_fmax_hz:g} Hz "
            f"and native fs={native_fs_hz:g} Hz."
        )

    return (
        low1,
        low2,
        high1,
        high2,
    )


# ==============================================================================
# DAS
# ==============================================================================

def load_das_observables(
    *,
    origin: UTCDateTime,
    das_channel: int,
    fmin_hz: float,
    fmax_hz: float,
    fir_numtaps: int,
) -> dict:

    for path in DAS_FILES:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    # Same read call as Ettore's Oct-03 notebook.
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

    i0 = (
        das_channel
        - DAS_MEDIAN_HALF_WINDOW
    )

    i1 = (
        das_channel
        + DAS_MEDIAN_HALF_WINDOW
        + 1
    )

    if (
        i0 < 0
        or i1 > das_data.shape[0]
    ):
        raise RuntimeError(
            f"DAS channel window "
            f"[{i0}:{i1}] outside "
            f"shape {das_data.shape}."
        )

    # ------------------------------------------------------------------
    # Oct-03 Sintela units.
    #
    # Ettore's notebook labels the readFile_protobuf(..., diff=False)
    # output directly as strain rate [um/m/s].
    #
    # Convert that to SI [1/s] for the exact same downstream logic
    # as the old SCEC script:
    #
    #     1 um/m/s = 1e-6 1/s
    # ------------------------------------------------------------------

    strain_rate_si_channels = (
        das_data[
            i0:i1,
            :
        ]
        * 1.0e-6
    )

    strain_rate_si_channels = detrend_demean(
        strain_rate_si_channels,
        axis=1,
    )

    strain_rate_si_channels = edge_taper(
        strain_rate_si_channels,
        axis=1,
    )

    if DAS_MEDIAN_HALF_WINDOW == 0:

        strain_rate_prepared = (
            strain_rate_si_channels[
                0,
                :
            ]
        )

    else:

        strain_rate_prepared = np.median(
            strain_rate_si_channels,
            axis=0,
        )

    # ------------------------------------------------------------------
    # EXACT SAME causal linear-phase FIR workflow as old code.
    # ------------------------------------------------------------------

    strain_rate_filtered_si = (
        causal_linear_phase_bandpass(
            strain_rate_prepared,
            fs_hz,
            fmin_hz,
            fmax_hz,
            fir_numtaps,
        )
    )

    strain_rate_um_per_m_per_s = (
        strain_rate_filtered_si
        * 1.0e6
    )

    # ------------------------------------------------------------------
    # Strain: integrate prepared SI strain rate, then apply SAME FIR.
    # ------------------------------------------------------------------

    strain = cumulative_trapezoid(
        strain_rate_prepared,
        dx=(
            1.0
            / fs_hz
        ),
        initial=0.0,
    )

    strain = detrend_demean(
        strain
    )

    strain_filtered = (
        causal_linear_phase_bandpass(
            strain,
            fs_hz,
            fmin_hz,
            fmax_hz,
            fir_numtaps,
        )
    )

    return {
        "time_s": (
            time_s
        ),

        "fs_hz": (
            fs_hz
        ),

        "dx_m": float(
            info["dx"]
        ),

        "strain_rate_1_per_s": (
            strain_rate_filtered_si
        ),

        "strain_rate_um_per_m_per_s": (
            strain_rate_um_per_m_per_s
        ),

        "strain": (
            strain_filtered
        ),

        "beg_time": (
            beg_time
        ),

        "end_time": (
            beg_time
            + (
                das_data.shape[1]
                - 1
            )
            / fs_hz
        ),

        "channel_first": (
            i0
        ),

        "channel_last": (
            i1 - 1
        ),
    }


# ==============================================================================
# Geophone
# ==============================================================================

def load_geophone_observables(
    *,
    client: Client,
    origin: UTCDateTime,
    das_time_s: np.ndarray,
    das_fs_hz: float,
    fmin_hz: float,
    fmax_hz: float,
    fir_numtaps: int,
) -> dict:

    waveform_seed_id = (
        f"{NETWORK}."
        f"{STATION}."
        f"{LOCATION}."
        f"{GEOPHONE_CHANNEL}"
    )

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

    # ------------------------------------------------------------------
    # Current MH030 waveform.
    # ------------------------------------------------------------------

    try:

        stream = client.get_waveforms(
            network=NETWORK,
            station=STATION,
            location=LOCATION,
            channel=GEOPHONE_CHANNEL,
            starttime=absolute_start,
            endtime=absolute_end,
            attach_response=False,
        )

    except FDSNNoDataException as exc:

        raise RuntimeError(
            f"No waveform available "
            f"for {waveform_seed_id}."
        ) from exc

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

    tr.data = edge_taper(
        tr.data
    )

    native_fs_hz = float(
        tr.stats.sampling_rate
    )

    # ------------------------------------------------------------------
    # Response metadata:
    # current MH030 first, historical MH029 Pod-1 fallback second.
    # ------------------------------------------------------------------

    metadata_station = (
        STATION
    )

    metadata_fallback = False

    try:

        inventory = client.get_stations(
            network=NETWORK,
            station=STATION,
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

    # ------------------------------------------------------------------
    # Counts -> physical ground velocity [m/s].
    # SAME response-removal logic as old code.
    # ------------------------------------------------------------------

    tr_velocity = tr.copy()

    if metadata_fallback:
        # Make the temporary trace ID match the fallback StationXML.
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

    orientation = inventory.get_orientation(
        metadata_seed_id,
        datetime=origin,
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
            "the complete DAS UTC interval."
        )

    # Same as old code: put geophone velocity onto the DAS time grid first.
    velocity = np.interp(
        das_time_s,
        geo_time_s,
        np.asarray(
            tr_velocity.data,
            dtype=np.float64,
        ),
    )

    # ------------------------------------------------------------------
    # EXACT SAME final causal linear-phase FIR as DAS.
    # ------------------------------------------------------------------

    velocity_filtered = (
        causal_linear_phase_bandpass(
            velocity,
            das_fs_hz,
            fmin_hz,
            fmax_hz,
            fir_numtaps,
        )
    )

    # ------------------------------------------------------------------
    # Velocity -> acceleration AFTER common final filtering.
    # ------------------------------------------------------------------

    acceleration_m_per_s2 = np.gradient(
        velocity_filtered,
        (
            1.0
            / das_fs_hz
        ),
    )

    acceleration_cm_per_s2 = (
        acceleration_m_per_s2
        * 100.0
    )

    return {
        "velocity_m_per_s": (
            velocity_filtered
        ),

        "acceleration_m_per_s2": (
            acceleration_m_per_s2
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

        "azimuth_deg": float(
            orientation["azimuth"]
        ),

        "dip_deg": float(
            orientation["dip"]
        ),

        "native_fs_hz": (
            native_fs_hz
        ),

        "response_prefilter_hz": (
            pre_filt
        ),
    }


# ==============================================================================
# Plot helpers
# ==============================================================================

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


def symmetric_ylim(
    ax,
    data: np.ndarray,
) -> None:

    limit = (
        1.05
        * float(
            np.max(
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


# ==============================================================================
# Figure 1: strain vs velocity
# ==============================================================================

def plot_strain_vs_velocity(
    *,
    time_s: np.ndarray,
    strain: np.ndarray,
    velocity: np.ndarray,
    das_channel: int,
    plot_tmin_s: float,
    plot_tmax_s: float,
    output_path: Path,
) -> None:

    mask = (
        (time_s >= plot_tmin_s)
        & (time_s <= plot_tmax_s)
    )

    t = time_s[mask]
    das = strain[mask]
    geo = velocity[mask]

    fig, ax1 = plt.subplots(
        figsize=PLOT_FIGSIZE,
    )

    line_das, = ax1.plot(
        t,
        das,
        color=DAS_COLOR,
        linewidth=DAS_LINEWIDTH,
        label=f"DAS channel {das_channel}",
    )

    ax1.set_xlabel(
        "Time from catalog origin [s]",
        fontsize=13,
        fontweight="bold",
    )

    ax1.set_ylabel(
        "Axial strain\n[dimensionless]",
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
        plot_tmin_s,
        plot_tmax_s,
    )

    ax2 = ax1.twinx()

    line_geo, = ax2.plot(
        t,
        geo,
        color=GEOPHONE_COLOR,
        linewidth=GEOPHONE_LINEWIDTH,
        label="MH030 GP1 velocity",
    )

    ax2.set_ylabel(
        "GP1 ground velocity\n[m/s]",
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
        output_path.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    plt.close(fig)


# ==============================================================================
# Figure 2: strain rate vs acceleration
# ==============================================================================

def plot_strain_rate_vs_acceleration(
    *,
    time_s: np.ndarray,
    strain_rate_um_per_m_per_s: np.ndarray,
    acceleration_cm_per_s2: np.ndarray,
    das_channel: int,
    plot_tmin_s: float,
    plot_tmax_s: float,
    output_path: Path,
) -> None:

    mask = (
        (time_s >= plot_tmin_s)
        & (time_s <= plot_tmax_s)
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
        label=f"DAS channel {das_channel}",
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
        plot_tmin_s,
        plot_tmax_s,
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
        output_path.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    plt.close(fig)


# ==============================================================================
# Arguments
# ==============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Generate Oct-03 SAFOD DAS / MH030 GP1 "
            "physical comparison figures using the same "
            "processing logic as the earlier SCEC script."
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
        "--fir-numtaps",
        type=int,
        default=DEFAULT_FIR_NUMTAPS,
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
        "--corr-tmin",
        type=float,
        default=DEFAULT_CORR_TMIN_S,
    )

    parser.add_argument(
        "--corr-tmax",
        type=float,
        default=DEFAULT_CORR_TMAX_S,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser.parse_args()


# ==============================================================================
# Main
# ==============================================================================

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
        fir_numtaps=args.fir_numtaps,
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
        fir_numtaps=args.fir_numtaps,
    )

    # ------------------------------------------------------------------
    # Diagnostic correlation only.
    # No shift is applied.
    # ------------------------------------------------------------------

    corr_mask = (
        (das["time_s"] >= args.corr_tmin)
        & (das["time_s"] <= args.corr_tmax)
    )

    corr_strain_velocity = zero_lag_corr(
        das["strain"][
            corr_mask
        ],
        geo["velocity_m_per_s"][
            corr_mask
        ],
    )

    corr_rate_acceleration = zero_lag_corr(
        das[
            "strain_rate_um_per_m_per_s"
        ][corr_mask],
        geo[
            "acceleration_cm_per_s2"
        ][corr_mask],
    )

    # ------------------------------------------------------------------
    # Figures.
    # ------------------------------------------------------------------

    band_tag = (
        f"{args.fmin:g}-"
        f"{args.fmax:g}Hz"
    )

    fig1_path = (
        args.output_dir
        / (
            "01_das_strain_vs_"
            "mh030_gp1_velocity_"
            f"{band_tag}.png"
        )
    )

    plot_strain_vs_velocity(
        time_s=(
            das["time_s"]
        ),
        strain=(
            das["strain"]
        ),
        velocity=(
            geo["velocity_m_per_s"]
        ),
        das_channel=args.channel,
        plot_tmin_s=args.tmin,
        plot_tmax_s=args.tmax,
        output_path=(
            fig1_path
        ),
    )

    fig2_path = (
        args.output_dir
        / (
            "02_das_strain_rate_vs_"
            "mh030_gp1_acceleration_"
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
        plot_tmin_s=args.tmin,
        plot_tmax_s=args.tmax,
        output_path=(
            fig2_path
        ),
    )

    # ------------------------------------------------------------------
    # Processing delay.
    # ------------------------------------------------------------------

    fir_group_delay_s = (
        (args.fir_numtaps - 1)
        / (
            2.0
            * das["fs_hz"]
        )
    )

    # ------------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------------

    npz_path = (
        args.output_dir
        / (
            "das_geophone_physical_pairs_"
            f"{band_tag}.npz"
        )
    )

    np.savez_compressed(
        npz_path,

        event_id=np.array(
            EVENT_ID_LABEL
        ),

        catalog_origin=np.array(
            str(origin)
        ),

        time_from_catalog_origin_s=(
            das["time_s"]
        ),

        das_axial_strain=(
            das["strain"]
        ),

        das_axial_strain_rate_1_per_s=(
            das[
                "strain_rate_1_per_s"
            ]
        ),

        das_axial_strain_rate_um_per_m_per_s=(
            das[
                "strain_rate_um_per_m_per_s"
            ]
        ),

        gp1_velocity_m_per_s=(
            geo[
                "velocity_m_per_s"
            ]
        ),

        gp1_acceleration_m_per_s2=(
            geo[
                "acceleration_m_per_s2"
            ]
        ),

        gp1_acceleration_cm_per_s2=(
            geo[
                "acceleration_cm_per_s2"
            ]
        ),

        das_channel_center=np.array(
            args.channel
        ),

        das_dx_m=np.array(
            das["dx_m"]
        ),

        waveform_seed_id=np.array(
            geo[
                "waveform_seed_id"
            ]
        ),

        metadata_seed_id=np.array(
            geo[
                "metadata_seed_id"
            ]
        ),

        metadata_fallback=np.array(
            geo[
                "metadata_fallback"
            ]
        ),

        gp1_azimuth_deg=np.array(
            geo[
                "azimuth_deg"
            ]
        ),

        gp1_dip_deg=np.array(
            geo[
                "dip_deg"
            ]
        ),

        common_fmin_hz=np.array(
            args.fmin
        ),

        common_fmax_hz=np.array(
            args.fmax
        ),

        fir_numtaps=np.array(
            args.fir_numtaps
        ),

        fir_group_delay_s=np.array(
            fir_group_delay_s
        ),

        common_filter_description=np.array(
            "causal one-pass linear-phase Hamming FIR bandpass"
        ),

        zero_lag_corr_strain_velocity=np.array(
            corr_strain_velocity
        ),

        zero_lag_corr_strain_rate_acceleration=np.array(
            corr_rate_acceleration
        ),
    )

    # ------------------------------------------------------------------
    # Console.
    # ------------------------------------------------------------------

    print()
    print(
        "OCT-03 PHYSICAL DAS / GEOPHONE COMPARISON"
    )
    print(
        "=========================================="
    )

    print(
        f"event                        : "
        f"{EVENT_ID_LABEL}"
    )

    print(
        f"catalog origin               : "
        f"{origin}"
    )

    print(
        f"DAS absolute start           : "
        f"{das['beg_time']}"
    )

    print(
        f"DAS absolute end             : "
        f"{das['end_time']}"
    )

    print(
        f"DAS current Sintela channel  : "
        f"{args.channel}"
    )

    print(
        f"DAS fs                       : "
        f"{das['fs_hz']:.3f} Hz"
    )

    print(
        f"DAS dx                       : "
        f"{das['dx_m']:.4f} m"
    )

    print(
        f"waveform                     : "
        f"{geo['waveform_seed_id']}"
    )

    print(
        f"response/orientation metadata: "
        f"{geo['metadata_seed_id']}"
    )

    print(
        f"metadata fallback            : "
        f"{geo['metadata_fallback']}"
    )

    print(
        f"GP1 orientation              : "
        f"az={geo['azimuth_deg']:+.2f} deg, "
        f"dip={geo['dip_deg']:+.2f} deg"
    )

    print(
        f"response prefilter           : "
        f"{geo['response_prefilter_hz']}"
    )

    print(
        f"common final filter          : "
        f"{args.fmin:g}-{args.fmax:g} Hz"
    )

    print(
        f"filter                       : "
        f"causal linear-phase FIR, "
        f"{args.fir_numtaps} taps"
    )

    print(
        f"FIR group delay              : "
        f"{fir_group_delay_s:.3f} s"
    )

    print(
        "zero-phase filtering          : NO"
    )

    print(
        "manual relative time shift    : NONE"
    )

    print(
        "polarity flip                 : NONE"
    )

    print(
        "normalization                  : NONE"
    )

    print(
        "amplitude fitting              : NONE"
    )

    print(
        "DAS strain-rate units         : um/m/s"
    )

    print(
        "geophone acceleration units   : cm/s^2"
    )

    print(
        f"corr(strain, velocity)        : "
        f"{corr_strain_velocity:+.4f}"
    )

    print(
        f"corr(strain rate, accel.)     : "
        f"{corr_rate_acceleration:+.4f}"
    )

    if geo["metadata_fallback"]:
        print()
        print(
            "WARNING: MH030 waveform is current, but "
            "MH029 Pod-1 response metadata are provisional."
        )

    print()
    print("Saved")
    print("=====")

    print(fig1_path)
    print(fig1_path.with_suffix(".pdf"))
    print(fig2_path)
    print(fig2_path.with_suffix(".pdf"))
    print(npz_path)


if __name__ == "__main__":
    main()
