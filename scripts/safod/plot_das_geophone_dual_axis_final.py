# scripts/safod/plot_das_geophone_two_physical_comparisons.py
#
# Final SAFOD DAS / borehole-geophone comparison for NC75336802.
#
# Produces TWO physically consistent figures:
#
#   1) DAS axial strain        vs MH029 GP1 axial ground velocity
#      [dimensionless]            [m/s]
#
#   2) DAS axial strain rate   vs MH029 GP1 axial ground acceleration
#      [1/s]                      [m/s^2]
#
# The two quantities in each figure are related for a locally plane wave:
#
#       epsilon_parallel  ~  - v_parallel / c_app
#       epsilon_dot       ~  - a_parallel / c_app
#
# We DO NOT estimate or apply c_app here. Therefore each observable stays on
# its own physical y-axis. There is:
#
#   - no RMS normalization
#   - no least-squares amplitude scaling
#   - no relative time shift
#   - no correlation-based polarity flip
#
# GP1 is used directly because its StationXML orientation is almost parallel
# to the local fiber direction near physical channel 1694 (dot ~ 0.996).
#
# Time axis:
#
#       t = absolute UTC sample time - NCEDC catalog origin
#
# so x=0 is exactly the earthquake catalog origin for BOTH instruments.
#
# Default display window is -0.5 ... 3.5 s, i.e. a 4-s window comparable to
# a 0...400 sample plot at 100 Hz, but expressed in physical seconds.

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
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, detrend, sosfiltfilt
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

EVENT_ID = "NC75336802"
ORIGIN = UTCDateTime("2026-04-01T04:57:57.470000Z")

DAS_FILE = Path(
    "/oak/stanford/groups/ettore88/data/SAFOD/SAFOD_events/"
    "SAFOD-Deep-10mGL-1000HzFs-2mChDualPulse_2026-04-01T045735Z.h5"
)

GEO_XLSX = Path(
    "/home/groups/ettore88/alina/SAFOD/"
    "SAFOD_Phase2_GeoReferenced_Channels.xlsx"
)

# Exact colocated DAS channel used for the final comparison.
DAS_CHANNEL = 1694

# 0 = exact ch.1694 only.
# Set to 2 if you intentionally want a 5-channel median (1692...1696).
DAS_MEDIAN_HALF_WINDOW = 0

NETWORK = "SF"
STATION = "MH029"
LOCATION = "01"
GEOPHONE_CHANNEL = "GP1"

# Common final bandwidth for BOTH measurements.
FMIN_HZ = 1.0
FMAX_HZ = 20.0
FILTER_ORDER = 12

# Wider response-removal prefilter for the geophone.
PREFILT_HZ = (0.2, 0.4, 30.0, 40.0)

# 5% Hann taper on each edge of the full processing trace.
EDGE_TAPER_FRACTION = 0.05

# Final display window.
PLOT_TMIN_S = -0.5
PLOT_TMAX_S = 2.0

# Optional correlation diagnostic window. Never used to modify the data.
CORR_TMIN_S = 0.0
CORR_TMAX_S = 1.0

DEFAULT_OUTPUT_DIR = Path(
    "results/real_event_20260401_75336802/"
    "das_geophone_physical_pairs"
)

DPI = 300


# ==============================================================================
# Generic helpers
# ==============================================================================

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


def detrend_demean(
    x: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    y = detrend(
        np.asarray(x, dtype=np.float64),
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
    """
    5% Hann-like taper on each end.

    scipy Tukey alpha is the TOTAL tapered fraction, so alpha=0.10 gives
    approximately 5% at the left edge and 5% at the right edge.
    """
    y = np.asarray(
        x,
        dtype=np.float64,
    )

    n = y.shape[axis]

    window = tukey(
        n,
        alpha=2.0 * EDGE_TAPER_FRACTION,
    )

    shape = [1] * y.ndim
    shape[axis] = n

    return (
        y
        * window.reshape(shape)
    )


def zero_phase_bandpass(
    x: np.ndarray,
    fs_hz: float,
    axis: int = -1,
) -> np.ndarray:
    nyquist = (
        0.5
        * float(fs_hz)
    )

    if not (
        0.0
        < FMIN_HZ
        < FMAX_HZ
        < nyquist
    ):
        raise ValueError(
            f"Invalid band {FMIN_HZ:g}-{FMAX_HZ:g} Hz "
            f"for fs={fs_hz:g} Hz."
        )

    sos = butter(
        FILTER_ORDER,
        [
            FMIN_HZ / nyquist,
            FMAX_HZ / nyquist,
        ],
        btype="bandpass",
        output="sos",
    )

    return sosfiltfilt(
        sos,
        np.asarray(
            x,
            dtype=np.float64,
        ),
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


def component_unit_vector_enu(
    azimuth_deg: float,
    dip_deg: float,
) -> np.ndarray:
    """
    StationXML convention:
      azimuth = clockwise from North
      dip     = degrees down from horizontal

    Returns [East, North, Up].
    """
    azimuth = np.deg2rad(
        float(azimuth_deg)
    )
    dip = np.deg2rad(
        float(dip_deg)
    )

    vector = np.array(
        [
            np.cos(dip) * np.sin(azimuth),
            np.cos(dip) * np.cos(azimuth),
            -np.sin(dip),
        ],
        dtype=np.float64,
    )

    return (
        vector
        / np.linalg.norm(vector)
    )


# ==============================================================================
# Local fiber direction
# ==============================================================================

def load_local_fiber_tangent_enu() -> tuple[np.ndarray, dict]:
    """
    Central-difference local fiber tangent at physical channel 1694.
    """
    if not GEO_XLSX.exists():
        raise FileNotFoundError(
            GEO_XLSX
        )

    df = pd.read_excel(
        GEO_XLSX
    )

    required = {
        "Channel",
        "UTM_E_m",
        "UTM_N_m",
        "TVD_m",
    }

    missing = (
        required
        .difference(
            df.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"Missing geometry columns: {sorted(missing)}"
        )

    channel_numeric = pd.to_numeric(
        df["Channel"],
        errors="coerce",
    )

    def get_row(channel: int):
        rows = df.loc[
            channel_numeric == channel
        ]

        if len(rows) != 1:
            raise RuntimeError(
                f"Expected one geometry row for channel {channel}; "
                f"found {len(rows)}."
            )

        return rows.iloc[0]

    before = get_row(
        DAS_CHANNEL - 1
    )
    center = get_row(
        DAS_CHANNEL
    )
    after = get_row(
        DAS_CHANNEL + 1
    )

    tangent = np.array(
        [
            float(after["UTM_E_m"])
            - float(before["UTM_E_m"]),

            float(after["UTM_N_m"])
            - float(before["UTM_N_m"]),

            -(
                float(after["TVD_m"])
                - float(before["TVD_m"])
            ),
        ],
        dtype=np.float64,
    )

    tangent /= np.linalg.norm(
        tangent
    )

    return tangent, {
        "tvd_m": float(
            center["TVD_m"]
        ),
    }


# ==============================================================================
# DAS: strain rate AND strain
# ==============================================================================

def load_das_observables() -> dict:
    """
    Read DAS once and return BOTH:
        axial strain rate [1/s]
        axial strain      [dimensionless]

    Processing:
        DASutils strain-rate conversion
        -> detrend/demean
        -> 5% edge taper

    Variant B:
        prepared strain rate -> 1-20 Hz zero-phase filter

    Variant A:
        prepared strain rate -> time integration -> detrend/demean
        -> 1-20 Hz zero-phase filter
    """
    if not DAS_FILE.exists():
        raise FileNotFoundError(
            DAS_FILE
        )

    das_data, info = DASutils.readFile_HDF(
        [str(DAS_FILE)],
        0.01,
        500.0,
        verbose=1,
        diff=True,
        detrend=False,
        tapering=False,
        filter=False,
        median=True,
        desampling=False,
        nChbuffer=1000,
        system="OptaSense",
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

    # --------------------------------------------------------------
    # True DAS time axis:
    # absolute UTC sample time - catalog origin.
    # --------------------------------------------------------------
    time_s = (
        float(
            beg_time
            - ORIGIN
        )
        + np.arange(
            das_data.shape[1],
            dtype=np.float64,
        )
        / fs_hz
    )

    # --------------------------------------------------------------
    # Select exact channel or an optional small local median.
    # --------------------------------------------------------------
    i0 = (
        DAS_CHANNEL
        - DAS_MEDIAN_HALF_WINDOW
    )
    i1 = (
        DAS_CHANNEL
        + DAS_MEDIAN_HALF_WINDOW
        + 1
    )

    if (
        i0 < 0
        or i1 > das_data.shape[0]
    ):
        raise RuntimeError(
            f"DAS channel window [{i0}:{i1}] outside "
            f"shape {das_data.shape}."
        )

    # Existing SAFOD project convention:
    #
    #     DASutils output * 1e3 -> nm/m/s
    #
    # Convert nm/m/s to SI strain rate [1/s]:
    #
    #     * 1e-9
    #
    strain_rate_si_channels = (
        das_data[
            i0:i1,
            :
        ]
        * 1.0e3
        * 1.0e-9
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

    # --------------------------------------------------------------
    # Variant B observable:
    # band-limited axial strain rate [1/s].
    # --------------------------------------------------------------
    strain_rate_filtered = zero_phase_bandpass(
        strain_rate_prepared,
        fs_hz,
    )

    # --------------------------------------------------------------
    # Variant A observable:
    # integrate the prepared strain rate FIRST, then apply the common
    # final bandpass to axial strain.
    # --------------------------------------------------------------
    strain = cumulative_trapezoid(
        strain_rate_prepared,
        dx=1.0 / fs_hz,
        initial=0.0,
    )

    strain = detrend_demean(
        strain
    )

    strain_filtered = zero_phase_bandpass(
        strain,
        fs_hz,
    )

    return {
        "time_s": time_s,
        "fs_hz": fs_hz,

        "strain_rate_1_per_s": (
            strain_rate_filtered
        ),

        "strain": (
            strain_filtered
        ),

        "beg_time": beg_time,

        "end_time": (
            beg_time
            + (
                das_data.shape[1] - 1
            )
            / fs_hz
        ),

        "channel_first": i0,
        "channel_last": i1 - 1,
    }


# ==============================================================================
# Geophone: velocity AND acceleration
# ==============================================================================

def load_geophone_observables(
    *,
    das_time_s: np.ndarray,
    das_fs_hz: float,
    fiber_tangent_enu: np.ndarray,
) -> dict:
    """
    Return BOTH:
        GP1 axial ground velocity     [m/s]
        GP1 axial ground acceleration [m/s^2]

    GP1 is response-corrected to velocity first.

    Variant A:
        response-corrected velocity -> common 1-20 Hz zero-phase filter

    Variant B:
        derivative of the SAME band-limited velocity -> acceleration

    No waveform fitting or time shifting is applied.
    """
    client = Client(
        "https://service.ncedc.org"
    )

    seed_id = (
        f"{NETWORK}.{STATION}.{LOCATION}.{GEOPHONE_CHANNEL}"
    )

    # Request margin beyond the full DAS interval for response removal.
    absolute_start = (
        ORIGIN
        + float(
            das_time_s[0]
        )
        - 20.0
    )

    absolute_end = (
        ORIGIN
        + float(
            das_time_s[-1]
        )
        + 20.0
    )

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
            f"No waveform/response available for {seed_id}."
        ) from exc

    stream.merge(
        method=1,
        fill_value="interpolate",
    )

    if len(stream) != 1:
        raise RuntimeError(
            f"Expected one merged {seed_id} trace; "
            f"found {len(stream)}."
        )

    tr = stream[0].copy()

    # --------------------------------------------------------------
    # Pre-response preprocessing.
    # --------------------------------------------------------------
    tr.data = detrend_demean(
        np.asarray(
            tr.data,
            dtype=np.float64,
        )
    )

    tr.data = edge_taper(
        tr.data
    )

    # --------------------------------------------------------------
    # Counts -> calibrated ground velocity [m/s].
    # --------------------------------------------------------------
    tr.remove_response(
        inventory=inventory,
        output="VEL",
        pre_filt=PREFILT_HZ,
        water_level=60.0,
        taper=False,
        zero_mean=False,
    )

    # --------------------------------------------------------------
    # Confirm GP1 alignment with local fiber axis.
    # --------------------------------------------------------------
    orientation = inventory.get_orientation(
        seed_id,
        datetime=ORIGIN,
    )

    gp1_axis = component_unit_vector_enu(
        orientation["azimuth"],
        orientation["dip"],
    )

    dot_fiber = float(
        np.dot(
            gp1_axis,
            fiber_tangent_enu,
        )
    )

    fiber_axis_angle_deg = float(
        np.degrees(
            np.arccos(
                np.clip(
                    abs(dot_fiber),
                    0.0,
                    1.0,
                )
            )
        )
    )

    # --------------------------------------------------------------
    # Independent geophone time axis:
    # absolute UTC sample time - SAME catalog origin.
    # --------------------------------------------------------------
    geo_time_s = (
        float(
            tr.stats.starttime
            - ORIGIN
        )
        + np.arange(
            tr.stats.npts,
            dtype=np.float64,
        )
        / float(
            tr.stats.sampling_rate
        )
    )

    if (
        das_time_s[0]
        < geo_time_s[0]
        or das_time_s[-1]
        > geo_time_s[-1]
    ):
        raise RuntimeError(
            f"{seed_id} does not cover the complete DAS UTC interval."
        )

    # --------------------------------------------------------------
    # Interpolate calibrated velocity onto the exact DAS time grid.
    # --------------------------------------------------------------
    velocity = np.interp(
        das_time_s,
        geo_time_s,
        np.asarray(
            tr.data,
            dtype=np.float64,
        ),
    )

    # Put positive GP1 in the same geometric direction as the chosen
    # local fiber tangent. This is coordinate orientation, NOT data fitting.
    if dot_fiber < 0.0:
        velocity = -velocity
        dot_fiber = -dot_fiber

    # --------------------------------------------------------------
    # Variant A observable:
    # band-limited axial velocity [m/s].
    # --------------------------------------------------------------
    velocity_filtered = zero_phase_bandpass(
        velocity,
        das_fs_hz,
    )

    # --------------------------------------------------------------
    # Variant B observable:
    # derivative of the band-limited velocity [m/s^2].
    # --------------------------------------------------------------
    acceleration = np.gradient(
        velocity_filtered,
        1.0 / das_fs_hz,
    )

    return {
        "velocity_m_per_s": (
            velocity_filtered
        ),

        "acceleration_m_per_s2": (
            acceleration
        ),

        "seed_id": seed_id,

        "azimuth_deg": float(
            orientation["azimuth"]
        ),

        "dip_deg": float(
            orientation["dip"]
        ),

        "dot_fiber": float(
            dot_fiber
        ),

        "fiber_axis_angle_deg": (
            fiber_axis_angle_deg
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
        labelsize=12,
        width=1.1,
        length=4,
    )

    ax_right.tick_params(
        axis="y",
        labelsize=12,
        width=1.1,
        length=4,
    )

    for label in (
        ax_left.get_xticklabels()
        + ax_left.get_yticklabels()
        + ax_right.get_yticklabels()
    ):
        label.set_fontweight(
            "bold"
        )

    for spine in ax_left.spines.values():
        spine.set_linewidth(
            1.1
        )

    for spine in ax_right.spines.values():
        spine.set_linewidth(
            1.1
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
# Variant A: strain vs velocity
# ==============================================================================

def plot_strain_vs_velocity(
    *,
    time_s: np.ndarray,
    strain: np.ndarray,
    velocity: np.ndarray,
    output_path: Path,
) -> None:
    mask = (
        (time_s >= PLOT_TMIN_S)
        & (time_s <= PLOT_TMAX_S)
    )

    t = time_s[mask]
    das = strain[mask]
    geo = velocity[mask]

    fig, ax1 = plt.subplots(
        figsize=(11.5, 4.8),
    )

    line_das, = ax1.plot(
        t,
        das,
        linewidth=1.55,
        label="DAS",
    )

    ax1.set_xlabel(
        "Time from catalog origin [s]",
        fontsize=15,
        fontweight="bold",
    )

    ax1.set_ylabel(
        "Axial strain",
        fontsize=15,
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
        PLOT_TMIN_S,
        PLOT_TMAX_S,
    )

    ax2 = ax1.twinx()

    line_geo, = ax2.plot(
        t,
        geo,
        color="black",
        linewidth=1.35,
        label="MH029 GP1 axial velocity",
    )

    ax2.set_ylabel(
        "Axial velocity [m/s]",
        fontsize=15,
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
        fontsize=11,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


# ==============================================================================
# Variant B: strain rate vs acceleration
# ==============================================================================

def plot_strain_rate_vs_acceleration(
    *,
    time_s: np.ndarray,
    strain_rate: np.ndarray,
    acceleration: np.ndarray,
    output_path: Path,
) -> None:
    mask = (
        (time_s >= PLOT_TMIN_S)
        & (time_s <= PLOT_TMAX_S)
    )

    t = time_s[mask]
    das = strain_rate[mask]
    geo = acceleration[mask]

    fig, ax1 = plt.subplots(
        figsize=(11.5, 4.8),
    )

    line_das, = ax1.plot(
        t,
        das,
        linewidth=1.55,
        label="DAS",
    )

    ax1.set_xlabel(
        "Time from catalog origin [s]",
        fontsize=15,
        fontweight="bold",
    )

    ax1.set_ylabel(
        r"Axial strain rate [s$^{-1}$]",
        fontsize=15,
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
        PLOT_TMIN_S,
        PLOT_TMAX_S,
    )

    ax2 = ax1.twinx()

    line_geo, = ax2.plot(
        t,
        geo,
        color="black",
        linewidth=1.35,
        label="MH029 GP1",
    )

    ax2.set_ylabel(
        r"Ground acceleration [m/s$^2$]",
        fontsize=15,
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
        fontsize=11,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


# ==============================================================================
# Main
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the two physically consistent SAFOD DAS / MH029 GP1 "
            "comparison figures."
        )
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

    fiber_tangent, fiber_meta = (
        load_local_fiber_tangent_enu()
    )

    das = load_das_observables()

    geo = load_geophone_observables(
        das_time_s=das["time_s"],
        das_fs_hz=das["fs_hz"],
        fiber_tangent_enu=fiber_tangent,
    )

    # --------------------------------------------------------------------------
    # Optional diagnostics only; nothing is shifted/scaled/flipped from these.
    # --------------------------------------------------------------------------
    corr_mask = (
        (das["time_s"] >= CORR_TMIN_S)
        & (das["time_s"] <= CORR_TMAX_S)
    )

    corr_strain_velocity = zero_lag_corr(
        das["strain"][corr_mask],
        geo["velocity_m_per_s"][corr_mask],
    )

    corr_rate_acceleration = zero_lag_corr(
        das["strain_rate_1_per_s"][corr_mask],
        geo["acceleration_m_per_s2"][corr_mask],
    )

    # --------------------------------------------------------------------------
    # Figure 1.
    # --------------------------------------------------------------------------
    fig1_path = (
        args.output_dir
        / "01_das_strain_vs_mh029_gp1_velocity.png"
    )

    plot_strain_vs_velocity(
        time_s=das["time_s"],
        strain=das["strain"],
        velocity=geo["velocity_m_per_s"],
        output_path=fig1_path,
    )

    # --------------------------------------------------------------------------
    # Figure 2.
    # --------------------------------------------------------------------------
    fig2_path = (
        args.output_dir
        / "02_das_strain_rate_vs_mh029_gp1_acceleration.png"
    )

    plot_strain_rate_vs_acceleration(
        time_s=das["time_s"],
        strain_rate=das["strain_rate_1_per_s"],
        acceleration=geo["acceleration_m_per_s2"],
        output_path=fig2_path,
    )

    # --------------------------------------------------------------------------
    # Save both physical observables.
    # --------------------------------------------------------------------------
    npz_path = (
        args.output_dir
        / "das_geophone_physical_pairs.npz"
    )

    np.savez_compressed(
        npz_path,

        event_id=np.array(
            EVENT_ID
        ),

        catalog_origin=np.array(
            str(ORIGIN)
        ),

        time_from_catalog_origin_s=(
            das["time_s"]
        ),

        das_axial_strain=(
            das["strain"]
        ),

        das_axial_strain_rate_1_per_s=(
            das["strain_rate_1_per_s"]
        ),

        gp1_axial_velocity_m_per_s=(
            geo["velocity_m_per_s"]
        ),

        gp1_axial_acceleration_m_per_s2=(
            geo["acceleration_m_per_s2"]
        ),

        das_channel_center=np.array(
            DAS_CHANNEL
        ),

        das_channel_first=np.array(
            das["channel_first"]
        ),

        das_channel_last=np.array(
            das["channel_last"]
        ),

        fiber_tangent_enu=(
            fiber_tangent
        ),

        fiber_tvd_m=np.array(
            fiber_meta["tvd_m"]
        ),

        gp1_seed_id=np.array(
            geo["seed_id"]
        ),

        gp1_azimuth_deg=np.array(
            geo["azimuth_deg"]
        ),

        gp1_dip_deg=np.array(
            geo["dip_deg"]
        ),

        gp1_dot_fiber=np.array(
            geo["dot_fiber"]
        ),

        gp1_fiber_axis_angle_deg=np.array(
            geo["fiber_axis_angle_deg"]
        ),

        common_fmin_hz=np.array(
            FMIN_HZ
        ),

        common_fmax_hz=np.array(
            FMAX_HZ
        ),

        zero_lag_corr_strain_velocity=np.array(
            corr_strain_velocity
        ),

        zero_lag_corr_strain_rate_acceleration=np.array(
            corr_rate_acceleration
        ),
    )

    print()
    print("FINAL physical DAS / geophone comparisons")
    print("==========================================")
    print(f"event                        : {EVENT_ID}")
    print(f"catalog origin               : {ORIGIN}")
    print(f"DAS absolute start           : {das['beg_time']}")
    print(f"DAS absolute end             : {das['end_time']}")
    print(
        f"origin in DAS file           : "
        f"{-float(das['time_s'][0]):.6f} s after file start"
    )

    if DAS_MEDIAN_HALF_WINDOW == 0:
        print(
            f"DAS channel                  : "
            f"{DAS_CHANNEL}"
        )
    else:
        print(
            f"DAS median channels          : "
            f"{das['channel_first']}..{das['channel_last']}"
        )

    print(
        f"fiber tangent ENU            : "
        f"[{fiber_tangent[0]:+.6f}, "
        f"{fiber_tangent[1]:+.6f}, "
        f"{fiber_tangent[2]:+.6f}]"
    )

    print(
        f"fiber TVD                    : "
        f"{fiber_meta['tvd_m']:.2f} m"
    )

    print(
        f"geophone                     : "
        f"{geo['seed_id']}"
    )

    print(
        f"GP1 orientation              : "
        f"az={geo['azimuth_deg']:+.2f} deg, "
        f"dip={geo['dip_deg']:+.2f} deg"
    )

    print(
        f"GP1 dot(fiber)               : "
        f"{geo['dot_fiber']:+.6f}"
    )

    print(
        f"GP1-fiber axis angle         : "
        f"{geo['fiber_axis_angle_deg']:.3f} deg"
    )

    print(
        f"common filter                : "
        f"{FMIN_HZ:g}-{FMAX_HZ:g} Hz, zero phase"
    )

    print(
        f"corr(strain, velocity)       : "
        f"{corr_strain_velocity:+.4f}"
    )

    print(
        f"corr(strain rate, accel.)    : "
        f"{corr_rate_acceleration:+.4f}"
    )

    print(
        "x-axis                       : "
        "absolute UTC sample time - NCEDC catalog origin"
    )

    print("relative time shift           : NONE")
    print("normalization                 : NONE")
    print("amplitude fitting             : NONE")
    print("correlation-based sign flip   : NONE")

    print()
    print("Saved")
    print("=====")
    print(fig1_path)
    print(fig2_path)
    print(npz_path)


if __name__ == "__main__":
    main()