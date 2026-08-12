# scripts/safod/plot_das_geophone_two_physical_comparisons.py

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

EVENT_ID = "NC75336802"

ORIGIN = UTCDateTime(
    "2026-04-01T04:57:57.470000Z"
)

DAS_FILE = Path(
    "/oak/stanford/groups/ettore88/data/SAFOD/SAFOD_events/"
    "SAFOD-Deep-10mGL-1000HzFs-2mChDualPulse_2026-04-01T045735Z.h5"
)

GEO_XLSX = Path(
    "/home/groups/ettore88/alina/SAFOD/"
    "SAFOD_Phase2_GeoReferenced_Channels.xlsx"
)

DAS_CHANNEL = 1694
DAS_MEDIAN_HALF_WINDOW = 0

NETWORK = "SF"
STATION = "MH029"
LOCATION = "01"
GEOPHONE_CHANNEL = "GP1"


# ==============================================================================
# Processing
# ==============================================================================

FMIN_HZ = 1.0
FMAX_HZ = 20.0

# Odd number -> Type-I linear-phase FIR.
#
# Group delay:
#
#     (N - 1) / (2 fs)
#
# At fs=1000 Hz:
#
#     (401 - 1)/(2*1000) = 0.200 s
#
FIR_NUMTAPS = 401

PREFILT_HZ = (
    0.2,
    0.4,
    30.0,
    40.0,
)

EDGE_TAPER_FRACTION = 0.05

PLOT_TMIN_S = 0.0
PLOT_TMAX_S = 1.0

CORR_TMIN_S = 0.0
CORR_TMAX_S = 1.0

DEFAULT_OUTPUT_DIR = Path(
    "results/real_event_20260401_75336802/"
    "das_geophone_physical_pairs"
)

DPI = 300

# Poster plotting style
PLOT_FIGSIZE = (11.2, 3.1)
DAS_COLOR = "gray"  #"#8C1515"  # Stanford Cardinal
DAS_LINEWIDTH = 2.0
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
    axis: int = -1,
) -> np.ndarray:
    """
    Causal linear-phase FIR bandpass.

    Important properties:

      * one forward pass only
      * no filtfilt
      * therefore no backward/acausal filtering
      * linear phase
      * constant group delay

    The SAME filter is applied to DAS and MH029.
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
        < FMIN_HZ
        < FMAX_HZ
        < nyquist
    ):
        raise ValueError(
            f"Invalid band "
            f"{FMIN_HZ:g}-{FMAX_HZ:g} Hz "
            f"for fs={fs_hz:g} Hz."
        )

    taps = firwin(
        FIR_NUMTAPS,
        [
            FMIN_HZ,
            FMAX_HZ,
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


def component_unit_vector_enu(
    azimuth_deg: float,
    dip_deg: float,
) -> np.ndarray:

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
            f"Missing geometry columns: "
            f"{sorted(missing)}"
        )

    channel_numeric = pd.to_numeric(
        df["Channel"],
        errors="coerce",
    )

    def get_row(channel: int):
        rows = df.loc[
            channel_numeric
            == channel
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

    return (
        tangent,
        {
            "tvd_m": float(
                center["TVD_m"]
            ),
        },
    )


# ==============================================================================
# DAS
# ==============================================================================

def load_das_observables() -> dict:

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
            f"DAS channel window "
            f"[{i0}:{i1}] outside "
            f"shape {das_data.shape}."
        )

    # ------------------------------------------------------------------
    # Existing SAFOD DAS conversion.
    #
    # DASutils output * 1e3 -> nm/m/s
    #
    # nm/m/s -> m/m/s = 1/s:
    #
    #     * 1e-9
    #
    # Hence combined factor = 1e-6.
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Causal, linear-phase filtering.
    # ------------------------------------------------------------------

    strain_rate_filtered_si = (
        causal_linear_phase_bandpass(
            strain_rate_prepared,
            fs_hz,
        )
    )

    # ------------------------------------------------------------------
    # [1/s] = [m/m/s] -> [um/m/s]
    #
    # 1 m = 1e6 um.
    #
    # Therefore:
    #
    #     1e-6 /s = 1 um/m/s
    # ------------------------------------------------------------------

    strain_rate_um_per_m_per_s = (
        strain_rate_filtered_si
        * 1.0e6
    )

    # ------------------------------------------------------------------
    # Strain.
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
        )
    )

    return {
        "time_s": (
            time_s
        ),

        "fs_hz": (
            fs_hz
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
    das_time_s: np.ndarray,
    das_fs_hz: float,
    fiber_tangent_enu: np.ndarray,
) -> dict:

    client = Client(
        "https://service.ncedc.org"
    )

    seed_id = (
        f"{NETWORK}."
        f"{STATION}."
        f"{LOCATION}."
        f"{GEOPHONE_CHANNEL}"
    )

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
            f"No waveform/response available "
            f"for {seed_id}."
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

    tr.data = detrend_demean(
        np.asarray(
            tr.data,
            dtype=np.float64,
        )
    )

    tr.data = edge_taper(
        tr.data
    )

    # ------------------------------------------------------------------
    # Counts -> physical ground velocity [m/s].
    # ------------------------------------------------------------------

    tr.remove_response(
        inventory=inventory,
        output="VEL",
        pre_filt=PREFILT_HZ,
        water_level=60.0,
        taper=False,
        zero_mean=False,
    )

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
            f"{seed_id} does not cover "
            "the complete DAS UTC interval."
        )

    velocity = np.interp(
        das_time_s,
        geo_time_s,
        np.asarray(
            tr.data,
            dtype=np.float64,
        ),
    )

    # Same geometric direction as the local fiber.
    # NOT a correlation-based polarity change.
    if dot_fiber < 0.0:

        velocity = (
            -velocity
        )

        dot_fiber = (
            -dot_fiber
        )

    # ------------------------------------------------------------------
    # EXACT SAME causal linear-phase filter as DAS.
    # ------------------------------------------------------------------

    velocity_filtered = (
        causal_linear_phase_bandpass(
            velocity,
            das_fs_hz,
        )
    )

    # ------------------------------------------------------------------
    # Velocity -> acceleration.
    # ------------------------------------------------------------------

    acceleration_m_per_s2 = np.gradient(
        velocity_filtered,
        (
            1.0
            / das_fs_hz
        ),
    )

    # ------------------------------------------------------------------
    # m/s^2 -> cm/s^2
    # ------------------------------------------------------------------

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

        "seed_id": (
            seed_id
        ),

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
# Figure 1
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
        figsize=PLOT_FIGSIZE,
    )

    line_das, = ax1.plot(
        t,
        das,
        color=DAS_COLOR,
        linewidth=DAS_LINEWIDTH,
        label=f"DAS channel {DAS_CHANNEL}",
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
        PLOT_TMIN_S,
        PLOT_TMAX_S,
    )

    ax2 = ax1.twinx()

    line_geo, = ax2.plot(
        t,
        geo,
        color="#940025",
        linewidth=GEOPHONE_LINEWIDTH,
        label="MH029 GP1 axial velocity",
    )

    ax2.set_ylabel(
        "Axial velocity\n[m/s]",
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

    fig.tight_layout(pad=0.5)

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


# ==============================================================================
# Figure 2
# ==============================================================================

def plot_strain_rate_vs_acceleration(
    *,
    time_s: np.ndarray,
    strain_rate_um_per_m_per_s: np.ndarray,
    acceleration_cm_per_s2: np.ndarray,
    output_path: Path,
) -> None:

    mask = (
        (time_s >= PLOT_TMIN_S)
        & (time_s <= PLOT_TMAX_S)
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
        label=f"DAS channel {DAS_CHANNEL}",
    )

    ax1.set_xlabel(
        "Time from catalog origin [s]",
        fontsize=13,
        fontweight="bold",
    )

    ax1.set_ylabel(
        "Strain rate\n" + r"[$\mu$m/m/s]",
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
        PLOT_TMIN_S,
        PLOT_TMAX_S,
    )

    ax2 = ax1.twinx()

    line_geo, = ax2.plot(
        t,
        geo,
        color="#940025",
        linewidth=GEOPHONE_LINEWIDTH,
        label="MH029 GP1",
    )

    ax2.set_ylabel(
        "Axial ground acceleration\n" + r"[cm/s$^2$]",
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

    fig.tight_layout(pad=0.5)

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)


# ==============================================================================
# Arguments
# ==============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Generate SAFOD DAS / MH029 GP1 "
            "physical comparison figures."
        )
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

    fiber_tangent, fiber_meta = (
        load_local_fiber_tangent_enu()
    )

    das = load_das_observables()

    geo = load_geophone_observables(
        das_time_s=(
            das["time_s"]
        ),
        das_fs_hz=(
            das["fs_hz"]
        ),
        fiber_tangent_enu=(
            fiber_tangent
        ),
    )

    # ------------------------------------------------------------------
    # Diagnostic correlation only.
    # ------------------------------------------------------------------

    corr_mask = (
        (das["time_s"] >= CORR_TMIN_S)
        & (das["time_s"] <= CORR_TMAX_S)
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

    fig1_path = (
        args.output_dir
        / "01_das_strain_vs_mh029_gp1_velocity.png"
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
        output_path=(
            fig1_path
        ),
    )

    fig2_path = (
        args.output_dir
        / "02_das_strain_rate_vs_mh029_gp1_acceleration.png"
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
        output_path=(
            fig2_path
        ),
    )

    # ------------------------------------------------------------------
    # Processing delay.
    # ------------------------------------------------------------------

    fir_group_delay_s = (
        (FIR_NUMTAPS - 1)
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
            das[
                "strain_rate_1_per_s"
            ]
        ),

        das_axial_strain_rate_um_per_m_per_s=(
            das[
                "strain_rate_um_per_m_per_s"
            ]
        ),

        gp1_axial_velocity_m_per_s=(
            geo[
                "velocity_m_per_s"
            ]
        ),

        gp1_axial_acceleration_m_per_s2=(
            geo[
                "acceleration_m_per_s2"
            ]
        ),

        gp1_axial_acceleration_cm_per_s2=(
            geo[
                "acceleration_cm_per_s2"
            ]
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
            geo[
                "fiber_axis_angle_deg"
            ]
        ),

        common_fmin_hz=np.array(
            FMIN_HZ
        ),

        common_fmax_hz=np.array(
            FMAX_HZ
        ),

        fir_numtaps=np.array(
            FIR_NUMTAPS
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
        "FINAL physical DAS / geophone comparisons"
    )
    print(
        "=========================================="
    )

    print(
        f"event                        : "
        f"{EVENT_ID}"
    )

    print(
        f"catalog origin               : "
        f"{ORIGIN}"
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
        f"DAS channel                  : "
        f"{DAS_CHANNEL}"
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
        f"{FMIN_HZ:g}-{FMAX_HZ:g} Hz"
    )

    print(
        f"filter                       : "
        f"causal linear-phase FIR, "
        f"{FIR_NUMTAPS} taps"
    )

    print(
        f"FIR group delay              : "
        f"{fir_group_delay_s:.3f} s"
    )

    print(
        "zero-phase filtering          : "
        "NO"
    )

    print(
        "manual time shift             : "
        "NONE"
    )

    print(
        "DAS strain-rate units         : "
        "um/m/s"
    )

    print(
        "geophone acceleration units   : "
        "cm/s^2"
    )

    print(
        f"corr(strain, velocity)        : "
        f"{corr_strain_velocity:+.4f}"
    )

    print(
        f"corr(strain rate, accel.)     : "
        f"{corr_rate_acceleration:+.4f}"
    )

    print(
        "relative DAS/MH029 shift      : "
        "NONE"
    )

    print(
        "normalization                  : "
        "NONE"
    )

    print(
        "amplitude fitting              : "
        "NONE"
    )

    print()
    print("Saved")
    print("=====")

    print(fig1_path)
    print(fig2_path)
    print(npz_path)


if __name__ == "__main__":
    main()