# scripts/safod/plot_das_geophone_final.py
#
# Final SAFOD DAS / borehole-geophone waveform comparison for NC75336802.
#
# Reference processing logic:
# Lindsey, Rademacher & Ajo-Franklin (2020),
# "On the Broadband Instrument Response of Fiber-Optic DAS Arrays",
# JGR Solid Earth, doi:10.1029/2019JB018145.
#
# Final comparison:
#   DAS axial strain
#       DASutils strain rate
#       -> linear detrend + demean
#       -> 5% Hann edge taper
#       -> integrate in time to strain
#       -> detrend + demean
#       -> zero-phase 1-20 Hz bandpass
#
#   Borehole geophone axial velocity
#       SF.MH029.01.GP1
#       -> linear detrend + demean
#       -> 5% Hann edge taper
#       -> remove instrument response to ground velocity [m/s]
#       -> zero-phase 1-20 Hz bandpass
#
# GP1 is used directly because its StationXML axis is essentially parallel to
# the local SAFOD fiber direction near physical channel 1694
# (dot product ~0.996 for the April-1 event). This avoids an unnecessarily
# ill-conditioned 3C reconstruction.
#
# Plot:
#   both traces are independently RMS-normalized for waveform comparison.
#   no relative time shift is applied.
#
# Time axis:
#   t = absolute UTC sample time - NCEDC catalog origin time.
#   Therefore x=0 is exactly 2026-04-01T04:57:57.470000Z for BOTH instruments.

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
# Configuration
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

DAS_CHANNEL = 1694

NETWORK = "SF"
STATION = "MH029"
LOCATION = "01"
GEOPHONE_CHANNEL = "GP1"

# Common final filter.
FMIN_HZ = 1.0
FMAX_HZ = 20.0
FILTER_ORDER = 4

# Wider prefilter only for instrument-response removal.
PREFILT_HZ = (0.2, 0.4, 30.0, 40.0)

# 5% taper on each edge.
EDGE_TAPER_FRACTION = 0.05

# Final clean figure window.
PLOT_TMIN_S = -0.10
PLOT_TMAX_S = 1.20

# Same interval used for independent RMS normalization and diagnostics.
COMPARE_TMIN_S = 0.05
COMPARE_TMAX_S = 0.80

# Diagnostic lag only; NEVER applied.
MAX_DIAGNOSTIC_LAG_S = 0.10

DEFAULT_OUTPUT_DIR = Path(
    "results/real_event_20260401_75336802/"
    "das_geophone_final"
)

DPI = 300


# ==============================================================================
# Generic utilities
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


def detrend_demean(trace: np.ndarray) -> np.ndarray:
    x = detrend(
        np.asarray(trace, dtype=np.float64),
        type="linear",
    )
    return x - np.mean(x)


def five_percent_hann_taper(trace: np.ndarray) -> np.ndarray:
    """
    Symmetric 5% edge taper on each side.

    scipy Tukey alpha is the TOTAL tapered fraction, so alpha=0.10 gives
    approximately 5% Hann taper at each edge.
    """
    x = np.asarray(trace, dtype=np.float64)

    window = tukey(
        x.size,
        alpha=2.0 * EDGE_TAPER_FRACTION,
    )

    return x * window


def bandpass(trace: np.ndarray, fs_hz: float) -> np.ndarray:
    nyquist = 0.5 * float(fs_hz)

    if not 0.0 < FMIN_HZ < FMAX_HZ < nyquist:
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
        np.asarray(trace, dtype=np.float64),
    )


def zero_lag_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)

    x = x - np.mean(x)
    y = y - np.mean(y)

    denom = np.linalg.norm(x) * np.linalg.norm(y)

    if denom <= 0.0:
        return np.nan

    return float(np.dot(x, y) / denom)


def rms_normalize(trace: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x = np.asarray(trace, dtype=np.float64)

    scale = float(
        np.sqrt(
            np.mean(
                x[mask] ** 2
            )
        )
    )

    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError(
            "Cannot RMS-normalize an invalid/zero waveform."
        )

    return x / scale


def diagnostic_best_lag(
    a: np.ndarray,
    b: np.ndarray,
    dt_s: float,
    mask: np.ndarray,
) -> tuple[float, float]:
    """
    Diagnostic only. No shift is ever applied.

    Positive returned lag means b would have to move later to best match a.
    """
    x = np.asarray(a[mask], dtype=np.float64)
    y = np.asarray(b[mask], dtype=np.float64)

    x = x - np.mean(x)
    y = y - np.mean(y)

    max_lag_samples = int(
        round(
            MAX_DIAGNOSTIC_LAG_S / dt_s
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
            xx = x[lag:]
            yy = y[:-lag]
        elif lag < 0:
            xx = x[:lag]
            yy = y[-lag:]
        else:
            xx = x
            yy = y

        if xx.size < 20:
            continue

        corr = zero_lag_corr(xx, yy)

        if np.isfinite(corr) and abs(corr) > best_abs_corr:
            best_abs_corr = abs(corr)
            best_corr = corr
            best_lag_samples = lag

    return (
        best_lag_samples * dt_s,
        float(best_corr),
    )


# ==============================================================================
# Local fiber geometry
# ==============================================================================

def component_unit_vector_enu(
    azimuth_deg: float,
    dip_deg: float,
) -> np.ndarray:
    """
    StationXML convention:
      azimuth = clockwise from North
      dip     = degrees down from horizontal

    Return [East, North, Up].
    """
    az = np.deg2rad(
        float(azimuth_deg)
    )
    dip = np.deg2rad(
        float(dip_deg)
    )

    vec = np.array(
        [
            np.cos(dip) * np.sin(az),
            np.cos(dip) * np.cos(az),
            -np.sin(dip),
        ],
        dtype=np.float64,
    )

    return vec / np.linalg.norm(vec)


def load_local_fiber_tangent_enu() -> tuple[np.ndarray, dict]:
    """
    Central-difference fiber tangent at physical channel 1694.
    """
    if not GEO_XLSX.exists():
        raise FileNotFoundError(GEO_XLSX)

    df = pd.read_excel(GEO_XLSX)

    required = {
        "Channel",
        "UTM_E_m",
        "UTM_N_m",
        "TVD_m",
    }

    missing = required.difference(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing geometry columns: {sorted(missing)}"
        )

    channel_numeric = pd.to_numeric(
        df["Channel"],
        errors="coerce",
    )

    def row_for(channel: int):
        rows = df.loc[
            channel_numeric == channel
        ]

        if len(rows) != 1:
            raise RuntimeError(
                f"Expected one geometry row for channel {channel}; "
                f"found {len(rows)}."
            )

        return rows.iloc[0]

    before = row_for(
        DAS_CHANNEL - 1
    )
    center = row_for(
        DAS_CHANNEL
    )
    after = row_for(
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
        "tvd_m": float(center["TVD_m"]),
        "utm_e_m": float(center["UTM_E_m"]),
        "utm_n_m": float(center["UTM_N_m"]),
    }


# ==============================================================================
# DAS
# ==============================================================================

def load_das_strain() -> dict:
    """
    Read April-1 DAS with the same DASutils workflow used in the project.
    """
    if not DAS_FILE.exists():
        raise FileNotFoundError(DAS_FILE)

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

    if (
        DAS_CHANNEL < 0
        or DAS_CHANNEL >= das_data.shape[0]
    ):
        raise RuntimeError(
            f"DAS row {DAS_CHANNEL} is outside shape "
            f"{das_data.shape}."
        )

    # Definitive DAS time axis:
    # absolute UTC sample time - catalog origin.
    time_s = (
        float(
            beg_time - ORIGIN
        )
        + np.arange(
            das_data.shape[1],
            dtype=np.float64,
        )
        / fs_hz
    )

    # Existing SAFOD convention:
    # DASutils output * 1e3 -> strain rate [nm/m/s].
    # nm/m/s -> SI strain rate [1/s] gives another 1e-9.
    strain_rate_si = (
        das_data[
            DAS_CHANNEL,
            :
        ]
        * 1.0e3
        * 1.0e-9
    )

    strain_rate_si = detrend_demean(
        strain_rate_si
    )

    strain_rate_si = five_percent_hann_taper(
        strain_rate_si
    )

    strain = cumulative_trapezoid(
        strain_rate_si,
        dx=1.0 / fs_hz,
        initial=0.0,
    )

    # Remove integration drift before the common final bandpass.
    strain = detrend_demean(
        strain
    )

    strain = bandpass(
        strain,
        fs_hz,
    )

    return {
        "time_s": time_s,
        "strain": strain,
        "fs_hz": fs_hz,
        "beg_time": beg_time,
        "end_time": (
            beg_time
            + (
                das_data.shape[1] - 1
            )
            / fs_hz
        ),
    }


# ==============================================================================
# Borehole geophone GP1
# ==============================================================================

def load_gp1_velocity(
    *,
    das_time_s: np.ndarray,
    das_fs_hz: float,
    fiber_tangent_enu: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    Load response-corrected GP1 ground velocity.

    GP1 is already nearly parallel to the local fiber axis, so it is used
    directly rather than reconstructing ENU from the poorly conditioned
    GP1/GP2/GP3 orientation matrix.
    """
    client = Client(
        "https://service.ncedc.org"
    )

    # Request extra margin for stable response removal.
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

    seed_id = (
        f"{NETWORK}.{STATION}.{LOCATION}.{GEOPHONE_CHANNEL}"
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

    # Same edge preparation as DAS before response removal.
    tr.data = detrend_demean(
        np.asarray(
            tr.data,
            dtype=np.float64,
        )
    )

    tr.data = five_percent_hann_taper(
        tr.data
    )

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

    axis_angle_deg = float(
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

    # Independent geophone time axis:
    # absolute UTC sample time - the SAME catalog origin.
    geo_time_s = (
        float(
            tr.stats.starttime - ORIGIN
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
        das_time_s[0] < geo_time_s[0]
        or das_time_s[-1] > geo_time_s[-1]
    ):
        raise RuntimeError(
            f"{seed_id} does not cover the full DAS UTC interval."
        )

    velocity = np.interp(
        das_time_s,
        geo_time_s,
        np.asarray(
            tr.data,
            dtype=np.float64,
        ),
    )

    # Put GP1 positive axis in the same geometric direction as the local fiber.
    if dot_fiber < 0.0:
        velocity = -velocity
        dot_fiber = -dot_fiber

    velocity = bandpass(
        velocity,
        das_fs_hz,
    )

    return velocity, {
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
        "axis_angle_deg": axis_angle_deg,
        "native_start": tr.stats.starttime,
        "native_end": tr.stats.endtime,
    }


# ==============================================================================
# Clean final figure
# ==============================================================================

def make_figure(
    *,
    time_s: np.ndarray,
    das_strain: np.ndarray,
    gp1_velocity: np.ndarray,
    output_path: Path,
) -> None:
    plot_mask = (
        (time_s >= PLOT_TMIN_S)
        & (time_s <= PLOT_TMAX_S)
    )

    compare_mask = (
        (time_s >= COMPARE_TMIN_S)
        & (time_s <= COMPARE_TMAX_S)
    )

    das_normalized = rms_normalize(
        das_strain,
        compare_mask,
    )

    gp1_normalized = rms_normalize(
        gp1_velocity,
        compare_mask,
    )

    # For visualization only, align the global polarity of the two different
    # observables. No time shift and no amplitude fitting are applied.
    if zero_lag_corr(
        das_normalized[compare_mask],
        gp1_normalized[compare_mask],
    ) < 0.0:
        gp1_normalized = -gp1_normalized

    fig, ax = plt.subplots(
        figsize=(10.5, 4.4),
    )

    ax.plot(
        time_s[plot_mask],
        das_normalized[plot_mask],
        color="black",
        linewidth=1.55,
        label="DAS axial strain",
    )

    ax.plot(
        time_s[plot_mask],
        gp1_normalized[plot_mask],
        color="0.52",
        linestyle="--",
        linewidth=1.55,
        label="MH029 GP1 axial velocity",
    )

    ax.axvline(
        0.0,
        color="0.55",
        linestyle=":",
        linewidth=0.9,
    )

    ax.axhline(
        0.0,
        color="0.82",
        linewidth=0.7,
    )

    ax.set_xlim(
        PLOT_TMIN_S,
        PLOT_TMAX_S,
    )

    ax.set_xlabel(
        "Time from catalog origin [s]",
        fontsize=15,
        fontweight="bold",
    )

    ax.set_ylabel(
        "Normalized amplitude",
        fontsize=15,
        fontweight="bold",
    )

    ax.legend(
        loc="upper right",
        fontsize=11,
        frameon=False,
    )

    ax.tick_params(
        axis="both",
        labelsize=12,
        width=1.1,
        length=4,
    )

    for label in (
        ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        label.set_fontweight(
            "bold"
        )

    for spine in ax.spines.values():
        spine.set_linewidth(
            1.1
        )

    fig.tight_layout()

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
# Main
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Final April-1 SAFOD DAS strain / borehole GP1 velocity "
            "waveform validation plot."
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

    fiber_tangent, fiber_metadata = (
        load_local_fiber_tangent_enu()
    )

    das = load_das_strain()

    gp1_velocity, gp1_metadata = (
        load_gp1_velocity(
            das_time_s=das["time_s"],
            das_fs_hz=das["fs_hz"],
            fiber_tangent_enu=fiber_tangent,
        )
    )

    compare_mask = (
        (das["time_s"] >= COMPARE_TMIN_S)
        & (das["time_s"] <= COMPARE_TMAX_S)
    )

    corr_native = zero_lag_corr(
        das["strain"][compare_mask],
        gp1_velocity[compare_mask],
    )

    # Diagnostic waveform polarity for direct shape comparison.
    display_sign = (
        -1.0
        if corr_native < 0.0
        else 1.0
    )

    corr_display = zero_lag_corr(
        das["strain"][compare_mask],
        display_sign
        * gp1_velocity[compare_mask],
    )

    dt_s = (
        1.0 / das["fs_hz"]
    )

    lag_s, lag_corr = diagnostic_best_lag(
        das["strain"],
        display_sign * gp1_velocity,
        dt_s,
        compare_mask,
    )

    print()
    print("FINAL DAS / geophone waveform validation")
    print("========================================")
    print(f"event                   : {EVENT_ID}")
    print(f"catalog origin          : {ORIGIN}")
    print(f"DAS absolute start      : {das['beg_time']}")
    print(f"DAS absolute end        : {das['end_time']}")
    print(
        f"origin in DAS file      : "
        f"{-float(das['time_s'][0]):.6f} s after file start"
    )
    print(f"DAS physical channel    : {DAS_CHANNEL}")
    print(
        f"fiber tangent ENU       : "
        f"[{fiber_tangent[0]:+.6f}, "
        f"{fiber_tangent[1]:+.6f}, "
        f"{fiber_tangent[2]:+.6f}]"
    )
    print(
        f"fiber TVD               : "
        f"{fiber_metadata['tvd_m']:.2f} m"
    )
    print(f"geophone                : {gp1_metadata['seed_id']}")
    print(
        f"GP1 orientation         : "
        f"az={gp1_metadata['azimuth_deg']:+.2f} deg, "
        f"dip={gp1_metadata['dip_deg']:+.2f} deg"
    )
    print(
        f"GP1 dot(fiber)          : "
        f"{gp1_metadata['dot_fiber']:+.6f}"
    )
    print(
        f"GP1-fiber angle         : "
        f"{gp1_metadata['axis_angle_deg']:.3f} deg"
    )
    print(
        f"common filter           : "
        f"{FMIN_HZ:g}-{FMAX_HZ:g} Hz, zero phase"
    )
    print(
        f"native zero-lag corr    : "
        f"{corr_native:+.4f}"
    )
    print(
        f"display polarity sign   : "
        f"{display_sign:+.0f}"
    )
    print(
        f"display zero-lag corr   : "
        f"{corr_display:+.4f}"
    )
    print(
        f"diagnostic best lag     : "
        f"{lag_s*1000.0:+.2f} ms "
        f"(r={lag_corr:+.4f})"
    )
    print("relative time shift     : NONE")
    print(
        "x-axis                  : "
        "absolute UTC sample time - NCEDC catalog origin"
    )

    figure_path = (
        args.output_dir
        / "das_strain_vs_gp1_velocity_clean.png"
    )

    make_figure(
        time_s=das["time_s"],
        das_strain=das["strain"],
        gp1_velocity=gp1_velocity,
        output_path=figure_path,
    )

    npz_path = (
        args.output_dir
        / "das_strain_vs_gp1_velocity_clean.npz"
    )

    np.savez_compressed(
        npz_path,
        event_id=np.array(
            EVENT_ID
        ),
        catalog_origin=np.array(
            str(ORIGIN)
        ),
        time_from_catalog_origin_s=das["time_s"],
        das_axial_strain=das["strain"],
        gp1_velocity_m_per_s=gp1_velocity,
        das_channel=np.array(
            DAS_CHANNEL
        ),
        fiber_tangent_enu=fiber_tangent,
        fiber_tvd_m=np.array(
            fiber_metadata["tvd_m"]
        ),
        gp1_seed_id=np.array(
            gp1_metadata["seed_id"]
        ),
        gp1_azimuth_deg=np.array(
            gp1_metadata["azimuth_deg"]
        ),
        gp1_dip_deg=np.array(
            gp1_metadata["dip_deg"]
        ),
        gp1_dot_fiber=np.array(
            gp1_metadata["dot_fiber"]
        ),
        gp1_fiber_angle_deg=np.array(
            gp1_metadata["axis_angle_deg"]
        ),
        common_fmin_hz=np.array(
            FMIN_HZ
        ),
        common_fmax_hz=np.array(
            FMAX_HZ
        ),
        native_zero_lag_corr=np.array(
            corr_native
        ),
        display_sign=np.array(
            display_sign
        ),
        display_zero_lag_corr=np.array(
            corr_display
        ),
        diagnostic_best_lag_s=np.array(
            lag_s
        ),
        diagnostic_corr_at_best_lag=np.array(
            lag_corr
        ),
    )

    print()
    print("Saved")
    print("=====")
    print(figure_path)
    print(figure_path.with_suffix(".pdf"))
    print(npz_path)


if __name__ == "__main__":
    main()