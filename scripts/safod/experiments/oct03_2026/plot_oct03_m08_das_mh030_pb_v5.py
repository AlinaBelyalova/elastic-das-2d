#!/usr/bin/env python3
"""
SAFOD Oct-03-2026 M0.8 (nc75443276)
DAS <-> MH030 comparison with MH029 metadata fallback, adapted directly from Ettore's
Plotting_Events_Oct2026_Deep.ipynb and the earlier SCEC comparison scripts.

Key differences from the April/SCEC HDF5 script
------------------------------------------------
* October DAS is Sintela protobuf, not OptaSense HDF5.
* It is read exactly as in Ettore's notebook:
      readFile_protobuf(..., diff=False, system="Sintela")
* The protobuf reader output is treated as strain rate [um/m/s], matching
  Ettore's notebook/colorbar. No HDF5-specific 1e3*1e-9 conversion is applied.
* Two bands are made: 5-40 Hz and 5-200 Hz.
* Filtering uses the exact DASutils.bandpass2D_c(..., zerophase=False)
  call used in Ettore's notebook. The SAME filter is applied to DAS and
  geophone after bringing them to the common DAS time grid.
* Both physical pairs are saved:
      DAS strain      <-> MH030 particle velocity
      DAS strain rate <-> MH030 particle acceleration
* GP1/GP2/GP3 are inspected. If GP1 is nearly parallel to the local fiber,
  GP1 is used directly, as in the SCEC workflow. Otherwise a 3-C projection
  onto the local fiber is attempted if the orientation matrix is well
  conditioned.
* A small channel scan around the nominal co-located channel is diagnostic
  only; it does NOT silently change the channel used in the figures.

Default channel
---------------
1694 is used as the default because the earlier MH029 "Pod 1" comparison
used channel 1694 and MH030 is the current "Pod 1" station code. The script
prints orientation/correlation diagnostics and a local channel scan so this
can be verified rather than assumed.

Run
---
python -m scripts.safod.experiments.oct03_2026.plot_oct03_m08_das_mh030_pb_v5

or explicitly:
python -m scripts.safod.experiments.oct03_2026.plot_oct03_m08_das_mh030_pb_v5 --das-channel 1694
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
import pandas as pd

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
# Event / data configuration
# =============================================================================

EVENT_ID = "nc75443276"
FALLBACK_ORIGIN = UTCDateTime("2026-10-03T04:52:31.000000Z")
ORIGIN = FALLBACK_ORIGIN

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

GEO_XLSX = Path(
    "/home/groups/ettore88/alina/SAFOD/"
    "SAFOD_Phase2_GeoReferenced_Channels.xlsx"
)

DEFAULT_DAS_CHANNEL = 1694

NETWORK = "SF"
STATION = "MH030"
METADATA_FALLBACK_STATION = "MH029"
LOCATION = "01"
GEO_CHANNELS = ("GP1", "GP2", "GP3")

BANDS_HZ = (
    (5.0, 40.0),
    (5.0, 200.0),
)

EDGE_TAPER_FRACTION = 0.05

PLOT_TMIN_S = 0.40
PLOT_TMAX_S = 1.40
ONSET_TMIN_S = 0.25
ONSET_TMAX_S = 1.20
ONSET_NORM_TMIN_S = 0.45
ONSET_NORM_TMAX_S = 1.10
MAX_DIAGNOSTIC_LAG_S = 0.20

GATHER_TMIN_S = 0.40
GATHER_TMAX_S = 2.00
GATHER_MIN_CH = 0
GATHER_MAX_CH = 2500
GATHER_PCLIP = 96.0

SECONDARY_TMIN_S = 0.75
SECONDARY_TMAX_S = 1.05

SCAN_RADIUS_CH = 30
SCAN_TMIN_S = 0.45
SCAN_TMAX_S = 1.20

RESPONSE_MARGIN_S = 20.0
DPI = 300

DEFAULT_OUTPUT_DIR = Path(
    "results/events/20261003_75443276/compare/"
    "das_mh030_comparison_v5_causal"
)


# =============================================================================
# Generic helpers
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


def detrend_demean(
    x: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    y = detrend(
        np.asarray(x, dtype=np.float64),
        axis=axis,
        type="linear",
    )
    return y - np.mean(
        y,
        axis=axis,
        keepdims=True,
    )


def edge_taper(
    x: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64)
    n = y.shape[axis]

    window = tukey(
        n,
        alpha=2.0 * EDGE_TAPER_FRACTION,
    )

    shape = [1] * y.ndim
    shape[axis] = n

    return y * window.reshape(shape)


def ettore_bandpass(
    x: np.ndarray,
    fs_hz: float,
    fmin_hz: float,
    fmax_hz: float,
    axis: int = -1,
) -> np.ndarray:
    """
    Exact SAFOD processing used in Ettore's Oct-2026 notebook:

        DASutils.bandpass2D_c(..., zerophase=False)

    The function expects time on the last axis.  For 1-D input we temporarily
    add a channel dimension.  For N-D input, the requested time axis is moved
    to the end, flattened to (ntraces, nt), filtered, and restored.
    """
    arr = np.asarray(x, dtype=np.float64)

    if axis < 0:
        axis = arr.ndim + axis

    moved = np.moveaxis(arr, axis, -1)
    original_shape = moved.shape
    flat = np.ascontiguousarray(
        moved.reshape(-1, original_shape[-1])
    )

    filtered = DASutils.bandpass2D_c(
        flat,
        float(fmin_hz),
        float(fmax_hz),
        1.0 / float(fs_hz),
        zerophase=False,
    )

    filtered = np.asarray(filtered, dtype=np.float64)
    filtered = filtered.reshape(original_shape)
    return np.moveaxis(filtered, -1, axis)


def zero_lag_corr(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)

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


def diagnostic_best_lag(
    a: np.ndarray,
    b: np.ndarray,
    dt_s: float,
    mask: np.ndarray,
) -> tuple[float, float]:
    """Diagnostic only. No shift is applied."""
    x = np.asarray(a[mask], dtype=np.float64)
    y = np.asarray(b[mask], dtype=np.float64)

    x = x - np.mean(x)
    y = y - np.mean(y)

    max_lag_samples = int(round(MAX_DIAGNOSTIC_LAG_S / dt_s))

    best_abs_corr = -np.inf
    best_corr = np.nan
    best_lag_samples = 0

    for lag in range(-max_lag_samples, max_lag_samples + 1):
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

    return best_lag_samples * dt_s, float(best_corr)


def rms_normalize(
    x: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64)

    scale = float(
        np.sqrt(
            np.mean(
                y[mask] ** 2
            )
        )
    )

    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError(
            "Cannot RMS-normalize zero/invalid trace."
        )

    return y / scale


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


def update_origin_from_ncedc(
    client: Client,
) -> UTCDateTime:
    """
    Use the exact NCEDC origin time when available.

    Ettore's notebook uses 04:52:31Z rounded to the second.  For the
    intermediate-phase timing we prefer the catalog origin with its fractional
    seconds, while retaining the notebook value as a fallback.
    """
    global ORIGIN

    try:
        catalog = client.get_events(
            eventid=75443276
        )

        if len(catalog) < 1:
            raise RuntimeError(
                "NCEDC event query returned no event."
            )

        event = catalog[0]
        origin_obj = (
            event.preferred_origin()
            or event.origins[0]
        )

        ORIGIN = origin_obj.time

        print(
            f"exact NCEDC origin         : "
            f"{ORIGIN}"
        )

        return ORIGIN

    except Exception as exc:
        ORIGIN = FALLBACK_ORIGIN

        print(
            "WARNING: could not retrieve exact "
            f"NCEDC origin ({exc})."
        )
        print(
            f"using fallback origin      : "
            f"{ORIGIN}"
        )

        return ORIGIN


# =============================================================================
# Fiber geometry
# =============================================================================

def load_local_fiber_tangent_enu(
    das_channel: int,
) -> tuple[np.ndarray, dict]:

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

    missing = required.difference(
        df.columns
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
            channel_numeric == channel
        ]

        if len(rows) != 1:
            raise RuntimeError(
                f"Expected one geometry row "
                f"for channel {channel}; "
                f"found {len(rows)}."
            )

        return rows.iloc[0]

    before = get_row(
        das_channel - 1
    )
    center = get_row(
        das_channel
    )
    after = get_row(
        das_channel + 1
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
            "utm_e_m": float(
                center["UTM_E_m"]
            ),
            "utm_n_m": float(
                center["UTM_N_m"]
            ),
        },
    )


# =============================================================================
# DAS: Sintela protobuf
# =============================================================================

def load_das_protobuf() -> dict:
    """
    Read the Oct-03 DAS exactly following Ettore's notebook.

    Important:
    The notebook plots the returned DAS_data directly as
    strain rate [um/m/s], so this script preserves those units.
    """

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
            beg_time - ORIGIN
        )
        + np.arange(
            das_data.shape[1],
            dtype=np.float64,
        )
        / fs_hz
    )

    return {
        "data_strain_rate_um_per_m_per_s": das_data,
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
        "info": info,
    }


def process_one_das_channel(
    *,
    das_data: np.ndarray,
    channel: int,
    fs_hz: float,
    fmin_hz: float,
    fmax_hz: float,
) -> dict:

    if (
        channel < 0
        or channel >= das_data.shape[0]
    ):
        raise RuntimeError(
            f"DAS channel {channel} "
            f"outside shape {das_data.shape}."
        )

    # Reader output is strain rate [um/m/s].
    strain_rate_um = detrend_demean(
        das_data[
            channel,
            :
        ]
    )

    strain_rate_um = edge_taper(
        strain_rate_um
    )

    strain_rate_um_f = ettore_bandpass(
        strain_rate_um,
        fs_hz,
        fmin_hz,
        fmax_hz,
    )

    # Integrating [um/m/s] gives [um/m] = microstrain.
    strain_microstrain = cumulative_trapezoid(
        strain_rate_um,
        dx=1.0 / fs_hz,
        initial=0.0,
    )

    strain_microstrain = detrend_demean(
        strain_microstrain
    )

    strain_microstrain = ettore_bandpass(
        strain_microstrain,
        fs_hz,
        fmin_hz,
        fmax_hz,
    )

    return {
        "strain_rate_um_per_m_per_s": (
            strain_rate_um_f
        ),
        "strain_microstrain": (
            strain_microstrain
        ),
    }


# =============================================================================
# MH030 geophone
# =============================================================================

def response_prefilter(
    native_fs_hz: float,
    target_fmax_hz: float,
) -> tuple[float, float, float, float]:

    nyquist = (
        0.5
        * float(native_fs_hz)
    )

    if target_fmax_hz >= 0.90 * nyquist:
        raise RuntimeError(
            f"Requested fmax={target_fmax_hz:g} Hz "
            f"is too close to MH030 Nyquist="
            f"{nyquist:g} Hz."
        )

    low1 = 0.5
    low2 = 1.0

    high1 = min(
        1.15 * target_fmax_hz,
        0.82 * nyquist,
    )

    high2 = min(
        1.30 * target_fmax_hz,
        0.92 * nyquist,
    )

    if not (
        low1
        < low2
        < target_fmax_hz
        < high1
        < high2
        < nyquist
    ):
        raise RuntimeError(
            "Could not construct a safe "
            "response-removal prefilter for "
            f"fs={native_fs_hz:g} Hz and "
            f"target fmax={target_fmax_hz:g} Hz."
        )

    return (
        low1,
        low2,
        high1,
        high2,
    )


def load_geophone_native(
    *,
    client: Client,
    channel_code: str,
    absolute_start: UTCDateTime,
    absolute_end: UTCDateTime,
    max_target_fmax_hz: float,
) -> dict:
    """
    Waveform comes from current station code MH030.

    As of 2026-10-03 the NCEDC waveform service already serves
    SF.MH030.01.GP*, but StationXML response metadata are not yet available
    for MH030.  We therefore try MH030 metadata first, then fall back to
    historical Pod-1 metadata under MH029.

    IMPORTANT:
    The MH029 fallback is provisional. It is appropriate for timing/waveform
    work only if MH030 is the same physical Pod-1 sensor/orientation/response.
    The script prints the metadata source so this is never hidden.
    """

    waveform_seed_id = (
        f"{NETWORK}."
        f"{STATION}."
        f"{LOCATION}."
        f"{channel_code}"
    )

    # ------------------------------------------------------------------
    # 1) Current waveform: MH030
    # ------------------------------------------------------------------
    try:
        stream = client.get_waveforms(
            network=NETWORK,
            station=STATION,
            location=LOCATION,
            channel=channel_code,
            starttime=absolute_start,
            endtime=absolute_end,
            attach_response=False,
        )
    except FDSNNoDataException as exc:
        raise RuntimeError(
            f"No waveform available for {waveform_seed_id}."
        ) from exc

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

    # ------------------------------------------------------------------
    # 2) Response/orientation metadata:
    #    first try MH030, then historical MH029 Pod-1 metadata.
    # ------------------------------------------------------------------
    metadata_station = STATION
    metadata_seed_id = waveform_seed_id
    metadata_is_fallback = False

    try:
        inventory = client.get_stations(
            network=NETWORK,
            station=STATION,
            location=LOCATION,
            channel=channel_code,
            starttime=absolute_start,
            endtime=absolute_end,
            level="response",
        )

    except FDSNNoDataException:
        metadata_station = METADATA_FALLBACK_STATION
        metadata_seed_id = (
            f"{NETWORK}."
            f"{metadata_station}."
            f"{LOCATION}."
            f"{channel_code}"
        )
        metadata_is_fallback = True

        try:
            inventory = client.get_stations(
                network=NETWORK,
                station=metadata_station,
                location=LOCATION,
                channel=channel_code,
                starttime=ORIGIN,
                endtime=ORIGIN + 1.0,
                level="response",
            )
        except FDSNNoDataException as exc:
            raise RuntimeError(
                f"Waveform exists for {waveform_seed_id}, but no response "
                f"metadata were found for either MH030 or fallback "
                f"{metadata_seed_id}."
            ) from exc

    # ------------------------------------------------------------------
    # 3) Preprocess counts.
    # ------------------------------------------------------------------
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

    pre_filt = response_prefilter(
        native_fs_hz,
        max_target_fmax_hz,
    )

    # ------------------------------------------------------------------
    # 4) Remove response.
    #
    # If using MH029 StationXML for an MH030 waveform, ObsPy will not match
    # the trace ID automatically.  Use a temporary trace whose station code
    # matches the metadata, remove the response, then keep only the resulting
    # physical velocity samples.
    # ------------------------------------------------------------------
    tr_response = tr.copy()

    if metadata_is_fallback:
        tr_response.stats.station = metadata_station

    tr_response.remove_response(
        inventory=inventory,
        output="VEL",
        pre_filt=pre_filt,
        water_level=60.0,
        taper=False,
        zero_mean=False,
    )

    orientation = inventory.get_orientation(
        metadata_seed_id,
        datetime=ORIGIN,
    )

    try:
        coordinates = inventory.get_coordinates(
            metadata_seed_id,
            datetime=ORIGIN,
        )
    except Exception:
        coordinates = {}

    geo_time_s = (
        float(
            tr.stats.starttime
            - ORIGIN
        )
        + np.arange(
            tr.stats.npts,
            dtype=np.float64,
        )
        / native_fs_hz
    )

    return {
        "seed_id": waveform_seed_id,
        "waveform_seed_id": waveform_seed_id,
        "metadata_seed_id": metadata_seed_id,
        "metadata_station": metadata_station,
        "metadata_is_fallback": metadata_is_fallback,
        "time_s": geo_time_s,
        "velocity_native_m_per_s": np.asarray(
            tr_response.data,
            dtype=np.float64,
        ),
        "raw_counts": np.asarray(
            tr.data,
            dtype=np.float64,
        ),
        "native_fs_hz": native_fs_hz,
        "orientation": orientation,
        "coordinates": coordinates,
        "response_prefilter_hz": pre_filt,
    }


def interpolate_and_filter_geophone(
    *,
    native: dict,
    das_time_s: np.ndarray,
    das_fs_hz: float,
    fmin_hz: float,
    fmax_hz: float,
) -> np.ndarray:

    geo_time_s = native[
        "time_s"
    ]

    if (
        das_time_s[0]
        < geo_time_s[0]
        or das_time_s[-1]
        > geo_time_s[-1]
    ):
        raise RuntimeError(
            f"{native['seed_id']} "
            "does not cover the full "
            "DAS UTC interval."
        )

    velocity = np.interp(
        das_time_s,
        geo_time_s,
        native[
            "velocity_native_m_per_s"
        ],
    )

    velocity = ettore_bandpass(
        velocity,
        das_fs_hz,
        fmin_hz,
        fmax_hz,
    )

    return velocity


def interpolate_and_filter_raw_counts(
    *,
    native: dict,
    das_time_s: np.ndarray,
    das_fs_hz: float,
    fmin_hz: float,
    fmax_hz: float,
) -> np.ndarray:
    """
    Timing-only diagnostic.  No instrument-response correction is applied.
    MH030 raw counts are interpolated to the DAS time grid and passed through
    the exact same non-zero-phase DASutils filter as the DAS trace.
    """
    if (
        das_time_s[0] < native["time_s"][0]
        or das_time_s[-1] > native["time_s"][-1]
    ):
        raise RuntimeError(
            f"{native['seed_id']} does not cover the full DAS interval."
        )

    counts = np.interp(
        das_time_s,
        native["time_s"],
        native["raw_counts"],
    )

    return ettore_bandpass(
        counts,
        das_fs_hz,
        fmin_hz,
        fmax_hz,
    )


def build_axial_geophone(
    *,
    native_by_channel: dict,
    das_time_s: np.ndarray,
    das_fs_hz: float,
    fiber_tangent_enu: np.ndarray,
    fmin_hz: float,
    fmax_hz: float,
) -> dict:
    """
    Prefer GP1 directly when it is nearly parallel to the fiber.

    Otherwise try a 3-C ENU reconstruction if the orientation matrix is
    sufficiently well-conditioned.
    """

    velocities = {}
    raw_counts_filtered = {}

    axes = {}

    for channel_code in GEO_CHANNELS:

        native = native_by_channel[
            channel_code
        ]

        velocities[channel_code] = (
            interpolate_and_filter_geophone(
                native=native,
                das_time_s=das_time_s,
                das_fs_hz=das_fs_hz,
                fmin_hz=fmin_hz,
                fmax_hz=fmax_hz,
            )
        )

        raw_counts_filtered[channel_code] = (
            interpolate_and_filter_raw_counts(
                native=native,
                das_time_s=das_time_s,
                das_fs_hz=das_fs_hz,
                fmin_hz=fmin_hz,
                fmax_hz=fmax_hz,
            )
        )

        orientation = native[
            "orientation"
        ]

        axes[channel_code] = (
            component_unit_vector_enu(
                orientation[
                    "azimuth"
                ],
                orientation[
                    "dip"
                ],
            )
        )

    gp1_dot_signed = float(
        np.dot(
            axes["GP1"],
            fiber_tangent_enu,
        )
    )

    gp1_angle_deg = float(
        np.degrees(
            np.arccos(
                np.clip(
                    abs(
                        gp1_dot_signed
                    ),
                    0.0,
                    1.0,
                )
            )
        )
    )

    gp1_velocity = (
        velocities["GP1"].copy()
    )
    gp1_counts = (
        raw_counts_filtered["GP1"].copy()
    )

    if gp1_dot_signed < 0.0:
        gp1_velocity *= -1.0
        gp1_counts *= -1.0

    orientation_matrix = np.vstack(
        [
            axes[channel]
            for channel in GEO_CHANNELS
        ]
    )

    condition_number = float(
        np.linalg.cond(
            orientation_matrix
        )
    )

    # Same philosophy as the SCEC comparison:
    # if one physical component is already essentially axial,
    # do not amplify noise through an unnecessary inversion.
    if gp1_angle_deg <= 15.0:

        axial_velocity = (
            gp1_velocity
        )

        method = (
            "GP1 direct "
            "(nearly parallel to fiber)"
        )

    elif (
        np.isfinite(
            condition_number
        )
        and condition_number <= 20.0
    ):

        component_matrix = np.vstack(
            [
                velocities[channel]
                for channel in GEO_CHANNELS
            ]
        )

        velocity_enu = np.linalg.solve(
            orientation_matrix,
            component_matrix,
        )

        axial_velocity = (
            fiber_tangent_enu
            @ velocity_enu
        )

        method = (
            "3C reconstruction projected "
            "onto local fiber tangent"
        )

    else:

        best_channel = max(
            GEO_CHANNELS,
            key=lambda channel: abs(
                float(
                    np.dot(
                        axes[channel],
                        fiber_tangent_enu,
                    )
                )
            ),
        )

        best_dot = float(
            np.dot(
                axes[best_channel],
                fiber_tangent_enu,
            )
        )

        axial_velocity = (
            velocities[
                best_channel
            ].copy()
        )

        if best_dot < 0.0:
            axial_velocity *= -1.0

        method = (
            f"{best_channel} direct "
            "(3C orientation matrix "
            "ill-conditioned)"
        )

    axial_acceleration = np.gradient(
        axial_velocity,
        1.0 / das_fs_hz,
    )

    return {
        "velocity_m_per_s": axial_velocity,
        "acceleration_m_per_s2": axial_acceleration,
        "acceleration_cm_per_s2": (
            axial_acceleration
            * 100.0
        ),
        "component_velocity_m_per_s": (
            velocities
        ),
        "gp1_raw_counts_filtered": (
            gp1_counts
        ),
        "component_axes_enu": axes,
        "gp1_dot_fiber_signed": (
            gp1_dot_signed
        ),
        "gp1_fiber_angle_deg": (
            gp1_angle_deg
        ),
        "orientation_condition_number": (
            condition_number
        ),
        "method": method,
    }


# =============================================================================
# Diagnostic channel scan
# =============================================================================

def scan_nearby_channels(
    *,
    das_data: np.ndarray,
    time_s: np.ndarray,
    fs_hz: float,
    nominal_channel: int,
    geophone_velocity: np.ndarray,
    fmin_hz: float,
    fmax_hz: float,
) -> pd.DataFrame:

    scan_min = max(
        0,
        nominal_channel
        - SCAN_RADIUS_CH,
    )

    scan_max = min(
        das_data.shape[0] - 1,
        nominal_channel
        + SCAN_RADIUS_CH,
    )

    channels = np.arange(
        scan_min,
        scan_max + 1,
    )

    rate_um = np.asarray(
        das_data[
            channels,
            :
        ],
        dtype=np.float64,
    )

    rate_um = detrend_demean(
        rate_um,
        axis=1,
    )

    rate_um = edge_taper(
        rate_um,
        axis=1,
    )

    # Integrate each candidate to strain [microstrain].
    strain = cumulative_trapezoid(
        rate_um,
        dx=1.0 / fs_hz,
        axis=1,
        initial=0.0,
    )

    strain = detrend_demean(
        strain,
        axis=1,
    )

    strain = ettore_bandpass(
        strain,
        fs_hz,
        fmin_hz,
        fmax_hz,
        axis=1,
    )

    mask = (
        (time_s >= SCAN_TMIN_S)
        & (time_s <= SCAN_TMAX_S)
    )

    rows = []

    for row_index, channel in enumerate(
        channels
    ):

        corr = zero_lag_corr(
            strain[
                row_index,
                mask,
            ],
            geophone_velocity[
                mask
            ],
        )

        rows.append(
            {
                "channel": int(
                    channel
                ),
                "corr_signed": float(
                    corr
                ),
                "corr_abs": float(
                    abs(corr)
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "corr_abs",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# =============================================================================
# Figures
# =============================================================================

def plot_gather(
    *,
    das_data: np.ndarray,
    time_s: np.ndarray,
    fs_hz: float,
    selected_channel: int,
    fmin_hz: float,
    fmax_hz: float,
    output_path: Path,
) -> None:

    min_ch = GATHER_MIN_CH

    max_ch = min(
        GATHER_MAX_CH,
        das_data.shape[0],
    )

    channel_slice = np.asarray(
        das_data[
            min_ch:max_ch,
            :
        ],
        dtype=np.float64,
    )

    channel_slice = ettore_bandpass(
        channel_slice,
        fs_hz,
        fmin_hz,
        fmax_hz,
        axis=1,
    )

    it_min = int(
        np.argmin(
            np.abs(
                time_s
                - GATHER_TMIN_S
            )
        )
    )

    it_max = int(
        np.argmin(
            np.abs(
                time_s
                - GATHER_TMAX_S
            )
        )
    )

    clip_value = float(
        np.percentile(
            np.abs(
                channel_slice[
                    :,
                    it_min:it_max,
                ]
            ),
            GATHER_PCLIP,
        )
    )

    fig, ax = plt.subplots(
        figsize=(15, 8)
    )

    im = ax.imshow(
        channel_slice[
            :,
            it_min:it_max,
        ].T,
        extent=[
            min_ch,
            max_ch,
            time_s[it_max],
            time_s[it_min],
        ],
        aspect="auto",
        vmin=-clip_value,
        vmax=clip_value,
        cmap="seismic",
    )

    ax.axvline(
        selected_channel,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=(
            f"comparison ch "
            f"{selected_channel}"
        ),
    )

    ax.set_xlabel(
        "Channel number",
        fontweight="bold",
    )

    ax.set_ylabel(
        "Time from 2026-10-03 "
        "04:52:31Z [s]",
        fontweight="bold",
    )

    ax.set_title(
        f"DAS strain rate | "
        f"{fmin_hz:g}-{fmax_hz:g} Hz",
        fontweight="bold",
    )

    ax.legend(
        frameon=False,
        loc="lower left",
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
        pad=0.012,
    )

    cbar.set_label(
        r"strain rate [$\mu$m/m/s]",
        fontweight="bold",
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


def plot_normalized_overlay(
    *,
    time_s: np.ndarray,
    das_trace: np.ndarray,
    geophone_trace: np.ndarray,
    das_label: str,
    geophone_label: str,
    title: str,
    output_path: Path,
    tmin_s: float = PLOT_TMIN_S,
    tmax_s: float = PLOT_TMAX_S,
    shade_secondary: bool = True,
) -> dict:

    plot_mask = (
        (time_s >= tmin_s)
        & (time_s <= tmax_s)
    )

    das_normalized = rms_normalize(
        das_trace,
        plot_mask,
    )

    geo_normalized = rms_normalize(
        geophone_trace,
        plot_mask,
    )

    corr_native = zero_lag_corr(
        das_normalized[
            plot_mask
        ],
        geo_normalized[
            plot_mask
        ],
    )

    # Only a global polarity convention for visual comparison.
    # No time shift and no waveform/amplitude fit.
    display_sign = (
        -1.0
        if (
            np.isfinite(
                corr_native
            )
            and corr_native < 0.0
        )
        else 1.0
    )

    geo_display = (
        display_sign
        * geo_normalized
    )

    corr_display = zero_lag_corr(
        das_normalized[
            plot_mask
        ],
        geo_display[
            plot_mask
        ],
    )

    secondary_mask = (
        (time_s >= SECONDARY_TMIN_S)
        & (time_s <= SECONDARY_TMAX_S)
    )

    corr_secondary = zero_lag_corr(
        das_normalized[
            secondary_mask
        ],
        geo_display[
            secondary_mask
        ],
    )

    dt_s = float(time_s[1] - time_s[0])
    best_lag_full_s, best_corr_full = diagnostic_best_lag(
        das_normalized,
        geo_display,
        dt_s,
        plot_mask,
    )
    best_lag_secondary_s, best_corr_secondary = diagnostic_best_lag(
        das_normalized,
        geo_display,
        dt_s,
        secondary_mask,
    )

    fig, ax = plt.subplots(
        figsize=(11.2, 4.0)
    )

    if shade_secondary:
        ax.axvspan(
            SECONDARY_TMIN_S,
            SECONDARY_TMAX_S,
            color="0.93",
            zorder=0,
        )

    ax.plot(
        time_s[
            plot_mask
        ],
        das_normalized[
            plot_mask
        ],
        color="black",
        linewidth=1.65,
        label=das_label,
    )

    ax.plot(
        time_s[
            plot_mask
        ],
        geo_display[
            plot_mask
        ],
        color="#940025",
        linewidth=1.55,
        label=geophone_label,
    )

    ax.axhline(
        0.0,
        color="0.82",
        linewidth=0.7,
    )

    ax.set_xlim(
        tmin_s,
        tmax_s,
    )

    ax.set_xlabel(
        "Time from catalog origin [s]",
        fontsize=13,
        fontweight="bold",
    )

    ax.set_ylabel(
        "Normalized amplitude",
        fontsize=13,
        fontweight="bold",
    )

    ax.set_title(
        title,
        fontsize=13,
        fontweight="bold",
    )

    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=10,
    )

    if shade_secondary:
        ax.text(
            0.01,
            0.03,
            "gray: 0.75-1.05 s "
            "secondary/intermediate phase window",
            transform=ax.transAxes,
            fontsize=9,
            color="0.35",
        )

    fig.tight_layout(
        pad=0.6
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

    return {
        "corr_native": (
            corr_native
        ),
        "display_sign": (
            display_sign
        ),
        "corr_display": (
            corr_display
        ),
        "corr_secondary_window": (
            corr_secondary
        ),
        "best_lag_full_s": best_lag_full_s,
        "best_corr_full": best_corr_full,
        "best_lag_secondary_s": best_lag_secondary_s,
        "best_corr_secondary": best_corr_secondary,
    }


def plot_onset_timing_diagnostic(
    *,
    time_s: np.ndarray,
    das_strain_rate: np.ndarray,
    mh030_raw_counts: np.ndarray,
    fmin_hz: float,
    fmax_hz: float,
    das_channel: int,
    output_path: Path,
) -> dict:
    """
    Show the earthquake onset without relying on the provisional MH029
    instrument response.  Both series have already been passed through the
    exact same DASutils non-zero-phase bandpass.  The two traces are stacked,
    not overlaid, because raw counts and DAS strain rate are different
    observables; this figure is for timing/onset only.
    """
    plot_mask = (
        (time_s >= ONSET_TMIN_S)
        & (time_s <= ONSET_TMAX_S)
    )
    norm_mask = (
        (time_s >= ONSET_NORM_TMIN_S)
        & (time_s <= ONSET_NORM_TMAX_S)
    )

    d = rms_normalize(das_strain_rate, norm_mask)
    g = rms_normalize(mh030_raw_counts, norm_mask)

    # Global polarity only for easier visual comparison; no time shift.
    corr0 = zero_lag_corr(d[norm_mask], g[norm_mask])
    sign = -1.0 if np.isfinite(corr0) and corr0 < 0.0 else 1.0
    g = sign * g

    lag_s, lag_corr = diagnostic_best_lag(
        d, g, float(time_s[1] - time_s[0]), norm_mask
    )

    fig, ax = plt.subplots(figsize=(11.2, 4.6))

    ax.plot(
        time_s[plot_mask],
        0.38 * d[plot_mask] + 1.0,
        color="black",
        linewidth=1.2,
    )
    ax.plot(
        time_s[plot_mask],
        0.38 * g[plot_mask] + 0.0,
        color="#940025",
        linewidth=1.2,
    )

    ax.axhline(1.0, color="0.80", linewidth=0.6)
    ax.axhline(0.0, color="0.80", linewidth=0.6)
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(
        ["MH030 GP1 raw counts", f"DAS ch {das_channel} strain rate"],
        fontweight="bold",
    )
    ax.set_xlim(ONSET_TMIN_S, ONSET_TMAX_S)
    ax.set_xlabel(
        "Time from catalog origin [s]",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_title(
        f"Onset timing diagnostic | {fmin_hz:g}-{fmax_hz:g} Hz | "
        "same DASutils causal filter",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.25)
    ax.text(
        0.01, 0.03,
        "MH030 shown in raw counts: no provisional response correction; "
        "no relative time shift applied",
        transform=ax.transAxes,
        fontsize=9,
        color="0.35",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    return {
        "zero_lag_corr": corr0,
        "best_lag_s": lag_s,
        "best_lag_corr": lag_corr,
        "display_sign": sign,
    }


def plot_three_components(
    *,
    time_s: np.ndarray,
    component_velocity: dict,
    fmin_hz: float,
    fmax_hz: float,
    output_path: Path,
) -> None:

    mask = (
        (time_s >= PLOT_TMIN_S)
        & (time_s <= PLOT_TMAX_S)
    )

    offsets = {
        "GP1": 2.0,
        "GP2": 1.0,
        "GP3": 0.0,
    }

    fig, ax = plt.subplots(
        figsize=(11.2, 4.5)
    )

    ax.axvspan(
        SECONDARY_TMIN_S,
        SECONDARY_TMAX_S,
        color="0.93",
        zorder=0,
    )

    for channel in GEO_CHANNELS:

        normalized = rms_normalize(
            component_velocity[
                channel
            ],
            mask,
        )

        ax.plot(
            time_s[
                mask
            ],
            0.34
            * normalized[
                mask
            ]
            + offsets[
                channel
            ],
            color="black",
            linewidth=1.0,
        )

    ax.set_yticks(
        [
            0.0,
            1.0,
            2.0,
        ]
    )

    ax.set_yticklabels(
        [
            "GP3",
            "GP2",
            "GP1",
        ],
        fontweight="bold",
    )

    ax.set_xlim(
        PLOT_TMIN_S,
        PLOT_TMAX_S,
    )

    ax.set_xlabel(
        "Time from catalog origin [s]",
        fontweight="bold",
    )

    ax.set_title(
        f"MH030 response-corrected velocity | "
        f"{fmin_hz:g}-{fmax_hz:g} Hz",
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


# =============================================================================
# Arguments / main
# =============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Oct-03-2026 SAFOD M0.8 "
            "Sintela DAS / MH030 comparison."
        )
    )

    parser.add_argument(
        "--das-channel",
        type=int,
        default=DEFAULT_DAS_CHANNEL,
        help=(
            "Nominal DAS channel co-located "
            "with MH030. Default: 1694."
        ),
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

    update_origin_from_ncedc(
        client
    )

    fiber_tangent, fiber_meta = (
        load_local_fiber_tangent_enu(
            args.das_channel
        )
    )

    das = load_das_protobuf()

    print()
    print(
        "OCT-03-2026 SAFOD M0.8"
    )
    print(
        "DAS / MH030 comparison"
    )
    print(
        "=========================="
    )
    print(
        f"event                     : "
        f"{EVENT_ID}"
    )
    print(
        f"catalog origin            : "
        f"{ORIGIN}"
    )
    print(
        f"DAS files                 : "
        f"{len(DAS_FILES)} protobuf files"
    )
    print(
        f"DAS shape                 : "
        f"{das['data_strain_rate_um_per_m_per_s'].shape}"
    )
    print(
        f"DAS fs                    : "
        f"{das['fs_hz']:.3f} Hz"
    )
    print(
        f"DAS dx                    : "
        f"{das['dx_m']:.4f} m"
    )
    print(
        f"DAS units                 : "
        "strain rate [um/m/s] "
        "(following Ettore notebook)"
    )
    print(
        f"nominal DAS channel       : "
        f"{args.das_channel}"
    )
    print(
        f"nominal channel TVD       : "
        f"{fiber_meta['tvd_m']:.2f} m"
    )
    print(
        "fiber tangent ENU         : "
        f"[{fiber_tangent[0]:+.6f}, "
        f"{fiber_tangent[1]:+.6f}, "
        f"{fiber_tangent[2]:+.6f}]"
    )

    absolute_start = (
        das["beg_time"]
        - RESPONSE_MARGIN_S
    )

    absolute_end = (
        das["end_time"]
        + RESPONSE_MARGIN_S
    )

    max_target_fmax_hz = max(
        fmax
        for _, fmax in BANDS_HZ
    )

    geo_native = {}

    for channel_code in GEO_CHANNELS:

        geo_native[
            channel_code
        ] = load_geophone_native(
            client=client,
            channel_code=channel_code,
            absolute_start=absolute_start,
            absolute_end=absolute_end,
            max_target_fmax_hz=(
                max_target_fmax_hz
            ),
        )

        native = geo_native[
            channel_code
        ]

        orientation = native[
            "orientation"
        ]

        print()
        print(
            f"{native['waveform_seed_id']}"
        )
        print(
            f"  waveform source         : "
            f"{native['waveform_seed_id']}"
        )
        print(
            f"  metadata source         : "
            f"{native['metadata_seed_id']}"
        )
        print(
            f"  metadata fallback       : "
            f"{native['metadata_is_fallback']}"
        )
        print(
            f"  native fs               : "
            f"{native['native_fs_hz']:.3f} Hz"
        )
        print(
            f"  azimuth                 : "
            f"{orientation['azimuth']:+.3f} deg"
        )
        print(
            f"  dip                     : "
            f"{orientation['dip']:+.3f} deg"
        )
        print(
            f"  response prefilter      : "
            f"{native['response_prefilter_hz']}"
        )
        print(
            f"  coordinates             : "
            f"{native['coordinates']}"
        )

    if any(
        geo_native[ch]["metadata_is_fallback"]
        for ch in GEO_CHANNELS
    ):
        print()
        print("WARNING")
        print("=======")
        print(
            "MH030 waveforms are current, but NCEDC has not yet "
            "published MH030 response metadata."
        )
        print(
            "Response/orientation are provisionally taken from MH029 "
            "(historical Main Hole Pod 1)."
        )
        print(
            "Use the response-corrected amplitudes/orientation only after "
            "confirming MH030 is the same physical Pod-1 installation."
        )

    summary_rows = []

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

        das_one = process_one_das_channel(
            das_data=(
                das[
                    "data_strain_rate_um_per_m_per_s"
                ]
            ),
            channel=args.das_channel,
            fs_hz=das["fs_hz"],
            fmin_hz=fmin_hz,
            fmax_hz=fmax_hz,
        )

        geo = build_axial_geophone(
            native_by_channel=geo_native,
            das_time_s=das["time_s"],
            das_fs_hz=das["fs_hz"],
            fiber_tangent_enu=(
                fiber_tangent
            ),
            fmin_hz=fmin_hz,
            fmax_hz=fmax_hz,
        )

        print()
        print(
            f"--- {band_label} ---"
        )
        print(
            f"axial geophone method     : "
            f"{geo['method']}"
        )
        print(
            f"GP1 dot(fiber), signed    : "
            f"{geo['gp1_dot_fiber_signed']:+.6f}"
        )
        print(
            f"GP1-fiber angle           : "
            f"{geo['gp1_fiber_angle_deg']:.3f} deg"
        )
        print(
            f"3C orientation cond(A)    : "
            f"{geo['orientation_condition_number']:.3f}"
        )

        gather_path = (
            args.output_dir
            / (
                "00_das_gather_"
                f"{band_tag}.png"
            )
        )

        plot_gather(
            das_data=(
                das[
                    "data_strain_rate_um_per_m_per_s"
                ]
            ),
            time_s=das["time_s"],
            fs_hz=das["fs_hz"],
            selected_channel=(
                args.das_channel
            ),
            fmin_hz=fmin_hz,
            fmax_hz=fmax_hz,
            output_path=gather_path,
        )

        strain_velocity_result = (
            plot_normalized_overlay(
                time_s=das["time_s"],
                das_trace=(
                    das_one[
                        "strain_microstrain"
                    ]
                ),
                geophone_trace=(
                    geo[
                        "velocity_m_per_s"
                    ]
                ),
                das_label=(
                    f"DAS ch "
                    f"{args.das_channel}: "
                    "axial strain"
                ),
                geophone_label=(
                    "MH030: axial "
                    "particle velocity"
                ),
                title=(
                    "DAS strain vs MH030 "
                    "axial velocity | "
                    f"{band_label}"
                ),
                output_path=(
                    args.output_dir
                    / (
                        "01_strain_vs_"
                        "mh030_velocity_"
                        f"{band_tag}.png"
                    )
                ),
            )
        )

        onset_result = plot_onset_timing_diagnostic(
            time_s=das["time_s"],
            das_strain_rate=(
                das_one["strain_rate_um_per_m_per_s"]
            ),
            mh030_raw_counts=(
                geo["gp1_raw_counts_filtered"]
            ),
            fmin_hz=fmin_hz,
            fmax_hz=fmax_hz,
            das_channel=args.das_channel,
            output_path=(
                args.output_dir
                / f"01a_onset_timing_rawcounts_{band_tag}.png"
            ),
        )

        rate_accel_result = (
            plot_normalized_overlay(
                time_s=das["time_s"],
                das_trace=(
                    das_one[
                        "strain_rate_um_per_m_per_s"
                    ]
                ),
                geophone_trace=(
                    geo[
                        "acceleration_cm_per_s2"
                    ]
                ),
                das_label=(
                    f"DAS ch "
                    f"{args.das_channel}: "
                    "strain rate"
                ),
                geophone_label=(
                    "MH030: axial "
                    "particle acceleration"
                ),
                title=(
                    "DAS strain rate vs "
                    "MH030 axial acceleration | "
                    f"{band_label}"
                ),
                output_path=(
                    args.output_dir
                    / (
                        "02_strainrate_vs_"
                        "mh030_acceleration_"
                        f"{band_tag}.png"
                    )
                ),
            )
        )

        plot_normalized_overlay(
            time_s=das["time_s"],
            das_trace=(
                das_one[
                    "strain_microstrain"
                ]
            ),
            geophone_trace=(
                geo[
                    "velocity_m_per_s"
                ]
            ),
            das_label=(
                f"DAS ch "
                f"{args.das_channel}"
            ),
            geophone_label=(
                "MH030 axial velocity"
            ),
            title=(
                "Secondary/intermediate "
                "phase zoom | "
                f"{band_label}"
            ),
            output_path=(
                args.output_dir
                / (
                    "03_secondary_phase_zoom_"
                    f"{band_tag}.png"
                )
            ),
            tmin_s=0.65,
            tmax_s=1.10,
            shade_secondary=True,
        )

        plot_three_components(
            time_s=das["time_s"],
            component_velocity=(
                geo[
                    "component_velocity_m_per_s"
                ]
            ),
            fmin_hz=fmin_hz,
            fmax_hz=fmax_hz,
            output_path=(
                args.output_dir
                / (
                    "04_mh030_3c_velocity_"
                    f"{band_tag}.png"
                )
            ),
        )

        scan = scan_nearby_channels(
            das_data=(
                das[
                    "data_strain_rate_um_per_m_per_s"
                ]
            ),
            time_s=das["time_s"],
            fs_hz=das["fs_hz"],
            nominal_channel=(
                args.das_channel
            ),
            geophone_velocity=(
                geo[
                    "velocity_m_per_s"
                ]
            ),
            fmin_hz=fmin_hz,
            fmax_hz=fmax_hz,
        )

        scan_path = (
            args.output_dir
            / (
                "channel_scan_"
                f"{band_tag}.csv"
            )
        )

        scan.to_csv(
            scan_path,
            index=False,
        )

        print(
            "top local channel correlations:"
        )
        print(
            scan.head(10).to_string(
                index=False
            )
        )

        print(
            "processing filter           : "
            "DASutils.bandpass2D_c, zerophase=False"
        )
        print(
            "corr(strain, velocity), "
            "full window             : "
            f"{strain_velocity_result['corr_display']:+.4f}"
        )
        print(
            "corr(strain, velocity), "
            "secondary window        : "
            f"{strain_velocity_result['corr_secondary_window']:+.4f}"
        )
        print(
            "best lag(strain, velocity), full : "
            f"{1000.0*strain_velocity_result['best_lag_full_s']:+.1f} ms "
            f"(r={strain_velocity_result['best_corr_full']:+.4f})"
        )
        print(
            "best lag(strain, velocity), sec. : "
            f"{1000.0*strain_velocity_result['best_lag_secondary_s']:+.1f} ms "
            f"(r={strain_velocity_result['best_corr_secondary']:+.4f})"
        )
        print(
            "corr(rate, acceleration), "
            "full window             : "
            f"{rate_accel_result['corr_display']:+.4f}"
        )
        print(
            "corr(rate, acceleration), "
            "secondary window        : "
            f"{rate_accel_result['corr_secondary_window']:+.4f}"
        )
        print(
            "best lag(rate, acceleration), full : "
            f"{1000.0*rate_accel_result['best_lag_full_s']:+.1f} ms "
            f"(r={rate_accel_result['best_corr_full']:+.4f})"
        )
        print(
            "best lag(rate, acceleration), sec. : "
            f"{1000.0*rate_accel_result['best_lag_secondary_s']:+.1f} ms "
            f"(r={rate_accel_result['best_corr_secondary']:+.4f})"
        )
        print(
            "onset timing raw-count lag         : "
            f"{1000.0*onset_result['best_lag_s']:+.1f} ms "
            f"(r={onset_result['best_lag_corr']:+.4f})"
        )
        print(
            "relative DAS/MH030 shift applied : NONE (diagnostic lag only)"
        )

        summary_rows.append(
            {
                "band": band_label,
                "das_channel": (
                    args.das_channel
                ),
                "geophone_method": (
                    geo["method"]
                ),
                "gp1_fiber_angle_deg": (
                    geo[
                        "gp1_fiber_angle_deg"
                    ]
                ),
                "orientation_condition_number": (
                    geo[
                        "orientation_condition_number"
                    ]
                ),
                "corr_strain_velocity_full": (
                    strain_velocity_result[
                        "corr_display"
                    ]
                ),
                "corr_strain_velocity_secondary": (
                    strain_velocity_result[
                        "corr_secondary_window"
                    ]
                ),
                "corr_rate_accel_full": (
                    rate_accel_result[
                        "corr_display"
                    ]
                ),
                "corr_rate_accel_secondary": (
                    rate_accel_result[
                        "corr_secondary_window"
                    ]
                ),
                "best_lag_strain_velocity_full_ms": 1000.0 * strain_velocity_result["best_lag_full_s"],
                "best_lag_strain_velocity_secondary_ms": 1000.0 * strain_velocity_result["best_lag_secondary_s"],
                "best_lag_rate_accel_full_ms": 1000.0 * rate_accel_result["best_lag_full_s"],
                "best_lag_rate_accel_secondary_ms": 1000.0 * rate_accel_result["best_lag_secondary_s"],
                "onset_rawcounts_best_lag_ms": 1000.0 * onset_result["best_lag_s"],
                "onset_rawcounts_best_corr": onset_result["best_lag_corr"],
                "best_scan_channel": int(
                    scan.iloc[0][
                        "channel"
                    ]
                ),
                "best_scan_abs_corr": float(
                    scan.iloc[0][
                        "corr_abs"
                    ]
                ),
            }
        )

        np.savez_compressed(
            args.output_dir
            / (
                "waveforms_"
                f"{band_tag}.npz"
            ),
            event_id=np.array(
                EVENT_ID
            ),
            catalog_origin=np.array(
                str(ORIGIN)
            ),
            time_from_origin_s=(
                das["time_s"]
            ),
            das_channel=np.array(
                args.das_channel
            ),
            das_strain_microstrain=(
                das_one[
                    "strain_microstrain"
                ]
            ),
            das_strain_rate_um_per_m_per_s=(
                das_one[
                    "strain_rate_um_per_m_per_s"
                ]
            ),
            mh030_axial_velocity_m_per_s=(
                geo[
                    "velocity_m_per_s"
                ]
            ),
            mh030_axial_acceleration_cm_per_s2=(
                geo[
                    "acceleration_cm_per_s2"
                ]
            ),
            mh030_gp1_velocity_m_per_s=(
                geo[
                    "component_velocity_m_per_s"
                ][
                    "GP1"
                ]
            ),
            mh030_gp2_velocity_m_per_s=(
                geo[
                    "component_velocity_m_per_s"
                ][
                    "GP2"
                ]
            ),
            mh030_gp3_velocity_m_per_s=(
                geo[
                    "component_velocity_m_per_s"
                ][
                    "GP3"
                ]
            ),
            fmin_hz=np.array(
                fmin_hz
            ),
            fmax_hz=np.array(
                fmax_hz
            ),
            fiber_tangent_enu=(
                fiber_tangent
            ),
            fiber_tvd_m=np.array(
                fiber_meta["tvd_m"]
            ),
            gp1_fiber_angle_deg=np.array(
                geo[
                    "gp1_fiber_angle_deg"
                ]
            ),
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        args.output_dir
        / "comparison_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("Saved")
    print("=====")
    print(
        args.output_dir
    )
    print(
        summary_path
    )
    print()
    print(
        summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
