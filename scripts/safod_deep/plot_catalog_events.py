# ==============================================================================
# scripts/safod_deep/plot_catalog_events.py
#
# Adaptive full-cable DAS QC plots for every event in catalog_recorded.csv.
#
# This is a visual catalog-screening workflow. It does not pick phases and it
# does not alter the scientific event catalog.
#
# Adaptive display recipe
# -----------------------
# minimum 3-D source-cable distance < 8 km:
#     5-30 Hz,  -0.5 to 8 s
#
# 8 km <= distance < 20 km:
#     3-15 Hz,  -0.5 to at least 12.5 s
#
# distance >= 20 km:
#     1-12 Hz,  -0.5 to a distance-dependent end time (20-30 s)
#
# The default filter is causal (zerophase=False), matching Ettore's event-QC
# notebook. CLI values such as --fmin or --display-tmax override only the
# requested fields of the adaptive recipe.
#
# Coordinate convention
# ---------------------
# The horizontal axis is the immutable GEO_XLSX physical reference channel.
# HDF5 row numbering changes between standard/no-spool and all-channel
# configurations, so rows are registered through:
#
#   physical channel = first_channel_index + HDF5 row
#                      - reference_locus_offset
#
# and cropped to the physical GEO_XLSX channel range.
#
# Examples
# --------
# Test five events:
#
#   python -m scripts.safod_deep.plot_catalog_events \
#       --limit 5 --overwrite
#
# One event:
#
#   python -m scripts.safod_deep.plot_catalog_events \
#       --event-id 75334912 --overwrite
#
# Eight independent shards:
#
#   python -m scripts.safod_deep.plot_catalog_events \
#       --num-shards 8 --shard-index "$SLURM_ARRAY_TASK_ID"
#
# Build/rebuild the HTML gallery after all shards finish:
#
#   python -m scripts.safod_deep.plot_catalog_events --build-index-only
# ==============================================================================

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import datetime as dt
import html
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import dateutil.parser
import matplotlib

# Headless plotting on Sherlock compute nodes.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==============================================================================
# DAS-UTILITIES IMPORT
# ==============================================================================

DAS_UTILITIES_ROOT = Path(
    "/home/groups/ettore88/alina/packages/DAS-utilities"
)
DAS_UTILITIES_BUILD = DAS_UTILITIES_ROOT / "build"
DAS_UTILITIES_PYTHON = DAS_UTILITIES_ROOT / "python"

existing_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = (
    f"{existing_ld_path}:{DAS_UTILITIES_BUILD}"
    if existing_ld_path
    else str(DAS_UTILITIES_BUILD)
)

sys.path.insert(0, str(DAS_UTILITIES_BUILD))
sys.path.insert(0, str(DAS_UTILITIES_PYTHON))

import DASutils  # noqa: E402


# ==============================================================================
# DEFAULT PATHS / DISPLAY SETTINGS
# ==============================================================================

DEFAULT_CATALOG = Path(
    "results/safod_deep/catalog/catalog_recorded.csv"
)
DEFAULT_MANIFEST = Path(
    "results/safod_deep/catalog/recording_manifest.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "results/safod_deep/event_qc_adaptive"
)

GEO_XLSX = Path(
    "/home/groups/ettore88/alina/SAFOD/"
    "SAFOD_Phase2_GeoReferenced_Channels.xlsx"
)

DEFAULT_DISPLAY_TMIN_S = -0.5
DEFAULT_PCLIP = 96.0
DEFAULT_DPI = 180

# After filtering, the raster does not need the full temporal sample rate.
# The decimation guard below also enforces sufficient Nyquist margin.
MAX_DISPLAY_FS_HZ = 500.0


# ==============================================================================
# ADAPTIVE RECIPE
# ==============================================================================

@dataclass(frozen=True)
class DisplayRecipe:
    name: str
    fmin_hz: float
    fmax_hz: float
    display_tmin_s: float
    display_tmax_s: float
    filter_pad_s: float
    pclip: float


def choose_adaptive_recipe(
    min_source_cable_distance_km: float,
) -> DisplayRecipe:
    """
    Choose a first-pass catalog-QC recipe from source-cable distance.

    The time window is deliberately conservative. For intermediate/far events,
    the end time includes an approximate S-arrival allowance based on 2.7 km/s
    plus several seconds of coda. This is a display heuristic, not a phase pick.
    """
    distance_km = float(min_source_cable_distance_km)

    if not np.isfinite(distance_km):
        return DisplayRecipe(
            name="fallback_3_15Hz",
            fmin_hz=3.0,
            fmax_hz=15.0,
            display_tmin_s=DEFAULT_DISPLAY_TMIN_S,
            display_tmax_s=12.5,
            filter_pad_s=5.0,
            pclip=DEFAULT_PCLIP,
        )

    if distance_km < 8.0:
        return DisplayRecipe(
            name="near_5_30Hz",
            fmin_hz=5.0,
            fmax_hz=30.0,
            display_tmin_s=DEFAULT_DISPLAY_TMIN_S,
            display_tmax_s=8.0,
            filter_pad_s=3.0,
            pclip=DEFAULT_PCLIP,
        )

    if distance_km < 20.0:
        approximate_s_plus_coda_s = (
            distance_km / 2.7
            + 5.0
        )
        display_tmax_s = min(
            16.0,
            max(
                12.5,
                approximate_s_plus_coda_s,
            ),
        )

        return DisplayRecipe(
            name="intermediate_3_15Hz",
            fmin_hz=3.0,
            fmax_hz=15.0,
            display_tmin_s=DEFAULT_DISPLAY_TMIN_S,
            display_tmax_s=float(display_tmax_s),
            filter_pad_s=5.0,
            pclip=DEFAULT_PCLIP,
        )

    approximate_s_plus_coda_s = (
        distance_km / 2.7
        + 5.0
    )
    display_tmax_s = min(
        30.0,
        max(
            20.0,
            approximate_s_plus_coda_s,
        ),
    )

    return DisplayRecipe(
        name="far_1_12Hz",
        fmin_hz=1.0,
        fmax_hz=12.0,
        display_tmin_s=DEFAULT_DISPLAY_TMIN_S,
        display_tmax_s=float(display_tmax_s),
        filter_pad_s=10.0,
        pclip=DEFAULT_PCLIP,
    )


def resolve_recipe(
    event_row: pd.Series,
    args: argparse.Namespace,
) -> DisplayRecipe:
    """Apply optional CLI overrides to one adaptive recipe."""
    distance_km = finite_float(
        event_row.get(
            "min_3d_distance_to_cable_km",
            np.nan,
        )
    )

    base = choose_adaptive_recipe(
        distance_km
    )

    values = asdict(base)
    overridden = False

    override_map = {
        "fmin_hz": args.fmin,
        "fmax_hz": args.fmax,
        "display_tmin_s": args.display_tmin,
        "display_tmax_s": args.display_tmax,
        "filter_pad_s": args.filter_pad,
        "pclip": args.pclip,
    }

    for field_name, value in override_map.items():
        if value is not None:
            values[field_name] = float(value)
            overridden = True

    if overridden:
        values["name"] = (
            f"custom_from_{base.name}"
        )

    recipe = DisplayRecipe(
        **values
    )

    if not (
        0.0
        < recipe.fmin_hz
        < recipe.fmax_hz
    ):
        raise ValueError(
            "Resolved frequency band must satisfy "
            f"0 < fmin < fmax; got "
            f"{recipe.fmin_hz}-{recipe.fmax_hz} Hz."
        )

    if not (
        recipe.display_tmin_s
        < recipe.display_tmax_s
    ):
        raise ValueError(
            "Resolved display interval must satisfy tmin < tmax; got "
            f"{recipe.display_tmin_s} to {recipe.display_tmax_s} s."
        )

    if recipe.filter_pad_s < 0.0:
        raise ValueError(
            "filter_pad_s must be non-negative."
        )

    if not 0.0 < recipe.pclip <= 100.0:
        raise ValueError(
            f"pclip must be in (0, 100]; got {recipe.pclip}."
        )

    return recipe


# ==============================================================================
# BASIC HELPERS
# ==============================================================================

def parse_beg_time_from_info(
    info: dict,
) -> dt.datetime:
    value = info["begTime"]

    if isinstance(value, dt.datetime):
        parsed = value
    else:
        parsed = dateutil.parser.parse(
            str(value)
        )

    if parsed.tzinfo is None:
        parsed = parsed.replace(
            tzinfo=dt.timezone.utc
        )

    return parsed.astimezone(
        dt.timezone.utc
    )


def safe_token(
    value: Any,
) -> str:
    text = str(value).strip()

    return "".join(
        character
        if character.isalnum() or character in "-_"
        else "_"
        for character in text
    )


def format_number_token(
    value: float,
) -> str:
    value = float(value)

    nearest_integer = round(
        value
    )
    if math.isclose(
        value,
        nearest_integer,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        return str(
            int(nearest_integer)
        )

    return (
        f"{value:.1f}"
        .replace(".", "p")
    )


def finite_float(
    value: Any,
    default: float = np.nan,
) -> float:
    try:
        result = float(
            value
        )
    except (TypeError, ValueError):
        return float(
            default
        )

    if not np.isfinite(
        result
    ):
        return float(
            default
        )

    return result


def robust_clip(
    data: np.ndarray,
    percentile: float,
) -> float:
    clip_value = float(
        np.percentile(
            np.abs(data),
            percentile,
        )
    )

    if (
        not np.isfinite(clip_value)
        or clip_value <= 0.0
    ):
        clip_value = float(
            np.max(
                np.abs(data)
            )
        )

    if (
        not np.isfinite(clip_value)
        or clip_value <= 0.0
    ):
        clip_value = 1.0

    return clip_value


def manifest_error_is_empty(
    series: pd.Series,
) -> pd.Series:
    text = (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )
    return text.eq("")


def parse_bool_series(
    series: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.fillna(
            False
        )

    text = (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return text.isin(
        {
            "true",
            "1",
            "yes",
            "y",
        }
    )


# ==============================================================================
# CATALOG / MANIFEST LOADING
# ==============================================================================

def load_catalog(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Catalog not found: {path}"
        )

    catalog = pd.read_csv(
        path
    )

    required = {
        "event_id",
        "origin_time_utc",
        "magnitude",
        "depth_km",
        "primary_recording",
        "min_3d_distance_to_cable_km",
    }

    missing = sorted(
        required.difference(
            catalog.columns
        )
    )

    if missing:
        raise ValueError(
            f"Catalog is missing columns {missing}: {path}"
        )

    catalog = catalog.copy()

    catalog["origin_time_utc"] = pd.to_datetime(
        catalog["origin_time_utc"],
        utc=True,
        errors="coerce",
    )

    catalog = catalog[
        catalog["origin_time_utc"].notna()
    ].copy()

    if "window_covered" in catalog.columns:
        catalog = catalog[
            parse_bool_series(
                catalog["window_covered"]
            )
        ].copy()

    return (
        catalog
        .sort_values(
            [
                "origin_time_utc",
                "event_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def load_manifest(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Recording manifest not found: {path}"
        )

    manifest = pd.read_csv(
        path
    )

    required = {
        "recording_label",
        "file_path",
        "start_time_utc",
        "end_time_utc",
        "sample_rate_hz",
        "n_channels",
        "first_channel_index",
        "error",
    }

    missing = sorted(
        required.difference(
            manifest.columns
        )
    )

    if missing:
        raise ValueError(
            f"Manifest is missing columns {missing}: {path}"
        )

    manifest = manifest.copy()

    manifest["start_time_utc"] = pd.to_datetime(
        manifest["start_time_utc"],
        utc=True,
        errors="coerce",
    )
    manifest["end_time_utc"] = pd.to_datetime(
        manifest["end_time_utc"],
        utc=True,
        errors="coerce",
    )

    valid = (
        manifest["start_time_utc"].notna()
        & manifest["end_time_utc"].notna()
        & manifest_error_is_empty(
            manifest["error"]
        )
    )

    return (
        manifest.loc[
            valid
        ]
        .sort_values(
            [
                "start_time_utc",
                "file_path",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ==============================================================================
# RECORDING SELECTION / COVERAGE
# ==============================================================================

def select_event_files(
    *,
    manifest: pd.DataFrame,
    event_row: pd.Series,
    read_start_utc: pd.Timestamp,
    read_end_utc: pd.Timestamp,
) -> pd.DataFrame:
    """
    Select files only from the event's primary recording configuration.

    This prevents duplicate coverage from overlapping archived configurations
    during acquisition-setting transitions.
    """
    recording_label = str(
        event_row[
            "primary_recording"
        ]
    )

    candidates = manifest[
        manifest[
            "recording_label"
        ].astype(str)
        == recording_label
    ].copy()

    selected = candidates[
        (
            candidates[
                "end_time_utc"
            ]
            > read_start_utc
        )
        & (
            candidates[
                "start_time_utc"
            ]
            < read_end_utc
        )
    ].copy()

    if selected.empty:
        raise RuntimeError(
            "No manifest files overlap the requested event window for "
            f"recording {recording_label!r}."
        )

    return (
        selected
        .drop_duplicates(
            subset=[
                "file_path"
            ]
        )
        .sort_values(
            [
                "start_time_utc",
                "file_path",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def coverage_summary(
    selected: pd.DataFrame,
    *,
    required_start: pd.Timestamp,
    required_end: pd.Timestamp,
    tolerance_s: float = 0.01,
) -> tuple[bool, str]:
    intervals = sorted(
        [
            (
                max(
                    pd.Timestamp(
                        row.start_time_utc
                    ),
                    required_start,
                ),
                min(
                    pd.Timestamp(
                        row.end_time_utc
                    ),
                    required_end,
                ),
            )
            for row in selected.itertuples()
        ],
        key=lambda pair: pair[0],
    )

    covered_until = required_start
    tolerance = pd.to_timedelta(
        tolerance_s,
        unit="s",
    )

    for interval_start, interval_end in intervals:
        if (
            interval_start
            > covered_until + tolerance
        ):
            return (
                False,
                (
                    "gap from "
                    f"{covered_until.isoformat()} to "
                    f"{interval_start.isoformat()}"
                ),
            )

        if interval_end > covered_until:
            covered_until = interval_end

        if covered_until >= required_end:
            return True, "complete"

    return (
        False,
        (
            "coverage ends at "
            f"{covered_until.isoformat()}, before "
            f"{required_end.isoformat()}"
        ),
    )


# ==============================================================================
# DATA READING / FILTERING
# ==============================================================================

def read_event_files(
    files: list[str],
) -> tuple[np.ndarray, dict]:
    """
    Use the same OptaSense reader/conversion path as prepare_event.py.
    """
    data, info = DASutils.readFile_HDF(
        files,
        0.01,
        500.0,
        verbose=0,
        diff=True,
        detrend=False,
        tapering=False,
        filter=False,
        median=True,
        desampling=False,
        nChbuffer=1000,
        system="OptaSense",
    )

    data = np.asarray(
        data
    )

    if data.ndim != 2:
        raise ValueError(
            f"DAS reader returned shape {data.shape}; expected 2D."
        )

    return data, info


def filter_event_window(
    *,
    data: np.ndarray,
    time_s: np.ndarray,
    fs_hz: float,
    recipe: DisplayRecipe,
    zero_phase: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    nyquist_hz = (
        0.5
        * float(fs_hz)
    )

    effective_fmax_hz = min(
        recipe.fmax_hz,
        0.90 * nyquist_hz,
    )

    if not (
        0.0
        < recipe.fmin_hz
        < effective_fmax_hz
        < nyquist_hz
    ):
        raise ValueError(
            "Invalid event bandpass after Nyquist guard: "
            f"requested={recipe.fmin_hz:.3f}-"
            f"{recipe.fmax_hz:.3f} Hz, "
            f"effective fmax={effective_fmax_hz:.3f} Hz, "
            f"fs={fs_hz:.3f} Hz."
        )

    display_mask = (
        (
            time_s
            >= recipe.display_tmin_s
        )
        & (
            time_s
            <= recipe.display_tmax_s
        )
    )

    if np.count_nonzero(
        display_mask
    ) < 2:
        raise RuntimeError(
            "Loaded data do not contain the requested display interval."
        )

    filtered = DASutils.bandpass2D_c(
        np.ascontiguousarray(
            data
        ),
        recipe.fmin_hz,
        effective_fmax_hz,
        1.0 / float(fs_hz),
        zerophase=bool(
            zero_phase
        ),
    )

    filtered = np.asarray(
        filtered,
        dtype=np.float64,
    ) * 1.0e3

    return (
        filtered[
            :,
            display_mask,
        ],
        time_s[
            display_mask
        ],
        float(
            effective_fmax_hz
        ),
    )


def decimate_for_display(
    *,
    data: np.ndarray,
    time_s: np.ndarray,
    fs_hz: float,
    fmax_hz: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Integer decimation for raster display only.

    The signal is already band-limited. The guard keeps displayed Nyquist at
    least 1.2 times the upper band corner.
    """
    maximum_factor_from_band = int(
        math.floor(
            float(fs_hz)
            / (
                2.4
                * float(fmax_hz)
            )
        )
    )
    maximum_factor_from_target = int(
        math.floor(
            float(fs_hz)
            / MAX_DISPLAY_FS_HZ
        )
    )

    factor = max(
        1,
        min(
            maximum_factor_from_band,
            maximum_factor_from_target,
        ),
    )

    if factor <= 1:
        return (
            data,
            time_s,
            float(fs_hz),
        )

    return (
        data[
            :,
            ::factor,
        ],
        time_s[
            ::factor
        ],
        float(fs_hz) / factor,
    )


# ==============================================================================
# PLOTTING
# ==============================================================================

def load_reference_channel_limits(
    geometry_path: Path,
) -> tuple[float, float]:
    """
    Read the immutable physical channel range from GEO_XLSX.
    """
    if not geometry_path.exists():
        raise FileNotFoundError(
            f"Physical cable geometry not found: {geometry_path}"
        )

    geometry = pd.read_excel(
        geometry_path
    )

    if "Channel" not in geometry.columns:
        raise ValueError(
            f"Geometry file has no 'Channel' column: {geometry_path}"
        )

    channels = pd.to_numeric(
        geometry["Channel"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    channels = channels[
        np.isfinite(channels)
    ]

    if channels.size < 2:
        raise ValueError(
            "Geometry contains fewer than two finite physical channels."
        )

    return (
        float(np.min(channels)),
        float(np.max(channels)),
    )


def infer_reference_locus_offset(
    manifest: pd.DataFrame,
) -> float:
    """
    Infer the fixed interrogator-locus offset of the original standard/no-spool
    acquisition.

    In that configuration HDF5 row i matches GEO_XLSX reference channel i.
    Therefore:

        physical reference channel
        = first_channel_index + HDF5 row - reference_locus_offset

    and reference_locus_offset is the modal first_channel_index among the
    standard ~3200-channel recordings.

    The all-channel configuration normally has first_channel_index near zero
    and includes leading reference/spool loci. Applying the same offset maps
    the unchanged physical cable back onto the GEO_XLSX channel axis.
    """
    first = pd.to_numeric(
        manifest["first_channel_index"],
        errors="coerce",
    )
    n_channels = pd.to_numeric(
        manifest["n_channels"],
        errors="coerce",
    )

    standard_mask = (
        first.notna()
        & n_channels.between(
            3000,
            3300,
            inclusive="both",
        )
    )

    candidates = first[
        standard_mask
    ].to_numpy(dtype=np.float64)
    candidates = candidates[
        np.isfinite(candidates)
    ]

    if candidates.size == 0:
        raise RuntimeError(
            "Could not infer the reference-locus offset: no standard "
            "3000-3300 channel recordings have first_channel_index metadata."
        )

    rounded = np.rint(
        candidates
    ).astype(np.int64)
    values, counts = np.unique(
        rounded,
        return_counts=True,
    )
    modal_value = int(
        values[
            np.argmax(counts)
        ]
    )

    close = candidates[
        np.abs(
            candidates - modal_value
        ) <= 0.5
    ]

    if close.size == 0:
        raise RuntimeError(
            "Could not robustly estimate the reference-locus offset."
        )

    return float(
        np.median(close)
    )


def physical_reference_axis_and_crop(
    *,
    selected_manifest: pd.DataFrame,
    n_channels: int,
    reference_locus_offset: float,
    physical_channel_min: float,
    physical_channel_max: float,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    """
    Map HDF5 rows to the unchanged GEO_XLSX physical channel coordinate.

    Coordinates are kept distinct:

      data_row:
          zero-based row in the loaded HDF5 array;

      global_locus:
          first_channel_index + data_row;

      physical_reference_channel:
          global_locus - reference_locus_offset.
    """
    if int(n_channels) < 1:
        raise ValueError(
            f"n_channels must be positive; got {n_channels}."
        )

    first_values = pd.to_numeric(
        selected_manifest["first_channel_index"],
        errors="coerce",
    ).dropna().to_numpy(dtype=np.float64)

    if first_values.size == 0:
        raise RuntimeError(
            "Selected files have no finite first_channel_index metadata."
        )

    if (
        np.max(first_values)
        - np.min(first_values)
        > 0.5
    ):
        raise RuntimeError(
            "Selected files use inconsistent first_channel_index values: "
            f"{first_values.tolist()}."
        )

    first_channel_index = float(
        np.median(first_values)
    )

    data_row = np.arange(
        int(n_channels),
        dtype=np.float64,
    )
    global_locus = (
        first_channel_index
        + data_row
    )
    physical_channel = (
        global_locus
        - float(reference_locus_offset)
    )

    tolerance = 0.51
    physical_mask = (
        physical_channel
        >= float(physical_channel_min) - tolerance
    ) & (
        physical_channel
        <= float(physical_channel_max) + tolerance
    )

    if np.count_nonzero(physical_mask) < 100:
        raise RuntimeError(
            "The inferred acquisition-to-physical registration leaves fewer "
            "than 100 physical cable rows. "
            f"first_channel_index={first_channel_index}, "
            f"reference_locus_offset={reference_locus_offset}, "
            f"mapped range={physical_channel.min():.1f} to "
            f"{physical_channel.max():.1f}, "
            f"GEO_XLSX range={physical_channel_min:.1f} to "
            f"{physical_channel_max:.1f}."
        )

    return (
        physical_channel[
            physical_mask
        ],
        physical_mask,
        "Physical reference channel (GEO_XLSX)",
        first_channel_index,
    )


def event_plot_filename(
    event_row: pd.Series,
    recipe: DisplayRecipe,
) -> str:
    origin = pd.Timestamp(
        event_row[
            "origin_time_utc"
        ]
    )
    timestamp_token = origin.strftime(
        "%Y%m%dT%H%M%S"
    )
    event_id_token = safe_token(
        event_row[
            "event_id"
        ]
    )
    magnitude = finite_float(
        event_row.get(
            "magnitude",
            np.nan,
        )
    )
    band_token = (
        f"{format_number_token(recipe.fmin_hz)}_"
        f"{format_number_token(recipe.fmax_hz)}Hz"
    )

    return (
        f"{timestamp_token}_event_"
        f"{event_id_token}_"
        f"M{magnitude:.1f}_"
        f"{safe_token(recipe.name)}_"
        f"{band_token}.png"
    )


def plot_event(
    *,
    data: np.ndarray,
    time_s: np.ndarray,
    x_axis: np.ndarray,
    x_label: str,
    event_row: pd.Series,
    recording_label: str,
    recipe: DisplayRecipe,
    effective_fmax_hz: float,
    zero_phase: bool,
    out_path: Path,
    dpi: int,
) -> float:
    clip_value = robust_clip(
        data,
        recipe.pclip,
    )

    event_id = str(
        event_row[
            "event_id"
        ]
    )
    magnitude = finite_float(
        event_row.get(
            "magnitude",
            np.nan,
        )
    )
    magnitude_type = str(
        event_row.get(
            "magnitude_type",
            "",
        )
    )
    depth_km = finite_float(
        event_row.get(
            "depth_km",
            np.nan,
        )
    )
    distance_km = finite_float(
        event_row.get(
            "min_3d_distance_to_cable_km",
            np.nan,
        )
    )
    crossline_km = (
        abs(
            finite_float(
                event_row.get(
                    "source_crossline_m",
                    np.nan,
                )
            )
        )
        / 1000.0
    )
    geometry_class = str(
        event_row.get(
            "geometry_2d_class",
            "unknown",
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            20,
            11,
        )
    )

    image = ax.imshow(
        data.T,
        extent=[
            float(
                x_axis[0]
            ),
            float(
                x_axis[-1] + 1.0
            ),
            float(
                time_s[-1]
            ),
            float(
                time_s[0]
            ),
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
        linewidth=1.1,
        linestyle="--",
        label="Catalog origin",
    )

    phase_text = (
        "zero phase"
        if zero_phase
        else "causal"
    )

    ax.set_title(
        f"Real DAS event {event_id}, "
        f"M{magnitude:.1f} {magnitude_type}, "
        f"{recipe.fmin_hz:g}-"
        f"{effective_fmax_hz:g} Hz\n"
        f"depth={depth_km:.2f} km, "
        f"min source-cable distance="
        f"{distance_km:.2f} km, "
        f"|crossline|={crossline_km:.2f} km, "
        f"2-D class={geometry_class}"
    )

    ax.set_xlabel(
        x_label
    )
    ax.set_ylabel(
        "Time from catalog origin [s]"
    )
    ax.set_xlim(
        float(
            x_axis[0]
        ),
        float(
            x_axis[-1] + 1.0
        ),
    )
    ax.set_ylim(
        float(
            time_s[-1]
        ),
        float(
            time_s[0]
        ),
    )
    ax.legend(
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
    )

    annotation = (
        f"recording: {recording_label}\n"
        f"recipe: {recipe.name}\n"
        f"display: {recipe.display_tmin_s:g} to "
        f"{recipe.display_tmax_s:g} s\n"
        f"clip: ±P{recipe.pclip:g} = "
        f"{clip_value:.3g} nm/m/s\n"
        f"filter: {phase_text}"
    )

    ax.text(
        0.01,
        0.99,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={
            "facecolor": "white",
            "alpha": 0.84,
            "edgecolor": "none",
        },
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        pad=0.015,
    )
    colorbar.set_label(
        "strain rate [nm/m/s]"
    )

    fig.tight_layout()
    fig.savefig(
        out_path,
        dpi=int(
            dpi
        ),
        bbox_inches="tight",
    )
    plt.close(
        fig
    )

    return clip_value


# ==============================================================================
# PER-EVENT PROCESSING
# ==============================================================================

def process_event(
    *,
    event_index: int,
    event_row: pd.Series,
    manifest: pd.DataFrame,
    output_dir: Path,
    recipe: DisplayRecipe,
    reference_locus_offset: float,
    physical_channel_min: float,
    physical_channel_max: float,
    zero_phase: bool,
    overwrite: bool,
    dpi: int,
) -> dict[str, Any]:
    started = time.perf_counter()

    origin = pd.Timestamp(
        event_row[
            "origin_time_utc"
        ]
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

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    metadata_dir = (
        output_dir
        / "metadata"
    )
    metadata_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_path = (
        output_dir
        / event_plot_filename(
            event_row,
            recipe,
        )
    )
    metadata_path = (
        metadata_dir
        / (
            image_path.stem
            + ".json"
        )
    )

    if (
        image_path.exists()
        and metadata_path.exists()
        and not overwrite
    ):
        existing = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )
        existing["status"] = "exists"
        existing["elapsed_s"] = 0.0
        return existing

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
            "Requested padded plot window is not continuously covered: "
            f"{coverage_message}"
        )

    files = (
        selected[
            "file_path"
        ]
        .astype(str)
        .tolist()
    )

    data_full, info = read_event_files(
        files
    )

    fs_hz = float(
        info[
            "fs"
        ]
    )

    if (
        not np.isfinite(fs_hz)
        or fs_hz <= 0.0
    ):
        raise ValueError(
            f"Invalid read sampling rate: {fs_hz}"
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
        (
            time_full_s
            >= filter_tmin_s
        )
        & (
            time_full_s
            <= filter_tmax_s
        )
    )

    if np.count_nonzero(
        filter_mask
    ) < 16:
        raise RuntimeError(
            "Too few samples remain in the padded filter window."
        )

    data_filter = np.ascontiguousarray(
        data_full[
            :,
            filter_mask,
        ]
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
        display_fs_hz,
    ) = decimate_for_display(
        data=data_display,
        time_s=time_display_s,
        fs_hz=fs_hz,
        fmax_hz=effective_fmax_hz,
    )

    (
        x_axis,
        physical_row_mask,
        x_label,
        first_channel_index,
    ) = physical_reference_axis_and_crop(
        selected_manifest=selected,
        n_channels=data_display.shape[0],
        reference_locus_offset=reference_locus_offset,
        physical_channel_min=physical_channel_min,
        physical_channel_max=physical_channel_max,
    )

    # Remove leading/trailing all-channel/reference-spool loci that lie outside
    # the immutable GEO_XLSX physical cable. Every configuration is therefore
    # displayed on the same physical x coordinate.
    data_display = data_display[
        physical_row_mask,
        :,
    ]

    recording_label = str(
        event_row[
            "primary_recording"
        ]
    )

    clip_value = plot_event(
        data=data_display,
        time_s=time_display_s,
        x_axis=x_axis,
        x_label=x_label,
        event_row=event_row,
        recording_label=recording_label,
        recipe=recipe,
        effective_fmax_hz=effective_fmax_hz,
        zero_phase=zero_phase,
        out_path=image_path,
        dpi=dpi,
    )

    result = {
        "status": "ok",
        "event_index": int(
            event_index
        ),
        "event_id": str(
            event_row[
                "event_id"
            ]
        ),
        "origin_time_utc": origin.isoformat(),
        "magnitude": finite_float(
            event_row.get(
                "magnitude",
                np.nan,
            )
        ),
        "depth_km": finite_float(
            event_row.get(
                "depth_km",
                np.nan,
            )
        ),
        "min_3d_distance_to_cable_km": finite_float(
            event_row.get(
                "min_3d_distance_to_cable_km",
                np.nan,
            )
        ),
        "source_crossline_m": finite_float(
            event_row.get(
                "source_crossline_m",
                np.nan,
            )
        ),
        "geometry_2d_class": str(
            event_row.get(
                "geometry_2d_class",
                "unknown",
            )
        ),
        "recording_label": recording_label,
        "first_channel_index": float(
            first_channel_index
        ),
        "reference_locus_offset": float(
            reference_locus_offset
        ),
        "physical_channel_min": float(
            x_axis[0]
        ),
        "physical_channel_max": float(
            x_axis[-1]
        ),
        "channel_axis_definition": (
            "physical_reference_channel = "
            "first_channel_index + HDF5_data_row - "
            "reference_locus_offset"
        ),
        "recipe": asdict(
            recipe
        ),
        "effective_fmax_hz": float(
            effective_fmax_hz
        ),
        "zero_phase": bool(
            zero_phase
        ),
        "display_clip_nm_per_m_per_s": float(
            clip_value
        ),
        "files": files,
        "n_files": len(
            files
        ),
        "n_channels": int(
            data_display.shape[0]
        ),
        "read_fs_hz": float(
            fs_hz
        ),
        "display_fs_hz": float(
            display_fs_hz
        ),
        "image_path": str(
            image_path
        ),
        "elapsed_s": float(
            time.perf_counter()
            - started
        ),
    }

    metadata_path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    return result


# ==============================================================================
# HTML GALLERY
# ==============================================================================

def _catalog_lookup_by_event_id(
    catalog_path: Path,
) -> dict[str, dict[str, Any]]:
    """Build fallback metadata for gallery cards from catalog_recorded.csv."""
    if not catalog_path.exists():
        return {}

    catalog = pd.read_csv(catalog_path)

    if "event_id" not in catalog.columns:
        return {}

    lookup: dict[str, dict[str, Any]] = {}

    for _, row in catalog.iterrows():
        event_id = str(row["event_id"]).strip()
        lookup[event_id] = {
            "event_id": event_id,
            "origin_time_utc": str(row.get("origin_time_utc", "")),
            "magnitude": finite_float(row.get("magnitude", np.nan)),
            "min_3d_distance_to_cable_km": finite_float(
                row.get("min_3d_distance_to_cable_km", np.nan)
            ),
            "geometry_2d_class": str(
                row.get("geometry_2d_class", "unknown")
            ),
        }

    return lookup


def _event_id_from_image_name(
    image_name: str,
) -> str:
    """Extract event id from YYYY..._event_<ID>_M... filenames."""
    marker = "_event_"

    if marker not in image_name:
        return ""

    suffix = image_name.split(marker, 1)[1]

    if "_M" not in suffix:
        return ""

    return suffix.split("_M", 1)[0]


def build_html_gallery(
    *,
    output_dir: Path,
    catalog_path: Path = DEFAULT_CATALOG,
) -> Path:
    """
    Build the gallery from all PNG files that actually exist.

    JSON metadata are used when available but are not required. The previous
    implementation scanned only metadata/*.json, so many PNG files plus one
    JSON produced a one-card HTML page.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_dir = output_dir / "metadata"
    metadata_by_image_name: dict[str, dict[str, Any]] = {}

    if metadata_dir.exists():
        for metadata_path in sorted(metadata_dir.glob("*.json")):
            try:
                record = json.loads(
                    metadata_path.read_text(encoding="utf-8")
                )
            except Exception:
                continue

            image_value = str(record.get("image_path", "")).strip()

            if not image_value:
                continue

            metadata_by_image_name[Path(image_value).name] = record

    catalog_lookup = _catalog_lookup_by_event_id(catalog_path)
    image_paths = sorted(output_dir.glob("*.png"))

    cards: list[str] = []
    gallery_rows: list[dict[str, Any]] = []

    for image_path in image_paths:
        record = dict(
            metadata_by_image_name.get(
                image_path.name,
                {},
            )
        )

        event_id = str(record.get("event_id", "")).strip()

        if not event_id:
            event_id = _event_id_from_image_name(image_path.name)

        catalog_record = catalog_lookup.get(event_id, {})

        origin = str(
            record.get(
                "origin_time_utc",
                catalog_record.get("origin_time_utc", ""),
            )
        )
        magnitude = finite_float(
            record.get(
                "magnitude",
                catalog_record.get("magnitude", np.nan),
            )
        )
        distance_km = finite_float(
            record.get(
                "min_3d_distance_to_cable_km",
                catalog_record.get(
                    "min_3d_distance_to_cable_km",
                    np.nan,
                ),
            )
        )
        geometry_class = str(
            record.get(
                "geometry_2d_class",
                catalog_record.get(
                    "geometry_2d_class",
                    "unknown",
                ),
            )
        )

        recipe = record.get("recipe", {})
        recipe_name = (
            str(recipe.get("name", ""))
            if isinstance(recipe, dict)
            else str(recipe)
        )

        image_name_html = html.escape(image_path.name)
        event_id_html = html.escape(event_id or "unknown")
        origin_html = html.escape(origin)
        geometry_html = html.escape(geometry_class)
        recipe_html = html.escape(recipe_name)

        cards.append(
            f"""
            <article class="card">
              <a href="{image_name_html}">
                <img loading="lazy"
                     src="{image_name_html}"
                     alt="Event {event_id_html}">
              </a>
              <div class="meta">
                <strong>{event_id_html}</strong>
                &nbsp; M{magnitude:.1f}
                &nbsp; d={distance_km:.2f} km
                &nbsp; 2-D={geometry_html}<br>
                <span>{origin_html}</span><br>
                <span>{recipe_html}</span>
              </div>
            </article>
            """
        )

        gallery_rows.append(
            {
                "image_name": image_path.name,
                "event_id": event_id,
                "origin_time_utc": origin,
                "magnitude": magnitude,
                "min_3d_distance_to_cable_km": distance_km,
                "geometry_2d_class": geometry_class,
                "recipe_name": recipe_name,
                "has_json_metadata": (
                    image_path.name in metadata_by_image_name
                ),
            }
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SAFOD deep-DAS adaptive event QC</title>
<style>
body {{
  font-family: system-ui, sans-serif;
  margin: 20px;
  background: #f5f5f5;
}}
h1 {{
  margin-bottom: 6px;
}}
.grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(430px, 1fr));
  gap: 16px;
}}
.card {{
  background: white;
  border: 1px solid #ddd;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 1px 4px rgba(0,0,0,.08);
}}
.card img {{
  display: block;
  width: 100%;
  height: auto;
}}
.meta {{
  padding: 10px 12px;
  line-height: 1.4;
}}
.meta span {{
  color: #555;
  font-size: 0.9em;
}}
</style>
</head>
<body>
<h1>SAFOD deep-DAS adaptive event QC</h1>
<p>{len(cards)} rendered PNG files. Click a thumbnail for the full image.</p>
<div class="grid">
{''.join(cards)}
</div>
</body>
</html>
"""

    index_path = output_dir / "index.html"
    index_path.write_text(page, encoding="utf-8")

    pd.DataFrame(gallery_rows).to_csv(
        output_dir / "gallery_manifest.csv",
        index=False,
    )

    print("Gallery inventory")
    print("-----------------")
    print(f"PNG files found       : {len(image_paths)}")
    print(
        "JSON metadata matched : "
        f"{sum(row['has_json_metadata'] for row in gallery_rows)}"
    )
    print(f"HTML cards written    : {len(cards)}")

    return index_path


# ==============================================================================
# CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot adaptive full-cable DAS QC gathers for events in "
            "catalog_recorded.csv."
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
        default=DEFAULT_OUTPUT_DIR,
    )

    # Optional overrides. None means use the adaptive value for each event.
    parser.add_argument(
        "--fmin",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--display-tmin",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--display-tmax",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--filter-pad",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--pclip",
        type=float,
        default=None,
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--event-id",
        default=None,
        help="Process one exact catalog event id.",
    )
    parser.add_argument(
        "--min-magnitude",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--geometry-class",
        choices=[
            "good",
            "borderline",
            "poor",
        ],
        default=None,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N selected events.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of independent processing shards.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help=(
            "Zero-based shard number. If omitted, SLURM_ARRAY_TASK_ID is "
            "used when present; otherwise 0."
        ),
    )
    parser.add_argument(
        "--build-index-only",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_shards < 1:
        raise ValueError(
            "--num-shards must be >= 1."
        )

    if args.shard_index is None:
        shard_index = int(
            os.environ.get(
                "SLURM_ARRAY_TASK_ID",
                "0",
            )
        )
    else:
        shard_index = int(
            args.shard_index
        )

    if not (
        0
        <= shard_index
        < args.num_shards
    ):
        raise ValueError(
            f"shard index {shard_index} is outside "
            f"[0, {args.num_shards})."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.build_index_only:
        index_path = build_html_gallery(
            output_dir=args.output_dir,
            catalog_path=args.catalog,
        )
        print(
            f"Saved gallery: {index_path}"
        )
        return

    catalog = load_catalog(
        args.catalog
    )

    if args.event_id is not None:
        catalog = catalog[
            catalog[
                "event_id"
            ].astype(str)
            == str(
                args.event_id
            )
        ].copy()

    if args.min_magnitude is not None:
        catalog = catalog[
            pd.to_numeric(
                catalog[
                    "magnitude"
                ],
                errors="coerce",
            )
            >= float(
                args.min_magnitude
            )
        ].copy()

    if args.geometry_class is not None:
        catalog = catalog[
            catalog[
                "geometry_2d_class"
            ].astype(str)
            == args.geometry_class
        ].copy()

    catalog = catalog.reset_index(
        drop=True
    )

    if args.limit is not None:
        catalog = catalog.iloc[
            : int(
                args.limit
            )
        ].copy()

    manifest = load_manifest(
        args.manifest
    )

    (
        physical_channel_min,
        physical_channel_max,
    ) = load_reference_channel_limits(
        GEO_XLSX
    )
    reference_locus_offset = infer_reference_locus_offset(
        manifest
    )

    selected_positions = [
        position
        for position in range(
            len(catalog)
        )
        if (
            position
            % args.num_shards
            == shard_index
        )
    ]

    print("\nSAFOD adaptive catalog event plotting")
    print("-------------------------------------")
    print(
        f"catalog events selected : {len(catalog)}"
    )
    print(
        f"num shards              : {args.num_shards}"
    )
    print(
        f"this shard              : {shard_index}"
    )
    print(
        f"events in this shard    : {len(selected_positions)}"
    )
    print(
        "filter default          : "
        f"{'zero phase' if args.zero_phase else 'causal'}"
    )
    print(
        f"output                  : {args.output_dir}"
    )
    print(
        "reference locus offset  : "
        f"{reference_locus_offset:.1f}"
    )
    print(
        "physical channel range  : "
        f"{physical_channel_min:.1f} to "
        f"{physical_channel_max:.1f}"
    )

    statuses: list[
        dict[str, Any]
    ] = []
    total_started = time.perf_counter()

    for local_count, position in enumerate(
        selected_positions,
        start=1,
    ):
        event_row = catalog.iloc[
            position
        ]
        recipe = resolve_recipe(
            event_row,
            args,
        )

        print(
            f"\n[{local_count}/{len(selected_positions)}] "
            f"event={event_row['event_id']} "
            f"origin={event_row['origin_time_utc']} "
            f"M={finite_float(event_row.get('magnitude', np.nan)):.1f}"
        )
        print(
            f"recipe={recipe.name}, "
            f"band={recipe.fmin_hz:g}-{recipe.fmax_hz:g} Hz, "
            f"display={recipe.display_tmin_s:g} to "
            f"{recipe.display_tmax_s:g} s, "
            f"pad={recipe.filter_pad_s:g} s"
        )

        try:
            result = process_event(
                event_index=position,
                event_row=event_row,
                manifest=manifest,
                output_dir=args.output_dir,
                recipe=recipe,
                reference_locus_offset=reference_locus_offset,
                physical_channel_min=physical_channel_min,
                physical_channel_max=physical_channel_max,
                zero_phase=args.zero_phase,
                overwrite=args.overwrite,
                dpi=args.dpi,
            )

            print(
                f"status={result['status']}, "
                f"elapsed={result['elapsed_s']:.1f} s"
            )

        except Exception as exc:
            result = {
                "status": "error",
                "event_index": int(
                    position
                ),
                "event_id": str(
                    event_row[
                        "event_id"
                    ]
                ),
                "origin_time_utc": str(
                    event_row[
                        "origin_time_utc"
                    ]
                ),
                "recipe": asdict(
                    recipe
                ),
                "error_type": type(
                    exc
                ).__name__,
                "error": str(
                    exc
                ),
            }

            print(
                f"ERROR: {type(exc).__name__}: {exc}"
            )

        statuses.append(
            result
        )

    status_path = (
        args.output_dir
        / (
            f"status_shard_"
            f"{shard_index:03d}_of_"
            f"{args.num_shards:03d}.csv"
        )
    )

    pd.DataFrame(
        statuses
    ).to_csv(
        status_path,
        index=False,
    )

    elapsed_s = (
        time.perf_counter()
        - total_started
    )

    successful_count = sum(
        status.get(
            "status"
        )
        in {
            "ok",
            "exists",
        }
        for status in statuses
    )
    error_count = sum(
        status.get(
            "status"
        )
        == "error"
        for status in statuses
    )

    print("\nShard summary")
    print("-------------")
    print(
        f"successful/existing : {successful_count}"
    )
    print(
        f"errors              : {error_count}"
    )
    print(
        f"elapsed             : {elapsed_s / 60.0:.1f} min"
    )
    print(
        f"status CSV          : {status_path}"
    )

    if args.num_shards == 1:
        index_path = build_html_gallery(
            output_dir=args.output_dir,
            catalog_path=args.catalog,
        )
        print(
            f"HTML gallery        : {index_path}"
        )


if __name__ == "__main__":
    main()