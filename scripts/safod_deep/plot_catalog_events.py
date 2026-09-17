#!/usr/bin/env python3
"""
SAFOD deep-DAS catalog QC with one immutable physical cable geometry.

Physical geometry is read only from SAFOD_Phase2_GeoReferenced_Channels.xlsx.
Each acquisition configuration is identified from manifest metadata and only
its turnaround DataRow is calibrated. Scientific x-axis is physical cumulative
distance along the unchanged borehole fibre [m].

Future interrogator changes do not require source-code changes: rerun
--prepare-geometry after updating the manifest/catalog.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import datetime as dt
import hashlib
import html
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

import dateutil.parser
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DAS_UTILITIES_ROOT = Path("/home/groups/ettore88/alina/packages/DAS-utilities")
DAS_UTILITIES_BUILD = DAS_UTILITIES_ROOT / "build"
DAS_UTILITIES_PYTHON = DAS_UTILITIES_ROOT / "python"
_old_ld = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = (
    f"{_old_ld}:{DAS_UTILITIES_BUILD}" if _old_ld else str(DAS_UTILITIES_BUILD)
)
sys.path.insert(0, str(DAS_UTILITIES_BUILD))
sys.path.insert(0, str(DAS_UTILITIES_PYTHON))
import DASutils  # noqa: E402

DEFAULT_CATALOG = Path("results/safod_deep/catalog/catalog_recorded.csv")
DEFAULT_MANIFEST = Path("results/safod_deep/catalog/recording_manifest.csv")
DEFAULT_OUTPUT_DIR = Path("results/safod_deep/event_qc")
DEFAULT_GEOMETRY_DIR = Path("results/safod_deep/geometry")
GEO_XLSX = Path(
    "/home/groups/ettore88/alina/SAFOD/SAFOD_Phase2_GeoReferenced_Channels.xlsx"
)

REGISTRY_NAME = "acquisition_registration.json"
REGISTRY_QC_NAME = "acquisition_registration_qc.png"
REGISTRY_VERSION = 2

DEFAULT_DISPLAY_TMIN_S = -0.5
DEFAULT_PCLIP = 96.0
DEFAULT_DPI = 180
MAX_DISPLAY_FS_HZ = 500.0

DEFAULT_CALIBRATION_ATTEMPTS = 8
DEFAULT_TARGET_CALIBRATIONS = 3
DEFAULT_MIN_CALIBRATION_SCORE = 0.30
DEFAULT_MAX_PEAK_DEVIATION_ROWS = 2.0

SINGLE_EVENT_MIN_SCORE = 0.85
SINGLE_EVENT_MIN_ABS_CORR = 0.80
SINGLE_EVENT_MIN_ENV_CORR = 0.80


@dataclass(frozen=True)
class DisplayRecipe:
    name: str
    fmin_hz: float
    fmax_hz: float
    display_tmin_s: float
    display_tmax_s: float
    filter_pad_s: float
    pclip: float


@dataclass(frozen=True)
class AcquisitionSignature:
    n_channels: int
    channel_spacing_m: float
    gauge_length_m: float
    first_channel_index: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_channels": int(self.n_channels),
            "channel_spacing_m": float(self.channel_spacing_m),
            "gauge_length_m": float(self.gauge_length_m),
            "first_channel_index": (
                None if self.first_channel_index is None
                else float(self.first_channel_index)
            ),
        }

    @property
    def config_id(self) -> str:
        first = (
            "na" if self.first_channel_index is None
            else number_token(self.first_channel_index, 3)
        )
        return (
            f"n{self.n_channels}"
            f"_dx{number_token(self.channel_spacing_m, 6)}"
            f"_gl{number_token(self.gauge_length_m, 6)}"
            f"_first{first}"
        )

    @property
    def family_id(self) -> str:
        return (
            f"dx{number_token(self.channel_spacing_m, 6)}"
            f"_gl{number_token(self.gauge_length_m, 6)}"
        )


@dataclass(frozen=True)
class RegisteredAxis:
    data_rows: np.ndarray
    fibre_distance_m: np.ndarray
    turnaround_m: float
    metadata: dict[str, Any]


def finite_float(value: Any, default: float = np.nan) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float(default)
    return x if np.isfinite(x) else float(default)


def safe_token(value: Any) -> str:
    return "".join(
        c if c.isalnum() or c in "-_" else "_"
        for c in str(value).strip()
    )


def number_token(value: float, decimals: int) -> str:
    text = f"{float(value):.{decimals}f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def format_number_token(value: float) -> str:
    value = float(value)
    nearest = round(value)
    if math.isclose(value, nearest, rel_tol=0.0, abs_tol=1.0e-9):
        return str(int(nearest))
    return f"{value:.1f}".replace(".", "p")


def parse_beg_time_from_info(info: dict) -> dt.datetime:
    value = info["begTime"]
    parsed = value if isinstance(value, dt.datetime) else dateutil.parser.parse(str(value))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def manifest_error_is_empty(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().eq("")


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.fillna("").astype(str).str.strip().str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def robust_clip(data: np.ndarray, percentile: float) -> float:
    clip = float(np.percentile(np.abs(data), percentile))
    if not np.isfinite(clip) or clip <= 0.0:
        clip = float(np.max(np.abs(data)))
    return clip if np.isfinite(clip) and clip > 0.0 else 1.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_adaptive_recipe(distance_km: float) -> DisplayRecipe:
    distance_km = float(distance_km)

    if not np.isfinite(distance_km):
        return DisplayRecipe(
            "fallback_3_15Hz", 3.0, 15.0,
            DEFAULT_DISPLAY_TMIN_S, 12.5, 5.0, DEFAULT_PCLIP
        )
    if distance_km < 8.0:
        return DisplayRecipe(
            "near_5_30Hz", 5.0, 30.0,
            DEFAULT_DISPLAY_TMIN_S, 8.0, 3.0, DEFAULT_PCLIP
        )
    if distance_km < 20.0:
        tmax = min(16.0, max(12.5, distance_km / 2.7 + 5.0))
        return DisplayRecipe(
            "intermediate_3_15Hz", 3.0, 15.0,
            DEFAULT_DISPLAY_TMIN_S, float(tmax), 5.0, DEFAULT_PCLIP
        )

    tmax = min(30.0, max(20.0, distance_km / 2.7 + 5.0))
    return DisplayRecipe(
        "far_1_12Hz", 1.0, 12.0,
        DEFAULT_DISPLAY_TMIN_S, float(tmax), 10.0, DEFAULT_PCLIP
    )


def resolve_recipe(event_row: pd.Series, args: argparse.Namespace) -> DisplayRecipe:
    base = choose_adaptive_recipe(
        finite_float(event_row.get("min_3d_distance_to_cable_km", np.nan))
    )
    values = asdict(base)
    overridden = False

    for key, value in {
        "fmin_hz": args.fmin,
        "fmax_hz": args.fmax,
        "display_tmin_s": args.display_tmin,
        "display_tmax_s": args.display_tmax,
        "filter_pad_s": args.filter_pad,
        "pclip": args.pclip,
    }.items():
        if value is not None:
            values[key] = float(value)
            overridden = True

    if overridden:
        values["name"] = f"custom_from_{base.name}"

    recipe = DisplayRecipe(**values)

    if not 0.0 < recipe.fmin_hz < recipe.fmax_hz:
        raise ValueError("Frequency band must satisfy 0 < fmin < fmax.")
    if not recipe.display_tmin_s < recipe.display_tmax_s:
        raise ValueError("Display interval must satisfy tmin < tmax.")
    if recipe.filter_pad_s < 0.0:
        raise ValueError("filter_pad_s must be non-negative.")
    if not 0.0 < recipe.pclip <= 100.0:
        raise ValueError("pclip must be in (0,100].")
    return recipe


def load_catalog(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    catalog = pd.read_csv(path)
    required = {
        "event_id", "origin_time_utc", "magnitude", "depth_km",
        "primary_recording", "min_3d_distance_to_cable_km",
    }
    missing = sorted(required.difference(catalog.columns))
    if missing:
        raise ValueError(f"Catalog missing columns {missing}: {path}")

    catalog = catalog.copy()
    catalog["origin_time_utc"] = pd.to_datetime(
        catalog["origin_time_utc"], utc=True, errors="coerce"
    )
    catalog = catalog[catalog["origin_time_utc"].notna()].copy()

    if "window_covered" in catalog.columns:
        catalog = catalog[parse_bool_series(catalog["window_covered"])].copy()

    return catalog.sort_values(
        ["origin_time_utc", "event_id"]
    ).reset_index(drop=True)


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    manifest = pd.read_csv(path)
    required = {
        "recording_label", "file_path", "start_time_utc", "end_time_utc",
        "sample_rate_hz", "n_channels", "first_channel_index",
        "channel_spacing_m", "gauge_length_m", "error",
    }
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Manifest missing columns {missing}: {path}")

    manifest = manifest.copy()
    manifest["start_time_utc"] = pd.to_datetime(
        manifest["start_time_utc"], utc=True, errors="coerce"
    )
    manifest["end_time_utc"] = pd.to_datetime(
        manifest["end_time_utc"], utc=True, errors="coerce"
    )

    valid = (
        manifest["start_time_utc"].notna()
        & manifest["end_time_utc"].notna()
        & manifest_error_is_empty(manifest["error"])
    )

    return manifest.loc[valid].sort_values(
        ["start_time_utc", "file_path"]
    ).reset_index(drop=True)


def signature_from_values(
    *,
    n_channels: Any,
    channel_spacing_m: Any,
    gauge_length_m: Any,
    first_channel_index: Any,
) -> AcquisitionSignature:
    n = finite_float(n_channels)
    dx = finite_float(channel_spacing_m)
    gl = finite_float(gauge_length_m)
    first = finite_float(first_channel_index)

    if not np.isfinite(n) or int(round(n)) < 1:
        raise ValueError(f"Invalid n_channels={n_channels!r}.")
    if not np.isfinite(dx) or dx <= 0.0:
        raise ValueError(f"Invalid channel_spacing_m={channel_spacing_m!r}.")
    if not np.isfinite(gl) or gl <= 0.0:
        raise ValueError(f"Invalid gauge_length_m={gauge_length_m!r}.")

    return AcquisitionSignature(
        n_channels=int(round(n)),
        channel_spacing_m=round(float(dx), 6),
        gauge_length_m=round(float(gl), 6),
        first_channel_index=(
            None if not np.isfinite(first) else round(float(first), 3)
        ),
    )


def annotate_manifest_configurations(manifest: pd.DataFrame) -> pd.DataFrame:
    output = manifest.copy()
    ids = []
    errors = []

    for row in output.itertuples():
        try:
            sig = signature_from_values(
                n_channels=row.n_channels,
                channel_spacing_m=row.channel_spacing_m,
                gauge_length_m=row.gauge_length_m,
                first_channel_index=row.first_channel_index,
            )
            ids.append(sig.config_id)
            errors.append("")
        except Exception as exc:
            ids.append("")
            errors.append(f"{type(exc).__name__}: {exc}")

    output["_config_id"] = ids
    output["_config_error"] = errors
    return output


def configuration_signature_from_rows(rows: pd.DataFrame) -> AcquisitionSignature:
    if rows.empty:
        raise RuntimeError("Empty manifest selection.")

    ids = [
        x for x in rows["_config_id"].fillna("").astype(str).str.strip().unique()
        if x
    ]
    if len(ids) != 1:
        raise RuntimeError(
            f"Selected files do not have one acquisition configuration: {ids}."
        )

    row = rows.iloc[0]
    sig = signature_from_values(
        n_channels=row["n_channels"],
        channel_spacing_m=row["channel_spacing_m"],
        gauge_length_m=row["gauge_length_m"],
        first_channel_index=row["first_channel_index"],
    )
    if sig.config_id != ids[0]:
        raise RuntimeError("Internal signature mismatch.")
    return sig


def event_configuration(
    event_row: pd.Series,
    manifest: pd.DataFrame,
) -> AcquisitionSignature:
    primary_file = str(event_row.get("primary_file", "")).strip()

    if primary_file:
        exact = manifest[manifest["file_path"].astype(str) == primary_file]
        if not exact.empty:
            return configuration_signature_from_rows(exact)

    origin = pd.Timestamp(event_row["origin_time_utc"])
    label = str(event_row["primary_recording"])

    candidates = manifest[
        (manifest["recording_label"].astype(str) == label)
        & (manifest["start_time_utc"] <= origin)
        & (manifest["end_time_utc"] > origin)
    ].copy()

    if candidates.empty:
        raise RuntimeError(
            f"Cannot map event {event_row['event_id']} to acquisition configuration."
        )

    candidates = candidates.sort_values(
        "start_time_utc", ascending=False
    ).reset_index(drop=True)

    cid = str(candidates.iloc[0]["_config_id"])
    if not cid:
        raise RuntimeError("Event configuration metadata are invalid.")

    return configuration_signature_from_rows(
        candidates[candidates["_config_id"] == cid]
    )


def select_event_files(
    *,
    manifest: pd.DataFrame,
    event_row: pd.Series,
    read_start_utc: pd.Timestamp,
    read_end_utc: pd.Timestamp,
) -> pd.DataFrame:
    label = str(event_row["primary_recording"])

    selected = manifest[
        (manifest["recording_label"].astype(str) == label)
        & (manifest["end_time_utc"] > read_start_utc)
        & (manifest["start_time_utc"] < read_end_utc)
    ].copy()

    if selected.empty:
        raise RuntimeError(f"No files overlap event window for {label!r}.")

    selected = selected.drop_duplicates("file_path").sort_values(
        ["start_time_utc", "file_path"]
    ).reset_index(drop=True)

    configuration_signature_from_rows(selected)
    return selected


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
                max(pd.Timestamp(r.start_time_utc), required_start),
                min(pd.Timestamp(r.end_time_utc), required_end),
            )
            for r in selected.itertuples()
        ],
        key=lambda p: p[0],
    )

    covered_until = required_start
    tolerance = pd.to_timedelta(tolerance_s, unit="s")

    for start, end in intervals:
        if start > covered_until + tolerance:
            return False, f"gap {covered_until.isoformat()} -> {start.isoformat()}"
        if end > covered_until:
            covered_until = end
        if covered_until >= required_end:
            return True, "complete"

    return False, f"coverage ends at {covered_until.isoformat()}"


def calibration_file_for_event(
    *,
    event_row: pd.Series,
    manifest: pd.DataFrame,
    signature: AcquisitionSignature,
) -> Path:
    origin = pd.Timestamp(event_row["origin_time_utc"])
    required_start = origin - pd.to_timedelta(0.5, unit="s")
    required_end = origin + pd.to_timedelta(2.2, unit="s")

    candidates = manifest[
        (manifest["_config_id"] == signature.config_id)
        & (manifest["start_time_utc"] <= required_start)
        & (manifest["end_time_utc"] >= required_end)
    ].copy()

    if candidates.empty:
        raise RuntimeError(
            "No single HDF5 contains calibration window with margin."
        )

    candidates = candidates.sort_values(
        "start_time_utc", ascending=False
    ).reset_index(drop=True)

    path = Path(str(candidates.iloc[0]["file_path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_reference_geometry(path: Path) -> dict[str, Any]:
    """
    Load the ONE authoritative physical cable geometry.

    The scientific borehole-fibre coordinate is constructed from the workbook:
      down-leg: s = MD
      up-leg:   s = 2*MD_bottom - MD

    Surface spool is excluded. Repeated co-located up-leg MD=0 channels are
    collapsed at the first surface channel reached by the borehole trajectory.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    g = pd.read_excel(path).copy()
    required = {
        "Channel", "Section", "MD_m", "TVD_m", "Lat_WGS84", "Lon_WGS84"
    }
    missing = sorted(required.difference(g.columns))
    if missing:
        raise ValueError(f"Reference geometry missing {missing}: {path}")

    for col in [
        "Channel", "MD_m", "TVD_m", "Lat_WGS84", "Lon_WGS84",
        "UTM_E_m", "UTM_N_m",
    ]:
        if col in g.columns:
            g[col] = pd.to_numeric(g[col], errors="coerce")

    g["Section"] = g["Section"].astype(str).str.strip()
    g = (
        g.dropna(subset=["Channel", "MD_m", "TVD_m"])
        .sort_values("Channel")
        .drop_duplicates("Channel")
        .reset_index(drop=True)
    )

    down = g[g["Section"] == "Down-leg"].copy().sort_values("Channel")
    up = g[g["Section"] == "Up-leg"].copy().sort_values("Channel")
    spool = g[g["Section"] == "Surface Spool"].copy().sort_values("Channel")

    if down.empty or up.empty:
        raise RuntimeError("Reference geometry lacks Down-leg or Up-leg.")

    bottom = down.loc[down["MD_m"].idxmax()].copy()
    bottom_md = float(bottom["MD_m"])
    bottom_tvd = float(bottom["TVD_m"])
    turn_channel = float(bottom["Channel"])

    surface_md = float(up["MD_m"].min())
    up_surface_channel = float(
        up[np.isclose(up["MD_m"], surface_md, atol=1e-9, rtol=0.0)]["Channel"].min()
    )

    # Only the resolved borehole path. Channels after the first MD=0 up-leg
    # point are co-located at surface and add no physical borehole distance.
    up_borehole = up[up["Channel"] <= up_surface_channel].copy()
    down_borehole = down.copy()

    down_borehole["fibre_distance_m"] = down_borehole["MD_m"].astype(float)
    up_borehole["fibre_distance_m"] = (
        2.0 * bottom_md - up_borehole["MD_m"].astype(float)
    )

    borehole = (
        pd.concat([down_borehole, up_borehole], ignore_index=True, sort=False)
        .sort_values("Channel")
        .reset_index(drop=True)
    )

    reference_channels = borehole["Channel"].to_numpy(float)
    reference_fibre = borehole["fibre_distance_m"].to_numpy(float)

    if np.any(np.diff(reference_channels) <= 0.0):
        raise RuntimeError("Reference borehole Channel axis is not increasing.")
    if np.any(np.diff(reference_fibre) <= 0.0):
        raise RuntimeError("Reference physical fibre axis is not increasing.")

    full_channel_min = float(g["Channel"].min())
    full_channel_max = float(g["Channel"].max())

    return {
        "geometry": g,
        "spool": spool,
        "down": down_borehole,
        "up": up_borehole,
        "borehole": borehole,
        "reference_channels": reference_channels,
        "reference_fibre_m": reference_fibre,
        "reference_borehole_channel_min": float(reference_channels.min()),
        "reference_borehole_channel_max": float(reference_channels.max()),
        "full_channel_min": full_channel_min,
        "full_channel_max": full_channel_max,
        "full_channel_span": full_channel_max - full_channel_min,
        "bottom_md_m": bottom_md,
        "bottom_tvd_m": bottom_tvd,
        "turn_channel": turn_channel,
        "up_surface_channel": up_surface_channel,
        "full_borehole_fibre_length_m": 2.0 * bottom_md,
        "geometry_sha256": sha256_file(path),
    }


def candidate_events_for_config(
    *,
    catalog: pd.DataFrame,
    manifest: pd.DataFrame,
    signature: AcquisitionSignature,
) -> pd.DataFrame:
    """
    Rank calibration events for mirrored down/up coherence.

    Geometry matters more than magnitude for this registration. Nearby events
    are therefore tried first; magnitude breaks ties inside distance tiers.
    """
    rows = []

    for idx, event in catalog.iterrows():
        try:
            sig = event_configuration(event, manifest)
        except Exception:
            continue

        if sig.config_id != signature.config_id:
            continue

        mag = finite_float(event.get("magnitude", np.nan), -99.0)
        dist = finite_float(
            event.get("min_3d_distance_to_cable_km", np.nan), 999.0
        )

        if dist < 5.0:
            distance_tier = 0
        elif dist < 10.0:
            distance_tier = 1
        elif dist < 20.0:
            distance_tier = 2
        else:
            distance_tier = 3

        rows.append(
            {
                "catalog_index": int(idx),
                "event_id": str(event["event_id"]),
                "magnitude": mag,
                "distance_km": dist,
                "distance_tier": distance_tier,
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["distance_tier", "magnitude", "distance_km"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def run_one_calibration(
    *,
    event_row: pd.Series,
    manifest: pd.DataFrame,
    signature: AcquisitionSignature,
    reference_xlsx: Path,
    output_dir: Path,
) -> dict[str, Any]:
    h5 = calibration_file_for_event(
        event_row=event_row,
        manifest=manifest,
        signature=signature,
    )
    origin = pd.Timestamp(event_row["origin_time_utc"]).isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.safod.calibrate_channel_registration",
            "--h5", str(h5),
            "--origin-time", origin,
            "--reference-xlsx", str(reference_xlsx),
            "--out-dir", str(output_dir),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "calibrate_channel_registration failed:\n" + result.stderr[-3000:]
        )

    reg = json.loads(
        (output_dir / "channel_registration.json").read_text()
    )
    scores = pd.read_csv(
        output_dir / "turnaround_score.csv"
    )[["centre_row", "score"]].copy()

    scores["centre_row"] = pd.to_numeric(scores["centre_row"], errors="coerce")
    scores["score"] = pd.to_numeric(scores["score"], errors="coerce")
    scores = scores.replace([np.inf, -np.inf], np.nan).dropna().sort_values(
        "centre_row"
    )

    if len(scores) < 5:
        raise RuntimeError("Too few finite turnaround scores.")

    imax = scores["score"].idxmax()
    peak_row = float(scores.loc[imax, "centre_row"])
    peak_score = float(scores.loc[imax, "score"])
    baseline = float(scores["score"].median())
    scale = peak_score - baseline
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("Invalid score normalization.")

    if not np.isclose(
        float(reg["new_channel_spacing_m"]),
        signature.channel_spacing_m,
        atol=1e-5,
        rtol=0.0,
    ):
        raise RuntimeError("Calibration dx differs from manifest signature.")

    if not np.isclose(
        float(reg["new_gauge_length_m"]),
        signature.gauge_length_m,
        atol=1e-5,
        rtol=0.0,
    ):
        raise RuntimeError("Calibration GL differs from manifest signature.")

    return {
        "event_id": str(event_row["event_id"]),
        "origin_time_utc": origin,
        "h5": str(h5),
        "peak_row": peak_row,
        "peak_score": peak_score,
        "median_abs_signed_corr": float(
            reg.get("median_abs_signed_corr", np.nan)
        ),
        "median_envelope_corr": float(
            reg.get("median_envelope_corr", np.nan)
        ),
        "n_pairs": int(reg.get("n_pairs", 0)),
        "curve": pd.DataFrame(
            {
                "centre_row": scores["centre_row"].to_numpy(float),
                "normalized_score": ((scores["score"] - baseline) / scale).to_numpy(float),
            }
        ),
    }


def stack_calibration_curves(
    calibrations: list[dict[str, Any]],
) -> tuple[pd.DataFrame, float, float]:
    stack = None

    for item in calibrations:
        col = "event_" + safe_token(item["event_id"])
        curve = item["curve"].rename(columns={"normalized_score": col})
        stack = curve if stack is None else stack.merge(
            curve, on="centre_row", how="inner"
        )

    if stack is None or len(stack) < 5:
        raise RuntimeError("Insufficient common calibration support.")

    event_cols = [c for c in stack.columns if c != "centre_row"]
    stack["stack_score"] = stack[event_cols].mean(axis=1)
    stack = stack.sort_values("centre_row").reset_index(drop=True)

    grid_i = stack["stack_score"].idxmax()
    grid_peak = float(stack.loc[grid_i, "centre_row"])
    canonical = grid_peak

    local = stack[
        (stack["centre_row"] >= grid_peak - 1.5)
        & (stack["centre_row"] <= grid_peak + 1.5)
    ]
    if len(local) >= 3:
        x = local["centre_row"].to_numpy(float)
        y = local["stack_score"].to_numpy(float)
        a, b, _ = np.polyfit(x, y, 2)
        if np.isfinite(a) and np.isfinite(b) and a < 0.0:
            vertex = float(-b / (2.0 * a))
            if x.min() <= vertex <= x.max():
                canonical = vertex

    return stack, canonical, grid_peak


def calibrate_configuration(
    *,
    signature: AcquisitionSignature,
    catalog: pd.DataFrame,
    manifest: pd.DataFrame,
    reference: dict[str, Any],
    reference_xlsx: Path,
    temporary_root: Path,
    max_attempts: int,
    target_calibrations: int,
    min_score: float,
    max_peak_deviation_rows: float,
) -> dict[str, Any]:
    candidates = candidate_events_for_config(
        catalog=catalog,
        manifest=manifest,
        signature=signature,
    )

    attempts = []
    qualified = []

    if candidates.empty:
        return {
            "status": "unresolved",
            "reason": "no recorded events use this configuration",
            "config_id": signature.config_id,
            "signature": signature.to_dict(),
            "attempts": [],
        }

    # Deliberately use all allowed attempts. Do not stop merely because several
    # events exceed a weak absolute score threshold: they may form different
    # spurious peak clusters. The consistent cluster is chosen afterwards.
    for attempt_no, candidate in enumerate(
        candidates.itertuples(index=False),
        start=1,
    ):
        if attempt_no > max_attempts:
            break

        event = catalog.iloc[int(candidate.catalog_index)]
        event_id = str(event["event_id"])

        print(
            f"    attempt {attempt_no}: event={event_id}, "
            f"M={candidate.magnitude:.1f}, d={candidate.distance_km:.2f} km"
        )

        try:
            result = run_one_calibration(
                event_row=event,
                manifest=manifest,
                signature=signature,
                reference_xlsx=reference_xlsx,
                output_dir=(
                    temporary_root
                    / signature.config_id
                    / f"event_{safe_token(event_id)}"
                ),
            )

            passes = result["peak_score"] >= min_score
            attempts.append(
                {
                    "event_id": event_id,
                    "origin_time_utc": result["origin_time_utc"],
                    "h5": result["h5"],
                    "peak_row": float(result["peak_row"]),
                    "peak_score": float(result["peak_score"]),
                    "median_abs_signed_corr": float(result["median_abs_signed_corr"]),
                    "median_envelope_corr": float(result["median_envelope_corr"]),
                    "n_pairs": int(result["n_pairs"]),
                    "passes_score": bool(passes),
                    "accepted_final": False,
                    "error": "",
                }
            )
            if passes:
                qualified.append(result)

        except Exception as exc:
            attempts.append(
                {
                    "event_id": event_id,
                    "origin_time_utc": pd.Timestamp(
                        event["origin_time_utc"]
                    ).isoformat(),
                    "h5": None,
                    "peak_row": None,
                    "peak_score": None,
                    "median_abs_signed_corr": None,
                    "median_envelope_corr": None,
                    "n_pairs": None,
                    "passes_score": False,
                    "accepted_final": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if not qualified:
        return {
            "status": "unresolved",
            "reason": "no event passed minimum coherence score",
            "config_id": signature.config_id,
            "signature": signature.to_dict(),
            "attempts": attempts,
        }

    # Find the densest 1-D turnaround cluster. This is more robust than taking
    # the median of all score-qualified events when some earthquakes have a
    # strong but physically unrelated coherence maximum.
    best_cluster = []
    best_cluster_score = -np.inf

    for centre_result in qualified:
        centre = float(centre_result["peak_row"])
        cluster = [
            result
            for result in qualified
            if abs(float(result["peak_row"]) - centre)
            <= max_peak_deviation_rows
        ]
        cluster_score = float(sum(result["peak_score"] for result in cluster))

        if (
            len(cluster) > len(best_cluster)
            or (
                len(cluster) == len(best_cluster)
                and cluster_score > best_cluster_score
            )
        ):
            best_cluster = cluster
            best_cluster_score = cluster_score

    accepted = best_cluster
    accepted_ids = {result["event_id"] for result in accepted}

    for row in attempts:
        row["accepted_final"] = row["event_id"] in accepted_ids
        if row["passes_score"] and not row["accepted_final"] and not row["error"]:
            row["error"] = "rejected as inconsistent turnaround-row peak"

    confidence = None
    if len(accepted) >= max(2, min(target_calibrations, 2)):
        confidence = "validated_multi_event"
    elif len(accepted) == 1:
        result = accepted[0]
        if (
            result["peak_score"] >= SINGLE_EVENT_MIN_SCORE
            and result["median_abs_signed_corr"] >= SINGLE_EVENT_MIN_ABS_CORR
            and result["median_envelope_corr"] >= SINGLE_EVENT_MIN_ENV_CORR
        ):
            confidence = "validated_single_high_confidence"

    if confidence is None:
        return {
            "status": "unresolved",
            "reason": (
                "no repeatable turnaround cluster: need >=2 consistent "
                "events or one exceptionally high-confidence event"
            ),
            "config_id": signature.config_id,
            "signature": signature.to_dict(),
            "attempts": attempts,
        }

    _, canonical, grid_peak = stack_calibration_curves(accepted)
    accepted_peaks = np.asarray([r["peak_row"] for r in accepted], float)

    dx = float(signature.channel_spacing_m)
    bottom_md = float(reference["bottom_md_m"])
    rows_all = np.arange(signature.n_channels, dtype=float)
    fibre = bottom_md + (rows_all - canonical) * dx
    keep = (fibre >= 0.0) & (fibre <= 2.0 * bottom_md)

    if np.count_nonzero(keep) < 100:
        raise RuntimeError("Too few samples map onto physical borehole.")

    uncertainty_rows = float(np.max(np.abs(accepted_peaks - canonical)))

    return {
        "status": "validated",
        "method": "turnaround_waveform_calibration",
        "confidence": confidence,
        "config_id": signature.config_id,
        "family_id": signature.family_id,
        "signature": signature.to_dict(),
        "canonical_turnaround_data_row": float(canonical),
        "stack_grid_peak_row": float(grid_peak),
        "accepted_turnaround_rows": accepted_peaks.tolist(),
        "peak_spread_rows": float(accepted_peaks.max() - accepted_peaks.min()),
        "turnaround_uncertainty_rows": uncertainty_rows,
        "turnaround_uncertainty_m": float(uncertainty_rows * dx),
        "data_row_min_borehole": int(rows_all[keep].min()),
        "data_row_max_borehole": int(rows_all[keep].max()),
        "fibre_distance_min_m": float(fibre[keep].min()),
        "fibre_distance_max_m": float(fibre[keep].max()),
        "n_borehole_rows": int(np.count_nonzero(keep)),
        "accepted_calibration_events": [
            {
                "event_id": r["event_id"],
                "origin_time_utc": r["origin_time_utc"],
                "h5": r["h5"],
                "peak_row": float(r["peak_row"]),
                "peak_score": float(r["peak_score"]),
                "median_abs_signed_corr": float(r["median_abs_signed_corr"]),
                "median_envelope_corr": float(r["median_envelope_corr"]),
                "n_pairs": int(r["n_pairs"]),
            }
            for r in accepted
        ],
        "attempts": attempts,
    }


def discover_used_configurations(
    *,
    catalog: pd.DataFrame,
    manifest: pd.DataFrame,
) -> dict[str, AcquisitionSignature]:
    used = {}
    failures = []

    for _, event in catalog.iterrows():
        try:
            sig = event_configuration(event, manifest)
            used[sig.config_id] = sig
        except Exception as exc:
            failures.append(
                {
                    "event_id": str(event["event_id"]),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if failures:
        raise RuntimeError(
            "Some catalog events cannot be assigned to acquisition "
            "configurations:\n"
            + json.dumps(failures[:10], indent=2)
        )

    return used


def load_existing_registry(
    *,
    path: Path,
    geometry_sha256: str,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        registry = json.loads(path.read_text())
    except Exception:
        return None

    if registry.get("registry_version") != REGISTRY_VERSION:
        return None
    if registry.get("geometry_sha256") != geometry_sha256:
        return None
    return registry


def write_registry_qc(
    *,
    registry: dict[str, Any],
    output_path: Path,
) -> None:
    validated = [
        x for x in registry["configurations"].values()
        if x.get("status") == "validated"
    ]
    if not validated:
        return

    fig, ax = plt.subplots(
        figsize=(12, max(4.5, 0.8 * len(validated) + 2.0))
    )

    labels = []
    for y, entry in enumerate(validated):
        labels.append(entry["config_id"])
        canonical = entry["canonical_turnaround_data_row"]
        peaks = np.asarray(entry["accepted_turnaround_rows"], float)
        delta = peaks - canonical

        ax.scatter(delta, np.full(delta.shape, y), s=45)
        u = entry["turnaround_uncertainty_rows"]
        ax.plot([-u, u], [y, y], lw=3, alpha=0.3)

    ax.axvline(0.0, color="black", ls="--", lw=1)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Calibration-event turnaround row minus canonical row")
    ax.set_title(
        "SAFOD acquisition-registration QC\n"
        "Physical cable geometry fixed by reference Excel"
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    tmp = output_path.parent / (output_path.name + ".tmp.png")
    fig.savefig(tmp, dpi=180)
    plt.close(fig)
    os.replace(tmp, output_path)


def detect_reference_configuration(
    *,
    used: dict[str, AcquisitionSignature],
    reference: dict[str, Any],
) -> AcquisitionSignature:
    """
    Detect the acquisition configuration on which the reference Channel index
    is defined. It must span essentially the same number of channel intervals
    as the full reference workbook. No interrogator parameter is hard-coded.
    """
    span = float(reference["full_channel_span"])
    candidates = []

    for signature in used.values():
        if signature.first_channel_index is None:
            continue

        mismatch = abs(float(signature.n_channels) - span)
        if mismatch <= 2.0:
            candidates.append((mismatch, signature.n_channels, signature))

    if not candidates:
        raise RuntimeError(
            "Could not identify a reference-channel acquisition automatically. "
            "Expected one catalog configuration whose n_channels matches the "
            "reference workbook Channel span within 2 samples."
        )

    candidates.sort(key=lambda item: (item[0], item[1]))
    best = candidates[0][2]

    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        raise RuntimeError(
            "Reference acquisition is ambiguous; multiple configurations "
            "match the reference Channel span equally well."
        )

    return best


def make_reference_family_entry(
    *,
    signature: AcquisitionSignature,
    reference_signature: AcquisitionSignature,
    reference: dict[str, Any],
) -> dict[str, Any]:
    """
    Register any configuration from the original reference-channel family.

    In the detected reference configuration, DataRow 0 corresponds to the
    minimum reference Channel. first_channel_index therefore fixes the family
    global-locus offset. Other crops from the same dx/GL family inherit it.
    """
    if signature.family_id != reference_signature.family_id:
        raise ValueError("Configuration is not in reference family.")

    if (
        signature.first_channel_index is None
        or reference_signature.first_channel_index is None
    ):
        raise RuntimeError("Reference-family first_channel_index is required.")

    reference_channel_origin = float(reference["full_channel_min"])
    locus_offset = (
        float(reference_signature.first_channel_index)
        - reference_channel_origin
    )

    turn_row = (
        float(reference["turn_channel"])
        + locus_offset
        - float(signature.first_channel_index)
    )

    rows_all = np.arange(signature.n_channels, dtype=np.int64)
    physical_channel = (
        float(signature.first_channel_index)
        + rows_all.astype(float)
        - locus_offset
    )

    ch_min = float(reference["reference_borehole_channel_min"])
    ch_max = float(reference["reference_borehole_channel_max"])
    keep = (physical_channel >= ch_min) & (physical_channel <= ch_max)

    if np.count_nonzero(keep) < 100:
        raise RuntimeError(
            "Reference-family configuration does not overlap enough of the "
            "georeferenced borehole."
        )

    fibre = np.interp(
        physical_channel[keep],
        reference["reference_channels"],
        reference["reference_fibre_m"],
    )

    if np.any(np.diff(fibre) <= 0.0):
        raise RuntimeError("Reference-family physical fibre axis is invalid.")

    return {
        "status": "validated",
        "method": "reference_channel_locus",
        "confidence": "reference_geometry_registration",
        "config_id": signature.config_id,
        "family_id": signature.family_id,
        "signature": signature.to_dict(),
        "reference_config_id": reference_signature.config_id,
        "family_locus_offset": float(locus_offset),
        "canonical_turnaround_data_row": float(turn_row),
        "accepted_turnaround_rows": [float(turn_row)],
        "peak_spread_rows": 0.0,
        "turnaround_uncertainty_rows": 0.0,
        "turnaround_uncertainty_m": 0.0,
        "data_row_min_borehole": int(rows_all[keep].min()),
        "data_row_max_borehole": int(rows_all[keep].max()),
        "physical_reference_channel_min": float(physical_channel[keep].min()),
        "physical_reference_channel_max": float(physical_channel[keep].max()),
        "fibre_distance_min_m": float(fibre.min()),
        "fibre_distance_max_m": float(fibre.max()),
        "n_borehole_rows": int(np.count_nonzero(keep)),
        "accepted_calibration_events": [],
        "attempts": [],
    }


def prepare_geometry_registry(
    *,
    catalog_path: Path,
    manifest_path: Path,
    geometry_path: Path,
    geometry_dir: Path,
    force_recalibrate: bool,
    max_attempts: int,
    target_calibrations: int,
    min_score: float,
    max_peak_deviation_rows: float,
) -> None:
    catalog = load_catalog(catalog_path)
    manifest = annotate_manifest_configurations(load_manifest(manifest_path))
    reference = load_reference_geometry(geometry_path)
    used = discover_used_configurations(catalog=catalog, manifest=manifest)

    reference_signature = detect_reference_configuration(
        used=used,
        reference=reference,
    )
    reference_family_id = reference_signature.family_id

    geometry_dir.mkdir(parents=True, exist_ok=True)
    registry_path = geometry_dir / REGISTRY_NAME
    qc_path = geometry_dir / REGISTRY_QC_NAME

    existing = (
        None
        if force_recalibrate
        else load_existing_registry(
            path=registry_path,
            geometry_sha256=reference["geometry_sha256"],
        )
    )
    previous = {} if existing is None else existing.get("configurations", {})

    print("\nFixed SAFOD physical geometry")
    print("-----------------------------")
    print(f"geometry file             : {geometry_path}")
    print(f"turnaround reference ch   : {reference['turn_channel']:.0f}")
    print(f"turnaround MD             : {reference['bottom_md_m']:.6f} m")
    print(f"turnaround TVD            : {reference['bottom_tvd_m']:.6f} m")
    print(
        f"full borehole fibre       : "
        f"{reference['full_borehole_fibre_length_m']:.6f} m"
    )
    print(f"catalog configurations    : {len(used)}")
    print(f"reference configuration   : {reference_signature.config_id}")
    print(f"reference family          : {reference_family_id}")

    configurations = {}

    with tempfile.TemporaryDirectory(prefix="safod_registration_") as tmp:
        temp_root = Path(tmp)

        for i, (cid, sig) in enumerate(sorted(used.items()), start=1):
            print(f"\n[{i}/{len(used)}] {cid}")
            print(
                f"    n={sig.n_channels}, dx={sig.channel_spacing_m:.6f} m, "
                f"GL={sig.gauge_length_m:.6f} m, first={sig.first_channel_index}"
            )

            # Original/reference interrogator family is anchored directly to
            # the Channel coordinate in the immutable geometry workbook. It
            # must not be re-estimated from earthquake waveform coherence.
            if sig.family_id == reference_family_id:
                entry = make_reference_family_entry(
                    signature=sig,
                    reference_signature=reference_signature,
                    reference=reference,
                )
                configurations[cid] = entry
                print(
                    f"    VALIDATED from reference Channel locus: "
                    f"turn_row={entry['canonical_turnaround_data_row']:.3f}"
                )
                continue

            old = previous.get(cid)
            if (
                old is not None
                and old.get("status") == "validated"
                and old.get("signature") == sig.to_dict()
                and old.get("method") == "turnaround_waveform_calibration"
            ):
                configurations[cid] = old
                print("    reused validated waveform registration")
                continue

            entry = calibrate_configuration(
                signature=sig,
                catalog=catalog,
                manifest=manifest,
                reference=reference,
                reference_xlsx=geometry_path,
                temporary_root=temp_root,
                max_attempts=max_attempts,
                target_calibrations=target_calibrations,
                min_score=min_score,
                max_peak_deviation_rows=max_peak_deviation_rows,
            )
            configurations[cid] = entry

            if entry["status"] == "validated":
                print(
                    f"    VALIDATED by waveform turn: "
                    f"turn_row={entry['canonical_turnaround_data_row']:.3f}, "
                    f"uncertainty≈{entry['turnaround_uncertainty_m']:.2f} m, "
                    f"{entry['confidence']}"
                )
            else:
                print(f"    UNRESOLVED: {entry.get('reason', '')}")

    registry = {
        "registry_version": REGISTRY_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "geometry_file": str(geometry_path),
        "geometry_sha256": reference["geometry_sha256"],
        "physical_geometry": {
            "turnaround_reference_channel": float(reference["turn_channel"]),
            "turnaround_MD_m": float(reference["bottom_md_m"]),
            "turnaround_TVD_m": float(reference["bottom_tvd_m"]),
            "full_borehole_fibre_length_m": float(
                reference["full_borehole_fibre_length_m"]
            ),
            "surface_spool_excluded_from_scientific_axis": True,
            "up_leg_surface_reference_channel_for_qc": float(
                reference["up_surface_channel"]
            ),
        },
        "reference_configuration_id": reference_signature.config_id,
        "reference_family_id": reference_family_id,
        "configuration_definition": (
            "n_channels + channel_spacing_m + gauge_length_m + first_channel_index"
        ),
        "configurations": configurations,
    }

    tmp_registry = geometry_dir / (REGISTRY_NAME + ".tmp")
    tmp_registry.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    os.replace(tmp_registry, registry_path)

    write_registry_qc(registry=registry, output_path=qc_path)

    unresolved = [
        x for x in configurations.values()
        if x.get("status") != "validated"
    ]

    print("\nRegistration summary")
    print("--------------------")
    print(
        f"validated configurations : "
        f"{len(configurations) - len(unresolved)}"
    )
    print(f"unresolved configurations: {len(unresolved)}")
    print(f"registry                 : {registry_path}")
    print(f"QC                       : {qc_path}")

    if unresolved:
        for entry in unresolved:
            print(
                f"  - {entry['config_id']}: "
                f"{entry.get('reason', 'unresolved')}"
            )
        raise RuntimeError(
            f"{len(unresolved)} acquisition configuration(s) remain unvalidated."
        )


def load_registry(
    *,
    geometry_dir: Path,
    reference: dict[str, Any],
) -> dict[str, Any]:
    path = geometry_dir / REGISTRY_NAME

    if not path.exists():
        raise FileNotFoundError(
            f"{path}\nRun --prepare-geometry first."
        )

    registry = json.loads(path.read_text())

    if registry.get("registry_version") != REGISTRY_VERSION:
        raise RuntimeError("Registry version changed; rerun --prepare-geometry.")
    if registry.get("geometry_sha256") != reference["geometry_sha256"]:
        raise RuntimeError(
            "Authoritative geometry Excel changed; rerun --prepare-geometry."
        )
    return registry


def register_axis(
    *,
    selected_manifest: pd.DataFrame,
    n_channels_loaded: int,
    reference: dict[str, Any],
    registry: dict[str, Any],
) -> RegisteredAxis:
    sig = configuration_signature_from_rows(selected_manifest)

    if n_channels_loaded != sig.n_channels:
        raise RuntimeError(
            f"Loaded n_channels={n_channels_loaded}, manifest={sig.n_channels}."
        )

    entry = registry["configurations"].get(sig.config_id)
    if entry is None or entry.get("status") != "validated":
        raise RuntimeError(
            f"Configuration {sig.config_id} is not validated."
        )
    if entry.get("signature") != sig.to_dict():
        raise RuntimeError("Registry signature differs from selected HDF5.")

    rows_all = np.arange(sig.n_channels, dtype=np.int64)
    method = entry.get("method")

    if method == "reference_channel_locus":
        if sig.first_channel_index is None:
            raise RuntimeError("Reference-locus registration needs first_channel_index.")

        locus_offset = float(entry["family_locus_offset"])
        physical_channel = (
            float(sig.first_channel_index)
            + rows_all.astype(float)
            - locus_offset
        )

        keep = (
            (physical_channel >= reference["reference_borehole_channel_min"])
            & (physical_channel <= reference["reference_borehole_channel_max"])
        )

        rows = rows_all[keep]
        fibre = np.interp(
            physical_channel[keep],
            reference["reference_channels"],
            reference["reference_fibre_m"],
        )

        registration_method = "reference_channel_locus_to_fixed_geometry"

    elif method == "turnaround_waveform_calibration":
        turn = float(entry["canonical_turnaround_data_row"])
        dx = float(sig.channel_spacing_m)
        bottom_md = float(reference["bottom_md_m"])

        fibre_all = bottom_md + (rows_all.astype(float) - turn) * dx
        keep = (
            (fibre_all >= 0.0)
            & (fibre_all <= 2.0 * bottom_md)
        )
        rows = rows_all[keep]
        fibre = fibre_all[keep]
        registration_method = "turnaround_to_fixed_geometry"

    else:
        raise RuntimeError(
            f"Unknown acquisition registration method {method!r}."
        )

    if len(rows) < 100:
        raise RuntimeError("Too few HDF5 rows map to physical borehole.")
    if not np.all(np.diff(rows) == 1):
        raise RuntimeError("Registered rows are not contiguous.")
    if not np.all(np.isfinite(fibre)) or np.any(np.diff(fibre) <= 0.0):
        raise RuntimeError("Physical fibre axis is invalid.")

    return RegisteredAxis(
        data_rows=rows,
        fibre_distance_m=fibre,
        turnaround_m=float(reference["bottom_md_m"]),
        metadata={
            "registration_method": registration_method,
            "registry_method": method,
            "config_id": sig.config_id,
            "family_id": sig.family_id,
            "signature": sig.to_dict(),
            "canonical_turnaround_data_row": float(
                entry["canonical_turnaround_data_row"]
            ),
            "registration_confidence": entry["confidence"],
            "turnaround_uncertainty_rows": float(
                entry["turnaround_uncertainty_rows"]
            ),
            "turnaround_uncertainty_m": float(
                entry["turnaround_uncertainty_m"]
            ),
            "physical_turnaround_reference_channel": float(
                reference["turn_channel"]
            ),
            "physical_turnaround_MD_m": float(reference["bottom_md_m"]),
            "physical_turnaround_TVD_m": float(reference["bottom_tvd_m"]),
            "data_row_min": int(rows.min()),
            "data_row_max": int(rows.max()),
            "fibre_distance_min_m": float(fibre.min()),
            "fibre_distance_max_m": float(fibre.max()),
        },
    )


def read_event_files(files: list[str]) -> tuple[np.ndarray, dict]:
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
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"DAS reader returned {data.shape}; expected 2-D.")
    return data, info


def filter_event_window(
    *,
    data: np.ndarray,
    time_s: np.ndarray,
    fs_hz: float,
    recipe: DisplayRecipe,
    zero_phase: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    nyquist = 0.5 * float(fs_hz)
    effective_fmax = min(recipe.fmax_hz, 0.90 * nyquist)

    if not 0.0 < recipe.fmin_hz < effective_fmax < nyquist:
        raise ValueError("Invalid band after Nyquist guard.")

    display = (
        (time_s >= recipe.display_tmin_s)
        & (time_s <= recipe.display_tmax_s)
    )
    if np.count_nonzero(display) < 2:
        raise RuntimeError("Display interval unavailable.")

    filtered = DASutils.bandpass2D_c(
        np.ascontiguousarray(data),
        recipe.fmin_hz,
        effective_fmax,
        1.0 / float(fs_hz),
        zerophase=bool(zero_phase),
    )

    filtered = np.asarray(filtered, dtype=np.float64) * 1.0e3
    return filtered[:, display], time_s[display], float(effective_fmax)


def decimate_for_display(
    *,
    data: np.ndarray,
    time_s: np.ndarray,
    fs_hz: float,
    fmax_hz: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    max_band = int(
        math.floor(float(fs_hz) / (2.4 * float(fmax_hz)))
    )
    max_target = int(
        math.floor(float(fs_hz) / MAX_DISPLAY_FS_HZ)
    )
    factor = max(1, min(max_band, max_target))

    if factor <= 1:
        return data, time_s, float(fs_hz)

    return (
        data[:, ::factor],
        time_s[::factor],
        float(fs_hz) / factor,
    )


def raster_edges(x: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    if x.size < 2:
        raise ValueError("Need >=2 spatial samples.")
    dx = float(np.median(np.diff(x)))
    if not np.isfinite(dx) or dx <= 0.0:
        raise ValueError(f"Invalid spatial sample {dx}.")
    return x[0] - 0.5 * dx, x[-1] + 0.5 * dx, dx


def event_plot_filename(
    event_row: pd.Series,
    recipe: DisplayRecipe,
) -> str:
    origin = pd.Timestamp(event_row["origin_time_utc"])
    magnitude = finite_float(event_row.get("magnitude", np.nan))
    band = (
        f"{format_number_token(recipe.fmin_hz)}_"
        f"{format_number_token(recipe.fmax_hz)}Hz"
    )
    return (
        f"{origin.strftime('%Y%m%dT%H%M%S')}_event_"
        f"{safe_token(event_row['event_id'])}_"
        f"M{magnitude:.1f}_{safe_token(recipe.name)}_{band}.png"
    )


def plot_event(
    *,
    data: np.ndarray,
    time_s: np.ndarray,
    registration: RegisteredAxis,
    event_row: pd.Series,
    recording_label: str,
    recipe: DisplayRecipe,
    effective_fmax_hz: float,
    zero_phase: bool,
    output_path: Path,
    dpi: int,
) -> float:
    x = registration.fibre_distance_m
    clip = robust_clip(data, recipe.pclip)
    left, right, dx = raster_edges(x)

    magnitude = finite_float(event_row.get("magnitude", np.nan))
    magnitude_type = str(event_row.get("magnitude_type", ""))
    depth_km = finite_float(event_row.get("depth_km", np.nan))
    distance_km = finite_float(
        event_row.get("min_3d_distance_to_cable_km", np.nan)
    )
    crossline_km = abs(
        finite_float(event_row.get("source_crossline_m", np.nan))
    ) / 1000.0
    geometry_class = str(event_row.get("geometry_2d_class", "unknown"))

    fig, ax = plt.subplots(figsize=(20, 11))
    image = ax.imshow(
        data.T,
        extent=[left, right, float(time_s[-1]), float(time_s[0])],
        aspect="auto",
        cmap="seismic",
        vmin=-clip,
        vmax=clip,
        interpolation="none",
    )

    ax.axhline(
        0.0, color="black", lw=1.1, ls="--", label="Catalog origin"
    )
    ax.axvline(
        registration.turnaround_m,
        color="black",
        lw=1.0,
        ls=":",
        label=f"Physical turnaround ({registration.turnaround_m:.1f} m)",
    )

    ax.set_title(
        f"Real DAS event {event_row['event_id']}, "
        f"M{magnitude:.1f} {magnitude_type}, "
        f"{recipe.fmin_hz:g}-{effective_fmax_hz:g} Hz\n"
        f"depth={depth_km:.2f} km, "
        f"min source-cable distance={distance_km:.2f} km, "
        f"|crossline|={crossline_km:.2f} km, "
        f"2-D class={geometry_class}"
    )
    ax.set_xlabel("Distance along borehole fibre [m]")
    ax.set_ylabel("Time from catalog origin [s]")
    ax.set_xlim(left, right)
    ax.set_ylim(float(time_s[-1]), float(time_s[0]))
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    m = registration.metadata
    phase = "zero phase" if zero_phase else "causal"
    ax.text(
        0.01,
        0.99,
        (
            f"recording: {recording_label}\n"
            f"configuration: {m['config_id']}\n"
            f"turn DataRow: {m['canonical_turnaround_data_row']:.3f}\n"
            f"registration uncertainty: ~{m['turnaround_uncertainty_m']:.2f} m\n"
            f"recipe: {recipe.name}\n"
            f"spatial sample: ~{dx:.3f} m\n"
            f"clip: ±P{recipe.pclip:g} = {clip:.3g} nm/m/s\n"
            f"filter: {phase}"
        ),
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

    cb = fig.colorbar(image, ax=ax, pad=0.015)
    cb.set_label("strain rate [nm/m/s]")

    fig.tight_layout()
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return clip


def process_event(
    *,
    event_index: int,
    event_row: pd.Series,
    manifest: pd.DataFrame,
    output_dir: Path,
    recipe: DisplayRecipe,
    reference: dict[str, Any],
    registry: dict[str, Any],
    zero_phase: bool,
    overwrite: bool,
    dpi: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    origin = pd.Timestamp(event_row["origin_time_utc"])

    filter_tmin = recipe.display_tmin_s - recipe.filter_pad_s
    filter_tmax = recipe.display_tmax_s + recipe.filter_pad_s

    read_start = origin + pd.to_timedelta(filter_tmin, unit="s")
    read_end = origin + pd.to_timedelta(filter_tmax, unit="s")

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    image_path = output_dir / event_plot_filename(event_row, recipe)
    metadata_path = metadata_dir / f"{image_path.stem}.json"

    if image_path.exists() and metadata_path.exists() and not overwrite:
        record = json.loads(metadata_path.read_text())
        record["status"] = "exists"
        record["elapsed_s"] = 0.0
        return record

    selected = select_event_files(
        manifest=manifest,
        event_row=event_row,
        read_start_utc=read_start,
        read_end_utc=read_end,
    )

    ok, message = coverage_summary(
        selected,
        required_start=read_start,
        required_end=read_end,
    )
    if not ok:
        raise RuntimeError(f"Incomplete padded event window: {message}")

    files = selected["file_path"].astype(str).tolist()
    data_full, info = read_event_files(files)
    fs_hz = float(info["fs"])

    if not np.isfinite(fs_hz) or fs_hz <= 0.0:
        raise ValueError(f"Invalid read fs={fs_hz}.")

    beg = parse_beg_time_from_info(info)
    relative_start = (beg - origin.to_pydatetime()).total_seconds()
    time_full = (
        relative_start
        + np.arange(data_full.shape[1], dtype=float) / fs_hz
    )

    filter_mask = (
        (time_full >= filter_tmin)
        & (time_full <= filter_tmax)
    )
    if np.count_nonzero(filter_mask) < 16:
        raise RuntimeError("Too few samples in padded filter window.")

    data_filter = np.ascontiguousarray(data_full[:, filter_mask])
    time_filter = time_full[filter_mask]
    del data_full

    data_display, time_display, effective_fmax = filter_event_window(
        data=data_filter,
        time_s=time_filter,
        fs_hz=fs_hz,
        recipe=recipe,
        zero_phase=zero_phase,
    )
    del data_filter

    data_display, time_display, display_fs = decimate_for_display(
        data=data_display,
        time_s=time_display,
        fs_hz=fs_hz,
        fmax_hz=effective_fmax,
    )

    registration = register_axis(
        selected_manifest=selected,
        n_channels_loaded=data_display.shape[0],
        reference=reference,
        registry=registry,
    )
    data_display = data_display[registration.data_rows, :]

    recording_label = str(event_row["primary_recording"])
    clip = plot_event(
        data=data_display,
        time_s=time_display,
        registration=registration,
        event_row=event_row,
        recording_label=recording_label,
        recipe=recipe,
        effective_fmax_hz=effective_fmax,
        zero_phase=zero_phase,
        output_path=image_path,
        dpi=dpi,
    )

    result = {
        "status": "ok",
        "event_index": int(event_index),
        "event_id": str(event_row["event_id"]),
        "origin_time_utc": origin.isoformat(),
        "magnitude": finite_float(event_row.get("magnitude", np.nan)),
        "depth_km": finite_float(event_row.get("depth_km", np.nan)),
        "min_3d_distance_to_cable_km": finite_float(
            event_row.get("min_3d_distance_to_cable_km", np.nan)
        ),
        "source_crossline_m": finite_float(
            event_row.get("source_crossline_m", np.nan)
        ),
        "geometry_2d_class": str(
            event_row.get("geometry_2d_class", "unknown")
        ),
        "recording_label": recording_label,
        "axis_name": "fibre_distance_m",
        "axis_definition": (
            "physical cumulative distance along unchanged georeferenced "
            "SAFOD borehole fibre; surface spool excluded"
        ),
        "fibre_distance_min_m": float(
            registration.fibre_distance_m.min()
        ),
        "fibre_distance_max_m": float(
            registration.fibre_distance_m.max()
        ),
        "turnaround_fibre_distance_m": float(
            registration.turnaround_m
        ),
        "registration": registration.metadata,
        "recipe": asdict(recipe),
        "effective_fmax_hz": float(effective_fmax),
        "zero_phase": bool(zero_phase),
        "display_clip_nm_per_m_per_s": float(clip),
        "files": files,
        "n_files": len(files),
        "n_channels": int(data_display.shape[0]),
        "read_fs_hz": float(fs_hz),
        "display_fs_hz": float(display_fs),
        "image_path": str(image_path),
        "elapsed_s": float(time.perf_counter() - started),
    }

    metadata_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def build_html_gallery(
    *,
    output_dir: Path,
    catalog_path: Path,
) -> Path:
    """
    Build a fully local, searchable HTML gallery.

    The gallery does not alter any processing products. It only indexes the
    existing PNG + JSON metadata and adds client-side search/filter/sort.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = output_dir / "metadata"

    # Catalog is used only as a fallback for descriptive fields. The figure
    # inventory itself is defined by existing PNG + JSON products.
    catalog_lookup = {}
    if catalog_path.exists():
        catalog = pd.read_csv(catalog_path)

        if "event_id" in catalog.columns:
            catalog["event_id"] = (
                catalog["event_id"]
                .astype(str)
            )

            if "origin_time_utc" in catalog.columns:
                catalog["origin_time_utc"] = pd.to_datetime(
                    catalog["origin_time_utc"],
                    utc=True,
                    errors="coerce",
                )

            catalog_lookup = {
                str(row["event_id"]): row
                for _, row in catalog.iterrows()
            }

    metadata = {}

    if metadata_dir.exists():
        for path in metadata_dir.glob("*.json"):
            try:
                row = json.loads(
                    path.read_text()
                )
            except Exception:
                continue

            image_name = Path(
                str(
                    row.get(
                        "image_path",
                        "",
                    )
                )
            ).name

            if image_name:
                metadata[
                    image_name
                ] = row

    pngs = sorted(
        output_dir.glob(
            "*.png"
        )
    )

    cards = []
    rows = []

    recordings = set()
    configurations = set()

    for image in pngs:
        row = metadata.get(
            image.name,
            {},
        )

        event_id = str(
            row.get(
                "event_id",
                "unknown",
            )
        )

        catalog_row = (
            catalog_lookup.get(
                event_id
            )
        )

        def fallback(
            key: str,
            default: Any,
        ) -> Any:
            if key in row:
                return row[key]

            if (
                catalog_row is not None
                and key in catalog_row.index
            ):
                value = catalog_row[key]

                if pd.isna(value):
                    return default

                return value

            return default

        magnitude = finite_float(
            fallback(
                "magnitude",
                np.nan,
            )
        )

        depth_km = finite_float(
            fallback(
                "depth_km",
                np.nan,
            )
        )

        distance = finite_float(
            fallback(
                "min_3d_distance_to_cable_km",
                np.nan,
            )
        )

        geometry_class = str(
            fallback(
                "geometry_2d_class",
                "unknown",
            )
        )

        recording = str(
            fallback(
                "recording_label",
                fallback(
                    "primary_recording",
                    "",
                ),
            )
        )

        origin_raw = fallback(
            "origin_time_utc",
            "",
        )

        origin = pd.to_datetime(
            origin_raw,
            utc=True,
            errors="coerce",
        )

        if pd.isna(origin):
            origin_iso = ""
            date_token = ""
            display_time = ""
            sort_time = 0
        else:
            origin_iso = origin.isoformat()
            date_token = origin.strftime(
                "%Y-%m-%d"
            )
            display_time = origin.strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
            sort_time = int(
                origin.timestamp()
            )

        recipe = row.get(
            "recipe",
            {},
        )

        recipe_name = (
            str(
                recipe.get(
                    "name",
                    "",
                )
            )
            if isinstance(
                recipe,
                dict,
            )
            else str(
                recipe
            )
        )

        registration = row.get(
            "registration",
            {},
        )

        config_id = (
            str(
                registration.get(
                    "config_id",
                    "",
                )
            )
            if isinstance(
                registration,
                dict,
            )
            else ""
        )

        fibre_min = finite_float(
            row.get(
                "fibre_distance_min_m",
                np.nan,
            )
        )

        fibre_max = finite_float(
            row.get(
                "fibre_distance_max_m",
                np.nan,
            )
        )

        if recording:
            recordings.add(
                recording
            )

        if config_id:
            configurations.add(
                config_id
            )

        escaped_image = html.escape(
            image.name,
            quote=True,
        )

        escaped_event = html.escape(
            event_id,
            quote=True,
        )

        escaped_recording = html.escape(
            recording,
            quote=True,
        )

        escaped_config = html.escape(
            config_id,
            quote=True,
        )

        escaped_recipe = html.escape(
            recipe_name,
            quote=True,
        )

        escaped_geometry = html.escape(
            geometry_class,
            quote=True,
        )

        escaped_origin = html.escape(
            display_time,
            quote=True,
        )

        search_text = " ".join(
            [
                event_id,
                origin_iso,
                date_token,
                recording,
                config_id,
                recipe_name,
                geometry_class,
                f"M{magnitude:.1f}",
            ]
        ).lower()

        cards.append(
            f"""
            <article
              class="card"
              data-event-id="{escaped_event}"
              data-date="{html.escape(date_token, quote=True)}"
              data-time="{sort_time}"
              data-magnitude="{magnitude if np.isfinite(magnitude) else -999}"
              data-distance="{distance if np.isfinite(distance) else 999999}"
              data-recording="{escaped_recording}"
              data-config="{escaped_config}"
              data-search="{html.escape(search_text, quote=True)}"
            >
              <a
                class="image-link"
                href="{escaped_image}"
                title="Open full-resolution figure"
              >
                <img
                  loading="lazy"
                  src="{escaped_image}"
                  alt="Event {escaped_event}"
                >
              </a>

              <div class="meta">
                <div class="event-line">
                  <strong class="event-id">{escaped_event}</strong>
                  <span class="badge">M{magnitude:.1f}</span>
                  <span class="badge">d={distance:.2f} km</span>
                  <span class="badge">{escaped_geometry}</span>
                </div>

                <div class="origin">{escaped_origin}</div>

                <div class="details">
                  depth={depth_km:.2f} km<br>
                  recording: {escaped_recording}<br>
                  config: <span class="mono">{escaped_config}</span><br>
                  recipe: {escaped_recipe}<br>
                  fibre: {fibre_min:.1f}–{fibre_max:.1f} m
                </div>
              </div>
            </article>
            """
        )

        rows.append(
            {
                "image_name": image.name,
                "event_id": event_id,
                "origin_time_utc": origin_iso,
                "magnitude": magnitude,
                "depth_km": depth_km,
                "min_3d_distance_to_cable_km": distance,
                "geometry_2d_class": geometry_class,
                "recording_label": recording,
                "config_id": config_id,
                "recipe_name": recipe_name,
                "fibre_distance_min_m": fibre_min,
                "fibre_distance_max_m": fibre_max,
                "has_json_metadata": (
                    image.name
                    in metadata
                ),
            }
        )

    recording_options = "\n".join(
        (
            '<option value="'
            + html.escape(
                value,
                quote=True,
            )
            + '">'
            + html.escape(
                value
            )
            + "</option>"
        )
        for value in sorted(
            recordings
        )
    )

    configuration_options = "\n".join(
        (
            '<option value="'
            + html.escape(
                value,
                quote=True,
            )
            + '">'
            + html.escape(
                value
            )
            + "</option>"
        )
        for value in sorted(
            configurations
        )
    )

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SAFOD deep-DAS event QC</title>

<style>
:root {{
  color-scheme: light;
}}

* {{
  box-sizing: border-box;
}}

body {{
  font-family:
    system-ui,
    -apple-system,
    BlinkMacSystemFont,
    "Segoe UI",
    sans-serif;
  margin: 0;
  background: #f5f5f5;
  color: #202124;
}}

header {{
  position: sticky;
  top: 0;
  z-index: 20;
  background: rgba(245, 245, 245, 0.97);
  border-bottom: 1px solid #d9d9d9;
  padding: 16px 20px 14px;
  backdrop-filter: blur(8px);
}}

h1 {{
  margin: 0 0 6px;
  font-size: 1.45rem;
}}

.subtitle {{
  margin: 0 0 14px;
  color: #555;
  font-size: 0.92rem;
}}

.controls {{
  display: grid;
  grid-template-columns:
    minmax(220px, 2fr)
    repeat(2, minmax(150px, 1fr))
    repeat(2, minmax(140px, 0.8fr))
    minmax(170px, 1fr);
  gap: 8px;
  align-items: end;
}}

.control {{
  display: flex;
  flex-direction: column;
  gap: 4px;
}}

.control label {{
  color: #555;
  font-size: 0.75rem;
  font-weight: 600;
}}

input,
select,
button {{
  width: 100%;
  min-height: 38px;
  border: 1px solid #c8c8c8;
  border-radius: 6px;
  background: white;
  color: inherit;
  padding: 7px 9px;
  font: inherit;
}}

button {{
  cursor: pointer;
  font-weight: 600;
}}

button:hover {{
  background: #eeeeee;
}}

.status-line {{
  display: flex;
  gap: 16px;
  align-items: center;
  flex-wrap: wrap;
  margin-top: 11px;
  color: #555;
  font-size: 0.88rem;
}}

#visibleCount {{
  font-weight: 700;
  color: #202124;
}}

main {{
  padding: 18px 20px 30px;
}}

.grid {{
  display: grid;
  grid-template-columns:
    repeat(
      auto-fill,
      minmax(420px, 1fr)
    );
  gap: 16px;
}}

.card {{
  background: white;
  border: 1px solid #ddd;
  border-radius: 8px;
  overflow: hidden;
  box-shadow:
    0 1px 4px
    rgba(0, 0, 0, 0.08);
}}

.card.hidden {{
  display: none;
}}

.card img {{
  display: block;
  width: 100%;
  height: auto;
}}

.image-link {{
  display: block;
  background: #eee;
}}

.meta {{
  padding: 10px 12px 12px;
  line-height: 1.42;
}}

.event-line {{
  display: flex;
  gap: 6px;
  align-items: center;
  flex-wrap: wrap;
}}

.event-id {{
  margin-right: 4px;
  font-size: 1.02rem;
}}

.badge {{
  display: inline-block;
  padding: 2px 6px;
  border-radius: 999px;
  background: #efefef;
  font-size: 0.78rem;
  white-space: nowrap;
}}

.origin {{
  margin-top: 5px;
  font-weight: 600;
  font-size: 0.9rem;
}}

.details {{
  margin-top: 5px;
  color: #555;
  font-size: 0.82rem;
}}

.mono {{
  font-family:
    ui-monospace,
    SFMono-Regular,
    Menlo,
    Consolas,
    monospace;
  font-size: 0.94em;
}}

#emptyState {{
  display: none;
  margin: 30px auto;
  max-width: 620px;
  padding: 20px;
  background: white;
  border: 1px solid #ddd;
  border-radius: 8px;
  text-align: center;
  color: #555;
}}

@media (max-width: 1100px) {{
  .controls {{
    grid-template-columns:
      repeat(3, minmax(180px, 1fr));
  }}
}}

@media (max-width: 700px) {{
  header {{
    position: static;
  }}

  .controls {{
    grid-template-columns: 1fr;
  }}

  .grid {{
    grid-template-columns: 1fr;
  }}

  main {{
    padding: 12px;
  }}
}}
</style>
</head>

<body>
<header>
  <h1>SAFOD deep-DAS event QC</h1>

  <p class="subtitle">
    {len(cards)} plotted events ·
    horizontal axis = physical distance along the unchanged borehole fibre [m].
    Search accepts event ID, date, recording, configuration, recipe, or magnitude.
  </p>

  <div class="controls">
    <div class="control">
      <label for="search">Search</label>
      <input
        id="search"
        type="search"
        placeholder="e.g. 75336802 or 2026-04-01"
        autocomplete="off"
      >
    </div>

    <div class="control">
      <label for="recording">Recording</label>
      <select id="recording">
        <option value="">All recordings</option>
        {recording_options}
      </select>
    </div>

    <div class="control">
      <label for="configuration">Configuration</label>
      <select id="configuration">
        <option value="">All configurations</option>
        {configuration_options}
      </select>
    </div>

    <div class="control">
      <label for="dateFrom">Date from</label>
      <input
        id="dateFrom"
        type="date"
      >
    </div>

    <div class="control">
      <label for="dateTo">Date to</label>
      <input
        id="dateTo"
        type="date"
      >
    </div>

    <div class="control">
      <label for="sort">Sort</label>
      <select id="sort">
        <option value="time-asc">Time: oldest first</option>
        <option value="time-desc">Time: newest first</option>
        <option value="mag-desc">Magnitude: high to low</option>
        <option value="mag-asc">Magnitude: low to high</option>
        <option value="dist-asc">Distance: near to far</option>
      </select>
    </div>
  </div>

  <div class="status-line">
    <span id="visibleCount">{len(cards)} / {len(cards)} shown</span>
    <button
      id="reset"
      type="button"
      style="width:auto; min-height:32px; padding:4px 10px;"
    >
      Reset filters
    </button>
  </div>
</header>

<main>
  <div
    id="gallery"
    class="grid"
  >
    {''.join(cards)}
  </div>

  <div id="emptyState">
    No events match the current filters.
  </div>
</main>

<script>
(() => {{
  const gallery = document.getElementById("gallery");
  const cards = Array.from(
    gallery.querySelectorAll(".card")
  );

  const search = document.getElementById("search");
  const recording = document.getElementById("recording");
  const configuration = document.getElementById("configuration");
  const dateFrom = document.getElementById("dateFrom");
  const dateTo = document.getElementById("dateTo");
  const sort = document.getElementById("sort");
  const reset = document.getElementById("reset");
  const visibleCount = document.getElementById("visibleCount");
  const emptyState = document.getElementById("emptyState");

  const total = cards.length;

  function numberValue(card, key, fallback) {{
    const value = Number(card.dataset[key]);
    return Number.isFinite(value) ? value : fallback;
  }}

  function applyFilters() {{
    const query = search.value
      .trim()
      .toLowerCase();

    const recordingValue = recording.value;
    const configurationValue = configuration.value;
    const fromValue = dateFrom.value;
    const toValue = dateTo.value;

    let visible = 0;

    cards.forEach(card => {{
      const matchesSearch =
        !query
        || card.dataset.search.includes(query);

      const matchesRecording =
        !recordingValue
        || card.dataset.recording === recordingValue;

      const matchesConfiguration =
        !configurationValue
        || card.dataset.config === configurationValue;

      const cardDate = card.dataset.date;

      const matchesFrom =
        !fromValue
        || (
          cardDate
          && cardDate >= fromValue
        );

      const matchesTo =
        !toValue
        || (
          cardDate
          && cardDate <= toValue
        );

      const show =
        matchesSearch
        && matchesRecording
        && matchesConfiguration
        && matchesFrom
        && matchesTo;

      card.classList.toggle(
        "hidden",
        !show
      );

      if (show) {{
        visible += 1;
      }}
    }});

    visibleCount.textContent =
      `${{visible}} / ${{total}} shown`;

    emptyState.style.display =
      visible === 0
      ? "block"
      : "none";
  }}

  function applySort() {{
    const mode = sort.value;

    const ordered = [...cards].sort(
      (a, b) => {{
        if (mode === "time-desc") {{
          return (
            numberValue(b, "time", 0)
            - numberValue(a, "time", 0)
          );
        }}

        if (mode === "mag-desc") {{
          return (
            numberValue(b, "magnitude", -999)
            - numberValue(a, "magnitude", -999)
          );
        }}

        if (mode === "mag-asc") {{
          return (
            numberValue(a, "magnitude", 999)
            - numberValue(b, "magnitude", 999)
          );
        }}

        if (mode === "dist-asc") {{
          return (
            numberValue(a, "distance", 999999)
            - numberValue(b, "distance", 999999)
          );
        }}

        return (
          numberValue(a, "time", 0)
          - numberValue(b, "time", 0)
        );
      }}
    );

    ordered.forEach(
      card => gallery.appendChild(card)
    );
  }}

  function refresh() {{
    applySort();
    applyFilters();
  }}

  [
    search,
    recording,
    configuration,
    dateFrom,
    dateTo,
  ].forEach(
    element => element.addEventListener(
      "input",
      applyFilters
    )
  );

  sort.addEventListener(
    "change",
    refresh
  );

  reset.addEventListener(
    "click",
    () => {{
      search.value = "";
      recording.value = "";
      configuration.value = "";
      dateFrom.value = "";
      dateTo.value = "";
      sort.value = "time-asc";
      refresh();
      search.focus();
    }}
  );

  refresh();
}})();
</script>
</body>
</html>
"""

    index = output_dir / "index.html"

    index.write_text(
        page,
        encoding="utf-8",
    )

    pd.DataFrame(
        rows
    ).to_csv(
        output_dir
        / "gallery_manifest.csv",
        index=False,
    )

    print(
        "Gallery inventory"
    )
    print(
        "-----------------"
    )
    print(
        f"PNG files found       : {len(pngs)}"
    )
    print(
        "JSON metadata matched : "
        f"{sum(row['has_json_metadata'] for row in rows)}"
    )
    print(
        f"HTML cards written    : {len(cards)}"
    )
    print(
        "Search/filter/sort    : enabled"
    )

    return index

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--geometry", type=Path, default=GEO_XLSX)
    parser.add_argument("--geometry-dir", type=Path, default=DEFAULT_GEOMETRY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--prepare-geometry", action="store_true")
    parser.add_argument("--force-recalibrate", action="store_true")
    parser.add_argument(
        "--calibration-attempts",
        type=int,
        default=DEFAULT_CALIBRATION_ATTEMPTS,
    )
    parser.add_argument(
        "--target-calibrations",
        type=int,
        default=DEFAULT_TARGET_CALIBRATIONS,
    )
    parser.add_argument(
        "--min-calibration-score",
        type=float,
        default=DEFAULT_MIN_CALIBRATION_SCORE,
    )
    parser.add_argument(
        "--max-peak-deviation-rows",
        type=float,
        default=DEFAULT_MAX_PEAK_DEVIATION_ROWS,
    )

    parser.add_argument("--fmin", type=float, default=None)
    parser.add_argument("--fmax", type=float, default=None)
    parser.add_argument("--display-tmin", type=float, default=None)
    parser.add_argument("--display-tmax", type=float, default=None)
    parser.add_argument("--filter-pad", type=float, default=None)
    parser.add_argument("--pclip", type=float, default=None)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--zero-phase", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--event-id", default=None)
    parser.add_argument("--min-magnitude", type=float, default=None)
    parser.add_argument(
        "--geometry-class",
        choices=["good", "borderline", "poor"],
        default=None,
    )
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--build-index-only", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.prepare_geometry:
        prepare_geometry_registry(
            catalog_path=args.catalog,
            manifest_path=args.manifest,
            geometry_path=args.geometry,
            geometry_dir=args.geometry_dir,
            force_recalibrate=args.force_recalibrate,
            max_attempts=args.calibration_attempts,
            target_calibrations=args.target_calibrations,
            min_score=args.min_calibration_score,
            max_peak_deviation_rows=args.max_peak_deviation_rows,
        )
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.build_index_only:
        print(
            "HTML gallery:",
            build_html_gallery(
                output_dir=args.output_dir,
                catalog_path=args.catalog,
            ),
        )
        return

    catalog = load_catalog(args.catalog)

    if args.event_id is not None:
        catalog = catalog[
            catalog["event_id"].astype(str) == str(args.event_id)
        ].copy()
    if args.min_magnitude is not None:
        catalog = catalog[
            pd.to_numeric(catalog["magnitude"], errors="coerce")
            >= float(args.min_magnitude)
        ].copy()
    if args.geometry_class is not None:
        catalog = catalog[
            catalog["geometry_2d_class"].astype(str)
            == args.geometry_class
        ].copy()

    catalog = catalog.reset_index(drop=True)
    if args.limit is not None:
        catalog = catalog.iloc[: int(args.limit)].copy()

    if catalog.empty:
        raise RuntimeError("No catalog events selected.")

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >=1.")

    shard_index = (
        int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        if args.shard_index is None
        else int(args.shard_index)
    )
    if not 0 <= shard_index < args.num_shards:
        raise ValueError("Invalid shard index.")

    indices = [
        i for i in range(len(catalog))
        if i % args.num_shards == shard_index
    ]

    manifest = annotate_manifest_configurations(
        load_manifest(args.manifest)
    )
    reference = load_reference_geometry(args.geometry)
    registry = load_registry(
        geometry_dir=args.geometry_dir,
        reference=reference,
    )

    shard_configs = {}
    for i in indices:
        sig = event_configuration(catalog.iloc[i], manifest)
        shard_configs[sig.config_id] = sig

    invalid = [
        cid
        for cid in shard_configs
        if (
            registry["configurations"].get(cid) is None
            or registry["configurations"][cid].get("status") != "validated"
        )
    ]
    if invalid:
        raise RuntimeError(
            "Unvalidated configurations:\n  "
            + "\n  ".join(sorted(invalid))
            + "\nRun --prepare-geometry."
        )

    print("\nSAFOD catalog event plotting")
    print("----------------------------")
    print(f"catalog events selected : {len(catalog)}")
    print(f"num shards              : {args.num_shards}")
    print(f"this shard              : {shard_index}")
    print(f"events in this shard    : {len(indices)}")
    print(f"configurations in shard : {len(shard_configs)}")
    print("scientific x-axis       : distance along borehole fibre [m]")
    print(f"fixed physical geometry : {args.geometry}")
    print(f"physical turnaround     : {reference['bottom_md_m']:.3f} m")
    print(f"output                  : {args.output_dir}")

    statuses = []
    started = time.perf_counter()

    for pos, event_index in enumerate(indices, start=1):
        event = catalog.iloc[event_index]
        recipe = resolve_recipe(event, args)

        print(
            f"\n[{pos}/{len(indices)}] event={event['event_id']} "
            f"origin={event['origin_time_utc']} "
            f"M={finite_float(event.get('magnitude', np.nan)):.1f}"
        )

        try:
            result = process_event(
                event_index=event_index,
                event_row=event,
                manifest=manifest,
                output_dir=args.output_dir,
                recipe=recipe,
                reference=reference,
                registry=registry,
                zero_phase=args.zero_phase,
                overwrite=args.overwrite,
                dpi=args.dpi,
            )
            print(
                f"status={result['status']}, "
                f"config={result['registration']['config_id']}, "
                f"fibre={result['fibre_distance_min_m']:.1f}.."
                f"{result['fibre_distance_max_m']:.1f} m, "
                f"elapsed={result['elapsed_s']:.1f} s"
            )
        except Exception as exc:
            result = {
                "status": "error",
                "event_index": int(event_index),
                "event_id": str(event["event_id"]),
                "origin_time_utc": pd.Timestamp(
                    event["origin_time_utc"]
                ).isoformat(),
                "recording_label": str(event["primary_recording"]),
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_s": 0.0,
            }
            print("status=ERROR:", result["error"])

        statuses.append(result)

    status_path = (
        args.output_dir
        / f"status_shard_{shard_index:03d}_of_{args.num_shards:03d}.csv"
    )
    pd.DataFrame(statuses).to_csv(status_path, index=False)

    n_errors = sum(x.get("status") == "error" for x in statuses)
    elapsed = time.perf_counter() - started

    print("\nShard summary")
    print("-------------")
    print(f"successful/existing : {len(statuses) - n_errors}")
    print(f"errors              : {n_errors}")
    print(f"elapsed             : {elapsed / 60.0:.1f} min")
    print(f"status CSV          : {status_path}")

    if args.num_shards == 1:
        print(
            "HTML gallery        :",
            build_html_gallery(
                output_dir=args.output_dir,
                catalog_path=args.catalog,
            ),
        )


if __name__ == "__main__":
    main()
