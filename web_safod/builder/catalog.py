from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {
    "event_id",
    "origin_time_utc",
    "latitude",
    "longitude",
    "depth_km",
}


def finite_or_none(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    return text or None


def safe_event_id(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("Empty event_id.")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def canonical_public_event_id(event_id: str) -> str:
    event_id = str(event_id).strip()

    if event_id.isdigit():
        return f"nc{event_id}"

    return event_id.lower()


def load_catalog(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(
            f"Catalog is missing required columns {missing}: {path}"
        )

    frame = frame.copy()
    frame["event_id"] = frame["event_id"].astype(str).str.strip()
    # Parse each value independently. A vectorized pd.to_datetime call can
    # infer one strict ISO format from the first row and coerce otherwise valid
    # rows that differ only by fractional-second precision.
    frame["origin_time_utc"] = frame["origin_time_utc"].map(
        lambda value: pd.to_datetime(
            value,
            utc=True,
            errors="coerce",
        )
    )

    for column in (
        "latitude",
        "longitude",
        "depth_km",
        "magnitude",
        "min_3d_distance_to_cable_km",
        "source_crossline_m",
        "source_section_x_m",
        "out_of_plane_angle_deg",
    ):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame[
        frame["event_id"].ne("")
        & frame["origin_time_utc"].notna()
        & frame["latitude"].notna()
        & frame["longitude"].notna()
        & frame["depth_km"].notna()
    ].copy()

    if frame["event_id"].duplicated().any():
        duplicates = (
            frame.loc[frame["event_id"].duplicated(False), "event_id"]
            .tolist()
        )
        raise ValueError(
            "Catalog contains duplicate event_id values: "
            + ", ".join(sorted(set(duplicates))[:20])
        )

    return frame.sort_values("origin_time_utc").reset_index(drop=True)


def load_metadata_inventory(metadata_dir: Path) -> dict[str, list[dict[str, Any]]]:
    inventory: dict[str, list[dict[str, Any]]] = {}
    if not metadata_dir.exists():
        return inventory

    for path in sorted(metadata_dir.glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        event_id = text_or_none(row.get("event_id"))
        if not event_id:
            continue
        row["_metadata_path"] = str(path)
        row["_metadata_mtime_ns"] = path.stat().st_mtime_ns
        inventory.setdefault(event_id, []).append(row)

    for event_id, rows in inventory.items():
        rows.sort(
            key=lambda item: (
                item.get("status") not in {"ok", "exists"},
                -int(item.get("_metadata_mtime_ns", 0)),
            )
        )
    return inventory


def official_event_url(event_id: str, template: str) -> str:
    return template.format(event_id=event_id)


def event_record_from_row(
    row: pd.Series,
    *,
    metadata_rows: list[dict[str, Any]],
    plots: list[dict[str, Any]],
    event_page_template: str,
) -> dict[str, Any]:
    raw_event_id = safe_event_id(row["event_id"])
    event_id = canonical_public_event_id(raw_event_id)
    origin = pd.Timestamp(row["origin_time_utc"])

    recording = text_or_none(row.get("primary_recording"))
    config_id = None
    if metadata_rows:
        first = metadata_rows[0]
        recording = text_or_none(first.get("recording_label")) or recording
        registration = first.get("registration")
        if isinstance(registration, dict):
            config_id = text_or_none(registration.get("config_id"))

    location_name = (
        text_or_none(row.get("location_name"))
        or text_or_none(row.get("place"))
    )

    return {
        "event_id": event_id,
        "origin_time_utc": origin.isoformat().replace("+00:00", "Z"),
        "latitude": finite_or_none(row.get("latitude")),
        "longitude": finite_or_none(row.get("longitude")),
        "depth_km": finite_or_none(row.get("depth_km")),
        "magnitude": finite_or_none(row.get("magnitude")),
        "magnitude_type": text_or_none(row.get("magnitude_type")),
        "location_name": location_name,
        "min_3d_distance_to_cable_km": finite_or_none(
            row.get("min_3d_distance_to_cable_km")
        ),
        "source_crossline_m": finite_or_none(row.get("source_crossline_m")),
        "source_section_x_m": finite_or_none(
            row.get("source_section_x_m")
        ),
        "out_of_plane_angle_deg": finite_or_none(
            row.get("out_of_plane_angle_deg")
        ),
        "geometry_2d_class": (
            text_or_none(row.get("geometry_2d_class")) or "unknown"
        ),
        "recording_label": recording,
        "recording_config": config_id,
        "plot_available": bool(plots),
        "plot_url": plots[0]["url"] if plots else None,
        "plots": plots,
        "event_url": official_event_url(
            event_id,
            event_page_template,
        ),
    }
