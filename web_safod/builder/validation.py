from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


PRIVATE_PATH_TOKENS = (
    "/home/",
    "/oak/",
    "/scratch/",
    "file_path",
    ".h5",
    ".hdf5",
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _walk_strings(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)
    elif isinstance(value, str):
        yield value


def validate_site(site_dir: Path) -> list[str]:
    errors: list[str] = []

    required = [
        site_dir / "index.html",
        site_dir / "app.js",
        site_dir / "style.css",
        site_dir / "data/events.json",
        site_dir / "data/site_status.json",
        site_dir / "data/site_config.json",
        site_dir / "data/safod_geometry.geojson",
    ]
    for path in required:
        if not path.exists():
            errors.append(f"Missing required site file: {path}")

    if errors:
        return errors

    events = _load_json(site_dir / "data/events.json")
    if not isinstance(events, list):
        errors.append("events.json must contain a list.")
        return errors

    ids = []
    for i, event in enumerate(events):
        if not isinstance(event, dict):
            errors.append(f"Event #{i} is not an object.")
            continue

        event_id = str(event.get("event_id", "")).strip()
        if not event_id:
            errors.append(f"Event #{i} has empty event_id.")
        ids.append(event_id)

        for key, lo, hi in (
            ("latitude", -90.0, 90.0),
            ("longitude", -180.0, 180.0),
        ):
            value = event.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                errors.append(f"{event_id}: invalid {key}={value!r}")
            elif not lo <= float(value) <= hi:
                errors.append(f"{event_id}: out-of-range {key}={value!r}")

        depth = event.get("depth_km")
        if not isinstance(depth, (int, float)) or not math.isfinite(depth):
            errors.append(f"{event_id}: invalid depth_km={depth!r}")

        if event.get("plot_available"):
            plots = event.get("plots")
            if not isinstance(plots, list) or not plots:
                errors.append(f"{event_id}: plot_available but plots is empty.")
            else:
                for plot in plots:
                    url = str(plot.get("url", ""))
                    if not url or url.startswith(("http://", "https://", "/")):
                        errors.append(
                            f"{event_id}: plot URL must be relative: {url!r}"
                        )
                        continue
                    if not (site_dir / url).exists():
                        errors.append(
                            f"{event_id}: plot file does not exist: {url}"
                        )

                    arrivals_url = str(plot.get("arrivals_url", "") or "")
                    if arrivals_url:
                        if arrivals_url.startswith(("http://", "https://", "/")):
                            errors.append(
                                f"{event_id}: arrivals URL must be relative: "
                                f"{arrivals_url!r}"
                            )
                        elif not (site_dir / arrivals_url).exists():
                            errors.append(
                                f"{event_id}: arrivals plot file does not exist: "
                                f"{arrivals_url}"
                            )

    duplicates = sorted({x for x in ids if ids.count(x) > 1 and x})
    if duplicates:
        errors.append(
            "Duplicate event IDs: " + ", ".join(duplicates[:20])
        )

    focal_path = site_dir / "data/focal_mechanisms.json"
    if focal_path.exists():
        focal = _load_json(focal_path)
        if not isinstance(focal, dict):
            errors.append("focal_mechanisms.json must contain an object keyed by event_id.")
        else:
            event_ids = set(ids)
            for event_id, mechanism in focal.items():
                if event_id not in event_ids:
                    errors.append(f"Focal mechanism references unknown event: {event_id}")
                    continue
                if not isinstance(mechanism, dict):
                    errors.append(f"{event_id}: focal mechanism must be an object.")
                    continue
                for plane_name in ("np1", "np2"):
                    plane = mechanism.get(plane_name)
                    if not isinstance(plane, dict):
                        errors.append(f"{event_id}: missing {plane_name}.")
                        continue
                    for field, lo, hi in (
                        ("strike", 0.0, 360.0),
                        ("dip", 0.0, 90.0),
                        ("rake", -180.0, 180.0),
                    ):
                        value = plane.get(field)
                        if not isinstance(value, (int, float)) or not math.isfinite(value):
                            errors.append(f"{event_id}: invalid {plane_name}.{field}={value!r}")
                        elif not lo <= float(value) <= hi:
                            errors.append(f"{event_id}: out-of-range {plane_name}.{field}={value!r}")

    # Public JSON must not leak internal filesystem / raw-data paths.
    public_json_paths = [
        site_dir / "data/events.json",
        site_dir / "data/site_status.json",
        site_dir / "data/site_config.json",
        site_dir / "data/safod_geometry.geojson",
    ]
    if focal_path.exists():
        public_json_paths.append(focal_path)

    for json_path in public_json_paths:
        payload = _load_json(json_path)
        for text in _walk_strings(payload):
            lowered = text.lower()
            for token in PRIVATE_PATH_TOKENS:
                if token.lower() in lowered:
                    errors.append(
                        f"Private/raw path token {token!r} leaked into "
                        f"{json_path.name}: {text[:180]!r}"
                    )
                    break

    geometry = _load_json(site_dir / "data/safod_geometry.geojson")
    if geometry.get("type") != "FeatureCollection":
        errors.append("safod_geometry.geojson is not a FeatureCollection.")
    if not geometry.get("features"):
        errors.append("safod_geometry.geojson contains no features.")

    status = _load_json(site_dir / "data/site_status.json")
    if int(status.get("recorded_events", -1)) != len(events):
        errors.append(
            "site_status recorded_events does not match events.json."
        )
    plots_count = sum(bool(x.get("plot_available")) for x in events)
    if int(status.get("plots_available", -1)) != plots_count:
        errors.append(
            "site_status plots_available does not match events.json."
        )

    return errors
