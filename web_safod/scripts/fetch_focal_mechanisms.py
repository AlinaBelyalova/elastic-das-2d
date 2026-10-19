from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

DEFAULT_EVENTS = Path("web_safod/site/data/events.json")
DEFAULT_OUTPUT = Path("results/safod_deep/catalog/focal_mechanisms.json")
DEFAULT_SITE_COPY = Path("web_safod/site/data/focal_mechanisms.json")

WELL_LAT = 35.9742039
WELL_LON = -120.5521414
RADIUS_KM = 50.0
USER_AGENT = "SAFOD-event-explorer/1.0"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _get_json(url: str, params: dict[str, Any] | None = None, *, attempts: int = 5) -> Any:
    if params:
        url = url + "?" + urllib.parse.urlencode(params)

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            last_error = exc
            retryable = exc.code == 429 or 500 <= exc.code <= 599
            if not retryable or attempt == attempts:
                raise
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt == attempts:
                raise

        delay = min(16.0, 1.5 * (2 ** (attempt - 1)))
        print(f"    retry {attempt}/{attempts} after {type(last_error).__name__}; sleeping {delay:.1f}s")
        time.sleep(delay)

    assert last_error is not None
    raise last_error


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _first(props: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in props and props[name] not in ("", None):
            return props[name]
    return None


def _preferred_focal_product(detail: dict[str, Any]) -> dict[str, Any] | None:
    products = detail.get("properties", {}).get("products", {})
    mechanisms = products.get("focal-mechanism", [])
    if not mechanisms:
        return None
    return max(
        mechanisms,
        key=lambda product: (
            float(product.get("preferredWeight", 0) or 0),
            int(product.get("updateTime", 0) or 0),
        ),
    )


def _parse_mechanism(event_id: str, product: dict[str, Any]) -> dict[str, Any] | None:
    props = product.get("properties", {})
    np1 = {
        "strike": _as_float(_first(props, "nodal-plane-1-strike")),
        "dip": _as_float(_first(props, "nodal-plane-1-dip")),
        "rake": _as_float(_first(props, "nodal-plane-1-rake", "nodal-plane-1-slip")),
    }
    np2 = {
        "strike": _as_float(_first(props, "nodal-plane-2-strike")),
        "dip": _as_float(_first(props, "nodal-plane-2-dip")),
        "rake": _as_float(_first(props, "nodal-plane-2-rake", "nodal-plane-2-slip")),
    }
    if not all(value is not None for plane in (np1, np2) for value in plane.values()):
        return None

    return {
        "np1": np1,
        "np2": np2,
        "source": product.get("source"),
        "code": product.get("code"),
        "review_status": props.get("review-status"),
        "evaluation_status": props.get("evaluation-status"),
        "num_stations_used": props.get("num-stations-used"),
        "event_url": f"https://earthquake.usgs.gov/earthquakes/eventpage/{event_id}/focal-mechanism",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch preferred ComCat focal-mechanism products for events already "
            "present in the SAFOD web explorer. Existing mechanisms are preserved "
            "and only missing entries are fetched unless --refresh-existing is used."
        )
    )
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--site-copy", type=Path, default=DEFAULT_SITE_COPY)
    parser.add_argument("--refresh-existing", action="store_true")
    parser.add_argument("--attempts", type=int, default=5)
    args = parser.parse_args()

    events = _read_json(args.events)
    if not isinstance(events, list) or not events:
        raise SystemExit(f"No events found in {args.events}")

    viewer_ids = {str(event["event_id"]).strip() for event in events}
    event_by_id = {str(event["event_id"]).strip(): event for event in events}
    times = [
        datetime.fromisoformat(str(event["origin_time_utc"]).replace("Z", "+00:00"))
        for event in events
    ]
    start = min(times)
    end = max(times) + timedelta(seconds=1)

    existing: dict[str, Any] = {}
    if args.output.exists():
        loaded = _read_json(args.output)
        if isinstance(loaded, dict):
            existing = loaded
    elif args.site_copy.exists():
        loaded = _read_json(args.site_copy)
        if isinstance(loaded, dict):
            existing = loaded

    # Drop entries that are no longer part of the viewer catalog.
    result = {event_id: value for event_id, value in existing.items() if event_id in viewer_ids}

    params = {
        "format": "geojson",
        "starttime": start.isoformat(),
        "endtime": end.isoformat(),
        "latitude": WELL_LAT,
        "longitude": WELL_LON,
        "maxradiuskm": RADIUS_KM,
        "producttype": "focal-mechanism",
        "limit": 20000,
        "orderby": "time-asc",
    }

    print("Querying USGS ComCat focal-mechanism inventory...")
    catalog = _get_json(
        "https://earthquake.usgs.gov/fdsnws/event/1/query",
        params,
        attempts=max(1, args.attempts),
    )
    features = catalog.get("features", [])
    matching = [feature for feature in features if feature.get("id") in viewer_ids]

    print(f"Viewer events                  : {len(viewer_ids)}")
    print(f"ComCat focal mechanisms        : {len(features)}")
    print(f"Matching viewer events         : {len(matching)}")
    print(f"Existing usable mechanisms     : {len(result)}")

    failures: list[str] = []
    fetched = 0
    skipped = 0

    for index, feature in enumerate(matching, 1):
        event_id = str(feature["id"])
        if event_id in result and not args.refresh_existing:
            skipped += 1
            continue

        detail_url = feature.get("properties", {}).get("detail")
        if not detail_url:
            failures.append(event_id)
            print(f"[{index:3d}/{len(matching)}] {event_id}: no detail URL")
            continue

        try:
            detail = _get_json(detail_url, attempts=max(1, args.attempts))
            product = _preferred_focal_product(detail)
            if product is None:
                failures.append(event_id)
                print(f"[{index:3d}/{len(matching)}] {event_id}: no focal product")
                continue
            mechanism = _parse_mechanism(event_id, product)
            if mechanism is None:
                failures.append(event_id)
                print(f"[{index:3d}/{len(matching)}] {event_id}: incomplete nodal planes")
                continue

            result[event_id] = mechanism
            fetched += 1
            np1 = mechanism["np1"]
            magnitude = event_by_id[event_id].get("magnitude")
            mag_text = f" M{float(magnitude):.1f}" if magnitude is not None else ""
            print(
                f"[{index:3d}/{len(matching)}] {event_id}{mag_text}: "
                f"NP1 {np1['strike']:.0f}/{np1['dip']:.0f}/{np1['rake']:.0f}"
            )
        except Exception as exc:
            failures.append(event_id)
            print(f"[{index:3d}/{len(matching)}] {event_id}: ERROR {type(exc).__name__}: {exc}")

    ordered = {event_id: result[event_id] for event_id in sorted(result)}
    _write_json(args.output, ordered)
    args.site_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.output, args.site_copy)

    print()
    print("========================================")
    print(f"Skipped existing            : {skipped}")
    print(f"Fetched/updated             : {fetched}")
    print(f"Usable focal mechanisms     : {len(ordered)} / {len(viewer_ids)}")
    print(f"Coverage                    : {100.0 * len(ordered) / len(viewer_ids):.1f}%")
    print(f"Canonical file              : {args.output}")
    print(f"Site copy                   : {args.site_copy}")
    if failures:
        print(f"Still missing after retries : {len(failures)}")
        print("                               " + ", ".join(failures))
    print("========================================")


if __name__ == "__main__":
    main()
