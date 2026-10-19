from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .assets import ensure_web_plots, load_cache, save_cache
from .catalog import event_record_from_row, load_catalog, load_metadata_inventory
from .config import resolve_path
from .geometry import export_safod_geojson, fault_query_url


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _copy_templates(
    template_dir: Path,
    site_dir: Path,
    build_id: str,
) -> None:
    site_dir.mkdir(parents=True, exist_ok=True)
    for name in ("app.js", "style.css"):
        shutil.copy2(template_dir / name, site_dir / name)

    html = (template_dir / "index.html").read_text(encoding="utf-8")
    html = html.replace("__BUILD_ID__", build_id)
    (site_dir / "index.html").write_text(html, encoding="utf-8")
    (site_dir / ".nojekyll").write_text("", encoding="utf-8")


def build_site(
    config: dict[str, Any],
    *,
    overwrite_images: bool = False,
) -> dict[str, Any]:
    repo_root = Path(config["_repo_root"])
    catalog_path = resolve_path(config, "catalog")
    metadata_dir = resolve_path(config, "event_metadata_dir")
    event_qc_dir = resolve_path(config, "event_qc_dir")
    geometry_xlsx = resolve_path(config, "geometry_xlsx")
    site_dir = resolve_path(config, "site_dir")
    fault_geojson = resolve_path(config, "fault_geojson")

    template_dir = repo_root / "web_safod/templates"
    cache_path = repo_root / "web_safod/cache/event_inventory.json"
    report_path = repo_root / "web_safod/cache/build_report.json"

    frame = load_catalog(catalog_path)
    inventory = load_metadata_inventory(metadata_dir)

    build_time = datetime.now(timezone.utc)
    build_id = build_time.strftime("%Y%m%dT%H%M%SZ")

    _copy_templates(template_dir, site_dir, build_id)

    data_dir = site_dir / "data"
    plots_dir = site_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Focal mechanisms are maintained as a small scientific sidecar catalog.
    # Keep the canonical copy outside site/ so a normal site rebuild cannot
    # silently erase downloaded mechanisms.
    focal_catalog = repo_root / "results/safod_deep/catalog/focal_mechanisms.json"
    focal_site = data_dir / "focal_mechanisms.json"
    focal_count = 0
    if focal_catalog.exists():
        shutil.copy2(focal_catalog, focal_site)
        try:
            focal_payload = json.loads(focal_catalog.read_text(encoding="utf-8"))
            if isinstance(focal_payload, dict):
                focal_count = len(focal_payload)
        except Exception:
            focal_count = 0
    elif focal_site.exists():
        focal_site.unlink()

    export_safod_geojson(
        geometry_xlsx,
        data_dir / "safod_geometry.geojson",
    )

    if fault_geojson.exists():
        shutil.copy2(
            fault_geojson,
            data_dir / "san_andreas_fault.geojson",
        )
    else:
        # The browser has a remote USGS fallback. Do not create a fake fault file.
        target = data_dir / "san_andreas_fault.geojson"
        if target.exists():
            target.unlink()

    cache = load_cache(cache_path)
    records: list[dict[str, Any]] = []
    warnings: list[str] = []

    for _, row in frame.iterrows():
        event_id = str(row["event_id"]).strip()
        metadata_rows = inventory.get(event_id, [])

        plots, plot_warnings = ensure_web_plots(
            event_id,
            metadata_rows,
            repo_root=repo_root,
            event_qc_dir=event_qc_dir,
            output_dir=plots_dir,
            cache=cache,
            max_long_edge_px=int(config["web_image_max_long_edge_px"]),
            quality=int(config["web_image_quality"]),
            overwrite=overwrite_images,
        )
        warnings.extend(plot_warnings)

        records.append(
            event_record_from_row(
                row,
                metadata_rows=metadata_rows,
                plots=plots,
                event_page_template=str(
                    config["public_event_page_template"]
                ),
            )
        )

    save_cache(cache_path, cache)

    # Keep only plot files currently referenced by the event inventory.
    referenced = set()
    for event in records:
        for plot in event.get("plots", []):
            if plot.get("url"):
                referenced.add(Path(plot["url"]).name)
            if plot.get("arrivals_url"):
                referenced.add(Path(plot["arrivals_url"]).name)
    for path in plots_dir.glob("*.webp"):
        if path.name not in referenced:
            path.unlink()

    _write_json(data_dir / "events.json", records)

    recording_start = (
        min((r["origin_time_utc"] for r in records), default=None)
    )
    recording_end = (
        max((r["origin_time_utc"] for r in records), default=None)
    )

    site_status = {
        "build_id": build_id,
        "generated_utc": build_time.isoformat().replace("+00:00", "Z"),
        "event_time_start_utc": recording_start,
        "event_time_end_utc": recording_end,
        "recorded_events": len(records),
        "plots_available": sum(
            bool(record.get("plot_available")) for record in records
        ),
        "focal_mechanisms_available": int(focal_count),
        "catalog_sha256": _sha256(catalog_path),
        "catalog_name": catalog_path.name,
        "elastic_das_git_commit": _git_commit(repo_root),
        "web_product_version": 2,
    }
    _write_json(data_dir / "site_status.json", site_status)

    remote_fault_url = fault_query_url(
        str(config["fault_arcgis_layer"]),
        list(config["fault_bbox"]),
    )

    site_config = {
        "build_id": build_id,
        "map_center": list(config["map_center"]),
        "map_default_zoom": int(config["map_default_zoom"]),
        "fault_bbox": list(config["fault_bbox"]),
        "fault_remote_query_url": remote_fault_url,
        "web_product_version": 2,
    }
    _write_json(data_dir / "site_config.json", site_config)

    report = {
        "build_id": build_id,
        "catalog": str(catalog_path),
        "metadata_dir": str(metadata_dir),
        "event_qc_dir": str(event_qc_dir),
        "geometry_xlsx": str(geometry_xlsx),
        "site_dir": str(site_dir),
        "events": len(records),
        "plots_available": site_status["plots_available"],
        "warnings": warnings,
    }
    _write_json(report_path, report)
    return report
