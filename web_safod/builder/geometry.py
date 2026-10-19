from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


def export_safod_geojson(
    xlsx_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if not xlsx_path.exists():
        raise FileNotFoundError(xlsx_path)

    frame = pd.read_excel(xlsx_path).copy()
    required = {
        "Channel",
        "Section",
        "MD_m",
        "TVD_m",
        "Lat_WGS84",
        "Lon_WGS84",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"SAFOD geometry is missing columns {missing}: {xlsx_path}"
        )

    for col in ("Channel", "MD_m", "TVD_m", "Lat_WGS84", "Lon_WGS84"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame["Section"] = frame["Section"].astype(str).str.strip()
    frame = frame.dropna(
        subset=["Channel", "Lat_WGS84", "Lon_WGS84"]
    ).sort_values("Channel")

    features: list[dict[str, Any]] = []

    section_order = ["Surface Spool", "Down-leg", "Up-leg"]
    for section in section_order:
        part = frame[frame["Section"] == section].copy()
        if len(part) < 2:
            continue
        coords = [
            [float(row.Lon_WGS84), float(row.Lat_WGS84)]
            for row in part.itertuples()
        ]
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "kind": "fibre_section",
                    "section": section,
                    "name": f"SAFOD {section}",
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords,
                },
            }
        )

    down = frame[frame["Section"] == "Down-leg"].copy()
    if not down.empty:
        down = down.dropna(subset=["MD_m"])
        if not down.empty:
            wellhead = down.loc[down["MD_m"].idxmin()]
            turn = down.loc[down["MD_m"].idxmax()]

            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "kind": "wellhead",
                        "name": "SAFOD wellhead",
                        "channel": float(wellhead["Channel"]),
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [
                            float(wellhead["Lon_WGS84"]),
                            float(wellhead["Lat_WGS84"]),
                        ],
                    },
                }
            )
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "kind": "turnaround",
                        "name": "SAFOD turnaround",
                        "channel": float(turn["Channel"]),
                        "md_m": float(turn["MD_m"]),
                        "tvd_m": (
                            None
                            if pd.isna(turn["TVD_m"])
                            else float(turn["TVD_m"])
                        ),
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [
                            float(turn["Lon_WGS84"]),
                            float(turn["Lat_WGS84"]),
                        ],
                    },
                }
            )

    geojson = {
        "type": "FeatureCollection",
        "name": "SAFOD georeferenced fibre geometry",
        "features": features,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(geojson, indent=2),
        encoding="utf-8",
    )
    return geojson


def fault_query_url(
    arcgis_layer: str,
    bbox: list[float],
) -> str:
    if len(bbox) != 4:
        raise ValueError("fault_bbox must be [west, south, east, north].")
    west, south, east, north = [float(x) for x in bbox]
    params = {
        "where": "1=1",
        "geometry": f"{west},{south},{east},{north}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
    }
    return arcgis_layer + "?" + urlencode(params)


def download_san_andreas_fault(
    *,
    arcgis_layer: str,
    bbox: list[float],
    output_path: Path,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    url = fault_query_url(arcgis_layer, bbox)
    request = Request(
        url,
        headers={
            "User-Agent": "SAFOD-Deep-DAS-Event-Explorer/1.0"
        },
    )
    with urlopen(request, timeout=float(timeout_s)) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if payload.get("type") != "FeatureCollection":
        raise RuntimeError(
            "USGS ArcGIS response is not a GeoJSON FeatureCollection."
        )

    selected = []
    for feature in payload.get("features", []):
        props = feature.get("properties") or {}
        name = str(props.get("fault_name", ""))
        if "san andreas" in name.lower():
            selected.append(feature)

    if not selected:
        raise RuntimeError(
            "USGS query returned no San Andreas Fault features "
            "inside the configured bounding box."
        )

    output = {
        "type": "FeatureCollection",
        "name": "San Andreas Fault near SAFOD",
        "source": (
            "USGS Quaternary Fault and Fold Database, California layer"
        ),
        "source_query": url,
        "features": selected,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    return output
