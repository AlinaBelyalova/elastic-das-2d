"""Read-only check that a catalog matches the pinned SAFOD horizontal datum."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.safod.nad27 import provenance
from src.safod.reference import SAFOD_WELLHEAD_LAT_WGS84, SAFOD_WELLHEAD_LON_WGS84
from src.safod_deep.catalog import CableGeometry3D


# All differences reported in metres, including distance stored in kilometres.
FIELDS = {
    "source_east_m": 1., "source_north_m": 1., "source_elevation_m": 1.,
    "source_section_x_m": 1., "source_crossline_m": 1.,
    "min_3d_distance_to_cable_km": 1000.,
}


def audit_catalog(catalog_path: Path, geometry_path: Path,
                  wellhead_elevation_m: float = 660.5) -> dict:
    frame = pd.read_csv(catalog_path)
    if frame.empty or frame.event_id.duplicated().any():
        raise ValueError("Catalog must be nonempty with unique event IDs.")
    cable = CableGeometry3D.from_file(
        geometry_path, wellhead_elevation_m=wellhead_elevation_m,
        profile_origin=(SAFOD_WELLHEAD_LAT_WGS84, SAFOD_WELLHEAD_LON_WGS84),
    )
    maxima = {field: 0. for field in FIELDS}
    mismatches = []
    for _, row in frame.iterrows():
        metrics = cable.event_metrics(longitude=row.longitude,
            latitude=row.latitude, depth_km=row.depth_km)
        failed = []
        for field, scale in FIELDS.items():
            delta = abs(float(row[field]) - float(metrics[field])) * scale
            if not np.isfinite(delta):
                raise ValueError(f"Nonfinite {field} for event {row.event_id}")
            maxima[field] = max(maxima[field], delta)
            if delta > 1e-6:  # numerical reproduction tolerance, not survey accuracy
                failed.append(field)
        if row.geometry_2d_class != metrics["geometry_2d_class"]:
            failed.append("geometry_2d_class")
        if failed:
            mismatches.append({"event_id": str(row.event_id), "fields": failed})
    return {
        "checked_events": len(frame), "passed": not mismatches,
        "horizontal_transform": provenance(), "maximum_difference_m": maxima,
        "mismatch_count": len(mismatches), "first_mismatches": mismatches[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path,
        default=Path("results/safod_deep/catalog/catalog_recorded.csv"))
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--wellhead-elevation-m", type=float, default=660.5)
    args = parser.parse_args()
    report = audit_catalog(args.catalog, args.geometry, args.wellhead_elevation_m)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
