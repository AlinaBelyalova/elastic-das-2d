"""Reproducible WGS84 -> SAFOD NAD27/UTM 10N horizontal transformation.

EPSG:1750 inverse + EPSG:1241 inverse + UTM 10N on Clarke 1866.
Use the same California-South NADCON/HARN route as the verified 431-event
catalog. The two small NOAA grids are vendored and checked by SHA256.
No environment-dependent operation selection, network lookup or fallback.
Vertical coordinates are deliberately not transformed by this module.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pyproj
from pyproj import Transformer


TRANSFORM_ID = "safod_wgs84_nad27_noaa_cs_v1"
ACCURACY_M = 2.15  # EPSG's stated operation accuracy, not an error guarantee.
SOURCE_BOUNDS = (-121.98, 32.53, -114.12, 36.50)  # W, S, E, N; EPSG:1750.
GRID_DIR = Path(__file__).resolve().with_name("nad27_grids")
GRID_SHA256 = {
    "us_noaa_conus.tif": "44611d823c48e5347500ee6afe40ff33d2b88cf817bf59f705ed4a4c3bd687d7",
    "us_noaa_cshpgn.tif": "d54183c832e0366fc558147961bd8ae7d0dd3b5ae945069afd6a37a5a5c604c9",
}


def require_source_coverage(longitude: float, latitude: float) -> None:
    """Keep this regional operation inside its documented area of use."""
    west, south, east, north = SOURCE_BOUNDS
    if not (math.isfinite(longitude) and math.isfinite(latitude)
            and west <= longitude <= east and south <= latitude <= north):
        raise ValueError(
            f"Source ({longitude}, {latitude}) outside pinned SAFOD NAD27 "
            f"operation bounds {SOURCE_BOUNDS}; no automatic fallback is allowed."
        )


def make_transformer() -> Transformer:
    for name, expected in GRID_SHA256.items():
        path = GRID_DIR / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned SAFOD datum grid: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch for pinned SAFOD datum grid: {path}")
    # Absolute paths prevent PROJ_DATA, a user's grid cache, or a newer grid
    # package from silently selecting different files. Quotes permit spaces.
    return Transformer.from_pipeline(
        "+proj=pipeline "
        "+step +proj=unitconvert +xy_in=deg +xy_out=rad "
        f'+step +inv +proj=hgridshift +grids="{GRID_DIR / "us_noaa_cshpgn.tif"}" '
        f'+step +inv +proj=hgridshift +grids="{GRID_DIR / "us_noaa_conus.tif"}" '
        "+step +proj=utm +zone=10 +ellps=clrk66"
    )


def provenance() -> dict:
    return {
        "id": TRANSFORM_ID,
        "source_crs": "EPSG:4326",
        "target_crs": "EPSG:26710",
        "operation": "Inverse EPSG:1750 + inverse EPSG:1241 + UTM zone 10N",
        "stated_accuracy_m": ACCURACY_M,
        "source_bounds_wsen": list(SOURCE_BOUNDS),
        "grid_sha256": dict(GRID_SHA256),
        "runtime": {"pyproj": pyproj.__version__, "proj": pyproj.proj_version_str},
        "vertical_coordinates": "unchanged; cable MSL elevation = wellhead elevation - TVD",
    }
