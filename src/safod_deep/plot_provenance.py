"""Keep rendered catalog labels and their JSON consistent with catalog rows."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


NUMERIC_LABEL_FIELDS = (
    "magnitude", "depth_km", "min_3d_distance_to_cable_km", "source_crossline_m",
)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def require_current_plot_labels(record: Mapping[str, Any], catalog_row) -> None:
    """Reject stale products; changing JSON alone cannot repair a PNG title.

    Missing values compare equal. Tolerance only permits CSV/JSON round-off,
    not geometry changes. Older metadata missing a finite label must rerender.
    """
    changed = []
    for key in NUMERIC_LABEL_FIELDS:
        if key not in catalog_row:
            continue
        old, new = _finite(record.get(key)), _finite(catalog_row[key])
        if old is None or new is None:
            equal = old is new
        else:
            equal = math.isclose(old, new, rel_tol=1e-12, abs_tol=1e-9)
        if not equal:
            changed.append(key)
    if "geometry_2d_class" in catalog_row:
        if (str(record.get("geometry_2d_class", "unknown"))
                != str(catalog_row["geometry_2d_class"])):
            changed.append("geometry_2d_class")
    if changed:
        raise ValueError(
            f"Stale plot labels for event {catalog_row.get('event_id', '?')}: "
            f"{', '.join(changed)}. Rerun scripts.safod_deep.plot_catalog_events "
            "with --overwrite in the existing output directory before building "
            "the gallery/site."
        )
