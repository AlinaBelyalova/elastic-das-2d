"""Shared configuration for the SAFOD real-event modelling workflow.

The physical cable geometry is unchanged between events.  What can change
between interrogator configurations is the mapping from HDF5 data rows to that
physical geometry, together with acquisition parameters such as channel
spacing and gauge length.

Select an event with:

    SAFOD_EVENT_KEY=nc75379261 python -m scripts.safod.prepare_event
"""

from __future__ import annotations

import glob
import math
import os
from pathlib import Path
from typing import Any


# ==============================================================================
# EXTERNAL PATHS
# ==============================================================================

SAFOD_EVENT_DATA_ROOT = Path(
    "/oak/stanford/groups/ettore88/data/SAFOD/SAFOD_events"
)

# One unchanged physical cable-geometry template for all events.
GEO_XLSX = Path(
    "/home/groups/ettore88/alina/SAFOD/"
    "SAFOD_Phase2_GeoReferenced_Channels.xlsx"
)

DAS_UTILITIES_ROOT = Path(
    "/home/groups/ettore88/alina/packages/DAS-utilities"
)

DAS_UTILITIES_BUILD = DAS_UTILITIES_ROOT / "build"
DAS_UTILITIES_PYTHON = DAS_UTILITIES_ROOT / "python"
DAS_DB_PY = DAS_UTILITIES_PYTHON / "DAS_db.py"
DESAMPLE_DAS_PY = DAS_UTILITIES_PYTHON / "Desample_DAS.py"

GEO_PYTHON = Path(
    "/home/groups/ettore88/alina/.envs/geo/bin/python3"
)

DAS_SYSTEM_NAME = "SAFOD_QuantX"


# ==============================================================================
# EVENT REGISTRY
# ==============================================================================

DEFAULT_EVENT_KEY = "nc75379261"

EVENTS: dict[str, dict[str, Any]] = {
    "nc75336802": {
        "event_id": "NC75336802",
        "ncedc_event_id": 75336802,
        "origin_time": "2026-04-01T04:57:57.470000Z",
        "magnitude": 0.77,
        "magnitude_type": "Md",
        "depth_km": 1.57,
        "event_tag": "20260401_75336802",
        "input_patterns": [
            str(
                SAFOD_EVENT_DATA_ROOT
                / (
                    "SAFOD-Deep-10mGL-1000HzFs-2mChDualPulse_"
                    "2026-04-01T045735Z.h5"
                )
            ),
        ],
        # April channel numbers already match the reference Excel geometry.
        "channel_mapping_csv": None,
    },
    "nc75379261": {
        "event_id": "NC75379261",
        "ncedc_event_id": 75379261,
        "origin_time": "2026-06-18T21:04:42.290000Z",
        "magnitude": 1.61,
        "magnitude_type": "Md",
        "depth_km": 3.43,
        "event_tag": "20260618_75379261",
        "input_patterns": [
            str(
                SAFOD_EVENT_DATA_ROOT
                / "SAFOD_QuantX-2026-06-18T210347Z.h5"
            ),
        ],
        # Same physical cable geometry, but June HDF5 rows use a different
        # interrogator registration.  This CSV maps June rows to the unchanged
        # physical MD/TVD/X/Z trajectory.
        "channel_mapping_csv": (
            "results/real_event_20260618_75379261/"
            "channel_registration_qc/"
            "mapped_downleg_model_geometry.csv"
        ),
    },
}


def normalize_event_key(value: str) -> str:
    """Normalize an event selector to a configured key."""
    raw = str(value).strip().lower()

    aliases = {
        "75336802": "nc75336802",
        "nc75336802": "nc75336802",
        "75379261": "nc75379261",
        "nc75379261": "nc75379261",
    }

    if raw not in aliases:
        configured = ", ".join(sorted(EVENTS))
        raise ValueError(
            f"Unknown SAFOD event key {value!r}. "
            f"Configured keys: {configured}."
        )

    return aliases[raw]


def get_event_config(event_key: str) -> dict[str, Any]:
    """Return a defensive copy of one event configuration."""
    return dict(EVENTS[normalize_event_key(event_key)])


def resolve_event_input_files(
    event_key: str,
    *,
    require_exists: bool = True,
) -> list[str]:
    """Expand configured HDF5 paths or glob patterns."""
    config = get_event_config(event_key)

    resolved: list[str] = []
    unmatched: list[str] = []

    for pattern in config["input_patterns"]:
        matches = sorted(glob.glob(str(pattern)))

        if matches:
            resolved.extend(matches)
        else:
            unmatched.append(str(pattern))

    resolved = list(dict.fromkeys(resolved))

    if require_exists and unmatched:
        formatted = "\n".join(
            f"  - {pattern}"
            for pattern in unmatched
        )
        raise FileNotFoundError(
            "No files matched one or more configured event inputs:\n"
            f"{formatted}"
        )

    if require_exists and not resolved:
        raise FileNotFoundError(
            f"No HDF5 inputs resolved for event {event_key!r}."
        )

    return resolved


ACTIVE_EVENT_KEY = normalize_event_key(
    os.environ.get(
        "SAFOD_EVENT_KEY",
        DEFAULT_EVENT_KEY,
    )
)

ACTIVE_EVENT = get_event_config(ACTIVE_EVENT_KEY)

EVENT = {
    "event_id": ACTIVE_EVENT["event_id"],
    "ncedc_event_id": ACTIVE_EVENT["ncedc_event_id"],
    "origin_time": ACTIVE_EVENT["origin_time"],
}

EVENT_TAG = str(ACTIVE_EVENT["event_tag"])

SELECTED_FILES = resolve_event_input_files(
    ACTIVE_EVENT_KEY,
    require_exists=True,
)

_mapping_value = ACTIVE_EVENT.get("channel_mapping_csv")

CHANNEL_MAPPING_CSV = (
    Path(_mapping_value)
    if _mapping_value is not None
    else None
)


# ==============================================================================
# PREPARED REAL EVENT
# ==============================================================================

REAL_EVENT_DIR = Path(
    f"results/real_event_{EVENT_TAG}"
)

REAL_EVENT_PACKAGE = (
    REAL_EVENT_DIR
    / "real_das_event_window.npz"
)

# Event-specific geometry exported by prepare_event.py and consumed by
# run_forward.py / compare_event.py.
GEOMETRY_CSV = (
    REAL_EVENT_DIR
    / "SAFOD_Phase2_projected_from_georef.csv"
)


# ==============================================================================
# REAL-DATA PREPARATION / COMMON COMPARISON FILTER
# ==============================================================================

PREP_TMIN_S = -2.0
PREP_TMAX_S = 15.0

COMMON_FMIN_HZ = 1.0
COMMON_FMAX_HZ = 20.0
FILTER_ORDER = 4
FILTER_TAPER_FRAC = 0.05


# ==============================================================================
# FORWARD / COMPARISON RUN NAMING
# ==============================================================================

DEFAULT_THETA_DEG = 35.0
FORWARD_BASE_TAG = "n120_g80_xplus500"

FORWARD_ROOT = Path(
    f"results/forward_real_event_{EVENT_TAG}"
)

COMPARISON_ROOT = Path(
    f"results/compare_real_synthetic_{EVENT_TAG}"
)


def format_theta_tag(theta_deg: float) -> str:
    """Return a deterministic filesystem-safe source-angle tag."""
    theta_deg = float(theta_deg)

    if not math.isfinite(theta_deg):
        raise ValueError(
            f"theta_deg must be finite; got {theta_deg!r}."
        )

    if not 0.0 <= theta_deg < 90.0:
        raise ValueError(
            "For the current 2D double-couple parameterisation, "
            "theta_deg must satisfy 0 <= theta_deg < 90."
        )

    nearest_integer = round(theta_deg)

    if math.isclose(
        theta_deg,
        nearest_integer,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        return f"{int(nearest_integer):03d}"

    return f"{theta_deg:05.1f}".replace(".", "p")


def forward_run_tag(theta_deg: float) -> str:
    return (
        f"{FORWARD_BASE_TAG}_dc"
        f"{format_theta_tag(theta_deg)}"
    )


def forward_dir_for_theta(theta_deg: float) -> Path:
    return (
        FORWARD_ROOT
        / f"dc{format_theta_tag(theta_deg)}"
    )


def forward_package_for_theta(theta_deg: float) -> Path:
    return (
        forward_dir_for_theta(theta_deg)
        / "outputs_safod_initial_forward.npz"
    )


def comparison_dir_for_theta(theta_deg: float) -> Path:
    return (
        COMPARISON_ROOT
        / f"dc{format_theta_tag(theta_deg)}"
    )


FORWARD_RUN_TAG = forward_run_tag(DEFAULT_THETA_DEG)
FORWARD_DIR = forward_dir_for_theta(DEFAULT_THETA_DEG)
FORWARD_PACKAGE = forward_package_for_theta(DEFAULT_THETA_DEG)
COMPARISON_DIR = comparison_dir_for_theta(DEFAULT_THETA_DEG)