"""Shared configuration for the current SAFOD event workflow."""

from __future__ import annotations

from pathlib import Path


EVENT = {
    "event_id": "NC75336802",
    "ncedc_event_id": 75336802,
    "origin_time": "2026-04-01T04:57:57.470000Z",
}

EVENT_TAG = "20260401_75336802"
FORWARD_RUN_TAG = "n120_g80"

SELECTED_FILES = [
    (
        "/oak/stanford/groups/ettore88/data/SAFOD/SAFOD_events/"
        "SAFOD-Deep-10mGL-1000HzFs-2mChDualPulse_"
        "2026-04-01T045735Z.h5"
    ),
]

GEO_XLSX = (
    "/home/groups/ettore88/alina/SAFOD/"
    "SAFOD_Phase2_GeoReferenced_Channels.xlsx"
)
DAS_DB_PY = (
    "/home/groups/ettore88/alina/packages/"
    "DAS-utilities/python/DAS_db.py"
)
DAS_SYSTEM_NAME = "SAFOD_QuantX"

REAL_EVENT_DIR = Path(f"results/real_event_{EVENT_TAG}")
REAL_EVENT_PACKAGE = REAL_EVENT_DIR / "real_das_event_window.npz"
GEOMETRY_CSV = REAL_EVENT_DIR / "SAFOD_Phase2_projected_from_georef.csv"

FORWARD_DIR = Path(
    f"results/forward_real_event_{EVENT_TAG}_{FORWARD_RUN_TAG}"
)
FORWARD_PACKAGE = FORWARD_DIR / "outputs_safod_initial_forward.npz"

COMPARISON_DIR = Path(
    f"results/compare_real_synthetic_{EVENT_TAG}_{FORWARD_RUN_TAG}"
)

PREP_TMIN_S = -2.0
PREP_TMAX_S = 15.0

COMMON_FMIN_HZ = 1.0
COMMON_FMAX_HZ = 20.0
FILTER_ORDER = 4
FILTER_TAPER_FRAC = 0.05
