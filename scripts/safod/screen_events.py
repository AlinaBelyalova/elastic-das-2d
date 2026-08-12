from __future__ import annotations

from pathlib import Path
import importlib.util
import re

import numpy as np
import pandas as pd

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import URL_MAPPINGS

from scripts.safod.settings import (
    DAS_DB_PY,
    DAS_SYSTEM_NAME,
    SAFOD_WELLHEAD_LAT_WGS84,
    SAFOD_WELLHEAD_LON_WGS84,
)

from scripts.safod.prepare_event import (
    _load_reference_geometry_context,
    latlon_to_local_enu_m,
)


# ==============================================================================
# SETTINGS
# ==============================================================================

DATA_DIR = Path(
    "/oak/stanford/groups/ettore88/data/SAFOD/SAFOD_events"
)

OUT_CSV = Path(
    "results/safod_event_screening.csv"
)

# Search radius around SAFOD wellhead.
# 0.20 deg is ~22 km in latitude, similar to prepare_event fallback.
MAX_RADIUS_DEG = 0.20

MIN_MAG = 0.0
MAX_DEPTH_KM = 20.0

# Required real-data coverage around origin for our current preparation window.
REQUIRED_PRE_S = 2.0
REQUIRED_POST_S = 15.0

# Treat files separated by <= this as continuous recording.
CONTIGUOUS_GAP_TOL_S = 0.05


# ==============================================================================
# HDF5 HEADER READER
# ==============================================================================

def load_das_db_module():
    spec = importlib.util.spec_from_file_location(
        "das_db_external",
        DAS_DB_PY,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not import DAS_db.py from {DAS_DB_PY}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def read_file_intervals(files: list[Path]) -> pd.DataFrame:
    """
    Read true start/end times from DAS headers rather than assuming
    that every HDF5 filename represents exactly one minute.
    """
    das_db = load_das_db_module()

    rows = []

    for path in files:
        try:
            df = das_db.get_header_df(
                str(path),
                DAS_SYSTEM_NAME,
            )

            if df.empty:
                print(f"WARNING: empty header: {path}")
                continue

            row = df.iloc[0]

            start = pd.Timestamp(row["startTime"])
            end = pd.Timestamp(row["endTime"])

            if start.tzinfo is None:
                start = start.tz_localize("UTC")
            else:
                start = start.tz_convert("UTC")

            if end.tzinfo is None:
                end = end.tz_localize("UTC")
            else:
                end = end.tz_convert("UTC")

            rows.append(
                {
                    "file": str(path),
                    "filename": path.name,
                    "start": start,
                    "end": end,
                    "duration_s": (
                        end - start
                    ).total_seconds(),
                }
            )

        except Exception as exc:
            print(
                f"WARNING: failed to read header for {path}: {exc}"
            )

    if not rows:
        raise RuntimeError(
            f"No valid HDF5 headers found in {DATA_DIR}"
        )

    return (
        pd.DataFrame(rows)
        .sort_values("start")
        .reset_index(drop=True)
    )


# ==============================================================================
# MERGE CONTIGUOUS RECORDING WINDOWS
# ==============================================================================

def merge_recording_windows(
    file_table: pd.DataFrame,
) -> list[dict]:

    windows = []

    current = None

    for row in file_table.itertuples(index=False):
        start = row.start
        end = row.end

        if current is None:
            current = {
                "start": start,
                "end": end,
                "files": [row.file],
            }
            continue

        gap_s = (
            start - current["end"]
        ).total_seconds()

        if gap_s <= CONTIGUOUS_GAP_TOL_S:
            current["end"] = max(
                current["end"],
                end,
            )
            current["files"].append(row.file)

        else:
            windows.append(current)

            current = {
                "start": start,
                "end": end,
                "files": [row.file],
            }

    if current is not None:
        windows.append(current)

    return windows


# ==============================================================================
# EVENT HELPERS
# ==============================================================================

def extract_nc_event_id(event) -> str:
    """
    Try to recover NC event id from ObsPy/QuakeML resource strings.
    """
    strings = []

    if event.resource_id is not None:
        strings.append(str(event.resource_id))

    for origin in event.origins:
        if origin.resource_id is not None:
            strings.append(str(origin.resource_id))

    for magnitude in event.magnitudes:
        if magnitude.resource_id is not None:
            strings.append(str(magnitude.resource_id))

    for text in strings:
        match = re.search(
            r"(nc)?(\d{7,9})",
            text,
            flags=re.IGNORECASE,
        )

        if match:
            return f"NC{match.group(2)}"

    return "UNKNOWN"


def event_projection(
    *,
    lat: float,
    lon: float,
    depth_km: float,
    context: dict,
) -> dict:

    east, north = latlon_to_local_enu_m(
        lat,
        lon,
        SAFOD_WELLHEAD_LAT_WGS84,
        SAFOD_WELLHEAD_LON_WGS84,
    )

    east = float(east)
    north = float(north)

    along = (
        east * context["u_e"]
        + north * context["u_n"]
    )

    cross = (
        -east * context["u_n"]
        + north * context["u_e"]
    )

    horizontal = float(
        np.hypot(east, north)
    )

    return {
        "east_m": east,
        "north_m": north,
        "horizontal_m": horizontal,
        "along_m": float(along),
        "crossline_m": float(cross),
        "abs_crossline_m": float(abs(cross)),
        "depth_km": float(depth_km),
    }


def candidate_score(
    *,
    mag: float,
    abs_crossline_m: float,
    horizontal_m: float,
    has_full_window: bool,
) -> float:
    """
    Ranking heuristic only.

    Geometry matters more than magnitude for a 2-D experiment.
    Lower score = better.
    """

    score = 0.0

    # Strong penalty for leaving the 2-D model plane.
    score += abs_crossline_m / 500.0

    # Mild penalty for total horizontal distance.
    score += horizontal_m / 5000.0

    # Reward magnitude.
    score -= 1.5 * mag

    # Hard-ish penalty if our full -2/+15 s window is unavailable.
    if not has_full_window:
        score += 5.0

    return float(score)


# ==============================================================================
# MAIN
# ==============================================================================

def main():

    files = sorted(DATA_DIR.glob("*.h5"))

    if not files:
        raise FileNotFoundError(
            f"No *.h5 files found in {DATA_DIR}"
        )

    print(f"Found {len(files)} HDF5 files.")

    # --------------------------------------------------------------------------
    # 1. Read exact recording intervals
    # --------------------------------------------------------------------------

    file_table = read_file_intervals(files)

    print("\nRecorded files")
    print("--------------")

    print(
        file_table[
            [
                "filename",
                "start",
                "end",
                "duration_s",
            ]
        ].to_string(index=False)
    )

    windows = merge_recording_windows(file_table)

    print("\nContinuous recording windows")
    print("----------------------------")

    for i, window in enumerate(windows, start=1):
        duration = (
            window["end"] - window["start"]
        ).total_seconds()

        print(
            f"{i:02d}: "
            f"{window['start']} -> "
            f"{window['end']} "
            f"({duration:.1f} s, "
            f"{len(window['files'])} file(s))"
        )

    # --------------------------------------------------------------------------
    # 2. Load the exact same SAFOD profile system as prepare_event.py
    # --------------------------------------------------------------------------

    context = _load_reference_geometry_context()

    print("\nSAFOD 2-D profile")
    print("-----------------")
    print(
        "profile unit vector EN : "
        f"({context['u_e']:.4f}, {context['u_n']:.4f})"
    )

    # --------------------------------------------------------------------------
    # 3. Query NCEDC for every recorded time interval
    # --------------------------------------------------------------------------

    URL_MAPPINGS["NCEDC"] = "https://service.ncedc.org"

    client_ncedc = Client("NCEDC")
    client_usgs = Client("USGS")

    candidates = []

    for i, window in enumerate(windows, start=1):

        start = UTCDateTime(
            window["start"].to_pydatetime()
        )
        end = UTCDateTime(
            window["end"].to_pydatetime()
        )

        print(
            f"\nQuerying window {i:02d}: "
            f"{start.isoformat()} -> {end.isoformat()}"
        )

        query_kwargs = dict(
            starttime=start,
            endtime=end,
            latitude=SAFOD_WELLHEAD_LAT_WGS84,
            longitude=SAFOD_WELLHEAD_LON_WGS84,
            maxradius=MAX_RADIUS_DEG,
            minmagnitude=MIN_MAG,
            maxdepth=MAX_DEPTH_KM,
            orderby="time",
        )

        try:
            catalog = client_ncedc.get_events(
                **query_kwargs
            )
            catalog_source = "NCEDC"

        except Exception as exc:
            print(
                f"  NCEDC failed: {exc}"
            )
            print(
                "  Retrying same window with USGS..."
            )

            try:
                catalog = client_usgs.get_events(
                    **query_kwargs
                )
                catalog_source = "USGS"

            except Exception as exc2:
                print(
                    f"WARNING: USGS fallback also failed: {exc2}"
                )
                continue

        print(
            f"  events found: {len(catalog)} "
            f"[{catalog_source}]"
        )

        for event in catalog:

            origin = (
                event.preferred_origin()
                or event.origins[0]
            )

            magnitude = (
                event.preferred_magnitude()
                or event.magnitudes[0]
            )

            if origin.latitude is None:
                continue
            if origin.longitude is None:
                continue
            if origin.depth is None:
                continue
            if magnitude.mag is None:
                continue

            event_time = pd.Timestamp(
                origin.time.datetime,
                tz="UTC",
            )

            mag = float(magnitude.mag)
            mag_type = str(
                magnitude.magnitude_type or ""
            )

            depth_km = (
                float(origin.depth) / 1000.0
            )

            projection = event_projection(
                lat=float(origin.latitude),
                lon=float(origin.longitude),
                depth_km=depth_km,
                context=context,
            )

            pre_s = (
                event_time - window["start"]
            ).total_seconds()

            post_s = (
                window["end"] - event_time
            ).total_seconds()

            has_full_window = (
                pre_s >= REQUIRED_PRE_S
                and post_s >= REQUIRED_POST_S
            )

            event_id = extract_nc_event_id(event)

            score = candidate_score(
                mag=mag,
                abs_crossline_m=projection[
                    "abs_crossline_m"
                ],
                horizontal_m=projection[
                    "horizontal_m"
                ],
                has_full_window=has_full_window,
            )

            candidates.append(
                {
                    "event_id": event_id,
                    "origin_time": origin.time.isoformat(),
                    "mag": mag,
                    "mag_type": mag_type,
                    "lat": float(origin.latitude),
                    "lon": float(origin.longitude),
                    "depth_km": depth_km,

                    "east_m": projection["east_m"],
                    "north_m": projection["north_m"],
                    "horizontal_m": projection["horizontal_m"],
                    "along_m": projection["along_m"],
                    "crossline_m": projection["crossline_m"],
                    "abs_crossline_m": projection[
                        "abs_crossline_m"
                    ],

                    "pre_available_s": pre_s,
                    "post_available_s": post_s,
                    "full_prep_window": has_full_window,

                    "n_files_in_window": len(
                        window["files"]
                    ),
                    "files": " | ".join(
                        window["files"]
                    ),

                    "score": score,
                }
            )

    if not candidates:
        print("\nNo catalog events found in recorded windows.")
        return

    # --------------------------------------------------------------------------
    # 4. Deduplicate + rank
    # --------------------------------------------------------------------------

    df = pd.DataFrame(candidates)

    # Same event could theoretically appear in adjacent/overlapping windows.
    df = (
        df.sort_values(
            [
                "event_id",
                "full_prep_window",
                "post_available_s",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
        .drop_duplicates(
            subset=[
                "event_id",
                "origin_time",
            ],
            keep="first",
        )
    )

    df = (
        df.sort_values(
            [
                "score",
                "abs_crossline_m",
                "mag",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    # Useful categorical interpretation.
    df["geometry_class"] = np.select(
        [
            df["abs_crossline_m"] <= 500.0,
            df["abs_crossline_m"] <= 1000.0,
            df["abs_crossline_m"] <= 2000.0,
        ],
        [
            "excellent",
            "good",
            "marginal",
        ],
        default="poor",
    )

    # --------------------------------------------------------------------------
    # 5. Print compact ranked table
    # --------------------------------------------------------------------------

    display_cols = [
        "event_id",
        "origin_time",
        "mag",
        "depth_km",
        "along_m",
        "crossline_m",
        "horizontal_m",
        "geometry_class",
        "full_prep_window",
        "score",
    ]

    display = df[display_cols].copy()

    for col in [
        "depth_km",
        "along_m",
        "crossline_m",
        "horizontal_m",
        "score",
    ]:
        display[col] = display[col].round(1)

    print("\n")
    print("=" * 120)
    print("RANKED SAFOD EVENT CANDIDATES")
    print("=" * 120)

    print(
        display.to_string(
            index=False,
        )
    )

    # --------------------------------------------------------------------------
    # 6. Shortlist
    # --------------------------------------------------------------------------

    shortlist = df[
        (df["full_prep_window"])
        & (df["abs_crossline_m"] <= 1000.0)
        & (df["mag"] >= 0.8)
    ].copy()

    print("\n")
    print("=" * 120)
    print(
        "SHORTLIST: full window + |crossline| <= 1 km + M >= 0.8"
    )
    print("=" * 120)

    if shortlist.empty:
        print("No events satisfy all shortlist criteria.")
    else:
        print(
            shortlist[
                display_cols
            ].to_string(
                index=False,
            )
        )

    # --------------------------------------------------------------------------
    # 7. Save
    # --------------------------------------------------------------------------

    OUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUT_CSV,
        index=False,
    )

    print(
        f"\nSaved full screening table: {OUT_CSV}"
    )


if __name__ == "__main__":
    main()
