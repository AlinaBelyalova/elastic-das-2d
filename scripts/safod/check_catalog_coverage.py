from __future__ import annotations

from pathlib import Path
import os
import sys
import glob
import importlib.util
import datetime
import dateutil.parser

import numpy as np
import pandas as pd


# ==============================================================================
# DAS-utilities path
# ==============================================================================

pyDAS_path = "/home/groups/ettore88/alina/packages/DAS-utilities/build"
pyDAS_python_path = "/home/groups/ettore88/alina/packages/DAS-utilities/python"

try:
    os.environ["LD_LIBRARY_PATH"] += ":" + pyDAS_path
except KeyError:
    os.environ["LD_LIBRARY_PATH"] = pyDAS_path

sys.path.insert(0, pyDAS_path)
sys.path.insert(0, pyDAS_python_path)


# ==============================================================================
# SETTINGS
# ==============================================================================

CANDIDATE_CSV = "results/catalog_event_search_2d/safod_catalog_events_since_april_2026_ranked.csv"

DAS_DB_PY = "/home/groups/ettore88/alina/packages/DAS-utilities/python/DAS_db.py"
DAS_SYSTEM_NAME = "SAFOD_QuantX"

# Broader roots because data may not only be in SAFOD_events.
DAS_GLOBS = [
    "/oak/stanford/groups/ettore88/data/SAFOD/SAFOD_events/*.h5",
    "/oak/stanford/groups/ettore88/data/SAFOD/ActiveJune2026/**/*.h5",
    "/oak/stanford/groups/ettore88/data/SAFOD/**/*.h5",
]

OUT_DIR = Path("results/catalog_event_search_2d")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_BEFORE_S = 2.0
WINDOW_AFTER_S = 15.0

# Keep geometrically useful candidates.
MAX_ABS_CROSSLINE_M = 3000.0
MAX_ABS_ALONG_M = 5000.0
MIN_MAG = 0.5


# ==============================================================================
# HELPERS
# ==============================================================================

def read_das_db_headers(files: list[str]) -> pd.DataFrame:
    spec = importlib.util.spec_from_file_location("das_db_external", DAS_DB_PY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import DAS_db.py from {DAS_DB_PY}")

    das_db = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(das_db)

    rows = []
    for i, fn in enumerate(files):
        if i % 100 == 0:
            print(f"Reading headers {i}/{len(files)}")
        df_one = das_db.get_header_df(fn, DAS_SYSTEM_NAME)
        rows.append(df_one)

    df = pd.concat(rows, ignore_index=True)
    return df


def parse_time(x) -> datetime.datetime:
    dt = dateutil.parser.parse(str(x))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt


def files_cover_window(headers: pd.DataFrame, origin_time: datetime.datetime):
    need_start = origin_time - datetime.timedelta(seconds=WINDOW_BEFORE_S)
    need_end = origin_time + datetime.timedelta(seconds=WINDOW_AFTER_S)

    overlap = headers[
        (headers["end_dt"] >= need_start)
        & (headers["start_dt"] <= need_end)
    ].copy()

    if overlap.empty:
        return False, [], need_start, need_end

    covered_start = overlap["start_dt"].min()
    covered_end = overlap["end_dt"].max()

    ok = (covered_start <= need_start) and (covered_end >= need_end)

    return bool(ok), overlap["file"].tolist(), need_start, need_end


def main():
    candidates = pd.read_csv(CANDIDATE_CSV)

    # Filter to plausible events first.
    candidates = candidates[
        (candidates["magnitude"] >= MIN_MAG)
        & (candidates["abs_crossline_m"] <= MAX_ABS_CROSSLINE_M)
        & (np.abs(candidates["event_along_m"]) <= MAX_ABS_ALONG_M)
    ].copy()

    candidates = candidates.sort_values(
        ["abs_crossline_m", "magnitude"],
        ascending=[True, False],
    ).reset_index(drop=True)

    print("\nCandidate events after geometry filter")
    print("--------------------------------------")
    print(f"count: {len(candidates)}")
    print(
        candidates[
            [
                "origin_time",
                "event_id",
                "magnitude",
                "depth_km",
                "event_along_m",
                "event_crossline_m",
                "abs_crossline_m",
            ]
        ].head(30).to_string(index=False)
    )

    # Find H5 files.
    files = []
    for pat in DAS_GLOBS:
        found = glob.glob(pat, recursive=True)
        print(f"glob {pat}: {len(found)} files")
        files.extend(found)

    files = sorted(set(files))
    print(f"\nTotal unique H5 files: {len(files)}")

    if len(files) == 0:
        raise RuntimeError("No H5 files found.")

    headers_cache = OUT_DIR / "all_candidate_das_headers.csv"

    if headers_cache.exists():
        print(f"\nReading cached headers: {headers_cache}")
        headers = pd.read_csv(headers_cache)
    else:
        print("\nReading DAS headers. This may take a while...")
        headers = read_das_db_headers(files)
        headers.to_csv(headers_cache, index=False)
        print(f"Saved headers: {headers_cache}")

    if headers.empty:
        raise RuntimeError("Header dataframe is empty.")

    headers["start_dt"] = headers["startTime"].apply(parse_time)
    headers["end_dt"] = headers["endTime"].apply(parse_time)

    for col in ["nSamples", "fs", "Desample", "nChannels", "dCh", "GaugeLen"]:
        headers[col] = pd.to_numeric(headers[col], errors="coerce")

    rows = []

    for _, row in candidates.iterrows():
        origin = parse_time(row["origin_time"])

        ok, overlap_files, need_start, need_end = files_cover_window(headers, origin)

        out = row.to_dict()
        out["has_das_coverage"] = ok
        out["needed_start"] = need_start.isoformat()
        out["needed_end"] = need_end.isoformat()
        out["n_overlap_files"] = len(overlap_files)
        out["overlap_files"] = ";".join(overlap_files)

        if overlap_files:
            hsel = headers[headers["file"].isin(overlap_files)]
            out["median_dCh_m"] = float(np.nanmedian(hsel["dCh"]))
            out["median_GaugeLen_m"] = float(np.nanmedian(hsel["GaugeLen"]))
            out["median_raw_fs"] = float(np.nanmedian(hsel["fs"]))
            out["median_desample"] = float(np.nanmedian(hsel["Desample"]))
        else:
            out["median_dCh_m"] = np.nan
            out["median_GaugeLen_m"] = np.nan
            out["median_raw_fs"] = np.nan
            out["median_desample"] = np.nan

        rows.append(out)

    out_df = pd.DataFrame(rows)

    out_df = out_df.sort_values(
        ["has_das_coverage", "abs_crossline_m", "magnitude"],
        ascending=[False, True, False],
    ).reset_index(drop=True)

    out_csv = OUT_DIR / "catalog_candidates_with_das_coverage.csv"
    out_df.to_csv(out_csv, index=False)

    covered = out_df[out_df["has_das_coverage"]].copy()
    covered_csv = OUT_DIR / "catalog_candidates_with_das_coverage_only.csv"
    covered.to_csv(covered_csv, index=False)

    print("\nSaved:")
    print(f"  {out_csv}")
    print(f"  {covered_csv}")

    cols = [
        "origin_time",
        "event_id",
        "magnitude",
        "depth_km",
        "event_along_m",
        "event_crossline_m",
        "abs_crossline_m",
        "min_3d_distance_to_cable_m",
        "has_das_coverage",
        "n_overlap_files",
        "median_dCh_m",
        "median_GaugeLen_m",
    ]

    print("\nCandidates WITH DAS coverage:")
    if covered.empty:
        print("None found with current file roots.")
    else:
        print(covered[cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()