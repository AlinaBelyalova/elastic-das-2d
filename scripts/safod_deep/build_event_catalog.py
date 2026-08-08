# ==============================================================================
# scripts/safod_deep/build_event_catalog.py
#
# One command to:
#   1. scan all SAFOD-deep HDF5 files without loading waveforms;
#   2. determine exact recording coverage/settings;
#   3. query the official NCEDC event service;
#   4. compute 3-D source-to-cable distances and 2-D section diagnostics;
#   5. write machine-readable catalogs and overview plots.
#
# Example:
#
# python -m scripts.safod_deep.build_event_catalog \
#   --roots-config config/safod_deep/roots.json \
#   --geometry /path/to/SAFOD_Phase2_GeoReferenced_Channels.xlsx \
#   --wellhead-elevation-m 684.0
#
# Do not guess wellhead elevation. Omit the option if the geometry file already
# contains absolute cable elevation; otherwise 3-D distances are marked unknown.
# ==============================================================================

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.safod_deep.catalog import (
    CableGeometry3D,
    H5RecordingScanner,
    NCEDCCatalogClient,
    SAFODCatalogBuilder,
    load_roots_config,
    plot_catalog_overview,
    write_catalog_notes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--roots-config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--geometry",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--geometry-sheet",
        default="0",
        help="Excel sheet index or name. Default: 0.",
    )
    parser.add_argument(
        "--wellhead-elevation-m",
        type=float,
        default=None,
        help=(
            "Required only when geometry has TVD but no absolute elevation. "
            "Must use the same vertical datum as NCEDC event depth."
        ),
    )
    parser.add_argument(
        "--geometry-min-tvd-m",
        type=float,
        default=10.0,
        help=(
            "Exclude shallow reference-spool geometry from the best-fit "
            "vertical cable plane. Set a negative value to disable filtering."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/safod_deep/catalog"),
    )
    parser.add_argument(
        "--wellhead-latitude",
        type=float,
        default=35.97243650644933,
    )
    parser.add_argument(
        "--wellhead-longitude",
        type=float,
        default=-120.5511469244454,
    )
    parser.add_argument(
        "--max-radius-km",
        type=float,
        default=50.0,
    )
    parser.add_argument(
        "--min-magnitude",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--query-start",
        default=None,
        help="UTC ISO time. By default use earliest valid recording start.",
    )
    parser.add_argument(
        "--query-end",
        default=None,
        help="UTC ISO time. By default use latest valid recording end.",
    )
    parser.add_argument(
        "--pre-s",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--post-s",
        type=float,
        default=12.0,
    )
    parser.add_argument(
        "--reuse-manifest",
        action="store_true",
        help="Reuse out-dir/recording_manifest.csv instead of rescanning HDF5.",
    )
    parser.add_argument(
        "--reuse-events",
        action="store_true",
        help="Reuse out-dir/ncedc_events_raw.csv instead of querying NCEDC.",
    )

    return parser.parse_args()


def _parse_sheet(value: str):
    try:
        return int(value)
    except ValueError:
        return value


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    output = frame.copy()
    for column in output.columns:
        if pd.api.types.is_datetime64_any_dtype(output[column]):
            output[column] = output[column].dt.strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            )
    output.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.out_dir / "recording_manifest.csv"
    events_path = args.out_dir / "ncedc_events_raw.csv"

    roots = load_roots_config(args.roots_config)

    if args.reuse_manifest and manifest_path.exists():
        print(f"Loading existing manifest: {manifest_path}")
        manifest = pd.read_csv(manifest_path)
        manifest["start_time_utc"] = pd.to_datetime(
            manifest["start_time_utc"], utc=True, errors="coerce"
        )
        manifest["end_time_utc"] = pd.to_datetime(
            manifest["end_time_utc"], utc=True, errors="coerce"
        )
    else:
        print("Scanning SAFOD-deep HDF5 headers...")
        manifest = H5RecordingScanner().scan_roots(roots)
        _write_csv(manifest, manifest_path)

    valid_manifest = manifest[
        manifest["start_time_utc"].notna()
        & manifest["end_time_utc"].notna()
        & manifest["error"].isna()
    ].copy()

    print("\nRecording inventory")
    print("-------------------")
    print(f"rows/files          : {len(manifest)}")
    print(f"valid timed files   : {len(valid_manifest)}")
    print(f"scan errors         : {manifest['error'].notna().sum()}")

    if valid_manifest.empty and (
        args.query_start is None or args.query_end is None
    ):
        raise RuntimeError(
            "No valid recording intervals were found. Inspect "
            "recording_manifest.csv and provide --query-start/--query-end if "
            "the HDF5 start-time metadata use an unsupported convention."
        )

    query_start = (
        pd.to_datetime(args.query_start, utc=True)
        if args.query_start is not None
        else valid_manifest["start_time_utc"].min()
    )
    query_end = (
        pd.to_datetime(args.query_end, utc=True)
        if args.query_end is not None
        else valid_manifest["end_time_utc"].max()
    )

    if args.reuse_events and events_path.exists():
        print(f"Loading existing NCEDC events: {events_path}")
        events = pd.read_csv(events_path)
        events["origin_time_utc"] = pd.to_datetime(
            events["origin_time_utc"], utc=True, errors="coerce"
        )
    else:
        print(
            "\nQuerying NCEDC catalog "
            f"{query_start.isoformat()} -> {query_end.isoformat()}, "
            f"radius={args.max_radius_km:.1f} km..."
        )
        events = NCEDCCatalogClient().query(
            start=query_start,
            end=query_end,
            latitude=args.wellhead_latitude,
            longitude=args.wellhead_longitude,
            max_radius_km=args.max_radius_km,
            min_magnitude=args.min_magnitude,
        )
        _write_csv(events, events_path)

    print(f"NCEDC events returned: {len(events)}")

    cable = CableGeometry3D.from_file(
        args.geometry,
        wellhead_elevation_m=args.wellhead_elevation_m,
        sheet_name=_parse_sheet(args.geometry_sheet),
        min_tvd_m=(
            None
            if args.geometry_min_tvd_m < 0.0
            else args.geometry_min_tvd_m
        ),
    )

    print(f"Vertical datum: {cable.vertical_datum_status}")

    catalog = SAFODCatalogBuilder(
        cable=cable,
        manifest=manifest,
        pre_s=args.pre_s,
        post_s=args.post_s,
    ).build(events)

    _write_csv(
        catalog,
        args.out_dir / "catalog_all.csv",
    )

    recorded = catalog[
        catalog["origin_covered"]
    ].copy()
    _write_csv(
        recorded,
        args.out_dir / "catalog_recorded.csv",
    )

    direct_2d = recorded[
        recorded["suitable_for_direct_2d_source"]
    ].copy()
    _write_csv(
        direct_2d,
        args.out_dir / "catalog_2d_direct_source_candidates.csv",
    )

    complete_windows = recorded[
        recorded["window_covered"]
    ].copy()
    _write_csv(
        complete_windows,
        args.out_dir / "event_windows_for_plotting.csv",
    )

    plot_catalog_overview(
        catalog,
        cable,
        args.out_dir,
    )

    write_catalog_notes(
        args.out_dir / "CATALOG_NOTES.md",
        catalog=catalog,
        manifest=manifest,
        cable=cable,
        query_start=query_start,
        query_end=query_end,
        radius_km=args.max_radius_km,
        min_magnitude=args.min_magnitude,
    )

    print("\nCatalog summary")
    print("---------------")
    print(f"all catalog events      : {len(catalog)}")
    print(f"origin recorded         : {len(recorded)}")
    print(f"complete plot windows   : {len(complete_windows)}")
    print(f"direct 2-D candidates   : {len(direct_2d)}")

    if not recorded.empty:
        print("\n2-D geometry classes")
        print(recorded["geometry_2d_class"].value_counts(dropna=False).to_string())

    print(f"\nSaved to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
