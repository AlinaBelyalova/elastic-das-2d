from __future__ import annotations

import argparse
from pathlib import Path

from web_safod.builder.config import load_config, resolve_path
from web_safod.builder.geometry import download_san_andreas_fault


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download San Andreas Fault geometry from the USGS "
            "Quaternary Fault and Fold Database ArcGIS service."
        )
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output = (
        args.output
        if args.output is not None
        else resolve_path(config, "fault_geojson")
    )

    geojson = download_san_andreas_fault(
        arcgis_layer=str(config["fault_arcgis_layer"]),
        bbox=list(config["fault_bbox"]),
        output_path=output,
    )
    print(f"features: {len(geojson['features'])}")
    print(f"saved   : {output}")


if __name__ == "__main__":
    main()
