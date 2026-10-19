from __future__ import annotations

import argparse
from pathlib import Path

from web_safod.builder.config import load_config, resolve_path
from web_safod.builder.geometry import export_safod_geojson


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    source = resolve_path(config, "geometry_xlsx")
    output = (
        args.output
        if args.output is not None
        else resolve_path(config, "site_dir")
        / "data/safod_geometry.geojson"
    )
    geojson = export_safod_geojson(source, output)
    print(f"features: {len(geojson['features'])}")
    print(f"saved   : {output}")


if __name__ == "__main__":
    main()
