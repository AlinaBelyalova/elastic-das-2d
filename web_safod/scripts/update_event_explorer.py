from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from web_safod.builder.config import load_config, resolve_path
from web_safod.builder.geometry import download_san_andreas_fault
from web_safod.builder.site_builder import build_site
from web_safod.builder.validation import validate_site


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build, validate, and optionally publish the SAFOD explorer."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--overwrite-images", action="store_true")
    parser.add_argument(
        "--refresh-fault",
        action="store_true",
        help="Refresh local San Andreas geometry from the USGS ArcGIS service.",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish to the configured local GitHub Pages checkout.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    fault_path = resolve_path(config, "fault_geojson")
    if args.refresh_fault or not fault_path.exists():
        try:
            fault = download_san_andreas_fault(
                arcgis_layer=str(config["fault_arcgis_layer"]),
                bbox=list(config["fault_bbox"]),
                output_path=fault_path,
            )
            print(
                f"USGS San Andreas geometry: "
                f"{len(fault['features'])} feature(s)"
            )
        except Exception as exc:
            print(
                "WARNING: local fault download failed. "
                "The public site will try the same USGS service in the browser."
            )
            print(f"         {type(exc).__name__}: {exc}")

    report = build_site(
        config,
        overwrite_images=args.overwrite_images,
    )
    site_dir = resolve_path(config, "site_dir")
    errors = validate_site(site_dir)
    if errors:
        for error in errors:
            print("VALIDATION ERROR:", error)
        raise SystemExit(2)

    print("\nUpdate complete")
    print("---------------")
    print(f"events          : {report['events']}")
    print(f"plots available : {report['plots_available']}")
    print(f"site            : {site_dir}")
    print("validation      : PASSED")

    if args.publish:
        script = (
            Path(config["_repo_root"])
            / "web_safod/scripts/publish_github_pages.sh"
        )
        env = os.environ.copy()
        env["PYTHON"] = sys.executable

        subprocess.run(
            [
                "bash",
                str(script),
                str(config["pages_checkout"]),
            ],
            check=True,
            cwd=config["_repo_root"],
            env=env,
        )

if __name__ == "__main__":
    main()
