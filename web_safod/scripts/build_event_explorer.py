from __future__ import annotations

import argparse
from pathlib import Path

from web_safod.builder.config import load_config
from web_safod.builder.site_builder import build_site


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the static SAFOD Deep-DAS event explorer."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--overwrite-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    report = build_site(
        config,
        overwrite_images=args.overwrite_images,
    )

    print("\nSAFOD event explorer build")
    print("--------------------------")
    print(f"events           : {report['events']}")
    print(f"plots available  : {report['plots_available']}")
    print(f"warnings         : {len(report['warnings'])}")
    print(f"site             : {report['site_dir']}")
    if report["warnings"]:
        print("\nFirst warnings:")
        for warning in report["warnings"][:20]:
            print("  -", warning)


if __name__ == "__main__":
    main()
