from __future__ import annotations

import argparse
from pathlib import Path

from web_safod.builder.config import load_config, resolve_path
from web_safod.builder.validation import validate_site


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--site-dir", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    site_dir = (
        args.site_dir
        if args.site_dir is not None
        else resolve_path(config, "site_dir")
    )

    errors = validate_site(site_dir)
    if errors:
        print("SITE VALIDATION FAILED")
        print("----------------------")
        for error in errors:
            print(" -", error)
        raise SystemExit(2)

    print("SITE VALIDATION PASSED")
    print(site_dir)


if __name__ == "__main__":
    main()
