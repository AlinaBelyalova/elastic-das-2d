from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    # repo/web_safod/builder/config.py -> repo
    return Path(__file__).resolve().parents[2]


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    root = repo_root()
    path = (
        Path(config_path)
        if config_path is not None
        else root / "web_safod/config/explorer.json"
    )
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise FileNotFoundError(path)

    config = json.loads(path.read_text(encoding="utf-8"))

    local_path = path.parent / "paths.local.json"
    if local_path.exists():
        config = _merge(
            config,
            json.loads(local_path.read_text(encoding="utf-8")),
        )

    config["_repo_root"] = str(root)
    config["_config_path"] = str(path)
    return config


def resolve_path(config: dict[str, Any], key: str) -> Path:
    if key not in config:
        raise KeyError(f"Missing config key: {key}")
    path = Path(str(config[key])).expanduser()
    if path.is_absolute():
        return path
    return Path(config["_repo_root"]) / path
