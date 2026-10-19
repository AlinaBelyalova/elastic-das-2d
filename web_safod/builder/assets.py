from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


WEB_PRODUCT_VERSION = 2


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text.strip("_.") or "plot"


def _resolve_source_image(
    metadata: dict[str, Any],
    *,
    repo_root: Path,
    event_qc_dir: Path,
    field: str = "image_path",
) -> Path | None:
    raw = str(metadata.get(field, "") or "").strip()
    candidates: list[Path] = []

    if raw:
        p = Path(raw).expanduser()
        candidates.append(p if p.is_absolute() else repo_root / p)
        candidates.append(event_qc_dir / p.name)

    metadata_path = metadata.get("_metadata_path")
    if metadata_path and raw:
        mp = Path(str(metadata_path))
        candidates.append(mp.parent.parent / Path(raw).name)

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    return None


def _source_signature(path: Path) -> str:
    stat = path.stat()
    payload = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _plot_descriptor(metadata: dict[str, Any]) -> dict[str, Any]:
    recipe = metadata.get("recipe")
    if not isinstance(recipe, dict):
        recipe = {}

    fmin = recipe.get("fmin_hz")
    fmax = metadata.get("effective_fmax_hz", recipe.get("fmax_hz"))
    recipe_name = recipe.get("name") or "DAS event"

    label = str(recipe_name)
    try:
        if fmin is not None and fmax is not None:
            label += f" · {float(fmin):g}–{float(fmax):g} Hz"
    except (TypeError, ValueError):
        pass

    return {
        "label": label,
        "recipe_name": recipe_name,
        "fmin_hz": fmin,
        "fmax_hz": fmax,
        "zero_phase": metadata.get("zero_phase"),
    }


def _render_web_image(
    *,
    source: Path,
    destination: Path,
    cache_key: str,
    cache: dict[str, Any],
    max_long_edge_px: int,
    quality: int,
    overwrite: bool,
) -> None:
    signature = _source_signature(source)
    cached = cache.get(cache_key, {})
    current = (
        destination.exists()
        and cached.get("source_signature") == signature
        and cached.get("web_product_version") == WEB_PRODUCT_VERSION
        and int(cached.get("max_long_edge_px", -1)) == int(max_long_edge_px)
        and int(cached.get("quality", -1)) == int(quality)
    )

    if overwrite or not current:
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            width, height = image.size
            scale = min(
                1.0,
                float(max_long_edge_px) / float(max(width, height)),
            )
            if scale < 1.0:
                image = image.resize(
                    (
                        max(1, round(width * scale)),
                        max(1, round(height * scale)),
                    ),
                    Image.Resampling.LANCZOS,
                )
            tmp = destination.with_suffix(".tmp.webp")
            image.save(
                tmp,
                "WEBP",
                quality=int(quality),
                method=6,
            )
            os.replace(tmp, destination)

        cache[cache_key] = {
            "source_signature": signature,
            "source_name": source.name,
            "destination": str(destination),
            "web_product_version": WEB_PRODUCT_VERSION,
            "max_long_edge_px": int(max_long_edge_px),
            "quality": int(quality),
        }


def ensure_web_plots(
    event_id: str,
    metadata_rows: list[dict[str, Any]],
    *,
    repo_root: Path,
    event_qc_dir: Path,
    output_dir: Path,
    cache: dict[str, Any],
    max_long_edge_px: int,
    quality: int,
    overwrite: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Build one clean base WebP per DAS plot plus an optional P/S-arrivals
    companion WebP.  The companion is referenced by arrivals_url and is not
    exposed as a second DAS recipe in the selector.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plots: list[dict[str, Any]] = []
    warnings: list[str] = []

    for metadata in metadata_rows:
        source = _resolve_source_image(
            metadata,
            repo_root=repo_root,
            event_qc_dir=event_qc_dir,
            field="image_path",
        )
        if source is None:
            warnings.append(
                f"{event_id}: source image not found for "
                f"{metadata.get('_metadata_path', '<metadata>')}"
            )
            continue

        token = hashlib.sha256(
            (
                event_id
                + "|"
                + source.name
                + "|"
                + str(metadata.get("_metadata_path", ""))
            ).encode("utf-8")
        ).hexdigest()[:10]

        if len(metadata_rows) == 1:
            base_name = f"{_slug(event_id)}.webp"
            arrivals_name = f"{_slug(event_id)}__arrivals.webp"
        else:
            base_name = f"{_slug(event_id)}__{token}.webp"
            arrivals_name = f"{_slug(event_id)}__{token}__arrivals.webp"

        destination = output_dir / base_name
        _render_web_image(
            source=source,
            destination=destination,
            cache_key=f"{event_id}:{token}:base",
            cache=cache,
            max_long_edge_px=max_long_edge_px,
            quality=quality,
            overwrite=overwrite,
        )

        descriptor = _plot_descriptor(metadata)
        descriptor.update(
            {
                "url": f"plots/{destination.name}",
                "source_name": source.name,
                "arrivals_url": None,
                "arrivals_source_name": None,
            }
        )

        arrivals_source = _resolve_source_image(
            metadata,
            repo_root=repo_root,
            event_qc_dir=event_qc_dir,
            field="arrivals_image_path",
        )
        if arrivals_source is not None:
            arrivals_destination = output_dir / arrivals_name
            _render_web_image(
                source=arrivals_source,
                destination=arrivals_destination,
                cache_key=f"{event_id}:{token}:arrivals",
                cache=cache,
                max_long_edge_px=max_long_edge_px,
                quality=quality,
                overwrite=overwrite,
            )
            descriptor["arrivals_url"] = (
                f"plots/{arrivals_destination.name}"
            )
            descriptor["arrivals_source_name"] = arrivals_source.name
        elif metadata.get("arrivals_available"):
            warnings.append(
                f"{event_id}: arrivals image was expected but not found for "
                f"{metadata.get('_metadata_path', '<metadata>')}"
            )

        plots.append(descriptor)

    return plots, warnings


def load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def save_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    os.replace(tmp, path)
