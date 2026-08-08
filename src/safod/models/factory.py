from __future__ import annotations

from pathlib import Path

from .smooth_prior import build_smooth_prior_model
from .digitized_log import build_digitized_log_model

SMOOTH_PRIOR = "smooth_prior"
DIGITIZED_LOG = "digitized_log"
AVAILABLE_INITIAL_MODELS = (SMOOTH_PRIOR, DIGITIZED_LOG)


def build_initial_model(
    *,
    model_name: str,
    geom_file: str | Path,
    digitized_log_csv: str | Path | None = None,
    **kwargs,
):
    name = str(model_name).strip().lower()

    if name == SMOOTH_PRIOR:
        return build_smooth_prior_model(
            geom_file=geom_file,
            build_initial_model=True,
            **kwargs,
        )

    if name == DIGITIZED_LOG:
        if digitized_log_csv is None:
            raise ValueError(
                "digitized_log_csv is required for model_name='digitized_log'."
            )
        return build_digitized_log_model(
            geom_file=geom_file,
            log_csv=digitized_log_csv,
            build_initial_model=True,
            **kwargs,
        )

    raise ValueError(
        f"Unknown SAFOD initial model {model_name!r}. "
        f"Available: {AVAILABLE_INITIAL_MODELS}."
    )
