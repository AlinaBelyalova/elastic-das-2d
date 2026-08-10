from __future__ import annotations

from pathlib import Path

from .bill_logs import build_bill_logs_model
from .boness_zoback2006 import (
    DEFAULT_BONESS_ZOBACK2006_CSV,
    build_zhang2009_boness2006_model,
)
from .hybrid_zhang2009_bill_logs import (
    build_hybrid_zhang2009_bill_logs_model,
)
from .hybrid_zhang2009_boness2006_bill_logs import (
    build_hybrid_zhang2009_boness2006_bill_logs_model,
)
from .smooth_prior import build_smooth_prior_model
from .zhang2009 import (
    DEFAULT_ZHANG_SECTION,
    build_zhang2009_model,
)

SMOOTH_PRIOR = "smooth_prior"
BILL_LOGS = "bill_logs"
ZHANG2009 = "zhang2009"
HYBRID_ZHANG2009_BILL_LOGS = "hybrid_zhang2009_bill_logs"
ZHANG2009_BONESS2006 = "zhang2009_boness2006"
HYBRID_ZHANG2009_BONESS2006_BILL_LOGS = (
    "hybrid_zhang2009_boness2006_bill_logs"
)

AVAILABLE_INITIAL_MODELS = (
    SMOOTH_PRIOR,
    BILL_LOGS,
    ZHANG2009,
    HYBRID_ZHANG2009_BILL_LOGS,
    ZHANG2009_BONESS2006,
    HYBRID_ZHANG2009_BONESS2006_BILL_LOGS,
)

DEFAULT_BILL_LOGS_CSV = Path(
    "data/safod/velocity_models/"
    "ellsworth_malin_2011/"
    "fig3a_digitized.csv"
)

def build_initial_model(
    *,
    model_name: str,
    geom_file: str | Path,
    bill_logs_csv: str | Path | None = None,
    digitized_log_csv: str | Path | None = None,
    zhang_section_npz: str | Path | None = None,
    boness_log_csv: str | Path | None = None,
    **kwargs,
):
    name = str(model_name).strip().lower()

    if bill_logs_csv is None:
        bill_logs_csv = digitized_log_csv
    if bill_logs_csv is None:
        bill_logs_csv = DEFAULT_BILL_LOGS_CSV
    if zhang_section_npz is None:
        zhang_section_npz = DEFAULT_ZHANG_SECTION
    if boness_log_csv is None:
        boness_log_csv = DEFAULT_BONESS_ZOBACK2006_CSV

    if name == SMOOTH_PRIOR:
        return build_smooth_prior_model(
            geom_file=geom_file,
            build_initial_model=True,
            **kwargs,
        )

    if name == BILL_LOGS:
        return build_bill_logs_model(
            geom_file=geom_file,
            log_csv=bill_logs_csv,
            build_initial_model=True,
            **kwargs,
        )

    if name == ZHANG2009:
        return build_zhang2009_model(
            geom_file=geom_file,
            section_npz=zhang_section_npz,
            build_initial_model=True,
            **kwargs,
        )

    if name == HYBRID_ZHANG2009_BILL_LOGS:
        return build_hybrid_zhang2009_bill_logs_model(
            geom_file=geom_file,
            bill_logs_csv=bill_logs_csv,
            section_npz=zhang_section_npz,
            build_initial_model=True,
            **kwargs,
        )

    if name == ZHANG2009_BONESS2006:
        return build_zhang2009_boness2006_model(
            geom_file=geom_file,
            boness_log_csv=boness_log_csv,
            section_npz=zhang_section_npz,
            build_initial_model=True,
            **kwargs,
        )

    if name == HYBRID_ZHANG2009_BONESS2006_BILL_LOGS:
        return build_hybrid_zhang2009_boness2006_bill_logs_model(
            geom_file=geom_file,
            bill_logs_csv=bill_logs_csv,
            boness_log_csv=boness_log_csv,
            section_npz=zhang_section_npz,
            build_initial_model=True,
            **kwargs,
        )

    raise ValueError(
        f"Unknown SAFOD initial model {model_name!r}. "
        f"Available: {AVAILABLE_INITIAL_MODELS}."
    )
