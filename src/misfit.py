# ==============================================================================
# src/misfit.py — DAS data-misfit functions
#
# This module contains data-space objective functions only.
# It does not time-reverse residuals and does not inject adjoint sources into
# the elastic solver. Its returned gradient is dJ / d(synthetic_data).
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DASMisfitResult:
    """
    Result of a DAS data-space misfit evaluation.
    """

    objective: float
    residual: np.ndarray
    data_gradient: np.ndarray
    quadrature_weight: float

    def __post_init__(self) -> None:
        residual = np.asarray(
            self.residual,
            dtype=np.float64,
        )
        data_gradient = np.asarray(
            self.data_gradient,
            dtype=np.float64,
        )

        if residual.ndim != 2:
            raise ValueError(
                f"residual must be 2D; got shape {residual.shape}."
            )

        if data_gradient.shape != residual.shape:
            raise ValueError(
                "data_gradient must match residual shape: "
                f"{data_gradient.shape} != {residual.shape}."
            )

        if not np.isfinite(self.objective):
            raise ValueError(
                f"objective must be finite; got {self.objective}."
            )

        if (
            not np.isfinite(self.quadrature_weight)
            or self.quadrature_weight <= 0.0
        ):
            raise ValueError(
                "quadrature_weight must be finite and positive; "
                f"got {self.quadrature_weight}."
            )

        residual = np.array(
            residual,
            copy=True,
        )
        data_gradient = np.array(
            data_gradient,
            copy=True,
        )

        residual.flags.writeable = False
        data_gradient.flags.writeable = False

        object.__setattr__(
            self,
            "residual",
            residual,
        )
        object.__setattr__(
            self,
            "data_gradient",
            data_gradient,
        )
        object.__setattr__(
            self,
            "objective",
            float(self.objective),
        )
        object.__setattr__(
            self,
            "quadrature_weight",
            float(self.quadrature_weight),
        )


def l2_das_misfit(
    synthetic: np.ndarray,
    observed: np.ndarray,
    *,
    dt_s: float,
    channel_spacing_m: float,
    weights: np.ndarray | float | None = None,
) -> DASMisfitResult:
    """
    Weighted least-squares DAS objective.

    J = 0.5 * dt * dCh * sum(w * (synthetic - observed)^2)
    """
    synthetic = np.asarray(
        synthetic,
        dtype=np.float64,
    )
    observed = np.asarray(
        observed,
        dtype=np.float64,
    )

    if synthetic.shape != observed.shape:
        raise ValueError(
            "synthetic and observed must have identical shapes; "
            f"got {synthetic.shape} and {observed.shape}."
        )

    if synthetic.ndim != 2:
        raise ValueError(
            "synthetic and observed must be 2D arrays "
            f"(nchannel, nt); got {synthetic.shape}."
        )

    if not np.all(np.isfinite(synthetic)):
        raise ValueError(
            "synthetic contains NaN or Inf."
        )

    if not np.all(np.isfinite(observed)):
        raise ValueError(
            "observed contains NaN or Inf."
        )

    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError(
            f"dt_s must be finite and positive; got {dt_s}."
        )

    if (
        not np.isfinite(channel_spacing_m)
        or channel_spacing_m <= 0.0
    ):
        raise ValueError(
            "channel_spacing_m must be finite and positive; "
            f"got {channel_spacing_m}."
        )

    if weights is None:
        weight_array = 1.0
    else:
        weight_array = np.asarray(
            weights,
            dtype=np.float64,
        )

        try:
            np.broadcast_shapes(
                synthetic.shape,
                weight_array.shape,
            )
        except ValueError as exc:
            raise ValueError(
                "weights must be broadcastable to the data shape; "
                f"got weights={weight_array.shape}, "
                f"data={synthetic.shape}."
            ) from exc

        if not np.all(np.isfinite(weight_array)):
            raise ValueError(
                "weights contains NaN or Inf."
            )

        if np.any(weight_array < 0.0):
            raise ValueError(
                "weights must be non-negative."
            )

    residual = synthetic - observed

    quadrature_weight = float(
        dt_s
        * channel_spacing_m
    )

    weighted_residual = (
        weight_array
        * residual
    )

    objective = 0.5 * quadrature_weight * float(
        np.sum(
            residual
            * weighted_residual,
            dtype=np.float64,
        )
    )

    data_gradient = (
        quadrature_weight
        * weighted_residual
    )

    return DASMisfitResult(
        objective=objective,
        residual=residual,
        data_gradient=data_gradient,
        quadrature_weight=quadrature_weight,
    )


def _self_test() -> None:
    rng = np.random.default_rng(20260721)

    synthetic = rng.standard_normal(
        (7, 13)
    )
    observed = rng.standard_normal(
        (7, 13)
    )
    weights = rng.random(
        (7, 1)
    )

    result = l2_das_misfit(
        synthetic,
        observed,
        dt_s=0.002,
        channel_spacing_m=2.532124,
        weights=weights,
    )

    direction = rng.standard_normal(
        synthetic.shape
    )

    analytic = float(
        np.vdot(
            result.data_gradient,
            direction,
        )
    )

    epsilon = 1.0e-7

    plus = l2_das_misfit(
        synthetic + epsilon * direction,
        observed,
        dt_s=0.002,
        channel_spacing_m=2.532124,
        weights=weights,
    ).objective

    minus = l2_das_misfit(
        synthetic - epsilon * direction,
        observed,
        dt_s=0.002,
        channel_spacing_m=2.532124,
        weights=weights,
    ).objective

    finite_difference = (
        plus - minus
    ) / (
        2.0 * epsilon
    )

    relative_error = abs(
        analytic - finite_difference
    ) / max(
        abs(analytic),
        abs(finite_difference),
        1.0e-30,
    )

    assert relative_error < 1.0e-7, (
        "L2 misfit directional-derivative test failed: "
        f"analytic={analytic:.16e}, "
        f"finite_difference={finite_difference:.16e}, "
        f"relative_error={relative_error:.3e}"
    )

    print(
        "L2 DAS misfit directional derivative: OK "
        f"(relative error={relative_error:.3e})"
    )
    print("misfit.py: all self-tests passed")


if __name__ == "__main__":
    _self_test()
