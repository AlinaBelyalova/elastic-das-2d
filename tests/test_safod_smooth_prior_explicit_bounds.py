from __future__ import annotations

import csv

import numpy as np
import pytest

from scripts.safod.settings import (
    SAFOD_SCIENTIFIC_X_MAX_M,
    SAFOD_SCIENTIFIC_X_MIN_M,
)
from src.safod.models.smooth_prior import build_smooth_prior_model


DX_M = 5.0
N_SIDE_SPONGE_CELLS = 120


@pytest.fixture
def geometry_csv(tmp_path):
    path = tmp_path / "geometry.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("X_2D_m", "Z_2D_m"),
        )
        writer.writeheader()
        writer.writerows(
            (
                {"X_2D_m": 0.0, "Z_2D_m": 0.0},
                {"X_2D_m": 10.0, "Z_2D_m": 10.0},
            )
        )
    return path


def _build(geometry_csv, **overrides):
    kwargs = {
        "geom_file": geometry_csv,
        "build_initial_model": True,
        "x_column": "X_2D_m",
        "z_column": "Z_2D_m",
        "dx": DX_M,
        "dz": DX_M,
        "nt": 2,
        "z_max_m": 20.0,
        "z_padding_bottom_m": 0.0,
        "fault_offset_from_cable_m": 0.0,
        "fault_dip_deg": 82.0,
        "fault_dip_sign": -1.0,
        "x_padding_m": 20.0,
        "min_x_width_m": 0.0,
        "initial_cross_fault_contrast": 0.0,
        "initial_fault_zone_velocity_reduction": 0.0,
        "include_pilot_hole_lvz_in_initial": False,
        "smooth_initial_sigma_m": 0.0,
    }
    kwargs.update(overrides)
    return build_smooth_prior_model(**kwargs)


def test_explicit_safod_bounds_have_exact_endpoints_and_dimensions(
    geometry_csv,
):
    sponge_width_m = N_SIDE_SPONGE_CELLS * DX_M
    computational_min_m = SAFOD_SCIENTIFIC_X_MIN_M - sponge_width_m
    computational_max_m = SAFOD_SCIENTIFIC_X_MAX_M + sponge_width_m

    grid, model, *_ = _build(
        geometry_csv,
        x_grid_min_m=computational_min_m,
        x_grid_max_m=computational_max_m,
    )

    assert grid.x[0] == -2100.0
    assert grid.x[-1] == 4100.0
    assert grid.nx == 1241
    assert model.vp.shape == (1241, 5)

    scientific_min_m = grid.x[0] + sponge_width_m
    scientific_max_m = grid.x[-1] - sponge_width_m
    scientific_nx = int(
        round((scientific_max_m - scientific_min_m) / grid.dx)
    ) + 1

    assert scientific_min_m == -1500.0
    assert scientific_max_m == 3500.0
    assert scientific_nx == 1001


@pytest.mark.parametrize(
    "bounds",
    (
        {"x_grid_min_m": -20.0},
        {"x_grid_max_m": 30.0},
    ),
)
def test_partial_explicit_bounds_are_rejected(geometry_csv, bounds):
    with pytest.raises(ValueError, match="must be supplied together"):
        _build(geometry_csv, **bounds)


@pytest.mark.parametrize(
    ("x_min_m", "x_max_m", "message"),
    (
        (30.0, -20.0, "must exceed"),
        (-20.0, -20.0, "must exceed"),
        (np.nan, 30.0, "must be finite"),
        (-20.0, np.nan, "must be finite"),
        (-np.inf, 30.0, "must be finite"),
        (-20.0, np.inf, "must be finite"),
    ),
)
def test_invalid_explicit_bounds_are_rejected(
    geometry_csv,
    x_min_m,
    x_max_m,
    message,
):
    with pytest.raises(ValueError, match=message):
        _build(
            geometry_csv,
            x_grid_min_m=x_min_m,
            x_grid_max_m=x_max_m,
        )


def test_explicit_span_incompatible_with_dx_is_rejected(geometry_csv):
    with pytest.raises(ValueError, match="integer number of dx intervals"):
        _build(
            geometry_csv,
            x_grid_min_m=-20.0,
            x_grid_max_m=31.0,
        )


def test_cable_outside_explicit_grid_is_rejected(geometry_csv):
    with pytest.raises(ValueError, match="Cable geometry lies outside"):
        _build(
            geometry_csv,
            x_grid_min_m=-20.0,
            x_grid_max_m=-5.0,
        )


def test_fault_outside_explicit_grid_is_rejected(geometry_csv):
    with pytest.raises(ValueError, match="Fault geometry lies outside"):
        _build(
            geometry_csv,
            x_grid_min_m=-20.0,
            x_grid_max_m=30.0,
            fault_offset_from_cable_m=100.0,
        )


def test_legacy_automatic_bounds_are_unchanged(geometry_csv):
    grid, model, *_ = _build(geometry_csv)

    assert grid.x[0] == -20.0
    assert grid.x[-1] == 35.0
    assert grid.nx == 12
    assert model.vp.shape == (12, 5)
