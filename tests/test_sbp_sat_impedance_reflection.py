"""An independent continuum check of the actual boundary law's limitations."""

import numpy as np
import pytest

from scripts.validation.sbp_sat.impedance_reflection import reflection, run_checks


@pytest.mark.parametrize("wave", ["P", "SV"])
@pytest.mark.parametrize("angle", [0, 15, 30, 45, 60, 75])
def test_passive_boundary_flux_and_normal_exactness(wave, angle):
    row = reflection(wave, angle)
    assert row["flux_balance_relative"] < 1e-12
    assert row["boundary_residual_relative"] < 1e-12
    assert 0 <= row["reflected_energy_fraction"] <= 1
    if angle == 0:
        assert row["reflected_energy_fraction"] == 0
    else:
        assert row["reflected_energy_fraction"] > 0


def test_density_scaling_does_not_change_reflected_fraction():
    for kind in ("P", "SV"):
        a = reflection(kind, 60)
        b = reflection(kind, 60, rho=5000)
        assert a["reflected_energy_fraction"] == pytest.approx(
            b["reflected_energy_fraction"], rel=2e-13
        )
    assert run_checks()["passed"]


@pytest.mark.parametrize(
    "parameters",
    [
        {"vp": -5000},
        {"vp": 0},
        {"vs": 0},
        {"rho": -1},
        {"vp": np.nan},
        {"vp": 4000, "vs": 3000},
    ],
)
def test_invalid_material_is_rejected(parameters):
    with pytest.raises(ValueError, match="material"):
        reflection("P", 60, **parameters)
