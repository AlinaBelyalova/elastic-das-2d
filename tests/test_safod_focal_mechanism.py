"""Sign, orientation and saved mechanism checks for the ComCat source."""

import json

import numpy as np
import pytest

from src.safod.focal_mechanism import (
    COMCAT_MECHANISM,
    double_couple_ned,
    load_projected_comcat_source,
    validate_saved_source_mechanism,
)


def test_first_motion_conventions():
    # Vertical right-hand-rule N-S plane, positive strike-slip.
    np.testing.assert_allclose(
        double_couple_ned(0, 90, 0),
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]], atol=1e-14,
    )
    # Positive rake is up-dip: a reverse source has Mdd > 0 here.
    np.testing.assert_allclose(
        double_couple_ned(0, 45, 90),
        np.diag([0, -1, 1]), atol=1e-14,
    )


def test_projected_comcat_mechanism(tmp_path):
    path = tmp_path / "mechanisms.json"
    path.write_text(json.dumps({
        "nc75379261": {"code": "nc75379261_fm1", "np1": {"strike": 55, "dip": 55, "rake": 30}}
    }))
    kwargs = dict(path=path, event_id="75379261", profile_u_e=1, profile_u_n=0, scalar_moment=2)
    tensor, meta = load_projected_comcat_source(**kwargs)
    # Independent NED tensor reference: N-E-D → E-D picks rows/columns 1,2.
    m_ned = double_couple_ned(55, 55, 30)
    np.testing.assert_allclose(meta["source_comcat_unit_moment_ned"], m_ned)
    np.testing.assert_allclose(tensor.as_matrix(), 2 * m_ned[1:, 1:])
    assert abs(tensor.trace()) > 0.1  # Must not silently remove the trace.
    reversed_tensor, _ = load_projected_comcat_source(**dict(kwargs, profile_u_e=-1))
    assert (reversed_tensor.Mxx, reversed_tensor.Mzz, reversed_tensor.Mxz) == (
        tensor.Mxx, tensor.Mzz, -tensor.Mxz,
    )
    with pytest.raises(ValueError, match="No ComCat mechanism"):
        load_projected_comcat_source(**dict(kwargs, event_id="NC123"))
    with pytest.raises(ValueError, match="Rerun scripts.safod.prepare_event"):
        load_projected_comcat_source(**dict(kwargs, profile_u_n=np.nan))
    with pytest.raises(FileNotFoundError, match="cache missing"):
        load_projected_comcat_source(**dict(kwargs, path=tmp_path / "missing.json"))


def test_saved_manual_and_comcat_sources_are_distinct():
    manual = {"source_theta_deg": np.array(35)}
    comcat = {
        "source_mechanism": np.array(COMCAT_MECHANISM),
        "source_comcat_product_code": np.array("nc75379261_fm1"),
    }
    assert not validate_saved_source_mechanism(manual, 35)
    assert validate_saved_source_mechanism(comcat, None) == "nc75379261_fm1"
    with pytest.raises(ValueError, match="differs from saved"):
        validate_saved_source_mechanism(manual, None)
    with pytest.raises(ValueError, match="differs from saved"):
        validate_saved_source_mechanism(comcat, 35)
