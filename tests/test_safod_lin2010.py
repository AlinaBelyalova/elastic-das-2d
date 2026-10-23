"""Original-field interpolation, datum preservation, and coverage-only fallback."""
import numpy as np
import pytest

from src.safod import travel_times as tt
from src.safod.lin2010 import Lin3D, LinCoordinates, LinGrid


def write_field(path, *, coarse=False, phase="vp"):
    axis = np.linspace(-40., 40., 3 if coarse else 5)
    z = np.array([-1., 1., 4., 8., 14., 20., 45.])
    zz, yy, xx = np.meshgrid(z, axis, axis, indexing="ij")
    lat, lon = LinCoordinates().geographic(xx*1000, yy*1000)
    # Separate functions, not a constant Vp/Vs construction.
    speed = (6.0 + 0.001*xx + 0.002*yy + 0.01*zz if phase == "vp"
             else 3.3 + 0.0005*xx + 0.001*yy + 0.005*zz)
    rows = np.column_stack([a.ravel() for a in (lon, lat, zz, xx, yy, speed, np.ones_like(xx))])
    # Node order is immaterial; write in a reproducible shuffled order.
    np.random.default_rng(3).shuffle(rows)
    np.savetxt(path, rows, fmt="%.12f")
    return rows


@pytest.fixture
def fields(tmp_path):
    vp, vs = tmp_path / "out.vp", tmp_path / "out.vs"
    write_field(vp)
    write_field(vs, coarse=True, phase="vs")
    return vp, vs


def test_original_independent_fields_and_inverse_cell_mapping(fields):
    model = Lin3D(*fields)
    for phase, grid in enumerate(model.interpolators):
        j, i, u, v, depth = 1, 1, 0.31, 0.72, 10.42
        weights = np.array([[(1-v)*(1-u), (1-v)*u], [v*(1-u), v*u]])
        lat = np.sum(weights*grid.lat[j:j+2, i:i+2])
        lon = np.sum(weights*grid.lon[j:j+2, i:i+2])
        x = grid.x[i] + u*(grid.x[i+1]-grid.x[i])
        y = grid.y[j] + v*(grid.y[j+1]-grid.y[j])
        nx, ny = grid.native_xy_km([lat, lat], [lon, lon])
        np.testing.assert_allclose(nx, x, atol=1e-8)
        np.testing.assert_allclose(ny, y, atol=1e-8)
        expected = (6 + 0.001*x + 0.002*y + 0.01*depth if phase == 0
                    else 3.3 + 0.0005*x + 0.001*y + 0.005*depth)
        np.testing.assert_allclose(grid.sample(lat, lon, depth), expected, atol=1e-10)
    x, y = np.array([-17000., 23000.]), np.array([13000., -19000.])
    lat, lon = model.coordinates.geographic(x, y)
    xx, yy = model.coordinates.xy_m(lat, lon)
    np.testing.assert_allclose(xx, x, atol=1e-7)
    np.testing.assert_allclose(yy, y, atol=1e-7)


@pytest.mark.parametrize("change", ["duplicate", "missing", "invalid"])
def test_bad_native_grids_fail(fields, change):
    path = fields[0]
    raw = np.loadtxt(path)
    if change == "duplicate": raw[-1] = raw[0]
    elif change == "missing": raw = raw[:-1]
    else: raw[0, 5] = 0.
    np.savetxt(path, raw)
    with pytest.raises(ValueError):
        LinGrid(path)


def test_lin_no_geographic_or_depth_extrapolation(fields):
    grid = LinGrid(fields[0])
    with pytest.raises(tt.CoverageError, match="geographic cells"):
        grid.sample(20., -120., 10.)
    with pytest.raises(tt.CoverageError, match="depth range"):
        grid.sample(35.97, -120.55, 46.)
    # Exact geographic cell edges are valid despite floating-point inversion.
    x, y = grid.native_xy_km(grid.lat.ravel(), grid.lon.ravel())
    xx, yy = np.meshgrid(grid.x, grid.y)
    np.testing.assert_allclose(x, xx.ravel(), atol=1e-8)
    np.testing.assert_allclose(y, yy.ravel(), atol=1e-8)


def test_lin_si_units_and_negative_catalogue_depth_are_preserved(fields, monkeypatch):
    model = Lin3D(*fields)
    monkeypatch.setattr(tt, "_backend", lambda *a: (None, None, {}))
    captured = []
    def solve(velocity, axes, source, receivers, **kwargs):
        captured.append((velocity.copy(), source.copy(), receivers.copy()))
        return np.array([0.2]) * (1 if len(captured) == 1 else 1.8), {}
    monkeypatch.setattr(tt, "solve_eikonal", solve)
    lat, lon = model.coordinates.geographic([100.], [100.])
    p, s, meta = model.predict(source_lat=model.coordinates.origin_lat,
        source_lon=model.coordinates.origin_lon, source_depth_km=-0.67,
        receiver_lat=lat, receiver_lon=lon, receiver_tvd_m=[1000.],
        spacing_m=200., padding_m=400., cache_dir=None)
    vp, source, receivers = captured[0]
    assert source[2] == -670.
    assert receivers[0, 2] == pytest.approx(339.54)
    assert 5500 < vp.min() < vp.max() < 7000
    assert 3000 < captured[1][0].min() < captured[1][0].max() < 4000
    assert meta["source_above_local_flat_ground"] is True
    assert meta["model_name"] == "lin2010_original_vp_vs_3d"
    assert len(meta["lin_original_fields"]) == 2
    assert np.all(s > p)


def test_selector_preserves_primary_then_uses_lin_only_for_coverage(monkeypatch):
    import src.safod.lin2010 as lin
    calls = []
    class Primary:
        def __init__(self, path): pass
        def predict(self, **kwargs):
            if kwargs["source_depth_km"] > 10:
                raise tt.CoverageError("source below Zhang")
            calls.append("zhang")
            return np.array([1.]), np.array([2.]), {"model_name": "zhang"}
    class Regional:
        def __init__(self, *paths): calls.append("lin_load")
        def predict(self, **kwargs):
            calls.append(("lin", kwargs["spacing_m"]))
            return np.array([3.]), np.array([5.]), {"model_name": "lin"}
    monkeypatch.setattr(tt, "Zhang3D", Primary)
    monkeypatch.setattr(lin, "Lin3D", Regional)
    model = tt.Regional3D(lin_spacing_m=200.)
    _, _, first = model.predict(source_depth_km=3., spacing_m=100.)
    assert calls == ["zhang"]
    assert first["fallback_used"] is False
    _, _, second = model.predict(source_depth_km=10.42, spacing_m=100.)
    assert calls[-2:] == ["lin_load", ("lin", 200.)]
    assert second["fallback_used"] is True
    assert second["fallback_reason"] == "source below Zhang"
    model.predict(source_depth_km=18., spacing_m=100.)
    assert calls.count("lin_load") == 1


@pytest.mark.parametrize("error", [RuntimeError("solver"), ValueError("node limit"), FileNotFoundError("input")])
def test_selector_does_not_hide_noncoverage_errors(monkeypatch, error):
    class Primary:
        def __init__(self, path): pass
        def predict(self, **kwargs): raise error
    monkeypatch.setattr(tt, "Zhang3D", Primary)
    model = tt.Regional3D()
    with pytest.raises(type(error), match=str(error)):
        model.predict()
    assert model._lin is None


def test_both_model_hashes_identify_the_result(fields):
    first = Lin3D(*fields)
    raw = np.loadtxt(fields[1]); raw[:, 5] *= 0.99
    np.savetxt(fields[1], raw)
    second = Lin3D(*fields)
    assert first.sha256 != second.sha256


def test_explicit_lin_does_not_require_a_zhang_file(fields, monkeypatch):
    model = tt.Regional3D(native_grid="does_not_exist.npz", lin_vp=fields[0], lin_vs=fields[1], mode="lin")
    assert model.primary is None
    assert model._lin is None


def test_gallery_lin_label_and_actual_spacing(monkeypatch):
    import importlib
    import sys
    from types import ModuleType, SimpleNamespace
    import pandas as pd
    monkeypatch.setitem(sys.modules, "DASutils", ModuleType("DASutils"))
    gallery = importlib.import_module("scripts.safod_deep.plot_catalog_events")
    class Predictor:
        def predict(self, **kwargs):
            return np.array([1., 1.1]), np.array([2., 2.1]), dict(
                source_xyz_m=[100., 200., 10420.], method="3-D eikonal FMM",
                model_name="lin2010_original_vp_vs_3d", model_label="Lin 2010 3-D (original Vp/Vs)",
                spacing_m=200., fallback_used=True, fallback_reason="below Zhang")
    context = SimpleNamespace(zhang_3d=Predictor(), failures={})
    reference = dict(borehole=pd.DataFrame(dict(fibre_distance_m=[0., 100.],
        Lat_WGS84=[35.97, 35.97], Lon_WGS84=[-120.55, -120.55], TVD_m=[0., 100.])))
    args = SimpleNamespace(no_travel_time_guides=False, phase_guide_mode="3d",
        phase_3d_spacing_m=100., phase_3d_padding_m=3000., phase_max_nodes=12000000,
        eiko_root="unused", phase_cache_dir="unused")
    event = pd.Series(dict(event_id="qc", latitude=35.97, longitude=-120.55, depth_km=10.42))
    guide = gallery.choose_phase_travel_time_guide(event_row=event, reference=reference,
        context=context, fibre_distance_m=np.array([0., 100.]), args=args)
    assert guide.model_label.startswith("Lin 2010")
    assert guide.grid_dx_m == guide.grid_dz_m == 200.
    assert guide.details["fallback_used"] is True
