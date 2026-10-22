"""Coordinate, unit, coverage and model-identity checks for arrival predictions."""
from types import SimpleNamespace

import numpy as np
import pytest
from pyproj import Geod

from src.safod import travel_times as tt


def geographic(x, y, *, azimuth=35.0):
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    a, b = np.deg2rad([azimuth, azimuth + 90])
    east, north = x * np.sin(a) + y * np.sin(b), x * np.cos(a) + y * np.cos(b)
    lon, lat, _ = Geod(ellps="WGS84").fwd(
        np.full(x.shape, -120.55), np.full(x.shape, 35.97),
        np.rad2deg(np.arctan2(east, north)), np.hypot(east, north),
    )
    return lat, lon


@pytest.fixture
def native(tmp_path):
    x = np.array([-240., -3., 0., 4., 240.])
    y = np.array([-240., -3., 0., 4., 240.])
    z = np.array([-150., -0.5, 0., 2., 10., 340.])
    xx, yy = np.meshgrid(x * 1000, y * 1000)
    lat, lon = geographic(xx, yy)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    vp = 5.5 + 0.01 * xx + 0.02 * yy + 0.03 * zz
    path = tmp_path / "native.npz"
    np.savez(path, x_km=x, y_km=y, z_km=z, lat_deg=lat, lon_deg=lon,
             vp_km_s=vp, vs_km_s=vp / 1.75)
    return tt.Zhang3D(path)


def test_rotated_geographic_coordinates_and_model_axis_order(native):
    expected_x = np.array([-1100., 700., 2100.])
    expected_y = np.array([900., -1300., 400.])
    lat, lon = geographic(expected_x, expected_y)
    x, y = native.coordinates.xy_m(lat, lon)
    np.testing.assert_allclose(x, expected_x, atol=1e-7)
    np.testing.assert_allclose(y, expected_y, atol=1e-7)
    q = np.column_stack((x, y, [300., 1100., 1900.]))
    expected_vp = 5500 + 0.01 * q[:, 0] + 0.02 * q[:, 1] + 0.03 * q[:, 2]
    np.testing.assert_allclose(native.interpolators[0](q), expected_vp, atol=1e-9)
    assert [a[[0, -1]].tolist() for a in native.axes] == [[-3000., 4000.], [-3000., 4000.], [-500., 10000.]]


def test_msl_depth_and_explicit_top_extension(native, monkeypatch):
    observed = []
    monkeypatch.setattr(tt, "_backend", lambda *a: (None, None, {"test": "backend"}))
    def inspect(velocity, axes, source, receivers, **kwargs):
        observed.append((velocity.copy(), axes, source.copy(), receivers.copy()))
        # Test double used only to inspect geometry and interpolation, not to
        # claim validation of FMM accuracy (the validation script does that).
        return np.array([0.1, 0.2]) * (1 if len(observed) == 1 else 1.75), {}
    monkeypatch.setattr(tt, "solve_eikonal", inspect)
    lat, lon = geographic(np.array([0., 200.]), np.array([0., -200.]))
    p, s, meta = native.predict(source_lat=35.97, source_lon=-120.55, source_depth_km=1.0,
        receiver_lat=lat, receiver_lon=lon, receiver_tvd_m=[0., 1000.],
        spacing_m=100., padding_m=500., cache_dir=None)
    velocity, axes, source, receivers = observed[0]
    np.testing.assert_allclose(source, [0., 0., 1000.], atol=1e-7)
    np.testing.assert_allclose(receivers[:, 2], [-660.46, 339.54])
    assert axes[2][0] == -660.46
    # Below z=-500 m in depth coordinates: explicit constant top-node value.
    np.testing.assert_allclose(velocity[:, :, 0], velocity[:, :, 1])
    assert meta["vertical_datum"].startswith("depth below MSL")
    assert np.all(s > p)


def test_no_lateral_or_bottom_extrapolation(native):
    lat, lon = geographic(np.array([0.]), np.array([0.]))
    options = dict(source_lat=35.97, source_lon=-120.55,
                   receiver_lat=lat, receiver_lon=lon, receiver_tvd_m=[1000.], cache_dir=None)
    with pytest.raises(tt.CoverageError, match="outside physical"):
        native.predict(source_depth_km=11., **options)
    far_lat, far_lon = geographic(np.array([5000.]), np.array([0.]))
    options.update(source_lat=float(far_lat[0]), source_lon=float(far_lon[0]))
    with pytest.raises(tt.CoverageError, match="outside physical"):
        native.predict(source_depth_km=1., **options)


def test_backend_axis_permutation_and_boundary_rejection(monkeypatch):
    class Vector:
        def __init__(self, array): self.array = array; self.shape = array.shape
        def getNdArray(self): return self.array
    called = []
    class Operator:
        def __init__(self, vv, result, *args, **kwargs):
            called.append((vv.getNdArray(), args, kwargs)); self.result = result
        def forward(self, add, model, result): result.array[:] = 1.0
    monkeypatch.setattr(tt, "_backend", lambda *a: (
        SimpleNamespace(EikonalTT_3D=Operator), SimpleNamespace(vectorIC=Vector), {}))
    axes = (np.arange(4.) * 10, np.arange(5.) * 20, np.arange(6.) * 30)
    cube = np.arange(120.).reshape(4, 5, 6) + 1000
    tt.solve_eikonal(cube, axes, [10., 20., 30.], [[15., 25., 35.]])
    values, args, kwargs = called[0]
    assert values.shape == (1, 5, 4, 6)
    np.testing.assert_array_equal(values[0, 3, 2], cube[2, 3])
    assert args[:6] == (0., 0., 0., 20., 10., 30.)
    np.testing.assert_allclose(args[6], [[10., 20., 30.]])
    assert kwargs["Acc_Inj"] is True
    with pytest.raises(tt.CoverageError):
        tt.solve_eikonal(cube, axes, [10., 20., 30.], [[30., 25., 35.]])


def test_physical_reference_channels_are_not_acquisition_rows(tmp_path):
    geometry = tmp_path / "geometry.csv"
    geometry.write_text("Channel,ReferenceChannel,Lat_WGS84,Lon_WGS84,TVD_m\n"
                        "0,210,35,-120,0\n1,220,36,-121,1000\n")
    lat, lon, tvd = tt.receivers_from_channels(geometry, [215.])
    np.testing.assert_allclose([lat[0], lon[0], tvd[0]], [35.5, -120.5, 500.])
    with pytest.raises(tt.CoverageError):
        tt.receivers_from_channels(geometry, [0.])
