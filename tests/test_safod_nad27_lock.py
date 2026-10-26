"""Real control points and failure cases for the fixed NADCON/HARN operation."""
from __future__ import annotations

import shutil

import numpy as np
import pytest
from pyproj import Transformer

from src.safod import nad27
from src.safod_deep.catalog import CableGeometry3D
from test_safod_catalog_crs import cable_frame


def test_authoritative_wellhead_pair():
    east, north = nad27.make_transformer().transform(-120.5521414, 35.9742039, errcheck=True)
    # Independent supplied PDF: WGS84 coordinate and NAD27 E/N. The remaining
    # 0.226 m is a consistency check, not centimetre survey accuracy.
    assert np.hypot(east - 720807.06, north - 3983663.97) < 0.30


def test_control_event_matches_verified_sherlock_operation():
    geometry = CableGeometry3D(cable_frame(), wellhead_elevation_m=660.5)
    result = geometry._source_coordinates(-120.54166, 35.9835, 3.43)
    assert result['source_east_m'] == pytest.approx(721726.3934939585, abs=1e-6, rel=0.)
    assert result['source_north_m'] == pytest.approx(3984719.019528135, abs=1e-6, rel=0.)
    assert result['source_elevation_m'] == -3430.
    assert geometry.horizontal_transform_provenance['id'] == nad27.TRANSFORM_ID


def test_operation_does_not_call_automatic_selection(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError('Pinned operation must not select from CRS database')
    monkeypatch.setattr(Transformer, 'from_crs', unexpected)
    for lon, lat in [(-120.55,35.97),(-120.9,36.3),(-120.1,35.6)]:
        t = nad27.make_transformer()
        east, north, z = t.transform(lon, lat, 1234., errcheck=True)
        reconstructed = t.transform(east, north, direction='INVERSE', errcheck=True)
        np.testing.assert_allclose(reconstructed, [lon,lat], atol=1e-9, rtol=0.)
        assert z == 1234.


def test_missing_grid_fails_instead_of_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(nad27, 'GRID_DIR', tmp_path)
    with pytest.raises(FileNotFoundError, match='Missing pinned SAFOD datum grid'):
        nad27.make_transformer()


def test_changed_grid_fails_instead_of_fallback(tmp_path, monkeypatch):
    for name in nad27.GRID_SHA256:
        shutil.copy2(nad27.GRID_DIR / name, tmp_path / name)
    (tmp_path / 'us_noaa_conus.tif').write_bytes(b'changed datum grid')
    monkeypatch.setattr(nad27, 'GRID_DIR', tmp_path)
    with pytest.raises(RuntimeError, match='SHA256 mismatch'):
        nad27.make_transformer()


def test_absolute_grid_paths_allow_spaces(tmp_path, monkeypatch):
    directory = tmp_path / 'project with spaces'
    directory.mkdir()
    for name in nad27.GRID_SHA256:
        shutil.copy2(nad27.GRID_DIR / name, directory / name)
    monkeypatch.setattr(nad27, 'GRID_DIR', directory)
    result = nad27.make_transformer().transform(-120.54166,35.9835,errcheck=True)
    np.testing.assert_allclose(result,[721726.3934939585,3984719.019528135],atol=1e-6,rtol=0.)


@pytest.mark.parametrize('lon,lat', [(-122.,36.),(-120.,37.),(float('nan'),36.)])
def test_source_outside_operation_area_is_rejected(lon,lat):
    geometry = CableGeometry3D(cable_frame(), wellhead_elevation_m=660.5)
    with pytest.raises(ValueError, match='outside pinned SAFOD NAD27'):
        geometry._source_coordinates(lon,lat,3.)


@pytest.mark.parametrize('tampered', [False,True])
def test_read_only_catalog_audit_detects_changed_distance(tmp_path,tampered):
    import pandas as pd
    from scripts.validation.safod.catalog_geodesy import audit_catalog
    frame = cable_frame()
    geometry_path = tmp_path / 'geometry.csv'
    frame.to_csv(geometry_path,index=False)
    geometry = CableGeometry3D.from_file(geometry_path,wellhead_elevation_m=660.5,
        profile_origin=(35.9742039,-120.5521414))
    source = dict(longitude=-120.54166,latitude=35.9835,depth_km=3.43)
    record = dict(event_id='qc', **source, **geometry.event_metrics(**source))
    if tampered:
        record['min_3d_distance_to_cable_km'] += .001
    catalog_path = tmp_path / 'catalog.csv'
    pd.DataFrame([record]).to_csv(catalog_path,index=False)
    original = catalog_path.read_bytes()
    report = audit_catalog(catalog_path,geometry_path)
    assert report['passed'] == (not tampered)
    assert report['mismatch_count'] == int(tampered)
    assert catalog_path.read_bytes() == original
