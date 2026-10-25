"""Catalog distances must compare source and cable in the same UTM datum."""
import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer

from src.safod_deep.catalog import CableGeometry3D


def cable_frame(epsg=26710):
    """A known trajectory, anchored at the authoritative SAFOD NAD27 wellhead."""
    fraction = np.linspace(0., 1., 21)
    east = 720807.06 + 600. * fraction
    north = 3983663.97 + 800. * fraction
    lon, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform(east, north)
    return pd.DataFrame(dict(
        Channel=210 + np.arange(21), UTM_E_m=east, UTM_N_m=north,
        MD_m=3000. * fraction, TVD_m=2500. * fraction,
        Lat_WGS84=lat, Lon_WGS84=lon,
    ))


@pytest.mark.parametrize("receiver", [0, 10, 20])
def test_source_on_cable_has_zero_distance(receiver):
    frame = cable_frame()
    geometry = CableGeometry3D(frame, wellhead_elevation_m=660.46)
    row = frame.iloc[receiver]
    metrics = geometry.event_metrics(
        latitude=row.Lat_WGS84, longitude=row.Lon_WGS84,
        depth_km=(row.TVD_m - 660.46) / 1000.,
    )
    # Millimetre tolerance allows the PROJ geographic-datum round trip.
    assert metrics["min_3d_distance_to_cable_km"] * 1000. < 0.005
    assert metrics["closest_cable_channel"] == pytest.approx(row.Channel, abs=1e-4)
    if receiver == 0:
        assert metrics["wellhead_epicentral_distance_km"] * 1000. < 0.005
        assert metrics["wellhead_hypocentral_distance_km"] * 1000. < 0.005


def test_file_loader_uses_the_cable_datum(tmp_path):
    frame = cable_frame()
    path = tmp_path / "geometry.csv"
    frame.to_csv(path, index=False)
    geometry = CableGeometry3D.from_file(
        path, wellhead_elevation_m=660.46, min_tvd_m=None,
    )
    source = geometry._source_coordinates(
        longitude=frame.Lon_WGS84.iloc[0], latitude=frame.Lat_WGS84.iloc[0],
        depth_km=-0.66046,
    )
    np.testing.assert_allclose(
        [source["source_east_m"], source["source_north_m"]],
        [720807.06, 3983663.97], atol=0.005, rtol=0.,
    )


def test_explicit_wgs84_utm_geometry_remains_supported():
    frame = cable_frame(32610)
    geometry = CableGeometry3D(frame, wellhead_elevation_m=660.46, utm_epsg=32610)
    row = frame.iloc[10]
    metrics = geometry.event_metrics(
        latitude=row.Lat_WGS84, longitude=row.Lon_WGS84,
        depth_km=(row.TVD_m - 660.46) / 1000.,
    )
    assert metrics["min_3d_distance_to_cable_km"] * 1000. < 0.005


def test_shallow_cut_preserves_wellhead_and_full_profile(tmp_path):
    frame = cable_frame()
    # Curved shallow section makes percentile/direction changes detectable.
    frame.loc[1:3, 'UTM_E_m'] += 50.
    lon, lat = Transformer.from_crs(26710, 4326, always_xy=True).transform(
        frame.UTM_E_m.to_numpy(), frame.UTM_N_m.to_numpy())
    frame['Lon_WGS84'], frame['Lat_WGS84'] = lon, lat
    path = tmp_path / 'geometry.csv'
    frame.to_csv(path, index=False)
    full = CableGeometry3D.from_file(path, wellhead_elevation_m=660.5, min_tvd_m=None)
    cut = CableGeometry3D.from_file(path, wellhead_elevation_m=660.5, min_tvd_m=200.)
    assert cut.wellhead_elevation == 660.5
    assert cut.wellhead_east == full.wellhead_east
    np.testing.assert_array_equal(cut.along_unit, full.along_unit)
    np.testing.assert_array_equal(cut.channel_valid, frame.loc[frame.TVD_m >= 200., 'Channel'])
    metrics = cut.event_metrics(longitude=lon[0], latitude=lat[0], depth_km=-.6605)
    assert metrics['wellhead_hypocentral_distance_km'] * 1000. < .005
    # Distance is intentionally to the retained borehole, not the removed spool.
    assert metrics['min_3d_distance_to_cable_km'] > .2


def test_profile_origin_is_explicit_and_independent_of_distance_cut(tmp_path):
    frame = cable_frame()
    origin = (35.9742039, -120.5521414)
    path = tmp_path / 'geometry.csv'; frame.to_csv(path, index=False)
    cut = CableGeometry3D.from_file(path, wellhead_elevation_m=660.5,
                                  profile_origin=origin)
    source = cut._source_coordinates(longitude=origin[1], latitude=origin[0], depth_km=0.)
    assert source['source_section_x_m'] == 0.
    assert source['source_crossline_m'] == 0.
    # Reproduce prepare_event's profile formula independently on the full down-leg.
    east = np.deg2rad(frame.Lon_WGS84.to_numpy() - origin[1]) * 6371000. * np.cos(np.deg2rad(origin[0]))
    north = np.deg2rad(frame.Lat_WGS84.to_numpy() - origin[0]) * 6371000.
    deep = frame.TVD_m.to_numpy() > np.percentile(frame.TVD_m, 90)
    if deep.sum() < 5:
        deep = frame.TVD_m.to_numpy() > np.percentile(frame.TVD_m, 80)
    direction = np.array([np.median(east[deep]), np.median(north[deep])])
    np.testing.assert_allclose(cut.along_unit, direction / np.linalg.norm(direction), atol=1e-15)


def test_invalid_shallow_cut_is_explicit(tmp_path):
    path = tmp_path / 'geometry.csv'; cable_frame().to_csv(path, index=False)
    with pytest.raises(ValueError, match='fewer than two cable points'):
        CableGeometry3D.from_file(path, wellhead_elevation_m=660.5, min_tvd_m=9999.)


def test_query_defaults_use_canonical_wellhead(monkeypatch):
    import sys
    from scripts.safod_deep.build_event_catalog import parse_args
    from src.safod.reference import SAFOD_WELLHEAD_LAT_WGS84, SAFOD_WELLHEAD_LON_WGS84
    monkeypatch.setattr(sys, 'argv', ['build_event_catalog', '--roots-config', 'roots.json',
                                     '--geometry', 'geometry.xlsx'])
    args = parse_args()
    assert args.wellhead_latitude == SAFOD_WELLHEAD_LAT_WGS84
    assert args.wellhead_longitude == SAFOD_WELLHEAD_LON_WGS84
