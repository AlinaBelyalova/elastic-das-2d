"""A catalog correction must not publish old geometry labels on PNGs."""
import json
import importlib
import sys
from types import ModuleType

import pandas as pd
import pytest

from src.safod_deep.plot_provenance import require_current_plot_labels
from web_safod.builder.catalog import event_record_from_row


def event():
    return pd.Series(dict(event_id='75379261', origin_time_utc='2026-06-18T21:04:42Z',
        magnitude=1.61, depth_km=3.43, min_3d_distance_to_cable_km=1.5702,
        source_crossline_m=50., geometry_2d_class='good'))


def test_label_roundoff_and_missing_values():
    row = event(); record = row.to_dict()
    record['min_3d_distance_to_cable_km'] += 1e-13
    require_current_plot_labels(record, row)
    row['magnitude'] = float('nan'); record['magnitude'] = None
    require_current_plot_labels(record, row)


@pytest.mark.parametrize('key,value', [('min_3d_distance_to_cable_km', 1.58985),
    ('source_crossline_m', 51.), ('geometry_2d_class', 'borderline'), ('depth_km', 3.5)])
def test_site_refuses_stale_png_labels(key, value):
    row = event(); metadata = row.to_dict(); metadata[key] = value
    with pytest.raises(ValueError, match='Stale plot labels'):
        event_record_from_row(row, metadata_rows=[metadata], plots=[], event_page_template='')


def test_reuse_guard_runs_before_waveform_io(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'DASutils', ModuleType('DASutils'))
    gallery = importlib.import_module('scripts.safod_deep.plot_catalog_events')
    row = event(); recipe = gallery.DisplayRecipe('test', 1., 20., -.5, 2., 1., 96.)
    image = tmp_path / gallery.event_plot_filename(row, recipe); image.touch()
    (tmp_path / 'metadata').mkdir()
    metadata = row.to_dict(); metadata['min_3d_distance_to_cable_km'] = 1.58985
    (tmp_path / 'metadata' / f'{image.stem}.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='Stale plot labels'):
        gallery.process_event(event_index=0, event_row=row, manifest=None,
            output_dir=tmp_path, recipe=recipe, reference=None, registry=None,
            zero_phase=False, overwrite=False, dpi=72, phase_guide_full=None)


def test_gallery_refuses_stale_metadata(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'DASutils', ModuleType('DASutils'))
    gallery = importlib.import_module('scripts.safod_deep.plot_catalog_events')
    row = event(); metadata = row.to_dict()
    metadata.update(image_path='event.png', min_3d_distance_to_cable_km=1.58985)
    (tmp_path / 'event.png').touch(); (tmp_path / 'metadata').mkdir()
    (tmp_path / 'metadata' / 'event.json').write_text(json.dumps(metadata))
    catalog = tmp_path / 'catalog.csv'; pd.DataFrame([row]).to_csv(catalog, index=False)
    with pytest.raises(ValueError, match='Stale plot labels'):
        gallery.build_html_gallery(output_dir=tmp_path, catalog_path=catalog)


def test_gallery_indexes_base_and_arrivals_with_same_metadata(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'DASutils', ModuleType('DASutils'))
    gallery = importlib.import_module('scripts.safod_deep.plot_catalog_events')
    row = event(); metadata = row.to_dict()
    metadata.update(image_path='event.png', arrivals_image_path='event_arrivals.png')
    (tmp_path / 'event.png').touch(); (tmp_path / 'event_arrivals.png').touch()
    (tmp_path / 'metadata').mkdir()
    (tmp_path / 'metadata' / 'event.json').write_text(json.dumps(metadata))
    catalog = tmp_path / 'catalog.csv'; pd.DataFrame([row]).to_csv(catalog, index=False)
    gallery.build_html_gallery(output_dir=tmp_path, catalog_path=catalog)
    inventory = pd.read_csv(tmp_path / 'gallery_manifest.csv')
    assert len(inventory) == 2
    assert inventory.has_json_metadata.all()
    assert set(inventory.event_id.astype(str)) == {'75379261'}
    assert (inventory.min_3d_distance_to_cable_km == 1.5702).all()
    event_record_from_row(row, metadata_rows=[metadata], plots=[], event_page_template='')
