"""A failed 3-D solve cannot silently become an old prediction or fallback."""
import importlib
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def gallery(monkeypatch):
    # Only the external recording reader is stubbed. These checks stop before
    # waveform I/O; the real eikonal backend has its own numerical QC.
    monkeypatch.setitem(sys.modules, "DASutils", ModuleType("DASutils"))
    return importlib.import_module("scripts.safod_deep.plot_catalog_events")


def test_failed_3d_prediction_has_explicit_reason_and_no_fallback(gallery, monkeypatch):
    class Unsupported:
        def predict(self, **kwargs): raise ValueError("outside physical tomography")
    def unexpected(**kwargs): raise AssertionError("Fallback must not be called")
    monkeypatch.setattr(gallery, "predict_straight_ray_phase_travel_times", unexpected)
    context = SimpleNamespace(zhang_3d=Unsupported(), failures={})
    reference = {"borehole": pd.DataFrame(dict(fibre_distance_m=[0., 100.],
                 Lat_WGS84=[35., 35.], Lon_WGS84=[-120., -120.], TVD_m=[0., 100.]))}
    args = SimpleNamespace(no_travel_time_guides=False, phase_guide_mode="3d",
             phase_3d_spacing_m=100., phase_3d_padding_m=3000., phase_max_nodes=12000000,
             eiko_root="unused", phase_cache_dir="unused")
    event = pd.Series(dict(event_id="qc", latitude=35., longitude=-120., depth_km=3.))
    result = gallery.choose_phase_travel_time_guide(event_row=event, reference=reference,
                context=context, fibre_distance_m=np.array([0., 100.]), args=args)
    assert result is None
    assert "outside physical tomography" in context.failures["qc"]


def test_existing_plot_cannot_reuse_different_prediction(gallery, tmp_path):
    event = pd.Series(dict(event_id="qc", origin_time_utc="2026-10-21T00:00:00Z", magnitude=1.))
    recipe = gallery.DisplayRecipe("test", 1., 20., -0.5, 2., 1., 96.)
    image = tmp_path / gallery.event_plot_filename(event, recipe)
    companion = tmp_path / gallery.event_arrivals_plot_filename(event, recipe)
    image.touch(); companion.touch()
    (tmp_path / "metadata").mkdir()
    metadata = dict(travel_time_guide=dict(method="old 2-D", model_name="old_model"))
    (tmp_path / "metadata" / f"{image.stem}.json").write_text(json.dumps(metadata))
    guide = gallery.PhaseTravelTimeGuide(np.array([0., 100.]), np.array([.1,.2]),
        np.array([.2,.3]), "3-D eikonal FMM", "zhang2009_native_3d", "Zhang 3D",
        0., 1000., details={"cache_key": "new_prediction"})
    with pytest.raises(RuntimeError, match="different P/S provenance"):
        gallery.process_event(event_index=0, event_row=event, manifest=None,
            output_dir=tmp_path, recipe=recipe, reference=None, registry=None,
            zero_phase=False, overwrite=False, dpi=72, phase_guide_full=guide)
