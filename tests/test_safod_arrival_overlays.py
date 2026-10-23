"""Real and synthetic panels share one origin-relative 3-D reference."""
import json
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from scripts.safod import compare_event as comparison


@pytest.mark.parametrize("envelope", [False, True])
@pytest.mark.parametrize("real_model", ["Zhang 2009 3-D eikonal", "Lin 2010 3-D (original Vp/Vs) eikonal"])
def test_shared_guides_and_no_double_source_time_shift(tmp_path, monkeypatch, envelope, real_model):
    channels = np.array([210., 220., 230.])
    times = np.linspace(-0.3, 2.0, 80)
    synthetic_times = np.linspace(-0.12, 1.8, 80)
    synthetic_channels = np.array([215., 220., 225.])
    data = np.zeros((3, 80))
    p3, s3 = np.full(3, 0.4), np.full(3, 0.8)
    real = dict(arrival_channels=channels, p_arrivals_s=p3, s_arrivals_s=s3,
                model_label=real_model)
    captured = []
    original_close = plt.close
    monkeypatch.setattr(plt, "close", lambda fig: captured.append(fig))
    options = dict(real_time_s=times, real_channels=channels,
        synthetic_time_display_s=synthetic_times, synthetic_channels=synthetic_channels,
        event_id="synthetic_QC", source_f0_hz=5., source_theta_deg=35.,
        synthetic_display_shift_s=-0.25, initial_model="test_2d_model",
        out_path=tmp_path / "comparison.png", shared_arrivals=real)
    try:
        if envelope:
            comparison.plot_envelope_side_by_side(real_envelope_normalized=data,
                synthetic_envelope_normalized=data, **options)
        else:
            comparison.plot_signed_side_by_side(real_signed_normalized=data,
                synthetic_signed_normalized=data, **options)
        fig = captured[-1]
        for ax in fig.axes[:2]:
            # Gauges remove end receivers; the old shared guide expanded the
            # synthetic axes and left blank strips. Both images must cover
            # their entire common viewport, on unchanged physical coordinates.
            np.testing.assert_array_equal(ax.get_xlim(), [215., 225.])
            np.testing.assert_array_equal(ax.get_ylim(), [1.8, 0.])
            assert not ax.get_autoscalex_on()
            assert not ax.get_autoscaley_on()
            left, right, bottom, top = ax.images[0].get_extent()
            assert left <= 215. < 225. <= right
            assert top <= 0. < 1.8 <= bottom
            line = next(line for line in ax.lines if line.get_label().startswith("Predicted P"))
            assert line.get_label() == "Predicted P"
            assert any(item.get_label() == "Predicted S" for item in ax.lines)
            np.testing.assert_array_equal(line.get_xdata(), channels)
            np.testing.assert_array_equal(line.get_ydata(), p3)
            s_line = next(item for item in ax.lines if item.get_label() == "Predicted S")
            np.testing.assert_array_equal(s_line.get_xdata(), channels)
            np.testing.assert_array_equal(s_line.get_ydata(), s3)
        # The viewport is cropped; data are not stretched to new coordinates.
        np.testing.assert_array_equal(fig.axes[0].images[0].get_extent(), [210., 230., 2., -0.3])
        np.testing.assert_array_equal(fig.axes[1].images[0].get_extent(), [215., 225., 1.8, -0.12])
    finally:
        for fig in captured:
            original_close(fig)


@pytest.mark.parametrize("model_name, model_label, fallback", [
    ("zhang2009_native_3d", "Zhang 2009 3-D", False),
    ("lin2010_original_vp_vs_3d", "Lin 2010 3-D (original Vp/Vs)", True),
])
def test_predict_once_for_both_panels(tmp_path, monkeypatch, model_name, model_label, fallback):
    """The shared guide needs no synthetic grid, velocities or receiver projection."""
    channels = np.array([230., 210., 220.])
    p = np.array([0.5, 0.3, 0.4])
    s = np.array([0.9, 0.7, 0.8])
    lat, lon = np.array([35.9, 35.91, 35.92]), np.array([-120.5, -120.51, -120.52])
    tvd = np.array([100., 200., 300.])
    calls = []

    def predict(**options):
        calls.append(options)
        return p, s, dict(model_name=model_name, model_label=model_label,
                         method="3-D eikonal FMM (EikoHypoTomo/pykonal)",
                         fallback_used=fallback,
                         fallback_reason="outside Zhang coverage" if fallback else None)

    coordinates = SimpleNamespace(xy_m=lambda lat, lon: (1000 * lat, 1000 * lon))
    model = SimpleNamespace(predict=predict, coordinates=coordinates,
                            primary=SimpleNamespace(coordinates=coordinates))

    def receivers(path, requested_channels):
        np.testing.assert_array_equal(requested_channels, channels)
        return lat, lon, tvd

    monkeypatch.setattr(comparison, "regional_model_from_args", lambda args: model)
    monkeypatch.setattr(comparison, "receivers_from_channels", receivers)
    monkeypatch.setattr("sys.argv", ["compare_event"])
    args = comparison.parse_args()
    assert args.phase_3d_model == "zhang-lin"
    np.savez(tmp_path / "real.npz", raw_channels=channels,
             ev_lat=35.95, ev_lon=-120.55, ev_depth_km=3.43)
    # There is deliberately no synthetic NPZ: reference calculation must be
    # independent of a projected 2-D forward grid and its source-time function.
    with np.load(tmp_path / "real.npz") as real:
        guides, provenance = comparison.predict_comparison_arrivals(real, args, tmp_path)

    assert len(calls) == 1
    assert calls[0]["source_lat"] == 35.95
    assert calls[0]["source_lon"] == -120.55
    assert calls[0]["source_depth_km"] == 3.43
    assert calls[0]["surface_elevation_m"] == 660.46
    np.testing.assert_array_equal(calls[0]["receiver_lat"], lat)
    np.testing.assert_array_equal(calls[0]["receiver_lon"], lon)
    np.testing.assert_array_equal(calls[0]["receiver_tvd_m"], tvd)
    np.testing.assert_array_equal(guides["arrival_channels"], [210., 220., 230.])
    np.testing.assert_array_equal(guides["p_arrivals_s"], [0.3, 0.4, 0.5])
    np.testing.assert_array_equal(guides["s_arrivals_s"], [0.7, 0.8, 0.9])
    saved = json.loads((tmp_path / "predicted_arrivals_metadata.json").read_text())
    assert saved == provenance
    assert saved["shared_reference"] is True
    assert saved["overlay_panels"] == ["real", "synthetic"]
    assert saved["real"] == saved["synthetic"]
    assert saved["synthetic"]["model_name"] == model_name
    assert saved["synthetic"]["fallback_used"] is fallback
    assert saved["synthetic"]["time_reference"] == "catalog origin; no STF shift"
    assert saved["synthetic_waveform_dimension"] == 2
    assert saved["synthetic_waveform_model"] == args.initial_model
    table = pd.read_csv(tmp_path / saved["reference_csv"])
    np.testing.assert_array_equal(table["channel"], channels)
    np.testing.assert_allclose(table["p_time_s"], p)
    np.testing.assert_allclose(table["s_time_s"], s)
    np.testing.assert_allclose(table["depth_msl_m"], tvd - 660.46)
    assert not (tmp_path / "predicted_arrivals_synthetic_2d.csv").exists()


@pytest.mark.parametrize("real_channels, synthetic_channels, real_time, synthetic_time, message", [
    ([210., 220.], [230., 240.], [-0.3, 2.], [-0.12, 2.], "channel ranges"),
    ([210., 220.], [210., 220.], [-0.3, 2.], [0.1, 2.], "catalog origin"),
    ([210., 220.], [210., 220.], [-0.3, -0.1], [-0.12, 2.], "post-origin"),
])
def test_no_fabricated_coverage(real_channels, synthetic_channels, real_time, synthetic_time, message):
    with pytest.raises(ValueError, match=message):
        comparison.comparison_view_limits(
            real_channels=real_channels, synthetic_channels=synthetic_channels,
            real_time_s=real_time, synthetic_time_display_s=synthetic_time,
        )


def test_real_only_retains_recorded_pre_origin_window(tmp_path, monkeypatch):
    channels, times = np.array([210., 220., 230.]), np.linspace(-0.3, 2., 80)
    # Long arrival times must not expand the display into unrecorded time.
    guides = dict(arrival_channels=channels, p_arrivals_s=np.full(3, 3.),
                  s_arrivals_s=np.full(3, 5.), model_label="Zhang 2009 3-D eikonal")
    captured = []
    original_close = plt.close
    monkeypatch.setattr(plt, "close", lambda fig: captured.append(fig))
    try:
        comparison.plot_real_with_arrivals(
            real_signed_normalized=np.zeros((3, 80)), real_time_s=times,
            real_channels=channels, event_id="test", out_path=tmp_path / "real.png",
            shared_arrivals=guides,
        )
        ax = captured[-1].axes[0]
        np.testing.assert_array_equal(ax.get_xlim(), [210., 230.])
        np.testing.assert_array_equal(ax.get_ylim(), [2., -0.3])
    finally:
        for fig in captured:
            original_close(fig)
