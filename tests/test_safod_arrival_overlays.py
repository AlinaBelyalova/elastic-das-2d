"""Real/synthetic guides must retain separate models and time origins."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from scripts.safod import compare_event as comparison


@pytest.mark.parametrize("envelope", [False, True])
@pytest.mark.parametrize("real_model", ["Zhang 2009 3-D eikonal", "Lin 2010 3-D (original Vp/Vs) eikonal"])
def test_distinct_guides_and_no_double_source_time_shift(tmp_path, monkeypatch, envelope, real_model):
    channels = np.array([210., 220., 230.])
    times = np.linspace(-0.2, 1.2, 80)
    data = np.zeros((3, 80))
    p2, s2 = np.full(3, 0.3), np.full(3, 0.6)
    p3, s3 = np.full(3, 0.4), np.full(3, 0.8)
    real = dict(arrival_channels=channels, p_arrivals_s=p3, s_arrivals_s=s3,
                model_label=real_model)
    captured = []
    original_close = plt.close
    monkeypatch.setattr(plt, "close", lambda fig: captured.append(fig))
    options = dict(real_time_s=times, real_channels=channels,
        synthetic_time_display_s=times, synthetic_channels=channels,
        arrival_channels=channels, p_arrivals_s=p2, s_arrivals_s=s2,
        event_id="synthetic_QC", source_f0_hz=5., source_theta_deg=35.,
        synthetic_display_shift_s=-0.25, initial_model="test_2d_model",
        out_path=tmp_path / "comparison.png", real_arrivals=real)
    try:
        if envelope:
            comparison.plot_envelope_side_by_side(real_envelope_normalized=data,
                synthetic_envelope_normalized=data, **options)
        else:
            comparison.plot_signed_side_by_side(real_signed_normalized=data,
                synthetic_signed_normalized=data, **options)
        fig = captured[-1]
        for ax, expected, label in ((fig.axes[0], p3, "3-D"), (fig.axes[1], p2, "2-D")):
            line = next(line for line in ax.lines if line.get_label().startswith("Predicted P"))
            assert label in line.get_label()
            if ax is fig.axes[0]:
                assert real_model in line.get_label()
            np.testing.assert_array_equal(line.get_ydata(), expected)
    finally:
        for fig in captured:
            original_close(fig)
