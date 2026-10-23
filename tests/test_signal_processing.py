"""Check frequency rejection and phase for configured comparison filters."""
import numpy as np
import pytest

from src.common.signal_processing import bandpass_traces


@pytest.mark.parametrize("lower,low_expected", [(0., True), (5., False)])
def test_comparison_filter_frequency_content_and_zero_phase(lower, low_expected):
    fs = 200.
    t = np.arange(4000)/fs
    frequencies = (1., 10., 80.)
    signal = sum(np.sin(2*np.pi*f*t) for f in frequencies)
    filtered = bandpass_traces(signal[None, :], fs_hz=fs, fmin_hz=lower,
                               fmax_hz=20., order=4, taper_frac=0.05)[0]
    # The central ten seconds exclude the taper and edge transients. Project
    # onto independent sinusoidal components, not the filter implementation.
    tt, yy = t[1000:3000], filtered[1000:3000]
    sine = [2*np.mean(yy*np.sin(2*np.pi*f*tt)) for f in frequencies]
    quadrature = [2*np.mean(yy*np.cos(2*np.pi*f*tt)) for f in frequencies]
    assert 0.98 < sine[1] < 1.01
    assert abs(sine[2]) < 0.01
    if low_expected:
        assert 0.98 < sine[0] < 1.01
    else:
        assert abs(sine[0]) < 0.01
    assert max(abs(np.asarray(quadrature))) < 0.001


def test_negative_lower_corner_is_still_invalid():
    with pytest.raises(ValueError, match="0 <= fmin"):
        bandpass_traces(np.zeros((1, 100)), fs_hz=100., fmin_hz=-1., fmax_hz=20.)
