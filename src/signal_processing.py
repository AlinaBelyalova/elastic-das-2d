"""Shared signal-processing utilities for SAFOD comparison and future FWI."""

from __future__ import annotations

import numpy as np

try:
    from scipy.signal import butter, detrend, sosfiltfilt, welch, windows
except ImportError as exc:
    raise ImportError("src.signal_processing requires scipy.signal.") from exc


def _as_trace_matrix(data: np.ndarray, *, caller: str) -> np.ndarray:
    array = np.asarray(data, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(
            f"{caller} expects a 2D (trace, time) array; got {array.shape}."
        )
    if array.shape[1] < 2:
        raise ValueError(f"{caller} requires at least two time samples.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{caller} input contains NaN or Inf.")
    return array


def bandpass_traces(
    data: np.ndarray,
    *,
    fs_hz: float,
    fmin_hz: float,
    fmax_hz: float,
    order: int = 4,
    taper_frac: float = 0.05,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass along the time axis."""
    array = _as_trace_matrix(data, caller="bandpass_traces")

    if not np.isfinite(fs_hz) or fs_hz <= 0.0:
        raise ValueError(f"Invalid sampling rate: {fs_hz}")
    if order < 1:
        raise ValueError(f"Filter order must be >= 1; got {order}.")
    if not 0.0 <= taper_frac < 0.5:
        raise ValueError(
            f"taper_frac must be in [0, 0.5); got {taper_frac}."
        )

    nyquist_hz = 0.5 * fs_hz
    if not 0.0 < fmin_hz < fmax_hz < nyquist_hz:
        raise ValueError(
            "Require 0 < fmin < fmax < Nyquist; "
            f"received {fmin_hz}, {fmax_hz}, {nyquist_hz} Hz."
        )

    filtered_input = detrend(array, axis=1, type="linear")
    n_time = filtered_input.shape[1]
    n_taper = int(round(taper_frac * n_time))

    if n_taper >= 1:
        taper = np.ones(n_time, dtype=np.float64)
        ramp = windows.hann(2 * n_taper, sym=True)
        taper[:n_taper] = ramp[:n_taper]
        taper[-n_taper:] = ramp[n_taper:]
        filtered_input = filtered_input * taper[None, :]

    sos = butter(
        order,
        [fmin_hz, fmax_hz],
        btype="bandpass",
        fs=fs_hz,
        output="sos",
    )
    return sosfiltfilt(sos, filtered_input, axis=1)


def median_welch_psd(
    data: np.ndarray,
    time_s: np.ndarray,
    *,
    fs_hz: float,
    tmin_s: float,
    tmax_s: float,
    trace_mask: np.ndarray | None = None,
    segment_length_s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return median and 10th/90th-percentile Welch PSD over traces."""
    array = _as_trace_matrix(data, caller="median_welch_psd")
    time_s = np.asarray(time_s, dtype=np.float64)

    if time_s.ndim != 1 or time_s.size != array.shape[1]:
        raise ValueError(
            "time_s must be 1D and match the trace time dimension."
        )
    if not tmin_s < tmax_s:
        raise ValueError("PSD window requires tmin_s < tmax_s.")

    time_mask = (time_s >= tmin_s) & (time_s <= tmax_s)
    if np.count_nonzero(time_mask) < 8:
        raise ValueError(
            f"PSD window {tmin_s}..{tmax_s} s has too few samples."
        )

    if trace_mask is None:
        trace_mask = np.ones(array.shape[0], dtype=bool)
    else:
        trace_mask = np.asarray(trace_mask, dtype=bool)
        if trace_mask.shape != (array.shape[0],):
            raise ValueError("trace_mask must match the trace dimension.")

    if np.count_nonzero(trace_mask) < 1:
        raise ValueError("PSD selection contains no traces.")

    selected = array[trace_mask][:, time_mask]
    requested_nperseg = max(8, int(round(segment_length_s * fs_hz)))
    nperseg = min(requested_nperseg, selected.shape[1])

    frequency_hz, psd = welch(
        selected,
        fs=fs_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="linear",
        scaling="density",
        axis=1,
    )

    return (
        frequency_hz,
        np.nanmedian(psd, axis=0),
        np.nanpercentile(psd, 10.0, axis=0),
        np.nanpercentile(psd, 90.0, axis=0),
    )
