# ==============================================================================
# scripts/safod/compare_event.py
#
# Compare one prepared real SAFOD DAS event with one synthetic forward run.
#
# This version deliberately contains NO observed-ridge picker and NO ridge
# residuals. The current observed-ridge procedure was not sufficiently robust
# for interpretation, so the comparison is restricted to:
#
#   1. common zero-phase filtering of real and synthetic DAS;
#   2. signed trace-normalized gathers;
#   3. Hilbert-envelope gathers;
#   4. model-based approximate P/S arrival overlays;
#   5. pre-filter frequency-content QC.
#
# Synthetic channels are read from the exact das_raw_channels array saved by
# scripts.safod.run_forward. No nearest-neighbour remapping is used.
# ==============================================================================

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.signal import hilbert
except ImportError as exc:
    raise ImportError(
        "This script requires scipy.signal. "
        "Run it in the fwi/geo environment where SciPy is installed."
    ) from exc

from scripts.safod.settings import (
    COMMON_FMAX_HZ,
    COMMON_FMIN_HZ,
    DEFAULT_THETA_DEG,
    FILTER_ORDER,
    FILTER_TAPER_FRAC,
    REAL_EVENT_PACKAGE,
    comparison_dir_for_theta,
    forward_package_for_theta,
    forward_run_tag,
)
from src.signal_processing import bandpass_traces, median_welch_psd


# ==============================================================================
# INPUTS AND DISPLAY SETTINGS
# ==============================================================================

# Display-only shift applied to synthetic time. It is not used to alter the
# forward output, source time function, or predicted physical travel times.
SYN_DISPLAY_TIME_SHIFT_S = -0.20

TMIN = -0.30
TMAX = 2.00

FMIN_COMPARE_HZ = COMMON_FMIN_HZ
FMAX_COMPARE_HZ = COMMON_FMAX_HZ

TRACE_NORMALIZATION_PERCENTILE = 99.0
VMAX_SIGNED = 1.0
VMAX_ENVELOPE = 1.0

# Frequency-content QC windows; these are diagnostics, not phase picks.
PSD_CHANNEL_MIN = 550.0
PSD_CHANNEL_MAX = 1550.0

REAL_NOISE_TMIN_S = -1.8
REAL_NOISE_TMAX_S = -0.5

REAL_SIGNAL_TMIN_S = 0.2
REAL_SIGNAL_TMAX_S = 1.8

SYN_SIGNAL_TMIN_S = 0.0
SYN_SIGNAL_TMAX_S = 1.8

PSD_SEGMENT_LENGTH_S = 1.0
PSD_PLOT_FMAX_HZ = 100.0


# ==============================================================================
# GENERAL HELPERS
# ==============================================================================

def get_scalar(npz, name: str, default=np.nan):
    """Read a scalar value from an NPZ file."""
    if name not in npz.files:
        return default

    value = np.asarray(npz[name])

    if value.shape == ():
        return value.item()

    if value.size == 1:
        return value.reshape(-1)[0].item()

    return value


def trace_normalize(
    data: np.ndarray,
    percentile: float = TRACE_NORMALIZATION_PERCENTILE,
    eps: float = 1.0e-12,
) -> np.ndarray:
    """Normalize every trace independently for display only."""
    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError(
            f"trace_normalize expects a 2D array, got shape {data.shape}."
        )

    scale = np.percentile(
        np.abs(data),
        percentile,
        axis=1,
        keepdims=True,
    )
    scale = np.maximum(scale, eps)

    return data / scale


def compute_envelope(data: np.ndarray) -> np.ndarray:
    """
    Compute the Hilbert envelope along time.

    The envelope is evaluated before cropping the short display interval so
    Hilbert-transform edge effects stay away from the analysed window.
    """
    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError(
            f"compute_envelope expects a 2D array, got shape {data.shape}."
        )

    return np.abs(
        hilbert(
            data,
            axis=1,
        )
    )


def sort_by_channel(
    channels: np.ndarray,
    *arrays: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Sort a channel axis and all corresponding 1D or 2D arrays."""
    channels = np.asarray(channels, dtype=np.float64)

    if channels.ndim != 1:
        raise ValueError(
            f"channels must be 1D, got shape {channels.shape}."
        )

    if not np.all(np.isfinite(channels)):
        raise ValueError("Channel axis contains NaN or Inf.")

    order = np.argsort(channels)

    output: list[np.ndarray] = [
        channels[order],
    ]

    for array in arrays:
        array = np.asarray(array)

        if array.shape[0] != channels.size:
            raise ValueError(
                "The first dimension of every array must match the channel "
                f"axis: {array.shape[0]} != {channels.size}."
            )

        if array.ndim == 1:
            output.append(array[order])

        elif array.ndim == 2:
            output.append(array[order, :])

        else:
            raise ValueError(
                f"Unsupported array dimension {array.ndim}; expected 1 or 2."
            )

    return tuple(output)


def collapse_duplicate_channels(
    channels: np.ndarray,
    *arrays: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """
    Collapse scalar values assigned to duplicate raw channels.

    This is used only for P/S arrival curves. Waveform traces are never averaged.
    """
    channels = np.asarray(channels, dtype=np.float64)

    if channels.ndim != 1:
        raise ValueError(
            f"channels must be 1D, got shape {channels.shape}."
        )

    unique_channels = np.unique(channels)

    output: list[np.ndarray] = [
        unique_channels,
    ]

    for array in arrays:
        array = np.asarray(array, dtype=np.float64)

        if array.ndim != 1:
            raise ValueError(
                "collapse_duplicate_channels only supports 1D arrays."
            )

        if array.size != channels.size:
            raise ValueError(
                "Arrival array length must match channel-axis length: "
                f"{array.size} != {channels.size}."
            )

        collapsed = np.full(
            unique_channels.shape,
            np.nan,
            dtype=np.float64,
        )

        for i, channel in enumerate(unique_channels):
            values = array[channels == channel]

            if np.any(np.isfinite(values)):
                collapsed[i] = float(
                    np.nanmedian(values)
                )

        output.append(collapsed)

    return tuple(output)


def _relative_db(
    psd: np.ndarray,
    *,
    reference_peak: float | None = None,
    eps: float = 1.0e-300,
) -> np.ndarray:
    """Convert a non-negative PSD curve to dB relative to a chosen peak."""
    psd = np.asarray(psd, dtype=np.float64)

    if reference_peak is None:
        reference_peak = float(np.nanmax(psd))

    if not np.isfinite(reference_peak) or reference_peak <= 0.0:
        raise ValueError(
            f"Invalid PSD reference peak: {reference_peak}"
        )

    return 10.0 * np.log10(
        np.maximum(psd, eps)
        / max(reference_peak, eps)
    )


# ==============================================================================
# FREQUENCY-CONTENT QC
# ==============================================================================

def plot_frequency_qc(
    *,
    real_data_unfiltered: np.ndarray,
    real_time_s: np.ndarray,
    real_channels: np.ndarray,
    real_fs_hz: float,
    synthetic_data_unfiltered: np.ndarray,
    synthetic_time_shifted_s: np.ndarray,
    synthetic_channels: np.ndarray,
    synthetic_fs_hz: float,
    out_figure: Path,
    out_csv: Path,
) -> None:
    """
    Compare pre-filter real noise, real event signal, and synthetic spectra.

    Curves are independently normalized because absolute amplitudes from a 2D
    line-source calculation are not directly calibrated to a 3D earthquake.
    """
    real_trace_mask = (
        np.isfinite(real_channels)
        & (real_channels >= PSD_CHANNEL_MIN)
        & (real_channels <= PSD_CHANNEL_MAX)
    )
    synthetic_trace_mask = (
        np.isfinite(synthetic_channels)
        & (synthetic_channels >= PSD_CHANNEL_MIN)
        & (synthetic_channels <= PSD_CHANNEL_MAX)
    )

    if np.count_nonzero(real_trace_mask) < 2:
        raise RuntimeError(
            "Too few real channels remain in the PSD channel interval."
        )

    if np.count_nonzero(synthetic_trace_mask) < 2:
        raise RuntimeError(
            "Too few synthetic channels remain in the PSD channel interval."
        )

    f_real, real_signal_psd, real_signal_p10, real_signal_p90 = (
        median_welch_psd(
            real_data_unfiltered,
            real_time_s,
            fs_hz=real_fs_hz,
            tmin_s=REAL_SIGNAL_TMIN_S,
            tmax_s=REAL_SIGNAL_TMAX_S,
            trace_mask=real_trace_mask,
            segment_length_s=PSD_SEGMENT_LENGTH_S,
        )
    )

    f_noise, real_noise_psd, _, _ = median_welch_psd(
        real_data_unfiltered,
        real_time_s,
        fs_hz=real_fs_hz,
        tmin_s=REAL_NOISE_TMIN_S,
        tmax_s=REAL_NOISE_TMAX_S,
        trace_mask=real_trace_mask,
        segment_length_s=PSD_SEGMENT_LENGTH_S,
    )

    f_syn, synthetic_psd, synthetic_p10, synthetic_p90 = median_welch_psd(
        synthetic_data_unfiltered,
        synthetic_time_shifted_s,
        fs_hz=synthetic_fs_hz,
        tmin_s=SYN_SIGNAL_TMIN_S,
        tmax_s=SYN_SIGNAL_TMAX_S,
        trace_mask=synthetic_trace_mask,
        segment_length_s=PSD_SEGMENT_LENGTH_S,
    )

    if not np.array_equal(f_real, f_noise):
        real_noise_psd = np.interp(
            f_real,
            f_noise,
            real_noise_psd,
        )

    eps = 1.0e-300

    real_snr_db = 10.0 * np.log10(
        np.maximum(real_signal_psd, eps)
        / np.maximum(real_noise_psd, eps)
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
    )
    ax0, ax1 = axes

    real_signal_peak = float(
        np.nanmax(real_signal_psd)
    )
    real_noise_peak = float(
        np.nanmax(real_noise_psd)
    )
    synthetic_peak = float(
        np.nanmax(synthetic_psd)
    )

    ax0.plot(
        f_real,
        _relative_db(
            real_signal_psd,
            reference_peak=real_signal_peak,
        ),
        label="Real event window",
        linewidth=1.5,
    )
    ax0.fill_between(
        f_real,
        _relative_db(
            real_signal_p10,
            reference_peak=real_signal_peak,
        ),
        _relative_db(
            real_signal_p90,
            reference_peak=real_signal_peak,
        ),
        alpha=0.15,
    )

    ax0.plot(
        f_real,
        _relative_db(
            real_noise_psd,
            reference_peak=real_noise_peak,
        ),
        label="Real pre-event noise",
        linewidth=1.2,
    )

    ax0.plot(
        f_syn,
        _relative_db(
            synthetic_psd,
            reference_peak=synthetic_peak,
        ),
        label="Synthetic",
        linewidth=1.5,
    )
    ax0.fill_between(
        f_syn,
        _relative_db(
            synthetic_p10,
            reference_peak=synthetic_peak,
        ),
        _relative_db(
            synthetic_p90,
            reference_peak=synthetic_peak,
        ),
        alpha=0.12,
    )

    ax0.axvspan(
        FMIN_COMPARE_HZ,
        FMAX_COMPARE_HZ,
        alpha=0.12,
        label=(
            f"Common band "
            f"{FMIN_COMPARE_HZ:g}–{FMAX_COMPARE_HZ:g} Hz"
        ),
    )

    ax0.set_ylabel(
        "Median PSD relative to peak [dB]"
    )
    ax0.set_ylim(
        -80.0,
        5.0,
    )
    ax0.grid(
        alpha=0.3
    )
    ax0.legend(
        fontsize=9
    )

    ax1.plot(
        f_real,
        real_snr_db,
        linewidth=1.4,
    )
    ax1.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )
    ax1.axvspan(
        FMIN_COMPARE_HZ,
        FMAX_COMPARE_HZ,
        alpha=0.12,
    )
    ax1.set_xlabel(
        "Frequency [Hz]"
    )
    ax1.set_ylabel(
        "Real signal / noise PSD [dB]"
    )
    ax1.grid(
        alpha=0.3
    )

    max_f = min(
        PSD_PLOT_FMAX_HZ,
        float(np.nanmax(f_real)),
        float(np.nanmax(f_syn)),
    )
    ax1.set_xlim(
        0.0,
        max_f,
    )

    fig.suptitle(
        "SAFOD real/synthetic frequency-content QC\n"
        "Curves are independently normalized; compare spectral shape, "
        "not absolute scale."
    )
    fig.tight_layout()
    fig.savefig(
        out_figure,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(
        fig
    )

    synthetic_interp = np.interp(
        f_real,
        f_syn,
        synthetic_psd,
        left=np.nan,
        right=np.nan,
    )

    pd.DataFrame(
        {
            "frequency_hz": f_real,
            "real_signal_psd": real_signal_psd,
            "real_noise_psd": real_noise_psd,
            "real_signal_to_noise_db": real_snr_db,
            "synthetic_psd_interpolated": synthetic_interp,
        }
    ).to_csv(
        out_csv,
        index=False,
    )


# ==============================================================================
# PLOTTING HELPERS
# ==============================================================================

def add_arrival_overlays(
    *,
    ax,
    arrival_channels: np.ndarray,
    p_arrivals_s: np.ndarray,
    s_arrivals_s: np.ndarray,
    arrival_time_shift_s: float = 0.0,
    include_labels: bool = True,
    line_color: str = "black",
) -> None:
    """Add model-based approximate P/S curves to a gather plot."""
    arrival_channels = np.asarray(
        arrival_channels,
        dtype=np.float64,
    )
    p_arrivals_s = np.asarray(
        p_arrivals_s,
        dtype=np.float64,
    )
    s_arrivals_s = np.asarray(
        s_arrivals_s,
        dtype=np.float64,
    )

    if not (
        arrival_channels.shape
        == p_arrivals_s.shape
        == s_arrivals_s.shape
    ):
        raise ValueError(
            "Arrival channel/P/S arrays must have identical shapes."
        )

    valid_p = (
        np.isfinite(arrival_channels)
        & np.isfinite(p_arrivals_s)
    )
    valid_s = (
        np.isfinite(arrival_channels)
        & np.isfinite(s_arrivals_s)
    )

    ax.plot(
        arrival_channels[valid_p],
        p_arrivals_s[valid_p]
        + float(arrival_time_shift_s),
        color=line_color,
        linestyle="--",
        linewidth=1.6,
        label="Predicted P" if include_labels else None,
        zorder=20,
    )

    ax.plot(
        arrival_channels[valid_s],
        s_arrivals_s[valid_s]
        + float(arrival_time_shift_s),
        color=line_color,
        linestyle=":",
        linewidth=1.9,
        label="Predicted S" if include_labels else None,
        zorder=20,
    )


def plot_real_with_arrivals(
    *,
    real_signed_normalized: np.ndarray,
    real_time_s: np.ndarray,
    real_channels: np.ndarray,
    arrival_channels: np.ndarray,
    p_arrivals_s: np.ndarray,
    s_arrivals_s: np.ndarray,
    event_id: str,
    out_path: Path,
) -> None:
    """Real signed DAS gather with predicted P/S arrivals only."""
    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    image = ax.imshow(
        real_signed_normalized.T,
        extent=[
            float(real_channels.min()),
            float(real_channels.max()),
            float(real_time_s[-1]),
            float(real_time_s[0]),
        ],
        aspect="auto",
        cmap="seismic",
        vmin=-VMAX_SIGNED,
        vmax=VMAX_SIGNED,
        interpolation="none",
    )

    ax.axhline(
        0.0,
        color="black",
        linewidth=1.0,
        linestyle="-.",
        label="Catalogue origin",
    )

    add_arrival_overlays(
        ax=ax,
        arrival_channels=arrival_channels,
        p_arrivals_s=p_arrivals_s,
        s_arrivals_s=s_arrivals_s,
        arrival_time_shift_s=0.0,
        include_labels=True,
        line_color="black",
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Trace-normalized amplitude",
    )

    ax.set_title(
        f"{event_id} real DAS: predicted P/S arrivals"
    )
    ax.set_xlabel(
        "Raw channel number"
    )
    ax.set_ylabel(
        "Time from catalogue origin [s]"
    )
    ax.set_xlim(
        float(real_channels.min()),
        float(real_channels.max()),
    )
    ax.set_ylim(
        float(real_time_s[-1]),
        float(real_time_s[0]),
    )
    ax.grid(
        alpha=0.25
    )
    ax.legend(
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
    )

    fig.tight_layout()
    fig.savefig(
        out_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(
        fig
    )


def plot_signed_side_by_side(
    *,
    real_signed_normalized: np.ndarray,
    real_time_s: np.ndarray,
    real_channels: np.ndarray,
    synthetic_signed_normalized: np.ndarray,
    synthetic_time_shifted_s: np.ndarray,
    synthetic_channels: np.ndarray,
    arrival_channels: np.ndarray,
    p_arrivals_s: np.ndarray,
    s_arrivals_s: np.ndarray,
    event_id: str,
    source_f0_hz: float,
    source_theta_deg: float,
    out_path: Path,
) -> None:
    """Signed, trace-normalized real/synthetic DAS comparison."""
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(18, 7),
        sharey=True,
    )

    ax_real, ax_syn = axes

    ax_real.imshow(
        real_signed_normalized.T,
        extent=[
            float(real_channels.min()),
            float(real_channels.max()),
            float(real_time_s[-1]),
            float(real_time_s[0]),
        ],
        aspect="auto",
        cmap="seismic",
        vmin=-VMAX_SIGNED,
        vmax=VMAX_SIGNED,
        interpolation="none",
    )

    ax_real.axhline(
        0.0,
        color="black",
        linewidth=1.0,
        linestyle="-.",
        label="Catalogue origin",
    )

    add_arrival_overlays(
        ax=ax_real,
        arrival_channels=arrival_channels,
        p_arrivals_s=p_arrivals_s,
        s_arrivals_s=s_arrivals_s,
        arrival_time_shift_s=0.0,
        include_labels=True,
        line_color="black",
    )

    ax_real.set_title(
        "Real DAS, down-going pass"
    )
    ax_real.set_xlabel(
        "Raw channel number"
    )
    ax_real.set_ylabel(
        "Time from origin [s]"
    )
    ax_real.set_xlim(
        float(real_channels.min()),
        float(real_channels.max()),
    )
    ax_real.grid(
        alpha=0.25
    )
    ax_real.legend(
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
    )

    synthetic_image = ax_syn.imshow(
        synthetic_signed_normalized.T,
        extent=[
            float(synthetic_channels.min()),
            float(synthetic_channels.max()),
            float(synthetic_time_shifted_s[-1]),
            float(synthetic_time_shifted_s[0]),
        ],
        aspect="auto",
        cmap="seismic",
        vmin=-VMAX_SIGNED,
        vmax=VMAX_SIGNED,
        interpolation="none",
    )

    ax_syn.axhline(
        0.0,
        color="black",
        linewidth=1.0,
        linestyle="-.",
    )

    add_arrival_overlays(
        ax=ax_syn,
        arrival_channels=arrival_channels,
        p_arrivals_s=p_arrivals_s,
        s_arrivals_s=s_arrivals_s,
        arrival_time_shift_s=SYN_DISPLAY_TIME_SHIFT_S,
        include_labels=True,
        line_color="black",
    )

    ax_syn.set_title(
        "Synthetic DAS, down-going pass"
    )
    ax_syn.set_xlabel(
        "Exact raw channel number"
    )
    ax_syn.set_xlim(
        float(synthetic_channels.min()),
        float(synthetic_channels.max()),
    )
    ax_syn.grid(
        alpha=0.25
    )
    ax_syn.legend(
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
    )

    fig.colorbar(
        synthetic_image,
        ax=axes.ravel().tolist(),
        label="Trace-normalized amplitude",
        shrink=0.85,
    )

    fig.suptitle(
        f"{event_id} real vs synthetic signed DAS\n"
        f"f0={source_f0_hz:.1f} Hz, "
        f"theta={source_theta_deg:.1f}°, "
        f"synthetic display shift={SYN_DISPLAY_TIME_SHIFT_S:+.2f} s",
        y=0.98,
    )

    fig.savefig(
        out_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(
        fig
    )


def plot_envelope_side_by_side(
    *,
    real_envelope_normalized: np.ndarray,
    real_time_s: np.ndarray,
    real_channels: np.ndarray,
    synthetic_envelope_normalized: np.ndarray,
    synthetic_time_shifted_s: np.ndarray,
    synthetic_channels: np.ndarray,
    arrival_channels: np.ndarray,
    p_arrivals_s: np.ndarray,
    s_arrivals_s: np.ndarray,
    event_id: str,
    source_f0_hz: float,
    source_theta_deg: float,
    out_path: Path,
) -> None:
    """Real/synthetic Hilbert-envelope comparison without ridge picks."""
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(18, 7),
        sharey=True,
    )

    ax_real, ax_syn = axes

    ax_real.imshow(
        real_envelope_normalized.T,
        extent=[
            float(real_channels.min()),
            float(real_channels.max()),
            float(real_time_s[-1]),
            float(real_time_s[0]),
        ],
        aspect="auto",
        cmap="magma",
        vmin=0.0,
        vmax=VMAX_ENVELOPE,
        interpolation="none",
    )

    ax_real.axhline(
        0.0,
        color="white",
        linewidth=1.0,
        linestyle="-.",
        label="Catalogue origin",
    )

    add_arrival_overlays(
        ax=ax_real,
        arrival_channels=arrival_channels,
        p_arrivals_s=p_arrivals_s,
        s_arrivals_s=s_arrivals_s,
        arrival_time_shift_s=0.0,
        include_labels=True,
        line_color="white",
    )

    ax_real.set_title(
        "Real DAS envelope"
    )
    ax_real.set_xlabel(
        "Raw channel number"
    )
    ax_real.set_ylabel(
        "Time from origin [s]"
    )
    ax_real.set_xlim(
        float(real_channels.min()),
        float(real_channels.max()),
    )
    ax_real.grid(
        alpha=0.20
    )
    ax_real.legend(
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
    )

    synthetic_image = ax_syn.imshow(
        synthetic_envelope_normalized.T,
        extent=[
            float(synthetic_channels.min()),
            float(synthetic_channels.max()),
            float(synthetic_time_shifted_s[-1]),
            float(synthetic_time_shifted_s[0]),
        ],
        aspect="auto",
        cmap="magma",
        vmin=0.0,
        vmax=VMAX_ENVELOPE,
        interpolation="none",
    )

    ax_syn.axhline(
        0.0,
        color="white",
        linewidth=1.0,
        linestyle="-.",
    )

    add_arrival_overlays(
        ax=ax_syn,
        arrival_channels=arrival_channels,
        p_arrivals_s=p_arrivals_s,
        s_arrivals_s=s_arrivals_s,
        arrival_time_shift_s=SYN_DISPLAY_TIME_SHIFT_S,
        include_labels=True,
        line_color="white",
    )

    ax_syn.set_title(
        "Synthetic DAS envelope"
    )
    ax_syn.set_xlabel(
        "Exact raw channel number"
    )
    ax_syn.set_xlim(
        float(synthetic_channels.min()),
        float(synthetic_channels.max()),
    )
    ax_syn.grid(
        alpha=0.20
    )
    ax_syn.legend(
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
    )

    fig.colorbar(
        synthetic_image,
        ax=axes.ravel().tolist(),
        label="Trace-normalized envelope",
        shrink=0.85,
    )

    fig.suptitle(
        f"{event_id} real vs synthetic DAS envelopes\n"
        f"band={FMIN_COMPARE_HZ:.0f}–{FMAX_COMPARE_HZ:.0f} Hz, "
        f"f0={source_f0_hz:.1f} Hz, "
        f"theta={source_theta_deg:.1f}°",
        y=0.98,
    )

    fig.savefig(
        out_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(
        fig
    )


# ==============================================================================
# COMMAND-LINE INTERFACE
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the prepared real SAFOD DAS event with one synthetic "
            "forward-model run. No observed-ridge picks are used."
        )
    )

    parser.add_argument(
        "--theta-deg",
        type=float,
        default=DEFAULT_THETA_DEG,
        help=(
            "Effective 2D double-couple orientation identifying the synthetic "
            f"run. Default: {DEFAULT_THETA_DEG:.1f}."
        ),
    )

    return parser.parse_args()


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    args = parse_args()

    if not 0.0 <= args.theta_deg < 90.0:
        raise ValueError(
            "--theta-deg must satisfy 0 <= theta < 90 for the current "
            "2D double-couple parameterisation."
        )

    requested_run_tag = forward_run_tag(
        args.theta_deg
    )
    real_pkg = REAL_EVENT_PACKAGE
    syn_pkg = forward_package_for_theta(
        args.theta_deg
    )
    out_dir = comparison_dir_for_theta(
        args.theta_deg
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nComparison inputs")
    print("-----------------")
    print(f"requested theta : {args.theta_deg:.1f} deg")
    print(f"run tag         : {requested_run_tag}")
    print(f"real package    : {real_pkg}")
    print(f"synthetic pkg   : {syn_pkg}")
    print(f"output dir      : {out_dir}")

    if not real_pkg.exists():
        raise FileNotFoundError(
            f"Real-event package not found: {real_pkg}"
        )

    if not syn_pkg.exists():
        raise FileNotFoundError(
            f"Synthetic package not found: {syn_pkg}"
        )

    with np.load(
        real_pkg,
        allow_pickle=True,
    ) as real, np.load(
        syn_pkg,
        allow_pickle=True,
    ) as synthetic:

        # ----------------------------------------------------------------------
        # Metadata
        # ----------------------------------------------------------------------
        event_id = str(
            get_scalar(
                real,
                "ev_id",
                "unknown",
            )
        )

        source_f0_hz = float(
            get_scalar(
                synthetic,
                "source_f0_hz",
                np.nan,
            )
        )

        source_theta_deg = float(
            get_scalar(
                synthetic,
                "source_theta_deg",
                np.nan,
            )
        )

        if (
            not np.isfinite(source_f0_hz)
            or source_f0_hz <= 0.0
        ):
            raise ValueError(
                "Synthetic package contains invalid source_f0_hz: "
                f"{source_f0_hz}."
            )

        if not np.isfinite(source_theta_deg):
            raise ValueError(
                "Synthetic package does not contain a finite "
                "source_theta_deg."
            )

        if not np.isclose(
            source_theta_deg,
            args.theta_deg,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError(
                "Requested mechanism angle does not match the synthetic "
                "package: "
                f"requested={args.theta_deg:.6f} deg, "
                f"package={source_theta_deg:.6f} deg, "
                f"path={syn_pkg}."
            )

        package_run_tag = str(
            get_scalar(
                synthetic,
                "run_tag",
                requested_run_tag,
            )
        )

        if package_run_tag != requested_run_tag:
            raise ValueError(
                "Synthetic package run_tag does not match the requested path: "
                f"requested={requested_run_tag!r}, "
                f"package={package_run_tag!r}."
            )

        # ----------------------------------------------------------------------
        # Real DAS
        # ----------------------------------------------------------------------
        if "das_data_unfiltered" not in real.files:
            raise RuntimeError(
                "Real-event package is from the old preprocessing workflow. "
                "Rerun: python -m scripts.safod.prepare_event"
            )

        real_data_unfiltered_full = np.asarray(
            real["das_data_unfiltered"],
            dtype=np.float64,
        )
        real_time_full = np.asarray(
            real["t"],
            dtype=np.float64,
        )
        real_channels = np.asarray(
            real["raw_channels"],
            dtype=np.float64,
        )
        real_fs = float(
            get_scalar(
                real,
                "fs",
            )
        )

        if real_data_unfiltered_full.shape != (
            real_channels.size,
            real_time_full.size,
        ):
            raise ValueError(
                "Real DAS shape does not match channel/time axes: "
                f"{real_data_unfiltered_full.shape} != "
                f"({real_channels.size}, {real_time_full.size})."
            )

        (
            real_channels,
            real_data_unfiltered_full,
        ) = sort_by_channel(
            real_channels,
            real_data_unfiltered_full,
        )

        # ----------------------------------------------------------------------
        # Synthetic DAS and exact raw-channel registration
        # ----------------------------------------------------------------------
        if "das_raw_channels" not in synthetic.files:
            raise RuntimeError(
                "Synthetic package does not contain exact das_raw_channels. "
                "Rerun scripts.safod.run_forward with the current exact "
                "registered-channel implementation."
            )

        synthetic_data_unfiltered_full = np.asarray(
            synthetic["das_data"],
            dtype=np.float64,
        )
        synthetic_time_full = np.asarray(
            synthetic["t"],
            dtype=np.float64,
        )
        synthetic_channels = np.asarray(
            synthetic["das_raw_channels"],
            dtype=np.float64,
        )
        predicted_p = np.asarray(
            synthetic["arrival_p_das"],
            dtype=np.float64,
        )
        predicted_s = np.asarray(
            synthetic["arrival_swave_das"],
            dtype=np.float64,
        )

        if synthetic_data_unfiltered_full.ndim != 2:
            raise ValueError(
                "Synthetic DAS must be 2D; got "
                f"{synthetic_data_unfiltered_full.shape}."
            )

        if synthetic_data_unfiltered_full.shape != (
            synthetic_channels.size,
            synthetic_time_full.size,
        ):
            raise ValueError(
                "Synthetic DAS shape does not match exact channel/time axes: "
                f"{synthetic_data_unfiltered_full.shape} != "
                f"({synthetic_channels.size}, "
                f"{synthetic_time_full.size})."
            )

        if not (
            predicted_p.shape
            == predicted_s.shape
            == synthetic_channels.shape
        ):
            raise ValueError(
                "Synthetic P/S arrival arrays must match das_raw_channels: "
                f"channels={synthetic_channels.shape}, "
                f"P={predicted_p.shape}, S={predicted_s.shape}."
            )

        (
            synthetic_channels,
            synthetic_data_unfiltered_full,
            predicted_p,
            predicted_s,
        ) = sort_by_channel(
            synthetic_channels,
            synthetic_data_unfiltered_full,
            predicted_p,
            predicted_s,
        )

        synthetic_dt = float(
            get_scalar(
                synthetic,
                "dt",
                np.median(
                    np.diff(
                        synthetic_time_full
                    )
                ),
            )
        )

        if not np.isfinite(synthetic_dt) or synthetic_dt <= 0.0:
            raise ValueError(
                f"Invalid synthetic dt: {synthetic_dt}"
            )

        synthetic_fs = 1.0 / synthetic_dt

    # NPZ files are closed from this point onward.

    # --------------------------------------------------------------------------
    # Common zero-phase filtering
    # --------------------------------------------------------------------------
    print(
        "Filtering real DAS to "
        f"{FMIN_COMPARE_HZ:.1f}–{FMAX_COMPARE_HZ:.1f} Hz, zero phase..."
    )

    real_data_filtered_full = bandpass_traces(
        real_data_unfiltered_full,
        fs_hz=real_fs,
        fmin_hz=FMIN_COMPARE_HZ,
        fmax_hz=FMAX_COMPARE_HZ,
        order=FILTER_ORDER,
        taper_frac=FILTER_TAPER_FRAC,
    )

    print(
        "Filtering synthetic DAS to "
        f"{FMIN_COMPARE_HZ:.1f}–{FMAX_COMPARE_HZ:.1f} Hz, zero phase..."
    )

    synthetic_data_filtered_full = bandpass_traces(
        synthetic_data_unfiltered_full,
        fs_hz=synthetic_fs,
        fmin_hz=FMIN_COMPARE_HZ,
        fmax_hz=FMAX_COMPARE_HZ,
        order=FILTER_ORDER,
        taper_frac=FILTER_TAPER_FRAC,
    )

    # --------------------------------------------------------------------------
    # Frequency QC uses unfiltered data
    # --------------------------------------------------------------------------
    synthetic_time_shifted_full = (
        synthetic_time_full
        + SYN_DISPLAY_TIME_SHIFT_S
    )

    frequency_figure_path = (
        out_dir
        / "00_frequency_content_qc.png"
    )
    frequency_csv_path = (
        out_dir
        / "frequency_content_qc.csv"
    )

    plot_frequency_qc(
        real_data_unfiltered=real_data_unfiltered_full,
        real_time_s=real_time_full,
        real_channels=real_channels,
        real_fs_hz=real_fs,
        synthetic_data_unfiltered=synthetic_data_unfiltered_full,
        synthetic_time_shifted_s=synthetic_time_shifted_full,
        synthetic_channels=synthetic_channels,
        synthetic_fs_hz=synthetic_fs,
        out_figure=frequency_figure_path,
        out_csv=frequency_csv_path,
    )

    # --------------------------------------------------------------------------
    # Envelopes are computed before display-window cropping
    # --------------------------------------------------------------------------
    real_envelope_full = compute_envelope(
        real_data_filtered_full
    )
    synthetic_envelope_full = compute_envelope(
        synthetic_data_filtered_full
    )

    real_time_mask = (
        (real_time_full >= TMIN)
        & (real_time_full <= TMAX)
    )
    synthetic_time_mask = (
        (synthetic_time_shifted_full >= TMIN)
        & (synthetic_time_shifted_full <= TMAX)
    )

    if np.count_nonzero(real_time_mask) < 2:
        raise RuntimeError(
            "Requested real-data display interval contains fewer than "
            "two samples."
        )

    if np.count_nonzero(synthetic_time_mask) < 2:
        raise RuntimeError(
            "Requested synthetic display interval contains fewer than "
            "two samples."
        )

    real_time = real_time_full[
        real_time_mask
    ]
    real_data = real_data_filtered_full[
        :,
        real_time_mask,
    ]
    real_envelope = real_envelope_full[
        :,
        real_time_mask,
    ]

    synthetic_time_shifted = synthetic_time_shifted_full[
        synthetic_time_mask
    ]
    synthetic_data = synthetic_data_filtered_full[
        :,
        synthetic_time_mask,
    ]
    synthetic_envelope = synthetic_envelope_full[
        :,
        synthetic_time_mask,
    ]

    real_signed_normalized = trace_normalize(
        real_data
    )
    real_envelope_normalized = trace_normalize(
        real_envelope
    )
    synthetic_signed_normalized = trace_normalize(
        synthetic_data
    )
    synthetic_envelope_normalized = trace_normalize(
        synthetic_envelope
    )

    # --------------------------------------------------------------------------
    # Strictly increasing channel axis for plotting arrival curves
    # --------------------------------------------------------------------------
    (
        arrival_channels,
        predicted_p_unique,
        predicted_s_unique,
    ) = collapse_duplicate_channels(
        synthetic_channels,
        predicted_p,
        predicted_s,
    )

    arrival_range_mask = (
        np.isfinite(arrival_channels)
        & np.isfinite(predicted_p_unique)
        & np.isfinite(predicted_s_unique)
        & (arrival_channels >= real_channels.min())
        & (arrival_channels <= real_channels.max())
    )

    arrival_channels = arrival_channels[
        arrival_range_mask
    ]
    predicted_p_unique = predicted_p_unique[
        arrival_range_mask
    ]
    predicted_s_unique = predicted_s_unique[
        arrival_range_mask
    ]

    if arrival_channels.size < 2:
        raise RuntimeError(
            "Too few valid exact-channel arrival samples remain after "
            "restricting to the real-data channel range."
        )

    # --------------------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------------------
    print("\nReal/Synthetic comparison QC")
    print("----------------------------")
    print(f"event id                 : {event_id}")
    print(f"source f0                : {source_f0_hz:.2f} Hz")
    print(f"source theta             : {source_theta_deg:.2f} deg")
    print(
        "common comparison band   : "
        f"{FMIN_COMPARE_HZ:.1f} to {FMAX_COMPARE_HZ:.1f} Hz, zero phase"
    )
    print(f"real DAS shape           : {real_data.shape}")
    print(f"synthetic DAS shape      : {synthetic_data.shape}")
    print(
        "real time range          : "
        f"{real_time.min():.3f} to {real_time.max():.3f} s"
    )
    print(
        "synthetic display range  : "
        f"{synthetic_time_shifted.min():.3f} to "
        f"{synthetic_time_shifted.max():.3f} s"
    )
    print(
        "real channel range       : "
        f"{real_channels.min():.1f} to {real_channels.max():.1f}"
    )
    print(
        "synthetic channel range  : "
        f"{synthetic_channels.min():.1f} to "
        f"{synthetic_channels.max():.1f}"
    )
    print(
        "P arrival range          : "
        f"{np.nanmin(predicted_p_unique):.3f} to "
        f"{np.nanmax(predicted_p_unique):.3f} s"
    )
    print(
        "S arrival range          : "
        f"{np.nanmin(predicted_s_unique):.3f} to "
        f"{np.nanmax(predicted_s_unique):.3f} s"
    )
    print(
        "synthetic display shift  : "
        f"{SYN_DISPLAY_TIME_SHIFT_S:+.3f} s"
    )
    print("observed ridge picker     : disabled")

    summary_csv = (
        out_dir
        / "comparison_summary.csv"
    )

    pd.DataFrame(
        [
            {
                "event_id": event_id,
                "run_tag": requested_run_tag,
                "requested_theta_deg": float(args.theta_deg),
                "source_f0_hz": source_f0_hz,
                "source_theta_deg": source_theta_deg,
                "synthetic_display_shift_s": SYN_DISPLAY_TIME_SHIFT_S,
                "common_fmin_hz": FMIN_COMPARE_HZ,
                "common_fmax_hz": FMAX_COMPARE_HZ,
                "filter_order": FILTER_ORDER,
                "filter_taper_frac": FILTER_TAPER_FRAC,
                "filter_phase": "zero_phase_sosfiltfilt_both",
                "synthetic_channel_mapping": (
                    "exact_das_raw_channels_from_forward_package"
                ),
                "observed_ridge_picker": "disabled",
                "comparison_scope": (
                    "signed gather, Hilbert envelope, predicted P/S overlays, "
                    "and pre-filter frequency-content QC"
                ),
                "amplitude_note": (
                    "Trace-normalized display only; absolute 2D line-source "
                    "amplitudes are not calibrated to a 3D point earthquake"
                ),
            }
        ]
    ).to_csv(
        summary_csv,
        index=False,
    )

    # --------------------------------------------------------------------------
    # Remove stale ridge products from previous runs
    # --------------------------------------------------------------------------
    stale_names = [
        "01_real_with_arrivals_and_observed_ridge.png",
        "04_real_vs_synthetic_envelope_ridge.png",
        "04_observed_ridge_minus_predicted_S.png",
        "observed_ridge_and_arrival_residuals.csv",
    ]

    for name in stale_names:
        stale_path = out_dir / name
        if stale_path.exists():
            stale_path.unlink()
            print(f"Removed stale ridge product: {stale_path}")

    # --------------------------------------------------------------------------
    # Figures
    # --------------------------------------------------------------------------
    real_overlay_path = (
        out_dir
        / "01_real_with_predicted_arrivals.png"
    )
    signed_comparison_path = (
        out_dir
        / "02_real_vs_synthetic_signed.png"
    )
    envelope_comparison_path = (
        out_dir
        / "03_real_vs_synthetic_envelopes.png"
    )

    plot_real_with_arrivals(
        real_signed_normalized=real_signed_normalized,
        real_time_s=real_time,
        real_channels=real_channels,
        arrival_channels=arrival_channels,
        p_arrivals_s=predicted_p_unique,
        s_arrivals_s=predicted_s_unique,
        event_id=event_id,
        out_path=real_overlay_path,
    )

    plot_signed_side_by_side(
        real_signed_normalized=real_signed_normalized,
        real_time_s=real_time,
        real_channels=real_channels,
        synthetic_signed_normalized=synthetic_signed_normalized,
        synthetic_time_shifted_s=synthetic_time_shifted,
        synthetic_channels=synthetic_channels,
        arrival_channels=arrival_channels,
        p_arrivals_s=predicted_p_unique,
        s_arrivals_s=predicted_s_unique,
        event_id=event_id,
        source_f0_hz=source_f0_hz,
        source_theta_deg=source_theta_deg,
        out_path=signed_comparison_path,
    )

    plot_envelope_side_by_side(
        real_envelope_normalized=real_envelope_normalized,
        real_time_s=real_time,
        real_channels=real_channels,
        synthetic_envelope_normalized=synthetic_envelope_normalized,
        synthetic_time_shifted_s=synthetic_time_shifted,
        synthetic_channels=synthetic_channels,
        arrival_channels=arrival_channels,
        p_arrivals_s=predicted_p_unique,
        s_arrivals_s=predicted_s_unique,
        event_id=event_id,
        source_f0_hz=source_f0_hz,
        source_theta_deg=source_theta_deg,
        out_path=envelope_comparison_path,
    )

    print("\nSaved outputs")
    print("-------------")
    print(frequency_figure_path)
    print(frequency_csv_path)
    print(real_overlay_path)
    print(signed_comparison_path)
    print(envelope_comparison_path)
    print(summary_csv)


if __name__ == "__main__":
    main()