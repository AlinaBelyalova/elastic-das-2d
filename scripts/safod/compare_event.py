from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.signal import hilbert
    from scipy.spatial import cKDTree
except ImportError as exc:
    raise ImportError(
        "This script requires scipy.signal and scipy.spatial. "
        "Run it in the fwi/geo environment where SciPy is installed."
    ) from exc

from scripts.safod.settings import (
    COMMON_FMAX_HZ,
    COMMON_FMIN_HZ,
    COMPARISON_DIR,
    FILTER_ORDER,
    FILTER_TAPER_FRAC,
    FORWARD_PACKAGE,
    GEOMETRY_CSV,
    REAL_EVENT_PACKAGE,
)
from src.signal_processing import bandpass_traces, median_welch_psd


# ==============================================================================
# INPUTS AND SETTINGS
# ==============================================================================

REAL_PKG = REAL_EVENT_PACKAGE
SYN_PKG = FORWARD_PACKAGE
GEOM_CSV = GEOMETRY_CSV
OUT_DIR = COMPARISON_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYN_DISPLAY_TIME_SHIFT_S = -0.20

TMIN = -0.30
TMAX = 2.00

FMIN_COMPARE_HZ = COMMON_FMIN_HZ
FMAX_COMPARE_HZ = COMMON_FMAX_HZ

TRACE_NORMALIZATION_PERCENTILE = 99.0
VMAX_SIGNED = 1.0
VMAX_ENVELOPE = 1.0

# Guided dominant-envelope ridge picker.
RIDGE_ANCHORS = [
    (500.0, 1.18),
    (700.0, 1.02),
    (1000.0, 0.82),
    (1300.0, 0.62),
    (1600.0, 0.36),
]

RIDGE_PICK_CH_MIN = 500.0
RIDGE_PICK_CH_MAX = 1650.0
RIDGE_SEARCH_HALF_WIDTH_S = 0.16
RIDGE_SMOOTH_WINDOW = 31

# Interval used for apparent-velocity and residual statistics.
FIT_CH_MIN = 550.0
FIT_CH_MAX = 1550.0

# Frequency-content QC windows (diagnostic only, not picks)
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
    """
    Read a scalar value from an NPZ file.
    """
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
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Normalize every trace independently for display only.
    """
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
    Compute Hilbert envelope along the time axis.

    This should be called before cropping the short display interval so that
    Hilbert edge effects remain as far as possible from the analysed window.
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


def nearest_geometry_channel_for_receivers(
    receiver_x: np.ndarray,
    receiver_z: np.ndarray,
    geom_csv: Path,
) -> np.ndarray:
    """
    Map synthetic receiver positions to approximate raw downleg channel numbers.
    """
    geom = pd.read_csv(geom_csv)

    required = [
        "Channel",
        "X_2D_m",
        "Z_2D_m",
    ]

    for column in required:
        if column not in geom.columns:
            raise ValueError(
                f"Missing column {column!r} in {geom_csv}"
            )

    channels = pd.to_numeric(
        geom["Channel"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    geom_x = pd.to_numeric(
        geom["X_2D_m"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    geom_z = pd.to_numeric(
        geom["Z_2D_m"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    valid = (
        np.isfinite(channels)
        & np.isfinite(geom_x)
        & np.isfinite(geom_z)
    )

    channels = channels[valid]
    geom_x = geom_x[valid]
    geom_z = geom_z[valid]

    receiver_x = np.asarray(receiver_x, dtype=np.float64)
    receiver_z = np.asarray(receiver_z, dtype=np.float64)

    if receiver_x.shape != receiver_z.shape:
        raise ValueError(
            "receiver_x and receiver_z must have matching shapes; "
            f"got {receiver_x.shape} and {receiver_z.shape}."
        )

    if channels.size == 0:
        raise ValueError(
            f"No valid geometry rows found in {geom_csv}"
        )

    tree = cKDTree(
        np.column_stack(
            [
                geom_x,
                geom_z,
            ]
        )
    )

    _, nearest = tree.query(
        np.column_stack(
            [
                receiver_x,
                receiver_z,
            ]
        )
    )

    return channels[nearest]


def sort_by_channel(
    channels: np.ndarray,
    *arrays: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """
    Sort a channel axis and every corresponding 1D or 2D array.
    """
    channels = np.asarray(channels, dtype=np.float64)
    order = np.argsort(channels)

    output = [
        channels[order],
    ]

    for array in arrays:
        array = np.asarray(array)

        if array.shape[0] != channels.size:
            raise ValueError(
                "The first dimension of every array must match the channel axis: "
                f"{array.shape[0]} != {channels.size}."
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
    Collapse scalar values assigned to duplicate mapped raw channels.

    This function is used only for scalar P/S arrival times. It does not average
    waveform traces. Duplicate scalar values are collapsed with the median.
    """
    channels = np.asarray(channels, dtype=np.float64)

    if channels.ndim != 1:
        raise ValueError(
            f"channels must be 1D, got shape {channels.shape}."
        )

    unique_channels = np.unique(channels)

    output = [
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


# ==============================================================================
# OBSERVED RIDGE DIAGNOSTICS
# ==============================================================================

def pick_real_ridge_guided(
    *,
    real_envelope_normalized: np.ndarray,
    time_s: np.ndarray,
    raw_channels: np.ndarray,
    anchor_ch_t: list[tuple[float, float]],
    half_width_s: float,
    smooth_window: int,
    pick_ch_min: float,
    pick_ch_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pick the strongest real-data envelope ridge around a guide curve.

    The guide is defined using manually selected channel/time anchors. For every
    channel, the picker selects the envelope maximum within a narrow time window
    around the interpolated guide.

    This is a guided dominant-envelope ridge, not an automatic P/S first-break
    picker.
    """
    real_envelope_normalized = np.asarray(
        real_envelope_normalized,
        dtype=np.float64,
    )
    time_s = np.asarray(
        time_s,
        dtype=np.float64,
    )
    raw_channels = np.asarray(
        raw_channels,
        dtype=np.float64,
    )

    if real_envelope_normalized.shape != (
        raw_channels.size,
        time_s.size,
    ):
        raise ValueError(
            "Envelope shape must match channel/time axes: "
            f"{real_envelope_normalized.shape} != "
            f"({raw_channels.size}, {time_s.size})."
        )

    anchors = np.asarray(
        anchor_ch_t,
        dtype=np.float64,
    )

    if anchors.ndim != 2 or anchors.shape[1] != 2:
        raise ValueError(
            "anchor_ch_t must be a list of (channel, time) pairs."
        )

    anchor_channels = anchors[:, 0]
    anchor_times = anchors[:, 1]

    anchor_order = np.argsort(anchor_channels)
    anchor_channels = anchor_channels[anchor_order]
    anchor_times = anchor_times[anchor_order]

    guide_times = np.interp(
        raw_channels,
        anchor_channels,
        anchor_times,
    )

    ridge_times = np.full(
        raw_channels.shape,
        np.nan,
        dtype=np.float64,
    )

    ridge_amplitudes = np.full(
        raw_channels.shape,
        np.nan,
        dtype=np.float64,
    )

    for i, guide_time in enumerate(guide_times):
        channel = raw_channels[i]

        if channel < pick_ch_min or channel > pick_ch_max:
            continue

        search_mask = (
            (time_s >= guide_time - half_width_s)
            & (time_s <= guide_time + half_width_s)
        )

        if np.count_nonzero(search_mask) < 3:
            continue

        local_times = time_s[search_mask]
        local_envelope = real_envelope_normalized[
            i,
            search_mask,
        ]

        if not np.any(np.isfinite(local_envelope)):
            continue

        local_index = int(
            np.nanargmax(local_envelope)
        )

        ridge_times[i] = float(
            local_times[local_index]
        )

        ridge_amplitudes[i] = float(
            local_envelope[local_index]
        )

    if smooth_window > 3:
        min_periods = max(
            3,
            smooth_window // 4,
        )

        ridge_times = (
            pd.Series(ridge_times)
            .rolling(
                window=smooth_window,
                center=True,
                min_periods=min_periods,
            )
            .median()
            .to_numpy(dtype=np.float64)
        )

        ridge_amplitudes = (
            pd.Series(ridge_amplitudes)
            .rolling(
                window=smooth_window,
                center=True,
                min_periods=min_periods,
            )
            .median()
            .to_numpy(dtype=np.float64)
        )

    return (
        raw_channels,
        ridge_times,
        ridge_amplitudes,
    )


def estimate_apparent_velocity(
    *,
    channels: np.ndarray,
    times_s: np.ndarray,
    channel_spacing_m: float,
    ch_min: float,
    ch_max: float,
) -> dict:
    """
    Estimate apparent along-fibre velocity from a linear time/channel fit.

        v_app = channel_spacing / abs(dt / dchannel)
    """
    channels = np.asarray(
        channels,
        dtype=np.float64,
    )
    times_s = np.asarray(
        times_s,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(channels)
        & np.isfinite(times_s)
        & (channels >= ch_min)
        & (channels <= ch_max)
    )

    n_valid = int(
        np.count_nonzero(valid)
    )

    if n_valid < 10:
        return {
            "slope_s_per_channel": np.nan,
            "intercept_s": np.nan,
            "apparent_velocity_m_s": np.nan,
            "n": n_valid,
        }

    slope, intercept = np.polyfit(
        channels[valid],
        times_s[valid],
        deg=1,
    )

    slope = float(slope)
    intercept = float(intercept)

    if slope == 0.0 or not np.isfinite(slope):
        apparent_velocity = np.nan
    else:
        apparent_velocity = abs(
            channel_spacing_m / slope
        )

    return {
        "slope_s_per_channel": slope,
        "intercept_s": intercept,
        "apparent_velocity_m_s": float(apparent_velocity),
        "n": n_valid,
    }


def _relative_db(
    psd: np.ndarray,
    *,
    reference_peak: float | None = None,
    eps: float = 1e-300,
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
        np.maximum(psd, eps) / max(reference_peak, eps)
    )


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
    Compare real pre-event noise, real event signal, and synthetic spectral
    content before the common comparison bandpass is applied.
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

    f_real, real_signal_psd, real_signal_p10, real_signal_p90 = median_welch_psd(
        real_data_unfiltered,
        real_time_s,
        fs_hz=real_fs_hz,
        tmin_s=REAL_SIGNAL_TMIN_S,
        tmax_s=REAL_SIGNAL_TMAX_S,
        trace_mask=real_trace_mask,
        segment_length_s=PSD_SEGMENT_LENGTH_S,
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
        real_noise_psd = np.interp(f_real, f_noise, real_noise_psd)

    eps = 1e-300
    real_snr_db = 10.0 * np.log10(
        np.maximum(real_signal_psd, eps) / np.maximum(real_noise_psd, eps)
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax0, ax1 = axes

    # Spectrum-shape comparison. Real signal, real noise, and synthetic
    # are independently normalized because absolute 2D synthetic amplitudes
    # are not calibrated. Percentile bands use the corresponding median peak.
    real_signal_peak = float(np.nanmax(real_signal_psd))
    real_noise_peak = float(np.nanmax(real_noise_psd))
    synthetic_peak = float(np.nanmax(synthetic_psd))

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
        label=f"Common band {FMIN_COMPARE_HZ:g}–{FMAX_COMPARE_HZ:g} Hz",
    )

    ax0.set_ylabel("Median PSD relative to peak [dB]")
    ax0.set_ylim(-80.0, 5.0)
    ax0.grid(alpha=0.3)
    ax0.legend(fontsize=9)

    # Real SNR
    ax1.plot(f_real, real_snr_db, linewidth=1.4)
    ax1.axhline(0.0, linestyle="--", linewidth=1.0)
    ax1.axvspan(FMIN_COMPARE_HZ, FMAX_COMPARE_HZ, alpha=0.12)
    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel("Real signal / noise PSD [dB]")
    ax1.grid(alpha=0.3)

    max_f = min(
        PSD_PLOT_FMAX_HZ,
        float(np.nanmax(f_real)),
        float(np.nanmax(f_syn)),
    )
    ax1.set_xlim(0.0, max_f)

    fig.suptitle(
        "SAFOD real/synthetic frequency-content QC\n"
        "Curves are independently normalized; compare spectral shape, not absolute scale."
    )
    fig.tight_layout()
    fig.savefig(out_figure, dpi=220, bbox_inches="tight")
    plt.close(fig)

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
    ).to_csv(out_csv, index=False)

# ==============================================================================
# PLOTTING HELPERS
# ==============================================================================

def add_arrival_and_ridge_overlays(
    *,
    ax,
    arrival_channels: np.ndarray,
    p_arrivals_s: np.ndarray,
    s_arrivals_s: np.ndarray,
    ridge_channels: np.ndarray | None = None,
    ridge_times_s: np.ndarray | None = None,
    arrival_time_shift_s: float = 0.0,
    include_labels: bool = True,
) -> None:
    """
    Add predicted P/S curves and an optional observed ridge.
    """
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
        p_arrivals_s[valid_p] + arrival_time_shift_s,
        color="black",
        linestyle="--",
        linewidth=1.6,
        label="Predicted P" if include_labels else None,
        zorder=20,
    )

    ax.plot(
        arrival_channels[valid_s],
        s_arrivals_s[valid_s] + arrival_time_shift_s,
        color="black",
        linestyle=":",
        linewidth=1.9,
        label="Predicted S" if include_labels else None,
        zorder=20,
    )

    if ridge_channels is not None and ridge_times_s is not None:
        ridge_channels = np.asarray(
            ridge_channels,
            dtype=np.float64,
        )
        ridge_times_s = np.asarray(
            ridge_times_s,
            dtype=np.float64,
        )

        valid_ridge = (
            np.isfinite(ridge_channels)
            & np.isfinite(ridge_times_s)
        )

        ax.plot(
            ridge_channels[valid_ridge],
            ridge_times_s[valid_ridge],
            color="magenta",
            linestyle="-",
            linewidth=2.0,
            label="Observed ridge" if include_labels else None,
            zorder=25,
        )


def plot_real_with_arrivals_and_ridge(
    *,
    real_signed_normalized: np.ndarray,
    real_time_s: np.ndarray,
    real_channels: np.ndarray,
    arrival_channels: np.ndarray,
    p_arrivals_s: np.ndarray,
    s_arrivals_s: np.ndarray,
    ridge_channels: np.ndarray,
    ridge_times_s: np.ndarray,
    event_id: str,
    out_path: Path,
) -> None:
    """
    Real signed DAS gather with P/S arrivals and observed envelope ridge.
    """
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
        linestyle="--",
        label="Catalogue origin",
    )

    add_arrival_and_ridge_overlays(
        ax=ax,
        arrival_channels=arrival_channels,
        p_arrivals_s=p_arrivals_s,
        s_arrivals_s=s_arrivals_s,
        ridge_channels=ridge_channels,
        ridge_times_s=ridge_times_s,
        arrival_time_shift_s=0.0,
        include_labels=True,
    )

    fig.colorbar(
        image,
        ax=ax,
        label="trace-normalized amplitude",
    )

    ax.set_title(
        f"{event_id} real DAS: predicted arrivals and guided envelope ridge"
    )
    ax.set_xlabel("Raw channel number")
    ax.set_ylabel("Time from catalogue origin [s]")

    ax.set_xlim(
        float(real_channels.min()),
        float(real_channels.max()),
    )
    ax.set_ylim(
        float(real_time_s[-1]),
        float(real_time_s[0]),
    )

    ax.grid(alpha=0.25)

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

    plt.close(fig)


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
    ridge_channels: np.ndarray,
    ridge_times_s: np.ndarray,
    event_id: str,
    source_f0_hz: float,
    source_theta_deg: float,
    out_path: Path,
) -> None:
    """
    Signed real/synthetic DAS comparison.
    """
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
        linestyle="--",
        label="Catalogue origin",
    )

    add_arrival_and_ridge_overlays(
        ax=ax_real,
        arrival_channels=arrival_channels,
        p_arrivals_s=p_arrivals_s,
        s_arrivals_s=s_arrivals_s,
        ridge_channels=ridge_channels,
        ridge_times_s=ridge_times_s,
        arrival_time_shift_s=0.0,
        include_labels=True,
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

    ax_real.grid(alpha=0.25)

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
        linestyle="--",
    )

    add_arrival_and_ridge_overlays(
        ax=ax_syn,
        arrival_channels=arrival_channels,
        p_arrivals_s=p_arrivals_s,
        s_arrivals_s=s_arrivals_s,
        ridge_channels=None,
        ridge_times_s=None,
        arrival_time_shift_s=SYN_DISPLAY_TIME_SHIFT_S,
        include_labels=False,
    )

    ax_syn.set_title(
        "Synthetic DAS, down-going pass"
    )
    ax_syn.set_xlabel(
        "Approx. raw channel number"
    )

    ax_syn.set_xlim(
        float(synthetic_channels.min()),
        float(synthetic_channels.max()),
    )

    ax_syn.grid(alpha=0.25)

    fig.colorbar(
        synthetic_image,
        ax=axes.ravel().tolist(),
        label="trace-normalized amplitude",
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

    plt.close(fig)


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
    ridge_channels: np.ndarray,
    ridge_times_s: np.ndarray,
    event_id: str,
    source_f0_hz: float,
    source_theta_deg: float,
    out_path: Path,
) -> None:
    """
    Real/synthetic Hilbert-envelope comparison.
    """
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
        linestyle="--",
        label="Catalogue origin",
    )

    add_arrival_and_ridge_overlays(
        ax=ax_real,
        arrival_channels=arrival_channels,
        p_arrivals_s=p_arrivals_s,
        s_arrivals_s=s_arrivals_s,
        ridge_channels=ridge_channels,
        ridge_times_s=ridge_times_s,
        arrival_time_shift_s=0.0,
        include_labels=True,
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

    ax_real.grid(alpha=0.20)

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
        linestyle="--",
    )

    add_arrival_and_ridge_overlays(
        ax=ax_syn,
        arrival_channels=arrival_channels,
        p_arrivals_s=p_arrivals_s,
        s_arrivals_s=s_arrivals_s,
        ridge_channels=None,
        ridge_times_s=None,
        arrival_time_shift_s=SYN_DISPLAY_TIME_SHIFT_S,
        include_labels=False,
    )

    ax_syn.set_title(
        "Synthetic DAS envelope"
    )
    ax_syn.set_xlabel(
        "Approx. raw channel number"
    )

    ax_syn.set_xlim(
        float(synthetic_channels.min()),
        float(synthetic_channels.max()),
    )

    ax_syn.grid(alpha=0.20)

    fig.colorbar(
        synthetic_image,
        ax=axes.ravel().tolist(),
        label="trace-normalized envelope",
        shrink=0.85,
    )

    fig.suptitle(
        f"{event_id} real vs synthetic DAS envelopes\n"
        f"nominal band={FMIN_COMPARE_HZ:.0f}–{FMAX_COMPARE_HZ:.0f} Hz, "
        f"f0={source_f0_hz:.1f} Hz, "
        f"theta={source_theta_deg:.1f}°",
        y=0.98,
    )

    fig.savefig(
        out_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_s_residual(
    *,
    channels: np.ndarray,
    residual_s: np.ndarray,
    median_residual_s: float,
    residual_slope: float,
    residual_intercept: float,
    event_id: str,
    out_path: Path,
) -> None:
    """
    Plot guided-envelope-ridge minus predicted-S diagnostic residual.
    """
    channels = np.asarray(
        channels,
        dtype=np.float64,
    )
    residual_s = np.asarray(
        residual_s,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(channels)
        & np.isfinite(residual_s)
    )

    fig, ax = plt.subplots(
        figsize=(10, 4.5)
    )

    ax.plot(
        channels[valid],
        residual_s[valid],
        color="magenta",
        linewidth=0.8,
        alpha=0.6,
    )

    ax.scatter(
        channels[valid],
        residual_s[valid],
        s=7,
        color="magenta",
        alpha=0.7,
        label="Guided ridge − predicted S",
    )

    ax.axhline(
        median_residual_s,
        color="black",
        linestyle="--",
        linewidth=1.3,
        label=f"Median = {median_residual_s:.3f} s",
    )

    if (
        np.isfinite(residual_slope)
        and np.isfinite(residual_intercept)
    ):
        x_fit = np.linspace(
            float(np.nanmin(channels[valid])),
            float(np.nanmax(channels[valid])),
            200,
        )

        y_fit = (
            residual_slope * x_fit
            + residual_intercept
        )

        ax.plot(
            x_fit,
            y_fit,
            color="tab:blue",
            linewidth=1.4,
            label="Linear residual trend",
        )

    ax.set_xlabel(
        "Raw channel number"
    )
    ax.set_ylabel(
        "Guided ridge − predicted S [s]"
    )
    ax.set_title(
        f"{event_id}: dominant-envelope-ridge delay relative to straight-ray S"
    )

    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    if not REAL_PKG.exists():
        raise FileNotFoundError(
            f"Real-event package not found: {REAL_PKG}"
        )

    if not SYN_PKG.exists():
        raise FileNotFoundError(
            f"Synthetic package not found: {SYN_PKG}"
        )

    if not GEOM_CSV.exists():
        raise FileNotFoundError(
            f"Geometry CSV not found: {GEOM_CSV}"
        )

    real = np.load(
        REAL_PKG,
        allow_pickle=True,
    )

    synthetic = np.load(
        SYN_PKG,
        allow_pickle=True,
    )

    # --------------------------------------------------------------------------
    # Metadata
    # --------------------------------------------------------------------------
    event_id = str(
        get_scalar(
            real,
            "ev_id",
            "NC75336802",
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

    real_channel_spacing_m = float(
        get_scalar(
            real,
            "channel_spacing_m",
        )
    )

    # --------------------------------------------------------------------------
    # Real DAS: load unfiltered data and apply the common zero-phase filter
    # --------------------------------------------------------------------------
    if "das_data_unfiltered" not in real.files:
        raise RuntimeError(
            "Real-event package is from the old preprocessing workflow. "
            "Rerun: python -m scripts.safod.prepare_event"
        )

    real_data_unfiltered_full = np.asarray(
        real["das_data_unfiltered"], dtype=np.float64
    )
    real_time_full = np.asarray(real["t"], dtype=np.float64)
    real_channels = np.asarray(real["raw_channels"], dtype=np.float64)
    real_fs = float(get_scalar(real, "fs"))

    if real_data_unfiltered_full.shape != (
        real_channels.size,
        real_time_full.size,
    ):
        raise ValueError(
            "Real DAS shape does not match channel/time axes: "
            f"{real_data_unfiltered_full.shape} != "
            f"({real_channels.size}, {real_time_full.size})."
        )

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

    real_time_mask = (
        (real_time_full >= TMIN)
        & (real_time_full <= TMAX)
    )
    if np.count_nonzero(real_time_mask) < 2:
        raise RuntimeError(
            "Requested real-data display interval contains fewer than "
            "two samples."
        )

    real_envelope_full = compute_envelope(real_data_filtered_full)
    real_data = real_data_filtered_full[:, real_time_mask]
    real_time = real_time_full[real_time_mask]
    real_envelope = real_envelope_full[:, real_time_mask]
    real_signed_normalized = trace_normalize(real_data)
    real_envelope_normalized = trace_normalize(real_envelope)

    # --------------------------------------------------------------------------
    # Synthetic DAS: apply exactly the same common zero-phase filter
    # --------------------------------------------------------------------------
    synthetic_data_unfiltered_full = np.asarray(
        synthetic["das_data"], dtype=np.float64
    )
    synthetic_time_full = np.asarray(synthetic["t"], dtype=np.float64)

    if synthetic_data_unfiltered_full.shape[1] != synthetic_time_full.size:
        raise ValueError(
            "Synthetic DAS time dimension does not match its time axis: "
            f"{synthetic_data_unfiltered_full.shape[1]} != "
            f"{synthetic_time_full.size}."
        )

    synthetic_dt = float(
        get_scalar(
            synthetic,
            "dt",
            np.median(np.diff(synthetic_time_full)),
        )
    )
    if synthetic_dt <= 0.0:
        raise ValueError(f"Invalid synthetic dt: {synthetic_dt}")
    synthetic_fs = 1.0 / synthetic_dt

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

    synthetic_envelope_full = compute_envelope(
        synthetic_data_filtered_full
    )
    synthetic_time_shifted_full = (
        synthetic_time_full + SYN_DISPLAY_TIME_SHIFT_S
    )
    synthetic_time_mask = (
        (synthetic_time_shifted_full >= TMIN)
        & (synthetic_time_shifted_full <= TMAX)
    )
    if np.count_nonzero(synthetic_time_mask) < 2:
        raise RuntimeError(
            "Requested synthetic display interval contains fewer than "
            "two samples."
        )

    synthetic_data = synthetic_data_filtered_full[:, synthetic_time_mask]
    synthetic_envelope = synthetic_envelope_full[:, synthetic_time_mask]
    synthetic_time_shifted = synthetic_time_shifted_full[
        synthetic_time_mask
    ]

    # --------------------------------------------------------------------------
    # Map synthetic receivers to raw downleg channel numbers
    # --------------------------------------------------------------------------
    receiver_x = np.asarray(
        synthetic["receiver_x"],
        dtype=np.float64,
    )

    receiver_z = np.asarray(
        synthetic["receiver_z"],
        dtype=np.float64,
    )

    das_channel_indices = np.asarray(
        synthetic["das_channel_indices"],
        dtype=np.int64,
    )

    receiver_raw_channels = nearest_geometry_channel_for_receivers(
        receiver_x=receiver_x,
        receiver_z=receiver_z,
        geom_csv=GEOM_CSV,
    )

    synthetic_channels = receiver_raw_channels[
        das_channel_indices
    ]

    predicted_p = np.asarray(
        synthetic["arrival_p_das"],
        dtype=np.float64,
    )

    predicted_s = np.asarray(
        synthetic["arrival_swave_das"],
        dtype=np.float64,
    )

    expected_traces = synthetic_data.shape[0]

    for name, array in [
        ("synthetic_channels", synthetic_channels),
        ("predicted_p", predicted_p),
        ("predicted_s", predicted_s),
        ("synthetic_envelope", synthetic_envelope),
    ]:
        if array.shape[0] != expected_traces:
            raise ValueError(
                f"{name} first dimension {array.shape[0]} does not match "
                f"synthetic DAS traces {expected_traces}."
            )

    # --------------------------------------------------------------------------
    # Frequency-content QC before the common comparison bandpass
    # --------------------------------------------------------------------------
    frequency_figure_path = (
        OUT_DIR / "00_frequency_content_qc.png"
    )
    frequency_csv_path = (
        OUT_DIR / "frequency_content_qc.csv"
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

    # Sort waveform, envelope and arrival curves using exactly the same order.
    (
        synthetic_channels,
        synthetic_data,
        synthetic_envelope,
        predicted_p,
        predicted_s,
    ) = sort_by_channel(
        synthetic_channels,
        synthetic_data,
        synthetic_envelope,
        predicted_p,
        predicted_s,
    )

    synthetic_signed_normalized = trace_normalize(
        synthetic_data
    )

    synthetic_envelope_normalized = trace_normalize(
        synthetic_envelope
    )

    # Build a clean, strictly increasing raw-channel axis for P/S interpolation.
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
            "Too few valid mapped arrival channels remain after filtering."
        )

    # --------------------------------------------------------------------------
    # Pick the guided dominant-envelope ridge
    # --------------------------------------------------------------------------
    (
        ridge_channels,
        ridge_times,
        ridge_amplitudes,
    ) = pick_real_ridge_guided(
        real_envelope_normalized=real_envelope_normalized,
        time_s=real_time,
        raw_channels=real_channels,
        anchor_ch_t=RIDGE_ANCHORS,
        half_width_s=RIDGE_SEARCH_HALF_WIDTH_S,
        smooth_window=RIDGE_SMOOTH_WINDOW,
        pick_ch_min=RIDGE_PICK_CH_MIN,
        pick_ch_max=RIDGE_PICK_CH_MAX,
    )

    apparent_velocity_result = estimate_apparent_velocity(
        channels=ridge_channels,
        times_s=ridge_times,
        channel_spacing_m=real_channel_spacing_m,
        ch_min=FIT_CH_MIN,
        ch_max=FIT_CH_MAX,
    )

    # --------------------------------------------------------------------------
    # Interpolate predicted arrivals to observed ridge channels
    # --------------------------------------------------------------------------
    predicted_p_on_ridge = np.interp(
        ridge_channels,
        arrival_channels,
        predicted_p_unique,
        left=np.nan,
        right=np.nan,
    )

    predicted_s_on_ridge = np.interp(
        ridge_channels,
        arrival_channels,
        predicted_s_unique,
        left=np.nan,
        right=np.nan,
    )

    s_residual = (
        ridge_times
        - predicted_s_on_ridge
    )

    residual_mask = (
        np.isfinite(ridge_channels)
        & np.isfinite(ridge_times)
        & np.isfinite(predicted_s_on_ridge)
        & np.isfinite(s_residual)
        & (ridge_channels >= FIT_CH_MIN)
        & (ridge_channels <= FIT_CH_MAX)
    )

    n_residual = int(
        np.count_nonzero(residual_mask)
    )

    if n_residual < 10:
        raise RuntimeError(
            f"Too few valid ridge/S residual samples: {n_residual}"
        )

    residual_values = s_residual[
        residual_mask
    ]

    residual_channels = ridge_channels[
        residual_mask
    ]

    residual_median = float(
        np.nanmedian(residual_values)
    )

    residual_mean = float(
        np.nanmean(residual_values)
    )

    residual_std = float(
        np.nanstd(residual_values)
    )

    residual_mad = float(
        np.nanmedian(
            np.abs(
                residual_values
                - residual_median
            )
        )
    )

    residual_slope, residual_intercept = np.polyfit(
        residual_channels,
        residual_values,
        deg=1,
    )

    residual_slope = float(
        residual_slope
    )

    residual_intercept = float(
        residual_intercept
    )

    # --------------------------------------------------------------------------
    # Print QC summary
    # --------------------------------------------------------------------------
    print("\nReal/Synthetic comparison QC")
    print("----------------------------")
    print(f"event id                 : {event_id}")
    print(f"source f0                : {source_f0_hz:.2f} Hz")
    print(f"source theta             : {source_theta_deg:.2f} deg")
    print(
        "common comparison band   : "
        f"{FMIN_COMPARE_HZ:.1f} to {FMAX_COMPARE_HZ:.1f} Hz, "
        "zero phase"
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

    print("\nGuided dominant-envelope ridge")
    print("------------------------------")

    print(
        "fit channel interval     : "
        f"{FIT_CH_MIN:.0f} to {FIT_CH_MAX:.0f}"
    )

    print(
        "ridge slope              : "
        f"{apparent_velocity_result['slope_s_per_channel']:.6e} "
        "s/channel"
    )

    print(
        "apparent along-fibre vel.: "
        f"{apparent_velocity_result['apparent_velocity_m_s']:.1f} m/s"
    )

    print(
        "fit samples              : "
        f"{apparent_velocity_result['n']}"
    )

    print("\nGuided ridge minus predicted S")
    print("--------------------------------")

    print(
        f"median residual          : {residual_median:.4f} s"
    )
    print(
        f"mean residual            : {residual_mean:.4f} s"
    )
    print(
        f"standard deviation       : {residual_std:.4f} s"
    )
    print(
        f"median absolute deviation: {residual_mad:.4f} s"
    )
    print(
        "residual trend           : "
        f"{residual_slope:.6e} s/channel"
    )

    print(
        "\nInterpretation note: this residual compares a guided dominant-envelope "
        "ridge with a straight-ray S first-arrival estimate. It is a QC "
        "diagnostic, not a formal S-wave travel-time residual."
    )

    # --------------------------------------------------------------------------
    # Save diagnostic tables
    # --------------------------------------------------------------------------
    diagnostic_table = pd.DataFrame(
        {
            "raw_channel": ridge_channels,
            "guided_envelope_ridge_time_s": ridge_times,
            "guided_envelope_ridge_amplitude": ridge_amplitudes,
            "predicted_P_time_s": predicted_p_on_ridge,
            "predicted_S_time_s": predicted_s_on_ridge,
            "ridge_minus_P_s": (
                ridge_times
                - predicted_p_on_ridge
            ),
            "ridge_minus_S_s": s_residual,
        }
    )

    diagnostic_csv = (
        OUT_DIR
        / "observed_ridge_and_arrival_residuals.csv"
    )

    diagnostic_table.to_csv(
        diagnostic_csv,
        index=False,
    )

    summary_table = pd.DataFrame(
        [
            {
                "event_id": event_id,
                "source_f0_hz": source_f0_hz,
                "source_theta_deg": source_theta_deg,
                "synthetic_display_shift_s": SYN_DISPLAY_TIME_SHIFT_S,
                "common_fmin_hz": FMIN_COMPARE_HZ,
                "common_fmax_hz": FMAX_COMPARE_HZ,
                "filter_order": FILTER_ORDER,
                "filter_taper_frac": FILTER_TAPER_FRAC,
                "filter_phase": "zero_phase_sosfiltfilt_both",
                "ridge_type": "guided_dominant_envelope_ridge",
                "ridge_fit_ch_min": FIT_CH_MIN,
                "ridge_fit_ch_max": FIT_CH_MAX,
                "ridge_slope_s_per_channel": (
                    apparent_velocity_result[
                        "slope_s_per_channel"
                    ]
                ),
                "ridge_apparent_velocity_m_s": (
                    apparent_velocity_result[
                        "apparent_velocity_m_s"
                    ]
                ),
                "ridge_minus_S_median_s": residual_median,
                "ridge_minus_S_mean_s": residual_mean,
                "ridge_minus_S_std_s": residual_std,
                "ridge_minus_S_mad_s": residual_mad,
                "ridge_minus_S_slope_s_per_channel": residual_slope,
                "residual_interpretation": (
                    "QC diagnostic only; dominant-envelope ridge minus "
                    "straight-ray S first arrival"
                ),
            }
        ]
    )

    summary_csv = (
        OUT_DIR
        / "comparison_summary.csv"
    )

    summary_table.to_csv(
        summary_csv,
        index=False,
    )

    # --------------------------------------------------------------------------
    # Figures
    # --------------------------------------------------------------------------
    real_overlay_path = (
        OUT_DIR
        / "01_real_with_arrivals_and_observed_ridge.png"
    )

    signed_comparison_path = (
        OUT_DIR
        / "02_real_vs_synthetic_signed.png"
    )

    envelope_comparison_path = (
        OUT_DIR
        / "03_real_vs_synthetic_envelopes.png"
    )

    residual_path = (
        OUT_DIR
        / "04_observed_ridge_minus_predicted_S.png"
    )

    plot_real_with_arrivals_and_ridge(
        real_signed_normalized=real_signed_normalized,
        real_time_s=real_time,
        real_channels=real_channels,
        arrival_channels=arrival_channels,
        p_arrivals_s=predicted_p_unique,
        s_arrivals_s=predicted_s_unique,
        ridge_channels=ridge_channels,
        ridge_times_s=ridge_times,
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
        ridge_channels=ridge_channels,
        ridge_times_s=ridge_times,
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
        ridge_channels=ridge_channels,
        ridge_times_s=ridge_times,
        event_id=event_id,
        source_f0_hz=source_f0_hz,
        source_theta_deg=source_theta_deg,
        out_path=envelope_comparison_path,
    )

    plot_s_residual(
        channels=residual_channels,
        residual_s=residual_values,
        median_residual_s=residual_median,
        residual_slope=residual_slope,
        residual_intercept=residual_intercept,
        event_id=event_id,
        out_path=residual_path,
    )

    print("\nSaved outputs")
    print("-------------")
    print(frequency_figure_path)
    print(frequency_csv_path)
    print(real_overlay_path)
    print(signed_comparison_path)
    print(envelope_comparison_path)
    print(residual_path)
    print(diagnostic_csv)
    print(summary_csv)


if __name__ == "__main__":
    main()