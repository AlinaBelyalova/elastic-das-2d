from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.signal import hilbert
    from scipy.spatial import cKDTree
except ImportError as exc:
    raise ImportError("compare_event.py requires scipy.") from exc

from scripts.safod.settings import (
    COMMON_FMAX_HZ,
    COMMON_FMIN_HZ,
    DEFAULT_THETA_DEG,
    FILTER_ORDER,
    FILTER_TAPER_FRAC,
    GEOMETRY_CSV,
    REAL_EVENT_PACKAGE,
    SAFOD_SURFACE_ELEVATION_M,
    comparison_dir_for_theta,
    forward_package_for_theta,
    forward_run_tag,
)
from src.common.signal_processing import bandpass_traces, median_welch_psd
from src.safod.models import AVAILABLE_INITIAL_MODELS
from src.safod.travel_times import (
    add_travel_time_arguments, prediction_options, regional_model_from_args,
    receivers_from_channels,
)


# =============================================================================
# SETTINGS
# =============================================================================

INITIAL_MODEL_CHOICES = AVAILABLE_INITIAL_MODELS
DEFAULT_INITIAL_MODEL = "hybrid_zhang2009_boness2006_bill_logs"

TMIN_REAL_S = -0.30
TMAX_REAL_S = 2.00

TMIN_SYN_DISPLAY_S = -0.20
TMAX_SYN_DISPLAY_S = 2.00

FMIN_COMPARE_HZ = float(COMMON_FMIN_HZ)
FMAX_COMPARE_HZ = float(COMMON_FMAX_HZ)

TRACE_NORMALIZATION_PERCENTILE = 99.0
VMAX_SIGNED = 1.0
VMAX_ENVELOPE = 1.0

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

PREDICTED_ARRIVAL_COLOR = "k" ##00843D"
PREDICTED_ARRIVAL_COLOR_LIGHT = "#32CD32"
PREDICTED_P_LINEWIDTH = 2.7
PREDICTED_S_LINEWIDTH = 2.5


# =============================================================================
# HELPERS
# =============================================================================

def get_scalar(npz, name: str, default=np.nan):
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
    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError(f"Expected 2-D gather, got {data.shape}.")

    scale = np.percentile(
        np.abs(data),
        percentile,
        axis=1,
        keepdims=True,
    )
    scale = np.maximum(scale, eps)

    return data / scale


def compute_envelope(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError(f"Expected 2-D gather, got {data.shape}.")

    return np.abs(hilbert(data, axis=1))


def sort_by_channel(channels: np.ndarray, *arrays: np.ndarray):
    channels = np.asarray(channels, dtype=np.float64)
    order = np.argsort(channels)

    out = [channels[order]]

    for array in arrays:
        array = np.asarray(array)

        if array.shape[0] != channels.size:
            raise ValueError(
                "Array first dimension does not match channel axis: "
                f"{array.shape[0]} != {channels.size}."
            )

        out.append(array[order, ...])

    return tuple(out)


def collapse_duplicate_channels(
    channels: np.ndarray,
    *arrays: np.ndarray,
):
    channels = np.asarray(channels, dtype=np.float64)
    unique = np.unique(channels)
    out = [unique]

    for array in arrays:
        array = np.asarray(array, dtype=np.float64)

        if array.ndim != 1 or array.size != channels.size:
            raise ValueError(
                "collapse_duplicate_channels expects 1-D arrays matching "
                "the channel axis."
            )

        collapsed = np.full(unique.shape, np.nan, dtype=np.float64)

        for i, ch in enumerate(unique):
            values = array[channels == ch]
            if np.any(np.isfinite(values)):
                collapsed[i] = float(np.nanmedian(values))

        out.append(collapsed)

    return tuple(out)


def nearest_geometry_channel_for_receivers(
    receiver_x: np.ndarray,
    receiver_z: np.ndarray,
    geom_csv: Path,
) -> np.ndarray:
    """
    Legacy fallback only. New forward packages should save das_raw_channels.
    """
    geom_csv = Path(geom_csv)

    if not geom_csv.exists():
        raise FileNotFoundError(f"Geometry CSV not found: {geom_csv}")

    geom = pd.read_csv(geom_csv)

    for column in ("Channel", "X_2D_m", "Z_2D_m"):
        if column not in geom.columns:
            raise ValueError(f"Missing {column!r} in {geom_csv}")

    channels = pd.to_numeric(
        geom["Channel"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    gx = pd.to_numeric(
        geom["X_2D_m"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    gz = pd.to_numeric(
        geom["Z_2D_m"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    valid = np.isfinite(channels) & np.isfinite(gx) & np.isfinite(gz)

    channels = channels[valid]
    gx = gx[valid]
    gz = gz[valid]

    tree = cKDTree(np.column_stack([gx, gz]))

    _, nearest = tree.query(
        np.column_stack(
            [
                np.asarray(receiver_x, dtype=np.float64),
                np.asarray(receiver_z, dtype=np.float64),
            ]
        )
    )

    return channels[nearest]


# =============================================================================
# MODEL-AWARE PATHS
# =============================================================================

def synthetic_package_for_model(
    theta_deg: float,
    initial_model: str,
) -> Path:
    """
    settings.forward_package_for_theta() still resolves the common theta folder.
    Insert the model-name directory below that folder.
    """
    base = Path(forward_package_for_theta(theta_deg))

    return (
        base.parent
        / initial_model
        / base.name
    )


def comparison_dir_for_model(
    theta_deg: float,
    initial_model: str,
) -> Path:
    return (
        Path(comparison_dir_for_theta(theta_deg))
        / initial_model
    )


def infer_package_model_name(synthetic) -> str:
    if "initial_model_name" in synthetic.files:
        return str(
            get_scalar(
                synthetic,
                "initial_model_name",
                "",
            )
        ).strip()

    model_type = str(
        get_scalar(
            synthetic,
            "model_type",
            "",
        )
    ).lower()

    if "hybrid" in model_type:
        return "hybrid_zhang2009_bill_logs"

    if "zhang2009" in model_type or "zhang" in model_type:
        return "zhang2009"

    if (
        "bill_logs" in model_type
        or "ellsworth" in model_type
        or "digitized" in model_type
    ):
        return "bill_logs"

    if "smooth" in model_type:
        return "smooth_prior"

    return ""
# =============================================================================
# FREQUENCY QC
# =============================================================================

def _relative_db(
    psd: np.ndarray,
    reference_peak: float | None = None,
    eps: float = 1.0e-300,
) -> np.ndarray:
    psd = np.asarray(psd, dtype=np.float64)

    if reference_peak is None:
        reference_peak = float(np.nanmax(psd))

    if not np.isfinite(reference_peak) or reference_peak <= 0.0:
        raise ValueError(f"Invalid PSD reference peak: {reference_peak}")

    return 10.0 * np.log10(
        np.maximum(psd, eps)
        / max(reference_peak, eps)
    )


def plot_frequency_qc(
    *,
    real_data_unfiltered: np.ndarray,
    real_time_s: np.ndarray,
    real_channels: np.ndarray,
    real_fs_hz: float,
    synthetic_data_unfiltered: np.ndarray,
    synthetic_time_physical_s: np.ndarray,
    synthetic_channels: np.ndarray,
    synthetic_fs_hz: float,
    initial_model: str,
    out_figure: Path,
    out_csv: Path,
) -> None:
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

    (
        f_real,
        real_signal_psd,
        real_signal_p10,
        real_signal_p90,
    ) = median_welch_psd(
        real_data_unfiltered,
        real_time_s,
        fs_hz=real_fs_hz,
        tmin_s=REAL_SIGNAL_TMIN_S,
        tmax_s=REAL_SIGNAL_TMAX_S,
        trace_mask=real_trace_mask,
        segment_length_s=PSD_SEGMENT_LENGTH_S,
    )

    (
        f_noise,
        real_noise_psd,
        _,
        _,
    ) = median_welch_psd(
        real_data_unfiltered,
        real_time_s,
        fs_hz=real_fs_hz,
        tmin_s=REAL_NOISE_TMIN_S,
        tmax_s=REAL_NOISE_TMAX_S,
        trace_mask=real_trace_mask,
        segment_length_s=PSD_SEGMENT_LENGTH_S,
    )

    (
        f_syn,
        synthetic_psd,
        synthetic_p10,
        synthetic_p90,
    ) = median_welch_psd(
        synthetic_data_unfiltered,
        synthetic_time_physical_s,
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

    real_signal_peak = float(np.nanmax(real_signal_psd))
    real_noise_peak = float(np.nanmax(real_noise_psd))
    synthetic_peak = float(np.nanmax(synthetic_psd))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
    )

    ax0, ax1 = axes

    ax0.plot(
        f_real,
        _relative_db(real_signal_psd, real_signal_peak),
        label="Real event window",
        linewidth=PREDICTED_P_LINEWIDTH,
    )
    ax0.fill_between(
        f_real,
        _relative_db(real_signal_p10, real_signal_peak),
        _relative_db(real_signal_p90, real_signal_peak),
        alpha=0.15,
    )

    ax0.plot(
        f_real,
        _relative_db(real_noise_psd, real_noise_peak),
        label="Real pre-event noise",
        linewidth=1.2,
    )

    ax0.plot(
        f_syn,
        _relative_db(synthetic_psd, synthetic_peak),
        label=f"Synthetic: {initial_model}",
        linewidth=PREDICTED_P_LINEWIDTH,
    )
    ax0.fill_between(
        f_syn,
        _relative_db(synthetic_p10, synthetic_peak),
        _relative_db(synthetic_p90, synthetic_peak),
        alpha=0.12,
    )

    ax0.axvspan(
        FMIN_COMPARE_HZ,
        FMAX_COMPARE_HZ,
        alpha=0.12,
        label=f"Common band {FMIN_COMPARE_HZ:g}-{FMAX_COMPARE_HZ:g} Hz",
    )

    ax0.set_ylabel("Median PSD relative to peak [dB]")
    ax0.set_ylim(-80.0, 5.0)
    ax0.grid(alpha=0.3)
    ax0.legend(fontsize=9)

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
        "Curves are independently normalized; compare spectral shape, "
        "not absolute scale."
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


# =============================================================================
# PLOTTING
# =============================================================================

def comparison_view_limits(
    *, real_channels, synthetic_channels, real_time_s, synthetic_time_display_s,
):
    """Use the same supported channel/time window on both panels, from t=0.

    Cropping the viewport preserves sample coordinates. In particular, do not
    stretch the synthetic image to the real extent or shift its time axis again.
    """
    bounds = []
    for name, values in (
        ("real channels", real_channels),
        ("synthetic channels", synthetic_channels),
        ("real time", real_time_s),
        ("synthetic display time", synthetic_time_display_s),
    ):
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain at least two finite coordinates.")
        bounds.append((float(values.min()), float(values.max())))

    real_x, syn_x, real_t, syn_t = bounds
    x_min, x_max = max(real_x[0], syn_x[0]), min(real_x[1], syn_x[1])
    if x_min >= x_max:
        raise ValueError("Real and synthetic channel ranges do not overlap.")
    if real_t[0] > 0.0 or syn_t[0] > 0.0:
        raise ValueError("Both displayed waveform time ranges must include catalog origin (0 s).")
    t_max = min(real_t[1], syn_t[1], TMAX_REAL_S, TMAX_SYN_DISPLAY_S)
    if t_max <= 0.0:
        raise ValueError("Real and synthetic waveforms have no common post-origin time interval.")
    return (x_min, x_max), (t_max, 0.0)


def add_arrival_overlays(
    *,
    ax,
    arrival_channels: np.ndarray,
    p_arrivals_s: np.ndarray,
    s_arrivals_s: np.ndarray,
    time_shift_s: float = 0.0,
    light: bool = False,
    include_labels: bool = True,
    model_label: str = "3-D eikonal",
) -> None:
    colour = (
        PREDICTED_ARRIVAL_COLOR_LIGHT
        if light
        else PREDICTED_ARRIVAL_COLOR
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
        p_arrivals_s[valid_p] + time_shift_s,
        color=colour,
        linestyle=":",
        linewidth=PREDICTED_P_LINEWIDTH,
        label="Predicted P" if include_labels else None,
        zorder=20,
    )

    ax.plot(
        arrival_channels[valid_s],
        s_arrivals_s[valid_s] + time_shift_s,
        color=colour,
        linestyle="--",
        linewidth=PREDICTED_S_LINEWIDTH,
        label="Predicted S" if include_labels else None,
        zorder=20,
    )


def plot_real_with_arrivals(
    *,
    real_signed_normalized,
    real_time_s,
    real_channels,
    event_id,
    out_path,
    shared_arrivals,
):
    fig, ax = plt.subplots(figsize=(11, 8))

    im = ax.imshow(
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
        label="Catalog origin",
    )

    add_arrival_overlays(
        ax=ax,
        **shared_arrivals,
        include_labels=True,
    )

    # Model guides outside the recorded window must not expand the image axes.
    ax.set_xlim(float(real_channels.min()), float(real_channels.max()))
    ax.set_ylim(float(real_time_s[-1]), float(real_time_s[0]))

    fig.colorbar(
        im,
        ax=ax,
        label="Trace-normalized amplitude",
    )

    ax.set_title(
        f"{event_id} real DAS with predicted arrivals\n"
        f"{shared_arrivals['model_label']}"
    )
    ax.set_xlabel("Physical reference channel")
    ax.set_ylabel("Time from catalog origin [s]")
    ax.legend(
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_signed_side_by_side(
    *,
    real_signed_normalized,
    real_time_s,
    real_channels,
    synthetic_signed_normalized,
    synthetic_time_display_s,
    synthetic_channels,
    event_id,
    source_f0_hz,
    source_theta_deg,
    synthetic_display_shift_s,
    initial_model,
    out_path,
    shared_arrivals,
):
    channel_limits, time_limits = comparison_view_limits(
        real_channels=real_channels, synthetic_channels=synthetic_channels,
        real_time_s=real_time_s, synthetic_time_display_s=synthetic_time_display_s,
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(18, 7),
        sharex=True,
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
        label="Catalog origin",
    )

    add_arrival_overlays(
        ax=ax_real,
        **shared_arrivals,
        include_labels=True,
    )

    ax_real.set_title("Real DAS")
    ax_real.set_xlabel("Physical reference channel")
    ax_real.set_ylabel("Time from origin [s]")
    ax_real.legend(
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
    )

    syn_im = ax_syn.imshow(
        synthetic_signed_normalized.T,
        extent=[
            float(synthetic_channels.min()),
            float(synthetic_channels.max()),
            float(synthetic_time_display_s[-1]),
            float(synthetic_time_display_s[0]),
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

    add_arrival_overlays(
        ax=ax_syn,
        **shared_arrivals,
        # T is relative to source origin. The displayed waveform has already
        # been shifted by -STF.t0; applying that shift to T again is incorrect.
        include_labels=True,
    )

    ax_syn.set_title(f"Synthetic DAS: {initial_model}")
    ax_syn.legend(loc="upper right", fontsize=8)
    ax_syn.set_xlabel("Physical reference channel")

    # Set limits after images and guides: shared 3-D curves can extend beyond
    # the synthetic gauge centres and otherwise trigger autoscale margins.
    for ax in axes:
        ax.set_xlim(*channel_limits)
        ax.set_ylim(*time_limits)

    fig.colorbar(
        syn_im,
        ax=axes.ravel().tolist(),
        label="Trace-normalized amplitude",
        shrink=0.85,
    )

    fig.suptitle(
        f"{event_id} real vs synthetic signed DAS",
        y=0.98,
    )

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_envelope_side_by_side(
    *,
    real_envelope_normalized,
    real_time_s,
    real_channels,
    synthetic_envelope_normalized,
    synthetic_time_display_s,
    synthetic_channels,
    event_id,
    source_f0_hz,
    source_theta_deg,
    synthetic_display_shift_s,
    initial_model,
    out_path,
    shared_arrivals,
):
    channel_limits, time_limits = comparison_view_limits(
        real_channels=real_channels, synthetic_channels=synthetic_channels,
        real_time_s=real_time_s, synthetic_time_display_s=synthetic_time_display_s,
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(18, 7),
        sharex=True,
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
        label="Catalog origin",
    )

    add_arrival_overlays(
        ax=ax_real,
        **shared_arrivals,
        light=True,
        include_labels=True,
    )

    ax_real.set_title("Real DAS envelope")
    ax_real.set_xlabel("Physical reference channel")
    ax_real.set_ylabel("Time from origin [s]")
    ax_real.legend(
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
    )

    syn_im = ax_syn.imshow(
        synthetic_envelope_normalized.T,
        extent=[
            float(synthetic_channels.min()),
            float(synthetic_channels.max()),
            float(synthetic_time_display_s[-1]),
            float(synthetic_time_display_s[0]),
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

    add_arrival_overlays(
        ax=ax_syn,
        **shared_arrivals,
        light=True,
        include_labels=True,
    )

    ax_syn.set_title(f"Synthetic envelope: {initial_model}")
    ax_syn.legend(loc="upper right", fontsize=8)
    ax_syn.set_xlabel("Physical reference channel")

    for ax in axes:
        ax.set_xlim(*channel_limits)
        ax.set_ylim(*time_limits)

    fig.colorbar(
        syn_im,
        ax=axes.ravel().tolist(),
        label="Trace-normalized envelope",
        shrink=0.85,
    )

    fig.suptitle(
        f"{event_id} real vs synthetic DAS envelopes",
        y=0.98,
    )

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================

def predict_comparison_arrivals(real, args, out_dir):
    """Compute one origin-relative 3-D reference for both comparison panels."""
    channels = np.asarray(real["raw_channels"], dtype=float)
    lat, lon, tvd = receivers_from_channels(GEOMETRY_CSV, channels)
    model3 = regional_model_from_args(args)
    p3, s3, metadata3 = model3.predict(
        source_lat=float(get_scalar(real, "ev_lat")),
        source_lon=float(get_scalar(real, "ev_lon")),
        source_depth_km=float(get_scalar(real, "ev_depth_km")),
        receiver_lat=lat, receiver_lon=lon, receiver_tvd_m=tvd,
        surface_elevation_m=SAFOD_SURFACE_ELEVATION_M,
        **prediction_options(args),
    )
    p3, s3 = np.asarray(p3, dtype=float), np.asarray(s3, dtype=float)
    if p3.shape != channels.shape or s3.shape != channels.shape:
        raise ValueError("3-D predictions do not match the physical reference channels.")
    valid = np.isfinite(channels) & np.isfinite(p3) & np.isfinite(s3)
    if np.count_nonzero(valid) < 2:
        raise RuntimeError("Too few valid 3-D arrival channels.")
    metadata3 = dict(metadata3, time_reference="catalog origin; no STF shift")
    order = np.argsort(channels)
    shared_arrivals = dict(
        arrival_channels=channels[order], p_arrivals_s=p3[order],
        s_arrivals_s=s3[order], model_label=metadata3["model_label"] + " eikonal",
    )
    receiver_x, receiver_y = model3.coordinates.xy_m(lat, lon)
    # Retain the existing Zhang-coordinate CSV columns with their actual
    # meaning; Lin EN coordinates get separate, explicitly named columns.
    if model3.primary is not None:
        zhang_x, zhang_y = model3.primary.coordinates.xy_m(lat, lon)
    else:
        zhang_x = zhang_y = np.full(channels.shape, np.nan)
    pd.DataFrame(dict(channel=channels, latitude=lat, longitude=lon, tvd_m=tvd,
                      zhang_x_m=zhang_x, zhang_y_m=zhang_y,
                      receiver_model_x_m=receiver_x, receiver_model_y_m=receiver_y,
                      depth_msl_m=tvd-SAFOD_SURFACE_ELEVATION_M,
                      p_time_s=p3, s_time_s=s3)).to_csv(
        out_dir / "predicted_arrivals_real_3d.csv", index=False
    )
    # Keep the established real/synthetic metadata keys, now describing the
    # very same reference. The elastic waveform still uses its saved 2-D model.
    provenance = dict(
        real=metadata3, synthetic=metadata3, observed_picks=False,
        shared_reference=True, reference_csv="predicted_arrivals_real_3d.csv",
        overlay_panels=["real", "synthetic"],
        synthetic_waveform_model=args.initial_model, synthetic_waveform_dimension=2,
    )
    (out_dir / "predicted_arrivals_metadata.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(f"Shared real/synthetic guides: {metadata3['model_label']} eikonal; "
          "synthetic waveform: saved 2-D elastic model")
    return shared_arrivals, provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare prepared real SAFOD DAS with one model-aware "
            "synthetic forward run."
        )
    )

    parser.add_argument(
        "--theta-deg",
        type=float,
        default=DEFAULT_THETA_DEG,
        help=f"Effective 2-D source orientation. Default: {DEFAULT_THETA_DEG:.1f}.",
    )

    parser.add_argument(
        "--initial-model",
        choices=INITIAL_MODEL_CHOICES,
        default=DEFAULT_INITIAL_MODEL,
        help=f"Initial velocity model. Default: {DEFAULT_INITIAL_MODEL}.",
    )

    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--arrivals-only", action="store_true",
                        help="Save shared 3-D predictions for real and synthetic panels without filtering waveforms.")
    add_travel_time_arguments(parser)
    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    if not 0.0 <= args.theta_deg < 90.0:
        raise ValueError("--theta-deg must satisfy 0 <= theta < 90.")

    requested_run_tag = forward_run_tag(args.theta_deg)

    real_pkg = Path(REAL_EVENT_PACKAGE)

    syn_pkg = synthetic_package_for_model(
        args.theta_deg,
        args.initial_model,
    )

    out_dir = args.output_dir or comparison_dir_for_model(
        args.theta_deg,
        args.initial_model,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nComparison inputs")
    print("=================")
    print(f"requested theta : {args.theta_deg:.1f} deg")
    print(f"initial model   : {args.initial_model}")
    print(f"run tag         : {requested_run_tag}")
    print(f"real package    : {real_pkg}")
    print(f"synthetic pkg   : {syn_pkg}")
    print(f"output dir      : {out_dir}")

    if not real_pkg.exists():
        raise FileNotFoundError(f"Real package not found: {real_pkg}")

    if not syn_pkg.exists():
        raise FileNotFoundError(
            "Synthetic package not found:\n"
            f"    {syn_pkg}\n"
            "Run the matching forward model first."
        )

    with np.load(real_pkg, allow_pickle=True) as real, np.load(
        syn_pkg,
        allow_pickle=True,
    ) as synthetic:

        # ---------------------------------------------------------------------
        # Identity
        # ---------------------------------------------------------------------
        event_id = str(get_scalar(real, "ev_id", "unknown"))
        synthetic_event_id = str(
            get_scalar(synthetic, "event_id", "")
        )

        if synthetic_event_id and synthetic_event_id != event_id:
            raise ValueError(
                "Real/synthetic event mismatch: "
                f"{event_id} != {synthetic_event_id}."
            )

        source_f0_hz = float(
            get_scalar(synthetic, "source_f0_hz", np.nan)
        )

        source_theta_deg = float(
            get_scalar(synthetic, "source_theta_deg", np.nan)
        )

        if not np.isfinite(source_f0_hz) or source_f0_hz <= 0.0:
            raise ValueError(
                f"Invalid source_f0_hz: {source_f0_hz}."
            )

        source_time_function_t0_s = float(
            get_scalar(
                synthetic,
                "source_time_function_t0_s",
                np.nan,
            )
        )

        if (
            not np.isfinite(source_time_function_t0_s)
            or source_time_function_t0_s < 0.0
        ):
            raise RuntimeError(
                "Synthetic package is missing a valid "
                "source_time_function_t0_s. Rerun "
                "scripts.safod.run_forward after saving source.stf.t0 "
                "into the synthetic NPZ package."
            )

        # Display-only correction: place the peak of the source-time
        # function at catalogue origin. No empirical/manual alignment.
        synthetic_display_shift_s = -source_time_function_t0_s

        if not np.isfinite(source_theta_deg):
            raise ValueError("Missing/invalid source_theta_deg.")

        if not np.isclose(
            source_theta_deg,
            args.theta_deg,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError(
                "Requested theta does not match synthetic package."
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
                "Synthetic run_tag does not match requested run."
            )

        package_model = infer_package_model_name(synthetic)

        if package_model and package_model != args.initial_model:
            raise ValueError(
                "Requested initial model does not match package: "
                f"{args.initial_model!r} != {package_model!r}."
            )

        model_type = str(
            get_scalar(synthetic, "model_type", "")
        )

        shared_arrivals, arrival_provenance = predict_comparison_arrivals(
            real, args, out_dir
        )
        if args.arrivals_only:
            print(f"Predictions and provenance saved to {out_dir}")
            return

        # ---------------------------------------------------------------------
        # Real DAS
        # ---------------------------------------------------------------------
        if "das_data_unfiltered" not in real.files:
            raise RuntimeError(
                "Real package lacks das_data_unfiltered; rerun prepare_event."
            )

        real_unfiltered = np.asarray(
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
        real_fs_hz = float(get_scalar(real, "fs"))

        if real_unfiltered.shape != (
            real_channels.size,
            real_time_full.size,
        ):
            raise ValueError(
                "Real DAS shape does not match channel/time axes."
            )

        print(
            f"Filtering real DAS to "
            f"{FMIN_COMPARE_HZ:.1f}-{FMAX_COMPARE_HZ:.1f} Hz, "
            "zero phase..."
        )

        real_filtered_full = bandpass_traces(
            real_unfiltered,
            fs_hz=real_fs_hz,
            fmin_hz=FMIN_COMPARE_HZ,
            fmax_hz=FMAX_COMPARE_HZ,
            order=FILTER_ORDER,
            taper_frac=FILTER_TAPER_FRAC,
        )

        real_envelope_full = compute_envelope(real_filtered_full)

        real_mask = (
            (real_time_full >= TMIN_REAL_S)
            & (real_time_full <= TMAX_REAL_S)
        )

        real_data = real_filtered_full[:, real_mask]
        real_time = real_time_full[real_mask]
        real_envelope = real_envelope_full[:, real_mask]

        real_signed_norm = trace_normalize(real_data)
        real_envelope_norm = trace_normalize(real_envelope)

        # ---------------------------------------------------------------------
        # Synthetic DAS
        # ---------------------------------------------------------------------
        synthetic_unfiltered = np.asarray(
            synthetic["das_data"],
            dtype=np.float64,
        )

        synthetic_time_physical_full = np.asarray(
            synthetic["t"],
            dtype=np.float64,
        )

        synthetic_dt_s = float(
            get_scalar(synthetic, "dt", np.nan)
        )

        if not np.isfinite(synthetic_dt_s) or synthetic_dt_s <= 0.0:
            synthetic_dt_s = float(
                np.median(
                    np.diff(synthetic_time_physical_full)
                )
            )

        synthetic_fs_hz = 1.0 / synthetic_dt_s

        # Authoritative channel axis first; legacy fallbacks second.
        if "das_raw_channels" in synthetic.files:
            synthetic_channels_full = np.asarray(
                synthetic["das_raw_channels"],
                dtype=np.float64,
            )
            channel_mapping_source = "synthetic das_raw_channels"

        elif (
            "receiver_raw_channels" in synthetic.files
            and "das_channel_indices" in synthetic.files
        ):
            receiver_raw_channels = np.asarray(
                synthetic["receiver_raw_channels"],
                dtype=np.float64,
            )
            das_idx = np.asarray(
                synthetic["das_channel_indices"],
                dtype=np.int64,
            )
            synthetic_channels_full = receiver_raw_channels[das_idx]
            channel_mapping_source = (
                "receiver_raw_channels[das_channel_indices]"
            )

        else:
            receiver_x = np.asarray(
                synthetic["receiver_x"],
                dtype=np.float64,
            )
            receiver_z = np.asarray(
                synthetic["receiver_z"],
                dtype=np.float64,
            )
            das_idx = np.asarray(
                synthetic["das_channel_indices"],
                dtype=np.int64,
            )

            receiver_channels = nearest_geometry_channel_for_receivers(
                receiver_x,
                receiver_z,
                Path(GEOMETRY_CSV),
            )
            synthetic_channels_full = receiver_channels[das_idx]
            channel_mapping_source = "legacy nearest-geometry mapping"

        ntr = synthetic_unfiltered.shape[0]

        if synthetic_channels_full.shape[0] != ntr:
            raise ValueError(
                f"synthetic_channels length {synthetic_channels_full.shape[0]} "
                f"!= ntraces {ntr}."
            )

        print(
            f"Filtering synthetic DAS to "
            f"{FMIN_COMPARE_HZ:.1f}-{FMAX_COMPARE_HZ:.1f} Hz, "
            "zero phase..."
        )

        synthetic_filtered_full = bandpass_traces(
            synthetic_unfiltered,
            fs_hz=synthetic_fs_hz,
            fmin_hz=FMIN_COMPARE_HZ,
            fmax_hz=FMAX_COMPARE_HZ,
            order=FILTER_ORDER,
            taper_frac=FILTER_TAPER_FRAC,
        )

        synthetic_envelope_full = compute_envelope(
            synthetic_filtered_full
        )

        synthetic_time_display_full = (
            synthetic_time_physical_full
            + synthetic_display_shift_s
        )

        syn_mask = (
            (synthetic_time_display_full >= TMIN_SYN_DISPLAY_S)
            & (synthetic_time_display_full <= TMAX_SYN_DISPLAY_S)
        )

        synthetic_data = synthetic_filtered_full[:, syn_mask]
        synthetic_envelope = synthetic_envelope_full[:, syn_mask]
        synthetic_time_display = synthetic_time_display_full[syn_mask]

        (
            synthetic_channels,
            synthetic_data,
            synthetic_envelope,
        ) = sort_by_channel(
            synthetic_channels_full,
            synthetic_data,
            synthetic_envelope,
        )

        synthetic_signed_norm = trace_normalize(synthetic_data)
        synthetic_envelope_norm = trace_normalize(synthetic_envelope)

        channel_limits, time_limits = comparison_view_limits(
            real_channels=real_channels, synthetic_channels=synthetic_channels,
            real_time_s=real_time, synthetic_time_display_s=synthetic_time_display,
        )

        # ---------------------------------------------------------------------
        # Frequency QC
        # ---------------------------------------------------------------------
        frequency_figure = (
            out_dir / "00_frequency_content_qc.png"
        )
        frequency_csv = (
            out_dir / "frequency_content_qc.csv"
        )

        plot_frequency_qc(
            real_data_unfiltered=real_unfiltered,
            real_time_s=real_time_full,
            real_channels=real_channels,
            real_fs_hz=real_fs_hz,
            synthetic_data_unfiltered=synthetic_unfiltered,
            synthetic_time_physical_s=synthetic_time_physical_full,
            synthetic_channels=synthetic_channels_full,
            synthetic_fs_hz=synthetic_fs_hz,
            initial_model=args.initial_model,
            out_figure=frequency_figure,
            out_csv=frequency_csv,
        )

        p_finite = shared_arrivals["p_arrivals_s"]
        p_finite = p_finite[np.isfinite(p_finite)]
        s_finite = shared_arrivals["s_arrivals_s"]
        s_finite = s_finite[np.isfinite(s_finite)]

        p_min = float(np.min(p_finite))
        p_max = float(np.max(p_finite))
        s_min = float(np.min(s_finite))
        s_max = float(np.max(s_finite))

        # ---------------------------------------------------------------------
        # Print / save summary
        # ---------------------------------------------------------------------
        print("\nReal/Synthetic comparison QC")
        print("============================")
        print(f"event id                 : {event_id}")
        print(f"initial model            : {args.initial_model}")
        print(f"model type               : {model_type}")
        print(f"source f0                : {source_f0_hz:.2f} Hz")
        print(f"source STF t0            : {source_time_function_t0_s:.6f} s")
        print(f"source theta             : {source_theta_deg:.2f} deg")
        print(
            "common comparison band   : "
            f"{FMIN_COMPARE_HZ:.1f} to "
            f"{FMAX_COMPARE_HZ:.1f} Hz, zero phase"
        )
        print(f"channel mapping          : {channel_mapping_source}")
        print(f"real DAS shape           : {real_data.shape}")
        print(f"synthetic DAS shape      : {synthetic_data.shape}")
        print(
            f"real time range          : "
            f"{real_time.min():.3f} to {real_time.max():.3f} s"
        )
        print(
            f"synthetic display range  : "
            f"{synthetic_time_display.min():.3f} to "
            f"{synthetic_time_display.max():.3f} s"
        )
        print(
            f"real channel range       : "
            f"{real_channels.min():.1f} to {real_channels.max():.1f}"
        )
        print(
            f"synthetic channel range  : "
            f"{synthetic_channels.min():.1f} to "
            f"{synthetic_channels.max():.1f}"
        )
        print(f"P arrival range          : {p_min:.3f} to {p_max:.3f} s")
        print(f"S arrival range          : {s_min:.3f} to {s_max:.3f} s")
        print(f"common plotted channels  : {channel_limits[0]:.1f} to {channel_limits[1]:.1f}")
        print(f"common plotted time      : {time_limits[1]:.3f} to {time_limits[0]:.3f} s")
        print(
            f"synthetic display shift  : "
            f"{synthetic_display_shift_s:+.3f} s"
        )
        print("observed ridge picker    : disabled")

        summary_csv = out_dir / "comparison_summary.csv"

        pd.DataFrame(
            [
                {
                    "event_id": event_id,
                    "initial_model": args.initial_model,
                    "model_type": model_type,
                    "run_tag": requested_run_tag,
                    "theta_deg": source_theta_deg,
                    "source_f0_hz": source_f0_hz,
                    "source_time_function_t0_s": source_time_function_t0_s,
                    "common_fmin_hz": FMIN_COMPARE_HZ,
                    "common_fmax_hz": FMAX_COMPARE_HZ,
                    "synthetic_display_shift_s": synthetic_display_shift_s,
                    "real_ntraces": int(real_data.shape[0]),
                    "synthetic_ntraces": int(synthetic_data.shape[0]),
                    "real_channel_min": float(np.nanmin(real_channels)),
                    "real_channel_max": float(np.nanmax(real_channels)),
                    "synthetic_channel_min": float(
                        np.nanmin(synthetic_channels)
                    ),
                    "synthetic_channel_max": float(
                        np.nanmax(synthetic_channels)
                    ),
                    "predicted_p_min_s": p_min,
                    "predicted_p_max_s": p_max,
                    "predicted_s_min_s": s_min,
                    "predicted_s_max_s": s_max,
                    "channel_mapping_source": channel_mapping_source,
                    "real_package": str(real_pkg),
                    "synthetic_package": str(syn_pkg),
                    "observed_ridge_picker": "disabled",
                    "real_prediction_method": arrival_provenance["real"]["method"],
                    "real_prediction_model": arrival_provenance["real"]["model_name"],
                    "synthetic_prediction_method": arrival_provenance["synthetic"]["method"],
                    "synthetic_prediction_model": arrival_provenance["synthetic"]["model_name"],
                    "shared_3d_prediction_reference": True,
                    "comparison_channel_min": channel_limits[0],
                    "comparison_channel_max": channel_limits[1],
                    "comparison_time_min_s": time_limits[1],
                    "comparison_time_max_s": time_limits[0],
                }
            ]
        ).to_csv(summary_csv, index=False)

        # ---------------------------------------------------------------------
        # Figures
        # ---------------------------------------------------------------------
        real_overlay = (
            out_dir / "01_real_with_predicted_arrivals.png"
        )
        signed_comparison = (
            out_dir / "02_real_vs_synthetic_signed.png"
        )
        envelope_comparison = (
            out_dir / "03_real_vs_synthetic_envelopes.png"
        )

        plot_real_with_arrivals(
            real_signed_normalized=real_signed_norm,
            real_time_s=real_time,
            real_channels=real_channels,
            event_id=event_id,
            out_path=real_overlay,
            shared_arrivals=shared_arrivals,
        )

        plot_signed_side_by_side(
            real_signed_normalized=real_signed_norm,
            real_time_s=real_time,
            real_channels=real_channels,
            synthetic_signed_normalized=synthetic_signed_norm,
            synthetic_time_display_s=synthetic_time_display,
            synthetic_channels=synthetic_channels,
            event_id=event_id,
            source_f0_hz=source_f0_hz,
            source_theta_deg=source_theta_deg,
            synthetic_display_shift_s=synthetic_display_shift_s,
            initial_model=args.initial_model,
            out_path=signed_comparison,
            shared_arrivals=shared_arrivals,
        )

        plot_envelope_side_by_side(
            real_envelope_normalized=real_envelope_norm,
            real_time_s=real_time,
            real_channels=real_channels,
            synthetic_envelope_normalized=synthetic_envelope_norm,
            synthetic_time_display_s=synthetic_time_display,
            synthetic_channels=synthetic_channels,
            event_id=event_id,
            source_f0_hz=source_f0_hz,
            source_theta_deg=source_theta_deg,
            synthetic_display_shift_s=synthetic_display_shift_s,
            initial_model=args.initial_model,
            out_path=envelope_comparison,
            shared_arrivals=shared_arrivals,
        )

    print("\nSaved outputs")
    print("=============")
    print(frequency_figure)
    print(frequency_csv)
    print(real_overlay)
    print(signed_comparison)
    print(envelope_comparison)
    print(summary_csv)


if __name__ == "__main__":
    main()
