from __future__ import annotations

import argparse
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
    comparison_dir_for_theta,
    forward_package_for_theta,
    forward_run_tag,
)
from src.signal_processing import bandpass_traces, median_welch_psd


# =============================================================================
# SETTINGS
# =============================================================================

INITIAL_MODEL_CHOICES = ("smooth_prior", "digitized_log")
DEFAULT_INITIAL_MODEL = "digitized_log"

SYN_DISPLAY_TIME_SHIFT_S = -0.20

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

    if "digitized" in model_type or "ellsworth" in model_type:
        return "digitized_log"

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
        linewidth=1.5,
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
        linewidth=1.5,
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

def add_arrival_overlays(
    *,
    ax,
    arrival_channels: np.ndarray,
    p_arrivals_s: np.ndarray,
    s_arrivals_s: np.ndarray,
    time_shift_s: float = 0.0,
    light: bool = False,
    include_labels: bool = True,
) -> None:
    colour = "white" if light else "black"

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
        linestyle="--",
        linewidth=1.5,
        label="Predicted P" if include_labels else None,
        zorder=20,
    )

    ax.plot(
        arrival_channels[valid_s],
        s_arrivals_s[valid_s] + time_shift_s,
        color=colour,
        linestyle=":",
        linewidth=1.8,
        label="Predicted S" if include_labels else None,
        zorder=20,
    )


def plot_real_with_arrivals(
    *,
    real_signed_normalized,
    real_time_s,
    real_channels,
    arrival_channels,
    p_arrivals_s,
    s_arrivals_s,
    event_id,
    initial_model,
    out_path,
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
        arrival_channels=arrival_channels,
        p_arrivals_s=p_arrivals_s,
        s_arrivals_s=s_arrivals_s,
        include_labels=True,
    )

    fig.colorbar(
        im,
        ax=ax,
        label="Trace-normalized amplitude",
    )

    ax.set_title(
        f"{event_id} real DAS with predicted arrivals\n"
        f"initial model: {initial_model}"
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
    arrival_channels,
    p_arrivals_s,
    s_arrivals_s,
    event_id,
    source_f0_hz,
    source_theta_deg,
    initial_model,
    out_path,
):
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
        label="Catalog origin",
    )

    add_arrival_overlays(
        ax=ax_real,
        arrival_channels=arrival_channels,
        p_arrivals_s=p_arrivals_s,
        s_arrivals_s=s_arrivals_s,
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
        arrival_channels=arrival_channels,
        p_arrivals_s=p_arrivals_s,
        s_arrivals_s=s_arrivals_s,
        time_shift_s=SYN_DISPLAY_TIME_SHIFT_S,
        include_labels=False,
    )

    ax_syn.set_title(f"Synthetic DAS: {initial_model}")
    ax_syn.set_xlabel("Physical reference channel")

    fig.colorbar(
        syn_im,
        ax=axes.ravel().tolist(),
        label="Trace-normalized amplitude",
        shrink=0.85,
    )

    fig.suptitle(
        f"{event_id} real vs synthetic signed DAS\n"
        f"f0={source_f0_hz:.1f} Hz, "
        f"theta={source_theta_deg:.1f} deg, "
        f"display shift={SYN_DISPLAY_TIME_SHIFT_S:+.2f} s",
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
    arrival_channels,
    p_arrivals_s,
    s_arrivals_s,
    event_id,
    source_f0_hz,
    source_theta_deg,
    initial_model,
    out_path,
):
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
        label="Catalog origin",
    )

    add_arrival_overlays(
        ax=ax_real,
        arrival_channels=arrival_channels,
        p_arrivals_s=p_arrivals_s,
        s_arrivals_s=s_arrivals_s,
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
        arrival_channels=arrival_channels,
        p_arrivals_s=p_arrivals_s,
        s_arrivals_s=s_arrivals_s,
        time_shift_s=SYN_DISPLAY_TIME_SHIFT_S,
        light=True,
        include_labels=False,
    )

    ax_syn.set_title(f"Synthetic envelope: {initial_model}")
    ax_syn.set_xlabel("Physical reference channel")

    fig.colorbar(
        syn_im,
        ax=axes.ravel().tolist(),
        label="Trace-normalized envelope",
        shrink=0.85,
    )

    fig.suptitle(
        f"{event_id} real vs synthetic DAS envelopes\n"
        f"{FMIN_COMPARE_HZ:g}-{FMAX_COMPARE_HZ:g} Hz, "
        f"f0={source_f0_hz:.1f} Hz, "
        f"theta={source_theta_deg:.1f} deg",
        y=0.98,
    )

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================

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

    out_dir = comparison_dir_for_model(
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

        predicted_p_full = np.asarray(
            synthetic["arrival_p_das"],
            dtype=np.float64,
        )
        predicted_s_full = np.asarray(
            synthetic["arrival_swave_das"],
            dtype=np.float64,
        )

        ntr = synthetic_unfiltered.shape[0]

        for name, array in (
            ("synthetic_channels", synthetic_channels_full),
            ("arrival_p_das", predicted_p_full),
            ("arrival_swave_das", predicted_s_full),
        ):
            if array.shape[0] != ntr:
                raise ValueError(
                    f"{name} length {array.shape[0]} != ntraces {ntr}."
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
            + SYN_DISPLAY_TIME_SHIFT_S
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
            predicted_p,
            predicted_s,
        ) = sort_by_channel(
            synthetic_channels_full,
            synthetic_data,
            synthetic_envelope,
            predicted_p_full,
            predicted_s_full,
        )

        synthetic_signed_norm = trace_normalize(synthetic_data)
        synthetic_envelope_norm = trace_normalize(synthetic_envelope)

        (
            arrival_channels,
            predicted_p_unique,
            predicted_s_unique,
        ) = collapse_duplicate_channels(
            synthetic_channels,
            predicted_p,
            predicted_s,
        )

        arrival_mask = (
            np.isfinite(arrival_channels)
            & np.isfinite(predicted_p_unique)
            & np.isfinite(predicted_s_unique)
            & (arrival_channels >= np.nanmin(real_channels))
            & (arrival_channels <= np.nanmax(real_channels))
        )

        arrival_channels = arrival_channels[arrival_mask]
        predicted_p_unique = predicted_p_unique[arrival_mask]
        predicted_s_unique = predicted_s_unique[arrival_mask]

        if arrival_channels.size < 2:
            raise RuntimeError("Too few valid arrival channels.")

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

        p_finite = predicted_p_unique[
            np.isfinite(predicted_p_unique)
        ]
        s_finite = predicted_s_unique[
            np.isfinite(predicted_s_unique)
        ]

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
        print(
            f"synthetic display shift  : "
            f"{SYN_DISPLAY_TIME_SHIFT_S:+.3f} s"
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
                    "common_fmin_hz": FMIN_COMPARE_HZ,
                    "common_fmax_hz": FMAX_COMPARE_HZ,
                    "synthetic_display_shift_s": SYN_DISPLAY_TIME_SHIFT_S,
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
            arrival_channels=arrival_channels,
            p_arrivals_s=predicted_p_unique,
            s_arrivals_s=predicted_s_unique,
            event_id=event_id,
            initial_model=args.initial_model,
            out_path=real_overlay,
        )

        plot_signed_side_by_side(
            real_signed_normalized=real_signed_norm,
            real_time_s=real_time,
            real_channels=real_channels,
            synthetic_signed_normalized=synthetic_signed_norm,
            synthetic_time_display_s=synthetic_time_display,
            synthetic_channels=synthetic_channels,
            arrival_channels=arrival_channels,
            p_arrivals_s=predicted_p_unique,
            s_arrivals_s=predicted_s_unique,
            event_id=event_id,
            source_f0_hz=source_f0_hz,
            source_theta_deg=source_theta_deg,
            initial_model=args.initial_model,
            out_path=signed_comparison,
        )

        plot_envelope_side_by_side(
            real_envelope_normalized=real_envelope_norm,
            real_time_s=real_time,
            real_channels=real_channels,
            synthetic_envelope_normalized=synthetic_envelope_norm,
            synthetic_time_display_s=synthetic_time_display,
            synthetic_channels=synthetic_channels,
            arrival_channels=arrival_channels,
            p_arrivals_s=predicted_p_unique,
            s_arrivals_s=predicted_s_unique,
            event_id=event_id,
            source_f0_hz=source_f0_hz,
            source_theta_deg=source_theta_deg,
            initial_model=args.initial_model,
            out_path=envelope_comparison,
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