"""Controlled, signed DAS P-window comparison of manual and ComCat sources."""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.safod import settings
from scripts.safod.compare_event import (
    DEFAULT_INITIAL_MODEL,
    comparison_dir_for_model,
    synthetic_package_for_model,
)
from src.common.signal_processing import bandpass_traces
from src.safod.focal_mechanism import validate_saved_source_mechanism
from src.safod.models import AVAILABLE_INITIAL_MODELS


# No fitted time shifts. P/S predictions serve as window guides, not picks.
P_BEFORE_S = 0.04
P_AFTER_S = 0.12
S_GUARD_S = 0.02
MIN_WINDOW_S = 0.08
NOISE_S = (-0.24, -0.06)
MIN_RMS_RATIO = 3.0
BACKEND = "sbp_sat"
ARRIVALS = "predicted_arrivals_real_3d.csv"
# source scale, wavelet, location, model, sampling, and boundary conditions
# must agree. Only the tensor orientation/components should differ.
FIXED = (
    "event_id", "event_origin_time", "initial_model_name", "solver_backend",
    "source_x", "source_z", "source_spreading", "source_f0_hz",
    "source_time_mode", "source_stf_kind", "source_stf_sampling_offset_steps",
    "source_time_function_t0_s", "source_scalar_moment", "source_spatial_normalization",
    "dt", "nt", "t", "grid_x", "grid_z", "dx", "dz", "vp", "vs", "rho",
    "receiver_x", "receiver_z", "receiver_s", "das_channel_indices",
    "das_gauge_length_m", "das_raw_channels", "boundary_scheme_version",
    "boundary_top", "boundary_bottom", "boundary_left", "boundary_right",
    "sponge_side_width_m", "sponge_bottom_width_m", "sponge_rate_per_s",
    "sponge_time_split",
)


def verify_control(manual, comcat, real, theta_deg, model):
    code = validate_saved_source_mechanism(comcat, None)
    validate_saved_source_mechanism(manual, theta_deg)
    for name in FIXED:
        if name not in manual or name not in comcat:
            raise ValueError(f"Missing {name!r} in a forward NPZ; rerun both with the same current setup.")
        if not np.array_equal(manual[name], comcat[name]):
            raise ValueError(f"Forward inputs differ in {name!r}; this is not a source-only control.")
    for name, expected in (
        ("event_id", str(real["ev_id"].item())),
        ("event_origin_time", str(real["ev_origin_time"].item())),
        ("initial_model_name", model),
        ("solver_backend", BACKEND),
    ):
        if str(manual[name].item()) != expected:
            raise ValueError(f"Saved {name!r} differs from the requested real event/model/backend.")
    if not np.isclose(float(real["gauge_length_m"]), float(manual["das_gauge_length_m"]),
                      atol=1e-8, rtol=0):
        raise ValueError("Real and synthetic DAS gauge lengths differ.")
    return code


def match_channels(reference, targets):
    """Find the same physical reference channels, allowing CSV roundoff only."""
    reference = np.asarray(reference, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if (reference.ndim != 1 or not reference.size or targets.ndim != 1 or not targets.size
            or not np.isfinite(reference).all() or not np.isfinite(targets).all()):
        raise ValueError("Invalid physical reference channel axis.")
    order = np.argsort(reference)
    ref = reference[order]
    if np.any(np.diff(ref) <= 2e-6) or np.unique(targets).size != targets.size:
        raise ValueError("Ambiguous physical reference channels.")
    right = np.clip(np.searchsorted(ref, targets), 0, len(ref) - 1)
    left = np.maximum(right - 1, 0)
    nearest = np.where(abs(ref[left] - targets) < abs(ref[right] - targets), left, right)
    if np.any(abs(ref[nearest] - targets) > 1e-6):
        raise ValueError("Real, synthetic or predicted-arrival physical channels do not match.")
    return order[nearest]


def load_guides(path, channels):
    frame = pd.read_csv(path)
    i = match_channels(frame["channel"], channels)
    arrivals = frame[["p_time_s", "s_time_s"]].to_numpy()[i]
    if not np.isfinite(arrivals).all() or np.any(arrivals[:, 1] <= arrivals[:, 0]):
        raise ValueError(f"Invalid P/S guides in {path}.")
    return arrivals


def filtered_das(data, t, fs_hz, n_channels):
    t = np.asarray(t, dtype=float)
    if (t.ndim != 1 or t.size < 2 or not np.isfinite(t).all()
            or not np.isfinite(fs_hz) or fs_hz <= 0
            or not np.allclose(np.diff(t), 1 / fs_hz, atol=1e-10, rtol=1e-6)
            or data.shape != (n_channels, t.size)):
        raise ValueError("Invalid DAS shape or sampling axis.")
    return bandpass_traces(
        data, fs_hz=fs_hz, fmin_hz=settings.COMMON_FMIN_HZ,
        fmax_hz=settings.COMMON_FMAX_HZ, order=settings.FILTER_ORDER,
        taper_frac=settings.FILTER_TAPER_FRAC,
    )


def demean(a):
    return a - np.mean(a)


def rms(a):
    return float(np.linalg.norm(a) / np.sqrt(a.size))


def score_p_windows(t_real, t_syn, channels, guides, real, manual, comcat):
    rows, traces = [], {}
    noise = (t_real >= NOISE_S[0]) & (t_real <= NOISE_S[1])
    if t_real[0] > NOISE_S[0] or t_real[-1] < NOISE_S[1] or noise.sum() < 8:
        raise ValueError(f"The real record does not cover the pre-event noise window {NOISE_S}.")
    for i, (channel, (p, s)) in enumerate(zip(channels, guides)):
        lo, hi = p - P_BEFORE_S, min(p + P_AFTER_S, s - S_GUARD_S)
        row = dict(channel=channel, p_time_s=p, s_time_s=s,
                   start_s=lo, end_s=hi, n_samples=0, real_p_to_noise_rms=np.nan,
                   dc_rms=np.nan, comcat_rms=np.nan, dc_score=np.nan,
                   comcat_score=np.nan, delta_score=np.nan)
        rows.append(row)
        mask = (t_real >= lo) & (t_real <= hi)
        row["n_samples"] = int(mask.sum())
        if (hi - lo < MIN_WINDOW_S or mask.sum() < 8
                or lo < max(t_real[0], t_syn[0]) or hi > min(t_real[-1], t_syn[-1])):
            row["status"] = "window_outside_record_or_too_short"
            continue
        t = t_real[mask]
        d = demean(real[i, mask])
        noise_rms = rms(demean(real[i, noise]))
        d_rms = rms(d)
        ratio = d_rms / noise_rms if noise_rms > 0 else np.nan
        row["real_p_to_noise_rms"] = ratio
        if not np.isfinite(ratio) or ratio < MIN_RMS_RATIO:
            row["status"] = "insufficient_real_p_to_noise"
            continue

        # Both synthetics have the same native time axis and STF correction.
        a = demean(np.interp(t, t_syn, manual[i]))
        b = demean(np.interp(t, t_syn, comcat[i]))
        a_rms, b_rms = rms(a), rms(b)
        row.update(dc_rms=a_rms, comcat_rms=b_rms)
        if min(a_rms, b_rms) == 0:
            row["status"] = "zero_synthetic_window"
            continue
        # Signed cosine similarity of demeaned waveforms, at fixed timing.
        d, a, b = d / d_rms, a / a_rms, b / b_rms
        sa = float(np.clip(np.mean(d * a), -1, 1))
        sb = float(np.clip(np.mean(d * b), -1, 1))
        row.update(dc_score=sa, comcat_score=sb, delta_score=sb - sa, status="scored")
        traces[i] = (t - p, d, a, b)

    return pd.DataFrame(rows), traces


def plot_diagnostic(out, metrics, traces, theta_deg, code):
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    ax = axes.flat[0]
    ax.plot(metrics.channel, metrics.dc_score, label=f"Manual {theta_deg:g}°")
    ax.plot(metrics.channel, metrics.comcat_score, label="ComCat")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set(xlabel="Physical reference channel", ylabel="Signed similarity", ylim=(-1.05, 1.05))
    ax.legend(loc="best")
    for ax, target in zip(list(axes.flat)[1:], (400, 700, 1000, 1400, 1600)):
        i = int(np.argmin(abs(metrics.channel.to_numpy() - target)))
        channel = metrics.channel.iloc[i]
        ax.set_title(f"Reference channel {channel:.2f}")
        if i not in traces:
            ax.text(0.5, 0.5, metrics.status.iloc[i].replace("_", " "),
                    ha="center", va="center", transform=ax.transAxes, wrap=True)
            ax.set_axis_off()
            continue
        t, d, a, b = traces[i]
        ax.plot(t, d, label="Observed", c="black")
        ax.plot(t, a, label=f"Manual {theta_deg:g}°")
        ax.plot(t, b, label="ComCat")
        ax.axvline(0, color="0.6", lw=0.8)
        ax.set(xlabel="Time from predicted P (s)", ylabel="Window RMS units")
        ax.legend(loc="best", fontsize="small")
    fig.suptitle(f"Source comparison • {code} • P windows at fixed timing")
    fig.savefig(out / "p_source_comparison.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta-deg", type=float, default=settings.DEFAULT_THETA_DEG)
    parser.add_argument("--initial-model", choices=AVAILABLE_INITIAL_MODELS,
                        default=DEFAULT_INITIAL_MODEL)
    args = parser.parse_args()
    theta, model = args.theta_deg, args.initial_model
    settings.format_theta_tag(theta)
    manual_path = synthetic_package_for_model(theta, model, BACKEND)
    comcat_path = synthetic_package_for_model(None, model, BACKEND)
    manual_compare = comparison_dir_for_model(theta, model, BACKEND)
    comcat_compare = comparison_dir_for_model(None, model, BACKEND)

    with np.load(settings.REAL_EVENT_PACKAGE, allow_pickle=False) as real, \
         np.load(manual_path, allow_pickle=False) as manual, \
         np.load(comcat_path, allow_pickle=False) as comcat:
        code = verify_control(manual, comcat, real, theta, model)
        channels = manual["das_raw_channels"]
        i_real = match_channels(real["raw_channels"], channels)
        guides = load_guides(comcat_compare / ARRIVALS, channels)
        if not np.allclose(guides, load_guides(manual_compare / ARRIVALS, channels),
                           atol=1e-8, rtol=0):
            raise ValueError("Saved P/S guides differ. Run compare_event with matching eikonal options.")
        t_real, t_syn = real["t"], manual["t"]
        d = filtered_das(real["das_data_unfiltered"][i_real], t_real, float(real["fs"]), len(channels))
        fs_syn = 1 / float(manual["dt"])
        a = filtered_das(manual["das_data"], t_syn, fs_syn, len(channels))
        b = filtered_das(comcat["das_data"], t_syn, fs_syn, len(channels))
        t_syn = t_syn - float(manual["source_time_function_t0_s"])
        metrics, traces = score_p_windows(t_real, t_syn, channels, guides, d, a, b)
        event_id = str(real["ev_id"].item())

    out = comcat_compare / f"source_qc_vs_{settings.theta_dir_name(theta)}"
    out.mkdir(parents=True, exist_ok=True)
    scored = metrics[metrics.status == "scored"]
    summary = dict(
        event_id=event_id, comcat_product=code, manual_theta_deg=theta,
        initial_model=model, manual_package=str(manual_path), comcat_package=str(comcat_path),
        common_band_hz=[settings.COMMON_FMIN_HZ, settings.COMMON_FMAX_HZ],
        noise_window_s=list(NOISE_S), min_real_p_to_noise_rms=MIN_RMS_RATIO,
        n_channels=len(metrics), n_scored=len(scored),
        status_counts={k: int(v) for k, v in metrics.status.value_counts().items()},
        median_dc_score=float(scored.dc_score.median()) if len(scored) else None,
        median_comcat_score=float(scored.comcat_score.median()) if len(scored) else None,
        median_paired_delta=float(scored.delta_score.median()) if len(scored) else None,
        n_comcat_better=int((scored.delta_score > 0).sum()),
        n_dc_better=int((scored.delta_score < 0).sum()),
    )
    metrics.to_csv(out / "p_source_metrics.csv", index=False)
    plot_diagnostic(out, metrics, traces, theta, code)
    report = json.dumps(summary, indent=2, allow_nan=False)
    (out / "source_qc_summary.json").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"Saved source QC to {out}")


if __name__ == "__main__":
    main()
