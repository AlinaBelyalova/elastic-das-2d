#!/usr/bin/env python3
"""
Find the CURRENT Sintela DAS channel(s) nearest the MH030 downhole pod
using first-arrival timing, without relying on the old OptaSense channel map.

Event: nc75443276, 2026-10-03
DAS: two Sintela protobuf files used by Ettore's notebook
Seismometer: SF.MH030.01.GP1 waveform in raw counts

Why this script exists
----------------------
The Oct-2026 Sintela data have 3236 channels and dx ~2.5321 m.  The older
SAFOD_Phase2_GeoReferenced_Channels.xlsx channel numbers should therefore NOT
be assumed to map one-to-one onto the current Sintela channel indices.

This diagnostic:
  * reads the current protobuf data exactly as Ettore does;
  * gets the exact NCEDC origin;
  * downloads MH030 GP1 raw counts (no provisional MH029 response);
  * resamples GP1 to DAS fs;
  * applies the SAME DASutils causal filter to both;
  * picks the first strong onset independently on every DAS channel;
  * finds channels whose onset is closest to the GP1 onset;
  * makes an early-time gather + onset-vs-channel plot + trace panel.

No relative time shift is applied anywhere.
"""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path
import sys

import dateutil.parser
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from scipy.signal import detrend, hilbert, resample_poly


# -----------------------------------------------------------------------------
# DASutils
# -----------------------------------------------------------------------------
DAS_UTILITIES_ROOT = Path("/home/groups/ettore88/alina/packages/DAS-utilities")
DAS_UTILITIES_BUILD = DAS_UTILITIES_ROOT / "build"
DAS_UTILITIES_PYTHON = DAS_UTILITIES_ROOT / "python"

existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = (
    f"{existing_ld}:{DAS_UTILITIES_BUILD}"
    if existing_ld else str(DAS_UTILITIES_BUILD)
)
sys.path.insert(0, str(DAS_UTILITIES_BUILD))
sys.path.insert(0, str(DAS_UTILITIES_PYTHON))
import DASutils  # noqa: E402


EVENT_ID_NUMERIC = 75443276
FALLBACK_ORIGIN = UTCDateTime("2026-10-03T04:52:31.320000Z")

DAS_FILES = [
    Path(
        "/oak/stanford/groups/ettore88/data/SAFOD/SAFOD_events/"
        "timeseriesScaler_2026-10-03_04.51.40_UTC_073863.pb"
    ),
    Path(
        "/oak/stanford/groups/ettore88/data/SAFOD/SAFOD_events/"
        "timeseriesScaler_2026-10-03_04.52.40_UTC_073864.pb"
    ),
]

NETWORK = "SF"
STATION = "MH030"
LOCATION = "01"
CHANNEL = "GP1"

FMIN_HZ = 5.0
FMAX_HZ = 40.0

NOISE_TMIN_S = -0.20
NOISE_TMAX_S = 0.05
PICK_TMIN_S = 0.05
PICK_TMAX_S = 0.60

# threshold = median(noise envelope) + THRESHOLD_MAD * robust_sigma
THRESHOLD_MAD = 8.0
MIN_SUSTAIN_MS = 8.0

GATHER_TMIN_S = 0.00
GATHER_TMAX_S = 0.65
DPI = 250


def parse_beg_time(info) -> UTCDateTime:
    value = info["begTime"]
    if isinstance(value, datetime.datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return UTCDateTime(dt)

    dt = dateutil.parser.parse(str(value))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return UTCDateTime(dt)


def get_origin(client: Client) -> UTCDateTime:
    try:
        cat = client.get_events(eventid=EVENT_ID_NUMERIC)
        ev = cat[0]
        origin = ev.preferred_origin() or ev.origins[0]
        print(f"Exact NCEDC origin: {origin.time}")
        return origin.time
    except Exception as exc:
        print(f"WARNING: event query failed: {exc}")
        print(f"Using fallback origin: {FALLBACK_ORIGIN}")
        return FALLBACK_ORIGIN


def load_das(origin: UTCDateTime):
    data, info = DASutils.readFile_protobuf(
        [str(p) for p in DAS_FILES],
        0.01,
        500.0,
        verbose=1,
        diff=False,
        detrend=False,
        tapering=False,
        filter=False,
        median=False,
        desampling=False,
        nChbuffer=3000,
        system="Sintela",
    )

    data = np.asarray(data, dtype=np.float64)
    fs = float(info["fs"])
    beg = parse_beg_time(info)
    t = float(beg - origin) + np.arange(data.shape[1]) / fs

    return data, t, fs, info


def common_causal_filter(x: np.ndarray, fs_hz: float, axis: int = -1):
    """
    Use DASutils.bandpass2D_c with zerophase=False, like Ettore's notebook.

    DASutils expects 2D channel x time input, so 1D traces are wrapped.
    """
    y = np.asarray(x, dtype=np.float64)

    if y.ndim == 1:
        yy = y[None, :].copy()
        DASutils.bandpass2D_c(
            yy,
            fs_hz,
            FMIN_HZ,
            FMAX_HZ,
            zerophase=False,
        )
        return yy[0]

    if axis != 1:
        raise ValueError("This helper expects 2D arrays with time on axis=1.")

    yy = y.copy()
    DASutils.bandpass2D_c(
        yy,
        fs_hz,
        FMIN_HZ,
        FMAX_HZ,
        zerophase=False,
    )
    return yy


def load_gp1_raw_counts(
    client: Client,
    origin: UTCDateTime,
    das_t: np.ndarray,
    das_fs: float,
):
    start = origin + float(das_t[0]) - 2.0
    end = origin + float(das_t[-1]) + 2.0

    st = client.get_waveforms(
        network=NETWORK,
        station=STATION,
        location=LOCATION,
        channel=CHANNEL,
        starttime=start,
        endtime=end,
        attach_response=False,
    )
    st.merge(method=1, fill_value="interpolate")

    tr = st[0].copy()
    raw = detrend(np.asarray(tr.data, dtype=np.float64), type="linear")
    raw -= np.mean(raw)

    geo_fs = float(tr.stats.sampling_rate)
    geo_t = (
        float(tr.stats.starttime - origin)
        + np.arange(tr.stats.npts) / geo_fs
    )

    # Interpolate directly onto DAS absolute-relative time grid.  This preserves
    # timing and avoids any hidden relative shift.
    raw_on_das_grid = np.interp(das_t, geo_t, raw)

    return common_causal_filter(raw_on_das_grid, das_fs)


def smooth_envelope(x: np.ndarray, fs: float, smooth_ms: float = 10.0):
    env = np.abs(hilbert(x, axis=-1))
    n = max(1, int(round(smooth_ms * 1e-3 * fs)))
    kernel = np.ones(n, dtype=np.float64) / n

    if env.ndim == 1:
        return np.convolve(env, kernel, mode="same")

    out = np.empty_like(env)
    for i in range(env.shape[0]):
        out[i] = np.convolve(env[i], kernel, mode="same")
    return out


def robust_threshold(envelope: np.ndarray, noise_mask: np.ndarray):
    noise = envelope[..., noise_mask]
    med = np.median(noise, axis=-1)
    mad = np.median(
        np.abs(noise - np.expand_dims(med, -1)),
        axis=-1,
    )
    robust_sigma = 1.4826 * mad
    return med + THRESHOLD_MAD * robust_sigma


def first_sustained_crossing(
    envelope: np.ndarray,
    threshold,
    t: np.ndarray,
    pick_mask: np.ndarray,
    fs: float,
):
    idxs = np.flatnonzero(pick_mask)
    n_sustain = max(1, int(round(MIN_SUSTAIN_MS * 1e-3 * fs)))

    if envelope.ndim == 1:
        above = envelope[idxs] > float(threshold)
        run = np.convolve(
            above.astype(np.int16),
            np.ones(n_sustain, dtype=np.int16),
            mode="valid",
        )
        hit = np.flatnonzero(run >= n_sustain)
        if len(hit) == 0:
            return np.nan
        return float(t[idxs[hit[0]]])

    picks = np.full(envelope.shape[0], np.nan)

    for i in range(envelope.shape[0]):
        above = envelope[i, idxs] > float(threshold[i])
        run = np.convolve(
            above.astype(np.int16),
            np.ones(n_sustain, dtype=np.int16),
            mode="valid",
        )
        hit = np.flatnonzero(run >= n_sustain)
        if len(hit):
            picks[i] = t[idxs[hit[0]]]

    return picks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ch-min", type=int, default=1050)
    parser.add_argument("--ch-max", type=int, default=1450)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/events/20261003_75443276/compare/"
            "current_channel_timing"
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    client = Client("https://service.ncedc.org")
    origin = get_origin(client)

    das_all, t, fs, info = load_das(origin)

    ch_min = max(0, args.ch_min)
    ch_max = min(das_all.shape[0] - 1, args.ch_max)
    channels = np.arange(ch_min, ch_max + 1)

    print(f"DAS shape: {das_all.shape}")
    print(f"fs={fs} Hz, dx={float(info['dx']):.4f} m")
    print(f"Scanning current Sintela channels {ch_min}:{ch_max}")

    # Current DAS strain-rate traces.
    das = das_all[ch_min:ch_max + 1].astype(np.float64)
    das = detrend(das, axis=1, type="linear")
    das -= np.mean(das, axis=1, keepdims=True)
    das_f = common_causal_filter(das, fs, axis=1)

    # Current MH030 GP1 RAW COUNTS: no response fallback involved.
    gp1_f = load_gp1_raw_counts(client, origin, t, fs)

    noise_mask = (t >= NOISE_TMIN_S) & (t <= NOISE_TMAX_S)
    pick_mask = (t >= PICK_TMIN_S) & (t <= PICK_TMAX_S)

    das_env = smooth_envelope(das_f, fs)
    gp1_env = smooth_envelope(gp1_f, fs)

    das_threshold = robust_threshold(das_env, noise_mask)
    gp1_threshold = robust_threshold(gp1_env, noise_mask)

    das_pick = first_sustained_crossing(
        das_env, das_threshold, t, pick_mask, fs
    )
    gp1_pick = first_sustained_crossing(
        gp1_env, gp1_threshold, t, pick_mask, fs
    )

    result = pd.DataFrame(
        {
            "channel": channels,
            "das_onset_s": das_pick,
            "delta_to_gp1_ms": 1000.0 * (das_pick - gp1_pick),
            "abs_delta_to_gp1_ms": 1000.0 * np.abs(das_pick - gp1_pick),
        }
    ).sort_values("abs_delta_to_gp1_ms")

    result.to_csv(
        args.output_dir / "current_channel_onset_scan.csv",
        index=False,
    )

    print()
    print(f"MH030 GP1 onset: {gp1_pick:.4f} s after origin")
    print()
    print("Channels with onset closest to MH030:")
    print(result.head(20).to_string(index=False))

    # ------------------------------------------------------------------
    # Figure 1: early-time gather
    # ------------------------------------------------------------------
    tm = (t >= GATHER_TMIN_S) & (t <= GATHER_TMAX_S)
    clip = np.percentile(np.abs(das_f[:, tm]), 98.0)

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(
        das_f[:, tm].T,
        extent=[
            ch_min,
            ch_max,
            t[tm][-1],
            t[tm][0],
        ],
        aspect="auto",
        cmap="seismic",
        vmin=-clip,
        vmax=clip,
    )
    ax.axhline(gp1_pick, color="black", ls="--", lw=1.5, label="MH030 GP1 onset")
    for ch in result.head(5)["channel"].astype(int):
        ax.axvline(ch, color="black", alpha=0.35, lw=0.8)

    ax.set_xlabel("Current Sintela channel")
    ax.set_ylabel("Time from exact NCEDC origin [s]")
    ax.set_title("Early DAS wavefield | 5-40 Hz | causal filter")
    ax.legend(frameon=False)
    fig.colorbar(im, ax=ax, label="DAS strain rate [reader units]")
    fig.tight_layout()
    fig.savefig(
        args.output_dir / "01_early_gather_current_channels.png",
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: onset time vs channel
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(channels, das_pick, "k.", ms=4, label="DAS onset")
    ax.axhline(gp1_pick, color="#940025", lw=2, label="MH030 GP1 onset")
    ax.set_xlabel("Current Sintela channel")
    ax.set_ylabel("Picked onset [s after origin]")
    ax.set_title("First-arrival timing: current DAS vs MH030")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        args.output_dir / "02_onset_vs_channel.png",
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 3: GP1 plus the 6 timing-nearest DAS traces
    # ------------------------------------------------------------------
    best = result.head(6)["channel"].astype(int).tolist()
    plot_mask = (t >= 0.05) & (t <= 0.60)

    fig, ax = plt.subplots(figsize=(12, 7))
    offsets = np.arange(len(best) + 1)[::-1].astype(float)

    # GP1 at the top.
    gp = gp1_f.copy()
    gp_scale = np.sqrt(np.mean(gp[plot_mask] ** 2))
    if gp_scale > 0:
        gp = gp / gp_scale
    ax.plot(
        t[plot_mask],
        0.30 * gp[plot_mask] + offsets[0],
        color="#940025",
        lw=1.4,
    )

    labels = ["MH030 GP1 raw counts"]

    for j, ch in enumerate(best, start=1):
        row = ch - ch_min
        tr = das_f[row].copy()
        scale = np.sqrt(np.mean(tr[plot_mask] ** 2))
        if scale > 0:
            tr /= scale
        ax.plot(
            t[plot_mask],
            0.30 * tr[plot_mask] + offsets[j],
            color="black",
            lw=1.0,
        )
        labels.append(f"DAS ch {ch}")

    ax.set_yticks(offsets)
    ax.set_yticklabels(labels, fontweight="bold")
    ax.axvline(gp1_pick, color="#940025", ls="--", lw=1.0)
    ax.set_xlim(0.05, 0.60)
    ax.set_xlabel("Time from exact NCEDC origin [s]")
    ax.set_title("Onset comparison: MH030 and timing-nearest DAS channels")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        args.output_dir / "03_best_timing_channels_vs_gp1.png",
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
