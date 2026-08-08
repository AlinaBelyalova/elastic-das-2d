# scripts/safod/digitize_ellsworth_malin_fig3a.py
#
# Reproduce the raster digitization of Ellsworth & Malin (2011) Fig. 3a
# from the supplied image.
#
# Usage:
#   python -m scripts.safod.digitize_ellsworth_malin_fig3a \
#       --image path/to/ellsworth_malin_fig3a.png \
#       --csv data/safod/ellsworth_malin_2011_fig3a_digitized.csv \
#       --qc results/ellsworth_malin_2011_fig3a_digitization_qc.png
#
# The calibration values below come from the visible figure ticks.

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


STRUCTURES_MD = {
    "GBF": 3150.0,
    "SDZ": 3192.0,
    "CDZ": 3302.0,
    "NBF": 3413.0,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--qc", type=Path, required=True)
    args = p.parse_args()

    rgb = np.array(Image.open(args.image).convert("RGB"))
    gray = np.array(Image.open(args.image).convert("L"))

    # Main-axis calibration.
    x_tick_px = np.array([212, 373, 534, 695, 856, 1017], float)
    md_tick_m = np.array([3100, 3200, 3300, 3400, 3500, 3600], float)
    md_slope, md_intercept = np.polyfit(x_tick_px, md_tick_m, 1)

    y_tick_px = np.array([108, 188, 268, 348.5, 428.5, 509], float)
    vel_tick_kms = np.array([6, 5, 4, 3, 2, 1], float)
    vel_slope, vel_intercept = np.polyfit(y_tick_px, vel_tick_kms, 1)

    # Lower TVD-axis calibration.
    tvd_tick_x_px = np.array([137.5, 270.5, 428.5, 581.0, 717.0, 853.0, 989.0])
    tvd_tick_m = np.array([2550, 2600, 2650, 2700, 2750, 2800, 2850], float)

    def x_from_md(md):
        return (md - md_intercept) / md_slope

    def trace_segment(xs, y_start, y_min, y_max, forbidden_bands=()):
        ys = []
        prev = float(y_start)

        for x in xs:
            yy = np.arange(
                max(y_min, int(prev - 9)),
                min(y_max, int(prev + 9)) + 1,
            )

            ok = np.ones(len(yy), dtype=bool)
            for a, b in forbidden_bands:
                ok &= ~((yy >= a) & (yy <= b))

            dark = (gray[yy, x] < 180) & ok

            if dark.any():
                cand = yy[dark]
                score = np.abs(cand - prev) + 0.015 * gray[cand, x]
                chosen = float(cand[np.argmin(score)])
            else:
                yy = np.arange(
                    max(y_min, int(prev - 60)),
                    min(y_max, int(prev + 60)) + 1,
                )

                ok = np.ones(len(yy), dtype=bool)
                for a, b in forbidden_bands:
                    ok &= ~((yy >= a) & (yy <= b))

                cand = yy[(gray[yy, x] < 200) & ok]

                if len(cand):
                    score = 0.35 * np.abs(cand - prev) + 0.02 * gray[cand, x]
                    chosen = float(cand[np.argmin(score)])
                else:
                    chosen = prev

            prev = chosen
            ys.append(chosen)

        return np.asarray(ys)

    xs = np.arange(133, 1017)

    vp_y = np.full(xs.size, np.nan)
    vp_segments = [
        (133, 292, 160, 135, 280, ()),
        (294, 360, 245, 170, 280, ()),
        (362, 537, 250, 220, 300, ()),
        (539, 716, 330, 205, 345, ((246, 255),)),
        (718, 1016, 228, 200, 290, ((246, 255),)),
    ]

    for xa, xb, y0, ymin, ymax, forbidden in vp_segments:
        yy = trace_segment(
            np.arange(xa, xb + 1),
            y0,
            ymin,
            ymax,
            forbidden,
        )
        mask = (xs >= xa) & (xs <= xb)
        vp_y[mask] = yy

    vp_y = pd.Series(vp_y).interpolate(limit_direction="both").to_numpy()
    vs_y = trace_segment(xs, 340, 325, 455)

    md_px = md_slope * xs + md_intercept
    vp_px = vel_slope * vp_y + vel_intercept
    vs_px = vel_slope * vs_y + vel_intercept

    md_grid = np.arange(np.ceil(md_px.min()), np.floor(md_px.max()) + 1.0, 1.0)
    vp_kms = np.interp(md_grid, md_px, vp_px)
    vs_kms = np.interp(md_grid, md_px, vs_px)

    x_grid = (md_grid - md_intercept) / md_slope

    def interp_extrap(xnew, xp, fp):
        out = np.interp(xnew, xp, fp)
        left = xnew < xp[0]
        right = xnew > xp[-1]
        out[left] = fp[0] + (xnew[left] - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])
        out[right] = fp[-1] + (xnew[right] - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        return out

    tvd_m = interp_extrap(x_grid, tvd_tick_x_px, tvd_tick_m)

    # Preserve narrow, near-vertical core minima.
    def local_core_min(md0, ylo, yhi):
        xc = x_from_md(md0)
        xx = np.arange(int(round(xc)) - 2, int(round(xc)) + 3)
        ys = []
        for x in xx:
            yy = np.where(gray[ylo:yhi + 1, x] < 120)[0] + ylo
            ys.extend(yy.tolist())
        if not ys:
            return np.nan
        return vel_slope * max(ys) + vel_intercept

    for name, md0 in STRUCTURES_MD.items():
        i = int(np.argmin(np.abs(md_grid - md0)))
        vp_kms[i] = local_core_min(md0, 145, 330)
        vs_kms[i] = local_core_min(md0, 330, 460)

    # Reconstruct approximate horizontal section coordinate from MD and TVD.
    cable_end_tvd_m = 2549.5
    iref = int(np.argmin(np.abs(tvd_m - cable_end_tvd_m)))

    x_rel = np.zeros_like(md_grid)
    for i in range(iref + 1, len(md_grid)):
        dmd = md_grid[i] - md_grid[i - 1]
        dz = tvd_m[i] - tvd_m[i - 1]
        x_rel[i] = x_rel[i - 1] + np.sqrt(max(dmd*dmd - dz*dz, 0.0))

    for i in range(iref - 1, -1, -1):
        dmd = md_grid[i + 1] - md_grid[i]
        dz = tvd_m[i + 1] - tvd_m[i]
        x_rel[i] = x_rel[i + 1] - np.sqrt(max(dmd*dmd - dz*dz, 0.0))

    i_sdz = int(np.argmin(np.abs(md_grid - STRUCTURES_MD["SDZ"])))
    offset_sdz = x_rel - x_rel[i_sdz]

    df = pd.DataFrame({
        "measured_depth_m": md_grid,
        "tvd_m": tvd_m,
        "section_x_from_cable_end_m": x_rel,
        "section_offset_from_sdz_m": offset_sdz,
        "vp_mps": vp_kms * 1000.0,
        "vs_mps": vs_kms * 1000.0,
    })

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False, float_format="%.6f")

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.imshow(rgb)
    ax.plot(xs, vp_y, lw=1.2, label="digitized Vp")
    ax.plot(xs, vs_y, lw=1.2, label="digitized Vs")
    for md0 in STRUCTURES_MD.values():
        ax.axvline(x_from_md(md0), lw=0.7, ls=":")
    ax.set_xlim(120, 1025)
    ax.set_ylim(550, 95)
    ax.set_axis_off()
    ax.legend(loc="lower right")
    fig.tight_layout()

    args.qc.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.qc, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(args.csv)
    print(args.qc)


if __name__ == "__main__":
    main()
