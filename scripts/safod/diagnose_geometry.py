# ==============================================================================
# scripts/diagnose_channel_geometry.py
#
# Diagnose whether the georeferenced channel table (GEO_XLSX) is monotonic
# in TVD when sorted by Channel number. Non-monotonicity would corrupt:
#   - the arc-length parameterization used by build_das_cable()
#     (observed: 6000 m arc length vs a physically max-possible ~3662 m
#      for the reported x/TVD extents -- see prior conversation)
#   - the lat0/lon0 reference point (surf = geo.iloc[0]) used to anchor
#     the entire along/cross-profile projection, including the event
#     projection in project_event_to_model()
# ==============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd

GEO_XLSX = "/home/groups/ettore88/alina/SAFOD/SAFOD_Phase2_GeoReferenced_Channels.xlsx"


def main() -> None:
    geo = pd.read_excel(GEO_XLSX)

    required = ["Channel", "Lat_WGS84", "Lon_WGS84", "TVD_m", "MD_m", "Horiz_Disp_m"]
    for col in required:
        if col not in geo.columns:
            raise ValueError(f"Missing column {col!r}")
        geo[col] = pd.to_numeric(geo[col], errors="coerce")

    geo = geo.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["Channel", "Lat_WGS84", "Lon_WGS84", "TVD_m"]
    )
    geo = geo.sort_values("Channel")
    geo = geo.groupby("Channel", as_index=False).median(numeric_only=True)

    channel = geo["Channel"].to_numpy()
    tvd = geo["TVD_m"].to_numpy()
    md = geo["MD_m"].to_numpy()

    print("Basic range check")
    print("------------------")
    print(f"n rows                    : {len(geo)}")
    print(f"channel range             : {channel.min():.1f} to {channel.max():.1f}")
    print(f"TVD at first channel (iloc[0])  : {tvd[0]:.1f} m")
    print(f"TVD at last channel (iloc[-1])  : {tvd[-1]:.1f} m")
    print(f"TVD min / max                    : {tvd.min():.1f} / {tvd.max():.1f} m")
    print()

    # ── monotonicity check ────────────────────────────────────────────────────
    d_tvd = np.diff(tvd)
    n_decreasing = int(np.sum(d_tvd < -1.0))   # allow 1 m noise tolerance
    max_backward_jump = float(np.min(d_tvd)) if len(d_tvd) else 0.0

    print("TVD monotonicity vs Channel order")
    print("----------------------------------")
    print(f"segments with TVD decreasing >1m : {n_decreasing} / {len(d_tvd)}")
    print(f"largest backward jump in TVD     : {max_backward_jump:.1f} m")
    print(f"is fully monotonic (non-decreasing): {n_decreasing == 0}")
    print()

    # ── arc length check: does MD_m (measured depth along fiber) already
    #    give the TRUE along-fiber distance, independent of channel order? ────
    if not np.all(np.isnan(md)):
        md_valid = md[~np.isnan(md)]
        print("Measured depth (MD_m) cross-check")
        print("----------------------------------")
        print(f"MD_m range        : {np.nanmin(md):.1f} to {np.nanmax(md):.1f} m")
        print(f"MD_m span (should match true along-fiber cable length): "
              f"{np.nanmax(md) - np.nanmin(md):.1f} m")
        print("If this MD_m span differs a lot from the ~6000 m arc length seen")
        print("in the gather plots, MD_m is the trustworthy along-fiber distance")
        print("and channel order is NOT giving a clean physical path.")
    else:
        print("MD_m column is all-NaN for these rows; cannot cross-check.")

    print()
    # ── straight-line vs reported arc-length sanity ───────────────────────────
    x_range = float(np.nanmax(geo["Channel"]) - np.nanmin(geo["Channel"]))  # placeholder
    print("Reminder: max possible arc length for ANY smooth monotonic path")
    print("is (x_range + TVD_range). Compare this to receivers.s[-1]-receivers.s[0]")
    print("printed by build_das_cable() / prepare_real_event script.")


if __name__ == "__main__":
    main()