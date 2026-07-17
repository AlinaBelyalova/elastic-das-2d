from __future__ import annotations

from pathlib import Path
import os
import sys
import importlib.util
import datetime
import dateutil.parser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==============================================================================
# DAS-utilities path
# ==============================================================================

pyDAS_path = "/home/groups/ettore88/alina/packages/DAS-utilities/build"
pyDAS_python_path = "/home/groups/ettore88/alina/packages/DAS-utilities/python"

try:
    os.environ["LD_LIBRARY_PATH"] += ":" + pyDAS_path
except KeyError:
    os.environ["LD_LIBRARY_PATH"] = pyDAS_path

sys.path.insert(0, pyDAS_path)
sys.path.insert(0, pyDAS_python_path)

import DASutils  # noqa: E402


# ==============================================================================
# USER SETTINGS
# ==============================================================================

EVENT = {
    "event_id": "NC75343317",
    "origin_time": "2026-04-13T06:35:49.710000Z",
    "lat": 36.019,
    "lon": -120.583333333333,
    "depth_km": 4.59,
    "mag": 1.53,
    "mag_type": "Md",
}

SELECTED_FILES = [
    "/oak/stanford/groups/ettore88/data/SAFOD/SAFOD_events/"
    "SAFOD-Deep-10mGL-1000HzFs-2mChDualPulse_2026-04-13T063449Z.h5",

    "/oak/stanford/groups/ettore88/data/SAFOD/SAFOD_events/"
    "SAFOD-Deep-10mGL-1000HzFs-2mChDualPulse_2026-04-13T063549Z.h5",
]

GEO_XLSX = "/home/groups/ettore88/alina/SAFOD/SAFOD_Phase2_GeoReferenced_Channels.xlsx"

# This is the projected 2D geometry file used by build_safod_model.
PROJECTED_CSV = "/home/groups/ettore88/alina/imaging/SAFOD_downleg_Projected_2D.csv"

# DAS_db.py from the same DAS-utilities package.
DAS_DB_PY = "/home/groups/ettore88/alina/packages/DAS-utilities/python/DAS_db.py"
DAS_SYSTEM_NAME = "SAFOD_QuantX"

OUT_DIR = Path("results/real_event_20260413_M153")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Real DAS preprocessing.
FMIN = 1.0
FMAX = 20.0

# Event window relative to TRUE catalog origin.
TMIN = 0.0
TMAX = 15.0

# Plot clipping.
PCLIP = 96.0


# ==============================================================================
# HEADER / METADATA HELPERS
# ==============================================================================

def read_das_db_headers(selected_files: list[str]) -> pd.DataFrame:
    """
    Read DAS headers using the same parser as DAS_db.py.

    This is important because readFile_HDF(info) exposes dx/fs, but not
    always GaugeLen. For OptaSense files, DAS_db.py reads:
        Acquisition.attrs["SpatialSamplingInterval"]
        Acquisition.attrs["GaugeLength"]
    """
    spec = importlib.util.spec_from_file_location("das_db_external", DAS_DB_PY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import DAS_db.py from {DAS_DB_PY}")

    das_db = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(das_db)

    rows = []
    for fn in selected_files:
        df_one = das_db.get_header_df(fn, DAS_SYSTEM_NAME)
        rows.append(df_one)

    df = pd.concat(rows, ignore_index=True)

    if df.empty:
        raise RuntimeError("DAS_db-style header read returned empty dataframe.")

    print("\nDAS_db-style file headers")
    print("-------------------------")
    print(df.to_string(index=False))

    return df


def parse_beg_time_from_info(info) -> datetime.datetime:
    """
    Return timezone-aware file begin time from DASutils info dict.
    """
    beg = info["begTime"]

    if isinstance(beg, datetime.datetime):
        if beg.tzinfo is None:
            beg = beg.replace(tzinfo=datetime.timezone.utc)
        return beg

    beg = dateutil.parser.parse(str(beg))
    if beg.tzinfo is None:
        beg = beg.replace(tzinfo=datetime.timezone.utc)
    return beg


# ==============================================================================
# PROJECTION HELPERS
# ==============================================================================

def latlon_to_local_enu_m(lat, lon, lat0, lon0):
    """
    Small-area local tangent-plane approximation.

    Returns east, north offsets in metres relative to lat0/lon0.
    Accurate enough for Parkfield-scale event/cable projection.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    r_earth = 6371000.0
    lat0_rad = np.deg2rad(lat0)

    east = np.deg2rad(lon - lon0) * r_earth * np.cos(lat0_rad)
    north = np.deg2rad(lat - lat0) * r_earth
    return east, north


def build_channel_projection_mapping():
    """
    Build real-event 2D geometry directly from the georeferenced channel table.

    For the real OptaSense event, the Excel file is the source of truth:
        raw DAS channel -> lat/lon/TVD/MD.

    We define the 2D model coordinates as:
        model x = along-profile horizontal coordinate [m]
        model z = TVD_m [m]

    This avoids the incorrect exact merge / fit with the older PROJECTED_CSV.
    """
    geo = pd.read_excel(GEO_XLSX)

    required_geo = [
        "Channel",
        "Lat_WGS84",
        "Lon_WGS84",
        "TVD_m",
        "MD_m",
        "Horiz_Disp_m",
    ]

    for col in required_geo:
        if col not in geo.columns:
            raise ValueError(f"Missing column {col!r} in {GEO_XLSX}")

    geo = geo.copy()

    for col in required_geo:
        geo[col] = pd.to_numeric(geo[col], errors="coerce")

    geo = geo.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["Channel", "Lat_WGS84", "Lon_WGS84", "TVD_m"]
    )

    if len(geo) < 10:
        raise RuntimeError(f"Too few georeferenced rows: {len(geo)}")

    # Sort by channel and remove duplicate channels if present.
    geo = geo.sort_values("Channel")
    geo = geo.groupby("Channel", as_index=False).median(numeric_only=True)

    # Surface reference.
    surf = geo.iloc[0]
    lat0 = float(surf["Lat_WGS84"])
    lon0 = float(surf["Lon_WGS84"])

    east, north = latlon_to_local_enu_m(
        geo["Lat_WGS84"].to_numpy(dtype=np.float64),
        geo["Lon_WGS84"].to_numpy(dtype=np.float64),
        lat0,
        lon0,
    )

    tvd = geo["TVD_m"].to_numpy(dtype=np.float64)

    # Define profile direction from surface to deepest/deviated part.
    deep_mask = tvd > np.nanpercentile(tvd, 90.0)
    if np.count_nonzero(deep_mask) < 5:
        deep_mask = tvd > np.nanpercentile(tvd, 80.0)

    e_deep = float(np.nanmedian(east[deep_mask]))
    n_deep = float(np.nanmedian(north[deep_mask]))

    norm = float(np.hypot(e_deep, n_deep))
    if norm <= 0.0 or not np.isfinite(norm):
        raise RuntimeError("Could not define profile direction from georeferenced cable.")

    u_e = e_deep / norm
    u_n = n_deep / norm

    along = east * u_e + north * u_n
    cross = -east * u_n + north * u_e

    # For real-event modelling, use direct 2D geometry:
    # x = along-profile horizontal coordinate, z = TVD.
    model_x = along
    model_z = tvd

    out = geo.copy()
    out["east_m_from_surface"] = east
    out["north_m_from_surface"] = north
    out["along_profile_m"] = along
    out["cross_profile_m"] = cross
    out["X_2D_m"] = model_x
    out["Z_2D_m"] = model_z

    # Save a model-ready geometry file.
    # IMPORTANT: for this file, use x_column="X_2D_m", z_column="Z_2D_m".
    out_csv = OUT_DIR / "SAFOD_Phase2_projected_from_georef.csv"
    out.to_csv(out_csv, index=False)

    print("\nGeoreferenced channel projection QC")
    print("-----------------------------------")
    print(f"geo rows                 : {len(out)}")
    print(f"surface lat/lon          : {lat0:.7f}, {lon0:.7f}")
    print(f"profile unit vector EN   : ({u_e:.4f}, {u_n:.4f})")
    print(f"channel range            : {out['Channel'].min():.1f} to {out['Channel'].max():.1f}")
    print(f"TVD range                : {np.nanmin(model_z):.1f} to {np.nanmax(model_z):.1f} m")
    print(f"model x range            : {np.nanmin(model_x):.1f} to {np.nanmax(model_x):.1f} m")
    print(f"crossline cable range    : {np.nanmin(cross):.1f} to {np.nanmax(cross):.1f} m")
    print(f"saved model geometry     : {out_csv}")

    # Quick geometry plot.
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(model_x, model_z, "k-", lw=1.5)
    ax.scatter(model_x[0], model_z[0], c="white", edgecolors="black", s=50, label="start")
    ax.scatter(model_x[-1], model_z[-1], c="cyan", edgecolors="black", s=50, label="end")
    ax.set_xlabel("2D model x = along-profile coordinate [m]")
    ax.set_ylabel("TVD depth [m]")
    ax.set_title("SAFOD Phase2 georeferenced cable projected to 2D")
    ax.set_ylim(np.nanmax(model_z), 0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "real_cable_projected_geometry_from_georef.png", dpi=220)
    plt.close(fig)

    fit_rms_m = float(np.sqrt(np.nanmean(cross**2)))

    print(f"cable-to-profile-line RMS: {fit_rms_m:.2f} m")

    return {
        "lat0": lat0,
        "lon0": lon0,
        "u_e": float(u_e),
        "u_n": float(u_n),
        "geometry_csv": str(out_csv),
        "mapping_table": out,
        "fit_rms_m": fit_rms_m,
    }
    


def project_event_to_model(mapping):
    """
    Project event lat/lon/depth into the real-event 2D model coordinates.

    x_model = event along-profile coordinate [m]
    z_model = event depth [m]

    crossline distance is retained as QC. If it is large, the 2D synthetic
    will be only approximate because the real source is out of plane.
    """
    east_ev, north_ev = latlon_to_local_enu_m(
        EVENT["lat"],
        EVENT["lon"],
        mapping["lat0"],
        mapping["lon0"],
    )

    along_ev = east_ev * mapping["u_e"] + north_ev * mapping["u_n"]
    cross_ev = -east_ev * mapping["u_n"] + north_ev * mapping["u_e"]

    x_model = float(along_ev)
    z_model = float(EVENT["depth_km"] * 1000.0)

    print("\nEvent projection into 2D model")
    print("------------------------------")
    print(f"event id                  : {EVENT['event_id']}")
    print(f"origin                    : {EVENT['origin_time']}")
    print(f"lat/lon/depth             : {EVENT['lat']:.6f}, {EVENT['lon']:.6f}, {EVENT['depth_km']:.2f} km")
    print(f"magnitude                 : M{EVENT['mag']:.2f} {EVENT['mag_type']}")
    print(f"event east/north from surf: {float(east_ev):.1f}, {float(north_ev):.1f} m")
    print(f"event along profile       : {float(along_ev):.1f} m")
    print(f"event crossline distance  : {float(cross_ev):.1f} m")
    print(f"event model x             : {x_model:.1f} m")
    print(f"event model z             : {z_model:.1f} m")

    if abs(float(cross_ev)) > 3000.0:
        print(
            "WARNING: event is >3 km out of the 2D profile plane. "
            "A 2D synthetic will ignore this crossline distance and will likely "
            "predict arrivals too early unless treated only as a qualitative moveout test."
        )

    return {
        "event_x_model_m": x_model,
        "event_z_model_m": z_model,
        "event_along_profile_m": float(along_ev),
        "event_crossline_m": float(cross_ev),
    }

# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    eq_time = dateutil.parser.isoparse(EVENT["origin_time"])

    # --------------------------------------------------------------------------
    # 1. Read DAS event files
    # --------------------------------------------------------------------------
    print("Reading real DAS files...")
    DAS_data, info = DASutils.readFile_HDF(
        SELECTED_FILES,
        0.01,
        500.0,
        verbose=1,
        diff=True,
        detrend=False,
        tapering=False,
        filter=False,
        median=True,
        desampling=False,
        nChbuffer=1000,
        system="OptaSense",
    )

    # Read acquisition headers using DAS_db.py logic.
    db_headers = read_das_db_headers(SELECTED_FILES)

    valid_headers = db_headers[
        (pd.to_numeric(db_headers["dCh"], errors="coerce") > 0.0)
        & (pd.to_numeric(db_headers["GaugeLen"], errors="coerce") > 0.0)
    ].copy()

    if valid_headers.empty:
        raise RuntimeError(
            "Could not find valid dCh/GaugeLen in DAS_db headers. "
            "Check DAS_db parsing for these OptaSense files."
        )

    raw_fs_db = float(np.median(pd.to_numeric(valid_headers["fs"], errors="coerce")))
    desample_db = float(np.median(pd.to_numeric(valid_headers["Desample"], errors="coerce")))
    dx_ch_db = float(np.median(pd.to_numeric(valid_headers["dCh"], errors="coerce")))
    gauge_length_db = float(np.median(pd.to_numeric(valid_headers["GaugeLen"], errors="coerce")))

    fs = float(info["fs"])
    dt = 1.0 / fs
    dx_ch = dx_ch_db
    gauge_length_m = gauge_length_db
    nt = DAS_data.shape[1]

    beg_time = parse_beg_time_from_info(info)
    ot = (beg_time - eq_time).total_seconds()
    t_ax = ot + np.arange(nt, dtype=np.float64) * dt

    print("\nDAS acquisition parameters from DAS_db")
    print("-------------------------------------")
    print(f"raw fs              : {raw_fs_db:.3f} Hz")
    print(f"desample factor     : {desample_db:.3f}")
    print(f"read fs             : {fs:.3f} Hz")
    print(f"channel spacing dCh : {dx_ch:.6f} m")
    print(f"gauge length        : {gauge_length_m:.6f} m")

    print("\nReal DAS metadata")
    print("-----------------")
    print(f"fs                  : {fs:.3f} Hz")
    print(f"dt                  : {dt:.6f} s")
    print(f"dx channel           : {dx_ch:.6f} m")
    print(f"gauge length         : {gauge_length_m:.6f} m")
    print(f"shape               : {DAS_data.shape}")
    print(f"begTime             : {info['begTime']}")
    print(f"endTime             : {info['endTime']}")
    print(f"origin              : {EVENT['origin_time']}")
    print(f"file start rel origin: {ot:.6f} s")

    # --------------------------------------------------------------------------
    # 2. Bandpass and crop
    # --------------------------------------------------------------------------
    print("\nFiltering real DAS...")
    DAS_proc = DASutils.bandpass2D_c(
        DAS_data[:, :],
        FMIN,
        FMAX,
        1.0 / fs,
        zerophase=False,
    ) * 1e3

    it0 = int(np.searchsorted(t_ax, TMIN))
    it1 = int(np.searchsorted(t_ax, TMAX))

    D_event = DAS_proc[:, it0:it1]
    t_event = t_ax[it0:it1]

    print("\nReal event window")
    print("-----------------")
    print(f"time range          : {t_event[0]:.3f} to {t_event[-1]:.3f} s")
    print(f"data shape          : {D_event.shape}")
    print(f"bandpass            : {FMIN:.1f} to {FMAX:.1f} Hz")

    # --------------------------------------------------------------------------
    # 3. Build channel/event projection mapping
    # --------------------------------------------------------------------------
    mapping = build_channel_projection_mapping()
    ev_proj = project_event_to_model(mapping)

    # --------------------------------------------------------------------------
    # 4. Plot real event gather
    # --------------------------------------------------------------------------
    clip = np.percentile(np.abs(D_event), PCLIP)

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(
        D_event.T,
        extent=[0, D_event.shape[0] - 1, t_event[-1], t_event[0]],
        aspect="auto",
        cmap="seismic",
        vmin=-clip,
        vmax=clip,
        interpolation="none",
    )
    fig.colorbar(im, ax=ax, label="strain rate [nm/m/s]")
    ax.axhline(0.0, color="k", lw=1.1, ls="--", label="Catalog origin")
    ax.set_xlabel("Raw channel number")
    ax.set_ylabel(f"Time from {EVENT['origin_time']} [s]")
    ax.set_title(
        f"Real SAFOD DAS event {EVENT['event_id']} "
        f"M{EVENT['mag']:.2f}, {FMIN:.0f}-{FMAX:.0f} Hz"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "real_das_event_0_15s_raw_channels.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Trace-normalized display.
    scale = np.percentile(np.abs(D_event), 99.0, axis=1, keepdims=True)
    scale = np.maximum(scale, 1e-12)
    D_norm = D_event / scale

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(
        D_norm.T,
        extent=[0, D_norm.shape[0] - 1, t_event[-1], t_event[0]],
        aspect="auto",
        cmap="seismic",
        vmin=-1,
        vmax=1,
        interpolation="none",
    )
    fig.colorbar(im, ax=ax, label="trace-normalized amplitude")
    ax.axhline(0.0, color="k", lw=1.1, ls="--", label="Catalog origin")
    ax.set_xlabel("Raw channel number")
    ax.set_ylabel(f"Time from {EVENT['origin_time']} [s]")
    ax.set_title(f"Real SAFOD DAS event {EVENT['event_id']} trace-normalized")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "real_das_event_0_15s_trace_normalized.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # 5. Save real-event package
    # --------------------------------------------------------------------------
    np.savez_compressed(
        OUT_DIR / "real_das_event_window_0_15s.npz",
        das_data=D_event,
        t=t_event,
        fs=np.array(fs),
        dt=np.array(dt),
        channel_spacing_m=np.array(dx_ch),
        gauge_length_m=np.array(gauge_length_m),
        raw_fs=np.array(raw_fs_db),
        desample_factor=np.array(desample_db),

        ev_id=np.array(EVENT["event_id"]),
        ev_origin_time=np.array(EVENT["origin_time"]),
        ev_lat=np.array(EVENT["lat"]),
        ev_lon=np.array(EVENT["lon"]),
        ev_depth_km=np.array(EVENT["depth_km"]),
        ev_mag=np.array(EVENT["mag"]),
        ev_mag_type=np.array(EVENT["mag_type"]),

        event_x_model_m=np.array(ev_proj["event_x_model_m"]),
        event_z_model_m=np.array(ev_proj["event_z_model_m"]),
        event_along_profile_m=np.array(ev_proj["event_along_profile_m"]),
        event_crossline_m=np.array(ev_proj["event_crossline_m"]),
        projection_fit_rms_m=np.array(mapping["fit_rms_m"]),

        selected_files=np.array(SELECTED_FILES),
        fmin=np.array(FMIN),
        fmax=np.array(FMAX),
    )

    print(f"\nSaved real-event package to: {OUT_DIR.absolute()}")
    print("Next synthetic source coordinates:")
    print(f"    x_src = {ev_proj['event_x_model_m']:.3f} m")
    print(f"    z_src = {ev_proj['event_z_model_m']:.3f} m")
    print(f"    gauge_length_m = {gauge_length_m:.6f}")
    print(f"    channel_spacing_m = {dx_ch:.6f}")


if __name__ == "__main__":
    main()