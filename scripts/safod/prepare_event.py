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

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import URL_MAPPINGS


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
from scripts.safod.settings import (
    COMMON_FMAX_HZ,
    COMMON_FMIN_HZ,
    DAS_DB_PY,
    DAS_SYSTEM_NAME,
    EVENT,
    FILTER_ORDER,
    FILTER_TAPER_FRAC,
    GEO_XLSX,
    PREP_TMAX_S,
    PREP_TMIN_S,
    REAL_EVENT_DIR,
    REAL_EVENT_PACKAGE,
    SELECTED_FILES,
)
from src.signal_processing import bandpass_traces

OUT_DIR = REAL_EVENT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

FMIN = COMMON_FMIN_HZ
FMAX = COMMON_FMAX_HZ
TMIN = PREP_TMIN_S
TMAX = PREP_TMAX_S
PCLIP = 96.0

# ==============================================================================
# HEADER / METADATA HELPERS
# ==============================================================================

def read_das_db_headers(selected_files: list[str]) -> pd.DataFrame:
    """
    Read DAS headers using the same parser as DAS_db.py.

    For OptaSense files, this reads:
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
    beg = info["begTime"]

    if isinstance(beg, datetime.datetime):
        if beg.tzinfo is None:
            beg = beg.replace(tzinfo=datetime.timezone.utc)
        return beg

    beg = dateutil.parser.parse(str(beg))
    if beg.tzinfo is None:
        beg = beg.replace(tzinfo=datetime.timezone.utc)

    return beg


def parse_catalog_time_to_datetime(t: str) -> datetime.datetime:
    dt = dateutil.parser.isoparse(t)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt


# ==============================================================================
# NCEDC EVENT METADATA
# ==============================================================================

def fetch_event_from_ncedc() -> dict:
    """
    Fetch event metadata from NCEDC using event id, with time-window fallback.
    """
    URL_MAPPINGS["NCEDC"] = "https://service.ncedc.org"

    client = Client("NCEDC")

    event_id_raw = str(EVENT["event_id"])
    ncedc_event_id = int(EVENT["ncedc_event_id"])

    cat = None

    try:
        cat = client.get_events(eventid=ncedc_event_id)
    except Exception as exc:
        print(
            "NCEDC eventid lookup failed for "
            f"{ncedc_event_id!r}: {exc}"
        )

    if cat is None or len(cat) == 0:
        t0 = UTCDateTime(EVENT["origin_time"])
        print("Falling back to time-window NCEDC query around origin time.")

        cat = client.get_events(
            starttime=t0 - 10.0,
            endtime=t0 + 10.0,
            latitude=35.9741971,
            longitude=-120.5521278,
            maxradius=0.20,
            minmagnitude=0.0,
            maxdepth=20.0,
            orderby="time",
        )

    if len(cat) == 0:
        raise RuntimeError("NCEDC query returned no events for NC75336802.")

    expected_t = UTCDateTime(EVENT["origin_time"])

    best_ev = None
    best_dt = np.inf

    for ev in cat:
        ori = ev.preferred_origin() or ev.origins[0]
        dt_abs = abs(float(ori.time - expected_t))

        if dt_abs < best_dt:
            best_dt = dt_abs
            best_ev = ev

    if best_ev is None:
        raise RuntimeError("Could not select event from NCEDC catalog response.")

    ori = best_ev.preferred_origin() or best_ev.origins[0]
    mag = best_ev.preferred_magnitude() or best_ev.magnitudes[0]

    meta = {
        "event_id": event_id_raw,
        "origin_time": ori.time.isoformat(),
        "lat": float(ori.latitude),
        "lon": float(ori.longitude),
        "depth_km": float(ori.depth) / 1000.0,
        "mag": float(mag.mag),
        "mag_type": str(mag.magnitude_type or ""),
    }

    print("\nNCEDC event metadata")
    print("--------------------")
    print(f"event id       : {meta['event_id']}")
    print(f"origin         : {meta['origin_time']}")
    print(f"lat/lon/depth  : {meta['lat']:.6f}, {meta['lon']:.6f}, {meta['depth_km']:.3f} km")
    print(f"magnitude      : M{meta['mag']:.2f} {meta['mag_type']}")
    print(f"time mismatch  : {best_dt:.3f} s")

    if best_dt > 2.0:
        print(
            "WARNING: selected NCEDC event differs from EVENT['origin_time'] "
            f"by {best_dt:.3f} s."
        )

    return meta


# ==============================================================================
# GEOMETRY / PROJECTION HELPERS
# ==============================================================================

def latlon_to_local_enu_m(lat, lon, lat0, lon0):
    """
    Small-area local tangent-plane approximation.

    Returns east/north offsets in metres relative to lat0/lon0.
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

    SAFOD Phase2 / DualPulse channel table contains:
        surface spool -> down-going borehole pass -> up-going return pass.

    For 2D elastic forward modelling we keep only:
        down-going borehole pass,
    and remove surface spool / repeated near-surface zero-TVD channels.

    Resulting model coordinates:
        X_2D_m = along-profile horizontal coordinate [m]
        Z_2D_m = TVD depth [m]
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
        subset=["Channel", "Lat_WGS84", "Lon_WGS84", "TVD_m", "MD_m"]
    )

    geo = geo.sort_values("Channel")
    geo = geo.groupby("Channel", as_index=False).median(numeric_only=True)

    if len(geo) < 10:
        raise RuntimeError(f"Too few georeferenced rows: {len(geo)}")

    # --------------------------------------------------------------------------
    # 1. Detect DualPulse turn-around and keep only down-going pass
    # --------------------------------------------------------------------------
    tvd_full = geo["TVD_m"].to_numpy(dtype=np.float64)
    md_full = geo["MD_m"].to_numpy(dtype=np.float64)
    channel_full = geo["Channel"].to_numpy(dtype=np.float64)

    turn_idx = int(np.nanargmax(tvd_full))
    turn_channel = float(channel_full[turn_idx])
    turn_tvd_m = float(tvd_full[turn_idx])
    turn_md_m = float(md_full[turn_idx])

    n_rows_full = len(geo)

    geo_down = geo.iloc[: turn_idx + 1].copy()
    n_rows_downleg_including_spool = len(geo_down)

    # --------------------------------------------------------------------------
    # 2. Remove surface spool / repeated zero-depth channels
    # --------------------------------------------------------------------------
    MIN_BOREHOLE_TVD_M = 10.0
    MIN_BOREHOLE_MD_M = 10.0

    borehole_mask = (
        (geo_down["TVD_m"].to_numpy(dtype=np.float64) >= MIN_BOREHOLE_TVD_M)
        & (geo_down["MD_m"].to_numpy(dtype=np.float64) >= MIN_BOREHOLE_MD_M)
    )

    if np.count_nonzero(borehole_mask) < 10:
        raise RuntimeError(
            "Too few borehole rows after removing surface spool. "
            f"Rows left: {np.count_nonzero(borehole_mask)}"
        )

    first_borehole_pos = int(np.argmax(borehole_mask))
    first_borehole_channel = float(geo_down.iloc[first_borehole_pos]["Channel"])

    geo_model = geo_down.iloc[first_borehole_pos:].copy()
    n_rows_downleg_model = len(geo_model)

    d_tvd_model = np.diff(geo_model["TVD_m"].to_numpy(dtype=np.float64))
    n_decreasing = int(np.sum(d_tvd_model < -1.0))

    print("\nDual-pass / surface-spool geometry fix")
    print("--------------------------------------")
    print(f"rows before truncation       : {n_rows_full}")
    print(f"turn-around row index        : {turn_idx}")
    print(f"turn-around channel          : {turn_channel:.1f}")
    print(f"turn-around TVD / MD         : {turn_tvd_m:.1f} / {turn_md_m:.1f} m")
    print(f"rows down-pass incl. spool   : {n_rows_downleg_including_spool}")
    print(f"first borehole channel kept  : {first_borehole_channel:.1f}")
    print(f"rows after spool trimming    : {n_rows_downleg_model}")
    print(f"TVD-decreasing segments left : {n_decreasing}")

    if n_decreasing > 0:
        print(
            "WARNING: down-going pass is still not perfectly monotonic in TVD. "
            "Small local survey/interpolation noise may be okay, but check geometry."
        )

    # --------------------------------------------------------------------------
    # 3. Projection reference and profile direction
    # --------------------------------------------------------------------------
    # Use original channel 0 as physical surface reference.
    surf = geo.iloc[0]
    lat0 = float(surf["Lat_WGS84"])
    lon0 = float(surf["Lon_WGS84"])

    east_down, north_down = latlon_to_local_enu_m(
        geo_down["Lat_WGS84"].to_numpy(dtype=np.float64),
        geo_down["Lon_WGS84"].to_numpy(dtype=np.float64),
        lat0,
        lon0,
    )

    tvd_down = geo_down["TVD_m"].to_numpy(dtype=np.float64)

    deep_mask = tvd_down > np.nanpercentile(tvd_down, 90.0)
    if np.count_nonzero(deep_mask) < 5:
        deep_mask = tvd_down > np.nanpercentile(tvd_down, 80.0)

    e_deep = float(np.nanmedian(east_down[deep_mask]))
    n_deep = float(np.nanmedian(north_down[deep_mask]))

    norm = float(np.hypot(e_deep, n_deep))
    if norm <= 0.0 or not np.isfinite(norm):
        raise RuntimeError("Could not define profile direction from georeferenced cable.")

    u_e = e_deep / norm
    u_n = n_deep / norm

    east, north = latlon_to_local_enu_m(
        geo_model["Lat_WGS84"].to_numpy(dtype=np.float64),
        geo_model["Lon_WGS84"].to_numpy(dtype=np.float64),
        lat0,
        lon0,
    )

    tvd = geo_model["TVD_m"].to_numpy(dtype=np.float64)

    along = east * u_e + north * u_n
    cross = -east * u_n + north * u_e

    model_x = along
    model_z = tvd

    out = geo_model.copy()
    out["east_m_from_surface"] = east
    out["north_m_from_surface"] = north
    out["along_profile_m"] = along
    out["cross_profile_m"] = cross
    out["X_2D_m"] = model_x
    out["Z_2D_m"] = model_z

    out_csv = OUT_DIR / "SAFOD_Phase2_projected_from_georef.csv"
    out.to_csv(out_csv, index=False)

    # --------------------------------------------------------------------------
    # 4. Geometry QC
    # --------------------------------------------------------------------------
    dx_seg = np.diff(model_x)
    dz_seg = np.diff(model_z)
    arc_length_m = float(np.sum(np.sqrt(dx_seg * dx_seg + dz_seg * dz_seg)))

    x_range = float(np.nanmax(model_x) - np.nanmin(model_x))
    z_range = float(np.nanmax(model_z) - np.nanmin(model_z))

    straight_bound = float(np.hypot(x_range, z_range))
    monotonic_bound = float(x_range + z_range)

    fit_rms_m = float(np.sqrt(np.nanmean(cross ** 2)))

    print("\nGeoreferenced channel projection QC: down-going borehole only")
    print("------------------------------------------------------------")
    print(f"geo rows used             : {len(out)}")
    print(f"surface lat/lon           : {lat0:.7f}, {lon0:.7f}")
    print(f"profile unit vector EN    : ({u_e:.4f}, {u_n:.4f})")
    print(f"channel range used        : {out['Channel'].min():.1f} to {out['Channel'].max():.1f}")
    print(f"TVD range used            : {np.nanmin(model_z):.1f} to {np.nanmax(model_z):.1f} m")
    print(f"model x range used        : {np.nanmin(model_x):.1f} to {np.nanmax(model_x):.1f} m")
    print(f"computed cable arc length : {arc_length_m:.1f} m")
    print(f"arc length sanity bounds  : {straight_bound:.1f} to {monotonic_bound:.1f} m")
    print(f"crossline cable range     : {np.nanmin(cross):.1f} to {np.nanmax(cross):.1f} m")
    print(f"cable-to-profile-line RMS : {fit_rms_m:.2f} m")
    print(f"saved model geometry      : {out_csv}")

    if arc_length_m > 1.15 * monotonic_bound:
        print(
            "WARNING: computed arc length is much larger than the monotonic bound. "
            "Geometry may still include duplicated/reversed path segments."
        )

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(model_x, model_z, "k-", lw=1.5)
    ax.scatter(
        model_x[0],
        model_z[0],
        c="white",
        edgecolors="black",
        s=50,
        label="first borehole receiver",
    )
    ax.scatter(
        model_x[-1],
        model_z[-1],
        c="cyan",
        edgecolors="black",
        s=50,
        label="bottom / turn-around",
    )
    ax.set_xlabel("2D model x = along-profile coordinate [m]")
    ax.set_ylabel("TVD depth [m]")
    ax.set_title("SAFOD Phase2 down-going borehole pass projected to 2D")
    ax.set_ylim(np.nanmax(model_z), 0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "real_cable_projected_geometry_downpass_only.png", dpi=220)
    plt.close(fig)

    return {
        "lat0": lat0,
        "lon0": lon0,
        "u_e": float(u_e),
        "u_n": float(u_n),
        "geometry_csv": str(out_csv),
        "mapping_table": out,
        "fit_rms_m": fit_rms_m,
        "turn_channel": turn_channel,
        "turn_tvd_m": turn_tvd_m,
        "turn_md_m": turn_md_m,
        "first_borehole_channel": first_borehole_channel,
        "arc_length_m": arc_length_m,
        "n_rows_full": n_rows_full,
        "n_rows_downleg_including_spool": n_rows_downleg_including_spool,
        "n_rows_downleg_model": n_rows_downleg_model,
    }


def project_event_to_model(mapping: dict, event_meta: dict) -> dict:
    """
    Project real NCEDC event lat/lon/depth into the current 2D model coordinates.
    """
    east_ev, north_ev = latlon_to_local_enu_m(
        event_meta["lat"],
        event_meta["lon"],
        mapping["lat0"],
        mapping["lon0"],
    )

    along_ev = float(east_ev * mapping["u_e"] + north_ev * mapping["u_n"])
    cross_ev = float(-east_ev * mapping["u_n"] + north_ev * mapping["u_e"])

    x_model = along_ev
    z_model = event_meta["depth_km"] * 1000.0

    print("\nEvent projection into 2D model")
    print("------------------------------")
    print(f"event id                  : {event_meta['event_id']}")
    print(f"origin                    : {event_meta['origin_time']}")
    print(f"lat/lon/depth             : {event_meta['lat']:.6f}, {event_meta['lon']:.6f}, {event_meta['depth_km']:.3f} km")
    print(f"magnitude                 : M{event_meta['mag']:.2f} {event_meta['mag_type']}")
    print(f"event east/north from surf: {float(east_ev):.1f}, {float(north_ev):.1f} m")
    print(f"event along profile       : {along_ev:.1f} m")
    print(f"event crossline distance  : {cross_ev:.1f} m")
    print(f"event model x             : {x_model:.1f} m")
    print(f"event model z             : {z_model:.1f} m")

    return {
        "event_x_model_m": float(x_model),
        "event_z_model_m": float(z_model),
        "event_along_profile_m": float(along_ev),
        "event_crossline_m": float(cross_ev),
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    # --------------------------------------------------------------------------
    # 0. Fetch real catalog event metadata first
    # --------------------------------------------------------------------------
    event_meta = fetch_event_from_ncedc()
    eq_time = parse_catalog_time_to_datetime(event_meta["origin_time"])

    # --------------------------------------------------------------------------
    # 1. Read DAS event file
    # --------------------------------------------------------------------------
    print("\nReading real DAS files...")
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

    db_headers = read_das_db_headers(SELECTED_FILES)

    valid_headers = db_headers[
        (pd.to_numeric(db_headers["dCh"], errors="coerce") > 0.0)
        & (pd.to_numeric(db_headers["GaugeLen"], errors="coerce") > 0.0)
    ].copy()

    if valid_headers.empty:
        raise RuntimeError("Could not find valid dCh/GaugeLen in DAS_db headers.")

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
    print(f"catalog origin      : {event_meta['origin_time']}")
    print(f"file start rel origin: {ot:.6f} s")

    # --------------------------------------------------------------------------
    # 2. Crop the temporally unfiltered event window
    # --------------------------------------------------------------------------
    it0 = int(np.searchsorted(t_ax, TMIN))
    it1 = int(np.searchsorted(t_ax, TMAX))

    D_event_unfiltered_full = (
        np.asarray(DAS_data[:, it0:it1], dtype=np.float64) * 1e3
    )
    t_event = t_ax[it0:it1]

    print("\nReal event window before temporal filtering")
    print("-------------------------------------------")
    print(f"time range      : {t_event[0]:.3f} to {t_event[-1]:.3f} s")
    print(f"full data shape : {D_event_unfiltered_full.shape}")
    print("temporal filter : none in saved event package")

    # --------------------------------------------------------------------------
    # 3. Build down-going geometry and live event projection
    # --------------------------------------------------------------------------
    mapping = build_channel_projection_mapping()
    ev_proj = project_event_to_model(mapping, event_meta)

    # --------------------------------------------------------------------------
    # 3b. Crop real DAS to down-going borehole pass only
    # --------------------------------------------------------------------------
    raw_channels_full = np.arange(
        D_event_unfiltered_full.shape[0], dtype=np.float64
    )

    first_borehole_channel = float(mapping["first_borehole_channel"])
    turn_channel = float(mapping["turn_channel"])

    raw_max = float(raw_channels_full[-1])
    downleg_min_channel = max(first_borehole_channel, float(raw_channels_full[0]))
    downleg_max_channel = min(turn_channel, raw_max)

    downleg_mask = (
        (raw_channels_full >= downleg_min_channel)
        & (raw_channels_full <= downleg_max_channel)
    )

    D_event_unfiltered = D_event_unfiltered_full[downleg_mask, :]
    raw_channels_event = raw_channels_full[downleg_mask]

    # Preview only. Quantitative comparison filters both datasets
    # with this same zero-phase implementation.
    D_event = bandpass_traces(
        D_event_unfiltered,
        fs_hz=fs,
        fmin_hz=FMIN,
        fmax_hz=FMAX,
        order=FILTER_ORDER,
        taper_frac=FILTER_TAPER_FRAC,
    )

    print("\nReal DAS downleg crop")
    print("---------------------")
    print(f"full real data shape       : {D_event_unfiltered_full.shape}")
    print(f"first borehole channel     : {first_borehole_channel:.1f}")
    print(f"turn channel from geometry : {turn_channel:.1f}")
    print(f"downleg channel range      : {raw_channels_event[0]:.1f} to {raw_channels_event[-1]:.1f}")
    print(f"downleg data shape         : {D_event.shape}")

    # --------------------------------------------------------------------------
    # 4. Plot real event gather
    # --------------------------------------------------------------------------
    clip = np.percentile(np.abs(D_event), PCLIP)

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(
        D_event.T,
        extent=[
            float(raw_channels_event[0]),
            float(raw_channels_event[-1]),
            float(t_event[-1]),
            float(t_event[0]),
        ],
        aspect="auto",
        cmap="seismic",
        vmin=-clip,
        vmax=clip,
        interpolation="none",
    )
    fig.colorbar(im, ax=ax, label="strain rate [nm/m/s]")
    ax.axhline(0.0, color="k", lw=1.1, ls="--", label="Catalog origin")
    ax.set_xlabel("Raw channel number")
    ax.set_ylabel(f"Time from {event_meta['origin_time']} [s]")
    ax.set_title(
        f"Real SAFOD DAS event {event_meta['event_id']} "
        f"M{event_meta['mag']:.2f}, {FMIN:.0f}-{FMAX:.0f} Hz"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "real_das_event_preview_raw_channels.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    scale = np.percentile(np.abs(D_event), 99.0, axis=1, keepdims=True)
    scale = np.maximum(scale, 1e-12)
    D_norm = D_event / scale

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(
        D_norm.T,
        extent=[
            float(raw_channels_event[0]),
            float(raw_channels_event[-1]),
            float(t_event[-1]),
            float(t_event[0]),
        ],
        aspect="auto",
        cmap="seismic",
        vmin=-1,
        vmax=1,
        interpolation="none",
    )
    fig.colorbar(im, ax=ax, label="trace-normalized amplitude")
    ax.axhline(0.0, color="k", lw=1.1, ls="--", label="Catalog origin")
    ax.set_xlabel("Raw channel number")
    ax.set_ylabel(f"Time from {event_meta['origin_time']} [s]")
    ax.set_title(f"Real SAFOD DAS event {event_meta['event_id']} trace-normalized")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "real_das_event_preview_trace_normalized.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # 5. Save real-event package
    # --------------------------------------------------------------------------
    np.savez_compressed(
        REAL_EVENT_PACKAGE,

        das_data_unfiltered=D_event_unfiltered,
        das_data_preview=D_event,
        t=t_event,
        fs=np.array(fs),
        dt=np.array(dt),
        channel_spacing_m=np.array(dx_ch),
        gauge_length_m=np.array(gauge_length_m),
        raw_fs=np.array(raw_fs_db),
        desample_factor=np.array(desample_db),

        raw_channels=raw_channels_event,
        first_borehole_channel=np.array(first_borehole_channel),
        turn_channel=np.array(turn_channel),
        turn_tvd_m=np.array(mapping["turn_tvd_m"]),
        turn_md_m=np.array(mapping["turn_md_m"]),
        geometry_arc_length_m=np.array(mapping["arc_length_m"]),
        geometry_csv=np.array(mapping["geometry_csv"]),
        n_rows_full_geometry=np.array(mapping["n_rows_full"]),
        n_rows_downleg_including_spool=np.array(mapping["n_rows_downleg_including_spool"]),
        n_rows_downleg_model=np.array(mapping["n_rows_downleg_model"]),

        ev_id=np.array(event_meta["event_id"]),
        ev_origin_time=np.array(event_meta["origin_time"]),
        ev_lat=np.array(event_meta["lat"]),
        ev_lon=np.array(event_meta["lon"]),
        ev_depth_km=np.array(event_meta["depth_km"]),
        ev_mag=np.array(event_meta["mag"]),
        ev_mag_type=np.array(event_meta["mag_type"]),

        event_x_model_m=np.array(ev_proj["event_x_model_m"]),
        event_z_model_m=np.array(ev_proj["event_z_model_m"]),
        event_along_profile_m=np.array(ev_proj["event_along_profile_m"]),
        event_crossline_m=np.array(ev_proj["event_crossline_m"]),
        projection_fit_rms_m=np.array(mapping["fit_rms_m"]),

        selected_files=np.array(SELECTED_FILES),
        temporal_filter=np.array("none"),
        preview_fmin_hz=np.array(FMIN),
        preview_fmax_hz=np.array(FMAX),
        preview_filter_order=np.array(FILTER_ORDER),
        preview_filter_taper_frac=np.array(FILTER_TAPER_FRAC),
    )

    print(f"\nSaved real-event package to: {OUT_DIR.absolute()}")
    print("Next synthetic source coordinates:")
    print(f"    x_src = {ev_proj['event_x_model_m']:.3f} m")
    print(f"    z_src = {ev_proj['event_z_model_m']:.3f} m")
    print(f"    gauge_length_m = {gauge_length_m:.6f}")
    print(f"    channel_spacing_m = {dx_ch:.6f}")
    print(f"    geometry_csv = {mapping['geometry_csv']}")


if __name__ == "__main__":
    main()