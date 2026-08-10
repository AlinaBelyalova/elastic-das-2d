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
    CHANNEL_MAPPING_CSV,
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
    SAFOD_SURFACE_ELEVATION_M,
    SAFOD_WELLHEAD_LAT_WGS84,
    SAFOD_WELLHEAD_LON_WGS84,
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

# Full registered deep-cable visual QC. The plotted x coordinate is the
# immutable physical reference-channel coordinate from GEO_XLSX, not the
# HDF5 row number. This makes April and June directly comparable even though
# dCh and interrogator registration changed.
FULL_CABLE_QC_TMIN_S = -0.50
FULL_CABLE_QC_TMAX_S = 3.00
FULL_CABLE_QC_FILTER_PAD_S = 2.00
FULL_CABLE_QC_PCLIP = 96.0

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
            latitude=SAFOD_WELLHEAD_LAT_WGS84,
            longitude=SAFOD_WELLHEAD_LON_WGS84,
            maxradius=0.20,
            minmagnitude=0.0,
            maxdepth=20.0,
            orderby="time",
        )

    if len(cat) == 0:
        raise RuntimeError(
            f"NCEDC query returned no events for {event_id_raw}."
        )

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

MIN_BOREHOLE_TVD_M = 10.0
MIN_BOREHOLE_MD_M = 10.0


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


def _integer_index(values, *, name: str) -> np.ndarray:
    """Convert an integer-valued column to a validated int64 array."""
    values_float = pd.to_numeric(
        values,
        errors="raise",
    ).to_numpy(dtype=np.float64)

    values_int = np.rint(values_float).astype(np.int64)

    if not np.allclose(
        values_float,
        values_int,
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise ValueError(
            f"{name} values must be integer-valued."
        )

    return values_int


def _load_reference_geometry_context() -> dict:
    """
    Load the one unchanged physical SAFOD cable geometry and define the
    common 2D profile coordinate system.

    This function is independent of the interrogator configuration.
    """
    geo = pd.read_excel(
        GEO_XLSX,
        engine="openpyxl",
    )

    required_numeric = [
        "Channel",
        "Lat_WGS84",
        "Lon_WGS84",
        "TVD_m",
        "MD_m",
    ]
    required_columns = [
        *required_numeric,
        "Section",
    ]

    missing = sorted(
        set(required_columns).difference(geo.columns)
    )

    if missing:
        raise ValueError(
            f"Reference geometry is missing columns {missing}: {GEO_XLSX}"
        )

    geo = geo.copy()

    for column in required_numeric:
        geo[column] = pd.to_numeric(
            geo[column],
            errors="coerce",
        )

    geo["Section"] = (
        geo["Section"]
        .astype(str)
        .str.strip()
    )

    geo = (
        geo
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=required_numeric)
        .sort_values("Channel")
        .reset_index(drop=True)
    )

    # Channel is the immutable physical cable coordinate. Do not collapse the
    # table with median(numeric_only=True): that would discard the categorical
    # Section column required to distinguish spool, down-leg, and up-leg.
    duplicated_channels = geo["Channel"].duplicated(
        keep=False
    )

    if np.any(duplicated_channels):
        duplicate_values = (
            geo.loc[duplicated_channels, "Channel"]
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(
            "Reference geometry contains duplicate physical Channel values: "
            f"{duplicate_values[:10]}. Resolve the geometry table explicitly "
            "instead of averaging rows and losing Section identity."
        )

    expected_sections = {
        "Surface Spool",
        "Down-leg",
        "Up-leg",
    }
    found_sections = set(
        geo["Section"].unique()
    )
    unexpected_sections = sorted(
        found_sections.difference(expected_sections)
    )

    if unexpected_sections:
        raise ValueError(
            "Reference geometry contains unexpected Section labels: "
            f"{unexpected_sections}. Expected {sorted(expected_sections)}."
        )

    for section_name in sorted(expected_sections):
        if not np.any(
            geo["Section"].to_numpy() == section_name
        ):
            raise ValueError(
                f"Reference geometry contains no rows for Section "
                f"{section_name!r}."
            )

    if len(geo) < 10:
        raise RuntimeError(
            f"Too few valid rows in reference geometry: {len(geo)}"
        )

    tvd_full = geo["TVD_m"].to_numpy(dtype=np.float64)
    md_full = geo["MD_m"].to_numpy(dtype=np.float64)
    channel_full = geo["Channel"].to_numpy(dtype=np.float64)

    turn_idx = int(np.nanargmax(tvd_full))

    turn_section = str(
        geo.iloc[turn_idx]["Section"]
    )
    if turn_section != "Down-leg":
        raise RuntimeError(
            "Deepest reference-cable point is not labelled Down-leg: "
            f"channel={channel_full[turn_idx]:.1f}, "
            f"Section={turn_section!r}."
        )

    geo_down = geo.iloc[: turn_idx + 1].copy()

    # The down-going table must not contain up-leg samples.
    if np.any(
        geo_down["Section"].to_numpy() == "Up-leg"
    ):
        raise RuntimeError(
            "Reference cable ordering is inconsistent: Up-leg rows appear "
            "before the deepest/turn-around channel."
        )

    borehole_mask = (
        geo_down["TVD_m"].to_numpy(dtype=np.float64)
        >= MIN_BOREHOLE_TVD_M
    ) & (
        geo_down["MD_m"].to_numpy(dtype=np.float64)
        >= MIN_BOREHOLE_MD_M
    )

    if np.count_nonzero(borehole_mask) < 10:
        raise RuntimeError(
            "Too few reference down-leg rows after removing the "
            f"surface spool: {np.count_nonzero(borehole_mask)}"
        )

    first_borehole_pos = int(np.flatnonzero(borehole_mask)[0])
    geo_model = geo_down.iloc[first_borehole_pos:].copy()

    # The 2-D profile origin is the canonical SAFOD wellhead, not the
    # first DAS row.  The Surface Spool rows in GEO_XLSX are collapsed to
    # one MD=TVD=0 coordinate and do not represent a resolved surface path.
    lat0 = float(SAFOD_WELLHEAD_LAT_WGS84)
    lon0 = float(SAFOD_WELLHEAD_LON_WGS84)

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

    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError(
            "Could not define the 2D profile direction from the "
            "reference cable geometry."
        )

    u_e = e_deep / norm
    u_n = n_deep / norm

    return {
        "geo_full": geo,
        "geo_down": geo_down,
        "geo_model": geo_model,
        "lat0": lat0,
        "lon0": lon0,
        "u_e": float(u_e),
        "u_n": float(u_n),
        "reference_turn_channel": float(channel_full[turn_idx]),
        "reference_turn_tvd_m": float(tvd_full[turn_idx]),
        "reference_turn_md_m": float(md_full[turn_idx]),
        "reference_first_borehole_channel": float(
            geo_down.iloc[first_borehole_pos]["Channel"]
        ),
    }


def _finalize_channel_mapping(
    geo_model: pd.DataFrame,
    *,
    context: dict,
    source_description: str,
    first_borehole_channel: float,
    turn_channel: float,
    turn_tvd_m: float,
    turn_md_m: float,
) -> dict:
    """
    Project one event's receiver table into the common 2D coordinates,
    save the standard geometry CSV, and return the schema expected by main().
    """
    out = geo_model.copy()

    required = [
        "Channel",
        "data_idx",
        "Lat_WGS84",
        "Lon_WGS84",
        "TVD_m",
        "MD_m",
    ]

    missing = sorted(
        set(required).difference(out.columns)
    )

    if missing:
        raise ValueError(
            f"Receiver mapping is missing columns: {missing}"
        )

    for column in [
        "Channel",
        "Lat_WGS84",
        "Lon_WGS84",
        "TVD_m",
        "MD_m",
    ]:
        out[column] = pd.to_numeric(
            out[column],
            errors="raise",
        )

    out["data_idx"] = _integer_index(
        out["data_idx"],
        name="data_idx",
    )

    data_idx = out["data_idx"].to_numpy(dtype=np.int64)

    if np.any(np.diff(data_idx) <= 0):
        raise ValueError(
            "Receiver data_idx values must increase strictly."
        )

    if not np.all(np.diff(data_idx) == 1):
        raise ValueError(
            "The current down-leg receiver mapping must be contiguous."
        )

    md_m = out["MD_m"].to_numpy(dtype=np.float64)
    tvd_m = out["TVD_m"].to_numpy(dtype=np.float64)

    if not np.all(np.isfinite(md_m)) or not np.all(np.isfinite(tvd_m)):
        raise ValueError(
            "MD_m and TVD_m must contain only finite values."
        )

    if np.any(np.diff(md_m) <= 0.0):
        raise ValueError(
            "Mapped down-leg MD_m must increase strictly."
        )

    if np.any(np.diff(tvd_m) < -1.0e-6):
        raise ValueError(
            "Mapped down-leg TVD_m must not decrease."
        )

    east, north = latlon_to_local_enu_m(
        out["Lat_WGS84"].to_numpy(dtype=np.float64),
        out["Lon_WGS84"].to_numpy(dtype=np.float64),
        context["lat0"],
        context["lon0"],
    )

    along = (
        east * context["u_e"]
        + north * context["u_n"]
    )
    cross = (
        -east * context["u_n"]
        + north * context["u_e"]
    )

    out["east_m_from_wellhead"] = east
    out["north_m_from_wellhead"] = north
    out["along_profile_m"] = along
    out["cross_profile_m"] = cross
    out["X_2D_m"] = along
    out["Z_2D_m"] = tvd_m

    model_x = out["X_2D_m"].to_numpy(dtype=np.float64)
    model_z = out["Z_2D_m"].to_numpy(dtype=np.float64)

    dx_seg = np.diff(model_x)
    dz_seg = np.diff(model_z)

    arc_length_m = float(
        np.sum(
            np.hypot(dx_seg, dz_seg)
        )
    )

    x_range = float(
        np.nanmax(model_x) - np.nanmin(model_x)
    )
    z_range = float(
        np.nanmax(model_z) - np.nanmin(model_z)
    )

    straight_bound = float(
        np.hypot(x_range, z_range)
    )
    monotonic_bound = float(
        x_range + z_range
    )

    fit_rms_m = float(
        np.sqrt(
            np.nanmean(cross ** 2)
        )
    )

    out_csv = (
        OUT_DIR
        / "SAFOD_Phase2_projected_from_georef.csv"
    )

    out.to_csv(
        out_csv,
        index=False,
    )

    print("\nReceiver geometry / channel registration")
    print("----------------------------------------")
    print(f"source                     : {source_description}")
    print(f"geometry rows              : {len(out)}")
    print(
        "data-row range             : "
        f"{data_idx[0]} to {data_idx[-1]}"
    )
    print(
        "physical reference channels: "
        f"{out['Channel'].iloc[0]:.1f} to "
        f"{out['Channel'].iloc[-1]:.1f}"
    )
    print(
        "MD range                   : "
        f"{np.nanmin(md_m):.1f} to "
        f"{np.nanmax(md_m):.1f} m"
    )
    print(
        "TVD range                  : "
        f"{np.nanmin(model_z):.1f} to "
        f"{np.nanmax(model_z):.1f} m"
    )
    print(
        "model x range              : "
        f"{np.nanmin(model_x):.1f} to "
        f"{np.nanmax(model_x):.1f} m"
    )
    print(
        "wellhead lat/lon           : "
        f"{context['lat0']:.7f}, "
        f"{context['lon0']:.7f}"
    )
    print(
        "profile unit vector EN     : "
        f"({context['u_e']:.4f}, "
        f"{context['u_n']:.4f})"
    )
    print(
        "computed cable arc length  : "
        f"{arc_length_m:.1f} m"
    )
    print(
        "arc-length sanity bounds   : "
        f"{straight_bound:.1f} to "
        f"{monotonic_bound:.1f} m"
    )
    print(
        "crossline cable range      : "
        f"{np.nanmin(cross):.1f} to "
        f"{np.nanmax(cross):.1f} m"
    )
    print(
        "cable-to-profile-line RMS  : "
        f"{fit_rms_m:.2f} m"
    )
    print(f"saved model geometry       : {out_csv}")

    if arc_length_m > 1.15 * monotonic_bound:
        print(
            "WARNING: computed arc length is much larger than the "
            "monotonic bound. Check for duplicated or reversed geometry."
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
    ax.set_xlabel(
        "2D model x = along-profile coordinate [m]"
    )
    ax.set_ylabel("TVD depth [m]")
    ax.set_title(
        "SAFOD down-going borehole pass projected to 2D"
    )
    ax.set_ylim(np.nanmax(model_z), 0.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUT_DIR
        / "real_cable_projected_geometry_downpass_only.png",
        dpi=220,
    )
    plt.close(fig)

    return {
        "lat0": context["lat0"],
        "lon0": context["lon0"],
        "u_e": context["u_e"],
        "u_n": context["u_n"],
        "geometry_csv": str(out_csv),
        "mapping_table": out,
        "fit_rms_m": fit_rms_m,
        "turn_channel": float(turn_channel),
        "turn_tvd_m": float(turn_tvd_m),
        "turn_md_m": float(turn_md_m),
        "first_borehole_channel": float(first_borehole_channel),
        "arc_length_m": arc_length_m,
        "n_rows_full": len(context["geo_full"]),
        "n_rows_downleg_including_spool": len(
            context["geo_down"]
        ),
        "n_rows_downleg_model": len(out),
    }


def _reference_channels_from_mapping_table(
    table: pd.DataFrame,
    *,
    context: dict,
) -> np.ndarray:
    """
    Return the immutable physical channel coordinate from GEO_XLSX.

    DataRow/AcquisitionChannel identify the current interrogator recording and
    can change between acquisitions. ReferenceChannel identifies position on
    the unchanged physical cable and is therefore the correct common x-axis.
    """
    if "ReferenceChannel" in table.columns:
        reference_channels = pd.to_numeric(
            table["ReferenceChannel"],
            errors="raise",
        ).to_numpy(dtype=np.float64)
    else:
        # Backward-compatible reconstruction from unchanged down-leg MD.
        reference_down = (
            context["geo_full"]
            .loc[
                context["geo_full"]["Section"] == "Down-leg"
            ]
            .sort_values("MD_m")
        )

        reference_md = pd.to_numeric(
            reference_down["MD_m"],
            errors="raise",
        ).to_numpy(dtype=np.float64)
        reference_channel_axis = pd.to_numeric(
            reference_down["Channel"],
            errors="raise",
        ).to_numpy(dtype=np.float64)
        target_md = pd.to_numeric(
            table["MD_m"],
            errors="raise",
        ).to_numpy(dtype=np.float64)

        if (
            np.nanmin(target_md) < np.nanmin(reference_md) - 1.0e-6
            or np.nanmax(target_md) > np.nanmax(reference_md) + 1.0e-6
        ):
            raise ValueError(
                "Mapped down-leg MD lies outside the unchanged reference "
                "geometry and ReferenceChannel is absent."
            )

        reference_channels = np.interp(
            target_md,
            reference_md,
            reference_channel_axis,
        )

    if not np.all(np.isfinite(reference_channels)):
        raise ValueError(
            "Physical ReferenceChannel contains NaN or Inf."
        )

    if np.any(np.diff(reference_channels) <= 0.0):
        raise ValueError(
            "Physical down-leg ReferenceChannel must increase strictly."
        )

    return reference_channels


def load_event_channel_mapping(
    path: Path,
    *,
    context: dict,
) -> dict:
    """
    Load an event-specific HDF5-row registration onto the unchanged cable.

    The two coordinates are kept separate:

      data_idx / DataRow
          row index in the current HDF5 file;

      Channel / ReferenceChannel
          physical channel coordinate of the unchanged GEO_XLSX cable.

    They are identical for the April registration but not for June.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Channel-mapping CSV not found: {path}"
        )

    geo = pd.read_csv(path).copy()

    required = [
        "DataRow",
        "MD_m",
        "TVD_m",
        "Lat_WGS84",
        "Lon_WGS84",
    ]

    missing = sorted(
        set(required).difference(geo.columns)
    )

    if missing:
        raise ValueError(
            "Channel-mapping CSV is missing required columns "
            f"{missing}: {path}"
        )

    data_rows = _integer_index(
        geo["DataRow"],
        name="DataRow",
    )

    if np.any(np.diff(data_rows) <= 0):
        raise ValueError(
            "DataRow values must increase strictly."
        )

    if not np.all(np.diff(data_rows) == 1):
        raise ValueError(
            "Mapped down-leg DataRow values must be contiguous."
        )

    reference_channels = _reference_channels_from_mapping_table(
        geo,
        context=context,
    )

    geo["data_idx"] = data_rows
    geo["ReferenceChannel"] = reference_channels

    # Standard geometry schema consumed by run_forward.py. Channel now always
    # means the stable physical cable coordinate, never the HDF5 row number.
    geo["Channel"] = reference_channels

    return _finalize_channel_mapping(
        geo,
        context=context,
        source_description=(
            "event-specific HDF5-row -> physical reference-channel "
            f"registration: {path}"
        ),
        first_borehole_channel=float(reference_channels[0]),
        turn_channel=float(reference_channels[-1]),
        turn_tvd_m=float(
            pd.to_numeric(
                geo["TVD_m"],
                errors="raise",
            ).iloc[-1]
        ),
        turn_md_m=float(
            pd.to_numeric(
                geo["MD_m"],
                errors="raise",
            ).iloc[-1]
        ),
    )

def build_channel_projection_mapping() -> dict:
    """
    Build one consistent receiver mapping for the active event.

    April:
        reference Excel channel numbers already index the HDF5 rows.

    June:
        CHANNEL_MAPPING_CSV maps the new HDF5 rows onto the same physical
        reference cable geometry.
    """
    context = _load_reference_geometry_context()

    if CHANNEL_MAPPING_CSV is not None:
        return load_event_channel_mapping(
            CHANNEL_MAPPING_CSV,
            context=context,
        )

    geo_model = context["geo_model"].copy()

    channel_indices = _integer_index(
        geo_model["Channel"],
        name="reference Channel",
    )

    geo_model["data_idx"] = channel_indices

    return _finalize_channel_mapping(
        geo_model,
        context=context,
        source_description=(
            f"reference Excel channel registration: {GEO_XLSX}"
        ),
        first_borehole_channel=float(
            channel_indices[0]
        ),
        turn_channel=float(
            channel_indices[-1]
        ),
        turn_tvd_m=float(
            geo_model["TVD_m"].iloc[-1]
        ),
        turn_md_m=float(
            geo_model["MD_m"].iloc[-1]
        ),
    )



def _reference_channels_for_full_table(
    table: pd.DataFrame,
    *,
    context: dict,
) -> np.ndarray:
    """Recover physical reference channels for a full down/up table."""
    if "ReferenceChannel" in table.columns:
        values = pd.to_numeric(
            table["ReferenceChannel"],
            errors="raise",
        ).to_numpy(dtype=np.float64)
        return values

    if "Section" not in table.columns or "MD_m" not in table.columns:
        raise ValueError(
            "Full registration requires ReferenceChannel, or Section + MD_m "
            "for reconstruction from GEO_XLSX."
        )

    output = np.full(
        len(table),
        np.nan,
        dtype=np.float64,
    )

    section_text = (
        table["Section"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    for section_name, reference_name in [
        ("down-leg", "Down-leg"),
        ("up-leg", "Up-leg"),
    ]:
        mask = section_text == section_name
        if not np.any(mask):
            continue

        reference = (
            context["geo_full"]
            .loc[
                context["geo_full"]["Section"] == reference_name
            ]
            .sort_values("MD_m")
        )

        reference_md = pd.to_numeric(
            reference["MD_m"],
            errors="raise",
        ).to_numpy(dtype=np.float64)
        reference_channels = pd.to_numeric(
            reference["Channel"],
            errors="raise",
        ).to_numpy(dtype=np.float64)
        target_md = pd.to_numeric(
            table.loc[mask, "MD_m"],
            errors="raise",
        ).to_numpy(dtype=np.float64)

        output[mask.to_numpy()] = np.interp(
            target_md,
            reference_md,
            reference_channels,
        )

    if not np.all(np.isfinite(output)):
        raise ValueError(
            "Could not reconstruct all full-cable ReferenceChannel values."
        )

    return output


def build_full_cable_qc_registration(
    *,
    n_hdf5_channels: int,
) -> dict:
    """
    Register the physical down-going + up-going deep cable for plotting.

    The physical channel coordinate comes from GEO_XLSX. HDF5 DataRow is used
    only to select samples. This is why the turn-around remains at reference
    channel 1700 for both April and June, while its HDF5 row changes.
    """
    context = _load_reference_geometry_context()

    reference_dual = (
        context["geo_full"]
        .loc[
            context["geo_full"]["Section"].isin(
                ["Down-leg", "Up-leg"]
            )
        ]
        .sort_values("Channel")
        .reset_index(drop=True)
    )

    reference_min = float(
        pd.to_numeric(
            reference_dual["Channel"],
            errors="raise",
        ).min()
    )
    reference_max = float(
        pd.to_numeric(
            reference_dual["Channel"],
            errors="raise",
        ).max()
    )
    reference_turn = float(
        context["reference_turn_channel"]
    )

    if CHANNEL_MAPPING_CSV is None:
        table = reference_dual.copy()
        data_rows = _integer_index(
            table["Channel"],
            name="reference full-cable Channel",
        )
        reference_channels = pd.to_numeric(
            table["Channel"],
            errors="raise",
        ).to_numpy(dtype=np.float64)
        registration_source = (
            f"identity reference registration: {GEO_XLSX}"
        )
        registration_path = str(GEO_XLSX)

    else:
        full_path = (
            Path(CHANNEL_MAPPING_CSV).parent
            / "mapped_full_dual_pass_geometry.csv"
        )

        if full_path.exists():
            table = pd.read_csv(full_path).copy()

            required = {"DataRow", "Section"}
            missing = sorted(
                required.difference(table.columns)
            )
            if missing:
                raise ValueError(
                    "Full dual-pass registration is missing columns "
                    f"{missing}: {full_path}"
                )

            section_text = (
                table["Section"]
                .astype(str)
                .str.strip()
                .str.lower()
            )
            table = (
                table.loc[
                    section_text.isin(["down-leg", "up-leg"])
                ]
                .copy()
                .sort_values("DataRow")
                .reset_index(drop=True)
            )

            data_rows = _integer_index(
                table["DataRow"],
                name="full-cable DataRow",
            )
            reference_channels = _reference_channels_for_full_table(
                table,
                context=context,
            )
            table["ReferenceChannel"] = reference_channels
            registration_source = (
                f"full dual-pass event registration: {full_path}"
            )
            registration_path = str(full_path)

        else:
            # Safe fallback: channelisation is affine along a continuous fibre.
            # Fit DataRow -> ReferenceChannel using the calibrated downleg, then
            # extend only over the immutable dual-pass reference range.
            down = pd.read_csv(
                CHANNEL_MAPPING_CSV
            ).copy()
            down_rows = _integer_index(
                down["DataRow"],
                name="downleg DataRow",
            )
            down_reference = _reference_channels_from_mapping_table(
                down,
                context=context,
            )

            slope, intercept = np.polyfit(
                down_rows.astype(np.float64),
                down_reference,
                deg=1,
            )
            predicted = (
                slope * down_rows.astype(np.float64)
                + intercept
            )
            fit_rms = float(
                np.sqrt(
                    np.mean(
                        (predicted - down_reference) ** 2
                    )
                )
            )

            if not np.isfinite(slope) or slope <= 0.0:
                raise RuntimeError(
                    "Invalid HDF5-row -> ReferenceChannel slope."
                )

            if fit_rms > 0.10:
                raise RuntimeError(
                    "Downleg registration is not sufficiently affine to "
                    "extend to the full cable: "
                    f"RMS={fit_rms:.4f} reference channels."
                )

            all_rows = np.arange(
                int(n_hdf5_channels),
                dtype=np.int64,
            )
            all_reference = (
                slope * all_rows.astype(np.float64)
                + intercept
            )
            keep = (
                (all_reference >= reference_min - 0.5 * slope)
                & (all_reference <= reference_max + 0.5 * slope)
            )

            data_rows = all_rows[keep]
            reference_channels = all_reference[keep]
            table = pd.DataFrame(
                {
                    "DataRow": data_rows,
                    "ReferenceChannel": reference_channels,
                    "Section": np.where(
                        reference_channels <= reference_turn,
                        "Down-leg",
                        "Up-leg",
                    ),
                }
            )
            registration_source = (
                "affine extension of calibrated downleg because "
                f"{full_path} was absent; fit RMS={fit_rms:.4f} channels"
            )
            registration_path = str(CHANNEL_MAPPING_CSV)

    valid = (
        (data_rows >= 0)
        & (data_rows < int(n_hdf5_channels))
        & np.isfinite(reference_channels)
        & (reference_channels >= reference_min - 1.0e-6)
        & (reference_channels <= reference_max + 1.0e-6)
    )

    data_rows = data_rows[valid]
    reference_channels = reference_channels[valid]
    table = table.loc[valid].reset_index(drop=True)

    if data_rows.size < 10:
        raise RuntimeError(
            "Too few registered dual-pass rows remain for full-cable QC."
        )

    if np.any(np.diff(data_rows) <= 0):
        raise ValueError(
            "Registered full-cable DataRow must increase strictly."
        )

    if not np.all(np.diff(data_rows) == 1):
        raise ValueError(
            "Registered full-cable DataRow must be contiguous."
        )

    if np.any(np.diff(reference_channels) <= 0.0):
        raise ValueError(
            "Physical ReferenceChannel must increase strictly from downleg "
            "through upleg."
        )

    # The reference-channel step should be almost constant because both axes
    # sample the same continuous fibre uniformly. imshow uses endpoint extent.
    channel_step = float(
        np.median(
            np.diff(reference_channels)
        )
    )
    affine_axis = (
        reference_channels[0]
        + channel_step * np.arange(reference_channels.size)
    )
    axis_rms = float(
        np.sqrt(
            np.mean(
                (reference_channels - affine_axis) ** 2
            )
        )
    )

    if axis_rms > 0.10:
        raise RuntimeError(
            "Reference-channel axis is too nonuniform for imshow: "
            f"RMS={axis_rms:.4f} channels."
        )

    turn_data_row = float(
        np.interp(
            reference_turn,
            reference_channels,
            data_rows.astype(np.float64),
        )
    )

    table["DataRow"] = data_rows
    table["ReferenceChannel"] = reference_channels

    registration_csv = (
        OUT_DIR
        / "full_cable_qc_registration.csv"
    )
    table.to_csv(
        registration_csv,
        index=False,
    )

    print("\nFull registered deep-cable geometry")
    print("-----------------------------------")
    print(f"source                    : {registration_source}")
    print(
        "HDF5 data-row range       : "
        f"{data_rows[0]} to {data_rows[-1]}"
    )
    print(
        "physical reference channel: "
        f"{reference_channels[0]:.2f} to "
        f"{reference_channels[-1]:.2f}"
    )
    print(
        "turn-around               : "
        f"reference channel {reference_turn:.2f}, "
        f"HDF5 row {turn_data_row:.2f}"
    )
    print(f"registered dual-pass rows : {data_rows.size}")
    print(f"saved registration        : {registration_csv}")

    return {
        "data_rows": data_rows,
        "reference_channels": reference_channels,
        "turn_reference_channel": reference_turn,
        "turn_data_row": turn_data_row,
        "registration_path": registration_path,
        "registration_source": registration_source,
        "registration_csv": str(registration_csv),
    }


def plot_full_cable_event_qc(
    *,
    data_unfiltered_full: np.ndarray,
    time_s: np.ndarray,
    fs_hz: float,
    event_meta: dict,
    registration: dict,
    out_path: Path,
) -> None:
    """Plot the registered physical deep cable down and back up."""
    data_unfiltered_full = np.asarray(
        data_unfiltered_full,
        dtype=np.float64,
    )
    time_s = np.asarray(
        time_s,
        dtype=np.float64,
    )
    data_rows = np.asarray(
        registration["data_rows"],
        dtype=np.int64,
    )
    reference_channels = np.asarray(
        registration["reference_channels"],
        dtype=np.float64,
    )
    turn_reference_channel = float(
        registration["turn_reference_channel"]
    )

    if data_unfiltered_full.ndim != 2:
        raise ValueError(
            "Full DAS array must be 2D; "
            f"got shape {data_unfiltered_full.shape}."
        )

    if data_unfiltered_full.shape[1] != time_s.size:
        raise ValueError(
            "Full DAS/time-axis mismatch."
        )

    if data_rows[0] < 0 or data_rows[-1] >= data_unfiltered_full.shape[0]:
        raise IndexError(
            "Full-cable registration points outside the HDF5 data array."
        )

    filter_tmin = max(
        float(time_s[0]),
        FULL_CABLE_QC_TMIN_S - FULL_CABLE_QC_FILTER_PAD_S,
    )
    filter_tmax = min(
        float(time_s[-1]),
        FULL_CABLE_QC_TMAX_S + FULL_CABLE_QC_FILTER_PAD_S,
    )
    filter_mask = (
        (time_s >= filter_tmin)
        & (time_s <= filter_tmax)
    )

    data_filter_window = np.ascontiguousarray(
        data_unfiltered_full[data_rows][:, filter_mask],
        dtype=np.float64,
    )
    time_filter_window = time_s[filter_mask]

    print("\nFiltering registered full-cable QC window")
    print("------------------------------------------")
    print(
        "filter interval          : "
        f"{time_filter_window[0]:.3f} to "
        f"{time_filter_window[-1]:.3f} s"
    )
    print(
        "display interval         : "
        f"{FULL_CABLE_QC_TMIN_S:.3f} to "
        f"{FULL_CABLE_QC_TMAX_S:.3f} s"
    )
    print(f"registered data shape    : {data_filter_window.shape}")
    print(f"bandpass                 : {FMIN:.1f} to {FMAX:.1f} Hz")

    filtered = bandpass_traces(
        data_filter_window,
        fs_hz=fs_hz,
        fmin_hz=FMIN,
        fmax_hz=FMAX,
        order=FILTER_ORDER,
        taper_frac=FILTER_TAPER_FRAC,
    )

    display_mask = (
        (time_filter_window >= FULL_CABLE_QC_TMIN_S)
        & (time_filter_window <= FULL_CABLE_QC_TMAX_S)
    )
    display = filtered[:, display_mask]
    display_time = time_filter_window[display_mask]

    clip = float(
        np.percentile(
            np.abs(display),
            FULL_CABLE_QC_PCLIP,
        )
    )
    if not np.isfinite(clip) or clip <= 0.0:
        clip = 1.0

    fig, ax = plt.subplots(
        figsize=(20, 12)
    )

    image = ax.imshow(
        display.T,
        extent=[
            float(reference_channels[0]),
            float(reference_channels[-1]),
            float(display_time[-1]),
            float(display_time[0]),
        ],
        aspect="auto",
        cmap="seismic",
        vmin=-clip,
        vmax=clip,
        interpolation="none",
    )

    ax.axhline(
        0.0,
        color="black",
        linewidth=1.1,
        linestyle="--",
        label="Catalog origin",
    )
    ax.axvline(
        turn_reference_channel,
        color="black",
        linewidth=1.0,
        linestyle=":",
        label="Cable turn-around",
    )

    ax.set_xlabel(
        "Physical reference channel (GEO_XLSX)"
    )
    ax.set_ylabel(
        "Time from catalog origin [s]"
    )
    ax.set_title(
        f"Real DAS {event_meta['event_id']} M{event_meta['mag']:.2f}, "
        f"registered down + up cable, {FMIN:.0f}-{FMAX:.0f} Hz"
    )
    ax.set_xlim(
        float(reference_channels[0]),
        float(reference_channels[-1]),
    )
    ax.set_ylim(
        float(display_time[-1]),
        float(display_time[0]),
    )
    ax.grid(alpha=0.20)
    ax.legend(
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
    )

    position = ax.get_position()
    colorbar_ax = fig.add_axes([0.0, 0.0, 0.1, 0.1])
    colorbar_ax.set_position(
        [
            position.x0 + position.width + 0.01,
            position.y0,
            0.02,
            position.height,
        ]
    )
    colorbar = plt.colorbar(
        image,
        cax=colorbar_ax,
    )
    colorbar.set_label(
        "strain rate [nm/m/s]"
    )

    fig.savefig(
        out_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved registered full-cable QC plot: {out_path}")

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

    # NCEDC reports earthquake depth relative to the geoid (approximately
    # mean sea level).  The elastic solver uses depth below the local SAFOD
    # ground surface, so the site elevation must be added.
    catalog_depth_geoid_m = event_meta["depth_km"] * 1000.0
    z_model = catalog_depth_geoid_m + SAFOD_SURFACE_ELEVATION_M

    print("\nEvent projection into 2D model")
    print("------------------------------")
    print(f"event id                  : {event_meta['event_id']}")
    print(f"origin                    : {event_meta['origin_time']}")
    print(f"lat/lon/depth             : {event_meta['lat']:.6f}, {event_meta['lon']:.6f}, {event_meta['depth_km']:.3f} km")
    print(f"magnitude                 : M{event_meta['mag']:.2f} {event_meta['mag_type']}")
    print(f"event east/north wellhead : {float(east_ev):.1f}, {float(north_ev):.1f} m")
    print(f"event along profile       : {along_ev:.1f} m")
    print(f"event crossline distance  : {cross_ev:.1f} m")
    print(f"event model x             : {x_model:.1f} m")
    print(
        f"catalog geoid depth       : "
        f"{catalog_depth_geoid_m:.1f} m"
    )
    print(
        f"SAFOD surface elevation   : "
        f"{SAFOD_SURFACE_ELEVATION_M:.2f} m MSL"
    )
    print(
        f"event model z             : "
        f"{z_model:.1f} m below solver surface"
    )

    return {
        "event_x_model_m": float(x_model),
        "event_z_model_m": float(z_model),
        "event_along_profile_m": float(along_ev),
        "event_crossline_m": float(cross_ev),
        "event_depth_geoid_m": float(catalog_depth_geoid_m),
        "surface_elevation_m": float(SAFOD_SURFACE_ELEVATION_M),
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
    # 3a. Plot only the physically registered down-going + up-going cable.
    #     HDF5 rows outside this registration are not cable geometry.
    # --------------------------------------------------------------------------
    full_cable_registration = build_full_cable_qc_registration(
        n_hdf5_channels=D_event_unfiltered_full.shape[0],
    )

    stale_full_plot = (
        OUT_DIR / "00_real_das_full_cable_preview.png"
    )
    if stale_full_plot.exists():
        stale_full_plot.unlink()

    plot_full_cable_event_qc(
        data_unfiltered_full=D_event_unfiltered_full,
        time_s=t_event,
        fs_hz=fs,
        event_meta=event_meta,
        registration=full_cable_registration,
        out_path=(
            OUT_DIR
            / "00_real_das_registered_dual_pass_preview.png"
        ),
    )

    # --------------------------------------------------------------------------
    # 3b. Select the exact HDF5 rows represented by the downleg geometry
    # --------------------------------------------------------------------------
    mapping_table = mapping["mapping_table"]

    data_indices = _integer_index(
        mapping_table["data_idx"],
        name="mapping data_idx",
    )

    n_raw_channels = D_event_unfiltered_full.shape[0]

    if data_indices[0] < 0 or data_indices[-1] >= n_raw_channels:
        raise IndexError(
            "Geometry mapping references HDF5 rows outside the data array: "
            f"{data_indices[0]} to {data_indices[-1]}, "
            f"but the file contains rows 0 to {n_raw_channels - 1}."
        )

    D_event_unfiltered = D_event_unfiltered_full[
        data_indices,
        :,
    ]

    reference_channels_event = mapping_table[
        "Channel"
    ].to_numpy(dtype=np.float64)

    acquisition_channels_event = (
        pd.to_numeric(
            mapping_table["AcquisitionChannel"],
            errors="raise",
        ).to_numpy(dtype=np.float64)
        if "AcquisitionChannel" in mapping_table.columns
        else data_indices.astype(np.float64)
    )

    if len(reference_channels_event) != D_event_unfiltered.shape[0]:
        raise RuntimeError(
            "Geometry/data alignment failure: "
            f"{len(reference_channels_event)} geometry rows versus "
            f"{D_event_unfiltered.shape[0]} DAS traces."
        )

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

    first_borehole_channel = float(
        mapping["first_borehole_channel"]
    )
    turn_channel = float(
        mapping["turn_channel"]
    )

    print("\nReal DAS downleg selection")
    print("--------------------------")
    print(
        "full real data shape       : "
        f"{D_event_unfiltered_full.shape}"
    )
    print(
        "mapped HDF5-row range      : "
        f"{data_indices[0]} to {data_indices[-1]}"
    )
    print(
        "physical reference channels: "
        f"{reference_channels_event[0]:.2f} to "
        f"{reference_channels_event[-1]:.2f}"
    )
    print(
        "selected downleg shape     : "
        f"{D_event.shape}"
    )
    print(
        "geometry/data row match    : "
        f"{len(mapping_table)} == {D_event.shape[0]}"
    )

    # --------------------------------------------------------------------------
    # 4. Plot real event gather
    # --------------------------------------------------------------------------
    old_preview = OUT_DIR / "real_das_event_preview_raw_channels.png"
    if old_preview.exists():
        old_preview.unlink()

    clip = np.percentile(np.abs(D_event), PCLIP)

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(
        D_event.T,
        extent=[
            float(reference_channels_event[0]),
            float(reference_channels_event[-1]),
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
    ax.set_xlabel("Physical reference channel (GEO_XLSX)")
    ax.set_ylabel(f"Time from {event_meta['origin_time']} [s]")
    ax.set_title(
        f"Real SAFOD DAS event {event_meta['event_id']} "
        f"M{event_meta['mag']:.2f}, {FMIN:.0f}-{FMAX:.0f} Hz"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "real_das_event_preview_reference_channels.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    scale = np.percentile(np.abs(D_event), 99.0, axis=1, keepdims=True)
    scale = np.maximum(scale, 1e-12)
    D_norm = D_event / scale

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(
        D_norm.T,
        extent=[
            float(reference_channels_event[0]),
            float(reference_channels_event[-1]),
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
    ax.set_xlabel("Physical reference channel (GEO_XLSX)")
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

        # Backward-compatible key consumed by run_forward/compare_event.
        # Its values are now the stable physical reference-channel coordinate.
        raw_channels=reference_channels_event,
        reference_channels=reference_channels_event,
        hdf5_data_rows=data_indices,
        acquisition_channels=acquisition_channels_event,
        channel_axis_definition=np.array(
            "physical ReferenceChannel from unchanged GEO_XLSX; "
            "HDF5 DataRow stored separately"
        ),
        first_borehole_channel=np.array(first_borehole_channel),
        turn_channel=np.array(turn_channel),
        turn_tvd_m=np.array(mapping["turn_tvd_m"]),
        turn_md_m=np.array(mapping["turn_md_m"]),
        geometry_arc_length_m=np.array(mapping["arc_length_m"]),
        geometry_csv=np.array(mapping["geometry_csv"]),
        n_rows_full_geometry=np.array(mapping["n_rows_full"]),
        n_rows_downleg_including_spool=np.array(mapping["n_rows_downleg_including_spool"]),
        n_rows_downleg_model=np.array(mapping["n_rows_downleg_model"]),

        full_cable_qc_data_rows=full_cable_registration["data_rows"],
        full_cable_qc_reference_channels=(
            full_cable_registration["reference_channels"]
        ),
        full_cable_qc_turn_reference_channel=np.array(
            full_cable_registration["turn_reference_channel"]
        ),
        full_cable_qc_turn_data_row=np.array(
            full_cable_registration["turn_data_row"]
        ),
        full_cable_qc_registration_path=np.array(
            full_cable_registration["registration_path"]
        ),

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
        profile_origin_lat=np.array(mapping["lat0"]),
        profile_origin_lon=np.array(mapping["lon0"]),
        profile_u_e=np.array(mapping["u_e"]),
        profile_u_n=np.array(mapping["u_n"]),
        profile_azimuth_deg=np.array(
            (
                np.rad2deg(
                    np.arctan2(
                        mapping["u_e"],
                        mapping["u_n"],
                    )
                )
                % 360.0
            )
        ),
        event_depth_geoid_m=np.array(ev_proj["event_depth_geoid_m"]),
        surface_elevation_m=np.array(ev_proj["surface_elevation_m"]),
        horizontal_origin_definition=np.array(
            "canonical SAFOD wellhead WGS84"
        ),
        vertical_datum_definition=np.array(
            "NCEDC depth positive downward from geoid; "
            "solver z positive downward from SAFOD ground surface"
        ),
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