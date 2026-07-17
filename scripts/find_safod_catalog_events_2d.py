from __future__ import annotations

from pathlib import Path
import datetime

import numpy as np
import pandas as pd

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import URL_MAPPINGS


# ==============================================================================
# SETTINGS
# ==============================================================================

GEO_XLSX = "/home/groups/ettore88/alina/SAFOD/SAFOD_Phase2_GeoReferenced_Channels.xlsx"

OUT_DIR = Path("results/catalog_event_search_2d")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# DAS has been recording since April 2026.
START_TIME = UTCDateTime("2026-04-01T00:00:00")

# Use current time when script runs.
END_TIME = UTCDateTime()

# Catalog search around SAFOD surface location.
MAX_RADIUS_DEG = 0.35      # roughly 35–40 km
MIN_MAG = 0.5
MIN_DEPTH_KM = 0.0
MAX_DEPTH_KM = 15.0

# Query in chunks to avoid large FDSN requests.
CHUNK_DAYS = 14

# Ranking thresholds.
GOOD_CROSSLINE_M = 1000.0
OK_CROSSLINE_M = 2000.0
MARGINAL_CROSSLINE_M = 3000.0

URL_MAPPINGS["NCEDC"] = "https://service.ncedc.org"


# ==============================================================================
# GEOMETRY HELPERS
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


def build_real_2d_geometry() -> dict:
    """
    Define the SAFOD 2D profile from the georeferenced channel table.

    2D coordinates:
        x = along-profile horizontal coordinate [m]
        z = TVD depth [m]
    """
    geo = pd.read_excel(GEO_XLSX)

    required = ["Channel", "Lat_WGS84", "Lon_WGS84", "TVD_m"]
    for col in required:
        if col not in geo.columns:
            raise ValueError(f"Missing column {col!r} in {GEO_XLSX}")

    geo = geo.copy()
    for col in required:
        geo[col] = pd.to_numeric(geo[col], errors="coerce")

    geo = geo.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["Channel", "Lat_WGS84", "Lon_WGS84", "TVD_m"]
    )

    geo = geo.sort_values("Channel")
    geo = geo.groupby("Channel", as_index=False).median(numeric_only=True)

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
    ch = geo["Channel"].to_numpy(dtype=np.float64)

    # Profile direction from surface to deepest/deviated part of cable.
    deep_mask = tvd > np.nanpercentile(tvd, 90.0)
    if np.count_nonzero(deep_mask) < 5:
        deep_mask = tvd > np.nanpercentile(tvd, 80.0)

    e_deep = float(np.nanmedian(east[deep_mask]))
    n_deep = float(np.nanmedian(north[deep_mask]))

    norm = float(np.hypot(e_deep, n_deep))
    if norm <= 0.0 or not np.isfinite(norm):
        raise RuntimeError("Could not define SAFOD 2D profile direction.")

    u_e = e_deep / norm
    u_n = n_deep / norm

    along = east * u_e + north * u_n
    cross = -east * u_n + north * u_e

    out = geo.copy()
    out["east_m_from_surface"] = east
    out["north_m_from_surface"] = north
    out["along_profile_m"] = along
    out["cross_profile_m"] = cross
    out["X_2D_m"] = along
    out["Z_2D_m"] = tvd
    out.to_csv(OUT_DIR / "safod_real_2d_profile_geometry.csv", index=False)

    print("\nSAFOD 2D profile geometry")
    print("-------------------------")
    print(f"surface lat/lon        : {lat0:.7f}, {lon0:.7f}")
    print(f"profile unit vector EN : ({u_e:.4f}, {u_n:.4f})")
    print(f"channel range          : {np.nanmin(ch):.0f} to {np.nanmax(ch):.0f}")
    print(f"along range            : {np.nanmin(along):.1f} to {np.nanmax(along):.1f} m")
    print(f"TVD range              : {np.nanmin(tvd):.1f} to {np.nanmax(tvd):.1f} m")
    print(f"cable crossline range  : {np.nanmin(cross):.1f} to {np.nanmax(cross):.1f} m")

    return {
        "lat0": lat0,
        "lon0": lon0,
        "u_e": float(u_e),
        "u_n": float(u_n),
        "channel": ch,
        "along": along,
        "cross": cross,
        "tvd": tvd,
    }


def event_to_profile(lat: float, lon: float, depth_km: float, geom: dict) -> dict:
    east, north = latlon_to_local_enu_m(lat, lon, geom["lat0"], geom["lon0"])

    along = float(east * geom["u_e"] + north * geom["u_n"])
    cross = float(-east * geom["u_n"] + north * geom["u_e"])
    depth_m = float(depth_km * 1000.0)

    dx = geom["along"] - along
    dy = geom["cross"] - cross
    dz = geom["tvd"] - depth_m

    dist3d = np.sqrt(dx * dx + dy * dy + dz * dz)
    i_min = int(np.nanargmin(dist3d))

    along_min = float(np.nanmin(geom["along"]))
    along_max = float(np.nanmax(geom["along"]))

    return {
        "event_along_m": along,
        "event_crossline_m": cross,
        "abs_crossline_m": abs(cross),
        "event_depth_m": depth_m,
        "min_3d_distance_to_cable_m": float(dist3d[i_min]),
        "closest_channel": float(geom["channel"][i_min]),
        "closest_channel_along_m": float(geom["along"][i_min]),
        "closest_channel_tvd_m": float(geom["tvd"][i_min]),
        "inside_cable_along_range": bool((along >= along_min) and (along <= along_max)),
    }


# ==============================================================================
# CATALOG HELPERS
# ==============================================================================

def get_event_id(ev) -> str:
    rid = str(ev.resource_id)
    return rid.split("/")[-1]


def get_origin(ev):
    return ev.preferred_origin() or ev.origins[0]


def get_magnitude(ev):
    if ev.preferred_magnitude() is not None:
        return ev.preferred_magnitude()
    if ev.magnitudes:
        return ev.magnitudes[0]
    return None


def crossline_quality(abs_crossline_m: float) -> str:
    if abs_crossline_m <= GOOD_CROSSLINE_M:
        return "good_<1km"
    if abs_crossline_m <= OK_CROSSLINE_M:
        return "ok_<2km"
    if abs_crossline_m <= MARGINAL_CROSSLINE_M:
        return "marginal_<3km"
    return "bad_>3km"


def suitability_score(row: dict) -> float:
    """
    Lower is better.

    Crossline is the main penalty because we need a 2D-compatible event.
    Magnitude helps, but cannot compensate for a huge out-of-plane distance.
    """
    cross_km = row["abs_crossline_m"] / 1000.0
    min_dist_km = row["min_3d_distance_to_cable_m"] / 1000.0
    mag = row["magnitude"]

    inside_bonus = -0.5 if row["inside_cable_along_range"] else 0.5

    return 3.0 * cross_km + 0.5 * min_dist_km - 0.8 * mag + inside_bonus


def query_catalog_in_chunks(client: Client, geom: dict):
    rows = []

    t0 = START_TIME
    while t0 < END_TIME:
        t1 = min(t0 + CHUNK_DAYS * 24 * 3600, END_TIME)

        print(f"\nQuerying NCEDC: {t0.isoformat()} to {t1.isoformat()}")

        try:
            cat = client.get_events(
                starttime=t0,
                endtime=t1,
                latitude=geom["lat0"],
                longitude=geom["lon0"],
                maxradius=MAX_RADIUS_DEG,
                minmagnitude=MIN_MAG,
                mindepth=MIN_DEPTH_KM,
                maxdepth=MAX_DEPTH_KM,
                orderby="time",
            )
        except Exception as e:
            print(f"  WARNING: query failed: {e}")
            t0 = t1
            continue

        print(f"  events returned: {len(cat)}")

        for ev in cat:
            try:
                ori = get_origin(ev)
                mag = get_magnitude(ev)
                if mag is None or ori.depth is None:
                    continue

                ev_lat = float(ori.latitude)
                ev_lon = float(ori.longitude)
                ev_depth_km = float(ori.depth) / 1000.0
                ev_mag = float(mag.mag)
                mag_type = str(mag.magnitude_type or "")

                proj = event_to_profile(ev_lat, ev_lon, ev_depth_km, geom)

                row = {
                    "event_id": get_event_id(ev),
                    "origin_time": ori.time.isoformat(),
                    "latitude": ev_lat,
                    "longitude": ev_lon,
                    "depth_km": ev_depth_km,
                    "magnitude": ev_mag,
                    "magnitude_type": mag_type,
                    **proj,
                }

                row["crossline_quality"] = crossline_quality(row["abs_crossline_m"])
                row["suitability_score"] = suitability_score(row)

                rows.append(row)

            except Exception as e:
                print(f"  WARNING: skipped event because of error: {e}")

        t0 = t1

    return rows


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    geom = build_real_2d_geometry()

    print("\nCatalog search settings")
    print("-----------------------")
    print(f"start time     : {START_TIME.isoformat()}")
    print(f"end time       : {END_TIME.isoformat()}")
    print(f"max radius     : {MAX_RADIUS_DEG:.2f} deg")
    print(f"min magnitude  : {MIN_MAG:.1f}")
    print(f"depth range    : {MIN_DEPTH_KM:.1f} to {MAX_DEPTH_KM:.1f} km")

    client = Client("NCEDC")

    rows = query_catalog_in_chunks(client, geom)

    if len(rows) == 0:
        print("\nNo catalog events found.")
        return

    df = pd.DataFrame(rows)

    # Sort by actual numeric score.
    df = df.sort_values(
        ["suitability_score", "abs_crossline_m", "magnitude"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    out_all = OUT_DIR / "safod_catalog_events_since_april_2026_ranked.csv"
    df.to_csv(out_all, index=False)

    lt1 = df[df["abs_crossline_m"] <= 1000.0].copy()
    lt2 = df[df["abs_crossline_m"] <= 2000.0].copy()
    lt3 = df[df["abs_crossline_m"] <= 3000.0].copy()

    lt1.to_csv(OUT_DIR / "candidates_crossline_lt1km.csv", index=False)
    lt2.to_csv(OUT_DIR / "candidates_crossline_lt2km.csv", index=False)
    lt3.to_csv(OUT_DIR / "candidates_crossline_lt3km.csv", index=False)

    print("\nSaved:")
    print(f"  {out_all}")
    print(f"  {OUT_DIR / 'candidates_crossline_lt1km.csv'}")
    print(f"  {OUT_DIR / 'candidates_crossline_lt2km.csv'}")
    print(f"  {OUT_DIR / 'candidates_crossline_lt3km.csv'}")

    cols = [
        "origin_time",
        "event_id",
        "magnitude",
        "magnitude_type",
        "depth_km",
        "event_along_m",
        "event_crossline_m",
        "abs_crossline_m",
        "min_3d_distance_to_cable_m",
        "closest_channel",
        "inside_cable_along_range",
        "crossline_quality",
        "suitability_score",
    ]

    print("\nBest candidates with crossline < 1 km:")
    if lt1.empty:
        print("  None")
    else:
        print(lt1[cols].head(20).to_string(index=False))

    print("\nBest candidates with crossline < 2 km:")
    if lt2.empty:
        print("  None")
    else:
        print(lt2[cols].head(20).to_string(index=False))

    print("\nBest candidates with crossline < 3 km:")
    if lt3.empty:
        print("  None")
    else:
        print(lt3[cols].head(20).to_string(index=False))

    print("\nBest 30 overall:")
    print(df[cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()