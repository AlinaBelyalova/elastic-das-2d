#!/usr/bin/env python3
"""
Extract the current SAFOD 2-D modelling section from the native
Zhang, Thurber & Bedrosian (2009) 3-D Vp/Vs tomography.

This is a MODEL-PREPARATION / QC script. It does not modify run_forward.py
or the model factory.

Horizontal registration
-----------------------
- Native Zhang XY axes come directly from inversion_grid.dat.
- The current SAFOD section orientation is recovered from the same
  georeferenced cable CSV used by the real-event workflow.
- No assumption is made that the SAFOD section is exactly Zhang X=constant
  or Y=constant.

Vertical registration
---------------------
The Zhang z coordinate is interpreted as depth relative to mean sea level,
positive downward. The current elastic-DAS solver uses depth below local
ground surface, positive downward.

Therefore:

    z_zhang_km = (solver_depth_m - surface_elevation_m) / 1000

Default surface elevation:
    660.46 m MSL

The shallow interval above the first physical Zhang node (z=-0.5 km) is
clipped to that top node. This is explicit constant extrapolation; no extra
shallow resolution is invented.

Inputs
------
1. processed/zhang2009_native_grid.npz
2. current event geometry CSV from scripts.safod.settings.GEOMETRY_CSV

Outputs
-------
processed/zhang2009_safod_section_2d.npz
processed/zhang2009_safod_section_coordinates.csv
qc/zhang2009_safod_section_registration.txt
qc/zhang2009_safod_section_map.png
qc/zhang2009_safod_section_vp.png
qc/zhang2009_safod_section_vs.png
qc/zhang2009_safod_section_vpvs.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Geod
from scipy.interpolate import RegularGridInterpolator

from scripts.safod.settings import (
    GEOMETRY_CSV,
    SAFOD_SURFACE_ELEVATION_M,
    SAFOD_WELLHEAD_LAT_WGS84,
    SAFOD_WELLHEAD_LON_WGS84,
)


DEFAULT_GRID = Path(
    "data/safod/velocity_models/"
    "zhang_thurber_bedrosian_2009/"
    "processed/zhang2009_native_grid.npz"
)

DEFAULT_OUTPUT_DIR = Path(
    "data/safod/velocity_models/"
    "zhang_thurber_bedrosian_2009"
)

DEFAULT_SURFACE_ELEVATION_M = SAFOD_SURFACE_ELEVATION_M

DEFAULT_X_MIN_M = -1100.0
DEFAULT_X_MAX_M = 2680.0
DEFAULT_Z_MAX_M = 5000.0
DEFAULT_DX_M = 5.0
DEFAULT_DZ_M = 5.0


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def geodetic_offset_en(
    geod: Geod,
    *,
    lon0: float,
    lat0: float,
    lon1: float,
    lat1: float,
) -> tuple[float, float]:
    az_deg, _, distance_m = geod.inv(
        lon0,
        lat0,
        lon1,
        lat1,
    )

    az = np.deg2rad(az_deg)

    east_m = distance_m * np.sin(az)
    north_m = distance_m * np.cos(az)

    return float(east_m), float(north_m)


def forward_from_en(
    geod: Geod,
    *,
    lon0: float,
    lat0: float,
    east_m: float,
    north_m: float,
) -> tuple[float, float]:
    distance_m = float(
        np.hypot(
            east_m,
            north_m,
        )
    )

    if distance_m == 0.0:
        return float(lon0), float(lat0)

    azimuth_deg = float(
        np.rad2deg(
            np.arctan2(
                east_m,
                north_m,
            )
        )
    )

    lon, lat, _ = geod.fwd(
        lon0,
        lat0,
        azimuth_deg,
        distance_m,
    )

    return float(lon), float(lat)


def infer_profile_from_geometry(
    geometry_csv: Path,
    geod: Geod,
) -> dict:
    """
    Read the exact SAFOD 2-D coordinate frame produced by prepare_event.

    The geographic origin is the canonical SAFOD wellhead.  The profile
    direction is recovered algebraically from the stored EN/along/cross
    coordinates, rather than re-fitting a straight line to the borehole.
    """
    geometry_csv = Path(geometry_csv)

    if not geometry_csv.exists():
        raise FileNotFoundError(geometry_csv)

    geo = pd.read_csv(geometry_csv)

    required = {
        "X_2D_m",
        "TVD_m",
        "Lat_WGS84",
        "Lon_WGS84",
        "east_m_from_wellhead",
        "north_m_from_wellhead",
        "along_profile_m",
        "cross_profile_m",
    }

    missing = sorted(required.difference(geo.columns))
    if missing:
        raise ValueError(
            f"Geometry CSV is missing {missing}: {geometry_csv}"
        )

    geo = geo.copy()

    for column in required:
        geo[column] = pd.to_numeric(
            geo[column],
            errors="coerce",
        )

    geo = (
        geo.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=list(required))
        .reset_index(drop=True)
    )

    if len(geo) < 10:
        raise ValueError("Too few valid geometry rows.")

    east = geo["east_m_from_wellhead"].to_numpy(
        dtype=np.float64
    )
    north = geo["north_m_from_wellhead"].to_numpy(
        dtype=np.float64
    )
    along = geo["along_profile_m"].to_numpy(
        dtype=np.float64
    )
    cross = geo["cross_profile_m"].to_numpy(
        dtype=np.float64
    )

    # Exact inverse of:
    #
    # along = E*u_E + N*u_N
    # cross = -E*u_N + N*u_E
    #
    # The values were written by prepare_event using the common SAFOD frame.
    denominator = float(
        np.sum(along ** 2 + cross ** 2)
    )

    if not np.isfinite(denominator) or denominator <= 0.0:
        raise RuntimeError(
            "Cannot recover SAFOD profile direction from geometry."
        )

    u_e = float(
        np.sum(along * east + cross * north)
        / denominator
    )

    u_n = float(
        np.sum(along * north - cross * east)
        / denominator
    )

    norm = float(np.hypot(u_e, u_n))

    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError(
            "Recovered SAFOD profile direction is invalid."
        )

    u_e /= norm
    u_n /= norm

    # Algebraic reconstruction residual. This is not a fit defining the
    # section; it only checks consistency of the stored coordinates.
    east_reconstructed = (
        along * u_e
        - cross * u_n
    )

    north_reconstructed = (
        along * u_n
        + cross * u_e
    )

    residual_m = np.hypot(
        east - east_reconstructed,
        north - north_reconstructed,
    )

    return {
        "geometry": geo,
        "lat0": float(SAFOD_WELLHEAD_LAT_WGS84),
        "lon0": float(SAFOD_WELLHEAD_LON_WGS84),
        "u_e": float(u_e),
        "u_n": float(u_n),
        "profile_azimuth_deg": float(
            np.rad2deg(
                np.arctan2(u_e, u_n)
            )
            % 360.0
        ),
        "horizontal_fit_rms_m": float(
            np.sqrt(
                np.mean(
                    residual_m ** 2
                )
            )
        ),
        "horizontal_fit_max_m": float(
            np.max(residual_m)
        ),
    }


def infer_zhang_axes(
    *,
    x_km: np.ndarray,
    y_km: np.ndarray,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    geod: Geod,
) -> dict:
    ix0 = nearest_index(
        x_km,
        0.0,
    )

    iy0 = nearest_index(
        y_km,
        0.0,
    )

    ix_pos = np.where(
        x_km > 0.0
    )[0][0]

    iy_pos = np.where(
        y_km > 0.0
    )[0][0]

    origin_lat = float(
        lat_deg[
            iy0,
            ix0,
        ]
    )

    origin_lon = float(
        lon_deg[
            iy0,
            ix0,
        ]
    )

    x_azimuth_deg, _, _ = geod.inv(
        origin_lon,
        origin_lat,
        float(
            lon_deg[
                iy0,
                ix_pos,
            ]
        ),
        float(
            lat_deg[
                iy0,
                ix_pos,
            ]
        ),
    )

    y_azimuth_deg, _, _ = geod.inv(
        origin_lon,
        origin_lat,
        float(
            lon_deg[
                iy_pos,
                ix0,
            ]
        ),
        float(
            lat_deg[
                iy_pos,
                ix0,
            ]
        ),
    )

    x_azimuth_deg %= 360.0
    y_azimuth_deg %= 360.0

    return {
        "origin_lat": origin_lat,
        "origin_lon": origin_lon,
        "x_azimuth_deg": float(
            x_azimuth_deg
        ),
        "y_azimuth_deg": float(
            y_azimuth_deg
        ),
    }


def en_to_zhang_xy(
    *,
    east_m: np.ndarray,
    north_m: np.ndarray,
    x_azimuth_deg: float,
    y_azimuth_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_az = np.deg2rad(
        x_azimuth_deg
    )

    y_az = np.deg2rad(
        y_azimuth_deg
    )

    zhang_x_m = (
        east_m
        * np.sin(
            x_az
        )
        + north_m
        * np.cos(
            x_az
        )
    )

    zhang_y_m = (
        east_m
        * np.sin(
            y_az
        )
        + north_m
        * np.cos(
            y_az
        )
    )

    return (
        np.asarray(
            zhang_x_m,
            dtype=np.float64,
        ),
        np.asarray(
            zhang_y_m,
            dtype=np.float64,
        ),
    )


def make_interpolator(
    *,
    z_km: np.ndarray,
    y_km: np.ndarray,
    x_km: np.ndarray,
    cube: np.ndarray,
) -> tuple[RegularGridInterpolator, np.ndarray]:
    """
    Exclude the artificial extreme vertical bounding nodes -150 and +340 km.
    The local horizontal nodes are retained; interpolation within the SAFOD
    section never approaches the +/-240 km horizontal bounds.
    """
    physical_z_mask = (
        (z_km >= -0.5)
        & (z_km <= 10.0)
    )

    z_local = z_km[
        physical_z_mask
    ]

    cube_local = cube[
        physical_z_mask,
        :,
        :,
    ]

    interpolator = RegularGridInterpolator(
        (
            z_local,
            y_km,
            x_km,
        ),
        cube_local,
        method="linear",
        bounds_error=True,
    )

    return (
        interpolator,
        z_local,
    )


def plot_section(
    *,
    field: np.ndarray,
    x_model_m: np.ndarray,
    depth_m: np.ndarray,
    cable_x_m: np.ndarray,
    cable_z_m: np.ndarray,
    quantity: str,
    units: str,
    output: Path,
) -> None:
    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    mesh = ax.pcolormesh(
        x_model_m,
        depth_m,
        field.T,
        shading="auto",
    )

    ax.plot(
        cable_x_m,
        cable_z_m,
        linewidth=1.5,
        label="SAFOD DAS cable",
    )

    ax.set_xlabel(
        "Current SAFOD 2-D model x [m]"
    )

    ax.set_ylabel(
        "Depth below local ground surface [m]"
    )

    ax.set_title(
        f"Zhang et al. (2009) {quantity} sampled on current SAFOD section"
    )

    ax.set_ylim(
        float(
            depth_m.max()
        ),
        0.0,
    )

    ax.legend()

    fig.colorbar(
        mesh,
        ax=ax,
        label=units,
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--grid",
        type=Path,
        default=DEFAULT_GRID,
    )

    parser.add_argument(
        "--geometry-csv",
        type=Path,
        default=Path(
            GEOMETRY_CSV
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--surface-elevation-m",
        type=float,
        default=DEFAULT_SURFACE_ELEVATION_M,
    )

    parser.add_argument(
        "--x-min-m",
        type=float,
        default=DEFAULT_X_MIN_M,
    )

    parser.add_argument(
        "--x-max-m",
        type=float,
        default=DEFAULT_X_MAX_M,
    )

    parser.add_argument(
        "--z-max-m",
        type=float,
        default=DEFAULT_Z_MAX_M,
    )

    parser.add_argument(
        "--dx-m",
        type=float,
        default=DEFAULT_DX_M,
    )

    parser.add_argument(
        "--dz-m",
        type=float,
        default=DEFAULT_DZ_M,
    )

    args = parser.parse_args()

    if not args.grid.exists():
        raise FileNotFoundError(
            args.grid
        )

    if not args.geometry_csv.exists():
        raise FileNotFoundError(
            args.geometry_csv
        )

    processed_dir = (
        args.output_dir
        / "processed"
    )

    qc_dir = (
        args.output_dir
        / "qc"
    )

    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    qc_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    geod = Geod(
        ellps="WGS84"
    )

    with np.load(
        args.grid,
        allow_pickle=True,
    ) as pkg:
        x_km = np.asarray(
            pkg["x_km"],
            dtype=np.float64,
        )

        y_km = np.asarray(
            pkg["y_km"],
            dtype=np.float64,
        )

        z_km = np.asarray(
            pkg["z_km"],
            dtype=np.float64,
        )

        lat_deg = np.asarray(
            pkg["lat_deg"],
            dtype=np.float64,
        )

        lon_deg = np.asarray(
            pkg["lon_deg"],
            dtype=np.float64,
        )

        vp_km_s = np.asarray(
            pkg["vp_km_s"],
            dtype=np.float64,
        )

        vs_km_s = np.asarray(
            pkg["vs_km_s"],
            dtype=np.float64,
        )

        vpvs_joint = np.asarray(
            pkg["vpvs_joint"],
            dtype=np.float64,
        )

    profile = infer_profile_from_geometry(
        args.geometry_csv,
        geod,
    )

    zhang = infer_zhang_axes(
        x_km=x_km,
        y_km=y_km,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        geod=geod,
    )

    # Current SAFOD profile origin relative to native Zhang origin.
    (
        profile_origin_e_m,
        profile_origin_n_m,
    ) = geodetic_offset_en(
        geod,
        lon0=zhang["origin_lon"],
        lat0=zhang["origin_lat"],
        lon1=profile["lon0"],
        lat1=profile["lat0"],
    )

    (
        profile_origin_zhang_x_m,
        profile_origin_zhang_y_m,
    ) = en_to_zhang_xy(
        east_m=np.array(
            [
                profile_origin_e_m
            ]
        ),
        north_m=np.array(
            [
                profile_origin_n_m
            ]
        ),
        x_azimuth_deg=zhang[
            "x_azimuth_deg"
        ],
        y_azimuth_deg=zhang[
            "y_azimuth_deg"
        ],
    )

    profile_origin_zhang_x_m = float(
        profile_origin_zhang_x_m[0]
    )

    profile_origin_zhang_y_m = float(
        profile_origin_zhang_y_m[0]
    )

    # How one metre along our current 2-D profile maps into native Zhang XY.
    (
        dzhang_x_ds,
        dzhang_y_ds,
    ) = en_to_zhang_xy(
        east_m=np.array(
            [
                profile["u_e"]
            ]
        ),
        north_m=np.array(
            [
                profile["u_n"]
            ]
        ),
        x_azimuth_deg=zhang[
            "x_azimuth_deg"
        ],
        y_azimuth_deg=zhang[
            "y_azimuth_deg"
        ],
    )

    dzhang_x_ds = float(
        dzhang_x_ds[0]
    )

    dzhang_y_ds = float(
        dzhang_y_ds[0]
    )

    # Solver coordinates.
    x_model_m = np.arange(
        args.x_min_m,
        args.x_max_m
        + 0.5
        * args.dx_m,
        args.dx_m,
        dtype=np.float64,
    )

    depth_m = np.arange(
        0.0,
        args.z_max_m
        + 0.5
        * args.dz_m,
        args.dz_m,
        dtype=np.float64,
    )

    zhang_x_km = (
        profile_origin_zhang_x_m
        + dzhang_x_ds
        * x_model_m
    ) / 1000.0

    zhang_y_km = (
        profile_origin_zhang_y_m
        + dzhang_y_ds
        * x_model_m
    ) / 1000.0

    # Positive downward from MSL.
    zhang_depth_km_raw = (
        depth_m
        - args.surface_elevation_m
    ) / 1000.0

    # Explicit shallow constant extrapolation to the first physical
    # tomography node.
    z_top_km = -0.5

    zhang_depth_km = np.maximum(
        zhang_depth_km_raw,
        z_top_km,
    )

    shallow_clipped_mask = (
        zhang_depth_km_raw
        < z_top_km
    )

    vp_interp, vp_z_nodes = make_interpolator(
        z_km=z_km,
        y_km=y_km,
        x_km=x_km,
        cube=vp_km_s,
    )

    vs_interp, _ = make_interpolator(
        z_km=z_km,
        y_km=y_km,
        x_km=x_km,
        cube=vs_km_s,
    )

    vpvs_interp, _ = make_interpolator(
        z_km=z_km,
        y_km=y_km,
        x_km=x_km,
        cube=vpvs_joint,
    )

    # Build all 2-D query points:
    # field shape will be (nx_section, nz_section).
    xx, zz = np.meshgrid(
        zhang_x_km,
        zhang_depth_km,
        indexing="ij",
    )

    yy = np.broadcast_to(
        zhang_y_km[
            :,
            None,
        ],
        xx.shape,
    )

    query = np.column_stack(
        [
            zz.ravel(),
            yy.ravel(),
            xx.ravel(),
        ]
    )

    vp_section_km_s = vp_interp(
        query
    ).reshape(
        xx.shape
    )

    vs_section_km_s = vs_interp(
        query
    ).reshape(
        xx.shape
    )

    vpvs_section_joint = vpvs_interp(
        query
    ).reshape(
        xx.shape
    )

    # Save section.
    section_npz = (
        processed_dir
        / "zhang2009_safod_section_2d.npz"
    )

    np.savez_compressed(
        section_npz,
        source=np.array(
            "Zhang, Thurber & Bedrosian (2009); "
            "sampled onto current SAFOD 2-D section"
        ),
        horizontal_registration=np.array(
            "Zhang inversion_grid.dat + current prepared georeferenced "
            "SAFOD cable geometry"
        ),
        vertical_registration=np.array(
            "z_zhang_km=(solver_depth_m-surface_elevation_m)/1000; "
            "positive down from MSL; values shallower than -0.5 km clipped "
            "to top physical Zhang node"
        ),
        surface_elevation_m=np.array(
            args.surface_elevation_m,
            dtype=np.float64,
        ),
        x_model_m=x_model_m,
        depth_m=depth_m,
        zhang_x_km=zhang_x_km,
        zhang_y_km=zhang_y_km,
        zhang_depth_km=zhang_depth_km,
        vp_mps=vp_section_km_s
        * 1000.0,
        vs_mps=vs_section_km_s
        * 1000.0,
        vpvs_joint=vpvs_section_joint,
        profile_origin_lat=np.array(
            profile["lat0"],
            dtype=np.float64,
        ),
        profile_origin_lon=np.array(
            profile["lon0"],
            dtype=np.float64,
        ),
        profile_u_e=np.array(
            profile["u_e"],
            dtype=np.float64,
        ),
        profile_u_n=np.array(
            profile["u_n"],
            dtype=np.float64,
        ),
        profile_azimuth_deg=np.array(
            profile[
                "profile_azimuth_deg"
            ],
            dtype=np.float64,
        ),
        zhang_origin_lat=np.array(
            zhang["origin_lat"],
            dtype=np.float64,
        ),
        zhang_origin_lon=np.array(
            zhang["origin_lon"],
            dtype=np.float64,
        ),
        zhang_x_azimuth_deg=np.array(
            zhang[
                "x_azimuth_deg"
            ],
            dtype=np.float64,
        ),
        zhang_y_azimuth_deg=np.array(
            zhang[
                "y_azimuth_deg"
            ],
            dtype=np.float64,
        ),
        dzhang_x_ds=np.array(
            dzhang_x_ds,
            dtype=np.float64,
        ),
        dzhang_y_ds=np.array(
            dzhang_y_ds,
            dtype=np.float64,
        ),
    )

    coordinate_csv = (
        processed_dir
        / "zhang2009_safod_section_coordinates.csv"
    )

    pd.DataFrame(
        {
            "x_model_m": x_model_m,
            "zhang_x_km": zhang_x_km,
            "zhang_y_km": zhang_y_km,
        }
    ).to_csv(
        coordinate_csv,
        index=False,
    )

    geo = profile[
        "geometry"
    ]

    cable_x_m = geo[
        "X_2D_m"
    ].to_numpy(
        dtype=np.float64
    )

    cable_z_m = geo[
        "TVD_m"
    ].to_numpy(
        dtype=np.float64
    )

    # Registration summary.
    summary = "\n".join(
        [
            "Zhang et al. (2009) -> current SAFOD 2-D section",
            "================================================",
            "",
            "Current SAFOD section:",
            f"  geometry CSV          : {args.geometry_csv}",
            f"  origin lat/lon        : {profile['lat0']:.12f}, "
            f"{profile['lon0']:.12f}",
            f"  profile u_E,u_N       : {profile['u_e']:.8f}, "
            f"{profile['u_n']:.8f}",
            f"  profile azimuth       : {profile['profile_azimuth_deg']:.6f} deg",
            f"  frame reconstruction RMS: {profile['horizontal_fit_rms_m']:.3f} m",
            f"  frame reconstruction max: {profile['horizontal_fit_max_m']:.3f} m",
            "",
            "Native Zhang coordinates:",
            f"  origin lat/lon        : {zhang['origin_lat']:.12f}, "
            f"{zhang['origin_lon']:.12f}",
            f"  +X azimuth            : {zhang['x_azimuth_deg']:.6f} deg",
            f"  +Y azimuth            : {zhang['y_azimuth_deg']:.6f} deg",
            "",
            "Section in Zhang XY:",
            f"  profile origin X      : {profile_origin_zhang_x_m:.3f} m",
            f"  profile origin Y      : {profile_origin_zhang_y_m:.3f} m",
            f"  dZhangX / ds          : {dzhang_x_ds:.8f}",
            f"  dZhangY / ds          : {dzhang_y_ds:.8f}",
            f"  Zhang X range         : {zhang_x_km.min():.4f} .. "
            f"{zhang_x_km.max():.4f} km",
            f"  Zhang Y range         : {zhang_y_km.min():.4f} .. "
            f"{zhang_y_km.max():.4f} km",
            "",
            "Vertical registration:",
            f"  ground elevation      : {args.surface_elevation_m:.2f} m MSL",
            "  Zhang z convention    : depth relative to MSL, positive downward",
            f"  physical z nodes used : {vp_z_nodes.tolist()} km",
            f"  raw Zhang z at surface: {zhang_depth_km_raw[0]:.5f} km",
            f"  top node clipping     : z < {z_top_km:.3f} km -> {z_top_km:.3f} km",
            f"  clipped solver depth  : 0 .. "
            f"{max(0.0, args.surface_elevation_m - 500.0):.2f} m",
            f"  model bottom Zhang z  : {zhang_depth_km_raw[-1]:.5f} km",
            "",
            "Output section:",
            f"  x range               : {x_model_m.min():.1f} .. "
            f"{x_model_m.max():.1f} m",
            f"  depth range           : {depth_m.min():.1f} .. "
            f"{depth_m.max():.1f} m",
            f"  dx,dz                 : {args.dx_m:.1f}, {args.dz_m:.1f} m",
            f"  shape                  : {vp_section_km_s.shape} = (x, depth)",
            f"  Vp range              : {vp_section_km_s.min():.3f} .. "
            f"{vp_section_km_s.max():.3f} km/s",
            f"  Vs range              : {vs_section_km_s.min():.3f} .. "
            f"{vs_section_km_s.max():.3f} km/s",
            f"  joint Vp/Vs range     : {vpvs_section_joint.min():.3f} .. "
            f"{vpvs_section_joint.max():.3f}",
            "",
            "Important:",
            "  5 m sampling is only interpolation of the coarse tomography;",
            "  it does not imply 5 m tomographic resolution.",
            "",
        ]
    )

    summary_path = (
        qc_dir
        / "zhang2009_safod_section_registration.txt"
    )

    summary_path.write_text(
        summary,
        encoding="utf-8",
    )

    # Map-view QC in native Zhang XY.
    local_x_mask = (
        np.abs(
            x_km
        )
        <= 10.0
    )

    local_y_mask = (
        np.abs(
            y_km
        )
        <= 8.0
    )

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    for y_value in y_km[
        local_y_mask
    ]:
        ax.plot(
            x_km[
                local_x_mask
            ],
            np.full(
                np.count_nonzero(
                    local_x_mask
                ),
                y_value,
            ),
            linewidth=0.7,
        )

    for x_value in x_km[
        local_x_mask
    ]:
        ax.plot(
            np.full(
                np.count_nonzero(
                    local_y_mask
                ),
                x_value,
            ),
            y_km[
                local_y_mask
            ],
            linewidth=0.7,
        )

    ax.plot(
        zhang_x_km,
        zhang_y_km,
        linewidth=2.0,
        label="Current SAFOD 2-D section",
    )

    cable_zhang_x_km = (
        profile_origin_zhang_x_m
        + dzhang_x_ds
        * cable_x_m
    ) / 1000.0

    cable_zhang_y_km = (
        profile_origin_zhang_y_m
        + dzhang_y_ds
        * cable_x_m
    ) / 1000.0

    ax.scatter(
        cable_zhang_x_km,
        cable_zhang_y_km,
        s=8,
        label="DAS cable horizontal projection",
    )

    ax.set_xlabel(
        "Zhang X [km]"
    )

    ax.set_ylabel(
        "Zhang Y [km]"
    )

    ax.set_title(
        "Current SAFOD modelling section in Zhang et al. (2009) XY grid"
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.legend()

    fig.tight_layout()

    map_path = (
        qc_dir
        / "zhang2009_safod_section_map.png"
    )

    fig.savefig(
        map_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    plot_section(
        field=vp_section_km_s,
        x_model_m=x_model_m,
        depth_m=depth_m,
        cable_x_m=cable_x_m,
        cable_z_m=cable_z_m,
        quantity="Vp",
        units="km/s",
        output=(
            qc_dir
            / "zhang2009_safod_section_vp.png"
        ),
    )

    plot_section(
        field=vs_section_km_s,
        x_model_m=x_model_m,
        depth_m=depth_m,
        cable_x_m=cable_x_m,
        cable_z_m=cable_z_m,
        quantity="Vs",
        units="km/s",
        output=(
            qc_dir
            / "zhang2009_safod_section_vs.png"
        ),
    )

    plot_section(
        field=vpvs_section_joint,
        x_model_m=x_model_m,
        depth_m=depth_m,
        cable_x_m=cable_x_m,
        cable_z_m=cable_z_m,
        quantity="joint Vp/Vs",
        units="ratio",
        output=(
            qc_dir
            / "zhang2009_safod_section_vpvs.png"
        ),
    )

    print(summary)
    print("Saved:")
    print(f"  {section_npz}")
    print(f"  {coordinate_csv}")
    print(f"  {summary_path}")
    print(f"  {map_path}")
    print(
        "  "
        + str(
            qc_dir
            / "zhang2009_safod_section_vp.png"
        )
    )
    print(
        "  "
        + str(
            qc_dir
            / "zhang2009_safod_section_vs.png"
        )
    )
    print(
        "  "
        + str(
            qc_dir
            / "zhang2009_safod_section_vpvs.png"
        )
    )


if __name__ == "__main__":
    main()
