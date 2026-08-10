from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from src.grid import Grid2D
from src.model import ElasticModel2D
from src.solver_numpy import max_stable_dt

from .zhang2009 import (
    DEFAULT_ZHANG_SECTION,
    build_zhang2009_model,
)


DEFAULT_BONESS_ZOBACK2006_CSV = Path(
    "data/safod/velocity_models/"
    "boness_zoback_2006/"
    "processed/"
    "boness_zoback2006_fig3_4_vp_vs_digitized.csv"
)


def _check_elastic_physicality(
    vp: np.ndarray,
    vs: np.ndarray,
    rho: np.ndarray,
) -> None:
    """Strict isotropic-elastic sanity checks."""
    for name, arr in (("vp", vp), ("vs", vs), ("rho", rho)):
        if np.any(~np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values.")
        if np.any(arr <= 0.0):
            raise ValueError(f"{name} contains non-positive values.")

    if np.any(vp <= vs):
        raise ValueError("Model contains Vp <= Vs.")

    mu = rho * vs**2
    lam = rho * (vp**2 - 2.0 * vs**2)

    if np.any(mu <= 0.0):
        raise ValueError(f"Non-positive mu; min={mu.min():.6e}")

    if np.any(lam < 0.0):
        ratio = vp / vs
        raise ValueError(
            "Negative lambda after Boness-Zoback correction. "
            f"min(Vp/Vs)={ratio.min():.6f}, "
            f"min(lambda)={lam.min():.6e}."
        )


def _make_safe_grid_like(
    spatial_grid,
    *,
    vp: np.ndarray,
    requested_dt: float | None,
    half_order: int,
    cfl_safety: float,
) -> Grid2D:
    """Keep x/z/nt unchanged and choose a CFL-safe dt."""
    dt_max = max_stable_dt(
        float(np.max(vp)),
        float(spatial_grid.dx),
        float(spatial_grid.dz),
        int(half_order),
        safety=float(cfl_safety),
        use_ts_sfd=False,
    )

    if requested_dt is None:
        dt = float(dt_max)
    else:
        dt = float(requested_dt)
        if dt > dt_max:
            raise ValueError(
                f"Requested dt={dt:.6e} s exceeds "
                f"CFL limit {dt_max:.6e} s."
            )

    return Grid2D(
        nx=int(spatial_grid.nx),
        nz=int(spatial_grid.nz),
        dx=float(spatial_grid.dx),
        dz=float(spatial_grid.dz),
        nt=int(spatial_grid.nt),
        dt=dt,
        x0=float(spatial_grid.x0),
        z0=float(spatial_grid.z0),
    )


def _cosine_window(
    coordinate: np.ndarray,
    *,
    inner_min: float,
    inner_max: float,
    taper: float,
) -> np.ndarray:
    """Unit window with cosine tapers outside the inner interval."""
    coordinate = np.asarray(coordinate, dtype=np.float64)

    if inner_max <= inner_min:
        raise ValueError("inner_max must exceed inner_min.")
    if taper <= 0.0:
        raise ValueError("taper must be positive.")

    w = np.zeros_like(coordinate, dtype=np.float64)

    inside = (coordinate >= inner_min) & (coordinate <= inner_max)
    w[inside] = 1.0

    left = (
        (coordinate >= inner_min - taper)
        & (coordinate < inner_min)
    )
    if np.any(left):
        u = (
            coordinate[left]
            - (inner_min - taper)
        ) / taper
        w[left] = 0.5 * (1.0 - np.cos(np.pi * u))

    right = (
        (coordinate > inner_max)
        & (coordinate <= inner_max + taper)
    )
    if np.any(right):
        u = (
            coordinate[right]
            - inner_max
        ) / taper
        w[right] = 0.5 * (1.0 + np.cos(np.pi * u))

    return w


def _load_boness_log(
    csv_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the digitized Boness & Zoback (2006) Main-Hole Vp/Vs log.

    The source figure uses measured depth (MD).  Prefer the 20-m-smoothed
    raster digitization; fall back to raw values if necessary.
    """
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Boness-Zoback digitized CSV not found: {path}"
        )

    df = pd.read_csv(path)

    if "md_m" not in df.columns:
        raise ValueError(
            f"Boness-Zoback CSV has no 'md_m' column: {path}"
        )

    vp_candidates = (
        "vp_mps_smooth20m",
        "vp_mps_raw",
    )
    vs_candidates = (
        "vs_mps_smooth20m",
        "vs_mps_raw",
    )

    vp_col = next(
        (c for c in vp_candidates if c in df.columns),
        None,
    )
    vs_col = next(
        (c for c in vs_candidates if c in df.columns),
        None,
    )

    if vp_col is None or vs_col is None:
        raise ValueError(
            "Boness-Zoback CSV must contain smoothed or raw Vp/Vs "
            f"columns. Columns: {list(df.columns)}"
        )

    md = pd.to_numeric(df["md_m"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    vp = pd.to_numeric(df[vp_col], errors="coerce").to_numpy(
        dtype=np.float64
    )
    vs = pd.to_numeric(df[vs_col], errors="coerce").to_numpy(
        dtype=np.float64
    )

    valid = (
        np.isfinite(md)
        & np.isfinite(vp)
        & np.isfinite(vs)
        & (vp > 0.0)
        & (vs > 0.0)
    )

    md = md[valid]
    vp = vp[valid]
    vs = vs[valid]

    if md.size < 20:
        raise ValueError(
            "Too few valid Boness-Zoback Vp/Vs samples."
        )

    order = np.argsort(md)
    md = md[order]
    vp = vp[order]
    vs = vs[order]

    if np.any(np.diff(md) <= 0.0):
        unique_md, unique_index = np.unique(
            md,
            return_index=True,
        )
        md = unique_md
        vp = vp[unique_index]
        vs = vs[unique_index]

    return md, vp, vs


def _load_downleg_geometry(
    geometry_csv: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load solver geometry as MD -> X_2D/TVD.

    The event-specific projected geometry is already registered to physical
    reference channels, so this converts the published Main-Hole MD log into
    the same 2-D solver frame without treating MD as vertical depth.
    """
    path = Path(geometry_csv)

    if not path.exists():
        raise FileNotFoundError(path)

    geo = pd.read_csv(path)

    required = (
        "MD_m",
        "X_2D_m",
        "Z_2D_m",
    )

    missing = [
        c for c in required
        if c not in geo.columns
    ]
    if missing:
        raise ValueError(
            f"Geometry CSV missing {missing}: {path}"
        )

    work = pd.DataFrame(
        {
            c: pd.to_numeric(
                geo[c],
                errors="coerce",
            )
            for c in required
        }
    ).dropna()

    if len(work) < 20:
        raise ValueError(
            "Too few valid MD/X/Z geometry rows."
        )

    # Event mapping can contain interpolated physical channels; collapse any
    # accidental duplicate MD values before interpolation.
    work = (
        work.groupby("MD_m", as_index=False)
        .mean(numeric_only=True)
        .sort_values("MD_m")
        .reset_index(drop=True)
    )

    md = work["MD_m"].to_numpy(dtype=np.float64)
    x = work["X_2D_m"].to_numpy(dtype=np.float64)
    z = work["Z_2D_m"].to_numpy(dtype=np.float64)

    if np.any(np.diff(md) <= 0.0):
        raise RuntimeError(
            "Geometry MD must be strictly increasing after cleanup."
        )

    return md, x, z


def _smooth_ratio_along_md(
    md_m: np.ndarray,
    ratio: np.ndarray,
    sigma_m: float,
) -> np.ndarray:
    """Smooth a 1-D log ratio in measured-depth coordinates."""
    if sigma_m <= 0.0:
        return np.asarray(ratio, dtype=np.float64).copy()

    dmd = float(np.median(np.diff(md_m)))
    if not np.isfinite(dmd) or dmd <= 0.0:
        raise ValueError("Invalid Boness-Zoback MD sampling.")

    sigma_samples = float(sigma_m) / dmd

    return gaussian_filter1d(
        np.asarray(ratio, dtype=np.float64),
        sigma=sigma_samples,
        mode="nearest",
    )


def _build_borehole_correction(
    *,
    grid,
    zhang_model,
    geometry_csv: str | Path,
    boness_csv: str | Path,
    borehole_sigma_m: float,
    vertical_taper_m: float,
    ratio_smooth_sigma_m: float,
    correction_strength: float,
    support_cutoff_sigma: float,
) -> dict:
    """
    Build smooth Vp/Vs multiplicative corrections around the Main Hole.

    Boness/Zoback absolute velocities are first converted to ratios relative
    to Zhang sampled at the same physical borehole locations.  Those ratios
    are then spread smoothly around the borehole with a Gaussian distance
    weight, while preserving the Zhang 2-D regional structure.
    """
    if borehole_sigma_m <= 0.0:
        raise ValueError("borehole_sigma_m must be positive.")
    if vertical_taper_m <= 0.0:
        raise ValueError("vertical_taper_m must be positive.")
    if not (0.0 <= correction_strength <= 1.5):
        raise ValueError(
            "correction_strength should be between 0 and 1.5."
        )
    if support_cutoff_sigma <= 0.0:
        raise ValueError("support_cutoff_sigma must be positive.")

    log_md, log_vp, log_vs = _load_boness_log(
        boness_csv
    )
    geom_md, geom_x, geom_z = _load_downleg_geometry(
        geometry_csv
    )

    if log_md.min() < geom_md.min() or log_md.max() > geom_md.max():
        raise ValueError(
            "Boness-Zoback MD interval lies outside available "
            "down-leg geometry: "
            f"log={log_md.min():.1f}..{log_md.max():.1f} m, "
            f"geometry={geom_md.min():.1f}..{geom_md.max():.1f} m."
        )

    log_x = np.interp(
        log_md,
        geom_md,
        geom_x,
    )
    log_z = np.interp(
        log_md,
        geom_md,
        geom_z,
    )

    vp_interp = RegularGridInterpolator(
        (
            np.asarray(grid.x, dtype=np.float64),
            np.asarray(grid.z, dtype=np.float64),
        ),
        np.asarray(zhang_model.vp, dtype=np.float64),
        method="linear",
        bounds_error=True,
    )
    vs_interp = RegularGridInterpolator(
        (
            np.asarray(grid.x, dtype=np.float64),
            np.asarray(grid.z, dtype=np.float64),
        ),
        np.asarray(zhang_model.vs, dtype=np.float64),
        method="linear",
        bounds_error=True,
    )

    log_points = np.column_stack(
        [log_x, log_z]
    )

    zhang_vp_log = vp_interp(log_points)
    zhang_vs_log = vs_interp(log_points)

    vp_ratio_log_raw = log_vp / zhang_vp_log
    vs_ratio_log_raw = log_vs / zhang_vs_log

    vp_ratio_log = _smooth_ratio_along_md(
        log_md,
        vp_ratio_log_raw,
        ratio_smooth_sigma_m,
    )
    vs_ratio_log = _smooth_ratio_along_md(
        log_md,
        vs_ratio_log_raw,
        ratio_smooth_sigma_m,
    )

    X, Z = np.meshgrid(
        np.asarray(grid.x, dtype=np.float64),
        np.asarray(grid.z, dtype=np.float64),
        indexing="ij",
    )
    points = np.column_stack(
        [X.ravel(), Z.ravel()]
    )

    tree = cKDTree(log_points)

    k = min(4, log_points.shape[0])
    distance, index = tree.query(
        points,
        k=k,
    )

    if k == 1:
        distance = distance[:, None]
        index = index[:, None]

    # Smooth interpolation along the discretized borehole path.
    interpolation_weight = 1.0 / (
        distance + 2.5
    ) ** 2

    interpolation_weight /= np.sum(
        interpolation_weight,
        axis=1,
        keepdims=True,
    )

    vp_ratio_near = np.sum(
        interpolation_weight
        * vp_ratio_log[index],
        axis=1,
    )
    vs_ratio_near = np.sum(
        interpolation_weight
        * vs_ratio_log[index],
        axis=1,
    )

    dmin = distance[:, 0]

    radial_weight = np.exp(
        -0.5
        * (
            dmin
            / float(borehole_sigma_m)
        ) ** 2
    )

    radial_weight[
        dmin
        > support_cutoff_sigma
        * float(borehole_sigma_m)
    ] = 0.0

    z_min = float(np.min(log_z))
    z_max = float(np.max(log_z))

    vertical_weight = _cosine_window(
        Z.ravel(),
        inner_min=z_min,
        inner_max=z_max,
        taper=float(vertical_taper_m),
    )

    weight = (
        radial_weight
        * vertical_weight
        * float(correction_strength)
    )

    vp_ratio_field = (
        1.0
        + weight
        * (
            vp_ratio_near
            - 1.0
        )
    ).reshape(X.shape)

    vs_ratio_field = (
        1.0
        + weight
        * (
            vs_ratio_near
            - 1.0
        )
    ).reshape(X.shape)

    return {
        "vp_ratio_field": vp_ratio_field,
        "vs_ratio_field": vs_ratio_field,
        "weight": weight.reshape(X.shape),
        "log_md_m": log_md,
        "log_x_m": log_x,
        "log_z_m": log_z,
        "log_vp_mps": log_vp,
        "log_vs_mps": log_vs,
        "zhang_vp_log_mps": zhang_vp_log,
        "zhang_vs_log_mps": zhang_vs_log,
        "vp_ratio_log_raw": vp_ratio_log_raw,
        "vs_ratio_log_raw": vs_ratio_log_raw,
        "vp_ratio_log": vp_ratio_log,
        "vs_ratio_log": vs_ratio_log,
        "z_min_m": z_min,
        "z_max_m": z_max,
    }


def build_zhang2009_boness2006_model(
    *,
    geom_file: str | Path,
    boness_log_csv: str | Path = DEFAULT_BONESS_ZOBACK2006_CSV,
    section_npz: str | Path = DEFAULT_ZHANG_SECTION,
    borehole_sigma_m: float = 500.0,
    vertical_taper_m: float = 300.0,
    ratio_smooth_sigma_m: float = 30.0,
    correction_strength: float = 1.0,
    support_cutoff_sigma: float = 3.0,
    build_initial_model: bool = True,
    verbose: bool = True,
    **base_builder_kwargs,
):
    """
    Zhang regional tomography calibrated to Boness & Zoback (2006) Vp/Vs.

    The published Main-Hole log is treated as a 1-D borehole constraint,
    not as a new regional 2-D model.  Absolute log velocities are converted
    to multiplicative corrections relative to Zhang at the same physical
    MD -> X_2D/TVD locations, then spread smoothly around the borehole.

    This preserves Zhang's lateral structure while enforcing the measured
    borehole velocity trend near the Main Hole.
    """
    if not build_initial_model:
        raise ValueError(
            "zhang2009_boness2006 is an initial-model builder."
        )

    kwargs = dict(base_builder_kwargs)
    requested_dt = kwargs.pop("dt", None)
    half_order = int(kwargs.get("half_order", 2))
    cfl_safety = float(kwargs.get("cfl_safety", 0.80))

    (
        zhang_grid,
        zhang_model,
        x_cable,
        z_cable,
        zhang_metadata,
    ) = build_zhang2009_model(
        geom_file=geom_file,
        section_npz=section_npz,
        build_initial_model=True,
        verbose=False,
        dt=None,
        **kwargs,
    )

    correction = _build_borehole_correction(
        grid=zhang_grid,
        zhang_model=zhang_model,
        geometry_csv=geom_file,
        boness_csv=boness_log_csv,
        borehole_sigma_m=borehole_sigma_m,
        vertical_taper_m=vertical_taper_m,
        ratio_smooth_sigma_m=ratio_smooth_sigma_m,
        correction_strength=correction_strength,
        support_cutoff_sigma=support_cutoff_sigma,
    )

    vp = (
        np.asarray(zhang_model.vp, dtype=np.float64)
        * correction["vp_ratio_field"]
    )
    vs = (
        np.asarray(zhang_model.vs, dtype=np.float64)
        * correction["vs_ratio_field"]
    )
    rho = np.asarray(
        zhang_model.rho,
        dtype=np.float64,
    ).copy()

    _check_elastic_physicality(
        vp,
        vs,
        rho,
    )

    grid = _make_safe_grid_like(
        zhang_grid,
        vp=vp,
        requested_dt=requested_dt,
        half_order=half_order,
        cfl_safety=cfl_safety,
    )

    model = ElasticModel2D(
        grid=grid,
        vp=vp,
        vs=vs,
        rho=rho,
    )

    notes = (
        "Zhang et al. (2009) direct Vp/Vs regional tomography calibrated "
        "to the digitized Boness & Zoback (2006) SAFOD Main-Hole sonic "
        "Vp/Vs log. Published measured depth is converted through the "
        "registered SAFOD down-leg geometry to solver X_2D/TVD. Log "
        "velocities are applied as multiplicative ratios relative to Zhang "
        "at the borehole, with a Gaussian distance weight around the hole. "
        f"Borehole sigma={float(borehole_sigma_m):.1f} m; "
        f"vertical taper={float(vertical_taper_m):.1f} m; "
        f"log-ratio smoothing sigma={float(ratio_smooth_sigma_m):.1f} m; "
        f"correction strength={float(correction_strength):.3f}."
    )

    metadata = replace(
        zhang_metadata,
        model_type="initial_zhang2009_boness2006",
        dt_s=float(grid.dt),
        notes=notes,
    )

    if verbose:
        active = correction["weight"] > 0.05

        print()
        print("SAFOD Zhang 2009 + Boness/Zoback 2006")
        print("======================================")
        print(f"log file              : {Path(boness_log_csv)}")
        print(
            f"log MD                 : "
            f"{correction['log_md_m'].min():.1f} .. "
            f"{correction['log_md_m'].max():.1f} m"
        )
        print(
            f"log TVD                : "
            f"{correction['log_z_m'].min():.1f} .. "
            f"{correction['log_z_m'].max():.1f} m"
        )
        print(
            f"log X_2D               : "
            f"{correction['log_x_m'].min():.1f} .. "
            f"{correction['log_x_m'].max():.1f} m"
        )
        print(
            f"Vp ratio at borehole   : "
            f"{correction['vp_ratio_log'].min():.3f} .. "
            f"{correction['vp_ratio_log'].max():.3f}"
        )
        print(
            f"Vs ratio at borehole   : "
            f"{correction['vs_ratio_log'].min():.3f} .. "
            f"{correction['vs_ratio_log'].max():.3f}"
        )
        print(
            f"borehole sigma         : "
            f"{float(borehole_sigma_m):.1f} m"
        )
        print(
            f"vertical taper         : "
            f"{float(vertical_taper_m):.1f} m"
        )
        print(
            f"active grid >0.05      : "
            f"{100.0 * np.mean(active):.2f}%"
        )
        print(
            f"final Vp range         : "
            f"{vp.min()/1000.0:.3f} .. "
            f"{vp.max()/1000.0:.3f} km/s"
        )
        print(
            f"final Vs range         : "
            f"{vs.min()/1000.0:.3f} .. "
            f"{vs.max()/1000.0:.3f} km/s"
        )
        print(
            f"final Vp/Vs            : "
            f"{(vp/vs).min():.3f} .. "
            f"{(vp/vs).max():.3f}"
        )
        print(f"dt                      : {grid.dt:.6e} s")

    return (
        grid,
        model,
        x_cable,
        z_cable,
        metadata,
    )
