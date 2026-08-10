from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from src.grid import Grid2D
from src.model import ElasticModel2D
from src.solver_numpy import max_stable_dt

from .bill_logs import build_bill_logs_model
from .smooth_prior import build_smooth_prior_model, fault_x_at_z
from .zhang2009 import (
    DEFAULT_ZHANG_SECTION,
    build_zhang2009_model,
)


def _assert_same_spatial_grid(a, b, label: str) -> None:
    same = (
        int(a.nx) == int(b.nx)
        and int(a.nz) == int(b.nz)
        and np.array_equal(np.asarray(a.x), np.asarray(b.x))
        and np.array_equal(np.asarray(a.z), np.asarray(b.z))
    )
    if not same:
        raise RuntimeError(f"Spatial-grid mismatch: {label}.")


def _check_elastic_physicality(
    vp: np.ndarray,
    vs: np.ndarray,
    rho: np.ndarray,
) -> None:
    for name, arr in (("vp", vp), ("vs", vs), ("rho", rho)):
        if np.any(~np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values.")
        if np.any(arr <= 0.0):
            raise ValueError(f"{name} contains non-positive values.")

    if np.any(vp <= vs):
        raise ValueError("Hybrid model contains Vp <= Vs.")

    mu = rho * vs**2
    lam = rho * (vp**2 - 2.0 * vs**2)

    if np.any(mu <= 0.0):
        raise ValueError(f"Non-positive mu; min={mu.min():.6e}")

    if np.any(lam < 0.0):
        ratio = vp / vs
        raise ValueError(
            "Negative lambda in hybrid model. "
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
                f"hybrid CFL limit {dt_max:.6e} s."
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
    coordinate = np.asarray(coordinate, dtype=np.float64)

    if inner_max <= inner_min:
        raise ValueError("inner_max must exceed inner_min.")
    if taper <= 0.0:
        raise ValueError("taper must be positive.")

    weight = np.zeros_like(coordinate, dtype=np.float64)

    inside = (coordinate >= inner_min) & (coordinate <= inner_max)
    weight[inside] = 1.0

    left = (
        (coordinate >= inner_min - taper)
        & (coordinate < inner_min)
    )
    if np.any(left):
        u = (
            coordinate[left] - (inner_min - taper)
        ) / taper
        weight[left] = 0.5 * (1.0 - np.cos(np.pi * u))

    right = (
        (coordinate > inner_max)
        & (coordinate <= inner_max + taper)
    )
    if np.any(right):
        u = (
            coordinate[right] - inner_max
        ) / taper
        weight[right] = 0.5 * (1.0 + np.cos(np.pi * u))

    return weight


def _pick_numeric_column(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
    label: str,
) -> np.ndarray:
    for name in candidates:
        if name in frame.columns:
            values = pd.to_numeric(
                frame[name],
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size:
                return values

    raise ValueError(
        f"Could not find a usable {label} column. "
        f"Tried {candidates}. Columns are: {list(frame.columns)}"
    )


def _load_bill_support(
    bill_logs_csv: str | Path,
) -> tuple[float, float, float, float]:
    path = Path(bill_logs_csv)

    if not path.exists():
        raise FileNotFoundError(
            f"Bill-log CSV not found: {path}"
        )

    log = pd.read_csv(path)

    if len(log) < 20:
        raise ValueError(
            f"Too few Bill-log rows for localization: {len(log)}"
        )

    offsets = _pick_numeric_column(
        log,
        (
            "section_offset_from_sdz_m",
            "offset_from_sdz_m",
        ),
        "section-offset",
    )

    tvd = _pick_numeric_column(
        log,
        (
            "tvd_m",
            "TVD_m",
            "tvd",
            "TVD",
        ),
        "TVD",
    )

    return (
        float(np.min(offsets)),
        float(np.max(offsets)),
        float(np.min(tvd)),
        float(np.max(tvd)),
    )


def build_hybrid_zhang2009_bill_logs_model(
    *,
    geom_file: str | Path,
    bill_logs_csv: str | Path,
    section_npz: str | Path = DEFAULT_ZHANG_SECTION,

    # Localize the measured Bill anomaly around the fault-zone section.
    anomaly_taper_m: float = 150.0,

    # Smooth continuation in depth:
    # sigma_z = half measured TVD span + this pad.
    depth_gaussian_pad_m: float = 175.0,

    # Mild anomaly-only smoothing.
    smooth_sigma_x_m: float = 15.0,
    smooth_sigma_z_m: float = 40.0,

    build_initial_model: bool = True,
    verbose: bool = True,
    **base_builder_kwargs,
):
    """
    Zhang regional tomography plus a smooth localized Bill-log anomaly.

    The Bill Vp/Vs anomaly is defined relative to the exact smooth background
    from which the standalone Bill model was constructed.  It is then applied
    multiplicatively to Zhang.

    The anomaly is localized laterally with a cosine taper and vertically with
    a Gaussian envelope centered on the measured Bill-log TVD interval.
    Gaussian smoothing is applied only to the localized anomaly, not to the
    full Zhang background.
    """
    if not build_initial_model:
        raise ValueError(
            "hybrid_zhang2009_bill_logs is an initial-model builder."
        )

    if anomaly_taper_m <= 0.0:
        raise ValueError("anomaly_taper_m must be positive.")
    if depth_gaussian_pad_m < 0.0:
        raise ValueError("depth_gaussian_pad_m must be >= 0.")
    if smooth_sigma_x_m < 0.0 or smooth_sigma_z_m < 0.0:
        raise ValueError("Smoothing sigmas must be >= 0.")

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

    (
        bill_grid,
        bill_model,
        _,
        _,
        bill_metadata,
    ) = build_bill_logs_model(
        geom_file=geom_file,
        log_csv=bill_logs_csv,
        build_initial_model=True,
        verbose=False,
        dt=None,
        **kwargs,
    )

    background_kwargs = dict(kwargs)
    background_kwargs["initial_cross_fault_contrast"] = 0.0
    background_kwargs["initial_fault_zone_velocity_reduction"] = 0.0

    (
        background_grid,
        bill_background,
        _,
        _,
        _,
    ) = build_smooth_prior_model(
        geom_file=geom_file,
        build_initial_model=True,
        dt=None,
        **background_kwargs,
    )

    _assert_same_spatial_grid(
        zhang_grid,
        bill_grid,
        "Zhang vs Bill",
    )
    _assert_same_spatial_grid(
        zhang_grid,
        background_grid,
        "Zhang vs Bill background",
    )

    vp_ratio_raw = (
        np.asarray(bill_model.vp, dtype=np.float64)
        / np.asarray(bill_background.vp, dtype=np.float64)
    )
    vs_ratio_raw = (
        np.asarray(bill_model.vs, dtype=np.float64)
        / np.asarray(bill_background.vs, dtype=np.float64)
    )

    if np.any(~np.isfinite(vp_ratio_raw)):
        raise ValueError("Non-finite Bill Vp anomaly ratio.")
    if np.any(~np.isfinite(vs_ratio_raw)):
        raise ValueError("Non-finite Bill Vs anomaly ratio.")

    (
        offset_min_m,
        offset_max_m,
        tvd_min_m,
        tvd_max_m,
    ) = _load_bill_support(bill_logs_csv)

    X, Z = np.meshgrid(
        np.asarray(zhang_grid.x, dtype=np.float64),
        np.asarray(zhang_grid.z, dtype=np.float64),
        indexing="ij",
    )

    x_sdz = fault_x_at_z(
        Z,
        x_tie_m=float(zhang_metadata.x_tie_m),
        z_tie_m=float(zhang_metadata.z_tie_m),
        fault_dip_deg=float(zhang_metadata.fault_dip_deg),
        fault_dip_sign=float(zhang_metadata.fault_dip_sign),
    )

    signed_offset_from_sdz_m = X - x_sdz

    lateral_weight = _cosine_window(
        signed_offset_from_sdz_m,
        inner_min=offset_min_m,
        inner_max=offset_max_m,
        taper=float(anomaly_taper_m),
    )

    tvd_center_m = 0.5 * (tvd_min_m + tvd_max_m)
    tvd_half_width_m = 0.5 * (tvd_max_m - tvd_min_m)
    tvd_sigma_m = (
        tvd_half_width_m
        + float(depth_gaussian_pad_m)
    )

    if tvd_sigma_m <= 0.0:
        raise ValueError(
            f"Non-positive vertical Gaussian sigma: {tvd_sigma_m}"
        )

    depth_weight = np.exp(
        -0.5
        * (
            (Z - tvd_center_m)
            / tvd_sigma_m
        ) ** 2
    )

    support_weight = lateral_weight * depth_weight

    if not np.any(support_weight > 0.0):
        raise RuntimeError(
            "Hybrid localization produced no active Bill-anomaly cells."
        )

    vp_delta_local = (
        support_weight
        * (vp_ratio_raw - 1.0)
    )
    vs_delta_local = (
        support_weight
        * (vs_ratio_raw - 1.0)
    )

    sigma_x_cells = (
        float(smooth_sigma_x_m)
        / float(zhang_grid.dx)
    )
    sigma_z_cells = (
        float(smooth_sigma_z_m)
        / float(zhang_grid.dz)
    )

    if sigma_x_cells > 0.0 or sigma_z_cells > 0.0:
        vp_delta = gaussian_filter(
            vp_delta_local,
            sigma=(sigma_x_cells, sigma_z_cells),
            mode="nearest",
        )
        vs_delta = gaussian_filter(
            vs_delta_local,
            sigma=(sigma_x_cells, sigma_z_cells),
            mode="nearest",
        )
    else:
        vp_delta = vp_delta_local
        vs_delta = vs_delta_local

    vp_ratio = 1.0 + vp_delta
    vs_ratio = 1.0 + vs_delta

    vp = (
        np.asarray(zhang_model.vp, dtype=np.float64)
        * vp_ratio
    )
    vs = (
        np.asarray(zhang_model.vs, dtype=np.float64)
        * vs_ratio
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

    active = support_weight > 0.05
    strong = support_weight > 0.50

    notes = (
        "Hybrid SAFOD initial model: Zhang et al. (2009) direct Vp/Vs "
        "tomography supplies the regional background. Ellsworth & Malin "
        "(2011) / Bill-log velocity anomalies are applied multiplicatively "
        "as a smooth localized perturbation. "
        f"Bill section-offset interval: {offset_min_m:.1f}.."
        f"{offset_max_m:.1f} m from SDZ; "
        f"lateral taper={float(anomaly_taper_m):.1f} m. "
        f"Bill TVD interval={tvd_min_m:.1f}..{tvd_max_m:.1f} m; "
        f"vertical Gaussian sigma={tvd_sigma_m:.1f} m "
        f"(pad={float(depth_gaussian_pad_m):.1f} m). "
        f"Localized anomaly smoothing sigmas: "
        f"sigma_x={float(smooth_sigma_x_m):.1f} m, "
        f"sigma_z={float(smooth_sigma_z_m):.1f} m. "
        "No direct smoothing of the regional Zhang background."
    )

    metadata = replace(
        zhang_metadata,
        model_type="initial_hybrid_zhang2009_bill_logs",
        dt_s=float(grid.dt),
        cross_fault_contrast=float(
            bill_metadata.cross_fault_contrast
        ),
        fault_zone_width_m=float(
            bill_metadata.fault_zone_width_m
        ),
        fault_zone_velocity_reduction=float(
            bill_metadata.fault_zone_velocity_reduction
        ),
        pilot_hole_lvz_strength=0.0,
        smoothing_sigma_m=0.0,
        notes=notes,
    )

    if verbose:
        print()
        print("SAFOD hybrid Zhang 2009 + Bill logs")
        print("====================================")
        print(
            f"Bill section interval : "
            f"{offset_min_m:+.1f} .. {offset_max_m:+.1f} m from SDZ"
        )
        print(
            f"cross-fault taper     : "
            f"{float(anomaly_taper_m):.1f} m each side"
        )
        print(
            f"Bill log TVD interval : "
            f"{tvd_min_m:.1f} .. {tvd_max_m:.1f} m"
        )
        print(
            f"vertical Gaussian σ   : "
            f"{tvd_sigma_m:.1f} m"
        )
        print(
            f"depth Gaussian pad    : "
            f"{float(depth_gaussian_pad_m):.1f} m"
        )
        print(
            f"anomaly smoothing σx,z: "
            f"{float(smooth_sigma_x_m):.1f} m, "
            f"{float(smooth_sigma_z_m):.1f} m"
        )
        print(
            f"support fraction >0.05: "
            f"{100.0 * np.mean(active):.2f}%"
        )
        print(
            f"support fraction >0.50: "
            f"{100.0 * np.mean(strong):.2f}%"
        )
        print(
            f"Vp ratio final        : "
            f"{vp_ratio.min():.3f} .. {vp_ratio.max():.3f}"
        )
        print(
            f"Vs ratio final        : "
            f"{vs_ratio.min():.3f} .. {vs_ratio.max():.3f}"
        )
        print(
            f"final Vp range        : "
            f"{vp.min()/1000.0:.3f} .. "
            f"{vp.max()/1000.0:.3f} km/s"
        )
        print(
            f"final Vs range        : "
            f"{vs.min()/1000.0:.3f} .. "
            f"{vs.max()/1000.0:.3f} km/s"
        )
        print(
            f"direct Vp/Vs          : "
            f"{(vp/vs).min():.3f} .. "
            f"{(vp/vs).max():.3f}"
        )
        print(
            f"dt                     : "
            f"{grid.dt:.6e} s"
        )
        print(
            "post-hybrid smoothing  : anomaly only"
        )

    return (
        grid,
        model,
        x_cable,
        z_cable,
        metadata,
    )
