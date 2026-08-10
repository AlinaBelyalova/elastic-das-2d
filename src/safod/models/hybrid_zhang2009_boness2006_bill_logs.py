from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from src.grid import Grid2D
from src.model import ElasticModel2D
from src.solver_numpy import max_stable_dt

from .boness_zoback2006 import (
    DEFAULT_BONESS_ZOBACK2006_CSV,
    build_zhang2009_boness2006_model,
)
from .hybrid_zhang2009_bill_logs import (
    build_hybrid_zhang2009_bill_logs_model,
)
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
        raise ValueError("Model contains Vp <= Vs.")

    mu = rho * vs**2
    lam = rho * (vp**2 - 2.0 * vs**2)

    if np.any(mu <= 0.0):
        raise ValueError(f"Non-positive mu; min={mu.min():.6e}")

    if np.any(lam < 0.0):
        ratio = vp / vs
        raise ValueError(
            "Negative lambda in Zhang+Boness+Bill model. "
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


def build_hybrid_zhang2009_boness2006_bill_logs_model(
    *,
    geom_file: str | Path,
    bill_logs_csv: str | Path,
    boness_log_csv: str | Path = DEFAULT_BONESS_ZOBACK2006_CSV,
    section_npz: str | Path = DEFAULT_ZHANG_SECTION,

    # Boness/Zoback regional borehole calibration.
    boness_borehole_sigma_m: float = 500.0,
    boness_vertical_taper_m: float = 300.0,
    boness_ratio_smooth_sigma_m: float = 30.0,
    boness_correction_strength: float = 1.0,
    boness_support_cutoff_sigma: float = 3.0,

    # Bill local fault-zone anomaly. These match the smoothed hybrid model.
    bill_anomaly_taper_m: float = 150.0,
    bill_depth_gaussian_pad_m: float = 175.0,
    bill_smooth_sigma_x_m: float = 15.0,
    bill_smooth_sigma_z_m: float = 40.0,

    build_initial_model: bool = True,
    verbose: bool = True,
    **base_builder_kwargs,
):
    """
    Build the combined SAFOD initial model:

        Zhang 2009 regional tomography
        + Boness & Zoback 2006 broad Main-Hole Vp/Vs calibration
        + Bill/Ellsworth-Malin local fault-zone anomaly.

    The Bill component is obtained as the exact multiplicative anomaly of the
    existing smoothed Zhang+Bill hybrid relative to pure Zhang. Applying that
    ratio to Zhang+Boness keeps the already validated Bill localization and
    smoothing unchanged.
    """
    if not build_initial_model:
        raise ValueError(
            "hybrid_zhang2009_boness2006_bill_logs is an "
            "initial-model builder."
        )

    kwargs = dict(base_builder_kwargs)
    requested_dt = kwargs.pop("dt", None)
    half_order = int(kwargs.get("half_order", 2))
    cfl_safety = float(kwargs.get("cfl_safety", 0.80))

    # ------------------------------------------------------------------
    # 1. Zhang + Boness/Zoback broad borehole calibration.
    # ------------------------------------------------------------------
    (
        boness_grid,
        boness_model,
        x_cable,
        z_cable,
        boness_metadata,
    ) = build_zhang2009_boness2006_model(
        geom_file=geom_file,
        boness_log_csv=boness_log_csv,
        section_npz=section_npz,
        borehole_sigma_m=boness_borehole_sigma_m,
        vertical_taper_m=boness_vertical_taper_m,
        ratio_smooth_sigma_m=boness_ratio_smooth_sigma_m,
        correction_strength=boness_correction_strength,
        support_cutoff_sigma=boness_support_cutoff_sigma,
        build_initial_model=True,
        verbose=False,
        dt=None,
        **kwargs,
    )

    # ------------------------------------------------------------------
    # 2. Existing smoothed Zhang + Bill hybrid.
    # ------------------------------------------------------------------
    (
        bill_grid,
        zhang_bill_model,
        _,
        _,
        bill_metadata,
    ) = build_hybrid_zhang2009_bill_logs_model(
        geom_file=geom_file,
        bill_logs_csv=bill_logs_csv,
        section_npz=section_npz,
        anomaly_taper_m=bill_anomaly_taper_m,
        depth_gaussian_pad_m=bill_depth_gaussian_pad_m,
        smooth_sigma_x_m=bill_smooth_sigma_x_m,
        smooth_sigma_z_m=bill_smooth_sigma_z_m,
        build_initial_model=True,
        verbose=False,
        dt=None,
        **kwargs,
    )

    # ------------------------------------------------------------------
    # 3. Pure Zhang reference, only to isolate the Bill multiplicative
    #    anomaly already produced by the existing hybrid builder.
    # ------------------------------------------------------------------
    (
        zhang_grid,
        zhang_model,
        _,
        _,
        _,
    ) = build_zhang2009_model(
        geom_file=geom_file,
        section_npz=section_npz,
        build_initial_model=True,
        verbose=False,
        dt=None,
        **kwargs,
    )

    _assert_same_spatial_grid(
        boness_grid,
        bill_grid,
        "Boness vs Zhang+Bill",
    )
    _assert_same_spatial_grid(
        boness_grid,
        zhang_grid,
        "Boness vs pure Zhang",
    )

    bill_vp_ratio = (
        np.asarray(zhang_bill_model.vp, dtype=np.float64)
        / np.asarray(zhang_model.vp, dtype=np.float64)
    )
    bill_vs_ratio = (
        np.asarray(zhang_bill_model.vs, dtype=np.float64)
        / np.asarray(zhang_model.vs, dtype=np.float64)
    )

    vp = (
        np.asarray(boness_model.vp, dtype=np.float64)
        * bill_vp_ratio
    )
    vs = (
        np.asarray(boness_model.vs, dtype=np.float64)
        * bill_vs_ratio
    )
    rho = np.asarray(
        boness_model.rho,
        dtype=np.float64,
    ).copy()

    _check_elastic_physicality(
        vp,
        vs,
        rho,
    )

    grid = _make_safe_grid_like(
        boness_grid,
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
        "Combined SAFOD initial model: Zhang et al. (2009) direct Vp/Vs "
        "regional tomography, broad Boness & Zoback (2006) Main-Hole "
        "Vp/Vs calibration in registered MD -> X_2D/TVD coordinates, and "
        "the existing smoothed/localized Ellsworth-Malin/Bill fault-zone "
        "multiplicative anomaly. "
        + boness_metadata.notes
        + " "
        + bill_metadata.notes
    )

    metadata = replace(
        boness_metadata,
        model_type=(
            "initial_hybrid_zhang2009_boness2006_bill_logs"
        ),
        dt_s=float(grid.dt),
        notes=notes,
    )

    if verbose:
        bill_active = (
            np.abs(bill_vp_ratio - 1.0) > 1.0e-3
        ) | (
            np.abs(bill_vs_ratio - 1.0) > 1.0e-3
        )

        print()
        print("SAFOD Zhang + Boness/Zoback + Bill")
        print("===================================")
        print(
            f"Boness borehole sigma  : "
            f"{float(boness_borehole_sigma_m):.1f} m"
        )
        print(
            f"Boness correction      : "
            f"{float(boness_correction_strength):.3f}"
        )
        print(
            f"Bill active fraction   : "
            f"{100.0 * np.mean(bill_active):.2f}%"
        )
        print(
            f"Bill Vp ratio          : "
            f"{bill_vp_ratio.min():.3f} .. "
            f"{bill_vp_ratio.max():.3f}"
        )
        print(
            f"Bill Vs ratio          : "
            f"{bill_vs_ratio.min():.3f} .. "
            f"{bill_vs_ratio.max():.3f}"
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
