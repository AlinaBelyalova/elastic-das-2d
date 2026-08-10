
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from src.grid import Grid2D
from src.model import ElasticModel2D
from src.solver_numpy import max_stable_dt

from .smooth_prior import build_smooth_prior_model


DEFAULT_ZHANG_SECTION = Path(
    "data/safod/velocity_models/"
    "zhang_thurber_bedrosian_2009/"
    "processed/zhang2009_safod_section_2d.npz"
)


def _check_elastic_physicality(
    vp: np.ndarray,
    vs: np.ndarray,
    rho: np.ndarray,
) -> None:
    vp = np.asarray(vp, dtype=np.float64)
    vs = np.asarray(vs, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)

    if (
        np.any(~np.isfinite(vp))
        or np.any(~np.isfinite(vs))
        or np.any(~np.isfinite(rho))
    ):
        raise ValueError("vp/vs/rho contain non-finite values.")

    if (
        np.any(vp <= 0.0)
        or np.any(vs <= 0.0)
        or np.any(rho <= 0.0)
    ):
        raise ValueError("vp/vs/rho must be positive.")

    if np.any(vp <= vs):
        raise ValueError("Vp must be strictly greater than Vs.")

    mu = rho * vs**2
    lam = rho * (vp**2 - 2.0 * vs**2)

    if np.any(mu <= 0.0):
        raise ValueError(
            f"Non-positive mu; min={float(np.min(mu)):.6e}"
        )

    if np.any(lam < 0.0):
        ratio = vp / vs
        raise ValueError(
            "Direct Zhang Vp/Vs gives negative lambda somewhere: "
            f"min(Vp/Vs)={float(np.min(ratio)):.6f}, "
            f"min(lambda)={float(np.min(lam)):.6e}. "
            "Do not silently modify the supplied Vp or Vs."
        )


def _load_section(
    section_npz: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = Path(section_npz)

    if not path.exists():
        raise FileNotFoundError(
            f"Zhang SAFOD section not found: {path}"
        )

    with np.load(path, allow_pickle=True) as pkg:
        required = (
            "x_model_m",
            "depth_m",
            "vp_mps",
            "vs_mps",
        )

        missing = [
            name
            for name in required
            if name not in pkg.files
        ]

        if missing:
            raise ValueError(
                f"Zhang section missing arrays: {missing}"
            )

        x = np.asarray(
            pkg["x_model_m"],
            dtype=np.float64,
        )
        z = np.asarray(
            pkg["depth_m"],
            dtype=np.float64,
        )
        vp = np.asarray(
            pkg["vp_mps"],
            dtype=np.float64,
        )
        vs = np.asarray(
            pkg["vs_mps"],
            dtype=np.float64,
        )

    if x.ndim != 1 or z.ndim != 1:
        raise ValueError(
            "x_model_m and depth_m must be one-dimensional."
        )

    expected_shape = (x.size, z.size)

    if vp.shape != expected_shape:
        raise ValueError(
            f"Vp shape {vp.shape} != {expected_shape}."
        )

    if vs.shape != expected_shape:
        raise ValueError(
            f"Vs shape {vs.shape} != {expected_shape}."
        )

    if (
        np.any(np.diff(x) <= 0.0)
        or np.any(np.diff(z) <= 0.0)
    ):
        raise ValueError(
            "Zhang section coordinates must be strictly increasing."
        )

    return x, z, vp, vs


def _sample_section_on_grid(
    *,
    grid,
    x_section: np.ndarray,
    z_section: np.ndarray,
    field: np.ndarray,
) -> np.ndarray:
    """
    Sample the extracted Zhang section on the FD grid.

    The extracted section spans the undamped scientific domain. Coordinates
    outside it are clipped only to provide constant nearest-edge extension
    into the absorbing side/bottom sponges.
    """
    interp = RegularGridInterpolator(
        (x_section, z_section),
        field,
        method="linear",
        bounds_error=True,
    )

    x_query = np.clip(
        np.asarray(grid.x, dtype=np.float64),
        float(x_section[0]),
        float(x_section[-1]),
    )
    z_query = np.clip(
        np.asarray(grid.z, dtype=np.float64),
        float(z_section[0]),
        float(z_section[-1]),
    )

    xx, zz = np.meshgrid(
        x_query,
        z_query,
        indexing="ij",
    )

    points = np.column_stack(
        (xx.ravel(), zz.ravel())
    )

    return interp(points).reshape(xx.shape)


def _grid_with_cfl_safe_dt(
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
        dt_s = float(dt_max)
    else:
        dt_s = float(requested_dt)

        if dt_s > dt_max:
            raise ValueError(
                f"Requested dt={dt_s:.6e} s exceeds the Zhang-model "
                f"CFL limit {dt_max:.6e} s."
            )

    return Grid2D(
        nx=int(spatial_grid.nx),
        nz=int(spatial_grid.nz),
        dx=float(spatial_grid.dx),
        dz=float(spatial_grid.dz),
        nt=int(spatial_grid.nt),
        dt=dt_s,
        x0=float(spatial_grid.x0),
        z0=float(spatial_grid.z0),
    )


def build_zhang2009_model(
    *,
    geom_file: str | Path,
    section_npz: str | Path = DEFAULT_ZHANG_SECTION,
    build_initial_model: bool = True,
    verbose: bool = True,
    **base_builder_kwargs,
):
    """
    Build the pure Zhang et al. (2009) SAFOD tomography initial model.

    Vp and Vs come directly from the extracted Zhang section. Density is the
    common smooth-background density used by the log-constrained model; Zhang
    does not supply density.
    """
    if not build_initial_model:
        raise ValueError(
            "zhang2009 is an initial-model builder; "
            "use build_initial_model=True."
        )

    kwargs = dict(base_builder_kwargs)

    requested_dt = kwargs.pop("dt", None)
    half_order = int(kwargs.get("half_order", 2))
    cfl_safety = float(kwargs.get("cfl_safety", 0.80))

    # Same density background used by the Bill-log builder: keep the depth
    # prior, but disable the hand-tuned lateral SAF velocity/density features.
    background_kwargs = dict(kwargs)
    background_kwargs["initial_cross_fault_contrast"] = 0.0
    background_kwargs[
        "initial_fault_zone_velocity_reduction"
    ] = 0.0

    (
        spatial_grid,
        background_model,
        x_cable,
        z_cable,
        base_metadata,
    ) = build_smooth_prior_model(
        geom_file=geom_file,
        build_initial_model=True,
        dt=None,
        **background_kwargs,
    )

    (
        x_section,
        z_section,
        vp_section,
        vs_section,
    ) = _load_section(section_npz)

    vp = _sample_section_on_grid(
        grid=spatial_grid,
        x_section=x_section,
        z_section=z_section,
        field=vp_section,
    )

    vs = _sample_section_on_grid(
        grid=spatial_grid,
        x_section=x_section,
        z_section=z_section,
        field=vs_section,
    )

    rho = np.asarray(
        background_model.rho,
        dtype=np.float64,
    ).copy()

    _check_elastic_physicality(
        vp,
        vs,
        rho,
    )

    grid = _grid_with_cfl_safe_dt(
        spatial_grid,
        vp=vp,
        requested_dt=requested_dt,
        half_order=half_order,
        cfl_safety=cfl_safety,
    )

    model = ElasticModel2D(
        grid=grid,
        vp=np.asarray(vp, dtype=np.float64),
        vs=np.asarray(vs, dtype=np.float64),
        rho=rho,
    )

    metadata = replace(
        base_metadata,
        model_type="initial_zhang2009_tomography",
        dt_s=float(grid.dt),
        cross_fault_contrast=0.0,
        fault_zone_width_m=0.0,
        fault_zone_velocity_reduction=0.0,
        smoothing_sigma_m=0.0,
        notes=(
            "Pure Zhang, Thurber & Bedrosian (2009) SAFOD tomography. "
            "Direct supplied Vp and direct supplied Vs are sampled onto the "
            "canonical SAFOD 2-D section. The 5 m grid is numerical "
            "interpolation only, not tomographic resolution. Values are "
            "constant-extended at the section edges only to fill absorbing "
            "sponges. No Bill/Ellsworth-Malin local velocity anomalies are "
            "added. Density uses the common smooth-background density prior."
        ),
    )

    if verbose:
        ratio = vp / vs

        print()
        print("SAFOD Zhang 2009 initial model")
        print("================================")
        print(f"section file : {Path(section_npz)}")
        print(
            "Vp range     : "
            f"{vp.min()/1000.0:.3f} .. "
            f"{vp.max()/1000.0:.3f} km/s"
        )
        print(
            "Vs range     : "
            f"{vs.min()/1000.0:.3f} .. "
            f"{vs.max()/1000.0:.3f} km/s"
        )
        print(
            "direct Vp/Vs : "
            f"{ratio.min():.3f} .. {ratio.max():.3f}"
        )
        print(
            f"grid         : {grid.nx} x {grid.nz}"
        )
        print(
            f"dt           : {grid.dt:.6e} s"
        )
        print(
            "duration     : "
            f"{(grid.nt - 1) * grid.dt:.3f} s"
        )

    return (
        grid,
        model,
        x_cable,
        z_cable,
        metadata,
    )
