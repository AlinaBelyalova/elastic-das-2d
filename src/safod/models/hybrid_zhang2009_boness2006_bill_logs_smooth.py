from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from src.grid import Grid2D
from src.model import ElasticModel2D
from src.solver_numpy import max_stable_dt

from .boness_zoback2006 import (
    DEFAULT_BONESS_ZOBACK2006_CSV,
    _load_boness_log,
    _load_downleg_geometry,
    _smooth_ratio_along_md,
)
from .hybrid_zhang2009_bill_logs import (
    build_hybrid_zhang2009_bill_logs_model,
)
from .zhang2009 import (
    DEFAULT_ZHANG_SECTION,
    build_zhang2009_model,
)


def _check_elastic_physicality(
    vp: np.ndarray,
    vs: np.ndarray,
    rho: np.ndarray,
) -> None:
    for name, arr in (
        ("vp", vp),
        ("vs", vs),
        ("rho", rho),
    ):
        if np.any(~np.isfinite(arr)):
            raise ValueError(
                f"{name} contains non-finite values."
            )
        if np.any(arr <= 0.0):
            raise ValueError(
                f"{name} contains non-positive values."
            )

    if np.any(vp <= vs):
        raise ValueError(
            "Model contains Vp <= Vs."
        )

    mu = rho * vs**2
    lam = rho * (
        vp**2
        - 2.0 * vs**2
    )

    if np.any(mu <= 0.0):
        raise ValueError(
            f"Non-positive mu; min={mu.min():.6e}"
        )

    if np.any(lam < 0.0):
        ratio = vp / vs
        raise ValueError(
            "Negative lambda in smooth combined model. "
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


def _cosine_window(
    coordinate: np.ndarray,
    *,
    inner_min: float,
    inner_max: float,
    taper: float,
) -> np.ndarray:

    coordinate = np.asarray(
        coordinate,
        dtype=np.float64,
    )

    if inner_max <= inner_min:
        raise ValueError(
            "inner_max must exceed inner_min."
        )

    if taper <= 0.0:
        raise ValueError(
            "taper must be positive."
        )

    w = np.zeros_like(
        coordinate,
        dtype=np.float64,
    )

    inside = (
        (coordinate >= inner_min)
        & (coordinate <= inner_max)
    )

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

        w[left] = (
            0.5
            * (
                1.0
                - np.cos(np.pi * u)
            )
        )

    right = (
        (coordinate > inner_max)
        & (coordinate <= inner_max + taper)
    )

    if np.any(right):
        u = (
            coordinate[right]
            - inner_max
        ) / taper

        w[right] = (
            0.5
            * (
                1.0
                + np.cos(np.pi * u)
            )
        )

    return w


def _sample_model(
    grid,
    field: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:

    interp = RegularGridInterpolator(
        (
            np.asarray(
                grid.x,
                dtype=np.float64,
            ),
            np.asarray(
                grid.z,
                dtype=np.float64,
            ),
        ),
        np.asarray(
            field,
            dtype=np.float64,
        ),
        method="linear",
        bounds_error=True,
    )

    points = np.column_stack(
        [
            np.asarray(
                x,
                dtype=np.float64,
            ),
            np.asarray(
                z,
                dtype=np.float64,
            ),
        ]
    )

    return interp(points)


def _build_smooth_boness_correction(
    *,
    grid,
    zhang_model,
    geometry_csv: str | Path,
    boness_csv: str | Path,

    borehole_sigma_m: float,
    vertical_taper_m: float,

    ratio_smooth_sigma_m: float,

    dense_step_m: float,

    spatial_smooth_sigma_x_m: float,
    spatial_smooth_sigma_z_m: float,

    anchor_sigma_m: float,

    correction_strength: float,
    support_cutoff_sigma: float,
) -> dict:
    """
    Smooth low-wavenumber Boness/Zoback borehole correction.

    Key difference from the previous implementation:

    1. Map published MD log to the registered 2-D borehole.
    2. Convert measured Vp/Vs to ratios relative to Zhang.
    3. Smooth ratios along measured depth.
    4. Resample the borehole densely.
    5. Assign each grid cell the ratio at its nearest position
       along the continuous borehole trajectory.
    6. Apply Gaussian radial support around the well.
    7. Spatially smooth ONLY the anomaly field.
    8. Close to the actual borehole, retain the unsmeared
       log-derived correction so the data constraint is preserved.

    Zhang itself is never spatially smoothed.
    """

    if borehole_sigma_m <= 0.0:
        raise ValueError(
            "borehole_sigma_m must be positive."
        )

    if vertical_taper_m <= 0.0:
        raise ValueError(
            "vertical_taper_m must be positive."
        )

    if ratio_smooth_sigma_m < 0.0:
        raise ValueError(
            "ratio_smooth_sigma_m must be >= 0."
        )

    if dense_step_m <= 0.0:
        raise ValueError(
            "dense_step_m must be positive."
        )

    if spatial_smooth_sigma_x_m < 0.0:
        raise ValueError(
            "spatial_smooth_sigma_x_m must be >= 0."
        )

    if spatial_smooth_sigma_z_m < 0.0:
        raise ValueError(
            "spatial_smooth_sigma_z_m must be >= 0."
        )

    if anchor_sigma_m <= 0.0:
        raise ValueError(
            "anchor_sigma_m must be positive."
        )

    if not (
        0.0
        <= correction_strength
        <= 1.5
    ):
        raise ValueError(
            "correction_strength should be between 0 and 1.5."
        )

    # --------------------------------------------------------------
    # Published Phase-1 / Boness velocity log.
    # --------------------------------------------------------------

    log_md, log_vp, log_vs = (
        _load_boness_log(
            boness_csv
        )
    )

    geom_md, geom_x, geom_z = (
        _load_downleg_geometry(
            geometry_csv
        )
    )

    if (
        log_md.min() < geom_md.min()
        or log_md.max() > geom_md.max()
    ):
        raise ValueError(
            "Boness-Zoback MD interval lies outside "
            "available down-leg geometry: "
            f"log={log_md.min():.1f}.."
            f"{log_md.max():.1f} m, "
            f"geometry={geom_md.min():.1f}.."
            f"{geom_md.max():.1f} m."
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

    # --------------------------------------------------------------
    # Zhang values at exactly the same borehole locations.
    # --------------------------------------------------------------

    zhang_vp_log = _sample_model(
        grid,
        zhang_model.vp,
        log_x,
        log_z,
    )

    zhang_vs_log = _sample_model(
        grid,
        zhang_model.vs,
        log_x,
        log_z,
    )

    vp_ratio_raw = (
        log_vp
        / zhang_vp_log
    )

    vs_ratio_raw = (
        log_vs
        / zhang_vs_log
    )

    # --------------------------------------------------------------
    # Keep low-wavenumber information from the well logs.
    #
    # The original CSV already prefers the 20-m-smoothed
    # digitization. We additionally smooth the MODEL CORRECTION
    # along MD, rather than imposing metre-scale raster structure.
    # --------------------------------------------------------------

    vp_ratio_smooth = (
        _smooth_ratio_along_md(
            log_md,
            vp_ratio_raw,
            ratio_smooth_sigma_m,
        )
    )

    vs_ratio_smooth = (
        _smooth_ratio_along_md(
            log_md,
            vs_ratio_raw,
            ratio_smooth_sigma_m,
        )
    )

    # --------------------------------------------------------------
    # Densify the trajectory.
    #
    # This removes the k=4 / nearest-sparse-sample geometry that
    # produced ray / wedge-looking regions in the previous model.
    # --------------------------------------------------------------

    md_span = (
        float(log_md[-1])
        - float(log_md[0])
    )

    n_dense = max(
        2,
        int(
            np.ceil(
                md_span
                / float(dense_step_m)
            )
        )
        + 1,
    )

    dense_md = np.linspace(
        float(log_md[0]),
        float(log_md[-1]),
        n_dense,
    )

    dense_x = np.interp(
        dense_md,
        log_md,
        log_x,
    )

    dense_z = np.interp(
        dense_md,
        log_md,
        log_z,
    )

    dense_vp_ratio = np.interp(
        dense_md,
        log_md,
        vp_ratio_smooth,
    )

    dense_vs_ratio = np.interp(
        dense_md,
        log_md,
        vs_ratio_smooth,
    )

    dense_points = np.column_stack(
        [
            dense_x,
            dense_z,
        ]
    )

    # --------------------------------------------------------------
    # Model grid.
    # --------------------------------------------------------------

    X, Z = np.meshgrid(
        np.asarray(
            grid.x,
            dtype=np.float64,
        ),
        np.asarray(
            grid.z,
            dtype=np.float64,
        ),
        indexing="ij",
    )

    grid_points = np.column_stack(
        [
            X.ravel(),
            Z.ravel(),
        ]
    )

    tree = cKDTree(
        dense_points
    )

    distance, nearest = tree.query(
        grid_points,
        k=1,
    )

    vp_ratio_near = (
        dense_vp_ratio[
            nearest
        ]
    )

    vs_ratio_near = (
        dense_vs_ratio[
            nearest
        ]
    )

    # --------------------------------------------------------------
    # Radial support around the borehole.
    # --------------------------------------------------------------

    radial_weight = np.exp(
        -0.5
        * (
            distance
            / float(
                borehole_sigma_m
            )
        ) ** 2
    )

    radial_weight[
        distance
        > (
            float(
                support_cutoff_sigma
            )
            * float(
                borehole_sigma_m
            )
        )
    ] = 0.0

    # Do not let the shallow/deep end of a 1-D log create
    # abrupt horizontal boundaries.
    vertical_weight = _cosine_window(
        Z.ravel(),
        inner_min=float(
            np.min(log_z)
        ),
        inner_max=float(
            np.max(log_z)
        ),
        taper=float(
            vertical_taper_m
        ),
    )

    support = (
        radial_weight
        * vertical_weight
        * float(
            correction_strength
        )
    )

    # --------------------------------------------------------------
    # Raw low-wavenumber correction field.
    # --------------------------------------------------------------

    vp_delta_raw = (
        support
        * (
            vp_ratio_near
            - 1.0
        )
    ).reshape(
        X.shape
    )

    vs_delta_raw = (
        support
        * (
            vs_ratio_near
            - 1.0
        )
    ).reshape(
        X.shape
    )

    # --------------------------------------------------------------
    # Smooth ONLY the log-derived anomaly, not Zhang.
    # --------------------------------------------------------------

    sigma_x_cells = (
        float(
            spatial_smooth_sigma_x_m
        )
        / float(
            grid.dx
        )
    )

    sigma_z_cells = (
        float(
            spatial_smooth_sigma_z_m
        )
        / float(
            grid.dz
        )
    )

    if (
        sigma_x_cells > 0.0
        or sigma_z_cells > 0.0
    ):
        vp_delta_smooth = gaussian_filter(
            vp_delta_raw,
            sigma=(
                sigma_x_cells,
                sigma_z_cells,
            ),
            mode="nearest",
        )

        vs_delta_smooth = gaussian_filter(
            vs_delta_raw,
            sigma=(
                sigma_x_cells,
                sigma_z_cells,
            ),
            mode="nearest",
        )
    else:
        vp_delta_smooth = (
            vp_delta_raw.copy()
        )

        vs_delta_smooth = (
            vs_delta_raw.copy()
        )

    # --------------------------------------------------------------
    # Preserve the actual borehole constraint.
    #
    # Near the logged trajectory:
    #     use the original log-derived correction.
    #
    # Away from the well:
    #     use the smoother spatial continuation.
    #
    # This is the key step allowing BOTH:
    #     - fidelity to borehole velocities
    #     - a natural-looking 2-D model.
    # --------------------------------------------------------------

    anchor = np.exp(
        -0.5
        * (
            distance
            / float(
                anchor_sigma_m
            )
        ) ** 2
    ).reshape(
        X.shape
    )

    anchor *= (
        vertical_weight.reshape(
            X.shape
        )
    )

    vp_delta = (
        anchor
        * vp_delta_raw
        + (
            1.0
            - anchor
        )
        * vp_delta_smooth
    )

    vs_delta = (
        anchor
        * vs_delta_raw
        + (
            1.0
            - anchor
        )
        * vs_delta_smooth
    )

    vp_ratio_field = (
        1.0
        + vp_delta
    )

    vs_ratio_field = (
        1.0
        + vs_delta
    )

    # Smoothed target velocities along the actual well.
    target_vp_log = (
        zhang_vp_log
        * vp_ratio_smooth
    )

    target_vs_log = (
        zhang_vs_log
        * vs_ratio_smooth
    )

    return {
        "vp_ratio_field": (
            vp_ratio_field
        ),
        "vs_ratio_field": (
            vs_ratio_field
        ),
        "support": (
            support.reshape(
                X.shape
            )
        ),
        "anchor": anchor,
        "distance_m": (
            distance.reshape(
                X.shape
            )
        ),

        "log_md_m": log_md,
        "log_x_m": log_x,
        "log_z_m": log_z,

        "log_vp_mps": log_vp,
        "log_vs_mps": log_vs,

        "target_vp_log_mps": (
            target_vp_log
        ),
        "target_vs_log_mps": (
            target_vs_log
        ),

        "vp_ratio_raw": (
            vp_ratio_raw
        ),
        "vs_ratio_raw": (
            vs_ratio_raw
        ),

        "vp_ratio_smooth": (
            vp_ratio_smooth
        ),
        "vs_ratio_smooth": (
            vs_ratio_smooth
        ),
    }


def _print_log_qc(
    *,
    grid,
    vp: np.ndarray,
    vs: np.ndarray,
    correction: dict,
) -> None:

    x = correction[
        "log_x_m"
    ]

    z = correction[
        "log_z_m"
    ]

    vp_model = _sample_model(
        grid,
        vp,
        x,
        z,
    )

    vs_model = _sample_model(
        grid,
        vs,
        x,
        z,
    )

    vp_target = correction[
        "target_vp_log_mps"
    ]

    vs_target = correction[
        "target_vs_log_mps"
    ]

    vp_raw = correction[
        "log_vp_mps"
    ]

    vs_raw = correction[
        "log_vs_mps"
    ]

    vp_target_error = (
        100.0
        * np.abs(
            vp_model
            - vp_target
        )
        / vp_target
    )

    vs_target_error = (
        100.0
        * np.abs(
            vs_model
            - vs_target
        )
        / vs_target
    )

    # How much low-pass modelling changes the actual digitized log.
    vp_smoothing_change = (
        100.0
        * np.abs(
            vp_target
            - vp_raw
        )
        / vp_raw
    )

    vs_smoothing_change = (
        100.0
        * np.abs(
            vs_target
            - vs_raw
        )
        / vs_raw
    )

    print()
    print(
        "Boness log fidelity"
    )
    print(
        "-------------------"
    )

    print(
        "final vs smoothed-log Vp : "
        f"median="
        f"{np.median(vp_target_error):.2f}%, "
        f"p95="
        f"{np.percentile(vp_target_error, 95):.2f}%"
    )

    print(
        "final vs smoothed-log Vs : "
        f"median="
        f"{np.median(vs_target_error):.2f}%, "
        f"p95="
        f"{np.percentile(vs_target_error, 95):.2f}%"
    )

    print(
        "smoothing change raw Vp   : "
        f"median="
        f"{np.median(vp_smoothing_change):.2f}%, "
        f"p95="
        f"{np.percentile(vp_smoothing_change, 95):.2f}%"
    )

    print(
        "smoothing change raw Vs   : "
        f"median="
        f"{np.median(vs_smoothing_change):.2f}%, "
        f"p95="
        f"{np.percentile(vs_smoothing_change, 95):.2f}%"
    )


def build_hybrid_zhang2009_boness2006_bill_logs_smooth_model(
    *,
    geom_file: str | Path,
    bill_logs_csv: str | Path,
    boness_log_csv: str | Path = (
        DEFAULT_BONESS_ZOBACK2006_CSV
    ),
    section_npz: str | Path = (
        DEFAULT_ZHANG_SECTION
    ),

    # --------------------------------------------------------------
    # Phase 1 / Boness.
    #
    # Compared with the existing model:
    #   lateral sigma: 500 -> 350 m
    #   MD smoothing : 30  -> 75 m
    #
    # and new gentle anomaly-only 2-D smoothing.
    # --------------------------------------------------------------

    boness_borehole_sigma_m: float = 350.0,
    boness_vertical_taper_m: float = 250.0,

    boness_ratio_smooth_sigma_m: float = 75.0,

    boness_dense_step_m: float = 5.0,

    boness_spatial_smooth_sigma_x_m: float = 50.0,
    boness_spatial_smooth_sigma_z_m: float = 70.0,

    boness_anchor_sigma_m: float = 60.0,

    boness_correction_strength: float = 1.0,
    boness_support_cutoff_sigma: float = 3.0,

    # --------------------------------------------------------------
    # Phase 2 / Bill-Ellsworth-Malin.
    #
    # Existing model is already localized and sensible.
    # Only make its spatial continuation slightly softer.
    # --------------------------------------------------------------

    bill_anomaly_taper_m: float = 300.0,
    bill_depth_gaussian_pad_m: float = 325.0,

    bill_smooth_sigma_x_m: float = 55.0,
    bill_smooth_sigma_z_m: float = 110.0,

    build_initial_model: bool = True,
    verbose: bool = True,

    **base_builder_kwargs,
):
    """
    Preferred smooth SAFOD initial model.

    Data hierarchy:

        1. Zhang et al. (2009)
           regional Vp/Vs tomography

        2. Boness & Zoback (2006)
           Phase-1 Main-Hole Vp/Vs calibration

        3. Ellsworth-Malin / Bill
           Phase-2 local fault-zone Vp/Vs anomaly

    The regional Zhang structure is never spatially smoothed.

    Only log-derived corrections are smoothed.

    Boness correction is anchored close to the measured
    borehole so the low-wavenumber log trend is preserved,
    while its off-borehole continuation is made smooth.

    The Bill fault-zone anomaly uses the already validated
    multiplicative formulation with slightly gentler smoothing.
    """

    if not build_initial_model:
        raise ValueError(
            "This is an initial-model builder."
        )

    kwargs = dict(
        base_builder_kwargs
    )

    requested_dt = kwargs.pop(
        "dt",
        None,
    )

    half_order = int(
        kwargs.get(
            "half_order",
            2,
        )
    )

    cfl_safety = float(
        kwargs.get(
            "cfl_safety",
            0.80,
        )
    )

    # ==============================================================
    # 1. Pure regional Zhang Vp/Vs.
    # ==============================================================

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

    # ==============================================================
    # 2. Improved Phase-1 / Boness correction.
    # ==============================================================

    boness = _build_smooth_boness_correction(
        grid=zhang_grid,
        zhang_model=zhang_model,
        geometry_csv=geom_file,
        boness_csv=boness_log_csv,

        borehole_sigma_m=(
            boness_borehole_sigma_m
        ),

        vertical_taper_m=(
            boness_vertical_taper_m
        ),

        ratio_smooth_sigma_m=(
            boness_ratio_smooth_sigma_m
        ),

        dense_step_m=(
            boness_dense_step_m
        ),

        spatial_smooth_sigma_x_m=(
            boness_spatial_smooth_sigma_x_m
        ),

        spatial_smooth_sigma_z_m=(
            boness_spatial_smooth_sigma_z_m
        ),

        anchor_sigma_m=(
            boness_anchor_sigma_m
        ),

        correction_strength=(
            boness_correction_strength
        ),

        support_cutoff_sigma=(
            boness_support_cutoff_sigma
        ),
    )

    vp_boness = (
        np.asarray(
            zhang_model.vp,
            dtype=np.float64,
        )
        * boness[
            "vp_ratio_field"
        ]
    )

    vs_boness = (
        np.asarray(
            zhang_model.vs,
            dtype=np.float64,
        )
        * boness[
            "vs_ratio_field"
        ]
    )

    # ==============================================================
    # 3. Phase-2 / Bill anomaly.
    #
    # Build Zhang+Bill using the existing validated formulation,
    # but with slightly smoother parameters.
    # ==============================================================

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

        anomaly_taper_m=(
            bill_anomaly_taper_m
        ),

        depth_gaussian_pad_m=(
            bill_depth_gaussian_pad_m
        ),

        smooth_sigma_x_m=(
            bill_smooth_sigma_x_m
        ),

        smooth_sigma_z_m=(
            bill_smooth_sigma_z_m
        ),

        build_initial_model=True,
        verbose=False,
        dt=None,
        **kwargs,
    )

    if (
        int(zhang_grid.nx)
        != int(bill_grid.nx)
        or int(zhang_grid.nz)
        != int(bill_grid.nz)
        or not np.array_equal(
            np.asarray(
                zhang_grid.x
            ),
            np.asarray(
                bill_grid.x
            ),
        )
        or not np.array_equal(
            np.asarray(
                zhang_grid.z
            ),
            np.asarray(
                bill_grid.z
            ),
        )
    ):
        raise RuntimeError(
            "Spatial-grid mismatch between "
            "Zhang and Zhang+Bill."
        )

    bill_vp_ratio = (
        np.asarray(
            zhang_bill_model.vp,
            dtype=np.float64,
        )
        / np.asarray(
            zhang_model.vp,
            dtype=np.float64,
        )
    )

    bill_vs_ratio = (
        np.asarray(
            zhang_bill_model.vs,
            dtype=np.float64,
        )
        / np.asarray(
            zhang_model.vs,
            dtype=np.float64,
        )
    )

    # ==============================================================
    # 4. Final combination.
    #
    # Zhang
    #   × smooth Phase-1 Boness correction
    #   × smooth localized Phase-2 Bill correction
    # ==============================================================

    vp = (
        vp_boness
        * bill_vp_ratio
    )

    vs = (
        vs_boness
        * bill_vs_ratio
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

    # ==============================================================
    # 5. Solver-safe timestep.
    # ==============================================================

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

    # ==============================================================
    # Metadata.
    # ==============================================================

    notes = (
        "Preferred smooth SAFOD initial model. "
        "Zhang et al. (2009) direct regional Vp/Vs tomography "
        "is retained without spatial smoothing. "
        "Boness & Zoback (2006) Phase-1 Main-Hole Vp/Vs "
        "are converted to multiplicative corrections relative "
        "to Zhang at the registered MD->X_2D/TVD trajectory. "
        "The Boness correction is smoothed along measured depth, "
        "densely resampled along the borehole, and its off-borehole "
        "anomaly is spatially smoothed while remaining anchored "
        "to the log-derived trend near the well. "
        "Ellsworth-Malin/Bill Phase-2 fault-zone velocities are "
        "applied using the existing localized multiplicative "
        "anomaly formulation with slightly gentler spatial smoothing. "
        + bill_metadata.notes
    )

    metadata = replace(
        zhang_metadata,
        model_type=(
            "initial_hybrid_zhang2009_"
            "boness2006_bill_logs_smooth"
        ),
        dt_s=float(
            grid.dt
        ),
        notes=notes,
    )

    # ==============================================================
    # QC.
    # ==============================================================

    if verbose:

        ratio = (
            vp / vs
        )

        bill_active = (
            np.abs(
                bill_vp_ratio
                - 1.0
            )
            > 1.0e-3
        ) | (
            np.abs(
                bill_vs_ratio
                - 1.0
            )
            > 1.0e-3
        )

        boness_active = (
            np.abs(
                boness[
                    "vp_ratio_field"
                ]
                - 1.0
            )
            > 1.0e-3
        ) | (
            np.abs(
                boness[
                    "vs_ratio_field"
                ]
                - 1.0
            )
            > 1.0e-3
        )

        print()
        print(
            "SAFOD preferred smooth initial model"
        )
        print(
            "===================================="
        )

        print(
            f"Boness MD smoothing sigma : "
            f"{boness_ratio_smooth_sigma_m:.1f} m"
        )

        print(
            f"Boness radial sigma       : "
            f"{boness_borehole_sigma_m:.1f} m"
        )

        print(
            f"Boness dense MD step      : "
            f"{boness_dense_step_m:.1f} m"
        )

        print(
            f"Boness anomaly smooth x,z : "
            f"{boness_spatial_smooth_sigma_x_m:.1f}, "
            f"{boness_spatial_smooth_sigma_z_m:.1f} m"
        )

        print(
            f"Boness anchor sigma       : "
            f"{boness_anchor_sigma_m:.1f} m"
        )

        print(
            f"Bill anomaly taper        : "
            f"{bill_anomaly_taper_m:.1f} m"
        )

        print(
            f"Bill anomaly smooth x,z   : "
            f"{bill_smooth_sigma_x_m:.1f}, "
            f"{bill_smooth_sigma_z_m:.1f} m"
        )

        print(
            f"Boness active fraction    : "
            f"{100.0*np.mean(boness_active):.2f}%"
        )

        print(
            f"Bill active fraction      : "
            f"{100.0*np.mean(bill_active):.2f}%"
        )

        print(
            f"Vp range                  : "
            f"{vp.min()/1000.0:.3f} .. "
            f"{vp.max()/1000.0:.3f} km/s"
        )

        print(
            f"Vs range                  : "
            f"{vs.min()/1000.0:.3f} .. "
            f"{vs.max()/1000.0:.3f} km/s"
        )

        print(
            f"Vp/Vs range               : "
            f"{ratio.min():.3f} .. "
            f"{ratio.max():.3f}"
        )

        print(
            f"dt                        : "
            f"{grid.dt:.6e} s"
        )

        _print_log_qc(
            grid=grid,
            vp=vp,
            vs=vs,
            correction=boness,
        )

    return (
        grid,
        model,
        x_cable,
        z_cable,
        metadata,
    )
