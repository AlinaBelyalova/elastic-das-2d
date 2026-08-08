# src/safod_digitized_log_model.py
#
# SAFOD initial elastic model constrained directly by the digitized
# Ellsworth & Malin (2011) Fig. 3a open-hole Vp/Vs logs.
#
# This replaces the previous hand-tuned Gaussian / piecewise velocity factors
# by the pointwise published Vp and Vs profile.
#
# Scientific idea
# ---------------
# 1. Build the existing smooth SAFOD background model, but DISABLE its old
#    lateral cross-fault contrast and Gaussian SAF damage zone.
#
# 2. Read the digitized Phase-II open-hole Vp/Vs log:
#
#       measured_depth_m
#       tvd_m
#       section_x_from_cable_end_m
#       section_offset_from_sdz_m
#       vp_mps
#       vs_mps
#
# 3. Convert the absolute digitized velocities to multiplicative Vp and Vs
#    anomalies relative to the smooth background at the actual log TVDs.
#
# 4. Apply those anomalies as a function of signed section distance from the
#    SDZ line.  The existing model fault-tie line is interpreted as the SDZ,
#    because its ~105 m cable-to-fault offset agrees with the digitized
#    cable-end -> SDZ distance (~114 m).
#
# 5. Preserve the sharp metre-scale GBF/SDZ/CDZ/NBF velocity minima by widening
#    each unresolved fault core to at least one FD x-cell.  NO lateral Gaussian
#    smoothing is applied after the log profile is inserted.
#
# 6. Density remains the background density because the published figure
#    constrains Vp and Vs, not rho.
#
# The model therefore exactly reproduces the digitized Vp/Vs cross-fault
# profile at the log depth range to raster-digitization accuracy, while using
# the original depth trend away from that depth range.
#
# References
# ----------
# Ellsworth, W. L., & Malin, P. E. (2011),
# "Deep rock damage in the San Andreas Fault revealed by P- and S-type
# fault-zone-guided waves", Geological Society, London, Special Publications,
# 359, 39-53. doi:10.1144/SP359.3.
#
# Figure 3a is explicitly described as SAFOD open-hole seismic P- and S-wave
# velocity logs.  The published figure shows GBF, SDZ, CDZ, NBF, the broad
# damage zone, and the ILVZ.
#
# Usage
# -----
# from src.safod_digitized_log_model import build_safod_digitized_log_model
#
# grid, model, x_cable, z_cable, metadata = build_safod_digitized_log_model(
#     geom_file=GEOM_FILE,
#     log_csv="data/safod/ellsworth_malin_2011_fig3a_digitized.csv",
#     dx=5.0,
#     dz=5.0,
#     nt=...,
#     ...
# )
#
# Return signature matches build_safod_model().
# ==============================================================================

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.model import ElasticModel2D
from src.safod_builder import (
    build_safod_model,
    fault_x_at_z,
)


STRUCTURE_MD_M = {
    "GBF": 3150.0,
    "SDZ": 3192.0,
    "CDZ": 3302.0,
    "NBF": 3413.0,
}


def _grid_coordinates(grid):
    x = (
        float(grid.x0)
        + np.arange(
            int(grid.nx),
            dtype=np.float64,
        )
        * float(grid.dx)
    )

    z = (
        float(grid.z0)
        + np.arange(
            int(grid.nz),
            dtype=np.float64,
        )
        * float(grid.dz)
    )

    return np.meshgrid(
        x,
        z,
        indexing="ij",
    )


def _load_digitized_log(log_csv: str | Path) -> pd.DataFrame:
    path = Path(log_csv)

    if not path.exists():
        raise FileNotFoundError(
            f"Digitized Ellsworth-Malin log not found: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "measured_depth_m",
        "tvd_m",
        "section_offset_from_sdz_m",
        "vp_mps",
        "vs_mps",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            f"Digitized log is missing columns: {sorted(missing)}"
        )

    for column in required:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df.dropna(
            subset=list(required)
        )
        .sort_values(
            "section_offset_from_sdz_m"
        )
        .reset_index(drop=True)
    )

    if len(df) < 20:
        raise ValueError(
            f"Digitized log contains only {len(df)} valid points."
        )

    if np.any(
        np.diff(
            df["section_offset_from_sdz_m"].to_numpy()
        )
        <= 0.0
    ):
        raise ValueError(
            "section_offset_from_sdz_m must be strictly increasing."
        )

    if (
        (df["vp_mps"] <= 0.0).any()
        or (df["vs_mps"] <= 0.0).any()
    ):
        raise ValueError(
            "Digitized Vp/Vs must be positive."
        )

    return df


def _background_depth_profiles(base_model):
    """
    Lateral median of the background model.

    The old lateral SAF structure is disabled before this function is called,
    so this is effectively the smooth depth trend used by the original model.
    """
    vp_z = np.median(
        np.asarray(
            base_model.vp,
            dtype=np.float64,
        ),
        axis=0,
    )

    vs_z = np.median(
        np.asarray(
            base_model.vs,
            dtype=np.float64,
        ),
        axis=0,
    )

    return vp_z, vs_z


def _interp_depth_profile(
    *,
    grid,
    values_z: np.ndarray,
    target_z_m: np.ndarray,
) -> np.ndarray:
    z_grid = (
        float(grid.z0)
        + np.arange(
            int(grid.nz),
            dtype=np.float64,
        )
        * float(grid.dz)
    )

    return np.interp(
        np.asarray(
            target_z_m,
            dtype=np.float64,
        ),
        z_grid,
        np.asarray(
            values_z,
            dtype=np.float64,
        ),
    )


def _structure_row(
    df: pd.DataFrame,
    md_m: float,
) -> pd.Series:
    i = int(
        np.argmin(
            np.abs(
                df["measured_depth_m"].to_numpy()
                - float(md_m)
            )
        )
    )

    return df.iloc[i]


def _check_elastic_physicality(
    vp: np.ndarray,
    vs: np.ndarray,
    rho: np.ndarray,
) -> None:
    if (
        np.any(~np.isfinite(vp))
        or np.any(~np.isfinite(vs))
        or np.any(~np.isfinite(rho))
    ):
        raise ValueError(
            "vp/vs/rho contain non-finite values."
        )

    if (
        np.any(vp <= 0.0)
        or np.any(vs <= 0.0)
        or np.any(rho <= 0.0)
    ):
        raise ValueError(
            "vp/vs/rho must be positive."
        )

    mu = rho * vs**2

    lam = (
        rho
        * (
            vp**2
            - 2.0
            * vs**2
        )
    )

    if np.any(mu <= 0.0):
        raise ValueError(
            f"Non-positive mu; min={mu.min():.6e}"
        )

    if np.any(lam < 0.0):
        raise ValueError(
            f"Negative lambda; min={lam.min():.6e}. "
            "Check the digitized Vp/Vs ratio."
        )


def build_safod_digitized_log_model(
    *,
    geom_file: str | Path,
    log_csv: str | Path,

    build_initial_model: bool = True,

    # Preserve sub-grid fault cores as one FD cell.
    resolve_fault_cores: bool = True,

    # The digitized log begins ~114 m NE of the current cable end at the SDZ.
    # The existing model tie line is already ~105 m from the cable end, so the
    # cleanest convention is to identify the existing tie line with the SDZ.
    tie_line_is_sdz: bool = True,

    verbose: bool = True,

    **base_builder_kwargs,
):
    """
    Build the SAFOD initial model from pointwise digitized Vp/Vs logs.

    Returns
    -------
    grid, model, x_cable, z_cable, metadata
        Same return convention as src.safod_builder.build_safod_model.
    """
    if not build_initial_model:
        raise ValueError(
            "This builder defines the log-constrained INITIAL model; "
            "use build_initial_model=True."
        )

    if not tie_line_is_sdz:
        raise NotImplementedError(
            "Current implementation uses the existing tie line as the SDZ. "
            "This is the geometry supported by the present cable-to-fault "
            "offset and digitized log reconstruction."
        )

    log = _load_digitized_log(
        log_csv
    )

    kwargs = dict(
        base_builder_kwargs
    )

    # --------------------------------------------------------------
    # Remove the OLD hand-tuned lateral fault model.
    # --------------------------------------------------------------
    kwargs[
        "initial_cross_fault_contrast"
    ] = 0.0

    kwargs[
        "initial_fault_zone_velocity_reduction"
    ] = 0.0

    (
        grid,
        base_model,
        x_cable,
        z_cable,
        base_metadata,
    ) = build_safod_model(
        geom_file=geom_file,
        build_initial_model=True,
        **kwargs,
    )

    # --------------------------------------------------------------
    # Background velocity at the actual TVD of every digitized point.
    # --------------------------------------------------------------
    vp_bg_z, vs_bg_z = _background_depth_profiles(
        base_model
    )

    log_tvd = log[
        "tvd_m"
    ].to_numpy(
        dtype=np.float64
    )

    vp_bg_at_log = _interp_depth_profile(
        grid=grid,
        values_z=vp_bg_z,
        target_z_m=log_tvd,
    )

    vs_bg_at_log = _interp_depth_profile(
        grid=grid,
        values_z=vs_bg_z,
        target_z_m=log_tvd,
    )

    vp_log = log[
        "vp_mps"
    ].to_numpy(
        dtype=np.float64
    )

    vs_log = log[
        "vs_mps"
    ].to_numpy(
        dtype=np.float64
    )

    # Pointwise anomalies relative to the original depth trend.
    vp_ratio_log = (
        vp_log
        / vp_bg_at_log
    )

    vs_ratio_log = (
        vs_log
        / vs_bg_at_log
    )

    offset_log = log[
        "section_offset_from_sdz_m"
    ].to_numpy(
        dtype=np.float64
    )

    # --------------------------------------------------------------
    # Signed section coordinate relative to the SDZ line.
    # --------------------------------------------------------------
    X, Z = _grid_coordinates(
        grid
    )

    x_sdz = fault_x_at_z(
        Z,
        x_tie_m=float(
            base_metadata.x_tie_m
        ),
        z_tie_m=float(
            base_metadata.z_tie_m
        ),
        fault_dip_deg=float(
            base_metadata.fault_dip_deg
        ),
        fault_dip_sign=float(
            base_metadata.fault_dip_sign
        ),
    )

    d = (
        X
        - x_sdz
    )

    # np.interp clamps outside the digitized profile to the SW/NE edge
    # values.  This naturally gives separate host-rock velocities on the
    # two sides without inventing another arbitrary contrast.
    vp_ratio = np.interp(
        d.ravel(),
        offset_log,
        vp_ratio_log,
    ).reshape(
        d.shape
    )

    vs_ratio = np.interp(
        d.ravel(),
        offset_log,
        vs_ratio_log,
    ).reshape(
        d.shape
    )

    # --------------------------------------------------------------
    # Explicitly resolve the very narrow GBF/SDZ/CDZ/NBF minima.
    #
    # The raster figure shows these as near-vertical metre-scale drops.
    # At dx=5 m they are sub-grid, so broaden them to one x-cell while
    # retaining the digitized minimum Vp and Vs ratio.
    # --------------------------------------------------------------
    structure_info = {}

    if resolve_fault_cores:
        half_width = (
            0.5
            * float(
                grid.dx
            )
        )

        for name, md0 in STRUCTURE_MD_M.items():
            row = _structure_row(
                log,
                md0,
            )

            offset0 = float(
                row[
                    "section_offset_from_sdz_m"
                ]
            )

            tvd0 = float(
                row[
                    "tvd_m"
                ]
            )

            vp_bg0 = float(
                _interp_depth_profile(
                    grid=grid,
                    values_z=vp_bg_z,
                    target_z_m=np.array([tvd0]),
                )[0]
            )

            vs_bg0 = float(
                _interp_depth_profile(
                    grid=grid,
                    values_z=vs_bg_z,
                    target_z_m=np.array([tvd0]),
                )[0]
            )

            vp_core_ratio = float(
                row[
                    "vp_mps"
                ]
                / vp_bg0
            )

            vs_core_ratio = float(
                row[
                    "vs_mps"
                ]
                / vs_bg0
            )

            core_mask = (
                np.abs(
                    d - offset0
                )
                <= half_width
            )

            # A core should only reduce velocity relative to the surrounding
            # interpolated profile, never increase it.
            vp_ratio[
                core_mask
            ] = np.minimum(
                vp_ratio[
                    core_mask
                ],
                vp_core_ratio,
            )

            vs_ratio[
                core_mask
            ] = np.minimum(
                vs_ratio[
                    core_mask
                ],
                vs_core_ratio,
            )

            structure_info[
                name
            ] = {
                "md_m": float(
                    row[
                        "measured_depth_m"
                    ]
                ),
                "tvd_m": tvd0,
                "offset_from_sdz_m": offset0,
                "vp_mps": float(
                    row[
                        "vp_mps"
                    ]
                ),
                "vs_mps": float(
                    row[
                        "vs_mps"
                    ]
                ),
                "effective_model_width_m": float(
                    grid.dx
                ),
            }

    # --------------------------------------------------------------
    # Apply the digitized anomaly field to the background.
    # --------------------------------------------------------------
    vp = (
        np.asarray(
            base_model.vp,
            dtype=np.float64,
        )
        * vp_ratio
    )

    vs = (
        np.asarray(
            base_model.vs,
            dtype=np.float64,
        )
        * vs_ratio
    )

    # Published figure constrains velocity, not density.
    rho = np.asarray(
        base_model.rho,
        dtype=np.float64,
    ).copy()

    # CRITICAL: do not smooth after insertion.
    _check_elastic_physicality(
        vp,
        vs,
        rho,
    )

    model = ElasticModel2D(
        grid=grid,
        vp=vp,
        vs=vs,
        rho=rho,
    )

    sdz_row = _structure_row(
        log,
        STRUCTURE_MD_M[
            "SDZ"
        ],
    )

    nbf_row = _structure_row(
        log,
        STRUCTURE_MD_M[
            "NBF"
        ],
    )

    damage_width = float(
        nbf_row[
            "section_offset_from_sdz_m"
        ]
        - sdz_row[
            "section_offset_from_sdz_m"
        ]
    )

    sw_vp = float(
        np.median(
            log.loc[
                log[
                    "section_offset_from_sdz_m"
                ]
                < -50.0,
                "vp_mps",
            ]
        )
    )

    ne_vp = float(
        np.median(
            log.loc[
                log[
                    "section_offset_from_sdz_m"
                ]
                > 220.0,
                "vp_mps",
            ]
        )
    )

    representative_contrast = (
        ne_vp / sw_vp - 1.0
    )

    notes = (
        "SAFOD initial model constrained pointwise by digitization of "
        "Ellsworth & Malin (2011) Fig. 3a open-hole Vp/Vs logs. "
        "The original smooth cross-fault contrast and Gaussian SAF damage "
        "zone are disabled. Digitized Vp and Vs are converted to anomalies "
        "relative to the original smooth depth trend and mapped by section "
        "distance from the SDZ. GBF/SDZ/CDZ/NBF metre-scale minima are "
        "resolved as one FD x-cell because they are sub-grid at dx=5 m. "
        "No smoothing is applied after the log anomaly is inserted. "
        "Density remains the original background density."
    )

    metadata = replace(
        base_metadata,

        model_type=(
            "initial_digitized_ellsworth_malin_2011"
        ),

        cross_fault_contrast=float(
            representative_contrast
        ),

        fault_zone_width_m=float(
            damage_width
        ),

        fault_zone_velocity_reduction=float(
            1.0
            - np.min(
                vp_ratio_log
            )
        ),

        smoothing_sigma_m=0.0,

        notes=notes,
    )

    if verbose:
        print()
        print("SAFOD digitized-log initial model")
        print("=================================")

        print(
            f"log points                 : {len(log)}"
        )

        print(
            f"log MD range               : "
            f"{log['measured_depth_m'].min():.1f} .. "
            f"{log['measured_depth_m'].max():.1f} m"
        )

        print(
            f"log TVD range              : "
            f"{log['tvd_m'].min():.1f} .. "
            f"{log['tvd_m'].max():.1f} m"
        )

        print(
            f"digitized Vp range        : "
            f"{vp_log.min()/1000.0:.3f} .. "
            f"{vp_log.max()/1000.0:.3f} km/s"
        )

        print(
            f"digitized Vs range        : "
            f"{vs_log.min()/1000.0:.3f} .. "
            f"{vs_log.max()/1000.0:.3f} km/s"
        )

        print(
            f"SDZ -> NBF section width  : "
            f"{damage_width:.1f} m"
        )

        print(
            f"SW median Vp              : "
            f"{sw_vp/1000.0:.3f} km/s"
        )

        print(
            f"NE median Vp              : "
            f"{ne_vp/1000.0:.3f} km/s"
        )

        print(
            f"representative NE/SW dVp  : "
            f"{100.0*representative_contrast:+.1f}%"
        )

        print()
        print("Digitized fault positions relative to SDZ:")

        for name in (
            "GBF",
            "SDZ",
            "CDZ",
            "NBF",
        ):
            if name in structure_info:
                info = structure_info[name]

                print(
                    f"  {name}: "
                    f"offset={info['offset_from_sdz_m']:+.1f} m, "
                    f"TVD={info['tvd_m']:.1f} m, "
                    f"Vp={info['vp_mps']/1000.0:.3f} km/s, "
                    f"Vs={info['vs_mps']/1000.0:.3f} km/s, "
                    f"model width={info['effective_model_width_m']:.1f} m"
                )

        print()
        print(
            "post-log smoothing          : NONE"
        )

    return (
        grid,
        model,
        x_cable,
        z_cable,
        metadata,
    )


def save_digitized_model_profile_qc(
    *,
    grid,
    model,
    metadata,
    log_csv: str | Path,
    output_png: str | Path,
    depth_m: float | None = None,
) -> None:
    """
    QC plot comparing the model cross-fault profile with the digitized log.

    PNG only.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log = _load_digitized_log(
        log_csv
    )

    if depth_m is None:
        # Use the middle of the actual log interval.
        depth_m = float(
            np.median(
                log[
                    "tvd_m"
                ]
            )
        )

    x_grid = (
        float(grid.x0)
        + np.arange(
            int(grid.nx),
            dtype=np.float64,
        )
        * float(grid.dx)
    )

    z_grid = (
        float(grid.z0)
        + np.arange(
            int(grid.nz),
            dtype=np.float64,
        )
        * float(grid.dz)
    )

    iz = int(
        np.argmin(
            np.abs(
                z_grid
                - float(depth_m)
            )
        )
    )

    x_sdz_here = float(
        fault_x_at_z(
            z_grid[iz],
            x_tie_m=float(
                metadata.x_tie_m
            ),
            z_tie_m=float(
                metadata.z_tie_m
            ),
            fault_dip_deg=float(
                metadata.fault_dip_deg
            ),
            fault_dip_sign=float(
                metadata.fault_dip_sign
            ),
        )
    )

    d_grid = (
        x_grid
        - x_sdz_here
    )

    fig, ax = plt.subplots(
        figsize=(10.5, 4.8)
    )

    ax.plot(
        d_grid,
        model.vp[:, iz] / 1000.0,
        linewidth=1.8,
        label="Model Vp",
    )

    ax.plot(
        d_grid,
        model.vs[:, iz] / 1000.0,
        linewidth=1.8,
        label="Model Vs",
    )

    ax.plot(
        log[
            "section_offset_from_sdz_m"
        ],
        log[
            "vp_mps"
        ] / 1000.0,
        linewidth=0.9,
        alpha=0.75,
        label="Digitized Vp log",
    )

    ax.plot(
        log[
            "section_offset_from_sdz_m"
        ],
        log[
            "vs_mps"
        ] / 1000.0,
        linewidth=0.9,
        alpha=0.75,
        label="Digitized Vs log",
    )

    for name, md0 in STRUCTURE_MD_M.items():
        row = _structure_row(
            log,
            md0,
        )

        offset = float(
            row[
                "section_offset_from_sdz_m"
            ]
        )

        ax.axvline(
            offset,
            color="0.45",
            linestyle=":",
            linewidth=0.8,
        )

        ax.text(
            offset,
            0.98,
            name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlim(
        float(
            log[
                "section_offset_from_sdz_m"
            ].min()
        ),
        float(
            log[
                "section_offset_from_sdz_m"
            ].max()
        ),
    )

    ax.set_xlabel(
        "Section offset from SDZ [m]",
        fontsize=13,
        fontweight="bold",
    )

    ax.set_ylabel(
        "Velocity [km/s]",
        fontsize=13,
        fontweight="bold",
    )

    ax.legend(
        frameon=False,
        fontsize=9,
        ncol=2,
    )

    ax.tick_params(
        axis="both",
        labelsize=11,
    )

    for tick in (
        ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        tick.set_fontweight(
            "bold"
        )

    fig.tight_layout()

    output_png = Path(
        output_png
    )

    output_png.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_png,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )