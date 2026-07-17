# ==============================================================================
# scripts/validation/boundary_absorption.py
#
# Objective side-boundary absorption QC for the Gaussian sponge.
#
# Scientific design
# -----------------
# 1. Homogeneous elastic medium.
# 2. Isotropic source: predominantly P-wave radiation, so double-couple
#    radiation-pattern zeros do not bias the boundary test.
# 3. The undamped interior is fixed for every configuration. Increasing
#    n_boundary enlarges the grid outward; it never moves the sponge entrance
#    toward the source or receivers.
# 4. The interior is strongly rectangular (tall in z) so the selected record
#    contains left/right reflections but excludes top/bottom contamination.
# 5. Every sponge width has a matched gamma_s=0 control on the identical grid.
# 6. A fixed receiver loop records both Cartesian velocity components.
# 7. Two complementary metrics are reported:
#
#       actual contamination = late energy / direct energy
#       matched suppression  = late energy / late energy(gamma_s=0, same grid)
#
#    The first includes the practical benefit of a wider domain. The second
#    isolates attenuation by the sponge from extra propagation distance.
#
# Lower ratios are better. Energy ratios are also reported in dB using
# 10*log10(ratio).
# ==============================================================================

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.grid import Grid2D
from src.model import ElasticModel2D
from src.receivers import build_das_cable
from src.simulator import run_forward_simulation
from src.solver_numpy import max_stable_dt
from src.source import build_isotropic_source


OUT_DIR = Path("results/boundary_absorption_qc")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------------
# Sponge configurations
# ------------------------------------------------------------------------------
# Each width MUST include gamma_s=0 as a matched undamped control. Do not use
# n_boundary=0: both production solvers require n_boundary > half_order.
#
# Start with this moderate sweep. If the optimum lies on the edge of the tested
# range, extend the corresponding width or gamma range in a second pass.
SPONGE_FAMILIES: dict[int, tuple[float, ...]] = {
    100: (0.0, 80.0, 100.0, 120.0),
    120: (0.0, 40.0, 60.0, 80.0, 100.0),
    140: (0.0, 40.0, 60.0, 80.0),
}


# ------------------------------------------------------------------------------
# Numerical and physical parameters
# ------------------------------------------------------------------------------
DX_M = 5.0
DZ_M = 5.0
HALF_ORDER = 2
CFL_SAFETY = 0.80

VP_M_S = 5000.0
VS_M_S = 2900.0
RHO_KG_M3 = 2600.0

SOURCE_F0_HZ = 12.0
SOURCE_SCALAR_MOMENT = 1.0e12
SOURCE_OFFSET_X_CELLS = 0.37
SOURCE_OFFSET_Z_CELLS = 0.61

# Fixed UNDAMPED interior dimensions, including both endpoints.
#
# X: 201 cells -> 1000 m; source is approximately 500 m from each side sponge.
# Z: 641 cells -> 3200 m; top/bottom sponge entries are far enough away that
#    their earliest possible backscatter occurs after the selected record.
INTERIOR_NX = 201
INTERIOR_NZ = 661

# Fixed square receiver contour around the source. The loop is small relative
# to the source-to-side-boundary distance and remains inside the undamped area.
RECEIVER_LOOP_HALF_SIZE_M = 150.0
RECEIVER_SPACING_M = 10.0
GAUGE_LENGTH_M = 20.0

# Continue recording briefly after the latest expected side P reflection from
# the outer edge of the widest tested domain.
TAIL_AFTER_LATEST_SIDE_REFLECTION_S = 0.12

# Required quiet-time separation between record end and the earliest possible
# top/bottom backscatter from the sponge entrance.
MIN_VERTICAL_GUARD_S = 0.05

# Cost-aware recommendation: among configurations whose actual contamination is
# within this factor of the best tested result, prefer the cheapest grid.
RECOMMEND_WITHIN_FACTOR_OF_BEST = 2.0

# Direct arrivals should be almost unchanged relative to the matched control.
# Larger differences suggest that receivers/source are too close to damping or
# that the configurations are no longer physically comparable.
DIRECT_CONTROL_RELATIVE_TOLERANCE = 0.02


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def build_configs() -> list[dict[str, float | int | str]]:
    """Expand SPONGE_FAMILIES into validated run configurations."""
    configs: list[dict[str, float | int | str]] = []

    for n_boundary, gammas in sorted(SPONGE_FAMILIES.items()):
        if n_boundary <= HALF_ORDER:
            raise ValueError(
                f"n_boundary={n_boundary} must exceed HALF_ORDER={HALF_ORDER}."
            )

        gamma_values = tuple(float(gamma) for gamma in gammas)
        if gamma_values.count(0.0) != 1:
            raise ValueError(
                f"n_boundary={n_boundary} must contain exactly one gamma_s=0 "
                "matched control."
            )
        if any(gamma < 0.0 for gamma in gamma_values):
            raise ValueError("gamma_s values must be non-negative.")
        if len(set(gamma_values)) != len(gamma_values):
            raise ValueError(
                f"Duplicate gamma_s values for n_boundary={n_boundary}."
            )

        for gamma_s in gamma_values:
            gamma_tag = f"{gamma_s:g}".replace(".", "p")
            suffix = "_control" if gamma_s == 0.0 else ""
            configs.append(
                {
                    "n_boundary": int(n_boundary),
                    "gamma_s": gamma_s,
                    "label": f"n{n_boundary}_g{gamma_tag}{suffix}",
                }
            )

    return configs


def build_homogeneous_model(
    grid: Grid2D,
    *,
    vp_m_s: float,
    vs_m_s: float,
    rho_kg_m3: float,
) -> ElasticModel2D:
    """Build a homogeneous isotropic elastic model."""
    shape = grid.shape

    return ElasticModel2D(
        grid=grid,
        vp=np.full(shape, vp_m_s, dtype=np.float64),
        vs=np.full(shape, vs_m_s, dtype=np.float64),
        rho=np.full(shape, rho_kg_m3, dtype=np.float64),
    )


def build_central_receiver_loop(
    grid: Grid2D,
    *,
    x_center_m: float,
    z_center_m: float,
    half_size_m: float,
    spacing_m: float,
):
    """Build a fixed square receiver loop centred on the source."""
    h = float(half_size_m)

    waypoints_x = [
        x_center_m - h,
        x_center_m + h,
        x_center_m + h,
        x_center_m - h,
        x_center_m - h,
    ]
    waypoints_z = [
        z_center_m - h,
        z_center_m - h,
        z_center_m + h,
        z_center_m + h,
        z_center_m - h,
    ]

    return build_das_cable(
        grid=grid,
        waypoints_x=waypoints_x,
        waypoints_z=waypoints_z,
        channel_spacing_m=spacing_m,
        n_pml=0,
    )


def integrate_energy(
    energy: np.ndarray,
    time_s: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Integrate a non-negative energy proxy over a selected time window."""
    if np.count_nonzero(mask) < 2:
        raise ValueError("Energy-integration window contains fewer than 2 samples.")

    return float(np.trapezoid(energy[mask], time_s[mask]))


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return a finite positive ratio with protection against zero division."""
    return float(numerator / max(denominator, 1.0e-300))


def energy_ratio_db(ratio: float) -> float:
    """Convert a positive energy ratio to decibels."""
    return float(10.0 * np.log10(max(ratio, 1.0e-300)))


def mark_pareto_efficient(
    costs: np.ndarray,
    contamination: np.ndarray,
) -> np.ndarray:
    """Mark points not dominated in both grid cost and contamination."""
    n = costs.size
    efficient = np.ones(n, dtype=bool)

    for i in range(n):
        dominated = (
            (costs <= costs[i])
            & (contamination <= contamination[i])
            & ((costs < costs[i]) | (contamination < contamination[i]))
        )
        if np.any(dominated):
            efficient[i] = False

    return efficient


def validate_geometry_and_timing(
    configs: list[dict[str, float | int | str]],
) -> dict[str, float | int]:
    """Validate the fixed geometry and construct physically defined windows."""
    for name, value in (
        ("INTERIOR_NX", INTERIOR_NX),
        ("INTERIOR_NZ", INTERIOR_NZ),
    ):
        if value < 5 or value % 2 == 0:
            raise ValueError(f"{name} must be an odd integer >= 5, got {value}.")

    if not np.isclose(DX_M, DZ_M):
        raise ValueError("Current elastic solver requires DX_M == DZ_M.")
    if RECEIVER_LOOP_HALF_SIZE_M <= 0.0:
        raise ValueError("RECEIVER_LOOP_HALF_SIZE_M must be positive.")
    if RECEIVER_SPACING_M <= 0.0:
        raise ValueError("RECEIVER_SPACING_M must be positive.")
    if GAUGE_LENGTH_M / RECEIVER_SPACING_M < 2.0:
        raise ValueError(
            "GAUGE_LENGTH_M must span at least two receiver spacings because "
            "run_forward_simulation also constructs the DAS output."
        )

    max_n_boundary = max(int(cfg["n_boundary"]) for cfg in configs)

    interior_half_width_m = 0.5 * (INTERIOR_NX - 1) * DX_M
    interior_half_height_m = 0.5 * (INTERIOR_NZ - 1) * DZ_M

    source_offset_x_m = SOURCE_OFFSET_X_CELLS * DX_M
    source_offset_z_m = SOURCE_OFFSET_Z_CELLS * DZ_M

    if RECEIVER_LOOP_HALF_SIZE_M >= (
        interior_half_width_m - abs(source_offset_x_m)
    ):
        raise ValueError("Receiver loop reaches the side sponge entrance.")
    if RECEIVER_LOOP_HALF_SIZE_M >= (
        interior_half_height_m - abs(source_offset_z_m)
    ):
        raise ValueError("Receiver loop reaches the top/bottom sponge entrance.")

    dt = max_stable_dt(
        vp_max=VP_M_S,
        dx=DX_M,
        dz=DZ_M,
        half_order=HALF_ORDER,
        safety=CFL_SAFETY,
    )

    source_t0_s = 1.2 / SOURCE_F0_HZ

    # Earliest possible side backscatter: nearest side sponge entrance.
    nearest_side_entry_distance_m = (
        interior_half_width_m - abs(source_offset_x_m)
    )
    earliest_side_path_m = (
        2.0 * nearest_side_entry_distance_m
        - RECEIVER_LOOP_HALF_SIZE_M
    )
    late_start_s = source_t0_s + earliest_side_path_m / VP_M_S

    # Latest expected side return: farthest outer edge for the widest sponge.
    farthest_side_outer_distance_m = (
        interior_half_width_m
        + max_n_boundary * DX_M
        + abs(source_offset_x_m)
    )
    latest_side_path_m = (
        2.0 * farthest_side_outer_distance_m
        - RECEIVER_LOOP_HALF_SIZE_M
    )
    latest_side_return_s = source_t0_s + latest_side_path_m / VP_M_S
    end_time_s = (
        latest_side_return_s
        + TAIL_AFTER_LATEST_SIDE_REFLECTION_S
    )

    # Conservative contamination bound: top/bottom sponge can backscatter from
    # its entrance, not only from the outer numerical edge.
    nearest_vertical_entry_distance_m = (
        interior_half_height_m - abs(source_offset_z_m)
    )
    earliest_vertical_path_m = (
        2.0 * nearest_vertical_entry_distance_m
        - RECEIVER_LOOP_HALF_SIZE_M
    )
    earliest_vertical_contamination_s = (
        source_t0_s + earliest_vertical_path_m / VP_M_S
    )

    vertical_guard_s = earliest_vertical_contamination_s - end_time_s
    if vertical_guard_s < MIN_VERTICAL_GUARD_S:
        raise ValueError(
            "The record is not side-isolated. Earliest top/bottom sponge "
            f"contamination is at {earliest_vertical_contamination_s:.3f} s, "
            f"but the record ends at {end_time_s:.3f} s "
            f"(guard={vertical_guard_s:.3f} s; required "
            f">={MIN_VERTICAL_GUARD_S:.3f} s). Increase INTERIOR_NZ, shorten "
            "the tail, or reduce the tested maximum n_boundary."
        )

    nt = int(np.ceil(end_time_s / dt)) + 1
    duration_s = float((nt - 1) * dt)

    return {
        "max_n_boundary": max_n_boundary,
        "dt": dt,
        "nt": nt,
        "duration_s": duration_s,
        "source_t0_s": source_t0_s,
        "interior_half_width_m": interior_half_width_m,
        "interior_half_height_m": interior_half_height_m,
        "late_start_s": late_start_s,
        "latest_side_return_s": latest_side_return_s,
        "end_time_s": end_time_s,
        "earliest_vertical_contamination_s": earliest_vertical_contamination_s,
        "vertical_guard_s": vertical_guard_s,
    }


# ------------------------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------------------------
def main() -> None:
    configs = build_configs()
    setup = validate_geometry_and_timing(configs)

    dt = float(setup["dt"])
    nt = int(setup["nt"])
    duration_s = float(setup["duration_s"])
    late_start_s = float(setup["late_start_s"])

    print("Side-boundary absorption QC setup")
    print("---------------------------------")
    print(f"dx, dz                         : {DX_M:.1f}, {DZ_M:.1f} m")
    print(f"dt                             : {dt:.6e} s")
    print(f"nt / duration                  : {nt} / {duration_s:.3f} s")
    print(
        "fixed undamped interior         : "
        f"{(INTERIOR_NX - 1) * DX_M:.1f} x "
        f"{(INTERIOR_NZ - 1) * DZ_M:.1f} m"
    )
    print(
        "source-to-side sponge entry     : "
        f"~{float(setup['interior_half_width_m']):.1f} m"
    )
    print(f"receiver-loop half-size        : {RECEIVER_LOOP_HALF_SIZE_M:.1f} m")
    print(f"late/reflection window starts  : {late_start_s:.3f} s")
    print(
        "latest expected side return     : "
        f"{float(setup['latest_side_return_s']):.3f} s"
    )
    print(
        "earliest vertical contamination : "
        f"{float(setup['earliest_vertical_contamination_s']):.3f} s"
    )
    print(f"vertical timing guard          : {float(setup['vertical_guard_s']):.3f} s")
    print(f"number of runs                 : {len(configs)}")

    results: list[dict[str, float | int | str | bool]] = []
    energy_curves: list[dict[str, object]] = []

    for cfg in configs:
        n_boundary = int(cfg["n_boundary"])
        gamma_s = float(cfg["gamma_s"])
        label = str(cfg["label"])
        is_control = gamma_s == 0.0

        # Keep the undamped interior fixed. A wider sponge enlarges the grid.
        nx = INTERIOR_NX + 2 * n_boundary
        nz = INTERIOR_NZ + 2 * n_boundary

        grid = Grid2D(
            nx=nx,
            nz=nz,
            dx=DX_M,
            dz=DZ_M,
            nt=nt,
            dt=dt,
        )

        model = build_homogeneous_model(
            grid,
            vp_m_s=VP_M_S,
            vs_m_s=VS_M_S,
            rho_kg_m3=RHO_KG_M3,
        )

        ix_center = grid.nx // 2
        iz_center = grid.nz // 2
        x_center = float(grid.x[ix_center])
        z_center = float(grid.z[iz_center])

        # The same sub-cell offset is used for every grid to exercise the
        # production bilinear source path without changing physical geometry.
        x_src = x_center + SOURCE_OFFSET_X_CELLS * grid.dx
        z_src = z_center + SOURCE_OFFSET_Z_CELLS * grid.dz

        source = build_isotropic_source(
            grid=grid,
            x_m=x_src,
            z_m=z_src,
            scalar_moment=SOURCE_SCALAR_MOMENT,
            nt=grid.nt,
            dt=grid.dt,
            f0_hz=SOURCE_F0_HZ,
            derivative_order=0,
            spreading="bilinear",
        )

        receivers = build_central_receiver_loop(
            grid,
            x_center_m=x_src,
            z_center_m=z_src,
            half_size_m=RECEIVER_LOOP_HALF_SIZE_M,
            spacing_m=RECEIVER_SPACING_M,
        )

        side_outer_distance_m = (
            float(setup["interior_half_width_m"])
            + n_boundary * DX_M
            + abs(SOURCE_OFFSET_X_CELLS * DX_M)
        )
        expected_outer_return_s = (
            float(setup["source_t0_s"])
            + (
                2.0 * side_outer_distance_m
                - RECEIVER_LOOP_HALF_SIZE_M
            )
            / VP_M_S
        )

        print("\nRunning:", label)
        print(
            f"  grid={nx}x{nz} ({nx * nz:,} cells), "
            f"sponge={n_boundary * DX_M:.0f} m, gamma={gamma_s:.1f}, "
            f"receivers={receivers.nrec}"
        )

        run_result, _ = run_forward_simulation(
            model=model,
            source=source,
            receivers=receivers,
            gauge_length_m=GAUGE_LENGTH_M,
            half_order=HALF_ORDER,
            use_ts_sfd=False,
            n_boundary=n_boundary,
            gamma_s=gamma_s,
            snapshot_stride=None,
            backend="numba_fused",
            free_surface=False,
        )

        time_s = np.asarray(run_result.t, dtype=np.float64)
        receiver_vx = np.asarray(run_result.receiver_vx, dtype=np.float64)
        receiver_vz = np.asarray(run_result.receiver_vz, dtype=np.float64)

        if not np.all(np.isfinite(receiver_vx)):
            raise RuntimeError(f"{label}: receiver_vx contains NaN/Inf.")
        if not np.all(np.isfinite(receiver_vz)):
            raise RuntimeError(f"{label}: receiver_vz contains NaN/Inf.")

        # Two-component receiver energy proxy on the identical physical contour.
        energy = np.sum(receiver_vx**2 + receiver_vz**2, axis=0)

        direct_mask = time_s < late_start_s
        late_mask = time_s >= late_start_s

        direct_integral = integrate_energy(energy, time_s, direct_mask)
        late_integral = integrate_energy(energy, time_s, late_mask)
        direct_peak = float(np.max(energy[direct_mask]))
        late_peak = float(np.max(energy[late_mask]))

        late_to_direct_integral_ratio = safe_ratio(
            late_integral,
            direct_integral,
        )
        late_to_direct_peak_ratio = safe_ratio(late_peak, direct_peak)

        results.append(
            {
                "label": label,
                "is_control": is_control,
                "n_boundary": n_boundary,
                "sponge_width_m": n_boundary * DX_M,
                "gamma_s": gamma_s,
                "nx": nx,
                "nz": nz,
                "grid_cells": nx * nz,
                "dt_s": dt,
                "nt": nt,
                "duration_s": duration_s,
                "late_start_s": late_start_s,
                "expected_side_outer_return_s": expected_outer_return_s,
                "earliest_vertical_contamination_s": float(
                    setup["earliest_vertical_contamination_s"]
                ),
                "direct_energy_integral": direct_integral,
                "late_energy_integral": late_integral,
                "direct_energy_peak": direct_peak,
                "late_energy_peak": late_peak,
                "late_to_direct_integral_ratio": late_to_direct_integral_ratio,
                "late_to_direct_integral_db": energy_ratio_db(
                    late_to_direct_integral_ratio
                ),
                "late_to_direct_peak_ratio": late_to_direct_peak_ratio,
                "late_to_direct_peak_db": energy_ratio_db(
                    late_to_direct_peak_ratio
                ),
            }
        )

        energy_curves.append(
            {
                "label": label,
                "is_control": is_control,
                "time_s": time_s,
                "energy_normalized": energy / max(direct_peak, 1.0e-300),
            }
        )

        print(
            "  late/direct integral ratio : "
            f"{late_to_direct_integral_ratio:.6e} "
            f"({energy_ratio_db(late_to_direct_integral_ratio):.2f} dB)"
        )
        print(
            "  late/direct peak ratio     : "
            f"{late_to_direct_peak_ratio:.6e} "
            f"({energy_ratio_db(late_to_direct_peak_ratio):.2f} dB)"
        )

    # ------------------------------------------------------------------
    # Matched-control metrics
    # ------------------------------------------------------------------
    summary = pd.DataFrame(results)

    controls = (
        summary.loc[
            summary["is_control"],
            [
                "n_boundary",
                "direct_energy_integral",
                "late_energy_integral",
                "direct_energy_peak",
                "late_energy_peak",
            ],
        ]
        .rename(
            columns={
                "direct_energy_integral": "control_direct_energy_integral",
                "late_energy_integral": "control_late_energy_integral",
                "direct_energy_peak": "control_direct_energy_peak",
                "late_energy_peak": "control_late_energy_peak",
            }
        )
    )

    if controls["n_boundary"].duplicated().any():
        raise RuntimeError("Matched controls are not unique by n_boundary.")

    summary = summary.merge(
        controls,
        on="n_boundary",
        how="left",
        validate="many_to_one",
    )

    if summary["control_late_energy_integral"].isna().any():
        raise RuntimeError("At least one configuration has no matched control.")

    summary["late_to_matched_control_ratio"] = (
        summary["late_energy_integral"]
        / summary["control_late_energy_integral"].clip(lower=1.0e-300)
    )
    summary["late_to_matched_control_db"] = 10.0 * np.log10(
        summary["late_to_matched_control_ratio"].clip(lower=1.0e-300)
    )
    summary["late_peak_to_matched_control_ratio"] = (
        summary["late_energy_peak"]
        / summary["control_late_energy_peak"].clip(lower=1.0e-300)
    )
    summary["late_peak_to_matched_control_db"] = 10.0 * np.log10(
        summary["late_peak_to_matched_control_ratio"].clip(lower=1.0e-300)
    )
    summary["direct_to_matched_control_ratio"] = (
        summary["direct_energy_integral"]
        / summary["control_direct_energy_integral"].clip(lower=1.0e-300)
    )
    summary["direct_control_relative_error"] = np.abs(
        summary["direct_to_matched_control_ratio"] - 1.0
    )

    active = summary.loc[~summary["is_control"]].copy()
    if active.empty:
        raise RuntimeError("No active gamma_s > 0 configurations were tested.")

    min_grid_cells = float(active["grid_cells"].min())
    active["relative_grid_cost"] = active["grid_cells"] / min_grid_cells
    active["pareto_efficient"] = mark_pareto_efficient(
        active["grid_cells"].to_numpy(dtype=float),
        active["late_to_direct_integral_ratio"].to_numpy(dtype=float),
    )

    # Merge active-only diagnostics back into the complete table.
    summary = summary.merge(
        active[
            [
                "label",
                "relative_grid_cost",
                "pareto_efficient",
            ]
        ],
        on="label",
        how="left",
        validate="one_to_one",
    )
    summary["pareto_efficient"] = summary["pareto_efficient"].astype("boolean")

    summary = summary.sort_values(
        ["n_boundary", "gamma_s"],
        ascending=[True, True],
    )
    active = summary.loc[~summary["is_control"]].copy()

    # ------------------------------------------------------------------
    # Sanity check: direct wave should be unaffected by damping.
    # ------------------------------------------------------------------
    direct_bad = active[
        active["direct_control_relative_error"]
        > DIRECT_CONTROL_RELATIVE_TOLERANCE
    ]
    if not direct_bad.empty:
        print("\nWARNING: direct-wave mismatch relative to matched control")
        print("----------------------------------------------------------")
        for _, row in direct_bad.iterrows():
            print(
                f"{row['label']}: relative direct-energy difference = "
                f"{row['direct_control_relative_error']:.3%}"
            )
        print(
            "The source/receiver contour may be too close to the sponge, or "
            "the compared runs may not be physically equivalent."
        )

    # ------------------------------------------------------------------
    # Selection summaries
    # ------------------------------------------------------------------
    best_actual = active.loc[
        active["late_to_direct_integral_ratio"].idxmin()
    ]
    best_suppression = active.loc[
        active["late_to_matched_control_ratio"].idxmin()
    ]

    actual_threshold = (
        float(best_actual["late_to_direct_integral_ratio"])
        * RECOMMEND_WITHIN_FACTOR_OF_BEST
    )
    recommendation_pool = active[
        active["late_to_direct_integral_ratio"] <= actual_threshold
    ].copy()
    recommended = recommendation_pool.sort_values(
        [
            "grid_cells",
            "late_to_direct_integral_ratio",
            "late_to_matched_control_ratio",
        ],
        ascending=[True, True, True],
    ).iloc[0]

    summary_path = OUT_DIR / "boundary_absorption_summary.csv"
    active_path = OUT_DIR / "boundary_absorption_active_ranking.csv"
    summary.to_csv(summary_path, index=False)
    active.sort_values(
        "late_to_direct_integral_ratio",
        ascending=True,
    ).to_csv(active_path, index=False)

    print("\nBest actual residual contamination")
    print("----------------------------------")
    print(f"label                         : {best_actual['label']}")
    print(f"sponge width                  : {best_actual['sponge_width_m']:.0f} m")
    print(f"gamma_s                       : {best_actual['gamma_s']:.1f}")
    print(
        "late/direct integral ratio     : "
        f"{best_actual['late_to_direct_integral_ratio']:.6e} "
        f"({best_actual['late_to_direct_integral_db']:.2f} dB)"
    )

    print("\nBest matched-control suppression")
    print("--------------------------------")
    print(f"label                         : {best_suppression['label']}")
    print(
        "late/matched-control ratio     : "
        f"{best_suppression['late_to_matched_control_ratio']:.6e} "
        f"({best_suppression['late_to_matched_control_db']:.2f} dB)"
    )

    print("\nCost-aware recommendation")
    print("-------------------------")
    print(f"label                         : {recommended['label']}")
    print(f"grid cells                    : {int(recommended['grid_cells']):,}")
    print(f"relative grid cost            : {recommended['relative_grid_cost']:.3f}")
    print(
        "late/direct integral ratio     : "
        f"{recommended['late_to_direct_integral_ratio']:.6e}"
    )
    print(
        "late/matched-control ratio     : "
        f"{recommended['late_to_matched_control_ratio']:.6e}"
    )
    print(
        "selection rule                 : cheapest configuration within "
        f"{RECOMMEND_WITHIN_FACTOR_OF_BEST:g}x of best actual contamination"
    )

    max_width = max(SPONGE_FAMILIES)
    max_gamma_for_recommended_width = max(
        SPONGE_FAMILIES[int(recommended["n_boundary"])]
    )
    if (
        int(recommended["n_boundary"]) == max_width
        or float(recommended["gamma_s"]) == max_gamma_for_recommended_width
    ):
        print(
            "\nEDGE WARNING: the recommendation lies on the tested parameter "
            "boundary. Extend the sweep before freezing the production setting."
        )

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    for curve in energy_curves:
        linestyle = "--" if bool(curve["is_control"]) else "-"
        ax.semilogy(
            np.asarray(curve["time_s"]),
            np.maximum(
                np.asarray(curve["energy_normalized"]),
                1.0e-30,
            ),
            linestyle=linestyle,
            label=str(curve["label"]),
        )

    ax.axvline(
        late_start_s,
        linestyle=":",
        linewidth=1.4,
        label="late-window start",
    )
    ax.axvline(
        float(setup["earliest_vertical_contamination_s"]),
        linestyle="-.",
        linewidth=1.2,
        label="earliest vertical contamination",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Two-component loop energy / direct peak")
    ax.set_title("Side-boundary absorption QC")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    curves_path = OUT_DIR / "boundary_reflection_energy_curves.png"
    fig.savefig(curves_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    ordered_actual = active.sort_values(
        "late_to_direct_integral_ratio",
        ascending=False,
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        ordered_actual["label"],
        ordered_actual["late_to_direct_integral_ratio"],
    )
    ax.set_xscale("log")
    ax.set_xlabel("Late reflected energy / direct energy")
    ax.set_title("Actual residual boundary contamination — lower is better")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    actual_ranking_path = OUT_DIR / "boundary_actual_contamination_ranking.png"
    fig.savefig(actual_ranking_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    ordered_suppression = active.sort_values(
        "late_to_matched_control_ratio",
        ascending=False,
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        ordered_suppression["label"],
        ordered_suppression["late_to_matched_control_ratio"],
    )
    ax.set_xscale("log")
    ax.set_xlabel("Late energy / matched gamma=0 late energy")
    ax.set_title("Sponge-only suppression — lower is better")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    suppression_path = OUT_DIR / "boundary_matched_suppression_ranking.png"
    fig.savefig(suppression_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        active["grid_cells"],
        active["late_to_direct_integral_ratio"],
    )
    for _, row in active.iterrows():
        ax.annotate(
            str(row["label"]),
            (
                float(row["grid_cells"]),
                float(row["late_to_direct_integral_ratio"]),
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Grid cells per run")
    ax.set_ylabel("Late reflected energy / direct energy")
    ax.set_title("Boundary absorption versus computational cost")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    pareto_path = OUT_DIR / "boundary_absorption_cost_tradeoff.png"
    fig.savefig(pareto_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved outputs")
    print("-------------")
    print(summary_path)
    print(active_path)
    print(curves_path)
    print(actual_ranking_path)
    print(suppression_path)
    print(pareto_path)


if __name__ == "__main__":
    main()