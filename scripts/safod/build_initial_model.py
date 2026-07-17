# ==============================================================================
# scripts/build_safod_initial_model.py
#
# Build a geologically informed SAFOD initial model for 2D elastic-DAS
# modelling / future FWI.
#
# One script, one builder:
#   src/safod_builder.py builds the model
#   src/plotting.py plots it
#
# Geometry convention
# -------------------
# The projected geometry CSV has columns:
#
#   X_2D_m, Z_2D_m
#
# but the plotted trajectory shows that:
#
#   X_2D_m behaves like depth / downhole-like coordinate
#   Z_2D_m behaves like horizontal projected offset
#
# Therefore:
#
#   model x <- Z_2D_m
#   model z <- X_2D_m
#
# Model philosophy
# ----------------
# This is not a pure 1D gradient model anymore. It includes:
#
#   - Pilot-Hole sonic-log-inspired depth trend
#   - broad Pilot-Hole low-velocity fractured intervals
#   - smooth cross-fault contrast
#   - broad low-velocity SAF damage zone
#
# It is still an INITIAL model, so everything is smooth and conservative.
# ==============================================================================

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.safod_builder import build_safod_model
from src.plotting import plot_safod_model


def main() -> None:
    geom_file = "/home/groups/ettore88/alina/imaging/SAFOD_downleg_Projected_2D.csv"

    out_dir = Path("results/safod_initial_model")
    out_dir.mkdir(parents=True, exist_ok=True)

    grid, model, x_cable, z_cable, metadata = build_safod_model(
        geom_file=geom_file,

        # Axis fix:
        # CSV Z_2D_m is horizontal section coordinate.
        # CSV X_2D_m is depth-positive coordinate.
        x_column="Z_2D_m",
        z_column="X_2D_m",

        # Build geologically informed smooth initial model.
        build_initial_model=True,

        dx=5.0,
        dz=5.0,
        dt=None,

        nt=3000,
        half_order=2,
        cfl_safety=0.80,

        # Anchor SAF prior to cable end because the DAS cable stops before
        # crossing the fault zone.
        z_tie_m=None,
        anchor_fault_to_cable_end=True,
        fault_offset_from_cable_m=105.0,

        # Near-vertical SAF. fault_dip_sign controls which way the line dips.
        fault_dip_deg=82.0,
        fault_dip_sign=-1.0,

        left_block_name="salinian",
        right_block_name="franciscan",

        # ------------------------------------------------------------------
        # Geological priors for the initial model
        # ------------------------------------------------------------------
        # Negative: right side of dashed fault is slower than left side.
        # If the geologic side is reversed in this 2D projection, flip sign.
        initial_cross_fault_contrast=-0.08,
        initial_cross_fault_transition_m=350.0,

        # Broad SAF low-velocity damage zone.
        # Keep this broad/smooth for an initial model.
        initial_fault_zone_width_m=160.0,
        initial_fault_zone_velocity_reduction=0.14,

        # Pilot-Hole fractured intervals as broad depth LVZs.
        include_pilot_hole_lvz_in_initial=True,
        initial_pilot_hole_lvz_strength=0.035,

        # Smooth but not completely 1D.
        smooth_initial_sigma_m=80.0,
    )

    print("\nSAFOD initial model")
    print("-------------------")
    print(f"model_type : {metadata.model_type}")
    print(f"grid       : nx={grid.nx}, nz={grid.nz}")
    print(f"dx, dz     : {grid.dx:.2f}, {grid.dz:.2f} m")
    print(f"dt         : {grid.dt:.6e} s")
    print(f"nt         : {grid.nt}")
    print(f"x range    : {grid.x[0]:.1f} to {grid.x[-1]:.1f} m")
    print(f"z range    : {grid.z[0]:.1f} to {grid.z[-1]:.1f} m")
    print(f"Vp range   : {model.vp.min():.1f} to {model.vp.max():.1f} m/s")
    print(f"Vs range   : {model.vs.min():.1f} to {model.vs.max():.1f} m/s")
    print(f"rho range  : {model.rho.min():.1f} to {model.rho.max():.1f} kg/m^3")

    print("\nCable / SAF geometry")
    print("--------------------")
    print(f"cable start: x={x_cable[0]:.1f} m, z={z_cable[0]:.1f} m")
    print(f"cable end  : x={metadata.x_cable_end_m:.1f} m, z={metadata.z_cable_end_m:.1f} m")
    print(f"anchor to cable end: {metadata.anchor_fault_to_cable_end}")
    print(f"x_tie      : {metadata.x_tie_m:.1f} m")
    print(f"z_tie      : {metadata.z_tie_m:.1f} m")
    print(f"cable@tie  : {metadata.x_cable_at_tie_m:.1f} m")
    print(f"fault offset from cable at tie: {metadata.fault_offset_from_cable_m:.1f} m")
    print(f"fault dip  : {metadata.fault_dip_deg:.1f} deg")
    print(f"fault sign : {metadata.fault_dip_sign:.1f}")

    # --------------------------------------------------------------------------
    # Plot Vp, Vs, rho.
    # --------------------------------------------------------------------------
    for field in ["vp", "vs", "rho"]:
        fig, ax = plot_safod_model(
            grid=grid,
            model=model,
            x_cable=x_cable,
            z_cable=z_cable,
            metadata=metadata,
            field=field,
            show_fault=True,
            show_tie_point=True,
            show_offset_segment=True,
        )

        fig.savefig(
            out_dir / f"safod_initial_model_{field}.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)

    # --------------------------------------------------------------------------
    # Save model.
    # --------------------------------------------------------------------------
    np.savez_compressed(
        out_dir / "safod_initial_model.npz",
        x=grid.x,
        z=grid.z,
        dx=np.array(grid.dx),
        dz=np.array(grid.dz),
        dt=np.array(grid.dt),
        nt=np.array(grid.nt),

        vp=model.vp,
        vs=model.vs,
        rho=model.rho,
        lam=model.lam,
        mu=model.mu,

        x_cable=x_cable,
        z_cable=z_cable,

        x_fault_line=metadata.x_fault_line,
        z_fault_line=metadata.z_fault_line,

        x_tie_m=np.array(metadata.x_tie_m),
        z_tie_m=np.array(metadata.z_tie_m),
        x_cable_at_tie_m=np.array(metadata.x_cable_at_tie_m),

        x_cable_end_m=np.array(metadata.x_cable_end_m),
        z_cable_end_m=np.array(metadata.z_cable_end_m),
        anchor_fault_to_cable_end=np.array(metadata.anchor_fault_to_cable_end),

        fault_offset_from_cable_m=np.array(metadata.fault_offset_from_cable_m),
        fault_dip_deg=np.array(metadata.fault_dip_deg),
        fault_dip_sign=np.array(metadata.fault_dip_sign),

        model_type=np.array(metadata.model_type),
        left_block_name=np.array(metadata.left_block_name),
        right_block_name=np.array(metadata.right_block_name),
        notes=np.array(metadata.notes),
    )

    print(f"\nSaved model to: {out_dir / 'safod_initial_model.npz'}")
    print(f"Saved figures to: {out_dir}")

    print("\nCHECK:")
    print("  1. Cable should go downward first, then deviate laterally.")
    print("  2. A broad low-velocity zone should be visible around the dashed SAF line.")
    print("  3. One side of the SAF should be slightly slower than the other.")
    print("  4. If the slower side is geologically wrong, flip:")
    print("         initial_cross_fault_contrast=+0.08")
    print("     or flip fault_offset_from_cable_m if the fault is on the wrong side.")


if __name__ == "__main__":
    main()