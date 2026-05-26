from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.safod_builder import build_safod_model
from src.plotting import plot_safod_model


def main() -> None:
    geom_file = "/home/groups/ettore88/alina/imaging/SAFOD_downleg_Projected_2D.csv"

    grid, model, x_cable, z_cable, metadata = build_safod_model(
        geom_file=geom_file,
        build_initial_model=False,
        dx=5.0,
        dz=5.0,
        dt=5.0e-4,
        nt=2000,
        z_tie_m=2100.0,
        fault_dip_deg=82.0,
        fault_dip_sign=-1.0,
        left_block_name="salinian",
        right_block_name="franciscan",
    )

    fig, ax = plot_safod_model(
        grid=grid,
        model=model,
        x_cable=x_cable,
        z_cable=z_cable,
        metadata=metadata,
        field="vp",
    )

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    fig.savefig(out_dir / "safod_true_model_vp.png", dpi=200, bbox_inches="tight")

    np.savez_compressed(
        out_dir / "safod_true_model.npz",
        x=grid.x,
        z=grid.z,
        vp=model.vp,
        vs=model.vs,
        rho=model.rho,
        lam=model.lam,
        mu=model.mu,
        x_cable=x_cable,
        z_cable=z_cable,
        x_fault_line=metadata.x_fault_line,
        z_fault_line=metadata.z_fault_line,
        x_tie_m=metadata.x_tie_m,
        z_tie_m=metadata.z_tie_m,
    )

    print("Saved model to results/safod_true_model.npz")
    print("Saved figure to results/safod_true_model_vp.png")

    plt.show()


if __name__ == "__main__":
    main()