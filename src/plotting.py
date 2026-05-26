from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from src.grid import Grid2D
from src.model import ElasticModel2D
from src.safod_builder import SafodBuildMetadata


def plot_safod_model(
    grid: Grid2D,
    model: ElasticModel2D,
    x_cable,
    z_cable,
    metadata: SafodBuildMetadata,
    field: str = "vp",
    ax: Axes | None = None,
):
    """
    Plot a SAFOD-like model together with cable geometry and the fault line.
    """
    if field not in {"vp", "vs", "rho"}:
        raise ValueError("field must be one of: 'vp', 'vs', 'rho'")

    arr = getattr(model, field)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    X, Z = grid.meshgrid()
    im = ax.pcolormesh(X, Z, arr, shading="auto", cmap="turbo")
    fig.colorbar(im, ax=ax, label=f"{field.upper()}")

    ax.plot(x_cable, z_cable, color="white", lw=2, label="DAS cable")

    if not metadata.build_initial_model:
        ax.plot(
            metadata.x_fault_line,
            metadata.z_fault_line,
            "k--",
            lw=2,
            label="San Andreas fault",
        )

    ax.scatter(
        [metadata.x_tie_m],
        [metadata.z_tie_m],
        c="magenta",
        s=50,
        zorder=10,
        label="Tie point",
    )

    ax.invert_yaxis()
    ax.set_xlabel("Projected section coordinate X [m]")
    ax.set_ylabel("Depth [m]")
    ax.set_title(
        f"SAFOD-like {'Initial' if metadata.build_initial_model else 'True'} model ({field.upper()})"
    )
    ax.legend(loc="upper right")
    fig.tight_layout()

    return fig, ax