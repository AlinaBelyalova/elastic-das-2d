from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from src.grid import Grid2D
from src.model import ElasticModel2D

if TYPE_CHECKING:
    from src.safod_builder import SafodBuildMetadata


_FIELD_LABELS = {
    "vp": "Vp [m/s]",
    "vs": "Vs [m/s]",
    "rho": "Density [kg/m³]",
    "lam": "λ [Pa]",
    "mu": "μ [Pa]",
}


_FIELD_CMAPS = {
    "vp": "turbo",
    "vs": "turbo",
    "rho": "viridis",
    "lam": "magma",
    "mu": "magma",
}


def _metadata_is_initial(metadata: Any) -> bool:
    if hasattr(metadata, "build_initial_model"):
        return bool(metadata.build_initial_model)

    model_type = str(getattr(metadata, "model_type", "")).lower()

    if "initial" in model_type:
        return True

    if (
        "reference" in model_type
        or "synthetic" in model_type
        or "true" in model_type
        or "geologic" in model_type
    ):
        return False

    return True


def _metadata_model_label(metadata: Any) -> str:
    model_type = str(getattr(metadata, "model_type", "")).strip()
    if model_type:
        return model_type.replace("_", " ")
    return "initial model" if _metadata_is_initial(metadata) else "reference model"


def _get_model_field(model: ElasticModel2D, field: str) -> np.ndarray:
    if field not in _FIELD_LABELS:
        raise ValueError(
            f"field must be one of {sorted(_FIELD_LABELS)}, got {field!r}."
        )

    if not hasattr(model, field):
        raise ValueError(f"ElasticModel2D has no field {field!r}.")

    arr = np.asarray(getattr(model, field), dtype=np.float64)

    if arr.ndim != 2:
        raise ValueError(f"Model field {field!r} must be 2D, got shape {arr.shape}.")

    return arr


def _plot_model_image(
    ax: Axes,
    grid: Grid2D,
    arr: np.ndarray,
    *,
    field: str,
):
    """
    Internal model convention:
        arr.shape == (nx, nz)

    Matplotlib image convention:
        image.shape == (ny, nx)

    Therefore plot arr.T with explicit physical extent.
    """
    expected_shape = (grid.nx, grid.nz)

    if arr.shape != expected_shape:
        raise ValueError(
            f"Model field shape {arr.shape} does not match grid shape "
            f"(nx, nz)=({grid.nx}, {grid.nz})."
        )

    extent = [
        float(grid.x[0]),
        float(grid.x[-1]),
        float(grid.z[-1]),
        float(grid.z[0]),
    ]

    im = ax.imshow(
        arr.T,
        extent=extent,
        origin="upper",
        aspect="equal",
        interpolation="nearest",
        cmap=_FIELD_CMAPS.get(field, "turbo"),
    )

    return im


def plot_safod_model(
    grid: Grid2D,
    model: ElasticModel2D,
    x_cable,
    z_cable,
    metadata: SafodBuildMetadata,
    field: str = "vp",
    ax: Axes | None = None,
    *,
    show_fault: bool = True,
    show_tie_point: bool = True,
    show_offset_segment: bool = True,
):
    """
    Plot SAFOD 2D model with projected DAS cable and SAF prior geometry.

    For initial models, the fault line is only geological metadata. It does not
    mean a sharp fault-zone velocity contrast is inserted into the model.
    """
    arr = _get_model_field(model, field)

    x_cable = np.asarray(x_cable, dtype=np.float64)
    z_cable = np.asarray(z_cable, dtype=np.float64)

    if x_cable.shape != z_cable.shape:
        raise ValueError(
            f"x_cable and z_cable must have same shape; "
            f"got {x_cable.shape} and {z_cable.shape}."
        )

    if x_cable.size < 2:
        raise ValueError("Cable geometry must contain at least two points.")

    is_initial = _metadata_is_initial(metadata)
    model_label = _metadata_model_label(metadata)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))
    else:
        fig = ax.figure

    im = _plot_model_image(
        ax=ax,
        grid=grid,
        arr=arr,
        field=field,
    )
    # Explicit fraction/pad keep colorbar width predictable, so the legend
    # offset below can reliably clear it (see bbox_to_anchor comment).
    cbar = fig.colorbar(im, ax=ax, label=_FIELD_LABELS[field], fraction=0.046, pad=0.04)

    # DAS cable
    ax.plot(
        x_cable,
        z_cable,
        color="white",
        lw=2.5,
        label="Projected DAS cable",
        zorder=5,
    )

    ax.scatter(
        x_cable[0],
        z_cable[0],
        c="white",
        edgecolors="black",
        s=45,
        zorder=8,
        label="Cable start",
    )

    ax.scatter(
        x_cable[-1],
        z_cable[-1],
        c="cyan",
        edgecolors="black",
        s=55,
        zorder=8,
        label="Cable end",
    )

    # SAF metadata / fault line
    has_fault_line = (
        hasattr(metadata, "x_fault_line")
        and hasattr(metadata, "z_fault_line")
        and metadata.x_fault_line is not None
        and metadata.z_fault_line is not None
    )

    if show_fault and has_fault_line:
        x_fault = np.asarray(metadata.x_fault_line, dtype=np.float64)
        z_fault = np.asarray(metadata.z_fault_line, dtype=np.float64)

        label = (
            "SAF prior / metadata line"
            if is_initial
            else "Synthetic SAF geometry"
        )

        ax.plot(
            x_fault,
            z_fault,
            "k--",
            lw=2.0,
            label=label,
            zorder=6,
        )

    # Tie point
    has_tie = hasattr(metadata, "x_tie_m") and hasattr(metadata, "z_tie_m")

    if show_tie_point and has_tie:
        ax.scatter(
            [float(metadata.x_tie_m)],
            [float(metadata.z_tie_m)],
            c="magenta",
            edgecolors="black",
            s=70,
            zorder=10,
            label="Fault tie point",
        )

    # Cable-to-fault offset at tie depth
    if (
        show_offset_segment
        and has_tie
        and hasattr(metadata, "x_cable_at_tie_m")
    ):
        x0 = float(metadata.x_cable_at_tie_m)
        x1 = float(metadata.x_tie_m)
        zt = float(metadata.z_tie_m)

        ax.plot(
            [x0, x1],
            [zt, zt],
            color="magenta",
            ls=":",
            lw=2.0,
            zorder=9,
            label="Cable-to-fault offset at tie depth",
        )

    ax.set_xlim(float(grid.x[0]), float(grid.x[-1]))
    ax.set_ylim(float(grid.z[-1]), float(grid.z[0]))

    ax.set_xlabel("Projected 2D section coordinate X [m]")
    ax.set_ylabel("Depth [m]")

    title_prefix = "SAFOD initial model" if is_initial else "SAFOD reference/synthetic model"
    ax.set_title(f"{title_prefix}: {model_label} ({field.upper()})")

    ax.grid(False)

    # bbox_to_anchor x=1.22 clears the colorbar footprint (fraction=0.046,
    # pad=0.04 above -> colorbar occupies roughly x in [1.0, 1.10] of ax
    # width). Verified empirically: bbox_to_anchor=(1.02, 1.0) overlaps the
    # colorbar by ~0.028 figure-fraction; 1.22 leaves a clear gap.
    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.22, 1.0), fontsize=8)

    # Pass the legend explicitly so tight_layout reserves room for it.
    # Without this, tight_layout only knows about ax + colorbar axes and
    # will not avoid clipping or overlapping the externally-anchored legend.
    fig.tight_layout()
    fig.canvas.draw()

    return fig, ax