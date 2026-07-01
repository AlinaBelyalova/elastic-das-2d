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
    "rho": "Density [kg/m\u00b3]",
    "lam": "\u03bb [Pa]",
    "mu": "\u03bc [Pa]",
}


_FIELD_CMAPS = {
    "vp": "turbo",
    "vs": "turbo",
    "rho": "viridis",
    "lam": "magma",
    "mu": "magma",
}

# Right margin reserved for ax+colorbar, fixed via subplots_adjust BEFORE the
# colorbar is created (NOT via tight_layout(), see note below).
_RIGHT_MARGIN = 0.75

# Legend anchor in FIGURE-fraction coordinates (NOT axes-fraction, and NOT
# relying on tight_layout to find room for it).
#
# Two independent problems were found and fixed here:
#
# 1. axes-fraction sensitivity: bbox_to_anchor defaults to axes-fraction
#    (ax.transAxes). A tall/narrow model (e.g. a deep-source SAFOD domain
#    with z_max_m=6500 under aspect="equal") shrinks ax's width by roughly
#    2x compared to a shallower model. The same axes-fraction offset then
#    maps to a much smaller absolute figure-fraction gap, letting the
#    legend creep toward the colorbar. Fixed by anchoring in figure-fraction
#    (bbox_transform=fig.transFigure), which is independent of ax's shape.
#
# 2. tight_layout() blindness to the externally-anchored legend: calling
#    fig.tight_layout() does not know about a legend placed via
#    bbox_to_anchor + transFigure (it only manages ax/colorbar). With
#    nothing else "competing" for space, tight_layout() greedily expands
#    ax+colorbar to fill nearly the whole canvas width, pushing the
#    colorbar's right edge (including its tick-label text, which can
#    extend ~0.05 figure-fraction beyond the colorbar patch itself for
#    4-digit values) directly into the legend's anchor point.
#
#    Fixed by reserving the right margin EXPLICITLY via
#    fig.subplots_adjust(right=_RIGHT_MARGIN) BEFORE fig.colorbar() is
#    called. Because colorbar(ax=ax, ...) shrinks ax to make room for the
#    new colorbar axes within ax's CURRENT bounding box, capping that
#    bounding box first guarantees ax+colorbar (patch AND tick labels)
#    never cross _RIGHT_MARGIN, regardless of model aspect ratio. No
#    fig.tight_layout() call is made afterward, since it would undo this.
#
# Both fixes were verified numerically and visually for both a tall
# deep-source domain (z_max=6500 m) and a shallower domain (z_max~3700 m):
# the colorbar's full rendered extent (patch + tick labels) and the
# legend's position are now IDENTICAL in figure-fraction across both
# shapes, with a verified positive clearance gap.
_LEGEND_BBOX_TO_ANCHOR = (0.83, 0.96)


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
        raise ValueError(f"field must be one of {sorted(_FIELD_LABELS)}, got {field!r}.")
    if not hasattr(model, field):
        raise ValueError(f"ElasticModel2D has no field {field!r}.")
    arr = np.asarray(getattr(model, field), dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Model field {field!r} must be 2D, got shape {arr.shape}.")
    return arr


def _plot_model_image(ax: Axes, grid: Grid2D, arr: np.ndarray, *, field: str):
    """
    Internal model convention: arr.shape == (nx, nz).
    Matplotlib image convention: image.shape == (ny, nx).
    Therefore plot arr.T with explicit physical extent.
    """
    expected_shape = (grid.nx, grid.nz)
    if arr.shape != expected_shape:
        raise ValueError(
            f"Model field shape {arr.shape} does not match grid shape "
            f"(nx, nz)=({grid.nx}, {grid.nz})."
        )
    extent = [float(grid.x[0]), float(grid.x[-1]), float(grid.z[-1]), float(grid.z[0])]
    im = ax.imshow(
        arr.T, extent=extent, origin="upper", aspect="equal",
        interpolation="nearest", cmap=_FIELD_CMAPS.get(field, "turbo"),
    )
    return im


def place_safod_legend(fig, ax, *, fontsize: int = 8):
    """
    Place SAFOD plot legend in figure-fraction coordinates.

    Use this after adding extra labelled artists, e.g. source markers.
    Do not call fig.tight_layout() afterwards — it would undo the right
    margin reserved by plot_safod_model's fig.subplots_adjust() call and
    re-expand ax+colorbar into the legend's space (see module note above
    _LEGEND_BBOX_TO_ANCHOR).
    """
    return ax.legend(
        loc="upper left",
        bbox_to_anchor=_LEGEND_BBOX_TO_ANCHOR,
        bbox_transform=fig.transFigure,
        fontsize=fontsize,
    )


def plot_safod_model(
    grid: Grid2D,
    model: ElasticModel2D,
    x_cable,
    z_cable,
    metadata: "SafodBuildMetadata",
    field: str = "vp",
    ax: Axes | None = None,
    *,
    show_fault: bool = True,
    show_tie_point: bool = True,
    show_offset_segment: bool = True,
):
    """
    Plot SAFOD 2D model with projected DAS cable and SAF prior geometry.

    For initial models, the fault line is only geological metadata. It does
    not mean a sharp fault-zone velocity contrast is inserted into the model.

    The returned legend is positioned in figure-fraction coordinates (see
    _LEGEND_BBOX_TO_ANCHOR) so it stays clear of the colorbar regardless of
    the model's aspect ratio. If the caller adds more labelled artists and
    re-calls ax.legend(...) afterwards (e.g. to add a source marker), use
    the SAME bbox_transform=fig.transFigure and the same anchor point —
    do not revert to the matplotlib default (axes-fraction), which is what
    caused the legend to creep toward the colorbar for tall/narrow models.
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
        # Reserve the right margin BEFORE creating the colorbar (see note
        # above _LEGEND_BBOX_TO_ANCHOR). This must happen before the first
        # fig.colorbar() call so colorbar() is forced to fit within it.
        fig.subplots_adjust(left=0.08, right=_RIGHT_MARGIN, top=0.92, bottom=0.10)
    else:
        fig = ax.figure

    im = _plot_model_image(ax=ax, grid=grid, arr=arr, field=field)
    cbar = fig.colorbar(im, ax=ax, label=_FIELD_LABELS[field], fraction=0.046, pad=0.04)

    ax.plot(x_cable, z_cable, color="white", lw=2.5, label="Projected DAS cable", zorder=5)
    ax.scatter(x_cable[0], z_cable[0], c="white", edgecolors="black", s=45, zorder=8, label="Cable start")
    ax.scatter(x_cable[-1], z_cable[-1], c="cyan", edgecolors="black", s=55, zorder=8, label="Cable end")

    has_fault_line = (
        hasattr(metadata, "x_fault_line")
        and hasattr(metadata, "z_fault_line")
        and metadata.x_fault_line is not None
        and metadata.z_fault_line is not None
    )
    if show_fault and has_fault_line:
        x_fault = np.asarray(metadata.x_fault_line, dtype=np.float64)
        z_fault = np.asarray(metadata.z_fault_line, dtype=np.float64)
        label = "SAF prior / metadata line" if is_initial else "Synthetic SAF geometry"
        ax.plot(x_fault, z_fault, "k--", lw=2.0, label=label, zorder=6)

    has_tie = hasattr(metadata, "x_tie_m") and hasattr(metadata, "z_tie_m")
    if show_tie_point and has_tie:
        ax.scatter(
            [float(metadata.x_tie_m)], [float(metadata.z_tie_m)],
            c="magenta", edgecolors="black", s=70, zorder=10, label="Fault tie point",
        )

    if show_offset_segment and has_tie and hasattr(metadata, "x_cable_at_tie_m"):
        x0 = float(metadata.x_cable_at_tie_m)
        x1 = float(metadata.x_tie_m)
        zt = float(metadata.z_tie_m)
        ax.plot(
            [x0, x1], [zt, zt], color="magenta", ls=":", lw=2.0, zorder=9,
            label="Cable-to-fault offset at tie depth",
        )

    ax.set_xlim(float(grid.x[0]), float(grid.x[-1]))
    ax.set_ylim(float(grid.z[-1]), float(grid.z[0]))
    ax.set_xlabel("Projected 2D section coordinate X [m]")
    ax.set_ylabel("Depth [m]")
    title_prefix = "SAFOD initial model" if is_initial else "SAFOD reference/synthetic model"
    ax.set_title(f"{title_prefix}: {model_label} ({field.upper()})")
    ax.grid(False)

    # Figure-fraction anchor with a pre-reserved right margin: robust to
    # axes aspect ratio AND immune to tight_layout's blindness to the
    # legend (see module note above _LEGEND_BBOX_TO_ANCHOR).
    legend = place_safod_legend(fig, ax, fontsize=8)

    # No fig.tight_layout() call here: it would override the subplots_adjust
    # margin set above and re-expand ax+colorbar into the legend's space.
    fig.canvas.draw()

    return fig, ax