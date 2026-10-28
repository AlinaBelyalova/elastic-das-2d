"""Off-grid moment stress-rate source and its exact algebraic transpose.

The discrete delta is H_stress^-1 times the bilinear sampling transpose.
Thus its quadrature integral is exactly one, including SBP closure cells.
In the uniform interior this is precisely the existing weight/(dx*dz).
The tensor scale and STF sign/sampling are supplied unchanged by source.py.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .observations import _Stencil, _readonly
from .rectangular_observations import _bounded_stencil


@dataclass(frozen=True)
class DistributedStressSource:
    normal_shape: tuple[int, int]
    shear_shape: tuple[int, int]
    dx: float
    dz: float
    normal: _Stencil
    shear: _Stencil
    mxx: float
    mzz: float
    mxz: float

    def validate_grid(self, solver):
        if (
            self.normal_shape != solver.model.vp.shape
            or self.shear_shape != solver.model.mu_MM.shape
            or self.dx != solver.op_x.h
            or self.dz != solver.op_z.h
        ):
            raise ValueError("Distributed source and bounded SBP grid differ.")

    def inject(self, xx, zz, xz, scale):
        self.normal.scatter(np.array([scale * self.mxx]), xx)
        self.normal.scatter(np.array([scale * self.mzz]), zz)
        self.shear.scatter(np.array([scale * self.mxz]), xz)

    def gather(self, xx, zz, xz):
        return float(
            self.mxx * self.normal.sample(xx)[0]
            + self.mzz * self.normal.sample(zz)[0]
            + self.mxz * self.shear.sample(xz)[0]
        )


def build_moment_source(solver, source, grid):
    """Adapt EmbeddedSource2D; never resample/shift its STF or snap bilinear x/z.

    Both stress grids must bracket the physical source (half a cell inside
    all faces). Reject unsupported points instead of moving the source.
    The inherited numerical 2-D moment scale is preserved: this does not
    introduce a 3-D earthquake-moment to line-source amplitude conversion.
    """
    if source.spreading not in ("nearest", "bilinear"):
        raise ValueError("Source spreading must be nearest or bilinear.")
    if source.stf.nt != grid.nt or source.stf.dt != grid.dt or solver.dt != grid.dt:
        raise ValueError("Source STF nt/dt and SBP time grid must agree exactly.")
    if (
        solver.model.vp.shape != grid.shape
        or solver.op_x.h != grid.dx
        or solver.op_z.h != grid.dz
    ):
        raise ValueError("Source coordinate grid and SBP grid differ.")
    x, z = float(source.x_embedded_m), float(source.z_embedded_m)
    if not (
        np.isfinite(x)
        and np.isfinite(z)
        and grid.x0 + grid.dx / 2 <= x <= grid.x[-1] - grid.dx / 2
        and grid.z0 + grid.dz / 2 <= z <= grid.z[-1] - grid.dz / 2
    ):
        raise ValueError(
            "Source is outside common bounded stress support; no clipping."
        )

    def delta(shape, shift_x, shift_z, norm):
        stencil = _bounded_stencil(
            np.array([x]),
            np.array([z]),
            shape,
            grid.dx,
            grid.dz,
            grid.x0 + shift_x,
            grid.z0 + shift_z,
        )
        return _Stencil(
            stencil.ix,
            stencil.iz,
            _readonly(stencil.weights / norm[stencil.ix, stencil.iz]),
        )

    components = (source.m2d.Mxx, source.m2d.Mzz, source.m2d.Mxz)
    if not np.isfinite(components).all():
        raise ValueError("Source moment components must be finite.")
    return DistributedStressSource(
        grid.shape,
        solver.model.mu_MM.shape,
        grid.dx,
        grid.dz,
        delta(grid.shape, 0, 0, solver.wn),
        delta(solver.model.mu_MM.shape, grid.dx / 2, grid.dz / 2, solver.ws),
        *components,
    )
