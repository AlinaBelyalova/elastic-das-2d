# ==============================================================================
# src/receivers.py — 2D Receiver / DAS cable geometry for elastic FWI
#
# Design:
#   Receivers2D — truly immutable dataclass with read-only validated arrays
#   build_das_cable() — arc-length parameterised resampling from waypoints
#   create_l_shape_cable() — convenience helper for vertical borehole + surface
#
# Staggered-grid note:
#   ix, iz map channels to INTEGER grid nodes (where sxx/szz and material
#   properties live). Particle velocities live on half-integer nodes:
#     vx at (i+1/2, j)  -> nearest INTEGER ix carries a dx/2 systematic bias
#     vz at (i, j+1/2)  -> nearest INTEGER iz carries a dz/2 systematic bias
#   For a prototype this is acceptable.
#   Production: bilinear interpolation from the surrounding staggered nodes.
#
# Gauge-length constraint:
#   DAS strain-rate requires gauge_k = floor(gauge_L / channel_spacing + 1e-9) >= 2.
#   Callers must verify this before constructing a DAS operator.
# ==============================================================================

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.grid import Grid2D


@dataclass(frozen=True)
class Receivers2D:
    """
    Geometry and grid mapping for 2D DAS channel centers.

    Attributes
    ----------
    x, z :
        Physical coordinates of channel centers [m].
    ix, iz :
        Nearest-neighbour INTEGER grid indices.
    tx, tz :
        Unit tangent vectors along the cable (direction of increasing s).
    s :
        Arc-length coordinate of channel centers [m].

    All arrays are strictly 1D, identical in length (= nrec), and read-only.
    """
    x: np.ndarray
    z: np.ndarray
    ix: np.ndarray
    iz: np.ndarray
    tx: np.ndarray
    tz: np.ndarray
    s: np.ndarray

    def __post_init__(self) -> None:
        names_float = ("x", "z", "tx", "tz", "s")
        names_int = ("ix", "iz")
        ref_size = None

        for name in names_float + names_int:
            dtype = np.float64 if name in names_float else int

            arr = np.array(getattr(self, name), dtype=dtype, copy=True)
            if arr.ndim != 1:
                raise ValueError(f"'{name}' must be a 1D array, got shape {arr.shape}.")

            if ref_size is None:
                ref_size = arr.size
            elif arr.size != ref_size:
                raise ValueError(
                    f"All receiver arrays must have the same length; "
                    f"'{name}' has {arr.size}, expected {ref_size}."
                )

            arr.flags.writeable = False
            object.__setattr__(self, name, arr)

        norms = np.sqrt(self.tx**2 + self.tz**2)
        if not np.allclose(norms, 1.0, atol=1e-6):
            raise ValueError(
                f"tx/tz must be unit vectors. "
                f"max |norm - 1| = {np.abs(norms - 1.0).max():.2e}"
            )

    @property
    def nrec(self) -> int:
        return int(self.x.size)

    @property
    def channel_spacing(self) -> float:
        """
        Uniform arc-length spacing between adjacent channel centres [m].

        Since the cable is resampled on an even arc-length grid, this is
        computed directly from the first and last channel centres.
        """
        if self.nrec < 2:
            return float("nan")
        return float((self.s[-1] - self.s[0]) / (self.nrec - 1))

    def gauge_samples(self, gauge_length_m: float) -> int:
        """
        Number of receiver samples spanning the gauge length.

        gauge_k = floor(gauge_length_m / channel_spacing + 1e-9)

        Must be >= 2 for the DAS finite-difference operator to be well-defined.
        """
        ds = self.channel_spacing
        if not np.isfinite(ds) or ds <= 0.0:
            raise ValueError("Cannot compute gauge_samples: invalid channel spacing.")

        k = int(np.floor(gauge_length_m / ds + 1e-9))
        if k < 2:
            raise ValueError(
                f"gauge_length_m={gauge_length_m} m with channel_spacing≈{ds:.2f} m "
                f"gives gauge_k={k} < 2. Increase gauge_length_m or use finer spacing."
            )
        return k

    def summary(self) -> str:
        return (
            f"Receivers2D: {self.nrec} channels\n"
            f"  Arc-length: [{self.s.min():.1f}, {self.s.max():.1f}] m  "
            f"(spacing ≈ {self.channel_spacing:.2f} m)\n"
            f"  X: [{self.x.min():.1f}, {self.x.max():.1f}] m\n"
            f"  Z: [{self.z.min():.1f}, {self.z.max():.1f}] m"
        )

    def __repr__(self) -> str:
        return (
            f"Receivers2D(nrec={self.nrec}, "
            f"s=[{self.s.min():.1f},{self.s.max():.1f}]m, "
            f"x=[{self.x.min():.1f},{self.x.max():.1f}]m)"
        )

    def __str__(self) -> str:
        return self.summary()


def build_das_cable(
    grid: "Grid2D",
    waypoints_x: list[float] | np.ndarray,
    waypoints_z: list[float] | np.ndarray,
    channel_spacing_m: float,
    n_pml: int = 0,
) -> Receivers2D:
    """
    Build a 2D DAS cable from physical waypoints.

    The trajectory is parameterised by arc length and resampled to evenly
    spaced channel centres separated by `channel_spacing_m`.

    Parameters
    ----------
    grid :
        Grid2D-like object used for bounds checking and nearest-neighbour mapping.
        Must provide x0, z0, dx, dz, nx, nz.
    waypoints_x, waypoints_z :
        Cable control points [m].
    channel_spacing_m :
        Arc-length spacing between adjacent channel centres [m].
    n_pml :
        PML/sponge thickness [cells]. Channels inside this region are rejected.
    """
    for attr in ("x0", "z0", "dx", "dz", "nx", "nz"):
        if not hasattr(grid, attr):
            raise TypeError(
                f"'grid' must be a Grid2D-like object with attribute '{attr}'; "
                f"got {type(grid).__name__}."
            )

    if grid.dx <= 0.0 or grid.dz <= 0.0:
        raise ValueError(f"grid.dx and grid.dz must be positive; got dx={grid.dx}, dz={grid.dz}.")
    if grid.nx <= 1 or grid.nz <= 1:
        raise ValueError(f"grid.nx and grid.nz must be > 1; got nx={grid.nx}, nz={grid.nz}.")

    wx = np.asarray(waypoints_x, dtype=np.float64)
    wz = np.asarray(waypoints_z, dtype=np.float64)

    if wx.ndim != 1 or wz.ndim != 1:
        raise ValueError("waypoints_x and waypoints_z must be 1D.")
    if wx.size < 2:
        raise ValueError("At least 2 waypoints are required.")
    if wx.shape != wz.shape:
        raise ValueError("waypoints_x and waypoints_z must have the same shape.")
    if channel_spacing_m <= 0.0:
        raise ValueError(f"channel_spacing_m must be positive, got {channel_spacing_m}.")

    ds_seg = np.sqrt(np.diff(wx) ** 2 + np.diff(wz) ** 2)
    if np.any(ds_seg <= 0.0):
        raise ValueError("Waypoints contain repeated or zero-length segments.")

    s_way = np.insert(np.cumsum(ds_seg), 0, 0.0)
    total_length = float(s_way[-1])

    if total_length < channel_spacing_m / 2.0:
        raise ValueError(
            f"Cable length {total_length:.2f} m is too short for a channel at "
            f"channel_spacing_m/2 = {channel_spacing_m / 2.0:.2f} m."
        )

    # Channel centres at:
    #   ds/2, 3ds/2, 5ds/2, ...
    n_channels = int((total_length - channel_spacing_m / 2.0 + 1e-9) // channel_spacing_m) + 1
    if n_channels < 2:
        raise ValueError(
            f"Resampling produced only {n_channels} channel(s). "
            "Reduce channel_spacing_m or lengthen the cable."
        )

    s_chann = channel_spacing_m / 2.0 + np.arange(n_channels, dtype=np.float64) * channel_spacing_m

    x_chann = np.interp(s_chann, s_way, wx)
    z_chann = np.interp(s_chann, s_way, wz)

    tx = np.gradient(x_chann, s_chann)
    tz = np.gradient(z_chann, s_chann)

    norm = np.sqrt(tx**2 + tz**2)
    if np.any(norm <= 1e-12):
        raise ValueError("Degenerate tangent: resampled points coincide.")
    tx /= norm
    tz /= norm

    ix = np.round((x_chann - grid.x0) / grid.dx).astype(int)
    iz = np.round((z_chann - grid.z0) / grid.dz).astype(int)

    if np.any(ix < 0) or np.any(ix >= grid.nx):
        n_bad = int(((ix < 0) | (ix >= grid.nx)).sum())
        raise ValueError(
            f"Cable extends outside grid X bounds [0, {grid.nx - 1}] "
            f"at {n_bad} channel(s)."
        )

    if np.any(iz < 0) or np.any(iz >= grid.nz):
        n_bad = int(((iz < 0) | (iz >= grid.nz)).sum())
        raise ValueError(
            f"Cable extends outside grid Z bounds [0, {grid.nz - 1}] "
            f"at {n_bad} channel(s)."
        )

    if n_pml > 0:
        in_pml = (
            (ix < n_pml) | (ix >= grid.nx - n_pml) |
            (iz < n_pml) | (iz >= grid.nz - n_pml)
        )
        if np.any(in_pml):
            raise ValueError(
                f"{int(in_pml.sum())} channel(s) fall inside the PML/sponge zone "
                f"(n_pml={n_pml} cells from each edge)."
            )

    if channel_spacing_m < min(grid.dx, grid.dz):
        node_keys = ix * grid.nz + iz
        n_dup = len(ix) - len(np.unique(node_keys))
        if n_dup > 0:
            warnings.warn(
                f"channel_spacing_m={channel_spacing_m} m < grid spacing "
                f"(dx={grid.dx}, dz={grid.dz}): "
                f"{n_dup} channel pair(s) share the same grid node. "
                "Consider coarsening channel spacing or refining the grid.",
                UserWarning,
                stacklevel=2,
            )

    return Receivers2D(
        x=x_chann,
        z=z_chann,
        ix=ix,
        iz=iz,
        tx=tx,
        tz=tz,
        s=s_chann,
    )


def create_l_shape_cable(
    grid: "Grid2D",
    x_well: float,
    z_well_bottom: float,
    channel_spacing_m: float,
    n_pml: int = 0,
) -> Receivers2D:
    """
    Build an L-shaped cable: horizontal surface trench + vertical well segment.
    """
    edge_offset = max(2, n_pml + 1)
    x_start = grid.x0 + edge_offset * grid.dx
    z_surface = grid.z0 + edge_offset * grid.dz

    if x_well <= x_start:
        raise ValueError(
            f"x_well={x_well} m must be to the right of x_start={x_start:.1f} m."
        )
    if z_well_bottom <= z_surface:
        raise ValueError(
            f"z_well_bottom={z_well_bottom} m must be deeper than z_surface={z_surface:.1f} m."
        )

    wx = np.array([x_start, x_well, x_well], dtype=np.float64)
    wz = np.array([z_surface, z_surface, z_well_bottom], dtype=np.float64)

    return build_das_cable(
        grid=grid,
        waypoints_x=wx,
        waypoints_z=wz,
        channel_spacing_m=channel_spacing_m,
        n_pml=n_pml,
    )


def _self_test() -> None:
    from types import SimpleNamespace

    def make_grid(x0, z0, dx, dz, nx, nz):
        return SimpleNamespace(x0=x0, z0=z0, dx=dx, dz=dz, nx=nx, nz=nz)

    g = make_grid(0.0, 0.0, 10.0, 10.0, 200, 400)
    rec = build_das_cable(g, [500.0, 500.0], [100.0, 3000.0], channel_spacing_m=1.0)

    # Read-only arrays
    try:
        rec.x[0] = 999.0
        raise AssertionError("Arrays are not read-only.")
    except ValueError as e:
        assert "read-only" in str(e)
    print("Array immutability: OK")

    # Vertical cable tangents
    assert np.allclose(rec.tx, 0.0, atol=1e-6)
    assert np.allclose(rec.tz, 1.0, atol=1e-6)
    print(f"Vertical cable tangent: OK ({rec.nrec} channels)")

    # Uniform spacing
    ds = np.diff(rec.s)
    assert np.allclose(ds, ds[0], rtol=1e-10)
    print(f"Uniform arc-length: OK (ds={ds.mean():.2f} m)")

    # 45-degree cable
    g2 = make_grid(-50.0, -50.0, 10.0, 10.0, 150, 150)
    r2 = build_das_cable(g2, [0.0, 1000.0], [0.0, 1000.0], channel_spacing_m=5.0)
    assert np.allclose(r2.tx, 1 / np.sqrt(2), atol=1e-6)
    assert np.allclose(r2.tz, 1 / np.sqrt(2), atol=1e-6)
    print("45° cable tangent: OK")

    # gauge_samples
    assert rec.gauge_samples(10.0) == 10
    assert rec.gauge_samples(10.6) == 10
    assert rec.gauge_samples(11.0) == 11
    print("gauge_samples logic: OK")

    # Fine spacing without float-slip
    g3 = make_grid(0.0, 0.0, 1.0, 1.0, 5000, 10)
    r3 = build_das_cable(g3, [0.0, 4999.0], [5.0, 5.0], channel_spacing_m=0.1)
    assert r3.nrec > 0
    print(f"Exact channel-centre construction: OK ({r3.nrec} channels)")

    # Aliasing warning
    g4 = make_grid(0.0, -5.0, 10.0, 10.0, 30, 5)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        build_das_cable(g4, [0.0, 200.0], [0.0, 0.0], channel_spacing_m=3.0)
    assert any("share the same grid node" in str(x.message) for x in w)
    print("Aliasing UserWarning: OK")

    # PML rejection
    g5 = make_grid(0.0, 0.0, 5.0, 5.0, 30, 30)
    try:
        build_das_cable(g5, [0.0, 100.0], [5.0, 5.0], channel_spacing_m=5.0, n_pml=5)
        raise AssertionError("Expected ValueError for cable in PML zone.")
    except ValueError:
        pass
    print("PML zone rejection: OK")

    # L-shape cable
    g6 = make_grid(0.0, 0.0, 10.0, 10.0, 200, 400)
    rec_l = create_l_shape_cable(
        g6,
        x_well=1000.0,
        z_well_bottom=3500.0,
        channel_spacing_m=5.0,
        n_pml=5,
    )
    z_surf_expected = g6.z0 + (5 + 1) * g6.dz
    assert rec_l.z.min() >= z_surf_expected - 0.5 * 5.0
    print(f"create_l_shape_cable clearance: OK ({rec_l.nrec} channels)")

    # Bad grid object
    try:
        build_das_cable({"x0": 0}, [0, 1], [0, 1], 1.0)
        raise AssertionError("Expected TypeError.")
    except TypeError:
        pass
    print("Bad grid object TypeError: OK")

    # repr/str
    assert "nrec=" in repr(rec)
    assert "channels" in str(rec)
    assert len(repr(rec)) < len(str(rec))
    print("repr/str: OK")

    print(f"\nSelf-test PASSED\n{rec}")


if __name__ == "__main__":
    _self_test()