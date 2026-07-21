# ==============================================================================
# src/receivers.py — physical 2D receiver / DAS-cable geometry
#
# This module owns only the physical receiver geometry. Staggered-grid
# interpolation belongs in src.sampling; finite-gauge DAS physics belongs in
# src.das.
#
# Two constructors are intentionally separate:
#   build_das_cable_from_waypoints(...)
#       Generate uniformly spaced synthetic channel centres along a polyline.
#   build_receivers_from_channel_centres(...)
#       Preserve known field channel centres exactly, with no resampling.
#
# ix/iz are retained only as deprecated compatibility metadata for older code.
# Production wavefield sampling must use src.sampling.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.grid import Grid2D


def _readonly_1d(value, *, name: str, dtype) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name!r} must be 1D; got shape {array.shape}.")
    array.flags.writeable = False
    return array


def _validate_grid(grid: "Grid2D") -> None:
    for attr in ("x0", "z0", "dx", "dz", "nx", "nz"):
        if not hasattr(grid, attr):
            raise TypeError(
                f"'grid' must provide {attr!r}; got {type(grid).__name__}."
            )

    if not np.isfinite(grid.dx) or grid.dx <= 0.0:
        raise ValueError(f"grid.dx must be finite and positive; got {grid.dx}.")
    if not np.isfinite(grid.dz) or grid.dz <= 0.0:
        raise ValueError(f"grid.dz must be finite and positive; got {grid.dz}.")
    if int(grid.nx) < 2 or int(grid.nz) < 2:
        raise ValueError(
            f"grid.nx and grid.nz must be >= 2; got {grid.nx}, {grid.nz}."
        )


def _nearest_integer_grid_indices(
    grid: "Grid2D",
    x: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Deprecated compatibility indices; not used for staggered sampling."""
    ix = np.rint((x - float(grid.x0)) / float(grid.dx)).astype(np.int64)
    iz = np.rint((z - float(grid.z0)) / float(grid.dz)).astype(np.int64)
    return ix, iz


def _validate_grid_bounds(
    *,
    grid: "Grid2D",
    x: np.ndarray,
    z: np.ndarray,
    n_pml: int,
) -> tuple[np.ndarray, np.ndarray]:
    _validate_grid(grid)

    if int(n_pml) != n_pml or n_pml < 0:
        raise ValueError(f"n_pml must be a non-negative integer; got {n_pml!r}.")
    n_pml = int(n_pml)

    ix, iz = _nearest_integer_grid_indices(grid, x, z)

    outside_x = (ix < 0) | (ix >= int(grid.nx))
    outside_z = (iz < 0) | (iz >= int(grid.nz))

    if np.any(outside_x):
        bad = np.flatnonzero(outside_x)[:5]
        raise ValueError(
            "Receiver geometry extends outside the grid in x. "
            f"First bad receiver indices: {bad.tolist()}."
        )

    if np.any(outside_z):
        bad = np.flatnonzero(outside_z)[:5]
        raise ValueError(
            "Receiver geometry extends outside the grid in z. "
            f"First bad receiver indices: {bad.tolist()}."
        )

    if n_pml > 0:
        in_pml = (
            (ix < n_pml)
            | (ix >= int(grid.nx) - n_pml)
            | (iz < n_pml)
            | (iz >= int(grid.nz) - n_pml)
        )
        if np.any(in_pml):
            bad = np.flatnonzero(in_pml)[:5]
            raise ValueError(
                f"{int(np.count_nonzero(in_pml))} receiver(s) lie inside "
                f"the sponge/PML region (n_pml={n_pml}). "
                f"First bad receiver indices: {bad.tolist()}."
            )

    return ix, iz


def _compute_projected_unit_tangents(
    *,
    x: np.ndarray,
    z: np.ndarray,
    s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute unit tangents of the cable projected into the x-z plane."""
    if x.size < 2:
        raise ValueError("At least two receiver centres are required.")

    edge_order = 2 if x.size >= 3 else 1
    dx_ds = np.gradient(x, s, edge_order=edge_order)
    dz_ds = np.gradient(z, s, edge_order=edge_order)
    norm = np.hypot(dx_ds, dz_ds)

    if np.any(~np.isfinite(norm)):
        raise ValueError("Projected tangent calculation produced NaN or Inf.")
    if np.any(norm <= 1.0e-12):
        bad = np.flatnonzero(norm <= 1.0e-12)[:5]
        raise ValueError(
            "Degenerate projected cable tangent. "
            f"First bad receiver indices: {bad.tolist()}."
        )

    return dx_ds / norm, dz_ds / norm


@dataclass(frozen=True)
class Receivers2D:
    """
    Immutable physical geometry of 2D receiver/channel centres.

    Parameters
    ----------
    x, z
        Physical receiver coordinates [m].
    tx, tz
        Unit tangent components in the modelling x-z plane.
    s
        Strictly increasing, uniformly sampled cable coordinate [m].
        For registered borehole DAS geometry, measured depth is appropriate.
    ix, iz
        Deprecated nearest integer-grid indices kept only for compatibility.
        Production staggered-grid sampling uses src.sampling instead.
    """

    x: np.ndarray
    z: np.ndarray
    tx: np.ndarray
    tz: np.ndarray
    s: np.ndarray
    ix: np.ndarray | None = None
    iz: np.ndarray | None = None

    def __post_init__(self) -> None:
        x = _readonly_1d(self.x, name="x", dtype=np.float64)
        z = _readonly_1d(self.z, name="z", dtype=np.float64)
        tx = _readonly_1d(self.tx, name="tx", dtype=np.float64)
        tz = _readonly_1d(self.tz, name="tz", dtype=np.float64)
        s = _readonly_1d(self.s, name="s", dtype=np.float64)

        nrec = x.size
        for name, array in (("z", z), ("tx", tx), ("tz", tz), ("s", s)):
            if array.size != nrec:
                raise ValueError(
                    "All receiver arrays must have identical lengths; "
                    f"{name!r} has {array.size}, expected {nrec}."
                )

        for name, array in (("x", x), ("z", z), ("tx", tx), ("tz", tz), ("s", s)):
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name!r} contains NaN or Inf.")

        if nrec > 1:
            ds = np.diff(s)
            if np.any(ds <= 0.0):
                raise ValueError("Receiver coordinate s must increase strictly.")

            mean_ds = float(np.mean(ds))
            if not np.allclose(ds, mean_ds, rtol=1.0e-5, atol=1.0e-8):
                raise ValueError(
                    "Receivers2D currently requires uniform spacing in s "
                    "because src.das uses uniform cable interpolation. "
                    f"diff(s) ranges from {ds.min():.9f} to {ds.max():.9f} m."
                )

        if nrec > 0:
            tangent_norm = np.hypot(tx, tz)
            if not np.allclose(tangent_norm, 1.0, rtol=0.0, atol=1.0e-6):
                raise ValueError(
                    "tx/tz must be unit vectors. Maximum norm error is "
                    f"{np.max(np.abs(tangent_norm - 1.0)):.3e}."
                )

        ix_value = np.full(nrec, -1, dtype=np.int64) if self.ix is None else self.ix
        iz_value = np.full(nrec, -1, dtype=np.int64) if self.iz is None else self.iz
        ix = _readonly_1d(ix_value, name="ix", dtype=np.int64)
        iz = _readonly_1d(iz_value, name="iz", dtype=np.int64)

        if ix.size != nrec or iz.size != nrec:
            raise ValueError(
                "Deprecated ix/iz arrays must match the receiver count; "
                f"got ix={ix.size}, iz={iz.size}, nrec={nrec}."
            )

        object.__setattr__(self, "x", x)
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "tx", tx)
        object.__setattr__(self, "tz", tz)
        object.__setattr__(self, "s", s)
        object.__setattr__(self, "ix", ix)
        object.__setattr__(self, "iz", iz)

    @property
    def nrec(self) -> int:
        return int(self.x.size)

    @property
    def channel_spacing(self) -> float:
        """Uniform spacing of receiver centres along s [m]."""
        if self.nrec < 2:
            return float("nan")
        return float(np.mean(np.diff(self.s)))

    @property
    def has_legacy_grid_indices(self) -> bool:
        return bool(
            self.nrec > 0
            and np.all(self.ix >= 0)
            and np.all(self.iz >= 0)
        )

    def summary(self) -> str:
        if self.nrec == 0:
            return "Receivers2D: 0 channels"
        return (
            f"Receivers2D: {self.nrec} channels\n"
            f"  s: [{self.s.min():.3f}, {self.s.max():.3f}] m\n"
            f"  spacing: {self.channel_spacing:.6f} m\n"
            f"  x: [{self.x.min():.3f}, {self.x.max():.3f}] m\n"
            f"  z: [{self.z.min():.3f}, {self.z.max():.3f}] m"
        )

    def __repr__(self) -> str:
        if self.nrec == 0:
            return "Receivers2D(nrec=0)"
        return (
            f"Receivers2D(nrec={self.nrec}, "
            f"ds={self.channel_spacing:.6f} m, "
            f"s=[{self.s.min():.3f},{self.s.max():.3f}] m)"
        )

    def __str__(self) -> str:
        return self.summary()


def build_receivers_from_channel_centres(
    x,
    z,
    s,
    *,
    grid: "Grid2D | None" = None,
    n_pml: int = 0,
) -> Receivers2D:
    """
    Build geometry from already known physical channel centres.

    No resampling is performed: input channel i remains receiver i. This is the
    correct constructor for registered field geometry such as SAFOD.
    """
    x_array = np.asarray(x, dtype=np.float64)
    z_array = np.asarray(z, dtype=np.float64)
    s_array = np.asarray(s, dtype=np.float64)

    for name, array in (("x", x_array), ("z", z_array), ("s", s_array)):
        if array.ndim != 1:
            raise ValueError(f"{name!r} must be 1D; got shape {array.shape}.")

    if not (x_array.size == z_array.size == s_array.size):
        raise ValueError(
            "x, z and s must have identical lengths; "
            f"got {x_array.size}, {z_array.size}, {s_array.size}."
        )
    if x_array.size < 2:
        raise ValueError("At least two channel centres are required.")

    for name, array in (("x", x_array), ("z", z_array), ("s", s_array)):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name!r} contains NaN or Inf.")

    if np.any(np.diff(s_array) <= 0.0):
        raise ValueError("s must increase strictly.")

    tx, tz = _compute_projected_unit_tangents(x=x_array, z=z_array, s=s_array)

    if grid is None:
        if n_pml != 0:
            raise ValueError("n_pml requires a grid for bounds checking.")
        ix = None
        iz = None
    else:
        ix, iz = _validate_grid_bounds(
            grid=grid,
            x=x_array,
            z=z_array,
            n_pml=n_pml,
        )

    return Receivers2D(
        x=x_array,
        z=z_array,
        tx=tx,
        tz=tz,
        s=s_array,
        ix=ix,
        iz=iz,
    )


def build_das_cable_from_waypoints(
    grid: "Grid2D",
    waypoints_x,
    waypoints_z,
    channel_spacing_m: float,
    n_pml: int = 0,
) -> Receivers2D:
    """
    Generate uniformly spaced synthetic channel centres along a polyline.

    The first centre is channel_spacing_m/2 from the first waypoint. Use
    build_receivers_from_channel_centres() when inputs already are field
    channel centres.
    """
    _validate_grid(grid)

    if not np.isfinite(channel_spacing_m) or channel_spacing_m <= 0.0:
        raise ValueError(
            f"channel_spacing_m must be finite and positive; got {channel_spacing_m}."
        )

    wx = np.asarray(waypoints_x, dtype=np.float64)
    wz = np.asarray(waypoints_z, dtype=np.float64)

    if wx.ndim != 1 or wz.ndim != 1:
        raise ValueError("waypoints_x and waypoints_z must be 1D.")
    if wx.shape != wz.shape:
        raise ValueError("waypoints_x and waypoints_z must have identical shapes.")
    if wx.size < 2:
        raise ValueError("At least two waypoints are required.")
    if not (np.all(np.isfinite(wx)) and np.all(np.isfinite(wz))):
        raise ValueError("Waypoints contain NaN or Inf.")

    segment_length = np.hypot(np.diff(wx), np.diff(wz))
    if np.any(segment_length <= 0.0):
        raise ValueError("Waypoints contain repeated or zero-length segments.")

    s_waypoint = np.insert(np.cumsum(segment_length), 0, 0.0)
    total_length = float(s_waypoint[-1])
    spacing = float(channel_spacing_m)
    half_spacing = spacing / 2.0

    if total_length < half_spacing + spacing:
        raise ValueError(
            "Cable is too short to contain two channel centres at spacing "
            f"{spacing:.6f} m; polyline length is {total_length:.6f} m."
        )

    n_channels = int(
        np.floor((total_length - half_spacing) / spacing + 1.0e-12)
    ) + 1
    if n_channels < 2:
        raise ValueError("Resampling produced fewer than two channels.")

    s_channel = half_spacing + np.arange(n_channels, dtype=np.float64) * spacing
    x_channel = np.interp(s_channel, s_waypoint, wx)
    z_channel = np.interp(s_channel, s_waypoint, wz)
    tx, tz = _compute_projected_unit_tangents(
        x=x_channel,
        z=z_channel,
        s=s_channel,
    )
    ix, iz = _validate_grid_bounds(
        grid=grid,
        x=x_channel,
        z=z_channel,
        n_pml=n_pml,
    )

    return Receivers2D(
        x=x_channel,
        z=z_channel,
        tx=tx,
        tz=tz,
        s=s_channel,
        ix=ix,
        iz=iz,
    )


def build_das_cable(
    grid: "Grid2D",
    waypoints_x,
    waypoints_z,
    channel_spacing_m: float,
    n_pml: int = 0,
) -> Receivers2D:
    """Backward-compatible alias for build_das_cable_from_waypoints()."""
    return build_das_cable_from_waypoints(
        grid=grid,
        waypoints_x=waypoints_x,
        waypoints_z=waypoints_z,
        channel_spacing_m=channel_spacing_m,
        n_pml=n_pml,
    )


def create_l_shape_cable(
    grid: "Grid2D",
    x_well: float,
    z_well_bottom: float,
    channel_spacing_m: float,
    n_pml: int = 0,
) -> Receivers2D:
    """Build a synthetic L-shaped surface-plus-borehole cable."""
    _validate_grid(grid)

    edge_offset = max(2, int(n_pml) + 1)
    x_start = float(grid.x0) + edge_offset * float(grid.dx)
    z_surface = float(grid.z0) + edge_offset * float(grid.dz)

    if x_well <= x_start:
        raise ValueError(
            f"x_well={x_well} m must exceed x_start={x_start:.3f} m."
        )
    if z_well_bottom <= z_surface:
        raise ValueError(
            "z_well_bottom must be deeper than the synthetic surface; "
            f"got bottom={z_well_bottom}, surface={z_surface:.3f} m."
        )

    return build_das_cable_from_waypoints(
        grid=grid,
        waypoints_x=np.array([x_start, x_well, x_well], dtype=np.float64),
        waypoints_z=np.array([z_surface, z_surface, z_well_bottom], dtype=np.float64),
        channel_spacing_m=channel_spacing_m,
        n_pml=n_pml,
    )


def _self_test() -> None:
    from types import SimpleNamespace

    def make_grid(x0, z0, dx, dz, nx, nz):
        return SimpleNamespace(x0=x0, z0=z0, dx=dx, dz=dz, nx=nx, nz=nz)

    grid = make_grid(0.0, 0.0, 10.0, 10.0, 200, 400)
    receivers = build_das_cable(
        grid,
        [500.0, 500.0],
        [100.0, 3000.0],
        channel_spacing_m=1.0,
    )

    try:
        receivers.x[0] = 999.0
        raise AssertionError("Receiver arrays are not read-only.")
    except ValueError:
        pass
    print("Array immutability: OK")

    assert np.allclose(receivers.tx, 0.0, atol=1.0e-6)
    assert np.allclose(receivers.tz, 1.0, atol=1.0e-6)
    print(f"Vertical cable tangent: OK ({receivers.nrec} channels)")

    assert np.allclose(
        np.diff(receivers.s),
        receivers.channel_spacing,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    print(f"Uniform spacing: OK (ds={receivers.channel_spacing:.6f} m)")

    x_known = np.array([10.0, 11.0, 12.0, 13.0])
    z_known = np.array([20.0, 21.0, 22.0, 23.0])
    s_known = np.array([100.0, 102.0, 104.0, 106.0])
    registered = build_receivers_from_channel_centres(
        x=x_known,
        z=z_known,
        s=s_known,
    )

    assert np.array_equal(registered.x, x_known)
    assert np.array_equal(registered.z, z_known)
    assert np.array_equal(registered.s, s_known)
    print("Known channel centres preserved exactly: OK")

    expected = 1.0 / np.sqrt(2.0)
    assert np.allclose(registered.tx, expected, atol=1.0e-12)
    assert np.allclose(registered.tz, expected, atol=1.0e-12)
    print("Projected 45-degree tangent: OK")

    assert receivers.has_legacy_grid_indices
    print("Legacy ix/iz compatibility metadata: OK")

    fine_grid = make_grid(0.0, 0.0, 1.0, 1.0, 5000, 10)
    fine = build_das_cable(
        fine_grid,
        [0.0, 4999.0],
        [5.0, 5.0],
        channel_spacing_m=0.1,
    )
    assert fine.nrec > 0
    print(f"Sub-grid receiver spacing: OK ({fine.nrec} channels)")

    pml_grid = make_grid(0.0, 0.0, 5.0, 5.0, 30, 30)
    try:
        build_das_cable(
            pml_grid,
            [0.0, 100.0],
            [5.0, 5.0],
            channel_spacing_m=5.0,
            n_pml=5,
        )
        raise AssertionError("Expected PML bounds failure.")
    except ValueError:
        pass
    print("PML bounds rejection: OK")

    l_grid = make_grid(0.0, 0.0, 10.0, 10.0, 200, 400)
    l_receivers = create_l_shape_cable(
        l_grid,
        x_well=1000.0,
        z_well_bottom=3500.0,
        channel_spacing_m=5.0,
        n_pml=5,
    )
    assert l_receivers.nrec >= 2
    print(f"L-shaped cable construction: OK ({l_receivers.nrec} channels)")

    empty = Receivers2D(
        x=np.array([]),
        z=np.array([]),
        tx=np.array([]),
        tz=np.array([]),
        s=np.array([]),
    )
    assert empty.nrec == 0
    print("Empty geometry container: OK")

    try:
        Receivers2D(
            x=np.array([0.0, 1.0, 2.0]),
            z=np.array([0.0, 0.0, 0.0]),
            tx=np.ones(3),
            tz=np.zeros(3),
            s=np.array([0.0, 1.0, 2.2]),
        )
        raise AssertionError("Expected non-uniform spacing failure.")
    except ValueError:
        pass
    print("Non-uniform spacing guard: OK")

    assert "nrec=" in repr(receivers)
    assert "channels" in str(receivers)
    print("repr/str: OK")
    print("\nreceivers.py: all self-tests passed")


if __name__ == "__main__":
    _self_test()