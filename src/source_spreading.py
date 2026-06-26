# ==============================================================================
# src/source_spreading.py — staggered-grid bilinear source spreading
#
# Purpose
#   Compute bilinear interpolation stencils for distributing a physical
#   point source at arbitrary coordinates (x_s, z_s) onto the four
#   surrounding nodes of each staggered field.
#
# Design
#   This module contains only low-level geometry and interpolation logic.
#   No solver time stepping, no DAS physics, no earthquake catalog logic.
#
# Staggered node positions, using grid.py conventions
#   sxx[i,j], szz[i,j]  at  (x0 + i*dx,           z0 + j*dz        )
#   sxz[i,j]            at  (x0 + (i+0.5)*dx,    z0 + (j+0.5)*dz  )
#   vx[i,j]             at  (x0 + (i+0.5)*dx,    z0 + j*dz        )
#   vz[i,j]             at  (x0 + i*dx,           z0 + (j+0.5)*dz  )
#
# Bilinear stencil
#   For a source at (x_s, z_s) and a field with grid-origin offset
#   (shift_x, shift_z), the fractional index into that field is:
#
#       fx = (x_s - x0 - shift_x) / dx
#       fz = (z_s - z0 - shift_z) / dz
#
#   The four surrounding nodes and their bilinear weights are:
#
#       node (i0,   j0  )  w00 = (1-wx)*(1-wz)  <- x-left,  z-index j0
#       node (i0+1, j0  )  w10 = wx*(1-wz)      <- x-right, z-index j0
#       node (i0,   j0+1)  w01 = (1-wx)*wz      <- x-left,  z-index j0+1
#       node (i0+1, j0+1)  w11 = wx*wz          <- x-right, z-index j0+1
#
#   where:
#
#       i0 = floor(fx)
#       j0 = floor(fz)
#       wx = fx - i0
#       wz = fz - j0
#
# Weight convention
#   Ordering is (w00, w10, w01, w11), stored as w[0..3].
#   This matches the receiver sampling convention in sampling.py.
#
# Invariant
#   sum(stencil.w) == 1.0
#   0 <= stencil.w[k] <= 1
#
# FWI note
#   For a fixed staggered field, bilinear scattering is the algebraic adjoint
#   of bilinear sampling if the same indices and weights are used:
#
#       sampling:   grid field -> off-grid value
#       scattering: off-grid residual/source -> grid field
#
#   Moment-tensor source spreading acts on stress grids (sxx, szz, sxz),
#   while DAS receiver sampling acts on velocity grids (vx, vz). They follow
#   the same interpolation/scatter principle, but they are not the same
#   physical operator and cannot be composed directly.
#
# Numba compatibility
#   BilinearStencil stores ix, iz, w as read-only numpy arrays.
#   Solvers should unpack and pass these as plain arrays to @njit kernels.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.grid import Grid2D


# ==============================================================================
# 1. STENCIL CONTAINER
# ==============================================================================

@dataclass(frozen=True)
class BilinearStencil:
    """
    Bilinear scattering stencil for a single staggered field.

    Describes how a unit source amplitude at a physical position is
    distributed onto four surrounding grid nodes with bilinear weights.

    Attributes
    ----------
    ix : np.ndarray, shape (4,), int64
        x-indices of the four surrounding grid nodes.
    iz : np.ndarray, shape (4,), int64
        z-indices of the four surrounding grid nodes.
    w : np.ndarray, shape (4,), float64
        Bilinear weights. They must be non-negative and sum to 1.

    Node ordering
    -------------
    k=0: (i0,   j0  )  w[0] = (1-wx)*(1-wz)  x-left,  z-index j0
    k=1: (i0+1, j0  )  w[1] = wx*(1-wz)      x-right, z-index j0
    k=2: (i0,   j0+1)  w[2] = (1-wx)*wz      x-left,  z-index j0+1
    k=3: (i0+1, j0+1)  w[3] = wx*wz          x-right, z-index j0+1

    Note
    ----
    z increases downward in this project. Therefore j0 and j0+1 are
    depth-index labels, not spatial "upper/lower" directions.
    """
    ix: np.ndarray
    iz: np.ndarray
    w: np.ndarray

    def __post_init__(self) -> None:
        for name, arr_in, dtype in [
            ("ix", self.ix, np.int64),
            ("iz", self.iz, np.int64),
            ("w", self.w, np.float64),
        ]:
            arr = np.array(arr_in, dtype=dtype, copy=True)

            if arr.shape != (4,):
                raise ValueError(
                    f"BilinearStencil.{name} must have shape (4,), got {arr.shape}."
                )

            arr.flags.writeable = False
            object.__setattr__(self, name, arr)

        wsum = float(self.w.sum())

        if abs(wsum - 1.0) > 1.0e-12:
            raise ValueError(
                f"BilinearStencil weights must sum to 1.0; got {wsum:.15e}."
            )

        if np.any(self.w < -1.0e-12):
            raise ValueError(
                f"BilinearStencil weights must be non-negative; got {self.w}."
            )

        if np.any(self.w > 1.0 + 1.0e-12):
            raise ValueError(
                f"BilinearStencil weights must be <= 1.0; got {self.w}."
            )

    def is_point_injection(self, tol: float = 1.0e-12) -> bool:
        """
        Return True if one weight equals 1 and the remaining weights are zero.
        """
        k = int(np.argmax(self.w))

        if abs(float(self.w[k]) - 1.0) > tol:
            return False

        mask = np.ones(4, dtype=bool)
        mask[k] = False

        return bool(np.all(np.abs(self.w[mask]) <= tol))

    def dominant_node(self) -> tuple[int, int]:
        """
        Return the (ix, iz) node with the largest weight.
        """
        k = int(np.argmax(self.w))
        return int(self.ix[k]), int(self.iz[k])

    def summary(self) -> str:
        labels = [
            "x-left  z-j0",
            "x-right z-j0",
            "x-left  z-j0+1",
            "x-right z-j0+1",
        ]

        lines = ["BilinearStencil:"]
        for k, label in enumerate(labels):
            lines.append(
                f"  k={k} ({label}): "
                f"ix={int(self.ix[k])}, iz={int(self.iz[k])}, w={float(self.w[k]):.6f}"
            )
        lines.append(f"  sum(w) = {float(self.w.sum()):.15e}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# ==============================================================================
# 2. GROUPED CONTAINERS
# ==============================================================================

@dataclass(frozen=True)
class StressSourceSpreading:
    """
    Bilinear source spreading for the three stress components.

    Each component has its own stencil because sxx/szz live on integer
    nodes while sxz lives on half-integer nodes.

    Attributes
    ----------
    sxx : BilinearStencil
        Stencil for sigma_xx on the normal-stress grid.
    szz : BilinearStencil
        Stencil for sigma_zz on the normal-stress grid.
    sxz : BilinearStencil
        Stencil for sigma_xz on the shear-stress grid.
    """
    sxx: BilinearStencil
    szz: BilinearStencil
    sxz: BilinearStencil

    def summary(self) -> str:
        return (
            "StressSourceSpreading:\n"
            f"  sxx:\n    {self.sxx.summary().replace(chr(10), chr(10) + '    ')}\n"
            f"  szz:\n    {self.szz.summary().replace(chr(10), chr(10) + '    ')}\n"
            f"  sxz:\n    {self.sxz.summary().replace(chr(10), chr(10) + '    ')}"
        )


@dataclass(frozen=True)
class ForceSourceSpreading:
    """
    Bilinear source spreading for the two velocity force components.

    Attributes
    ----------
    vx : BilinearStencil
        Stencil for an F_x body-force component on the vx grid.
    vz : BilinearStencil
        Stencil for an F_z body-force component on the vz grid.
    """
    vx: BilinearStencil
    vz: BilinearStencil

    def summary(self) -> str:
        return (
            "ForceSourceSpreading:\n"
            f"  vx:\n    {self.vx.summary().replace(chr(10), chr(10) + '    ')}\n"
            f"  vz:\n    {self.vz.summary().replace(chr(10), chr(10) + '    ')}"
        )


# ==============================================================================
# 3. LOW-LEVEL STENCIL BUILDER
# ==============================================================================

def _compute_bilinear_stencil(
    *,
    x_s: float,
    z_s: float,
    x0: float,
    z0: float,
    dx: float,
    dz: float,
    nx: int,
    nz: int,
    shift_x: float,
    shift_z: float,
    tol: float = 1.0e-9,
) -> BilinearStencil:
    """
    Compute a BilinearStencil for a physical source at (x_s, z_s) on a
    staggered sub-grid whose node (i, j) is at:

        x_node(i) = x0 + i*dx + shift_x
        z_node(j) = z0 + j*dz + shift_z

    The field is defined on an (nx, nz) array with indices:

        0 <= i <= nx-1
        0 <= j <= nz-1

    The bilinear stencil uses four neighbouring nodes. Therefore the base
    index must satisfy:

        0 <= i0 <= nx-2
        0 <= j0 <= nz-2

    If the physical point lies exactly on the last grid node, the base node
    is clamped to nx-2 or nz-2 and the corresponding weight becomes one on
    the boundary node.

    Parameters
    ----------
    x_s, z_s :
        Physical source position [m].
    x0, z0 :
        Grid origin [m].
    dx, dz :
        Grid spacing [m].
    nx, nz :
        Number of grid points in x and z.
    shift_x, shift_z :
        Staggered sub-grid offset [m].
    tol :
        Tolerance for boundary checks in fractional-index units.
    """
    if nx < 2 or nz < 2:
        raise ValueError(f"Need nx >= 2 and nz >= 2, got nx={nx}, nz={nz}.")

    if dx <= 0.0 or dz <= 0.0:
        raise ValueError(f"Grid spacing must be positive, got dx={dx}, dz={dz}.")

    if not np.isfinite(x_s) or not np.isfinite(z_s):
        raise ValueError(
            f"Source coordinates must be finite, got x_s={x_s}, z_s={z_s}."
        )

    if not np.isfinite(x0) or not np.isfinite(z0):
        raise ValueError(f"Grid origin must be finite, got x0={x0}, z0={z0}.")

    if not np.isfinite(shift_x) or not np.isfinite(shift_z):
        raise ValueError(
            f"Grid shifts must be finite, got shift_x={shift_x}, shift_z={shift_z}."
        )

    fx = (float(x_s) - float(x0) - float(shift_x)) / float(dx)
    fz = (float(z_s) - float(z0) - float(shift_z)) / float(dz)

    fx_min = -tol
    fx_max = (nx - 1) + tol
    fz_min = -tol
    fz_max = (nz - 1) + tol

    if not (fx_min <= fx <= fx_max):
        raise ValueError(
            f"Source x_s={x_s:.4f} m falls outside the bilinear range for "
            f"shift_x={shift_x:.4f} m: "
            f"x in [{x0 + shift_x:.4f}, {x0 + shift_x + (nx - 1) * dx:.4f}] m."
        )

    if not (fz_min <= fz <= fz_max):
        raise ValueError(
            f"Source z_s={z_s:.4f} m falls outside the bilinear range for "
            f"shift_z={shift_z:.4f} m: "
            f"z in [{z0 + shift_z:.4f}, {z0 + shift_z + (nz - 1) * dz:.4f}] m."
        )

    fx = float(np.clip(fx, 0.0, nx - 1.0))
    fz = float(np.clip(fz, 0.0, nz - 1.0))

    i0 = min(int(np.floor(fx)), nx - 2)
    j0 = min(int(np.floor(fz)), nz - 2)

    wx = fx - i0
    wz = fz - j0

    ix = np.array([i0, i0 + 1, i0, i0 + 1], dtype=np.int64)
    iz = np.array([j0, j0, j0 + 1, j0 + 1], dtype=np.int64)

    w = np.array(
        [
            (1.0 - wx) * (1.0 - wz),
            wx * (1.0 - wz),
            (1.0 - wx) * wz,
            wx * wz,
        ],
        dtype=np.float64,
    )

    return BilinearStencil(ix=ix, iz=iz, w=w)


# ==============================================================================
# 4. PUBLIC FACTORY FUNCTIONS
# ==============================================================================

def build_stress_source_spreading(
    grid: "Grid2D",
    x_s: float,
    z_s: float,
) -> StressSourceSpreading:
    """
    Build bilinear source spreading stencils for stress components.

    For a physical source at (x_s, z_s), returns stencils for:

    - sxx and szz on the normal-stress grid,
    - sxz on the shear-stress grid.

    Parameters
    ----------
    grid :
        Grid2D-like object with attributes x0, z0, dx, dz, nx, nz.
    x_s, z_s :
        Physical source coordinates [m].

    Returns
    -------
    StressSourceSpreading
        Stencils for sxx, szz, and sxz.

    Notes
    -----
    For an interior on-grid source exactly at a normal-stress node:

    - sxx/szz stencil: one weight = 1, others = 0,
    - sxz stencil: all four weights = 0.25 because the source is
      equidistant from four surrounding shear-stress nodes.

    This matches the intended legacy behaviour for a normal-stress-node source:
    point injection for sxx/szz and four-point equal-weight spreading for sxz.
    """
    x0 = float(grid.x0)
    z0 = float(grid.z0)
    dx = float(grid.dx)
    dz = float(grid.dz)
    nx = int(grid.nx)
    nz = int(grid.nz)

    stencil_normal = _compute_bilinear_stencil(
        x_s=x_s,
        z_s=z_s,
        x0=x0,
        z0=z0,
        dx=dx,
        dz=dz,
        nx=nx,
        nz=nz,
        shift_x=0.0,
        shift_z=0.0,
    )

    stencil_shear = _compute_bilinear_stencil(
        x_s=x_s,
        z_s=z_s,
        x0=x0,
        z0=z0,
        dx=dx,
        dz=dz,
        nx=nx,
        nz=nz,
        shift_x=0.5 * dx,
        shift_z=0.5 * dz,
    )

    return StressSourceSpreading(
        sxx=stencil_normal,
        szz=stencil_normal,
        sxz=stencil_shear,
    )


def build_force_source_spreading(
    grid: "Grid2D",
    x_s: float,
    z_s: float,
) -> ForceSourceSpreading:
    """
    Build bilinear source spreading stencils for velocity force components.

    Used for body-force sources, such as point-force validation sources.

    Parameters
    ----------
    grid :
        Grid2D-like object with attributes x0, z0, dx, dz, nx, nz.
    x_s, z_s :
        Physical source coordinates [m].

    Returns
    -------
    ForceSourceSpreading
        Stencils for vx and vz force injection.

    Notes
    -----
    - vx lives at (x0 + (i+0.5)*dx, z0 + j*dz).
    - vz lives at (x0 + i*dx, z0 + (j+0.5)*dz).
    """
    x0 = float(grid.x0)
    z0 = float(grid.z0)
    dx = float(grid.dx)
    dz = float(grid.dz)
    nx = int(grid.nx)
    nz = int(grid.nz)

    stencil_vx = _compute_bilinear_stencil(
        x_s=x_s,
        z_s=z_s,
        x0=x0,
        z0=z0,
        dx=dx,
        dz=dz,
        nx=nx,
        nz=nz,
        shift_x=0.5 * dx,
        shift_z=0.0,
    )

    stencil_vz = _compute_bilinear_stencil(
        x_s=x_s,
        z_s=z_s,
        x0=x0,
        z0=z0,
        dx=dx,
        dz=dz,
        nx=nx,
        nz=nz,
        shift_x=0.0,
        shift_z=0.5 * dz,
    )

    return ForceSourceSpreading(vx=stencil_vx, vz=stencil_vz)


# ==============================================================================
# 5. SELF-TEST
# ==============================================================================

def _self_test() -> None:
    from types import SimpleNamespace

    dx = 10.0
    dz = 10.0
    nx = 51
    nz = 51
    x0 = 0.0
    z0 = 0.0

    grid = SimpleNamespace(x0=x0, z0=z0, dx=dx, dz=dz, nx=nx, nz=nz)

    # --------------------------------------------------------------------------
    # 1. Interior on-grid source: sxx/szz point injection, sxz equal spreading
    # --------------------------------------------------------------------------
    x_on = x0 + 10 * dx
    z_on = z0 + 20 * dz

    ss = build_stress_source_spreading(grid, x_on, z_on)

    assert np.allclose(ss.sxx.w.sum(), 1.0, atol=1.0e-12)
    assert ss.sxx.is_point_injection()
    dominant_ix, dominant_iz = ss.sxx.dominant_node()
    assert dominant_ix == 10 and dominant_iz == 20, (
        f"sxx on-grid dominant node wrong: got ({dominant_ix}, {dominant_iz}), "
        "expected (10, 20)."
    )

    assert np.allclose(ss.sxz.w.sum(), 1.0, atol=1.0e-12)
    assert np.allclose(ss.sxz.w, 0.25, atol=1.0e-12)

    expected_ix_set = {9, 10}
    expected_iz_set = {19, 20}

    assert set(ss.sxz.ix.tolist()) == expected_ix_set
    assert set(ss.sxz.iz.tolist()) == expected_iz_set

    print("Interior on-grid stress source spreading: OK")

    # --------------------------------------------------------------------------
    # 2. Off-grid source at cell midpoint on normal-stress grid
    # --------------------------------------------------------------------------
    x_off = 125.0
    z_off = 175.0

    ss_off = build_stress_source_spreading(grid, x_off, z_off)

    assert np.allclose(ss_off.sxx.w.sum(), 1.0, atol=1.0e-12)
    assert np.allclose(ss_off.szz.w.sum(), 1.0, atol=1.0e-12)
    assert np.allclose(ss_off.sxz.w.sum(), 1.0, atol=1.0e-12)
    assert np.allclose(ss_off.sxx.w, 0.25, atol=1.0e-12)

    print("Off-grid midpoint normal-stress spreading: OK")

    # --------------------------------------------------------------------------
    # 3. General off-grid analytic check on normal-stress grid
    # --------------------------------------------------------------------------
    x_gen = 123.0
    z_gen = 178.0

    ss_gen = build_stress_source_spreading(grid, x_gen, z_gen)

    wx = 0.3
    wz = 0.8

    expected_w = np.array(
        [
            (1.0 - wx) * (1.0 - wz),
            wx * (1.0 - wz),
            (1.0 - wx) * wz,
            wx * wz,
        ],
        dtype=np.float64,
    )

    assert np.allclose(ss_gen.sxx.w, expected_w, atol=1.0e-12), (
        f"sxx analytic check failed: got {ss_gen.sxx.w}, expected {expected_w}."
    )
    assert int(ss_gen.sxx.ix[0]) == 12 and int(ss_gen.sxx.iz[0]) == 17

    print("General off-grid analytic stress-grid check: OK")

    # --------------------------------------------------------------------------
    # 4. Force source spreading: check vx/vz staggered origins
    # --------------------------------------------------------------------------
    fs = build_force_source_spreading(grid, 150.0, 200.0)

    assert np.allclose(fs.vx.w.sum(), 1.0, atol=1.0e-12)
    assert np.allclose(fs.vz.w.sum(), 1.0, atol=1.0e-12)

    assert np.allclose(fs.vx.w[:2], 0.5, atol=1.0e-12)
    assert np.allclose(fs.vx.w[2:], 0.0, atol=1.0e-12)

    assert np.allclose(fs.vz.w[0], 0.5, atol=1.0e-12)
    assert np.allclose(fs.vz.w[1], 0.0, atol=1.0e-12)
    assert np.allclose(fs.vz.w[2], 0.5, atol=1.0e-12)
    assert np.allclose(fs.vz.w[3], 0.0, atol=1.0e-12)

    print("Force source spreading staggered-origin check: OK")

    # --------------------------------------------------------------------------
    # 5. Read-only arrays
    # --------------------------------------------------------------------------
    try:
        ss.sxx.w[0] = 999.0
        raise AssertionError("w should be read-only.")
    except ValueError:
        pass

    try:
        ss.sxx.ix[0] = 999
        raise AssertionError("ix should be read-only.")
    except ValueError:
        pass

    print("Read-only array check: OK")

    # --------------------------------------------------------------------------
    # 6. Out-of-domain sources raise ValueError
    # --------------------------------------------------------------------------
    try:
        build_stress_source_spreading(grid, -100.0, 200.0)
        raise AssertionError("Expected ValueError for out-of-domain x source.")
    except ValueError:
        pass

    try:
        build_stress_source_spreading(grid, 200.0, -100.0)
        raise AssertionError("Expected ValueError for out-of-domain z source.")
    except ValueError:
        pass

    print("Out-of-domain source check: OK")

    # --------------------------------------------------------------------------
    # 7. Random source positions: weights are non-negative and sum to one
    # --------------------------------------------------------------------------
    rng = np.random.default_rng(42)

    for _ in range(100):
        x_rand = rng.uniform(x0 + 0.5 * dx, x0 + (nx - 1.5) * dx)
        z_rand = rng.uniform(z0 + 0.5 * dz, z0 + (nz - 1.5) * dz)

        ss_rand = build_stress_source_spreading(grid, x_rand, z_rand)

        for stencil in [ss_rand.sxx, ss_rand.szz, ss_rand.sxz]:
            assert np.all(stencil.w >= -1.0e-15), f"Negative weight: {stencil.w}"
            assert np.allclose(stencil.w.sum(), 1.0, atol=1.0e-12)

    print("Random source-position check: OK")

    # --------------------------------------------------------------------------
    # 8. Bilinear exactness for a linear field on the normal-stress grid
    # --------------------------------------------------------------------------
    a_coef = 3.7
    b_coef = -1.2

    sxx_field = np.add.outer(
        a_coef * np.arange(nx, dtype=np.float64),
        b_coef * np.arange(nz, dtype=np.float64),
    )

    x_test = 123.0
    z_test = 178.0

    ss_test = build_stress_source_spreading(grid, x_test, z_test)

    interpolated = float(
        np.sum(ss_test.sxx.w * sxx_field[ss_test.sxx.ix, ss_test.sxx.iz])
    )

    expected_val = a_coef * (x_test / dx) + b_coef * (z_test / dz)

    assert abs(interpolated - expected_val) < 1.0e-10, (
        f"Linear-field recovery failed: got {interpolated:.6e}, "
        f"expected {expected_val:.6e}."
    )

    print("Linear-field bilinear exactness check: OK")

    # --------------------------------------------------------------------------
    # 9. Invalid grid/source inputs
    # --------------------------------------------------------------------------
    try:
        _compute_bilinear_stencil(
            x_s=0.0,
            z_s=0.0,
            x0=0.0,
            z0=0.0,
            dx=-1.0,
            dz=1.0,
            nx=10,
            nz=10,
            shift_x=0.0,
            shift_z=0.0,
        )
        raise AssertionError("Expected ValueError for negative dx.")
    except ValueError:
        pass

    try:
        _compute_bilinear_stencil(
            x_s=np.nan,
            z_s=0.0,
            x0=0.0,
            z0=0.0,
            dx=1.0,
            dz=1.0,
            nx=10,
            nz=10,
            shift_x=0.0,
            shift_z=0.0,
        )
        raise AssertionError("Expected ValueError for non-finite x_s.")
    except ValueError:
        pass

    print("Invalid input checks: OK")

    print("\n✓ source_spreading.py: all tests passed")


if __name__ == "__main__":
    _self_test()