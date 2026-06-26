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
# Staggered node positions (using grid.py conventions)
#   sxx[i,j], szz[i,j]  at  (x0 + i*dx,         z0 + j*dz        )
#   sxz[i,j]            at  (x0 + (i+0.5)*dx,    z0 + (j+0.5)*dz  )
#   vx[i,j]             at  (x0 + (i+0.5)*dx,    z0 + j*dz        )
#   vz[i,j]             at  (x0 + i*dx,           z0 + (j+0.5)*dz  )
#
# Bilinear stencil (operator F: physical position → grid)
#   For a source at (x_s, z_s) and a field with grid origin offset
#   (shift_x, shift_z), the fractional index into the field is:
#
#       fx = (x_s - x0 - shift_x) / dx
#       fz = (z_s - z0 - shift_z) / dz
#
#   The four surrounding nodes and their bilinear weights are:
#
#       node (i0,   j0  )  w00 = (1-wx)*(1-wz)  ← lower-left
#       node (i0+1, j0  )  w10 = wx*(1-wz)       ← lower-right
#       node (i0,   j0+1)  w01 = (1-wx)*wz       ← upper-left
#       node (i0+1, j0+1)  w11 = wx*wz            ← upper-right
#
#   where  i0 = floor(fx),  j0 = floor(fz),  wx = fx - i0,  wz = fz - j0.
#
# Weight convention
#   Ordering is (w00, w10, w01, w11) stored as w[0..3] in BilinearStencil.
#   This matches the (w00, w10, w01, w11) ordering in sampling.py so that
#   forward (source spreading) and adjoint (receiver sampling) operators
#   share the same stencil arithmetic.
#
# Invariant
#   sum(stencil.w) == 1.0  for every stencil, always.
#
# FWI note
#   For a fixed staggered field, bilinear scattering is the algebraic adjoint
#   of bilinear sampling if the same indices and weights are used:
#
#       sampling:   grid field   → off-grid value      (sampling.py)
#       scattering: off-grid residual/source → grid field   (this file)
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
        Bilinear weights (sum to 1.0).

    Node ordering
    -------------
    k=0: (i0,   j0  )  w[0] = (1-wx)*(1-wz)  x-left,  z-index j0
    k=1: (i0+1, j0  )  w[1] = wx*(1-wz)       x-right, z-index j0
    k=2: (i0,   j0+1)  w[2] = (1-wx)*wz       x-left,  z-index j0+1
    k=3: (i0+1, j0+1)  w[3] = wx*wz            x-right, z-index j0+1

    Note: z increases downward (depth). j0 and j0+1 are depth-index labels,
    not spatial "upper/lower" directions.
    """
    ix: np.ndarray  # shape (4,), int64, read-only
    iz: np.ndarray  # shape (4,), int64, read-only
    w:  np.ndarray  # shape (4,), float64, read-only

    def __post_init__(self) -> None:
        for name, arr_in, dtype in [
            ("ix", self.ix, np.int64),
            ("iz", self.iz, np.int64),
            ("w",  self.w,  np.float64),
        ]:
            arr = np.array(arr_in, dtype=dtype, copy=True)
            if arr.shape != (4,):
                raise ValueError(
                    f"BilinearStencil.{name} must have shape (4,), got {arr.shape}."
                )
            arr.flags.writeable = False
            object.__setattr__(self, name, arr)

        wsum = float(self.w.sum())
        if abs(wsum - 1.0) > 1e-12:
            raise ValueError(
                f"BilinearStencil weights must sum to 1.0; got {wsum:.15e}."
            )
        if np.any(self.w < -1e-12):
            raise ValueError(
                f"BilinearStencil weights must be non-negative; got {self.w}."
            )
        if np.any(self.w > 1.0 + 1e-12):
            raise ValueError(
                f"BilinearStencil weights must be <= 1.0; got {self.w}."
            )

    def is_point_injection(self, tol: float = 1e-12) -> bool:
        """
        Return True if one weight equals 1 and the rest are zero.

        Correct because __post_init__ guarantees w >= 0 and sum(w) = 1.
        If any w[k] > 1-tol, then the remaining weights sum to < tol,
        so they are all approximately zero.
        """
        return bool(np.any(self.w > 1.0 - tol))

    def dominant_node(self) -> tuple[int, int]:
        """Return (ix, iz) of the node with the largest weight."""
        k = int(np.argmax(self.w))
        return int(self.ix[k]), int(self.iz[k])

    def summary(self) -> str:
        lines = ["BilinearStencil:"]
        labels = ["x-left  z-j0", "x-right z-j0", "x-left  z-j0+1", "x-right z-j0+1"]
        for k, lbl in enumerate(labels):
            lines.append(
                f"  k={k} ({lbl}): "
                f"ix={self.ix[k]}, iz={self.iz[k]}, w={self.w[k]:.6f}"
            )
        lines.append(f"  sum(w) = {self.w.sum():.15e}")
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
    nodes while sxz lives on half-integer nodes (different staggered origin).

    Attributes
    ----------
    sxx : BilinearStencil
        Stencil for the sigma_xx component (integer grid).
    szz : BilinearStencil
        Stencil for the sigma_zz component (integer grid; same as sxx).
    sxz : BilinearStencil
        Stencil for the sigma_xz component (half-integer grid).
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

    vx and vz live on different half-integer sub-grids.

    Attributes
    ----------
    vx : BilinearStencil
        Stencil for the F_x body-force component (half-x, integer-z grid).
    vz : BilinearStencil
        Stencil for the F_z body-force component (integer-x, half-z grid).
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
    tol: float = 1e-9,
) -> BilinearStencil:
    """
    Compute a BilinearStencil for a physical source at (x_s, z_s) on a
    staggered sub-grid whose node (i, j) is at:

        x_node(i) = x0 + i*dx + shift_x
        z_node(j) = z0 + j*dz + shift_z

    The field is defined on an (nx, nz) array (indices 0..nx-1, 0..nz-1).
    The bilinear stencil uses the four nearest nodes; the base index
    (i0, j0) must satisfy 0 <= i0 <= nx-2 and 0 <= j0 <= nz-2.

    Parameters
    ----------
    x_s, z_s :
        Physical source position [m].
    x0, z0 :
        Grid origin [m] (from grid.x0, grid.z0).
    dx, dz :
        Grid spacing [m].
    nx, nz :
        Number of grid points in x and z.
    shift_x, shift_z :
        Staggered sub-grid offset [m]:
          - sxx/szz: shift_x=0,      shift_z=0
          - sxz:     shift_x=0.5*dx, shift_z=0.5*dz
          - vx:      shift_x=0.5*dx, shift_z=0
          - vz:      shift_x=0,      shift_z=0.5*dz
    tol :
        Fractional tolerance for boundary check.
    """
    # Fractional index of source in this sub-grid's coordinate system
    fx = (x_s - x0 - shift_x) / dx
    fz = (z_s - z0 - shift_z) / dz

    # Domain check: the source must lie within the field's bilinear range
    fx_min = -tol
    fx_max = (nx - 1) + tol
    fz_min = -tol
    fz_max = (nz - 1) + tol

    if not (fx_min <= fx <= fx_max):
        raise ValueError(
            f"Source x_s={x_s:.4f} m falls outside the bilinear range for "
            f"shift_x={shift_x:.2f} m: "
            f"x in [{x0 + shift_x:.2f}, {x0 + shift_x + (nx-1)*dx:.2f}] m."
        )
    if not (fz_min <= fz <= fz_max):
        raise ValueError(
            f"Source z_s={z_s:.4f} m falls outside the bilinear range for "
            f"shift_z={shift_z:.2f} m: "
            f"z in [{z0 + shift_z:.2f}, {z0 + shift_z + (nz-1)*dz:.2f}] m."
        )

    # Clamp to valid stencil base range [0, n-2]
    fx = float(np.clip(fx, 0.0, nx - 1.0))
    fz = float(np.clip(fz, 0.0, nz - 1.0))

    i0 = min(int(np.floor(fx)), nx - 2)
    j0 = min(int(np.floor(fz)), nz - 2)

    wx = fx - i0   # fractional part in x: 0 <= wx <= 1
    wz = fz - j0   # fractional part in z: 0 <= wz <= 1

    # Four surrounding nodes in standard order:
    #   k=0: (i0,   j0  )  x-left,  z-index j0   w00 = (1-wx)*(1-wz)
    #   k=1: (i0+1, j0  )  x-right, z-index j0   w10 = wx*(1-wz)
    #   k=2: (i0,   j0+1)  x-left,  z-index j0+1 w01 = (1-wx)*wz
    #   k=3: (i0+1, j0+1)  x-right, z-index j0+1 w11 = wx*wz
    ix = np.array([i0,     i0 + 1, i0,     i0 + 1], dtype=np.int64)
    iz = np.array([j0,     j0,     j0 + 1, j0 + 1], dtype=np.int64)
    w  = np.array(
        [(1.0 - wx) * (1.0 - wz),
         wx         * (1.0 - wz),
         (1.0 - wx) * wz,
         wx         * wz],
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
    Build bilinear source spreading stencils for all three stress components.

    For a physical source at (x_s, z_s), returns the BilinearStencil for:
      - sxx and szz (live on the integer stress grid, same stencil)
      - sxz (lives on the half-integer shear grid, different stencil)

    Parameters
    ----------
    grid :
        Grid2D-like object with attributes x0, z0, dx, dz, nx, nz.
    x_s, z_s :
        Physical source coordinates [m].

    Returns
    -------
    StressSourceSpreading
        Contains stencils for sxx, szz, sxz.

    Notes
    -----
    For an on-grid source exactly at a normal-stress node (i*dx, j*dz):
      - sxx/szz stencil: one weight = 1, others = 0.
      - sxz stencil:     all four weights = 0.25 (source equidistant from
        four surrounding shear nodes).
    This reproduces the current nearest-node injection for sxx/szz and the
    four-point equal-weight injection for sxz.
    """
    x0, z0 = float(grid.x0), float(grid.z0)
    dx, dz = float(grid.dx), float(grid.dz)
    nx, nz = int(grid.nx), int(grid.nz)

    # sxx and szz are on the same integer grid (no shift)
    stencil_normal = _compute_bilinear_stencil(
        x_s=x_s, z_s=z_s,
        x0=x0, z0=z0,
        dx=dx, dz=dz,
        nx=nx, nz=nz,
        shift_x=0.0,
        shift_z=0.0,
    )

    # sxz lives on the half-integer grid (shifted by 0.5*dx and 0.5*dz)
    stencil_shear = _compute_bilinear_stencil(
        x_s=x_s, z_s=z_s,
        x0=x0, z0=z0,
        dx=dx, dz=dz,
        nx=nx, nz=nz,
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
    Build bilinear source spreading stencils for the two velocity components.

    Used for body-force sources (e.g., point-force validation solver).

    Parameters
    ----------
    grid :
        Grid2D-like object with attributes x0, z0, dx, dz, nx, nz.
    x_s, z_s :
        Physical source coordinates [m].

    Returns
    -------
    ForceSourceSpreading
        Contains stencils for vx and vz.

    Notes
    -----
    - vx lives at (x0 + (i+0.5)*dx, z0 + j*dz):   shift_x=0.5*dx, shift_z=0
    - vz lives at (x0 + i*dx,       z0 + (j+0.5)*dz): shift_x=0,    shift_z=0.5*dz
    """
    x0, z0 = float(grid.x0), float(grid.z0)
    dx, dz = float(grid.dx), float(grid.dz)
    nx, nz = int(grid.nx), int(grid.nz)

    stencil_vx = _compute_bilinear_stencil(
        x_s=x_s, z_s=z_s,
        x0=x0, z0=z0,
        dx=dx, dz=dz,
        nx=nx, nz=nz,
        shift_x=0.5 * dx,
        shift_z=0.0,
    )

    stencil_vz = _compute_bilinear_stencil(
        x_s=x_s, z_s=z_s,
        x0=x0, z0=z0,
        dx=dx, dz=dz,
        nx=nx, nz=nz,
        shift_x=0.0,
        shift_z=0.5 * dz,
    )

    return ForceSourceSpreading(vx=stencil_vx, vz=stencil_vz)


# ==============================================================================
# 5. SELF-TEST
# ==============================================================================

def _self_test() -> None:
    from types import SimpleNamespace

    dx, dz = 10.0, 10.0
    nx, nz = 51, 51
    x0, z0 = 0.0, 0.0
    grid = SimpleNamespace(x0=x0, z0=z0, dx=dx, dz=dz, nx=nx, nz=nz)

    # ── 1. On-grid source: sxx/szz → point injection ──────────────────────────
    # Source exactly at integer node (ix=10, iz=20) → (100.0, 200.0) m
    x_on = x0 + 10 * dx   # 100.0 m
    z_on = z0 + 20 * dz   # 200.0 m

    ss = build_stress_source_spreading(grid, x_on, z_on)

    # sxx/szz: one weight = 1, others = 0
    assert np.allclose(ss.sxx.w.sum(), 1.0, atol=1e-12), "sxx weights don't sum to 1"
    assert np.max(ss.sxx.w) > 1.0 - 1e-12, "sxx on-grid: expected dominant weight = 1"
    dominant_ix, dominant_iz = ss.sxx.dominant_node()
    assert dominant_ix == 10 and dominant_iz == 20, (
        f"sxx on-grid dominant node wrong: got ({dominant_ix},{dominant_iz}), "
        f"expected (10, 20)"
    )
    print("On-grid sxx/szz (point injection): OK")

    # sxz: source at (100.0, 200.0) is equidistant from 4 sxz nodes
    # sxz nodes: (9,19),(10,19),(9,20),(10,20) all at 0.5*dx distance
    assert np.allclose(ss.sxz.w.sum(), 1.0, atol=1e-12), "sxz weights don't sum to 1"
    assert np.allclose(ss.sxz.w, 0.25, atol=1e-12), (
        f"sxz on-grid: expected all weights = 0.25, got {ss.sxz.w}"
    )
    # Verify the 4 nodes are the correct ones surrounding (ix=10, iz=20)
    # sxz[i,j] at ((i+0.5)*dx, (j+0.5)*dz)
    # Surrounding (100.0, 200.0): nodes with i=9,10 and j=19,20
    expected_ix_set = {9, 10}
    expected_iz_set = {19, 20}
    assert set(ss.sxz.ix.tolist()) == expected_ix_set, (
        f"sxz ix nodes wrong: got {ss.sxz.ix.tolist()}"
    )
    assert set(ss.sxz.iz.tolist()) == expected_iz_set, (
        f"sxz iz nodes wrong: got {ss.sxz.iz.tolist()}"
    )
    print("On-grid sxz (4-node equal weight, correct nodes): OK")

    # ── 2. Off-grid source: bilinear weights ──────────────────────────────────
    # Source at (125.0, 175.0) m — between nodes
    x_off = 125.0   # between ix=12 (120.0) and ix=13 (130.0) → wx = 0.5
    z_off = 175.0   # between iz=17 (170.0) and iz=18 (180.0) → wz = 0.5

    ss_off = build_stress_source_spreading(grid, x_off, z_off)

    assert np.allclose(ss_off.sxx.w.sum(), 1.0, atol=1e-12), "off-grid sxx weights sum"
    assert np.allclose(ss_off.szz.w.sum(), 1.0, atol=1e-12), "off-grid szz weights sum"
    assert np.allclose(ss_off.sxz.w.sum(), 1.0, atol=1e-12), "off-grid sxz weights sum"

    # wx = (125 - 0) / 10 - 12 = 0.5, wz = (175 - 0) / 10 - 17 = 0.5
    # weights: all 0.25
    assert np.allclose(ss_off.sxx.w, 0.25, atol=1e-12), (
        f"sxx off-grid (midpoint) expected all 0.25, got {ss_off.sxx.w}"
    )
    print("Off-grid sxx/szz (midpoint → 0.25 each): OK")

    # ── 3. General off-grid: analytic check ───────────────────────────────────
    # Source at (123.0, 178.0): wx = 0.3, wz = 0.8 on integer grid
    x_gen = 123.0   # ix_frac = 12.3 → i0=12, wx=0.3
    z_gen = 178.0   # iz_frac = 17.8 → j0=17, wz=0.8

    ss_gen = build_stress_source_spreading(grid, x_gen, z_gen)
    wx, wz = 0.3, 0.8

    expected_w = np.array([
        (1-wx)*(1-wz),   # w00 = 0.14
        wx*(1-wz),        # w10 = 0.06
        (1-wx)*wz,        # w01 = 0.56
        wx*wz,            # w11 = 0.24
    ])
    assert np.allclose(ss_gen.sxx.w, expected_w, atol=1e-12), (
        f"sxx analytic check failed: got {ss_gen.sxx.w}, expected {expected_w}"
    )
    assert ss_gen.sxx.ix[0] == 12 and ss_gen.sxx.iz[0] == 17, (
        f"base node wrong: got ({ss_gen.sxx.ix[0]}, {ss_gen.sxx.iz[0]})"
    )
    print("Off-grid sxx analytic weight check (wx=0.3, wz=0.8): OK")

    # ── 4. Force source spreading: correct staggered origins ──────────────────
    # vx lives at (x0 + (i+0.5)*dx, z0 + j*dz)
    # vz lives at (x0 + i*dx,       z0 + (j+0.5)*dz)
    # For source at (150.0, 200.0):
    #   vx:  fx = (150 - 5) / 10 = 14.5 → i0=14, wx=0.5
    #         fz = (200 - 0) / 10 = 20.0 → j0=20, wz=0.0
    #         weights: (0.5, 0.5, 0, 0) at nodes (14,20),(15,20),(14,21),(15,21)
    #   vz:  fx = (150 - 0) / 10 = 15.0 → i0=15, wx=0.0
    #         fz = (200 - 5) / 10 = 19.5 → j0=19, wz=0.5
    #         weights: (0.5, 0, 0.5, 0) at nodes (15,19),(16,19),(15,20),(16,20)
    fs = build_force_source_spreading(grid, 150.0, 200.0)

    assert np.allclose(fs.vx.w.sum(), 1.0, atol=1e-12)
    assert np.allclose(fs.vz.w.sum(), 1.0, atol=1e-12)

    # vx: source mid-way between i=14 and i=15 in x, on-grid in z
    assert np.allclose(fs.vx.w[:2], 0.5, atol=1e-12), (
        f"vx spreading: expected w[0]=w[1]=0.5, got {fs.vx.w}"
    )
    assert np.allclose(fs.vx.w[2:], 0.0, atol=1e-12), (
        f"vx spreading: expected w[2]=w[3]=0.0, got {fs.vx.w}"
    )

    # vz: source on-grid in x, mid-way between j=19 and j=20 in z
    assert np.allclose(fs.vz.w[0], 0.5, atol=1e-12), f"vz w[0] wrong: {fs.vz.w}"
    assert np.allclose(fs.vz.w[1], 0.0, atol=1e-12), f"vz w[1] wrong: {fs.vz.w}"
    assert np.allclose(fs.vz.w[2], 0.5, atol=1e-12), f"vz w[2] wrong: {fs.vz.w}"
    assert np.allclose(fs.vz.w[3], 0.0, atol=1e-12), f"vz w[3] wrong: {fs.vz.w}"
    print("Force source spreading (vx, vz): staggered origins correct OK")

    # ── 5. Read-only arrays ────────────────────────────────────────────────────
    try:
        ss.sxx.w[0] = 999.0
        raise AssertionError("w should be read-only")
    except ValueError:
        pass
    try:
        ss.sxx.ix[0] = 999
        raise AssertionError("ix should be read-only")
    except ValueError:
        pass
    print("Read-only arrays: OK")

    # ── 6. Out-of-domain source raises ValueError ──────────────────────────────
    try:
        build_stress_source_spreading(grid, -100.0, 200.0)
        raise AssertionError("Expected ValueError for out-of-domain source in x")
    except ValueError:
        pass
    try:
        build_stress_source_spreading(grid, 200.0, -100.0)
        raise AssertionError("Expected ValueError for out-of-domain source in z")
    except ValueError:
        pass
    print("Out-of-domain ValueError: OK")

    # ── 7. All weights non-negative ────────────────────────────────────────────
    rng = np.random.default_rng(42)
    for _ in range(100):
        x_rand = rng.uniform(x0 + 0.5 * dx, x0 + (nx - 1.5) * dx)
        z_rand = rng.uniform(z0 + 0.5 * dz, z0 + (nz - 1.5) * dz)
        ss_r = build_stress_source_spreading(grid, x_rand, z_rand)
        for stencil in [ss_r.sxx, ss_r.szz, ss_r.sxz]:
            assert np.all(stencil.w >= -1e-15), f"Negative weight: {stencil.w}"
            assert np.allclose(stencil.w.sum(), 1.0, atol=1e-12), (
                f"Weights don't sum to 1: {stencil.w.sum()}"
            )
    print("Random source positions (100): all weights >= 0 and sum to 1: OK")

    # ── 8. Weight conservation under linear field ──────────────────────────────
    # If sxx = a*i + b*j (linear), then spreading at (x_s, z_s) should
    # exactly recover the linearly interpolated value:
    #   Σ_k w_k * sxx(ix_k, iz_k) = a*(x_s/dx) + b*(z_s/dz)
    a_coef, b_coef = 3.7, -1.2
    sxx_field = np.add.outer(
        a_coef * np.arange(nx, dtype=np.float64),
        b_coef * np.arange(nz, dtype=np.float64),
    )
    x_test, z_test = 123.0, 178.0
    ss_t = build_stress_source_spreading(grid, x_test, z_test)
    interpolated = float(
        np.sum(ss_t.sxx.w * sxx_field[ss_t.sxx.ix, ss_t.sxx.iz])
    )
    expected_val = a_coef * (x_test / dx) + b_coef * (z_test / dz)
    assert abs(interpolated - expected_val) < 1e-10, (
        f"Linear field interpolation: got {interpolated:.6e}, expected {expected_val:.6e}"
    )
    print("Linear field recovery (bilinear exactness): OK")

    print("\n✓ source_spreading.py: All tests passed")


if __name__ == "__main__":
    _self_test()