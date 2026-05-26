# ==============================================================================
# src/sampling.py — staggered-grid receiver sampling for elastic wavefields
#
# Purpose
#   Precompute bilinear interpolation indices and weights for extracting
#   staggered-grid velocity fields (vx, vz) at physical receiver coordinates.
#
# Design
#   - Receivers2D provides physical receiver coordinates (x, z)
#   - Grid2D provides base coordinates and spacing
#   - This module precomputes sampling metadata once
#   - solver_numpy.py can then use only raw NumPy arrays inside the time loop
#
# Staggered locations
#   vx lives at: (x0 + (i + 0.5) dx, z0 + j dz)
#   vz lives at: (x0 + i dx,         z0 + (j + 0.5) dz)
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np

from src.receivers import Receivers2D

if TYPE_CHECKING:
    from src.grid import Grid2D


# ==============================================================================
# 1. WEIGHT CONTAINER FOR A SINGLE FIELD
# ==============================================================================

@dataclass(frozen=True)
class BilinearSampling2D:
    """
    Precomputed bilinear interpolation weights for a single staggered field.
    All arrays are of shape (nrec,) and are strictly read-only.
    """
    ix:  np.ndarray
    iz:  np.ndarray
    w00: np.ndarray
    w10: np.ndarray
    w01: np.ndarray
    w11: np.ndarray

    def __post_init__(self) -> None:
        ref = None
        for name, dtype in [("ix",  np.int64),   ("iz",  np.int64),
                            ("w00", np.float64), ("w10", np.float64),
                            ("w01", np.float64), ("w11", np.float64)]:
            arr = np.array(getattr(self, name), dtype=dtype, copy=True)
            if arr.ndim != 1:
                raise ValueError(f"'{name}' must be 1D, got shape {arr.shape}.")
            if ref is None:
                ref = arr.size
            elif arr.size != ref:
                raise ValueError(
                    f"All BilinearSampling2D arrays must be the same size; "
                    f"'{name}' has {arr.size}, expected {ref}."
                )
            arr.flags.writeable = False
            object.__setattr__(self, name, arr)

        wsum = self.w00 + self.w10 + self.w01 + self.w11
        if not np.allclose(wsum, 1.0, atol=1e-12):
            raise ValueError(
                f"Weights must sum to 1. max|sum-1| = {np.abs(wsum - 1.0).max():.3e}"
            )

    @property
    def nrec(self) -> int:
        return int(self.ix.size)


# ==============================================================================
# 2. COMBINED CONTAINER FOR VX AND VZ
# ==============================================================================

@dataclass(frozen=True)
class ReceiverSampling2D:
    """
    Precomputed staggered interpolation metadata for both vx and vz.
    Created once before the time loop via build_receiver_sampling().
    """
    vx: BilinearSampling2D
    vz: BilinearSampling2D

    def __post_init__(self) -> None:
        if self.vx.nrec != self.vz.nrec:
            raise ValueError(
                f"vx and vz samplers must have identical nrec; "
                f"got {self.vx.nrec} vs {self.vz.nrec}."
            )

    @property
    def nrec(self) -> int:
        return self.vx.nrec


# ==============================================================================
# 3. INTERNAL WEIGHT BUILDER
# ==============================================================================

def _build_bilinear_sampler(
    x_phys: np.ndarray,
    z_phys: np.ndarray,
    *,
    x_origin: float,
    z_origin: float,
    dx: float,
    dz: float,
    nx: int,
    nz: int,
) -> BilinearSampling2D:
    """
    Builds bilinear weights for a field whose origin (0,0) node is at (x_origin, z_origin).
    """
    fx = (np.asarray(x_phys, dtype=np.float64) - x_origin) / dx
    fz = (np.asarray(z_phys, dtype=np.float64) - z_origin) / dz
    tol = 1e-12

    if np.any(fx < -tol) or np.any(fx > (nx - 1) + tol):
        bad = np.where((fx < -tol) | (fx > (nx - 1) + tol))[0][:5]
        raise ValueError(
            f"Some receiver x positions fall outside bilinear interpolation range "
            f"[{x_origin:.2f}, {x_origin + (nx-1)*dx:.2f}] m. "
            f"First bad indices: {bad.tolist()}"
        )

    if np.any(fz < -tol) or np.any(fz > (nz - 1) + tol):
        bad = np.where((fz < -tol) | (fz > (nz - 1) + tol))[0][:5]
        raise ValueError(
            f"Some receiver z positions fall outside bilinear interpolation range "
            f"[{z_origin:.2f}, {z_origin + (nz-1)*dz:.2f}] m. "
            f"First bad indices: {bad.tolist()}"
        )

    # Safely handle roundoff near the upper edge
    fx = np.minimum(np.maximum(fx, 0.0), nx - 1.0)
    fz = np.minimum(np.maximum(fz, 0.0), nz - 1.0)

    ix = np.minimum(np.floor(fx).astype(np.int64), nx - 2)
    iz = np.minimum(np.floor(fz).astype(np.int64), nz - 2)

    wx1 = fx - ix
    wz1 = fz - iz
    wx0 = 1.0 - wx1
    wz0 = 1.0 - wz1

    w00 = wx0 * wz0
    w10 = wx1 * wz0
    w01 = wx0 * wz1
    w11 = wx1 * wz1

    return BilinearSampling2D(
        ix=ix,
        iz=iz,
        w00=w00,
        w10=w10,
        w01=w01,
        w11=w11,
    )


# ==============================================================================
# 4. PUBLIC FACTORY
# ==============================================================================

def build_receiver_sampling(
    grid: "Grid2D",
    receivers: Receivers2D,
) -> ReceiverSampling2D:
    """
    Builds precomputed interpolation weights for all receivers.
    """
    if receivers.nrec == 0:
        raise ValueError("Cannot build sampling metadata for 0 receivers.")

    for attr in ("x0", "z0", "dx", "dz", "nx", "nz"):
        if not hasattr(grid, attr):
            raise TypeError(f"'grid' must have attribute '{attr}'.")
    if grid.dx <= 0.0 or grid.dz <= 0.0:
        raise ValueError("grid.dx and grid.dz must be > 0.")
    if grid.nx < 2 or grid.nz < 2:
        raise ValueError("grid.nx and grid.nz must be >= 2.")

    # vx staggered grid: shifted by +0.5 dx in x
    vx_sampler = _build_bilinear_sampler(
        receivers.x,
        receivers.z,
        x_origin=grid.x0 + 0.5 * grid.dx,
        z_origin=grid.z0,
        dx=grid.dx,
        dz=grid.dz,
        nx=grid.nx,
        nz=grid.nz,
    )

    # vz staggered grid: shifted by +0.5 dz in z
    vz_sampler = _build_bilinear_sampler(
        receivers.x,
        receivers.z,
        x_origin=grid.x0,
        z_origin=grid.z0 + 0.5 * grid.dz,
        dx=grid.dx,
        dz=grid.dz,
        nx=grid.nx,
        nz=grid.nz,
    )

    return ReceiverSampling2D(vx=vx_sampler, vz=vz_sampler)


# ==============================================================================
# 5. EXTRACTION CORE (Called in hot loop)
# ==============================================================================

def _interp_field(f: np.ndarray, s: BilinearSampling2D) -> np.ndarray:
    """Fast inline bilinear extraction using precomputed arrays (no branching)."""
    return (
        f[s.ix,     s.iz    ] * s.w00 +
        f[s.ix + 1, s.iz    ] * s.w10 +
        f[s.ix,     s.iz + 1] * s.w01 +
        f[s.ix + 1, s.iz + 1] * s.w11
    )

def sample_receivers(
    vx_field: np.ndarray,
    vz_field: np.ndarray,
    sampling: ReceiverSampling2D,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts vx and vz at receiver coordinates via strict bilinear interpolation.
    To be called strictly inside the time loop.
    """
    # Defensive programming: minimal fast check in the outer function
    if vx_field.ndim != 2 or vz_field.ndim != 2:
        raise ValueError("vx_field and vz_field must be 2D arrays.")

    return _interp_field(vx_field, sampling.vx), _interp_field(vz_field, sampling.vz)


# ==============================================================================
# 6. SELF-TEST
# ==============================================================================

def _self_test() -> None:
    from types import SimpleNamespace

    g = SimpleNamespace(x0=0.0, z0=0.0, dx=10.0, dz=10.0, nx=21, nz=21)

    x = np.array([15.0, 55.0, 95.0], dtype=np.float64)
    z = np.array([20.0, 60.0, 100.0], dtype=np.float64)
    nrec = x.size

    rec = Receivers2D(
        x=x,
        z=z,
        ix=np.zeros(nrec, dtype=int),
        iz=np.zeros(nrec, dtype=int),
        tx=np.ones(nrec, dtype=np.float64),
        tz=np.zeros(nrec, dtype=np.float64),
        s=np.arange(nrec, dtype=np.float64),
    )

    sampling = build_receiver_sampling(g, rec)

    # 1. Constant field -> exact recovery
    vx_c = np.ones((g.nx, g.nz), dtype=np.float64) * 7.5
    vz_c = np.ones((g.nx, g.nz), dtype=np.float64) * (-2.0)
    vx_r, vz_r = sample_receivers(vx_c, vz_c, sampling)
    assert np.allclose(vx_r, 7.5) and np.allclose(vz_r, -2.0)
    print("Constant field recovery: OK")

    # 2. Affine field on staggered coordinates -> exact recovery
    i = np.arange(g.nx)
    j = np.arange(g.nz)
    
    Xvx = g.x0 + (i + 0.5) * g.dx
    Zvx = g.z0 + j * g.dz
    vx_aff = np.add.outer(2.0 * Xvx, -0.5 * Zvx) + 3.0

    Xvz = g.x0 + i * g.dx
    Zvz = g.z0 + (j + 0.5) * g.dz
    vz_aff = np.add.outer(-1.0 * Xvz, 4.0 * Zvz) - 2.0

    vx_r2, vz_r2 = sample_receivers(vx_aff, vz_aff, sampling)
    assert np.allclose(vx_r2, 2.0 * x - 0.5 * z + 3.0, atol=1e-10)
    assert np.allclose(vz_r2, -1.0 * x + 4.0 * z - 2.0, atol=1e-10)
    print("Affine field (staggered coordinates): OK")

    # 3. Weights sum to 1
    for name, s in [("vx", sampling.vx), ("vz", sampling.vz)]:
        wsum = s.w00 + s.w10 + s.w01 + s.w11
        assert np.allclose(wsum, 1.0, atol=1e-12)
    print("Weight normalization: OK")

    # 4. Read-only arrays
    try:
        sampling.vx.ix[0] = 999
        raise AssertionError("ix must be read-only")
    except ValueError:
        pass
    print("Read-only arrays guard: OK")

    # 5. Boundary float tolerance
    x_e = np.array([g.x0 + 0.5 * g.dx - 1e-13])
    z_e = np.array([50.0])
    rec_e = Receivers2D(
        x=x_e, z=z_e,
        ix=np.zeros(1, dtype=int), iz=np.zeros(1, dtype=int),
        tx=np.ones(1), tz=np.zeros(1), s=np.zeros(1)
    )
    build_receiver_sampling(g, rec_e)
    print("Boundary tolerance guard: OK")

    # 6. Out of bounds -> ValueError
    x_b = np.array([g.x0 + 0.5 * g.dx - 1.0])
    z_b = np.array([50.0])
    rec_b = Receivers2D(
        x=x_b, z=z_b,
        ix=np.zeros(1, dtype=int), iz=np.zeros(1, dtype=int),
        tx=np.ones(1), tz=np.zeros(1), s=np.zeros(1)
    )
    try:
        build_receiver_sampling(g, rec_b)
        raise AssertionError("Should raise ValueError")
    except ValueError:
        pass
    print("Out of bounds guard: OK")

    # 7. Empty receivers guard
    rec_empty = Receivers2D(
        x=np.array([]), z=np.array([]),
        ix=np.array([], dtype=int), iz=np.array([], dtype=int),
        tx=np.array([]), tz=np.array([]), s=np.array([])
    )
    try:
        build_receiver_sampling(g, rec_empty)
        raise AssertionError("Should raise ValueError for 0 receivers")
    except ValueError:
        pass
    print("Empty receiver guard: OK")

    # 8. Use in loop (reproducible)
    rng = np.random.default_rng(42)
    nt = 10
    rec_vx = np.zeros((nrec, nt))
    rec_vz = np.zeros((nrec, nt))
    for it in range(nt):
        rec_vx[:, it], rec_vz[:, it] = sample_receivers(
            rng.random((g.nx, g.nz)), rng.random((g.nx, g.nz)), sampling
        )
    assert rec_vx.shape == (nrec, nt)
    print(f"Time loop execution: OK shape={rec_vx.shape}")

    print("\n✓ sampling.py: All tests passed")


if __name__ == "__main__":
    _self_test()