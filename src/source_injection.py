# ==============================================================================
# src/source_injection.py — source-to-solver adapter
#
# Purpose
#   Convert an EmbeddedSource2D into plain numpy arrays that the solver
#   can consume directly, without knowing about spreading mode or staggered
#   geometry.
#
# Position in the stack
#   source.py              — source physics: tensor, STF, physical position
#   source_spreading.py    — staggered-grid bilinear geometry
#   source_injection.py    — source object -> solver-ready arrays
#   solver_numba_fused.py  — production solver, receives only plain arrays
#
# Design
#   StressSourceInjection holds six read-only numpy arrays, shape (4,) each,
#   describing how the three stress components are scattered onto the grid.
#
#   Both spreading modes produce arrays of the same shape and semantics:
#
#       spreading="nearest"
#           normal_w = [1, 0, 0, 0]
#           shear_w  = [0.25, 0.25, 0.25, 0.25]
#
#           This reproduces the legacy moment-tensor source convention:
#           point injection for sxx/szz and four-node centroid correction
#           for sxz on the staggered shear grid.
#
#       spreading="bilinear"
#           weights are the bilinear coefficients computed from the actual
#           physical source position relative to each staggered sub-grid.
#
#   In both cases the solver calls the same inject_stress_source_numba kernel.
#   No branching on spreading mode is needed inside the time loop.
#
# Physical convention
#   The weights here only distribute a scalar stress-source amplitude.
#   They do NOT apply dt/(dx*dz). That scaling belongs in the solver:
#
#       amp_xx = stf_xx[it] * dt / (dx * dz)
#       sxx[ix[k], iz[k]] += amp_xx * w[k]
#
#   Because sum(w) = 1, the total injected amplitude is conserved.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.source_spreading import build_stress_source_spreading

if TYPE_CHECKING:
    from src.grid import Grid2D
    from src.source import EmbeddedSource2D


_VALID_SPREADING = frozenset({"nearest", "bilinear"})


# ==============================================================================
# 1. SOLVER-READY INJECTION DESCRIPTOR
# ==============================================================================

@dataclass(frozen=True)
class StressSourceInjection:
    """
    Solver-ready stress-source injection descriptor.

    The production solver should consume only these arrays, not source objects
    or staggered-grid geometry objects.

    Attributes
    ----------
    normal_ix, normal_iz : np.ndarray, shape (4,), int64
        Node indices for sxx and szz on the integer normal-stress grid.

    normal_w : np.ndarray, shape (4,), float64
        Weights for sxx and szz. Must be finite, non-negative, <= 1,
        and sum to 1.

    shear_ix, shear_iz : np.ndarray, shape (4,), int64
        Node indices for sxz on the half-integer shear-stress grid.

    shear_w : np.ndarray, shape (4,), float64
        Weights for sxz. Must be finite, non-negative, <= 1,
        and sum to 1.

    mode : str
        Source spreading mode used to build the descriptor:
        "nearest" or "bilinear". Stored for diagnostics only.

    Node ordering
    -------------
    Consistent with BilinearStencil in source_spreading.py:

        k=0: (i0,   j0  )  w = (1-wx)*(1-wz)
        k=1: (i0+1, j0  )  w = wx*(1-wz)
        k=2: (i0,   j0+1)  w = (1-wx)*wz
        k=3: (i0+1, j0+1)  w = wx*wz

    z is depth-increasing downward, so j0 and j0+1 are index labels,
    not "upper/lower" directions.
    """
    normal_ix: np.ndarray
    normal_iz: np.ndarray
    normal_w:  np.ndarray
    shear_ix:  np.ndarray
    shear_iz:  np.ndarray
    shear_w:   np.ndarray
    mode:      str = "unknown"

    def __post_init__(self) -> None:
        for name, arr_in, dtype in [
            ("normal_ix", self.normal_ix, np.int64),
            ("normal_iz", self.normal_iz, np.int64),
            ("normal_w",  self.normal_w,  np.float64),
            ("shear_ix",  self.shear_ix,  np.int64),
            ("shear_iz",  self.shear_iz,  np.int64),
            ("shear_w",   self.shear_w,   np.float64),
        ]:
            # Force a true copy so this descriptor never aliases source arrays.
            arr = np.array(arr_in, dtype=dtype, copy=True)
            arr = np.ascontiguousarray(arr)

            if arr.shape != (4,):
                raise ValueError(
                    f"StressSourceInjection.{name} must have shape (4,), "
                    f"got {arr.shape}."
                )

            arr.flags.writeable = False
            object.__setattr__(self, name, arr)

        mode = str(self.mode).lower()
        if mode not in _VALID_SPREADING and mode != "unknown":
            raise ValueError(
                f"StressSourceInjection.mode must be one of "
                f"{sorted(_VALID_SPREADING)} or 'unknown', got {self.mode!r}."
            )
        object.__setattr__(self, "mode", mode)

        for name, w in [("normal_w", self.normal_w), ("shear_w", self.shear_w)]:
            if not np.all(np.isfinite(w)):
                raise ValueError(
                    f"StressSourceInjection.{name} weights must be finite; got {w}."
                )

            wsum = float(w.sum())
            if abs(wsum - 1.0) > 1.0e-12:
                raise ValueError(
                    f"StressSourceInjection.{name} must sum to 1.0; "
                    f"got {wsum:.15e}."
                )

            if np.any(w < -1.0e-12):
                raise ValueError(
                    f"StressSourceInjection.{name} weights must be non-negative; "
                    f"got {w}."
                )

            if np.any(w > 1.0 + 1.0e-12):
                raise ValueError(
                    f"StressSourceInjection.{name} weights must be <= 1.0; "
                    f"got {w}."
                )

    def validate_bounds(self, nx: int, nz: int) -> None:
        """
        Check that all injection indices are valid for arrays of shape (nx, nz).
        """
        if nx <= 0 or nz <= 0:
            raise ValueError(f"nx and nz must be positive, got nx={nx}, nz={nz}.")

        for name, ix, iz in [
            ("normal", self.normal_ix, self.normal_iz),
            ("shear",  self.shear_ix,  self.shear_iz),
        ]:
            if int(ix.min()) < 0 or int(ix.max()) >= nx:
                raise ValueError(
                    f"{name} source ix indices out of bounds for nx={nx}: {ix}."
                )

            if int(iz.min()) < 0 or int(iz.max()) >= nz:
                raise ValueError(
                    f"{name} source iz indices out of bounds for nz={nz}: {iz}."
                )

    def dominant_node(self) -> tuple[int, int]:
        """
        Return the normal-stress node with the largest normal-source weight.
        """
        k = int(np.argmax(self.normal_w))
        return int(self.normal_ix[k]), int(self.normal_iz[k])

    def is_point_injection(self, tol: float = 1.0e-12) -> bool:
        """
        Return True if normal injection is exactly a single-node point injection.
        """
        k = int(np.argmax(self.normal_w))

        if abs(float(self.normal_w[k]) - 1.0) > tol:
            return False

        mask = np.ones(4, dtype=bool)
        mask[k] = False

        return bool(np.all(np.abs(self.normal_w[mask]) <= tol))

    def summary(self) -> str:
        lines = [f"StressSourceInjection(mode={self.mode}):"]

        for label, ix, iz, w in [
            ("normal (sxx/szz)", self.normal_ix, self.normal_iz, self.normal_w),
            ("shear  (sxz)    ", self.shear_ix,  self.shear_iz,  self.shear_w),
        ]:
            lines.append(f"  {label}:")
            for k in range(4):
                lines.append(
                    f"    k={k}: "
                    f"(ix={int(ix[k])}, iz={int(iz[k])})  "
                    f"w={float(w[k]):.6f}"
                )
            lines.append(f"    sum(w) = {float(w.sum()):.15e}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# ==============================================================================
# 2. FACTORY
# ==============================================================================

def build_stress_source_injection(
    source: "EmbeddedSource2D",
    grid: "Grid2D",
) -> StressSourceInjection:
    """
    Convert an EmbeddedSource2D into solver-ready stress-source arrays.

    Parameters
    ----------
    source :
        EmbeddedSource2D with spreading="nearest" or spreading="bilinear".
    grid :
        Grid2D. Used for constructing the nearest equivalent stencil and
        for index bounds validation.

    Returns
    -------
    StressSourceInjection
        Plain read-only arrays for the production solver.

    Notes
    -----
    spreading="nearest"
        The equivalent stencil is built from source.x_embedded_m and
        source.z_embedded_m, i.e. from the snapped nearest normal-stress node.
        This gives:

            normal_w = [1, 0, 0, 0]
            shear_w  = [0.25, 0.25, 0.25, 0.25]

        for an interior source. This reproduces the legacy staggered-grid
        moment-tensor injection.

    spreading="bilinear"
        The stencil is taken from source.spreading_stencil, already computed
        from the actual physical source position at source build time.
    """
    mode = str(source.spreading).lower()

    if mode not in _VALID_SPREADING:
        raise ValueError(
            f"Unsupported source.spreading={source.spreading!r}. "
            f"Expected one of {sorted(_VALID_SPREADING)}."
        )

    if mode == "bilinear":
        if source.spreading_stencil is None:
            raise ValueError(
                "EmbeddedSource2D has spreading='bilinear' but "
                "spreading_stencil is None. Rebuild the source with "
                "build_source_2d(..., spreading='bilinear')."
            )

        stencil = source.spreading_stencil

    else:
        # mode == "nearest"
        #
        # Build the equivalent staggered-grid stencil from the snapped
        # normal-stress-grid position. This avoids duplicating nearest-source
        # geometry here and guarantees consistency with source_spreading.py.
        stencil = build_stress_source_spreading(
            grid=grid,
            x_s=float(source.x_embedded_m),
            z_s=float(source.z_embedded_m),
        )

    injection = StressSourceInjection(
        normal_ix=stencil.sxx.ix,
        normal_iz=stencil.sxx.iz,
        normal_w=stencil.sxx.w,
        shear_ix=stencil.sxz.ix,
        shear_iz=stencil.sxz.iz,
        shear_w=stencil.sxz.w,
        mode=mode,
    )

    injection.validate_bounds(nx=int(grid.nx), nz=int(grid.nz))

    return injection


# ==============================================================================
# 3. SELF-TEST
# ==============================================================================

def _self_test() -> None:
    from types import SimpleNamespace

    # Minimal Grid2D-compatible namespace.
    dx = 10.0
    dz = 10.0
    nx = 51
    nz = 51
    x0 = 0.0
    z0 = 0.0

    grid = SimpleNamespace(
        x0=x0,
        z0=z0,
        dx=dx,
        dz=dz,
        nx=nx,
        nz=nz,
        x=x0 + np.arange(nx, dtype=np.float64) * dx,
        z=z0 + np.arange(nz, dtype=np.float64) * dz,
    )

    # --------------------------------------------------------------------------
    # 1. nearest: point normal injection + equal-weight shear centroid correction
    # --------------------------------------------------------------------------
    src_near = SimpleNamespace(
        spreading="nearest",
        spreading_stencil=None,
        x_embedded_m=100.0,  # ix=10 snapped normal-stress node
        z_embedded_m=200.0,  # iz=20 snapped normal-stress node
    )

    inj_near = build_stress_source_injection(src_near, grid)

    assert inj_near.mode == "nearest"
    assert inj_near.is_point_injection()
    assert inj_near.dominant_node() == (10, 20)
    assert np.allclose(inj_near.normal_w.sum(), 1.0, atol=1.0e-12)

    assert np.allclose(inj_near.shear_w, 0.25, atol=1.0e-12)
    assert set(inj_near.shear_ix.tolist()) == {9, 10}
    assert set(inj_near.shear_iz.tolist()) == {19, 20}

    print("nearest injection descriptor: OK")

    # --------------------------------------------------------------------------
    # 2. bilinear off-grid: valid conservative weights
    # --------------------------------------------------------------------------
    x_off = 123.0
    z_off = 178.0
    stencil_off = build_stress_source_spreading(
        grid=grid,
        x_s=x_off,
        z_s=z_off,
    )

    src_bil = SimpleNamespace(
        spreading="bilinear",
        spreading_stencil=stencil_off,
        x_embedded_m=x_off,
        z_embedded_m=z_off,
    )

    inj_bil = build_stress_source_injection(src_bil, grid)

    assert inj_bil.mode == "bilinear"
    assert np.allclose(inj_bil.normal_w.sum(), 1.0, atol=1.0e-12)
    assert np.allclose(inj_bil.shear_w.sum(), 1.0, atol=1.0e-12)
    assert np.all(inj_bil.normal_w >= -1.0e-15)
    assert np.all(inj_bil.shear_w >= -1.0e-15)
    assert not inj_bil.is_point_injection()

    print("bilinear off-grid injection descriptor: OK")

    # --------------------------------------------------------------------------
    # 3. read-only arrays
    # --------------------------------------------------------------------------
    try:
        inj_near.normal_w[0] = 999.0
        raise AssertionError("normal_w should be read-only.")
    except ValueError:
        pass

    try:
        inj_near.normal_ix[0] = 999
        raise AssertionError("normal_ix should be read-only.")
    except ValueError:
        pass

    print("read-only arrays: OK")

    # --------------------------------------------------------------------------
    # 4. bilinear on-grid source equals nearest source geometry
    # --------------------------------------------------------------------------
    x_on = 100.0
    z_on = 200.0

    stencil_on = build_stress_source_spreading(
        grid=grid,
        x_s=x_on,
        z_s=z_on,
    )

    src_bil_on = SimpleNamespace(
        spreading="bilinear",
        spreading_stencil=stencil_on,
        x_embedded_m=x_on,
        z_embedded_m=z_on,
    )

    inj_bil_on = build_stress_source_injection(src_bil_on, grid)

    assert inj_bil_on.mode == "bilinear"
    assert np.allclose(inj_bil_on.normal_ix, inj_near.normal_ix)
    assert np.allclose(inj_bil_on.normal_iz, inj_near.normal_iz)
    assert np.allclose(inj_bil_on.normal_w, inj_near.normal_w, atol=1.0e-12)
    assert np.allclose(inj_bil_on.shear_ix, inj_near.shear_ix)
    assert np.allclose(inj_bil_on.shear_iz, inj_near.shear_iz)
    assert np.allclose(inj_bil_on.shear_w, inj_near.shear_w, atol=1.0e-12)

    print("on-grid bilinear == nearest geometry: OK")

    # --------------------------------------------------------------------------
    # 5. bilinear with missing stencil raises ValueError
    # --------------------------------------------------------------------------
    src_missing = SimpleNamespace(
        spreading="bilinear",
        spreading_stencil=None,
        x_embedded_m=100.0,
        z_embedded_m=200.0,
    )

    try:
        build_stress_source_injection(src_missing, grid)
        raise AssertionError("Expected ValueError for bilinear source without stencil.")
    except ValueError:
        pass

    print("bilinear missing stencil ValueError: OK")

    # --------------------------------------------------------------------------
    # 6. invalid spreading mode raises ValueError
    # --------------------------------------------------------------------------
    src_invalid = SimpleNamespace(
        spreading="invalid",
        spreading_stencil=None,
        x_embedded_m=100.0,
        z_embedded_m=200.0,
    )

    try:
        build_stress_source_injection(src_invalid, grid)
        raise AssertionError("Expected ValueError for invalid spreading mode.")
    except ValueError:
        pass

    print("invalid spreading ValueError: OK")

    # --------------------------------------------------------------------------
    # 7. invalid weights are rejected
    # --------------------------------------------------------------------------
    try:
        StressSourceInjection(
            normal_ix=np.array([0, 0, 0, 0]),
            normal_iz=np.array([0, 0, 0, 0]),
            normal_w=np.array([1.0, 0.0, 0.0, np.nan]),
            shear_ix=np.array([0, 0, 0, 0]),
            shear_iz=np.array([0, 0, 0, 0]),
            shear_w=np.array([0.25, 0.25, 0.25, 0.25]),
            mode="nearest",
        )
        raise AssertionError("Expected ValueError for NaN normal_w.")
    except ValueError:
        pass

    try:
        StressSourceInjection(
            normal_ix=np.array([0, 0, 0, 0]),
            normal_iz=np.array([0, 0, 0, 0]),
            normal_w=np.array([1.2, -0.2, 0.0, 0.0]),
            shear_ix=np.array([0, 0, 0, 0]),
            shear_iz=np.array([0, 0, 0, 0]),
            shear_w=np.array([0.25, 0.25, 0.25, 0.25]),
            mode="nearest",
        )
        raise AssertionError("Expected ValueError for invalid normal_w.")
    except ValueError:
        pass

    print("invalid weight checks: OK")

    print("\n✓ source_injection.py: all tests passed")


if __name__ == "__main__":
    _self_test()