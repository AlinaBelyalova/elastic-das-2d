# ==============================================================================
# src/source.py — 2D seismic source definitions
#
# Moment tensor convention
#   Follows Aki & Richards (2002), Chapter 3.
#   In the 2D (x, z) plane the relevant components are Mxx, Mzz, Mxz,
#   forming the symmetric tensor
#
#       [ Mxx  Mxz ]
#       [ Mxz  Mzz ]
#
#   The double-couple base tensor (theta_deg=0) is
#
#       M_base = M0 * [[0, 1], [1, 0]]
#
#   which corresponds to fault normal n = z-hat and slip d = x-hat, i.e.
#   a horizontal fault with horizontal slip (pure shear, zero trace).
#   Rotating by theta_deg rotates the fault orientation in the x-z plane.
#
# Source time function convention
#   stf.values[n] is a dimensionless source-time factor sampled on the
#   integer stress time grid.
#
#   In the leapfrog solver it is injected into the updated stress field at
#   t_sigma[n+1] = (n+1)*dt.
#
#   Important physical note:
#   The current Ricker-based source time functions are band-limited modelling
#   wavelets. They are useful for controlled synthetic experiments and
#   validation. For absolute earthquake-amplitude modelling and FWI, a
#   separate unit-area moment-rate source time function should be added:
#
#       sum(moment_rate_stf) * dt = 1
#
#   so that M0 * moment_rate_stf(t) has physical moment-rate units.
#
# Source spreading
#   spreading="nearest" (default)
#       Source is snapped to the nearest integer stress-grid node (ix, iz).
#       sxx/szz inject at a single node; sxz is handled by the legacy solver
#       injection convention. This reproduces the original behaviour and all
#       existing scripts continue to work unchanged.
#
#   spreading="bilinear"
#       Source is distributed to surrounding nodes with bilinear weights
#       computed from the actual physical coordinates (x_m, z_m), not the
#       snapped position. Each stress component uses its own staggered
#       sub-grid origin:
#
#           Mxx -> sxx grid
#           Mzz -> szz grid
#           Mxz -> sxz grid
#
#       Use this mode for production SAFOD modelling and future FWI, where
#       source positions should not be forced to coincide with grid nodes.
#
#   The spreading stencil is computed once at build time and stored in
#   EmbeddedSource2D.spreading_stencil. Solvers that support bilinear
#   injection should read this field. Solvers that do not support it can
#   continue using ix/iz in the nearest-source path.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.grid import Grid2D
from src.source_spreading import (
    StressSourceSpreading,
    build_stress_source_spreading,
)


_VALID_SPREADING = frozenset({"nearest", "bilinear"})


# ==============================================================================
# 1. 2D MOMENT TENSOR
# ==============================================================================

@dataclass(frozen=True)
class MomentTensor2D:
    """
    Symmetric 2D moment tensor in the (x, z) plane.

    Matrix form
    -----------
        [ Mxx  Mxz ]
        [ Mxz  Mzz ]

    Units
    -----
    N·m, interpreted as a 2D modelling moment convention in this code.
    For direct comparison with real 3D earthquake moments, the 2D/3D
    amplitude convention must be treated carefully.
    """
    Mxx: float
    Mzz: float
    Mxz: float

    def as_matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.Mxx, self.Mxz],
                [self.Mxz, self.Mzz],
            ],
            dtype=np.float64,
        )

    def trace(self) -> float:
        """Return Mxx + Mzz, proportional to the isotropic part."""
        return float(self.Mxx + self.Mzz)

    def frobenius_norm(self) -> float:
        """Return ||M||_F = sqrt(Mxx^2 + Mzz^2 + 2*Mxz^2)."""
        return float(np.sqrt(self.Mxx**2 + self.Mzz**2 + 2.0 * self.Mxz**2))

    def scaled(self, factor: float) -> "MomentTensor2D":
        return MomentTensor2D(
            Mxx=float(factor * self.Mxx),
            Mzz=float(factor * self.Mzz),
            Mxz=float(factor * self.Mxz),
        )

    def summary(self) -> str:
        return (
            "MomentTensor2D:\n"
            f"  Mxx      = {self.Mxx:.6e}\n"
            f"  Mzz      = {self.Mzz:.6e}\n"
            f"  Mxz      = {self.Mxz:.6e}\n"
            f"  trace    = {self.trace():.6e}\n"
            f"  ||M||_F  = {self.frobenius_norm():.6e}"
        )

    def __repr__(self) -> str:
        return self.summary()


def rotate_moment_tensor_2d(
    mt: MomentTensor2D,
    theta_deg: float,
) -> MomentTensor2D:
    """
    Rotate a 2D symmetric moment tensor by angle theta_deg.

    Applies:

        M' = R M R^T

    where R is the 2D rotation matrix.

    For the double-couple base tensor (Mxx=Mzz=0, Mxz=M0):

        theta_deg =  0°  -> pure shear: fault normal = z, slip = x
        theta_deg = 45°  -> compressive: Mxx=-M0, Mzz=+M0, Mxz=0
        theta_deg = 90°  -> pure shear: fault normal = x, slip = z
    """
    theta = np.deg2rad(theta_deg)
    c2 = np.cos(2.0 * theta)
    s2 = np.sin(2.0 * theta)

    m_avg = 0.5 * (mt.Mxx + mt.Mzz)
    m_dev = 0.5 * (mt.Mxx - mt.Mzz)

    return MomentTensor2D(
        Mxx=float(m_avg + m_dev * c2 - mt.Mxz * s2),
        Mzz=float(m_avg - m_dev * c2 + mt.Mxz * s2),
        Mxz=float(m_dev * s2 + mt.Mxz * c2),
    )


# ==============================================================================
# 2. BASE 2D SOURCE TENSORS
# ==============================================================================

def base_double_couple_2d() -> MomentTensor2D:
    """
    Unit 2D double-couple tensor.

        M_base = [[0, 1], [1, 0]]

    This is pure shear with zero trace and Frobenius norm sqrt(2).
    Fault normal = z-hat, slip direction = x-hat.
    """
    return MomentTensor2D(Mxx=0.0, Mzz=0.0, Mxz=1.0)


def isotropic_tensor_2d() -> MomentTensor2D:
    """
    Unit 2D isotropic tensor.

        M_iso = [[1, 0], [0, 1]]

    Equal normal components, zero shear, Frobenius norm sqrt(2).
    """
    return MomentTensor2D(Mxx=1.0, Mzz=1.0, Mxz=0.0)


def build_rotated_double_couple_2d(
    theta_deg: float,
    scalar_moment: float,
) -> MomentTensor2D:
    """
    Build a scaled, rotated 2D double-couple tensor.

    Parameters
    ----------
    theta_deg :
        Fault rotation angle in the x-z plane [degrees].
    scalar_moment :
        Scalar moment M0 [N·m]. Must be positive.
    """
    if scalar_moment <= 0.0:
        raise ValueError(f"scalar_moment must be positive, got {scalar_moment}.")

    return rotate_moment_tensor_2d(
        base_double_couple_2d(),
        theta_deg,
    ).scaled(scalar_moment)


def build_isotropic_source_tensor_2d(scalar_moment: float) -> MomentTensor2D:
    """
    Build a scaled 2D isotropic tensor.

    Parameters
    ----------
    scalar_moment :
        Scalar moment M0 [N·m]. Must be positive.
    """
    if scalar_moment <= 0.0:
        raise ValueError(f"scalar_moment must be positive, got {scalar_moment}.")

    return isotropic_tensor_2d().scaled(scalar_moment)


# ==============================================================================
# 3. SOURCE TIME FUNCTION
# ==============================================================================

@dataclass(frozen=True)
class SourceTimeFunction:
    """
    Discrete source time function sampled on the solver integer time grid.

    values[n] is a unit-amplitude source-time factor at integer index n.
    In the current solver convention, values[n] is injected into the updated
    stress field at t_sigma[n+1] = (n+1)*dt.

    Note
    ----
    The current Ricker functions are modelling wavelets. They are not
    normalised as unit-area physical moment-rate functions.
    """
    values: np.ndarray
    dt: float
    t0: float
    kind: str

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=np.float64, copy=True)

        if values.ndim != 1:
            raise ValueError(f"values must be 1D, got shape {values.shape}.")
        if values.size < 2:
            raise ValueError("values must contain at least 2 samples.")
        if self.dt <= 0.0:
            raise ValueError(f"dt must be positive, got {self.dt}.")
        if self.t0 < 0.0:
            raise ValueError(f"t0 must be non-negative, got {self.t0}.")
        if not np.all(np.isfinite(values)):
            raise ValueError("source time function values must be finite.")

        values.flags.writeable = False

        object.__setattr__(self, "values", values)
        object.__setattr__(self, "dt", float(self.dt))
        object.__setattr__(self, "t0", float(self.t0))
        object.__setattr__(self, "kind", str(self.kind))

    @property
    def nt(self) -> int:
        return int(self.values.size)

    @property
    def t(self) -> np.ndarray:
        """
        Integer time axis: t[n] = n*dt.

        Note: values[n] is injected at t_sigma[n+1] = (n+1)*dt,
        not at t[n].
        """
        return np.arange(self.nt, dtype=np.float64) * self.dt

    def peak_amplitude(self) -> float:
        return float(np.max(np.abs(self.values)))

    def summary(self) -> str:
        return (
            f"SourceTimeFunction(kind={self.kind}, nt={self.nt}, "
            f"dt={self.dt:.3e}, t0={self.t0:.3e}, "
            f"peak={self.peak_amplitude():.3e})"
        )

    def __repr__(self) -> str:
        return self.summary()


def ricker_wavelet(
    *,
    nt: int,
    dt: float,
    f0_hz: float,
) -> SourceTimeFunction:
    """
    Ricker wavelet with unit peak scale.

    W(t) = (1 - 2*pi^2*f0^2*(t-t0)^2) exp(-pi^2*f0^2*(t-t0)^2)

    where:

        t0 = 1.2 / f0
    """
    if nt < 2:
        raise ValueError(f"nt must be >= 2, got {nt}.")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}.")
    if f0_hz <= 0.0:
        raise ValueError(f"f0_hz must be positive, got {f0_hz}.")

    t0 = 1.2 / f0_hz
    t = np.arange(nt, dtype=np.float64) * dt
    arg = (np.pi * f0_hz * (t - t0)) ** 2

    return SourceTimeFunction(
        values=(1.0 - 2.0 * arg) * np.exp(-arg),
        dt=dt,
        t0=t0,
        kind="ricker",
    )


def ricker_derivative_wavelet(
    *,
    nt: int,
    dt: float,
    f0_hz: float,
) -> SourceTimeFunction:
    """
    First time derivative of the Ricker wavelet.

    dW/dt = 2*pi*f0*x*(2*x^2 - 3)*exp(-x^2)

    where:

        x = pi*f0*(t - t0)
    """
    if nt < 2:
        raise ValueError(f"nt must be >= 2, got {nt}.")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}.")
    if f0_hz <= 0.0:
        raise ValueError(f"f0_hz must be positive, got {f0_hz}.")

    t0 = 1.2 / f0_hz
    t = np.arange(nt, dtype=np.float64) * dt
    x = np.pi * f0_hz * (t - t0)

    return SourceTimeFunction(
        values=(2.0 * np.pi * f0_hz * x * (2.0 * x**2 - 3.0)) * np.exp(-x**2),
        dt=dt,
        t0=t0,
        kind="ricker_derivative",
    )


def build_source_time_function(
    *,
    nt: int,
    dt: float,
    f0_hz: float,
    derivative_order: int = 0,
) -> SourceTimeFunction:
    """
    Build a source time function.

    derivative_order=0 : Ricker wavelet
    derivative_order=1 : first derivative of Ricker wavelet
    """
    if derivative_order == 0:
        return ricker_wavelet(nt=nt, dt=dt, f0_hz=f0_hz)

    if derivative_order == 1:
        return ricker_derivative_wavelet(nt=nt, dt=dt, f0_hz=f0_hz)

    raise ValueError(
        f"Unsupported derivative_order={derivative_order}. "
        "Only 0 and 1 are supported."
    )


# ==============================================================================
# 4. EMBEDDED 2D SOURCE
# ==============================================================================

@dataclass(frozen=True)
class EmbeddedSource2D:
    """
    A 2D point source embedded in a Grid2D.

    Attributes
    ----------
    x_m, z_m :
        Requested physical position [m].

    x_embedded_m, z_embedded_m :
        For spreading="nearest", this is the snapped nearest stress-grid node.
        For spreading="bilinear", this is the actual physical source position.

    ix, iz :
        Dominant nearest integer stress-grid node indices. Always set for
        backward compatibility and simple diagnostics.

    m2d :
        Scaled 2D moment tensor [N·m].

    stf :
        Source time function.

    scalar_moment :
        Scalar moment M0 [N·m], stored for bookkeeping.

    spreading :
        "nearest" or "bilinear".

    spreading_stencil :
        None for spreading="nearest".
        StressSourceSpreading for spreading="bilinear".

    label :
        Human-readable description.
    """
    x_m: float
    z_m: float
    x_embedded_m: float
    z_embedded_m: float
    ix: int
    iz: int
    m2d: MomentTensor2D
    stf: SourceTimeFunction
    scalar_moment: float
    spreading: str = "nearest"
    spreading_stencil: Optional[StressSourceSpreading] = None
    label: str = "2D source"

    def summary(self) -> str:
        lines = [
            f"{self.label}",
            f"  requested position : ({self.x_m:.4f}, {self.z_m:.4f}) m",
            f"  embedded position  : ({self.x_embedded_m:.4f}, {self.z_embedded_m:.4f}) m",
            f"  grid indices (ix,iz): ({self.ix}, {self.iz})",
            f"  spreading          : {self.spreading}",
            f"  scalar moment M0   : {self.scalar_moment:.6e} N·m",
            f"  STF                : {self.stf.summary()}",
            f"  tensor trace       : {self.m2d.trace():.6e}",
            f"  tensor ||M||_F     : {self.m2d.frobenius_norm():.6e}",
        ]

        if self.spreading_stencil is None:
            lines.append("  spreading stencil  : none")
        else:
            stn = self.spreading_stencil
            sxx_node = stn.sxx.dominant_node()
            sxz_node = stn.sxz.dominant_node()
            lines.extend(
                [
                    "  spreading stencil  : available",
                    f"  sxx dominant node  : (ix={sxx_node[0]}, iz={sxx_node[1]})",
                    f"  sxz dominant node  : (ix={sxz_node[0]}, iz={sxz_node[1]})",
                    f"  sxx weights        : {stn.sxx.w}",
                    f"  sxz weights        : {stn.sxz.w}",
                ]
            )

        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _normalise_spreading(spreading: str) -> str:
    spreading_norm = str(spreading).lower()

    if spreading_norm not in _VALID_SPREADING:
        raise ValueError(
            f"spreading must be one of {sorted(_VALID_SPREADING)}, "
            f"got {spreading!r}."
        )

    return spreading_norm


def build_source_2d(
    *,
    grid: Grid2D,
    x_m: float,
    z_m: float,
    mt2d: MomentTensor2D,
    scalar_moment: float,
    nt: int,
    dt: float,
    f0_hz: float,
    derivative_order: int = 0,
    spreading: str = "nearest",
    label: str = "2D source",
) -> EmbeddedSource2D:
    """
    Build an EmbeddedSource2D at physical position (x_m, z_m).

    Parameters
    ----------
    grid :
        Grid2D defining the computational domain.

    x_m, z_m :
        Physical source position [m].

    mt2d :
        Scaled 2D moment tensor [N·m].

    scalar_moment :
        Scalar moment M0 [N·m], stored for bookkeeping.

    nt, dt :
        Must match grid.nt and grid.dt.

    f0_hz :
        Dominant frequency of the source time function [Hz].

    derivative_order :
        0 -> Ricker wavelet.
        1 -> first derivative of Ricker wavelet.

    spreading :
        "nearest"  -> backward-compatible nearest-node injection.
        "bilinear" -> staggered-grid bilinear source spreading.

    label :
        Human-readable source label.
    """
    spreading_norm = _normalise_spreading(spreading)

    if nt != grid.nt:
        raise ValueError(f"nt={nt} must match grid.nt={grid.nt}.")

    if not np.isclose(dt, grid.dt):
        raise ValueError(f"dt={dt:.6e} must match grid.dt={grid.dt:.6e}.")

    if scalar_moment <= 0.0:
        raise ValueError(f"scalar_moment must be positive, got {scalar_moment}.")

    if not np.isfinite(x_m) or not np.isfinite(z_m):
        raise ValueError(f"Source position must be finite, got x_m={x_m}, z_m={z_m}.")

    if not isinstance(mt2d, MomentTensor2D):
        raise TypeError(f"mt2d must be MomentTensor2D, got {type(mt2d)!r}.")

    ix, iz, x_snapped, z_snapped = grid.get_closest_node(x_m, z_m)

    stf = build_source_time_function(
        nt=nt,
        dt=dt,
        f0_hz=f0_hz,
        derivative_order=derivative_order,
    )

    if spreading_norm == "nearest":
        return EmbeddedSource2D(
            x_m=float(x_m),
            z_m=float(z_m),
            x_embedded_m=float(x_snapped),
            z_embedded_m=float(z_snapped),
            ix=int(ix),
            iz=int(iz),
            m2d=mt2d,
            stf=stf,
            scalar_moment=float(scalar_moment),
            spreading="nearest",
            spreading_stencil=None,
            label=str(label),
        )

    spreading_stencil = build_stress_source_spreading(
        grid=grid,
        x_s=float(x_m),
        z_s=float(z_m),
    )

    return EmbeddedSource2D(
        x_m=float(x_m),
        z_m=float(z_m),
        x_embedded_m=float(x_m),
        z_embedded_m=float(z_m),
        ix=int(ix),
        iz=int(iz),
        m2d=mt2d,
        stf=stf,
        scalar_moment=float(scalar_moment),
        spreading="bilinear",
        spreading_stencil=spreading_stencil,
        label=str(label),
    )


# ==============================================================================
# 5. HIGH-LEVEL BUILDERS
# ==============================================================================

def build_dc_source(
    *,
    grid: Grid2D,
    x_m: float,
    z_m: float,
    theta_deg: float = 0.0,
    scalar_moment: float = 1.0e10,
    nt: int,
    dt: float,
    f0_hz: float,
    derivative_order: int = 0,
    spreading: str = "nearest",
) -> EmbeddedSource2D:
    """
    Build a 2D double-couple source.

    Parameters
    ----------
    theta_deg :
        Fault rotation in the x-z plane [degrees].

    scalar_moment :
        Scalar moment M0 [N·m].

    spreading :
        "nearest" or "bilinear".
    """
    spreading_norm = _normalise_spreading(spreading)

    mt2d = build_rotated_double_couple_2d(
        theta_deg=theta_deg,
        scalar_moment=scalar_moment,
    )

    return build_source_2d(
        grid=grid,
        x_m=x_m,
        z_m=z_m,
        mt2d=mt2d,
        scalar_moment=scalar_moment,
        nt=nt,
        dt=dt,
        f0_hz=f0_hz,
        derivative_order=derivative_order,
        spreading=spreading_norm,
        label=(
            f"2D double-couple "
            f"(theta={theta_deg:.1f}°, M0={scalar_moment:.2e} N·m, "
            f"spreading={spreading_norm})"
        ),
    )


def build_isotropic_source(
    *,
    grid: Grid2D,
    x_m: float,
    z_m: float,
    scalar_moment: float = 1.0e10,
    nt: int,
    dt: float,
    f0_hz: float,
    derivative_order: int = 0,
    spreading: str = "nearest",
) -> EmbeddedSource2D:
    """
    Build a 2D isotropic source.

    Parameters
    ----------
    scalar_moment :
        Scalar moment M0 [N·m].

    spreading :
        "nearest" or "bilinear".
    """
    spreading_norm = _normalise_spreading(spreading)

    mt2d = build_isotropic_source_tensor_2d(scalar_moment=scalar_moment)

    return build_source_2d(
        grid=grid,
        x_m=x_m,
        z_m=z_m,
        mt2d=mt2d,
        scalar_moment=scalar_moment,
        nt=nt,
        dt=dt,
        f0_hz=f0_hz,
        derivative_order=derivative_order,
        spreading=spreading_norm,
        label=(
            f"2D isotropic "
            f"(M0={scalar_moment:.2e} N·m, spreading={spreading_norm})"
        ),
    )


# ==============================================================================
# 6. SELF-TEST
# ==============================================================================

def _self_test() -> None:
    grid = Grid2D(
        nx=101,
        nz=81,
        dx=10.0,
        dz=10.0,
        nt=400,
        dt=1.0e-3,
        x0=0.0,
        z0=0.0,
    )

    # --------------------------------------------------------------------------
    # Moment tensor algebra
    # --------------------------------------------------------------------------
    mt0 = base_double_couple_2d()

    assert np.isclose(mt0.Mxx, 0.0, atol=1.0e-12)
    assert np.isclose(mt0.Mzz, 0.0, atol=1.0e-12)
    assert np.isclose(mt0.Mxz, 1.0, atol=1.0e-12)
    assert np.isclose(mt0.trace(), 0.0, atol=1.0e-12)

    mt45 = rotate_moment_tensor_2d(mt0, 45.0)

    assert np.isclose(mt45.Mxx, -1.0, atol=1.0e-12)
    assert np.isclose(mt45.Mzz, 1.0, atol=1.0e-12)
    assert np.isclose(mt45.Mxz, 0.0, atol=1.0e-12)

    mt30 = rotate_moment_tensor_2d(mt0, 30.0)
    assert np.isclose(mt0.frobenius_norm(), mt30.frobenius_norm(), atol=1.0e-12)

    print("MomentTensor2D + rotation: OK")

    # --------------------------------------------------------------------------
    # Source time functions
    # --------------------------------------------------------------------------
    stf = build_source_time_function(
        nt=grid.nt,
        dt=grid.dt,
        f0_hz=8.0,
        derivative_order=0,
    )

    assert stf.nt == grid.nt
    assert np.isclose(stf.dt, grid.dt)
    assert stf.peak_amplitude() > 0.0
    assert np.isclose(stf.t[0], 0.0)
    assert np.isclose(stf.t[-1], (grid.nt - 1) * grid.dt)

    stf_der = build_source_time_function(
        nt=grid.nt,
        dt=grid.dt,
        f0_hz=8.0,
        derivative_order=1,
    )

    assert stf_der.nt == grid.nt
    assert stf_der.peak_amplitude() > 0.0

    print("SourceTimeFunction: OK")

    # --------------------------------------------------------------------------
    # spreading='nearest' default, backward-compatible
    # --------------------------------------------------------------------------
    src_near = build_dc_source(
        grid=grid,
        x_m=400.0,
        z_m=300.0,
        theta_deg=30.0,
        scalar_moment=1.0e10,
        nt=grid.nt,
        dt=grid.dt,
        f0_hz=8.0,
    )

    assert isinstance(src_near, EmbeddedSource2D)
    assert src_near.spreading == "nearest"
    assert src_near.spreading_stencil is None
    assert 0 <= src_near.ix < grid.nx
    assert 0 <= src_near.iz < grid.nz
    assert np.isclose(src_near.x_embedded_m, grid.x[src_near.ix])
    assert np.isclose(src_near.z_embedded_m, grid.z[src_near.iz])

    print("build_dc_source(spreading='nearest'): OK")

    # --------------------------------------------------------------------------
    # spreading='bilinear' exactly on normal-stress grid node
    # --------------------------------------------------------------------------
    x_on = float(grid.x[40])
    z_on = float(grid.z[30])

    src_bil_on = build_dc_source(
        grid=grid,
        x_m=x_on,
        z_m=z_on,
        theta_deg=0.0,
        scalar_moment=1.0e10,
        nt=grid.nt,
        dt=grid.dt,
        f0_hz=8.0,
        spreading="bilinear",
    )

    assert src_bil_on.spreading == "bilinear"
    assert src_bil_on.spreading_stencil is not None
    assert src_bil_on.spreading_stencil.sxx.is_point_injection()
    assert src_bil_on.spreading_stencil.szz.is_point_injection()
    assert np.allclose(src_bil_on.spreading_stencil.sxz.w, 0.25, atol=1.0e-12)
    assert np.isclose(src_bil_on.x_embedded_m, x_on)
    assert np.isclose(src_bil_on.z_embedded_m, z_on)

    print("build_dc_source(spreading='bilinear', on-grid): OK")

    # --------------------------------------------------------------------------
    # spreading='bilinear' off normal-stress grid node
    # --------------------------------------------------------------------------
    x_off = float(grid.x[25] + 3.7)
    z_off = float(grid.z[20] + 6.1)

    src_bil_off = build_dc_source(
        grid=grid,
        x_m=x_off,
        z_m=z_off,
        theta_deg=45.0,
        scalar_moment=5.0e9,
        nt=grid.nt,
        dt=grid.dt,
        f0_hz=8.0,
        spreading="bilinear",
    )

    assert src_bil_off.spreading == "bilinear"
    assert src_bil_off.spreading_stencil is not None

    for component_name, stencil in [
        ("sxx", src_bil_off.spreading_stencil.sxx),
        ("szz", src_bil_off.spreading_stencil.szz),
        ("sxz", src_bil_off.spreading_stencil.sxz),
    ]:
        assert np.all(stencil.w >= -1.0e-15), f"{component_name} has negative weights."
        assert np.allclose(stencil.w.sum(), 1.0, atol=1.0e-12), (
            f"{component_name} weights do not sum to one."
        )

    assert not src_bil_off.spreading_stencil.sxx.is_point_injection()
    assert not src_bil_off.spreading_stencil.sxz.is_point_injection()

    print("build_dc_source(spreading='bilinear', off-grid): OK")

    # --------------------------------------------------------------------------
    # Case-insensitive spreading keyword
    # --------------------------------------------------------------------------
    src_case = build_dc_source(
        grid=grid,
        x_m=x_on,
        z_m=z_on,
        theta_deg=0.0,
        scalar_moment=1.0e10,
        nt=grid.nt,
        dt=grid.dt,
        f0_hz=8.0,
        spreading="BILINEAR",
    )

    assert src_case.spreading == "bilinear"
    assert src_case.spreading_stencil is not None

    print("Case-insensitive spreading keyword: OK")

    # --------------------------------------------------------------------------
    # Isotropic source with bilinear spreading
    # --------------------------------------------------------------------------
    iso = build_isotropic_source(
        grid=grid,
        x_m=500.0,
        z_m=400.0,
        scalar_moment=2.0e10,
        nt=grid.nt,
        dt=grid.dt,
        f0_hz=8.0,
        spreading="bilinear",
    )

    assert isinstance(iso, EmbeddedSource2D)
    assert iso.spreading == "bilinear"
    assert iso.spreading_stencil is not None
    assert np.isclose(iso.m2d.Mxx, iso.m2d.Mzz)
    assert np.isclose(iso.m2d.Mxz, 0.0)

    print("build_isotropic_source(spreading='bilinear'): OK")

    # --------------------------------------------------------------------------
    # Invalid spreading keyword
    # --------------------------------------------------------------------------
    try:
        build_dc_source(
            grid=grid,
            x_m=400.0,
            z_m=300.0,
            theta_deg=0.0,
            scalar_moment=1.0e10,
            nt=grid.nt,
            dt=grid.dt,
            f0_hz=8.0,
            spreading="invalid",
        )
        raise AssertionError("Expected ValueError for invalid spreading.")
    except ValueError:
        pass

    print("Invalid spreading ValueError: OK")

    print("\n✓ source.py self-test PASSED")


if __name__ == "__main__":
    _self_test()