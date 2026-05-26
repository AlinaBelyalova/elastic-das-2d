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
#   stf.values[n] is the moment-rate amplitude at step n.
#   In the leapfrog solver it is injected into the updated stress field at
#   t_sigma[n+1] = (n+1)*dt (see solver_numpy.py source convention).
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.grid import Grid2D


# ==============================================================================
# 1. 2D MOMENT TENSOR
# ==============================================================================

@dataclass(frozen=True)
class MomentTensor2D:
    """
    Symmetric 2D moment tensor in the (x, z) plane.

    Matrix form:
        [ Mxx  Mxz ]
        [ Mxz  Mzz ]

    Units: N·m (seismic moment).
    """
    Mxx: float
    Mzz: float
    Mxz: float

    def as_matrix(self) -> np.ndarray:
        return np.array(
            [[self.Mxx, self.Mxz],
             [self.Mxz, self.Mzz]],
            dtype=np.float64,
        )

    def trace(self) -> float:
        """Mxx + Mzz (proportional to isotropic part)."""
        return float(self.Mxx + self.Mzz)

    def frobenius_norm(self) -> float:
        """||M||_F = sqrt(Mxx² + 2·Mxz² + Mzz²)."""
        return float(np.sqrt(self.Mxx**2 + self.Mzz**2 + 2.0 * self.Mxz**2))

    def scaled(self, factor: float) -> "MomentTensor2D":
        return MomentTensor2D(
            Mxx=factor * self.Mxx,
            Mzz=factor * self.Mzz,
            Mxz=factor * self.Mxz,
        )

    def summary(self) -> str:
        return (
            "MomentTensor2D:\n"
            f"  Mxx = {self.Mxx:.6e}\n"
            f"  Mzz = {self.Mzz:.6e}\n"
            f"  Mxz = {self.Mxz:.6e}\n"
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

    Applies the standard tensor rotation M' = R M R^T, where R is the
    counterclockwise 2D rotation matrix for angle theta_deg.

    For the double-couple base tensor (Mxx=Mzz=0, Mxz=M0):
        theta_deg =  0°  →  pure shear: fault normal = z, slip = x
        theta_deg = 45°  →  compressive: Mxx=-M0, Mzz=+M0, Mxz=0
        theta_deg = 90°  →  pure shear: fault normal = x, slip = z
    """
    theta = np.deg2rad(theta_deg)
    c2    = np.cos(2.0 * theta)
    s2    = np.sin(2.0 * theta)

    m_avg = 0.5 * (mt.Mxx + mt.Mzz)
    m_dev = 0.5 * (mt.Mxx - mt.Mzz)

    return MomentTensor2D(
        Mxx=float(m_avg + m_dev * c2 - mt.Mxz * s2),
        Mzz=float(m_avg - m_dev * c2 + mt.Mxz * s2),
        Mxz=float(m_dev * s2         + mt.Mxz * c2),
    )


# ==============================================================================
# 2. BASE 2D SOURCE TENSORS
# ==============================================================================

def base_double_couple_2d() -> MomentTensor2D:
    """
    Unit 2D double-couple tensor (Aki & Richards convention).

        M_base = [[0, 1], [1, 0]]

    This is pure shear with zero trace and Frobenius norm sqrt(2).
    Fault normal = z-hat, slip direction = x-hat (horizontal fault,
    horizontal slip).

    Scale by scalar_moment to get physical units [N·m].
    Rotate with rotate_moment_tensor_2d() to change fault orientation.
    """
    return MomentTensor2D(Mxx=0.0, Mzz=0.0, Mxz=1.0)


def isotropic_tensor_2d() -> MomentTensor2D:
    """
    Unit 2D isotropic (explosive) tensor.

        M_iso = [[1, 0], [0, 1]]

    Equal P-wave radiation in all directions; zero shear (Mxz=0).
    Frobenius norm = sqrt(2).
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
        theta_deg=0  →  fault normal = z, slip = x.
        theta_deg=45 →  Mxx=-M0, Mzz=+M0, Mxz=0.
    scalar_moment :
        Scalar seismic moment M0 [N·m]. Must be positive.
    """
    if scalar_moment <= 0.0:
        raise ValueError(f"scalar_moment must be positive, got {scalar_moment}.")
    return rotate_moment_tensor_2d(base_double_couple_2d(), theta_deg).scaled(scalar_moment)


def build_isotropic_source_tensor_2d(scalar_moment: float) -> MomentTensor2D:
    """
    Build a scaled 2D isotropic (explosive/implosive) tensor.

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

    values[n] is the moment-rate amplitude at step n, injected into the
    updated stress field at t_sigma[n+1] = (n+1)*dt (see solver_numpy.py).

    The physical source history is:
        M_dot(t_sigma[n+1]) ≈ scalar_moment * values[n] / dt
    """
    values: np.ndarray
    dt:     float
    t0:     float
    kind:   str

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

        values.flags.writeable = False
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "dt",     float(self.dt))
        object.__setattr__(self, "t0",     float(self.t0))
        object.__setattr__(self, "kind",   str(self.kind))

    @property
    def nt(self) -> int:
        return int(self.values.size)

    @property
    def t(self) -> np.ndarray:
        """
        Integer time axis: t[n] = n*dt.

        Note: values[n] is injected at t_sigma[n+1] = (n+1)*dt,
        not at t[n]. See solver_numpy.py source convention.
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
    Ricker wavelet (second derivative of Gaussian) with unit amplitude.

    W(t) = (1 - 2π²f0²(t-t0)²) exp(-π²f0²(t-t0)²),   t0 = 1.2/f0
    """
    if nt < 2:
        raise ValueError(f"nt must be >= 2, got {nt}.")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}.")
    if f0_hz <= 0.0:
        raise ValueError(f"f0_hz must be positive, got {f0_hz}.")

    t0  = 1.2 / f0_hz
    t   = np.arange(nt, dtype=np.float64) * dt
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
    First time derivative of the Ricker wavelet, unit amplitude.

    dW/dt = 2πf0 · x · (2x² - 3) · exp(-x²),   x = πf0(t - t0)
    """
    if nt < 2:
        raise ValueError(f"nt must be >= 2, got {nt}.")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}.")
    if f0_hz <= 0.0:
        raise ValueError(f"f0_hz must be positive, got {f0_hz}.")

    t0 = 1.2 / f0_hz
    t  = np.arange(nt, dtype=np.float64) * dt
    x  = np.pi * f0_hz * (t - t0)

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
    Build a source time function of a given derivative order.

    derivative_order=0 : Ricker wavelet
    derivative_order=1 : first derivative of Ricker wavelet
    """
    if derivative_order == 0:
        return ricker_wavelet(nt=nt, dt=dt, f0_hz=f0_hz)
    if derivative_order == 1:
        return ricker_derivative_wavelet(nt=nt, dt=dt, f0_hz=f0_hz)
    raise ValueError(
        f"Unsupported derivative_order={derivative_order}. Only 0 or 1 are supported."
    )


# ==============================================================================
# 4. EMBEDDED 2D SOURCE
# ==============================================================================

@dataclass(frozen=True)
class EmbeddedSource2D:
    """
    A 2D point source embedded at the nearest grid node.

    Stores the moment tensor, source time function, and the snapped grid
    position. Supports DC, isotropic, and any other 2D moment tensor.

    Attributes
    ----------
    x_m, z_m :
        Requested physical position [m].
    x_embedded_m, z_embedded_m :
        Actual snapped position [m] (nearest grid node).
    ix, iz :
        Grid indices of the snapped position.
    m2d :
        Scaled moment tensor [N·m].
    stf :
        Source time function (unit-amplitude wavelet).
    scalar_moment :
        Seismic scalar moment M0 [N·m] used to scale m2d.
    label :
        Human-readable description.
    """
    x_m:           float
    z_m:           float
    x_embedded_m:  float
    z_embedded_m:  float
    ix:            int
    iz:            int
    m2d:           MomentTensor2D
    stf:           SourceTimeFunction
    scalar_moment: float
    label:         str = "2D source"

    def summary(self) -> str:
        return (
            f"{self.label}\n"
            f"  requested position : ({self.x_m:.2f}, {self.z_m:.2f}) m\n"
            f"  embedded position  : ({self.x_embedded_m:.2f}, {self.z_embedded_m:.2f}) m\n"
            f"  grid indices       : (ix={self.ix}, iz={self.iz})\n"
            f"  scalar moment M0   : {self.scalar_moment:.6e} N·m\n"
            f"  STF                : {self.stf.summary()}\n"
            f"  tensor trace       : {self.m2d.trace():.6e}\n"
            f"  tensor ||M||_F     : {self.m2d.frobenius_norm():.6e}"
        )

    def __repr__(self) -> str:
        return self.summary()


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
    label: str = "2D source",
) -> EmbeddedSource2D:
    """
    Embed a point source onto the nearest grid node.

    The moment tensor mt2d should already be scaled by scalar_moment.
    scalar_moment is stored separately for bookkeeping.
    """
    if nt != grid.nt:
        raise ValueError(f"nt={nt} must match grid.nt={grid.nt}.")
    if not np.isclose(dt, grid.dt):
        raise ValueError(f"dt={dt:.6e} must match grid.dt={grid.dt:.6e}.")
    if scalar_moment <= 0.0:
        raise ValueError(f"scalar_moment must be positive, got {scalar_moment}.")

    ix, iz, x_embedded_m, z_embedded_m = grid.get_closest_node(x_m, z_m)

    stf = build_source_time_function(
        nt=nt,
        dt=dt,
        f0_hz=f0_hz,
        derivative_order=derivative_order,
    )

    return EmbeddedSource2D(
        x_m=float(x_m),
        z_m=float(z_m),
        x_embedded_m=x_embedded_m,
        z_embedded_m=z_embedded_m,
        ix=ix,
        iz=iz,
        m2d=mt2d,
        stf=stf,
        scalar_moment=scalar_moment,
        label=label,
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
) -> EmbeddedSource2D:
    """
    Build a 2D double-couple source (Aki & Richards convention).

    Parameters
    ----------
    theta_deg :
        Fault rotation in the x-z plane [degrees].
        theta_deg=0  →  fault normal = z-hat, slip = x-hat.
        theta_deg=45 →  Mxx=-M0, Mzz=+M0, Mxz=0.
    scalar_moment :
        Scalar seismic moment M0 [N·m].
    """
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
        label=f"2D double-couple (theta={theta_deg:.1f}°, M0={scalar_moment:.2e} N·m)",
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
) -> EmbeddedSource2D:
    """
    Build a 2D isotropic (explosive) source.

    Parameters
    ----------
    scalar_moment :
        Scalar moment M0 [N·m]. Positive → explosive, would be negative
        for implosive (pass a negative value via scaled mt2d if needed).
    """
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
        label=f"2D isotropic source (M0={scalar_moment:.2e} N·m)",
    )


# ==============================================================================
# 6. SELF-TEST
# ==============================================================================

def _self_test() -> None:
    grid = Grid2D(
        nx=101, nz=81, dx=10.0, dz=10.0,
        nt=400, dt=1.0e-3, x0=0.0, z0=0.0,
    )

    # ── base DC tensor ─────────────────────────────────────────────────────────
    mt0 = base_double_couple_2d()
    assert np.isclose(mt0.Mxx,   0.0, atol=1e-12)
    assert np.isclose(mt0.Mzz,   0.0, atol=1e-12)
    assert np.isclose(mt0.Mxz,   1.0, atol=1e-12)
    assert np.isclose(mt0.trace(), 0.0, atol=1e-12)

    # ── rotation at 45° ────────────────────────────────────────────────────────
    mt45 = rotate_moment_tensor_2d(mt0, 45.0)
    assert np.isclose(mt45.Mxx, -1.0, atol=1e-12)
    assert np.isclose(mt45.Mzz,  1.0, atol=1e-12)
    assert np.isclose(mt45.Mxz,  0.0, atol=1e-12)
    assert np.isclose(mt45.trace(), 0.0, atol=1e-12)

    # ── Frobenius norm is rotation-invariant ───────────────────────────────────
    mt30 = rotate_moment_tensor_2d(mt0, 30.0)
    assert np.isclose(mt0.frobenius_norm(), mt30.frobenius_norm(), atol=1e-12)

    # ── STF ────────────────────────────────────────────────────────────────────
    stf = build_source_time_function(nt=grid.nt, dt=grid.dt, f0_hz=8.0, derivative_order=0)
    assert stf.nt == grid.nt
    assert np.isclose(stf.dt, grid.dt)
    assert stf.peak_amplitude() > 0.0
    assert np.isclose(stf.t[0], 0.0)
    assert np.isclose(stf.t[-1], (grid.nt - 1) * grid.dt)

    stf_d = build_source_time_function(nt=grid.nt, dt=grid.dt, f0_hz=8.0, derivative_order=1)
    assert stf_d.peak_amplitude() > 0.0

    # ── DC source ──────────────────────────────────────────────────────────────
    src = build_dc_source(
        grid=grid, x_m=400.0, z_m=300.0,
        theta_deg=30.0, scalar_moment=1.0e10,
        nt=grid.nt, dt=grid.dt, f0_hz=8.0, derivative_order=0,
    )
    assert isinstance(src, EmbeddedSource2D)
    assert 0 <= src.ix < grid.nx
    assert 0 <= src.iz < grid.nz
    assert np.isclose(src.stf.dt, grid.dt)
    assert np.isclose(src.scalar_moment, 1.0e10)

    # ── isotropic source ───────────────────────────────────────────────────────
    iso = build_isotropic_source(
        grid=grid, x_m=500.0, z_m=400.0,
        scalar_moment=2.0e10,
        nt=grid.nt, dt=grid.dt, f0_hz=8.0, derivative_order=1,
    )
    assert isinstance(iso, EmbeddedSource2D)
    assert np.isclose(iso.m2d.Mxx, iso.m2d.Mzz)
    assert np.isclose(iso.m2d.Mxz, 0.0)

    print("Base DC tensor      : OK")
    print("Tensor rotation     : OK")
    print("Frobenius invariance: OK")
    print("Source time function: OK")
    print("DC source builder   : OK")
    print("Isotropic builder   : OK")
    print("Self-test PASSED")


if __name__ == "__main__":
    _self_test()