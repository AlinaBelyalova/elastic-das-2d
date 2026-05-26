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

    Matrix form
    -----------
        [ Mxx  Mxz ]
        [ Mxz  Mzz ]
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
        return float(self.Mxx + self.Mzz)

    def frobenius_norm(self) -> float:
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
            f"  trace = {self.trace():.6e}\n"
            f"  ||M||_F = {self.frobenius_norm():.6e}"
        )

    def __repr__(self) -> str:
        return self.summary()


def rotate_moment_tensor_2d(
    mt: MomentTensor2D,
    theta_deg: float,
) -> MomentTensor2D:
    """
    Rotate a symmetric 2D tensor in the fixed x-z coordinate system.
    """
    theta = np.deg2rad(theta_deg)
    c2 = np.cos(2.0 * theta)
    s2 = np.sin(2.0 * theta)

    m_avg = 0.5 * (mt.Mxx + mt.Mzz)
    m_dev = 0.5 * (mt.Mxx - mt.Mzz)

    Mxx_rot = m_avg + m_dev * c2 - mt.Mxz * s2
    Mzz_rot = m_avg - m_dev * c2 + mt.Mxz * s2
    Mxz_rot = m_dev * s2 + mt.Mxz * c2

    return MomentTensor2D(
        Mxx=float(Mxx_rot),
        Mzz=float(Mzz_rot),
        Mxz=float(Mxz_rot),
    )


# ==============================================================================
# 2. BASE 2D SOURCE TENSORS
# ==============================================================================

def base_double_couple_2d() -> MomentTensor2D:
    """
    Base 2D double-couple-like tensor corresponding to pure shear:

        [ 0  1 ]
        [ 1  0 ]
    """
    return MomentTensor2D(
        Mxx=0.0,
        Mzz=0.0,
        Mxz=1.0,
    )


def isotropic_tensor_2d() -> MomentTensor2D:
    """
    2D isotropic tensor:

        [ 1  0 ]
        [ 0  1 ]
    """
    return MomentTensor2D(
        Mxx=1.0,
        Mzz=1.0,
        Mxz=0.0,
    )


def build_rotated_double_couple_2d(
    theta_deg: float,
    scalar_moment: float,
) -> MomentTensor2D:
    if scalar_moment <= 0.0:
        raise ValueError(f"scalar_moment must be positive, got {scalar_moment}.")

    mt0 = base_double_couple_2d()
    mt_rot = rotate_moment_tensor_2d(mt0, theta_deg)
    return mt_rot.scaled(scalar_moment)


def build_isotropic_source_tensor_2d(
    scalar_moment: float,
) -> MomentTensor2D:
    if scalar_moment <= 0.0:
        raise ValueError(f"scalar_moment must be positive, got {scalar_moment}.")
    return isotropic_tensor_2d().scaled(scalar_moment)


# ==============================================================================
# 3. SOURCE TIME FUNCTION
# ==============================================================================

@dataclass(frozen=True)
class SourceTimeFunction:
    """
    Discrete source time function sampled on the solver time grid.
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
        return np.arange(self.nt, dtype=np.float64) * self.dt

    def peak_amplitude(self) -> float:
        return float(np.max(np.abs(self.values)))

    def summary(self) -> str:
        return (
            f"SourceTimeFunction(kind={self.kind}, nt={self.nt}, dt={self.dt:.3e}, "
            f"t0={self.t0:.3e}, peak={self.peak_amplitude():.3e})"
        )

    def __repr__(self) -> str:
        return self.summary()


def ricker_wavelet(
    *,
    nt: int,
    dt: float,
    f0_hz: float,
    amplitude: float = 1.0,
) -> SourceTimeFunction:
    if nt < 2:
        raise ValueError(f"nt must be >= 2, got {nt}.")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}.")
    if f0_hz <= 0.0:
        raise ValueError(f"f0_hz must be positive, got {f0_hz}.")
    if amplitude <= 0.0:
        raise ValueError(f"amplitude must be positive, got {amplitude}.")

    t = np.arange(nt, dtype=np.float64) * dt
    t0 = 1.2 / f0_hz
    arg = (np.pi * f0_hz * (t - t0)) ** 2
    values = amplitude * (1.0 - 2.0 * arg) * np.exp(-arg)

    return SourceTimeFunction(
        values=values,
        dt=dt,
        t0=t0,
        kind="ricker",
    )


def ricker_derivative_wavelet(
    *,
    nt: int,
    dt: float,
    f0_hz: float,
    amplitude: float = 1.0,
) -> SourceTimeFunction:
    """
    Build the first time derivative of a Ricker wavelet analytically.

    If
        W(t) = (1 - 2 x^2) exp(-x^2),
        x = pi f0 (t - t0),

    then
        dW/dt = 2 pi f0 x (2 x^2 - 3) exp(-x^2)
    """
    if nt < 2:
        raise ValueError(f"nt must be >= 2, got {nt}.")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}.")
    if f0_hz <= 0.0:
        raise ValueError(f"f0_hz must be positive, got {f0_hz}.")
    if amplitude <= 0.0:
        raise ValueError(f"amplitude must be positive, got {amplitude}.")

    t = np.arange(nt, dtype=np.float64) * dt
    t0 = 1.2 / f0_hz

    x = np.pi * f0_hz * (t - t0)
    values = amplitude * (2.0 * np.pi * f0_hz * x * (2.0 * x**2 - 3.0)) * np.exp(-x**2)

    return SourceTimeFunction(
        values=values,
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
    amplitude: float = 1.0,
) -> SourceTimeFunction:
    if derivative_order == 0:
        return ricker_wavelet(
            nt=nt,
            dt=dt,
            f0_hz=f0_hz,
            amplitude=amplitude,
        )
    if derivative_order == 1:
        return ricker_derivative_wavelet(
            nt=nt,
            dt=dt,
            f0_hz=f0_hz,
            amplitude=amplitude,
        )

    raise ValueError(
        f"Unsupported derivative_order={derivative_order}. Only 0 or 1 are supported."
    )


# ==============================================================================
# 4. EMBEDDED 2D SOURCE OBJECT
# ==============================================================================

@dataclass(frozen=True)
class DoubleCoupleSource2D:
    """
    2D source embedded into a Grid2D.
    """
    x_m: float
    z_m: float
    x_embedded_m: float
    z_embedded_m: float
    ix: int
    iz: int
    m2d: MomentTensor2D
    stf: SourceTimeFunction
    label: str = "2D source"

    def summary(self) -> str:
        return (
            f"{self.label}\n"
            f"  requested position : ({self.x_m:.2f}, {self.z_m:.2f}) m\n"
            f"  embedded position  : ({self.x_embedded_m:.2f}, {self.z_embedded_m:.2f}) m\n"
            f"  grid indices       : (ix={self.ix}, iz={self.iz})\n"
            f"  STF                : {self.stf.summary()}\n"
            f"  tensor trace       : {self.m2d.trace():.6e}\n"
            f"  tensor norm        : {self.m2d.frobenius_norm():.6e}"
        )

    def __repr__(self) -> str:
        return self.summary()


def embed_point_on_grid(
    grid: Grid2D,
    *,
    x_m: float,
    z_m: float,
) -> tuple[int, int, float, float]:
    """
    Snap a physical point to the nearest solver grid support point.

    Delegates grid-specific logic to Grid2D.
    """
    return grid.get_closest_node(x_m, z_m)


def build_source_2d(
    *,
    grid: Grid2D,
    x_m: float,
    z_m: float,
    mt2d: MomentTensor2D,
    nt: int,
    dt: float,
    f0_hz: float,
    derivative_order: int = 0,
    stf_amplitude: float = 1.0,
    label: str = "2D source",
) -> DoubleCoupleSource2D:
    if nt != grid.nt:
        raise ValueError(f"nt={nt} must match grid.nt={grid.nt}.")
    if not np.isclose(dt, grid.dt):
        raise ValueError(f"dt={dt:.6e} must match grid.dt={grid.dt:.6e}.")
    if stf_amplitude <= 0.0:
        raise ValueError(f"stf_amplitude must be positive, got {stf_amplitude}.")

    ix, iz, x_embedded_m, z_embedded_m = embed_point_on_grid(
        grid,
        x_m=x_m,
        z_m=z_m,
    )

    stf = build_source_time_function(
        nt=nt,
        dt=dt,
        f0_hz=f0_hz,
        derivative_order=derivative_order,
        amplitude=stf_amplitude,
    )

    return DoubleCoupleSource2D(
        x_m=float(x_m),
        z_m=float(z_m),
        x_embedded_m=x_embedded_m,
        z_embedded_m=z_embedded_m,
        ix=ix,
        iz=iz,
        m2d=mt2d,
        stf=stf,
        label=label,
    )


# ==============================================================================
# 5. HIGH-LEVEL CONVENIENCE BUILDERS
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
    stf_amplitude: float = 1.0,
) -> DoubleCoupleSource2D:
    mt2d = build_rotated_double_couple_2d(
        theta_deg=theta_deg,
        scalar_moment=scalar_moment,
    )

    return build_source_2d(
        grid=grid,
        x_m=x_m,
        z_m=z_m,
        mt2d=mt2d,
        nt=nt,
        dt=dt,
        f0_hz=f0_hz,
        derivative_order=derivative_order,
        stf_amplitude=stf_amplitude,
        label=f"2D double-couple source (theta={theta_deg:.1f} deg)",
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
    stf_amplitude: float = 1.0,
) -> DoubleCoupleSource2D:
    mt2d = build_isotropic_source_tensor_2d(
        scalar_moment=scalar_moment,
    )

    return build_source_2d(
        grid=grid,
        x_m=x_m,
        z_m=z_m,
        mt2d=mt2d,
        nt=nt,
        dt=dt,
        f0_hz=f0_hz,
        derivative_order=derivative_order,
        stf_amplitude=stf_amplitude,
        label="2D isotropic source",
    )


# ==============================================================================
# 6. SELF-TEST
# ==============================================================================

def _self_test() -> None:
    from src.grid import Grid2D

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

    mt0 = base_double_couple_2d()
    assert np.isclose(mt0.Mxx, 0.0, atol=1e-12)
    assert np.isclose(mt0.Mzz, 0.0, atol=1e-12)
    assert np.isclose(mt0.Mxz, 1.0, atol=1e-12)
    assert np.isclose(mt0.trace(), 0.0, atol=1e-12)

    mt45 = rotate_moment_tensor_2d(mt0, 45.0)
    assert np.isclose(mt45.Mxx, -1.0, atol=1e-12)
    assert np.isclose(mt45.Mzz,  1.0, atol=1e-12)
    assert np.isclose(mt45.Mxz,  0.0, atol=1e-12)
    assert np.isclose(mt45.trace(), 0.0, atol=1e-12)

    mt30 = rotate_moment_tensor_2d(mt0, 30.0)
    assert np.isclose(mt0.frobenius_norm(), mt30.frobenius_norm(), atol=1e-12)

    stf = build_source_time_function(
        nt=grid.nt,
        dt=grid.dt,
        f0_hz=8.0,
        derivative_order=0,
    )
    assert stf.nt == grid.nt
    assert np.isclose(stf.dt, grid.dt)
    assert stf.peak_amplitude() > 0.0

    stf_d = build_source_time_function(
        nt=grid.nt,
        dt=grid.dt,
        f0_hz=8.0,
        derivative_order=1,
    )
    assert stf_d.nt == grid.nt
    assert np.isclose(stf_d.dt, grid.dt)
    assert stf_d.peak_amplitude() > 0.0

    src = build_dc_source(
        grid=grid,
        x_m=400.0,
        z_m=300.0,
        theta_deg=30.0,
        scalar_moment=1.0e10,
        nt=grid.nt,
        dt=grid.dt,
        f0_hz=8.0,
        derivative_order=0,
    )
    assert 0 <= src.ix < grid.nx
    assert 0 <= src.iz < grid.nz
    assert np.isclose(src.stf.dt, grid.dt)

    iso = build_isotropic_source(
        grid=grid,
        x_m=500.0,
        z_m=400.0,
        scalar_moment=2.0e10,
        nt=grid.nt,
        dt=grid.dt,
        f0_hz=8.0,
        derivative_order=1,
    )
    assert np.isclose(iso.m2d.Mxx, iso.m2d.Mzz)
    assert np.isclose(iso.m2d.Mxz, 0.0)

    print("2D base tensor: OK")
    print("2D tensor rotation: OK")
    print("Source time function: OK")
    print("2D source builders: OK")
    print("Self-test PASSED")


if __name__ == "__main__":
    _self_test()