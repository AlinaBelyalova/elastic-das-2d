# ==============================================================================
# src/analytical_moment_tensor.py
#
# Homogeneous full-space analytical reference for a symmetric moment tensor.
# The 2D solution is obtained by integrating the 3D point-source solution
# along the invariant y direction. Based on the near/intermediate/far-field
# expressions in Haipeng Li's AnalyticalSolution.py (Aki & Richards, 2002).
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LineIntegrationInfo:
    y_max_m: float
    dy_m: float
    n_y: int
    points_per_s_wavelength: float
    line_extent_factor: float


def ricker_moment(
    time_s: np.ndarray,
    *,
    f0_hz: float,
    t0_s: float | None = None,
) -> np.ndarray:
    """Unit-amplitude Ricker moment history."""
    t = np.asarray(time_s, dtype=np.float64)
    if not np.isfinite(f0_hz) or f0_hz <= 0.0:
        raise ValueError("f0_hz must be finite and positive.")
    if t0_s is None:
        t0_s = 1.2 / float(f0_hz)
    a_tau = np.pi * float(f0_hz) * (t - float(t0_s))
    return (1.0 - 2.0 * a_tau**2) * np.exp(-a_tau**2)


def ricker_moment_derivative(
    time_s: np.ndarray,
    *,
    f0_hz: float,
    t0_s: float | None = None,
) -> np.ndarray:
    """Exact first derivative of the unit-amplitude Ricker history."""
    t = np.asarray(time_s, dtype=np.float64)
    if t0_s is None:
        t0_s = 1.2 / float(f0_hz)
    a = np.pi * float(f0_hz)
    tau = t - float(t0_s)
    q = a * tau
    return -2.0 * a**2 * tau * (3.0 - 2.0 * q**2) * np.exp(-q**2)


def _near_field_integral(
    time_s: np.ndarray,
    *,
    lower_s: np.ndarray,
    upper_s: np.ndarray,
    f0_hz: float,
    t0_s: float,
) -> np.ndarray:
    r"""Evaluate integral_lower^upper tau*m(time-tau) d tau exactly."""
    t = np.asarray(time_s, dtype=np.float64)[None, :]
    lower = np.asarray(lower_s, dtype=np.float64)
    upper = np.asarray(upper_s, dtype=np.float64)
    if lower.ndim != 1 or upper.ndim != 1 or lower.shape != upper.shape:
        raise ValueError("lower_s and upper_s must be matching 1D arrays.")
    if np.any(upper < lower):
        raise ValueError("upper_s must be >= lower_s.")

    a = np.pi * float(f0_hz)
    inv_two_a2 = 1.0 / (2.0 * a**2)

    def i0(u: np.ndarray) -> np.ndarray:
        x = u - float(t0_s)
        return x * np.exp(-(a * x) ** 2)

    def i1(u: np.ndarray) -> np.ndarray:
        x = u - float(t0_s)
        e = np.exp(-(a * x) ** 2)
        return e * (x**2 + inv_two_a2 + float(t0_s) * x)

    u_hi = t - lower[:, None]
    u_lo = t - upper[:, None]
    return t * (i0(u_hi) - i0(u_lo)) - (i1(u_hi) - i1(u_lo))


def _validate(
    vp_m_s: float,
    vs_m_s: float,
    rho_kg_m3: float,
    time_s: np.ndarray,
    f0_hz: float,
    moment_tensor_nm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    for name, value in (
        ("vp_m_s", vp_m_s),
        ("vs_m_s", vs_m_s),
        ("rho_kg_m3", rho_kg_m3),
        ("f0_hz", f0_hz),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    if vp_m_s <= vs_m_s:
        raise ValueError("vp_m_s must exceed vs_m_s.")

    t = np.asarray(time_s, dtype=np.float64)
    if t.ndim != 1 or t.size < 3 or np.any(np.diff(t) <= 0.0):
        raise ValueError("time_s must be a strictly increasing 1D array.")

    m = np.array(moment_tensor_nm, dtype=np.float64, copy=True)
    if m.shape != (3, 3) or not np.all(np.isfinite(m)):
        raise ValueError("moment_tensor_nm must be a finite (3,3) array.")
    tolerance = 1.0e-12 * max(1.0, float(np.max(np.abs(m))))
    if not np.allclose(m, m.T, rtol=0.0, atol=tolerance):
        raise ValueError("moment_tensor_nm must be symmetric.")
    return t, m


def displacement_3d_many_y(
    *,
    vp_m_s: float,
    vs_m_s: float,
    rho_kg_m3: float,
    x_m: float,
    y_m: np.ndarray,
    z_m: float,
    time_s: np.ndarray,
    f0_hz: float,
    moment_tensor_nm: np.ndarray,
    t0_s: float | None = None,
) -> np.ndarray:
    """3D full-space displacement, shape (ny,3,nt)."""
    t, m = _validate(vp_m_s, vs_m_s, rho_kg_m3, time_s, f0_hz, moment_tensor_nm)
    if t0_s is None:
        t0_s = 1.2 / float(f0_hz)

    y = np.asarray(y_m, dtype=np.float64)
    if y.ndim != 1 or not np.all(np.isfinite(y)):
        raise ValueError("y_m must be a finite 1D array.")

    xyz = np.column_stack((np.full(y.size, x_m), y, np.full(y.size, z_m)))
    radius = np.linalg.norm(xyz, axis=1)
    if np.any(radius <= 0.0):
        raise ValueError("The Green function is singular at the source.")
    r = xyz / radius[:, None]

    trace_m = float(np.trace(m))
    mr = np.einsum("ij,nj->ni", m, r)
    q = np.einsum("ni,ij,nj->n", r, m, r)
    rq = r * q[:, None]

    an = 15.0 * rq - 3.0 * trace_m * r - 6.0 * mr
    aip = 6.0 * rq - trace_m * r - 2.0 * mr
    ais = -6.0 * rq + trace_m * r + 3.0 * mr
    afp = rq
    afs = mr - rq

    tp = radius / float(vp_m_s)
    ts = radius / float(vs_m_s)
    near_h = _near_field_integral(
        t, lower_s=tp, upper_s=ts, f0_hz=f0_hz, t0_s=float(t0_s)
    )
    ip_h = ricker_moment(t[None, :] - tp[:, None], f0_hz=f0_hz, t0_s=t0_s)
    is_h = ricker_moment(t[None, :] - ts[:, None], f0_hz=f0_hz, t0_s=t0_s)
    fp_h = ricker_moment_derivative(
        t[None, :] - tp[:, None], f0_hz=f0_hz, t0_s=t0_s
    )
    fs_h = ricker_moment_derivative(
        t[None, :] - ts[:, None], f0_hz=f0_hz, t0_s=t0_s
    )

    c = 1.0 / (4.0 * np.pi * float(rho_kg_m3))
    rr = radius[:, None, None]
    return (
        c * an[:, :, None] * near_h[:, None, :] / rr**4
        + c / vp_m_s**2 * aip[:, :, None] * ip_h[:, None, :] / rr**2
        + c / vs_m_s**2 * ais[:, :, None] * is_h[:, None, :] / rr**2
        + c / vp_m_s**3 * afp[:, :, None] * fp_h[:, None, :] / rr
        + c / vs_m_s**3 * afs[:, :, None] * fs_h[:, None, :] / rr
    )


def displacement_2d(
    *,
    vp_m_s: float,
    vs_m_s: float,
    rho_kg_m3: float,
    x_m: float,
    z_m: float,
    time_s: np.ndarray,
    f0_hz: float,
    moment_tensor_nm: np.ndarray,
    t0_s: float | None = None,
    points_per_s_wavelength: float = 20.0,
    line_extent_factor: float = 1.25,
    chunk_size: int = 128,
) -> tuple[np.ndarray, LineIntegrationInfo]:
    """2D line-source displacement (Ux,Uz), shape (2,nt)."""
    t, m = _validate(vp_m_s, vs_m_s, rho_kg_m3, time_s, f0_hz, moment_tensor_nm)
    if t0_s is None:
        t0_s = 1.2 / float(f0_hz)
    if points_per_s_wavelength < 8.0:
        raise ValueError("points_per_s_wavelength must be >= 8.")
    if line_extent_factor <= 1.0:
        raise ValueError("line_extent_factor must be > 1.")
    if int(chunk_size) != chunk_size or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")

    base_radius = float(np.hypot(x_m, z_m))
    if base_radius <= 0.0:
        raise ValueError("Receiver cannot coincide with source.")

    target_dy = float(vs_m_s) / float(f0_hz) / float(points_per_s_wavelength)
    effective_latest = max(
        float(t[-1]) - float(t0_s) + 4.0 / float(f0_hz),
        1.0 / float(f0_hz),
    )
    causal_radius = max(
        float(line_extent_factor) * float(vp_m_s) * effective_latest,
        1.05 * base_radius,
    )
    y_max = float(np.sqrt(max(causal_radius**2 - base_radius**2, target_dy**2)))
    n_intervals = max(2, int(np.ceil(2.0 * y_max / target_dy)))
    y = np.linspace(-y_max, y_max, n_intervals + 1)
    dy = float(y[1] - y[0])
    weights = np.full(y.size, dy)
    weights[[0, -1]] *= 0.5

    integrated = np.zeros((3, t.size), dtype=np.float64)
    for start in range(0, y.size, int(chunk_size)):
        stop = min(start + int(chunk_size), y.size)
        chunk = displacement_3d_many_y(
            vp_m_s=vp_m_s,
            vs_m_s=vs_m_s,
            rho_kg_m3=rho_kg_m3,
            x_m=x_m,
            y_m=y[start:stop],
            z_m=z_m,
            time_s=t,
            f0_hz=f0_hz,
            moment_tensor_nm=m,
            t0_s=t0_s,
        )
        integrated += np.einsum("n,nit->it", weights[start:stop], chunk)

    info = LineIntegrationInfo(
        y_max_m=y_max,
        dy_m=dy,
        n_y=int(y.size),
        points_per_s_wavelength=float(points_per_s_wavelength),
        line_extent_factor=float(line_extent_factor),
    )
    return integrated[[0, 2], :], info


def kinematics_2d(**kwargs) -> tuple[dict[str, np.ndarray], LineIntegrationInfo]:
    """Return displacement, velocity, acceleration; each shape (2,nt)."""
    time_s = np.asarray(kwargs["time_s"], dtype=np.float64)
    displacement, info = displacement_2d(**kwargs)
    velocity = np.gradient(displacement, time_s, axis=1, edge_order=2)
    acceleration = np.gradient(velocity, time_s, axis=1, edge_order=2)
    return {
        "displacement": displacement,
        "velocity": velocity,
        "acceleration": acceleration,
    }, info


def _self_test() -> None:
    f0 = 8.0
    t0 = 1.2 / f0
    t = np.linspace(0.0, 0.8, 2001)
    lower = np.array([0.08])
    upper = np.array([0.17])
    exact = _near_field_integral(
        t, lower_s=lower, upper_s=upper, f0_hz=f0, t0_s=t0
    )[0]
    tau = np.linspace(lower[0], upper[0], 2001)
    direct = np.trapezoid(
        tau[:, None]
        * ricker_moment(t[None, :] - tau[:, None], f0_hz=f0, t0_s=t0),
        tau,
        axis=0,
    )
    rel = np.linalg.norm(exact - direct) / max(np.linalg.norm(direct), 1e-30)
    assert rel < 5e-6, rel

    m = np.array(
        [[0.6e12, 0.0, 0.8e12], [0.0, 0.0, 0.0], [0.8e12, 0.0, -0.6e12]]
    )
    result, info = kinematics_2d(
        vp_m_s=4000.0,
        vs_m_s=2300.0,
        rho_kg_m3=2500.0,
        x_m=300.0,
        z_m=200.0,
        time_s=t,
        f0_hz=f0,
        moment_tensor_nm=m,
        t0_s=t0,
        points_per_s_wavelength=16.0,
        line_extent_factor=1.2,
        chunk_size=64,
    )
    for value in result.values():
        assert value.shape == (2, t.size)
        assert np.all(np.isfinite(value))
    print(f"Near-field primitive: OK (relative error={rel:.3e})")
    print(f"2D line integration: OK (n_y={info.n_y}, dy={info.dy_m:.3f} m)")
    print("analytical_moment_tensor.py: all self-tests passed")


if __name__ == "__main__":
    _self_test()
