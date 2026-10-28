"""Bounded-grid SAT energy, exact time-step transpose and impedance derivatives."""

from dataclasses import replace
import numpy as np
import pytest

from src.das.receivers import Receivers2D
from src.das.strain import compute_axial_strain_rate
from src.sbp_sat import PointReceiver, StressPointSource
from src.sbp_sat.rectangular import (
    BoundaryConditions,
    RectangularSBPSAT,
    prepare_rectangular_model,
)
from src.sbp_sat.rectangular_observations import build_rectangular_das_observation

FIELDS = ("vx", "vz", "sxx", "szz", "sxz")
FREE = BoundaryConditions(top="free", bottom="free", left="free", right="free")


def case(boundaries=None):
    x, z = np.indices((11, 12))
    model = prepare_rectangular_model(
        3.2 + 0.1 * np.sin(x + z), 1.5 + 0.06 * np.cos(z), 2.2 + 0.1 * np.sin(x - z)
    )
    return RectangularSBPSAT(model, 1.2, 0.9, 0.03, boundaries)


def random_state(f, rng):
    s = f.zero_state()
    for name in FIELDS:
        a = getattr(s, name)
        a[:] = rng.normal(size=a.shape)
    return s


def state_dot(a, b):
    return sum(np.vdot(getattr(a, name), getattr(b, name)) for name in FIELDS)


def boundary_flux(f, vx, vz):
    """Independent sum of Zp vn² + Zs vt², using native face quadratures."""
    x, z, Z = f.op_x, f.op_z, f._Z
    result = 0.0
    for side, p, index in (("left", x.p_left, 0), ("right", x.p_right, -1)):
        result += np.sum(z.H_N * Z[side + "_p"] * (p @ vx) ** 2)
        result += np.sum(z.H_M * Z[side + "_s"] * vz[index, :] ** 2)
    for side, p, index in (("top", z.p_left, 0), ("bottom", z.p_right, -1)):
        result += np.sum(x.H_N * Z[side + "_p"] * (vz @ p) ** 2)
        result += np.sum(x.H_M * Z[side + "_s"] * vx[:, index] ** 2)
    return float(result)


@pytest.mark.parametrize(
    "face", ["none", "top", "bottom", "left", "right", "all", "production"]
)
def test_exact_dissipation_identity_each_face_and_corners(face):
    bc = (
        BoundaryConditions(top="absorbing")
        if face == "all"
        else None
        if face == "production"
        else FREE
        if face == "none"
        else replace(FREE, **{face: "absorbing"})
    )
    f = case(bc)
    rng = np.random.default_rng(391)
    s = random_state(f, rng)
    for _ in range(20):
        e0 = f.energy(s)
        out = f.step(s)
        vx, vz = (s.vx + out.vx) / 2, (s.vz + out.vz) / 2
        expected = boundary_flux(f, vx, vz)
        assert expected >= 0 and f.energy(out) > 0
        np.testing.assert_allclose(
            f.dissipation(vx, vz), expected, atol=2e-13, rtol=3e-15
        )
        assert abs(f.energy(out) - e0 + f.dt * expected) < 3e-13 * e0
        assert f.energy(out) <= e0 * (1 + 3e-14)
        s = out


def test_rank_solves_and_transposes_against_actual_linear_operator():
    f = case(BoundaryConditions(top="absorbing"))
    rng = np.random.default_rng(87)
    bx = rng.normal(size=f.model.vx_shape)
    bz = rng.normal(size=f.model.vz_shape)
    vx, vz = f.solve_velocity(bx, bz)
    cx, cz = f.damping_force(vx, vz)
    np.testing.assert_allclose(
        f.model.rho_vx * vx + 0.5 * f.dt * cx, bx, atol=9e-16, rtol=2e-15
    )
    np.testing.assert_allclose(
        f.model.rho_vz * vz + 0.5 * f.dt * cz, bz, atol=9e-16, rtol=2e-15
    )
    qx = rng.normal(size=bx.shape)
    qz = rng.normal(size=bz.shape)
    ax, az = f.solve_velocity_transpose(qx, qz)
    np.testing.assert_allclose(
        np.vdot(vx, qx) + np.vdot(vz, qz), np.vdot(bx, ax) + np.vdot(bz, az), rtol=2e-14
    )
    ax, az = f.damping_transpose(qx, qz)
    dx, dz = f.damping_force(bx, bz)
    np.testing.assert_allclose(
        np.vdot(dx, qx) + np.vdot(dz, qz), np.vdot(bx, ax) + np.vdot(bz, az), rtol=2e-14
    )


def test_sbp_tensor_identity_and_force_constitutive_transposes():
    f = case()
    rng = np.random.default_rng(918)
    s = random_state(f, rng)
    fx, fz = f.force(s)
    xx, zz, xz = f.stress_rhs(s.vx, s.vz)
    m = f.model
    det = m.l2m_NN**2 - m.lam_NN**2
    potential = np.sum(
        f.wn
        * (
            s.sxx * (m.l2m_NN * xx - m.lam_NN * zz)
            + s.szz * (m.l2m_NN * zz - m.lam_NN * xx)
        )
        / det
    )
    potential += np.sum(f.ws * s.sxz * xz / m.mu_MM)
    kinetic = np.sum(f.wx * s.vx * fx) + np.sum(f.wz * s.vz * fz)
    assert abs(kinetic + potential) < 2e-13
    q = random_state(f, rng)
    ax, az, at = f.force_transpose(q.vx, q.vz)
    np.testing.assert_allclose(
        np.vdot(fx, q.vx) + np.vdot(fz, q.vz),
        np.vdot(s.sxx, ax) + np.vdot(s.szz, az) + np.vdot(s.sxz, at),
        rtol=2e-14,
    )
    vx, vz = f.stress_transpose(q.sxx, q.szz, q.sxz)
    np.testing.assert_allclose(
        np.vdot(xx, q.sxx) + np.vdot(zz, q.szz) + np.vdot(xz, q.sxz),
        np.vdot(s.vx, vx) + np.vdot(s.vz, vz),
        rtol=2e-14,
    )


@pytest.mark.parametrize(
    "boundaries", [FREE, BoundaryConditions(), BoundaryConditions(top="absorbing")]
)
def test_complete_state_step_transpose(boundaries):
    f = case(boundaries)
    rng = np.random.default_rng(71)
    s, q = random_state(f, rng), random_state(f, rng)
    out = f.step(s)
    rec = [PointReceiver("vx", 3, 4)]
    transpose, _ = f._transpose_step(q, np.zeros(1), rec)
    np.testing.assert_allclose(state_dot(out, q), state_dot(s, transpose), rtol=2e-14)


def observation(f):
    theta = np.linspace(-0.8, 0.8, 21)
    R = 2.0
    rec = Receivers2D(
        x=5.1 + R * np.sin(theta),
        z=4.1 + R * np.cos(theta),
        tx=np.cos(theta),
        tz=-np.sin(theta),
        s=R * (theta - theta[0]),
    )
    return build_rectangular_das_observation(f, rec, 0.67)


def test_bounded_das_affine_field_and_original_observable():
    f = case()
    o = observation(f)
    r = o.receivers
    x, z = f.op_x, f.op_z
    vx = (
        0.7 * ((np.arange(x.n_M) + 0.5) * x.h)[:, None]
        - 0.3 * (np.arange(z.n_N) * z.h)[None, :]
        + 2
    )
    vz = (
        -0.2 * (np.arange(x.n_N) * x.h)[:, None]
        + 0.4 * ((np.arange(z.n_M) + 0.5) * z.h)[None, :]
        - 3
    )
    ax = 0.7 * r.x - 0.3 * r.z + 2
    az = -0.2 * r.x + 0.4 * r.z - 3
    np.testing.assert_allclose(o._vx.sample(vx), ax, atol=2e-15)
    np.testing.assert_allclose(o._vz.sample(vz), az, atol=2e-15)
    ref = compute_axial_strain_rate(ax[:, None], az[:, None], r, o.gauge_length_m)
    np.testing.assert_allclose(o.sample(vx, vz), ref.data[:, 0], atol=8e-15)
    np.testing.assert_array_equal(o.channel_indices, ref.channel_indices)


@pytest.mark.parametrize(
    "source_components", [(1.0, -0.3, 0.0), (0.0, 0.0, 1.0), (1.0, -0.3, 0.5)]
)
def test_full_bounded_source_das_transpose_and_timing(source_components):
    f = case()
    o = observation(f)
    rng = np.random.default_rng(57)
    source = StressPointSource(4, 5, *source_components)
    s = rng.normal(size=180)
    q = rng.normal(size=(len(o), 180))
    out = f.run_forward(s, source, o)
    a = f.run_adjoint_to_source(q, source, o)
    np.testing.assert_allclose(np.vdot(out.data, q), np.vdot(s, a), rtol=4e-13)
    assert a[-1] == 0
    np.testing.assert_array_equal(out.data[:, 0], 0)


@pytest.mark.parametrize("direction", ["vp", "vs", "both", "boundary"])
def test_das_model_taylor_including_impedance_dependencies(direction):
    f = case(BoundaryConditions(top="absorbing"))
    o = observation(f)
    m = f.model
    source = StressPointSource(4, 5, 1.0, -0.3, 0.5)
    t = np.arange(220) * f.dt
    stf = 100 * np.exp(-(((t - 0.4) / 0.15) ** 2))
    truth = prepare_rectangular_model(m.vp * 1.015, m.vs * 0.985, m.rho)
    ft = RectangularSBPSAT(truth, f.op_x.h, f.op_z.h, f.dt, f.boundaries)
    observed = ft.run_forward(stf, source, o).data
    J, _, gp, gs = f.objective_and_gradient_log_vp_vs(stf, observed, source, o)
    rng = np.random.default_rng(26)
    p = rng.normal(size=m.vp.shape)
    s = rng.normal(size=m.vs.shape)
    if direction == "vp":
        s *= 0
    if direction == "vs":
        p *= 0
    if direction == "boundary":
        p[1:-1, 1:-1] = 0
        s[1:-1, 1:-1] = 0
    p /= max(np.linalg.norm(p), 1)
    s /= max(np.linalg.norm(s), 1)
    derivative = np.sum(gp * p + gs * s)
    assert abs(derivative) > 1e-8
    remainders = []
    # The boundary-only direction has a small quadratic coefficient: the
    # cubic term is still visible at eps=.02 (slope 2.136). Refinement to
    # .01..00125 enters the quadratic regime; the tolerance is unchanged.
    for eps in (0.01, 0.005, 0.0025, 0.00125):
        mm = prepare_rectangular_model(
            m.vp * np.exp(eps * p), m.vs * np.exp(eps * s), m.rho
        )
        ff = RectangularSBPSAT(mm, f.op_x.h, f.op_z.h, f.dt, f.boundaries)
        d = ff.run_forward(stf, source, o).data
        j = 0.5 * f.dt * np.sum((d - observed) ** 2)
        remainders.append(abs(j - J - eps * derivative))
    slopes = np.log2(np.array(remainders[:-1]) / remainders[1:])
    assert np.all((slopes > 1.9) & (slopes < 2.1)), (direction, slopes)


def test_ignoring_impedance_derivatives_is_detected(monkeypatch):
    f = case(BoundaryConditions(top="absorbing"))
    o = observation(f)
    m = f.model
    source = StressPointSource(4, 5, 1.0, -0.3, 0.5)
    stf = 100 * np.exp(-(((np.arange(220) * f.dt - 0.4) / 0.15) ** 2))
    observed = np.zeros((len(o), len(stf)))
    J, _, gp, gs = f.objective_and_gradient_log_vp_vs(stf, observed, source, o)
    direction = np.zeros_like(m.vs)
    direction[[0, -1], :] = 1
    direction[:, [0, -1]] = 1
    direction /= np.linalg.norm(direction)

    def objective(eps):
        mm = prepare_rectangular_model(m.vp, m.vs * np.exp(eps * direction), m.rho)
        ff = RectangularSBPSAT(mm, f.op_x.h, f.op_z.h, f.dt, f.boundaries)
        d = ff.run_forward(stf, source, o).data
        return 0.5 * f.dt * np.sum(d * d)

    fd = (objective(1e-5) - objective(-1e-5)) / 2e-5
    full = np.sum(gs * direction)
    assert abs(fd - full) / abs(fd) < 2e-6
    monkeypatch.setattr(
        f, "_impedance_vjp", lambda *args: (np.zeros_like(m.vp), np.zeros_like(m.vs))
    )
    _, _, _, bad = f.objective_and_gradient_log_vp_vs(stf, observed, source, o)
    assert abs(np.sum(bad * direction) - fd) / abs(fd) > 0.01


def test_geometry_source_and_model_contracts():
    f = case()
    o = observation(f)
    m = f.model
    with pytest.raises(ValueError):
        m.vp[0, 0] = 1
    with pytest.raises(AttributeError):
        f.dt *= 2
    for source in (StressPointSource(0, 3), StressPointSource(f.op_x.n_M, 3)):
        with pytest.raises(ValueError):
            f.run_forward([1.0], source, o)
    with pytest.raises(ValueError):
        BoundaryConditions(left="pml")
    with pytest.raises(ValueError):
        RectangularSBPSAT(m, 1.2, 0.9, f.dt_limit)
    r = o.receivers
    bad = Receivers2D(x=r.x * 0, z=r.z, tx=r.tx, tz=r.tz, s=r.s)
    with pytest.raises(ValueError):
        build_rectangular_das_observation(f, bad, 0.67)


@pytest.mark.parametrize("face", ["top", "bottom", "left", "right"])
def test_wrong_absorption_sign_produces_detectable_energy_growth(face):
    f = case(replace(FREE, **{face: "absorbing"}))
    # Deliberate in-memory negative control; the public boundary API has no
    # negative impedance option. Rebuild its small implicit solve consistently.
    f._Z[face + "_p"] *= -1
    f._Z[face + "_s"] *= -1
    f._prepare_velocity_solve()
    state = random_state(f, np.random.default_rng(88))
    assert f.energy(f.step(state)) > f.energy(state) * 1.001


def test_two_bounded_axes_timestep_bound_against_independent_spectrum():
    x, z = np.indices((9, 10))
    m = prepare_rectangular_model(
        3.2 + 0.1 * np.sin(x), 1.5 + 0.05 * np.cos(z), 2.1 + 0.1 * np.sin(x + z)
    )
    f = RectangularSBPSAT(m, 1.3, 0.7, 0.01, FREE)
    nv = m.rho_vx.size + m.rho_vz.size
    ns = 2 * m.vp.size + m.mu_MM.size
    A, B = np.zeros((nv, ns)), np.zeros((ns, nv))

    def fill(state, names, values):
        start = 0
        for name in names:
            array = getattr(state, name)
            array[:] = values[start : start + array.size].reshape(array.shape)
            start += array.size

    for j in range(ns):
        state = f.zero_state()
        u = np.zeros(ns)
        u[j] = 1
        fill(state, FIELDS[2:], u)
        a, b = f.force(state)
        A[:, j] = np.r_[(a / m.rho_vx).ravel(), (b / m.rho_vz).ravel()]
    for j in range(nv):
        state = f.zero_state()
        u = np.zeros(nv)
        u[j] = 1
        fill(state, FIELDS[:2], u)
        B[:, j] = np.concatenate([a.ravel() for a in f.stress_rhs(state.vx, state.vz)])
    mass = np.r_[(f.wx * m.rho_vx).ravel(), (f.wz * m.rho_vz).ravel()]
    L = -np.sqrt(mass[:, None]) * (A @ B) / np.sqrt(mass[None, :])
    np.testing.assert_allclose(L, L.T, atol=2e-13)
    omega = np.sqrt(np.linalg.eigvalsh(L).max())
    assert f.dt_limit * omega < 2


def normal_packet(h, bottom="absorbing"):
    # lambda approximately zero isolates the exact 1-D normal-P subsystem
    # in the actual bounded 2-D engine with free lateral faces.
    c = np.sqrt(2.0)
    rho = 2.0
    dt = 0.12 * h / c
    shape = (int(8 / h) + 1, int(80 / h) + 1)
    m = prepare_rectangular_model(
        np.full(shape, c), np.ones(shape), np.full(shape, rho)
    )
    f = RectangularSBPSAT(
        m,
        h,
        h,
        dt,
        BoundaryConditions(top="free", bottom=bottom, left="free", right="free"),
    )
    s = f.zero_state()
    zn = np.arange(shape[1]) * h
    zm = (np.arange(shape[1] - 1) + 0.5) * h
    s.vz[:] = np.exp(-0.5 * ((zm + c * dt / 2 - 30) / 5) ** 2)
    s.szz[:] = -rho * c * np.exp(-0.5 * ((zn - 30) / 5) ** 2)
    initial = f.energy(s)
    for _ in range(int(np.ceil((100 / c) / dt))):
        s = f._advance(s)
    return f.energy(s) / initial


def test_normal_outgoing_packet_absorption_and_refinement():
    coarse, fine = normal_packet(1.0), normal_packet(0.5)
    assert fine < coarse < 1e-4
    assert normal_packet(1.0, "free") == pytest.approx(1.0, abs=2e-13)
