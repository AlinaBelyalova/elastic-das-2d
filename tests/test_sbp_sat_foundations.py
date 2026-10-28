"""Independent polynomial, boundary-flux, spectrum and transpose checks."""

import numpy as np
import pytest

from src.sbp_sat import (
    PointReceiver,
    StressPointSource,
    advance_one_step,
    prepare_model,
    run_adjoint_to_source,
    run_forward,
    semidiscrete_energy_rate,
    stable_timestep_bound,
    staggered_sbp_4_2,
    zero_state,
)
from src.sbp_sat import elastic2d as elastic
from src.sbp_sat.boundaries import traction_free_sat, traction_free_sat_transpose
from src.sbp_sat.operators import periodic_D_M_to_N, periodic_D_N_to_M


def small_model(n=8, nx=5, h=0.9):
    op = staggered_sbp_4_2(n, h)
    x, z = np.indices((nx, n + 1))
    model = prepare_model(
        3.2 + 0.1 * np.sin(x + z), 1.6 + 0.08 * np.cos(z), 2.1 + 0.15 * np.sin(x - z)
    )
    return model, op


@pytest.mark.parametrize("n", [8, 9, 12, 40])
@pytest.mark.parametrize("h", [0.125, 3.7])
def test_polynomials_norms_traces_and_sbp(n, h):
    op = staggered_sbp_4_2(n, h)
    xN, xM = np.arange(n + 1) / n, (np.arange(n) + 0.5) / n
    assert np.all(op.H_N > 0) and np.all(op.H_M > 0)
    np.testing.assert_allclose(op.H_N.sum(), n * h, rtol=2e-15)
    np.testing.assert_allclose(op.H_M.sum(), n * h, rtol=2e-15)
    assert np.max(np.abs(op.sbp_residual())) < 1e-14
    for degree in range(3):
        dN = np.zeros(n + 1) if degree == 0 else degree * xN ** (degree - 1) / (n * h)
        dM = np.zeros(n) if degree == 0 else degree * xM ** (degree - 1) / (n * h)
        np.testing.assert_allclose(op.D_M_to_N @ xM**degree, dN, atol=2e-13, rtol=2e-13)
        np.testing.assert_allclose(op.D_N_to_M @ xN**degree, dM, atol=2e-13, rtol=2e-13)
        np.testing.assert_allclose(
            op.p_left @ xM**degree, 1.0 if degree == 0 else 0.0, atol=2e-15
        )
        np.testing.assert_allclose(op.p_right @ xM**degree, 1.0, atol=2e-15)
    for degree in (3, 4):
        dN = degree * xN ** (degree - 1) / (n * h)
        dM = degree * xM ** (degree - 1) / (n * h)
        np.testing.assert_allclose(
            (op.D_M_to_N @ xM**degree)[4:-4], dN[4:-4], atol=2e-13
        )
        np.testing.assert_allclose(
            (op.D_N_to_M @ xN**degree)[3:-3], dM[3:-3], atol=2e-13
        )


@pytest.mark.parametrize("n", [7, 8.5, True, np.nan])
def test_interval_count_is_not_silently_truncated(n):
    with pytest.raises(ValueError):
        staggered_sbp_4_2(n, 1.0)


@pytest.mark.parametrize("side", ["top", "bottom"])
@pytest.mark.parametrize("pair", ["shear", "normal"])
def test_each_boundary_flux_sign_and_negative_controls(side, pair):
    op = staggered_sbp_4_2(8, 2.3)
    boundary = 0 if side == "top" else -1
    sign = -1 if side == "top" else 1
    n = np.zeros(op.n_N)
    m = np.zeros(op.n_M)
    n[boundary] = m[boundary] = 1.0
    # Shear: v on N, tau on M. Normal: v on M, stress on N.
    # Material multipliers cancel in rho*v*dv + tau*dtau/modulus.
    raw = n @ (op.H_N * (op.D_M_to_N @ m)) + m @ (op.H_M * (op.D_N_to_M @ n))
    expected = sign * 15 / 8
    assert raw == pytest.approx(expected, abs=1e-14)
    sx, sz = traction_free_sat(m, n, op)
    penalty = n @ (op.H_N * sx) if pair == "shear" else m @ (op.H_M * sz)
    assert raw + penalty == pytest.approx(0.0, abs=1e-14)
    assert abs(raw) > 1  # SAT disabled MUST fail
    assert abs(raw - penalty) > 2  # reversed sign MUST fail


def test_sat_and_periodic_derivative_transposes():
    rng = np.random.default_rng(9)
    op = staggered_sbp_4_2(9, 0.7)
    szz, sxz = rng.normal(size=(7, 10)), rng.normal(size=(7, 9))
    qx, qz = rng.normal(size=szz.shape), rng.normal(size=sxz.shape)
    fx, fz = traction_free_sat(sxz, szz, op)
    txz, tzz = traction_free_sat_transpose(qx, qz, op)
    np.testing.assert_allclose(
        np.vdot(fx, qx) + np.vdot(fz, qz),
        np.vdot(sxz, txz) + np.vdot(szz, tzz),
        rtol=3e-15,
    )
    a, b = rng.normal(size=(7, 11)), rng.normal(size=(7, 11))
    np.testing.assert_allclose(
        np.vdot(periodic_D_M_to_N(a, 0.8), b),
        -np.vdot(a, periodic_D_N_to_M(b, 0.8)),
        rtol=3e-15,
    )


@pytest.mark.parametrize("control", ["disabled", "wrong_sign"])
def test_actual_2d_energy_detects_bad_top_sat(monkeypatch, control):
    model, op = small_model()
    rng = np.random.default_rng(16)
    state = zero_state(model)
    for name in ("vx", "vz", "sxx", "szz", "sxz"):
        a = getattr(state, name)
        a[:] = rng.normal(size=a.shape)
    dE, scale = semidiscrete_energy_rate(state, model, op, 1.3)
    assert abs(dE) / scale < 1e-14
    original = elastic.traction_free_sat

    def incorrect(sxz, szz, op):
        fx, fz = original(sxz, szz, op)
        factor = 1 if control == "disabled" else 2
        fx[:, 0] -= factor * (sxz @ op.p_left) / op.H_N[0]
        fz -= factor * szz[:, 0, None] * (op.p_left / op.H_M)
        return fx, fz

    monkeypatch.setattr(elastic, "traction_free_sat", incorrect)
    dE, scale = semidiscrete_energy_rate(state, model, op, 1.3)
    assert abs(dE) / scale > 1e-3


def test_timestep_bound_against_independently_assembled_spectrum():
    model, op = small_model()
    dx = 1.3
    template = zero_state(model)
    nv = template.vx.size + template.vz.size
    ns = template.sxx.size + template.szz.size + template.sxz.size

    def set_fields(s, fields, vector):
        start = 0
        for name in fields:
            a = getattr(s, name)
            a[:] = vector[start : start + a.size].reshape(a.shape)
            start += a.size

    A, B = np.zeros((nv, ns)), np.zeros((ns, nv))
    for j in range(ns):
        s = zero_state(model)
        u = np.zeros(ns)
        u[j] = 1
        set_fields(s, ("sxx", "szz", "sxz"), u)
        vx, vz = elastic.velocity_rhs(s, model, op, dx)
        A[:, j] = np.r_[vx.ravel(), vz.ravel()]
    for j in range(nv):
        s = zero_state(model)
        u = np.zeros(nv)
        u[j] = 1
        set_fields(s, ("vx", "vz"), u)
        B[:, j] = np.concatenate(
            [a.ravel() for a in elastic.stress_rhs(s.vx, s.vz, model, op, dx)]
        )
    mass = np.r_[
        (dx * op.H_N * model.rho_vx).ravel(), (dx * op.H_M * model.rho_vz).ravel()
    ]
    L = -np.sqrt(mass[:, None]) * (A @ B) / np.sqrt(mass[None, :])
    np.testing.assert_allclose(L, L.T, atol=3e-14)
    omega = np.sqrt(np.linalg.eigvalsh(L).max())
    limit = stable_timestep_bound(model, op, dx)
    assert limit * omega < 2

    # Independent full-step matrix: below the bound all modes stay on |r|=1.
    def step_matrix(dt):
        return np.block(
            [[np.eye(nv), dt * A], [dt * B, np.eye(ns) + dt * dt * (B @ A)]]
        )

    assert np.abs(np.linalg.eigvals(step_matrix(0.9 * limit))).max() < 1 + 2e-7
    assert np.abs(np.linalg.eigvals(step_matrix(2.05 / omega))).max() > 1.1


def test_one_step_exact_transpose_on_heterogeneous_model():
    model, op = small_model()
    rng = np.random.default_rng(94)
    s, q = zero_state(model), zero_state(model)
    fields = ("vx", "vz", "sxx", "szz", "sxz")
    for state in (s, q):
        for name in fields:
            a = getattr(state, name)
            a[:] = rng.normal(size=a.shape)
    dt = 0.4 * stable_timestep_bound(model, op, 1.1)
    out = advance_one_step(s, model, op, 1.1, dt)
    adj = q.copy()
    x, z = elastic._stress_update_transpose(
        adj.sxx, adj.szz, adj.sxz, model, op, 1.1, dt
    )
    adj.vx += x
    adj.vz += z
    xx, zz, xz = elastic._velocity_update_transpose(adj.vx, adj.vz, model, op, 1.1, dt)
    adj.sxx += xx
    adj.szz += zz
    adj.sxz += xz
    dot = lambda a, b: sum(np.vdot(getattr(a, n), getattr(b, n)) for n in fields)
    np.testing.assert_allclose(dot(out, q), dot(s, adj), rtol=3e-14)


@pytest.mark.parametrize("dt", [0.0, -1.0, np.nan, np.inf, 1e4])
def test_forward_and_adjoint_reject_uncertified_timestep(dt):
    model, op = small_model()
    source = StressPointSource(2, 3)
    rec = [PointReceiver("vx", 1, 0)]
    with pytest.raises(ValueError):
        run_forward(np.ones(3), model, op, 1.0, dt, source, rec)
    with pytest.raises(ValueError):
        run_adjoint_to_source(np.ones((1, 3)), model, op, 1.0, dt, source, rec)


def test_source_and_receiver_time_levels_and_repeated_receivers():
    model, op = small_model()
    source = StressPointSource(2, 3, 1.0, -0.3)
    rec = [
        PointReceiver("vx", 1, 3),
        PointReceiver("vx", 1, 3),
        PointReceiver("vz", 2, 2),
    ]
    dt = 0.4 * stable_timestep_bound(model, op, 1.0)
    stf = np.array([1.0, 0.0, 0.0, 0.0])
    out = run_forward(stf, model, op, 1.0, dt, source, rec)
    np.testing.assert_array_equal(out.data[:, 0], 0)
    np.testing.assert_array_equal(out.data[0], out.data[1])
    q = np.random.default_rng(5).normal(size=(3, 4))
    a = run_adjoint_to_source(q, model, op, 1.0, dt, source, rec)
    assert a[-1] == 0  # last injection affects final stress, not receiver history
    np.testing.assert_allclose(np.vdot(out.data, q), np.vdot(stf, a), rtol=3e-14)


def test_shear_source_centre_conservation_and_no_boundary_clipping():
    model, op = small_model()
    source = StressPointSource(0, 3, 0.0, 0.0, 2.0)
    out = advance_one_step(
        zero_state(model), model, op, 1.0, 0.01, source=source, source_amplitude=3.0
    )
    assert out.sxz.sum() == pytest.approx(0.01 * 3 * 2)
    expected = np.zeros_like(out.sxz)
    expected[[model.nx - 1, 0], 2] = 0.015
    expected[[model.nx - 1, 0], 3] = 0.015
    np.testing.assert_array_equal(out.sxz, expected)
    for iz in (0, model.nz_M):
        with pytest.raises(ValueError):
            run_forward(
                [1.0],
                model,
                op,
                1.0,
                0.01,
                StressPointSource(0, iz, 0.0, 0.0, 1.0),
                [PointReceiver("vx", 0, 0)],
            )


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_finite_input_contracts(bad):
    model, op = small_model()
    src = StressPointSource(2, 3)
    rec = [PointReceiver("vx", 1, 0)]
    with pytest.raises(ValueError):
        run_forward([bad], model, op, 1.0, 0.01, src, rec)
    with pytest.raises(ValueError):
        StressPointSource(2, 3, bad, 1)
    with pytest.raises(ValueError):
        PointReceiver("vx", 2.5, 3)
    state = zero_state(model)
    state.vz[0, 0] = bad
    with pytest.raises(ValueError):
        run_forward([1.0], model, op, 1.0, 0.01, src, rec, initial_state=state)
