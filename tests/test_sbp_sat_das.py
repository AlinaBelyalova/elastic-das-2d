"""Curved off-grid receivers, finite gauges, full transpose and model gradient."""

import numpy as np
import pytest

from src.das.receivers import Receivers2D
from src.das.strain import compute_axial_strain_rate
from src.sbp_sat import (
    StressPointSource,
    build_das_observation,
    objective_and_gradient_log_vp_vs,
    prepare_model,
    run_adjoint_to_source,
    run_forward,
    stable_timestep_bound,
    staggered_sbp_4_2,
    zero_state,
)


def setup_case(L=1.37):
    nx, n = 16, 16
    x, z = np.indices((nx, n + 1))
    model = prepare_model(
        3.4 + 0.12 * np.sin(x + z), 1.8 + 0.08 * np.cos(z), 2.2 + 0.1 * np.cos(x - z)
    )
    op = staggered_sbp_4_2(n, 1.0)
    theta = np.linspace(-0.8, 0.8, 21)
    R = 4.0
    rec = Receivers2D(
        x=7.2 + R * np.sin(theta),
        z=4.1 + R * np.cos(theta),
        tx=np.cos(theta),
        tz=-np.sin(theta),
        s=R * (theta - theta[0]),
    )
    obs = build_das_observation(model, op, 1.1, rec, L)
    return model, op, obs


@pytest.mark.parametrize("L", [0.17, 1.37, 2.56])
def test_matches_existing_das_convention_for_affine_velocity(L):
    model, op, obs = setup_case(L)
    xv = (np.arange(model.nx) + 0.5) * obs.dx
    xn = np.arange(model.nx) * obs.dx
    zn = np.arange(model.nz_N) * op.h
    zm = (np.arange(model.nz_M) + 0.5) * op.h
    vx = 0.7 * xv[:, None] - 0.3 * zn[None, :] + 2
    vz = -0.2 * xn[:, None] + 0.4 * zm[None, :] - 3
    r = obs.receivers
    exact_x = 0.7 * r.x - 0.3 * r.z + 2
    exact_z = -0.2 * r.x + 0.4 * r.z - 3
    # Analytic affine fields independently verify native-grid interpolation.
    np.testing.assert_allclose(obs._vx.sample(vx), exact_x, atol=3e-15)
    np.testing.assert_allclose(obs._vz.sample(vz), exact_z, atol=3e-15)
    reference = compute_axial_strain_rate(exact_x[:, None], exact_z[:, None], r, L)
    np.testing.assert_array_equal(obs.channel_indices, reference.channel_indices)
    np.testing.assert_allclose(obs.sample(vx, vz), reference.data[:, 0], atol=3e-14)


def test_rigid_translation_has_zero_das_on_curved_cable():
    model, op, obs = setup_case()
    np.testing.assert_allclose(
        obs.sample(np.full(obs.vx_shape, 3.0), np.full(obs.vz_shape, -7.0)),
        0,
        atol=3e-15,
    )


def test_sampling_das_transpose_with_overlapping_gauges():
    model, op, obs = setup_case()
    rng = np.random.default_rng(25)
    vx = rng.normal(size=obs.vx_shape)
    vz = rng.normal(size=obs.vz_shape)
    q = rng.normal(size=len(obs))
    ax = np.zeros_like(vx)
    az = np.zeros_like(vz)
    obs.scatter(q, ax, az)
    np.testing.assert_allclose(
        np.vdot(obs.sample(vx, vz), q), np.vdot(vx, ax) + np.vdot(vz, az), rtol=2e-14
    )


def test_periodic_x_interpolation_wraps_native_stencil_only():
    model, op, _ = setup_case()
    rec = Receivers2D(
        x=[0.0, 0.1, 0.2],
        z=[3.0, 3.0, 3.0],
        tx=[1.0, 1.0, 1.0],
        tz=[0.0, 0.0, 0.0],
        s=[0.0, 0.1, 0.2],
    )
    obs = build_das_observation(model, op, 1.1, rec, 0.1)
    vx = np.zeros(obs.vx_shape)
    vx[-1, :] = 2.0
    vx[0, :] = 4.0
    assert obs._vx.sample(vx)[0] == pytest.approx(3.0)


def test_rejects_geometry_extrapolation_and_grid_mismatch():
    model, op, obs = setup_case()
    r = obs.receivers
    for x, z in ((r.x - 100, r.z), (r.x, r.z * 0), (r.x, r.z + 100)):
        rec = Receivers2D(x=x, z=z, tx=r.tx, tz=r.tz, s=r.s)
        with pytest.raises(ValueError):
            build_das_observation(model, op, obs.dx, rec, 1.0)
    with pytest.raises(ValueError):
        build_das_observation(model, op, obs.dx, r, 100.0)
    with pytest.raises(ValueError):
        run_forward([1.0], model, op, 1.2, 0.01, StressPointSource(5, 5), obs)


@pytest.mark.parametrize(
    "coefficients", [(1.0, -0.4, 0.0), (0.0, 0.0, 1.0), (1.0, -0.4, 0.7)]
)
def test_complete_source_to_das_dot_product(coefficients):
    model, op, obs = setup_case()
    source = StressPointSource(7, 6, *coefficients)
    dt = 0.7 * stable_timestep_bound(model, op, obs.dx)
    rng = np.random.default_rng(67)
    s = rng.normal(size=100)
    q = rng.normal(size=(len(obs), 100))
    d = run_forward(s, model, op, obs.dx, dt, source, obs).data
    a = run_adjoint_to_source(q, model, op, obs.dx, dt, source, obs)
    np.testing.assert_allclose(np.vdot(d, q), np.vdot(s, a), rtol=2e-13)
    assert np.linalg.norm(d) > 1e-5


@pytest.mark.parametrize("parameter", ["vp", "vs", "both"])
def test_das_model_gradient_taylor_with_fixed_density_and_dt(parameter):
    model, op, obs = setup_case()
    source = StressPointSource(7, 6, 1.0, -0.4, 0.7)
    dt = 0.6 * stable_timestep_bound(model, op, obs.dx)
    t = np.arange(120) * dt
    stf = 100 * np.exp(-(((t - 0.4) / 0.15) ** 2))
    truth = prepare_model(model.vp * 1.015, model.vs * 0.985, model.rho)
    observed = run_forward(stf, truth, op, obs.dx, dt, source, obs).data
    J, data, gp, gs = objective_and_gradient_log_vp_vs(
        stf, observed, model, op, obs.dx, dt, source, obs
    )
    rng = np.random.default_rng(39)
    p = rng.normal(size=model.vp.shape)
    q = rng.normal(size=model.vs.shape)
    p /= np.linalg.norm(p)
    q /= np.linalg.norm(q)
    if parameter == "vp":
        q *= 0
    if parameter == "vs":
        p *= 0
    derivative = np.sum(gp * p + gs * q)
    assert abs(derivative) > 1e-9
    remainders = []
    for eps in [0.02, 0.01, 0.005, 0.0025]:
        perturbed = prepare_model(
            model.vp * np.exp(eps * p), model.vs * np.exp(eps * q), model.rho
        )
        d = run_forward(stf, perturbed, op, obs.dx, dt, source, obs).data
        j = 0.5 * dt * np.sum((d - observed) ** 2)
        remainders.append(abs(j - J - eps * derivative))
    slopes = np.log2(np.array(remainders[:-1]) / remainders[1:])
    assert np.all((slopes > 1.9) & (slopes < 2.1)), (parameter, slopes)
