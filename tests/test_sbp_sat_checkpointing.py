"""Exact replay, time-boundary indexing, model VJP and history allocations."""

from dataclasses import replace

import numpy as np
import pytest

from src.common.grid import Grid2D
from src.common.model import ElasticModel2D
from src.das.receivers import Receivers2D
from src.elastic.source import build_dc_source
from src.sbp_sat.checkpointing import plan_checkpoints
from src.sbp_sat.elastic2d import SBPSATState
from src.sbp_sat.rectangular import BoundaryConditions
from src.sbp_sat.simulator import (
    objective_and_gradient_log_vp_vs,
    prepare_simulation,
)


def problem(nt=137, execution="numba", bc=None):
    g = Grid2D(13, 14, 1.2, 0.9, nt, 0.005, x0=-3.7, z0=2.3)
    x, z = np.indices(g.shape)
    m = ElasticModel2D(
        g, 3.2 + 0.05 * np.sin(x), 1.5 + 0.02 * np.cos(z), 2.2 + 0.02 * np.cos(x + z)
    )
    source = build_dc_source(
        grid=g,
        x_m=g.x0 + 5.2 * g.dx,
        z_m=g.z0 + 5.3 * g.dz,
        theta_deg=35,
        scalar_moment=20,
        nt=nt,
        dt=g.dt,
        f0_hz=5,
        spreading="bilinear",
        source_time_mode="ricker_moment",
    )
    theta = np.linspace(-0.7, 0.7, 21)
    rec = Receivers2D(
        x=1.5 + 2 * np.sin(theta),
        z=6.5 + 2 * np.cos(theta),
        tx=np.cos(theta),
        tz=-np.sin(theta),
        s=2 * (theta - theta[0]),
    )
    p = prepare_simulation(m, source, rec, 0.57, execution=execution)
    if bc is not None:
        from src.sbp_sat.numba_solver import NumbaSBPSAT
        from src.sbp_sat.rectangular import RectangularSBPSAT

        cls = NumbaSBPSAT if execution == "numba" else RectangularSBPSAT
        kwargs = (
            {} if execution == "numba" else {"sparse_derivatives": execution == "csr"}
        )
        f = cls(p.solver.model, g.dx, g.dz, g.dt, bc, **kwargs)
        p = replace(p, solver=f)
    return m, source, rec, p


@pytest.mark.parametrize("execution", ["numba", "csr", "numpy"])
@pytest.mark.parametrize("interval", [1, 2, 7, 37, "auto"])
def test_checkpoint_matches_full_history_exactly_across_block_boundaries(
    execution, interval
):
    m, _src, _rec, p = problem(37, execution)
    rng = np.random.default_rng(1931)
    stf = rng.normal(size=37)
    observed = rng.normal(scale=0.001, size=(len(p.observation), 37))
    full = p.solver.objective_and_gradient_log_vp_vs(
        stf, observed, p.source, p.observation
    )
    saved = tuple(np.array(x, copy=True) for x in full)
    actual = p.solver.objective_and_gradient_log_vp_vs(
        stf, observed, p.source, p.observation, checkpoint_interval=interval
    )
    for expected, value, original in zip(full, actual, saved):
        np.testing.assert_array_equal(value, expected)
        np.testing.assert_array_equal(expected, original)
    assert m.grid.nt == 37 and m.grid.dt == 0.005


@pytest.mark.parametrize("mask", range(16))
def test_checkpoint_gradient_matches_reference_for_all_free_impedance_faces(mask):
    names = ("top", "bottom", "left", "right")
    bc = BoundaryConditions(
        **{
            key: "absorbing" if mask & (1 << i) else "free"
            for i, key in enumerate(names)
        }
    )
    _, _, _, p = problem(19, "numba", bc)
    stf = np.linspace(-0.7, 1, 19)
    obs = np.full((len(p.observation), 19), 0.001)
    full = p.solver.objective_and_gradient_log_vp_vs(stf, obs, p.source, p.observation)
    replay = p.solver.objective_and_gradient_log_vp_vs(
        stf, obs, p.source, p.observation, checkpoint_interval=6
    )
    for x, y in zip(full, replay):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("nt", [1, 2, 3])
def test_last_source_sample_and_zero_initial_velocity_at_short_records(nt):
    _, _, _, p = problem(max(2, nt))
    stf = np.ones(nt)
    obs = np.ones((len(p.observation), nt)) * 0.01
    old = p.solver.objective_and_gradient_log_vp_vs(stf, obs, p.source, p.observation)
    new = p.solver.objective_and_gradient_log_vp_vs(
        stf, obs, p.source, p.observation, checkpoint_interval=1
    )
    for x, y in zip(old, new):
        np.testing.assert_array_equal(x, y)
    if nt == 1:
        np.testing.assert_array_equal(new[1], 0)
        np.testing.assert_array_equal(new[2], 0)
        np.testing.assert_array_equal(new[3], 0)


def test_checkpoints_are_full_states_and_forward_steps_are_replayed_once(monkeypatch):
    _, _, _, p = problem(37)
    f = p.solver
    stf = np.arange(37, dtype=float) / 37
    obs = np.zeros((len(p.observation), 37))
    copies, calls = [], []
    old_copy, old_advance = SBPSATState.copy, f.advance_inplace

    def copy(state):
        result = old_copy(state)
        arrays = [getattr(result, key) for key in ("vx", "vz", "sxx", "szz", "sxz")]
        copies.append(sum(a.nbytes for a in arrays))
        for key in ("vx", "vz", "sxx", "szz", "sxz"):
            assert not np.shares_memory(getattr(state, key), getattr(result, key))
        return result

    def advance(state, source, amplitude):
        calls.append(amplitude)
        return old_advance(state, source, amplitude)

    monkeypatch.setattr(SBPSATState, "copy", copy)
    monkeypatch.setattr(f, "advance_inplace", advance)
    monkeypatch.setattr(
        f, "run_forward", lambda *a, **kw: pytest.fail("Full-history forward called")
    )
    f.objective_and_gradient_log_vp_vs(
        stf, obs, p.source, p.observation, checkpoint_interval=7
    )
    plan = plan_checkpoints(13, 14, 37, 7)
    assert len(copies) == plan.checkpoint_count
    assert sum(copies) == plan.checkpoint_bytes
    expected_replay = np.concatenate(
        [stf[i : min(i + 7, 37)] for i in reversed(range(0, 37, 7))]
    )
    np.testing.assert_array_equal(calls[:37], stf)
    np.testing.assert_array_equal(calls[37:], expected_replay)
    assert len(calls) == 74


@pytest.mark.parametrize("nt", [1, 2, 7, 37, 137])
def test_uniform_plan_minimizes_independently_counted_history_bytes(nt):
    p = plan_checkpoints(13, 14, nt)
    v = 8 * (12 * 14 + 13 * 13)
    s = 8 * (2 * 13 * 14 + 12 * 13 + 12 * 14 + 13 * 13)
    assert p.history_bytes == min(
        len(range(0, nt, k)) * s + (k + 1) * v for k in range(1, nt + 1)
    )
    assert p.full_velocity_history_bytes == nt * v


def test_event_history_estimate_requires_no_event_arrays():
    p = plan_checkpoints(1241, 1121, 19062)
    assert p.full_velocity_history_bytes / 2**30 == pytest.approx(394.8181683421135)
    assert p.history_bytes / 2**30 < 10
    assert p.history_bytes * 40 < p.full_velocity_history_bytes


def test_generic_grid_source_and_native_point_receivers():
    from src.sbp_sat.elastic2d import PointReceiver, StressPointSource

    _, _, _, p = problem(23)
    src = StressPointSource(5, 5, mxx=0.7, mzz=-0.3, mxz=0.2)
    rec = (PointReceiver("vx", 4, 5), PointReceiver("vz", 5, 4))
    stf = np.sin(np.arange(23))
    obs = np.zeros((2, 23))
    full = p.solver.objective_and_gradient_log_vp_vs(stf, obs, src, rec)
    replay = p.solver.objective_and_gradient_log_vp_vs(
        stf, obs, src, rec, checkpoint_interval=5
    )
    for x, y in zip(full, replay):
        np.testing.assert_array_equal(x, y)


def test_parallel_replay_parity_and_thread_mask_restoration(monkeypatch):
    from numba import config, get_num_threads

    from src.sbp_sat.numba_solver import NumbaSBPSAT

    if config.NUMBA_NUM_THREADS < 2:
        pytest.skip("Two Numba threads are unavailable")
    m, src, _rec, p = problem(31)
    try:
        f = NumbaSBPSAT(
            p.solver.model, m.grid.dx, m.grid.dz, m.grid.dt, numba_threads=2
        )
    except ValueError as exc:
        if "supports" in str(exc):
            pytest.skip(str(exc))
        raise
    obs = np.zeros((len(p.observation), 31))
    old_mask = get_num_threads()
    full = f.objective_and_gradient_log_vp_vs(
        src.stf.values, obs, p.source, p.observation
    )
    replay = f.objective_and_gradient_log_vp_vs(
        src.stf.values, obs, p.source, p.observation, checkpoint_interval=7
    )
    for x, y in zip(full, replay):
        np.testing.assert_array_equal(x, y)
    assert get_num_threads() == old_mask

    def fail(*args, **kwargs):
        raise RuntimeError("deliberate reverse failure")

    monkeypatch.setattr(f, "_reverse_model_gradient_step", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        f.objective_and_gradient_log_vp_vs(
            src.stf.values, obs, p.source, p.observation, checkpoint_interval=7
        )
    assert get_num_threads() == old_mask


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "bad", 38])
def test_bad_interval_rejected_before_propagation(monkeypatch, value):
    _, src, _, p = problem(37)
    monkeypatch.setattr(
        p.solver, "advance_inplace", lambda *a: pytest.fail("propagation started")
    )
    with pytest.raises(ValueError):
        p.solver.objective_and_gradient_log_vp_vs(
            src.stf.values,
            np.zeros((len(p.observation), 37)),
            p.source,
            p.observation,
            checkpoint_interval=value,
        )


@pytest.mark.parametrize("parameter", ["both", "boundary_vp", "boundary_vs"])
def test_checkpointed_physical_source_das_taylor_and_finite_difference(parameter):
    m, src, rec, p = problem(400)
    observed = np.zeros((len(p.observation), m.grid.nt))
    J, _, gp, gs = objective_and_gradient_log_vp_vs(
        m, src, rec, observed, 0.57, checkpoint_interval=37
    )
    rng = np.random.default_rng(16)
    u, v = rng.normal(size=m.grid.shape), rng.normal(size=m.grid.shape)
    if parameter != "both":
        u[:], v[:] = 0, 0
        active = u if parameter == "boundary_vp" else v
        active[0] = rng.normal(size=m.grid.nz)
        active[-1] = rng.normal(size=m.grid.nz)
        active[:, -1] += rng.normal(size=m.grid.nx)
    if np.linalg.norm(u):
        u /= np.linalg.norm(u)
    if np.linalg.norm(v):
        v /= np.linalg.norm(v)
    derivative = float(np.sum(gp * u + gs * v))
    assert abs(derivative) > 1e-12

    def objective(eps):
        mm = ElasticModel2D(
            m.grid, m.vp * np.exp(eps * u), m.vs * np.exp(eps * v), m.rho
        )
        pp = prepare_simulation(mm, src, rec, 0.57, execution="numba")
        d = pp.solver.run_forward(src.stf.values, pp.source, pp.observation).data
        return 0.5 * m.grid.dt * np.sum((d - observed) ** 2)

    steps = np.array([0.01, 0.005, 0.0025, 0.00125])
    remainders = [abs(objective(e) - J - e * derivative) for e in steps]
    slopes = np.log2(np.array(remainders[:-1]) / remainders[1:])
    assert np.all((slopes > 1.9) & (slopes < 2.1)), slopes
    eps = 1e-5
    fd = (objective(eps) - objective(-eps)) / (2 * eps)
    assert abs(fd - derivative) < 2e-6 * abs(derivative)
