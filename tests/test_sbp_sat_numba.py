"""Compiled bounded step against an independently executed CSR reference."""

import itertools
import os
import numpy as np
import pytest
from numba import get_num_threads, config

from src.sbp_sat import prepare_rectangular_model, StressPointSource, PointReceiver
from src.sbp_sat.rectangular import RectangularSBPSAT, BoundaryConditions
from src.sbp_sat.numba_solver import NumbaSBPSAT

FIELDS = ("vx", "vz", "sxx", "szz", "sxz")


def pair(bc, shape=(10, 11), threads=1):
    x, z = np.indices(shape)
    model = prepare_rectangular_model(
        3.2 + 0.06 * np.sin(x + z), 1.5 + 0.03 * np.cos(z), 2.2 + 0.02 * np.sin(x - z)
    )
    return (
        NumbaSBPSAT(model, 1.3, 0.8, 0.015, bc, numba_threads=threads),
        RectangularSBPSAT(model, 1.3, 0.8, 0.015, bc, sparse_derivatives=True),
    )


def random_state(f):
    s = f.zero_state()
    rng = np.random.default_rng(771)
    for name in FIELDS:
        a = getattr(s, name)
        a[:] = rng.normal(size=a.shape)
    return s


@pytest.mark.parametrize(
    "faces", list(itertools.product(("free", "absorbing"), repeat=4))
)
def test_all_face_combinations_same_step_and_exact_energy_loss(faces):
    f, r = pair(BoundaryConditions(*faces), shape=(9, 10))
    a = random_state(f)
    for _ in range(5):
        before = a.copy()
        out = f.step(a)
        ref = r.step(a)
        for name in FIELDS:
            np.testing.assert_allclose(
                getattr(out, name), getattr(ref, name), rtol=2e-14, atol=2e-15
            )
            np.testing.assert_array_equal(getattr(a, name), getattr(before, name))
        loss = f.dt * f.dissipation((a.vx + out.vx) / 2, (a.vz + out.vz) / 2)
        assert abs(f.energy(out) - f.energy(a) + loss) < 4e-13 * f.energy(a)
        assert f.energy(out) > 0
        a = out


@pytest.mark.parametrize(
    "components", [(1.0, -0.3, 0.0), (0.0, 0.0, 1.0), (1.0, -0.3, 0.7)]
)
def test_compiled_forward_with_reference_transpose(components):
    f, r = pair(BoundaryConditions())
    src = StressPointSource(4, 5, *components)
    rec = [PointReceiver("vx", i, 5) for i in range(2, 7)] + [
        PointReceiver("vz", 4, j) for j in range(2, 7)
    ]
    rng = np.random.default_rng(126)
    stf = rng.normal(size=240)
    out = f.run_forward(stf, src, rec)
    ref = r.run_forward(stf, src, rec)
    np.testing.assert_allclose(out.data, ref.data, rtol=2e-12, atol=3e-15)
    q = rng.normal(size=out.data.shape)
    adj = f.run_adjoint_to_source(q, src, rec)
    np.testing.assert_allclose(np.vdot(out.data, q), np.vdot(stf, adj), rtol=2e-12)
    assert adj[-1] == 0


def test_owned_workspace_reuses_arrays_without_changing_the_step():
    f, r = pair(BoundaryConditions(top="absorbing"))
    source = StressPointSource(4, 5, 1.0, -0.3, 0.7)
    s = random_state(f)
    a = s.copy()
    stresses = [id(getattr(a, n)) for n in FIELDS[2:]]
    velocity_ids = set()
    with f.thread_context():
        for k in range(40):
            amplitude = float(np.sin(k))
            a = f.advance_inplace(a, source, amplitude)
            s = r.step(s, source=source, amplitude=amplitude)
            velocity_ids.add((id(a.vx), id(a.vz)))
            assert stresses == [id(getattr(a, n)) for n in FIELDS[2:]]
    assert len(velocity_ids) == 2
    for name in FIELDS:
        np.testing.assert_allclose(
            getattr(a, name), getattr(s, name), rtol=1e-12, atol=2e-14
        )


def test_parallel_threads_are_deterministic_and_thread_mask_is_restored():
    available = min(
        config.NUMBA_NUM_THREADS, int(os.environ.get("SLURM_CPUS_PER_TASK", 2))
    )
    if available < 2:
        pytest.skip("Two threads are outside this allocation/runtime.")
    f, r = pair(BoundaryConditions(), threads=2)
    one, _ = pair(BoundaryConditions(), threads=1)
    initial = random_state(f)
    source = StressPointSource(4, 5, 1.0, -0.3, 0.7)
    rec = [PointReceiver("vx", 4, 6)]
    old = get_num_threads()
    a = f.run_forward(np.ones(60), source, rec, initial_state=initial)
    b = one.run_forward(np.ones(60), source, rec, initial_state=initial)
    assert get_num_threads() == old
    np.testing.assert_array_equal(a.data, b.data)
    for name in FIELDS:
        np.testing.assert_array_equal(
            getattr(a.final_state, name), getattr(b.final_state, name)
        )


@pytest.mark.parametrize(
    "bc",
    [
        BoundaryConditions(),
        BoundaryConditions(top="free", bottom="free", left="free", right="free"),
    ],
)
def test_bulk_long_run_energy_identity(bc):
    f, _ = pair(bc, shape=(23, 27))
    state = random_state(f)
    initial = f.energy(state)
    for _ in range(500):
        new = f.step(state)
        loss = f.dt * f.dissipation((state.vx + new.vx) / 2, (state.vz + new.vz) / 2)
        assert abs(f.energy(new) - f.energy(state) + loss) < 5e-13 * initial
        assert 0 < f.energy(new) <= initial * (1 + 1e-11)
        state = new
    if bc.bottom == "free":
        assert abs(f.energy(state) / initial - 1) < 1e-11


def test_history_and_final_state_do_not_alias_reused_forward_workspace():
    f, r = pair(BoundaryConditions())
    initial = random_state(f)
    saved_initial = initial.copy()
    source = StressPointSource(4, 5, 1.0, -0.3, 0.7)
    rec = [PointReceiver("vx", 4, 6)]
    stf = np.sin(np.arange(80))
    a = f.run_forward(
        stf, source, rec, initial_state=initial, store_velocity_history=True
    )
    b = r.run_forward(
        stf, source, rec, initial_state=initial, store_velocity_history=True
    )
    saved_final = a.final_state.copy()
    f.run_forward(stf, source, rec)
    for name in FIELDS:
        np.testing.assert_array_equal(
            getattr(initial, name), getattr(saved_initial, name)
        )
        np.testing.assert_array_equal(
            getattr(a.final_state, name), getattr(saved_final, name)
        )
    for aa, bb in zip(a.velocity_history, b.velocity_history):
        for v, w in zip(aa, bb):
            np.testing.assert_allclose(v, w, atol=3e-14, rtol=2e-12)


def test_boundary_parameter_derivative_with_compiled_forward():
    f, _ = pair(BoundaryConditions(top="absorbing"))
    source = StressPointSource(4, 5, 1.0, -0.3, 0.7)
    rec = [PointReceiver("vx", 2, 3), PointReceiver("vz", 3, 4)]
    stf = 100 * np.exp(-(((np.arange(240) * f.dt - 0.4) / 0.15) ** 2))
    observed = np.zeros((len(rec), len(stf)))
    J, _, gp, gs = f.objective_and_gradient_log_vp_vs(stf, observed, source, rec)
    p = np.zeros_like(f.model.vp)
    p[[0, -1], :] = 1
    p[:, [0, -1]] = 1
    p /= np.linalg.norm(p)
    derivative = float(np.sum(gs * p))
    values = []
    for sign in (-1, 1):
        m = prepare_rectangular_model(
            f.model.vp, f.model.vs * np.exp(sign * 1e-5 * p), f.model.rho
        )
        ff = NumbaSBPSAT(m, f.op_x.h, f.op_z.h, f.dt, f.boundaries, numba_threads=1)
        data = ff.run_forward(stf, source, rec).data
        values.append(0.5 * f.dt * np.sum(data * data))
    fd = (values[1] - values[0]) / (2e-5)
    assert abs(fd - derivative) < 2e-6 * abs(fd)


def test_external_lower_precision_state_is_promoted_without_mutating_input():
    f, r = pair(BoundaryConditions())
    state = random_state(f)
    for name in FIELDS:
        setattr(state, name, getattr(state, name).astype(np.float32))
    before = state.copy()
    a, b = f.step(state), r.step(state)
    for name in FIELDS:
        assert getattr(a, name).dtype == np.float64
        np.testing.assert_array_equal(getattr(state, name), getattr(before, name))
        np.testing.assert_allclose(
            getattr(a, name), getattr(b, name), rtol=2e-14, atol=3e-15
        )
    with pytest.raises(TypeError, match="float64"):
        f.advance_inplace(state, None, 0.0)
