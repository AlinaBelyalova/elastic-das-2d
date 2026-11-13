"""Native legacy-sponge forward, exact reverse/VJP and checkpoint checks."""

from dataclasses import fields

import numpy as np
import pytest

from src.elastic.solver_numpy import make_sponge_mask
from src.sbp_sat.elastic2d import PointReceiver, StressPointSource
from src.sbp_sat.rectangular import RectangularSBPSAT, prepare_rectangular_model
from src.sbp_sat.sponge import GaussianSponge, NumbaSpongeSBPSAT, SpongeSBPSAT


def case(cls=SpongeSBPSAT, *, rate=4.0, model=None):
    x, z = np.indices((13, 14))
    model = model or prepare_rectangular_model(
        3.2 + 0.03 * np.sin(x), 1.5 + 0.01 * np.cos(z), 2.2 + 0.02 * np.sin(x + z)
    )
    kw = (
        dict(numba_threads=1)
        if cls is NumbaSpongeSBPSAT
        else dict(sparse_derivatives=True)
    )
    return cls(model, 1.1, 0.9, 0.015, sponge=GaussianSponge(2.2, 2.7, rate), **kw)


def random_state(f, rng):
    s = f.zero_state()
    for field in fields(s):
        a = getattr(s, field.name)
        a[:] = rng.normal(size=a.shape)
    return s


def test_normal_node_mask_exactly_matches_old_quadratic_gaussian_law():
    m = prepare_rectangular_model(
        np.full((17, 19), 3.0), np.full((17, 19), 1.5), np.ones((17, 19))
    )
    f = SpongeSBPSAT(m, 1, 1, 0.01, sponge=GaussianSponge(4, 4, 8))
    old = make_sponge_mask(17, 19, 4, 8, 0.01, free_surface=True)
    np.testing.assert_allclose(f.sponge_masks[2], old, rtol=2e-16, atol=2e-16)
    assert f.sponge_masks[2][8, 0] == 1  # free top, outside side strips
    assert f.sponge_masks[2][8, -1] < 1
    assert f.sponge_masks[2][0, -1] < f.sponge_masks[2][0, 8]
    assert f.sponge_masks[0][0, 8] > f.sponge_masks[2][0, 8]  # true half-x location


@pytest.mark.parametrize("cls", [SpongeSBPSAT, NumbaSpongeSBPSAT])
def test_step_is_exact_post_multiplier_of_original_sat_and_does_not_mutate_input(cls):
    f = case(cls)
    base = RectangularSBPSAT(f.model, 1.1, 0.9, f.dt, sparse_derivatives=True)
    s = random_state(f, np.random.default_rng(18))
    saved = s.copy()
    source = StressPointSource(1, 11, 1.0, -0.4, 0.2)  # algebraic test even in sponge
    actual = f.step(s, source=source, amplitude=0.3)
    raw = base.step(s, source=source, amplitude=0.3)
    for field, mask in zip(fields(s), f.sponge_masks):
        np.testing.assert_array_equal(
            getattr(s, field.name), getattr(saved, field.name)
        )
        np.testing.assert_allclose(
            getattr(actual, field.name),
            getattr(raw, field.name) * mask,
            rtol=5e-14,
            atol=1e-14,
        )


@pytest.mark.parametrize("cls", [SpongeSBPSAT, NumbaSpongeSBPSAT])
def test_zero_rate_reproduces_impedance_only_forward(cls):
    f = case(cls, rate=0.0)
    base = RectangularSBPSAT(f.model, 1.1, 0.9, f.dt, sparse_derivatives=True)
    stf = np.random.default_rng(42).normal(size=90)
    src, rec = (
        StressPointSource(5, 6),
        [PointReceiver("vx", 1, 11), PointReceiver("vz", 10, 2)],
    )
    a, b = f.run_forward(stf, src, rec), base.run_forward(stf, src, rec)
    np.testing.assert_allclose(a.data, b.data, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("cls", [SpongeSBPSAT, NumbaSpongeSBPSAT])
def test_full_source_data_transpose_includes_sponge_and_receivers_in_strips(cls):
    f = case(cls)
    rng = np.random.default_rng(43)
    stf = rng.normal(size=130)
    src = StressPointSource(1, 10, 1.0, -0.4, 0.3)
    rec = [
        PointReceiver("vx", 1, 11),
        PointReceiver("vz", 10, 12),
        PointReceiver("vx", 7, 5),
    ]
    data = f.run_forward(stf, src, rec).data
    q = rng.normal(size=data.shape)
    grad = f.run_adjoint_to_source(q, src, rec)
    np.testing.assert_allclose(
        np.vdot(data, q), np.vdot(stf, grad), rtol=2e-12, atol=1e-14
    )


@pytest.mark.parametrize("parameter", ["vp", "vs", "both"])
def test_full_gradient_and_checkpoints_include_damping_and_boundary_impedance(
    parameter,
):
    f = case(NumbaSpongeSBPSAT)
    rng = np.random.default_rng(44)
    stf = np.sin(np.arange(170) * 0.2)
    src = StressPointSource(5, 7, 1.0, -0.4, 0.3)
    rec = [
        PointReceiver("vx", 1, 11),
        PointReceiver("vz", 10, 12),
        PointReceiver("vx", 7, 5),
    ]
    observed = rng.normal(size=(3, len(stf))) * 0.0001
    full = f.objective_and_gradient_log_vp_vs(stf, observed, src, rec)
    for interval in [1, 19, "auto"]:
        checkpoint = f.objective_and_gradient_log_vp_vs(
            stf, observed, src, rec, checkpoint_interval=interval
        )
        for a, b in zip(full, checkpoint):
            np.testing.assert_array_equal(a, b)
    p, s = rng.normal(size=f.model.vp.shape), rng.normal(size=f.model.vp.shape)
    p /= np.linalg.norm(p)
    s /= np.linalg.norm(s)
    if parameter == "vp":
        s *= 0
    if parameter == "vs":
        p *= 0
    derivative = np.sum(full[2] * p + full[3] * s)

    def objective(epsilon):
        m = prepare_rectangular_model(
            f.model.vp * np.exp(epsilon * p),
            f.model.vs * np.exp(epsilon * s),
            f.model.rho,
        )
        ff = case(NumbaSpongeSBPSAT, model=m)
        residual = ff.run_forward(stf, src, rec).data - observed
        return 0.5 * ff.dt * np.sum(residual**2)

    eps = 1e-5
    fd = (objective(eps) - objective(-eps)) / (2 * eps)
    np.testing.assert_allclose(fd, derivative, rtol=2e-7, atol=1e-14)
    remainders = [
        abs(objective(e) - full[0] - e * derivative)
        for e in [0.01, 0.005, 0.0025, 0.00125]
    ]
    slopes = np.log2(np.array(remainders[:-1]) / remainders[1:])
    assert np.all((slopes > 1.9) & (slopes < 2.1)), slopes


def test_numba_csr_and_owned_restart_parity():
    f, g = case(NumbaSpongeSBPSAT), case()
    a = random_state(f, np.random.default_rng(45))
    b = a.copy()
    with f.thread_context():
        for _ in range(30):
            a = f.advance_inplace(a, None, 0.0)
            b = g.step(b)
    for field in fields(a):
        np.testing.assert_allclose(
            getattr(a, field.name), getattr(b, field.name), rtol=2e-12, atol=2e-14
        )
    c = a.copy()
    with f.thread_context():
        for _ in range(10):
            a = f.advance_inplace(a, None, 0.0)
            c = f.advance_inplace(c, None, 0.0)
    for field in fields(a):
        np.testing.assert_array_equal(getattr(a, field.name), getattr(c, field.name))


@pytest.mark.parametrize("execution", ["csr", "numba"])
def test_pipeline_native_snapshots_and_exact_registered_das_with_sponge(execution):
    from tests.test_sbp_sat_pipeline import case as pipeline_case
    from src.elastic.simulator import run_forward_simulation
    from src.sbp_sat.simulator import prepare_simulation

    model, source, receivers, length = pipeline_case(nt=100)
    sponge = GaussianSponge(2, 2, 4)
    run, das = run_forward_simulation(
        model,
        source,
        receivers,
        length,
        backend="sbp_sat",
        free_surface=True,
        sbp_execution=execution,
        sbp_sponge=sponge,
        snapshot_stride=20,
    )
    p = prepare_simulation(
        model, source, receivers, length, execution=execution, sponge=sponge
    )
    direct = p.solver.run_forward(source.stf.values, p.source, p.observation).data
    np.testing.assert_allclose(das.data, direct, atol=1e-14)
    np.testing.assert_array_equal(das.channel_indices, p.observation.channel_indices)
    np.testing.assert_array_equal(
        run.t, (np.arange(model.grid.nt) + 0.5) * model.grid.dt
    )
    assert run.snapshots_vz.shape == (5, model.grid.nx, model.grid.nz - 1)
    assert (
        run.solver_metadata["boundary_scheme_version"] == "impedance_gaussian_sponge_v1"
    )


@pytest.mark.parametrize("bad", ["source", "receivers"])
def test_production_support_is_rejected_instead_of_clipped(bad):
    from dataclasses import replace
    from tests.test_sbp_sat_pipeline import case as pipeline_case
    from src.sbp_sat.simulator import prepare_simulation

    m, source, receivers, length = pipeline_case(nt=40)
    if bad == "source":
        source = replace(source, x_embedded_m=m.grid.x0 + 1.8 * m.grid.dx)
    else:
        receivers = replace(receivers, x=receivers.x + 6.8)
    with pytest.raises(ValueError, match="intersects sponge"):
        prepare_simulation(m, source, receivers, length, sponge=GaussianSponge(2, 2, 4))


def test_old_boundary_control_refuses_new_sponge_packages(tmp_path):
    from tests.test_sbp_sat_boundary_control import make_reference
    from scripts.validation.sbp_sat.event_boundary_control import load_reference

    path = tmp_path / "reference.npz"
    make_reference(path)
    with np.load(path) as f:
        values = {k: f[k] for k in f.files}
    values["sponge_rate_per_s"] = 80.0
    np.savez(path, **values)
    with pytest.raises(ValueError, match="cannot replay a sponge run"):
        load_reference(path)


def test_sponge_alone_contracts_physical_elastic_energy_on_native_grids():
    f = case()
    s = random_state(f, np.random.default_rng(51))
    damped = f._multiply(s.copy())
    assert f.energy(damped, modified=False) < f.energy(s, modified=False)


def test_small_step_spectrum_and_long_free_decay():
    from scipy.linalg import eigvals

    m = prepare_rectangular_model(
        np.full((9, 9), 3.0), np.full((9, 9), 1.5), np.full((9, 9), 2.0)
    )
    f = SpongeSBPSAT(
        m, 1, 1, 0.04, sponge=GaussianSponge(2, 2, 3), sparse_derivatives=True
    )
    template = f.zero_state()
    names = [p.name for p in fields(template)]
    sizes = [getattr(template, n).size for n in names]
    count = sum(sizes)
    matrix = np.empty((count, count))
    for j in range(count):
        s = f.zero_state()
        i = j
        for name, size in zip(names, sizes):
            if i < size:
                getattr(s, name).flat[i] = 1
                break
            i -= size
        out = f.step(s)
        matrix[:, j] = np.concatenate([getattr(out, n).ravel() for n in names])
    assert np.max(abs(eigvals(matrix))) < 1 + 1e-10
    f = case(NumbaSpongeSBPSAT)
    state = random_state(f, np.random.default_rng(52))
    initial = f.energy(state, modified=False)
    with f.thread_context():
        for it in range(4000):
            state = f.advance_inplace(state, None, 0.0)
            if it % 20 == 0:
                assert f.energy(state, modified=False) < 2 * initial
    assert f.energy(state, modified=False) < 0.5 * initial


@pytest.mark.parametrize(
    "values", [(float("nan"), 1, 2), (-1, 1, 2), (7, 1, 2), (1, 12, 2), (1, 1, 100)]
)
def test_invalid_overlap_or_excessive_per_step_decay_is_rejected(values):
    f = case()
    with pytest.raises(ValueError):
        SpongeSBPSAT(f.model, 1.1, 0.9, f.dt, sponge=GaussianSponge(*values))
