"""Production source/STF -> bounded SBP -> receivers -> DAS -> saved comparison."""

import json
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

from src.common.grid import Grid2D
from src.common.model import ElasticModel2D
from src.das.receivers import Receivers2D
from src.elastic.source import build_dc_source
from src.elastic.source_injection import build_stress_source_injection
from src.elastic.simulator import run_forward_simulation
from src.sbp_sat.rectangular import BoundaryConditions
from src.sbp_sat.simulator import (
    prepare_simulation,
    configure_time_grid,
    objective_and_gradient_log_vp_vs,
)
from src.sbp_sat.sources import build_moment_source
from scripts.safod import settings, compare_event


def case(*, near_boundary=False, nt=320):
    g = Grid2D(13, 14, 1.2, 0.9, nt, 0.005, x0=-3.7, z0=2.3)
    x, z = np.indices(g.shape)
    m = ElasticModel2D(
        g, 3.2 + 0.05 * np.sin(x), 1.5 + 0.02 * np.cos(z), 2.2 + 0.02 * np.cos(x + z)
    )
    src = build_dc_source(
        grid=g,
        x_m=g.x0 + (0.7 if near_boundary else 5.2) * g.dx,
        z_m=g.z0 + (0.8 if near_boundary else 5.3) * g.dz,
        theta_deg=35,
        scalar_moment=20,
        nt=g.nt,
        dt=g.dt,
        f0_hz=5,
        spreading="bilinear",
        source_time_mode="ricker_moment",
    )
    theta = np.linspace(-0.7, 0.7, 21)
    r = Receivers2D(
        x=1.5 + 2 * np.sin(theta),
        z=6.5 + 2 * np.cos(theta),
        tx=np.cos(theta),
        tz=-np.sin(theta),
        s=2 * (theta - theta[0]),
    )
    return m, src, r, 0.57


@pytest.mark.parametrize("near_boundary", [False, True])
def test_source_quadrature_moment_and_injection_transpose(near_boundary):
    m, src, r, L = case(near_boundary=near_boundary)
    p = prepare_simulation(m, src, r, L)
    f, s = p.solver, p.source
    a = f.zero_state()
    s.inject(a.sxx, a.szz, a.sxz, 1)
    for field, H, moment in [
        (a.sxx, f.wn, src.m2d.Mxx),
        (a.szz, f.wn, src.m2d.Mzz),
        (a.sxz, f.ws, src.m2d.Mxz),
    ]:
        assert np.sum(H * field) == pytest.approx(moment, rel=2e-15)
    rng = np.random.default_rng(174)
    q = [rng.normal(size=x.shape) for x in (a.sxx, a.szz, a.sxz)]
    assert sum(
        np.vdot(x, y) for x, y in zip((a.sxx, a.szz, a.sxz), q)
    ) == pytest.approx(s.gather(*q))
    if not near_boundary:
        old = build_stress_source_injection(src, m.grid)
        np.testing.assert_array_equal(s.normal.ix[0], old.normal_ix)
        np.testing.assert_array_equal(s.shear.iz[0], old.shear_iz)
        np.testing.assert_allclose(
            s.normal.weights[0], old.normal_w / (m.grid.dx * m.grid.dz), rtol=2e-15
        )
        np.testing.assert_allclose(
            s.shear.weights[0], old.shear_w / (m.grid.dx * m.grid.dz), rtol=2e-15
        )
    else:
        # H^-1, not a uniform cell-area delta inside the SBP closure.
        assert not np.allclose(f.wn[s.normal.ix, s.normal.iz], m.grid.dx * m.grid.dz)


@pytest.mark.parametrize(
    "bc",
    [
        BoundaryConditions(),
        BoundaryConditions(top="absorbing"),
        BoundaryConditions(top="free", bottom="free", left="free", right="free"),
    ],
)
def test_sparse_matches_dense_state_adjoint_and_energy(bc):
    m, src, r, L = case()
    p = prepare_simulation(m, src, r, L)
    f = type(p.solver)(
        p.solver.model, m.grid.dx, m.grid.dz, m.grid.dt, bc, sparse_derivatives=True
    )
    d = type(f)(f.model, m.grid.dx, m.grid.dz, m.grid.dt, bc)
    rng = np.random.default_rng(771)
    a = f.zero_state()
    for name in ("vx", "vz", "sxx", "szz", "sxz"):
        getattr(a, name)[:] = rng.normal(size=getattr(a, name).shape)
    af, ad = (
        f.step(a, source=p.source, amplitude=0.7),
        d.step(a, source=p.source, amplitude=0.7),
    )
    for name in ("vx", "vz", "sxx", "szz", "sxz"):
        np.testing.assert_allclose(
            getattr(af, name), getattr(ad, name), atol=4e-16, rtol=3e-15
        )
    out = f.run_forward(src.stf.values, p.source, p.observation)
    ref = d.run_forward(src.stf.values, p.source, p.observation)
    np.testing.assert_allclose(out.data, ref.data, atol=1e-14, rtol=2e-12)
    q = rng.normal(size=out.data.shape)
    adj = f.run_adjoint_to_source(q, p.source, p.observation)
    np.testing.assert_allclose(
        np.vdot(out.data, q), np.vdot(src.stf.values, adj), rtol=5e-13
    )
    af = f.step(a)
    diss = f.dissipation((a.vx + af.vx) / 2, (a.vz + af.vz) / 2)
    assert abs(f.energy(af) - f.energy(a) + f.dt * diss) < 1e-13 * f.energy(a)


def test_dispatch_native_snapshots_time_axes_and_same_das_chain():
    m, src, r, L = case()
    run, das = run_forward_simulation(
        m, src, r, L, backend="sbp_sat", free_surface=True, snapshot_stride=37
    )
    p = prepare_simulation(m, src, r, L)
    direct = p.solver.run_forward(src.stf.values, p.source, p.observation)
    np.testing.assert_allclose(das.data, direct.data, atol=1e-14)
    np.testing.assert_array_equal(das.channel_indices, p.observation.channel_indices)
    np.testing.assert_array_equal(run.t, (np.arange(m.grid.nt) + 0.5) * m.grid.dt)
    np.testing.assert_array_equal(run.t_sigma, np.arange(m.grid.nt + 1) * m.grid.dt)
    np.testing.assert_array_equal(run.snapshot_times_v, run.t[::37])
    assert run.snapshots_vz.shape == (len(run.t[::37]), m.grid.nx, m.grid.nz - 1)
    assert run.solver_metadata["solver_backend"] == "sbp_sat"
    assert run.solver_metadata["source_stf_sampling_offset_steps"] == 0.5
    # No velocity is injected at the wrong time or doubled as a stress source.
    np.testing.assert_array_equal(run.receiver_vx[:, 0], 0)
    np.testing.assert_array_equal(run.receiver_vz[:, 0], 0)


def test_time_grid_is_certified_without_changing_space_or_shortening_duration():
    m, src, r, L = case()
    old = replace(m, grid=replace(m.grid, dt=0.2, nt=20))
    new, report = configure_time_grid(old)
    assert new.grid.dt < report["dt_sufficient_limit_s"]
    assert (new.grid.nt - 1) * new.grid.dt >= (old.grid.nt - 1) * old.grid.dt
    assert new.grid.nt > old.grid.nt
    np.testing.assert_array_equal(new.vp, old.vp)
    np.testing.assert_array_equal(new.grid.x, old.grid.x)
    np.testing.assert_array_equal(new.grid.z, old.grid.z)
    assert old.grid.dt == 0.2 and old.grid.nt == 20
    _, explicit = configure_time_grid(old, duration_s=2.1)
    assert (explicit["nt"] - 1) * explicit["dt_s"] >= 2.1


@pytest.mark.parametrize("parameter", ["vp", "vs", "both"])
def test_pipeline_model_gradient_with_physical_source_and_sparse_derivatives(parameter):
    m, src, r, L = case(nt=400)
    observed = np.zeros((len(prepare_simulation(m, src, r, L).observation), m.grid.nt))
    J, data, gp, gs = objective_and_gradient_log_vp_vs(m, src, r, observed, L)
    rng = np.random.default_rng(16)
    p, s = rng.normal(size=m.grid.shape), rng.normal(size=m.grid.shape)
    p /= np.linalg.norm(p)
    s /= np.linalg.norm(s)
    if parameter == "vp":
        s *= 0
    if parameter == "vs":
        p *= 0
    derivative = np.sum(p * gp + s * gs)
    rem = []
    for eps in (0.01, 0.005, 0.0025, 0.00125):
        mm = ElasticModel2D(
            m.grid, m.vp * np.exp(eps * p), m.vs * np.exp(eps * s), m.rho
        )
        ff = prepare_simulation(mm, src, r, L)
        d = ff.solver.run_forward(src.stf.values, ff.source, ff.observation).data
        j = 0.5 * m.grid.dt * np.sum(d * d)
        rem.append(abs(j - J - eps * derivative))
    slopes = np.log2(np.array(rem[:-1]) / rem[1:])
    assert np.all((slopes > 1.9) & (slopes < 2.1)), slopes
    eps = 1e-5
    values = []
    for sign in (-1, 1):
        mm = ElasticModel2D(
            m.grid, m.vp * np.exp(sign * eps * p), m.vs * np.exp(sign * eps * s), m.rho
        )
        pp = prepare_simulation(mm, src, r, L)
        d = pp.solver.run_forward(src.stf.values, pp.source, pp.observation).data
        values.append(0.5 * m.grid.dt * np.sum(d * d))
    assert abs((values[1] - values[0]) / (2 * eps) - derivative) < 2e-6 * abs(
        derivative
    )


def test_pipeline_rejects_unsupported_source_time_geometry_and_order():
    m, src, r, L = case()
    p = prepare_simulation(m, src, r, L)
    with pytest.raises(ValueError, match="support"):
        build_moment_source(p.solver, replace(src, x_embedded_m=m.grid.x0), m.grid)
    with pytest.raises(ValueError, match="nt/dt"):
        prepare_simulation(
            replace(m, grid=replace(m.grid, dt=m.grid.dt * 1.01)), src, r, L
        )
    with pytest.raises(ValueError, match="4-2"):
        run_forward_simulation(m, src, r, L, backend="sbp_sat", half_order=3)
    with pytest.raises(ValueError, match="snapshot_stride"):
        run_forward_simulation(m, src, r, L, backend="sbp_sat", snapshot_stride=0)


def test_safod_backend_paths_are_isolated_and_legacy_paths_are_unchanged():
    legacy = settings.FORWARD_EVENT_DIR / "dc035"
    assert settings.forward_dir_for_theta(35) == legacy
    assert (
        settings.forward_dir_for_theta(35, "sbp_sat")
        == settings.FORWARD_EVENT_DIR / "sbp_sat/dc035"
    )
    for backend in settings.FORWARD_BACKENDS:
        pkg = compare_event.synthetic_package_for_model(35, "bill_logs", backend)
        assert pkg.parent == settings.forward_dir_for_theta(35, backend) / "bill_logs"
        assert (
            compare_event.comparison_dir_for_model(35, "bill_logs", backend)
            == settings.comparison_dir_for_theta(35, backend) / "bill_logs"
        )
    assert settings.forward_run_tag(35) == "n120_g80_xplus500_dc035"
    assert "sbp_sat" in settings.forward_run_tag(35, "sbp_sat")
    with pytest.raises(ValueError):
        settings.forward_dir_for_theta(35, "unknown")


def _trial_comcat_cache(tmp_path):
    path = tmp_path / "focal_mechanisms.json"
    path.write_text(json.dumps({"nc123": {
        "code": "nc123_fm1",
        "np1": {"strike": 0, "dip": 90, "rake": 0},
        "review_status": "automatic",
    }}))
    return path


@pytest.mark.parametrize("use_comcat", [False, True])
def test_saved_sbp_package_passes_full_compare_with_shared_guides(
    tmp_path, monkeypatch, use_comcat
):
    # Only the external 3-D eikonal predictor is stubbed; actual SBP, source,
    # sampling, DAS, package loading, filtering and all comparison plots run.
    m, src, r, L = case(nt=520)
    source_metadata = {}
    theta = 35
    if use_comcat:
        from src.safod.focal_mechanism import COMCAT_MECHANISM, load_projected_comcat_source

        tensor, source_metadata = load_projected_comcat_source(
            path=_trial_comcat_cache(tmp_path),
            event_id="NC123",
            profile_u_e=2**-0.5,
            profile_u_n=2**-0.5,
            scalar_moment=src.scalar_moment,
        )
        src = replace(src, m2d=tensor)
        source_metadata["source_mechanism"] = COMCAT_MECHANISM
        theta = None
    run, das = run_forward_simulation(
        m, src, r, L, backend="sbp_sat", free_surface=True
    )
    channels = 650 + 2.5 * np.arange(r.nrec)
    syn = tmp_path / "synthetic.npz"
    real = tmp_path / "real.npz"
    np.savez(
        syn,
        **run.solver_metadata,
        t=run.t,
        dt=m.grid.dt,
        das_data=das.data,
        das_raw_channels=channels[das.channel_indices],
        event_id="NC123",
        source_f0_hz=5,
        source_theta_deg=np.nan if use_comcat else theta,
        **source_metadata,
        source_time_function_t0_s=src.stf.t0,
        initial_model_name="bill_logs",
        run_tag=settings.forward_run_tag(theta, "sbp_sat"),
        model_type="manufactured",
    )
    rt = np.arange(-2.0, 2.01, 0.005)
    np.savez(
        real,
        ev_id="NC123",
        t=rt,
        fs=200,
        raw_channels=channels,
        das_data_unfiltered=np.tile(np.sin(2 * np.pi * 5 * rt), (r.nrec, 1)),
    )
    out = tmp_path / "comparison"
    monkeypatch.setattr(compare_event, "REAL_EVENT_PACKAGE", real)
    monkeypatch.setattr(compare_event, "synthetic_package_for_model", lambda *a: syn)
    calls = []

    def predict(*args):
        calls.append(args)
        guides = dict(
            arrival_channels=channels,
            p_arrivals_s=np.full(r.nrec, 0.4),
            s_arrivals_s=np.full(r.nrec, 0.8),
            model_label="stub Zhang 3-D",
        )
        detail = dict(method="external_3d_stub", model_name="zhang2009_native_3d")
        return guides, dict(real=detail, synthetic=detail)

    monkeypatch.setattr(compare_event, "predict_comparison_arrivals", predict)
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_event",
            "--backend",
            "sbp_sat",
            "--initial-model",
            "bill_logs",
            *(["--comcat-mechanism"] if use_comcat else ["--theta-deg", "35"]),
            "--output-dir",
            str(out),
        ],
    )
    compare_event.main()
    assert len(calls) == 1
    summary = pd.read_csv(out / "comparison_summary.csv").iloc[0]
    assert summary.solver_backend == "sbp_sat"
    assert summary.synthetic_display_shift_s == pytest.approx(-src.stf.t0)
    assert summary.shared_3d_prediction_reference
    assert len(list(out.glob("*.png"))) == 4
    if use_comcat:
        assert summary.source_comcat_product_code == "nc123_fm1"
        assert np.isnan(summary.theta_deg)
    else:
        assert summary.theta_deg == 35
    # A misplaced legacy package must be rejected before predictions/plotting.
    with np.load(syn) as pkg:
        values = {k: pkg[k] for k in pkg.files if k != "solver_backend"}
    np.savez(syn, **values, solver_backend="numba_fused")
    with pytest.raises(ValueError, match="differs from saved"):
        compare_event.main()


@pytest.mark.parametrize(
    "dry_run,save_gif", [(True, False), (False, False), (False, True)]
)
@pytest.mark.parametrize("use_comcat", [False, True])
def test_actual_safod_driver_preflight_save_and_native_snapshot_axes(
    tmp_path, monkeypatch, dry_run, save_gif, use_comcat
):
    # Synthetic external model loader/geometry, actual driver/source/SBP/DAS/save.
    # The coarse anisotropic grid is an interface check, not a wave-accuracy QC.
    from scripts.safod import run_forward as driver
    import matplotlib.pyplot as plt

    channels = 650.0 + np.arange(21)
    xx, zz = np.linspace(-5, 5, 21), np.linspace(1800, 3000, 21)
    geometry = tmp_path / "geometry.csv"
    pd.DataFrame(dict(Channel=channels, X_2D_m=xx, Z_2D_m=zz)).to_csv(
        geometry, index=False
    )
    real = tmp_path / "real.npz"
    np.savez(
        real,
        geometry_csv=str(geometry),
        raw_channels=channels,
        ev_id="NC123",
        ev_origin_time="2026-06-18T21:04:42Z",
        ev_mag=1.6,
        ev_depth_km=2.5,
        profile_u_e=2**-0.5,
        profile_u_n=2**-0.5,
        event_x_model_m=3.7,
        event_z_model_m=2501.2,
        gauge_length_m=17.3,
        channel_spacing_m=60.0,
    )
    monkeypatch.setattr(driver, "REAL_EVENT_PACKAGE", real)
    monkeypatch.setattr(driver, "SAFOD_SCIENTIFIC_X_MIN_M", -50.0)
    monkeypatch.setattr(driver, "SAFOD_SCIENTIFIC_X_MAX_M", 50.0)
    monkeypatch.setattr(driver, "FORWARD_PADDING_CELLS", 1)
    monkeypatch.setattr(driver, "SBP_X_MIN_M", -55.0)
    monkeypatch.setattr(driver, "SBP_X_MAX_M", 55.0)
    monkeypatch.setattr(driver, "SBP_Z_MAX_M", 5005.0)
    monkeypatch.setattr(driver, "SBP_SPONGE_WIDTH_M", 5.0)
    root = tmp_path / "forward"
    source_dir = "comcat" if use_comcat else "dc035"
    monkeypatch.setattr(
        driver, "forward_dir_for_theta", lambda theta, backend: root / backend / settings.theta_dir_name(theta)
    )

    def build_model(**kwargs):
        # Legacy bounds and padding above must not rebuild a second SBP box.
        assert kwargs["x_grid_min_m"] == -55.0
        assert kwargs["x_grid_max_m"] == 55.0
        assert kwargs["z_max_m"] == 5005.0
        assert kwargs["z_padding_bottom_m"] == 0.0
        g = Grid2D(23, 33, 5.0, 5005 / 32, kwargs["nt"], 0.00015, x0=-55.0)
        m = ElasticModel2D(
            g,
            np.full(g.shape, 3200.0),
            np.full(g.shape, 1700.0),
            np.full(g.shape, 2200.0),
        )
        meta = SimpleNamespace(
            x_cable_end_m=xx[-1],
            z_cable_end_m=zz[-1],
            x_tie_m=0.0,
            z_tie_m=3000.0,
            fault_offset_from_cable_m=105.0,
            fault_dip_deg=82.0,
            fault_dip_sign=-1.0,
            model_type="initial_bill_logs",
            x_fault_line=np.array([0.0, 0.0]),
            z_fault_line=np.array([0.0, 5000.0]),
        )
        return g, m, xx.copy(), zz.copy(), meta

    monkeypatch.setattr(driver, "build_initial_model", build_model)
    monkeypatch.setattr(driver, "plot_safod_model", lambda **kwargs: plt.subplots())
    monkeypatch.setattr(driver, "place_safod_legend", lambda *a, **k: None)
    monkeypatch.setattr(driver, "plot_receiver_gather", lambda **kwargs: None)
    monkeypatch.setattr(driver, "plot_das_gather", lambda **kwargs: None)
    gif = []

    def make_gif(**kwargs):
        g, s = kwargs["grid"], kwargs["snapshots_vz"]
        assert s.shape[1:] == g.shape
        assert kwargs["plot_x_min_m"] == -50.0
        assert kwargs["plot_x_max_m"] == 50.0
        assert kwargs["plot_z_min_m"] == 0.0
        assert kwargs["plot_z_max_m"] == 5000.0
        gif.append(g)

    monkeypatch.setattr(driver, "make_wavefield_gif", make_gif)
    argv = [
        "run_forward",
        "--backend",
        "sbp_sat",
        "--initial-model",
        "bill_logs",
        *(
            ["--comcat-mechanism", "--focal-mechanisms-file", str(_trial_comcat_cache(tmp_path))]
            if use_comcat else ["--theta-deg", "35"]
        ),
        "--duration-s",
        "1.2",
        "--save-gif" if save_gif else "--no-gif",
    ]
    if dry_run:
        argv.append("--dry-run")
    monkeypatch.setattr("sys.argv", argv)
    driver.main()
    output = root / f"sbp_sat/{source_dir}/bill_logs/outputs_safod_initial_forward.npz"
    if dry_run:
        assert not output.exists() and not gif
        return
    with np.load(output) as pkg:
        assert str(pkg["solver_backend"]) == "sbp_sat"
        if use_comcat:
            assert pkg["source_comcat_product_code"].item() == "nc123_fm1"
            np.testing.assert_allclose(
                pkg["source_moment_tensor_xz"] / float(pkg["source_scalar_moment"]),
                [[1.0, 0.0], [0.0, 0.0]], atol=1e-14,
            )
        assert float(pkg["source_x"]) == 3.7 and float(pkg["source_z"]) == 2501.2
        assert float(pkg["source_time_function_t0_s"]) == 0.12
        assert float(pkg["velocity_time_offset_steps"]) == 0.5
        assert float(pkg["dt"]) < float(pkg["sbp_dt_sufficient_limit_s"])
        assert pkg["t_sigma"].size == int(pkg["nt"]) + 1
        assert pkg["das_data"].shape[1] == int(pkg["nt"])
        np.testing.assert_array_equal(
            pkg["das_raw_channels"], channels[pkg["das_channel_indices"]]
        )
        assert int(pkg["n_boundary"]) == 1 and float(pkg["gamma_s"]) == 80
        assert float(pkg["sponge_width_x_m"]) == 5
        assert str(pkg["boundary_scheme_version"]) == "impedance_gaussian_sponge_v1"
        assert int(pkg["material_padding_cells"]) == 0
        assert str(pkg["x_domain_mode"]) == "explicit_computational_bounds"
        assert float(pkg["scientific_x_min_m"]) == float(pkg["grid_x"][0])
        assert float(pkg["scientific_x_max_m"]) == float(pkg["grid_x"][-1])
        assert float(pkg["scientific_z_max_m"]) == float(pkg["grid_z"][-1])
        assert str(pkg["receiver_geometry_mode"]) == "exact_registered_channel_centres"
        if save_gif:
            with np.load(output.parent / "snapshots_vz.npz") as snap:
                np.testing.assert_array_equal(
                    snap["snapshot_grid_z"], pkg["grid_z"][:-1] + float(pkg["dz"]) / 2
                )
                assert snap["snapshots_vz"].shape[1:] == (23, 32)
                assert len(gif) == 1 and gif[0].nz == 32
        else:
            assert not gif and not (output.parent / "snapshots_vz.npz").exists()
