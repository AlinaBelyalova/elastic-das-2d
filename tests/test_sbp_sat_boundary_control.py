"""Frozen-material boundary movement, replay protection and full output smoke."""

import numpy as np
import pytest

from scripts.validation.sbp_sat.event_boundary_control import (
    extend_model,
    load_reference,
    rebuild_inputs,
    run_control,
)
from src.common.grid import Grid2D
from src.common.model import ElasticModel2D
from src.das.receivers import build_receivers_from_channel_centres
from src.elastic.source import build_dc_source
from src.sbp_sat.simulator import run_forward_simulation


def material():
    g = Grid2D(21, 23, 50, 40, 1100, 0.002, -300, 0)
    i, j = np.indices(g.shape)
    return ElasticModel2D(
        g, 3200 + 30 * np.sin(i), 1700 + 20 * np.cos(j), 2200 + 10 * np.cos(i + j)
    )


def make_reference(path):
    model = material()
    g = model.grid
    source = build_dc_source(
        grid=g,
        x_m=123.7,
        z_m=241.3,
        theta_deg=35,
        scalar_moment=1e8,
        nt=g.nt,
        dt=g.dt,
        f0_hz=10,
        spreading="bilinear",
        source_time_mode="ricker_moment",
    )
    receivers = build_receivers_from_channel_centres(
        x=np.linspace(75, 100, 15),
        z=np.linspace(100, 380, 15),
        s=np.arange(15) * 20.0,
        grid=g,
    )
    run, das = run_forward_simulation(model, source, receivers, 27.3)
    raw = 220 + np.arange(receivers.nrec)
    values = dict(
        run.solver_metadata,
        source_mode="catalog_event",
        source_time_mode="ricker_moment",
        source_spreading="bilinear",
        event_id="NC_TEST",
        initial_model_name="manufactured",
        run_tag="test_pad2",
        grid_x=g.x,
        grid_z=g.z,
        dx=g.dx,
        dz=g.dz,
        dt=g.dt,
        nt=g.nt,
        vp=model.vp,
        vs=model.vs,
        rho=model.rho,
        t=run.t,
        das_data=das.data,
        das_channel_indices=das.channel_indices,
        das_raw_channels=raw[das.channel_indices],
        das_gauge_length_m=das.gauge_length_m,
        receiver_x=receivers.x,
        receiver_z=receivers.z,
        receiver_s=receivers.s,
        receiver_raw_channels=raw,
        source_x=source.x_embedded_m,
        source_z=source.z_embedded_m,
        source_theta_deg=35,
        source_scalar_moment=1e8,
        source_f0_hz=10,
        source_time_function_t0_s=source.stf.t0,
        material_padding_cells=2,
    )
    np.savez_compressed(path, **values)
    return values


def test_extension_preserves_entire_overlap_and_exact_edge_rule():
    m = material()
    e = extend_model(m, 3)
    assert e.grid.shape == (27, 26)
    assert e.grid.x0 == -450 and e.grid.z0 == 0
    assert e.grid.dt == m.grid.dt and e.grid.nt == m.grid.nt
    for key in ("vp", "vs", "rho"):
        a, b = getattr(m, key), getattr(e, key)
        np.testing.assert_array_equal(b[3:-3, :23], a)
        np.testing.assert_array_equal(b[:3, :23], np.repeat(a[:1], 3, axis=0))
        np.testing.assert_array_equal(b[-3:, :23], np.repeat(a[-1:], 3, axis=0))
        np.testing.assert_array_equal(b[3:-3, 23:], np.repeat(a[:, -1:], 3, axis=1))
        assert not np.shares_memory(a, b)
    np.testing.assert_array_equal(e.grid.z[:23], m.grid.z)
    # Expected full event dimensions/bounds, independently from array padding.
    assert (1241 + 2 * 120, 1121 + 120) == (1481, 1241)
    assert (-2100 - 120 * 5, 4100 + 120 * 5, 5600 + 120 * 5) == (-2700, 4700, 6200)


@pytest.mark.parametrize("cells", [True, False, 0, -1, 1.5, "120"])
def test_extension_rejects_nonpositive_or_noninteger_padding(cells):
    with pytest.raises(ValueError, match="positive integer"):
        extend_model(material(), cells)


def test_replay_and_frozen_source_geometry_then_complete_control(tmp_path):
    path = tmp_path / "baseline.npz"
    values = make_reference(path)
    original_bytes = path.read_bytes()
    data, model = load_reference(path)
    extended = extend_model(model, 3)
    source, receivers = rebuild_inputs(data, model.grid)
    other, other_receivers = rebuild_inputs(data, extended.grid)
    np.testing.assert_array_equal(source.stf.values, other.stf.values)
    assert source.x_embedded_m == other.x_embedded_m
    assert source.z_embedded_m == other.z_embedded_m
    assert other.ix == source.ix + 3
    assert other.iz == source.iz
    for key in ("x", "z", "s", "tx", "tz"):
        np.testing.assert_array_equal(
            getattr(receivers, key), getattr(other_receivers, key)
        )
    out = tmp_path / "control"
    report = run_control(path, out, extra_cells=3)
    assert report["reference_replay_relative_error"] < 1e-13
    assert report["padding_cells"] == {"reference": 2, "control": 5}
    assert report["material_overlap"] == "bitwise_equal"
    assert report["completed"]
    assert len(report["comparison"]["windows"]) == 3
    assert (out / "01_boundary_control_signed.png").stat().st_size > 10000
    assert (out / "summary.json").is_file()
    with np.load(out / "extended_das.npz") as result:
        assert result["vp"].shape == extended.grid.shape
        np.testing.assert_array_equal(result["vp"][3:-3, : model.grid.nz], values["vp"])
        np.testing.assert_array_equal(
            result["das_raw_channels"], values["das_raw_channels"]
        )
        np.testing.assert_array_equal(result["t"], values["t"])
    assert path.read_bytes() == original_bytes
    with pytest.raises(FileExistsError):
        run_control(path, out, extra_cells=3)
    with pytest.raises(ValueError, match="separate"):
        run_control(path, path.parent, extra_cells=3)


def test_tampered_waveforms_abort_before_boundary_interpretation(tmp_path):
    path = tmp_path / "baseline.npz"
    values = make_reference(path)
    values["das_data"] = values["das_data"] * 1.01
    np.savez_compressed(path, **values)
    out = tmp_path / "bad_control"
    with pytest.raises(ValueError, match="Reference replay differs"):
        run_control(path, out, extra_cells=3)
    assert not out.exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("solver_backend", "numba_fused"),
        ("boundary_top", "absorbing"),
        ("source_spreading", "nearest"),
        ("source_stf_sampling_offset_steps", 0.0),
    ],
)
def test_wrong_reference_conventions_fail(tmp_path, key, value):
    # Use a saved minimal fixture; these checks precede grid reconstruction.
    from scripts.validation.sbp_sat.event_boundary_control import REFERENCE_KEYS

    values = {k: np.array(0) for k in REFERENCE_KEYS}
    values.update(
        solver_backend="sbp_sat",
        source_mode="catalog_event",
        source_time_mode="ricker_moment",
        source_spreading="bilinear",
        source_stf_kind="ricker_physical_moment_rate",
        boundary_top="free",
        boundary_bottom="absorbing",
        boundary_left="absorbing",
        boundary_right="absorbing",
        source_stf_sampling_offset_steps=0.5,
    )
    values[key] = value
    path = tmp_path / "wrong.npz"
    np.savez(path, **values)
    with pytest.raises(ValueError):
        load_reference(path)
