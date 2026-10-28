"""Saved-array rendering: full domain, staggered coordinates, no propagation."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from scripts.safod import run_forward as driver
from src.common.grid import Grid2D


def saved_case(root):
    root.mkdir(parents=True)
    grid = Grid2D(13, 15, 5, 7, 11, 0.001, x0=-30, z0=10)
    native_z = grid.z[:-1] + grid.dz / 2
    x, z = np.meshgrid(grid.x, native_z, indexing="ij")
    snapshots = np.stack([np.sin((x + i * 5) / 15) * np.cos(z / 30) for i in range(3)])
    values = {
        "solver_backend": "sbp_sat",
        "initial_model_name": "bill_logs",
        "source_theta_deg": 35,
        "grid_x": grid.x,
        "grid_z": grid.z,
        "dx": grid.dx,
        "dz": grid.dz,
        "dt": grid.dt,
        "nt": grid.nt,
        "vp": np.full(grid.shape, 3200.0),
        "vs": np.full(grid.shape, 1700.0),
        "rho": np.full(grid.shape, 2200.0),
        "source_x": 3.7,
        "source_z": 31.1,
        "receiver_x": np.array([-5.0, 5.0]),
        "receiver_z": np.array([20.0, 70.0]),
        "model_type": "initial_bill_logs",
        "scientific_x_min_m": -20.0,
        "scientific_x_max_m": 20.0,
        "scientific_z_max_m": 80.0,
        "free_surface": True,
        "das_data": np.arange(22.0).reshape(2, 11),
    }
    snap = {
        "snapshots_vz": snapshots,
        "snapshot_times_v": (np.array([0, 3, 6]) + 0.5) * grid.dt,
        "snapshot_grid_x": grid.x,
        "snapshot_grid_z": native_z,
    }
    np.savez_compressed(root / "outputs_safod_initial_forward.npz", **values)
    np.savez_compressed(root / "snapshots_vz.npz", **snap)
    return grid, values, snap


def test_render_only_preserves_arrays_and_full_domain_native_alignment(
    tmp_path, monkeypatch
):
    root = tmp_path / "bill_logs"
    grid, values, saved = saved_case(root)
    original_npz = (root / "outputs_safod_initial_forward.npz").read_bytes()
    original_snapshots = (root / "snapshots_vz.npz").read_bytes()
    figures = []
    close = plt.close

    def record_close(fig=None):
        if hasattr(fig, "axes"):
            figures.append(fig)
        close(fig)

    def forbidden(*args, **kwargs):
        pytest.fail("Rendering attempted propagation or external-data/model rebuilding")

    monkeypatch.setattr(driver.plt, "close", record_close)
    monkeypatch.setattr(driver, "run_forward_simulation", forbidden)
    monkeypatch.setattr(driver, "build_initial_model", forbidden)
    monkeypatch.setattr(driver, "load_real_event_package", forbidden)
    monkeypatch.setattr(driver, "forward_dir_for_theta", lambda *a: tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_forward",
            "--backend",
            "sbp_sat",
            "--initial-model",
            "bill_logs",
            "--render-only",
        ],
    )
    driver.main()
    assert len(figures) == 2
    for fig in figures:
        ax = fig.axes[0]
        assert ax.get_xlim() == (grid.x[0], grid.x[-1])
        assert ax.get_ylim() == (grid.z[-1], grid.z[0])
        assert len(ax.patches) == 0
        assert "Full computational domain" in ax.get_title()
    model_image = figures[0].axes[0].images[0]
    gif_image = figures[1].axes[0].images[0]
    np.testing.assert_array_equal(model_image.get_array(), values["vp"].T)
    assert tuple(model_image.get_extent()) == (-32.5, 32.5, 111.5, 6.5)
    assert tuple(gif_image.get_extent()) == (-32.5, 32.5, 108.0, 10.0)
    # Every pixel centre is exactly its native sample coordinate. No interpolation
    # or silent half-cell displacement at the free surface / bottom is allowed.
    left, right, bottom, top = gif_image.get_extent()
    np.testing.assert_array_equal(
        left + (np.arange(grid.nx) + 0.5) * (right - left) / grid.nx, grid.x
    )
    np.testing.assert_array_equal(
        top + (np.arange(grid.nz - 1) + 0.5) * (bottom - top) / (grid.nz - 1),
        saved["snapshot_grid_z"],
    )
    with Image.open(root / "05_wavefield_vz_moment_tensor.gif") as im:
        assert im.n_frames == 3
        for i in range(im.n_frames):
            im.seek(i)
            im.load()
    assert (root / "01_model_vp_with_source.png").is_file()
    assert (root / "outputs_safod_initial_forward.npz").read_bytes() == original_npz
    assert (root / "snapshots_vz.npz").read_bytes() == original_snapshots


@pytest.mark.parametrize(
    "fault", ["backend", "model", "angle", "z_axis", "shape", "time"]
)
def test_saved_identity_and_native_axes_checked_before_any_render(
    tmp_path, monkeypatch, fault
):
    root = tmp_path / "bill_logs"
    _, values, saved = saved_case(root)
    if fault in ("backend", "model", "angle"):
        key, value = {
            "backend": ("solver_backend", "numba_fused"),
            "model": ("initial_model_name", "wrong"),
            "angle": ("source_theta_deg", 45),
        }[fault]
        values[key] = value
        np.savez_compressed(root / "outputs_safod_initial_forward.npz", **values)
    else:
        if fault == "z_axis":
            saved["snapshot_grid_z"] = saved["snapshot_grid_z"] + 3.5
        elif fault == "shape":
            saved["snapshots_vz"] = saved["snapshots_vz"][:, :, :-1]
        else:
            saved["snapshot_times_v"] = saved["snapshot_times_v"] + 0.00025
        np.savez_compressed(root / "snapshots_vz.npz", **saved)
    monkeypatch.setattr(
        driver,
        "save_forward_domain_figures",
        lambda *a, **kw: pytest.fail("Invalid package was rendered"),
    )
    with pytest.raises((ValueError, AssertionError)):
        driver.render_saved_forward(
            root, backend="sbp_sat", theta_deg=35, initial_model="bill_logs"
        )


def test_missing_snapshots_gives_rerun_instruction(tmp_path):
    with pytest.raises(FileNotFoundError, match="--save-gif"):
        driver.render_saved_forward(
            tmp_path, backend="sbp_sat", theta_deg=35, initial_model="bill_logs"
        )
