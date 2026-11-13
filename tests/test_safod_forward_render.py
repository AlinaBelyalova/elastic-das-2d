"""Saved-array rendering: hide actual sponge strips, preserve native coordinates."""

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
        assert ax.get_title() == "SAFOD model"
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


@pytest.mark.parametrize("profile_keys", ["canonical", "legacy"])
def test_render_hides_saved_sponge_without_changing_grid_data_or_pixel_centres(
    tmp_path, monkeypatch, profile_keys
):
    root = tmp_path / "sponge"
    grid, values, snapshots = saved_case(root)
    if profile_keys == "canonical":
        values.update(sponge_rate_per_s=80., sponge_side_width_m=10.,
                      sponge_bottom_width_m=21.)
    else:
        values.update(gamma_s=80., sponge_width_x_m=10., sponge_width_z_m=21.)
    # Deliberately wrong historical display aliases and current configuration
    # must not define this saved run's window.
    values.update(scientific_x_min_m=-999., scientific_x_max_m=999.,
                  scientific_z_max_m=999.)
    monkeypatch.setattr(driver, "SBP_SPONGE_WIDTH_M", 10000.)
    raw = snapshots["snapshots_vz"]
    raw[:, :2, :] = 1e12
    raw[:, -2:, :] = 1e12
    raw[:, :, 11:] = 1e12  # the first such Vz pixel only touches z=87
    np.savez_compressed(root / "outputs_safod_initial_forward.npz", **values)
    np.savez_compressed(root / "snapshots_vz.npz", **snapshots)
    before = {p.name: p.read_bytes() for p in root.glob("*.npz")}
    figures = []
    close = plt.close
    def record(fig=None):
        if hasattr(fig, "axes"):
            figures.append(fig)
        close(fig)
    def forbidden(*args, **kwargs):
        pytest.fail("Display-only work attempted to rebuild/run the model")
    monkeypatch.setattr(driver.plt, "close", record)
    monkeypatch.setattr(driver, "run_forward_simulation", forbidden)
    monkeypatch.setattr(driver, "build_initial_model", forbidden)
    driver.render_saved_forward(root, backend="sbp_sat", theta_deg=35., initial_model="bill_logs")
    assert len(figures) == 2
    for fig in figures:
        ax = fig.axes[0]
        assert ax.get_xlim() == (-20., 20.)
        assert ax.get_ylim() == (87., 10.)
        assert not ax.patches
        assert "sponge" not in ax.get_title().lower()
        assert "boundary" not in ax.get_title().lower()
    model_image = figures[0].axes[0].images[0]
    gif_image = figures[1].axes[0].images[0]
    # Model pixels keep the FULL data-to-coordinate transform; the axes crop it.
    assert tuple(model_image.get_extent()) == (-32.5, 32.5, 111.5, 6.5)
    np.testing.assert_array_equal(model_image.get_array(), values["vp"].T)
    # GIF pixels are cropped, but stay at the SAME native physical centres.
    assert tuple(gif_image.get_extent()) == (-22.5, 22.5, 87., 10.)
    left, right, bottom, top = gif_image.get_extent()
    rows, columns = gif_image.get_array().shape
    np.testing.assert_array_equal(left+(np.arange(columns)+.5)*(right-left)/columns,
                                  grid.x[2:-2])
    np.testing.assert_array_equal(top+(np.arange(rows)+.5)*(bottom-top)/rows,
                                  snapshots["snapshot_grid_z"][:11])
    assert max(abs(v) for v in gif_image.get_clim()) < 2
    assert before == {p.name: p.read_bytes() for p in root.glob("*.npz")}
    with Image.open(root / "05_wavefield_vz_moment_tensor.gif") as gif:
        assert gif.n_frames == 3


def test_display_bounds_keep_top_and_use_physical_widths():
    grid = Grid2D(1241, 1121, 5., 5., 2, .0001, x0=-2100.)
    assert driver.forward_display_bounds(grid, sponge_side_width_m=600.,
                                        sponge_bottom_width_m=600.) == (-1500.,3500.,0.,5000.)
    assert driver.forward_display_bounds(grid) == (-2100.,4100.,0.,5600.)
    assert driver.saved_sponge_widths({"sponge_rate_per_s":0.,
                                      "sponge_side_width_m":600.,
                                      "sponge_bottom_width_m":600.}) == (0.,0.)


@pytest.mark.parametrize("side,bottom", [(-1.,0.), (float("nan"),0.), (30.,0.), (0.,98.)])
def test_invalid_display_strip_widths_fail_instead_of_resizing_the_grid(side,bottom):
    grid = Grid2D(13,15,5.,7.,2,.001,x0=-30.,z0=10.)
    with pytest.raises(ValueError):
        driver.forward_display_bounds(grid, sponge_side_width_m=side,
                                      sponge_bottom_width_m=bottom)


def test_active_saved_absorber_needs_its_own_widths():
    with pytest.raises(ValueError,match="lacks widths"):
        driver.saved_sponge_widths({"sponge_rate_per_s":80.})
