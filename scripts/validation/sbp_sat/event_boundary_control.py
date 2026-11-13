"""Move only the artificial sides/bottom of an existing SBP event model.

The complete saved material volume is frozen; extra cells repeat its edge
values. dt, nt, physical source, registered receivers, STF and DAS are fixed.
Replay of the saved reference is mandatory before interpreting differences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from numbers import Integral
from pathlib import Path

import numpy as np

from src.common.grid import Grid2D
from src.common.model import ElasticModel2D
from src.common.signal_processing import bandpass_traces
from src.das.receivers import build_receivers_from_channel_centres
from src.elastic.source import build_dc_source
from src.sbp_sat.simulator import estimate_storage, run_forward_simulation

REFERENCE_KEYS = (
    "solver_backend",
    "boundary_top",
    "boundary_bottom",
    "boundary_left",
    "boundary_right",
    "source_mode",
    "source_time_mode",
    "source_spreading",
    "event_id",
    "initial_model_name",
    "run_tag",
    "grid_x",
    "grid_z",
    "dx",
    "dz",
    "dt",
    "nt",
    "vp",
    "vs",
    "rho",
    "t",
    "das_data",
    "das_channel_indices",
    "das_raw_channels",
    "das_gauge_length_m",
    "receiver_x",
    "receiver_z",
    "receiver_s",
    "receiver_raw_channels",
    "source_x",
    "source_z",
    "source_theta_deg",
    "source_scalar_moment",
    "source_f0_hz",
    "source_time_function_t0_s",
    "material_padding_cells",
    "source_stf_kind",
    "source_stf_sampling_offset_steps",
)


def relative_error(value, reference):
    norm = float(np.linalg.norm(reference))
    error = float(np.linalg.norm(value - reference))
    return error / norm if norm else (0.0 if error == 0 else float("inf"))


def load_reference(path):
    with np.load(path, allow_pickle=False) as package:
        if "sponge_rate_per_s" in package.files and float(package["sponge_rate_per_s"]) != 0:
            raise ValueError("This historical impedance-only control cannot replay a sponge run. "
                             "Use the sponge validation; no automatic grid extension.")
        missing = set(REFERENCE_KEYS) - set(package.files)
        if missing:
            raise ValueError(f"Reference lacks required provenance: {sorted(missing)}")
        data = {key: package[key] for key in REFERENCE_KEYS}
    for key, expected in {
        "solver_backend": "sbp_sat",
        "source_mode": "catalog_event",
        "source_time_mode": "ricker_moment",
        "source_spreading": "bilinear",
        "source_stf_kind": "ricker_physical_moment_rate",
        "boundary_top": "free",
        "boundary_bottom": "absorbing",
        "boundary_left": "absorbing",
        "boundary_right": "absorbing",
    }.items():
        if data[key].item() != expected:
            raise ValueError(f"Unsupported reference {key}: {data[key].item()!r}")
    if data["source_stf_sampling_offset_steps"].item() != 0.5:
        raise ValueError("Reference STF is not sampled at the stress midpoint.")
    x, z = data["grid_x"], data["grid_z"]
    grid = Grid2D(
        len(x),
        len(z),
        float(data["dx"]),
        float(data["dz"]),
        int(data["nt"]),
        float(data["dt"]),
        float(x[0]),
        float(z[0]),
    )
    np.testing.assert_allclose(grid.x, x, rtol=0, atol=1e-9)
    np.testing.assert_allclose(grid.z, z, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(data["t"], (np.arange(grid.nt) + 0.5) * grid.dt)
    if not np.all(np.isfinite(data["das_data"])):
        raise ValueError("Reference DAS contains NaN/Inf.")
    return data, ElasticModel2D(grid, data["vp"], data["vs"], data["rho"])


def extend_model(model, extra_cells):
    """Exact overlap, constant edge extension; never pad the free top."""
    if (
        isinstance(extra_cells, (bool, np.bool_))
        or not isinstance(extra_cells, Integral)
        or extra_cells < 1
    ):
        raise ValueError("extra_cells must be a positive integer.")
    e = int(extra_cells)
    g = model.grid
    grid = replace(g, nx=g.nx + 2 * e, nz=g.nz + e, x0=g.x0 - e * g.dx)
    arrays = [
        np.pad(getattr(model, key), ((e, e), (0, e)), mode="edge")
        for key in ("vp", "vs", "rho")
    ]
    extended = ElasticModel2D(grid, *arrays)
    for key in ("vp", "vs", "rho"):
        np.testing.assert_array_equal(
            getattr(extended, key)[e : e + g.nx, : g.nz], getattr(model, key)
        )
    np.testing.assert_allclose(grid.x[e : e + g.nx], g.x, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(grid.z[: g.nz], g.z)
    return extended


def rebuild_inputs(data, grid):
    source = build_dc_source(
        grid=grid,
        x_m=float(data["source_x"]),
        z_m=float(data["source_z"]),
        theta_deg=float(data["source_theta_deg"]),
        scalar_moment=float(data["source_scalar_moment"]),
        f0_hz=float(data["source_f0_hz"]),
        nt=grid.nt,
        dt=grid.dt,
        spreading="bilinear",
        source_time_mode="ricker_moment",
    )
    if source.stf.t0 != float(data["source_time_function_t0_s"]):
        raise ValueError("Reconstructed STF t0 differs from saved reference.")
    receivers = build_receivers_from_channel_centres(
        data["receiver_x"],
        data["receiver_z"],
        data["receiver_s"],
        grid=grid,
    )
    return source, receivers


def check_observable(data, run, das):
    np.testing.assert_array_equal(run.t, data["t"])
    np.testing.assert_array_equal(das.channel_indices, data["das_channel_indices"])
    np.testing.assert_array_equal(
        data["receiver_raw_channels"][das.channel_indices], data["das_raw_channels"]
    )
    if das.data.shape != data["das_data"].shape or not np.all(np.isfinite(das.data)):
        raise ValueError("Control DAS shape/finite-value check failed.")


def plot_comparison(data, control, output_dir, control_padding_cells):
    # Use a shared scale from the reference, never independently normalize the
    # extended result or the difference. Filter complete records identically.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.safod.settings import (
        COMMON_FMAX_HZ,
        COMMON_FMIN_HZ,
        FILTER_ORDER,
        FILTER_TAPER_FRAC,
    )

    dt = float(data["dt"])
    time = data["t"] - float(data["source_time_function_t0_s"])
    mask = (time >= 0) & (time <= 2)
    if np.count_nonzero(mask) < 8:
        raise ValueError("Too few samples in the origin-relative 0..2 s view.")
    options = {
        "fs_hz": 1 / dt,
        "fmin_hz": COMMON_FMIN_HZ,
        "fmax_hz": COMMON_FMAX_HZ,
        "order": FILTER_ORDER,
        "taper_frac": FILTER_TAPER_FRAC,
    }
    old = bandpass_traces(data["das_data"], **options)[:, mask]
    new = bandpass_traces(control, **options)[:, mask]
    scale = np.percentile(np.abs(old), 99, axis=1, keepdims=True)
    scale = np.maximum(scale, 1e-12)
    old_padding = int(data["material_padding_cells"])
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharex=True, sharey=True)
    channels = data["das_raw_channels"]
    shown_time = time[mask]
    # Keep arrays saved at full resolution; only raster rendering is thinned.
    stride = max(1, int(np.ceil(len(shown_time) / 2500)))
    extent = (channels[0], channels[-1], shown_time[-1], shown_time[0])
    for ax, values, title in zip(
        axes,
        (old, new, new - old),
        (
            f"Reference: pad {old_padding} cells",
            f"Control: pad {control_padding_cells} cells",
            "Extended - reference",
        ),
    ):
        image = ax.imshow(
            (values / scale)[:, ::stride].T,
            aspect="auto",
            extent=extent,
            cmap="seismic",
            vmin=-1,
            vmax=1,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel("Physical reference channel")
    axes[0].set_ylabel("Time from catalog origin [s]")
    fig.colorbar(
        image,
        ax=axes,
        label="Amplitude / reference trace p99 (clipped at +/-1)",
        shrink=0.9,
    )
    fig.suptitle(
        f"{data['event_id'].item()}: SBP-SAT boundary sensitivity; {COMMON_FMIN_HZ:g}-{COMMON_FMAX_HZ:g} Hz"
    )
    fig.savefig(
        output_dir / "01_boundary_control_signed.png", dpi=160, bbox_inches="tight"
    )
    plt.close(fig)
    metrics = []
    for start, end in ((0, 0.8), (0.8, 1.2), (1.2, 2.0)):
        raw_mask = (time >= start) & (time < end)
        filt_mask = (shown_time >= start) & (shown_time < end)
        metrics.append(
            {
                "origin_time_window_s": [start, end],
                "raw_relative_difference": relative_error(
                    control[:, raw_mask], data["das_data"][:, raw_mask]
                ),
                "filtered_relative_difference": relative_error(
                    new[:, filt_mask], old[:, filt_mask]
                ),
            }
        )
    return {
        "filter": options,
        "normalization": "one reference p99 per trace, shared across all panels",
        "difference_sign": "extended_minus_reference",
        "windows": metrics,
    }


def run_control(reference_path, output_dir, *, extra_cells=120, overwrite=False):
    reference_path, output_dir = Path(reference_path), Path(output_dir)
    if output_dir.resolve() == reference_path.parent.resolve():
        raise ValueError(
            "Use a separate control directory; keep the forward reference intact."
        )
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Control directory is not empty: {output_dir}. Use --overwrite intentionally."
        )
    data, original = load_reference(reference_path)
    extended = extend_model(original, extra_cells)
    source, receivers = rebuild_inputs(data, original.grid)
    control_source, control_receivers = rebuild_inputs(data, extended.grid)
    np.testing.assert_array_equal(source.stf.values, control_source.stf.values)
    for key in ("x", "z", "s", "tx", "tz"):
        np.testing.assert_array_equal(
            getattr(receivers, key), getattr(control_receivers, key)
        )
    for key in ("Mxx", "Mzz", "Mxz"):
        assert getattr(source.m2d, key) == getattr(control_source.m2d, key)
    g, h = original.grid, extended.grid
    new_padding = int(data["material_padding_cells"]) + extra_cells
    print(
        f"Reference grid: {g.nx}x{g.nz}; x=[{g.x[0]}, {g.x[-1]}], z=[{g.z[0]}, {g.z[-1]}] m",
        flush=True,
    )
    print(
        f"Control grid:   {h.nx}x{h.nz}; x=[{h.x[0]}, {h.x[-1]}], z=[{h.z[0]}, {h.z[-1]}] m",
        flush=True,
    )
    print(f"dx={g.dx}, dz={g.dz} m; dt={g.dt:.17g} s; nt={g.nt}; all fixed", flush=True)
    print(
        "Material overlap is exact; exterior uses constant edge extension. Free top is unchanged.",
        flush=True,
    )
    storage = estimate_storage(h, receivers.nrec)
    print(
        f"Forward storage estimate: {storage['forward_estimated_array_gib']:.2f} GiB (no gradient history)",
        flush=True,
    )
    gauge = float(data["das_gauge_length_m"])
    # Mandatory full reference replay catches reconstruction or convention
    # mismatch before any domain-sensitivity interpretation is made.
    print("Replaying saved reference with Numba...", flush=True)
    run, das = run_forward_simulation(original, source, receivers, gauge)
    check_observable(data, run, das)
    replay_error = relative_error(das.data, data["das_data"])
    if replay_error > 1e-11:
        raise ValueError(
            f"Reference replay differs by {replay_error:.3e}; stop before interpreting boundaries."
        )
    print(f"Reference replay relative error: {replay_error:.3e}", flush=True)
    del run, das
    print("Running extended model with exactly the saved dt/nt...", flush=True)
    run, das = run_forward_simulation(
        extended, control_source, control_receivers, gauge
    )
    check_observable(data, run, das)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save the actual arrays and coordinates, not a stale copy of the original
    # package's grid-dependent metadata. No absolute 3-D amplitude claim.
    np.savez_compressed(
        output_dir / "extended_das.npz",
        das_data=das.data,
        t=run.t,
        das_raw_channels=data["das_raw_channels"],
        grid_x=h.x,
        grid_z=h.z,
        vp=extended.vp,
        vs=extended.vs,
        rho=extended.rho,
        dt=h.dt,
        nt=h.nt,
        dx=h.dx,
        dz=h.dz,
        source_stf_values=control_source.stf.values,
        receiver_x=receivers.x,
        receiver_z=receivers.z,
        receiver_s=receivers.s,
        receiver_tx=receivers.tx,
        receiver_tz=receivers.tz,
        material_padding_cells=new_padding,
        reference_package=str(reference_path),
        **run.solver_metadata,
    )
    digest = hashlib.sha256()
    with reference_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    report = {
        "completed": True,
        "reference_package": str(reference_path.resolve()),
        "reference_sha256": digest.hexdigest(),
        "reference_replay_relative_error": replay_error,
        "material_overlap": "bitwise_equal",
        "extension": "constant_edge_values",
        "source_receivers_stf": "unchanged; full reference replay checked",
        "padding_cells": {
            "reference": int(data["material_padding_cells"]),
            "control": new_padding,
        },
        "grid_shape": {"reference": [g.nx, g.nz], "control": [h.nx, h.nz]},
        "bounds_m": {
            "reference": [
                [float(g.x[0]), float(g.x[-1])],
                [float(g.z[0]), float(g.z[-1])],
            ],
            "control": [
                [float(h.x[0]), float(h.x[-1])],
                [float(h.z[0]), float(h.z[-1])],
            ],
        },
        "dt_s": g.dt,
        "nt": g.nt,
        "extra_cells": int(extra_cells),
        "extra_width_m": {"x": extra_cells * g.dx, "bottom": extra_cells * g.dz},
        "propagation_elapsed_s": run.solver_metadata["propagation_elapsed_s"],
        "storage": storage,
        "limitations": [
            "Domain sensitivity, not proof that every changed branch is a pure boundary reflection.",
            "Unchanged branches may include free-surface reflections; the top is not moved.",
            "The added region is an explicit constant continuation, not new geological information.",
        ],
    }
    # Free receiver histories before filtering/plotting the large DAS arrays.
    del run
    report["comparison"] = plot_comparison(data, das.data, output_dir, new_padding)
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    print(f"Boundary control completed: {output_dir}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-package", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--extra-padding-cells", type=int, default=120)
    parser.add_argument("--theta-deg", type=float, default=35)
    parser.add_argument(
        "--initial-model", default="hybrid_zhang2009_boness2006_bill_logs"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    from scripts.safod.settings import comparison_dir_for_theta, forward_dir_for_theta

    reference = args.reference_package or (
        forward_dir_for_theta(args.theta_deg, "sbp_sat")
        / args.initial_model
        / "outputs_safod_initial_forward.npz"
    )
    output = args.output_dir or (
        comparison_dir_for_theta(args.theta_deg, "sbp_sat")
        / args.initial_model
        / f"boundary_control_extra{args.extra_padding_cells}"
    )
    run_control(
        reference,
        output,
        extra_cells=args.extra_padding_cells,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
