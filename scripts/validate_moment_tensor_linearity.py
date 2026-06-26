from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from src.grid import Grid2D
from src.model import ElasticModel2D
from src.source import MomentTensor2D, build_source_2d
from src.receivers import build_das_cable
from src.simulator import run_forward_simulation
from src.solver_numpy import max_stable_dt


def relative_l2(reference: np.ndarray, test: np.ndarray, eps: float = 1.0e-30) -> float:
    return float(np.linalg.norm(test - reference) / (np.linalg.norm(reference) + eps))


def max_abs_error(reference: np.ndarray, test: np.ndarray) -> float:
    return float(np.max(np.abs(test - reference)))


def build_model(
    *,
    nx: int = 301,
    nz: int = 251,
    nt: int = 1000,
    dx: float = 10.0,
    dz: float = 10.0,
    vp: float = 3000.0,
    vs: float = 1700.0,
    rho: float = 2500.0,
    half_order: int = 2,
    cfl_safety: float = 0.85,
) -> ElasticModel2D:
    dt = max_stable_dt(
        vp,
        dx,
        dz,
        half_order,
        safety=cfl_safety,
        use_ts_sfd=False,
    )

    grid = Grid2D(
        nx=nx,
        nz=nz,
        dx=dx,
        dz=dz,
        nt=nt,
        dt=dt,
        x0=0.0,
        z0=0.0,
    )

    return ElasticModel2D(
        grid=grid,
        vp=np.full(grid.shape, vp, dtype=np.float64),
        vs=np.full(grid.shape, vs, dtype=np.float64),
        rho=np.full(grid.shape, rho, dtype=np.float64),
    )


def build_geometry(model: ElasticModel2D, *, n_boundary: int):
    grid = model.grid

    x_src = grid.x[grid.nx // 3]
    z_src = grid.z[grid.nz // 2]

    ix_cable = grid.nx - n_boundary - 25
    iz_top = n_boundary + 10
    iz_bot = grid.nz - n_boundary - 10

    receivers = build_das_cable(
        grid=grid,
        waypoints_x=[grid.x[ix_cable], grid.x[ix_cable]],
        waypoints_z=[grid.z[iz_top], grid.z[iz_bot]],
        channel_spacing_m=10.0,
        n_pml=n_boundary,
    )

    return x_src, z_src, receivers


def make_source(
    model: ElasticModel2D,
    *,
    x_src: float,
    z_src: float,
    mt: MomentTensor2D,
    scalar_moment: float,
    label: str,
):
    grid = model.grid

    return build_source_2d(
        grid=grid,
        x_m=x_src,
        z_m=z_src,
        mt2d=mt,
        scalar_moment=scalar_moment,
        nt=grid.nt,
        dt=grid.dt,
        f0_hz=8.0,
        derivative_order=0,
        label=label,
    )


def run_source(
    *,
    model: ElasticModel2D,
    source,
    receivers,
    backend: str,
    free_surface: bool,
    half_order: int,
    n_boundary: int,
):
    run_result, das_result = run_forward_simulation(
        model=model,
        source=source,
        receivers=receivers,
        gauge_length_m=20.0,
        half_order=half_order,
        use_ts_sfd=False,
        n_boundary=n_boundary,
        gamma_s=50.0,
        snapshot_stride=None,
        backend=backend,
        free_surface=free_surface,
    )
    return run_result, das_result


def validate_once(
    *,
    backend: str = "numba_fused",
    free_surface: bool,
    outdir: Path,
) -> dict:
    half_order = 2
    n_boundary = 50
    m0 = 1.0e10

    model = build_model(half_order=half_order)
    x_src, z_src, receivers = build_geometry(model, n_boundary=n_boundary)

    # Unit component sources, each scaled by m0.
    src_xx = make_source(
        model,
        x_src=x_src,
        z_src=z_src,
        mt=MomentTensor2D(Mxx=m0, Mzz=0.0, Mxz=0.0),
        scalar_moment=m0,
        label="Mxx component",
    )
    src_zz = make_source(
        model,
        x_src=x_src,
        z_src=z_src,
        mt=MomentTensor2D(Mxx=0.0, Mzz=m0, Mxz=0.0),
        scalar_moment=m0,
        label="Mzz component",
    )
    src_xz = make_source(
        model,
        x_src=x_src,
        z_src=z_src,
        mt=MomentTensor2D(Mxx=0.0, Mzz=0.0, Mxz=m0),
        scalar_moment=m0,
        label="Mxz component",
    )

    # Arbitrary linear combination.
    a_xx = 0.8
    a_zz = -0.35
    a_xz = 0.6

    src_combined = make_source(
        model,
        x_src=x_src,
        z_src=z_src,
        mt=MomentTensor2D(
            Mxx=a_xx * m0,
            Mzz=a_zz * m0,
            Mxz=a_xz * m0,
        ),
        scalar_moment=m0,
        label="combined moment tensor",
    )

    print(f"\nRunning moment-tensor linearity validation, free_surface={free_surface}")
    print(f"Backend: {backend}")
    print(f"Source: x={x_src:.1f} m, z={z_src:.1f} m")
    print(f"Receivers: {receivers.nrec}")

    run_xx, das_xx = run_source(
        model=model,
        source=src_xx,
        receivers=receivers,
        backend=backend,
        free_surface=free_surface,
        half_order=half_order,
        n_boundary=n_boundary,
    )
    run_zz, das_zz = run_source(
        model=model,
        source=src_zz,
        receivers=receivers,
        backend=backend,
        free_surface=free_surface,
        half_order=half_order,
        n_boundary=n_boundary,
    )
    run_xz, das_xz = run_source(
        model=model,
        source=src_xz,
        receivers=receivers,
        backend=backend,
        free_surface=free_surface,
        half_order=half_order,
        n_boundary=n_boundary,
    )
    run_c, das_c = run_source(
        model=model,
        source=src_combined,
        receivers=receivers,
        backend=backend,
        free_surface=free_surface,
        half_order=half_order,
        n_boundary=n_boundary,
    )

    pred_vx = (
        a_xx * run_xx.receiver_vx
        + a_zz * run_zz.receiver_vx
        + a_xz * run_xz.receiver_vx
    )
    pred_vz = (
        a_xx * run_xx.receiver_vz
        + a_zz * run_zz.receiver_vz
        + a_xz * run_xz.receiver_vz
    )
    pred_das = (
        a_xx * das_xx.data
        + a_zz * das_zz.data
        + a_xz * das_xz.data
    )

    metrics = {
        "backend": backend,
        "free_surface": bool(free_surface),
        "nx": model.grid.nx,
        "nz": model.grid.nz,
        "nt": model.grid.nt,
        "dt": model.grid.dt,
        "n_receivers": receivers.nrec,
        "source_x_m": x_src,
        "source_z_m": z_src,
        "coefficients": {
            "a_xx": a_xx,
            "a_zz": a_zz,
            "a_xz": a_xz,
        },
        "receiver_vx_rel_l2": relative_l2(run_c.receiver_vx, pred_vx),
        "receiver_vz_rel_l2": relative_l2(run_c.receiver_vz, pred_vz),
        "das_rel_l2": relative_l2(das_c.data, pred_das),
        "receiver_vx_max_abs": max_abs_error(run_c.receiver_vx, pred_vx),
        "receiver_vz_max_abs": max_abs_error(run_c.receiver_vz, pred_vz),
        "das_max_abs": max_abs_error(das_c.data, pred_das),
    }

    print("\nLinearity metrics:")
    for key in [
        "receiver_vx_rel_l2",
        "receiver_vz_rel_l2",
        "das_rel_l2",
        "receiver_vx_max_abs",
        "receiver_vz_max_abs",
        "das_max_abs",
    ]:
        print(f"  {key:22s}: {metrics[key]:.6e}")

    tag = "free_surface_true" if free_surface else "free_surface_false"
    np.savez_compressed(
        outdir / f"moment_tensor_linearity_{tag}.npz",
        t_v=run_c.t_v,
        receiver_vx_combined=run_c.receiver_vx,
        receiver_vx_pred=pred_vx,
        receiver_vz_combined=run_c.receiver_vz,
        receiver_vz_pred=pred_vz,
        das_combined=das_c.data,
        das_pred=pred_das,
    )

    return metrics


def main() -> None:
    outdir = Path("results/validation_moment_tensor")
    outdir.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    for free_surface in [False, True]:
        metrics = validate_once(
            backend="numba_fused",
            free_surface=free_surface,
            outdir=outdir,
        )
        all_metrics.append(metrics)

    out_json = outdir / "moment_tensor_linearity_metrics.json"
    with out_json.open("w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nSaved metrics to {out_json}")

    # Conservative pass/fail threshold.
    threshold = 1.0e-8
    worst = max(
        max(m["receiver_vx_rel_l2"], m["receiver_vz_rel_l2"], m["das_rel_l2"])
        for m in all_metrics
    )

    print(f"\nWorst relative L2 error: {worst:.6e}")

    if worst > threshold:
        raise RuntimeError(
            f"Moment-tensor linearity check failed: worst rel_L2={worst:.3e} > {threshold:.1e}"
        )

    print("Moment-tensor linearity check PASSED.")


if __name__ == "__main__":
    main()