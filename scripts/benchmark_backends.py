from __future__ import annotations

from time import perf_counter
import numpy as np

from src.grid import Grid2D
from src.model import ElasticModel2D
from src.source import build_dc_source
from src.receivers import build_das_cable
from src.simulator import run_forward_simulation
from src.solver_numpy import max_stable_dt


def build_homogeneous_model(
    *,
    nx: int,
    nz: int,
    nt: int,
    dx: float = 10.0,
    dz: float = 10.0,
    vp: float = 3000.0,
    vs: float = 1700.0,
    rho: float = 2500.0,
    cfl_safety: float = 0.90,
    half_order: int = 2,
    use_ts_sfd: bool = False,
) -> ElasticModel2D:
    dt = cfl_safety * max_stable_dt(
        vp, dx, dz, half_order, use_ts_sfd=use_ts_sfd
    )
    grid = Grid2D(nx=nx, nz=nz, dx=dx, dz=dz, nt=nt, dt=dt, x0=0.0, z0=0.0)
    return ElasticModel2D(
        grid=grid,
        vp=np.full(grid.shape, vp, dtype=np.float64),
        vs=np.full(grid.shape, vs, dtype=np.float64),
        rho=np.full(grid.shape, rho, dtype=np.float64),
    )


def build_geometry(
    model: ElasticModel2D,
    *,
    n_pml: int = 50,
    free_surface: bool = False,
):
    grid = model.grid

    x_src = grid.x[grid.nx // 3]
    z_src = grid.z[n_pml + 20] if free_surface else grid.z[grid.nz // 2]

    source = build_dc_source(
        grid=grid,
        x_m=x_src,
        z_m=z_src,
        theta_deg=0.0,
        scalar_moment=1.0e10,
        nt=grid.nt,
        dt=grid.dt,
        f0_hz=8.0,
        derivative_order=0,
    )

    ix_cable = grid.nx - n_pml - 25
    iz_top = n_pml + 10
    iz_bot = grid.nz - n_pml - 10

    receivers = build_das_cable(
        grid=grid,
        waypoints_x=[grid.x[ix_cable], grid.x[ix_cable]],
        waypoints_z=[grid.z[iz_top], grid.z[iz_bot]],
        channel_spacing_m=10.0,
        n_pml=n_pml,
    )
    return source, receivers


def relative_l2(a: np.ndarray, b: np.ndarray, eps: float = 1e-30) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + eps))


def run_timed(
    *,
    backend: str,
    model: ElasticModel2D,
    source,
    receivers,
    half_order: int,
    use_ts_sfd: bool,
    n_boundary: int,
    free_surface: bool,
    n_runs: int = 3,
):
    result = None
    times = []

    for _ in range(n_runs):
        t0 = perf_counter()
        run_result, das_result = run_forward_simulation(
            model=model,
            source=source,
            receivers=receivers,
            gauge_length_m=20.0,
            half_order=half_order,
            use_ts_sfd=use_ts_sfd,
            n_boundary=n_boundary,
            gamma_s=50.0,
            snapshot_stride=None,
            backend=backend,
            free_surface=free_surface,
        )
        times.append(perf_counter() - t0)
        result = (run_result, das_result)

    return *result, min(times)


def print_correctness_against_numpy(
    *,
    label: str,
    run_np,
    das_np,
    run_test,
    das_test,
) -> None:
    print(f"\n--- Correctness vs NumPy: {label} ---")
    print("t_v   match:", np.allclose(run_np.t_v, run_test.t_v))
    print("t_sig match:", np.allclose(run_np.t_sigma, run_test.t_sigma))

    for name, a, b in [
        ("receiver_vx", run_np.receiver_vx, run_test.receiver_vx),
        ("receiver_vz", run_np.receiver_vz, run_test.receiver_vz),
        ("DAS data   ", das_np.data, das_test.data),
    ]:
        print(
            f"{name}: max_abs={np.max(np.abs(a - b)):.3e}  "
            f"rel_L2={relative_l2(a, b):.3e}  "
            f"allclose={np.allclose(a, b, atol=1e-7, rtol=1e-5)}"
        )


def benchmark_case(
    *,
    free_surface: bool,
    nx: int,
    nz: int,
    nt: int,
    half_order: int,
    use_ts_sfd: bool,
    n_boundary: int,
) -> None:
    label = "free_surface=True" if free_surface else "free_surface=False"

    print(f"\n{'=' * 80}")
    print(f"Benchmark case: {label}")
    print(f"{'=' * 80}")

    model = build_homogeneous_model(
        nx=nx,
        nz=nz,
        nt=nt,
        half_order=half_order,
        use_ts_sfd=use_ts_sfd,
    )
    source, receivers = build_geometry(
        model,
        n_pml=n_boundary,
        free_surface=free_surface,
    )

    print("Running NumPy backend (3 runs)...")
    run_np, das_np, t_np = run_timed(
        backend="numpy",
        model=model,
        source=source,
        receivers=receivers,
        half_order=half_order,
        use_ts_sfd=use_ts_sfd,
        n_boundary=n_boundary,
        free_surface=free_surface,
    )

    print("Running Numba fused backend (3 runs)...")
    run_fused, das_fused, t_fused = run_timed(
        backend="numba_fused",
        model=model,
        source=source,
        receivers=receivers,
        half_order=half_order,
        use_ts_sfd=use_ts_sfd,
        n_boundary=n_boundary,
        free_surface=free_surface,
    )

    print_correctness_against_numpy(
        label="Numba fused",
        run_np=run_np,
        das_np=das_np,
        run_test=run_fused,
        das_test=das_fused,
    )

    print("\n--- Runtime (best of 3) ---")
    print(f"NumPy       : {t_np:.3f} s")
    print(f"Numba fused : {t_fused:.3f} s")

    print("\n--- Speedup over NumPy ---")
    print(f"Numba fused : {t_np / t_fused:.2f}x")


def main() -> None:
    HALF_ORDER = 2
    USE_TS_SFD = False
    N_BOUNDARY = 50
    TEST_NX, TEST_NZ, TEST_NT = 601, 601, 800

    print("Warm-up Numba backend...")
    for free_surface in (False, True):
        warm = build_homogeneous_model(
            nx=201,
            nz=201,
            nt=5,
            half_order=HALF_ORDER,
            use_ts_sfd=USE_TS_SFD,
        )
        warm_src, warm_rec = build_geometry(
            warm,
            n_pml=N_BOUNDARY,
            free_surface=free_surface,
        )
        run_timed(
            backend="numba_fused",
            model=warm,
            source=warm_src,
            receivers=warm_rec,
            half_order=HALF_ORDER,
            use_ts_sfd=USE_TS_SFD,
            n_boundary=N_BOUNDARY,
            free_surface=free_surface,
            n_runs=1,
        )

    print(
        f"\nBenchmark setup: nx={TEST_NX}, nz={TEST_NZ}, nt={TEST_NT}, "
        f"order={2 * HALF_ORDER}, ts_sfd={USE_TS_SFD}"
    )

    benchmark_case(
        free_surface=False,
        nx=TEST_NX,
        nz=TEST_NZ,
        nt=TEST_NT,
        half_order=HALF_ORDER,
        use_ts_sfd=USE_TS_SFD,
        n_boundary=N_BOUNDARY,
    )

    benchmark_case(
        free_surface=True,
        nx=TEST_NX,
        nz=TEST_NZ,
        nt=TEST_NT,
        half_order=HALF_ORDER,
        use_ts_sfd=USE_TS_SFD,
        n_boundary=N_BOUNDARY,
    )


if __name__ == "__main__":
    main()