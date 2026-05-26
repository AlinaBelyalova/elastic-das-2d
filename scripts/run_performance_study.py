from __future__ import annotations

from pathlib import Path
from time import perf_counter
import json
import numpy as np
import matplotlib.pyplot as plt

from src.grid import Grid2D
from src.model import ElasticModel2D
from src.source import build_dc_source
from src.receivers import build_das_cable
from src.simulator import run_forward_simulation
from src.solver_numpy import max_stable_dt


# ==============================================================================
# 1. PLOTTING STYLE
# ==============================================================================

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "figure.titlesize": 15,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
    }
)

COLOR_NUMPY = "#7F8C8D"
COLOR_NUMBA = "#16A085"
COLOR_ACCENT = "#D35400"


# ==============================================================================
# 2. MODEL / GEOMETRY HELPERS
# ==============================================================================

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
    """Build a homogeneous elastic model for performance benchmarking."""
    dt = cfl_safety * max_stable_dt(
        vp, dx, dz, half_order, use_ts_sfd=use_ts_sfd
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


def build_geometry(
    model: ElasticModel2D,
    *,
    n_pml: int = 50,
    free_surface: bool = True,
):
    """
    Build a simple source-receiver geometry for benchmarking:
    - one double-couple source,
    - one vertical DAS cable near the right side of the domain.
    """
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


# ==============================================================================
# 3. BENCHMARK KERNEL
# ==============================================================================

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
) -> float:
    """Run the forward model several times and return the best runtime."""
    times = []

    for _ in range(n_runs):
        t0 = perf_counter()
        run_forward_simulation(
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

    return float(min(times))


# ==============================================================================
# 4. PLOTTING UTILITIES
# ==============================================================================

def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, linestyle="--", alpha=0.5, color="#E0E0E0", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#BDC3C7")
    ax.spines["bottom"].set_color("#BDC3C7")


def generate_plots(data: dict[str, np.ndarray], outdir: Path) -> None:
    """
    Generate the final performance figures:
    1. combined scaling summary,
    2. runtime by discrete benchmark case,
    3. large-case summary.
    """
    x = data["mgrid_updates"]
    labels = data["labels"]

    marker_style_np = dict(
        marker="o",
        ms=7,
        lw=2,
        color=COLOR_NUMPY,
        mec="w",
        mew=1,
        zorder=3,
    )
    marker_style_nb = dict(
        marker="s",
        ms=7,
        lw=2,
        color=COLOR_NUMBA,
        mec="w",
        mew=1,
        zorder=3,
    )

    # ------------------------------------------------------------------
    # Figure 1: combined scaling profile
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    # Runtime scaling
    axes[0].plot(x, data["t_numpy"], label="NumPy baseline", **marker_style_np)
    axes[0].plot(x, data["t_numba"], label="Numba fused", **marker_style_nb)
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"Problem size: $n_x n_z n_t$ [million]")
    axes[0].set_ylabel("Runtime [s] (log scale)")
    axes[0].set_title("Runtime scaling")
    axes[0].legend(frameon=False)
    style_axis(axes[0])

    # Throughput
    axes[1].plot(x, data["thr_numpy"], label="NumPy baseline", **marker_style_np)
    axes[1].plot(x, data["thr_numba"], label="Numba fused", **marker_style_nb)
    axes[1].set_xlabel(r"Problem size: $n_x n_z n_t$ [million]")
    axes[1].set_ylabel("Throughput [MGUPS]")
    axes[1].set_title("Effective throughput")
    axes[1].legend(frameon=False)
    style_axis(axes[1])

    # Speedup
    axes[2].plot(
        x,
        data["speedup"],
        marker="^",
        ms=8,
        lw=2.5,
        color=COLOR_ACCENT,
        mec="w",
        mew=1,
        zorder=3,
    )
    for xi, yi in zip(x, data["speedup"]):
        axes[2].annotate(
            f"{yi:.1f}x",
            xy=(xi, yi),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#34495E",
        )
    axes[2].set_xlabel(r"Problem size: $n_x n_z n_t$ [million]")
    axes[2].set_ylabel("Speedup over NumPy [x]")
    axes[2].set_title("Relative performance gain")
    style_axis(axes[2])

    fig.suptitle("2D elastic solver performance profile (free_surface=True)", y=1.03)
    fig.tight_layout()
    fig.savefig(outdir / "performance_summary_free_surface_true.png", dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: runtime by benchmark case
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    idx_arr = np.arange(labels.size)
    width = 0.35

    b1 = ax.bar(
        idx_arr - width / 2,
        data["t_numpy"],
        width,
        label="NumPy baseline",
        color=COLOR_NUMPY,
        alpha=0.85,
        zorder=3,
    )
    b2 = ax.bar(
        idx_arr + width / 2,
        data["t_numba"],
        width,
        label="Numba fused",
        color=COLOR_NUMBA,
        alpha=0.95,
        zorder=3,
    )

    ax.set_yscale("log")
    ax.set_xticks(idx_arr)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Runtime [s] (log scale)")
    ax.set_title("Runtime by benchmark case")
    ax.legend(frameon=False)
    style_axis(ax)

    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.annotate(
                f"{h:.2f}s" if h >= 0.1 else f"{h:.3f}s",
                xy=(b.get_x() + b.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#2C3E50",
            )

    fig.tight_layout()
    fig.savefig(outdir / "runtime_by_case_free_surface_true.png", dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 3: largest-case summary
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    heavy_lbl = labels[-1].replace("\n", " ")
    t_np = data["t_numpy"][-1]
    t_nb = data["t_numba"][-1]
    sp = data["speedup"][-1]
    thr_np = data["thr_numpy"][-1]
    thr_nb = data["thr_numba"][-1]

    # Left: runtime comparison
    axes[0].bar(
        ["NumPy baseline", "Numba fused"],
        [t_np, t_nb],
        color=[COLOR_NUMPY, COLOR_NUMBA],
        width=0.45,
        zorder=3,
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Runtime [s]")
    axes[0].set_title("Runtime comparison")
    style_axis(axes[0])

    axes[0].text(
        0.5,
        (t_np * t_nb) ** 0.5,
        f"{sp:.1f}x faster",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=COLOR_ACCENT,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#FDEDEC",
            edgecolor=COLOR_ACCENT,
            lw=1.5,
        ),
    )

    # Right: throughput comparison
    bars = axes[1].bar(
        ["NumPy baseline", "Numba fused"],
        [thr_np, thr_nb],
        color=[COLOR_NUMPY, COLOR_NUMBA],
        width=0.45,
        zorder=3,
    )
    axes[1].set_ylabel("Throughput [MGUPS]")
    axes[1].set_title("Computational throughput")
    style_axis(axes[1])

    for b in bars:
        h = b.get_height()
        axes[1].annotate(
            f"{h:,.1f} MGUPS",
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=10.5,
            fontweight="bold",
        )

    fig.suptitle(f"Largest benchmark case ({heavy_lbl})", y=1.03)
    fig.tight_layout()
    fig.savefig(outdir / "largest_case_summary_free_surface_true.png", dpi=300)
    plt.close(fig)


# ==============================================================================
# 5. MAIN DRIVER
# ==============================================================================

def main() -> None:
    outdir_data = Path("results/scaling")
    outdir_figs = Path("results/final_figures")
    outdir_data.mkdir(parents=True, exist_ok=True)
    outdir_figs.mkdir(parents=True, exist_ok=True)

    HALF_ORDER = 2
    USE_TS_SFD = False
    N_BOUNDARY = 50
    FREE_SURFACE = True

    cases = [
        (201, 201, 400),
        (401, 401, 600),
        (601, 601, 800),
    ]

    print("\n" + "=" * 80)
    print("Stage 1: running performance study")
    print("=" * 80)

    print("Warm-up Numba backend...")
    warm_model = build_homogeneous_model(
        nx=201,
        nz=201,
        nt=5,
        half_order=HALF_ORDER,
        use_ts_sfd=USE_TS_SFD,
    )
    warm_src, warm_rec = build_geometry(
        warm_model,
        n_pml=N_BOUNDARY,
        free_surface=FREE_SURFACE,
    )
    _ = run_timed(
        backend="numba_fused",
        model=warm_model,
        source=warm_src,
        receivers=warm_rec,
        half_order=HALF_ORDER,
        use_ts_sfd=USE_TS_SFD,
        n_boundary=N_BOUNDARY,
        free_surface=FREE_SURFACE,
        n_runs=1,
    )
    print("Warm-up completed.\n")

    raw_results: list[dict] = []

    for nx, nz, nt in cases:
        print(f"Case: nx={nx}, nz={nz}, nt={nt}")

        model = build_homogeneous_model(
            nx=nx,
            nz=nz,
            nt=nt,
            half_order=HALF_ORDER,
            use_ts_sfd=USE_TS_SFD,
        )
        source, receivers = build_geometry(
            model,
            n_pml=N_BOUNDARY,
            free_surface=FREE_SURFACE,
        )

        t_numpy = run_timed(
            backend="numpy",
            model=model,
            source=source,
            receivers=receivers,
            half_order=HALF_ORDER,
            use_ts_sfd=USE_TS_SFD,
            n_boundary=N_BOUNDARY,
            free_surface=FREE_SURFACE,
            n_runs=3,
        )
        t_numba = run_timed(
            backend="numba_fused",
            model=model,
            source=source,
            receivers=receivers,
            half_order=HALF_ORDER,
            use_ts_sfd=USE_TS_SFD,
            n_boundary=N_BOUNDARY,
            free_surface=FREE_SURFACE,
            n_runs=3,
        )

        grid_updates = nx * nz * nt
        mgrid_updates = grid_updates / 1.0e6
        speedup = t_numpy / t_numba
        thr_np = mgrid_updates / t_numpy
        thr_nb = mgrid_updates / t_numba

        print(f"  NumPy       : {t_numpy:7.3f} s")
        print(f"  Numba fused : {t_numba:7.3f} s")
        print(f"  Speedup     : {speedup:7.2f}x")
        print(f"  Throughput  : {thr_np:7.1f} -> {thr_nb:7.1f} MGUPS\n")

        raw_results.append(
            {
                "nx": nx,
                "nz": nz,
                "nt": nt,
                "grid_updates": grid_updates,
                "mgrid_updates": mgrid_updates,
                "numpy_time_s": t_numpy,
                "numba_fused_time_s": t_numba,
                "speedup": speedup,
                "numpy_mgrid_updates_per_s": thr_np,
                "numba_fused_mgrid_updates_per_s": thr_nb,
            }
        )

    json_path = outdir_data / "benchmark_scaling_free_surface_true.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2)

    plot_data = {
        "labels": np.array(
            [f"{r['nx']}×{r['nz']}\n(nt={r['nt']})" for r in raw_results],
            dtype=object,
        ),
        "mgrid_updates": np.array([r["mgrid_updates"] for r in raw_results], dtype=float),
        "t_numpy": np.array([r["numpy_time_s"] for r in raw_results], dtype=float),
        "t_numba": np.array([r["numba_fused_time_s"] for r in raw_results], dtype=float),
        "speedup": np.array([r["speedup"] for r in raw_results], dtype=float),
        "thr_numpy": np.array(
            [r["numpy_mgrid_updates_per_s"] for r in raw_results], dtype=float
        ),
        "thr_numba": np.array(
            [r["numba_fused_mgrid_updates_per_s"] for r in raw_results], dtype=float
        ),
    }

    print("=" * 80)
    print("Stage 2: generating figures")
    print("=" * 80)
    generate_plots(plot_data, outdir_figs)

    print("\n" + "=" * 80)
    print("Performance summary")
    print("=" * 80)
    for i, row in enumerate(raw_results, start=1):
        print(f"Case {i}: {row['nx']}×{row['nz']}×{row['nt']}")
        print(f"  NumPy throughput       : {row['numpy_mgrid_updates_per_s']:6.1f} MGUPS")
        print(f"  Numba fused throughput : {row['numba_fused_mgrid_updates_per_s']:6.1f} MGUPS")
        print(f"  Speedup over NumPy     : {row['speedup']:6.1f}x")
    print("=" * 80)
    print(f"Saved JSON results to: {json_path}")
    print(f"Saved figures to: {outdir_figs}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()