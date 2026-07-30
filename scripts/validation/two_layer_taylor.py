# ==============================================================================
# scripts/validation/two_layer_taylor.py
#
# Directional-derivative + Taylor test of the checkpointed Numba DAS gradient.
#
# Parameterisation:
#     m_p = log(Vp)
#     m_s = log(Vs)
#
# Perturbation:
#     Vp(eps) = Vp0 * exp(eps * p_p)
#     Vs(eps) = Vs0 * exp(eps * p_s)
#
# Objective:
#     J = 0.5 * dt * dCh * || d_syn - d_obs ||^2
#
# Expected asymptotic behaviour:
#     R0(eps) = |J(m+eps p) - J(m)|              ~ O(eps)
#     R1(eps) = |J(m+eps p) - J(m) - eps g.p|    ~ O(eps^2)
#
# This test uses the heterogeneous two-layer true model as observations and the
# smooth-interface model as the point where the gradient is evaluated.
# ==============================================================================

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.fwi.gradient_numba import NumbaElasticDASGradient
from src.fwi.two_layer import TwoLayerFWIProblem


def _direction_fields(
    problem: TwoLayerFWIProblem,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Smooth deterministic dimensionless log-velocity perturbation direction.

    It is zeroed inside the sponge so the test targets the scientific model
    domain rather than boundary material changes.
    """
    g = problem.grid

    x = (
        np.asarray(
            g.x,
            dtype=np.float64,
        )
        - float(g.x[0])
    ) / (
        float(g.x[-1])
        - float(g.x[0])
    )

    z = (
        np.asarray(
            g.z,
            dtype=np.float64,
        )
        - float(g.z[0])
    ) / (
        float(g.z[-1])
        - float(g.z[0])
    )

    X, Z = np.meshgrid(
        x,
        z,
        indexing="ij",
    )

    p_vp = (
        np.sin(2.0 * np.pi * X)
        * np.sin(np.pi * Z)
        + 0.35
        * np.sin(3.0 * np.pi * Z)
    )

    p_vs = (
        np.cos(1.5 * np.pi * X)
        * np.sin(2.0 * np.pi * Z)
        - 0.25
        * np.sin(2.0 * np.pi * X)
    )

    n_boundary = int(
        problem.solver_spec.n_boundary
    )

    mask = np.zeros(
        (g.nx, g.nz),
        dtype=np.float64,
    )

    mask[
        n_boundary : g.nx - n_boundary,
        n_boundary : g.nz - n_boundary,
    ] = 1.0

    p_vp *= mask
    p_vs *= mask

    norm = float(
        np.sqrt(
            np.sum(p_vp**2)
            + np.sum(p_vs**2)
        )
    )

    if norm == 0.0:
        raise RuntimeError(
            "Perturbation direction has zero norm."
        )

    # Normalise by max absolute component instead of L2 so epsilon has a direct
    # interpretation as a maximum approximate fractional velocity perturbation.
    amplitude = max(
        float(np.max(np.abs(p_vp))),
        float(np.max(np.abs(p_vs))),
    )

    p_vp /= amplitude
    p_vs /= amplitude

    return (
        np.ascontiguousarray(p_vp),
        np.ascontiguousarray(p_vs),
    )


def _log_slope(
    eps: np.ndarray,
    residual: np.ndarray,
) -> float:
    mask = (
        np.isfinite(eps)
        & np.isfinite(residual)
        & (eps > 0.0)
        & (residual > 0.0)
    )

    if np.count_nonzero(mask) < 2:
        return np.nan

    coefficient = np.polyfit(
        np.log10(eps[mask]),
        np.log10(residual[mask]),
        deg=1,
    )

    return float(
        coefficient[0]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=96,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "results/validation/two_layer_taylor"
        ),
    )

    parser.add_argument(
        "--eps",
        type=float,
        nargs="*",
        default=[
            2.0e-2,
            1.0e-2,
            5.0e-3,
            2.0e-3,
            1.0e-3,
            5.0e-4,
            2.0e-4,
        ],
    )

    parser.add_argument(
        "--strict",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    eps_values = np.array(
        args.eps,
        dtype=np.float64,
    )

    if (
        eps_values.ndim != 1
        or eps_values.size < 3
        or np.any(eps_values <= 0.0)
    ):
        raise ValueError(
            "--eps must provide at least three positive values."
        )

    eps_values = np.sort(
        eps_values
    )[::-1]

    problem = TwoLayerFWIProblem.default()

    print(problem.summary())

    print(
        "\nGenerating observed DAS from the sharp true model..."
    )

    observed_result = problem.run_true()
    observed_das = observed_result.das.data

    initial = problem.smooth_initial_medium(
        transition_half_width_m=100.0,
    )

    gradient_engine = (
        NumbaElasticDASGradient.from_forward_engine(
            problem.engine,
            channel_spacing_m=problem.das_spec.channel_spacing_m,
            checkpoint_interval=args.checkpoint_interval,
        )
    )

    print(
        "\nComputing checkpointed discrete gradient at smooth initial model..."
    )

    gradient = gradient_engine.compute_gradient(
        medium=initial,
        source=problem.source,
        observed_das=observed_das,
    )

    print(
        f"J(m0)                 : {gradient.objective:.16e}"
    )
    print(
        f"gradient forward time : {gradient.forward_elapsed_s:.3f} s"
    )
    print(
        f"replay time           : {gradient.replay_elapsed_s:.3f} s"
    )
    print(
        f"reverse time          : {gradient.reverse_elapsed_s:.3f} s"
    )
    print(
        f"working memory approx : {gradient.checkpoint_memory_mb:.1f} MB"
    )

    p_log_vp, p_log_vs = _direction_fields(
        problem
    )

    analytic_directional = float(
        np.vdot(
            gradient.g_log_vp,
            p_log_vp,
        )
        + np.vdot(
            gradient.g_log_vs,
            p_log_vs,
        )
    )

    print(
        f"<g,p>                 : {analytic_directional:.16e}"
    )

    J_plus = np.empty_like(
        eps_values
    )
    J_minus = np.empty_like(
        eps_values
    )

    R0 = np.empty_like(
        eps_values
    )
    R1 = np.empty_like(
        eps_values
    )
    fd_derivative = np.empty_like(
        eps_values
    )
    derivative_rel_error = np.empty_like(
        eps_values
    )

    print(
        "\nTaylor / directional derivative sweep"
    )
    print(
        "-------------------------------------"
    )

    for i, eps in enumerate(
        eps_values
    ):
        vp_plus = (
            initial.vp
            * np.exp(
                eps
                * p_log_vp
            )
        )
        vs_plus = (
            initial.vs
            * np.exp(
                eps
                * p_log_vs
            )
        )

        vp_minus = (
            initial.vp
            * np.exp(
                -eps
                * p_log_vp
            )
        )
        vs_minus = (
            initial.vs
            * np.exp(
                -eps
                * p_log_vs
            )
        )

        plus = problem.medium_with_velocities(
            vp=vp_plus,
            vs=vs_plus,
            rho=initial.rho,
        )
        minus = problem.medium_with_velocities(
            vp=vp_minus,
            vs=vs_minus,
            rho=initial.rho,
        )

        J_plus[i] = gradient_engine.objective_only(
            medium=plus,
            source=problem.source,
            observed_das=observed_das,
        )

        J_minus[i] = gradient_engine.objective_only(
            medium=minus,
            source=problem.source,
            observed_das=observed_das,
        )

        fd_derivative[i] = (
            J_plus[i]
            - J_minus[i]
        ) / (
            2.0
            * eps
        )

        derivative_rel_error[i] = (
            abs(
                fd_derivative[i]
                - analytic_directional
            )
            / max(
                abs(fd_derivative[i]),
                abs(analytic_directional),
                1.0e-30,
            )
        )

        R0[i] = abs(
            J_plus[i]
            - gradient.objective
        )

        R1[i] = abs(
            J_plus[i]
            - gradient.objective
            - eps
            * analytic_directional
        )

        print(
            f"eps={eps:9.2e}  "
            f"J+={J_plus[i]:.8e}  "
            f"FD={fd_derivative[i]:+.8e}  "
            f"dir.err={derivative_rel_error[i]:.3e}  "
            f"R0={R0[i]:.3e}  "
            f"R1={R1[i]:.3e}"
        )

    # Use the four largest eps values for the asymptotic slope estimate.
    # Very small eps may eventually flatten because of floating-point noise.
    n_slope = min(
        4,
        eps_values.size,
    )

    slope_R0 = _log_slope(
        eps_values[:n_slope],
        R0[:n_slope],
    )
    slope_R1 = _log_slope(
        eps_values[:n_slope],
        R1[:n_slope],
    )

    best_derivative_error = float(
        np.min(
            derivative_rel_error
        )
    )

    print(
        "\nTaylor summary"
    )
    print(
        "--------------"
    )
    print(
        f"best directional derivative rel. error : "
        f"{best_derivative_error:.3e}"
    )
    print(
        f"R0 slope                              : "
        f"{slope_R0:.4f}  (target ~1)"
    )
    print(
        f"R1 slope                              : "
        f"{slope_R1:.4f}  (target ~2)"
    )

    np.savez_compressed(
        args.out_dir / "taylor_results.npz",
        eps=eps_values,
        J0=np.array(
            gradient.objective
        ),
        J_plus=J_plus,
        J_minus=J_minus,
        analytic_directional=np.array(
            analytic_directional
        ),
        fd_derivative=fd_derivative,
        derivative_rel_error=derivative_rel_error,
        R0=R0,
        R1=R1,
        slope_R0=np.array(
            slope_R0
        ),
        slope_R1=np.array(
            slope_R1
        ),
        g_log_vp=gradient.g_log_vp,
        g_log_vs=gradient.g_log_vs,
        p_log_vp=p_log_vp,
        p_log_vs=p_log_vs,
    )

    fig, ax = plt.subplots(
        figsize=(7.0, 5.5)
    )

    ax.loglog(
        eps_values,
        R0,
        "o-",
        label=f"R0, slope={slope_R0:.2f}",
    )
    ax.loglog(
        eps_values,
        R1,
        "o-",
        label=f"R1, slope={slope_R1:.2f}",
    )

    ax.set_xlabel("epsilon")
    ax.set_ylabel("Taylor remainder")
    ax.set_title("Two-layer Numba discrete-gradient Taylor test")
    ax.legend()
    ax.grid(
        True,
        which="both",
        alpha=0.3,
    )

    fig.tight_layout()
    fig.savefig(
        args.out_dir / "taylor_remainders.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    if args.strict:
        failures = []

        if best_derivative_error >= 1.0e-4:
            failures.append(
                "best directional derivative relative error "
                f"{best_derivative_error:.3e} >= 1e-4"
            )

        if not (
            0.85
            <= slope_R0
            <= 1.15
        ):
            failures.append(
                f"R0 slope {slope_R0:.3f} is not near first order"
            )

        if not (
            1.75
            <= slope_R1
            <= 2.25
        ):
            failures.append(
                f"R1 slope {slope_R1:.3f} is not near second order"
            )

        if failures:
            print(
                "\nSTRICT FAILURES:"
            )
            for failure in failures:
                print(
                    f"  - {failure}"
                )
            raise RuntimeError(
                "Two-layer Numba Taylor test FAILED."
            )

    print(
        f"\nSaved results to: {args.out_dir.resolve()}"
    )
    print(
        "Two-layer Numba Taylor test completed."
    )


if __name__ == "__main__":
    main()
