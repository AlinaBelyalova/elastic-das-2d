# ==============================================================================
# scripts/validation/two_layer_adjoint_dot_product.py
#
# Full heterogeneous discrete-adjoint validation for the production Numba path.
#
# Tests on the controlled two-layer model:
#
#   A) source sequences -> elastic solver -> receiver Vx/Vz
#
#      <F s, q> = <s, F^T q>
#
#   B) source sequences -> elastic solver -> bilinear receivers -> finite-gauge DAS
#
#      <D F s, r> = <s, F^T D^T r>
#
# Both tests include:
#   - heterogeneous lambda/mu/rho,
#   - staggered material averaging,
#   - the exact production FD coefficients,
#   - bilinear source spreading,
#   - multiplicative sponge,
#   - bilinear receiver sampling,
#   - reverse time stepping,
#   - and, in test B, the finite-gauge DAS operator.
#
# No analytical solution is used here: this is an algebraic transpose test.
# ==============================================================================

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np

from src.das import compute_axial_strain_rate
from src.fwi.adjoint_numba import NumbaElasticDASAdjoint
from src.fwi.two_layer import TwoLayerFWIProblem
from src.solver_numba_fused import run_elastic_solver_numba_fused


def _relative_dot_error(
    lhs: float,
    rhs: float,
) -> float:
    return float(
        abs(lhs - rhs)
        / max(
            abs(lhs),
            abs(rhs),
            1.0e-30,
        )
    )


def _smooth_random_series(
    rng: np.random.Generator,
    nt: int,
    scale: float,
) -> np.ndarray:
    """
    Random but temporally smooth sequence.

    Algebraic adjointness does not require band limitation, but smoothing keeps
    intermediate wavefield magnitudes closer to normal modelling amplitudes.
    """
    raw = rng.standard_normal(
        nt
    )

    kernel = np.array(
        [1.0, 4.0, 6.0, 4.0, 1.0],
        dtype=np.float64,
    )
    kernel /= kernel.sum()

    smooth = np.convolve(
        raw,
        kernel,
        mode="same",
    )

    rms = float(
        np.sqrt(
            np.mean(
                smooth**2
            )
        )
    )

    return (
        float(scale)
        * smooth
        / max(rms, 1.0e-30)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=20260725,
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0e-8,
        help=(
            "Relative dot-product tolerance. The kernels use fastmath and "
            "parallel execution, so 1e-8 is strict enough for the full "
            "thousands-of-step production chain."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    problem = TwoLayerFWIProblem.default()
    medium = problem.true_medium
    source = problem.source
    engine = problem.engine
    grid = problem.grid
    solver = problem.solver_spec

    if solver.free_surface:
        raise RuntimeError(
            "This validation currently requires free_surface=False."
        )

    print(problem.summary())

    rng = np.random.default_rng(
        args.seed
    )

    nt = int(
        grid.nt
    )

    # --------------------------------------------------------------------------
    # Random independent source-component sequences.
    #
    # We test the linear source-to-data map itself rather than only one physical
    # double-couple STF.  This excites all three stress-source components and
    # makes the transpose test substantially harder to pass accidentally.
    # --------------------------------------------------------------------------
    source_scale = 1.0e12

    stf_xx = _smooth_random_series(
        rng,
        nt,
        source_scale,
    )
    stf_zz = _smooth_random_series(
        rng,
        nt,
        source_scale,
    )
    stf_xz = _smooth_random_series(
        rng,
        nt,
        source_scale,
    )

    print(
        "\nRunning random-source forward through production Numba solver..."
    )

    t0 = perf_counter()

    forward = run_elastic_solver_numba_fused(
        vp=medium.vp,
        vs=medium.vs,
        rho=medium.rho,
        dx=grid.dx,
        dz=grid.dz,
        dt=grid.dt,
        nt=grid.nt,
        source_ix=source.source_ix,
        source_iz=source.source_iz,
        stf_xx=stf_xx,
        stf_zz=stf_zz,
        stf_xz=stf_xz,
        receiver_sampling=engine.receiver_sampling,
        half_order=solver.half_order,
        use_ts_sfd=solver.use_ts_sfd,
        n_boundary=solver.n_boundary,
        gamma_s=solver.gamma_s,
        snapshot_stride=None,
        free_surface=False,
        source_injection=source.injection,
    )

    forward_elapsed = (
        perf_counter()
        - t0
    )

    das = compute_axial_strain_rate(
        vx=forward.receiver_vx,
        vz=forward.receiver_vz,
        receivers=problem.receivers,
        gauge_length_m=problem.das_spec.gauge_length_m,
    )

    print(
        f"forward elapsed: {forward_elapsed:.3f} s"
    )

    adjoint = NumbaElasticDASAdjoint.from_forward_engine(
        engine
    )

    # ==========================================================================
    # TEST A: source -> receiver velocities
    # ==========================================================================
    qx = rng.standard_normal(
        forward.receiver_vx.shape
    )
    qz = rng.standard_normal(
        forward.receiver_vz.shape
    )

    lhs_receiver = float(
        np.vdot(
            forward.receiver_vx,
            qx,
        )
        + np.vdot(
            forward.receiver_vz,
            qz,
        )
    )

    print(
        "\nRunning receiver-space adjoint..."
    )

    source_adj_receiver = (
        adjoint.run_receiver_adjoint_to_source(
            medium=medium,
            source=source,
            receiver_qx=qx,
            receiver_qz=qz,
        )
    )

    rhs_receiver = float(
        np.vdot(
            stf_xx,
            source_adj_receiver.grad_stf_xx,
        )
        + np.vdot(
            stf_zz,
            source_adj_receiver.grad_stf_zz,
        )
        + np.vdot(
            stf_xz,
            source_adj_receiver.grad_stf_xz,
        )
    )

    err_receiver = _relative_dot_error(
        lhs_receiver,
        rhs_receiver,
    )

    print(
        "\nSource -> receiver velocity dot product"
    )
    print(
        "---------------------------------------"
    )
    print(
        f"lhs                 : {lhs_receiver:.16e}"
    )
    print(
        f"rhs                 : {rhs_receiver:.16e}"
    )
    print(
        f"relative error      : {err_receiver:.3e}"
    )
    print(
        f"adjoint elapsed     : "
        f"{source_adj_receiver.elapsed_s:.3f} s"
    )

    # ==========================================================================
    # TEST B: source -> receiver velocities -> finite-gauge DAS
    # ==========================================================================
    q_das = rng.standard_normal(
        das.data.shape
    )

    lhs_das = float(
        np.vdot(
            das.data,
            q_das,
        )
    )

    print(
        "\nRunning complete DAS observation adjoint..."
    )

    source_adj_das = (
        adjoint.run_das_adjoint_to_source(
            medium=medium,
            source=source,
            das_adjoint_data=q_das,
            das_template=das,
        )
    )

    rhs_das = float(
        np.vdot(
            stf_xx,
            source_adj_das.grad_stf_xx,
        )
        + np.vdot(
            stf_zz,
            source_adj_das.grad_stf_zz,
        )
        + np.vdot(
            stf_xz,
            source_adj_das.grad_stf_xz,
        )
    )

    err_das = _relative_dot_error(
        lhs_das,
        rhs_das,
    )

    print(
        "\nSource -> DAS dot product"
    )
    print(
        "-------------------------"
    )
    print(
        f"lhs                 : {lhs_das:.16e}"
    )
    print(
        f"rhs                 : {rhs_das:.16e}"
    )
    print(
        f"relative error      : {err_das:.3e}"
    )
    print(
        f"adjoint elapsed     : "
        f"{source_adj_das.elapsed_s:.3f} s"
    )

    # ==========================================================================
    # FINAL DECISION
    # ==========================================================================
    max_error = max(
        err_receiver,
        err_das,
    )

    print(
        "\nAdjoint validation summary"
    )
    print(
        "--------------------------"
    )
    print(
        f"receiver-chain error : {err_receiver:.3e}"
    )
    print(
        f"full DAS-chain error : {err_das:.3e}"
    )
    print(
        f"required tolerance   : {args.tolerance:.3e}"
    )

    if not np.isfinite(max_error):
        raise RuntimeError(
            "Adjoint dot-product error is non-finite."
        )

    if max_error >= args.tolerance:
        raise RuntimeError(
            "Two-layer Numba discrete-adjoint validation FAILED: "
            f"max relative error={max_error:.3e}, "
            f"required < {args.tolerance:.3e}."
        )

    print(
        "Two-layer Numba discrete-adjoint validation PASSED."
    )


if __name__ == "__main__":
    main()
