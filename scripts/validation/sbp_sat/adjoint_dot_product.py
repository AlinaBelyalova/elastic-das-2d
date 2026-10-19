"""
QC 3: exact discrete source-to-data adjoint dot-product test.

Forward operator:
    source time sequence -> SBP-SAT elastic propagation -> point receivers

Adjoint operator:
    receiver-space vector -> exact Euclidean transpose -> source sequence

Pass criterion:
    <F s, q> = <s, F^T q>
to near machine precision.
"""

from __future__ import annotations

import numpy as np

from src.sbp_sat import (
    PointReceiver,
    StressPointSource,
    prepare_model,
    run_adjoint_to_source,
    run_forward,
    staggered_sbp_4_2,
)


def main() -> None:
    nx = 36
    h = 10.0
    n_intervals = 32
    nz_N = n_intervals + 1

    op = staggered_sbp_4_2(n_intervals, h)

    rho = np.full((nx, nz_N), 2500.0)
    vp = np.full((nx, nz_N), 4200.0)
    vs = np.full((nx, nz_N), 2400.0)
    model = prepare_model(vp, vs, rho)

    dt = 0.12 * h / float(vp.max())
    nt = 120

    source = StressPointSource(
        ix=nx // 2,
        iz=nz_N // 2,
        mxx=1.0,
        mzz=-0.35,
    )
    receivers = (
        PointReceiver("vx", 4, 0),
        PointReceiver("vx", 12, 3),
        PointReceiver("vz", 20, 2),
        PointReceiver("vz", 28, 10),
    )

    rng = np.random.default_rng(20261001)
    s = rng.standard_normal(nt)
    q = rng.standard_normal((len(receivers), nt))

    d = run_forward(
        s, model, op, h, dt, source, receivers
    ).data
    Ftq = run_adjoint_to_source(
        q, model, op, h, dt, source, receivers
    )

    lhs = float(np.vdot(d, q))
    rhs = float(np.vdot(s, Ftq))
    rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0e-300)

    print("SBP-SAT source-to-data adjoint dot product")
    print("------------------------------------------")
    print(f"<F s, q>   = {lhs:.16e}")
    print(f"<s, F^T q> = {rhs:.16e}")
    print(f"relative error = {rel:.6e}")

    if rel > 1.0e-11:
        raise RuntimeError("SBP-SAT adjoint dot-product test FAILED.")

    print("\nSBP-SAT adjoint dot-product test PASSED.")


if __name__ == "__main__":
    main()
