"""
QC 1: verify the staggered SBP identity and the semi-discrete elastic energy rate.
"""

from __future__ import annotations

import numpy as np

from src.sbp_sat import (
    SBPSATState,
    prepare_model,
    semidiscrete_energy_rate,
    staggered_sbp_4_2,
)


def main() -> None:
    h = 5.0
    n_intervals = 40
    op = staggered_sbp_4_2(n_intervals, h)

    sbp_err = float(np.max(np.abs(op.sbp_residual())))

    nx = 48
    nz_N = n_intervals + 1

    # Dimensionless-ish positive elastic model to keep the energy diagnostic
    # numerically well-scaled.
    rho = np.full((nx, nz_N), 2.0)
    vp = np.full((nx, nz_N), 2.0)
    vs = np.full((nx, nz_N), 1.0)
    model = prepare_model(vp, vs, rho)

    rng = np.random.default_rng(20261001)
    state = SBPSATState(
        vx=rng.standard_normal((nx, model.nz_N)),
        vz=rng.standard_normal((nx, model.nz_M)),
        sxx=rng.standard_normal((nx, model.nz_N)),
        szz=rng.standard_normal((nx, model.nz_N)),
        sxz=rng.standard_normal((nx, model.nz_M)),
    )

    dE_dt, scale = semidiscrete_energy_rate(
        state, model, op, dx=h
    )
    relative_energy_residual = abs(dE_dt) / max(scale, 1.0)

    print("SBP-SAT operator / energy QC")
    print("--------------------------------")
    print(f"max SBP identity residual       : {sbp_err:.6e}")
    print(f"semi-discrete dE/dt             : {dE_dt:.6e}")
    print(f"relative energy-rate residual   : {relative_energy_residual:.6e}")

    if sbp_err > 1.0e-12:
        raise RuntimeError("SBP identity failed.")
    if relative_energy_residual > 1.0e-12:
        raise RuntimeError("Semi-discrete energy conservation failed.")

    print("\nSBP identity and SAT energy QC PASSED.")


if __name__ == "__main__":
    main()
