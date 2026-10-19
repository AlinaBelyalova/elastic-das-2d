"""
QC 2: physical free-surface reflection for the SBP-SAT forward solver.

Normal-incidence P and SV Gaussian plane-wave packets are propagated toward the
top free surface. For a homogeneous half-space, the exact particle velocity at
the free surface is twice the incident particle velocity.

Unlike the Graves ghost implementation, SBP-SAT imposes traction weakly:
surface traction is therefore not set to zero by assignment. Its convergence
toward zero is an additional physical/discrete check.
"""

from __future__ import annotations

import numpy as np

from src.sbp_sat import (
    SBPSATState,
    advance_one_step,
    prepare_model,
    staggered_sbp_4_2,
)


def gaussian(x: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * (x / sigma) ** 2)


def run_case(wave: str, h: float) -> tuple[float, float, float, float]:
    vp0 = 5000.0
    vs0 = 3000.0
    rho0 = 2600.0

    width = 1000.0
    depth = 450.0
    packet_depth = 150.0
    packet_sigma = 25.0
    duration = 0.08

    nx = int(round(width / h))
    n_intervals = int(round(depth / h))
    op = staggered_sbp_4_2(n_intervals, h)

    nz_N = n_intervals + 1
    vp = np.full((nx, nz_N), vp0)
    vs = np.full((nx, nz_N), vs0)
    rho = np.full((nx, nz_N), rho0)
    model = prepare_model(vp, vs, rho)

    # Conservative time step for the first SBP-SAT validation branch.
    dt = 0.15 * h / vp0
    nt = int(np.ceil(duration / dt))

    z_N = np.arange(model.nz_N, dtype=np.float64) * h
    z_M = (np.arange(model.nz_M, dtype=np.float64) + 0.5) * h

    state = SBPSATState(
        vx=np.zeros((nx, model.nz_N)),
        vz=np.zeros((nx, model.nz_M)),
        sxx=np.zeros((nx, model.nz_N)),
        szz=np.zeros((nx, model.nz_N)),
        sxz=np.zeros((nx, model.nz_M)),
    )

    if wave == "P":
        c = vp0
        impedance = rho0 * vp0

        p_sigma = gaussian(
            z_N - packet_depth,
            packet_sigma,
        )
        state.szz[:, :] = impedance * p_sigma[None, :]
        state.sxx[:, :] = (
            model.lam_NN / model.l2m_NN
        ) * state.szz

        # Velocity is staggered by half a time step.
        p_v = gaussian(
            z_M - (packet_depth + 0.5 * c * dt),
            packet_sigma,
        )
        state.vz[:, :] = p_v[None, :]

    elif wave == "SV":
        c = vs0
        impedance = rho0 * vs0

        p_sigma = gaussian(
            z_M - packet_depth,
            packet_sigma,
        )
        state.sxz[:, :] = impedance * p_sigma[None, :]

        p_v = gaussian(
            z_N - (packet_depth + 0.5 * c * dt),
            packet_sigma,
        )
        state.vx[:, :] = p_v[None, :]

    else:
        raise ValueError("wave must be P or SV.")

    t = (np.arange(nt, dtype=np.float64) + 0.5) * dt
    trace = np.zeros(nt)
    traction = np.zeros(nt)
    ix = nx // 2

    for it in range(nt):
        state = advance_one_step(
            state, model, op, h, dt
        )

        if wave == "P":
            # vz is M_z staggered. Use the SBP boundary projection.
            trace[it] = state.vz[ix] @ op.p_left
            traction[it] = abs(state.szz[ix, 0])
        else:
            # vx lives directly on the N_z top boundary.
            trace[it] = state.vx[ix, 0]
            traction[it] = abs(state.sxz[ix] @ op.p_left)

    analytical = (
        2.0
        * gaussian(
            -packet_depth + c * t,
            packet_sigma,
        )
    )
    mask = analytical >= 1.0e-5 * float(np.max(analytical))

    rel_l2 = float(
        np.linalg.norm(trace[mask] - analytical[mask])
        / np.linalg.norm(analytical[mask])
    )
    peak_ratio = float(
        np.max(np.abs(trace[mask]))
        / np.max(np.abs(analytical[mask]))
    )

    # Normalize traction by the natural impedance*velocity scale.
    traction_ratio = float(
        np.max(traction)
        / impedance
    )
    return rel_l2, peak_ratio, traction_ratio, dt


def main() -> None:
    hs = (10.0, 5.0, 2.5)

    print("SBP-SAT normal-incidence free-surface QC")
    print(
        "wave   h(m)       dt(s)        rel_L2      peak_ratio   "
        "max traction/(rho*c)"
    )
    print("-" * 84)

    all_results: dict[str, list[tuple[float, float]]] = {
        "P": [],
        "SV": [],
    }

    for wave in ("P", "SV"):
        for h in hs:
            err, peak, tr, dt = run_case(wave, h)
            all_results[wave].append((h, err))
            print(
                f"{wave:>3s} "
                f"{h:7.2f} "
                f"{dt:12.5e} "
                f"{err:12.5e} "
                f"{peak:12.6f} "
                f"{tr:18.5e}"
            )

    print("\nObserved refinement orders")
    for wave in ("P", "SV"):
        vals = all_results[wave]
        for (h0, e0), (h1, e1) in zip(vals[:-1], vals[1:]):
            p = np.log(e0 / e1) / np.log(h0 / h1)
            print(
                f"{wave}: h {h0:g} -> {h1:g} m, "
                f"error {e0:.4e} -> {e1:.4e}, p={p:.3f}"
            )

    for wave in ("P", "SV"):
        errors = [e for _, e in all_results[wave]]
        if not (errors[2] < errors[1] < errors[0]):
            raise RuntimeError(
                f"{wave} free-surface error did not decrease under refinement."
            )

    print("\nSBP-SAT physical free-surface QC PASSED.")


if __name__ == "__main__":
    main()
