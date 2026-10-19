"""
QC 4: Taylor test for the exact rho-fixed log(Vp), log(Vs) model gradient.

Parameterization:
    mp = log(Vp)
    ms = log(Vs)

Perturbation:
    Vp(eps) = Vp0 * exp(eps * p_p)
    Vs(eps) = Vs0 * exp(eps * p_s)

Objective:
    J = 0.5 * dt * ||d_syn - d_obs||^2

Expected:
    R0 = |J(m+eps p) - J(m)|                  ~ O(eps)
    R1 = |J(m+eps p) - J(m) - eps <g,p>|     ~ O(eps^2)
"""

from __future__ import annotations

import numpy as np

from src.sbp_sat import (
    PointReceiver,
    StressPointSource,
    objective_and_gradient_log_vp_vs,
    prepare_model,
    run_forward,
    staggered_sbp_4_2,
)


def ricker(t: np.ndarray, f0: float, t0: float) -> np.ndarray:
    a = np.pi * f0 * (t - t0)
    return (1.0 - 2.0 * a**2) * np.exp(-a**2)


def main() -> None:
    nx = 28
    h = 10.0
    n_intervals = 36
    nz_N = n_intervals + 1

    op = staggered_sbp_4_2(n_intervals, h)

    x = np.arange(nx, dtype=np.float64)[:, None]
    z = np.arange(nz_N, dtype=np.float64)[None, :]

    rho = np.full((nx, nz_N), 2500.0)

    vp0 = (
        4200.0
        + 120.0
        * np.sin(2.0 * np.pi * x / nx)
        * np.exp(-((z - 18.0) / 10.0) ** 2)
    )
    vs0 = (
        2400.0
        + 70.0
        * np.cos(2.0 * np.pi * x / nx)
        * np.exp(-((z - 18.0) / 10.0) ** 2)
    )

    # Smooth "true" perturbation.
    q_vp = np.exp(
        -((x - 14.0) / 5.0) ** 2
        -((z - 18.0) / 6.0) ** 2
    )
    q_vs = -0.8 * np.exp(
        -((x - 10.0) / 6.0) ** 2
        -((z - 21.0) / 5.0) ** 2
    )

    vp_true = vp0 * np.exp(0.035 * q_vp)
    vs_true = vs0 * np.exp(0.025 * q_vs)

    model0 = prepare_model(vp0, vs0, rho)
    model_true = prepare_model(vp_true, vs_true, rho)

    # IMPORTANT: dt is fixed for every perturbed model in the Taylor test.
    dt = 0.10 * h / float(np.max(vp_true))
    nt = 220
    t = np.arange(nt, dtype=np.float64) * dt

    stf = 1.0e10 * ricker(t, f0=18.0, t0=0.025)

    source = StressPointSource(
        ix=nx // 2,
        iz=22,
        mxx=1.0,
        mzz=0.65,
    )
    receivers = (
        PointReceiver("vx", 2, 0),
        PointReceiver("vx", 7, 0),
        PointReceiver("vx", 20, 0),
        PointReceiver("vz", 4, 2),
        PointReceiver("vz", 12, 4),
        PointReceiver("vz", 24, 6),
    )

    observed = run_forward(
        stf,
        model_true,
        op,
        h,
        dt,
        source,
        receivers,
    ).data

    J0, _, g_log_vp, g_log_vs = objective_and_gradient_log_vp_vs(
        stf,
        observed,
        model0,
        op,
        h,
        dt,
        source,
        receivers,
    )

    # Independent smooth directional perturbation.
    p_vp = np.exp(
        -((x - 9.0) / 6.0) ** 2
        -((z - 16.0) / 8.0) ** 2
    )
    p_vs = -0.7 * np.exp(
        -((x - 18.0) / 5.0) ** 2
        -((z - 20.0) / 7.0) ** 2
    )

    norm = np.sqrt(np.sum(p_vp**2) + np.sum(p_vs**2))
    p_vp /= norm
    p_vs /= norm

    gdotp = float(
        np.sum(g_log_vp * p_vp)
        + np.sum(g_log_vs * p_vs)
    )

    eps_values = np.array(
        [2.0e-2, 1.0e-2, 5.0e-3, 2.5e-3, 1.25e-3],
        dtype=np.float64,
    )
    R0 = []
    R1 = []

    print("SBP-SAT log(Vp), log(Vs) Taylor test")
    print("-------------------------------------")
    print(f"J0       = {J0:.16e}")
    print(f"<g,p>    = {gdotp:.16e}")
    print()
    print("eps          R0                 R1")

    for eps in eps_values:
        model_eps = prepare_model(
            vp0 * np.exp(eps * p_vp),
            vs0 * np.exp(eps * p_vs),
            rho,
        )
        d_eps = run_forward(
            stf,
            model_eps,
            op,
            h,
            dt,
            source,
            receivers,
        ).data
        J_eps = 0.5 * dt * float(
            np.sum((d_eps - observed) ** 2)
        )

        r0 = abs(J_eps - J0)
        r1 = abs(J_eps - J0 - eps * gdotp)
        R0.append(r0)
        R1.append(r1)

        print(f"{eps:9.3e}  {r0:17.9e}  {r1:17.9e}")

    R0 = np.asarray(R0)
    R1 = np.asarray(R1)

    slope0 = np.polyfit(
        np.log(eps_values[-4:]),
        np.log(R0[-4:]),
        1,
    )[0]
    slope1 = np.polyfit(
        np.log(eps_values[-4:]),
        np.log(R1[-4:]),
        1,
    )[0]

    # Centered finite-difference directional derivative at a small epsilon.
    eps_fd = 2.0e-5
    model_p = prepare_model(
        vp0 * np.exp(eps_fd * p_vp),
        vs0 * np.exp(eps_fd * p_vs),
        rho,
    )
    model_m = prepare_model(
        vp0 * np.exp(-eps_fd * p_vp),
        vs0 * np.exp(-eps_fd * p_vs),
        rho,
    )

    dp = run_forward(
        stf, model_p, op, h, dt, source, receivers
    ).data
    dm = run_forward(
        stf, model_m, op, h, dt, source, receivers
    ).data

    Jp = 0.5 * dt * float(np.sum((dp - observed) ** 2))
    Jm = 0.5 * dt * float(np.sum((dm - observed) ** 2))
    fd = (Jp - Jm) / (2.0 * eps_fd)
    rel_fd = abs(fd - gdotp) / max(abs(fd), abs(gdotp), 1.0e-300)

    print()
    print(f"R0 slope                    = {slope0:.6f}")
    print(f"R1 slope                    = {slope1:.6f}")
    print(f"FD directional derivative   = {fd:.16e}")
    print(f"gradient directional deriv. = {gdotp:.16e}")
    print(f"relative derivative error   = {rel_fd:.6e}")

    if not (0.8 < slope0 < 1.2):
        raise RuntimeError("R0 is not first order.")
    if not (1.7 < slope1 < 2.3):
        raise RuntimeError("R1 is not second order.")
    if rel_fd > 1.0e-5:
        raise RuntimeError("Directional derivative check FAILED.")

    print("\nSBP-SAT model-gradient Taylor test PASSED.")


if __name__ == "__main__":
    main()
