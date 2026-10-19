"""Oblique P/SV free-surface benchmark for the standalone SBP-SAT branch.

This is an analytic total-field evolution benchmark, not a source experiment:
the incident and reflected monochromatic waves are present in the initial
condition. No analytic field is imposed after initialisation. Four native
staggered grids and the leapfrog half-time offset are respected exactly.

The reference comes directly from continuum elasticity and zero traction,
without using SBP coefficients. The bottom boundary is deliberately distant;
comparison stops before its disturbance reaches the observation slab.
Only propagating, subcritical reflected P/SV waves are supported here.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from src.sbp_sat import (
    SBPSATState, advance_one_step, prepare_model, staggered_sbp_4_2,
)


@dataclass(frozen=True)
class PlaneWave:
    kind: str
    p: float
    q: float
    velocity: np.ndarray
    stress: np.ndarray  # xx, zz, xz; per 1 m/s polarisation amplitude


def plane_wave(kind: str, p: float, sign_z: int,
               vp: float, vs: float, rho: float) -> PlaneWave:
    """v=a exp[i omega(px+qz-t)]; sigma=-C:sym(slowness tensor a)."""
    c = vp if kind == "P" else vs
    if kind not in ("P", "SV") or sign_z not in (-1, 1):
        raise ValueError("Expected P/SV and sign_z=-1/+1.")
    if abs(p) >= 1.0 / c:
        raise ValueError("This benchmark requires propagating reflected waves.")
    q = sign_z * np.sqrt(1.0 / c**2 - p**2)
    a = c * (np.array([p, q]) if kind == "P" else np.array([q, -p]))
    lam, mu = rho * (vp**2 - 2 * vs**2), rho * vs**2
    divergence = p * a[0] + q * a[1]
    stress = -np.array([
        lam * divergence + 2 * mu * p * a[0],
        lam * divergence + 2 * mu * q * a[1],
        mu * (q * a[0] + p * a[1]),
    ])
    return PlaneWave(kind, p, float(q), a, stress)


def analytic_reflection(kind: str, angle: float, vp: float = 5000.0,
                        vs: float = 3000.0, rho: float = 2600.0):
    """Solve the two continuum traction equations; angle is from vertical.

    P polarisation = c(p,q); SV polarisation = c(q,-p), on both branches.
    R_P/R_S are signed particle-velocity amplitudes relative to the incident
    polarisation amplitude. Their signs depend on this stated convention.
    """
    if kind not in ("P", "SV") or not 0 < angle < 90:
        raise ValueError("Use P/SV and 0 < incidence angle < 90 degrees.")
    ci = vp if kind == "P" else vs
    p = np.sin(np.deg2rad(angle)) / ci
    inc = plane_wave(kind, p, -1, vp, vs, rho)
    reflected = [plane_wave(k, p, 1, vp, vs, rho) for k in ("P", "SV")]
    traction = np.column_stack([w.stress[[2, 1]] for w in reflected])
    coeff = np.linalg.solve(traction, -inc.stress[[2, 1]])

    # Independent closed forms and energy-flux check of the reference.
    qp, qs = reflected[0].q, reflected[1].q
    a = 1 / vs**2 - 2 * p**2
    den = a*a + 4*p*p*qp*qs
    r = (4*p*p*qp*qs - a*a) / den
    closed = (np.array([r, 4*(vp/vs)*p*qp*a/den]) if kind == "P"
              else np.array([-4*(vs/vp)*p*qs*a/den, r]))
    np.testing.assert_allclose(coeff, closed, rtol=2e-13, atol=2e-13)
    traction_error = float(np.linalg.norm(traction @ coeff + inc.stress[[2, 1]])
                           / (rho * ci))
    flux = float((vp**2 * qp * coeff[0]**2 + vs**2 * qs * coeff[1]**2)
                 / (ci**2 * abs(inc.q)))
    if traction_error > 1e-12 or abs(flux - 1) > 1e-12:
        raise RuntimeError("Continuum reflection reference failed its checks.")
    return inc, reflected, coeff, flux


def complex_field(waves, amplitudes, omega, x, z, component):
    out = np.zeros((len(x), len(z)), dtype=np.complex128)
    for w, amplitude in zip(waves, amplitudes):
        value = w.velocity[component] if component < 2 else w.stress[component-2]
        out += amplitude * value * np.exp(
            1j * omega * (w.p * x[:, None] + w.q * z[None, :]))
    return out


@dataclass
class Result:
    wave: str
    angle_deg: float
    h_m: float
    dt_s: float
    frequency_hz: float
    duration_s: float
    depth_m: float
    surface_rel_l2: float
    traction_rel_max: float
    r_p_exact: float
    r_s_exact: float
    r_p_numerical_real: float
    r_p_numerical_imag: float
    r_s_numerical_real: float
    r_s_numerical_imag: float
    reflected_coeff_max_error: float
    unwanted_incident_max: float
    mode_fit_rel_residual: float
    mode_fit_condition: float
    analytic_reflected_flux: float


def run_case(kind: str, angle: float, h: float, *, depth: float = 800.0):
    vp, vs, rho = 5000.0, 3000.0, 2600.0
    width, slab_min, slab_max = 200.0, 30.0, 140.0
    for length in (width, depth):
        if not np.isclose(length / h, round(length / h)):
            raise ValueError("h must divide both width=200 m and domain depth.")
    nx, nz = int(round(width/h)), int(round(depth/h))
    if nx < 8:
        raise ValueError("Need at least 8 periodic x samples.")
    op = staggered_sbp_4_2(nz, h)
    model = prepare_model(*[np.full((nx, nz+1), c) for c in (vp, vs, rho)])
    inc, reflected, coeff, flux = analytic_reflection(kind, angle, vp, vs, rho)
    omega = 2 * np.pi / (inc.p * width)  # exactly one periodic x wavelength
    period = 2 * np.pi / omega
    dt = 0.15 * h / vp
    nt = int(np.ceil(2 * period / dt))
    duration = nt * dt
    # 1.5 is a conservative additional buffer, not a discrete causality proof.
    if depth - slab_max <= 1.5 * vp * duration:
        raise ValueError("Domain too shallow: increase depth to isolate the bottom.")
    xN, zN = np.arange(nx)*h, np.arange(nz+1)*h
    xM, zM = xN + h/2, (np.arange(nz)+0.5)*h
    waves, amps = [inc, *reflected], [1.0, *coeff]
    cx = complex_field(waves, amps, omega, xM, zN, 0)
    cz = complex_field(waves, amps, omega, xN, zM, 1)
    initial_half_phase = np.exp(1j*omega*dt/2)
    state = SBPSATState(
        vx=np.real(cx*initial_half_phase),
        vz=np.real(cz*initial_half_phase),
        sxx=complex_field(waves, amps, omega, xN, zN, 2).real,
        szz=complex_field(waves, amps, omega, xN, zN, 3).real,
        sxz=complex_field(waves, amps, omega, xM, zM, 4).real,
    )
    # Recover the complex +kx mode from real fields at their own x coordinates.
    proj_x = (2/nx) * np.exp(-1j*omega*inc.p*xM)
    proj_z = (2/nx) * np.exp(-1j*omega*inc.p*xN)
    zn = (zN >= slab_min) & (zN <= slab_max)
    zm = (zM >= slab_min) & (zM <= slab_max)
    modes = [plane_wave(k, inc.p, s, vp, vs, rho)
             for s in (-1, 1) for k in ("P", "SV")]
    design = np.column_stack([
        np.r_[w.velocity[0]*np.exp(1j*omega*w.q*zN[zn]),
              w.velocity[1]*np.exp(1j*omega*w.q*zM[zm])] for w in modes
    ])
    pinv = np.linalg.pinv(design)
    condition = float(np.linalg.cond(design))
    if condition > 50:
        raise RuntimeError("Modal separation is ill-conditioned for this setup.")
    surface_exact = sum(a*w.velocity for w, a in zip(waves, amps))
    t = (np.arange(nt)+0.5)*dt
    surface = np.empty((nt, 2), dtype=np.complex128)
    traction = np.empty((nt, 2), dtype=np.complex128)
    fits, fit_residuals, fit_times = [], [], []
    for it in range(nt):
        state = advance_one_step(state, model, op, h, dt)
        surface[it] = [proj_x @ state.vx[:, 0], proj_z @ (state.vz @ op.p_left)]
        traction[it] = [proj_x @ (state.sxz @ op.p_left), proj_z @ state.szz[:, 0]]
        if t[it] >= period and (it % 8 == 0 or it == nt-1):
            profile = np.r_[proj_x @ state.vx[:, zn], proj_z @ state.vz[:, zm]]
            profile *= np.exp(1j*omega*t[it])
            fit = pinv @ profile
            fits.append(fit)
            fit_times.append(t[it])
            fit_residuals.append(np.linalg.norm(design @ fit-profile)/np.linalg.norm(profile))
    exact = np.exp(-1j*omega*t[:, None])*surface_exact[None, :]
    mask = t >= period/2
    surface_error = float(np.linalg.norm((surface-exact)[mask]) / np.linalg.norm(exact[mask]))
    ci = vp if kind == "P" else vs
    # Unit incident particle-velocity amplitude V0=1 m/s makes this dimensionless.
    traction_error = float(np.max(np.linalg.norm(traction[mask], axis=1))/(rho*ci*1.0))
    fits = np.asarray(fits)
    inc_index = 0 if kind == "P" else 1
    if np.min(np.abs(fits[:, inc_index])) < 0.5:
        raise RuntimeError("Incident-mode estimate is too small for coefficient ratios.")
    r_num = fits[:, 2:4] / fits[:, inc_index, None]
    r_mean = r_num.mean(axis=0)
    result = Result(
        kind, angle, h, dt, omega/(2*np.pi), duration, depth,
        surface_error, traction_error, float(coeff[0]), float(coeff[1]),
        float(r_mean[0].real), float(r_mean[0].imag),
        float(r_mean[1].real), float(r_mean[1].imag),
        float(np.max(np.abs(r_num-coeff[None, :]))),
        float(np.max(np.abs(fits[:, 1-inc_index]/fits[:, inc_index]))),
        float(np.max(fit_residuals)), condition, flux,
    )
    traces = dict(time=t, surface=surface, exact=exact, traction=traction/(rho*ci),
                  fit_time=np.asarray(fit_times), reflected_coefficients=r_num)
    return result, traces


def check_results(results):
    """Fixed acceptance gates; no automatic tolerance tuning."""
    failures = []
    for kind, angle in sorted({(r.wave, r.angle_deg) for r in results}):
        group = sorted([r for r in results if (r.wave, r.angle_deg) == (kind, angle)],
                       key=lambda r: -r.h_m)
        label = f"{kind} {angle:g} deg"
        for attr in ("surface_rel_l2", "traction_rel_max"):
            values = [getattr(r, attr) for r in group]
            if not all(np.isfinite(values)) or not all(b < a for a, b in zip(values, values[1:])):
                failures.append(f"{label}: {attr} did not decrease under refinement")
        fine = group[-1]
        for attr, tol in (("surface_rel_l2", 1e-2), ("traction_rel_max", 1e-2),
                          ("reflected_coeff_max_error", 1e-2),
                          ("unwanted_incident_max", 1e-2), ("mode_fit_rel_residual", 1e-2)):
            value = getattr(fine, attr)
            if not np.isfinite(value) or value > tol:
                failures.append(f"{label}: finest {attr}={value:.3e} exceeds {tol:.1e}")
    return failures


def save_plot(path, results, traces):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    cases = list(dict.fromkeys((r.wave, r.angle_deg) for r in results))
    fig, axes = plt.subplots(2, len(cases), figsize=(4*len(cases), 6.7), constrained_layout=True)
    for col, case in enumerate(cases):
        group = [r for r in results if (r.wave, r.angle_deg) == case]
        fine = min(group, key=lambda r: r.h_m)
        tr = traces[(fine.wave, fine.angle_deg, fine.h_m)]
        ax = axes[0, col]
        for j, name, colour in ((0, "vx", "#0072B2"), (1, "vz", "#D55E00")):
            ax.plot(tr["time"], tr["exact"][:, j].real, color=colour, label=f"{name} analytic")
            ax.plot(tr["time"], tr["surface"][:, j].real, "--", color=colour,
                    label=f"{name} numeric")
        ax.set(title=f"{case[0]} incidence: {case[1]:g} degrees", xlabel="Time (s)",
               ylabel="Surface velocity at x=0 (m/s)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
        ax = axes[1, col]
        for attr, name in (("surface_rel_l2", "Surface waveform"),
                           ("traction_rel_max", "Traction / (rho c V0)"),
                           ("reflected_coeff_max_error", "Reflected coefficients")):
            ax.loglog([r.h_m for r in group], [getattr(r, attr) for r in group], "o-", label=name)
        ax.set(xlabel="Grid spacing h (m)", ylabel="Error (dimensionless)")
        ticks = sorted(r.h_m for r in group)
        ax.set_xticks(ticks, labels=[f"{h:g}" for h in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(which="both", alpha=0.2)
        ax.legend(fontsize=7)
    fig.suptitle("SBP-SAT oblique total-field benchmark | initial incident + reflected waves", fontsize=14)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hs", type=float, nargs="+", default=[10.0, 5.0, 2.5, 1.25])
    parser.add_argument("--depth", type=float, default=800.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow writing into a non-empty --output-dir.")
    args = parser.parse_args()
    if len(args.hs) < 3 or any(not np.isfinite(h) or h <= 0 for h in args.hs):
        parser.error("Supply at least three finite positive grid spacings.")
    hs = sorted(set(args.hs), reverse=True)
    if len(hs) < 3:
        parser.error("Supply at least three distinct grid spacings.")
    if args.plot and args.output_dir is None:
        parser.error("--plot requires --output-dir.")
    if (args.output_dir is not None and args.output_dir.exists()
            and any(args.output_dir.iterdir()) and not args.overwrite):
        parser.error(f"{args.output_dir} is not empty; choose a new run name or pass --overwrite.")
    cases = [("P", 30.0), ("P", 60.0), ("SV", 20.0), ("SV", 30.0)]
    results, traces = [], {}
    print("SBP-SAT oblique total-field free-surface QC", flush=True)
    print("wave angle   h(m)    surface L2     traction rel      max |dR|", flush=True)
    for kind, angle in cases:
        for h in hs:
            result, trace = run_case(kind, angle, h, depth=args.depth)
            results.append(result)
            traces[(kind, angle, h)] = trace
            print(f"{kind:>3} {angle:5.1f} {h:7.2f}  {result.surface_rel_l2:12.5e}"
                  f"  {result.traction_rel_max:12.5e}  {result.reflected_coeff_max_error:12.5e}", flush=True)
    for kind, angle in cases:
        fine = min([r for r in results if (r.wave, r.angle_deg) == (kind, angle)], key=lambda r: r.h_m)
        print(f"{kind} {angle:g} deg: analytic (R_P, R_S)=({fine.r_p_exact:+.6f}, {fine.r_s_exact:+.6f});"
              f" fitted=({fine.r_p_numerical_real:+.6f}{fine.r_p_numerical_imag:+.2e}j,"
              f" {fine.r_s_numerical_real:+.6f}{fine.r_s_numerical_imag:+.2e}j)")
    failures = check_results(results)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {"benchmark": "initialised total-field evolution; not incident-only packet generation",
                   "passed": not failures, "failures": failures, "results": [asdict(r) for r in results]}
        (args.output_dir/"summary.json").write_text(json.dumps(payload, indent=2)+"\n")
        arrays = {f"{k[0]}_{k[1]:g}_{k[2]:g}_{name}": value for k, trace in traces.items()
                  for name, value in trace.items()}
        np.savez_compressed(args.output_dir/"traces.npz", **arrays)
        if args.plot:
            save_plot(args.output_dir/"oblique_qc.png", results, traces)
    if failures:
        raise RuntimeError("Oblique QC FAILED:\n" + "\n".join(failures))
    print("SBP-SAT oblique total-field QC PASSED.")


if __name__ == "__main__":
    main()
