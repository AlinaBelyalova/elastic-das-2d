"""Fully discrete, source-free leapfrog energy QC for the SBP-SAT prototype.

The simultaneous-time physical energy need not be exactly conserved by
leapfrog. For stored state (v^(n-1/2), sigma^n), the conserved quadratic is

    E_LF = .5 v^T M v + .5 sigma^T W sigma + .5 dt v^T M A sigma,

where A is velocity_rhs (including SAT), M is the kinetic energy weight,
and W contains stress compliance and spatial quadrature. M A = -B^T W.
This checks many reflections in a heterogeneous closed domain; it is not
a proof of the maximum stable timestep or of an absorbing-boundary scheme.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.sbp_sat import SBPSATState, advance_one_step, prepare_model, staggered_sbp_4_2
from src.sbp_sat.elastic2d import velocity_rhs


def energies(state, model, op, dx, dt):
    wn, wm = dx*op.H_N[None, :], dx*op.H_M[None, :]
    dvx, dvz = velocity_rhs(state, model, op, dx)
    det = model.l2m_NN**2 - model.lam_NN**2
    elastic = 0.5 * (
        np.sum(wn*(model.l2m_NN*(state.sxx**2+state.szz**2)
                   - 2*model.lam_NN*state.sxx*state.szz)/det)
        + np.sum(wm*state.sxz**2/model.mu_MM))
    kinetic = 0.5*(np.sum(wn*model.rho_vx*state.vx**2)
                   + np.sum(wm*model.rho_vz*state.vz**2))
    cross = 0.5*dt*(np.sum(wn*model.rho_vx*state.vx*dvx)
                    + np.sum(wm*model.rho_vz*state.vz*dvz))
    modified = kinetic + elastic + cross
    # v at the integer stress time = average of the adjacent half-time values.
    vxc, vzc = state.vx+0.5*dt*dvx, state.vz+0.5*dt*dvz
    centred = elastic + 0.5*(np.sum(wn*model.rho_vx*vxc**2)
                             + np.sum(wm*model.rho_vz*vzc**2))
    return float(modified), float(centred)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow writing into a non-empty --output-dir.")
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration < 0.3:
        parser.error("Use a finite duration >= 0.3 s for a multi-reflection check.")
    if args.plot and args.output_dir is None:
        parser.error("--plot requires --output-dir.")
    if (args.output_dir is not None and args.output_dir.exists()
            and any(args.output_dir.iterdir()) and not args.overwrite):
        parser.error(f"{args.output_dir} is not empty; choose a new run name or pass --overwrite.")
    nx, nz, h = 32, 40, 10.0
    x = 2*np.pi*np.arange(nx)[:, None]/nx
    z = np.pi*np.arange(nz+1)[None, :]/nz
    vp = 4400+350*np.cos(x)*np.cos(z)
    vs = 2500+180*np.sin(x)*np.cos(z)
    rho = 2550+100*np.cos(x)*np.sin(z)
    model, op = prepare_model(vp, vs, rho), staggered_sbp_4_2(nz, h)
    dt = 0.12*h/float(vp.max())
    nt = int(np.ceil(args.duration/dt))
    rng = np.random.default_rng(20261016)
    # Random native-grid fields excite boundary, interior and short-wave modes.
    impedance = 2600.0*4400.0
    state = SBPSATState(
        rng.standard_normal((nx, nz+1)), rng.standard_normal((nx, nz)),
        impedance*rng.standard_normal((nx, nz+1)),
        impedance*rng.standard_normal((nx, nz+1)),
        impedance*rng.standard_normal((nx, nz)),
    )
    time, modified, centred = [0.0], [], []
    e, ec = energies(state, model, op, h, dt)
    modified.append(e)
    centred.append(ec)
    for it in range(nt):
        state = advance_one_step(state, model, op, h, dt)
        # Measure every step, including the final state.
        e, ec = energies(state, model, op, h, dt)
        modified.append(e)
        centred.append(ec)
        time.append((it+1)*dt)
    modified, centred = np.asarray(modified), np.asarray(centred)
    drift = float(np.max(np.abs(modified/modified[0]-1)))
    physical_variation = float(np.max(np.abs(centred/centred[0]-1)))
    finite_positive = bool(np.all(np.isfinite(modified)) and np.all(modified > 0)
                           and np.all(np.isfinite(centred)) and np.all(centred > 0))
    passed = finite_positive and drift < 1e-9 and physical_variation < 0.05
    summary = dict(passed=passed, steps=nt, dt_s=dt, duration_s=nt*dt,
                   modified_energy_relative_drift=drift,
                   centred_energy_max_relative_variation=physical_variation,
                   finite_positive_energy=finite_positive,
                   modified_drift_tolerance=1e-9, centred_variation_tolerance=0.05)
    print("SBP-SAT fully discrete leapfrog energy QC")
    print(f"steps / duration        : {nt} / {nt*dt:.6f} s")
    print(f"modified-energy drift   : {drift:.6e}")
    print(f"centred-energy variation: {physical_variation:.6e}")
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        np.savez_compressed(args.output_dir/"traces.npz", time=np.asarray(time),
                            modified_energy=modified, centred_energy=centred)
        if args.plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
            axs[0].plot(time, modified/modified[0]-1, color="#0072B2")
            axs[0].set(ylabel="Modified energy: relative drift",
                       title="SBP-SAT leapfrog | heterogeneous model, no source or damping")
            axs[1].plot(time, centred/centred[0]-1, color="#D55E00", linewidth=0.7)
            axs[1].set(xlabel="Time (s)", ylabel="Centred physical energy: relative change")
            for ax in axs:
                ax.grid(alpha=0.25)
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            fig.savefig(args.output_dir/"time_energy_qc.png", dpi=160)
            plt.close(fig)
    if not passed:
        raise RuntimeError("Fully discrete energy QC FAILED.")
    print("SBP-SAT fully discrete energy QC PASSED.")


if __name__ == "__main__":
    main()
