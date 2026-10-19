"""
Physical free-surface QC: normally incident P and SV plane-wave packets.

Why this test
-------------
The algebraic Graves FS1 QC proves that the boundary equations are satisfied by
construction, but it does not yet prove that the time-stepping scheme reflects
elastic waves with the correct phase and amplitude.

For a homogeneous isotropic half-space, a normally incident plane P wave and a
normally incident plane SV wave have especially simple exact free-surface
solutions:

    normal traction:  sigma_zz = 0
    shear traction:   sigma_xz = 0

At normal incidence, the reflected particle-velocity pulse has the same sign
and amplitude as the incident pulse. Therefore the particle velocity exactly at
the free surface is twice the incident velocity:

    v_surface(t) = 2 v_incident(t).

This script embeds a 1-D plane wave in the 2-D production staggered grid
(constant in x), propagates it with the actual Numba Graves-FS1 kernels, and
compares the numerical surface trace with the analytical result.

Two tests are run:
  P  : vertical propagation, vertical particle velocity vz, sigma_zz traction
  SV : vertical propagation, horizontal particle velocity vx, sigma_xz traction

The test is repeated for several grid spacings while keeping the PHYSICAL
domain width/depth fixed, so that refinement changes only resolution and not
the distance to numerical side boundaries.

Important
---------
This is the first PHYSICAL free-surface benchmark.  It is still simpler than a
point-source half-space / reflectivity benchmark because normal incidence has
no P-SV conversion and no Rayleigh wave.  A point-source benchmark should come
next if this test is satisfactory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.elastic.solver_numpy import (
    c_sfd_coefficients,
    max_stable_dt,
    prepare_staggered_materials,
)
from src.elastic.solver_numba_fused import (
    fill_velocity_ghosts_free_surface_numba,
    mirror_stress_ghosts_free_surface_numba,
    update_velocity_fused_numba,
    update_stress_fused_free_surface_numba,
)


@dataclass(frozen=True)
class CaseResult:
    wave: str
    h: float
    dt: float
    t: np.ndarray
    numerical: np.ndarray
    analytical: np.ndarray
    relative_l2: float
    peak_ratio: float
    peak_time_error_s: float
    max_surface_traction_pa: float


def gaussian(x: np.ndarray | float, sigma_m: float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    return np.exp(-0.5 * (x_arr / sigma_m) ** 2)


def _surface_trace_p(vz: np.ndarray, k: int, i0: int) -> float:
    """
    FS1 places vz half a grid point below/above the surface.

    Average the two staggered values to obtain a surface-centered estimate.
    Gottschammer & Olsen (2001) discuss this averaging as a useful estimate at
    the actual surface.
    """
    return 0.5 * (vz[i0, k - 1] + vz[i0, k])


def _run_case(
    *,
    wave: str,
    h: float,
    vp: float,
    vs: float,
    rho0: float,
    packet_depth_m: float,
    packet_sigma_m: float,
    amplitude: float,
    duration_s: float,
    depth_m: float,
    width_m: float,
) -> CaseResult:
    if wave not in {"P", "SV"}:
        raise ValueError("wave must be 'P' or 'SV'.")

    M = 2
    a = c_sfd_coefficients(M)

    # Keep the PHYSICAL lateral width fixed under grid refinement.
    #
    # The production kernels intentionally leave M edge columns un-updated.
    # If nx were held fixed while h decreased, the physical distance from the
    # central receiver to those untouched edge columns would shrink, and an
    # artificial lateral disturbance would reach the receiver earlier and
    # earlier.  That was the flaw in the first version of this benchmark.
    nx = int(np.ceil(width_m / h)) + 1
    if nx < 2 * M + 9:
        raise ValueError("width_m is too small for the fourth-order stencil.")

    # The physical free surface is the normal-stress row j=M. Coordinates
    # below are measured relative to that surface, so z=0 at j=M.
    nz = int(np.ceil(depth_m / h)) + M + 3

    vp_arr = np.full((nx, nz), vp, dtype=np.float64)
    vs_arr = np.full((nx, nz), vs, dtype=np.float64)
    rho = np.full((nx, nz), rho0, dtype=np.float64)

    mu = rho * vs_arr**2
    lam = rho * (vp_arr**2 - 2.0 * vs_arr**2)
    l2m = lam + 2.0 * mu
    bx, bz, mu_xz = prepare_staggered_materials(rho, mu)

    dt = max_stable_dt(
        vp_max=vp,
        dx=h,
        dz=h,
        half_order=M,
        safety=0.55,
        use_ts_sfd=False,
    )
    nt = int(np.ceil(duration_s / dt))

    # Staggered z coordinates relative to the physical surface z=0.
    j = np.arange(nz, dtype=np.float64)
    z_normal = (j - M) * h
    z_half = (j + 0.5 - M) * h

    vx = np.zeros((nx, nz), dtype=np.float64)
    vz = np.zeros((nx, nz), dtype=np.float64)
    sxx = np.zeros((nx, nz), dtype=np.float64)
    szz = np.zeros((nx, nz), dtype=np.float64)
    sxz = np.zeros((nx, nz), dtype=np.float64)

    if wave == "P":
        c = vp
        impedance = rho0 * vp

        # Stress lives at t=0.
        packet_stress = gaussian(
            z_normal - packet_depth_m,
            packet_sigma_m,
        )
        szz[:, :] = (
            impedance * amplitude * packet_stress[None, :]
        )
        sxx[:, :] = (
            (lam / l2m)
            * szz
        )

        # Velocity lives at t=-dt/2 initially.  An upward pulse has its center
        # slightly DEEPER at negative time:
        #
        #   z_center(-dt/2) = z0 + c dt/2.
        packet_velocity = gaussian(
            z_half
            - (
                packet_depth_m
                + 0.5 * c * dt
            ),
            packet_sigma_m,
        )
        vz[:, :] = amplitude * packet_velocity[None, :]

    else:
        c = vs
        impedance = rho0 * vs

        # Upward normally incident SV:
        #   vx != 0, sigma_xz != 0.
        packet_stress = gaussian(
            z_half - packet_depth_m,
            packet_sigma_m,
        )
        sxz[:, :] = (
            impedance * amplitude * packet_stress[None, :]
        )

        packet_velocity = gaussian(
            z_normal
            - (
                packet_depth_m
                + 0.5 * c * dt
            ),
            packet_sigma_m,
        )
        vx[:, :] = amplitude * packet_velocity[None, :]

    # The exact half-space initial condition would already contain the
    # exponentially small reflected tail.  Our packet is many sigma below the
    # surface, so imposing the Graves stress image here supplies that boundary
    # state while changing the physical packet only negligibly.
    mirror_stress_ghosts_free_surface_numba(
        sxx, szz, sxz, M
    )

    t = (np.arange(nt, dtype=np.float64) + 0.5) * dt
    numerical = np.zeros(nt, dtype=np.float64)
    max_traction = 0.0
    i0 = nx // 2

    # The x-edge columns are not advanced by the production FD kernels.  For
    # this x-invariant plane-wave test they therefore act as a possible source
    # of lateral numerical contamination.  Require the comparison window to
    # end before even a P-wave-speed disturbance from the nearest active-edge
    # transition can reach the central receiver.
    edge_distance_m = (i0 - M) * h
    edge_contamination_time_s = edge_distance_m / vp
    if duration_s >= 0.90 * edge_contamination_time_s:
        raise ValueError(
            "Plane-wave QC is not laterally isolated: "
            f"duration={duration_s:.5f} s, estimated edge-arrival="
            f"{edge_contamination_time_s:.5f} s. Increase width_m or "
            "shorten duration_s."
        )

    for it in range(nt):
        # sigma^n -> v^(n+1/2)
        update_velocity_fused_numba(
            vx, vz, sxx, szz, sxz,
            bx, bz,
            h, h, dt, a,
        )

        # Construct Graves velocity values above the free surface from the NEW
        # half-time-level velocity.
        fill_velocity_ghosts_free_surface_numba(
            vx, vz, lam, l2m, h, h, M
        )

        # Record surface velocity at t_v[it].
        if wave == "P":
            numerical[it] = _surface_trace_p(vz, M, i0)
        else:
            # FS1 places horizontal velocity exactly at the surface.
            numerical[it] = vx[i0, M]

        # v^(n+1/2) -> sigma^(n+1)
        update_stress_fused_free_surface_numba(
            vx, vz, sxx, szz, sxz,
            lam, l2m, mu_xz,
            h, h, dt, a,
        )

        mirror_stress_ghosts_free_surface_numba(
            sxx, szz, sxz, M
        )

        if wave == "P":
            traction = abs(float(szz[i0, M]))
        else:
            # sigma_xz is half-grid staggered about the surface.  The
            # surface-centered traction is the average of the two values.
            traction = abs(
                0.5
                * float(
                    sxz[i0, M - 1]
                    + sxz[i0, M]
                )
            )
        max_traction = max(max_traction, traction)

    # Exact normally incident free-surface solution:
    #
    # incident center: z_c(t) = packet_depth - c t
    # incident value at z=0: G(-packet_depth + c t)
    # reflected particle velocity has R_v = +1, so v_surface = 2*v_incident.
    analytical = (
        2.0
        * amplitude
        * gaussian(
            -packet_depth_m + c * t,
            packet_sigma_m,
        )
    )

    # Compare only over the physically relevant arrival window; this avoids
    # making the relative error sensitive to floating-point noise where both
    # traces are effectively zero.
    threshold = 1.0e-5 * float(np.max(np.abs(analytical)))
    mask = np.abs(analytical) >= threshold

    relative_l2 = float(
        np.linalg.norm(numerical[mask] - analytical[mask])
        / np.linalg.norm(analytical[mask])
    )

    peak_num = float(np.max(np.abs(numerical[mask])))
    peak_ref = float(np.max(np.abs(analytical[mask])))
    peak_ratio = peak_num / peak_ref

    i_num = int(np.argmax(np.abs(numerical)))
    i_ref = int(np.argmax(np.abs(analytical)))
    peak_time_error_s = float(t[i_num] - t[i_ref])

    return CaseResult(
        wave=wave,
        h=h,
        dt=dt,
        t=t,
        numerical=numerical,
        analytical=analytical,
        relative_l2=relative_l2,
        peak_ratio=peak_ratio,
        peak_time_error_s=peak_time_error_s,
        max_surface_traction_pa=max_traction,
    )


def _print_table(results: list[CaseResult]) -> None:
    print(
        "\n"
        "wave   h(m)       dt(s)       rel_L2      peak_ratio   "
        "peak_dt(s)    max_surface_traction(Pa)"
    )
    print("-" * 92)

    for r in results:
        print(
            f"{r.wave:>3s} "
            f"{r.h:7.2f} "
            f"{r.dt:12.5e} "
            f"{r.relative_l2:12.5e} "
            f"{r.peak_ratio:12.6f} "
            f"{r.peak_time_error_s:12.5e} "
            f"{r.max_surface_traction_pa:18.5e}"
        )


def _print_convergence(
    results: list[CaseResult],
    wave: str,
) -> None:
    selected = sorted(
        [r for r in results if r.wave == wave],
        key=lambda r: r.h,
        reverse=True,
    )

    print(f"\n{wave}-wave refinement:")
    for coarse, fine in zip(selected[:-1], selected[1:]):
        if (
            coarse.relative_l2 > 0.0
            and fine.relative_l2 > 0.0
            and coarse.h > fine.h
        ):
            order = (
                np.log(coarse.relative_l2 / fine.relative_l2)
                / np.log(coarse.h / fine.h)
            )
        else:
            order = np.nan

        print(
            f"  h {coarse.h:g} -> {fine.h:g} m: "
            f"error {coarse.relative_l2:.4e} -> {fine.relative_l2:.4e}, "
            f"observed p = {order:.3f}"
        )


def _plot_finest(
    results: list[CaseResult],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for wave in ("P", "SV"):
        r = min(
            (item for item in results if item.wave == wave),
            key=lambda item: item.h,
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(
            r.t,
            r.analytical,
            label="analytical free surface",
        )
        ax.plot(
            r.t,
            r.numerical,
            "--",
            label=f"Graves FS1 FD, h={r.h:g} m",
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Particle velocity (arbitrary amplitude)")
        ax.set_title(
            f"Normal-incidence {wave} free-surface reflection"
        )
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()

        path = out_dir / f"normal_incidence_{wave.lower()}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)

        print(f"Saved {path}")


def main() -> None:
    vp = 5000.0
    vs = 3000.0
    rho = 2600.0

    # Keep the physical experiment identical at every resolution.
    packet_depth_m = 150.0
    packet_sigma_m = 25.0
    amplitude = 1.0

    # P reaches the surface at 0.03 s and SV at 0.05 s.  The 0.08 s window is
    # long enough to capture both reflected peaks but short enough that neither
    # bottom reflections nor untouched lateral-edge columns can reach the
    # central receiver.
    duration_s = 0.08
    depth_m = 450.0
    width_m = 1000.0

    grid_spacings = [10.0, 5.0, 2.5]

    results: list[CaseResult] = []

    print("Running homogeneous half-space normal-incidence benchmark...")
    print(
        f"Vp={vp:g} m/s, Vs={vs:g} m/s, rho={rho:g} kg/m^3, "
        f"packet depth={packet_depth_m:g} m, width={width_m:g} m"
    )

    for wave in ("P", "SV"):
        for h in grid_spacings:
            print(f"  {wave}: h={h:g} m")
            results.append(
                _run_case(
                    wave=wave,
                    h=h,
                    vp=vp,
                    vs=vs,
                    rho0=rho,
                    packet_depth_m=packet_depth_m,
                    packet_sigma_m=packet_sigma_m,
                    amplitude=amplitude,
                    duration_s=duration_s,
                    depth_m=depth_m,
                    width_m=width_m,
                )
            )

    _print_table(results)
    _print_convergence(results, "P")
    _print_convergence(results, "SV")

    _plot_finest(
        results,
        Path(
            "results/validation/elastic/free_surface_normal_incidence"
        ),
    )

    print(
        "\nInterpretation:"
        "\n  * peak_ratio should approach 1 as h decreases;"
        "\n  * peak_dt should approach 0;"
        "\n  * relative L2 error should decrease under refinement;"
        "\n  * free-surface traction should stay at roundoff / explicit zero."
        "\n"
        "\nIf these trends hold for both P and SV, the next validation is an"
        "\noblique/point-source half-space test, which exercises P-SV conversion"
        "\nand Rayleigh-wave generation."
    )


if __name__ == "__main__":
    main()
