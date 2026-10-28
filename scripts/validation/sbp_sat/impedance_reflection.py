"""Continuum reflection of the CURRENT local impedance boundary law.

This independent 2x2 plane-wave calculation explains the boundary-law error;
it is not a simulation of a SAFOD event or a discrete PML benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def reflection(kind, angle_deg, *, vp=5000.0, vs=3000.0, rho=2600.0):
    """Bottom n=+z; angle from the outward normal. Rotationally same at sides.

    v = a exp(i*omega*(p*x+q*z-t)); sigma = -C:sym(s tensor a).
    Reflected modes have negative real q, or negative imaginary q for an
    evanescent mode decaying into the interior z<0. Traction + Z*v = 0,
    Z=diag(rho*Vs,rho*Vp), in (tangential, normal) order.
    """
    if kind not in ("P", "SV") or not np.isfinite(angle_deg) or not 0 <= angle_deg < 90:
        raise ValueError(
            "Use P/SV and an angle from 0 up to, but excluding, 90 degrees."
        )
    if (
        not np.isfinite([vp, vs, rho]).all()
        or vp <= 0
        or rho <= 0
        or vs <= 0
        or vp**2 < 2 * vs**2
    ):
        raise ValueError("Use finite positive material with lambda >= 0.")
    lam, mu = rho * (vp**2 - 2 * vs**2), rho * vs**2
    speed = vp if kind == "P" else vs
    p = np.sin(np.deg2rad(angle_deg)) / speed

    def mode(wave, sign):
        c = vp if wave == "P" else vs
        q = sign * np.sqrt(complex(1 / c**2 - p**2))
        a = c * (np.array([p, q]) if wave == "P" else np.array([q, -p]))
        stress_zz = -lam * (p * a[0] + q * a[1]) - 2 * mu * q * a[1]
        stress_xz = -mu * (q * a[0] + p * a[1])
        traction = np.array([stress_xz, stress_zz])
        residual = traction + rho * np.array([vs, vp]) * a
        return c, q, a, residual

    inc = mode(kind, 1)
    reflected = [mode(wave, -1) for wave in ("P", "SV")]
    matrix = np.column_stack([wave[3] for wave in reflected])
    coefficients = np.linalg.solve(matrix, -inc[3])
    flux_in = 0.5 * rho * speed**2 * inc[1].real
    flux_reflected = np.array(
        [
            0.5 * rho * wave[0] ** 2 * (-wave[1].real) * abs(a) ** 2
            for wave, a in zip(reflected, coefficients)
        ]
    )
    total_velocity = inc[2] + sum(
        a * wave[2] for a, wave in zip(coefficients, reflected)
    )
    flux_absorbed = 0.5 * rho * np.sum(np.array([vs, vp]) * abs(total_velocity) ** 2)
    balance = float(abs(flux_in - flux_reflected.sum() - flux_absorbed) / flux_in)
    residual = float(np.linalg.norm(matrix @ coefficients + inc[3]) / (rho * speed))
    return {
        "wave": kind,
        "angle_from_normal_deg": float(angle_deg),
        "reflected_energy_fraction": float(flux_reflected.sum() / flux_in),
        "reflected_P_energy_fraction": float(flux_reflected[0] / flux_in),
        "reflected_SV_energy_fraction": float(flux_reflected[1] / flux_in),
        "absorbed_energy_fraction": float(flux_absorbed / flux_in),
        "boundary_residual_relative": residual,
        "flux_balance_relative": balance,
    }


def run_checks():
    cases = [
        reflection(wave, angle)
        for wave in ("P", "SV")
        for angle in (0, 15, 30, 45, 60, 75)
    ]
    passed = all(
        0 <= c["reflected_energy_fraction"] <= 1 + 1e-12
        and c["boundary_residual_relative"] < 1e-12
        and c["flux_balance_relative"] < 1e-12
        for c in cases
    )
    passed = passed and all(
        c["reflected_energy_fraction"] < 1e-28
        for c in cases
        if c["angle_from_normal_deg"] == 0
    )
    report = {
        "passed": passed,
        "calculation": "continuum plane waves at local impedance boundary",
        "material": {"vp_m_s": 5000, "vs_m_s": 3000, "rho_kg_m3": 2600},
        "boundary": "traction + diag(rho*Vs,rho*Vp) velocity = 0",
        "angles": "degrees from normal; postcritical reflected P may be evanescent",
        "cases": cases,
        "scope": "Boundary-law error only. Not the measured reflection of the SAFOD event or a discrete PML.",
    }
    if not passed:
        raise AssertionError(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if (
        args.output_dir
        and args.output_dir.exists()
        and any(args.output_dir.iterdir())
        and not args.overwrite
    ):
        raise FileExistsError(
            "Choose an empty output directory or use --overwrite intentionally."
        )
    report = run_checks()
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
