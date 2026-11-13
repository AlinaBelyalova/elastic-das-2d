"""Independent localized P/SV packet tests for the restored Gaussian sponge.

A larger SAME-MESH elastic benchmark excludes its distant boundary returns.
It is a validation only, not an added production domain or inner display.
Finite angular/frequency spread; error norm is NOT an exact reflection coefficient.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from src.sbp_sat.elastic2d import SBPSATState
from src.sbp_sat.numba_solver import NumbaSBPSAT
from src.sbp_sat.rectangular import prepare_rectangular_model
from src.sbp_sat.sponge import GaussianSponge, NumbaSpongeSBPSAT

PHYSICAL = ("vx", "vz", "sxx", "szz", "sxz")


def fourier_packet(
    kind, angle_deg, dt, *, face="bottom", wavelength=16.0, sigma=8.0, centre=None
):
    """Pure continuum P/SV plane-wave superposition on a large FFT lattice.

    sigma is the Gaussian spatial-envelope scale. Real parts of the analytic
    positive-frequency solution are sampled at each native staggered position
    and time: v(-dt/2), sigma(0). Only a cropped, localized initial packet is
    subsequently evolved by finite differences, not a periodic wave solver.
    """
    if kind not in ("P", "SV"):
        raise ValueError("Use P or SV.")
    if face not in ("left", "right", "bottom"):
        raise ValueError("Choose an artificial face.")
    if centre is None:
        centre = (
            (400.0, 35.0)
            if face == "bottom"
            else ((35.0, 400.0) if face == "right" else (65.0, 400.0))
        )
    n, h, fft_origin = 2048, 1.0, -512.0
    kx = (2 * np.pi * np.fft.fftfreq(n, h))[:, None]
    kz = (2 * np.pi * np.fft.fftfreq(n, h))[None, :]
    kn = np.hypot(kx, kz)
    safe_kn = np.where(kn == 0, np.inf, kn)
    angle = np.deg2rad(angle_deg)
    k0 = 2 * np.pi / wavelength
    direction = (
        (np.sin(angle), np.cos(angle))
        if face == "bottom"
        else ((1 if face == "right" else -1) * np.cos(angle), np.sin(angle))
    )
    ax, az = (
        (kx / safe_kn, kz / safe_kn) if kind == "P" else (kz / safe_kn, -kx / safe_kn)
    )
    speed, rho, vp, vs = 3.0 if kind == "P" else 1.5, 2.0, 3.0, 1.5
    mu, lam = rho * vs**2, rho * (vp**2 - 2 * vs**2)
    amplitude = np.exp(
        -(sigma**2)
        * ((kx - k0 * direction[0]) ** 2 + (kz - k0 * direction[1]) ** 2)
        / 2
    ).astype(complex)
    amplitude *= np.exp(
        -1j * (kx * (centre[0] - fft_origin) + kz * (centre[1] - fft_origin))
    )
    amplitude /= np.max(np.abs(np.fft.ifft2(amplitude)))
    omega = speed * kn
    safe_omega = np.where(omega == 0, np.inf, omega)
    divergence = kx * ax + kz * az
    spectra = [
        ax * np.exp(1j * (kx * h / 2 + omega * dt / 2)),
        az * np.exp(1j * (kz * h / 2 + omega * dt / 2)),
        -(lam * divergence + 2 * mu * kx * ax) / safe_omega,
        -(lam * divergence + 2 * mu * kz * az) / safe_omega,
        -mu * (kz * ax + kx * az) / safe_omega * np.exp(1j * (kx + kz) * h / 2),
    ]
    values = [np.fft.ifft2(amplitude * spec).real.copy() for spec in spectra]

    def sample(solver, x0=0.0, z0=0.0):
        out = solver.zero_state()
        ix, iz = round(x0 - fft_origin), round(z0 - fft_origin)
        for name, full in zip(PHYSICAL, values):
            target = getattr(out, name)
            if ix < 0 or iz < 0 or ix + target.shape[0] > n or iz + target.shape[1] > n:
                raise ValueError("FFT packet lattice does not cover benchmark grid.")
            target[:] = full[ix : ix + target.shape[0], iz : iz + target.shape[1]]
        return out

    return sample


def norm_energy(bulk, state, masks=None):
    """Positive instantaneous elastic norm; NOT a PML Lyapunov certificate."""
    m = bulk.model
    masks = [1.0, 1.0, 1.0, 1.0] if masks is None else masks
    normal = (
        m.l2m_NN * (state.sxx**2 + state.szz**2) - 2 * m.lam_NN * state.sxx * state.szz
    ) / (m.l2m_NN**2 - m.lam_NN**2)
    return 0.5 * float(
        np.sum(bulk.wx * m.rho_vx * state.vx**2 * masks[0])
        + np.sum(bulk.wz * m.rho_vz * state.vz**2 * masks[1])
        + np.sum(bulk.wn * normal * masks[2])
        + np.sum(bulk.ws * state.sxz**2 / m.mu_MM * masks[3])
    )


def subtract_crop(state, reference, extension):
    values = []
    for name in PHYSICAL:
        a, b = getattr(state, name), getattr(reference, name)
        values.append(a - b[extension : extension + a.shape[0], : a.shape[1]])
    return SBPSATState(*values)


def packet_case(wave="P", face="bottom", angle=75.0, *, threads=4):
    start = perf_counter()
    dt, extension = 0.05, 480
    shape = (801, 101) if face == "bottom" else (101, 801)

    def medium(shape):
        return prepare_rectangular_model(
            np.full(shape, 3.0), np.full(shape, 1.5), np.full(shape, 2.0)
        )

    sponge = GaussianSponge(30.0, 30.0, 0.96)
    f = NumbaSpongeSBPSAT(medium(shape), 1, 1, dt, sponge=sponge, numba_threads=threads)
    old = NumbaSBPSAT(f.model, 1, 1, dt, numba_threads=threads)
    ref = NumbaSBPSAT(
        medium((shape[0] + 2 * extension, shape[1] + extension)),
        1,
        1,
        dt,
        numba_threads=threads,
    )
    packet = fourier_packet(wave, angle, dt, face=face)
    state, control, reference = packet(f), packet(old), packet(ref, -extension, 0.0)
    mx, mz, mn, _, ms = f.sponge_masks
    masks = tuple(m == 1 for m in (mx, mz, mn, ms))
    initial = norm_energy(f, state)
    initial_layer = 1 - norm_energy(f, state, masks) / initial
    nt = 2800 if wave == "P" else 5200
    peak = peak_old = 0.0
    history = []
    with f.thread_context():
        for it in range(nt):
            state = f.advance_inplace(state, None, 0.0)
            control = old.advance_inplace(control, None, 0.0)
            reference = ref.advance_inplace(reference, None, 0.0)
            if (it + 1) % 10 == 0:
                error = (
                    norm_energy(f, subtract_crop(state, reference, extension), masks)
                    / initial
                )
                old_error = (
                    norm_energy(f, subtract_crop(control, reference, extension), masks)
                    / initial
                )
                peak, peak_old = max(peak, error), max(peak_old, old_error)
                history.append(((it + 1) * dt, error, old_error))
                if not np.isfinite(error) or error > 10:
                    raise FloatingPointError("Nonfinite/growing sponge packet.")
    return dict(
        wave=wave,
        face=face,
        central_angle_from_face_normal_deg=angle,
        maximum_error_energy_fraction=peak,
        impedance_only_error_energy_fraction=peak_old,
        initial_energy_in_strips_fraction=initial_layer,
        cells=list(shape),
        spacing_m=1.0,
        dt_s=dt,
        nt=nt,
        sponge_width_m=30.0,
        rate_per_s=0.96,
        normalization="one common initial elastic energy",
        benchmark_extension_m=extension,
        scope="manufactured medium and packet band; not a SAFOD event reflection coefficient",
        elapsed_s=perf_counter() - start,
        history=history,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wave", choices=("P", "SV"), default="P")
    p.add_argument("--face", choices=("left", "right", "bottom"), default="bottom")
    p.add_argument("--angle", type=float, choices=(0.0, 30.0, 60.0, 75.0), default=75.0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError(
            "Choose a new validation directory; earlier results are preserved."
        )
    report = packet_case(a.wave, a.face, a.angle, threads=a.threads)
    # Report the actual residual; do not turn an improvement check into a
    # promise of perfect transparency or silently relax a PML target.
    report["improves_impedance_control"] = (
        report["maximum_error_energy_fraction"]
        < report["impedance_only_error_energy_fraction"]
    )
    a.output_dir.mkdir(parents=True)
    (a.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))
    if not report["improves_impedance_control"]:
        raise AssertionError(
            "Sponge did not improve this packet; inspect saved measurements."
        )


if __name__ == "__main__":
    main()
