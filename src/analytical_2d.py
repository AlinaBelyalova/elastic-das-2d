# ==============================================================================
# src/analytical_2d.py — helpers for 2D analytical validation
#
# Purpose
#   Infrastructure for analytical comparison of the 2D elastic point-force
#   validation problem.
#
# Current status
#   This file provides:
#     - geometry helpers
#     - FFT / IFFT helpers
#     - source spectrum construction
#     - time-domain reconstruction
#     - a working candidate implementation of the 2D analytical Green-function
#       kernel for v_z due to a vertical point force
#
# Important
#   The Green-function kernel below is implemented as a practical analytical
#   reference candidate for the current validation workflow. If later the
#   convergence plot shows unexpected behaviour, the first thing to re-check is
#   the exact analytical kernel: prefactors, signs, and the near-field term.
#
# Time convention
#   The numerical point-force solver records velocity on the leapfrog half-step
#   time axis:
#
#       t_v[n] = (n + 1/2) * dt
#
#   If return_half_step_times=True, this module does NOT merely relabel the time
#   axis. Instead, it physically advances the analytical trace by dt/2 in the
#   frequency domain using the Fourier shift theorem:
#
#       V(ω) -> V(ω) * exp(+i ω dt / 2)
#
#   Sign convention: NumPy irfft computes x[n] ∝ Σ_k X[k] exp(+2πi kn/N),
#   so a forward time shift of τ corresponds to multiplying by exp(+iωτ).
#   This is consistent with NumPy's inverse FFT convention and avoids an
#   artificial O(dt) phase error when comparing against leapfrog velocity data.
#
# Source spectrum scaling
#   The continuous Fourier transform is approximated as
#       S(ω) ≈ dt * Σ_n s(n*dt) exp(-iω n*dt) = dt * rfft(s)
#   The dt factor is included in build_source_spectrum so that absolute
#   amplitudes are consistent with the Green-function prefactors.
# ==============================================================================

from __future__ import annotations

import numpy as np
import scipy.special as sp


# ==============================================================================
# 1. BASIC GEOMETRY HELPERS
# ==============================================================================

def receiver_offset(
    *,
    x_src: float,
    z_src: float,
    x_rec: float,
    z_rec: float,
) -> tuple[float, float, float]:
    """Return (dx, dz, r) between source and receiver."""
    dx = float(x_rec - x_src)
    dz = float(z_rec - z_src)
    r  = float(np.sqrt(dx * dx + dz * dz))
    return dx, dz, r


def direction_cosines_2d(
    *,
    x_src: float,
    z_src: float,
    x_rec: float,
    z_rec: float,
    eps: float = 1e-30,
) -> tuple[float, float, float]:
    """
    Return direction cosines (gamma_x, gamma_z) and distance r.

    gamma_x = (x_rec - x_src) / r
    gamma_z = (z_rec - z_src) / r
    """
    dx, dz, r = receiver_offset(
        x_src=x_src, z_src=z_src, x_rec=x_rec, z_rec=z_rec
    )
    if r <= eps:
        raise ValueError(
            "Source and receiver coincide; direction cosines are undefined."
        )
    return float(dx / r), float(dz / r), float(r)


# ==============================================================================
# 2. FFT HELPERS
# ==============================================================================

def next_pow2(n: int) -> int:
    """Return the smallest power of two >= n."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}.")
    return 1 << (int(n - 1).bit_length())


def choose_nfft(nt: int, pad_factor: int = 2) -> int:
    """
    Choose FFT length by zero-padding to at least pad_factor*nt and rounding
    up to the next power of two.
    """
    if nt <= 1:
        raise ValueError(f"nt must be > 1, got {nt}.")
    if pad_factor < 1:
        raise ValueError(f"pad_factor must be >= 1, got {pad_factor}.")
    return next_pow2(pad_factor * nt)


def rfft_frequency_axis(
    *,
    dt: float,
    nfft: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (freq_hz, omega_rad_s) for the rFFT convention."""
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}.")
    if nfft <= 1:
        raise ValueError(f"nfft must be > 1, got {nfft}.")

    freq_hz = np.fft.rfftfreq(nfft, d=dt)
    omega   = 2.0 * np.pi * freq_hz
    return np.asarray(freq_hz, dtype=np.float64), np.asarray(omega, dtype=np.float64)


# ==============================================================================
# 3. SOURCE SPECTRUM
# ==============================================================================

def build_source_spectrum(
    *,
    stf: np.ndarray,
    dt: float,
    pad_factor: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Build the source spectrum using rFFT with correct dt scaling.

    The continuous Fourier transform is approximated as
        S(ω) ≈ dt * Σ_n s(n*dt) exp(-iω n*dt) = dt * rfft(s)
    The dt factor is included here so that absolute amplitudes match the
    Green-function prefactors without additional normalisation at call sites.

    Parameters
    ----------
    stf :
        Discrete source time function, shape (nt,), sampled at n*dt.
    dt :
        Time step.
    pad_factor :
        Zero-padding factor before rounding to next power of two.

    Returns
    -------
    src_fft :
        rFFT of the padded source array, scaled by dt.
    freq_hz :
        Frequency axis in Hz.
    omega :
        Angular frequency axis in rad/s.
    nfft :
        FFT size used.
    """
    stf = np.asarray(stf, dtype=np.float64)

    if stf.ndim != 1:
        raise ValueError(f"stf must be 1D, got shape {stf.shape}.")
    if stf.size <= 1:
        raise ValueError("stf must contain at least two samples.")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}.")

    nt   = stf.size
    nfft = choose_nfft(nt, pad_factor=pad_factor)

    # dt factor approximates the continuous Fourier integral ∫ s(t) e^{-iωt} dt
    src_fft = np.fft.rfft(stf, n=nfft) * dt

    freq_hz, omega = rfft_frequency_axis(dt=dt, nfft=nfft)

    return (
        np.asarray(src_fft, dtype=np.complex128),
        freq_hz,
        omega,
        nfft,
    )


# ==============================================================================
# 4. TIME-DOMAIN RECONSTRUCTION
# ==============================================================================

def irfft_to_time(
    *,
    spectrum: np.ndarray,
    nfft: int,
    nt: int,
) -> np.ndarray:
    """Return the first nt samples of the inverse rFFT."""
    spectrum = np.asarray(spectrum, dtype=np.complex128)

    if spectrum.ndim != 1:
        raise ValueError(f"spectrum must be 1D, got shape {spectrum.shape}.")
    if nfft <= 1:
        raise ValueError(f"nfft must be > 1, got {nfft}.")
    if nt <= 1:
        raise ValueError(f"nt must be > 1, got {nt}.")

    return np.asarray(np.fft.irfft(spectrum, n=nfft)[:nt], dtype=np.float64)


# ==============================================================================
# 5. 2D ANALYTICAL GREEN FUNCTION (CANDIDATE IMPLEMENTATION)
# ==============================================================================

def green_2d_pointforce_velocity_zz(
    *,
    omega: np.ndarray,
    r: float,
    gamma_x: float,
    gamma_z: float,
    rho: float,
    vp: float,
    vs: float,
) -> np.ndarray:
    """
    Frequency-domain 2D Green-function kernel for v_z at the receiver due to a
    vertical point force in a homogeneous elastic medium (plane strain).

    The displacement kernel is (Kausel 2006):

        G_zz^u(r, ω) = (-i / 4ρ) [
            γ_z² / vp²  * H_0^(2)(kp r)
          + (1 - γ_z²) / vs²  * H_0^(2)(ks r)
          + (2γ_z² - 1) / (ω² r) * (ks H_1^(2)(ks r) - kp H_1^(2)(kp r))
        ]

    Velocity kernel: G_zz^v = iω G_zz^u.

    FFT convention
    --------------
    NumPy rfft computes X[k] = Σ_n x[n] exp(-2πi kn/N), corresponding to the
    e^{-iωt} time convention. Hankel functions of the 2nd kind are the correct
    outgoing-wave solutions for this convention.

    Parameters
    ----------
    omega :
        Angular frequency axis in rad/s, shape (nfft//2+1,).
    r :
        Source-receiver distance.
    gamma_x :
        Direction cosine in x. Accepted for API consistency with other
        Green-function components; not used in G_zz for an isotropic medium.
    gamma_z :
        Direction cosine in z.
    rho, vp, vs :
        Homogeneous medium properties.

    Notes
    -----
    - The ω=0 component is explicitly zeroed to avoid singular behaviour.
    - The near-field term ~1/(ω²r) may be large at low frequencies; the ω=0
      mask protects the singularity but behaviour near ω≈0 should be monitored.
    - This is a candidate implementation; prefactors/signs should be verified
      against the final reference derivation used for the project.
    """
    omega = np.asarray(omega, dtype=np.float64)

    if omega.ndim != 1:
        raise ValueError(f"omega must be 1D, got shape {omega.shape}.")
    if r <= 0.0:
        raise ValueError(f"r must be positive, got {r}.")
    if rho <= 0.0:
        raise ValueError(f"rho must be positive, got {rho}.")
    if vp <= 0.0 or vs <= 0.0:
        raise ValueError(f"vp and vs must be positive, got vp={vp}, vs={vs}.")

    # gamma_x: accepted for API completeness, not used in G_zz (isotropic medium)
    _ = gamma_x

    gzz_v = np.zeros(omega.shape, dtype=np.complex128)

    mask = omega > 0.0
    if not np.any(mask):
        return gzz_v

    w  = omega[mask]
    kp = w / vp
    ks = w / vs

    # Hankel functions of the 2nd kind (outgoing waves, e^{-iωt} convention)
    H0_p = sp.hankel2(0, kp * r)
    H0_s = sp.hankel2(0, ks * r)
    H1_p = sp.hankel2(1, kp * r)
    H1_s = sp.hankel2(1, ks * r)

    # Displacement Green-function terms for G_zz (Kausel 2006)
    far_p = (gamma_z**2       / vp**2) * H0_p
    far_s = ((1.0 - gamma_z**2) / vs**2) * H0_s
    near  = ((2.0 * gamma_z**2 - 1.0) / (w**2 * r)) * (ks * H1_s - kp * H1_p)

    Gzz_u = (-1j / (4.0 * rho)) * (far_p + far_s + near)

    # Displacement → velocity: v̂(ω) = iω û(ω)
    gzz_v[mask] = Gzz_u * (1j * w)

    return gzz_v


# ==============================================================================
# 6. FULL ANALYTICAL TRACE PIPELINE
# ==============================================================================

def analytical_vz_trace_from_pointforce(
    *,
    stf: np.ndarray,
    dt: float,
    x_src: float,
    z_src: float,
    x_rec: float,
    z_rec: float,
    rho: float,
    vp: float,
    vs: float,
    pad_factor: int = 2,
    return_half_step_times: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the analytical v_z(t) trace from source spectrum * Green function.

    Parameters
    ----------
    stf :
        Discrete source time function sampled on the integer time grid n*dt.
    dt :
        Time step.
    x_src, z_src :
        Physical source coordinates.
    x_rec, z_rec :
        Physical receiver coordinates.
    rho, vp, vs :
        Homogeneous medium properties.
    pad_factor :
        Zero-padding factor for FFT.
    return_half_step_times :
        If True, return the velocity half-step time axis
            t_v[n] = (n + 1/2) * dt
        and physically advance the analytical trace by dt/2 in the frequency
        domain using the Fourier shift theorem:
            V(ω) -> V(ω) * exp(+iω dt/2)
        Sign: NumPy irfft computes x[n] ∝ Σ_k X[k] exp(+2πi kn/N), so a
        forward time shift by τ corresponds to multiplying by exp(+iωτ).
        If False, return the standard integer axis t[n] = n*dt with no shift.

    Returns
    -------
    t :
        Time axis, shape (nt,).
    v_analytic :
        Analytical v_z velocity trace, shape (nt,).
    """
    stf = np.asarray(stf, dtype=np.float64)

    if stf.ndim != 1:
        raise ValueError(f"stf must be 1D, got shape {stf.shape}.")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}.")
    if rho <= 0.0 or vp <= 0.0 or vs <= 0.0:
        raise ValueError(
            f"rho, vp, vs must be positive, got rho={rho}, vp={vp}, vs={vs}."
        )

    nt = stf.size

    gamma_x, gamma_z, r = direction_cosines_2d(
        x_src=x_src, z_src=z_src, x_rec=x_rec, z_rec=z_rec
    )

    src_fft, _freq_hz, omega, nfft = build_source_spectrum(
        stf=stf, dt=dt, pad_factor=pad_factor
    )

    gzz_v = green_2d_pointforce_velocity_zz(
        omega=omega,
        r=r,
        gamma_x=gamma_x,
        gamma_z=gamma_z,
        rho=rho,
        vp=vp,
        vs=vs,
    )

    if gzz_v.shape != omega.shape:
        raise ValueError(
            f"Green-function kernel must return shape {omega.shape}, "
            f"got {gzz_v.shape}."
        )

    v_omega = src_fft * gzz_v

    if return_half_step_times:
        # Forward time shift by dt/2: x(t + dt/2) <-> X(ω) * exp(+iω dt/2)
        # consistent with NumPy irfft convention exp(+2πi kn/N)
        v_omega = v_omega * np.exp(1j * omega * (dt / 2.0))
        t = (np.arange(nt, dtype=np.float64) + 0.5) * dt
    else:
        t = np.arange(nt, dtype=np.float64) * dt

    v_analytic = irfft_to_time(spectrum=v_omega, nfft=nfft, nt=nt)

    v_analytic /= dt

    return t, v_analytic