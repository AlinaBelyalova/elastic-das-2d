"""A sufficient (conservative) leapfrog timestep certificate for this core.

Only periodic x, traction-free z, fixed positive material and no damping.
This is not the interior-stencil CFL formula, nor a dispersion guarantee.
See docs/sbp_sat/method.md for the norm bound and positivity argument.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from .operators import StaggeredSBP1D

if TYPE_CHECKING:
    from .elastic2d import SBPSATModel2D


def _weighted_derivative_bound(D, H_out, H_in) -> float:
    # ||K||_2 <= sqrt(||K||_1 ||K||_inf), K=H_out^1/2 D H_in^-1/2.
    # Uses existing 1-D matrices only; no 2-D matrix assembly or eigensolve.
    K = np.abs(D) * np.sqrt(H_out[:, None] / H_in[None, :])
    return float(np.sqrt(K.sum(axis=0).max() * K.sum(axis=1).max()))


def stable_timestep_bound(
    model: SBPSATModel2D,
    op_z: StaggeredSBP1D,
    dx: float,
) -> float:
    """Return dt_limit such that 0 < dt < dt_limit is sufficient for stability.

    rho_min bounds BOTH native velocity densities below. C_max bounds the
    normal stiffness eigenvalues and twice the averaged shear modulus above.
    The strict inequality ensures positivity of the leapfrog invariant.
    """
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("dx must be finite and positive.")
    if op_z.n_N != model.nz_N or op_z.n_M != model.nz_M:
        raise ValueError("SBP operator and model dimensions do not match.")
    rho_min = min(float(model.rho_vx.min()), float(model.rho_vz.min()))
    C_max = max(
        float((model.l2m_NN + model.lam_NN).max()),
        float((model.l2m_NN - model.lam_NN).max()),
        2.0 * float(model.mu_MM.max()),
    )
    kz = max(
        _weighted_derivative_bound(op_z.D_M_to_N, op_z.H_N, op_z.H_M),
        _weighted_derivative_bound(op_z.D_N_to_M, op_z.H_M, op_z.H_N),
    )
    kx = 7.0 / (3.0 * dx)  # l1 norm of the periodic fourth-order stencil
    omega_bound = np.sqrt(C_max / rho_min) * np.hypot(kx, kz)
    if not np.isfinite(omega_bound) or omega_bound <= 0:
        raise ValueError("Cannot certify a finite timestep for this material/grid.")
    return float(2.0 / omega_bound)


def require_stable_timestep(model, op_z, dx: float, dt: float) -> None:
    """Reject an uncertified step; never silently reduce dt or alter nt."""
    limit = stable_timestep_bound(model, op_z, dx)
    if not np.isfinite(dt) or not 0 < dt < limit:
        raise ValueError(
            f"dt={dt!r} is outside the certified range 0 < dt < {limit:.12g} s. "
            "This is a sufficient bound, not the sharp CFL limit."
        )
