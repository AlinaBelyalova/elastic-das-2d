"""
Algebraic/timing QC for the planar Graves (1996) FS1 free surface.

This test deliberately isolates the boundary operator. It verifies that:

1) the Graves second-order velocity boundary equations are satisfied after
   velocity continuation;
2) filling the velocity ghosts before the velocity update leaves stale
   half-time-level ghosts and violates those equations;
3) refilling after the velocity update restores the equations to roundoff;
4) the FS1 stress image equations are imposed exactly.

A physical half-space benchmark is still required after this algebraic QC.
"""

from __future__ import annotations

import numpy as np

from src.elastic.solver_numpy import (
    c_sfd_coefficients,
    prepare_staggered_materials,
)
from src.elastic.solver_numba_fused import (
    fill_velocity_ghosts_free_surface_numba,
    mirror_stress_ghosts_free_surface_numba,
    update_velocity_fused_numba,
    update_stress_fused_free_surface_numba,
)


def _velocity_bc_residuals(
    vx: np.ndarray,
    vz: np.ndarray,
    lam: np.ndarray,
    l2m: np.ndarray,
    dx: float,
    dz: float,
    M: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the two 2-D Graves FS1 velocity-boundary residuals."""
    k = M
    nx = vx.shape[0]

    # Dz(vz)^k + lambda/(lambda+2mu) Dx(vx)^k = 0
    dz_vz = (
        vz[M:nx - M + 1, k]
        - vz[M:nx - M + 1, k - 1]
    ) / dz
    dx_vx = (
        vx[M:nx - M + 1, k]
        - vx[M - 1:nx - M, k]
    ) / dx
    normal = (
        dz_vz
        + (lam[M:nx - M + 1, k] / l2m[M:nx - M + 1, k])
        * dx_vx
    )

    # [Dz(vx)+Dx(vz)]^(k-1/2)
    #   + [Dz(vx)+Dx(vz)]^(k+1/2) = 0
    dz_vx_above = (
        vx[M:nx - M, k] - vx[M:nx - M, k - 1]
    ) / dz
    dx_vz_above = (
        vz[M + 1:nx - M + 1, k - 1]
        - vz[M:nx - M, k - 1]
    ) / dx

    dz_vx_below = (
        vx[M:nx - M, k + 1] - vx[M:nx - M, k]
    ) / dz
    dx_vz_below = (
        vz[M + 1:nx - M + 1, k]
        - vz[M:nx - M, k]
    ) / dx

    shear = (
        dz_vx_above
        + dx_vz_above
        + dz_vx_below
        + dx_vz_below
    )

    return normal, shear


def _maxabs(arr: np.ndarray) -> float:
    return float(np.max(np.abs(arr)))


def _report(label: str, arr: np.ndarray) -> float:
    value = _maxabs(arr)
    print(f"{label:<48s}: {value:.6e}")
    return value


def main() -> None:
    rng = np.random.default_rng(20261001)

    nx = 80
    nz = 40
    dx = dz = 5.0
    dt = 2.0e-4
    M = 2
    a = c_sfd_coefficients(M)

    rho = np.full((nx, nz), 2600.0)
    vp = np.full((nx, nz), 5000.0)
    vs = np.full((nx, nz), 3000.0)

    mu = rho * vs**2
    lam = rho * (vp**2 - 2.0 * vs**2)
    l2m = lam + 2.0 * mu
    bx, bz, mu_xz = prepare_staggered_materials(rho, mu)

    # A. Direct algebraic check of the two velocity boundary equations.
    vx = 1.0e-5 * rng.standard_normal((nx, nz))
    vz = 1.0e-5 * rng.standard_normal((nx, nz))

    fill_velocity_ghosts_free_surface_numba(
        vx, vz, lam, l2m, dx, dz, M
    )

    r_normal, r_shear = _velocity_bc_residuals(
        vx, vz, lam, l2m, dx, dz, M
    )

    print("\nA. Graves FS1 velocity boundary equations")
    a_normal = _report("normal relation max |residual|", r_normal)
    a_shear = _report("shear relation max |residual|", r_shear)

    # B. Reproduce the old timing: build ghosts at n-1/2, then change the
    # interior velocity to n+1/2. The ghost values are now stale.
    vx = 1.0e-5 * rng.standard_normal((nx, nz))
    vz = 1.0e-5 * rng.standard_normal((nx, nz))
    sxx = 1.0e4 * rng.standard_normal((nx, nz))
    szz = 1.0e4 * rng.standard_normal((nx, nz))
    sxz = 1.0e4 * rng.standard_normal((nx, nz))

    mirror_stress_ghosts_free_surface_numba(
        sxx, szz, sxz, M
    )
    fill_velocity_ghosts_free_surface_numba(
        vx, vz, lam, l2m, dx, dz, M
    )

    update_velocity_fused_numba(
        vx, vz, sxx, szz, sxz,
        bx, bz, dx, dz, dt, a,
    )

    stale_normal, stale_shear = _velocity_bc_residuals(
        vx, vz, lam, l2m, dx, dz, M
    )

    print("\nB. Old ordering: ghosts filled BEFORE velocity update")
    b_normal = _report("normal relation max |residual|", stale_normal)
    b_shear = _report("shear relation max |residual|", stale_shear)

    # C. Correct ordering: construct ghosts from the new n+1/2 velocity.
    fill_velocity_ghosts_free_surface_numba(
        vx, vz, lam, l2m, dx, dz, M
    )

    fixed_normal, fixed_shear = _velocity_bc_residuals(
        vx, vz, lam, l2m, dx, dz, M
    )

    print("\nC. Correct ordering: refill AFTER velocity update")
    c_normal = _report("normal relation max |residual|", fixed_normal)
    c_shear = _report("shear relation max |residual|", fixed_shear)

    # D. Check the stress image equations.
    update_stress_fused_free_surface_numba(
        vx, vz, sxx, szz, sxz,
        lam, l2m, mu_xz,
        dx, dz, dt, a,
    )
    mirror_stress_ghosts_free_surface_numba(
        sxx, szz, sxz, M
    )

    k = M

    print("\nD. Graves FS1 stress imaging")
    d_szz0 = _report("surface max |szz[k]|", szz[:, k])
    d_szz = _report(
        "szz max |szz[k-1] + szz[k+1]|",
        szz[:, k - 1] + szz[:, k + 1],
    )
    d_sxz1 = _report(
        "sxz max |sxz[k-1] + sxz[k]|",
        sxz[:, k - 1] + sxz[:, k],
    )
    d_sxz2 = _report(
        "sxz max |sxz[k-2] + sxz[k+1]|",
        sxz[:, k - 2] + sxz[:, k + 1],
    )

    # The exact boundary equations should close to floating-point precision.
    tol = 1.0e-11
    exact_errors = [
        a_normal,
        a_shear,
        c_normal,
        c_shear,
        d_szz0,
        d_szz,
        d_sxz1,
        d_sxz2,
    ]

    if max(exact_errors) > tol:
        raise RuntimeError(
            "Graves FS1 algebraic QC FAILED: "
            f"max exact residual={max(exact_errors):.3e} > {tol:.3e}"
        )

    # The intentionally stale experiment should fail by a visible margin.
    if max(b_normal, b_shear) <= 100.0 * tol:
        raise RuntimeError(
            "Timing QC was inconclusive: stale ghosts did not create a "
            "resolvable boundary-condition violation."
        )

    print("\nGraves FS1 algebraic/timing QC PASSED.")
    print(
        "Next step: homogeneous elastic half-space validation against a "
        "reference free-surface solution."
    )


if __name__ == "__main__":
    main()
