"""
Fourth-order-interior / second-order-boundary staggered SBP operators.

Notation follows Gao et al. (2018):
    N-grid: boundary-aligned points, size n_intervals + 1
    M-grid: half-cell staggered points, size n_intervals

D_M_to_N approximates d/dz of M-grid data on the N-grid.
D_N_to_M approximates d/dz of N-grid data on the M-grid.

The diagonal norm matrices H_N and H_M and the boundary projection vectors
p_left / p_right satisfy the staggered SBP identity

    H_N D_M_to_N + (H_M D_N_to_M)^T
      = -e_left p_left^T + e_right p_right^T.

The interior derivative stencil is

    [1/24, -9/8, 9/8, -1/24] / h,

while special second-order closures are used near the boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class StaggeredSBP1D:
    h: float
    D_M_to_N: np.ndarray
    D_N_to_M: np.ndarray
    H_N: np.ndarray
    H_M: np.ndarray
    p_left: np.ndarray
    p_right: np.ndarray

    @property
    def n_N(self) -> int:
        return int(self.D_M_to_N.shape[0])

    @property
    def n_M(self) -> int:
        return int(self.D_M_to_N.shape[1])

    def sbp_residual(self) -> np.ndarray:
        e_left = np.zeros(self.n_N, dtype=np.float64)
        e_right = np.zeros(self.n_N, dtype=np.float64)
        e_left[0] = 1.0
        e_right[-1] = 1.0

        lhs = (
            np.diag(self.H_N) @ self.D_M_to_N
            + (np.diag(self.H_M) @ self.D_N_to_M).T
        )
        rhs = (
            -np.outer(e_left, self.p_left)
            + np.outer(e_right, self.p_right)
        )
        return lhs - rhs


def staggered_sbp_4_2(n_intervals: int, h: float) -> StaggeredSBP1D:
    """
    Build the diagonal-norm 4-2 staggered SBP pair used by Gao et al. (2018).

    Parameters
    ----------
    n_intervals
        Number of half-cell M-grid points. The boundary-aligned N-grid contains
        n_intervals + 1 points.
    h
        Uniform grid spacing.

    Notes
    -----
    The published coefficients are given for h=1. Derivative matrices are
    therefore divided by h and norm weights multiplied by h here.
    """
    if n_intervals < 8:
        raise ValueError(
            "n_intervals must be at least 8 for the 4-2 staggered SBP closure."
        )
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError(f"h must be finite and positive; got {h}.")

    n_M = int(n_intervals)
    n_N = n_M + 1

    D_MN = np.zeros((n_N, n_M), dtype=np.float64)
    D_NM = np.zeros((n_M, n_N), dtype=np.float64)

    # ------------------------------------------------------------------
    # M -> N derivative.  Boundary closures are Eq. (12a) of Gao et al.
    # ------------------------------------------------------------------
    D_MN[0, 0:3] = (-2.0, 3.0, -1.0)
    D_MN[1, 0:2] = (-1.0, 1.0)
    D_MN[2, 0:4] = (1.0 / 24.0, -9.0 / 8.0, 9.0 / 8.0, -1.0 / 24.0)
    D_MN[3, 0:5] = (
        -1.0 / 71.0,
        6.0 / 71.0,
        -83.0 / 71.0,
        81.0 / 71.0,
        -3.0 / 71.0,
    )

    for r in range(4, n_N - 4):
        D_MN[r, r - 2:r + 2] = (
            1.0 / 24.0,
            -9.0 / 8.0,
            9.0 / 8.0,
            -1.0 / 24.0,
        )

    D_MN[n_N - 4, n_M - 5:n_M] = (
        3.0 / 71.0,
        -81.0 / 71.0,
        83.0 / 71.0,
        -6.0 / 71.0,
        1.0 / 71.0,
    )
    D_MN[n_N - 3, n_M - 4:n_M] = (
        1.0 / 24.0,
        -9.0 / 8.0,
        9.0 / 8.0,
        -1.0 / 24.0,
    )
    D_MN[n_N - 2, n_M - 2:n_M] = (-1.0, 1.0)
    D_MN[n_N - 1, n_M - 3:n_M] = (1.0, -3.0, 2.0)

    # ------------------------------------------------------------------
    # N -> M derivative.  Boundary closures are Eq. (12b).
    # ------------------------------------------------------------------
    D_NM[0, 0:4] = (-79.0 / 78.0, 27.0 / 26.0, -1.0 / 26.0, 1.0 / 78.0)
    D_NM[1, 0:4] = (2.0 / 21.0, -9.0 / 7.0, 9.0 / 7.0, -2.0 / 21.0)
    D_NM[2, 0:5] = (1.0 / 75.0, 0.0, -27.0 / 25.0, 83.0 / 75.0, -1.0 / 25.0)

    for r in range(3, n_M - 3):
        D_NM[r, r - 1:r + 3] = (
            1.0 / 24.0,
            -9.0 / 8.0,
            9.0 / 8.0,
            -1.0 / 24.0,
        )

    D_NM[n_M - 3, n_N - 5:n_N] = (
        1.0 / 25.0,
        -83.0 / 75.0,
        27.0 / 25.0,
        0.0,
        -1.0 / 75.0,
    )
    D_NM[n_M - 2, n_N - 4:n_N] = (
        2.0 / 21.0,
        -9.0 / 7.0,
        9.0 / 7.0,
        -2.0 / 21.0,
    )
    D_NM[n_M - 1, n_N - 4:n_N] = (
        -1.0 / 78.0,
        1.0 / 26.0,
        -27.0 / 26.0,
        79.0 / 78.0,
    )

    # Diagonal SBP norms, Eqs. (12c)-(12d).
    H_N = np.ones(n_N, dtype=np.float64)
    H_M = np.ones(n_M, dtype=np.float64)

    H_N[:4] = (7.0 / 18.0, 9.0 / 8.0, 1.0, 71.0 / 72.0)
    H_N[-4:] = (71.0 / 72.0, 1.0, 9.0 / 8.0, 7.0 / 18.0)

    H_M[:3] = (13.0 / 12.0, 7.0 / 8.0, 25.0 / 24.0)
    H_M[-3:] = (25.0 / 24.0, 7.0 / 8.0, 13.0 / 12.0)

    D_MN /= h
    D_NM /= h
    H_N *= h
    H_M *= h

    # Second-order boundary projections obtained from the SBP boundary matrix Q.
    p_left = np.zeros(n_M, dtype=np.float64)
    p_right = np.zeros(n_M, dtype=np.float64)
    p_left[:3] = (15.0 / 8.0, -5.0 / 4.0, 3.0 / 8.0)
    p_right[-3:] = (3.0 / 8.0, -5.0 / 4.0, 15.0 / 8.0)

    op = StaggeredSBP1D(
        h=float(h),
        D_M_to_N=D_MN,
        D_N_to_M=D_NM,
        H_N=H_N,
        H_M=H_M,
        p_left=p_left,
        p_right=p_right,
    )

    err = float(np.max(np.abs(op.sbp_residual())))
    if err > 5.0e-13:
        raise RuntimeError(
            f"Internal SBP identity construction failed: max residual={err:.3e}."
        )
    return op


# -----------------------------------------------------------------------------
# Periodic x-direction staggered derivatives.
#
# There is no physical boundary in x in the first controlled SBP-SAT branch.
# Therefore H_x = h I and the standard fourth-order staggered stencil wraps
# around periodically.  With these definitions:
#
#     (D_M_to_N)^T = -D_N_to_M
#
# in the ordinary Euclidean inner product.
# -----------------------------------------------------------------------------

def periodic_D_M_to_N(f: np.ndarray, h: float) -> np.ndarray:
    """Derivative of M-grid data evaluated on the periodic N-grid."""
    return (
        np.roll(f, 2, axis=0) / 24.0
        - 9.0 * np.roll(f, 1, axis=0) / 8.0
        + 9.0 * f / 8.0
        - np.roll(f, -1, axis=0) / 24.0
    ) / h


def periodic_D_N_to_M(f: np.ndarray, h: float) -> np.ndarray:
    """Derivative of N-grid data evaluated on the periodic M-grid."""
    return (
        np.roll(f, 1, axis=0) / 24.0
        - 9.0 * f / 8.0
        + 9.0 * np.roll(f, -1, axis=0) / 8.0
        - np.roll(f, -2, axis=0) / 24.0
    ) / h
