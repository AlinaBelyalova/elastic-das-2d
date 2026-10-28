"""Traction-free SAT and its Euclidean transpose, with depth positive down.

These are force densities BEFORE division by native-grid density. The upper
boundary is the left endpoint of the z SBP pair; its penalty sign is +1.
The bottom sign is -1. No stresses are overwritten and no ghost cells exist.
"""

from __future__ import annotations

import numpy as np

from .operators import StaggeredSBP1D


def traction_free_sat(
    sxz: np.ndarray,
    szz: np.ndarray,
    op: StaggeredSBP1D,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (SAT_x on N_z, SAT_z on M_z); last axis is z.

    SAT_x = H_N^-1 (e_L p_L^T - e_R p_R^T) sxz
    SAT_z = H_M^-1 (p_L e_L^T - p_R e_R^T) szz
    """
    fx = np.zeros_like(szz, dtype=np.float64)
    fx[..., 0] = (sxz @ op.p_left) / op.H_N[0]
    fx[..., -1] = -(sxz @ op.p_right) / op.H_N[-1]
    fz = szz[..., 0, None] * (op.p_left / op.H_M)
    fz -= szz[..., -1, None] * (op.p_right / op.H_M)
    return fx, fz


def traction_free_sat_transpose(
    qx: np.ndarray,
    qz: np.ndarray,
    op: StaggeredSBP1D,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (adj_sxz, adj_szz) for force-density cotangents qx/qz.

    This is the ordinary array transpose, without a second multiplication by
    H or density. Those factors already belong to the forward map.
    """
    sxz = qx[..., 0, None] * (op.p_left / op.H_N[0])
    sxz -= qx[..., -1, None] * (op.p_right / op.H_N[-1])
    szz = np.zeros_like(qx, dtype=np.float64)
    szz[..., 0] = qz @ (op.p_left / op.H_M)
    szz[..., -1] = -(qz @ (op.p_right / op.H_M))
    return sxz, szz
