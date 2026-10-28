"""Bounded native-grid interpolation for the same finite-gauge DAS observable."""

from __future__ import annotations

import numpy as np

from src.das.strain import _interp_index_weight, _valid_gauge_centres
from .observations import DASObservation, _Stencil, _readonly


def _bounded_stencil(x, z, shape, dx, dz, x_origin, z_origin):
    fx, fz = (x - x_origin) / dx, (z - z_origin) / dz
    ix = np.minimum(np.floor(fx).astype(np.int64), shape[0] - 2)
    iz = np.minimum(np.floor(fz).astype(np.int64), shape[1] - 2)
    wx, wz = fx - ix, fz - iz
    return _Stencil(
        _readonly(np.stack((ix, ix + 1, ix, ix + 1), axis=1), np.int64),
        _readonly(np.stack((iz, iz, iz + 1, iz + 1), axis=1), np.int64),
        _readonly(
            np.stack(
                ((1 - wx) * (1 - wz), wx * (1 - wz), (1 - wx) * wz, wx * wz), axis=1
            )
        ),
    )


def build_rectangular_das_observation(
    solver, receivers, gauge_length_m, *, x0=0.0, z0=0.0
):
    """No periodic wrap, clipping, boundary extrapolation or cable resampling.

    Receivers must be bracketed by BOTH velocity grids. Thus dx/2 <= x-x0
    <= width-dx/2 and dz/2 <= z <= depth-dz/2. Native point receivers can
    still sample their own boundary nodes separately.
    """
    x, z = solver.op_x, solver.op_z
    dx, dz = x.h, z.h
    L = float(gauge_length_m)
    if not np.isfinite(x0) or not np.isfinite(z0) or not np.isfinite(L) or L <= 0:
        raise ValueError(
            "Grid origins must be finite; gauge length must be positive finite."
        )
    if receivers.nrec < 2:
        raise ValueError("DAS needs at least two physical receiver centres.")
    if np.any((receivers.x < x0 + dx / 2) | (receivers.x > x0 + (x.n_M - 0.5) * dx)):
        raise ValueError("Receiver x is outside the common bounded velocity support.")
    if np.any((receivers.z < z0 + dz / 2) | (receivers.z > z0 + (z.n_M - 0.5) * dz)):
        raise ValueError("Receiver z is outside the common bounded velocity support.")
    s, ds = receivers.s, receivers.channel_spacing
    centres = _valid_gauge_centres(s, L / 2, 1e-6 * ds)
    if not centres.size:
        raise ValueError("Gauge length leaves no valid cable centres.")
    jl, wl = _interp_index_weight(s[centres] - L / 2, s[0], ds, receivers.nrec)
    jr, wr = _interp_index_weight(s[centres] + L / 2, s[0], ds, receivers.nrec)
    return DASObservation(
        receivers,
        L,
        _readonly(centres, np.int64),
        solver.model.vx_shape,
        solver.model.vz_shape,
        dx,
        dz,
        _bounded_stencil(
            receivers.x, receivers.z, solver.model.vx_shape, dx, dz, x0 + dx / 2, z0
        ),
        _bounded_stencil(
            receivers.x, receivers.z, solver.model.vz_shape, dx, dz, x0, z0 + dz / 2
        ),
        _readonly(np.stack((jr, jr + 1, jl, jl + 1), axis=1), np.int64),
        _readonly(np.stack((1 - wr, wr, -(1 - wl), -wl), axis=1)),
    )
