"""Precomputed SBP native-grid interpolation -> existing finite-gauge DAS.

The observable is the project's established centre-tangent, Cartesian-
difference strain rate. This module does not change that physical convention.
The grid is periodic in x; receiver coordinates outside its physical period
are rejected. In z both velocity grids must bracket the receiver: no boundary
extrapolation, coordinate clipping, or hidden geometry resampling is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.das.receivers import Receivers2D
from src.das.strain import _interp_index_weight, _valid_gauge_centres


def _readonly(a, dtype=float):
    a = np.array(a, dtype=dtype, copy=True)
    a.flags.writeable = False
    return a


@dataclass(frozen=True)
class _Stencil:
    ix: np.ndarray
    iz: np.ndarray
    weights: np.ndarray

    def sample(self, field):
        return np.sum(field[self.ix, self.iz] * self.weights, axis=1)

    def scatter(self, values, field):
        np.add.at(field, (self.ix, self.iz), values[:, None] * self.weights)


def _stencil(x, z, nx, nz, dx, dz, x_origin, z_origin):
    fx = (x - x_origin) / dx
    fz = (z - z_origin) / dz
    # Exact upper-grid samples use the last interpolation interval with w=1.
    # Bounds have already been checked in physical coordinates by the factory.
    i = np.floor(fx).astype(np.int64)
    j = np.minimum(np.floor(fz).astype(np.int64), nz - 2)
    wx, wz = fx - i, fz - j
    ix = np.stack((i % nx, (i + 1) % nx, i % nx, (i + 1) % nx), axis=1)
    iz = np.stack((j, j, j + 1, j + 1), axis=1)
    weights = np.stack(
        ((1 - wx) * (1 - wz), wx * (1 - wz), (1 - wx) * wz, wx * wz), axis=1
    )
    return _Stencil(
        _readonly(ix, np.int64), _readonly(iz, np.int64), _readonly(weights)
    )


@dataclass(frozen=True)
class DASObservation:
    """Linear velocity-field -> axial strain-rate map and exact transpose.

    Build with build_das_observation. len(observation) is the number of valid
    gauge centres. Returned data are in 1/s when velocity is in m/s.
    """

    receivers: Receivers2D
    gauge_length_m: float
    channel_indices: np.ndarray
    vx_shape: tuple[int, int]
    vz_shape: tuple[int, int]
    dx: float
    dz: float
    _vx: _Stencil
    _vz: _Stencil
    _gauge_indices: np.ndarray
    _gauge_weights: np.ndarray

    def __len__(self):
        return int(self.channel_indices.size)

    def validate_grid(self, model, op_z, dx):
        if self.vx_shape != (model.nx, model.nz_N) or self.vz_shape != (
            model.nx,
            model.nz_M,
        ):
            raise ValueError("DAS sampling grid dimensions do not match the model.")
        if dx != self.dx or op_z.h != self.dz:
            raise ValueError("DAS sampling dx/dz do not match the propagation grid.")

    def sample(self, vx, vz):
        x = self._vx.sample(vx)
        z = self._vz.sample(vz)
        delta_x = np.sum(x[self._gauge_indices] * self._gauge_weights, axis=1)
        delta_z = np.sum(z[self._gauge_indices] * self._gauge_weights, axis=1)
        c = self.channel_indices
        return (
            self.receivers.tx[c] * delta_x + self.receivers.tz[c] * delta_z
        ) / self.gauge_length_m

    def scatter(self, q, adj_vx, adj_vz):
        q = np.asarray(q, dtype=float)
        if q.shape != (len(self),):
            raise ValueError("DAS cotangent shape does not match gauge centres.")
        c = self.channel_indices
        qx = np.zeros(self.receivers.nrec)
        qz = np.zeros(self.receivers.nrec)
        weights = self._gauge_weights * (q / self.gauge_length_m)[:, None]
        np.add.at(qx, self._gauge_indices, weights * self.receivers.tx[c, None])
        np.add.at(qz, self._gauge_indices, weights * self.receivers.tz[c, None])
        self._vx.scatter(qx, adj_vx)
        self._vz.scatter(qz, adj_vz)


def build_das_observation(
    model,
    op_z,
    dx: float,
    receivers: Receivers2D,
    gauge_length_m: float,
    *,
    x0: float = 0.0,
) -> DASObservation:
    """Build O(n_receivers) immutable sampling weights once, outside the loop.

    Physical x must lie in [x0, x0+nx*dx). Native M_x indices wrap because the
    solver itself is periodic. Physical z must lie in [dz/2, depth-dz/2],
    the shared support of vx and vz. Surface point receivers in the original
    prototype remain supported separately; this DAS map never extrapolates.
    """
    if not np.isfinite(dx) or dx <= 0 or not np.isfinite(x0):
        raise ValueError("dx must be positive finite and x0 finite.")
    if op_z.n_N != model.nz_N or op_z.n_M != model.nz_M:
        raise ValueError("SBP operator dimensions do not match model.")
    if receivers.nrec < 2:
        raise ValueError("DAS requires at least two receiver centres.")
    L = float(gauge_length_m)
    if not np.isfinite(L) or L <= 0:
        raise ValueError("Gauge length must be finite and positive.")
    h = op_z.h
    if np.any((receivers.x < x0) | (receivers.x >= x0 + model.nx * dx)):
        raise ValueError("Receiver x is outside the physical periodic domain.")
    zmax = (model.nz_M - 0.5) * h
    if np.any((receivers.z < 0.5 * h) | (receivers.z > zmax)):
        raise ValueError(
            f"DAS receivers require dz/2 <= z <= depth-dz/2, here [{h / 2}, {zmax}] m."
        )
    s, ds = receivers.s, receivers.channel_spacing
    centres = _valid_gauge_centres(s, L / 2, 1e-6 * ds)
    if centres.size == 0:
        raise ValueError("Gauge length leaves no valid receiver centres.")
    jl, wl = _interp_index_weight(s[centres] - L / 2, s[0], ds, receivers.nrec)
    jr, wr = _interp_index_weight(s[centres] + L / 2, s[0], ds, receivers.nrec)
    indices = np.stack((jr, jr + 1, jl, jl + 1), axis=1)
    weights = np.stack((1 - wr, wr, -(1 - wl), -wl), axis=1)
    return DASObservation(
        receivers,
        L,
        _readonly(centres, np.int64),
        (model.nx, model.nz_N),
        (model.nx, model.nz_M),
        float(dx),
        float(h),
        _stencil(receivers.x, receivers.z, model.nx, model.nz_N, dx, h, x0 + dx / 2, 0),
        _stencil(receivers.x, receivers.z, model.nx, model.nz_M, dx, h, x0, h / 2),
        _readonly(indices, np.int64),
        _readonly(weights),
    )
