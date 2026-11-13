"""Legacy-style Gaussian sponge on the five native SBP fields.

One post-step multiplication, as in solver_numpy.make_sponge_mask. This is
a first-order splitting inside the strips, not PML or a Strang claim. The
ordinary SBP-SAT step, free surface and exterior impedance closure remain.
The exact discrete source transpose/model VJP include the multiplication.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from .elastic2d import _source_transpose
from .numba_solver import NumbaSBPSAT
from .rectangular import RectangularSBPSAT
from .sources import DistributedStressSource

FIELDS = ("vx", "vz", "sxx", "szz", "sxz")


@dataclass(frozen=True)
class GaussianSponge:
    """Fixed numerical profile. Widths are metres; rate is 1/s, not per step."""

    side_width_m: float
    bottom_width_m: float
    rate_per_s: float

    def __post_init__(self):
        values = (self.side_width_m, self.bottom_width_m, self.rate_per_s)
        if not np.isfinite(values).all() or any(v < 0 for v in values):
            raise ValueError("Sponge widths/rate must be finite and nonnegative.")

    def masks(self, solver):
        x, z, h = solver.op_x, solver.op_z, solver.dt
        lx, lz = (x.n_N - 1) * x.h, (z.n_N - 1) * z.h
        if 2 * self.side_width_m >= lx or self.bottom_width_m >= lz:
            raise ValueError("Sponge strips overlap or consume the domain.")

        def factor(coordinates, extent, width, low):
            if not width or not self.rate_per_s:
                return np.ones_like(coordinates)
            ramp = np.maximum(0, (coordinates - (extent - width)) / width) ** 2
            if low:
                ramp += np.maximum(0, (width - coordinates) / width) ** 2
            return np.exp(-self.rate_per_s * h * ramp)

        xn, xm = np.arange(x.n_N) * x.h, (np.arange(x.n_M) + 0.5) * x.h
        zn, zm = np.arange(z.n_N) * z.h, (np.arange(z.n_M) + 0.5) * z.h
        axn, axm = (factor(v, lx, self.side_width_m, True) for v in (xn, xm))
        azn, azm = (factor(v, lz, self.bottom_width_m, False) for v in (zn, zm))
        normal = axn[:, None] * azn
        masks = (
            axm[:, None] * azn,
            axn[:, None] * azm,
            normal,
            normal,
            axm[:, None] * azm,
        )
        # Local inversion of the final velocity multiplier is used only to
        # evaluate the exact VJP from the existing velocity histories. Keep
        # that small algebraic operation well conditioned, with no wavefield
        # reconstruction, extra histories or clipping.
        if min(float(a.min()) for a in masks) < 0.25:
            raise ValueError("Sponge multiplier < 0.25: reduce rate or timestep.")
        for a in masks:
            a.flags.writeable = False
        return masks


@njit(cache=True, parallel=True)
def _multiply_native(vx, vz, xx, zz, xz, mx, mz, mn, ms):
    for i in prange(xx.shape[0]):
        for j in range(xx.shape[1]):
            xx[i, j] *= mn[i, j]
            zz[i, j] *= mn[i, j]
            if i < vx.shape[0]:
                vx[i, j] *= mx[i, j]
            if j < vz.shape[1]:
                vz[i, j] *= mz[i, j]
            if i < xz.shape[0] and j < xz.shape[1]:
                xz[i, j] *= ms[i, j]


class _SpongeMixin:
    def __init__(self, *args, sponge, **kwargs):
        if not isinstance(sponge, GaussianSponge):
            raise TypeError("sponge must be GaussianSponge.")
        super().__init__(*args, **kwargs)
        if self.boundaries.top != "free":
            raise ValueError("This three-face sponge requires a free top.")
        self.sponge = sponge
        self.sponge_masks = sponge.masks(self)

    def _multiply(self, state):
        for name, mask in zip(FIELDS, self.sponge_masks):
            getattr(state, name)[:] *= mask
        return state

    def _advance(self, state, source=None, amplitude=0.0):
        return self._multiply(super()._advance(state, source, amplitude))

    def _transpose_step(self, adj, q, receivers):
        # Observation is AFTER the sponge: Phi^T S^T (adj + O^T q).
        a = adj.copy()
        self._scatter(q, receivers, a.vx, a.vz)
        self._multiply(a)
        return super()._transpose_step(a, np.zeros_like(q), receivers)

    def run_adjoint_to_source(self, q_data, source, receivers):
        q = np.asarray(q_data, dtype=float)
        self._validate_source(source)
        self._validate_receivers(receivers)
        if (
            q.ndim != 2
            or q.shape[0] != len(receivers)
            or not q.shape[1]
            or not np.isfinite(q).all()
        ):
            raise ValueError("Adjoint data need finite shape (n_receivers,nt), nt>=1.")
        adj, gradient = self.zero_state(), np.zeros(q.shape[1])
        mn, ms = self.sponge_masks[2], self.sponge_masks[4]
        for it in range(q.shape[1] - 1, -1, -1):
            xx, zz, xz = mn * adj.sxx, mn * adj.szz, ms * adj.sxz
            gradient[it] = (
                self.dt * source.gather(xx, zz, xz)
                if isinstance(source, DistributedStressSource)
                else _source_transpose(xx, zz, xz, source, self.dt)
            )
            adj, _ = self._transpose_step(adj, q[:, it], receivers)
        return gradient

    def _reverse_model_gradient_step(
        self, adj, q, receivers, vx, vz, old_x, old_z, gradients
    ):
        gl, ga, gm, bp, bs = gradients
        # Stored velocities follow S. Recover only the two same-step
        # pre-multiplier velocities; no reverse propagation is performed.
        raw_x, raw_z = vx / self.sponge_masks[0], vz / self.sponge_masks[1]
        ex, ez, gamma = self.strains(raw_x, raw_z)
        axx, azz, axz = (
            adj.sxx * self.sponge_masks[2],
            adj.szz * self.sponge_masks[3],
            adj.sxz * self.sponge_masks[4],
        )
        gl += self.dt * (axx * ez + azz * ex)
        ga += self.dt * (axx * ex + azz * ez)
        gm += self.dt * axz * gamma
        adj, (rx, rz) = self._transpose_step(adj, q, receivers)
        p, s = self._impedance_vjp((old_x + raw_x) / 2, (old_z + raw_z) / 2, rx, rz)
        bp += p
        bs += s
        return adj

    def validate_undamped_support(self, source, observation):
        """Production source/DAS interpolation supports must avoid the strips."""
        if isinstance(source, DistributedStressSource):
            pairs = [(source.normal, self.sponge_masks[2])]
            if source.mxz:
                pairs.append((source.shear, self.sponge_masks[4]))
            for stencil, mask in pairs:
                if np.any(mask[stencil.ix, stencil.iz][stencil.weights != 0] != 1):
                    raise ValueError(
                        "Source interpolation support intersects sponge; change domain/width, no clipping."
                    )
        for stencil, mask in (
            (observation._vx, self.sponge_masks[0]),
            (observation._vz, self.sponge_masks[1]),
        ):
            if np.any(mask[stencil.ix, stencil.iz][stencil.weights != 0] != 1):
                raise ValueError(
                    "Receiver/DAS interpolation support intersects sponge; change domain/width, no clipping."
                )


class SpongeSBPSAT(_SpongeMixin, RectangularSBPSAT):
    """Independent matrix forward/reverse with the legacy-style sponge."""


class NumbaSpongeSBPSAT(_SpongeMixin, NumbaSBPSAT):
    """Same five-field map; compiled sponge on the owned forward state."""

    def advance_inplace(self, state, source, amplitude):
        out = super().advance_inplace(state, source, amplitude)
        mx, mz, mn, _, ms = self.sponge_masks
        _multiply_native(out.vx, out.vz, out.sxx, out.szz, out.sxz, mx, mz, mn, ms)
        return out
