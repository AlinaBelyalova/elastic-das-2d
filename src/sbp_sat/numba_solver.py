"""Compiled execution of the bounded 4-2 SBP/SAT step, float64, no fastmath.

The matrices/impedances/implicit rank solves come from RectangularSBPSAT.
Only execution changes. The independent CSR transpose and material VJP are
inherited, and checked against the compiled forward rather than rewritten.
"""

from __future__ import annotations

from collections import namedtuple
from contextlib import contextmanager
from numbers import Integral
import os
import numpy as np
from numba import njit, prange, get_num_threads, set_num_threads, config

from .elastic2d import ForwardResult, SBPSATState, StressPointSource
from .rectangular import RectangularSBPSAT
from .sources import DistributedStressSource


Derivatives = namedtuple("Derivatives", "fx_x fx_z fz_x fz_z ex_x ez_z gamma_x gamma_z")
Medium = namedtuple(
    "Medium", "rho_x rho_z inverse_x inverse_z lam l2m mu tangent_x tangent_z"
)
Faces = namedtuple("Faces", "px pz lx lz zx zz ux cx uz cz")
Source = namedtuple("Source", "ix iz w jx jz v mxx mzz mxz")


def _readonly(a, dtype=float):
    a = np.array(a, dtype=dtype, order="C", copy=True)
    a.flags.writeable = False
    return a


def _copy_state64(state):
    return SBPSATState(
        *(
            np.array(getattr(state, name), dtype=np.float64, order="C", copy=True)
            for name in ("vx", "vz", "sxx", "szz", "sxz")
        )
    )


def _compact(matrix):
    """Five exact taps per row; never threshold or drop a nonzero coefficient."""
    indices = np.zeros((matrix.shape[0], 5), dtype=np.int64)
    values = np.zeros_like(indices, dtype=float)
    for row in range(matrix.shape[0]):
        columns = np.flatnonzero(matrix[row])
        if columns.size > 5:
            raise ValueError("4-2 operator exceeds the certified five-tap storage.")
        indices[row, : columns.size] = columns
        values[row, : columns.size] = matrix[row, columns]
    return _readonly(indices, np.int64), _readonly(values)


def _derivatives(solver):
    def force_nm(op):
        D = op.D_N_to_M.copy()
        D[:, 0] += op.p_left / op.H_M
        D[:, -1] -= op.p_right / op.H_M
        return D

    def force_mn(op):
        D = op.D_M_to_N.copy()
        D[0, :] += op.p_left / op.H_N[0]
        D[-1, :] -= op.p_right / op.H_N[-1]
        return D

    x, z = solver.op_x, solver.op_z
    # Fold the SAME free-traction SAT into force rows. This is QF=-B^T W,
    # not a new boundary closure or a stress overwrite.
    return Derivatives(
        *map(
            _compact,
            (
                force_nm(x),
                force_mn(z),
                force_mn(x),
                force_nm(z),
                x.D_M_to_N,
                z.D_M_to_N,
                x.D_N_to_M,
                z.D_N_to_M,
            ),
        )
    )


def _interior_coefficients(solver):
    values = []
    for op in (solver.op_x, solver.op_z):
        c1, c2 = float(op.D_N_to_M[3, 4]), float(op.D_N_to_M[3, 5])
        weights = np.array([-c2, -c1, c1, c2])
        for row in range(4, op.n_N - 4):
            for matrix, start in ((op.D_N_to_M, row - 1), (op.D_M_to_N, row - 2)):
                if np.count_nonzero(matrix[row]) != 4 or not np.array_equal(
                    matrix[row, start : start + 4], weights
                ):
                    raise ValueError(
                        "SBP interior differs from the four-tap compiled stencil."
                    )
        values.extend((c1, c2))
    return tuple(values)


def _faces(solver):
    x, z, Z = solver.op_x, solver.op_z, solver._Z
    px = np.stack((x.p_left[:3], x.p_right[-3:]))
    pz = np.stack((z.p_left[:3], z.p_right[-3:]))
    lx = px / np.stack((x.H_M[:3], x.H_M[-3:]))
    lz = pz / np.stack((z.H_M[:3], z.H_M[-3:]))
    ux = np.stack((solver._rank_x[0][1][:3], solver._rank_x[1][1][-3:]))
    uz = np.stack((solver._rank_z[0][1][:, :3], solver._rank_z[1][1][:, -3:]))
    return Faces(
        *map(
            _readonly,
            (
                px,
                pz,
                lx,
                lz,
                np.stack((Z["left_p"], Z["right_p"])),
                np.stack((Z["top_p"], Z["bottom_p"])),
                ux,
                np.stack([r[2] for r in solver._rank_x]),
                uz,
                np.stack([r[2] for r in solver._rank_z]),
            ),
        )
    )


def _source(source):
    if source is None:
        ix = iz = jx = jz = np.empty(0, dtype=np.int64)
        w = v = np.empty(0)
        components = (0.0, 0.0, 0.0)
    elif isinstance(source, DistributedStressSource):
        ix, iz, w = (
            source.normal.ix.ravel(),
            source.normal.iz.ravel(),
            source.normal.weights.ravel(),
        )
        jx, jz, v = (
            source.shear.ix.ravel(),
            source.shear.iz.ravel(),
            source.shear.weights.ravel(),
        )
        components = source.mxx, source.mzz, source.mxz
    elif isinstance(source, StressPointSource):
        ix, iz, w = [source.ix], [source.iz], [1.0]
        jx = [source.ix - 1, source.ix, source.ix - 1, source.ix]
        jz = [source.iz - 1, source.iz - 1, source.iz, source.iz]
        v = [0.25] * 4
        components = source.mxx, source.mzz, source.mxz
    else:
        raise TypeError("Unsupported bounded SBP source.")
    return Source(
        _readonly(ix, np.int64),
        _readonly(iz, np.int64),
        _readonly(w),
        _readonly(jx, np.int64),
        _readonly(jz, np.int64),
        _readonly(v),
        *map(float, components),
    )


@njit(inline="always")
def _dx(field, indices, values, i, j):
    out = 0.0
    for k in range(5):
        out += values[i, k] * field[indices[i, k], j]
    return out


@njit(inline="always")
def _dz(field, indices, values, i, j):
    out = 0.0
    for k in range(5):
        out += values[j, k] * field[i, indices[j, k]]
    return out


@njit(cache=True, parallel=True, fastmath=False)
def _velocity(
    vx,
    vz,
    xx,
    zz,
    xz,
    out_x,
    out_z,
    fxx_i,
    fxx_w,
    fxz_i,
    fxz_w,
    fzx_i,
    fzx_w,
    fzz_i,
    fzz_w,
    rho_x,
    rho_z,
    inverse_x,
    inverse_z,
    tangent_x,
    tangent_z,
    px,
    pz,
    lx,
    lz,
    zx,
    zp_z,
    c1x,
    c2x,
    c1z,
    c2z,
    dt,
):
    nx, nz = xx.shape
    for row in prange(nx):
        i = np.int64(row)
        for j in range(nz):
            if 4 <= i < nx - 4 and 4 <= j < nz - 4:
                # Exact same four interior coefficients, paired differences;
                # no indirect indices, zero padding or damping in the bulk.
                fx = (
                    c1x * (xx[i + 1, j] - xx[i, j])
                    + c2x * (xx[i + 2, j] - xx[i - 1, j])
                    + c1z * (xz[i, j] - xz[i, j - 1])
                    + c2z * (xz[i, j + 1] - xz[i, j - 2])
                )
                fz = (
                    c1x * (xz[i, j] - xz[i - 1, j])
                    + c2x * (xz[i + 1, j] - xz[i - 2, j])
                    + c1z * (zz[i, j + 1] - zz[i, j])
                    + c2z * (zz[i, j + 2] - zz[i, j - 1])
                )
                out_x[i, j] = (rho_x[i, j] * vx[i, j] + dt * fx) * inverse_x[i, j]
                out_z[i, j] = (rho_z[i, j] * vz[i, j] + dt * fz) * inverse_z[i, j]
                continue
            if i < nx - 1:
                force = _dx(xx, fxx_i, fxx_w, i, j) + _dz(xz, fxz_i, fxz_w, i, j)
                damping = tangent_x[i, j] * vx[i, j]
                if i < 3:
                    trace = 0.0
                    for k in range(3):
                        trace += px[0, k] * vx[k, j]
                    damping += lx[0, i] * (zx[0, j] * trace)
                if i >= nx - 4:
                    trace = 0.0
                    for k in range(3):
                        trace += px[1, k] * vx[nx - 4 + k, j]
                    damping += lx[1, i - (nx - 4)] * (zx[1, j] * trace)
                out_x[i, j] = (
                    rho_x[i, j] * vx[i, j] - 0.5 * dt * damping + dt * force
                ) * inverse_x[i, j]
            if j < nz - 1:
                force = _dx(xz, fzx_i, fzx_w, i, j) + _dz(zz, fzz_i, fzz_w, i, j)
                damping = tangent_z[i, j] * vz[i, j]
                if j < 3:
                    trace = 0.0
                    for k in range(3):
                        trace += pz[0, k] * vz[i, k]
                    damping += (zp_z[0, i] * trace) * lz[0, j]
                if j >= nz - 4:
                    trace = 0.0
                    for k in range(3):
                        trace += pz[1, k] * vz[i, nz - 4 + k]
                    damping += (zp_z[1, i] * trace) * lz[1, j - (nz - 4)]
                out_z[i, j] = (
                    rho_z[i, j] * vz[i, j] - 0.5 * dt * damping + dt * force
                ) * inverse_z[i, j]


@njit(cache=True, parallel=True, fastmath=False)
def _rank_solve(vx, vz, px, pz, ux, cx, uz, cz):
    nx, nz = vz.shape[0], vx.shape[1]
    # Disjoint three-point supports on grids with >=8 intervals. Corners
    # already include the tangential diagonal through the cached inverse.
    for row in prange(nx + nz):
        line = np.int64(row)
        if line < nz:
            j = line
            for side in range(2):
                start = 0 if side == 0 else nx - 4
                trace = 0.0
                for k in range(3):
                    trace += px[side, k] * vx[start + k, j]
                factor = cx[side, j] * trace
                for k in range(3):
                    vx[start + k, j] -= ux[side, k, j] * factor
        else:
            i = line - nz
            for side in range(2):
                start = 0 if side == 0 else nz - 4
                trace = 0.0
                for k in range(3):
                    trace += pz[side, k] * vz[i, start + k]
                factor = cz[side, i] * trace
                for k in range(3):
                    vz[i, start + k] -= uz[side, i, k] * factor


@njit(cache=True, parallel=True, fastmath=False)
def _stress(
    vx,
    vz,
    xx,
    zz,
    xz,
    ex_i,
    ex_w,
    ez_i,
    ez_w,
    gx_i,
    gx_w,
    gz_i,
    gz_w,
    lam,
    l2m,
    mu,
    c1x,
    c2x,
    c1z,
    c2z,
    dt,
):
    nx, nz = xx.shape
    for row in prange(nx):
        i = np.int64(row)
        for j in range(nz):
            if 4 <= i < nx - 4 and 4 <= j < nz - 4:
                ex = c1x * (vx[i, j] - vx[i - 1, j]) + c2x * (
                    vx[i + 1, j] - vx[i - 2, j]
                )
                ez = c1z * (vz[i, j] - vz[i, j - 1]) + c2z * (
                    vz[i, j + 1] - vz[i, j - 2]
                )
                gamma = (
                    c1z * (vx[i, j + 1] - vx[i, j])
                    + c2z * (vx[i, j + 2] - vx[i, j - 1])
                    + c1x * (vz[i + 1, j] - vz[i, j])
                    + c2x * (vz[i + 2, j] - vz[i - 1, j])
                )
                xx[i, j] += dt * (l2m[i, j] * ex + lam[i, j] * ez)
                zz[i, j] += dt * (lam[i, j] * ex + l2m[i, j] * ez)
                xz[i, j] += dt * mu[i, j] * gamma
                continue
            ex, ez = _dx(vx, ex_i, ex_w, i, j), _dz(vz, ez_i, ez_w, i, j)
            xx[i, j] += dt * (l2m[i, j] * ex + lam[i, j] * ez)
            zz[i, j] += dt * (lam[i, j] * ex + l2m[i, j] * ez)
            if i < nx - 1 and j < nz - 1:
                gamma = _dz(vx, gz_i, gz_w, i, j) + _dx(vz, gx_i, gx_w, i, j)
                xz[i, j] += dt * mu[i, j] * gamma


@njit(cache=True, fastmath=False)
def _step(vx, vz, xx, zz, xz, out_x, out_z, d, m, b, interior, source, dt, amplitude):
    _velocity(
        vx,
        vz,
        xx,
        zz,
        xz,
        out_x,
        out_z,
        d.fx_x[0],
        d.fx_x[1],
        d.fx_z[0],
        d.fx_z[1],
        d.fz_x[0],
        d.fz_x[1],
        d.fz_z[0],
        d.fz_z[1],
        m.rho_x,
        m.rho_z,
        m.inverse_x,
        m.inverse_z,
        m.tangent_x,
        m.tangent_z,
        b.px,
        b.pz,
        b.lx,
        b.lz,
        b.zx,
        b.zz,
        interior[0],
        interior[1],
        interior[2],
        interior[3],
        dt,
    )
    _rank_solve(out_x, out_z, b.px, b.pz, b.ux, b.cx, b.uz, b.cz)
    _stress(
        out_x,
        out_z,
        xx,
        zz,
        xz,
        d.ex_x[0],
        d.ex_x[1],
        d.ez_z[0],
        d.ez_z[1],
        d.gamma_x[0],
        d.gamma_x[1],
        d.gamma_z[0],
        d.gamma_z[1],
        m.lam,
        m.l2m,
        m.mu,
        interior[0],
        interior[1],
        interior[2],
        interior[3],
        dt,
    )
    scale = dt * amplitude
    # Serial accumulating scatter: no races even for overlapping source taps.
    for k in range(len(source.w)):
        xx[source.ix[k], source.iz[k]] += (scale * source.mxx) * source.w[k]
        zz[source.ix[k], source.iz[k]] += (scale * source.mzz) * source.w[k]
    for k in range(len(source.v)):
        xz[source.jx[k], source.jz[k]] += (scale * source.mxz) * source.v[k]


@njit(cache=True, fastmath=False)
def _sample_receivers(vx, vz, ix, iz, wx, jx, jz, wz, out_x, out_z, it):
    for r in range(out_x.shape[0]):
        x, z = 0.0, 0.0
        for k in range(4):
            x += wx[r, k] * vx[ix[r, k], iz[r, k]]
            z += wz[r, k] * vz[jx[r, k], jz[r, k]]
        out_x[r, it] = x
        out_z[r, it] = z


def _thread_count(requested, cells):
    available = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", available))
    maximum = min(available, allocated, config.NUMBA_NUM_THREADS)
    if requested is None:
        requested = 1 if cells < 4096 else min(4, maximum)
    if (
        isinstance(requested, bool)
        or not isinstance(requested, Integral)
        or requested < 1
    ):
        raise ValueError("numba_threads must be a positive integer.")
    if requested > maximum:
        raise ValueError(
            f"Requested {requested} Numba threads; allocation/runtime supports {maximum}."
        )
    return int(requested)


class NumbaSBPSAT(RectangularSBPSAT):
    """Same bounded operator; compiled step and reusable forward workspace.

    step/_advance retain nonmutating reference semantics. advance_inplace is
    only for an owned forward state. Do not share its workspace concurrently.
    """

    def __init__(self, model, dx, dz, dt, boundaries=None, *, numba_threads=None):
        super().__init__(model, dx, dz, dt, boundaries, sparse_derivatives=True)
        self.numba_threads = _thread_count(numba_threads, model.vp.size)
        self._d = _derivatives(self)
        self._interior = _interior_coefficients(self)
        m = self.model
        self._m = Medium(
            m.rho_vx,
            m.rho_vz,
            self._inverse_x,
            self._inverse_z,
            m.lam_NN,
            m.l2m_NN,
            m.mu_MM,
            self._tangent_x,
            self._tangent_z,
        )
        self._b = _faces(self)
        self._cached_source = object()
        self._native_source = None
        self._work_x = np.empty(m.vx_shape)
        self._work_z = np.empty(m.vz_shape)

    @contextmanager
    def thread_context(self):
        old = get_num_threads()
        if old != self.numba_threads:
            set_num_threads(self.numba_threads)
        try:
            yield
        finally:
            if old != self.numba_threads:
                set_num_threads(old)

    def _source_data(self, source):
        if source is not self._cached_source:
            if source is not None:
                self._validate_source(source)
            self._native_source = _source(source)
            self._cached_source = source
        return self._native_source

    def _advance(self, state, source=None, amplitude=0.0):
        # Independent output: all existing energy/adjoint/gradient calls retain
        # their reference semantics. Only the owned production loop reuses data.
        out = _copy_state64(state)
        with self.thread_context():
            _step(
                state.vx,
                state.vz,
                out.sxx,
                out.szz,
                out.sxz,
                out.vx,
                out.vz,
                self._d,
                self._m,
                self._b,
                self._interior,
                self._source_data(source),
                self.dt,
                amplitude,
            )
        return out

    def advance_inplace(self, state, source, amplitude):
        """Advance an owned state; caller holds thread_context for its loop."""
        if any(
            getattr(state, name).dtype != np.float64
            for name in ("vx", "vz", "sxx", "szz", "sxz")
        ):
            raise TypeError(
                "Owned Numba state must be float64; use zero_state or run_forward."
            )
        _step(
            state.vx,
            state.vz,
            state.sxx,
            state.szz,
            state.sxz,
            self._work_x,
            self._work_z,
            self._d,
            self._m,
            self._b,
            self._interior,
            self._source_data(source),
            self.dt,
            amplitude,
        )
        state.vx, self._work_x = self._work_x, state.vx
        state.vz, self._work_z = self._work_z, state.vz
        return state

    @staticmethod
    def sample_receivers(state, observation, out_x, out_z, it):
        x, z = observation._vx, observation._vz
        _sample_receivers(
            state.vx,
            state.vz,
            x.ix,
            x.iz,
            x.weights,
            z.ix,
            z.iz,
            z.weights,
            out_x,
            out_z,
            it,
        )

    def run_forward(
        self,
        stf,
        source,
        receivers,
        *,
        store_velocity_history=False,
        initial_state=None,
    ):
        stf = np.asarray(stf, dtype=float)
        if stf.ndim != 1 or not stf.size or not np.isfinite(stf).all():
            raise ValueError("STF must be a nonempty finite vector.")
        self._validate_source(source)
        self._validate_receivers(receivers)
        state = (
            self.zero_state() if initial_state is None else _copy_state64(initial_state)
        )
        self._validate_state(state)
        data = np.empty((len(receivers), len(stf)))
        history = [] if store_velocity_history else None
        with self.thread_context():
            for it, amplitude in enumerate(stf):
                self.advance_inplace(state, source, amplitude)
                data[:, it] = self._sample(state.vx, state.vz, receivers)
                if history is not None:
                    history.append((state.vx.copy(), state.vz.copy()))
        return ForwardResult(data, state, None if history is None else tuple(history))
