"""Bounded Cartesian SBP-SAT elasticity with dissipative impedance faces.

Depth z increases down. Normal stresses/materials are N-N, vx is M-N,
vz N-M and sxz M-M; BOTH coordinates now have bounded 4-2 SBP pairs.
Top defaults to traction-free; sides/bottom to t_out = -diag(Zp,Zs) v.
The boundary damping is trapezoidal in the velocity half-step. Together with
the staggered stress update this gives an exact modified-energy dissipation
identity, not a post-step sponge. See docs/sbp_sat/absorbing.md.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .elastic2d import (
    ForwardResult,
    PointReceiver,
    SBPSATState,
    StressPointSource,
    _inject_source,
    _source_transpose,
    _validate_base_model,
)
from .observations import DASObservation
from .operators import staggered_sbp_4_2
from .stability import _weighted_derivative_bound
from .sources import DistributedStressSource


@dataclass(frozen=True)
class BoundaryConditions:
    top: str = "free"
    bottom: str = "absorbing"
    left: str = "absorbing"
    right: str = "absorbing"

    def __post_init__(self):
        if any(
            getattr(self, face) not in ("free", "absorbing")
            for face in ("top", "bottom", "left", "right")
        ):
            raise ValueError("Each face must be 'free' or 'absorbing'.")


@dataclass(frozen=True)
class RectangularModel:
    vp: np.ndarray
    vs: np.ndarray
    rho: np.ndarray
    lam_NN: np.ndarray
    l2m_NN: np.ndarray
    mu_NN: np.ndarray
    mu_MM: np.ndarray
    rho_vx: np.ndarray
    rho_vz: np.ndarray

    def __post_init__(self):
        for name in (
            "vp",
            "vs",
            "rho",
            "lam_NN",
            "l2m_NN",
            "mu_NN",
            "mu_MM",
            "rho_vx",
            "rho_vz",
        ):
            getattr(self, name).flags.writeable = False

    @property
    def vx_shape(self):
        return self.rho_vx.shape

    @property
    def vz_shape(self):
        return self.rho_vz.shape


def prepare_rectangular_model(vp, vs, rho):
    """Same isotropic material convention as the reference, bounded in x/z."""
    vp, vs, rho = (
        np.array(a, dtype=float, order="C", copy=True) for a in (vp, vs, rho)
    )
    _validate_base_model(vp, vs, rho)
    if vp.shape[0] < 9:
        raise ValueError("The bounded x N-grid needs at least 9 points.")
    mu = rho * vs**2
    return RectangularModel(
        vp,
        vs,
        rho,
        rho * (vp**2 - 2 * vs**2),
        rho * vp**2,
        mu,
        0.25 * (mu[:-1, :-1] + mu[1:, :-1] + mu[:-1, 1:] + mu[1:, 1:]),
        0.5 * (rho[:-1, :] + rho[1:, :]),
        0.5 * (rho[:, :-1] + rho[:, 1:]),
    )


def _material_average_transpose(g):
    out = np.zeros((g.shape[0] + 1, g.shape[1] + 1))
    out[:-1, :-1] += 0.25 * g
    out[1:, :-1] += 0.25 * g
    out[:-1, 1:] += 0.25 * g
    out[1:, 1:] += 0.25 * g
    return out


def rectangular_timestep_bound(model, op_x, op_z):
    """Sufficient strict bound for the positive leapfrog energy quadratic."""
    if model.vp.shape != (op_x.n_N, op_z.n_N):
        raise ValueError("Model and bounded SBP grid dimensions differ.")

    def bound(op):
        return max(
            _weighted_derivative_bound(op.D_M_to_N, op.H_N, op.H_M),
            _weighted_derivative_bound(op.D_N_to_M, op.H_M, op.H_N),
        )

    stiffness = max(
        float((model.l2m_NN + model.lam_NN).max()),
        float((model.l2m_NN - model.lam_NN).max()),
        2 * float(model.mu_MM.max()),
    )
    density = min(float(model.rho_vx.min()), float(model.rho_vz.min()))
    omega = np.sqrt(stiffness / density) * np.hypot(bound(op_x), bound(op_z))
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError("Cannot certify a finite rectangular timestep.")
    return float(2 / omega)


class RectangularSBPSAT:
    """NumPy reference with exact discrete source transpose and model gradient.

    This is an opt-in bounded-domain reference. It does not change run_forward
    in scripts/safod or automatically map earthquake moment to grid forcing.
    """

    def __init__(self, model, dx, dz, dt, boundaries=None, *, sparse_derivatives=False):
        self._model = model
        self.op_x = staggered_sbp_4_2(model.vp.shape[0] - 1, dx)
        self.op_z = staggered_sbp_4_2(model.vp.shape[1] - 1, dz)
        self._boundaries = BoundaryConditions() if boundaries is None else boundaries
        if not isinstance(self.boundaries, BoundaryConditions):
            raise TypeError("boundaries must be BoundaryConditions.")
        self.dt_limit = rectangular_timestep_bound(model, self.op_x, self.op_z)
        if not np.isfinite(dt) or not 0 < dt < self.dt_limit:
            raise ValueError(
                f"Require 0 < dt < {self.dt_limit:.12g} s (sufficient bound)."
            )
        self._dt = float(dt)
        x, z = self.op_x, self.op_z
        self.wx = x.H_M[:, None] * z.H_N[None, :]
        self.wz = x.H_N[:, None] * z.H_M[None, :]
        self.wn = x.H_N[:, None] * z.H_N[None, :]
        self.ws = x.H_M[:, None] * z.H_M[None, :]
        self._Z = self._impedances()
        self._prepare_velocity_solve()
        # Same coefficients and SATs; CSR only avoids dense O(nx*nz*(nx+nz))
        # derivative products in event runs. Keep the dense SBP certificate.
        self.sparse_derivatives = bool(sparse_derivatives)
        convert = np.asarray
        if sparse_derivatives:
            from scipy.sparse import csr_matrix

            convert = csr_matrix
        self._dx_mn = convert(x.D_M_to_N)
        self._dx_nm = convert(x.D_N_to_M)
        self._dz_mn = convert(z.D_M_to_N)
        self._dz_nm = convert(z.D_N_to_M)

    @property
    def model(self):
        return self._model

    @property
    def dt(self):
        return self._dt

    @property
    def boundaries(self):
        return self._boundaries

    def zero_state(self):
        m = self.model
        return SBPSATState(
            np.zeros(m.vx_shape),
            np.zeros(m.vz_shape),
            np.zeros(m.vp.shape),
            np.zeros(m.vp.shape),
            np.zeros(m.mu_MM.shape),
        )

    def _impedances(self):
        m = self.model
        p, s = m.rho * m.vp, m.rho * m.vs
        Z = {
            "left_p": p[0, :],
            "right_p": p[-1, :],
            "top_p": p[:, 0],
            "bottom_p": p[:, -1],
            "left_s": 0.5 * (s[0, :-1] + s[0, 1:]),
            "right_s": 0.5 * (s[-1, :-1] + s[-1, 1:]),
            "top_s": 0.5 * (s[:-1, 0] + s[1:, 0]),
            "bottom_s": 0.5 * (s[:-1, -1] + s[1:, -1]),
        }
        return {
            name: value.copy()
            if getattr(self.boundaries, name.split("_")[0]) == "absorbing"
            else np.zeros_like(value)
            for name, value in Z.items()
        }

    def force(self, state):
        """Undamped divergence plus traction-free SATs BEFORE rho division."""
        x, z = self.op_x, self.op_z
        fx = self._dx_nm @ state.sxx + (self._dz_mn @ state.sxz.T).T
        fz = self._dx_mn @ state.sxz + (self._dz_nm @ state.szz.T).T
        fx += (x.p_left / x.H_M)[:, None] * state.sxx[0, :]
        fx -= (x.p_right / x.H_M)[:, None] * state.sxx[-1, :]
        fx[:, 0] += (state.sxz @ z.p_left) / z.H_N[0]
        fx[:, -1] -= (state.sxz @ z.p_right) / z.H_N[-1]
        fz[0, :] += (x.p_left @ state.sxz) / x.H_N[0]
        fz[-1, :] -= (x.p_right @ state.sxz) / x.H_N[-1]
        fz += state.szz[:, 0, None] * (z.p_left / z.H_M)
        fz -= state.szz[:, -1, None] * (z.p_right / z.H_M)
        return fx, fz

    def force_transpose(self, qx, qz):
        x, z = self.op_x, self.op_z
        xx = self._dx_nm.T @ qx
        xx[0, :] += (x.p_left / x.H_M) @ qx
        xx[-1, :] -= (x.p_right / x.H_M) @ qx
        zz = (self._dz_nm.T @ qz.T).T
        zz[:, 0] += qz @ (z.p_left / z.H_M)
        zz[:, -1] -= qz @ (z.p_right / z.H_M)
        xz = (self._dz_mn.T @ qx.T).T + self._dx_mn.T @ qz
        xz += qx[:, 0, None] * (z.p_left / z.H_N[0])
        xz -= qx[:, -1, None] * (z.p_right / z.H_N[-1])
        xz += (x.p_left / x.H_N[0])[:, None] * qz[0, :]
        xz -= (x.p_right / x.H_N[-1])[:, None] * qz[-1, :]
        return xx, zz, xz

    def strains(self, vx, vz):
        return (
            self._dx_mn @ vx,
            (self._dz_mn @ vz.T).T,
            (self._dz_nm @ vx.T).T + self._dx_nm @ vz,
        )

    def stress_rhs(self, vx, vz):
        ex, ez, gamma = self.strains(vx, vz)
        m = self.model
        return (
            m.l2m_NN * ex + m.lam_NN * ez,
            m.lam_NN * ex + m.l2m_NN * ez,
            m.mu_MM * gamma,
        )

    def stress_transpose(self, xx, zz, xz):
        m = self.model
        ex = m.l2m_NN * xx + m.lam_NN * zz
        ez = m.lam_NN * xx + m.l2m_NN * zz
        gamma = m.mu_MM * xz
        return (
            self._dx_mn.T @ ex + (self._dz_nm.T @ gamma.T).T,
            (self._dz_mn.T @ ez.T).T + self._dx_nm.T @ gamma,
        )

    def damping_force(self, vx, vz):
        """D v, where volume_quadrature*D is symmetric positive semidefinite."""
        x, z, Z = self.op_x, self.op_z, self._Z
        cx, cz = self._tangent_x * vx, self._tangent_z * vz
        for side, p in (("left", x.p_left), ("right", x.p_right)):
            cx += (p / x.H_M)[:, None] * (Z[side + "_p"] * (p @ vx))
        for side, p in (("top", z.p_left), ("bottom", z.p_right)):
            cz += (Z[side + "_p"] * (vz @ p))[:, None] * (p / z.H_M)
        return cx, cz

    def damping_transpose(self, qx, qz):
        cx, cz = self.damping_force(qx / self.wx, qz / self.wz)
        return self.wx * cx, self.wz * cz

    def _prepare_velocity_solve(self):
        """Diagonal + disjoint boundary rank-one solves; no global factorization."""
        m, x, z, Z = self.model, self.op_x, self.op_z, self._Z
        self._tangent_x = np.zeros(m.vx_shape)
        self._tangent_z = np.zeros(m.vz_shape)
        self._tangent_x[:, 0] = Z["top_s"] / z.H_N[0]
        self._tangent_x[:, -1] = Z["bottom_s"] / z.H_N[-1]
        self._tangent_z[0, :] = Z["left_s"] / x.H_N[0]
        self._tangent_z[-1, :] = Z["right_s"] / x.H_N[-1]
        alpha = self.dt / 2
        self._inverse_x = 1 / (m.rho_vx + alpha * self._tangent_x)
        self._inverse_z = 1 / (m.rho_vz + alpha * self._tangent_z)
        self._rank_x, self._rank_z = [], []
        for side, p in (("left", x.p_left), ("right", x.p_right)):
            u = self._inverse_x * (p / x.H_M)[:, None]
            a = alpha * Z[side + "_p"]
            self._rank_x.append((p, u, a / (1 + a * (p @ u))))
        for side, p in (("top", z.p_left), ("bottom", z.p_right)):
            u = self._inverse_z * (p / z.H_M)
            a = alpha * Z[side + "_p"]
            self._rank_z.append((p, u, a / (1 + a * (u @ p))))

    def solve_velocity(self, bx, bz):
        """Solve (rho + dt*D/2) v = b on each native velocity grid."""
        vx, vz = bx * self._inverse_x, bz * self._inverse_z
        for p, u, c in self._rank_x:
            vx -= u * (c * (p @ vx))
        for p, u, c in self._rank_z:
            vz -= u * (c * (vz @ p))[:, None]
        return vx, vz

    def solve_velocity_transpose(self, qx, qz):
        x, z = self.solve_velocity(qx / self.wx, qz / self.wz)
        return self.wx * x, self.wz * z

    def _validate_state(self, state):
        shapes = (
            self.model.vx_shape,
            self.model.vz_shape,
            self.model.vp.shape,
            self.model.vp.shape,
            self.model.mu_MM.shape,
        )
        for name, shape in zip(("vx", "vz", "sxx", "szz", "sxz"), shapes):
            a = getattr(state, name)
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError(f"{name} must be finite with native shape {shape}.")

    def _validate_source(self, source):
        if isinstance(source, DistributedStressSource):
            source.validate_grid(self)
            return
        if not isinstance(source, StressPointSource):
            raise TypeError("source must be StressPointSource (grid stress rate).")
        if not (1 <= source.ix < self.op_x.n_M and 1 <= source.iz < self.op_z.n_M):
            raise ValueError(
                "Source requires interior N-N support; no clipping at a boundary."
            )

    def _validate_receivers(self, receivers):
        if len(receivers) == 0:
            raise ValueError("At least one receiver/gauge centre is required.")
        if isinstance(receivers, DASObservation):
            if (
                receivers.vx_shape != self.model.vx_shape
                or receivers.vz_shape != self.model.vz_shape
                or receivers.dx != self.op_x.h
                or receivers.dz != self.op_z.h
            ):
                raise ValueError("DAS observation and bounded grid differ.")
        else:
            for r in receivers:
                if not isinstance(r, PointReceiver):
                    raise TypeError("Expected native PointReceivers or DASObservation.")
                shape = (
                    self.model.vx_shape if r.component == "vx" else self.model.vz_shape
                )
                if not (0 <= r.ix < shape[0] and 0 <= r.iz < shape[1]):
                    raise ValueError("Point receiver is outside its native grid.")

    @staticmethod
    def _sample(vx, vz, receivers):
        if isinstance(receivers, DASObservation):
            return receivers.sample(vx, vz)
        return np.array(
            [
                vx[r.ix, r.iz] if r.component == "vx" else vz[r.ix, r.iz]
                for r in receivers
            ]
        )

    @staticmethod
    def _scatter(q, receivers, vx, vz):
        if isinstance(receivers, DASObservation):
            receivers.scatter(q, vx, vz)
        else:
            for value, r in zip(q, receivers):
                field = vx if r.component == "vx" else vz
                field[r.ix, r.iz] += value

    def _advance(self, state, source=None, amplitude=0.0):
        m, dt = self.model, self.dt
        fx, fz = self.force(state)
        cx, cz = self.damping_force(state.vx, state.vz)
        vx, vz = self.solve_velocity(
            m.rho_vx * state.vx - 0.5 * dt * cx + dt * fx,
            m.rho_vz * state.vz - 0.5 * dt * cz + dt * fz,
        )
        xx, zz, xz = self.stress_rhs(vx, vz)
        xx, zz, xz = state.sxx + dt * xx, state.szz + dt * zz, state.sxz + dt * xz
        if source is not None:
            if isinstance(source, DistributedStressSource):
                source.inject(xx, zz, xz, dt * amplitude)
            else:
                _inject_source(xx, zz, xz, source, dt, amplitude)
        return SBPSATState(vx, vz, xx, zz, xz)

    def step(self, state, *, source=None, amplitude=0.0):
        self._validate_state(state)
        if source is not None:
            self._validate_source(source)
        if not np.isfinite(amplitude):
            raise ValueError("Source amplitude must be finite.")
        return self._advance(state, source, amplitude)

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
        state = self.zero_state() if initial_state is None else initial_state.copy()
        self._validate_state(state)
        data = np.empty((len(receivers), stf.size))
        history = [] if store_velocity_history else None
        for it, amplitude in enumerate(stf):
            state = self._advance(state, source, amplitude)
            data[:, it] = self._sample(state.vx, state.vz, receivers)
            if history is not None:
                history.append((state.vx.copy(), state.vz.copy()))
        return ForwardResult(data, state, None if history is None else tuple(history))

    def _transpose_step(self, adj, q, receivers):
        dt, m = self.dt, self.model
        x, z = self.stress_transpose(adj.sxx, adj.szz, adj.sxz)
        vx, vz = adj.vx + dt * x, adj.vz + dt * z
        self._scatter(q, receivers, vx, vz)
        rx, rz = self.solve_velocity_transpose(vx, vz)
        cx, cz = self.damping_transpose(rx, rz)
        old_vx, old_vz = m.rho_vx * rx - 0.5 * dt * cx, m.rho_vz * rz - 0.5 * dt * cz
        xx, zz, xz = self.force_transpose(rx, rz)
        return SBPSATState(
            old_vx, old_vz, adj.sxx + dt * xx, adj.szz + dt * zz, adj.sxz + dt * xz
        ), (rx, rz)

    def run_adjoint_to_source(self, q_data, source, receivers):
        q_data = np.asarray(q_data, dtype=float)
        self._validate_source(source)
        self._validate_receivers(receivers)
        if (
            q_data.ndim != 2
            or q_data.shape[0] != len(receivers)
            or q_data.shape[1] < 1
            or not np.isfinite(q_data).all()
        ):
            raise ValueError("Adjoint data need finite shape (n_receivers,nt), nt>=1.")
        adj = self.zero_state()
        gradient = np.zeros(q_data.shape[1])
        for it in range(q_data.shape[1] - 1, -1, -1):
            gradient[it] = (
                self.dt * source.gather(adj.sxx, adj.szz, adj.sxz)
                if isinstance(source, DistributedStressSource)
                else _source_transpose(adj.sxx, adj.szz, adj.sxz, source, self.dt)
            )
            adj, _ = self._transpose_step(adj, q_data[:, it], receivers)
        return gradient

    def _impedance_vjp(self, vx, vz, rx, rz):
        """-dt * adjoint_rhs^T (partial D) v_average, rho fixed."""
        x, z, dt, m = self.op_x, self.op_z, self.dt, self.model
        gp, gs = np.zeros_like(m.vp), np.zeros_like(m.vs)
        for side, p, index in (("left", x.p_left, 0), ("right", x.p_right, -1)):
            if getattr(self.boundaries, side) == "free":
                continue
            normal = -dt * (p @ vx) * ((p / x.H_M) @ rx)
            tangent = -dt * vz[index, :] * rz[index, :] / x.H_N[index]
            gp[index, :] += m.rho[index, :] * normal
            gs[index, :-1] += 0.5 * m.rho[index, :-1] * tangent
            gs[index, 1:] += 0.5 * m.rho[index, 1:] * tangent
        for side, p, index in (("top", z.p_left, 0), ("bottom", z.p_right, -1)):
            if getattr(self.boundaries, side) == "free":
                continue
            normal = -dt * (vz @ p) * (rz @ (p / z.H_M))
            tangent = -dt * vx[:, index] * rx[:, index] / z.H_N[index]
            gp[:, index] += m.rho[:, index] * normal
            gs[:-1, index] += 0.5 * m.rho[:-1, index] * tangent
            gs[1:, index] += 0.5 * m.rho[1:, index] * tangent
        return gp, gs

    def _new_model_gradient(self):
        m = self.model
        return [
            np.zeros_like(m.vp), np.zeros_like(m.vp), np.zeros_like(m.mu_MM),
            np.zeros_like(m.vp), np.zeros_like(m.vs),
        ]

    def _reverse_model_gradient_step(
        self, adj, q, receivers, vx, vz, old_x, old_z, gradients
    ):
        """One original reverse/VJP step, shared by both history strategies."""
        gl, ga, gm, bp, bs = gradients
        dt = self.dt
        ex, ez, gamma = self.strains(vx, vz)
        gl += dt * (adj.sxx * ez + adj.szz * ex)
        ga += dt * (adj.sxx * ex + adj.szz * ez)
        gm += dt * adj.sxz * gamma
        adj, (rx, rz) = self._transpose_step(adj, q, receivers)
        p, s = self._impedance_vjp(0.5 * (old_x + vx), 0.5 * (old_z + vz), rx, rz)
        bp += p
        bs += s
        return adj

    def _finish_model_gradient(self, gradients):
        gl, ga, gm, bp, bs = gradients
        m = self.model
        gp = 2 * m.rho * m.vp**2 * (gl + ga) + m.vp * bp
        gs = (
            -4 * m.rho * m.vs**2 * gl
            + 2 * m.rho * m.vs**2 * _material_average_transpose(gm)
            + m.vs * bs
        )
        return gp, gs

    def objective_and_gradient_log_vp_vs(
        self, stf, observed, source, receivers, *, checkpoint_interval=None
    ):
        """J=.5*dt*||data-observed||²; includes all model-dependent impedances.

        None retains the full-history reference. "auto" or a positive integer
        selects exact checkpoint/replay, with the same return convention.
        """
        if checkpoint_interval is not None:
            from .checkpointing import objective_and_gradient_checkpointed

            return objective_and_gradient_checkpointed(
                self, stf, observed, source, receivers, interval=checkpoint_interval
            )
        fwd = self.run_forward(stf, source, receivers, store_velocity_history=True)
        observed = np.asarray(observed, dtype=float)
        if observed.shape != fwd.data.shape or not np.isfinite(observed).all():
            raise ValueError("Observed data must be finite and match synthetic shape.")
        residual = fwd.data - observed
        J = 0.5 * self.dt * float(np.sum(residual**2))
        gradients = self._new_model_gradient()
        adj = self.zero_state()
        for it in range(fwd.data.shape[1] - 1, -1, -1):
            vx, vz = fwd.velocity_history[it]
            old_x, old_z = (
                fwd.velocity_history[it - 1]
                if it
                else (np.zeros_like(vx), np.zeros_like(vz))
            )
            adj = self._reverse_model_gradient_step(
                adj, self.dt * residual[:, it], receivers,
                vx, vz, old_x, old_z, gradients,
            )
        gp, gs = self._finish_model_gradient(gradients)
        return J, fwd.data, gp, gs

    def energy(self, state, *, modified=True):
        m = self.model
        det = m.l2m_NN**2 - m.lam_NN**2
        value = 0.5 * float(
            np.sum(self.wx * m.rho_vx * state.vx**2)
            + np.sum(self.wz * m.rho_vz * state.vz**2)
            + np.sum(
                self.wn
                * (
                    m.l2m_NN * (state.sxx**2 + state.szz**2)
                    - 2 * m.lam_NN * state.sxx * state.szz
                )
                / det
            )
            + np.sum(self.ws * state.sxz**2 / m.mu_MM)
        )
        if modified:
            fx, fz = self.force(state)
            value += (
                0.5
                * self.dt
                * float(
                    np.sum(self.wx * state.vx * fx) + np.sum(self.wz * state.vz * fz)
                )
            )
        return value

    def dissipation(self, vx, vz):
        cx, cz = self.damping_force(vx, vz)
        return float(np.sum(self.wx * vx * cx) + np.sum(self.wz * vz * cz))
