"""Uniform full-state checkpoints and exact forward replay for the model VJP.

No backward wavefield reconstruction, interpolation, precision reduction or
change to the discrete adjoint. The independent full-history path is retained.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from numbers import Integral

import numpy as np


def _positive_integer(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


@dataclass(frozen=True)
class CheckpointPlan:
    nt: int
    interval: int
    checkpoint_count: int
    checkpoint_bytes: int
    block_velocity_bytes: int
    full_velocity_history_bytes: int

    @property
    def history_bytes(self):
        return self.checkpoint_bytes + self.block_velocity_bytes

    def summary(self):
        return {
            "nt": self.nt,
            "interval": self.interval,
            "checkpoint_count": self.checkpoint_count,
            "checkpoint_gib": self.checkpoint_bytes / 2**30,
            "block_velocity_gib": self.block_velocity_bytes / 2**30,
            "history_gib": self.history_bytes / 2**30,
            "full_velocity_history_gib": self.full_velocity_history_bytes / 2**30,
            "replay_steps": self.nt,
            "scope": "history arrays only; excludes operators, live/adjoint states, data, gradients and temporaries",
        }


def plan_checkpoints(nx, nz, nt, interval="auto"):
    """Minimize uniform-schedule history bytes, with all fields in float64.

    nx/nz are normal-stress node counts, not interval counts. Store complete
    states at steps 0,K,2K,... strictly before nt. One block retains K+1
    velocity pairs: its initial v_minus plus each resulting v_plus.
    """
    nx, nz, nt = (
        _positive_integer(v, n) for v, n in ((nx, "nx"), (nz, "nz"), (nt, "nt"))
    )
    if nx < 2 or nz < 2:
        raise ValueError("nx/nz must include at least two normal-stress nodes.")
    velocity_bytes = 8 * ((nx - 1) * nz + nx * (nz - 1))
    state_bytes = 8 * (2 * nx * nz + (nx - 1) * (nz - 1)) + velocity_bytes

    def cost(k):
        return ((nt + k - 1) // k) * state_bytes + (k + 1) * velocity_bytes

    if isinstance(interval, str) and interval == "auto":
        # Scalar O(nt) search, no arrays or solver construction. At 19,062
        # steps this is negligible and avoids an approximate sqrt heuristic.
        interval = min(range(1, nt + 1), key=cost)
    else:
        interval = _positive_integer(interval, "checkpoint interval")
        if interval > nt:
            raise ValueError("checkpoint interval must not exceed nt.")
    count = (nt + interval - 1) // interval
    return CheckpointPlan(
        nt,
        interval,
        count,
        count * state_bytes,
        (interval + 1) * velocity_bytes,
        nt * velocity_bytes,
    )


def objective_and_gradient_checkpointed(
    solver, stf, observed, source, receivers, *, interval="auto"
):
    """Same J, data, log-Vp and log-Vs gradients as the full-history method.

    Initial state is zero, matching the reference model-gradient API. Full
    state checkpoints are independent copies. Restarts advance FORWARD with
    exactly the original source indices and arithmetic. Each step is replayed
    once; the adjoint is run once. Reverse/VJP operations are the existing
    independent matrix implementation, including boundary impedances.
    """
    stf = np.asarray(stf, dtype=np.float64)
    if stf.ndim != 1 or not stf.size or not np.isfinite(stf).all():
        raise ValueError("STF must be a nonempty finite vector.")
    solver._validate_source(source)
    solver._validate_receivers(receivers)
    observed = np.asarray(observed, dtype=np.float64)
    if observed.shape != (len(receivers), len(stf)) or not np.isfinite(observed).all():
        raise ValueError("Observed data must be finite and match synthetic shape.")
    plan = plan_checkpoints(*solver.model.vp.shape, len(stf), interval)
    compiled = callable(getattr(solver, "advance_inplace", None))
    context = solver.thread_context() if compiled else nullcontext()

    def advance(state, amplitude):
        if compiled:
            return solver.advance_inplace(state, source, amplitude)
        return solver._advance(state, source, amplitude)

    state = solver.zero_state()
    data = np.empty(observed.shape, dtype=np.float64)
    checkpoints = []
    with context:
        for it, amplitude in enumerate(stf):
            if it % plan.interval == 0:
                checkpoints.append(state.copy())
            state = advance(state, amplitude)
            data[:, it] = solver._sample(state.vx, state.vz, receivers)
        # The final full state is not needed by the objective or gradient.
        del state
        residual = data - observed
        J = 0.5 * solver.dt * float(np.sum(residual**2))
        gradients = solver._new_model_gradient()
        adjoint = solver.zero_state()
        hx = np.empty((plan.interval + 1, *solver.model.vx_shape))
        hz = np.empty((plan.interval + 1, *solver.model.vz_shape))
        for block in range(plan.checkpoint_count - 1, -1, -1):
            start = block * plan.interval
            stop = min(start + plan.interval, len(stf))
            # Consume the independent checkpoint as an owned replay state;
            # there is no extra full-state copy at each block restart.
            state = checkpoints.pop()
            hx[0], hz[0] = state.vx, state.vz
            for j, it in enumerate(range(start, stop)):
                state = advance(state, stf[it])
                hx[j + 1], hz[j + 1] = state.vx, state.vz
            for j in range(stop - start - 1, -1, -1):
                it = start + j
                adjoint = solver._reverse_model_gradient_step(
                    adjoint,
                    solver.dt * residual[:, it],
                    receivers,
                    hx[j + 1],
                    hz[j + 1],
                    hx[j],
                    hz[j],
                    gradients,
                )
            del state
    gp, gs = solver._finish_model_gradient(gradients)
    return J, data, gp, gs
