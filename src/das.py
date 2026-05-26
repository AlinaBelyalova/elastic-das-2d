# ==============================================================================
# src/das.py — DAS forward operator for elastic FWI
#
# Computes axial strain or strain-rate from Cartesian wavefields.
#
# ── Physical formulation ─────────────────────────────────────────────────────
# DAS measures the change in optical path length between two Rayleigh
# backscatter points separated by the gauge length L. For a cable element
# centred at arc-length s:
#
#   ε̇_ss(s) = (
#       [vx(s+L/2) - vx(s-L/2)] * tx(s)
#     + [vz(s+L/2) - vz(s-L/2)] * tz(s)
#   ) / L
#
# where tx(s), tz(s) are the unit tangent components at the CENTRE of the gauge.
#
# Why "difference first, then project" matters:
#   If we project FIRST and then difference (the intuitive but wrong order),
#   a rigid translation (vx=const, vz=const) produces a non-zero result on a
#   curved cable because the tangent vector changes along the cable.
#   The correct formulation is:
#       1. difference in Cartesian components over the gauge
#       2. project the difference onto the local tangent at the gauge centre
#
# ── Adjoint operator (for FWI gradient) ──────────────────────────────────────
#   The forward operator F maps (vx, vz) ∈ R^{nrec×nt} → d ∈ R^{nchan_out×nt}.
#   Its adjoint F^T maps a residual r ∈ R^{nchan_out×nt} → (qx, qz) ∈ R^{nrec×nt}.
#   This scatter-back operator is included for future FWI use and is verified
#   by a dot-product test in _self_test().
#
# Design choice:
#   This implementation requires an EVEN number of gauge samples so that the
#   spatial finite-difference stencil is symmetric about the channel centre.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.receivers import Receivers2D


# ==============================================================================
# RESULT CONTAINER
# ==============================================================================

@dataclass(frozen=True)
class DASResult:
    """
    Output of a DAS forward operator.

    Attributes
    ----------
    data :
        DAS observable of shape (nchan_out, nt).
        Typically axial strain or axial strain-rate.
    channel_indices :
        Indices into the original receiver array for which the centred gauge
        operator is defined.
    gauge_samples :
        Integer number of receiver spacings spanning the gauge.
        Must be even in this implementation.
    gauge_length_m :
        Effective physical gauge length [m].
    kind :
        'axial_strain' or 'axial_strain_rate'.
    """
    data: np.ndarray
    channel_indices: np.ndarray
    gauge_samples: int
    gauge_length_m: float
    kind: str

    def __post_init__(self) -> None:
        data = np.asarray(self.data, dtype=np.float64)
        cidx = np.asarray(self.channel_indices, dtype=np.int64)

        if data.ndim != 2:
            raise ValueError(f"data must be 2D, got shape {data.shape}.")
        if cidx.ndim != 1:
            raise ValueError(f"channel_indices must be 1D, got shape {cidx.shape}.")
        if data.shape[0] != cidx.size:
            raise ValueError(
                f"data.shape[0]={data.shape[0]} does not match "
                f"len(channel_indices)={cidx.size}."
            )
        if self.gauge_samples < 2:
            raise ValueError("gauge_samples must be >= 2.")
        if self.gauge_samples % 2 != 0:
            raise ValueError(
                f"gauge_samples must be an EVEN integer for a symmetric spatial "
                f"difference, got {self.gauge_samples}."
            )
        if self.gauge_length_m <= 0.0:
            raise ValueError("gauge_length_m must be positive.")
        if self.kind not in {"axial_strain", "axial_strain_rate"}:
            raise ValueError(
                f"kind must be 'axial_strain' or 'axial_strain_rate', got {self.kind!r}."
            )

        object.__setattr__(self, "data", data)
        object.__setattr__(self, "channel_indices", cidx)

    @property
    def nchan_out(self) -> int:
        return int(self.data.shape[0])

    @property
    def nt(self) -> int:
        return int(self.data.shape[1])


# ==============================================================================
# CORE ENGINE
# ==============================================================================

def _compute_das_observable(
    x_component: np.ndarray,
    z_component: np.ndarray,
    receivers: Receivers2D,
    gauge_length_m: float,
    kind: str,
) -> DASResult:
    """
    Core DAS engine: finite difference first, projection second.

    Parameters
    ----------
    x_component, z_component :
        Arrays of shape (nrec, nt), e.g. velocity or displacement components
        sampled at receiver centres.
    receivers :
        Receivers2D geometry object.
    gauge_length_m :
        Physical gauge length [m].
    kind :
        Output label.

    Returns
    -------
    DASResult

    Notes
    -----
    This implementation requires an even discrete gauge length so that the
    difference stencil remains symmetric about the channel centre.
    """
    x_component = np.asarray(x_component, dtype=np.float64)
    z_component = np.asarray(z_component, dtype=np.float64)

    if x_component.shape != z_component.shape:
        raise ValueError(
            f"x_component and z_component must have the same shape; "
            f"got {x_component.shape} vs {z_component.shape}."
        )
    if x_component.ndim != 2:
        raise ValueError(
            f"Input arrays must be 2D (nrec, nt); got shape {x_component.shape}."
        )
    if x_component.shape[0] != receivers.nrec:
        raise ValueError(
            f"First dimension ({x_component.shape[0]}) must match "
            f"receivers.nrec ({receivers.nrec})."
        )

    gauge_k = receivers.gauge_samples(gauge_length_m)
    if gauge_k % 2 != 0:
        raise ValueError(
            f"Requested gauge length results in an odd number of samples ({gauge_k}). "
            f"This implementation requires an even number of samples for a "
            f"symmetric centred operator."
        )

    ds = receivers.channel_spacing
    half_step = gauge_k // 2

    start = half_step
    stop = receivers.nrec - half_step
    if stop <= start:
        raise ValueError(
            f"Gauge is too long for the receiver array: "
            f"nrec={receivers.nrec}, gauge_samples={gauge_k}."
        )

    centre_idx = np.arange(start, stop, dtype=np.int64)
    left_idx = centre_idx - half_step
    right_idx = centre_idx + half_step

    gauge_length_eff_m = float(gauge_k * ds)

    # Step 1: difference in Cartesian components over the gauge
    dx_comp = x_component[right_idx, :] - x_component[left_idx, :]
    dz_comp = z_component[right_idx, :] - z_component[left_idx, :]

    # Step 2: project the difference onto the local tangent at the gauge centre
    tx_c = receivers.tx[centre_idx, None]
    tz_c = receivers.tz[centre_idx, None]

    data = (dx_comp * tx_c + dz_comp * tz_c) / gauge_length_eff_m

    return DASResult(
        data=data,
        channel_indices=centre_idx,
        gauge_samples=gauge_k,
        gauge_length_m=gauge_length_eff_m,
        kind=kind,
    )


# ==============================================================================
# ADJOINT OPERATOR
# ==============================================================================

def das_adjoint(
    residual: np.ndarray,
    receivers: Receivers2D,
    result: DASResult,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Adjoint of the DAS forward operator.

    Parameters
    ----------
    residual :
        DAS residual of shape (nchan_out, nt).
    receivers :
        Receivers2D object.
    result :
        DASResult returned by the corresponding forward call.

    Returns
    -------
    qx, qz :
        Arrays of shape (nrec, nt) containing adjoint Cartesian components.
    """
    residual = np.asarray(residual, dtype=np.float64)
    if residual.shape != result.data.shape:
        raise ValueError(
            f"residual shape {residual.shape} does not match "
            f"result.data shape {result.data.shape}."
        )

    nt = residual.shape[1]
    nrec = receivers.nrec
    cidx = result.channel_indices
    gauge_k = result.gauge_samples
    L_eff = result.gauge_length_m

    half_step = gauge_k // 2
    left_idx = cidx - half_step
    right_idx = cidx + half_step

    tx_c = receivers.tx[cidx, None]
    tz_c = receivers.tz[cidx, None]
    scaled = residual / L_eff

    # Оптимизация: предрасчет умножений для предотвращения повторных аллокаций
    tx_scaled = tx_c * scaled
    tz_scaled = tz_c * scaled

    qx = np.zeros((nrec, nt), dtype=np.float64)
    qz = np.zeros((nrec, nt), dtype=np.float64)

    np.add.at(qx, right_idx,  tx_scaled)
    np.add.at(qx, left_idx,  -tx_scaled)
    np.add.at(qz, right_idx,  tz_scaled)
    np.add.at(qz, left_idx,  -tz_scaled)

    return qx, qz


# ==============================================================================
# DIAGNOSTIC PROJECTION ONLY
# ==============================================================================

def project_to_fibre_axis(
    x_component: np.ndarray,
    z_component: np.ndarray,
    receivers: Receivers2D,
) -> np.ndarray:
    """
    Project a Cartesian wavefield onto the local fibre axis at each channel.

    Returns
    -------
    parallel_component :
        Array of shape (nrec, nt):
            tx[i] * x_component[i, :] + tz[i] * z_component[i, :]

    Notes
    -----
    This is provided for diagnostics only. It is NOT the DAS forward operator.
    The actual DAS operator uses difference first, then projection.
    """
    x_component = np.asarray(x_component, dtype=np.float64)
    z_component = np.asarray(z_component, dtype=np.float64)

    if x_component.shape != z_component.shape:
        raise ValueError("x_component and z_component must have the same shape.")
    if x_component.ndim != 2:
        raise ValueError(f"Input must be 2D (nrec, nt); got {x_component.shape}.")
    if x_component.shape[0] != receivers.nrec:
        raise ValueError(
            f"First dimension ({x_component.shape[0]}) must match "
            f"receivers.nrec ({receivers.nrec})."
        )

    return receivers.tx[:, None] * x_component + receivers.tz[:, None] * z_component


# ==============================================================================
# PUBLIC API
# ==============================================================================

def compute_axial_strain_rate(
    vx: np.ndarray,
    vz: np.ndarray,
    receivers: Receivers2D,
    gauge_length_m: float,
) -> DASResult:
    """
    Compute DAS axial strain-rate from receiver-sampled particle velocities.
    """
    return _compute_das_observable(
        vx, vz, receivers, gauge_length_m, kind="axial_strain_rate"
    )


def compute_axial_strain(
    ux: np.ndarray,
    uz: np.ndarray,
    receivers: Receivers2D,
    gauge_length_m: float,
) -> DASResult:
    """
    Compute DAS axial strain from receiver-sampled displacements.
    """
    return _compute_das_observable(
        ux, uz, receivers, gauge_length_m, kind="axial_strain"
    )


# ==============================================================================
# SELF-TEST
# ==============================================================================

def _self_test() -> None:
    def make_straight(nrec: int) -> Receivers2D:
        s = np.arange(nrec, dtype=np.float64)
        tx = np.ones(nrec, dtype=np.float64)
        tz = np.zeros(nrec, dtype=np.float64)
        return Receivers2D(
            x=s,
            z=np.zeros(nrec, dtype=np.float64),
            ix=np.arange(nrec, dtype=np.int64),
            iz=np.zeros(nrec, dtype=np.int64),
            tx=tx,
            tz=tz,
            s=s,
        )

    def make_bent() -> Receivers2D:
        """L-cable: 6 horizontal + 5 vertical channels."""
        nrec = 11
        x = np.array([0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5], dtype=np.float64)
        z = np.array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5], dtype=np.float64)
        
        # Более чистый синтаксис вместо вложенных списков
        distances = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(z) ** 2))
        s = np.insert(distances, 0, 0.0)
        
        tx = np.array([1, 1, 1, 1, 1, 1 / np.sqrt(2), 0, 0, 0, 0, 0], dtype=np.float64)
        tz = np.array([0, 0, 0, 0, 0, 1 / np.sqrt(2), 1, 1, 1, 1, 1], dtype=np.float64)
        return Receivers2D(
            x=x,
            z=z,
            ix=np.zeros(nrec, dtype=np.int64),
            iz=np.zeros(nrec, dtype=np.int64),
            tx=tx,
            tz=tz,
            s=s,
        )

    nt = 50

    # 1. Rigid translation on bent cable -> zero strain-rate
    rec_bent = make_bent()
    vx = np.ones((rec_bent.nrec, nt), dtype=np.float64) * 3.0
    vz = np.ones((rec_bent.nrec, nt), dtype=np.float64) * 2.0
    res = compute_axial_strain_rate(vx, vz, rec_bent, gauge_length_m=2.0)
    assert np.allclose(res.data, 0.0, atol=1e-15), (
        f"Rigid translation should give zero strain; "
        f"max={np.abs(res.data).max():.2e}"
    )
    print("Rigid translation on bent cable: OK")

    # 2. Linear strain field on straight cable -> exact recovery
    rec_st = make_straight(21)
    alpha = 2.5e-6
    ux = alpha * rec_st.s[:, None] * np.ones((1, nt), dtype=np.float64)
    uz = np.zeros_like(ux)
    res2 = compute_axial_strain(ux, uz, rec_st, gauge_length_m=4.0)
    assert np.allclose(res2.data, alpha, atol=1e-12), (
        f"Linear strain should recover alpha; "
        f"max_err={np.abs(res2.data - alpha).max():.2e}"
    )
    print("Linear strain recovery on straight cable: OK")

    # 3. Linear strain-rate field on straight cable -> exact recovery
    beta = 1.2e-7
    vx2 = beta * rec_st.s[:, None] * np.ones((1, nt), dtype=np.float64)
    vz2 = np.zeros_like(vx2)
    res3 = compute_axial_strain_rate(vx2, vz2, rec_st, gauge_length_m=4.0)
    assert np.allclose(res3.data, beta, atol=1e-12)
    print("Linear strain-rate recovery on straight cable: OK")

    # 4. Dot-product test: <F(m), r> = <m, F^T(r)>
    rng = np.random.default_rng(42)
    rec = make_straight(31)
    vx_m = rng.standard_normal((rec.nrec, nt))
    vz_m = rng.standard_normal((rec.nrec, nt))
    res_fwd = compute_axial_strain_rate(vx_m, vz_m, rec, gauge_length_m=4.0)
    r_rnd = rng.standard_normal(res_fwd.data.shape)
    qx, qz = das_adjoint(r_rnd, rec, res_fwd)

    lhs = float(np.vdot(res_fwd.data, r_rnd))
    rhs = float(np.vdot(vx_m, qx) + np.vdot(vz_m, qz))
    err = abs(lhs - rhs) / (abs(lhs) + 1e-30)
    assert err < 1e-12, (
        f"Dot-product test failed: lhs={lhs:.6e}, rhs={rhs:.6e}, err={err:.2e}"
    )
    print("Adjoint dot-product test: OK")

    # 5. Shape mismatch guard
    try:
        compute_axial_strain(ux[:-1], uz, rec_st, gauge_length_m=2.0)
        raise AssertionError("Expected ValueError.")
    except ValueError:
        pass
    print("Shape mismatch guard: OK")

    # 6. DASResult validation: mismatched channel count
    try:
        DASResult(
            data=np.zeros((5, 10), dtype=np.float64),
            channel_indices=np.arange(6, dtype=np.int64),
            gauge_samples=2,
            gauge_length_m=1.0,
            kind="axial_strain",
        )
        raise AssertionError("Expected ValueError.")
    except ValueError:
        pass
    print("DASResult shape validation: OK")

    # 7. DASResult validation: odd gauge samples
    try:
        DASResult(
            data=np.zeros((6, 10), dtype=np.float64),
            channel_indices=np.arange(6, dtype=np.int64),
            gauge_samples=3,
            gauge_length_m=1.5,
            kind="axial_strain",
        )
        raise AssertionError("Expected ValueError for odd gauge_samples.")
    except ValueError:
        pass
    print("DASResult symmetry guard: OK")

    print("\nSelf-test PASSED")


if __name__ == "__main__":
    _self_test()