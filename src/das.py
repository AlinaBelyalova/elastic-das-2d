# ==============================================================================
# src/das.py — DAS forward operator for elastic FWI
#
# Computes axial strain or strain-rate from Cartesian wavefields.
#
# Physical formulation
#   DAS measures the change in optical path length between two Rayleigh
#   backscatter points separated by the gauge length L. For a cable element
#   centred at arc-length s:
#
#     eps_dot_ss(s) = (
#         [vx(s+L/2) - vx(s-L/2)] * tx(s)
#       + [vz(s+L/2) - vz(s-L/2)] * tz(s)
#     ) / L
#
#   where tx(s), tz(s) are the unit tangent components at the CENTRE of the
#   gauge.
#
# Why "difference first, then project" matters
#   If we project FIRST and then difference (the intuitive but wrong order),
#   a rigid translation (vx=const, vz=const) produces a non-zero result on a
#   curved cable because the tangent vector changes along the cable.
#   The correct formulation is:
#       1. difference in Cartesian components over the gauge
#       2. project the difference onto the local tangent at the gauge centre
#
# Continuous gauge length (this implementation)
#   Real DAS interrogators apply a gauge length L as a continuous software
#   parameter, independent of the channel spacing ds. L is essentially never
#   an exact integer multiple of ds in field data (e.g. L=16.6213 m).
#
#   The two virtual measurement points s +/- L/2 therefore generally fall
#   BETWEEN channels. This implementation evaluates v(s +/- L/2) by LINEAR
#   interpolation along the (uniformly spaced) channel array, instead of
#   requiring s +/- L/2 to land exactly on a channel.
#
#   This removes the previous "even number of channel spacings" restriction
#   entirely. Any positive gauge_length_m that fits inside the cable works.
#
#   Two properties are preserved EXACTLY (not approximately) under linear
#   interpolation:
#     1. Rigid translation (vx, vz constant) -> zero strain-rate.
#        Linear interpolation of a constant field returns that same constant,
#        so the left/right difference is exactly zero regardless of L.
#     2. A field linear in arc length s is recovered exactly.
#        Linear interpolation has zero error for linear functions, so the
#        gauge-length finite difference reduces to the exact analytic slope,
#        independent of how L relates to ds.
#   Both are verified in _self_test().
#
#   When L happens to be an even integer multiple of ds (the only case the
#   previous implementation supported), interpolation weights are exactly
#   0 or 1 and this implementation reduces to the old exact-indexing result.
#
# Adjoint operator (for FWI gradient)
#   The forward operator F maps (vx, vz) in R^{nrec x nt} -> d in R^{nchan_out x nt}.
#   Its adjoint F^T maps a residual r in R^{nchan_out x nt} -> (qx, qz) in R^{nrec x nt}.
#   Because the forward operator is now interpolate -> difference -> project
#   (three linear steps), the adjoint is project -> difference -> scatter,
#   using the SAME interpolation indices/weights computed by the forward call.
#   Verified by a dot-product test in _self_test() for non-integer gauge
#   lengths, not just the old integer-multiple special case.
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
        Indices into the original receiver array used as gauge centres.
    gauge_samples :
        Equivalent number of channel spacings spanned by the gauge,
        gauge_length_m / channel_spacing. Generally NOT an integer.
        Informational only; not used to reconstruct the operator (the
        operator recomputes interpolation weights from gauge_length_m).
    gauge_length_m :
        Exact requested physical gauge length [m]. Unlike the previous
        implementation, this is never silently snapped to a channel-spacing
        multiple.
    kind :
        'axial_strain' or 'axial_strain_rate'.
    """
    data: np.ndarray
    channel_indices: np.ndarray
    gauge_samples: float
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
        if not np.isfinite(self.gauge_length_m) or self.gauge_length_m <= 0.0:
            raise ValueError(
                f"gauge_length_m must be finite and positive, got {self.gauge_length_m}."
            )
        if not np.isfinite(self.gauge_samples) or self.gauge_samples <= 0.0:
            raise ValueError(
                f"gauge_samples must be finite and positive, got {self.gauge_samples}."
            )
        if self.kind not in {"axial_strain", "axial_strain_rate"}:
            raise ValueError(
                f"kind must be 'axial_strain' or 'axial_strain_rate', got {self.kind!r}."
            )

        object.__setattr__(self, "data", data)
        object.__setattr__(self, "channel_indices", cidx)
        object.__setattr__(self, "gauge_samples", float(self.gauge_samples))
        object.__setattr__(self, "gauge_length_m", float(self.gauge_length_m))

    @property
    def nchan_out(self) -> int:
        return int(self.data.shape[0])

    @property
    def nt(self) -> int:
        return int(self.data.shape[1])


# ==============================================================================
# LINEAR INTERPOLATION ALONG THE CABLE
# ==============================================================================

def _interp_index_weight(
    s_query: np.ndarray,
    s0: float,
    ds: float,
    nrec: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fractional-channel index/weight for linear interpolation along a
    uniformly spaced arc-length array.

    For query positions s_query, returns (j, w) such that for any field
    v sampled at channels 0..nrec-1:

        v(s_query) ~= (1 - w) * v[j] + w * v[j + 1]

    j is clipped to [0, nrec-2] so that j+1 is always a valid index.
    """
    frac = (s_query - s0) / ds
    frac = np.clip(frac, 0.0, nrec - 1.0)
    j = np.minimum(np.floor(frac).astype(np.int64), nrec - 2)
    w = frac - j
    return j, w


def _valid_gauge_centres(
    s: np.ndarray,
    half_gauge_m: float,
    tol: float,
) -> np.ndarray:
    """
    Indices of channels that can serve as a gauge centre, i.e. both
    s[i] - half_gauge_m and s[i] + half_gauge_m fall inside [s[0], s[-1]].
    """
    valid = (s - half_gauge_m >= s[0] - tol) & (s + half_gauge_m <= s[-1] + tol)
    return np.nonzero(valid)[0].astype(np.int64)


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
    Core DAS engine: interpolate -> difference -> project.

    Parameters
    ----------
    x_component, z_component :
        Arrays of shape (nrec, nt), e.g. velocity or displacement components
        sampled at receiver centres.
    receivers :
        Receivers2D geometry object. receivers.s must be uniformly spaced
        (guaranteed by build_das_cable's arc-length resampling).
    gauge_length_m :
        Physical gauge length [m]. Any positive value is accepted; it does
        NOT need to be a multiple of the channel spacing.
    kind :
        Output label.

    Returns
    -------
    DASResult
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
    if receivers.nrec < 2:
        raise ValueError(
            f"receivers.nrec={receivers.nrec} < 2; cannot interpolate along cable."
        )
    if not np.isfinite(gauge_length_m) or gauge_length_m <= 0.0:
        raise ValueError(
            f"gauge_length_m must be finite and positive, got {gauge_length_m}."
        )

    s = np.asarray(receivers.s, dtype=np.float64)
    nrec = receivers.nrec

    if s.ndim != 1:
        raise ValueError(f"receivers.s must be 1D, got shape {s.shape}.")

    if s.size != nrec:
        raise ValueError(
            f"receivers.s.size={s.size} must match receivers.nrec={nrec}."
        )

    ds_array = np.diff(s)

    if np.any(ds_array <= 0.0):
        raise ValueError("receivers.s must be strictly increasing.")

    ds = float(receivers.channel_spacing)

    if not np.allclose(ds_array, ds, rtol=1e-5, atol=1e-8):
        raise ValueError(
            "This DAS implementation assumes uniformly spaced receivers along s. "
            f"Expected spacing {ds:.6f} m, but diff(s) ranges from "
            f"{ds_array.min():.6f} to {ds_array.max():.6f} m. "
            "Use build_das_cable(...) to resample the cable uniformly."
    )

    s0 = float(s[0])

    L = float(gauge_length_m)
    half_L = L / 2.0
    tol = 1e-6 * ds

    centre_idx = _valid_gauge_centres(s, half_L, tol)
    if centre_idx.size == 0:
        cable_length = float(s[-1] - s[0])
        raise ValueError(
            f"Gauge length {L:.4f} m is too long for the receiver array "
            f"(cable length {cable_length:.2f} m, nrec={nrec}, "
            f"channel_spacing={ds:.4f} m)."
        )

    s_centre = s[centre_idx]
    s_left = s_centre - half_L
    s_right = s_centre + half_L

    j_left, w_left = _interp_index_weight(s_left, s0, ds, nrec)
    j_right, w_right = _interp_index_weight(s_right, s0, ds, nrec)

    # Step 1: interpolate Cartesian components at the two gauge endpoints,
    #         then difference (right - left).
    def _interp(field: np.ndarray, j: np.ndarray, w: np.ndarray) -> np.ndarray:
        return (1.0 - w)[:, None] * field[j, :] + w[:, None] * field[j + 1, :]

    v_left_x  = _interp(x_component, j_left,  w_left)
    v_right_x = _interp(x_component, j_right, w_right)
    v_left_z  = _interp(z_component, j_left,  w_left)
    v_right_z = _interp(z_component, j_right, w_right)

    dx_comp = v_right_x - v_left_x
    dz_comp = v_right_z - v_left_z

    # Step 2: project the difference onto the local tangent at the gauge centre
    tx_c = receivers.tx[centre_idx, None]
    tz_c = receivers.tz[centre_idx, None]

    data = (dx_comp * tx_c + dz_comp * tz_c) / L

    return DASResult(
        data=data,
        channel_indices=centre_idx,
        gauge_samples=L / ds,
        gauge_length_m=L,
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

    Recomputes the same interpolation indices/weights the forward call used
    (from result.gauge_length_m and result.channel_indices), then scatters
    the residual back using those weights. This is the exact algebraic
    transpose of interpolate -> difference -> project, verified by a
    dot-product test in _self_test() for non-integer gauge lengths.

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
    L = result.gauge_length_m
    half_L = L / 2.0

    s = receivers.s
    ds = receivers.channel_spacing
    s0 = float(s[0])

    s_centre = s[cidx]
    s_left = s_centre - half_L
    s_right = s_centre + half_L

    j_left, w_left = _interp_index_weight(s_left, s0, ds, nrec)
    j_right, w_right = _interp_index_weight(s_right, s0, ds, nrec)

    tx_c = receivers.tx[cidx, None]
    tz_c = receivers.tz[cidx, None]
    scaled = residual / L

    tx_scaled = tx_c * scaled
    tz_scaled = tz_c * scaled

    qx = np.zeros((nrec, nt), dtype=np.float64)
    qz = np.zeros((nrec, nt), dtype=np.float64)

    # Right endpoint contributes with +sign, split across (j_right, j_right+1)
    # by interpolation weights (1-w_right), w_right.
    np.add.at(qx, j_right,     (1.0 - w_right)[:, None] * tx_scaled)
    np.add.at(qx, j_right + 1, w_right[:, None]          * tx_scaled)
    np.add.at(qz, j_right,     (1.0 - w_right)[:, None] * tz_scaled)
    np.add.at(qz, j_right + 1, w_right[:, None]          * tz_scaled)

    # Left endpoint contributes with -sign, split across (j_left, j_left+1).
    np.add.at(qx, j_left,      -(1.0 - w_left)[:, None] * tx_scaled)
    np.add.at(qx, j_left + 1,  -w_left[:, None]          * tx_scaled)
    np.add.at(qz, j_left,      -(1.0 - w_left)[:, None] * tz_scaled)
    np.add.at(qz, j_left + 1,  -w_left[:, None]          * tz_scaled)

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
    The actual DAS operator interpolates, then differences, then projects.
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

    gauge_length_m may be any positive value; it is not required to be a
    multiple of the channel spacing.
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

    gauge_length_m may be any positive value; it is not required to be a
    multiple of the channel spacing.
    """
    return _compute_das_observable(
        ux, uz, receivers, gauge_length_m, kind="axial_strain"
    )


# ==============================================================================
# SELF-TEST
# ==============================================================================

def _self_test() -> None:
    def make_straight(nrec: int, ds: float = 1.0) -> Receivers2D:
        s = np.arange(nrec, dtype=np.float64) * ds
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

        distances = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(z) ** 2))
        s = np.insert(distances, 0, 0.0)

        tx = np.array([1, 1, 1, 1, 1, 1 / np.sqrt(2), 0, 0, 0, 0, 0], dtype=np.float64)
        tz = np.array([0, 0, 0, 0, 0, 1 / np.sqrt(2), 1, 1, 1, 1, 1], dtype=np.float64)
        return Receivers2D(
            x=x, z=z,
            ix=np.zeros(nrec, dtype=np.int64),
            iz=np.zeros(nrec, dtype=np.int64),
            tx=tx, tz=tz, s=s,
        )

    nt = 50

    # 1. Rigid translation on bent cable -> zero strain-rate, ARBITRARY gauge
    rec_bent = make_bent()
    vx = np.ones((rec_bent.nrec, nt), dtype=np.float64) * 3.0
    vz = np.ones((rec_bent.nrec, nt), dtype=np.float64) * 2.0
    for gl in (2.0, 2.37, 4.9):
        res = compute_axial_strain_rate(vx, vz, rec_bent, gauge_length_m=gl)
        assert np.allclose(res.data, 0.0, atol=1e-13), (
            f"Rigid translation should give zero strain at gl={gl}; "
            f"max={np.abs(res.data).max():.2e}"
        )
    print("Rigid translation on bent cable (arbitrary gauge): OK")

    # 2. Linear strain field on straight cable -> exact recovery,
    #    including NON-multiple gauge length (interpolation must be exact
    #    for linear fields).
    rec_st = make_straight(21, ds=1.0)
    alpha = 2.5e-6
    ux = alpha * rec_st.s[:, None] * np.ones((1, nt), dtype=np.float64)
    uz = np.zeros_like(ux)
    for gl in (4.0, 4.37, 5.9999, 3.0001):
        res2 = compute_axial_strain(ux, uz, rec_st, gauge_length_m=gl)
        assert np.allclose(res2.data, alpha, atol=1e-12), (
            f"Linear strain should recover alpha exactly at gl={gl}; "
            f"max_err={np.abs(res2.data - alpha).max():.2e}"
        )
    print("Linear strain recovery, non-multiple gauge lengths: OK")

    # 3. Linear strain-rate field on straight cable -> exact recovery
    beta = 1.2e-7
    vx2 = beta * rec_st.s[:, None] * np.ones((1, nt), dtype=np.float64)
    vz2 = np.zeros_like(vx2)
    res3 = compute_axial_strain_rate(vx2, vz2, rec_st, gauge_length_m=4.37)
    assert np.allclose(res3.data, beta, atol=1e-12)
    print("Linear strain-rate recovery, non-multiple gauge: OK")

    # 4. Dot-product test for several non-integer-multiple gauge lengths
    rng = np.random.default_rng(42)
    rec = make_straight(31, ds=1.0)
    vx_m = rng.standard_normal((rec.nrec, nt))
    vz_m = rng.standard_normal((rec.nrec, nt))
    for gl in (4.0, 4.37, 7.123, 2.9999):
        res_fwd = compute_axial_strain_rate(vx_m, vz_m, rec, gauge_length_m=gl)
        r_rnd = rng.standard_normal(res_fwd.data.shape)
        qx, qz = das_adjoint(r_rnd, rec, res_fwd)

        lhs = float(np.vdot(res_fwd.data, r_rnd))
        rhs = float(np.vdot(vx_m, qx) + np.vdot(vz_m, qz))
        err = abs(lhs - rhs) / (abs(lhs) + 1e-30)
        assert err < 1e-10, (
            f"Dot-product test failed at gl={gl}: lhs={lhs:.6e}, rhs={rhs:.6e}, err={err:.2e}"
        )
    print("Adjoint dot-product test, non-integer gauge lengths: OK")

    # 5. EXACT reproduction of the user's failure case:
    #    channel_spacing=5.0 m, gauge_length_m=16 -> old code crashed
    #    ("odd number of samples (3)"). Must now succeed.
    rec_5m = make_straight(40, ds=5.0)
    vx5 = rng.standard_normal((rec_5m.nrec, nt))
    vz5 = rng.standard_normal((rec_5m.nrec, nt))
    res5 = compute_axial_strain_rate(vx5, vz5, rec_5m, gauge_length_m=16.0)
    assert np.all(np.isfinite(res5.data))
    assert np.isclose(res5.gauge_length_m, 16.0)
    print(f"Reproduced user's failing case (ds=5.0, gl=16.0): now succeeds, "
          f"{res5.nchan_out} channels, gauge_samples={res5.gauge_samples:.3f}")

    # 6. Real-data-style fully fractional gauge length (e.g. 16.6213 m),
    #    not a multiple of channel spacing at all.
    res6 = compute_axial_strain_rate(vx5, vz5, rec_5m, gauge_length_m=16.6213)
    assert np.all(np.isfinite(res6.data))
    assert np.isclose(res6.gauge_length_m, 16.6213)
    r_rnd6 = rng.standard_normal(res6.data.shape)
    qx6, qz6 = das_adjoint(r_rnd6, rec_5m, res6)
    lhs6 = float(np.vdot(res6.data, r_rnd6))
    rhs6 = float(np.vdot(vx5, qx6) + np.vdot(vz5, qz6))
    err6 = abs(lhs6 - rhs6) / (abs(lhs6) + 1e-30)
    assert err6 < 1e-10, f"Dot-product test failed at gl=16.6213: err={err6:.2e}"
    print(f"Real-data-style fractional gauge length (16.6213 m): OK, "
          f"dot-product err={err6:.2e}")

    # 7. Backward consistency: when L is an exact even multiple of ds
    #    (the only case the old implementation supported), interpolation
    #    weights must be exactly 0, reproducing the old exact-indexing result.
    rec_consist = make_straight(25, ds=2.0)
    L_exact = 8.0  # = 4 * ds, matches old "gauge_k=4, half_step=2" case
    vx7 = rng.standard_normal((rec_consist.nrec, nt))
    vz7 = rng.standard_normal((rec_consist.nrec, nt))
    res7 = compute_axial_strain_rate(vx7, vz7, rec_consist, gauge_length_m=L_exact)

    half_step = int(round(L_exact / rec_consist.channel_spacing / 2))
    centre_idx_old = np.arange(half_step, rec_consist.nrec - half_step, dtype=np.int64)
    left_idx_old = centre_idx_old - half_step
    right_idx_old = centre_idx_old + half_step
    dx_old = vx7[right_idx_old, :] - vx7[left_idx_old, :]
    dz_old = vz7[right_idx_old, :] - vz7[left_idx_old, :]
    tx_old = rec_consist.tx[centre_idx_old, None]
    tz_old = rec_consist.tz[centre_idx_old, None]
    data_old = (dx_old * tx_old + dz_old * tz_old) / L_exact

    assert np.array_equal(res7.channel_indices, centre_idx_old)
    assert np.allclose(res7.data, data_old, atol=1e-13), (
        "New interpolation-based result must exactly reproduce old "
        "exact-indexing result when L is an even multiple of ds."
    )
    print("Backward consistency with old exact-indexing case (L=4*ds): OK")

    # 8. Shape mismatch guard
    try:
        compute_axial_strain(ux[:-1], uz, rec_st, gauge_length_m=2.0)
        raise AssertionError("Expected ValueError.")
    except ValueError:
        pass
    print("Shape mismatch guard: OK")

    # 9. DASResult validation: mismatched channel count
    try:
        DASResult(
            data=np.zeros((5, 10), dtype=np.float64),
            channel_indices=np.arange(6, dtype=np.int64),
            gauge_samples=2.0,
            gauge_length_m=1.0,
            kind="axial_strain",
        )
        raise AssertionError("Expected ValueError.")
    except ValueError:
        pass
    print("DASResult shape validation: OK")

    # 10. Non-positive / non-finite gauge_length_m guard
    for bad_gl in (0.0, -1.0, float("nan"), float("inf")):
        try:
            compute_axial_strain_rate(vx_m, vz_m, rec, gauge_length_m=bad_gl)
            raise AssertionError(f"Expected ValueError for gauge_length_m={bad_gl}.")
        except ValueError:
            pass
    print("Non-positive/non-finite gauge_length_m guard: OK")

    # 11. Gauge too long for array -> clear error
    try:
        compute_axial_strain_rate(vx_m, vz_m, rec, gauge_length_m=10_000.0)
        raise AssertionError("Expected ValueError for oversized gauge.")
    except ValueError:
        pass
    print("Oversized gauge ValueError: OK")

    print("\nSelf-test PASSED")


if __name__ == "__main__":
    _self_test()