# ==============================================================================
# scripts/validation/source_radiation.py
#
# Homogeneous full-space radiation-pattern validation for 2D moment-tensor
# sources.
#
# Purpose
# -------
# Validate the complete production source path:
#
#   source tensor
#       -> staggered-grid bilinear spreading
#       -> stress injection
#       -> elastic propagation
#       -> staggered-aware receiver sampling
#
# The test is deliberately independent of SAFOD structure, free-surface
# reflections, DAS gauge response, and real-data preprocessing.
#
# Method
# ------
# 1. Build a homogeneous isotropic elastic full-space model.
# 2. Place an off-grid source at the model centre.
# 3. Record vx and vz on a circular receiver array.
# 4. Rotate Cartesian velocities into:
#
#       radial     v_r =  v_x cos(phi) + v_z sin(phi)
#       transverse v_t = -v_x sin(phi) + v_z cos(phi)
#
# 5. Estimate signed P and SV amplitudes by projecting each trace onto a
#    common phase template.
# 6. Compare the numerical angular patterns with the 2D far-field tensor
#    predictions:
#
#       A_P(phi)  proportional to n^T M n
#       A_SV(phi) proportional to t^T M n
#
#    where n = (cos(phi), sin(phi)) and
#          t = (-sin(phi), cos(phi)).
#
# Expected behaviour
# ------------------
# - isotropic source:
#       P pattern nearly constant;
#       SV energy very small.
#
# - double couple:
#       P and SV patterns agree with the tensor predictions;
#       changing theta rotates the lobes.
#
# Important limitation
# --------------------
# The current Ricker source is a modelling wavelet, not a unit-area physical
# moment-rate function. This validation tests radiation pattern, polarity,
# relative component energy, and source injection. It does not validate an
# absolute 2D-to-3D earthquake-amplitude convention.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.grid import Grid2D
from src.model import ElasticModel2D
from src.receivers import Receivers2D
from src.simulator import run_forward_simulation
from src.solver_numpy import max_stable_dt
from src.source import (
    EmbeddedSource2D,
    build_dc_source,
    build_isotropic_source,
)


# ==============================================================================
# OUTPUT
# ==============================================================================

OUT_DIR = Path("results/validation/source_radiation")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# NUMERICAL / PHYSICAL CONFIGURATION
# ==============================================================================

DX_M = 10.0
DZ_M = 10.0

NX = 451
NZ = 451

HALF_ORDER = 2
CFL_SAFETY = 0.80

VP_M_S = 4000.0
VS_M_S = 2300.0
RHO_KG_M3 = 2500.0

SOURCE_F0_HZ = 15.0
SOURCE_SCALAR_MOMENT = 1.0e12
SOURCE_DERIVATIVE_ORDER = 0
SOURCE_SPREADING = "bilinear"

N_BOUNDARY = 60
GAMMA_S = 80.0
FREE_SURFACE = False

RECEIVER_RADIUS_M = 1000.0
N_RECEIVERS = 144

# Recording must contain direct P and SV but end before the earliest possible
# outer-boundary return reaches the receiver ring.
RECORD_END_S = 0.62

P_WINDOW_HALF_WIDTH_S = 0.045
S_WINDOW_HALF_WIDTH_S = 0.055

# run_forward_simulation computes a DAS observable even though this validation
# uses receiver vx/vz. A valid gauge is therefore supplied but not analysed.
RING_GAUGE_SPACINGS = 3.0

SOURCE_CASES = (
    {"label": "isotropic", "kind": "isotropic", "theta_deg": np.nan},
    {"label": "dc_theta_0", "kind": "dc", "theta_deg": 0.0},
    {"label": "dc_theta_22p5", "kind": "dc", "theta_deg": 22.5},
    {"label": "dc_theta_45", "kind": "dc", "theta_deg": 45.0},
)


# ==============================================================================
# CONSERVATIVE PASS / FAIL THRESHOLDS
# ==============================================================================

# These are deliberately loose enough to accommodate finite-distance effects,
# grid anisotropy, and staggered interpolation, while still detecting a gross
# source-tensor or injection error.
MIN_PATTERN_SIMILARITY = 0.70
MAX_ISOTROPIC_SV_TO_P_ENERGY = 0.10
MAX_ISOTROPIC_P_RELATIVE_L2 = 0.35


# ==============================================================================
# DATA CONTAINERS
# ==============================================================================

@dataclass(frozen=True)
class RadiationCaseResult:
    label: str
    kind: str
    theta_deg: float
    source: EmbeddedSource2D
    angles_rad: np.ndarray
    time_s: np.ndarray
    receiver_vx: np.ndarray
    receiver_vz: np.ndarray
    radial_velocity: np.ndarray
    transverse_velocity: np.ndarray
    theoretical_p: np.ndarray
    theoretical_sv: np.ndarray
    measured_p: np.ndarray
    measured_sv: np.ndarray
    p_template_receiver: int
    sv_template_receiver: int | None
    p_window: tuple[float, float]
    sv_window: tuple[float, float]
    metrics: dict[str, float]


# ==============================================================================
# MODEL / GEOMETRY
# ==============================================================================

def build_homogeneous_model() -> ElasticModel2D:
    """Build the homogeneous full-space validation model."""
    dt = max_stable_dt(
        vp_max=VP_M_S,
        dx=DX_M,
        dz=DZ_M,
        half_order=HALF_ORDER,
        safety=CFL_SAFETY,
        use_ts_sfd=False,
    )

    nt = int(np.ceil(RECORD_END_S / dt)) + 1

    grid = Grid2D(
        nx=NX,
        nz=NZ,
        dx=DX_M,
        dz=DZ_M,
        nt=nt,
        dt=dt,
        x0=0.0,
        z0=0.0,
    )

    shape = grid.shape

    return ElasticModel2D(
        grid=grid,
        vp=np.full(shape, VP_M_S, dtype=np.float64),
        vs=np.full(shape, VS_M_S, dtype=np.float64),
        rho=np.full(shape, RHO_KG_M3, dtype=np.float64),
    )


def source_position(grid: Grid2D) -> tuple[float, float]:
    """
    Return a deliberately off-grid source near the model centre.

    The physical source position, not the dominant integer node, is used as
    the centre of the receiver ring and the radiation-coordinate system.
    """
    ix_center = grid.nx // 2
    iz_center = grid.nz // 2

    x_src = float(grid.x[ix_center] + 0.37 * grid.dx)
    z_src = float(grid.z[iz_center] + 0.61 * grid.dz)

    return x_src, z_src


def build_receiver_ring(
    grid: Grid2D,
    *,
    x_src: float,
    z_src: float,
) -> tuple[Receivers2D, np.ndarray]:
    """
    Build uniformly spaced receivers on a circle around the source.

    Angles are measured from +x toward +z. Since z is depth-positive, increasing
    angle is clockwise in a conventional x-horizontal/depth-down plot.
    """
    if N_RECEIVERS < 16:
        raise ValueError("N_RECEIVERS must be >= 16.")

    angles = np.linspace(
        0.0,
        2.0 * np.pi,
        N_RECEIVERS,
        endpoint=False,
        dtype=np.float64,
    )

    cos_phi = np.cos(angles)
    sin_phi = np.sin(angles)

    x = x_src + RECEIVER_RADIUS_M * cos_phi
    z = z_src + RECEIVER_RADIUS_M * sin_phi

    # Tangent of increasing phi on the circular path.
    tx = -sin_phi
    tz = cos_phi

    ix = np.rint((x - grid.x0) / grid.dx).astype(np.int64)
    iz = np.rint((z - grid.z0) / grid.dz).astype(np.int64)

    ds = 2.0 * np.pi * RECEIVER_RADIUS_M / N_RECEIVERS
    s = np.arange(N_RECEIVERS, dtype=np.float64) * ds

    receivers = Receivers2D(
        x=x,
        z=z,
        ix=ix,
        iz=iz,
        tx=tx,
        tz=tz,
        s=s,
    )

    # Explicit physical-interior guard.
    x_min = grid.x[0] + N_BOUNDARY * grid.dx
    x_max = grid.x[-1] - N_BOUNDARY * grid.dx
    z_min = grid.z[0] + N_BOUNDARY * grid.dz
    z_max = grid.z[-1] - N_BOUNDARY * grid.dz

    if not (
        np.all(receivers.x > x_min)
        and np.all(receivers.x < x_max)
        and np.all(receivers.z > z_min)
        and np.all(receivers.z < z_max)
    ):
        raise ValueError(
            "Receiver ring intersects the absorbing boundary. "
            "Increase NX/NZ or reduce RECEIVER_RADIUS_M."
        )

    return receivers, angles


def build_source_for_case(
    grid: Grid2D,
    *,
    x_src: float,
    z_src: float,
    kind: str,
    theta_deg: float,
) -> EmbeddedSource2D:
    """Build one isotropic or double-couple validation source."""
    common = dict(
        grid=grid,
        x_m=x_src,
        z_m=z_src,
        scalar_moment=SOURCE_SCALAR_MOMENT,
        nt=grid.nt,
        dt=grid.dt,
        f0_hz=SOURCE_F0_HZ,
        derivative_order=SOURCE_DERIVATIVE_ORDER,
        spreading=SOURCE_SPREADING,
    )

    if kind == "isotropic":
        return build_isotropic_source(**common)

    if kind == "dc":
        if not np.isfinite(theta_deg):
            raise ValueError("Double-couple case requires finite theta_deg.")

        return build_dc_source(
            **common,
            theta_deg=float(theta_deg),
        )

    raise ValueError(f"Unsupported source kind: {kind!r}")


# ==============================================================================
# RADIATION THEORY / SIGNAL EXTRACTION
# ==============================================================================

def theoretical_radiation_patterns(
    source: EmbeddedSource2D,
    angles_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return unnormalised 2D far-field P and SV tensor radiation patterns.

    A_P  = n^T M n
    A_SV = t^T M n
    """
    angles_rad = np.asarray(angles_rad, dtype=np.float64)

    n = np.column_stack(
        [
            np.cos(angles_rad),
            np.sin(angles_rad),
        ]
    )

    t = np.column_stack(
        [
            -np.sin(angles_rad),
            np.cos(angles_rad),
        ]
    )

    moment = source.m2d.as_matrix()

    p_pattern = np.einsum(
        "ni,ij,nj->n",
        n,
        moment,
        n,
    )

    sv_pattern = np.einsum(
        "ni,ij,nj->n",
        t,
        moment,
        n,
    )

    return (
        np.asarray(p_pattern, dtype=np.float64),
        np.asarray(sv_pattern, dtype=np.float64),
    )


def rotate_velocity_to_ray_coordinates(
    receiver_vx: np.ndarray,
    receiver_vz: np.ndarray,
    angles_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate sampled Cartesian velocities into radial/transverse components."""
    receiver_vx = np.asarray(receiver_vx, dtype=np.float64)
    receiver_vz = np.asarray(receiver_vz, dtype=np.float64)
    angles_rad = np.asarray(angles_rad, dtype=np.float64)

    if receiver_vx.shape != receiver_vz.shape:
        raise ValueError("receiver_vx and receiver_vz must share shape.")

    if receiver_vx.ndim != 2:
        raise ValueError("Receiver velocity arrays must be 2D.")

    if receiver_vx.shape[0] != angles_rad.size:
        raise ValueError(
            "Receiver count does not match angular coordinate: "
            f"{receiver_vx.shape[0]} != {angles_rad.size}."
        )

    cos_phi = np.cos(angles_rad)[:, None]
    sin_phi = np.sin(angles_rad)[:, None]

    radial = receiver_vx * cos_phi + receiver_vz * sin_phi
    transverse = -receiver_vx * sin_phi + receiver_vz * cos_phi

    return radial, transverse


def phase_window(
    time_s: np.ndarray,
    *,
    centre_s: float,
    half_width_s: float,
    label: str,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Build and validate one direct-phase time mask."""
    time_s = np.asarray(time_s, dtype=np.float64)

    start_s = float(centre_s - half_width_s)
    end_s = float(centre_s + half_width_s)

    mask = (
        (time_s >= start_s)
        & (time_s <= end_s)
    )

    if np.count_nonzero(mask) < 8:
        raise ValueError(
            f"{label} window [{start_s:.4f}, {end_s:.4f}] s "
            "contains fewer than 8 samples."
        )

    return mask, (start_s, end_s)


def template_projection_amplitudes(
    traces: np.ndarray,
    mask: np.ndarray,
    theoretical_pattern: np.ndarray,
    *,
    phase_label: str,
) -> tuple[np.ndarray, int]:
    """
    Estimate signed angular amplitudes by projection onto a common template.

    The template receiver is selected where the theoretical pattern has its
    largest absolute value. A single template avoids using independent peak
    signs from an oscillatory Ricker waveform.
    """
    traces = np.asarray(traces, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    theoretical_pattern = np.asarray(
        theoretical_pattern,
        dtype=np.float64,
    )

    if traces.ndim != 2:
        raise ValueError("traces must be 2D (receiver, time).")

    if traces.shape[0] != theoretical_pattern.size:
        raise ValueError(
            "Pattern length must match receiver count."
        )

    if traces.shape[1] != mask.size:
        raise ValueError(
            "Time mask length must match trace time dimension."
        )

    if not np.any(np.abs(theoretical_pattern) > 0.0):
        raise ValueError(
            f"{phase_label} theoretical pattern is identically zero; "
            "a signed template projection is undefined."
        )

    reference_index = int(
        np.nanargmax(np.abs(theoretical_pattern))
    )

    windowed = traces[:, mask].copy()
    windowed -= np.mean(windowed, axis=1, keepdims=True)

    reference = windowed[reference_index].copy()

    denominator = float(np.dot(reference, reference))

    if not np.isfinite(denominator) or denominator <= 1.0e-300:
        raise RuntimeError(
            f"{phase_label} reference trace has zero/invalid energy."
        )

    amplitudes = windowed @ reference / denominator

    return (
        np.asarray(amplitudes, dtype=np.float64),
        reference_index,
    )


def normalize_pattern(pattern: np.ndarray) -> np.ndarray:
    """Normalize a signed angular pattern by its maximum absolute value."""
    pattern = np.asarray(pattern, dtype=np.float64)

    scale = float(np.nanmax(np.abs(pattern)))

    if not np.isfinite(scale) or scale <= 1.0e-300:
        return np.zeros_like(pattern)

    return pattern / scale


def fit_pattern_metrics(
    measured: np.ndarray,
    theoretical: np.ndarray,
) -> dict[str, float]:
    """
    Fit one global scale between measured and theoretical angular patterns.

    The absolute cosine similarity is insensitive to one global polarity
    reversal, which may arise from source/velocity sign conventions. Nodal
    locations and relative lobe signs still control the metric.
    """
    measured_n = normalize_pattern(measured)
    theoretical_n = normalize_pattern(theoretical)

    norm_m = float(np.linalg.norm(measured_n))
    norm_t = float(np.linalg.norm(theoretical_n))

    if norm_m <= 1.0e-300 or norm_t <= 1.0e-300:
        return {
            "scale": np.nan,
            "similarity": np.nan,
            "relative_l2": np.nan,
        }

    scale = float(
        np.dot(measured_n, theoretical_n)
        / np.dot(theoretical_n, theoretical_n)
    )

    fitted = scale * theoretical_n

    similarity_signed = float(
        np.dot(measured_n, theoretical_n)
        / (norm_m * norm_t)
    )

    relative_l2 = float(
        np.linalg.norm(measured_n - fitted)
        / norm_m
    )

    return {
        "scale": scale,
        "similarity": abs(similarity_signed),
        "relative_l2": relative_l2,
    }


def integrated_energy(
    traces: np.ndarray,
    mask: np.ndarray,
    *,
    dt: float,
) -> float:
    """Return summed trace energy over a time window."""
    traces = np.asarray(traces, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)

    return float(
        np.sum(traces[:, mask] ** 2) * dt
    )


def earliest_outer_boundary_return_s(
    grid: Grid2D,
    *,
    x_src: float,
    z_src: float,
) -> float:
    """
    Conservative earliest P-wave return from a sponge entrance to the ring.

    This is a geometry guard, not ray tracing. It uses the nearest side of the
    undamped interior and the receiver on that same side.
    """
    x_left = grid.x[0] + N_BOUNDARY * grid.dx
    x_right = grid.x[-1] - N_BOUNDARY * grid.dx
    z_top = grid.z[0] + N_BOUNDARY * grid.dz
    z_bottom = grid.z[-1] - N_BOUNDARY * grid.dz

    distances_to_entry = np.array(
        [
            x_src - x_left,
            x_right - x_src,
            z_src - z_top,
            z_bottom - z_src,
        ],
        dtype=np.float64,
    )

    nearest_entry_distance = float(
        np.min(distances_to_entry)
    )

    if nearest_entry_distance <= RECEIVER_RADIUS_M:
        raise ValueError(
            "Receiver ring reaches or crosses a sponge entrance."
        )

    path_m = (
        nearest_entry_distance
        + nearest_entry_distance
        - RECEIVER_RADIUS_M
    )

    source_t0_s = 1.2 / SOURCE_F0_HZ

    return float(
        source_t0_s + path_m / VP_M_S
    )


# ==============================================================================
# ONE CASE
# ==============================================================================

def run_case(
    *,
    model: ElasticModel2D,
    receivers: Receivers2D,
    angles_rad: np.ndarray,
    x_src: float,
    z_src: float,
    label: str,
    kind: str,
    theta_deg: float,
) -> RadiationCaseResult:
    """Run and analyse one source mechanism."""
    grid = model.grid

    source = build_source_for_case(
        grid,
        x_src=x_src,
        z_src=z_src,
        kind=kind,
        theta_deg=theta_deg,
    )

    print(f"\nRunning source-radiation case: {label}")
    print(source.summary())

    gauge_length_m = (
        RING_GAUGE_SPACINGS
        * receivers.channel_spacing
    )

    run_result, _ = run_forward_simulation(
        model=model,
        source=source,
        receivers=receivers,
        gauge_length_m=gauge_length_m,
        half_order=HALF_ORDER,
        use_ts_sfd=False,
        n_boundary=N_BOUNDARY,
        gamma_s=GAMMA_S,
        snapshot_stride=None,
        backend="numba_fused",
        free_surface=FREE_SURFACE,
    )

    time_s = np.asarray(
        run_result.t_v,
        dtype=np.float64,
    )

    receiver_vx = np.asarray(
        run_result.receiver_vx,
        dtype=np.float64,
    )

    receiver_vz = np.asarray(
        run_result.receiver_vz,
        dtype=np.float64,
    )

    if not np.all(np.isfinite(receiver_vx)):
        raise RuntimeError(f"{label}: receiver_vx contains NaN/Inf.")

    if not np.all(np.isfinite(receiver_vz)):
        raise RuntimeError(f"{label}: receiver_vz contains NaN/Inf.")

    radial, transverse = rotate_velocity_to_ray_coordinates(
        receiver_vx,
        receiver_vz,
        angles_rad,
    )

    theoretical_p, theoretical_sv = theoretical_radiation_patterns(
        source,
        angles_rad,
    )

    source_t0_s = float(source.stf.t0)

    p_centre_s = (
        source_t0_s
        + RECEIVER_RADIUS_M / VP_M_S
    )

    sv_centre_s = (
        source_t0_s
        + RECEIVER_RADIUS_M / VS_M_S
    )

    p_mask, p_window = phase_window(
        time_s,
        centre_s=p_centre_s,
        half_width_s=P_WINDOW_HALF_WIDTH_S,
        label="P",
    )

    sv_mask, sv_window = phase_window(
        time_s,
        centre_s=sv_centre_s,
        half_width_s=S_WINDOW_HALF_WIDTH_S,
        label="SV",
    )

    if p_window[1] >= sv_window[0]:
        raise ValueError(
            "P and SV analysis windows overlap. Increase receiver radius "
            "or reduce phase-window widths."
        )

    measured_p, p_reference = template_projection_amplitudes(
        radial,
        p_mask,
        theoretical_p,
        phase_label="P",
    )

    if np.any(np.abs(theoretical_sv) > 1.0e-12 * SOURCE_SCALAR_MOMENT):
        measured_sv, sv_reference = template_projection_amplitudes(
            transverse,
            sv_mask,
            theoretical_sv,
            phase_label="SV",
        )
    else:
        measured_sv = np.zeros_like(theoretical_sv)
        sv_reference = None

    p_fit = fit_pattern_metrics(
        measured_p,
        theoretical_p,
    )

    if sv_reference is not None:
        sv_fit = fit_pattern_metrics(
            measured_sv,
            theoretical_sv,
        )
    else:
        sv_fit = {
            "scale": np.nan,
            "similarity": np.nan,
            "relative_l2": np.nan,
        }

    p_radial_energy = integrated_energy(
        radial,
        p_mask,
        dt=grid.dt,
    )

    p_transverse_energy = integrated_energy(
        transverse,
        p_mask,
        dt=grid.dt,
    )

    sv_radial_energy = integrated_energy(
        radial,
        sv_mask,
        dt=grid.dt,
    )

    sv_transverse_energy = integrated_energy(
        transverse,
        sv_mask,
        dt=grid.dt,
    )

    metrics = {
        "p_pattern_scale": p_fit["scale"],
        "p_pattern_similarity": p_fit["similarity"],
        "p_pattern_relative_l2": p_fit["relative_l2"],
        "sv_pattern_scale": sv_fit["scale"],
        "sv_pattern_similarity": sv_fit["similarity"],
        "sv_pattern_relative_l2": sv_fit["relative_l2"],
        "p_radial_energy": p_radial_energy,
        "p_transverse_energy": p_transverse_energy,
        "sv_radial_energy": sv_radial_energy,
        "sv_transverse_energy": sv_transverse_energy,
        "p_transverse_to_radial_energy": (
            p_transverse_energy
            / max(p_radial_energy, 1.0e-300)
        ),
        "sv_radial_to_transverse_energy": (
            sv_radial_energy
            / max(sv_transverse_energy, 1.0e-300)
        ),
        "sv_transverse_to_p_radial_energy": (
            sv_transverse_energy
            / max(p_radial_energy, 1.0e-300)
        ),
    }

    print(
        "  P pattern similarity     : "
        f"{metrics['p_pattern_similarity']:.4f}"
    )
    print(
        "  P pattern relative L2    : "
        f"{metrics['p_pattern_relative_l2']:.4f}"
    )

    if sv_reference is not None:
        print(
            "  SV pattern similarity    : "
            f"{metrics['sv_pattern_similarity']:.4f}"
        )
        print(
            "  SV pattern relative L2   : "
            f"{metrics['sv_pattern_relative_l2']:.4f}"
        )
    else:
        print(
            "  Isotropic SV/P energy    : "
            f"{metrics['sv_transverse_to_p_radial_energy']:.4e}"
        )

    return RadiationCaseResult(
        label=label,
        kind=kind,
        theta_deg=float(theta_deg),
        source=source,
        angles_rad=np.asarray(angles_rad, dtype=np.float64),
        time_s=time_s,
        receiver_vx=receiver_vx,
        receiver_vz=receiver_vz,
        radial_velocity=radial,
        transverse_velocity=transverse,
        theoretical_p=theoretical_p,
        theoretical_sv=theoretical_sv,
        measured_p=measured_p,
        measured_sv=measured_sv,
        p_template_receiver=p_reference,
        sv_template_receiver=sv_reference,
        p_window=p_window,
        sv_window=sv_window,
        metrics=metrics,
    )


# ==============================================================================
# PLOTTING
# ==============================================================================

def plot_case_radiation(
    result: RadiationCaseResult,
    out_path: Path,
) -> None:
    """Plot absolute polar lobes and signed angular patterns for one case."""
    angles_closed = np.append(
        result.angles_rad,
        result.angles_rad[0] + 2.0 * np.pi,
    )

    measured_p_n = normalize_pattern(result.measured_p)
    theoretical_p_n = normalize_pattern(result.theoretical_p)

    measured_sv_n = normalize_pattern(result.measured_sv)
    theoretical_sv_n = normalize_pattern(result.theoretical_sv)

    p_scale = result.metrics["p_pattern_scale"]
    sv_scale = result.metrics["sv_pattern_scale"]

    p_fit = (
        p_scale * theoretical_p_n
        if np.isfinite(p_scale)
        else theoretical_p_n
    )

    sv_fit = (
        sv_scale * theoretical_sv_n
        if np.isfinite(sv_scale)
        else theoretical_sv_n
    )

    measured_p_closed = np.append(measured_p_n, measured_p_n[0])
    p_fit_closed = np.append(p_fit, p_fit[0])

    measured_sv_closed = np.append(measured_sv_n, measured_sv_n[0])
    sv_fit_closed = np.append(sv_fit, sv_fit[0])

    angle_deg = np.rad2deg(result.angles_rad)

    fig = plt.figure(figsize=(13, 10))

    ax_p_polar = fig.add_subplot(2, 2, 1, projection="polar")
    ax_sv_polar = fig.add_subplot(2, 2, 2, projection="polar")
    ax_p_signed = fig.add_subplot(2, 2, 3)
    ax_sv_signed = fig.add_subplot(2, 2, 4)

    ax_p_polar.plot(
        angles_closed,
        np.abs(measured_p_closed),
        linewidth=1.8,
        label="Numerical |P|",
    )
    ax_p_polar.plot(
        angles_closed,
        np.abs(p_fit_closed),
        linestyle="--",
        linewidth=1.5,
        label="Tensor theory |P|",
    )
    ax_p_polar.set_title("P radiation: radial velocity")
    ax_p_polar.set_ylim(0.0, 1.05)
    ax_p_polar.legend(
        loc="upper right",
        bbox_to_anchor=(1.35, 1.15),
        fontsize=8,
    )

    if result.sv_template_receiver is not None:
        ax_sv_polar.plot(
            angles_closed,
            np.abs(measured_sv_closed),
            linewidth=1.8,
            label="Numerical |SV|",
        )
        ax_sv_polar.plot(
            angles_closed,
            np.abs(sv_fit_closed),
            linestyle="--",
            linewidth=1.5,
            label="Tensor theory |SV|",
        )
    else:
        sv_window_mask = (
            (result.time_s >= result.sv_window[0])
            & (result.time_s <= result.sv_window[1])
        )
        numerical_sv_rms = np.sqrt(
            np.mean(
                result.transverse_velocity[:, sv_window_mask] ** 2,
                axis=1,
            )
        )

        
        p_window_mask = (
            (result.time_s >= result.p_window[0])
            & (result.time_s <= result.p_window[1])
        )

        p_reference_rms = float(
            np.max(
                np.sqrt(
                    np.mean(
                        result.radial_velocity[:, p_window_mask] ** 2,
                        axis=1,
                    )
                )
            )
        )

        numerical_sv_rms = (
            numerical_sv_rms
            / max(p_reference_rms, 1.0e-300)
        )


        numerical_sv_rms_closed = np.append(
            numerical_sv_rms,
            numerical_sv_rms[0],
        )
        ax_sv_polar.plot(
            angles_closed,
            numerical_sv_rms_closed,
            linewidth=1.8,
            label="Numerical SV-window RMS",
        )
        ax_sv_polar.plot(
            angles_closed,
            np.zeros_like(angles_closed),
            linestyle="--",
            linewidth=1.5,
            label="Tensor theory: zero SV",
        )

    ax_sv_polar.set_title("SV radiation: transverse velocity")
    ax_sv_polar.set_ylim(0.0, 1.05)
    ax_sv_polar.legend(
        loc="upper right",
        bbox_to_anchor=(1.35, 1.15),
        fontsize=8,
    )

    ax_p_signed.plot(
        angle_deg,
        measured_p_n,
        linewidth=1.6,
        label="Numerical P",
    )
    ax_p_signed.plot(
        angle_deg,
        p_fit,
        linestyle="--",
        linewidth=1.5,
        label="Scaled tensor theory",
    )
    ax_p_signed.axhline(0.0, linewidth=0.8)
    ax_p_signed.set_xlim(0.0, 360.0)
    ax_p_signed.set_ylim(-1.1, 1.1)
    ax_p_signed.set_xlabel("Azimuth from +x toward +z [deg]")
    ax_p_signed.set_ylabel("Signed normalized amplitude")
    ax_p_signed.set_title(
        "P signed pattern\n"
        f"similarity={result.metrics['p_pattern_similarity']:.3f}, "
        f"relative L2={result.metrics['p_pattern_relative_l2']:.3f}"
    )
    ax_p_signed.grid(alpha=0.3)
    ax_p_signed.legend(fontsize=8)

    if result.sv_template_receiver is not None:
        ax_sv_signed.plot(
            angle_deg,
            measured_sv_n,
            linewidth=1.6,
            label="Numerical SV",
        )
        ax_sv_signed.plot(
            angle_deg,
            sv_fit,
            linestyle="--",
            linewidth=1.5,
            label="Scaled tensor theory",
        )
        subtitle = (
            f"similarity={result.metrics['sv_pattern_similarity']:.3f}, "
            f"relative L2={result.metrics['sv_pattern_relative_l2']:.3f}"
        )
    else:
        ax_sv_signed.plot(
            angle_deg,
            np.zeros_like(angle_deg),
            linestyle="--",
            linewidth=1.5,
            label="Tensor theory: zero SV",
        )
        subtitle = (
            "SV/P energy="
            f"{result.metrics['sv_transverse_to_p_radial_energy']:.3e}"
        )

    ax_sv_signed.axhline(0.0, linewidth=0.8)
    ax_sv_signed.set_xlim(0.0, 360.0)
    ax_sv_signed.set_ylim(-1.1, 1.1)
    ax_sv_signed.set_xlabel("Azimuth from +x toward +z [deg]")
    ax_sv_signed.set_ylabel("Signed normalized amplitude")
    ax_sv_signed.set_title(
        "SV signed pattern\n"
        + subtitle
    )
    ax_sv_signed.grid(alpha=0.3)
    ax_sv_signed.legend(fontsize=8)

    theta_text = (
        "isotropic"
        if result.kind == "isotropic"
        else f"double couple, theta={result.theta_deg:.1f}°"
    )

    fig.suptitle(
        "2D homogeneous source-radiation validation\n"
        f"{theta_text}; bilinear spreading; full space",
        y=0.995,
    )

    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_reference_traces(
    result: RadiationCaseResult,
    out_path: Path,
) -> None:
    """Plot radial/transverse traces at the selected template azimuths."""
    indices = [result.p_template_receiver]

    if (
        result.sv_template_receiver is not None
        and result.sv_template_receiver not in indices
    ):
        indices.append(result.sv_template_receiver)

    fig, axes = plt.subplots(
        len(indices),
        1,
        figsize=(10, 3.4 * len(indices)),
        sharex=True,
        squeeze=False,
    )

    for ax, index in zip(axes[:, 0], indices):
        angle_deg = np.rad2deg(
            result.angles_rad[index]
        )

        ax.plot(
            result.time_s,
            result.radial_velocity[index],
            label=r"$v_r$",
            linewidth=1.3,
        )

        ax.plot(
            result.time_s,
            result.transverse_velocity[index],
            label=r"$v_t$",
            linewidth=1.3,
        )

        ax.axvspan(
            result.p_window[0],
            result.p_window[1],
            alpha=0.12,
            label="P window",
        )

        ax.axvspan(
            result.sv_window[0],
            result.sv_window[1],
            alpha=0.12,
            label="SV window",
        )

        ax.set_ylabel("Velocity [arb.]")
        ax.set_title(
            f"{result.label}: receiver {index}, azimuth={angle_deg:.1f}°"
        )
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    axes[-1, 0].set_xlabel("Time [s]")

    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_summary_metrics(
    summary: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot compact pattern-similarity and leakage diagnostics."""
    labels = summary["label"].astype(str).to_numpy()
    x = np.arange(labels.size, dtype=np.float64)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
    )

    ax_similarity, ax_leakage = axes

    ax_similarity.plot(
        x,
        summary["p_pattern_similarity"],
        marker="o",
        label="P pattern similarity",
    )

    sv_valid = np.isfinite(
        summary["sv_pattern_similarity"].to_numpy(dtype=np.float64)
    )

    if np.any(sv_valid):
        ax_similarity.plot(
            x[sv_valid],
            summary.loc[sv_valid, "sv_pattern_similarity"],
            marker="o",
            label="SV pattern similarity",
        )

    ax_similarity.axhline(
        MIN_PATTERN_SIMILARITY,
        linestyle="--",
        linewidth=1.0,
        label="Minimum QC threshold",
    )

    ax_similarity.set_ylim(0.0, 1.05)
    ax_similarity.set_ylabel("Absolute cosine similarity")
    ax_similarity.set_title("Numerical versus tensor radiation pattern")
    ax_similarity.grid(alpha=0.3)
    ax_similarity.legend(fontsize=8)

    ax_leakage.semilogy(
        x,
        np.maximum(
            summary["p_transverse_to_radial_energy"].to_numpy(
                dtype=np.float64
            ),
            1.0e-12,
        ),
        marker="o",
        label="P-window transverse/radial",
    )

    ax_leakage.semilogy(
        x,
        np.maximum(
            summary["sv_radial_to_transverse_energy"].to_numpy(
                dtype=np.float64
            ),
            1.0e-12,
        ),
        marker="o",
        label="SV-window radial/transverse",
    )

    ax_leakage.set_ylabel("Energy ratio")
    ax_leakage.set_title("Cross-polarization diagnostics")
    ax_leakage.set_xticks(x)
    ax_leakage.set_xticklabels(
        labels,
        rotation=20,
        ha="right",
    )
    ax_leakage.grid(alpha=0.3)
    ax_leakage.legend(fontsize=8)

    fig.tight_layout()

    fig.savefig(
        out_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


# ==============================================================================
# TABLES / QC
# ==============================================================================

def build_amplitude_table(
    results: list[RadiationCaseResult],
) -> pd.DataFrame:
    """Create one row per source case and receiver azimuth."""
    rows: list[dict] = []

    for result in results:
        for i, angle_rad in enumerate(result.angles_rad):
            rows.append(
                {
                    "label": result.label,
                    "kind": result.kind,
                    "theta_deg": result.theta_deg,
                    "receiver_index": i,
                    "azimuth_deg": float(np.rad2deg(angle_rad)),
                    "theoretical_P": float(result.theoretical_p[i]),
                    "theoretical_SV": float(result.theoretical_sv[i]),
                    "measured_P_projection": float(result.measured_p[i]),
                    "measured_SV_projection": float(result.measured_sv[i]),
                }
            )

    return pd.DataFrame(rows)


def build_summary_table(
    results: list[RadiationCaseResult],
) -> pd.DataFrame:
    """Create one summary row per source case."""
    rows: list[dict] = []

    for result in results:
        row = {
            "label": result.label,
            "kind": result.kind,
            "theta_deg": result.theta_deg,
            "Mxx": result.source.m2d.Mxx,
            "Mzz": result.source.m2d.Mzz,
            "Mxz": result.source.m2d.Mxz,
            "tensor_trace": result.source.m2d.trace(),
            "tensor_frobenius_norm": (
                result.source.m2d.frobenius_norm()
            ),
            "p_window_start_s": result.p_window[0],
            "p_window_end_s": result.p_window[1],
            "sv_window_start_s": result.sv_window[0],
            "sv_window_end_s": result.sv_window[1],
            **result.metrics,
        }

        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_qc(
    summary: pd.DataFrame,
) -> tuple[bool, list[str]]:
    """Apply conservative gross-error checks."""
    messages: list[str] = []
    passed = True

    iso = summary.loc[
        summary["kind"] == "isotropic"
    ]

    if len(iso) != 1:
        passed = False
        messages.append(
            "Expected exactly one isotropic case."
        )
    else:
        iso_row = iso.iloc[0]

        if (
            float(iso_row["p_pattern_relative_l2"])
            > MAX_ISOTROPIC_P_RELATIVE_L2
        ):
            passed = False
            messages.append(
                "Isotropic P pattern is not sufficiently circular: "
                f"relative L2={iso_row['p_pattern_relative_l2']:.3f} "
                f"> {MAX_ISOTROPIC_P_RELATIVE_L2:.3f}."
            )

        if (
            float(iso_row["sv_transverse_to_p_radial_energy"])
            > MAX_ISOTROPIC_SV_TO_P_ENERGY
        ):
            passed = False
            messages.append(
                "Isotropic source produces excessive SV energy: "
                "SV/P="
                f"{iso_row['sv_transverse_to_p_radial_energy']:.3e} "
                f"> {MAX_ISOTROPIC_SV_TO_P_ENERGY:.3e}."
            )

    dc = summary.loc[
        summary["kind"] == "dc"
    ]

    for _, row in dc.iterrows():
        p_similarity = float(
            row["p_pattern_similarity"]
        )

        sv_similarity = float(
            row["sv_pattern_similarity"]
        )

        if p_similarity < MIN_PATTERN_SIMILARITY:
            passed = False
            messages.append(
                f"{row['label']}: P pattern similarity "
                f"{p_similarity:.3f} < "
                f"{MIN_PATTERN_SIMILARITY:.3f}."
            )

        if sv_similarity < MIN_PATTERN_SIMILARITY:
            passed = False
            messages.append(
                f"{row['label']}: SV pattern similarity "
                f"{sv_similarity:.3f} < "
                f"{MIN_PATTERN_SIMILARITY:.3f}."
            )

    if passed:
        messages.append(
            "All conservative source-radiation QC thresholds passed."
        )

    return passed, messages


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    model = build_homogeneous_model()
    grid = model.grid

    x_src, z_src = source_position(grid)

    receivers, angles_rad = build_receiver_ring(
        grid,
        x_src=x_src,
        z_src=z_src,
    )

    source_t0_s = 1.2 / SOURCE_F0_HZ

    p_arrival_s = (
        source_t0_s
        + RECEIVER_RADIUS_M / VP_M_S
    )

    sv_arrival_s = (
        source_t0_s
        + RECEIVER_RADIUS_M / VS_M_S
    )

    earliest_boundary_s = earliest_outer_boundary_return_s(
        grid,
        x_src=x_src,
        z_src=z_src,
    )

    print("Source-radiation validation setup")
    print("---------------------------------")
    print(f"grid                    : {grid.nx} x {grid.nz}")
    print(f"dx, dz                  : {grid.dx:.1f}, {grid.dz:.1f} m")
    print(f"dt, nt                  : {grid.dt:.6e} s, {grid.nt}")
    print(f"record end              : {grid.t if hasattr(grid, 't') else RECORD_END_S}")
    print(f"Vp, Vs, rho             : {VP_M_S:.1f}, {VS_M_S:.1f}, {RHO_KG_M3:.1f}")
    print(f"source x,z              : {x_src:.2f}, {z_src:.2f} m")
    print(f"receiver radius         : {RECEIVER_RADIUS_M:.1f} m")
    print(f"receiver count          : {receivers.nrec}")
    print(f"receiver angular step   : {360.0 / receivers.nrec:.2f} deg")
    print(f"expected P centre       : {p_arrival_s:.4f} s")
    print(f"expected SV centre      : {sv_arrival_s:.4f} s")
    print(f"earliest boundary return: {earliest_boundary_s:.4f} s")
    print(f"record duration         : {(grid.nt - 1) * grid.dt:.4f} s")

    if (grid.nt - 1) * grid.dt >= earliest_boundary_s:
        raise ValueError(
            "Recording reaches the conservative boundary-return time. "
            "Shorten RECORD_END_S or enlarge the model."
        )

    if sv_arrival_s + S_WINDOW_HALF_WIDTH_S >= earliest_boundary_s:
        raise ValueError(
            "SV analysis window may contain a boundary return."
        )

    results: list[RadiationCaseResult] = []

    for case in SOURCE_CASES:
        result = run_case(
            model=model,
            receivers=receivers,
            angles_rad=angles_rad,
            x_src=x_src,
            z_src=z_src,
            label=str(case["label"]),
            kind=str(case["kind"]),
            theta_deg=float(case["theta_deg"]),
        )

        results.append(result)

        plot_case_radiation(
            result,
            OUT_DIR / f"{result.label}_radiation.png",
        )

        plot_reference_traces(
            result,
            OUT_DIR / f"{result.label}_reference_traces.png",
        )

    amplitude_table = build_amplitude_table(results)
    summary_table = build_summary_table(results)

    amplitude_path = (
        OUT_DIR
        / "source_radiation_amplitudes.csv"
    )

    summary_path = (
        OUT_DIR
        / "source_radiation_summary.csv"
    )

    amplitude_table.to_csv(
        amplitude_path,
        index=False,
    )

    summary_table.to_csv(
        summary_path,
        index=False,
    )

    plot_summary_metrics(
        summary_table,
        OUT_DIR / "source_radiation_metrics.png",
    )

    # Save receiver traces for reproducibility and later diagnostic work.
    np.savez_compressed(
        OUT_DIR / "source_radiation_traces.npz",
        time_s=results[0].time_s,
        angles_rad=angles_rad,
        angles_deg=np.rad2deg(angles_rad),
        receiver_x=receivers.x,
        receiver_z=receivers.z,
        source_x_m=x_src,
        source_z_m=z_src,
        labels=np.array(
            [result.label for result in results],
            dtype=object,
        ),
        receiver_vx=np.stack(
            [result.receiver_vx for result in results],
            axis=0,
        ),
        receiver_vz=np.stack(
            [result.receiver_vz for result in results],
            axis=0,
        ),
        radial_velocity=np.stack(
            [result.radial_velocity for result in results],
            axis=0,
        ),
        transverse_velocity=np.stack(
            [result.transverse_velocity for result in results],
            axis=0,
        ),
        theoretical_p=np.stack(
            [result.theoretical_p for result in results],
            axis=0,
        ),
        theoretical_sv=np.stack(
            [result.theoretical_sv for result in results],
            axis=0,
        ),
        measured_p=np.stack(
            [result.measured_p for result in results],
            axis=0,
        ),
        measured_sv=np.stack(
            [result.measured_sv for result in results],
            axis=0,
        ),
        dt_s=grid.dt,
        dx_m=grid.dx,
        dz_m=grid.dz,
        vp_m_s=VP_M_S,
        vs_m_s=VS_M_S,
        rho_kg_m3=RHO_KG_M3,
        source_f0_hz=SOURCE_F0_HZ,
        source_scalar_moment=SOURCE_SCALAR_MOMENT,
        n_boundary=N_BOUNDARY,
        gamma_s=GAMMA_S,
    )

    passed, qc_messages = evaluate_qc(
        summary_table
    )

    print("\nSource-radiation QC")
    print("-------------------")

    for message in qc_messages:
        print(message)

    print("\nSaved outputs")
    print("-------------")
    print(summary_path)
    print(amplitude_path)
    print(OUT_DIR / "source_radiation_metrics.png")
    print(OUT_DIR / "source_radiation_traces.npz")

    for result in results:
        print(OUT_DIR / f"{result.label}_radiation.png")
        print(OUT_DIR / f"{result.label}_reference_traces.png")

    if not passed:
        raise RuntimeError(
            "Source-radiation validation FAILED. "
            "Inspect the saved patterns and summary table."
        )

    print("\nSource-radiation validation PASSED.")


if __name__ == "__main__":
    main()
