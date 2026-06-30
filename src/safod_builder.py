# SAFOD 2D model builder for elastic-DAS forward modelling / future FWI.
#
# Purpose
# -------
# Build a geologically informed 2D elastic model around the projected SAFOD DAS
# cable geometry.
#
# Model types
# -----------
# build_initial_model=True
#     Smooth geologically informed initial/prior model:
#       - sonic-log-inspired depth trend
#       - broad Pilot-Hole fractured low-velocity intervals
#       - smooth cross-fault contrast
#       - broad SAF low-velocity damage zone
#
#     This is NOT a true model. It is intentionally smooth enough for forward
#     modelling and future FWI starts.
#
# build_initial_model=False
#     Stronger synthetic/reference model:
#       - stronger cross-fault contrast
#       - stronger/narrower SAF damage zone
#       - stronger Pilot-Hole LVZs
#
# Coordinate convention
# ---------------------
# x : horizontal coordinate in projected 2D section [m]
# z : depth-positive coordinate [m]
#
# Fault geometry
# --------------
# The San Andreas fault prior is represented as a dipping line:
#
#     x_fault(z) = x_tie + fault_dip_sign * (z - z_tie) / tan(dip)
#
# where dip is in degrees from horizontal. A vertical fault would be close
# to 90 degrees.
#
# Cable / fault relation
# ----------------------
# The projected DAS cable does not cross the SAF. By default, the SAF prior
# is anchored to the deepest/end point of the projected cable, offset laterally
# by fault_offset_from_cable_m.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math

import numpy as np

from src.grid import Grid2D
from src.model import ElasticModel2D
from src.solver_numpy import max_stable_dt


# ==============================================================================
# 1. METADATA
# ==============================================================================

@dataclass(frozen=True)
class SafodBuildMetadata:
    geom_file: str
    model_type: str

    # Fault tie point
    x_tie_m: float
    z_tie_m: float
    x_cable_at_tie_m: float

    # Cable end used for anchoring
    x_cable_end_m: float
    z_cable_end_m: float
    anchor_fault_to_cable_end: bool

    # Fault prior
    fault_offset_from_cable_m: float
    fault_dip_deg: float
    fault_dip_sign: float
    left_block_name: str
    right_block_name: str

    # Fault line for plotting / synthetic perturbations
    x_fault_line: np.ndarray
    z_fault_line: np.ndarray

    # Grid extent
    x0_m: float
    z0_m: float
    x1_m: float
    z1_m: float

    dx_m: float
    dz_m: float
    dt_s: float
    nt: int

    # Initial/reference-model parameter record
    cross_fault_contrast: float
    fault_zone_width_m: float
    fault_zone_velocity_reduction: float
    pilot_hole_lvz_strength: float
    smoothing_sigma_m: float

    notes: str


# ==============================================================================
# 2. CSV GEOMETRY HELPERS
# ==============================================================================

def _normalise_column_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _pick_column(
    columns: list[str],
    *,
    requested: str | None,
    candidates: tuple[str, ...],
    semantic_name: str,
) -> str:
    """
    Pick a column robustly from a CSV header.
    """
    if requested is not None:
        if requested in columns:
            return requested

        requested_norm = _normalise_column_name(requested)
        for col in columns:
            if _normalise_column_name(col) == requested_norm:
                return col

        raise ValueError(
            f"Requested {semantic_name} column {requested!r} not found. "
            f"Available columns: {columns}"
        )

    norm_to_col = {_normalise_column_name(col): col for col in columns}

    # Exact normalised match.
    for cand in candidates:
        cand_norm = _normalise_column_name(cand)
        if cand_norm in norm_to_col:
            return norm_to_col[cand_norm]

    # Controlled substring match. Avoid single-letter accidents like x in index.
    for col_norm, col in norm_to_col.items():
        for cand in candidates:
            cand_norm = _normalise_column_name(cand)
            if len(cand_norm) > 1 and cand_norm in col_norm:
                return col

    raise ValueError(
        f"Could not identify {semantic_name} column. "
        f"Available columns: {columns}. "
        f"Pass x_column=... or z_column=... explicitly."
    )


def load_projected_cable_geometry(
    geom_file: str | Path,
    *,
    x_column: str | None = None,
    z_column: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load projected 2D SAFOD cable geometry from a CSV file.

    Returns
    -------
    x_cable, z_cable : np.ndarray
        Horizontal section coordinate and depth-positive coordinate [m].

    Notes
    -----
    The caller can intentionally swap CSV columns if the file uses names that
    do not match this model convention.
    """
    geom_file = Path(geom_file)

    if not geom_file.exists():
        raise FileNotFoundError(f"Geometry file not found: {geom_file}")

    with geom_file.open("r", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames

        if columns is None:
            raise ValueError(f"CSV file has no header: {geom_file}")

        columns = list(columns)

        x_col = _pick_column(
            columns,
            requested=x_column,
            candidates=(
                "x",
                "x_m",
                "xm",
                "x_projected_m",
                "projected_x_m",
                "x_2d_m",
                "x2d_m",
                "section_x_m",
                "cross_section_x_m",
                "distance_m",
                "profile_x_m",
            ),
            semantic_name="projected x",
        )

        z_col = _pick_column(
            columns,
            requested=z_column,
            candidates=(
                "z",
                "z_m",
                "zm",
                "z_2d_m",
                "z2d_m",
                "projected_z_m",
                "profile_z_m",
                "depth",
                "depth_m",
                "tvd",
                "tvd_m",
                "vertical_depth_m",
            ),
            semantic_name="depth z",
        )

        x_vals: list[float] = []
        z_vals: list[float] = []

        for row in reader:
            try:
                x = float(row[x_col])
                z = float(row[z_col])
            except (TypeError, ValueError):
                continue

            if np.isfinite(x) and np.isfinite(z):
                x_vals.append(x)
                z_vals.append(z)

    if len(x_vals) < 2:
        raise ValueError(
            f"Need at least two valid geometry points in {geom_file}; "
            f"found {len(x_vals)}."
        )

    return np.asarray(x_vals, dtype=np.float64), np.asarray(z_vals, dtype=np.float64)


def _interp_x_at_depth(
    x: np.ndarray,
    z: np.ndarray,
    z_target: float,
) -> float:
    """
    Interpolate cable x-coordinate at target depth.

    This assumes z_target is inside the cable depth range. The caller should
    check that explicitly.
    """
    order = np.argsort(z)
    z_sorted = np.asarray(z[order], dtype=np.float64)
    x_sorted = np.asarray(x[order], dtype=np.float64)

    return float(np.interp(z_target, z_sorted, x_sorted))


# ==============================================================================
# 3. VELOCITY / DENSITY PARAMETERISATION
# ==============================================================================

def _depth_trend_from_pilot_hole_logs(
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Smooth depth trend inspired by SAFOD Pilot Hole sonic/density logs.

    Units
    -----
    Vp, Vs : m/s
    rho    : kg/m^3

    Approximate modelling constraints
    ---------------------------------
    - shallow section above ~768 m is lower velocity
    - Salinian granite below ~768 m
    - from ~775 to 2150 m:
          Vp ~ 5.0 to 5.65 km/s
          Vs ~ 2.8 to 3.25 km/s
          rho ~ 2.5 to 2.7 g/cm^3

    The trend is smooth and conservative. Sharp log-scale features are not
    inserted here.
    """
    z = np.asarray(z, dtype=np.float64)

    vp = np.empty_like(z)
    vs = np.empty_like(z)
    rho = np.empty_like(z)

    # --------------------------------------------------------------------------
    # 0–768 m: shallow sediments / transition into granite.
    # --------------------------------------------------------------------------
    m0 = z < 768.0
    u0 = np.clip(z[m0] / 768.0, 0.0, 1.0)

    vp[m0] = 2200.0 + (5000.0 - 2200.0) * u0**1.15
    vs[m0] = 900.0  + (2800.0 - 900.0)  * u0**1.10
    rho[m0] = 2050.0 + (2500.0 - 2050.0) * u0**0.90

    # --------------------------------------------------------------------------
    # 768–2150 m: Salinian granite interval.
    # --------------------------------------------------------------------------
    m1 = (z >= 768.0) & (z <= 2150.0)
    u1 = np.clip((z[m1] - 768.0) / (2150.0 - 768.0), 0.0, 1.0)

    vp[m1] = 5000.0 + (5650.0 - 5000.0) * u1
    vs[m1] = 2800.0 + (3250.0 - 2800.0) * u1
    rho[m1] = 2500.0 + (2700.0 - 2500.0) * u1

    # --------------------------------------------------------------------------
    # >2150 m: gentle continuation to SAFOD target depth.
    # --------------------------------------------------------------------------
    m2 = z > 2150.0
    dz2 = z[m2] - 2150.0

    vp[m2] = np.minimum(5650.0 + 0.12 * dz2, 6100.0)
    vs[m2] = np.minimum(3250.0 + 0.07 * dz2, 3550.0)
    rho[m2] = np.minimum(2700.0 + 0.015 * dz2, 2760.0)

    # Safety: preserve lambda >= 0, i.e. Vp/Vs >= sqrt(2).
    vs = np.minimum(vs, vp / 1.62)

    return vp, vs, rho


def _add_pilot_hole_lvz(
    vp_z: np.ndarray,
    vs_z: np.ndarray,
    rho_z: np.ndarray,
    z: np.ndarray,
    *,
    strength: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Add broad depth-only low-velocity perturbations at fractured intervals.

    These are intentionally broad, smooth priors, not narrow sonic-log spikes.
    """
    if strength <= 0.0:
        return vp_z, vs_z, rho_z

    zones = [
        # centre depth [m], sigma [m]
        (1175.0, 55.0),
        (1365.0, 85.0),
        (1858.0, 60.0),
    ]

    factor = np.ones_like(z, dtype=np.float64)

    for z0, sigma in zones:
        factor *= 1.0 - strength * np.exp(-0.5 * ((z - z0) / sigma) ** 2)

    vp2 = vp_z * factor
    vs2 = vs_z * factor

    # Density perturbation is weaker than velocity perturbation.
    rho2 = rho_z * (1.0 - 0.25 * (1.0 - factor))

    return vp2, vs2, rho2


def _fault_x_at_z(
    z: np.ndarray,
    *,
    x_tie_m: float,
    z_tie_m: float,
    fault_dip_deg: float,
    fault_dip_sign: float,
) -> np.ndarray:
    """
    Compute x-coordinate of dipping fault line at depths z.
    """
    if not (0.0 < fault_dip_deg < 90.0):
        raise ValueError(
            f"fault_dip_deg must be in (0, 90), got {fault_dip_deg}."
        )

    dip_rad = np.deg2rad(fault_dip_deg)

    return x_tie_m + fault_dip_sign * (z - z_tie_m) / np.tan(dip_rad)


def _apply_cross_fault_contrast(
    vp: np.ndarray,
    vs: np.ndarray,
    rho: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    *,
    x_tie_m: float,
    z_tie_m: float,
    fault_dip_deg: float,
    fault_dip_sign: float,
    transition_width_m: float,
    right_block_velocity_shift: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply smooth cross-fault lithologic velocity contrast.

    right_block_velocity_shift
        Negative means the right side of the plotted fault is slower.
        Positive means the right side is faster.

    This is deliberately smooth for an initial model.
    """
    if transition_width_m <= 0.0:
        raise ValueError(
            f"transition_width_m must be positive, got {transition_width_m}."
        )

    x_fault = _fault_x_at_z(
        Z,
        x_tie_m=x_tie_m,
        z_tie_m=z_tie_m,
        fault_dip_deg=fault_dip_deg,
        fault_dip_sign=fault_dip_sign,
    )

    signed_distance = X - x_fault

    # 0 on left side, 1 on right side, with smooth transition.
    side_right = 0.5 * (1.0 + np.tanh(signed_distance / transition_width_m))

    scale_v = 1.0 + right_block_velocity_shift * side_right
    scale_rho = 1.0 + 0.35 * right_block_velocity_shift * side_right

    return vp * scale_v, vs * scale_v, rho * scale_rho


def _apply_fault_damage_zone(
    vp: np.ndarray,
    vs: np.ndarray,
    rho: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    *,
    x_tie_m: float,
    z_tie_m: float,
    fault_dip_deg: float,
    fault_dip_sign: float,
    width_m: float,
    velocity_reduction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply broad Gaussian low-velocity damage zone around the SAF prior.

    For this near-vertical SAFOD section, horizontal distance to x_fault(z)
    is a practical approximation to fault-normal distance.
    """
    if width_m <= 0.0 or velocity_reduction <= 0.0:
        return vp, vs, rho

    if not (0.0 <= velocity_reduction < 1.0):
        raise ValueError(
            f"velocity_reduction must be in [0, 1), got {velocity_reduction}."
        )

    x_fault = _fault_x_at_z(
        Z,
        x_tie_m=x_tie_m,
        z_tie_m=z_tie_m,
        fault_dip_deg=fault_dip_deg,
        fault_dip_sign=fault_dip_sign,
    )

    horizontal_distance = X - x_fault
    damage = np.exp(-0.5 * (horizontal_distance / width_m) ** 2)

    scale_v = 1.0 - velocity_reduction * damage
    scale_rho = 1.0 - 0.20 * velocity_reduction * damage
    
    return vp * scale_v, vs * scale_v, rho * scale_rho


def _smooth_2d(
    arr: np.ndarray,
    sigma_x_cells: float,
    sigma_z_cells: float,
) -> np.ndarray:
    """
    Gaussian smooth. Requires scipy; if unavailable, returns input unchanged.
    """
    if sigma_x_cells <= 0.0 and sigma_z_cells <= 0.0:
        return arr

    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        return arr

    return gaussian_filter(
        arr,
        sigma=(sigma_x_cells, sigma_z_cells),
        mode="nearest",
    )


def _check_elastic_physicality(
    vp: np.ndarray,
    vs: np.ndarray,
    rho: np.ndarray,
) -> None:
    """
    Basic isotropic elastic sanity checks.
    """
    if (
        np.any(~np.isfinite(vp))
        or np.any(~np.isfinite(vs))
        or np.any(~np.isfinite(rho))
    ):
        raise ValueError("vp, vs, rho must be finite everywhere.")

    if np.any(vp <= 0.0) or np.any(vs <= 0.0) or np.any(rho <= 0.0):
        raise ValueError("vp, vs, rho must be positive everywhere.")

    mu = rho * vs**2
    lam = rho * (vp**2 - 2.0 * vs**2)

    if np.any(mu <= 0.0):
        raise ValueError(f"mu must be positive; min(mu)={mu.min():.3e}.")

    if np.any(lam < 0.0):
        raise ValueError(
            f"lambda must be non-negative for this solver; "
            f"min(lambda)={lam.min():.3e}. Reduce Vs or increase Vp."
        )


# ==============================================================================
# 4. MAIN BUILDER
# ==============================================================================

def build_safod_model(
    *,
    geom_file: str | Path,
    build_initial_model: bool = True,
    dx: float = 5.0,
    dz: float = 5.0,
    dt: float | None = None,
    nt: int = 3000,
    half_order: int = 2,
    cfl_safety: float = 0.80,
    x_column: str | None = None,
    z_column: str | None = None,

    # Fault / cable geometry
    z_tie_m: float | None = None,
    anchor_fault_to_cable_end: bool = True,
    fault_offset_from_cable_m: float = 105.0,
    fault_dip_deg: float = 82.0,
    fault_dip_sign: float = -1.0,
    left_block_name: str = "salinian",
    right_block_name: str = "franciscan",

    # Grid padding
    x_padding_m: float = 800.0,
    z_padding_bottom_m: float = 700.0,
    min_x_width_m: float = 2500.0,
    z_max_m: float | None = None,

    # Geologically informed initial-model priors
    initial_cross_fault_contrast: float = -0.08,
    initial_cross_fault_transition_m: float = 350.0,
    initial_fault_zone_width_m: float = 160.0,
    initial_fault_zone_velocity_reduction: float = 0.14,
    include_pilot_hole_lvz_in_initial: bool = True,
    initial_pilot_hole_lvz_strength: float = 0.035,
    smooth_initial_sigma_m: float = 80.0,

    # Stronger reference/synthetic perturbations
    reference_cross_fault_contrast: float = -0.12,
    reference_cross_fault_transition_m: float = 220.0,
    reference_fault_zone_width_m: float = 100.0,
    reference_fault_zone_velocity_reduction: float = 0.22,
    include_pilot_hole_lvz_in_reference: bool = True,
    reference_pilot_hole_lvz_strength: float = 0.050,
    smooth_reference_sigma_m: float = 25.0,
) -> tuple[Grid2D, ElasticModel2D, np.ndarray, np.ndarray, SafodBuildMetadata]:
    """
    Build a SAFOD 2D elastic model and projected cable geometry.

    Parameters
    ----------
    geom_file
        CSV file containing projected 2D cable coordinates.

    build_initial_model
        True:
            geologically informed smooth initial model.
        False:
            stronger geologic/reference synthetic model.

    x_column, z_column
        CSV columns to use as model x and model z. For the current projected
        SAFOD geometry file, the correct mapping appears to be:
            x_column="Z_2D_m"
            z_column="X_2D_m"

    dt
        If None, choose CFL-safe dt automatically. If provided and too large,
        raise ValueError.

    fault_offset_from_cable_m
        Horizontal distance from cable tie point to SAF prior line.
        Flip sign if the plotted SAF line appears on the wrong side.

    Returns
    -------
    grid, model, x_cable, z_cable, metadata
    """
    if dx <= 0.0 or dz <= 0.0:
        raise ValueError(f"dx and dz must be positive; got dx={dx}, dz={dz}.")

    if nt <= 1:
        raise ValueError(f"nt must be > 1, got {nt}.")

    x_cable, z_cable = load_projected_cable_geometry(
        geom_file,
        x_column=x_column,
        z_column=z_column,
    )

    if np.any(~np.isfinite(x_cable)) or np.any(~np.isfinite(z_cable)):
        raise ValueError("Cable coordinates must be finite.")

    if np.any(z_cable < -1.0):
        raise ValueError(
            "z_cable appears to contain negative elevations rather than "
            "positive depth. Convert to depth-positive z before building model."
        )

    # --------------------------------------------------------------------------
    # Fault anchor / tie point
    # --------------------------------------------------------------------------
    i_cable_end = int(np.argmax(z_cable))
    x_cable_end_m = float(x_cable[i_cable_end])
    z_cable_end_m = float(z_cable[i_cable_end])

    if anchor_fault_to_cable_end or z_tie_m is None:
        z_tie_m = z_cable_end_m
        x_cable_at_tie = x_cable_end_m
    else:
        z_min_cable = float(np.min(z_cable))
        z_max_cable = float(np.max(z_cable))

        if not (z_min_cable <= float(z_tie_m) <= z_max_cable):
            raise ValueError(
                f"Requested z_tie_m={z_tie_m:.1f} m is outside cable depth range "
                f"[{z_min_cable:.1f}, {z_max_cable:.1f}] m. "
                "Use anchor_fault_to_cable_end=True, or choose z_tie_m inside cable."
            )

        x_cable_at_tie = _interp_x_at_depth(
            x_cable,
            z_cable,
            float(z_tie_m),
        )

    x_tie_m = float(x_cable_at_tie) + float(fault_offset_from_cable_m)

    # --------------------------------------------------------------------------
    # Grid extents
    # --------------------------------------------------------------------------
    if z_max_m is None:
        z_max_m = max(
            float(np.max(z_cable)) + z_padding_bottom_m,
            float(z_tie_m) + z_padding_bottom_m,
            2600.0,
        )

    z0_m = 0.0
    z1_m = math.ceil(float(z_max_m) / dz) * dz

    z_fault_line = np.linspace(z0_m, z1_m, 700)
    x_fault_line = _fault_x_at_z(
        z_fault_line,
        x_tie_m=x_tie_m,
        z_tie_m=float(z_tie_m),
        fault_dip_deg=fault_dip_deg,
        fault_dip_sign=fault_dip_sign,
    )

    x_min_raw = min(float(np.min(x_cable)), float(np.min(x_fault_line))) - x_padding_m
    x_max_raw = max(float(np.max(x_cable)), float(np.max(x_fault_line))) + x_padding_m

    if x_max_raw - x_min_raw < min_x_width_m:
        x_mid = 0.5 * (x_min_raw + x_max_raw)
        x_min_raw = x_mid - 0.5 * min_x_width_m
        x_max_raw = x_mid + 0.5 * min_x_width_m

    x0_m = math.floor(x_min_raw / dx) * dx
    x1_m = math.ceil(x_max_raw / dx) * dx

    nx = int(round((x1_m - x0_m) / dx)) + 1
    nz = int(round((z1_m - z0_m) / dz)) + 1

    x = x0_m + np.arange(nx, dtype=np.float64) * dx
    z = z0_m + np.arange(nz, dtype=np.float64) * dz

    X, Z = np.meshgrid(x, z, indexing="ij")

    # --------------------------------------------------------------------------
    # Base 1D trend
    # --------------------------------------------------------------------------
    vp_z, vs_z, rho_z = _depth_trend_from_pilot_hole_logs(z)

    vp = np.tile(vp_z[None, :], (nx, 1))
    vs = np.tile(vs_z[None, :], (nx, 1))
    rho = np.tile(rho_z[None, :], (nx, 1))

    # --------------------------------------------------------------------------
    # Geological priors
    # --------------------------------------------------------------------------
    if build_initial_model:
        model_type = "initial_geologic_prior_smooth"

        cross_fault_contrast = float(initial_cross_fault_contrast)
        fault_zone_width_m = float(initial_fault_zone_width_m)
        fault_zone_velocity_reduction = float(initial_fault_zone_velocity_reduction)
        pilot_hole_lvz_strength = (
            float(initial_pilot_hole_lvz_strength)
            if include_pilot_hole_lvz_in_initial
            else 0.0
        )
        smoothing_sigma_m = float(smooth_initial_sigma_m)

        # 1. Broad Pilot-Hole fractured LVZs as depth priors.
        if include_pilot_hole_lvz_in_initial:
            vp_z_ini, vs_z_ini, rho_z_ini = _add_pilot_hole_lvz(
                vp_z,
                vs_z,
                rho_z,
                z,
                strength=initial_pilot_hole_lvz_strength,
            )

            vp *= (vp_z_ini / vp_z)[None, :]
            vs *= (vs_z_ini / vs_z)[None, :]
            rho *= (rho_z_ini / rho_z)[None, :]

        # 2. Smooth cross-fault contrast.
        if abs(initial_cross_fault_contrast) > 0.0:
            vp, vs, rho = _apply_cross_fault_contrast(
                vp,
                vs,
                rho,
                X,
                Z,
                x_tie_m=x_tie_m,
                z_tie_m=float(z_tie_m),
                fault_dip_deg=fault_dip_deg,
                fault_dip_sign=fault_dip_sign,
                transition_width_m=initial_cross_fault_transition_m,
                right_block_velocity_shift=initial_cross_fault_contrast,
            )

        # 3. Broad SAF low-velocity damage zone.
        if initial_fault_zone_velocity_reduction > 0.0:
            vp, vs, rho = _apply_fault_damage_zone(
                vp,
                vs,
                rho,
                X,
                Z,
                x_tie_m=x_tie_m,
                z_tie_m=float(z_tie_m),
                fault_dip_deg=fault_dip_deg,
                fault_dip_sign=fault_dip_sign,
                width_m=initial_fault_zone_width_m,
                velocity_reduction=initial_fault_zone_velocity_reduction,
            )

        sigma_x = smooth_initial_sigma_m / dx
        sigma_z = smooth_initial_sigma_m / dz

    else:
        model_type = "geologic_reference_synthetic"

        cross_fault_contrast = float(reference_cross_fault_contrast)
        fault_zone_width_m = float(reference_fault_zone_width_m)
        fault_zone_velocity_reduction = float(reference_fault_zone_velocity_reduction)
        pilot_hole_lvz_strength = (
            float(reference_pilot_hole_lvz_strength)
            if include_pilot_hole_lvz_in_reference
            else 0.0
        )
        smoothing_sigma_m = float(smooth_reference_sigma_m)

        # 1. Stronger reference cross-fault contrast.
        vp, vs, rho = _apply_cross_fault_contrast(
            vp,
            vs,
            rho,
            X,
            Z,
            x_tie_m=x_tie_m,
            z_tie_m=float(z_tie_m),
            fault_dip_deg=fault_dip_deg,
            fault_dip_sign=fault_dip_sign,
            transition_width_m=reference_cross_fault_transition_m,
            right_block_velocity_shift=reference_cross_fault_contrast,
        )

        # 2. Stronger synthetic SAF damage zone.
        vp, vs, rho = _apply_fault_damage_zone(
            vp,
            vs,
            rho,
            X,
            Z,
            x_tie_m=x_tie_m,
            z_tie_m=float(z_tie_m),
            fault_dip_deg=fault_dip_deg,
            fault_dip_sign=fault_dip_sign,
            width_m=reference_fault_zone_width_m,
            velocity_reduction=reference_fault_zone_velocity_reduction,
        )

        # 3. Stronger broad Pilot-Hole fractured LVZs.
        if include_pilot_hole_lvz_in_reference:
            vp_z_ref, vs_z_ref, rho_z_ref = _add_pilot_hole_lvz(
                vp_z,
                vs_z,
                rho_z,
                z,
                strength=reference_pilot_hole_lvz_strength,
            )

            vp *= (vp_z_ref / vp_z)[None, :]
            vs *= (vs_z_ref / vs_z)[None, :]
            rho *= (rho_z_ref / rho_z)[None, :]

        sigma_x = smooth_reference_sigma_m / dx
        sigma_z = smooth_reference_sigma_m / dz

    # --------------------------------------------------------------------------
    # Smooth final model.
    # --------------------------------------------------------------------------
    vp = _smooth_2d(vp, sigma_x, sigma_z)
    vs = _smooth_2d(vs, sigma_x, sigma_z)
    rho = _smooth_2d(rho, sigma_x, sigma_z)

    _check_elastic_physicality(vp, vs, rho)

    # --------------------------------------------------------------------------
    # Time step
    # --------------------------------------------------------------------------
    dt_max = max_stable_dt(
        float(np.max(vp)),
        dx,
        dz,
        half_order,
        safety=cfl_safety,
        use_ts_sfd=False,
    )

    if dt is None:
        dt_s = dt_max
    else:
        dt_s = float(dt)

        if dt_s > dt_max:
            raise ValueError(
                f"Requested dt={dt_s:.6e} s is too large for this model/grid. "
                f"CFL-safe dt <= {dt_max:.6e} s for dx={dx}, dz={dz}, "
                f"vp_max={float(np.max(vp)):.1f} m/s, half_order={half_order}. "
                "Use dt=None to choose automatically."
            )

    # --------------------------------------------------------------------------
    # Grid / model object
    # --------------------------------------------------------------------------
    grid = Grid2D(
        nx=nx,
        nz=nz,
        dx=dx,
        dz=dz,
        nt=nt,
        dt=dt_s,
        x0=x0_m,
        z0=z0_m,
    )

    model = ElasticModel2D(
        grid=grid,
        vp=vp.astype(np.float64),
        vs=vs.astype(np.float64),
        rho=rho.astype(np.float64),
    )

    notes = (
        "SAFOD model built by src.safod_builder. Initial model is a "
        "geologically informed smooth prior: sonic-log-inspired depth trend, "
        "broad Pilot-Hole fractured low-velocity intervals, smooth cross-fault "
        "velocity contrast, and broad SAF low-velocity damage zone. It is not "
        "a true model. SAF prior is anchored to the projected DAS cable end "
        "unless anchor_fault_to_cable_end=False."
    )

    metadata = SafodBuildMetadata(
        geom_file=str(geom_file),
        model_type=model_type,

        x_tie_m=float(x_tie_m),
        z_tie_m=float(z_tie_m),
        x_cable_at_tie_m=float(x_cable_at_tie),

        x_cable_end_m=float(x_cable_end_m),
        z_cable_end_m=float(z_cable_end_m),
        anchor_fault_to_cable_end=bool(anchor_fault_to_cable_end),

        fault_offset_from_cable_m=float(fault_offset_from_cable_m),
        fault_dip_deg=float(fault_dip_deg),
        fault_dip_sign=float(fault_dip_sign),
        left_block_name=str(left_block_name),
        right_block_name=str(right_block_name),

        x_fault_line=np.asarray(x_fault_line, dtype=np.float64),
        z_fault_line=np.asarray(z_fault_line, dtype=np.float64),

        x0_m=float(x0_m),
        z0_m=float(z0_m),
        x1_m=float(x1_m),
        z1_m=float(z1_m),

        dx_m=float(dx),
        dz_m=float(dz),
        dt_s=float(dt_s),
        nt=int(nt),

        cross_fault_contrast=float(cross_fault_contrast),
        fault_zone_width_m=float(fault_zone_width_m),
        fault_zone_velocity_reduction=float(fault_zone_velocity_reduction),
        pilot_hole_lvz_strength=float(pilot_hole_lvz_strength),
        smoothing_sigma_m=float(smoothing_sigma_m),

        notes=notes,
    )

    return grid, model, x_cable, z_cable, metadata