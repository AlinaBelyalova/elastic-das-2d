"""Project a ComCat point double couple into the existing SAFOD x-z section."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.elastic.source import MomentTensor2D


MANUAL_MECHANISM = "manual_2d_dc"
COMCAT_MECHANISM = "comcat_projected_dc"


def double_couple_ned(strike_deg, dip_deg, rake_deg):
    """Unit scalar moment in true north, east, down (Aki–Richards SDR)."""
    angles = np.array([strike_deg, dip_deg, rake_deg], dtype=float)
    if not np.isfinite(angles).all() or not (
        0 <= angles[0] <= 360 and 0 <= angles[1] <= 90 and -180 <= angles[2] <= 180
    ):
        raise ValueError(f"Invalid strike/dip/rake in degrees: {angles}.")
    strike, dip, rake = np.deg2rad(angles)
    along_strike = np.array([np.cos(strike), np.sin(strike), 0.0])
    right = np.array([-np.sin(strike), np.cos(strike), 0.0])
    down = np.array([0.0, 0.0, 1.0])
    down_dip = np.cos(dip) * right + np.sin(dip) * down
    upward_normal = np.sin(dip) * right - np.cos(dip) * down
    # Positive rake points up-dip (reverse slip), not down-dip.
    slip = np.cos(rake) * along_strike - np.sin(rake) * down_dip
    return np.outer(slip, upward_normal) + np.outer(upward_normal, slip)


def load_projected_comcat_source(
    *, path, event_id, profile_u_e, profile_u_n, scalar_moment
):
    """Return the scaled x-z tensor and NPZ provenance from the viewer cache."""
    path = Path(path)
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"ComCat mechanism cache missing: {path}. Populate the focal mechanism "
            "catalog or use --theta-deg explicitly."
        ) from exc
    if not isinstance(entries, dict):
        raise ValueError(f"Expected an event-keyed ComCat mechanism cache: {path}.")
    event_key = str(event_id).strip().lower()
    # Prepared NCEDC packages can use either NC753... or the numeric ID.
    event_key = f"nc{event_key}" if event_key.isdigit() else event_key
    entry = entries.get(event_key)
    if entry is None:
        raise ValueError(
            f"No ComCat mechanism for {event_key} in {path}. "
            "Use --theta-deg explicitly if a trial 2D source is appropriate."
        )
    try:
        sdr = np.array([entry["np1"][k] for k in ("strike", "dip", "rake")], dtype=float)
        code = entry["code"]
        if not isinstance(code, str) or not code.strip():
            raise ValueError("Missing product code.")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Incomplete ComCat mechanism for {event_key} in {path}.") from exc
    m_ned_unit = double_couple_ned(*sdr)

    # prepare_event stores a local geographic east/north unit direction for +x.
    # Use the same local tangent-plane convention, not UTM/grid north.
    along = np.array([profile_u_n, profile_u_e, 0.0], dtype=float)
    norm = np.linalg.norm(along)
    if not np.isfinite(norm) or not np.isclose(norm, 1.0, rtol=0, atol=1e-7):
        raise ValueError(
            "Missing or invalid prepared profile_u_e/profile_u_n. "
            "Rerun scripts.safod.prepare_event."
        )
    basis_xz = np.array([along / norm, [0.0, 0.0, 1.0]])
    projected = basis_xz @ m_ned_unit @ basis_xz.T
    if np.linalg.norm(projected) <= 1e-12:
        raise ValueError(
            f"ComCat mechanism {code} has negligible P-SV source projection "
            "onto this profile; a 2D P-SV run cannot represent it."
        )
    if not np.isfinite(scalar_moment) or scalar_moment <= 0:
        raise ValueError("The numerical source moment scale must be finite and positive.")
    # Preserve trace, polarity and relative amplitude. This is a submatrix
    # of a 3D double couple, not a fit to another trace-free 2D double couple.
    m_xz = projected * scalar_moment
    return MomentTensor2D(Mxx=m_xz[0, 0], Mzz=m_xz[1, 1], Mxz=m_xz[0, 1]), {
        "source_comcat_product_code": np.array(code),
        "source_comcat_event_key": np.array(event_key),
        "source_comcat_review_status": np.array(str(entry.get("review_status", ""))),
        "source_comcat_evaluation_status": np.array(str(entry.get("evaluation_status", ""))),
        "source_comcat_strike_dip_rake_deg": sdr,
        "source_comcat_unit_moment_ned": m_ned_unit,
        "source_profile_basis_xz_ned": basis_xz,
    }


def validate_saved_source_mechanism(package, theta_deg):
    """Check manual angle or projected ComCat identity; support old manual NPZs."""
    requested = COMCAT_MECHANISM if theta_deg is None else MANUAL_MECHANISM
    stored = str(package["source_mechanism"].item()) if "source_mechanism" in package else MANUAL_MECHANISM
    if stored != requested:
        raise ValueError(f"Requested source mechanism {requested!r} differs from saved {stored!r}.")
    if theta_deg is not None:
        if "source_theta_deg" not in package or not np.isclose(
            float(package["source_theta_deg"]), theta_deg, rtol=0.0, atol=1e-9
        ):
            raise ValueError("Requested theta does not match synthetic package.")
        return ""
    if "source_comcat_product_code" not in package:
        raise ValueError("Synthetic package is missing its ComCat mechanism product code.")
    code = str(package["source_comcat_product_code"].item())
    if not code.strip():
        raise ValueError("Synthetic package has an empty ComCat mechanism product code.")
    return code
