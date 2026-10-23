"""Check original Lin Vp/Vs source and physical receiver coverage without DAS/FMM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.safod.settings import GEO_XLSX, SAFOD_SURFACE_ELEVATION_M
from src.safod.lin2010 import Lin3D
from src.safod.travel_times import DEFAULT_LIN_VP, DEFAULT_LIN_VS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vp", type=Path, default=DEFAULT_LIN_VP)
    parser.add_argument("--vs", type=Path, default=DEFAULT_LIN_VS)
    parser.add_argument("--catalog", type=Path,
                        default=Path("results/safod_deep/catalog/catalog_recorded.csv"))
    parser.add_argument("--geometry", type=Path, default=GEO_XLSX)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/validation/safod/lin2010_coverage"))
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a new output directory for coverage validation.")
    model = Lin3D(args.vp, args.vs)
    events = pd.read_csv(args.catalog)
    if len(events) == 0:
        raise ValueError("The event catalogue is empty.")
    geometry = (pd.read_excel(args.geometry) if args.geometry.suffix.lower() in (".xlsx", ".xls")
                else pd.read_csv(args.geometry))
    if "Section" in geometry:
        geometry = geometry[geometry["Section"].astype(str).str.strip().isin(("Down-leg", "Up-leg"))]
    columns = ["Lat_WGS84", "Lon_WGS84", "TVD_m"]
    receivers = geometry[columns].to_numpy(dtype=float)
    if len(receivers) == 0 or not np.all(np.isfinite(receivers)):
        raise ValueError("Physical borehole geometry must contain finite, nonempty receiver coordinates.")
    rec_vp, rec_vs = [grid.sample(receivers[:, 0], receivers[:, 1],
                     (receivers[:, 2]-SAFOD_SURFACE_ELEVATION_M)/1000)
                     for grid in model.interpolators]
    if np.any(rec_vp <= rec_vs):
        raise ValueError("Lin Vp <= Vs at a physical receiver.")
    rows = []
    for _, event in events.iterrows():
        row = dict(event_id=str(event["event_id"]), covered=False)
        try:
            vp, vs = [float(grid.sample(float(event["latitude"]), float(event["longitude"]),
                      float(event["depth_km"]))) for grid in model.interpolators]
            if vp <= vs:
                raise ValueError("Vp <= Vs at source")
            row.update(covered=True, vp_km_s=vp, vs_km_s=vs)
        except (ValueError, RuntimeError) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    covered = sum(row["covered"] for row in rows)
    report = dict(passed=(covered == len(rows)), events=len(rows), events_covered=covered,
                  physical_receivers_covered=len(receivers), model_name=model.model_name,
                  model_fields=[grid.info() for grid in model.interpolators],
                  surface_elevation_m=SAFOD_SURFACE_ELEVATION_M,
                  scope="Source/receiver interpolation support only; no eikonal solution, DAS read, or publication")
    args.output_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "events.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Lin coverage: {covered}/{len(rows)} sources; {len(receivers)} physical receivers")
    if not report["passed"]:
        raise RuntimeError("Lin coverage failed; see events.csv.")


if __name__ == "__main__":
    main()
