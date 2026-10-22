"""Independent homogeneous-medium QC of the installed EikoHypoTomo wrapper."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from src.safod.travel_times import DEFAULT_EIKO_ROOT, solve_eikonal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eiko-root", type=Path, default=DEFAULT_EIKO_ROOT)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/validation/safod/travel_times_3d"))
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a new --output-dir; existing QC is never overwritten.")
    source = np.array([111., -237., 517.])
    receivers = np.array([[1300., 850., 1800.], [-900., 700., 1300.],
                          [800., -1500., 900.], [-1700., -1100., 1500.]])
    distance = np.linalg.norm(receivers - source, axis=1)
    results = []
    traces = {}
    for h in (200., 100., 50.):
        axes = (np.arange(-2000., 2000. + h/2, h),
                np.arange(-2000., 2000. + h/2, h),
                np.arange(-500., 3500. + h/2, h))
        for phase, speed in (("P", 5500.), ("S", 3200.)):
            values, backend = solve_eikonal(np.full(tuple(len(a) for a in axes), speed),
                                            axes, source, receivers, eiko_root=args.eiko_root)
            exact = distance / speed
            error = float(np.max(abs(values - exact)))
            results.append(dict(phase=phase, h_m=h, max_abs_error_s=error))
            traces[f"{phase}_{h:g}"] = values
            print(f"{phase} h={h:g} m: max |T-T_exact| = {error:.6g} s")
    for phase in ("P", "S"):
        errors = [r["max_abs_error_s"] for r in results if r["phase"] == phase]
        if not errors[-1] < errors[0] or errors[-1] > 0.005:
            raise RuntimeError(f"{phase}: homogeneous refinement/5-ms criterion failed: {errors}")
    np.testing.assert_allclose(traces["S_50"] / traces["P_50"], 5500 / 3200, rtol=1e-8)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "traces.npz", source_xyz_m=source,
                        receivers_xyz_m=receivers, **traces)
    report = dict(passed=True, backend=backend, results=results,
                   reference="Euclidean distance / constant velocity; off-grid source")
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print("EikoHypoTomo 3-D homogeneous travel-time QC PASSED")


if __name__ == "__main__":
    main()
