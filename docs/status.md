# Project status

Date: 2026-10-22. Only the current stage, the latest verified runs,
next steps and open questions are kept here. Where things are: [project map](project_map.md).

## Focus: SBP-SAT foundations

`src/sbp_sat/` is a separate NumPy prototype: periodic x, free top
and bottom boundaries, point receivers, exact discrete adjoint and gradient with
respect to log(Vp)/log(Vs) at fixed density. Connecting it to the production
SAFOD/DAS workflow is a later stage. Formulas: [method.md](sbp_sat/method.md),
checks and commands: [validation.md](sbp_sat/validation.md).

## Latest verified runs

| Run | Contents | Outcome |
| --- | --- | --- |
| [reference_v2](../results/validation/sbp_sat/reference_v2/validation_report.md) | results of the original QC v2 package (Python 3.12.14, NumPy 2.3.5): `oblique/`, `time_energy/`, `controls.json`, `sha256.json` | per the report, all 16 oblique cases and the fully discrete energy check passed; the control with the top SAT disabled fails, as it should |
| [sherlock_20261016](../results/validation/sbp_sat/sherlock_20261016/validation_report.md) | local Sherlock diagnostics: `oblique/`, `time_energy/` | `passed = true` in both `summary.json` |

- `sherlock_20261016/time_energy/summary.json` is byte-identical to
  `reference_v2/time_energy/summary.json`. Rerunning `time_energy_qc`
  during the 2026-10-16 restructure reproduced it exactly; a separate folder
  for that run was not kept in `results/`.
- The `oblique/summary.json` files of the two runs match in structure and
  `passed` flags; the numbers differ in the last digits (relative difference
  at most 4e-10).
- `reference_v2/controls.json` (top SAT disabled; bottom boundary
  isolation, depth 800 and 1000 m) is not produced by any project script:
  `oblique_qc` and `time_energy_qc` do not write it. There is no ready check for
  these controls: `oblique_qc` has no option to disable SAT, and the script does
  not itself compare depths 800 and 1000 m (there is only a `--depth` parameter).
- `operator_qc` 2026-10-16 passes in the `fwi` environment.

## Next steps

SBP-SAT already implements 2D forward, exact discrete adjoint and gradient.
The immediate task is to secure the foundations with small independent checks:

1. Check 1D derivatives on polynomials, positivity of norms, projections
   onto both boundaries and the SBP identity, including the minimum of 8 intervals.
2. For `rho * vx_t = sxz_z`, `sxz_t = mu * vx_z`, derive the energy flux and
   show that SAT cancels it. A wrong sign or disabled SAT must
   be detected by the check.
3. After that, split the responsibilities of `elastic2d.py`, comparing fields
   before/after and preserving the accuracy of the energy, dot-product and Taylor checks.

Absorbing boundaries, interpolation, DAS, acceleration and SAFOD are added
one at a time, with an adjoint operator check after each addition.

## Open questions

Zhang 2009 remains the first 3-D arrival model for observations; original
separate Lin 2010 Vp/Vs are now selected only after a Zhang coverage error.
Synthetic panels retain their matching 2-D model. Local original-Lin coverage
passed for all 291 published sources and 2991 fibre horizontal points.
Public-wrapper FMM passed at the wellhead/turnaround for a deep event,
the deepest/distant event, and a negative-depth event. For nc75379216,
200-to-100 m refinement differences were at most 1.785 ms P / 2.619 ms S.
The full-channel pilot and gallery regeneration with the new code still
require Sherlock. See [travel-time validation](safod/travel_times.md).

Ordinary compare_event runs now update the existing compare/dc<theta>/<model>/
PNGs and prediction CSV/JSON directly, without an eikonal_3d subdirectory.
The configured 0-20 Hz filter is supported as a 20 Hz low-pass; positive
bandpass output is bit-identical to the previous implementation. Local checks
passed: 49 tests and a full comparison/output-routing smoke run with
manufactured waveform packages and stubbed travel-time solves. Real-event
rerendering still requires the prepared Sherlock packages.

1. **Event 2026-10-03: v3 or v5.** For nc75443276 there are two versions of
   the DAS vs MH030 comparison: `plot_oct03_m08_das_mh030_pb_v3.py` (zero phase,
   `sosfiltfilt`; writes `compare/das_mh030_comparison_v3_zerophase`) and
   `plot_oct03_m08_das_mh030_pb_v5.py` (causal `DASutils.bandpass2D_c`;
   writes `compare/das_mh030_comparison_v5_causal`). Which version is final
   is the author's decision. The event is not in `settings.EVENTS`, and there is
   no `results/events/20261003_75443276/` folder in the project. Details:
   [experiments README](../scripts/safod/experiments/oct03_2026/README.md).
2. **Shared default initial model.** `run_forward` defaults to
   `--initial-model bill_logs`, `compare_event` to
   `hybrid_zhang2009_boness2006_bill_logs`. Until a choice is made, both scripts
   are run with the same explicit `--initial-model` and `--theta-deg`
   ([workflow.md](safod/workflow.md)).
3. **`scripts/legacy/safod_deep/plot_two_selected_events_pretty.py`** does not
   import: `scripts/safod_deep/plot_catalog_events.py` no longer has
   `infer_reference_locus_offset`, `load_reference_channel_limits` and
   `physical_reference_axis_and_crop`. Its output
   `results/safod_deep/event_qc_pretty_two/` is kept. Whether to restore the
   script or keep it as a record is the author's decision.
