# Project status

Date: 2026-10-28. Only the current stage, the latest verified runs,
next steps and open questions are kept here. Where things are: [project map](project_map.md).

## Focus: opt-in SBP-SAT backend in the SAFOD forward pipeline

`src/sbp_sat/` is a separate NumPy prototype: periodic x, free top
and bottom boundaries, point or off-grid finite-gauge DAS receivers,
exact discrete adjoint and gradient with
respect to log(Vp)/log(Vs) at fixed density. This periodic reference is retained;
the bounded backend below is connected to the SAFOD forward/DAS workflow.
Formulas: [method.md](sbp_sat/method.md),
checks and commands: [validation.md](sbp_sat/validation.md).

The foundation/DAS integration now has isolated SAT and transpose functions,
a conservative timestep certificate, optional shear stress-rate source and
precomputed observation weights. The full source-to-DAS transpose and
rho-fixed model gradient are checked. The existing centre-tangent DAS
convention is preserved, including its curved-gauge approximation.
Details and derivation: [integration.md](sbp_sat/integration.md).

An additive bounded-domain reference now uses SBP pairs in both coordinates,
a free top and local impedance SATs on the sides/bottom. Its trapezoidal
velocity-boundary update has an exact modified-energy dissipation identity.
The source-to-DAS transpose and model gradient include the implicit boundary
solve and model-dependent impedances. It is first-order impedance absorption,
not PML. Derivation and limits: [absorbing.md](sbp_sat/absorbing.md).

`run_forward` and `compare_event` now accept `--backend sbp_sat`, with one
stable solver directory and strict saved-backend checks. The adapter preserves
the existing source tensor/STF and exact registered receivers, uses native
SBP quadrature for off-grid stress-source normalization, and connects DAS,
NPZ and native Vz snapshots/GIF. Numba is now the default SBP execution;
CSR is retained as an independent reference with the same coefficients.
A certified dt is selected before the STF, preserving requested duration.
The same prepared source/DAS chain is available to source-adjoint and model
gradient calls. Runbook and scope: [pipeline.md](sbp_sat/pipeline.md).
Real-event SBP propagation and comparison have completed on Sherlock; the
inputs remain external to this repository.

The supplied Sherlock preflight passed on 1241x1121 nodes, dx=dz=5 m,
dt=0.000157391363 s, nt=19062, 1196 valid DAS gauges and zero receiver
coordinate mismatch. The initial CSR forward took 126.8 s for 1000 steps.
That motivated the [compiled execution patch](sbp_sat/numba.md): fused
float64 kernels, compact closures, owned workspace and native sampling,
with no timestep or physical-convention change. User-supplied job 49903580
completed with exit code 0: four threads, 7.68 s warmup, 88.3 s propagation
versus 2435.1 s CSR (27.58x), 3 min 19 s whole batch including compare.
The supplied image has additional crossing branches whose origin remains
unresolved. A [frozen-material boundary control](sbp_sat/boundary_control.md)
replays the exact saved reference and adds 600 m at the sides/bottom without
changing the existing material volume, dt, source or DAS. The script is
locally checked. The subsequent user-supplied control image shows that main
crossing branches persist while differences concentrate in late time,
approximately 1.6-2 s in the plotted scale. This is a qualitative image
assessment; the full numeric control report has not been supplied.

The bounded model-gradient adapter now defaults to exact uniform
[checkpoint/replay](sbp_sat/checkpointing.md), retaining the original
full-history option and shared reverse/VJP algebra. On the 1241x1121,
nt=19062 event, history allocation is estimated at 9.073 GiB instead of
394.818 GiB (84 complete checkpoints, interval 227). These exclude other
process allocations. Small-model parity and Taylor checks pass locally;
the actual event-size gradient and peak RSS have not been measured.
The reverse/VJP still uses the independent matrix implementation.

The subsequent Sherlock checkpoint report also passes all 178 SBP tests,
with bitwise full-history parity and the same Taylor/finite-difference values.
Wavefield/model rendering now shows the complete computational grid instead
of the inner-window crop. `run_forward --render-only` can regenerate existing
PNG/GIF from saved arrays with no propagation; it validates native axes/times
and prints the written frame count. For the default event the displayed and
computed bounds are x=-2100..4100 m, z=0..5600 m. Native Vz staggering is
preserved. Numerical propagation and boundary absorption are unchanged.

The subsequent boundary audit makes that grid explicit in SBP settings and
removes the inner reference overlay. No extra cells are appended. The default
actual domain stays x=-2100..4100 m, z=0..5600 m; legacy scientific-bound
NPZ fields now alias the same actual domain. This is a layout/provenance fix,
not an improved absorbing boundary.

The current local impedance law was independently checked with continuum
P/SV plane waves, including postcritical mode conversion and energy flux.
For the illustrative Vp=5000, Vs=3000 m/s medium it reflects 19.6% of P energy
and 35.4% of SV energy at 75 degrees from the normal. This is boundary-law
error before discretization, not a measured SAFOD reflection coefficient.
The previous normal-P packet test did not cover this limitation.
Fresh manufactured-model checks reproduce the energy identity (relative
balance error 2.19e-16), exact checkpoint parity and quadratic Taylor rates.
These validate the current operator, not boundary transparency. The next
gated work is an SBP-compatible CFS-PML on sides/bottom, then discrete
reflection tests and its complete auxiliary-state adjoint/checkpoints.
See [boundary_plan.md](sbp_sat/boundary_plan.md). No PML has been enabled.

## Latest verified runs

| Run | Contents | Outcome |
| --- | --- | --- |
| `checkpoint_20261027` | authoring environment: 52 new checkpoint cases, all SBP tests, full-history parity, physical-source DAS/boundary Taylor and original pipeline QC | 178 SBP tests pass; checkpoint/full-history objective, data and gradients bitwise equal; Taylor slopes 2.001135/2.000568/2.000284; central difference relative error 1.12e-10; event history estimate 9.073 GiB without allocating event fields |
| `numba065_median_20261027` | manufactured model at the full 1241x1121 pilot dimensions, 1200 receivers, three 100-step repeats; NumPy 2.0.1/Numba 0.65.1, Python 3.12 | local full suite 196 pass; CSR median 25.871 s, Numba 1/4/8 threads 2.183/1.410/1.690 s; 11.85x/18.34x/15.31x speedup; largest field difference 9.66e-16. Timings are not Sherlock event measurements |
| `pipeline_20261027` | authoring environment: source/STF adapter, native CSR forward/DAS, complete source transpose, model gradient, SAFOD driver preflight/save/snapshots and comparison smoke with synthetic external inputs | 170 Python and 13 JS tests pass; quadrature source error 1.62e-16; dense/CSR difference 1.57e-15; source/DAS transpose 1.13e-14; Taylor slopes 2.00039--2.00009 |
| Sherlock terminal report 2026-10-27, absorbing patch | user-supplied bounded-domain QC in fwi/Python 3.11 | 154 tests pass; energy-balance error 2.13e-16; transpose 3.57e-15; Taylor slopes 2.00208--2.00052; normal-P packet residual decreases from 1.92e-7 to 6.06e-10 |
| `absorbing_20261027` | authoring environment: 28 new bounded-domain tests, full Python suite and absorbing/DAS QC | 154 Python tests pass; dissipation-balance error 2.19e-16; source/DAS transpose error 2.53e-15; Taylor slopes 2.0021--2.0005; normal-P residual energy 1.92e-7 at h=1 and 6.06e-10 at h=0.5 |
| Sherlock terminal report 2026-10-27 | user-supplied run of the core patch in fwi/Python 3.11, without saved QC output | 126 tests pass; 24 pre-existing pyproj warnings; DAS transpose error 1.39e-16; Taylor slopes 1.9986--1.9996 |
| `core_20261026`, `das_chain_20261026` | authoring environment: 43 new mathematical tests; the full Python suite; original operator, source transpose, Taylor, normal/oblique reflection and energy checks | 126 Python tests pass; all original QC pass; DAS dot-product error 1.67e-15, Taylor slopes 1.9986--1.9996; leapfrog energy drift 2.22e-16 |
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

Polynomial/norm/trace checks, both 1-D flux cancellations and wrong-sign or
disabled-SAT controls are now automated. SAT extraction preserves the original
controlled solution and gradient to rounding error. Receiver interpolation
and DAS are connected with a full transpose and Taylor test.

Bounded x/z and dissipative sides/bottom now have full transpose and model
derivatives, including impedance derivatives. The normal-P absorption check
is validated; practical oblique P/S reflection performance still needs its
own benchmark because local impedance is not exact at oblique incidence.
Source normalization/STF, off-grid source support, compiled execution and opt-in
SAFOD forward/comparison are now connected and verified on manufactured
inputs. The actual Sherlock event and compiled timing are now supplied;
the boundary-control image is supplied and checkpoint/replay is implemented.
Next compile and verify reverse/VJP before full-event gradient timing and
a minimal FWI driver. Boundary receiver traces where needed remain follow-up work.
The periodic-x/free-bottom model is not used as a production open domain.

## Reproducible horizontal datum transformation

The default SAFOD catalog now fixes inverse EPSG:1750 + inverse EPSG:1241 +
UTM 10N with two SHA256-verified, vendored public-domain NOAA grids. Automatic
PROJ selection previously reproduced old events with a 7 m Helmert operation
in one environment and a 2.15 m grid operation in another. The catalog builder
writes `catalog_horizontal_transform.json` and includes the provenance in its
notes. Missing/modified grids or sources outside the operation area fail
explicitly. The fixed route reproduces the current Sherlock catalog; the
workbook, depths, profile, channel registration and 3-D arrivals are unchanged.
See [geodesy lock](safod/geodesy_lock.md) for numerical controls and limits.

## SAFOD geometry correction

The catalog now projects sources to NAD27 / UTM 10N (EPSG:26710), matching
stored workbook coordinates. The wellhead and full down-leg profile are
established before the distance-only shallow cut; the builder uses the same
canonical WGS84 profile origin as prepare_event. NCEDC query defaults now
use that wellhead as well. Pure reference constants are shared without
importing event-specific settings. Catalog tools, local gallery and site
builder reject PNG metadata with stale catalog labels and request an in-place
rerender. The original workbook, MD/TVD, registration and 3-D P/S backend are
unchanged. Rebuild commands and validation: [geometry correction](safod/geometry_correction.md).

## Open questions

Zhang 2009 remains the first 3-D arrival model for observations; original
separate Lin 2010 Vp/Vs are now selected only after a Zhang coverage error.
Real and synthetic comparison panels now share one 3-D P/S reference computed
from the catalogue source and real receiver geometry through EikoHypoTomo.
Synthetic waveforms retain their saved 2-D elastic model. Local original-Lin
coverage passed for all 291 published sources and 2991 fibre horizontal points.
Public-wrapper FMM passed at the wellhead/turnaround for a deep event,
the deepest/distant event, and a negative-depth event. For nc75379216,
200-to-100 m refinement differences were at most 1.785 ms P / 2.619 ms S.
The full-channel pilot and gallery regeneration with the new code still
require Sherlock. See [travel-time validation](safod/travel_times.md).

Ordinary compare_event runs now update the existing compare/dc<theta>/<model>/
PNGs and prediction CSV/JSON directly, without an eikonal_3d subdirectory.
The configured 0-20 Hz filter is supported as a 20 Hz low-pass; positive
bandpass output is bit-identical to the previous implementation. Local checks
passed: 55 tests and full comparison/output-routing smoke runs for Zhang and
Lin metadata with manufactured waveform packages and stubbed 3-D predictions.
All five overlays per comparison used exactly the same channel/time arrays;
summary metadata and arrivals-only output agreed. Real-event rerendering still
requires the prepared Sherlock packages.

Comparison axes are now fixed to the common physical-channel range and
0-to-common-end catalogue time (at most 2 s). Shared guides no longer expand
the synthetic viewport into blank margins. Regression checks cover signed
and envelope plots, unequal channel/time coverage, identical P/S curves,
unchanged image extents, and the real-only pre-origin window.

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
