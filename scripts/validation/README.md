# Solver checks

In short: one table of all checks from `scripts/validation/{sbp_sat,elastic,fwi}`,
the example in `scripts/examples` and the benchmarks in `scripts/benchmarks`:
what each checks, how long it runs, when it counts as passed and where it
writes.

## How to run

- From the project root, in the `fwi` environment ([docs/setup.md](../../docs/setup.md)):
  `python -m <module from the table>`.
- SBP-SAT commands with parameters (`--output-dir`, `--plot`, `--overwrite`)
  are given only in [docs/sbp_sat/validation.md](../../docs/sbp_sat/validation.md).
- On Sherlock, everything except short checks runs in a Slurm job on a
  compute node, not on a login node.
- Methods: [docs/elastic/method.md](../../docs/elastic/method.md) (staggered-grid
  solver, DAS, FWI), [docs/sbp_sat/method.md](../../docs/sbp_sat/method.md)
  (SBP-SAT). Layout and names of result folders:
  [results/README.md](../../results/README.md).
- Scripts with a fixed result folder create it themselves and overwrite files
  in it without warning. Only `sbp_sat.time_energy_qc` and `sbp_sat.oblique_qc`
  have protection against writing into a non-empty folder (`--overwrite`).

Times are measurements from 2026-10-16 on a Sherlock compute node: `fwi`
environment, `OPENBLAS_NUM_THREADS=1`, run without `--output-dir`. "Not
measured" means the check was not run when the table was compiled.

## Table

| Check (`python -m ...`) | What it checks | Time | Pass criterion | Writes to |
| --- | --- | --- | --- | --- |
| `scripts.validation.sbp_sat.operator_qc` | QC 1: SBP identity for the staggered operators and semi-discrete energy rate with SAT | ~0.2–0.3 s | max SBP residual ≤ 1e-12 and relative dE/dt residual ≤ 1e-12; prints `PASSED` | stdout only |
| `scripts.validation.sbp_sat.adjoint_dot_product` | QC 3: exact discrete adjoint operator source → data, `<F s, q> = <s, F^T q>` | ~0.2–0.3 s | relative error ≤ 1e-11 | stdout only |
| `scripts.validation.sbp_sat.taylor` | QC 4: Taylor test for the gradient with respect to `log(Vp)`, `log(Vs)` at fixed `rho` | ~0.9 s | R0 slope in (0.8, 1.2), R1 slope in (1.7, 2.3), relative error of the directional derivative against a central difference ≤ 1e-5 | stdout only |
| `scripts.validation.sbp_sat.normal_incidence_qc` | QC 2: reflection of P and SV Gaussian packets at normal incidence on the free surface; surface velocity equals twice the incident; h = 10, 5, 2.5 m | > 30 s (stopped by `timeout 30`; full time not measured) | relative L2 error strictly decreases with refinement for P and SV; peak and traction are printed without a threshold | stdout only |
| `scripts.validation.sbp_sat.time_energy_qc` | fully discrete leapfrog energy: heterogeneous 32×41 model, no source and no attenuation, 0.5 s | ~1.5 s | energies finite and positive, invariant drift < 1e-9, change of centered energy < 5% | stdout only; with `--output-dir`, `summary.json`, `traces.npz`; with `--plot`, PNG |
| `scripts.validation.sbp_sat.oblique_qc` | oblique incidence of P (30°, 60°) and SV (20°, 30°): full-field evolution against the continuum solution; h = 10, 5, 2.5, 1.25 m | > 30 s (stopped by `timeout 30`; full time not measured) | waveform and traction errors decrease at every refinement step; on the finest grid five metrics ≤ 0.01 (details in [validation.md](../../docs/sbp_sat/validation.md)) | stdout only; with `--output-dir`, `summary.json`, `traces.npz`; with `--plot`, PNG |
| `scripts.validation.elastic.boundary_absorption` | absorption by a Gaussian sponge on the side boundaries: homogeneous medium, isotropic source, various sponge widths and `gamma_s` with a `gamma_s=0` control on the same grid | not measured | no single threshold: error only on NaN/Inf or an inconsistent set of configurations; `WARNING` if the direct-wave energy differs from the control of the same width by more than 2%; prints the best configurations and a cost-aware recommendation | `results/validation/elastic/boundary_absorption/` |
| `scripts.validation.elastic.free_surface_graves_fs1_qc` | Graves FS1 algebra and step order: boundary equations for velocities after continuation; filling ghost nodes before the velocity update (old order) violates them; refilling after the update restores them; stress mirroring is exact | ~2 s with a warm Numba cache; the first run with JIT compilation takes longer (~4 s) | all exact residuals ≤ 1e-11; the old-order residual must be > 1e-9 | stdout only |
| `scripts.validation.elastic.free_surface_normal_incidence_qc` | reflection of plane P and SV packets at normal incidence on the production grid with Graves FS1 Numba kernels; surface = 2 × incident wave; h = 10, 5, 2.5 m at a fixed physical domain | not measured | no automatic criterion: prints rel L2, peak_ratio, peak_dt, traction and observed orders; expected trends are printed at the end | `results/validation/elastic/free_surface_normal_incidence/` |
| `scripts.validation.elastic.magnitude_scaling` | amplitude linearity in moment: Mw = 0, 1, 2, 3, slope of `log10` of the peak envelope against Mw for `vx`, `vz`, DAS; `numba_fused`, `free_surface=True` | not measured | deviation of all three slopes from 1.5 ≤ 1e-3 | `results/validation/elastic/magnitude_scaling/` |
| `scripts.validation.elastic.moment_tensor_analytical` | moment-tensor solver against an analytical line source in a homogeneous whole space: P/S kinematics, radiation pattern and polarity, which quantity (U, V, A) the velocity corresponds to, overall amplitude factor, convergence of the y integration | not measured | best candidate is `velocity`, correlation ≥ 0.999, factor deviation from 1 ≤ 3%, relative error ≤ 0.02, shift ≤ 1 sample, etc.; without `--strict` violations are printed as `QC warnings` without an error, with `--strict` they raise an error | `results/validation/elastic/moment_tensor_analytical/` (`--out-dir`) |
| `scripts.validation.elastic.moment_tensor_linearity` | the response to a linear combination of `Mxx`, `Mzz`, `Mxz` equals the combination of responses (`vx`, `vz`, DAS), with `free_surface` False and True | not measured | worst relative L2 error ≤ 1e-8 | `results/validation/elastic/moment_tensor_linearity/` |
| `scripts.validation.elastic.pointforce_2d` | NumPy point-force solver against the analytical 2-D `v_z` for a vertical force; orders 2, 4, 6, 8 | not measured | no automatic threshold: prints rel L2 and max abs per order and error reduction factors; error only on invalid parameters or mismatched time axes | `results/validation/elastic/pointforce_2d/` |
| `scripts.validation.elastic.reciprocity` | reciprocity `G_zz(x_A, x_B) = G_zz(x_B, x_A)` for a vertical point force (`solver_numpy_pointforce`) | not measured | worst relative L2 error ≤ 1e-8 | `results/validation/elastic/reciprocity/` |
| `scripts.validation.elastic.source_radiation` | radiation patterns of 2-D sources in a homogeneous whole space along the full path: tensor → bilinear spreading → injection → propagation → sampling; isotropic and double couple at `theta` = 0, 22.5, 45° | not measured | double couple: P and SV pattern similarity ≥ 0.70; isotropic: P pattern rel L2 ≤ 0.35 and SV/P energy ≤ 0.10 | `results/validation/elastic/source_radiation/` |
| `scripts.validation.fwi.two_layer_numba` | controlled two-layer model: Numba forward problem for the true and smooth initial models (`vx`, `vz`, DAS), prints the misfit | not measured | no NaN/Inf; with `--benchmark-warm` the repeated run matches with relative error ≤ 1e-12; prints `PASSED` | `results/validation/fwi/two_layer_numba/` (`--out-dir`) |
| `scripts.validation.fwi.two_layer_adjoint_dot_product` | exact discrete adjoint operator on the two-layer model: A) source → `vx`/`vz`; B) source → receivers → DAS with finite gauge length; `free_surface=False` | not measured | maximum relative dot-product error < `--tolerance` (default 1e-8) | stdout only |
| `scripts.validation.fwi.two_layer_taylor` | directional derivative and Taylor test for the DAS gradient with checkpointing with respect to `log(Vp)`, `log(Vs)` | not measured | only with `--strict`: best relative derivative error < 1e-4, R0 slope in [0.85, 1.15], R1 in [1.75, 2.25]; without `--strict` violations do not raise an error, the script prints `completed` | `results/validation/fwi/two_layer_taylor/` (`--out-dir`) |
| `scripts.examples.homogeneous_demo` | demonstration of the full chain: homogeneous model, off-grid double couple with bilinear spreading, `numba_fused`, straight DAS cable, strain rate, seismograms, rough comparison of arrival times, animation; second check: explosive source with NumPy; `free_surface=True` | not measured | no criterion: error only on NaN/Inf; prints the arrival-time comparison | `results/examples/homogeneous_demo/` |
| `scripts.benchmarks.backends` | NumPy vs Numba fused on a 601×601 grid, nt = 800, order 4, with `free_surface` False and True: agreement and time (best of 3) | not measured | no threshold: prints max abs, rel L2 and `allclose(atol=1e-7, rtol=1e-5)` against NumPy, time and speedup | stdout only |
| `scripts.benchmarks.performance` | scaling of NumPy and Numba fused with `free_surface=True` on 201², 401², 601² grids (nt = 400, 600, 800): time, speedup, throughput | not measured | no threshold | `results/benchmarks/performance/` |

Notes on the criteria: `PASSED` and `FAILED` are strings printed by the
script; "error" means an exception (`RuntimeError` etc.) and a non-zero exit
code.

## SAFOD travel times

`scripts.validation.safod.travel_times_3d` checks the installed EikoHypoTomo
wrapper against homogeneous analytical P/S times. Commands, criteria and
real-model follow-up: [docs/safod/travel_times.md](../../docs/safod/travel_times.md).

## Self-tests of `src` modules

The modules `src.elastic.source`, `src.elastic.source_spreading`,
`src.elastic.source_injection`, `src.elastic.sampling`, `src.das.receivers`,
`src.das.strain`, `src.analytical.moment_tensor` and `src.fwi.misfit` contain
`_self_test()`, which runs on `python -m <module>`. They write no files; time
not measured. Fast regression tests (`python -m pytest`) are described in
[docs/setup.md](../../docs/setup.md).
