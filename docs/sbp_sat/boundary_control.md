# Frozen-material SAFOD boundary control

The 2026-10-27 Sherlock run of NC75379261 completed with Numba on four CPU
threads: 7.68 s warmup, 88.3 s propagation, 3 min 19 s for the whole Slurm
job including outputs and comparison. The previous CSR propagation was
2435.1 s: the measured propagation speedup is 27.58x. These timings alone do
not isolate the contributions of fusion, compact operators and threading.

## Controlled change

`scripts.validation.sbp_sat.event_boundary_control` reads the existing
forward NPZ. It does not rebuild the geological model or change settings.
The complete saved Vp/Vs/rho arrays remain bitwise equal in their original
overlap. Extra side and bottom cells repeat the nearest outer-edge values
(`np.pad(..., mode="edge")`); no cells are added above the free surface.
This is an explicit numerical continuation, not additional geological data.

For the saved 1241x1121 event grid, 120 additional cells at dx=dz=5 m give:

| Quantity | Reference | Control |
| --- | --- | --- |
| Grid nodes | 1241 x 1121 | 1481 x 1241 |
| x bounds, m | -2100 .. 4100 | -2700 .. 4700 |
| z bounds, m | 0 .. 5600 | 0 .. 6200 |
| Total side/bottom material padding | 600 m | 1200 m |
| dt, s | 0.00015739136303307788 | identical |
| nt | 19062 | identical |

The physical source location, tensor, half-step Ricker moment-rate STF,
receiver centres/tangents, acquisition coordinate, gauge and time axes are
unchanged. Grid indices shift with x0; physical coordinates do not. The same
free-top / impedance-side-and-bottom SATs are applied at the new outer faces.
The enlarged-domain CFL certificate must accept the existing dt; no automatic
time adjustment is allowed in this experiment.

Before the extension run, the script reconstructs and replays the original
forward on its exact saved grid. It requires relative DAS L2 error <=1e-11
against the saved unfiltered data. Wrong source conventions, time axes,
channel selection, nonfinite values and replay disagreement stop the job.
This costs an additional reference forward, intentionally.

## Run on Sherlock

After applying `sbp_sat_boundary_control.patch` after the Numba patch:

```bash
FWI=/home/groups/ettore88/alina/.envs/fwi/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$FWI" -m pytest -q tests/test_sbp_sat_boundary_control.py &&
mkdir -p results/events/logs &&
SAFOD_EVENT_KEY=nc75379261 \
SAFOD_MODEL=hybrid_zhang2009_boness2006_bill_logs \
SAFOD_THETA_DEG=35 \
sbatch scripts/validation/sbp_sat/event_boundary_control.sbatch
```

The batch requests four CPUs and 16 GiB. Default output is one control folder:

```text
results/events/20260618_75379261/compare/sbp_sat/dc035/
  hybrid_zhang2009_boness2006_bill_logs/boundary_control_extra120/
```

It contains `extended_das.npz` (full-resolution DAS, extended materials and
grid, actual solver metadata, source STF and receiver geometry),
`01_boundary_control_signed.png`, and `summary.json` with the source-NPZ
SHA256, replay error, bounds, exact-overlap declaration, timings and windowed
differences. Existing forward and real/synthetic comparison outputs are
untouched. Nonempty control folders require explicit `--overwrite`.

The module also accepts `--reference-package`, `--output-dir` and
`--extra-padding-cells`. Source/model/geometry must satisfy its documented
SBP catalog-event, bilinear, physical Ricker moment-rate provenance checks.
It runs in `fwi` and does not call the 3-D eikonal backend.

## Read the diagnostic

Panels show reference, enlarged model, and enlarged minus reference. Both
complete records receive the existing 0-20 Hz zero-phase filter before the
0-2 s origin-relative crop; t0 is subtracted exactly once. All panels use the
same per-trace p99 scale from the reference. The difference is not normalized
independently. Display clipping is explicit; NPZ data remain unfiltered and
full resolution. Summary windows include both raw and filtered differences.
Use raw differences to assess onset: zero-phase filtering and detrending
can spread a later difference earlier in the displayed record.

Branches that move/change when artificial sides/bottom move indicate
sensitivity to truncation and the chosen exterior continuation. This is
not a formal decomposition of each branch into pure boundary reflection
and physical scattering. Unchanged branches are not automatically physical:
the common free top and all original material interfaces remain. No arbitrary
pass threshold or automatic physical classification is imposed on these
differences; the replay and invariance checks are the numerical gates.

The baseline plus 32% more nodes in the extension is roughly 2.32 reference
propagation workloads. At the supplied 88.3 s timing that is approximately
205 s for the two wave loops, excluding compilation, compressed I/O and
plotting. CPU type and memory effects can change this estimate. Gradient
history is not allocated.
