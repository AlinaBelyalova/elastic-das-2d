# Compiled bounded SBP-SAT execution

`--backend sbp_sat` now defaults to Numba; `--sbp-execution csr` keeps the
independent reference. Discretization, grid, dt certificate, tensor/STF,
source quadrature, registered geometry, DAS, output paths and shared 3-D
predictions are unchanged. Contracts: [pipeline](pipeline.md).

## Same operator, faster execution

The CSR reference repeatedly traverses full force/damping/velocity/strain
arrays and allocates temporaries. Even its boundary traces/lifts traverse
full-length vectors whose only nonzero support is three points per face.
The supplied Sherlock pilot also has 19,062 rather than 12,000 steps;
this adds 1.59x work separately from the execution cost per step.

Numba coefficients come from RectangularSBPSAT. Force rows include exactly
the reference free-traction SAT:

$$
\widehat D_{N\to M}=D_{N\to M}
 +H_M^{-1}p_L e_L^T-H_M^{-1}p_R e_R^T,
$$
$$
\widehat D_{M\to N}=D_{M\to N}
 +H_N^{-1}e_L p_L^T-H_N^{-1}e_R p_R^T.
$$

Every nonzero is retained in compact five-tap rows. Bulk rows use paired
four-tap differences after exact checks against both published operators.
All closure rows use the compact tables. Regrouping changes rounding, so
agreement with matrix products is checked numerically, not claimed bitwise.

The kernels implement the same velocity RHS, tangential damping diagonal,
cached inverse and two disjoint three-point rank corrections. They then
apply the same constitutive update and serial accumulating source scatter.
Receiver sampling is compiled; DAS uses the original finite-gauge operator.
Fields stay float64 and kernels explicitly set `fastmath=False`.
Parallel rows/face lines have barriers between dependent updates.

Production reuses two velocity buffers and updates owned stresses in place.
Public `step` remains nonmutating, external initial states are protected,
and histories/snapshots are copied when requested. The independent CSR
transpose and material/impedance VJP are retained. Compiled forward is
checked against them by source dot products, Vp/Vs Taylor tests and a
boundary-only finite-difference direction. Compiled reverse kernels and
gradient checkpointing are not added by this performance change.

## Threads and timing

Small grids use one thread; event grids default to at most four, bounded
by the Slurm allocation, CPU affinity and Numba runtime limit. Library calls
restore the previous thread mask. More threads do not necessarily help a
memory-intensive calculation; the benchmark permits explicit tuning.

Warmup compiles on zero fields with zero source amplitude, so no physical
step/sample is lost. Compilation is timed separately. NPZ records
`solver_execution`, `numba_threads`, `kernel_warmup_s`,
`propagation_elapsed_s`. Numba manages its own compilation cache.

## Running after the pipeline patch

```bash
git apply --check sbp_sat_numba.patch && git apply sbp_sat_numba.patch
FWI=/home/groups/ettore88/alina/.envs/fwi/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$FWI" -m pytest -q tests/test_sbp_sat_*.py &&
"$FWI" -m scripts.validation.sbp_sat.pipeline_qc
```

Existing forward commands now select Numba automatically. For an explicit
four-CPU, 16 GiB allocation and the matching comparison, submit from root:

```bash
mkdir -p results/events/logs
sbatch scripts/safod/run_forward_sbp_sat.sbatch --overwrite
```

Defaults: NC75379261, theta=35, existing hybrid model, three seconds, with
Vz snapshots/GIF as in the regular forward driver. Pass `--no-gif` explicitly
only when snapshots are not wanted.
`--overwrite` intentionally regenerates an existing forward NPZ. Event,
model and angle can be set with `SAFOD_EVENT_KEY`, `SAFOD_MODEL`,
`SAFOD_THETA_DEG`. A running Python process keeps its imported code;
applying a patch affects the next run. The separate website test isolation
issue is not changed here.

## Verification and benchmark scope

26 new cases cover all 16 face combinations, minimal grids, nonmutating
steps, workspace/history aliasing, 500-step bulk energy balance, source
transpose, boundary sensitivity and thread determinism. The full local suite
passes 196 tests with NumPy 2.0.1/Numba 0.65.1 on Python 3.12. This is not a
Sherlock run. The complete source/DAS Taylor QC now uses compiled forward.

The benchmark uses a manufactured heterogeneous SI medium, not the actual
SAFOD velocity inputs. It reports compilation separately, repeat medians,
field/receiver parity and CSR component timings. Component calls have
different allocation/cache conditions and must not be read as additive
percentages of a whole step. Example on a compute allocation:

```bash
python -m scripts.benchmarks.sbp_sat --nx 1241 --nz 1121 \
  --steps 100 --threads 1 4 --repeats 3 \
  --output-dir results/benchmarks/sbp_sat/<label>_<YYYYMMDD>
```

Projected 19,062-step time covers only this manufactured wave loop,
excluding model construction, I/O, plots and external eikonal calculation.
Local timing cannot guarantee Sherlock runtime. Parity is gated below
1e-11; there is no hardware-dependent timing pass threshold. Nonempty
output directories require explicit `--overwrite`.

Local full-size medians (three 100-step repeats, NumPy 2.0.1/Numba 0.65.1,
Python 3.12): CSR 25.871 s; Numba one thread 2.183 s (11.85x), four threads
1.410 s (18.34x), eight threads 1.690 s (15.31x). Largest field-relative
difference was 9.66e-16 and receiver difference 5.86e-16. Four threads are
the conservative default; these observations do not promise a particular
Sherlock runtime or a monotonic speedup with thread count.

The subsequent user-supplied Sherlock event run completed: four threads,
7.68 s warmup, 88.3 s propagation versus 2435.1 s CSR (27.58x), 3 min 19 s
whole batch, including comparison. This is an actual event measurement;
it does not separately attribute speedup to each optimization. The next
waveform diagnostic is the [frozen-material boundary control](boundary_control.md).
