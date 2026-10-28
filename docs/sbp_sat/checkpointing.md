# Exact checkpoint/replay model gradient

The bounded SBP-SAT model gradient now supports a uniform in-memory
checkpoint schedule. The physical-source/DAS adapter defaults to it:

```python
from src.sbp_sat.simulator import objective_and_gradient_log_vp_vs

J, data, gradient_log_vp, gradient_log_vs = objective_and_gradient_log_vp_vs(
    model, source, receivers, observed_das, gauge_length_m,
    execution="numba", checkpoint_interval="auto",
)
```

An integer fixes the interval. `checkpoint_interval=None` retains the original
full-velocity-history reference. The low-level
`RectangularSBPSAT.objective_and_gradient_log_vp_vs` keeps `None` as its default
for existing reference/QC callers. Four return values and the objective
`J=0.5*dt*sum((data-observed)**2)` are unchanged. Density, grid, dt, source and
receiver geometry remain fixed; each perturbed model rebuilds impedances.

## State and indexing

Before step n the complete state is `(v^(n-1/2), sigma^n)`. Store independent
float64 copies of all five fields at n=0,K,2K,... strictly before nt. The
stress update and source injection remain in their original order. A restart
therefore has both the correct stresses and the previous half-step velocities.

For a reverse block [a,b), restore its state at a and recompute steps
a,...,b-1 **forward**, using STF samples with those exact indices. A local
buffer stores v^(a-1/2) and every resulting v_plus: b-a+1 velocity pairs.
The initial pair is necessary for the impedance derivative at the first
step of each block. Process steps b-1,...,a with the existing transpose/VJP;
carry the adjoint across blocks without resetting it. Last short blocks,
K=1, K=nt, and nt=1 are covered by tests.

For each original step the shared VJP performs the same bulk Lamé/shear
updates, the same discrete transpose, and
`-dt * a^T * delta(D) * ((v_minus+v_plus)/2)` for boundary impedances.
The final fixed-rho log-Vp/log-Vs chain rule is unchanged. This module changes
which forward velocities are retained; it adds no new gradient formula.
The independent finite-difference/Taylor checks still test that formula.

No damped wave equation is inverted backward. There is no time interpolation,
lossy compression, reduced precision, or stale checkpoint reuse between models.
Numba replay uses the same owned-state step and restores its previous thread
mask, including on failure. Checkpoints are consumed in reverse order.

## Memory and work

Let nx,nz denote normal-stress node counts. A velocity pair contains

`V = (nx-1)*nz + nx*(nz-1)` doubles;

a complete state contains

`S = V + 2*nx*nz + (nx-1)*(nz-1)` doubles.

With C=ceil(nt/K), retained history bytes are

`8 * (C*S + (K+1)*V)`.

The auto plan minimizes this expression over integer K=1,...,nt using scalar
arithmetic, without allocating event arrays or constructing a solver. The
local velocity buffers are reused; full-history lists are not constructed.
Original forward steps and replay each cost nt steps; there is one reverse
pass. Memory scales approximately as O(sqrt(nt)*nx*nz), rather than O(nt*nx*nz).
This is a simple uniform schedule, not disk checkpointing or Revolve.

For the 1241x1121, nt=19062 event, auto chooses K=227 and C=84:

| History allocation | GiB |
| --- | ---: |
| Original complete velocity history | 394.8182 |
| Complete checkpoints | 4.3503 |
| One block of velocities | 4.7224 |
| Checkpointed history total | 9.0727 |

These are history-array bytes, **not total process RSS**. Model/operator
arrays, live forward/adjoint states, data, residuals, gradient arrays,
temporaries and Python add memory. Do not request only 9 GiB for the full
event; a future event-gradient job should use a larger allocation (32 GiB
is a conservative starting request) and measure actual peak RSS.
`estimate_storage` now prints the checkpoint plan alongside the old full-history
estimate; forward propagation does not allocate either gradient history.

## Verification and Sherlock command

Apply `sbp_sat_checkpointing.patch` after the previous SBP patches:

```bash
git apply --check sbp_sat_checkpointing.patch &&
git apply sbp_sat_checkpointing.patch
FWI=/home/groups/ettore88/alina/.envs/fwi/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$FWI" -m pytest -q tests/test_sbp_sat_*.py &&
"$FWI" -m scripts.validation.sbp_sat.checkpoint_qc \
  --output-dir results/validation/sbp_sat/checkpoint_qc
```

The QC uses a small manufactured SI medium and physical-source/curved-DAS
chain. It compares objective, data and both gradients against full history,
then checks Taylor order and a central finite difference. Its example event
memory calculation allocates **no event wavefields**. Nonempty output folders
require explicit `--overwrite`.

Tests cover NumPy/CSR/Numba, all 16 face combinations, multiple block sizes,
short records, full-state copy counts, exact original/replay source indices,
generic native-point and physical DAS observations, parallel thread-mask
restoration and bulk/boundary-only gradient directions. On the authoring
environment, checkpoint/full-history outputs are bitwise equal. This is an
observed result for those checks, not a cross-platform bitwise guarantee.

Authoring verification (NumPy 2.0.1, Numba 0.65.1, Python 3.12): 178 SBP
tests pass, including 52 new checkpoint cases. Checkpoint QC gives exactly
zero parity differences, Taylor slopes 2.001135 / 2.000568 / 2.000284, and
central-difference relative error 1.12e-10. A separate comparison against the
pre-refactor matrix-gradient implementation is also bitwise equal. The
original source/DAS pipeline QC remains passed (dot error 8.09e-15).

## Remaining scope

This addition solves history storage. Reverse propagation and the VJP retain
the validated matrix implementation; they are not Numba-accelerated here.
Do not infer full-event gradient runtime from the 88 s compiled forward.
The actual event-scale checkpoint gradient and its peak RSS have not been
measured. Compile and verify reverse/VJP before making full-event FWI routine.
Real-data preprocessing, absolute 2-D/3-D source amplitude calibration and
an optimization driver are separate tasks. Ordinary forward/compare/GIF
behaviour is unchanged.
