# SBP-SAT in the SAFOD event pipeline

`--backend sbp_sat` connects the bounded solver to the existing model,
source, registered receivers, DAS, forward NPZ, native Vz snapshots/GIF
and `compare_event`. The default remains `numba_fused`, with its existing
paths and numerical loop. SAT equations and exact transpose:
[absorbing.md](absorbing.md).

## Spatial, source and time conventions

Grid2D nodes are the N-N material/normal-stress points, including both
physical endpoints. Vx is `(nx-1,nz)`, Vz `(nx,nz-1)`, Sxz `(nx-1,nz-1)`.
There is no material resampling or geometry shift. Both x0 and z0 are
respected. Sources/receivers must lie within common native-grid support;
no clipping or boundary extrapolation is introduced.

The SBP driver uses one explicit domain from `SBP_X_MIN_M`, `SBP_X_MAX_M`
and `SBP_Z_MAX_M` in `scripts/safod/settings.py`; the top is z=0. No cells
are appended. Defaults preserve the previous actual grid: x=-2100..4100 m,
z=0..5600 m, dx=dz=5 m, 1241x1121 material nodes. Legacy sponge constants
do not define this grid. Model interpolation/extension rules are unchanged.
The event driver restores a 600 m Gaussian sponge with rate 80 1/s inside
the chosen sides/bottom. Saved `n_boundary/gamma_s` are 120/80,
`sponge_width_*` are 600 m, and `material_padding_*` remain zero. Legacy `scientific_*` keys alias these same actual bounds;
they do not describe a second region. Registered coordinates are unchanged;
the driver still reports its retained channel mask and shallow cutoff.

The top is traction-free; first-order impedance SATs act at the actual
sides/bottom. A native-grid Gaussian sponge now attenuates waves before
they reach those outer faces. There is no PML. Its exact discrete reverse,
first-order time splitting in the strips and measured limitations are in
[sponge.md](sponge.md). The earlier PML proposal is deferred. Do not interpret a passing energy
or adjoint check as a measurement of low oblique reflection.

The source reuses the existing rotated 2-D tensor and `ricker_moment`
STF, `-dW/dt` at `(n+1/2)*dt`, with `t0=1.2/f0`. Bilinear positions are
not snapped. A nearest source uses its already embedded coordinates.
Positive-down z, tensor signs and centre-tangent finite-gauge DAS
strain-rate are preserved.

For bilinear sample vector b and native stress quadrature H:

$$
\delta_h=H^{-1}b,\qquad {\bf1}^TH\delta_h=1,\qquad
\Delta\sigma_{ij}=dt\,M_{ij}\,\mathrm{STF}[n]\,\delta_h.
$$

Normal components use `H_Nx tensor H_Nz`; shear uses `H_Mx tensor H_Mz`.
The source transpose uses exactly these coefficients, including dt.
There is no additional factor of two for the one stored Sxz component.
In the uniform interior H=dx*dz: this is precisely the legacy bilinear
`dt/(dx*dz)` convention. In closure cells, actual SBP quadrature preserves
the integrated source tensor. Both cases and all components are tested.
Current source support excludes points within half a cell of a face.

The inherited numerical 2-D tensor amplitude is preserved. The existing
source object labels the scale N m, but a physical line-source description
requires moment per out-of-plane length. This adapter does not choose that
length or calibrate a 3-D earthquake moment from magnitude. Absolute
real/synthetic amplitudes still need that separate physical choice.

Before building the STF, `configure_time_grid` selects
`dt=min(dt_legacy,0.8*dt_sufficient_limit)` and
`nt=ceil(requested_duration/dt)+1`. Without `--duration-s`, requested duration
is the initial `(nt_legacy-1)*dt_legacy`. Thus a smaller dt cannot silently
shorten the record. The duration check includes the known t0 delay and
the existing approximate straight-ray S/tail estimate. It is not a bound
on all heterogeneous travel times.

Velocity/DAS times are `(arange(nt)+0.5)*dt`; stress times are
`arange(nt+1)*dt`. Update order remains velocity, stress, stress source.
Sampling velocity after stress injection commutes with that injection.
The last source sample affects final stress, not an uncomputed velocity.

The adapter defaults to [compiled Numba execution](numba.md), with
`--sbp-execution csr` retaining the matrix reference. Both use the same
4-2 coefficients and dt certificate, with O(nx*nz) derivative work per step.
There is no GPU solver or change of discretization.

Forward retains receiver histories and requested snapshots, not all states.
`--dry-run` prints array-storage estimates and validates the actual model,
source/receiver support and CFL, then exits before propagation or writing a
forward package. The estimate excludes Python/plotting/external inputs and
is not measured peak RSS. Native Vz snapshots use x=grid.x and
z=grid.z[:-1]+dz/2; the saved axes and GIF use those coordinates directly.

The full computational arrays are saved at x=-2100..4100 m, z=0..5600 m,
dx=dz=5 m, 1241x1121 material nodes. Model Vp and wavefield GIF now hide the
actual absorbing strips, giving x=-1500..3500 m, z=0..5000 m at the default
600 m widths. No strip shading, reference rectangle or boundary-law caption
is drawn. Plotting does not resize the grid or rescale data coordinates.

Vz samples remain on their native half-depth nodes (1241x1120 in this case).
Pixel centres match those coordinates, and pixel edges/axes display the
physical 0/5600 m domain endpoints without interpolating boundary values.
The fixed GIF colour scale uses only pixels with positive overlap with the
displayed region across frames, excluding hidden absorbing rows. Saved
rerendering reads widths and rate from the NPZ. Historical impedance-only
packages have no active sponge and keep their full view; stale `scientific_*`
aliases and current settings do not override the saved absorber provenance.
Default grid/materials, source/STF, DAS observable, dt and checkpoint
equations are preserved. Explicitly changing the chosen domain can change
retained receiver support; that mask is reported, never coordinate-clipped.

To regenerate the existing model PNG/GIF from saved NPZ/snapshots, without
propagation, model rebuilding, external data or eikonal queries:

```bash
SAFOD_EVENT_KEY=nc75379261 "$FWI" -m scripts.safod.run_forward \
  --backend sbp_sat --theta-deg 35 \
  --initial-model hybrid_zhang2009_boness2006_bill_logs --render-only
```

The same `01_model_vp_with_source.png` and
`05_wavefield_vz_moment_tensor.gif` are updated. Input NPZ and snapshots
remain byte-identical. Rendering validates backend/model/angle, spatial axes,
native shape and velocity half-step timestamps before drawing. Missing
snapshots require a forward run with `--save-gif`; a receiver gather cannot
reconstruct the whole wavefield. The renderer prints its saved GIF frame count.

## Sherlock commands

Apply `sbp_sat_pipeline.patch` after the core and absorbing patches:

```bash
git apply --check sbp_sat_pipeline.patch &&
git apply sbp_sat_pipeline.patch
FWI=/home/groups/ettore88/alina/.envs/fwi/bin/python
GEO=/home/groups/ettore88/alina/.envs/geo/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$FWI" -m pytest -q &&
"$FWI" -m scripts.validation.sbp_sat.pipeline_qc
```

Inspect the actual prepared NC75379261 model, record and storage first:

```bash
SAFOD_EVENT_KEY=nc75379261 "$FWI" -m scripts.safod.run_forward \
  --backend sbp_sat --theta-deg 35 \
  --initial-model hybrid_zhang2009_boness2006_bill_logs \
  --duration-s 3 --no-gif --dry-run
```

Run the same request without `--dry-run`, then compare:

```bash
SAFOD_EVENT_KEY=nc75379261 "$FWI" -m scripts.safod.run_forward \
  --backend sbp_sat --theta-deg 35 \
  --initial-model hybrid_zhang2009_boness2006_bill_logs \
  --duration-s 3 --no-gif &&
SAFOD_EVENT_KEY=nc75379261 "$GEO" -m scripts.safod.compare_event \
  --backend sbp_sat --theta-deg 35 \
  --initial-model hybrid_zhang2009_boness2006_bill_logs
```

Use `--save-gif` for snapshots/GIF. As before, replacing an existing forward
package requires `--overwrite`. Run on a compute node.

## Output identity, predictions and gradient

One stable solver directory separates SBP from legacy products:

```text
results/events/20260618_75379261/forward/sbp_sat/dc035/<model>/
results/events/20260618_75379261/compare/sbp_sat/dc035/<model>/
```

Metadata records backend, native shapes, dt certificate, face types,
source quadrature and STF/amplitude conventions. The run tag is
`sbp_sat_4_2_pad120_dc035`. `pad120` is a historical file identifier retained
for saved impedance runs, not a parameter of the explicit SBP domain.
Compare checks backend/event/model/angle/tag;
missing backend metadata is accepted as legacy only.

Compare still computes one shared **3-D Zhang/Lin eikonal reference** from
the catalogue hypocentre and real physical receiver geometry, independently
of the 2-D solver. STF display correction remains `-source.stf.t0`, once.
The forward QC plots retain explicitly approximate 2-D straight-ray guides.

`src.sbp_sat.simulator.prepare_simulation` exposes the same solver,
distributed source and DAS observation for `run_adjoint_to_source`.
`objective_and_gradient_log_vp_vs` in that module accepts the existing
model/source/receiver objects and observed DAS. It differentiates
`J=0.5*dt*sum((predicted-observed)^2)` at fixed rho, dt, source and geometry,
including impedance derivatives. For sponge data pass the same explicit
`GaussianSponge` to forward and gradient preparation; widths/rate are fixed
numerical parameters. During optimization, keep dt fixed and
check its certificate for each model. Do not differentiate automatic dt/nt
selection or change observed-data length.

The adapter defaults to [exact checkpoint/replay](checkpointing.md); explicit
`checkpoint_interval=None` retains the O(nt*N) full-history reference.
History estimates exclude the other process allocations. The reverse/VJP
is still the independent matrix implementation, so event-gradient timing
is not the compiled forward timing.
Existing `src/fwi/` two-layer Numba experiments retain their original solver.
There is no existing SAFOD inversion driver replaced by this forward/QC work.

Local manufactured-medium QC: source quadrature error 1.62e-16,
sparse/dense output difference 1.57e-15, source-to-DAS transpose error
1.13e-14, Taylor slopes 2.00039/2.00019/2.00009. Automated tests also run the
actual SAFOD driver with synthetic external model inputs (preflight, NPZ,
native snapshots) and actual compare filtering/figures with a stubbed
external eikonal predictor. Real Sherlock data and the external 3-D solver
are not rerun locally. See [validation](validation.md).
