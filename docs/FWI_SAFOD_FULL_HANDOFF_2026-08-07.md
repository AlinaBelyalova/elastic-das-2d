# SAFOD Elastic-DAS FWI Project — Full Handoff / Continuation Context

**Project:** Elastic 2-D DAS forward modelling and FWI for SAFOD  
**Repository root:**  
`/home/groups/ettore88/alina/classes/geophys245/Elastic_das_project`

**Primary environment:**  
`/home/groups/ettore88/alina/.envs/fwi`

**Current date/state of handoff:** 2026-08-07

---

# 1. Purpose of this document

This file is a **complete continuation/handoff document** for the SAFOD
elastic-DAS project.

It is intended to be pasted or uploaded into a new ChatGPT conversation so
work can continue without reconstructing the project history.

The next chat should treat this file as the authoritative project context and
should **not reopen already validated numerical/unit/geometry issues unless a
new failure specifically points to them**.

The immediate scientific goal is to build, compare, and eventually invert from
a controlled set of SAFOD initial velocity models:

```text
1. smooth_prior
2. bill_logs
3. zhang2009
4. hybrid_zhang2009_bill_logs
```

The project is currently between stages 2 and 3:

- the original smooth prior works;
- Bill Ellsworth's borehole-log model has been digitized and forward modelled;
- the Zhang, Thurber & Bedrosian (2009) 3-D tomography supplied by Clifford
  Thurber has now been parsed and horizontally registered;
- the next technical task is to verify the Zhang vertical datum and extract
  the exact SAFOD 2-D section from the 3-D tomography.

---

# 2. Scientific objective

The project aims to model local earthquakes recorded on the SAFOD borehole
DAS array with a 2-D elastic finite-difference solver and then move toward
elastic FWI.

The working strategy is:

```text
real SAFOD DAS
      +
accurate borehole geometry
      +
local earthquake source
      +
controlled initial velocity models
      ↓
elastic forward modelling
      ↓
real/synthetic comparison
      ↓
initial-model selection / hybridization
      ↓
elastic DAS FWI
```

The central current scientific question is whether synthetic phases and
travel times improve when the initial model includes more realistic SAFOD
fault-zone velocity structure.

Bill Ellsworth specifically noted that the previous smooth model appeared too
fast in the damage-zone core and that the real borehole logs contain a much
sharper velocity jump than the smooth gradient used in the original model.

---

# 3. Canonical repository structure

The repository was cleaned and reorganized in August 2026.

The intended active structure is:

```text
Elastic_das_project/
│
├── src/
│   ├── model.py
│   ├── plotting.py
│   ├── simulator.py
│   ├── ...
│   │
│   ├── safod/
│   │   ├── __init__.py
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── factory.py
│   │       ├── smooth_prior.py
│   │       └── digitized_log.py
│   │           # planned rename -> bill_logs.py
│   │
│   └── safod_deep/
│       └── catalog.py
│
├── scripts/
│   └── safod/
│       ├── settings.py
│       ├── prepare_event.py
│       ├── run_forward.py
│       ├── compare_event.py
│       │
│       └── models/
│           ├── __init__.py
│           ├── digitize_ellsworth_malin_fig3a.py
│           └── zhang2009/
│               ├── parse_model.py
│               └── qc_registration.py
│
├── data/
│   └── safod/
│       └── velocity_models/
│           ├── ellsworth_malin_2011/
│           │   ├── README.md
│           │   └── fig3a_digitized.csv
│           │
│           └── zhang_thurber_bedrosian_2009/
│               ├── README.md
│               ├── raw/
│               │   ├── MOD.head
│               │   ├── inversion_grid.dat
│               │   ├── Vp_model.dat
│               │   ├── Vs_model.dat
│               │   └── Vpvs_model.dat
│               ├── processed/
│               │   ├── zhang2009_native_grid.npz
│               │   └── zhang2009_native_nodes.csv
│               └── qc/
│                   ├── zhang2009_grid_summary.txt
│                   ├── zhang2009_xy_grid.png
│                   ├── zhang2009_vp_slices.png
│                   ├── zhang2009_vs_slices.png
│                   ├── zhang2009_vpvs_slices.png
│                   ├── zhang2009_registration_summary.txt
│                   └── zhang2009_local_geographic_grid.png
│
├── config/
│   └── safod_deep/
│       └── roots.json
│
├── results/
│   ├── events/
│   │   ├── 20260401_75336802/
│   │   │   ├── real/
│   │   │   ├── forward/
│   │   │   └── compare/
│   │   │
│   │   └── 20260618_75379261/
│   │       ├── real/
│   │       ├── forward/
│   │       └── compare/
│   │
│   └── _migration_manifests/
│
├── archive/
│   ├── legacy_models/
│   │   └── safod_log_constrained_model.py
│   └── legacy_scripts/
│       └── safod/
│           └── build_initial_model.py
│
└── refactor_backup/
    # temporary rollback snapshots from repository migration
```

## Important repository rule

Real model implementation now belongs under:

```text
src/safod/models/
```

Scientific model inputs belong under:

```text
data/safod/velocity_models/
```

Model-building/digitization/QC utilities belong under:

```text
scripts/safod/models/
```

Retired code belongs under:

```text
archive/
```

---

# 4. Repository cleanup already completed

The following refactor was performed:

```text
src/safod_builder.py
src/safod_digitized_log_model.py
```

were converted into temporary compatibility shims after moving real model code
to:

```text
src/safod/models/smooth_prior.py
src/safod/models/digitized_log.py
src/safod/models/factory.py
```

The shims were only 17 and 13 lines respectively after refactoring.

An audit showed old imports in:

```text
src/plotting.py
scripts/safod/build_initial_model.py
src/safod_log_constrained_model.py
```

The old hand-tuned model:

```text
src/safod_log_constrained_model.py
```

was moved to:

```text
archive/legacy_models/safod_log_constrained_model.py
```

The old standalone build script was intended to be archived under:

```text
archive/legacy_scripts/safod/build_initial_model.py
```

The final cleanup plan was:

- `src/plotting.py` should import `SafodBuildMetadata` from the new model API;
- old build script should stay archived;
- old compatibility shims should be removed once no active imports remain;
- empty `configs/events/` should be removed;
- SAFOD-deep roots configuration should live at:

```text
config/safod_deep/roots.json
```

The user stated that the cleanup/order was completed before starting the
current velocity-model work.

Do **not** reintroduce `src/safod_builder.py` as the main implementation.

---

# 5. Canonical results layout

The results tree was migrated from separate top-level names such as:

```text
results/real_event_20260618_75379261/
results/forward_real_event_20260618_75379261/
results/compare_real_synthetic_20260618_75379261/
```

to an event-centered layout:

```text
results/events/<EVENT_KEY>/
├── real/
├── forward/
└── compare/
```

The two currently important events are:

```text
results/events/20260401_75336802/
results/events/20260618_75379261/
```

## April event

The April runs predate explicit model naming, so old outputs were deliberately
moved under:

```text
legacy_unversioned/
```

Example:

```text
results/events/20260401_75336802/
├── real/
├── forward/
│   ├── dc000/legacy_unversioned/
│   ├── dc015/legacy_unversioned/
│   ├── dc030/legacy_unversioned/
│   ├── dc035/legacy_unversioned/
│   ├── dc045/legacy_unversioned/
│   ├── dc060/legacy_unversioned/
│   └── dc075/legacy_unversioned/
└── compare/
    └── dc035/legacy_unversioned/
```

Do not silently relabel these old outputs as a particular model.

## June event

The June results are already model-aware:

```text
results/events/20260618_75379261/
├── real/
├── forward/
│   └── dc035/
│       ├── smooth_prior/
│       └── digitized_log/
└── compare/
    └── dc035/
        ├── smooth_prior/
        └── digitized_log/
```

Eventually `digitized_log` should be renamed to `bill_logs`.

## Theta directory naming

The result directory uses:

```text
dc035
```

for `theta = 35°`.

The long string:

```text
n120_g80_xplus500_dc035
```

is a **run tag / metadata identifier**, not a directory name.

This was explicitly fixed in `scripts/safod/settings.py`.

---

# 6. Current result-path configuration

The canonical settings now resolve the June event as:

```text
EVENT_RESULT_KEY   = 20260618_75379261
REAL_EVENT_DIR     = results/events/20260618_75379261/real
REAL_EVENT_PACKAGE = results/events/20260618_75379261/real/real_das_event_window.npz
GEOMETRY_CSV       = results/events/20260618_75379261/real/SAFOD_Phase2_projected_from_georef.csv

FORWARD 35         = results/events/20260618_75379261/forward/dc035
COMPARE 35         = results/events/20260618_75379261/compare/dc035
```

Both:

```text
REAL_EVENT_PACKAGE.exists() == True
GEOMETRY_CSV.exists()       == True
```

were checked successfully.

`run_forward.py` was also patched so that if an older NPZ contains a stale
`geometry_csv` path from before the results migration, it relocates the same
geometry basename beside the current real-event package.

---

# 7. Event selection and settings

The current event workflow is configured through:

```text
scripts/safod/settings.py
```

and event selection is controlled by:

```text
SAFOD_EVENT_KEY
```

Important events:

## April 1, 2026

```text
event id : NC75336802
origin   : 2026-04-01T04:57:57.470Z
magnitude: 0.77 Md
depth    : 1.570 km
```

This event is particularly important because it has usable geophone data at
`SF.MH029`.

## June 18, 2026

```text
event id : NC75379261
origin   : 2026-06-18T21:04:42.290Z
magnitude: 1.61 Md
depth    : 3.430 km
```

For the current 2-D section:

```text
source x          = 1394.244 m
source z          = 3430.000 m
crossline offset  = -111.045 m
```

The June event is currently the main forward-model comparison event.

---

# 8. SAFOD main-hole DAS acquisition / geometry

Current main-hole DAS was installed in late March 2026.

Approximate acquisition parameters:

```text
interrogator       : OptaSense QuantX
nominal sampling   : 1 kHz
main-hole geometry : deviated borehole, ~3 km total
scientific depth   : ~2.55 km TVD at cable end
```

For the June event package:

```text
gauge length       = 8.862434 m
real channel step  = 2.532124 m
real channels      = 1202
reference channels = ~215.8 to 1700.0
```

The fixed physical geometry originates from:

```text
/home/groups/ettore88/alina/SAFOD/SAFOD_Phase2_GeoReferenced_Channels.xlsx
```

The event-specific projected geometry is stored with the real event results.

---

# 9. Exact registered June geometry

The June real package / forward run reported:

```text
registered rows      : 1202
x column             : X_2D_m
z column             : Z_2D_m
channel column       : Channel
raw-channel range    : 215.8 to 1700.0
x range              : 0.025 to 1112.817 m
z range              : 11.796 to 2549.543 m
```

The receiver model keeps exact registered channel centres.

After domain trimming:

```text
receivers              : 1200
receiver spacing       : 2.532124 m
gauge / spacing        : 3.500000
receiver x             : 0.056 to 1112.817 m
receiver z             : 16.860 to 2549.543 m
raw-channel range      : 218.2 to 1700.0
centre mismatch max    : 0.000000e+00 m
```

Gauge-curvature QC:

```text
valid gauge centres : 1196
median rotation     : 0.134119 deg
95th percentile     : 1.162211 deg
maximum rotation    : 1.803008 deg
```

The forward DAS output therefore contains:

```text
1196 DAS traces
```

after the finite-gauge operator.

---

# 10. Correct 2-D section-plane definition

The project has an established 2-D section convention used by
`prepare_event.py`.

The section is **not** a PCA plane and not an arbitrary map slice.

It is defined from the physical SAFOD geometry using an east/north tangent
plane and the direction from the wellhead toward the median deepest down-leg
geometry.

This exact plane is also used for catalog source projection.

Important rule for future Zhang-model extraction:

> Sample the 3-D Zhang model onto the **same exact SAFOD 2-D plane already
> used by `prepare_event.py`**.

Do not use a constant Zhang-X or constant Zhang-Y slice as the final model.

---

# 11. Numerical elastic solver status

The solver is an elastic 2-D finite-difference implementation with Numba
kernels.

The project is organized around:

```text
ElasticModel2D
```

in generic solver code. Generic numerical code should remain separate from
SAFOD-specific model construction.

## Validated numerical points

These issues have already been checked and should not be reopened without new
evidence:

### Exact receiver geometry

The June receiver/channel registration has zero centre mismatch at saved
receiver points.

### Homogeneous moment-tensor benchmark

Validation result:

```text
correlation  = 0.999991
relative L2  = 0.004265
lag          = 0
scale        = 0.9937842
```

### Layered/Numba benchmark

A two-layer Numba benchmark was completed successfully.

### Adjoint

An exact discrete Numba adjoint test was completed successfully.

### Source units / moment tensor

The source implementation and physical scaling were previously audited.
Do not reopen source-unit calibration unless a new failing test requires it.

---

# 12. Current forward numerical configuration

For the successful June `digitized_log` forward run:

```text
grid nx,nz       = 997 × 1121
dx,dz            = 5 m × 5 m
dt               = 2.310283e-04 s
nt               = 12000
duration         = 2.772 s
```

Scientific and computational domain:

```text
scientific model bottom = 5000 m
bottom sponge width     = 600 m
computational bottom    = 5600 m

side sponge width       = 600 m
undamped side margin    = 1100 m
extra scientific x      = 500 m
total x padding         = 1700 m
```

Scientific X extent for that run:

```text
-1100 m to 2680 m
```

The bottom sponge is outside the 5-km scientific model and is hidden in
scientific plots/GIFs.

---

# 13. June source used in current forward experiment

Current source:

```text
event id        : NC75379261
x_src           : 1394.244 m
z_src           : 3430.000 m
crossline       : -111.0 m
theta           : 35°
f0              : 10 Hz
scalar moment   : 1.0e12 N m
source type     : 2-D double couple
spreading       : bilinear
```

The source passes boundary-margin checks.

Record-duration QC for the current model:

```text
max straight-ray S arrival ≈ 1.565 s
tail after far S           ≈ 1.207 s
record duration            = 2.772 s
```

---

# 14. Forward output conventions

The June model-aware run saves into:

```text
results/events/20260618_75379261/
forward/dc035/<initial_model>/
```

Current model directories:

```text
smooth_prior/
digitized_log/
```

The eventual intended names are:

```text
smooth_prior/
bill_logs/
zhang2009/
hybrid_zhang2009_bill_logs/
```

The forward package should contain model identity metadata, including:

```text
run_tag
initial_model_name
model_type
```

and exact DAS physical channel mapping when available:

```text
das_raw_channels
```

The comparison script should prefer `das_raw_channels` and avoid unnecessary
approximate remapping.

---

# 15. Real/synthetic comparison workflow

Current script:

```text
scripts/safod/compare_event.py
```

Current model-aware behavior:

```bash
python -m scripts.safod.compare_event \
    --initial-model digitized_log
```

The comparison uses the same common filter for real and synthetic DAS:

```text
1–20 Hz
zero phase
```

Current display-only synthetic shift:

```text
-0.20 s
```

This shift is for visualization only and must not change saved physical travel
times.

Observed-ridge picking has been removed.

Predicted P/S overlays are retained only as model-based QC guides.

Output structure:

```text
results/events/20260618_75379261/
compare/dc035/<initial_model>/
├── 00_frequency_content_qc.png
├── frequency_content_qc.csv
├── 01_real_with_predicted_arrivals.png
├── 02_real_vs_synthetic_signed.png
├── 03_real_vs_synthetic_envelopes.png
└── comparison_summary.csv
```

---

# 16. April DAS vs geophone validation

This validation has already been completed and is scientifically useful.

Only the April 1 event currently has usable colocated geophone waveform data.

Station/channel:

```text
SF.MH029.01.GP1
```

The geophone response is removed to velocity.

The fiber tangent near DAS channel 1694 is approximately:

```text
[+0.503502, +0.610562, -0.611310] ENU
```

Alignment with GP1:

```text
GP1 azimuth ≈ +35.50°
GP1 dip     ≈ +41.70°
dot(fiber, GP1) ≈ 0.9961
```

so GP1 is essentially axial to the fiber.

Important physical relationships:

```text
strain      ↔ geophone velocity
strain-rate ↔ geophone acceleration
```

approximately through apparent wave speed:

```text
epsilon_parallel     ~ -v_parallel / c_app
epsilon_dot_parallel ~ -a_parallel / c_app
```

The strain-rate vs acceleration comparison was substantially cleaner and is
the primary validation.

Final script created previously:

```text
plot_das_geophone_two_physical_comparisons_png_only.py
```

Only PNG output is desired for this validation.

---

# 17. Initial model 1 — `smooth_prior`

This is the original/current smooth geological prior.

Implementation:

```text
src/safod/models/smooth_prior.py
```

The model contains a smooth depth-dependent background and broad lateral SAF
structure.

Approximate original depth trend:

```text
0–768 m:
    Vp  2.2 -> 5.0 km/s
    Vs  0.9 -> 2.8 km/s
    rho 2050 -> 2500 kg/m3

768–2150 m:
    Vp  5.0 -> 5.65 km/s
    Vs  2.8 -> 3.25 km/s
    rho 2500 -> 2700 kg/m3

>2150 m:
    gentle continuation
    Vp capped ~6.1 km/s
    Vs capped ~3.55 km/s
    rho capped ~2760 kg/m3
```

It also included broad Pilot-Hole-inspired low-velocity anomalies around:

```text
1175 m
1365 m
1858 m
```

with a relatively weak strength (`~0.035` in the current baseline).

Original smooth SAF structure included:

```text
cross-fault contrast            ≈ -8%
cross-fault transition width    ≈ 350 m
Gaussian damage-zone width      ≈ 160 m
velocity reduction              ≈ 14%
post-model Gaussian smoothing   ≈ 80 m
```

The original fault tie was approximately 105 m from the cable end, with:

```text
dip      ≈ 82°
dip sign = -1
```

This model is now the **baseline/control**, not the preferred physical final
model.

Official model name:

```text
smooth_prior
```

---

# 18. Bill Ellsworth feedback motivating model 2

Bill's key feedback was:

> the velocity model looked too fast inside the fault-zone core, and the
> phases missing from the synthetics could be related to the sharp velocity
> jump seen in the borehole logs compared with the gradient in the smooth
> model.

The relevant reference figure is from:

**Ellsworth & Malin (2011)**  
"Deep rock damage in the San Andreas Fault revealed by P- and S-type
fault-zone-guided waves"  
Geological Society Special Publication 359, 39–53  
DOI: `10.1144/SP359.3`

Relevant structures:

```text
GBF
SDZ
CDZ
NBF
ILVZ
damage zone
```

The paper describes a broad damaged interval roughly ~200 m wide and a
localized low-velocity interval beyond/NE of the CDZ, with significant local
velocity reductions.

Related later logging work (Jeppson et al. 2015) reports even stronger narrow
velocity reductions in metre-scale zones, but those exact numerical minima
should **not** be mixed into the raster-digitized Ellsworth & Malin model
unless explicitly comparing studies.

---

# 19. Initial model 2 — Bill borehole logs

The Ellsworth & Malin Figure 3a raster was digitized pointwise.

Current input:

```text
data/safod/velocity_models/ellsworth_malin_2011/
fig3a_digitized.csv
```

Digitization utility:

```text
scripts/safod/models/digitize_ellsworth_malin_fig3a.py
```

The original active implementation is currently named:

```text
src/safod/models/digitized_log.py
```

but the planned scientific name is:

```text
bill_logs
```

because "digitized" is an implementation detail, not a scientific model name.

## Raster calibration used

Main MD axis:

```text
pixels [212,373,534,695,856,1017]
→ MD [3100,3200,3300,3400,3500,3600] m
```

Velocity axis:

```text
pixels [108,188,268,348.5,428.5,509]
→ [6,5,4,3,2,1] km/s
```

TVD lower axis:

```text
pixels [137.5,270.5,428.5,581,717,853,989]
→ [2550,2600,2650,2700,2750,2800,2850] m
```

The digitized curves were resampled at approximately 1 m MD spacing.

## Digitized values at key boundaries

Approximate raster-derived values:

```text
GBF MD 3150 m:
    Vp ≈ 4.215 km/s
    Vs ≈ 2.532 km/s

SDZ MD 3192 m:
    Vp ≈ 3.317 km/s
    Vs ≈ 1.983 km/s

CDZ MD 3302 m:
    Vp ≈ 3.230 km/s
    Vs ≈ 1.759 km/s

NBF MD 3413 m:
    Vp ≈ 3.230 km/s
    Vs ≈ 1.821 km/s
```

These are **raster-derived Figure 3a values**, not LAS/DLIS log samples.

Do not overstate their precision.

## Approximate section offsets reconstructed from MD/TVD

Using:

```text
dx = sqrt(dMD^2 - dTVD^2)
```

and aligning section distance to the cable-end TVD:

```text
GBF ≈  78.1 m from cable end
SDZ ≈ 114.2 m from cable end
CDZ ≈ 208.1 m from cable end
NBF ≈ 298.8 m from cable end
```

Relative to SDZ:

```text
GBF  = -36.1 m
SDZ  =   0.0 m
CDZ  = +93.9 m
NBF  = +184.5 m
```

This suggested that the old ~105 m cable-to-fault tie is most naturally
associated with the **SDZ**, not the CDZ.

This is an approximate section-coordinate reconstruction, not an exact
3-D borehole projection.

---

# 20. Current Bill-log model implementation

The model-building approach was intentionally changed from an earlier
hand-tuned "factor" model to a pointwise raster-derived model.

The superseded hand-tuned implementation:

```text
safod_log_constrained_model.py
```

is archived and should not be used as an official model.

The current digitized-log builder:

- starts from the old smooth background;
- disables/replaces the old smooth lateral SAF contrast / Gaussian fault zone;
- reads pointwise Vp and Vs from the CSV;
- computes separate Vp and Vs anomaly ratios relative to the local background;
- applies those anomalies across signed section distance;
- preserves narrow GBF/SDZ/CDZ/NBF minima at approximately one FD cell;
- applies **no post-log smoothing**;
- leaves density unchanged;
- checks physical Lamé parameters.

The successful June run reported:

```text
log points          : 549
MD range            : 3051.0 .. 3599.0 m
TVD range           : 2548.3 .. 2859.7 m

digitized Vp range  : 3.205 .. 5.487 km/s
digitized Vs range  : 1.759 .. 3.155 km/s

SDZ -> NBF width    : 184.5 m

SW median Vp        : 5.302 km/s
NE median Vp        : 4.340 km/s
representative dVp  : -18.1%
```

Key model grid positions in that run:

```text
GBF:
    offset     = -36.1 m
    TVD        = 2607.0 m
    Vp         = 4.215 km/s
    Vs         = 2.532 km/s
    model width= 5 m

SDZ:
    offset     = 0.0 m
    TVD        = 2628.4 m
    Vp         = 3.317 km/s
    Vs         = 1.983 km/s
    model width= 5 m

CDZ:
    offset     = +93.9 m
    TVD        = 2685.6 m
    Vp         = 3.230 km/s
    Vs         = 1.759 km/s
    model width= 5 m

NBF:
    offset     = +184.5 m
    TVD        = 2749.6 m
    Vp         = 3.230 km/s
    Vs         = 1.821 km/s
    model width= 5 m
```

Post-log smoothing:

```text
NONE
```

---

# 21. Successful June Bill-log forward run

A complete June forward run succeeded with:

```bash
python -m scripts.safod.run_forward --overwrite
```

at a time when the default initial model was `digitized_log`.

The run reported:

```text
initial model : digitized_log
theta         : 35°
output dir    : .../forward/dc035/digitized_log
```

Model ranges:

```text
Vp  : 1280.8 to 5827.0 m/s
Vs  :  515.4 to 3355.4 m/s
rho : 2073.3 to 2751.3 kg/m3
```

DAS synthetic amplitudes:

```text
receiver_vx max_abs : 4.622375e+00
receiver_vz max_abs : 1.351621e+00
das_data max_abs    : 4.909699e-02
```

The simulation completed and saved the wavefield GIF.

This demonstrates that the refactored Bill-log model is numerically usable.

---

# 22. Planned official model naming

The current scientific naming should become:

```text
smooth_prior
bill_logs
zhang2009
hybrid_zhang2009_bill_logs
```

The current active name:

```text
digitized_log
```

should eventually be renamed to:

```text
bill_logs
```

in:

```text
src/safod/models/
factory.py
run_forward.py CLI
compare_event.py CLI
result directories
metadata
```

Do this carefully and deliberately; avoid simultaneously changing scientific
behavior.

The reason for renaming is that "digitized" describes how the current data
were extracted, while `bill_logs` identifies the scientific source.

---

# 23. Clifford Thurber / Steve Hickman model recommendation

Steve Hickman asked Clifford Thurber for a suitable local model for SAFOD DAS
analysis over a radius of roughly 2–3 km around the wellhead.

Cliff recommended:

**Zhang, H., C. Thurber, and P. Bedrosian (2009)**  
"Joint inversion for Vp, Vs, and Vp/Vs at SAFOD, Parkfield, California"  
*Geochemistry, Geophysics, Geosystems*, 10, Q11002  
DOI: `10.1029/2009GC002709`

Cliff explicitly noted that the model:

- is not perfect;
- may have earthquake depths biased somewhat too deep;
- includes shots into the Pilot Hole;
- includes the P/GSI borehole array;
- is appropriate for the local volume Steve described.

Steve explicitly copied the project team so the model could be used for both
shallow and deep SAFOD DAS analyses.

---

# 24. Zhang 2009 raw files

Cliff supplied five files:

```text
MOD.head
inversion_grid.dat
Vp_model.dat
Vs_model.dat
Vpvs_model.dat
```

They are stored unchanged under:

```text
data/safod/velocity_models/
zhang_thurber_bedrosian_2009/raw/
```

Raw files should never be edited in place.

---

# 25. Zhang 2009 raw grid format

`MOD.head` contains:

```text
0.1 13 13 10
-240. -6.00 -3.00 -1.00 0.00 0.70 1.40 2.00 3.00 5.00 7.00 10.00 240.0
-240.0 -8.0 -6.0 -4.0 -2.0 -1.0 -0.0 1.0 2.0 4.0 6.0 8.0 240.0
-150. -0.50 0.00 0.50 1.00 2.00 4.00 7.00 10.0 340.0
```

Thus:

```text
nx = 13
ny = 13
nz = 10
```

Native nodes:

```text
X [km]:
[-240, -6, -3, -1, 0, 0.7, 1.4, 2, 3, 5, 7, 10, 240]

Y [km]:
[-240, -8, -6, -4, -2, -1, 0, 1, 2, 4, 6, 8, 240]

Z [km]:
[-150, -0.5, 0, 0.5, 1, 2, 4, 7, 10, 340]
```

The extreme values are artificial bounding nodes, not the local Parkfield
volume of interest.

The leading scalar:

```text
0.1
```

has been preserved but its meaning has not yet been needed or relied upon.

---

# 26. Zhang model-file ordering — resolved

Cliff explained the ordering directly.

Each model-file line contains all X values for one fixed `(Y,Z)`.

Then:

```text
X varies across the line
↓
Y advances down lines
↓
after all Y values, Z advances
```

The correct NumPy reconstruction is:

```python
cube[z_index, y_index, x_index]
```

Each model file contains:

```text
130 lines × 13 values = 1690 values
```

which exactly matches:

```text
13 × 13 × 10 = 1690
```

This ordering is considered resolved.

---

# 27. Zhang parser created and run

Parser:

```text
scripts/safod/models/zhang2009/parse_model.py
```

It creates:

```text
processed/zhang2009_native_grid.npz
processed/zhang2009_native_nodes.csv
```

and QC products:

```text
qc/zhang2009_grid_summary.txt
qc/zhang2009_xy_grid.png
qc/zhang2009_vp_slices.png
qc/zhang2009_vs_slices.png
qc/zhang2009_vpvs_slices.png
```

The parser does **not** yet interpolate to the SAFOD 2-D model plane.

---

# 28. Zhang native parser QC results

Successful parse:

```text
MOD.head scalar       : 0.1
grid dimensions       : nx=13, ny=13, nz=10
native cube shape     : (10, 13, 13) = (z, y, x)
model values/file     : 1690
```

Native geographic origin:

```text
X=0, Y=0

latitude  = 35.974206542969
longitude = -120.552140299479
```

Full-grid ranges including bounding nodes:

```text
Vp range       : 1.000 .. 9.500 km/s
Vs range       : 0.571 .. 5.429 km/s
joint Vp/Vs    : 1.600 .. 2.100
```

The bounding-node extrema should not be interpreted as local SAFOD velocities.

---

# 29. Vp, Vs, and Vp/Vs handling

The three files are retained independently:

```text
Vp_model.dat
Vs_model.dat
Vpvs_model.dat
```

For elastic forward modelling:

```text
Vp_model.dat
Vs_model.dat
```

are the primary velocity fields.

Do **not** construct Vs as:

```text
Vp / VpVs_model
```

because the supplied Vp/Vs model is its own independently inverted / stored
field.

QC between:

```text
Vp_model / Vs_model
```

and:

```text
Vpvs_model
```

gave:

```text
median |Vp/Vs - joint Vp/Vs| = 0.001313
95th percentile              = 0.330961
maximum                      = 0.755161
```

Keep all three fields but use direct Vp and direct Vs for the elastic model.

---

# 30. Zhang horizontal registration — resolved

Registration script:

```text
scripts/safod/models/zhang2009/qc_registration.py
```

Successful QC result:

```text
Native Zhang origin:
latitude  = 35.974206542969
longitude = -120.552140299479
```

Axis directions inferred directly from `inversion_grid.dat`:

```text
+X azimuth = 50.002595°
+Y azimuth = 320.000020°
```

First positive interior steps:

```text
+X = 700.352 m
+Y = 1000.209 m
```

Orthogonality error:

```text
0.002575°
```

Therefore the horizontal coordinate system is considered correctly decoded.

---

# 31. Zhang origin and Pilot Hole

USGS Pilot Hole wellhead coordinate used in the QC:

```text
lat = 35.97425794
lon = -120.55210714
```

Relative to the Zhang `(X=0,Y=0)` origin:

```text
distance         = 6.440 m
geodetic azimuth = 27.673549°

Zhang X = +5.957 m
Zhang Y = +2.446 m
```

This near-coincidence is a strong independent verification that the Zhang
origin is effectively tied to the SAFOD Pilot Hole.

Do not reinterpret the origin as the current Main Hole wellhead.

---

# 32. Current Main Hole in Zhang coordinates

Project wellhead:

```text
UTM zone 10N
E = 720807.1 m
N = 3983664.0 m
```

Converted geographic position:

```text
lat = 35.972436506449
lon = -120.551146924445
```

Relative to the Zhang origin:

```text
distance         = 215.872 m
geodetic azimuth = 155.477585°

Zhang X = -57.598 m
Zhang Y = -208.043 m
```

This offset is expected.

---

# 33. Zhang QC slices

The current parsing QC includes horizontal slices for:

```text
Z = 0, 0.5, 1, 2, 4, 7 km
```

for:

```text
Vp
Vs
Vp/Vs
```

The local plots show broad kilometre-scale lateral heterogeneity and a strong
cross-fault velocity contrast.

The plots are consistent with a tomography-scale model and do not resolve
metre-to-hundred-metre sharp fault-zone boundaries.

This is exactly why the Zhang model and the Bill-log model are complementary.

---

# 34. Zhang model spatial resolution and scientific role

Near SAFOD the native grid spacing is roughly:

```text
X: ~0.7–2 km
Y: ~1–2 km
Z: ~0.5–1 km in the shallow local model
```

Therefore Zhang 2009 is appropriate as:

```text
regional / kilometre-scale background
```

It is not sufficient by itself to represent:

```text
GBF
SDZ
CDZ
NBF
ILVZ
sharp local fault-zone boundaries
```

Those should come from the borehole-log model.

---

# 35. Vertical Zhang datum — current unresolved item

This is the most important unresolved coordinate issue before 3-D → 2-D
extraction.

Native local Z nodes are:

```text
-0.5, 0, 0.5, 1, 2, 4, 7, 10 km
```

The project solver uses:

```text
z = 0 at the local Main Hole surface
positive z downward
```

The Zhang parser intentionally preserves the native `z_km` without applying a
vertical shift.

Do **not** yet assume:

```text
solver_depth_m = 1000 * z_km
```

Possible interpretation to verify:

- Zhang Z may be depth relative to sea level or another datum;
- the presence of `-0.5 km` strongly suggests a datum that allows nodes above
  the local surface/elevation reference;
- a surface/elevation conversion may be required before mapping to solver z.

This must be explicitly checked before constructing `zhang2009.py`.

---

# 36. Immediate next script — `extract_safod_section.py`

The next planned script is:

```text
scripts/safod/models/zhang2009/extract_safod_section.py
```

It should **not** be written as a simple constant-X or constant-Y slice.

Required workflow:

```text
native Zhang 3-D tomography
        ↓
verified horizontal registration
        ↓
verified vertical datum
        ↓
convert project SAFOD 2-D section plane to Zhang coordinates
        ↓
sample Vp(X,Y,Z), Vs(X,Y,Z) along that plane
        ↓
obtain Vp(section_distance, depth)
       Vs(section_distance, depth)
        ↓
interpolate to the solver's 5 m grid
        ↓
QC before solver integration
```

---

# 37. Required Zhang 2-D extraction QC

Before exposing the model through `factory.py`, the extracted 2-D section
should be plotted with at least:

```text
Vp(section_x, solver_depth)
Vs(section_x, solver_depth)
```

and overlays for:

```text
Main Hole cable
June earthquake source
SAF tie / fault reference
scientific model domain
```

The output should clearly distinguish:

```text
native tomography sampling
interpolated 5 m solver grid
```

Do not judge a 5 m interpolated raster as 5 m physical resolution.

---

# 38. Initial model 3 — `zhang2009`

Once the section extraction is validated, implement:

```text
src/safod/models/zhang2009.py
```

Scientific definition:

```text
Zhang, Thurber & Bedrosian (2009)
3-D SAFOD tomography
sampled onto the exact project 2-D plane
```

It should use:

```text
Vp from Vp_model.dat
Vs from Vs_model.dat
```

Density should remain controlled / consistent with the other models unless a
separate defensible density model is introduced.

Do not add local Bill-log anomalies to this model.

The purpose of the pure Zhang model is to isolate the effect of changing the
regional background.

---

# 39. Initial model 4 — `hybrid_zhang2009_bill_logs`

The planned final candidate starting model is:

```text
hybrid_zhang2009_bill_logs
```

Concept:

```text
Zhang 2009 = regional background
Bill logs  = local fault-zone anomaly
```

A preferred combination is multiplicative/anomaly based rather than replacing
the entire Zhang model with absolute log velocities everywhere.

Conceptually:

```text
Rp = Vp_log / Vp_local_background
Rs = Vs_log / Vs_local_background

Vp_hybrid = Vp_Zhang * Rp
Vs_hybrid = Vs_Zhang * Rs
```

with careful spatial localization around the logged fault-zone section.

Advantages:

- retains Zhang long-wavelength 3-D/tomographic heterogeneity;
- retains sharp borehole-log contrasts;
- avoids flattening the regional background;
- allows a controlled scientific comparison.

The exact blending/tapering rules should be documented and QC'd.

---

# 40. Planned model comparison experiment

The controlled forward comparison should eventually run:

```bash
python -m scripts.safod.run_forward \
    --initial-model smooth_prior

python -m scripts.safod.run_forward \
    --initial-model bill_logs

python -m scripts.safod.run_forward \
    --initial-model zhang2009

python -m scripts.safod.run_forward \
    --initial-model hybrid_zhang2009_bill_logs
```

and corresponding:

```bash
python -m scripts.safod.compare_event \
    --initial-model <MODEL>
```

All other controlled variables should remain unchanged:

```text
same event
same source
same theta
same f0
same geometry
same grid
same boundary conditions
same DAS operator
same comparison filter
same plotting normalization
```

The point is to isolate the effect of the initial velocity model.

---

# 41. Planned factory API

The desired public model API is:

```python
build_initial_model(
    model_name=...,
    ...
)
```

Official model constants / names should become:

```text
smooth_prior
bill_logs
zhang2009
hybrid_zhang2009_bill_logs
```

The generic solver should not know how individual SAFOD models are assembled.

`factory.py` should own model selection.

---

# 42. Density policy for controlled comparisons

For now, do not introduce independent density differences between the four
velocity-model experiments unless scientifically justified.

Preferred policy:

```text
smooth_prior:
    existing rho prior

bill_logs:
    same/background rho policy

zhang2009:
    same/background rho policy

hybrid:
    same/background rho policy
```

This makes Vp/Vs structure the primary controlled variable.

---

# 43. SAFOD deep archive / catalog context

There is also a separate deep-archive/catalog workflow under:

```text
src/safod_deep/catalog.py
```

Config:

```text
config/safod_deep/roots.json
```

The deep archive currently includes March and May roots; recorded data extend
from late March into early June 2026.

A previous catalog pass found approximately:

```text
95,903 valid H5 headers
199 NCEDC events within 50 km
181 events with recorded windows
```

This catalog work is separate from the current Zhang model extraction, but the
same 2-D section-plane convention must be used consistently when projecting
sources.

---

# 44. April event 2-D geometry quality

For April event NC75336802:

```text
source_section_x ≈ 1687.285 m
crossline        ≈ 116.359 m
minimum 3-D dist ≈ 0.807772 km
in-plane dist    ≈ 0.657133 km
out-of-plane angle ≈ 10.041289°
```

This event is somewhat borderline for a strict 2-D approximation but remains
useful for validation, especially because of the geophone comparison.

---

# 45. June event 2-D geometry quality

For June NC75379261:

```text
source x        = 1394.244 m
source z        = 3430 m
crossline       = -111.045 m
```

This event is currently used for the main initial-model forward comparison.

---

# 46. Active-survey context

A June 17, 2026 active-source SAFOD main-hole survey was also processed.

Approximate acquisition:

```text
4 in-compound shots
1 kHz
3200 channels
~2.04 m channel spacing in that acquisition
```

Useful band:

```text
~5–35 Hz
```

with strongest energy around:

```text
15–20 Hz
```

This work is not the immediate task now, but it remains relevant for future
velocity/geometry constraints and VSP/tomography development.

---

# 47. Important scientific caution about "resolution"

The project often interpolates models to:

```text
dx = dz = 5 m
```

for numerical simulation.

This does **not** mean the Zhang tomography has 5 m physical resolution.

Keep separate:

```text
numerical grid spacing
vs
observational/model resolution
```

For Zhang 2009:

```text
physical resolution = kilometre-scale tomography
numerical representation = 5 m FD grid
```

For the Bill raster/log model:

```text
fault-zone structure = metre-to-hundred-metre information
but current source = digitized published raster, not original LAS/DLIS
```

---

# 48. Important scientific caution about Bill-log values

The current Bill-log model uses values digitized from a published figure.

Therefore describe them as:

```text
raster-derived
digitized from Ellsworth & Malin (2011) Figure 3a
```

Do not describe them as exact original logging samples.

Do not mix numerical minima from Jeppson et al. (2015) into this model unless
performing an explicit cross-study comparison.

---

# 49. Important scientific caution about Zhang Vp/Vs

Do not force:

```text
Vp / Vs == Vpvs_model
```

at every node.

Keep:

```text
Vp
Vs
Vp/Vs
```

as separate products supplied by the inversion.

Use direct Vp and Vs for elastic modelling.

Use `Vpvs_model` for QC/interpretation unless a specific modelling reason is
established.

---

# 50. Important coding rule for the next chat

Do not simultaneously:

```text
refactor paths
rename models
change physics
change interpolation
change source
change solver
```

Make one class of change at a time and run import/syntax/QC checks between
steps.

The repository already underwent a substantial path migration. Avoid creating
new compatibility layers unless necessary.

---

# 51. Important "do not redo" list

The following have already been solved or validated:

```text
- basic elastic solver units
- source physical scaling
- exact receiver/channel registration
- finite-gauge DAS receiver geometry
- homogeneous moment-tensor validation
- Numba benchmark
- exact discrete adjoint
- project 2-D section-plane definition
- June event result-path migration
- event-centered results layout
- Zhang model-file reshape/order
- Zhang horizontal geographic registration
- Zhang origin association with Pilot Hole
```

Do not restart these analyses from scratch unless a specific inconsistency is
found.

---

# 52. Current unresolved items, in priority order

## Priority 1 — Zhang vertical datum

Determine precisely how native Zhang `Z` relates to:

```text
elevation
sea level
depth
local surface
```

and derive the correct transformation to the project's solver depth.

This is the immediate next scientific task.

## Priority 2 — exact 3-D → 2-D Zhang extraction

Implement:

```text
scripts/safod/models/zhang2009/extract_safod_section.py
```

using the exact project section plane.

## Priority 3 — extracted-section QC

Plot and inspect Vp/Vs section with:

```text
Main Hole cable
source
SAF tie
```

before solver use.

## Priority 4 — implement `src/safod/models/zhang2009.py`

Only after extraction passes QC.

## Priority 5 — rename `digitized_log` → `bill_logs`

Do a controlled naming migration without changing model physics.

## Priority 6 — build hybrid model

Implement:

```text
hybrid_zhang2009_bill_logs
```

with documented anomaly/blending rules.

## Priority 7 — run all four controlled forwards

Compare travel times, phase content, envelopes, and frequency content.

## Priority 8 — choose FWI starting model

Likely candidate:

```text
hybrid_zhang2009_bill_logs
```

but this should be decided from the controlled real/synthetic comparison, not
assumed in advance.

---

# 53. Suggested immediate next commands/checks

Before new modelling work, confirm the Zhang products exist:

```bash
ls -lh \
data/safod/velocity_models/zhang_thurber_bedrosian_2009/raw \
data/safod/velocity_models/zhang_thurber_bedrosian_2009/processed \
data/safod/velocity_models/zhang_thurber_bedrosian_2009/qc
```

Current Zhang parsing/QC scripts:

```bash
python -m py_compile \
scripts/safod/models/zhang2009/parse_model.py \
scripts/safod/models/zhang2009/qc_registration.py
```

Do not run a new forward model yet for Zhang because the vertical coordinate
conversion and exact section extraction are not implemented.

---

# 54. Current Zhang registration numbers to preserve

These should be treated as regression values for future scripts:

```text
Zhang origin:
lat = 35.974206542969
lon = -120.552140299479

+X azimuth = 50.002595°
+Y azimuth = 320.000020°

+X first step = 700.352 m
+Y first step = 1000.209 m

axis orthogonality error = 0.002575°

Pilot Hole relative to Zhang origin:
distance = 6.440 m
X = +5.957 m
Y = +2.446 m

Current Main Hole relative to Zhang origin:
distance = 215.872 m
X = -57.598 m
Y = -208.043 m
```

If a future parser produces materially different values, investigate before
continuing.

---

# 55. Current June forward regression values to preserve

For the successful Bill-log/digitized-log June run:

```text
grid nx,nz      = 997, 1121
dx,dz           = 5, 5 m
dt              = 2.310283e-04 s
nt              = 12000
duration        = 2.772 s

receivers       = 1200
DAS outputs     = 1196

Vp range        = 1280.8 .. 5827.0 m/s
Vs range        = 515.4 .. 3355.4 m/s
rho range       = 2073.3 .. 2751.3 kg/m3

source:
x,z             = 1394.244, 3430.000 m
theta           = 35°
f0              = 10 Hz

predicted P     = ~0.218 .. 0.857 s
predicted S     = ~0.387 .. 1.565 s
```

These are useful regression targets after model/path renaming.

---

# 56. Model provenance summary

## `smooth_prior`

Source:

```text
project-built smooth geological prior
```

Role:

```text
baseline/control
```

## `bill_logs`

Source:

```text
Ellsworth & Malin (2011) Figure 3a
digitized from published raster
```

Role:

```text
local high-resolution fault-zone structure
```

## `zhang2009`

Source:

```text
Zhang, Thurber & Bedrosian (2009)
files supplied directly by Clifford Thurber
```

Role:

```text
regional SAFOD tomography/background
```

## `hybrid_zhang2009_bill_logs`

Source:

```text
derived project model
Zhang regional background + Bill local anomalies
```

Role:

```text
candidate FWI starting model
```

---

# 57. Recommended final scientific comparison table

When all four models exist, generate a compact table containing at least:

```text
model
Vp min/max
Vs min/max
rho min/max
P arrival residual statistics
S arrival residual statistics
signed waveform similarity
envelope similarity
frequency-content mismatch
phase presence/absence notes
fault-zone phase behavior
```

Keep amplitude scaling and normalization policy identical across models.

---

# 58. Long-term FWI direction

Once the forward-model comparison is satisfactory:

```text
1. choose initial model;
2. define inversion parameterization;
3. define DAS misfit;
4. establish multiscale frequency schedule;
5. verify gradient with adjoint/finite-difference checks;
6. begin synthetic recovery tests;
7. then invert real event(s).
```

Possible inversion parameters:

```text
Vp / Vs
or
lambda / mu
or
log-parameterized elastic moduli
```

The exact choice should be made after the initial model and real/synthetic
forward behavior are stabilized.

Do not jump to real-data FWI before the Zhang/hybrid model comparison is
complete.

---

# 59. Communication / working style for continuation

For future work on this project:

- prefer exact scripts and concrete shell commands;
- keep code changes localized;
- preserve scientific provenance;
- distinguish clearly between raw data, processed products, QC, and solver
  inputs;
- do not invent path names that have not been checked;
- do not silently change established geometry conventions;
- provide model/QC outputs before integrating a new model into the solver.

The user prefers direct technical guidance in Russian, while code, filenames,
comments, and scientific terminology may remain in English.

---

# 60. One-paragraph state summary for a new chat

We have a working 2-D elastic DAS forward modelling pipeline for SAFOD with
validated numerical kernels, exact registered borehole geometry, event-based
results organization, and real/synthetic comparison. The original smooth
initial model (`smooth_prior`) works. A second model derived pointwise from
Bill Ellsworth's Ellsworth & Malin (2011) borehole-log figure has been
implemented and successfully forward modelled for the June 18, 2026 M1.61
event; it is currently called `digitized_log` but should be renamed
`bill_logs`. Clifford Thurber then supplied the Zhang, Thurber & Bedrosian
(2009) SAFOD 3-D Vp/Vs/VpVs tomography. Those raw files have been parsed into a
native `(z,y,x) = (10,13,13)` grid and horizontally registered: the model
origin is within 6.44 m of the SAFOD Pilot Hole, +X azimuth is 50.002595° and
+Y is 320.000020°, and the current Main Hole lies at approximately
Zhang `(X,Y)=(-57.6,-208.0) m`. The immediate unresolved task is to determine
the Zhang vertical datum, then extract the exact project SAFOD 2-D plane from
the 3-D tomography, QC it with cable/source/SAF overlays, implement pure
`zhang2009`, and finally build
`hybrid_zhang2009_bill_logs = Zhang background + Bill local fault-zone
anomalies`. After that, run a controlled four-model forward comparison and
select the FWI starting model.

---

# 61. Immediate continuation instruction to another ChatGPT chat

**Continue from here:**

1. Do **not** refactor the repository again.
2. Do **not** change the existing solver or source.
3. First resolve the **Zhang vertical datum**.
4. Then implement and QC:
   `scripts/safod/models/zhang2009/extract_safod_section.py`.
5. Only after that implement:
   `src/safod/models/zhang2009.py`.
6. Keep pure `zhang2009` separate from the future hybrid model.
7. Preserve the exact horizontal registration numbers listed above as
   regression checks.
8. When renaming `digitized_log` to `bill_logs`, do it as a naming-only
   migration and verify that the June forward regression numbers remain
   unchanged.
