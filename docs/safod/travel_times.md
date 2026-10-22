# Predicted P/S travel times

These are model-based first-arrival guides, not waveform picks or ground-truth
labels. The website and real-data panel of `compare_event` share
`src/safod/travel_times.py`: native Zhang, Thurber & Bedrosian (2009) 3-D Vp
and Vs through EikoHypoTomo/PyKonal. The synthetic panel uses a separate 2-D
eikonal calculation on the exact Vp/Vs arrays saved in its forward NPZ.
The elastic solver remains 2-D.

## Coordinates and model support

- Input: `data/safod/velocity_models/zhang_thurber_bedrosian_2009/processed/zhang2009_native_grid.npz`.
  Prepare it with `scripts.safod.velocity_models.zhang2009.parse_model` if
  absent. Raw/model files are local and excluded from Git.
- Native arrays are `[z,y,x]` in km/s. Calculations use metres, seconds,
  and m/s. EikoHypoTomo receives `[component,y,x,z]` and source/receiver
  columns `[x,y,z]`. Vp and Vs are independent primary fields; the separately
  inverted Vp/Vs field is not used to reconstruct Vs.
- Geographic conversion preserves the Zhang origin and positive-axis
  azimuth convention in `extract_safod_section.py`. Full source and receiver
  coordinates are used without projection onto the SAFOD profile.
- Vertical coordinates are depth below MSL, positive down:
  `source_z_m = 1000 * catalog_depth_km`,
  `receiver_z_m = TVD_m - 660.46`. The established flat local ground datum
  is retained; this is not a terrain-following calculation.
- Artificial horizontal nodes at +/-240 km are excluded by retaining
  nodes strictly inside (-200,200) km. Physical vertical nodes in [-0.5,10]
  km are retained, matching the established section extractor. Actual
  supported bounds are read from the remaining coordinates.
- Above the first physical depth node, velocity is extended constantly to
  local ground, preserving the existing section-preparation convention.
  No lateral/bottom extrapolation or source/receiver clipping is permitted.
- Only the original 3-D Zhang velocities are used. The hybrid model's 2-D
  borehole corrections and fault-zone anomaly are not extended into 3-D.

The initial numerical defaults are 100 m spacing and 3000 m padding around
the source/receiver bounding box, bounded by physical model support. Check
selected events at smaller spacing and larger padding before publication;
these defaults do not establish convergence or the model's physical
resolution. Events outside supported coverage receive no substituted
straight-ray prediction.

The default grid limit is 12,000,000 nodes. Bounds, shape and a rough
working-memory estimate (160 bytes/node, excluding DAS and other processes)
are printed. Velocity interpolation processes one x slab at a time.

## External backend and cache

The installed checkout must provide `python/EikonalTomography3DMod.py`,
`python/EikonalTomography2DMod.py`, and
`external/python-solver/GenericSolver/python/`. Use `--eiko-root` if it is
not at `/home/groups/ettore88/alina/packages/EikoHypoTomo`. Missing modules
fail explicitly; no external package is installed or modified by this code.

The 3-D adapter targets `EikonalTT_3D(vel, tt, oy, ox, oz, dy, dx, dz,
SouPos, RecPos, ...)` with accurate point-source injection, no ray tracing,
and no retained full travel-time maps. The interface used for local testing
is in the [AI4EPS workshop source](https://github.com/AI4EPS/DAS_Seismology_Workshop/blob/63b1ccea6622eda6075ce58be49e6f9568fb57ae/notebooks/lab4_eikonal_tomography/Scripts/EikonalTomography3DMod.py).
The installed Sherlock version must pass the QC below too.

The 2-D wrapper retains the existing grid-node injection required by its
singleton dimension. The injected coordinates are recorded; its source
position error is a separate grid-resolution effect.

3-D caches live in `results/safod/travel_times_3d/`. Keys include model hash,
coordinate/datum conventions, source and receiver coordinates, bounds,
spacing, padding, wrapper hash, and PyKonal version. Each NPZ contains times
and JSON provenance; concurrent writes atomically replace complete temporary
files. Model/backend changes invalidate the cache.

## First verification on Sherlock

Run from the project root on a compute node in `geo`:

```bash
GEO=/home/groups/ettore88/alina/.envs/geo/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
$GEO -m scripts.validation.safod.travel_times_3d \
  --output-dir results/validation/safod/travel_times_3d_sherlock_20261021
```

This tests the installed wrapper against Euclidean distance / velocity in a
homogeneous medium with an off-grid source and four spatially distributed
receivers, for P and S at h=200,100,50 m. The finest error must decrease from
the coarsest and be below 5 ms; homogeneous S/P scaling is also checked.
Use a new output directory. This checks the backend, not real Zhang
registration or agreement with SAFOD waveforms.

With the public wrapper and PyKonal 0.4.1, local finest-grid maximum errors
were 1.774 ms for P and 3.050 ms for S. The installed backend and real data
were not available for that test.

## Prepared real/synthetic event

```bash
SAFOD_EVENT_KEY=nc75379261 $GEO -m scripts.safod.compare_event \
  --initial-model hybrid_zhang2009_boness2006_bill_logs --theta-deg 35 \
  --arrivals-only
```

This calculates curves without loading/filtering waveforms or rerunning the
elastic simulation. The matching forward package and prepared channel
geometry must exist. A new `eikonal_3d/` subdirectory under the model
comparison directory receives `predicted_arrivals_real_3d.csv`,
`predicted_arrivals_synthetic_2d.csv`, and `predicted_arrivals_metadata.json`.

Use `--output-dir` for a separate experiment directory. Repeat with
`--phase-3d-spacing-m 50 --phase-3d-padding-m 6000` and compare predictions
in seconds channel by channel to assess grid/padding sensitivity.

Omit `--arrivals-only` for the full comparison. The real panel shows 3-D
Zhang guides and the synthetic panel matching-model 2-D guides. Both are
origin-relative: shift only the synthetic waveform axis by `-STF.t0`, not
the predicted travel times a second time. Existing `predicted_*_min/max_s`
summary fields still describe synthetic guides; added method/model columns
identify their provenance.

The existing full-comparison filter setting `COMMON_FMIN_HZ=0.0` remains
incompatible with `bandpass_traces`. Choosing a positive bandpass lower
corner versus a low-pass is a separate scientific decision. The
`--arrivals-only` path does not invoke filtering.

## Website pilot and regeneration

```bash
$GEO -m scripts.safod_deep.plot_catalog_events \
  --phase-guide-mode 3d --event-id 75379261 \
  --output-dir results/safod_deep/event_qc_3d_pilot_20261021
```

Use an event present in `catalog_recorded.csv`. Inspect its companion image
and `metadata/*.json`: `travel_time_guide.details` records model hash,
source, grid, datum, cache key and backend. Repeat with finer spacing and
larger padding, and compare with clear observed first arrivals, accounting
separately for the plot's filter phase.

The default gallery is now `results/safod_deep/event_qc_3d_eikonal/`;
the previous `event_qc/` is retained. `web_safod/config/explorer.json`
points to the new gallery, and the existing array job uses the 3-D default.
After pilot validation and gallery generation, follow the
[web runbook](../../web_safod/README.md) to build the site. Source changes
alone neither regenerate plots nor publish the website.

A missing backend fails before reading DAS. An unsupported event or a
failed solve retains its observed DAS plot without a P/S companion. The
reason is printed and saved as `travel_time_guide_error`; the shard summary
counts missing guides. Explicit `auto`, `eikonal`, and `straight` retain
legacy modes for reproducibility; `3d` never falls back to them.
