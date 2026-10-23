# Predicted P/S travel times

These are model-based first-arrival guides, not waveform picks or ground-truth
labels. The website and real-data panel of `compare_event` share
`src/safod/travel_times.py`: native Zhang, Thurber & Bedrosian (2009) 3-D Vp
and Vs, switching to the original separate Lin et al. (2010) fields only on
a Zhang coverage error. Both use EikoHypoTomo/PyKonal. The synthetic panel uses a separate 2-D
eikonal calculation on the exact Vp/Vs arrays saved in its forward NPZ.
The elastic solver remains 2-D.

## Zhang coordinates and model support

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

## Original Lin 2010 regional alternative

Default selection is `--phase-3d-model zhang-lin`: keep Zhang when it covers
the source and receivers; use one complete Lin model for both P and S after
`CoverageError`. This does not blend fields. Solver failures, node limits,
missing files, invalid data and backend errors do not trigger model switching.
Use `--phase-3d-model zhang` to reproduce a Zhang-only run, or `lin` to test
Lin explicitly without requiring a Zhang file.

Inputs, excluded from Git:

- `data/safod/velocity_models/lin2010/raw/out.vp`
- `data/safod/velocity_models/lin2010/raw/out.vs`

Override them with `--phase-lin-vp` / `--phase-lin-vs`. These are the author's
original seven-column files, from the
[model page](https://sites.google.com/view/guoqing-lin/products/ca_statewide).
Citation: Lin et al. (2010), BSSA 100, 225-240,
[doi:10.1785/0120090028](https://doi.org/10.1785/0120090028).
The current SCEC uwlinca distribution derives Vs empirically from Vp and
is not interchangeable with this original separate out.vs.

`src/safod/lin2010.py` validates complete unique native grids and positive
finite velocities. Native shapes [z,y,x] are (9,133,61) for Vp and (9,45,21)
for Vs. Both have depth nodes -1,1,4,8,14,20,27,35,45 km below MSL;
horizontal nodes are 10 km and 30 km apart respectively.

For interpolation, invert the published geographic quadrilaterals bilinearly
to locate each point on its native x/y grid. Interpolate each original field
trilinearly there. This avoids inventing an undocumented author projection.
Geographic coordinates have 0.0001 degree file precision. Only fractional-cell
roundoff within 1e-10 of an edge is rounded to the edge (3 micrometres for
30 km nodes); outside geographic cells and native depths fail explicitly.

For propagation, use a local WGS84 azimuthal-equidistant east/north frame
centred at the existing SAFOD wellhead (35.9742039 N, -120.5521414 E).
Its x/y coordinates are metres; depth remains positive down below MSL.
Uniform Cartesian FMM in this local frame neglects Earth curvature over
these local/regional paths. Geographic lookup and SI conversion are performed
before FMM; native x/y labels are not treated as physical EN coordinates.

Lin uses its original top node at -1 km MSL, with no top extrapolation.
An above-sea-level catalogue source retains its original depth, including
the source at -0.67 km; it is not moved to SAFOD's local ground elevation.
`source_above_local_flat_ground` records that case. This preserves catalogue
geometry but does not supply terrain, a free surface or an improved hypocentre.
The receiver convention remains z=TVD-660.46 m.

Lin defaults to `--phase-lin-spacing-m 200` and uses the same requested
padding as Zhang (3000 m by default). This independent numerical spacing
keeps regional volumes below the existing 12-million-node limit; it does not
increase the original model's resolution. The entire used volume must have
Vp > Vs. One distant native Vp node near 33.15 N, -115.56 E fails this
comparison against interpolated native Vs; its count is recorded, and any
used volume containing an inconsistency fails rather than changing values.

Metadata and labels identify the selected model, both original file hashes,
coordinate frame, actual grid spacing and reason for switching. Cache schema
2 includes those model/coordinate changes; schema-1 predictions are retained
on disk but are not reused. Real comparison CSVs retain their Zhang-coordinate
columns and add receiver_model_x_m / receiver_model_y_m; in explicit Lin-only
mode the unavailable Zhang-coordinate columns are NaN.

### Verify on Sherlock before regenerating the gallery

```bash
GEO=/home/groups/ettore88/alina/.envs/geo/bin/python
$GEO -m scripts.validation.safod.lin2010_coverage \
  --output-dir results/validation/safod/lin2010_coverage_sherlock
$GEO -m scripts.safod_deep.plot_catalog_events \
  --event-id 75379216 --phase-guide-mode 3d \
  --output-dir results/safod_deep/event_qc_lin_pilot
```

The coverage check reads the actual physical borehole geometry and recorded
catalogue, without DAS or FMM. Inspect the pilot PNG and metadata: model_name
must be lin2010_original_vp_vs_3d, fallback_used true, and fallback_reason a
Zhang coverage error. A companion image must actually exist.

Local patch verification used original files (SHA256 Vp
f2043b320a473ea4c078fa0487cdf9c54563fcd534ef5d5eeed7a4069a37102c,
Vs cafc52c4f83e9fab6606990eb4f0432c5eb9e5584feeb52736b8e26afe869f93),
291 published source coordinates and 2991 published fibre horizontal points.
All were covered; padded 200 m domains also lay within both geographic grids.
The largest regional domain was (131,255,111), 3,707,955 nodes, approximately
0.55 GiB at the shared 160-byte/node estimate.

Numerical FMM pilots used the public workshop wrapper and PyKonal 0.4.1,
with the exact published wellhead and turnaround receiver positions only.
For nc75379216, refinement 200 to 100 m changed predictions by at most
1.785 ms P and 2.619 ms S; padding 3 to 6 km at 200 m changed them by less
than 1e-6 ms. Deep/distant nc75414397 (17.98 km) and negative-depth
nc75387326 (-0.67 km) also solved with finite P/S and S after P.
This does not replace a full-channel Sherlock pilot or validate observed onsets.

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

A missing backend fails before reading DAS. An event outside both models or a
failed solve retains its observed DAS plot without a P/S companion. The
reason is printed and saved as `travel_time_guide_error`; the shard summary
counts missing guides. Explicit `auto`, `eikonal`, and `straight` retain
legacy modes for reproducibility; `3d` never falls back to them.
