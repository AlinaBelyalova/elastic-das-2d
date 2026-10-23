# SAFOD: workflow

Steps, commands, inputs and outputs of the SAFOD part of the project.
Scientific conventions (2-D section, vertical datum, comparison policy, model
choice) are not restated here: they are in
`docs/history/FWI_SAFOD_FULL_HANDOFF_2026-08-14.md`,
§5–6, §9, §24–32. `docs/history/` is kept locally and not versioned;
paths inside the handoff are as of the document date.

## Two different tasks

| | Event modelling | SAFOD-deep archive |
| --- | --- | --- |
| Purpose | A few selected earthquakes: DAS preparation, forward in the initial model, comparison | Inventory of continuous recordings, NCEDC catalog, QC plots of all recorded events |
| Code | `scripts/safod/`, `src/safod/` | `scripts/safod_deep/`, `src/safod_deep/`, `config/safod_deep/roots.json` |
| DAS data | event files in `/oak/stanford/groups/ettore88/data/SAFOD/SAFOD_events/` | roots from `roots.json` in `/oak/stanford/groups/ettore88/data/SAFOD/SAFOD-deep/` |
| Results | `results/events/<EVENT_KEY>/`, `results/safod/` | `results/safod_deep/` |
| Next | synthetics, FWI | `web_safod/` |

"deep" in `safod_deep` is the name of the continuous-recording archive of the
SAFOD deep fibre (`SAFOD-deep/`, files `SAFOD-Deep-...h5` and
`SAFOD_Deep_Nano-...h5`), not event depth: the catalog includes events of any
depth within the query radius (`build_event_catalog --max-radius-km`, default
50 km). The `deep_saf` mode in `run_forward.py` (an old synthetic source near
the fault; the code selects `catalog_event`) is unrelated to the archive.

## Common conditions

- Commands are run from the project root: the `results/` and `data/` paths in
  the code are relative.
- Environments `fwi` and `geo`; contents, check and external packages are in
  [setup.md](../setup.md). `geo` is needed by everything that imports
  `DASutils` (the built `pyDAS` from DAS-utilities, Python 3.9) directly or via
  `DAS_db.py` and `prepare_event`.
- Modules that import `scripts.safod.settings` (`prepare_event`,
  `run_forward`, `compare_event`, `selection.screen_events`,
  `velocity_models.zhang2009.qc_registration`,
  `velocity_models.zhang2009.extract_safod_section`) look for the HDF5 files
  of the active event on `/oak` at import time and fail with
  `FileNotFoundError` without them. Everything that reads `/oak` or computes
  is run on a compute node (`sh_dev` or `sbatch`).
- The active event is set by `SAFOD_EVENT_KEY`. Without the variable,
  `DEFAULT_EVENT_KEY = "nc75379261"` is silently used, so it is given in every
  command.
- The scripts query NCEDC (FDSN) for the catalog, origin time and geophone
  records.

Shorthands used in the commands below:

```bash
cd /home/groups/ettore88/alina/classes/geophys245/Elastic_das_project
FWI=/home/groups/ettore88/alina/.envs/fwi/bin/python
GEO=/home/groups/ettore88/alina/.envs/geo/bin/python
```

`<nc_key>` is the registry key in `scripts/safod/settings.py` (for example,
`nc75379261`); `<EVENT_KEY>` is the event folder name (`20260618_75379261`).
`GEO_XLSX` is `/home/groups/ettore88/alina/SAFOD/SAFOD_Phase2_GeoReferenced_Channels.xlsx`,
a single physical cable geometry for all events.

## A. Event modelling

Script paths are relative to `scripts/safod/`.

| Stage | Script | Env. | Input | Output | Command |
| --- | --- | --- | --- | --- | --- |
| 1. Catalog search | `selection/find_catalog_events.py` | fwi | NCEDC, from 2026-04-01 to the time of the run; `GEO_XLSX` | `results/safod/catalog_search/`: `safod_catalog_events_since_april_2026_ranked.csv`, `candidates_crossline_lt{1,2,3}km.csv`, `safod_real_2d_profile_geometry.csv` | `$FWI -m scripts.safod.selection.find_catalog_events` |
| 2. DAS recording coverage | `selection/check_catalog_coverage.py` | geo | `..._ranked.csv` from stage 1; HDF5 headers by recursive glob, including `/oak/stanford/groups/ettore88/data/SAFOD/**/*.h5` | `results/safod/catalog_search/`: `catalog_candidates_with_das_coverage.csv`, `..._coverage_only.csv`, `all_candidate_das_headers.csv` | `$GEO -m scripts.safod.selection.check_catalog_coverage` |
| 3. Screening by 2-D geometry | `selection/screen_events.py` | geo | HDF5 headers in `SAFOD_events/`; NCEDC | `results/safod/event_screening.csv` | `$GEO -m scripts.safod.selection.screen_events` |
| 4. Event registration | `settings.py`, by hand | — | selected event | entries in `EVENTS` and `aliases` | see below |
| 5. Channel registration (if the interrogator configuration changed) | `calibrate_channel_registration.py` | geo | event HDF5, origin time, `GEO_XLSX` | `results/events/<EVENT_KEY>/real/channel_registration_qc/` | see below |
| 6. Observation preparation | `prepare_event.py` | geo | `input_patterns`, `GEO_XLSX`, NCEDC, CSV from `channel_mapping_csv` | `results/events/<EVENT_KEY>/real/`: `real_das_event_window.npz`, `SAFOD_Phase2_projected_from_georef.csv`, `full_cable_qc_registration.csv`, PNG | `SAFOD_EVENT_KEY=<nc_key> $GEO -m scripts.safod.prepare_event` |
| 7. Initial-model inputs (once) | `velocity_models/...` | fwi | `data/safod/velocity_models/*/raw` etc. | `data/safod/velocity_models/*/{processed,qc}` | section "Initial models" |
| 8. Forward | `run_forward.py` | fwi | `real/real_das_event_window.npz`, model inputs | `results/events/<EVENT_KEY>/forward/dc<θ>/<model>/`: `outputs_safod_initial_forward.npz`, PNG; without `--no-gif` also `snapshots_vz.npz` and GIF | `SAFOD_EVENT_KEY=<nc_key> $FWI -m scripts.safod.run_forward --theta-deg <θ> --initial-model <model>` |
| 9. Comparison with synthetics | `compare_event.py` | fwi | `real/real_das_event_window.npz`, `real/SAFOD_Phase2_projected_from_georef.csv`, stage 8 package with the same θ and model | `results/events/<EVENT_KEY>/compare/dc<θ>/<model>/` | `SAFOD_EVENT_KEY=<nc_key> $FWI -m scripts.safod.compare_event --theta-deg <θ> --initial-model <model>` |
| 10. DAS and borehole geophone | `plot_das_geophone_two_physical_comparisons.py` | geo | set in the code: HDF5 of event NC75336802, channel 1694, SF.MH029.01.GP1; NCEDC; `GEO_XLSX` | `--output-dir`, default `results/events/20260401_75336802/compare/das_geophone_physical_pairs/` | `$GEO -m scripts.safod.plot_das_geophone_two_physical_comparisons --output-dir results/events/20260401_75336802/compare/<analysis>` |

Stages 1–3 and 6 have no command-line parameters: settings are constants at
the top of the files. Stage 2 walks the whole SAFOD tree on Oak, including the
SAFOD-deep archive, so it is run only in a Slurm job.

### Event registration in settings.py

- The registry key is `nc<NCEDC number>`. `normalize_event_key()` accepts only
  values from the explicit `aliases` dictionary (`"<number>"` and
  `"nc<number>"` for each event), so a new event is added in two places:
  `EVENTS` and `aliases`. Otherwise: `ValueError: Unknown SAFOD event key`.
- Entry fields (from existing entries): `event_id` (`"NC<number>"`),
  `ncedc_event_id` (int), `origin_time` (ISO, UTC), `magnitude`,
  `magnitude_type`, `depth_km`, `event_tag`, `input_patterns`,
  `channel_mapping_csv`.
  - `input_patterns` are HDF5 paths or glob patterns; each must be found when
    `settings` is imported.
  - `channel_mapping_csv` is `None` if the HDF5 rows match the channel numbers
    of the reference geometry (as for the April events); otherwise a path
    relative to `results/events/<EVENT_KEY>/real/`, as for `nc75379261`:
    `channel_registration_qc/mapped_downleg_model_geometry.csv`.
  - `event_tag` is required (`settings.py` reads it) but is not used anywhere
    else; in all entries it equals `EVENT_RESULT_KEY`.
- `EVENT_RESULT_KEY = <YYYYMMDD>_<number>` is computed from `origin_time` and
  `event_id` without the `NC` prefix and sets the folder
  `results/events/<EVENT_KEY>/`: `nc75379261` → `20260618_75379261`.
- Currently in the registry: `nc75336802` (`20260401_75336802`), `nc75379261`
  (`20260618_75379261`, `DEFAULT_EVENT_KEY`), `nc75336682`
  (`20260401_75336682`, only `real/` prepared). Which events are suitable
  for 2-D comparison: handoff §24, §28–30.
- `prepare_event` reads HDF5 (`DASutils.readFile_HDF`). The 2026-10-03 event
  is recorded in Sintela protobuf and is not in the registry (see
  "Experiments").

### Channel registration

Needed when the HDF5 rows of a new event do not match the channel numbers of
the reference geometry. The script does not import `settings`, so it can be
run before the event is added to the registry; `<EVENT_KEY>` for `--out-dir`
is computed by the rule above.

```bash
$GEO -m scripts.safod.calibrate_channel_registration \
    --h5 <event HDF5> \
    --origin-time <origin_time, UTC> \
    --reference-xlsx /home/groups/ettore88/alina/SAFOD/SAFOD_Phase2_GeoReferenced_Channels.xlsx \
    --out-dir results/events/<EVENT_KEY>/real/channel_registration_qc
```

`--out-dir` receives `channel_registration.json`,
`channel_registration_qc.png`, `turnaround_score.csv`,
`mapped_full_dual_pass_geometry.csv`, `mapped_downleg_geometry.csv` and
`mapped_downleg_model_geometry.csv`; the last one is given in
`channel_mapping_csv`. Other parameters (`--fmin`, `--fmax`, windows,
`--target-fs` etc.) are in `--help`. For the SAFOD-deep archive the same task
is solved separately: `plot_catalog_events --prepare-geometry` (part B).

### run_forward and compare_event

Predicted P/S now use native Zhang 2009 3-D eikonal times for real data,
switching to original Lin 2010 Vp/Vs on Zhang coverage errors,
and the saved forward model's 2-D eikonal times for synthetics. The new
`--arrivals-only` mode computes curves without filtering or rerunning the
elastic solver. Geometry, numerical checks, commands and output conventions
are documented in [travel_times.md](travel_times.md).

- `--initial-model` has different defaults: `run_forward` uses
  `bill_logs`, `compare_event` uses `hybrid_zhang2009_boness2006_bill_logs`
  (the default angle of both is `DEFAULT_THETA_DEG = 35.0`). `compare_event`
  looks for `forward/dc<θ>/<model>/outputs_safod_initial_forward.npz`: if the
  package is missing, it stops with `Synthetic package not found`; if the
  model in the package differs, with `ValueError`. So `--theta-deg` and
  `--initial-model` are always passed to both explicitly and identically. The
  choice of model and angle is a scientific decision (handoff §31–32); this
  document does not make it.
- Allowed models (`AVAILABLE_INITIAL_MODELS` in
  `src/safod/models/factory.py`): `smooth_prior`, `bill_logs`, `zhang2009`,
  `hybrid_zhang2009_bill_logs`, `zhang2009_boness2006`,
  `hybrid_zhang2009_boness2006_bill_logs`,
  `hybrid_zhang2009_boness2006_bill_logs_smooth`.
- `<θ>` is an integer number of degrees, `0 <= θ < 90`: the three-digit
  folder `dc<θ>` (`dc035`) is built by `theta_dir_name()`; a non-integer angle
  raises an error. All saved runs are currently in `dc035`.
- Without `--overwrite`, `run_forward` does not overwrite an existing package;
  `--no-gif` disables field snapshots and GIF; `--nt` is the number of steps
  (default 12000).
- `run_forward` is a full computation and is run in a Slurm job. There is no
  ready `.sbatch` in the project; time and memory are not recorded.
- The scripts do not write a log themselves. For `20260618_75379261` the output
  was saved by hand in `results/events/20260618_75379261/logs/` as
  `forward_<nc_key>_<model>.log` and `compare_<nc_key>_<model>.log`.

### Observation comparisons

Analyses without synthetics (DAS and borehole geophone) go to
`results/events/<EVENT_KEY>/compare/<analysis>/`; naming rules are in
[results/README.md](../../results/README.md).

- `plot_das_geophone_two_physical_comparisons.py` is configured for one event:
  NC75336802, DAS channel 1694, SF.MH029.01.GP1; the HDF5 file and the band
  (`FMIN_HZ`, `FMAX_HZ`, currently 5–30 Hz) are constants in the code; only
  `--output-dir` can be changed from the command line. The method and the
  1–20 Hz variant are described in handoff §26.
- `results/events/20260618_75379261/compare/das_geophone_common_workflow/`
  is created by the notebook `notebooks/NC75379261_DAS_geophone.ipynb`.
- The code that created `results/events/20260401_75336802/compare/das_geophone_5_30Hz_usgs_time/`
  was not found in the current project.

## B. SAFOD-deep archive and web_safod

Script paths are relative to `scripts/safod_deep/`, result folders are
relative to `results/safod_deep/`.

| Stage | Script | Env. | Input | Output | Command |
| --- | --- | --- | --- | --- | --- |
| 1. Inventory of Nano recordings | `scan_nano_manifest.py` | fwi | `--root`, default `/oak/stanford/groups/ettore88/data/SAFOD/SAFOD-deep/2026-06-Nano`; the default label, priority and note repeat the `2026-06_nano` entry in `roots.json` | `catalog/recording_manifest_nano.csv` | `$FWI -m scripts.safod_deep.scan_nano_manifest` |
| 2. Event catalog and coverage | `build_event_catalog.py` | fwi | `config/safod_deep/roots.json`, `GEO_XLSX`, NCEDC | `catalog/`: `recording_manifest.csv`, `ncedc_events_raw.csv`, `catalog_all.csv`, `catalog_recorded.csv`, `catalog_2d_direct_source_candidates.csv`, `event_windows_for_plotting.csv`, `CATALOG_NOTES.md`, PNG | see below |
| 3. Geometry registration per configuration | `plot_catalog_events.py --prepare-geometry` | geo | `catalog/catalog_recorded.csv`, `catalog/recording_manifest.csv`, `GEO_XLSX` | `geometry/acquisition_registration.json`, `geometry/acquisition_registration_qc.png` | `$GEO -m scripts.safod_deep.plot_catalog_events --prepare-geometry` |
| 4. Event QC plots | `plot_catalog_events_array.sbatch` | geo | the same plus the stage 3 registry | `event_qc_3d_eikonal/`: PNG, `metadata/*.json`, `status_shard_<i>_of_008.csv`; Slurm logs in `logs/` | `mkdir -p results/safod_deep/logs && sbatch scripts/safod_deep/plot_catalog_events_array.sbatch` |
| 5. Local HTML gallery | `plot_catalog_events.py --build-index-only` | geo | `event_qc_3d_eikonal/` | `event_qc_3d_eikonal/index.html`, `event_qc_3d_eikonal/gallery_manifest.csv` | `$GEO -m scripts.safod_deep.plot_catalog_events --build-index-only` |
| 6. Web viewer | `web_safod/` | fwi | `catalog/catalog_recorded.csv`, `event_qc_3d_eikonal/`, `event_qc_3d_eikonal/metadata/` (paths in `web_safod/config/explorer.json`) | `web_safod/site/` | [web_safod/README.md](../../web_safod/README.md) |

A new recording folder on Oak is added to `config/safod_deep/roots.json`
(`label`, `path`, `priority`, `notes`).

### Stage 2: build_event_catalog

```bash
$FWI -m scripts.safod_deep.build_event_catalog \
    --roots-config config/safod_deep/roots.json \
    --geometry /home/groups/ettore88/alina/SAFOD/SAFOD_Phase2_GeoReferenced_Channels.xlsx \
    [--wellhead-elevation-m <m>] [--reuse-manifest] [--reuse-events]
```

- Without `--reuse-manifest` the script rescans the HDF5 headers of all roots
  in `roots.json` and overwrites `catalog/recording_manifest.csv`; with the
  flag it reads the existing file. `--reuse-events` uses
  `catalog/ncedc_events_raw.csv` instead of querying NCEDC.
- `scan_nano_manifest` writes a separate file that neither
  `build_event_catalog` nor `plot_catalog_events` reads. The current
  `recording_manifest.csv` contains all 95 903 rows of
  `recording_manifest_pre_nano.csv` and 5 366 of the 5 368 rows of
  `recording_manifest_nano.csv`. How the merge was done is not recorded in
  the code; check with the author before rebuilding the manifest.
- `--wellhead-elevation-m` is needed if the geometry has TVD but no absolute
  elevation; without it 3-D distances are marked unknown (script header:
  "Do not guess wellhead elevation"). The example in the header uses
  `684.0`, `settings.py` has `SAFOD_SURFACE_ELEVATION_M = 660.46`, and in the
  current `catalog_recorded.csv`
  `closest_cable_elevation_m + closest_cable_tvd_m` = 660.5 m in all 291
  rows. Which value to use: check with the author.

### Stages 3–5: plot_catalog_events

- Stage 3 is repeated after the manifest or catalog changes: if the catalog
  has an interrogator configuration without a validated registration, stage 4
  stops with `Unvalidated configurations ... Run --prepare-geometry`.
- The `.sbatch` is submitted from the project root: the `--output` and
  `--error` paths are relative. It calls
  `plot_catalog_events --num-shards 8 --shard-index $SLURM_ARRAY_TASK_ID --overwrite`;
  resources and the job array are set in the file itself. With `--num-shards`
  greater than 1 the gallery is not built; stage 5 exists for it.
- Band, windows, P/S guides and event selection are set by flags
  (`--fmin`, `--fmax`, `--phase-guide-mode`, `--phase-model`, `--event-id`,
  `--geometry-class`, `--limit`, `--output-dir` etc., see `--help`).
  The default is `--phase-guide-mode 3d`: native Zhang 2009 through
  EikoHypoTomo, then original Lin 2010 only on coverage errors. The previous hybrid model
  remains available in explicitly selected legacy modes. See
  [travel_times.md](travel_times.md) for pilot checks and regeneration.
- For the folders `event_qc_1_20Hz/`, `event_qc_3_15Hz/`, `event_qc_5_30Hz/`
  and `event_qc_adaptive/` the commands were not saved; the processing
  parameters of each event are recorded in their `metadata/*.json`.
  `web_safod` now reads `event_qc_3d_eikonal/` by default; retain the older
  galleries as historical products.
- The code that created `geometry/nano_registration.json`,
  `geometry/nano_registration_qc.png` and `geometry/nano_mapping.csv` was not
  found in the current project.
- `web_safod/scripts/fetch_focal_mechanisms.py` writes
  `catalog/focal_mechanisms.json`. Building and publishing the site are
  described in [web_safod/README.md](../../web_safod/README.md).

## Initial models

Models are built in `src/safod/models/`: `factory.build_initial_model(model_name=...)`
selects the builder by name from `AVAILABLE_INITIAL_MODELS`, one module per
model. The modules read prepared files from `data/safod/velocity_models/`
(default paths are in `factory.py`, `zhang2009.py`, `boness_zoback2006.py`).
The models are needed by `run_forward` and by the P/S guides in
`plot_catalog_events`.

Model inputs are prepared once by the scripts in
`scripts/safod/velocity_models/` and written to `data/`, not to `results/`.
`data/safod/velocity_models/` is excluded from Git.

| Set | Script (`scripts/safod/velocity_models/`) | Env. | Output (`data/safod/velocity_models/`) |
| --- | --- | --- | --- |
| Zhang et al. 2009, native grid | `zhang2009/parse_model.py` | fwi | `zhang_thurber_bedrosian_2009/processed/zhang2009_native_grid.npz`, `.../zhang2009_native_nodes.csv`, `.../qc/` |
| Zhang 2009, horizontal registration check | `zhang2009/qc_registration.py` | fwi | `zhang_thurber_bedrosian_2009/qc/zhang2009_registration_summary.txt`, `.../zhang2009_local_geographic_grid.png` |
| Zhang 2009, SAFOD 2-D section | `zhang2009/extract_safod_section.py` | fwi | `zhang_thurber_bedrosian_2009/processed/zhang2009_safod_section_2d.npz`, `.../zhang2009_safod_section_coordinates.csv`, `.../qc/` |
| Ellsworth & Malin 2011, Fig. 3a | `digitize_ellsworth_malin_fig3a.py` | fwi | `ellsworth_malin_2011/fig3a_digitized.csv`, QC PNG |
| Boness & Zoback 2006 | no script in the project, `raw/` is empty | — | prepared `boness_zoback_2006/processed/boness_zoback2006_fig3_4_vp_vs_digitized.csv` |

```bash
Z=data/safod/velocity_models/zhang_thurber_bedrosian_2009
$FWI -m scripts.safod.velocity_models.zhang2009.parse_model --raw-dir $Z/raw --output-dir $Z
$FWI -m scripts.safod.velocity_models.zhang2009.qc_registration
SAFOD_EVENT_KEY=<nc_key> $FWI -m scripts.safod.velocity_models.zhang2009.extract_safod_section

$FWI -m scripts.safod.velocity_models.digitize_ellsworth_malin_fig3a \
    --image <Fig. 3a PNG> \
    --csv data/safod/velocity_models/ellsworth_malin_2011/fig3a_digitized.csv \
    --qc data/safod/velocity_models/ellsworth_malin_2011/qc/fig3a_digitization_qc.png
```

- `qc_registration` and `extract_safod_section` import `settings` and need
  `/oak`. `extract_safod_section` by default takes the `--geometry-csv` of the
  active event,
  `results/events/<EVENT_KEY>/real/SAFOD_Phase2_projected_from_georef.csv`,
  so it is run after `prepare_event` for that event.
- The source PNG for digitizing Fig. 3a is not in the project.
- Data sources are described in
  `data/safod/velocity_models/zhang_thurber_bedrosian_2009/README.md` and
  `data/safod/velocity_models/ellsworth_malin_2011/README.md` (local, outside
  Git); the scientific content of the models is in handoff §7–19.

## Experiments and legacy code

- `scripts/safod/experiments/`: one-off diagnostics outside the main
  workflow:
  - `diagnose_geometry.py` checks whether TVD in `GEO_XLSX` is monotonic when
    sorted by channel number and prints the result:
    `$FWI -m scripts.safod.experiments.diagnose_geometry`.
  - `oct03_2026/`: event 2026-10-03 (nc75443276), Sintela protobuf and
    MH030: [README](../../scripts/safod/experiments/oct03_2026/README.md).
- `scripts/legacy/safod_deep/plot_two_selected_events_pretty.py`: plots of
  two selected archive events. It does not import: `plot_catalog_events.py`
  no longer has three functions it needs. Its saved result is
  `results/safod_deep/event_qc_pretty_two/`. On legacy code in general:
  [scripts/legacy/README.md](../../scripts/legacy/README.md).

## External consumer: SAFOD_WaveSearch

The project `/home/groups/ettore88/alina/SAFOD/SAFOD_WaveSearch` reads the
geometry by the absolute path
`/home/groups/ettore88/alina/classes/geophys245/Elastic_das_project/results/events/20260618_75379261/real/SAFOD_Phase2_projected_from_georef.csv`
(checked with grep 2026-10-16): `scripts/wavesearch/run_feasibility_pipeline.sh`
(default value of `GEOMETRY`), `configs/data/oak_safod_wave_search.yaml`,
`configs/data/fwi_boness_dev.yaml`, `configs/paths/sherlock.example.yaml`,
and also `README.md` and `docs/NEXT_STEPS.md`.

Therefore `results/events/`, the folder `20260618_75379261` and this file are
not renamed or moved without a simultaneous edit of SAFOD_WaveSearch.
`real/channel_registration_qc/channel_registration.json` of the same event
also stores absolute paths inside the project. `results/` is excluded from
Git, so the file exists only in the working copy on Sherlock.
