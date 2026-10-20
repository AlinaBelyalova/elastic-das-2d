# Computed results

Everything computed lives here: reports, JSON/CSV, arrays, figures and logs.
The report, numbers, figures and log of a single run are kept in its folder.
Methodology is in `docs/`, code locations are in the
[project map](../docs/project_map.md).

The contents of `results/` are local and excluded from Git (`results/*` in
`.gitignore`); only this README is versioned.

## Sections

| Folder | Created by | Contents |
| --- | --- | --- |
| `validation/sbp_sat/<run>/` | `scripts.validation.sbp_sat.oblique_qc`, `time_energy_qc` with `--output-dir` | `oblique/`, `time_energy/` (`summary.json`, `traces.npz`, PNG with `--plot`); the run-level `validation_report.md` is written by hand, the scripts do not create it. Currently: `reference_v2/`, `sherlock_20261016/` |
| `validation/elastic/<check>/` | `scripts/validation/elastic/` | one folder per script that writes files: `boundary_absorption/`, `free_surface_normal_incidence/`, `magnitude_scaling/`, `moment_tensor_analytical/`, `moment_tensor_linearity/`, `pointforce_2d/`, `reciprocity/`, `source_radiation/` |
| `validation/fwi/<check>/` | `scripts/validation/fwi/` | `two_layer_numba/`, `two_layer_taylor/` |
| `examples/homogeneous_demo/` | `scripts.examples.homogeneous_demo` | homogeneous demo run, `explosive_validation/` |
| `benchmarks/performance/` | `scripts.benchmarks.performance` | currently only `benchmark_scaling_free_surface_true.json`; the script creates the PNG on every run (together with the JSON, overwriting both) |
| `events/<YYYYMMDD_number>/` | `scripts/safod/` pipeline | one SAFOD event, layout below |
| `safod/` | `scripts/safod/selection/` | `catalog_search/` (`find_catalog_events`, `check_catalog_coverage`); `event_screening.csv` (`screen_events`) |
| `safod_deep/` | `scripts/safod_deep/`, `web_safod` | SAFOD-deep archive, layout below |
| `legacy/` | earlier code versions | old runs and migration manifests, index below |

Checks that only print their result (for example `operator_qc`) create no
folders; the list is in
[scripts/validation/README.md](../scripts/validation/README.md).

## SAFOD event

The event is selected with the variable `SAFOD_EVENT_KEY=nc<number>` and must
be registered in `scripts/safod/settings.py` (the `EVENTS` registry and
`aliases`). The folder name is `EVENT_RESULT_KEY` = origin-time date and
number, for example `nc75379261` → `events/20260618_75379261/`. Run order is
in [docs/safod/workflow.md](../docs/safod/workflow.md).

```text
events/<YYYYMMDD_number>/
├── real/                            prepare_event: real_das_event_window.npz, geometry, previews
│   └── channel_registration_qc/     calibrate_channel_registration (--out-dir), if channel registration differs
├── forward/dc<θ>/<initial_model>/   run_forward
├── compare/dc<θ>/<initial_model>/   compare_event
├── compare/<analysis>[_YYYYMMDD]/   other comparisons, for example DAS vs geophone
└── logs/                            run output (the scripts do not write logs themselves)
```

`dc<θ>` is the value of `--theta-deg` (2-D double couple orientation) in whole
degrees, three digits (`dc035` for 35°).
`<initial_model>` is the value of `--initial-model`.

Kept per the 2026-08-14 handoff convention (not renamed):

- `events/20260401_75336802/{forward,compare}/dc035/legacy_unversioned/` —
  April runs from before the split by model name;
- `events/20260618_75379261/{forward,compare}/dc035/digitized_log/` — former
  model name `bill_logs`.

In `events/20260401_75336802/`, analyses with the `_20260807` suffix are the
2026-08-07 generation moved out of `real/`;
`compare/das_geophone_physical_pairs/` is the default folder of
`plot_das_geophone_two_physical_comparisons`. The files `real/*_0_15s*`
(2026-07-08) follow the naming scheme of the earlier preparation script (as in
`scripts/legacy/prepare_real_event_20260413.py`); the current `prepare_event`
does not write them.

## SAFOD-deep archive

| Folder | Created by | Contents |
| --- | --- | --- |
| `safod_deep/catalog/` | `scan_nano_manifest`, `build_event_catalog`; `focal_mechanisms.json` — `web_safod.scripts.fetch_focal_mechanisms` | recording manifests, NCEDC catalog, overview figures, `CATALOG_NOTES.md`; there is no source script in the project for `recording_manifest_pre_nano.csv` |
| `safod_deep/geometry/` | `plot_catalog_events --prepare-geometry` | `acquisition_registration.json` and a QC figure: registration of recording configurations to the physical cable geometry. There is no source script in the project for `nano_mapping.csv` and `nano_registration.*` (check with the author) |
| `safod_deep/event_qc/` | `plot_catalog_events`, `plot_catalog_events_array.sbatch` | event QC gallery and `metadata/`; read by `web_safod` (`web_safod/config/explorer.json`) |
| `safod_deep/event_qc_1_20Hz/`, `event_qc_3_15Hz/`, `event_qc_5_30Hz/`, `event_qc_adaptive/` | earlier runs (August 2026) | earlier gallery variants; `web_safod` does not read them |
| `safod_deep/event_qc_pretty_two/` | `scripts/legacy/safod_deep/plot_two_selected_events_pretty.py` | figures of two selected events; the script currently does not run |
| `safod_deep/logs/` | Slurm | logs of `plot_catalog_events_array.sbatch` |

## Legacy

| Folder | Contents |
| --- | --- |
| `legacy/mechanism_compare_20260718/` | real vs synthetic comparisons for 20260401_75336802 at θ = 0, 15, 30, 45, 60, 75° (July 2026, earlier naming scheme) |
| `legacy/migrations/` | `results_reorg_20260807_223722.txt` — manifest of the 2026-08-07 results migration |
| `legacy/numpy_homogeneous_baseline_20260604/` | earlier homogeneous NumPy run (2026-06-04) |
| `legacy/real_event_20260413_M153/` | 2026-04-13 event preparation by `scripts/legacy/prepare_real_event_20260413.py`; which files it writes is in [scripts/legacy/README.md](../scripts/legacy/README.md) |
| `legacy/safod_initial_model_20260629/` | earlier SAFOD initial model: `safod_initial_model.npz` and Vp, Vs, rho figures |

Index of old code: [scripts/legacy/README.md](../scripts/legacy/README.md);
earlier handoffs and migrations: [docs/history/README.md](../docs/history/README.md).

## Rules

- Result folder = script name without `_qc`: `free_surface_normal_incidence_qc`
  → `validation/elastic/free_surface_normal_incidence/`. For SBP-SAT, a run
  folder `validation/sbp_sat/<label>_<YYYYMMDD>/` with subfolders `oblique/`,
  `time_energy/`.
- Every new run gets a new name. `oblique_qc` and `time_energy_qc` refuse to
  write into a non-empty folder without `--overwrite`; `run_forward` without
  `--overwrite` does not replace an existing forward NPZ. The archived
  `reference_v2/` and `sherlock_20261016/` are not overwritten.
- Slurm logs go to `results/<area>/logs/` (for an event,
  `events/<YYYYMMDD_number>/logs/`). Slurm does not create the folder itself:
  run `mkdir -p` before `sbatch`. Site build logs go to `web_safod/cache/`, the
  site itself to `web_safod/site/`.
- Exception: QC of velocity model preparation sits next to the inputs in
  `data/safod/velocity_models/<source>/qc/`.
- Large runs on Sherlock can be done in `$SCRATCH` or `$GROUP_SCRATCH` with
  the same layout. These spaces are purged, so the only copy that must be kept
  has to be on permanent storage.

## SBP-SAT

- [Local Sherlock run 2026-10-16](validation/sbp_sat/sherlock_20261016/validation_report.md)
- [Original QC v2 reference](validation/sbp_sat/reference_v2/validation_report.md)
- Methodology and run commands: [docs/sbp_sat/validation.md](../docs/sbp_sat/validation.md);
  run comparison: [docs/status.md](../docs/status.md)
