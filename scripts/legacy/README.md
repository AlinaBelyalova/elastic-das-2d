# scripts/legacy — old code

The scripts are kept for history only: they show how old results were
obtained. The code is not maintained, is not updated to the current structure
and is excluded from import checks; `tests/test_project_layout.py` also skips
`scripts/legacy/`. No new code is added here. Current scripts are listed in
[docs/project_map.md](../../docs/project_map.md).

| File | What it did | Status | Result |
| --- | --- | --- | --- |
| `prepare_real_event_20260413.py` | Preparation of the real DAS recording of event NC75343317 (2026-04-13, Md 1.53): two HDF5 files from Oak, 1–20 Hz band, 0–15 s window from the catalog origin time, projection of channels onto the 2-D geometry | Replaced by `scripts.safod.prepare_event` | `results/legacy/real_event_20260413_M153/` |
| `safod_deep/plot_two_selected_events_pretty.py` | "Publication" figures of two SAFOD-deep archive events, 75343317 and 75371066: 5–30 Hz filter, physical reference channel on the horizontal axis | Broken: does not import | `results/safod_deep/event_qc_pretty_two/` |

## prepare_real_event_20260413.py

- Appeared in commit `3b3f1bf` (2026-07-16) together with
  `scripts/prepare_real_event_20260401_75336802.py`. The functions of this
  script (`read_das_db_headers`, `parse_beg_time_from_info`,
  `latlon_to_local_enu_m`, `build_channel_projection_mapping`,
  `project_event_to_model`, `main`) are also in the second one. The second one
  adds `parse_catalog_time_to_datetime` and `fetch_event_from_ncedc`: event
  coordinates and magnitude are queried from NCEDC, whereas here they are
  hard-coded. Commit `7f182aa` moved `prepare_real_event_20260401_75336802.py`
  to `scripts/safod/prepare_event.py`, and this script here.
- `scripts.safod.prepare_event` has since been generalized: the event is taken
  from `EVENTS` in `scripts/safod/settings.py` (selected via
  `SAFOD_EVENT_KEY`), the result is written to
  `results/events/<YYYYMMDD_number>/real/`. Run order is described in
  [docs/safod/workflow.md](../../docs/safod/workflow.md).
- Event NC75343317 is not in `settings.EVENTS`. In the event screening
  (handoff 2026-08-14, §28) it is marked `geometry: poor`.
- After the move in `7f182aa`, only the output path was changed in the code
  (pass `structure_20261017T004955Z`): it was `results/real_event_20260413_M153`,
  now `results/legacy/real_event_20260413_M153`.
- If you run it: the script imports `DASutils` from the DAS-utilities build
  (requires the `geo` environment, see [docs/setup.md](../../docs/setup.md)),
  and the paths to the HDF5 files on Oak, to `DAS_db.py` and to the geometry
  tables are hard-coded. It creates the output folder already at import time,
  relative to the current directory, and when run it overwrites files with the
  same names saved there.
- The script writes 5 files in the result folder:
  `SAFOD_Phase2_projected_from_georef.csv`,
  `real_cable_projected_geometry_from_georef.png`,
  `real_das_event_0_15s_raw_channels.png`,
  `real_das_event_0_15s_trace_normalized.png`,
  `real_das_event_window_0_15s.npz`. A sixth file,
  `channel_projection_mapping_qc.csv`, is not written by this script. Its
  origin is unknown: the name appears neither in the current code nor in the
  git history.

## safod_deep/plot_two_selected_events_pretty.py

- The script imports three functions from
  `scripts/safod_deep/plot_catalog_events.py` that are no longer there:
  `infer_reference_locus_offset`, `load_reference_channel_limits`,
  `physical_reference_axis_and_crop`. They exist in the version from commit
  `a7ffd71` and were removed by commit `26daccc`. They can be viewed with:
  `git show a7ffd71:scripts/safod_deep/plot_catalog_events.py`.
- By default (`--output-dir`) the result is written to
  `results/safod_deep/event_qc_pretty_two/`. In the code the display windows
  are 7.0 s for 75343317 and 2.5 s for 75371066. They correspond to the files
  `event_75343317_pretty_5_30Hz_tmax_7s.png` and
  `event_75371066_pretty_5_30Hz_tmax_2.5s.png`.
- All 7 PNGs in the folder were modified on 2026-08-05 (by file date). The
  other 5 (without the `tmax` suffix or with other values), judging by their
  names, were produced with other settings. No such script versions exist in
  git: there is only the version from `a7ffd71`, identical to the current one
  (except for the header comment).
- The header comment (7.5 s and 3.5 s) does not match the code.

## Former archive/ folder

`archive/` was moved out of the project to
`/home/groups/ettore88/alina/project_archives/Elastic_das_project/archive_code_20261017T020246Z/`.
Its contents match `archive/` in commit `a7ffd71`. Commit `b61b7d5` removed
this folder from the repository. To view the files from git:

```bash
git show a7ffd71:archive/legacy_models/safod_log_constrained_model.py
git show a7ffd71:archive/legacy_scripts/safod/build_initial_model.py
```

| File | What it is |
| --- | --- |
| `legacy_models/safod_log_constrained_model.py` | Earlier SAFOD initial model. File header: model from the digitized Vp/Vs logs of Ellsworth & Malin (2011), Fig. 3a; the header gives the old name `src/safod_digitized_log_model.py` |
| `legacy_scripts/safod/build_initial_model.py` | Earlier script that built the SAFOD initial model (formerly `scripts/build_safod_initial_model.py`, then `scripts/safod/build_initial_model.py`) |

Neither file runs: they import `src.safod_builder`, which no longer exists.
Its code was moved to `src/safod/models/smooth_prior.py` (commit `a7ffd71`).

The full table of old and new paths is in `docs/history/README.md`. The
`docs/history/` folder exists only in the local copy of the project; it is not
in git.
