# Environments, data and tests

How to work with the project on Sherlock: Python environments, local non-PyPI
packages, input data, external paths hard-coded in the code, running and
tests. Everything below was checked on 2026-10-16 in a job on a compute node
(import, `ls`, `grep`, `importlib.metadata`); no computations were run. Where
things are in the project:
[project_map.md](project_map.md), SAFOD steps: [safod/workflow.md](safod/workflow.md).

## 1. Environments

Two conda environments. They do not need to be activated: it is enough to call
the interpreter by its full path (`<environment>/bin/python`).

|  | `fwi` | `geo` |
|---|---|---|
| Path | `/home/groups/ettore88/alina/.envs/fwi` | `/home/groups/ettore88/alina/.envs/geo` |
| Python | 3.11.13 | 3.9.23 |
| numpy | 2.0.1 | 1.26.4 |
| scipy | 1.16.0 | 1.13.1 |
| matplotlib | 3.10.8 | 3.9.4 |
| numba | 0.65.1 | 0.60.0 |
| pandas | 3.0.2 | 2.3.3 |
| h5py | 3.14.0 | 3.14.0 |
| obspy | 1.5.0 | 1.4.2 |
| pyproj | 3.6.1 | 3.6.1 |
| python-dateutil | 2.9.0.post0 | 2.9.0.post0 |
| openpyxl | 3.1.5 | 3.1.5 |
| Pillow | 12.1.1 | 11.3.0 |
| pytest | 9.0.2 | no |
| pykonal | no | 0.4.1 |
| ipykernel / jupyterlab / notebook | 7.2.0 / 4.5.5 / 7.5.4 | 6.30.1 / 4.4.6 / no |

The `jupyter` metapackage is in neither environment, but the `jupyter` command
exists in both. `fwi` also has devito (editable install); project code does
not import it, but because of it a foreign `scripts` package is found outside
the project root (section 5).

**`fwi`** is the main environment: `src/`, `scripts/validation/`,
`scripts/examples/`, `scripts/benchmarks/`, `scripts.safod.run_forward`,
`scripts.safod.compare_event`, `web_safod/`, tests (pytest is only here).

**`geo`** is needed by everything that imports `DASutils`: it imports `pyDAS`,
and `pyDAS` is built only for Python 3.9
(`DAS-utilities/build/pyDAS.cpython-39-x86_64-linux-gnu.so`). In `fwi` these
modules fail with `ModuleNotFoundError: No module named 'pyDAS'`:

- already at import: `scripts.safod.prepare_event`,
  `scripts.safod.selection.screen_events` (imports `prepare_event`),
  `scripts.safod.plot_das_geophone_two_physical_comparisons`,
  `scripts.safod_deep.plot_catalog_events`, all
  `scripts.safod.experiments.oct03_2026.*`;
- at run time (import in `fwi` succeeds):
  `scripts.safod.calibrate_channel_registration` (`DASutils` when reading
  HDF5), `scripts.safod.selection.check_catalog_coverage` (`DAS_db.py`
  imports `DASutils`);
- the notebook `notebooks/NC75379261_DAS_geophone.ipynb`.

P/S travel-time curves in `scripts.safod_deep.plot_catalog_events` are computed
with EikoHypoTomo, which needs `pykonal` (only in `geo`).

In `geo` all modules of `src/`, `scripts/` (except `scripts/legacy/`) and
`web_safod/` import; in `fwi`, all except those listed above. The other
scripts do not need `pyDAS`; `fwi` is used for them. Importing does not mean
the script was run in that environment: for `scripts.safod_deep.scan_nano_manifest`,
`scripts.safod_deep.build_event_catalog`,
`scripts.safod.selection.find_catalog_events`,
`scripts.safod.experiments.diagnose_geometry` and
`scripts.safod.velocity_models.*` the environment of earlier runs is not
recorded anywhere.

### Notebooks and Jupyter kernels

| Notebook | `metadata.kernelspec` | `language_info.version` | Environment |
|---|---|---|---|
| `notebooks/final_demo.ipynb` | `name: python3`, `display_name: Python 3` | 3.11.13 | `fwi` |
| `notebooks/NC75379261_DAS_geophone.ipynb` | `name: python3`, `display_name: geo` | 3.9.23 | `geo` |

Jupyter selects the kernel by `name`, and `python3` is the built-in kernel of
the environment Jupyter itself was started from. So open `final_demo.ipynb`
in Jupyter from `fwi` (or with the `Python (fwi)` kernel), and
`NC75379261_DAS_geophone.ipynb` in Jupyter from `geo`
(`/home/groups/ettore88/alina/.envs/geo/bin/jupyter lab`). Both notebooks
find the project root themselves if opened from the root or from `notebooks/`.

User kernels are in `~/.local/share/jupyter/kernels/`; list them with
`jupyter kernelspec list`. As of 2026-10-16 the `geo` kernel there points to
the non-existent `~/miniconda3/envs/geo/bin/python`, and `geo_cuda/kernel.json`
cannot be read (invalid JSON); `fwi` points to
`/home/groups/ettore88/alina/.envs/fwi/bin/python`.

Jupyter is a server, so run it only in a job on a compute node
(section 5), not on a login node.

## 2. Local non-PyPI packages

Neither package is installed in any environment (`import DASutils`,
`import pyVector` fail outside the project). Scripts add their directories to
`sys.path` themselves, and the path is hard-coded in each script separately
(section 4).

| Package | What is used | How the code finds it |
|---|---|---|
| DAS-utilities, `/home/groups/ettore88/alina/packages/DAS-utilities` | `python/DASutils.py`, `python/DAS_db.py`; `build/pyDAS.cpython-39-x86_64-linux-gnu.so` (CMake build, Python 3.9 only, i.e. `geo`) | the script inserts `build/` and `python/` at the start of `sys.path` and adds `build/` to `LD_LIBRARY_PATH`; `DAS_db.py` is loaded by file path (`importlib.util.spec_from_file_location`): `settings.DAS_DB_PY` in `prepare_event` and `selection.screen_events`, its own constant in `selection.check_catalog_coverage`; for `calibrate_channel_registration` the root is set by `--utilities-root` (same path by default) |
| EikoHypoTomo, `/home/groups/ettore88/alina/packages/EikoHypoTomo` | `python/EikonalTomography2DMod.py` (imports `pykonal`), `external/python-solver/GenericSolver/python/pyVector.py` | only `scripts.safod_deep.plot_catalog_events`: `_load_eikonal_modules()` on the first eikonal computation adds both directories to `sys.path` and, if `DATAPATH` is not set, sets it to `$TMPDIR` (or `/tmp`); eikonal computation errors are caught (`phase-guide eikonal failed ...`) |

The packages' own dependencies are described in their `environment.yml`
(DAS-utilities also has `requirements.txt`).

## 3. Input data

| Input | Where | Who reads it |
|---|---|---|
| Per-event DAS records (HDF5; for the 2026-10-03 experiment, `.pb` files) | `/oak/stanford/groups/ettore88/data/SAFOD/SAFOD_events` | `scripts/safod/settings.py`: `SAFOD_EVENT_DATA_ROOT` and `EVENTS[<key>]["input_patterns"]` (keys `nc75336802`, `nc75379261` (default), `nc75336682`; selected by `SAFOD_EVENT_KEY`); own copies of the path in `scripts/safod/` `selection/check_catalog_coverage.py`, `selection/screen_events.py`, `experiments/oct03_2026/*`, `plot_das_geophone_two_physical_comparisons.py` |
| SAFOD-deep continuous archive | 8 recording roots in `config/safod_deep/roots.json` (all under `/oak/stanford/groups/ettore88/data/SAFOD/SAFOD-deep/`) | `scripts.safod_deep.build_event_catalog --roots-config` (required option); `scripts.safod_deep.scan_nano_manifest --root` (default root `2026-06-Nano`) |
| Table of georeferenced cable channels | `/home/groups/ettore88/alina/SAFOD/SAFOD_Phase2_GeoReferenced_Channels.xlsx` (outside the project and outside git) | `settings.GEO_XLSX` (`prepare_event`); own copies of the path in 6 scripts (section 4); `geometry_xlsx` in `web_safod/config/*.json`; `build_event_catalog --geometry` (required, no default) |
| Channel re-registration for event `nc75379261` | `results/events/20260618_75379261/real/channel_registration_qc/mapped_downleg_model_geometry.csv` | `settings.CHANNEL_MAPPING_CSV` (`prepare_event`); written by `scripts.safod.calibrate_channel_registration` |
| Velocity models | `data/safod/velocity_models/<model>/` | `src/safod/models/` (table below) |

Velocity models (paths inside `data/safod/velocity_models/`; the code reads
them by relative paths, so run only from the root):

| Model | `raw/` | What else is there | Who reads it | How it was produced |
|---|---|---|---|---|
| `zhang_thurber_bedrosian_2009/` | `MOD.head`, `inversion_grid.dat`, `Vp_model.dat`, `Vs_model.dat`, `Vpvs_model.dat` (provided by C. Thurber, see the model's `README.md`) | `processed/` (`.npz`, `.csv`), `qc/` (`.png`, `.txt`) | `src/safod/models/zhang2009.py` (`processed/zhang2009_safod_section_2d.npz`) | `scripts.safod.velocity_models.zhang2009.{parse_model, extract_safod_section, qc_registration}` |
| `boness_zoback_2006/` | empty | `processed/boness_zoback2006_fig3_4_vp_vs_digitized.csv`, `qc/boness_zoback2006_fig3_4_digitization_qc.png` | `src/safod/models/boness_zoback2006.py` | no digitization script in the project; ask the author |
| `ellsworth_malin_2011/` | none (no `raw/processed/qc`) | `fig3a_digitized.csv`, `README.md` | `src/safod/models/factory.py` (`DEFAULT_BILL_LOGS_CSV`) | `scripts.safod.velocity_models.digitize_ellsworth_malin_fig3a --image <figure>`; the source figure is not in the project |

What is not in git (`.gitignore`, `web_safod/.gitignore`):

- all of `data/safod/velocity_models/`: a clone of the repository has no
  velocity models; they are transferred separately;
- `results/*`, except `results/README.md`, including the channel
  re-registration CSV from the table above;
- everywhere: `*.npz`, `*.npy`, `*.gif`, `*.mp4`;
- `docs/history/`;
- in `web_safod/`: `config/paths.local.json`, `cache/*` and `site/*` (except
  their `README.md`), `reference/san_andreas_fault.geojson`.

`config/safod_deep/roots.json` is in git.

## 4. External paths in the code

Collected with this command (without `scripts/legacy/`; 64 matches):

```bash
grep -rnE '"/oak/|"/home/groups/|/oak/stanford|/home/groups/ettore88' \
  --include=*.py --include=*.sbatch --include=*.json --include=*.sh \
  scripts src web_safod/config web_safod/scripts config | grep -v '^scripts/legacy'
```

| Root | What | Where (file:line) |
|---|---|---|
| `/home/groups/ettore88/alina/packages/DAS-utilities` (and `build/`, `python/`, `python/DAS_db.py`) | DAS-utilities | `scripts/safod/settings.py:59`, `scripts/safod/prepare_event.py:23,24`, `scripts/safod/calibrate_channel_registration.py:42`, `scripts/safod/plot_das_geophone_two_physical_comparisons.py:34`, `scripts/safod/selection/check_catalog_coverage.py:19,20,37`, `scripts/safod_deep/plot_catalog_events.py:38`, `scripts/safod/experiments/oct03_2026/`: `compare_oct03_current_turnaround.py:54`, `compare_oct03_das_mh030_simple.py:58`, `diagnose_oct03_mh030_channels.py:75`, `find_current_mh030_das_channel.py:52`, `plot_oct03_das_geophone_like_scec.py:75`, `plot_oct03_m08_das_mh030_pb_v3.py:70`, `plot_oct03_m08_das_mh030_pb_v5.py:71` |
| `/home/groups/ettore88/alina/packages/EikoHypoTomo` | EikoHypoTomo | `scripts/safod_deep/plot_catalog_events.py:84` |
| `/home/groups/ettore88/alina/SAFOD/SAFOD_Phase2_GeoReferenced_Channels.xlsx` | channel table | `scripts/safod/settings.py:32`, `scripts/safod/plot_das_geophone_two_physical_comparisons.py:69`, `scripts/safod/selection/find_catalog_events.py:18`, `scripts/safod/experiments/diagnose_geometry.py:19`, `scripts/safod/experiments/oct03_2026/plot_oct03_m08_das_mh030_pb_v3.py:108`, `scripts/safod/experiments/oct03_2026/plot_oct03_m08_das_mh030_pb_v5.py:109`, `scripts/safod_deep/plot_catalog_events.py:57`, `web_safod/config/explorer.json:5`, `web_safod/config/paths.example.json:6`, `web_safod/config/paths.local.json:6` |
| `/oak/stanford/groups/ettore88/data/SAFOD/SAFOD_events` | event records | `scripts/safod/settings.py:27`, `scripts/safod/plot_das_geophone_two_physical_comparisons.py:64`, `scripts/safod/selection/check_catalog_coverage.py:42`, `scripts/safod/selection/screen_events.py:32`, `scripts/safod/experiments/oct03_2026/`: `compare_oct03_current_turnaround.py:81,85`, `compare_oct03_das_mh030_simple.py:87,91`, `diagnose_oct03_mh030_channels.py:106,110`, `find_current_mh030_das_channel.py:71,75`, `plot_oct03_das_geophone_like_scec.py:107,111`, `plot_oct03_m08_das_mh030_pb_v3.py:98,102`, `plot_oct03_m08_das_mh030_pb_v5.py:99,103` |
| `/oak/stanford/groups/ettore88/data/SAFOD` (`ActiveJune2026/**/*.h5`, `**/*.h5`) | extended HDF5 search | `scripts/safod/selection/check_catalog_coverage.py:43,44` |
| `/oak/stanford/groups/ettore88/data/SAFOD/SAFOD-deep` | SAFOD-deep archive | `config/safod_deep/roots.json:5,11,17,23,29,35,41,47`, `scripts/safod_deep/scan_nano_manifest.py:136` |
| `/home/groups/ettore88/alina/.envs/geo/bin/python`, `.../python3` | `geo` interpreter | `scripts/safod/settings.py:68`, `scripts/safod_deep/plot_catalog_events_array.sbatch:15` |
| `/home/groups/ettore88/alina/.envs/fwi/bin/python` | `fwi` interpreter | `web_safod/scripts/build_site.sbatch:13` |
| `/home/groups/ettore88/alina/classes/geophys245/Elastic_das_project` | project root (`cd` in the job) | `scripts/safod_deep/plot_catalog_events_array.sbatch:14`, `web_safod/scripts/build_site.sbatch:12` |
| `/home/groups/ettore88/alina/classes/geophys245/safod-event-explorer-pages` | GitHub Pages checkout (`pages_checkout`) | `web_safod/config/explorer.json:8` |
| `/home/groups/ettore88/alina/imaging/SAFOD_downleg_Projected_2D.csv` | old geometry | `scripts/safod/run_forward.py:1680` (comment), `:1686` (`deep_saf` branch; `source_mode` is fixed in the code as `"catalog_event"`) |

Outside this search: the cell code of `notebooks/NC75379261_DAS_geophone.ipynb`
contains the DAS-utilities and `SAFOD_events` paths.

Path constants in `scripts/safod/settings.py` that other modules do not
import: `DAS_UTILITIES_BUILD` (:62), `DESAMPLE_DAS_PY` (:65) and
`GEO_PYTHON` (:67) are not used anywhere; `DAS_UTILITIES_ROOT` (:58) and
`DAS_UTILITIES_PYTHON` (:63) are needed only inside `settings.py`, to
build `DAS_DB_PY`; `SAFOD_EVENT_DATA_ROOT` (:26) only for `EVENTS`.
Of the `settings.py` paths, other modules import only `DAS_DB_PY`
(`prepare_event`, `selection.screen_events`) and `GEO_XLSX` (`prepare_event`).
Collecting the paths in one config is a possible next step.

## 5. How to run

- From the project root, as a module: `python -m scripts.<...>`. Paths to
  `results/` and `data/` in the code are relative. Also, outside the root the
  `scripts` package in `fwi` resolves to devito: from `/tmp`
  `/home/groups/ettore88/alina/.envs/fwi/bin/python -c "import importlib.util; print(importlib.util.find_spec('scripts'))"`
  shows `/home/groups/ettore88/alina/packages/devito/scripts`.
- Only on a compute node: `sh_dev` (default partition `dev`, 1 core,
  4 GB, 1 h; options `-c`, `-m`, `-t`) or `sbatch` with an explicit `--time`.
  Partitions and limits: `sh_part`. On a login node, only editing and viewing.
- For short checks, limit threads:
  `export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.
  In `scripts/safod_deep/plot_catalog_events_array.sbatch` `OMP_NUM_THREADS`
  and `NUMBA_NUM_THREADS` are set to `$SLURM_CPUS_PER_TASK`.
- Importing `scripts/safod/settings.py` does not access Oak. It selects the
  event `SAFOD_EVENT_KEY` (default `nc75379261`); an unknown key raises
  `ValueError`. `prepare_event.main()` calls `get_selected_files()` before
  querying the catalog and reading data: missing HDF5 still raises
  `FileNotFoundError`. Independent model tests can run without `/oak`.
- `sbatch` only from the root: `#SBATCH --output/--error` in the project files
  are relative, Slurm resolves them from the submission directory and does not
  create the log directory:

  ```bash
  cd /home/groups/ettore88/alina/classes/geophys245/Elastic_das_project
  mkdir -p results/<area>/logs
  sbatch <path/to/file>.sbatch
  ```

  | File | Environment | Logs |
  |---|---|---|
  | `scripts/safod_deep/plot_catalog_events_array.sbatch` | `geo` | `results/safod_deep/logs/` |
  | `web_safod/scripts/build_site.sbatch` | `fwi` | `web_safod/cache/` |

  Both files `cd` into the absolute project root (section 4). Where to put
  logs of new jobs: [results/README.md](../results/README.md); when and with
  which parameters to run: [safod/workflow.md](safod/workflow.md) and
  [web_safod/README.md](../web_safod/README.md).

## 6. Tests

From the root, in `fwi`, on a compute node:

```bash
cd /home/groups/ettore88/alina/classes/geophys245/Elastic_das_project
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/groups/ettore88/alina/.envs/fwi/bin/python -m pytest -q
```

[`pytest.ini`](../pytest.ini): `testpaths = tests web_safod/tests`,
`pythonpath = .`, `norecursedirs` includes `results`, `data`, `notebooks`,
`web_safod/site`, `web_safod/cache`. The full suite takes a few seconds
(measured 2026-10-16).

| File | What it checks | Conditions |
|---|---|---|
| `tests/test_project_layout.py` | project text: no old module names and paths; `python -m` targets in `.md`, `.sbatch`, `.sh`, `.ipynb` resolve; relative links in `.md` are not broken and do not leave the project | does not execute project code |
| `tests/test_safod_smooth_prior_explicit_bounds.py` | explicit grid bounds in `src.safod.models.smooth_prior.build_smooth_prior_model` | synthetic geometry, `/oak` not needed |
| `tests/test_safod_settings.py` | importing settings without searching for data; strict file check when preparing an event | `/oak` not needed |
| `web_safod/tests/test_web_safod.py` | the `web_safod` builder ([web_safod/README.md](../web_safod/README.md)) | — |

Bare `pytest` works as long as `pytest.ini` is picked up (run from the root or
with paths inside the project). If the ini is not used (for example, `-c` with
another file), the root is not in `sys.path`, `scripts` resolves to devito and
collection fails with `No module named 'scripts.safod'`. Hence
`python -m pytest` from the root.

`web_safod` tests separately:

```bash
/home/groups/ettore88/alina/.envs/fwi/bin/python -m unittest web_safod.tests.test_web_safod
node --test web_safod/tests/test_app.cjs
```

There is no Node.js in `fwi`. To check whether `node` is in the shell:
`command -v node && node --version` (in the agent shell on 2026-10-16 it found
`/home/groups/ettore88/alina/.envs/cursor-node/bin/node`, v20.20.2; `geo`
has `bin/node`, v22.6.0). Sherlock modules: `ml spider nodejs` (as of
2026-10-16: 20.20.0, 24.13.0, 25.3.0; load with `ml devel nodejs/20.20.0`).
If the `ml` function is not available in the shell, use `bash -lc 'ml spider nodejs'`.
Interface checks are described in [web_safod/README.md](../web_safod/README.md).

## 7. `requirements.txt`

[`requirements.txt`](../requirements.txt) is a minimal list for `fwi`, not a
lock file and not a description of `geo`:

- `numpy`, `scipy`, `matplotlib`, `numba` (with lower version bounds):
  solvers and graphics; `jupyter`, `ipykernel`: notebooks;
- `pandas`, `h5py`, `obspy`, `pyproj`, `python-dateutil`, `openpyxl`,
  `Pillow`, `pytest`, without versions: SAFOD data (HDF5, FDSN, coordinates,
  Excel via `pandas.read_excel`), catalogs, `web_safod`, tests. The versions in
  `fwi` as of 2026-10-16 are recorded in a comment in the file;
- DAS-utilities and EikoHypoTomo are not installed from PyPI (section 2);
  `pykonal` for EikoHypoTomo is only in `geo`.

Differences from `fwi`: the `jupyter` metapackage is not in `fwi` (it has
`jupyterlab`, `notebook`, `ipykernel`); devito is in `fwi` but is not listed
in the file and is not imported by project code. `web_safod/requirements.txt`
is a subset for the site (`pandas`, `openpyxl`, `Pillow`).
