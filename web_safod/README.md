# SAFOD Deep-DAS Interactive Event Explorer

Статический сайт для просмотра событий архива SAFOD-deep: карта, каталог и QC-рисунки DAS.
Собирается из `results/safod_deep/`; обычная команда — `update_event_explorer` (ниже), публикация — [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

Static public earthquake/DAS viewer for SAFOD. "SAFOD-deep" is the name of the
continuous deep-fibre DAS archive, not a selection of deep events.

The viewer is separated from the heavy processing pipeline:

```text
SAFOD-deep pipeline (catalog, event QC plots)            results/safod_deep/
                    |
           web_safod (builder)                            web_safod/site/
                    |
             GitHub Pages                                 public URL
```

Collaborators only need a browser. They do not download raw DAS data.

- Public URL: `https://alinabelyalova.github.io/safod-event-explorer/`
- Source: this folder, `web_safod/`, inside the project repository (git
  remote `origin`: `https://github.com/AlinaBelyalova/elastic-das-2d.git`).
- Generated site: the separate Pages repository
  `AlinaBelyalova/safod-event-explorer` (see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)).

## Layout

```text
web_safod/
  builder/           library: catalog.py, assets.py (WebP), geometry.py,
                     validation.py, site_builder.py, config.py
  scripts/           entry points run as modules (table below),
                     build_site.sbatch, publish_github_pages.sh
  templates/         index.html, app.js, style.css: edit the front end here
  config/            explorer.json (defaults), paths.example.json,
                     paths.local.json (gitignored)
  reference/         san_andreas_fault.geojson, local USGS cache (gitignored)
  tests/             test_web_safod.py (Python), test_app.cjs (Node)
  docs/              DATA_MODEL.md, DEPLOYMENT.md
  site/              generated public bundle (gitignored except README.md)
  cache/             WebP render index, build report, Slurm logs
                     (gitignored except README.md)
  requirements.txt   pandas, openpyxl, Pillow
```

`site/` is rebuilt from `templates/` and the inputs below; never edit it by
hand ([site/README.md](site/README.md)). Cache contents:
[cache/README.md](cache/README.md). Fault layer:
[reference/README.md](reference/README.md). Public JSON fields:
[docs/DATA_MODEL.md](docs/DATA_MODEL.md).

## Running

All commands run from the project root with the `fwi` environment's
`python` (environments and tests overall: [../docs/setup.md](../docs/setup.md)).
The required packages (`web_safod/requirements.txt`) are already in that
environment; no Node.js build step is needed.

Building converts PNGs to WebP, so run it in a job: `sbatch` below, or an
interactive `sh_dev` session. No `/oak` access is needed; the inputs are in
the project and in the geometry workbook.

## Inputs

Paths come from `config/explorer.json`; relative paths are resolved against
the project root.

| Input | Config key | Produced by |
| --- | --- | --- |
| `results/safod_deep/catalog/catalog_recorded.csv` | `catalog` | SAFOD-deep catalog step |
| `results/safod_deep/event_qc/*.png` | `event_qc_dir` | SAFOD-deep event QC plots |
| `results/safod_deep/event_qc/metadata/*.json` | `event_metadata_dir` | same step, one JSON per plot |
| `results/safod_deep/catalog/focal_mechanisms.json` (optional) | hard-coded in `builder/site_builder.py` | `fetch_focal_mechanisms` (below) |
| `/home/groups/ettore88/alina/SAFOD/SAFOD_Phase2_GeoReferenced_Channels.xlsx` | `geometry_xlsx` | external workbook |
| `web_safod/reference/san_andreas_fault.geojson` (optional) | `fault_geojson` | `prepare_fault_geometry` or `update_event_explorer` |

The upstream steps that create the catalog and the event QC plots are
described in [../docs/safod/workflow.md](../docs/safod/workflow.md).

The catalog must contain `event_id`, `origin_time_utc`, `latitude`,
`longitude` and `depth_km`. Rows without these values are dropped; duplicate
`event_id` values stop the build. Optional columns are copied when present
(field list: [docs/DATA_MODEL.md](docs/DATA_MODEL.md)). A metadata JSON is
attached to an event through its `event_id`; its `image_path` and
`arrivals_image_path` point to the PNGs.

## Scripts

| Entry point | What it does | Writes |
| --- | --- | --- |
| `python -m web_safod.scripts.update_event_explorer` | Normal command. Downloads the fault layer if the local file is missing (or with `--refresh-fault`), builds the site, validates it. `--publish` also publishes (see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)). | `site/`, `cache/`, fault file in `reference/` |
| `python -m web_safod.scripts.build_event_explorer` | Build only: no fault download, no validation. | `site/`, `cache/` |
| `python -m web_safod.scripts.validate_site` | Checks the bundle: required files, coordinates, plot files, focal-mechanism angles, private-path strings, counts in `site_status.json`. Exit code 2 on errors; `--site-dir` selects another bundle. | nothing |
| `python -m web_safod.scripts.export_safod_geometry` | Writes only the fibre GeoJSON from the workbook (`--output` to redirect). | `site/data/safod_geometry.geojson` |
| `python -m web_safod.scripts.prepare_fault_geometry` | Downloads the San Andreas Fault features from USGS (network). | `reference/san_andreas_fault.geojson` |
| `python -m web_safod.scripts.fetch_focal_mechanisms` | Reads the events of the built site (`site/data/events.json`) and fetches preferred focal mechanisms from USGS ComCat (network). Existing entries are kept unless `--refresh-existing`; entries of events that are no longer on the site are dropped. Default paths are relative to the current directory: run from the project root. | `results/safod_deep/catalog/focal_mechanisms.json` and a copy in `site/data/` |
| `sbatch web_safod/scripts/build_site.sbatch` | Runs `update_event_explorer` (without `--publish`) as a job: 1 h, 2 CPUs, 8 GB. Submit from the project root. | as `update_event_explorer`, plus Slurm logs `web_safod/cache/safod_web_<jobid>.out/.err` |
| `bash web_safod/scripts/publish_github_pages.sh <pages checkout>` | Validates, then `rsync --delete` of `web_safod/site/` into the Pages checkout, `git commit` and `git push origin main`. Publishes immediately. Never run it casually: see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md). | Pages checkout, GitHub |

Common options: `--config <file>` (all Python entry points except
`fetch_focal_mechanisms`; another `explorer.json`, a `paths.local.json` next
to it is merged on top) and `--overwrite-images` (`build_event_explorer` and
`update_event_explorer`: re-render all WebP files).

## Normal update

After the SAFOD-deep catalog and the event QC plots have been extended
([../docs/safod/workflow.md](../docs/safod/workflow.md)):

```bash
sbatch web_safod/scripts/build_site.sbatch
```

or, inside an `sh_dev` session:

```bash
python -m web_safod.scripts.update_event_explorer
```

A build:

1. copies `templates/` into `site/`, writes the build id into `index.html`
   and creates `.nojekyll`;
2. copies `focal_mechanisms.json` from `results/safod_deep/catalog/`
   (and removes the site copy if the source file is missing);
3. exports the fibre geometry from the workbook;
4. copies the local fault layer (or removes the site copy if the file is
   missing; the browser then queries USGS directly);
5. renders WebP images for new or changed source PNGs only;
6. deletes `site/plots/*.webp` files that no event references;
7. writes `events.json`, `site_status.json`, `site_config.json` and
   `cache/build_report.json`.

`update_event_explorer` then runs the validator and exits with code 2 on
errors. It does not print the build warnings (for example missing source
images); read them in `cache/build_report.json` (`build_event_explorer`
also prints the first 20).

New event IDs are picked up automatically; old images are not re-rendered
unless their source PNG or the image settings changed. The public URL stays
the same.

Focal mechanisms for new events: after the build, run
`python -m web_safod.scripts.fetch_focal_mechanisms`, then build again so that
`site_status.json` reports the new count.

## Configuration

`config/explorer.json` holds the defaults. `config/paths.local.json`
(gitignored), if present next to it, is merged on top. Today it has the same
content as `paths.example.json`, which repeats the defaults, so it changes
nothing.

| Key | Used for |
| --- | --- |
| `catalog`, `event_qc_dir`, `event_metadata_dir`, `geometry_xlsx` | inputs (table above) |
| `site_dir` | build, validation, update, `export_safod_geometry`. `publish_github_pages.sh` validates it but always rsyncs `web_safod/site/`; `fetch_focal_mechanisms` does not read it. |
| `fault_geojson`, `fault_bbox`, `fault_arcgis_layer` | local fault cache, its download, and the browser fallback query |
| `pages_checkout` | local clone of the Pages repository, used by `--publish` |
| `web_image_max_long_edge_px`, `web_image_quality` | WebP size and quality; changing them re-renders all images |
| `web_image_format` | not read by the code; the output is always WebP |
| `map_center`, `map_default_zoom` | initial map view (via `site_config.json`) |
| `public_event_page_template` | `event_url` of each event |

Paths fixed in the code (not in the config): `web_safod/templates/`,
`web_safod/cache/event_inventory.json`, `web_safod/cache/build_report.json`,
`results/safod_deep/catalog/focal_mechanisms.json`, the site directory in
`publish_github_pages.sh`, and the defaults of `fetch_focal_mechanisms`.

## Preview

Do not open `site/index.html` through `file://`: browsers block `fetch()` of
the JSON files. Serve the directory over HTTP instead.

- On Sherlock, only inside an `sh_dev` job, never on a login node:

  ```bash
  python -m http.server 8000 --bind 127.0.0.1 --directory web_safod/site
  ```

  Reach it from your laptop with your own SSH port forward to that compute
  node through a login node (laptop -> login node -> compute node). Stop the
  server (Ctrl-C) and the forward when you are done.
- Or copy `web_safod/site/` (about 890 MB, mostly WebP) to your laptop and run
  the same `http.server` command there.

## Public interface

- Leaflet map with OpenStreetMap tiles, the SAFOD fibre (sections, wellhead,
  turnaround) and the USGS San Andreas Fault;
- event markers coloured by depth, radius by magnitude, hover summary, popup;
- filters: search (event ID, location, recording, geometry class), depth,
  magnitude, hypocentre-cable distance, date range, 2-D geometry class,
  recording, "only events with plots", reset;
- focal-mechanism beach balls (toggle), distance measurement tool;
- DAS dialog: plot selector when an event has several plots, P/S-arrivals
  toggle, full-image link, USGS event-page link;
- coverage, build time, plot and focal-mechanism counts;
- direct links `#event=nc75379261` (also accepted: `#event=75379261`).

## Tests

```bash
python -m pytest web_safod/tests
```

Four Python tests (catalog loading, geometry export, WebP generation,
private-path rejection). `python -m pytest` without arguments also runs them
(`pytest.ini`). The same file runs with
`python -m unittest web_safod.tests.test_web_safod`.

Front-end tests need Node.js (`ml devel nodejs/20.20.0`):

```bash
node --test web_safod/tests/test_app.cjs
```

The Node tests cover direct event links, delayed optional layers, event
switching, filter reset, dialog closing and malformed URLs. The fetch and
Leaflet stubs include the optional focal-mechanism catalog and current marker
methods. A missing focal-mechanism catalog (HTTP 404) is also tested.
Plot URLs include the build identifier used to invalidate browser caches.

On 2026-10-19 all 13 Node tests passed after the test-stub updates and fixes
to direct-link navigation. These are automated logic tests, not a live-site
browser check with the complete SAFOD catalog.

## Public-data rule

`validate_site` fails if `events.json`, `site_status.json`,
`site_config.json`, `safod_geometry.geojson` or `focal_mechanisms.json`
contains any of these strings (case-insensitive, in keys and values):

```text
/home/
/oak/
/scratch/
file_path
.h5
.hdf5
```

Raw DAS paths and internal storage details must never reach the public site.
The validator does not scan `site/README.md` or `san_andreas_fault.geojson`;
`site/README.md` is published too, so keep it free of internal paths.

## Storage

- Generated products stay in `web_safod/site/` and `web_safod/cache/`.
  Exceptions: the fault layer in `web_safod/reference/` and the canonical
  `results/safod_deep/catalog/focal_mechanisms.json`.
- The raw DAS data and Ettore's SAFOD data tree are read-only inputs. Do not
  copy them into `web_safod`, and never write web products into the raw
  acquisition directories.
- A separate web archive is needed only if the products become genuinely
  large, for example under `$OAK/alina/web_safod/` after confirming that
  this is the right writable parent.
