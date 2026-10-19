# GitHub Pages deployment

| What | Where |
| --- | --- |
| Public URL | `https://alinabelyalova.github.io/safod-event-explorer/` (responded on 2026-10-16) |
| Pages repository | `AlinaBelyalova/safod-event-explorer`, branch `main`, remote `git@github.com:AlinaBelyalova/safod-event-explorer.git` |
| Local Pages checkout | config key `pages_checkout`, currently `/home/groups/ettore88/alina/classes/geophys245/safod-event-explorer-pages` |
| Source | `web_safod/` in this project (see [../README.md](../README.md)) |

The Pages repository only holds the generated bundle. Each publish adds one
commit "Update SAFOD event explorer".

## One-time setup

Already done on Sherlock: the checkout above exists. To recreate it, clone the
Pages repository outside the main project:

```bash
cd /home/groups/ettore88/alina/classes/geophys245
git clone git@github.com:AlinaBelyalova/safod-event-explorer.git \
  safod-event-explorer-pages
```

GitHub repository settings, as set up originally (not re-checked here):

```text
Settings -> Pages -> Build and deployment -> Deploy from a branch -> main -> / (root)
```

The generated site includes `.nojekyll`. If the checkout lives elsewhere, set
`pages_checkout` in `config/paths.local.json`.

## Before publishing

Publishing goes public at once and has no confirmation step. Never run it
casually, from a batch job, or as part of `build_site.sbatch` (that job does
not publish).

1. Build and validate (`update_event_explorer`, see [../README.md](../README.md#normal-update)).
2. Read the build warnings and look at the site in a preview.
3. Check that the Pages checkout has no local work: `git status` inside it.
   Any change there is overwritten or deleted.

## What the publish script does

`scripts/publish_github_pages.sh <pages checkout>` (or `PAGES_CHECKOUT=...`):

1. runs `python -m web_safod.scripts.validate_site` with `$PYTHON` (default
   `python`) on the config `site_dir` and stops on errors;
2. stops if the target has no `.git/`;
3. `rsync -av --delete --exclude='.git/' web_safod/site/ <checkout>/`.
   The source directory is fixed to `web_safod/site/`, not the config
   `site_dir`. Everything in the checkout that is not in `site/` is deleted,
   except `.git/`;
4. `git add -A`; if nothing changed it stops, otherwise
   `git commit -m "Update SAFOD event explorer"` and `git push origin main`.

The push goes over SSH to GitHub, so it needs your GitHub SSH key: run it
interactively, in the same `sh_dev` session as the build.

## Publish

Rebuild, validate and publish in one step (uses `pages_checkout` from the
config and the current Python):

```bash
python -m web_safod.scripts.update_event_explorer --publish
```

Publish the existing build without rebuilding:

```bash
PYTHON=/home/groups/ettore88/alina/.envs/fwi/bin/python \
bash web_safod/scripts/publish_github_pages.sh \
  /home/groups/ettore88/alina/classes/geophys245/safod-event-explorer-pages
```

Both run from the project root. A bad publish can be undone with git in the
Pages checkout, since every publish is a separate commit.

## What is public

Everything in `web_safod/site/`:

```text
index.html, app.js, style.css, .nojekyll
README.md                      becomes the README of the Pages repository
data/events.json               compact event list
data/site_status.json, data/site_config.json
data/safod_geometry.geojson    SAFOD fibre, wellhead, turnaround
data/san_andreas_fault.geojson USGS fault layer (if cached locally)
data/focal_mechanisms.json     ComCat nodal planes (if present)
plots/*.webp                   WebP copies of the DAS QC figures
```

Not published: `web_safod/cache/`, the config, the builder, the PNG
originals, raw HDF5 and any internal paths. Field definitions:
[DATA_MODEL.md](DATA_MODEL.md).

The public page also loads, at run time, Leaflet 1.9.4 from `unpkg.com`,
OpenStreetMap tiles, and the USGS ArcGIS fault service if the local fault
file is missing. It links to USGS event pages.
