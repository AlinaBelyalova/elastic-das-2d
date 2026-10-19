# Build cache and Slurm logs

Everything here except this README is gitignored and never published.

| File | Written by | Content |
| --- | --- | --- |
| `event_inventory.json` | each build (`builder/assets.py`) | WebP render index: one entry per rendered image; the images themselves are in `site/plots/` |
| `build_report.json` | each build (`builder/site_builder.py`) | last build: build id, input and site paths (absolute), event and plot counts, warnings such as missing source images |
| `safod_web_<jobid>.out`, `safod_web_<jobid>.err` | `sbatch web_safod/scripts/build_site.sbatch` (submit from the project root) | Slurm logs of the build job |

`event_inventory.json` keys have the form `<catalog event_id>:<token>:base`
or `...:arrivals`. Each entry stores a signature of the source PNG (resolved
path, size, modification time), the product version, the maximum long edge
and the quality. An image is rendered again when any of these change, when
its WebP file in `site/plots/` is missing, or with `--overwrite-images`.

- Deleting `event_inventory.json` makes the next build render every WebP
  image again.
- The token and the signature include absolute paths, so moving the project
  or `results/safod_deep/` also causes a full re-render.
- Old entries are never pruned; the current file still holds entries of the
  previous product version (`web_product_version` 1, keys without the
  `:base`/`:arrivals` suffix).

`build_report.json` contains internal absolute paths. That is why it lives
here and not in `site/`.
