# Public data model

Files that `builder/site_builder.py` writes into `site/data/` and that
`templates/app.js` reads. Every field below was checked against the builder
code and the build of 2026-10-15 (291 events, 291 with plots, 54 focal
mechanisms). How the site is built: [../README.md](../README.md).

| File | Content from | Required by `validate_site` |
| --- | --- | --- |
| `events.json` | `builder/catalog.py`, `builder/assets.py` | yes |
| `site_status.json` | `builder/site_builder.py` | yes |
| `site_config.json` | `builder/site_builder.py` | yes |
| `safod_geometry.geojson` | `builder/geometry.py` | yes |
| `focal_mechanisms.json` | copy of `results/safod_deep/catalog/focal_mechanisms.json` | no (checked if present) |
| `san_andreas_fault.geojson` | copy of `reference/san_andreas_fault.geojson` | no |

No public JSON file may contain raw HDF5 paths, Oak/Sherlock absolute paths,
credentials or API tokens. The validator only checks a fixed list of strings
([../README.md](../README.md#public-data-rule)); credentials and tokens are not
detected automatically.

## `events.json`

A list of event objects sorted by `origin_time_utc`, one per valid row of
`catalog_recorded.csv`. Every object has all the keys below; values that are
missing or not finite are `null`.

| Field | Type | Source and meaning |
| --- | --- | --- |
| `event_id` | string | Public ID. A numeric catalog ID gets the prefix `nc` (`75334592` -> `nc75334592`); other IDs are lower-cased. Before that, each run of characters outside `[A-Za-z0-9_.-]` becomes one `_`. |
| `origin_time_utc` | string | ISO 8601, UTC, suffix `Z`. |
| `latitude`, `longitude` | number | Catalog values, degrees. |
| `depth_km` | number | Catalog `depth_km`. |
| `magnitude` | number or null | Catalog `magnitude`. |
| `magnitude_type` | string or null | Catalog `magnitude_type`, e.g. `Md`. |
| `location_name` | string or null | Catalog `location_name`, otherwise `place`. |
| `min_3d_distance_to_cable_km` | number or null | Catalog column of the same name. |
| `source_crossline_m` | number or null | Catalog column of the same name. |
| `source_section_x_m` | number or null | Catalog column of the same name. |
| `out_of_plane_angle_deg` | number or null | Catalog column of the same name. |
| `geometry_2d_class` | string | Catalog value, or `unknown`. The current build has `good`, `borderline`, `poor`. |
| `recording_label` | string or null | `recording_label` of the first metadata JSON, otherwise catalog `primary_recording`. |
| `recording_config` | string or null | `registration.config_id` of the first metadata JSON, e.g. `n3200_dx2p041905_gl10p209524_first1800`. Not used by `app.js`. |
| `plot_available` | boolean | `true` if `plots` is not empty. |
| `plot_url` | string or null | `plots[0].url`. Kept for convenience; `app.js` reads `plots`. |
| `plots` | list | One entry per metadata JSON of the event (below). |
| `event_url` | string | `public_event_page_template` from the config with the public `event_id`. |

The geometry fields are computed upstream when the catalog is built
(`src/safod_deep/catalog.py`; steps in
[../../docs/safod/workflow.md](../../docs/safod/workflow.md)); the builder
only copies them.

The validator requires a non-empty unique `event_id`, finite `latitude`
(-90..90), `longitude` (-180..180) and `depth_km`, and, when
`plot_available` is true, a non-empty `plots` list whose relative URLs point
to existing files.

### `plots[]`

`plots` is a list because the SAFOD-deep QC step may produce more than one
plot per event. Metadata JSONs with `status` `ok` or `exists` come first;
within each group newer files come first. A metadata JSON whose source PNG is
not found gives a build warning and no plot. The current build has exactly one
plot per event.

| Field | Type | Meaning |
| --- | --- | --- |
| `label` | string | `recipe_name` plus the band, e.g. `near_5_30Hz · 5–30 Hz`. |
| `recipe_name` | string | `recipe.name` from the metadata, or `DAS event`. |
| `fmin_hz` | number or null | `recipe.fmin_hz`. |
| `fmax_hz` | number or null | `effective_fmax_hz`, otherwise `recipe.fmax_hz`. |
| `zero_phase` | boolean or null | `zero_phase` from the metadata. |
| `url` | string | Relative path of the WebP image, `plots/<file>.webp`. |
| `source_name` | string | File name of the source PNG. |
| `arrivals_url` | string or null | Relative path of the companion image with P/S arrivals; shown by the arrivals toggle, not as a separate plot. |
| `arrivals_source_name` | string or null | File name of the arrivals PNG. |

File names use the catalog `event_id` as is, so numeric IDs have no `nc`
prefix: `plots/<id>.webp` and `plots/<id>__arrivals.webp` when the event has
one metadata JSON, `plots/<id>__<token>.webp` and
`plots/<id>__<token>__arrivals.webp` when it has several (`<token>` is a
10-character hash). For example, event `nc75334592` uses
`plots/75334592.webp`.

The source PNG is looked up from the metadata `image_path` (absolute, or
relative to the project root), then by file name in `event_qc_dir`, then by
file name in the parent of the metadata folder. `arrivals_image_path` is
resolved the same way.

## `focal_mechanisms.json`

An object keyed by public `event_id`. Written by
`scripts/fetch_focal_mechanisms.py` from the preferred USGS ComCat
`focal-mechanism` product (highest `preferredWeight`, then latest
`updateTime`). The ComCat search covers 50 km around the well coordinates
hard-coded in the script and the time span of `events.json`; only events of
`events.json` are kept. The canonical copy is
`results/safod_deep/catalog/focal_mechanisms.json` (under `results/`, so not
in git); each build copies it into `site/data/`.

| Field | Type | Meaning |
| --- | --- | --- |
| `np1`, `np2` | object | Nodal planes `{strike, dip, rake}` in degrees. |
| `source` | string | ComCat product source, e.g. `nc`. |
| `code` | string | ComCat product code, e.g. `nc75336682_fm1`. |
| `review_status` | string or null | ComCat `review-status`, e.g. `automatic`. |
| `evaluation_status` | string or null | ComCat `evaluation-status`, e.g. `preliminary`. |
| `num_stations_used` | string or null | ComCat `num-stations-used`, stored as the string ComCat returns. |
| `event_url` | string | USGS event page, focal-mechanism tab. |

The validator requires every key to exist in `events.json` and checks
strike 0..360, dip 0..90 and rake -180..180 for both planes. `app.js` draws
the beach ball from `np1` and shows both planes as text.

## `site_status.json`

| Field | Meaning |
| --- | --- |
| `build_id` | Build time, `YYYYMMDDTHHMMSSZ`; also substituted into `index.html` for cache busting. |
| `generated_utc` | Build time, ISO 8601 UTC. |
| `event_time_start_utc`, `event_time_end_utc` | First and last `origin_time_utc` in `events.json`. |
| `recorded_events` | Number of events; must equal the length of `events.json`. |
| `plots_available` | Number of events with `plot_available`; must match `events.json`. |
| `focal_mechanisms_available` | Number of entries in the focal-mechanism file at build time. |
| `catalog_sha256`, `catalog_name` | SHA-256 and file name of the catalog CSV. |
| `elastic_das_git_commit` | `git rev-parse HEAD` of the project at build time, or `null`. It identifies the builder code only if `web_safod/` was committed at that moment. |
| `web_product_version` | Constant `2` in `site_builder.py`. |

## `site_config.json`

| Field | Meaning |
| --- | --- |
| `build_id` | Same as in `site_status.json`. |
| `map_center`, `map_default_zoom` | From the config; initial map view. |
| `fault_bbox` | From the config, `[west, south, east, north]` in degrees. |
| `fault_remote_query_url` | USGS ArcGIS GeoJSON query for that box; `app.js` uses it when `san_andreas_fault.geojson` is missing. |
| `web_product_version` | Constant `2`. |

## `safod_geometry.geojson`

A `FeatureCollection` named `SAFOD georeferenced fibre geometry`, built from
the workbook columns `Channel`, `Section`, `MD_m`, `TVD_m`, `Lat_WGS84` and
`Lon_WGS84`. Coordinates are `[lon, lat]` (WGS84).

| Feature | Geometry | Properties |
| --- | --- | --- |
| one per section `Surface Spool`, `Down-leg`, `Up-leg` (sections with fewer than 2 points are skipped) | `LineString` | `kind: fibre_section`, `section`, `name` |
| wellhead: `Down-leg` channel with the smallest `MD_m` | `Point` | `kind: wellhead`, `name`, `channel` |
| turnaround: `Down-leg` channel with the largest `MD_m` | `Point` | `kind: turnaround`, `name`, `channel`, `md_m`, `tvd_m` |

## `san_andreas_fault.geojson`

A copy of `reference/san_andreas_fault.geojson`: a `FeatureCollection` with
`name`, `source`, `source_query` and the USGS features with their original
properties ([../reference/README.md](../reference/README.md)). If the local
file is missing, the build removes the site copy and the browser queries
`fault_remote_query_url` instead.
