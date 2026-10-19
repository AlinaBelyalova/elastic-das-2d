# San Andreas Fault reference layer

The explorer uses the U.S. Geological Survey **Quaternary Fault and Fold
Database of the United States**, California feature layer:

- Program page: https://www.usgs.gov/programs/earthquake-hazards/faults (not verified: HTTP 403 to `curl` on 2026-10-16)
- ArcGIS layer: https://earthquake.usgs.gov/arcgis/rest/services/haz/Qfaults/MapServer/4

The layer reports `fault_name`, `section_name`, `fault_id`, `age`,
`slip_rate` and other attributes and supports GeoJSON queries.

## Local copy

```text
web_safod/reference/san_andreas_fault.geojson
```

The file is gitignored: it can be reproduced from USGS. It is created by

```bash
python -m web_safod.scripts.prepare_fault_geometry
```

and also by `python -m web_safod.scripts.update_event_explorer` when the file
is missing or with `--refresh-fault`. Both need network access; a failed
download in `update_event_explorer` is only a warning.

The downloader (`builder/geometry.py`):

1. queries the bounding box `fault_bbox` from `config/explorer.json`
   (`[west, south, east, north]`) on `fault_arcgis_layer`;
2. requests WGS84 (`EPSG:4326`) GeoJSON;
3. keeps only features whose `fault_name` contains `San Andreas`
   (case-insensitive) and fails if there are none;
4. preserves the returned USGS feature properties and records the query in
   `source_query`.

Each build copies the file to `site/data/san_andreas_fault.geojson`. If the
local file is absent, the build removes that copy and the browser runs the
same USGS query at run time. The local copy is preferred for
reproducibility.

Do not manually digitize or redraw the fault.
