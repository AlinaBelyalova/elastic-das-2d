# SAFOD geometry correction — 2026-10-24

This complete patch is based on main `472a3ec6c70b0d640d27f834e9962cce2ddcded6`.
It includes the earlier EPSG correction; apply this patch once, rather than
stacking it on the earlier CRS-only patch.

## What changes

- Event WGS84 coordinates are projected to EPSG:26710, matching NAD27 / UTM
  10N Easting/Northing stored in the supplied workbook.
- Physical wellhead coordinates/elevation are obtained before any shallow cut.
  The 10 m cut still selects distance-computation cable points, but cannot
  redefine the wellhead as channel 215 at TVD 10.244553 m.
- Catalog profile coordinates use the canonical WGS84 wellhead and complete
  down-leg percentile/direction calculation used by prepare_event.
- The NCEDC radial search centre defaults to that same canonical wellhead.
- Shared geodetic constants are event-independent and remain re-exported from
  settings.py. Modelling values are unchanged.
- Existing plot reuse, local gallery and site building reject stale geometry
  labels in JSON. Matching arrivals PNGs are indexed using their base metadata.
  PNGs must be rendered again: editing metadata alone cannot change their titles.

## Physical conventions retained

Use supplied workbook MD/TVD as GL depths; no 31 ft (9.4488 m) KB correction.
The email exchange supports this working interpretation, but the original
survey-to-workbook TVD conversion cannot be reconstructed from these files alone.

Catalog elevation remains 660.5 m MSL, as in the existing 291-event catalog.
The modelling surface remains 660.46 m. The 4 cm difference is not silently
introduced into a catalog rebuild. Source MSL elevation is -1000*depth_km;
cable elevation is supplied wellhead elevation minus TVD.

Workbook receiver coordinates and channel registration are untouched.
Zhang 3-D / original Lin 3-D coverage fallback, EikoHypoTomo solver, saved
forward waveforms, shared real/synthetic P/S curves and compare axes are unchanged.
There is no reason to rerun prepare_event or the elastic forward for this patch.

## Apply and test (project root)

```bash
git apply --check safod_geometry_full_fix.patch &&
git apply safod_geometry_full_fix.patch &&
/home/groups/ettore88/alina/.envs/fwi/bin/python -m pytest -q &&
node --test web_safod/tests/test_app.cjs &&
git diff --check
```

If the check fails, stop and inspect the reported files. Do not force application
or overwrite uncommitted changes. The earlier CRS-only patch is already included.

## Rebuild existing catalog and plots on Sherlock

Run on a compute node, from the project root. Reuse the original manifest and
queried events: do not rescan the archive or change the event population.
The explicit checks prevent the existing --reuse options silently falling back
to a scan/query when their files are missing.

```bash
FWI=/home/groups/ettore88/alina/.envs/fwi/bin/python
GEO=/home/groups/ettore88/alina/.envs/geo/bin/python

test -s results/safod_deep/catalog/recording_manifest.csv &&
test -s results/safod_deep/catalog/ncedc_events_raw.csv &&
"$FWI" -m scripts.safod_deep.build_event_catalog \
  --roots-config config/safod_deep/roots.json \
  --geometry /home/groups/ettore88/alina/SAFOD/SAFOD_Phase2_GeoReferenced_Channels.xlsx \
  --wellhead-elevation-m 660.5 \
  --reuse-manifest --reuse-events
```

Rerender into the existing canonical directory; no eikonal_3d variant directories
are created. The existing array script already uses --overwrite and the default
3-D arrival mode. Submit only after the catalog command succeeds:

```bash
mkdir -p results/safod_deep/logs
sbatch --wait scripts/safod_deep/plot_catalog_events_array.sbatch
```

Wait for a successful exit of the entire array before building the gallery/site.
If any shard fails, inspect its logs/status CSV and rerun that shard first.
The 3-D travel-time disk cache can be reused; this correction affects catalog
metrics/profile diagnostics, not source/receiver inputs to the 3-D solver.

```bash
"$GEO" -m scripts.safod_deep.plot_catalog_events --build-index-only &&
mkdir -p web_safod/cache &&
sbatch --wait web_safod/scripts/build_site.sbatch
```

Check completeness before publishing (the paths below are the current canonical
ones; retain your configured overrides if different):

```bash
"$FWI" - <<'PY'
import json
from pathlib import Path
import pandas as pd
from src.safod_deep.plot_provenance import require_current_plot_labels

catalog = pd.read_csv('results/safod_deep/catalog/catalog_recorded.csv')
lookup = {str(r['event_id']).lower().removeprefix('nc'): r for _, r in catalog.iterrows()}
metadata_dir = Path('results/safod_deep/event_qc_3d_eikonal/metadata')
matched = set()
for path in metadata_dir.glob('*.json'):
    record = json.loads(path.read_text())
    key = str(record.get('event_id', '')).lower().removeprefix('nc')
    if key not in lookup:
        continue
    require_current_plot_labels(record, lookup[key])
    assert record.get('arrivals_available'), path
    assert not record.get('travel_time_guide_error'), path
    matched.add(key)
assert matched == set(lookup), f'Missing metadata: {set(lookup) - matched}'
site = Path('web_safod/site')
events = json.loads((site / 'data/events.json').read_text())
assert len(events) == len(catalog), (len(events), len(catalog))
assert {str(e['event_id']).lower().removeprefix('nc') for e in events} == set(lookup)
assert all(e.get('plots') and all(p.get('arrivals_url') for p in e['plots']) for e in events)
report = json.loads(Path('web_safod/cache/build_report.json').read_text())
assert not report.get('warnings'), report.get('warnings')
print(f'Ready: {len(events)} events, current geometry labels, all with P/S')
PY
```

After successful checks, use the existing publication command:

```bash
PAGES=/home/groups/ettore88/alina/classes/geophys245/safod-event-explorer-pages
test -z "$(git -C "$PAGES" status --porcelain)" &&
PYTHON="$FWI" bash web_safod/scripts/publish_github_pages.sh "$PAGES"
```

The publication script validates the site, commits the Pages checkout and pushes.
This patch itself does not publish anything.

## Validation and limits

Local verification: 72 Python tests and 13 Node tests passed. New regressions
cover source-on-cable controls, explicit alternative UTM datum, shallow filtering,
canonical profile formula, query defaults, and stale PNG reuse/gallery/site guards.

Independently checked the actual 3201-row workbook (SHA256
4ae1dd2bb47840ce895c511411ac1cd54528002eaeedb897a1dc9369fab98a5b)
and all 291 public event coordinates. Retained UTM/TVD arrays are identical
to the input subset; wellhead elevation is 660.5 m; the first retained channel
is 215. Full down-leg direction EN is approximately (0.61337375, 0.78979278),
matching prepare_event's formula. No 2-D classes changed among these events.

For NC75379261, catalog minimum distance changes from 1589.850641 m to
1570.197269 m. Independently transforming the entire cable to WGS84 / UTM
instead gives 1570.197437 m. Across 291 events the two consistent-CRS routes
differ by at most 0.109324 m (projection scale/numerical datum-operation effects).
These are consistency checks, not sub-metre absolute survey accuracy claims.

Locally PROJ selected NAD27 to WGS84 (6), with stated accuracy 7 m; the best
regional grid was unavailable. Actual operation/accuracy can differ on Sherlock.
No new transformation grid was installed and no artificial translation was added.
The small WGS84/UTM mismatch in the original workbook remains as supplied.

Raw Sherlock DAS, saved event/forward NPZs and a full 291-event rerender were
not available locally. The above rebuild and completeness checks are required
there before publication. The original Excel is never modified by this patch.
