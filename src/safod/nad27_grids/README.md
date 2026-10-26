# Pinned NOAA horizontal datum grids

These two public-domain NOAA grids are distributed by OSGeo PROJ-data:

- https://cdn.proj.org/us_noaa_conus.tif — NAD27 to NAD83 (1986), EPSG:1241.
- https://cdn.proj.org/us_noaa_cshpgn.tif — NAD83 to California-South HARN;
  EPSG:1750 uses these parameters with its documented WGS84/HARN approximation.

License/source documentation: https://cdn.proj.org/us_noaa_README.txt
(the NAD27/NAD83 and NAD83/HARN sections state Public Domain).

Files retrieved 2026-10-25. SHA256 hashes are locked in `src/safod/nad27.py`.
The runtime uses the grids by absolute path and verifies their bytes before
constructing a transformer. Do not replace them to repair a system PROJ install.
A future intentional change requires a new transform ID and validation.

This pins a horizontal reference operation; it neither transforms vertical
heights nor establishes centimetre absolute accuracy for the SAFOD survey.
