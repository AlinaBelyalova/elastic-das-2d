# Experiments 2026-10-03 (nc75443276)

One-off diagnostics of NCEDC event 75443276 (2026-10-03, M0.8 per the v3/v5
headers): Sintela DAS and the borehole instrument SF.MH030.01.GP1. DAS data
are two Sintela protobuf files in `/oak/stanford/groups/ettore88/data/SAFOD/SAFOD_events/`:
`timeseriesScaler_2026-10-03_04.51.40_UTC_073863.pb` and
`timeseriesScaler_2026-10-03_04.52.40_UTC_073864.pb`.

The event is not registered in `scripts/safod/settings.py`: `EVENTS` contains
only `nc75336802`, `nc75379261` and `nc75336682`. The main `prepare_event`
reads HDF5, so the [SAFOD workflow](../../../../docs/safod/workflow.md) does
not apply to this event. The scripts import neither `scripts.safod.settings`
nor each other; data paths, DAS-utilities and parameters are set in each file.

## Running

- `geo` environment: all scripts import `DASutils` (requires `pyDAS`).
  Environments are described in [setup.md](../../../../docs/setup.md).
- From the project root, on a compute node: the data are on `/oak`.
- NCEDC access is required: origin time, MH030 waveforms, metadata.
- All scripts use argparse; for the full list of options use `--help`.

```bash
cd /home/groups/ettore88/alina/classes/geophys245/Elastic_das_project
GEO=/home/groups/ettore88/alina/.envs/geo/bin/python
$GEO -m scripts.safod.experiments.oct03_2026.<script> [options]
```

Examples from the file headers, converted to the `-m` form (the header of
`diagnose_oct03_mh030_channels.py` gives the run by file path):

```bash
$GEO -m scripts.safod.experiments.oct03_2026.plot_oct03_m08_das_mh030_pb_v3 --das-channel 1694
$GEO -m scripts.safod.experiments.oct03_2026.plot_oct03_m08_das_mh030_pb_v5 --das-channel 1694
$GEO -m scripts.safod.experiments.oct03_2026.diagnose_oct03_mh030_channels \
    --fmin 5 --fmax 40 --pmin 0.17 --pmax 0.34
```

## Scripts

Purpose is taken from the file headers; options from `--help`, defaults from
the code.

| Script | Purpose | Options (default) | Output in `compare/` |
| --- | --- | --- | --- |
| `find_current_mh030_das_channel.py` | Find the current Sintela DAS channels closest to MH030 by first-arrival time, without the old OptaSense channel map | `--ch-min` (1050), `--ch-max` (1450), `--output-dir` | `current_channel_timing/` |
| `diagnose_oct03_mh030_channels.py` | Compare MH030 GP1 with current Sintela channels 1276–1281, one panel per channel; the lag in the P window is computed for diagnostics and not applied | `--fmin` (5), `--fmax` (40), `--tmin`, `--tmax`, `--pmin`, `--pmax`, `--output-dir` | `mh030_channel_diagnostic/` |
| `compare_oct03_current_turnaround.py` | Comparison around the current Sintela turnaround (near channels 1278–1280) with the exact NCEDC time 04:52:31.320Z instead of the rounded 04:52:31.000Z from Ettore's notebook | `--channel` (1279), `--output-dir` | `current_turnaround_comparison/` |
| `compare_oct03_das_mh030_simple.py` | Simple DAS / MH030 comparison in the style of the earlier SCEC figure: two Y axes, no normalization, amplitude fitting or shift | `--channel` (1279), `--fmin` (5), `--fmax` (40), `--tmin`, `--tmax`, `--output-dir` | `das_mh030_two_physical_scales/` |
| `plot_oct03_das_geophone_like_scec.py` | Processing as close as possible to `scripts/safod/plot_das_geophone_two_physical_comparisons.py`: causal FIR, 201 taps at 500 Hz (group delay 0.200 s, same as the earlier 401 taps at 1000 Hz) | `--channel` (1279), `--fmin` (5), `--fmax` (40), `--fir-numtaps` (201), `--tmin`, `--tmax`, `--corr-tmin`, `--corr-tmax`, `--output-dir` | `das_geophone_like_scec/` |
| `plot_oct03_m08_das_mh030_pb_v3.py` | DAS / MH030 comparison with fallback MH029 metadata; 5–40 and 5–200 Hz bands; zero-phase filter | `--das-channel` (1694), `--output-dir` | `das_mh030_comparison_v3_zerophase/` |
| `plot_oct03_m08_das_mh030_pb_v5.py` | Same, with the `DASutils.bandpass2D_c(..., zerophase=False)` filter, as in Ettore's notebook | `--das-channel` (1694), `--output-dir` | `das_mh030_comparison_v5_causal/` |

## v3 and v5

- Filter. v3: 4th-order Butterworth via `scipy.signal.sosfiltfilt`
  (zero-phase, times stay tied to the origin time). v5:
  `DASutils.bandpass2D_c(..., zerophase=False)`, the same causal filter for
  DAS and the geophone after resampling to the common DAS time grid.
- Per the headers, everything else is the same: protobuf reading
  `readFile_protobuf(..., diff=False, system="Sintela")`, 5–40 and 5–200 Hz
  bands, strain / velocity and strain rate / acceleration pairs, channel 1694
  by default.
- Output: previously both wrote the same file names to
  `compare/das_mh030_comparison/`; now v3 writes to
  `das_mh030_comparison_v3_zerophase/`, v5 to `das_mh030_comparison_v5_causal/`.
- Which variant is final: check with the author.

## Discrepancies between scripts

- Default channel: v3 and v5 use 1694, the channel number of the earlier
  comparison with MH029 "Pod 1". Per the headers of
  `find_current_mh030_das_channel.py` and
  `compare_oct03_current_turnaround.py`, the current Sintela channel numbers
  (3236 channels, spacing about 2.5321 m) do not match the old numbers from
  `SAFOD_Phase2_GeoReferenced_Channels.xlsx`.
  `compare_oct03_current_turnaround.py`, `compare_oct03_das_mh030_simple.py`
  and `plot_oct03_das_geophone_like_scec.py` use current channel 1279 by
  default.
- Fallback origin time in case NCEDC is unavailable: 04:52:31.000Z in v3 and
  v5, 04:52:31.320Z in the other scripts.
- MH030 response metadata (StationXML) were, per the headers, unavailable at
  the time. All scripts except `find_current_mh030_das_channel.py` (it works
  with raw GP1 counts) use MH029 Pod 1 metadata instead, either directly or as
  a fallback.

## Results

All outputs go by default to
`results/events/20261003_75443276/compare/<folder from the table>/`. The name
`20261003_75443276` follows the `EVENT_RESULT_KEY` rule even though the event
is not registered. This folder does not exist in the project yet, and no
earlier outputs were found in the project. Result layout rules:
[results/README.md](../../../../results/README.md).
