# ==============================================================================
# src/safod_deep/catalog.py
#
# SAFOD deep-DAS recording inventory + NCEDC event catalog + 2-D suitability
# diagnostics.
#
# This module deliberately separates four concerns:
#   1. HDF5 header/schema discovery and exact recording coverage;
#   2. NCEDC FDSN catalog retrieval;
#   3. 3-D cable/source geometry and projection into the SAME vertical
#      profile used by scripts.safod.prepare_event;
#   4. catalog assembly and overview figures.
#
# It does not load full DAS files. HDF5 scanning reads only metadata and dataset
# shapes, so it is safe to run over months of data.
# ==============================================================================

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import StringIO
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer


UTC = timezone.utc
NCEDC_EVENT_URL = "https://service.ncedc.org/fdsnws/event/1/query"


# ==============================================================================
# 1. BASIC HELPERS
# ==============================================================================

def _normalise_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _decode_scalar(value: Any) -> Any:
    """Convert common HDF5 scalar/byte/NumPy values to ordinary Python values."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _decode_scalar(value.reshape(-1)[0])
        if value.dtype.kind in {"S", "U", "O"}:
            return [_decode_scalar(v) for v in value.reshape(-1)]
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    return value


def _parse_utc(value: Any) -> pd.Timestamp | None:
    """Parse ISO strings and common Unix epoch units into a UTC Timestamp."""
    value = _decode_scalar(value)

    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        for item in value:
            parsed = _parse_utc(item)
            if parsed is not None:
                return parsed
        return None

    if isinstance(value, (int, float)) and np.isfinite(value):
        numeric = float(value)
        magnitude = abs(numeric)

        try:
            if magnitude >= 1.0e17:
                return pd.to_datetime(int(numeric), unit="ns", utc=True)
            if magnitude >= 1.0e14:
                return pd.to_datetime(int(numeric), unit="us", utc=True)
            if magnitude >= 1.0e11:
                return pd.to_datetime(int(numeric), unit="ms", utc=True)
            if magnitude >= 1.0e8:
                return pd.to_datetime(numeric, unit="s", utc=True)
        except Exception:
            return None

    text = str(value).strip()
    if not text:
        return None

    # Numeric epoch written as a string.
    try:
        numeric = float(text)
        parsed = _parse_utc(numeric)
        if parsed is not None:
            return parsed
    except ValueError:
        pass

    try:
        parsed = pd.to_datetime(text, utc=True, errors="raise")
        if isinstance(parsed, pd.DatetimeIndex):
            return parsed[0] if len(parsed) else None
        return pd.Timestamp(parsed)
    except Exception:
        return None


_TIMESTAMP_PATTERN = re.compile(
    r"(?P<date>20\d{2}-\d{2}-\d{2})[T_ -]?"
    r"(?P<hour>\d{2})[:\-]?(?P<minute>\d{2})[:\-]?(?P<second>\d{2})"
    r"(?:\.(?P<fraction>\d+))?Z?"
)


def _time_from_path(path: Path) -> pd.Timestamp | None:
    """
    Find an ISO-like timestamp in the FILE NAME only.

    We intentionally do not use a timestamp embedded only in a parent directory:
    otherwise every file in a long continuous recording could be assigned the
    directory-start time, silently corrupting coverage.
    """
    matches = list(_TIMESTAMP_PATTERN.finditer(path.name))
    if not matches:
        return None

    match = matches[-1]
    fraction = match.group("fraction") or ""
    fraction = (fraction + "000000")[:6]

    text = (
        f"{match.group('date')}T{match.group('hour')}:"
        f"{match.group('minute')}:{match.group('second')}."
        f"{fraction}Z"
    )
    return pd.Timestamp(text)


def _first_numeric(
    metadata: dict[str, Any],
    aliases: Iterable[str],
) -> float | None:
    aliases_norm = {_normalise_key(alias) for alias in aliases}

    # Exact normalized-key match first.
    for key, value in metadata.items():
        if _normalise_key(key.split("/")[-1]) not in aliases_norm:
            continue
        decoded = _decode_scalar(value)
        candidates = decoded if isinstance(decoded, list) else [decoded]
        for candidate in candidates:
            try:
                number = float(candidate)
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                return number

    # Then permit suffix/substring matches.
    for key, value in metadata.items():
        key_norm = _normalise_key(key)
        if not any(alias in key_norm for alias in aliases_norm):
            continue
        try:
            number = float(_decode_scalar(value))
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            return number

    return None


def _first_time(
    metadata: dict[str, Any],
    aliases: Iterable[str],
) -> pd.Timestamp | None:
    aliases_norm = {_normalise_key(alias) for alias in aliases}

    for key, value in metadata.items():
        key_leaf = _normalise_key(key.split("/")[-1])
        if key_leaf not in aliases_norm:
            continue
        parsed = _parse_utc(value)
        if parsed is not None:
            return parsed

    for key, value in metadata.items():
        key_norm = _normalise_key(key)
        if not any(alias in key_norm for alias in aliases_norm):
            continue
        parsed = _parse_utc(value)
        if parsed is not None:
            return parsed

    return None


# ==============================================================================
# 2. RECORDING ROOT / HDF5 INVENTORY
# ==============================================================================

@dataclass(frozen=True)
class RecordingRoot:
    label: str
    path: str
    priority: int
    notes: str = ""


@dataclass(frozen=True)
class RecordingFile:
    recording_label: str
    recording_priority: int
    recording_notes: str
    file_path: str
    dataset_path: str | None
    dataset_shape: str | None
    dtype: str | None
    channel_axis: int | None
    time_axis: int | None
    n_channels: int | None
    n_samples: int | None
    first_channel_index: int | None
    sample_rate_hz: float | None
    dt_s: float | None
    channel_spacing_m: float | None
    gauge_length_m: float | None
    start_time_utc: str | None
    end_time_utc: str | None
    duration_s: float | None
    start_time_source: str | None
    error: str | None


class H5RecordingScanner:
    """
    Header-only scanner for heterogeneous OptaSense/QuantX HDF5 archives.

    It searches all attributes, chooses the most plausible large 2-D numeric
    dataset, determines channel/time axes, and extracts timing/acquisition
    metadata using several common vendor naming conventions.
    """

    H5_SUFFIXES = {".h5", ".hdf5", ".hdf"}

    DATA_NAME_HINTS = (
        "rawdata",
        "raw_data",
        "strainrate",
        "strain_rate",
        "phase",
        "data",
    )

    START_TIME_ALIASES = (
        "partstarttime",
        "starttime",
        "start_time",
        "filestarttime",
        "file_start_time",
        "acquisitionstarttime",
        "acquisition_start_time",
        "gpsstarttime",
        "timestamp",
    )

    FS_ALIASES = (
        "outputdatarate",
        "samplingfrequency",
        "samplingrate",
        "samplerate",
        "sample_rate",
        "fs",
    )

    DT_ALIASES = (
        "samplinginterval",
        "sampleinterval",
        "sample_interval",
        "timeincrement",
        "dt",
    )

    DCH_ALIASES = (
        "spatialsamplinginterval",
        "spatialsampling",
        "channelspacing",
        "channel_spacing",
        "dch",
    )

    GL_ALIASES = (
        "gaugelength",
        "gauge_length",
        "gl",
    )

    NCHAN_ALIASES = (
        "numberofloci",
        "numberofchannels",
        "numchannels",
        "nchannels",
        "nchan",
    )

    FIRST_CHANNEL_ALIASES = (
        "startlocusindex",
        "firstlocusindex",
        "firstchannel",
        "startchannel",
        "channelstart",
        "locusindexstart",
    )

    def __init__(self, *, min_dataset_elements: int = 1000) -> None:
        self.min_dataset_elements = int(min_dataset_elements)

    @staticmethod
    def _collect_metadata(h5: h5py.File) -> dict[str, Any]:
        metadata: dict[str, Any] = {}

        def add_attrs(path: str, obj: h5py.Group | h5py.Dataset) -> None:
            for key, value in obj.attrs.items():
                metadata[f"{path}/@{key}"] = _decode_scalar(value)

        add_attrs("", h5)

        def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            add_attrs(name, obj)

        h5.visititems(visitor)
        return metadata

    def _choose_dataset(
        self,
        h5: h5py.File,
    ) -> tuple[str, h5py.Dataset] | tuple[None, None]:
        candidates: list[tuple[float, str, h5py.Dataset]] = []

        def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            if obj.ndim != 2:
                return
            if obj.dtype.kind not in {"i", "u", "f", "c"}:
                return

            elements = int(np.prod(obj.shape))
            if elements < self.min_dataset_elements:
                return

            name_norm = _normalise_key(name)
            hint_score = sum(
                1.0
                for hint in self.DATA_NAME_HINTS
                if _normalise_key(hint) in name_norm
            )
            # Size dominates; naming hints break ties.
            score = math.log10(max(elements, 1)) + 2.0 * hint_score
            candidates.append((score, name, obj))

        h5.visititems(visitor)

        if not candidates:
            return None, None

        _, name, dataset = max(candidates, key=lambda item: item[0])
        return name, dataset

    @staticmethod
    def _choose_axes(
        shape: tuple[int, int],
        nchan_meta: float | None,
    ) -> tuple[int, int]:
        if nchan_meta is not None:
            target = int(round(nchan_meta))
            errors = [abs(shape[0] - target), abs(shape[1] - target)]
            channel_axis = int(np.argmin(errors))
            if errors[channel_axis] <= max(4, int(0.02 * max(target, 1))):
                return channel_axis, 1 - channel_axis

        # A DAS file normally has fewer channels than time samples.
        if shape[0] != shape[1]:
            channel_axis = int(np.argmin(shape))
            return channel_axis, 1 - channel_axis

        # Ambiguous square dataset: default to channels x time.
        return 0, 1

    def scan_file(
        self,
        path: Path,
        root: RecordingRoot,
    ) -> RecordingFile:
        try:
            with h5py.File(path, "r") as h5:
                metadata = self._collect_metadata(h5)
                dataset_path, dataset = self._choose_dataset(h5)

                if dataset is None:
                    raise RuntimeError(
                        "No sufficiently large 2-D numeric dataset found."
                    )

                nchan_meta = _first_numeric(metadata, self.NCHAN_ALIASES)
                channel_axis, time_axis = self._choose_axes(
                    tuple(int(v) for v in dataset.shape),
                    nchan_meta,
                )

                n_channels = int(dataset.shape[channel_axis])
                n_samples = int(dataset.shape[time_axis])

                fs = _first_numeric(metadata, self.FS_ALIASES)
                dt = _first_numeric(metadata, self.DT_ALIASES)

                if fs is None and dt is not None and dt > 0.0:
                    fs = 1.0 / dt
                if dt is None and fs is not None and fs > 0.0:
                    dt = 1.0 / fs

                dch = _first_numeric(metadata, self.DCH_ALIASES)
                gl = _first_numeric(metadata, self.GL_ALIASES)
                first_channel = _first_numeric(
                    metadata,
                    self.FIRST_CHANNEL_ALIASES,
                )

                start = _first_time(metadata, self.START_TIME_ALIASES)
                start_source = "hdf5_attribute"
                if start is None:
                    start = _time_from_path(path)
                    start_source = "path_timestamp" if start is not None else None

                duration = (
                    float(n_samples / fs)
                    if fs is not None and fs > 0.0
                    else None
                )
                end = (
                    start + pd.to_timedelta(duration, unit="s")
                    if start is not None and duration is not None
                    else None
                )

                return RecordingFile(
                    recording_label=root.label,
                    recording_priority=int(root.priority),
                    recording_notes=root.notes,
                    file_path=str(path),
                    dataset_path=str(dataset_path),
                    dataset_shape=str(tuple(int(v) for v in dataset.shape)),
                    dtype=str(dataset.dtype),
                    channel_axis=int(channel_axis),
                    time_axis=int(time_axis),
                    n_channels=n_channels,
                    n_samples=n_samples,
                    first_channel_index=(
                        int(round(first_channel))
                        if first_channel is not None
                        else None
                    ),
                    sample_rate_hz=float(fs) if fs is not None else None,
                    dt_s=float(dt) if dt is not None else None,
                    channel_spacing_m=float(dch) if dch is not None else None,
                    gauge_length_m=float(gl) if gl is not None else None,
                    start_time_utc=(
                        start.isoformat().replace("+00:00", "Z")
                        if start is not None
                        else None
                    ),
                    end_time_utc=(
                        end.isoformat().replace("+00:00", "Z")
                        if end is not None
                        else None
                    ),
                    duration_s=duration,
                    start_time_source=start_source,
                    error=None,
                )

        except Exception as exc:
            return RecordingFile(
                recording_label=root.label,
                recording_priority=int(root.priority),
                recording_notes=root.notes,
                file_path=str(path),
                dataset_path=None,
                dataset_shape=None,
                dtype=None,
                channel_axis=None,
                time_axis=None,
                n_channels=None,
                n_samples=None,
                first_channel_index=None,
                sample_rate_hz=None,
                dt_s=None,
                channel_spacing_m=None,
                gauge_length_m=None,
                start_time_utc=None,
                end_time_utc=None,
                duration_s=None,
                start_time_source=None,
                error=f"{type(exc).__name__}: {exc}",
            )

    def scan_roots(
        self,
        roots: Iterable[RecordingRoot],
    ) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        seen: set[str] = set()

        for root in roots:
            root_path = Path(root.path)
            if not root_path.exists():
                records.append(
                    asdict(
                        RecordingFile(
                            recording_label=root.label,
                            recording_priority=root.priority,
                            recording_notes=root.notes,
                            file_path=str(root_path),
                            dataset_path=None,
                            dataset_shape=None,
                            dtype=None,
                            channel_axis=None,
                            time_axis=None,
                            n_channels=None,
                            n_samples=None,
                            first_channel_index=None,
                            sample_rate_hz=None,
                            dt_s=None,
                            channel_spacing_m=None,
                            gauge_length_m=None,
                            start_time_utc=None,
                            end_time_utc=None,
                            duration_s=None,
                            start_time_source=None,
                            error="Root directory does not exist.",
                        )
                    )
                )
                continue

            files = sorted(
                path
                for path in root_path.rglob("*")
                if path.is_file() and path.suffix.lower() in self.H5_SUFFIXES
            )

            for path in files:
                resolved = str(path.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                records.append(asdict(self.scan_file(path, root)))

        frame = pd.DataFrame.from_records(records)
        if frame.empty:
            return frame

        frame["start_time_utc"] = pd.to_datetime(
            frame["start_time_utc"],
            utc=True,
            errors="coerce",
        )
        frame["end_time_utc"] = pd.to_datetime(
            frame["end_time_utc"],
            utc=True,
            errors="coerce",
        )

        return frame.sort_values(
            ["start_time_utc", "recording_priority", "file_path"],
            na_position="last",
        ).reset_index(drop=True)


# ==============================================================================
# 3. NCEDC FDSN EVENT CLIENT
# ==============================================================================

class NCEDCCatalogClient:
    """
    Minimal NCEDC FDSN event client using the official pipe-delimited text
    response. Results are paginated to avoid the service's default limit.
    """

    def __init__(
        self,
        *,
        endpoint: str = NCEDC_EVENT_URL,
        timeout_s: float = 120.0,
        user_agent: str = "SAFOD-deep-catalog/1.0",
    ) -> None:
        self.endpoint = endpoint
        self.timeout_s = float(timeout_s)
        self.user_agent = str(user_agent)

    def _request_page(
        self,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        latitude: float,
        longitude: float,
        max_radius_km: float,
        min_magnitude: float | None,
        limit: int,
        offset: int,
    ) -> pd.DataFrame:
        max_radius_deg = float(max_radius_km) / 111.195

        params: dict[str, Any] = {
            "format": "text",
            "start": start.strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "end": end.strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "lat": float(latitude),
            "lon": float(longitude),
            "minradius": 0.0,
            "maxradius": max_radius_deg,
            "orderby": "time-asc",
            "limit": int(limit),
            "offset": int(offset),
            "nodata": 204,
        }

        if min_magnitude is not None:
            params["minmag"] = float(min_magnitude)

        url = f"{self.endpoint}?{urlencode(params)}"
        request = Request(
            url,
            headers={"User-Agent": self.user_agent},
        )

        with urlopen(request, timeout=self.timeout_s) as response:
            status = getattr(response, "status", 200)
            if status == 204:
                return pd.DataFrame()
            # utf-8-sig also removes a possible BOM before #EventID.
            text = response.read().decode("utf-8-sig", errors="replace")

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]
        if not lines:
            return pd.DataFrame()

        # NCEDC documents the text header with spaces around the pipe
        # separators, e.g. ``#EventID | Time | Latitude | ...``.  Reading that
        # line directly with pandas produces columns such as ``"EventID "``
        # and ``" Time "``.  Parse and strip every header field explicitly.
        header_fields = [
            field.replace("\ufeff", "").lstrip("#").strip()
            for field in lines[0].split("|")
        ]

        if not header_fields or any(not field for field in header_fields):
            raise RuntimeError(
                "Could not parse NCEDC text-response header: "
                f"{lines[0]!r}"
            )

        if len(set(header_fields)) != len(header_fields):
            raise RuntimeError(
                "NCEDC text-response header contains duplicate fields after "
                f"whitespace normalization: {header_fields}"
            )

        if len(lines) == 1:
            return pd.DataFrame(columns=header_fields)

        payload = "\n".join(lines[1:])
        frame = pd.read_csv(
            StringIO(payload),
            sep="|",
            names=header_fields,
            header=None,
            dtype=str,
            keep_default_na=False,
            on_bad_lines="error",
        )

        # Strip padding from every text field but preserve empty strings.
        for column in frame.columns:
            frame[column] = frame[column].map(
                lambda value: value.strip() if isinstance(value, str) else value
            )

        return frame

    @staticmethod
    def _standardise(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame

        frame = frame.copy()
        frame.columns = [
            str(column).replace("\ufeff", "").lstrip("#").strip()
            for column in frame.columns
        ]

        canonical_names = {
            _normalise_key("EventID"): "event_id",
            _normalise_key("Time"): "origin_time_utc",
            _normalise_key("Latitude"): "latitude",
            _normalise_key("Longitude"): "longitude",
            _normalise_key("Depth/km"): "depth_km",
            _normalise_key("Author"): "origin_author",
            _normalise_key("Catalog"): "catalog",
            _normalise_key("Contributor"): "contributor",
            _normalise_key("ContributorID"): "contributor_id",
            _normalise_key("MagType"): "magnitude_type",
            _normalise_key("Magnitude"): "magnitude",
            _normalise_key("MagAuthor"): "magnitude_author",
            _normalise_key("EventLocationName"): "location_name",
            _normalise_key("EventType"): "event_type",
        }

        frame = frame.rename(
            columns={
                column: canonical_names.get(_normalise_key(column), column)
                for column in frame.columns
            }
        )

        required = {
            "event_id",
            "origin_time_utc",
            "latitude",
            "longitude",
            "depth_km",
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise RuntimeError(
                "NCEDC text response is missing required fields after header "
                f"normalization: {missing}. Parsed columns: "
                f"{list(frame.columns)}"
            )

        for column in ("latitude", "longitude", "depth_km", "magnitude"):
            if column in frame:
                frame[column] = pd.to_numeric(
                    frame[column],
                    errors="coerce",
                )

        frame["origin_time_utc"] = pd.to_datetime(
            frame["origin_time_utc"],
            utc=True,
            errors="coerce",
        )

        invalid_time = frame["origin_time_utc"].isna()
        if np.any(invalid_time):
            bad_rows = frame.loc[
                invalid_time,
                ["event_id", "origin_time_utc"],
            ].head(5)
            raise RuntimeError(
                "NCEDC returned one or more unparsable origin times. "
                f"Examples:\n{bad_rows.to_string(index=False)}"
            )

        frame["event_id"] = (
            frame["event_id"]
            .astype(str)
            .str.strip()
        )

        empty_event_id = frame["event_id"].eq("")
        if np.any(empty_event_id):
            raise RuntimeError(
                "NCEDC returned one or more rows with an empty EventID."
            )

        return frame

    def query(
        self,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        latitude: float,
        longitude: float,
        max_radius_km: float,
        min_magnitude: float | None = None,
        page_limit: int = 10000,
    ) -> pd.DataFrame:
        pages: list[pd.DataFrame] = []
        offset = 1

        while True:
            page = self._request_page(
                start=start,
                end=end,
                latitude=latitude,
                longitude=longitude,
                max_radius_km=max_radius_km,
                min_magnitude=min_magnitude,
                limit=page_limit,
                offset=offset,
            )

            if page.empty:
                break

            pages.append(page)

            if len(page) < page_limit:
                break

            offset += page_limit

        if not pages:
            return pd.DataFrame(
                columns=[
                    "event_id",
                    "origin_time_utc",
                    "latitude",
                    "longitude",
                    "depth_km",
                    "magnitude",
                ]
            )

        frame = self._standardise(
            pd.concat(pages, ignore_index=True)
        )

        return (
            frame.drop_duplicates(subset=["event_id"])
            .sort_values("origin_time_utc")
            .reset_index(drop=True)
        )


# ==============================================================================
# 4. 3-D CABLE GEOMETRY AND 2-D SECTION METRICS
# ==============================================================================

def _resolve_column(
    columns: Iterable[str],
    aliases: Iterable[str],
    *,
    required: bool,
    label: str,
) -> str | None:
    column_list = list(columns)
    norm_to_original = {
        _normalise_key(column): column
        for column in column_list
    }

    for alias in aliases:
        norm = _normalise_key(alias)
        if norm in norm_to_original:
            return norm_to_original[norm]

    for column in column_list:
        norm_column = _normalise_key(column)
        if any(_normalise_key(alias) in norm_column for alias in aliases):
            return column

    if required:
        raise ValueError(
            f"Could not identify {label} column. Available columns: "
            f"{column_list}"
        )
    return None


def _point_to_polyline_distance(
    point: np.ndarray,
    vertices: np.ndarray,
) -> tuple[float, int, float, np.ndarray]:
    """
    Minimum Euclidean distance from one point to a polyline.

    Returns distance, segment index, interpolation fraction, closest point.
    """
    if vertices.ndim != 2 or vertices.shape[0] < 2:
        raise ValueError("Polyline requires at least two vertices.")

    start = vertices[:-1]
    segment = vertices[1:] - start
    length2 = np.sum(segment * segment, axis=1)

    delta = point[None, :] - start
    fraction = np.zeros(start.shape[0], dtype=np.float64)

    valid = length2 > 0.0
    fraction[valid] = (
        np.sum(delta[valid] * segment[valid], axis=1)
        / length2[valid]
    )
    fraction = np.clip(fraction, 0.0, 1.0)

    closest = start + fraction[:, None] * segment
    distances = np.linalg.norm(point[None, :] - closest, axis=1)

    index = int(np.argmin(distances))
    return (
        float(distances[index]),
        index,
        float(fraction[index]),
        closest[index],
    )


class CableGeometry3D:
    """
    Physical SAFOD cable geometry in UTM coordinates and an absolute vertical
    datum.

    The modelling section is the SAME vertical profile used by
    scripts.safod.prepare_event:

      - origin: shallowest cable point / wellhead;
      - horizontal direction: wellhead to the median deepest down-leg cable;
      - along/cross coordinates: local east/north tangent-plane projection.

    This is intentionally not a PCA best-fit line. A PCA plane is unstable for
    a nearly vertical, curved, double-pass borehole and previously misclassified
    known near-profile events such as NC75336802.
    """

    CHANNEL_ALIASES = (
        "channel",
        "channel_id",
        "chan",
        "raw_channel",
    )
    EAST_ALIASES = (
        "utm_e",
        "easting",
        "easting_m",
        "x_utm",
        "x",
    )
    NORTH_ALIASES = (
        "utm_n",
        "northing",
        "northing_m",
        "y_utm",
        "y",
    )
    MD_ALIASES = (
        "md_m",
        "measured_depth_m",
        "measureddepth",
        "md",
    )
    TVD_ALIASES = (
        "tvd_m",
        "true_vertical_depth_m",
        "trueverticaldepth",
        "tvd",
    )
    ELEVATION_ALIASES = (
        "elevation_m",
        "elevation",
        "z_elevation_m",
        "z_utm",
        "elev_m",
    )
    LATITUDE_ALIASES = (
        "lat_wgs84",
        "latitude",
        "lat",
    )
    LONGITUDE_ALIASES = (
        "lon_wgs84",
        "longitude",
        "lon",
    )

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        wellhead_elevation_m: float | None,
        source_epsg: int = 4326,
        utm_epsg: int = 32610,
    ) -> None:
        self.frame = frame.copy().reset_index(drop=True)
        self.source_epsg = int(source_epsg)
        self.utm_epsg = int(utm_epsg)
        self.transformer = Transformer.from_crs(
            f"EPSG:{self.source_epsg}",
            f"EPSG:{self.utm_epsg}",
            always_xy=True,
        )

        channel_col = _resolve_column(
            self.frame.columns,
            self.CHANNEL_ALIASES,
            required=False,
            label="channel",
        )
        east_col = _resolve_column(
            self.frame.columns,
            self.EAST_ALIASES,
            required=True,
            label="UTM easting",
        )
        north_col = _resolve_column(
            self.frame.columns,
            self.NORTH_ALIASES,
            required=True,
            label="UTM northing",
        )
        md_col = _resolve_column(
            self.frame.columns,
            self.MD_ALIASES,
            required=False,
            label="measured depth",
        )
        tvd_col = _resolve_column(
            self.frame.columns,
            self.TVD_ALIASES,
            required=False,
            label="TVD",
        )
        elevation_col = _resolve_column(
            self.frame.columns,
            self.ELEVATION_ALIASES,
            required=False,
            label="absolute elevation",
        )
        latitude_col = _resolve_column(
            self.frame.columns,
            self.LATITUDE_ALIASES,
            required=True,
            label="WGS84 latitude",
        )
        longitude_col = _resolve_column(
            self.frame.columns,
            self.LONGITUDE_ALIASES,
            required=True,
            label="WGS84 longitude",
        )

        self.channel = (
            pd.to_numeric(self.frame[channel_col], errors="coerce").to_numpy()
            if channel_col is not None
            else np.arange(len(self.frame), dtype=np.float64)
        )
        self.east = pd.to_numeric(
            self.frame[east_col], errors="coerce"
        ).to_numpy(dtype=np.float64)
        self.north = pd.to_numeric(
            self.frame[north_col], errors="coerce"
        ).to_numpy(dtype=np.float64)
        self.md = (
            pd.to_numeric(self.frame[md_col], errors="coerce").to_numpy(
                dtype=np.float64
            )
            if md_col is not None
            else np.full(len(self.frame), np.nan)
        )
        self.tvd = (
            pd.to_numeric(self.frame[tvd_col], errors="coerce").to_numpy(
                dtype=np.float64
            )
            if tvd_col is not None
            else np.full(len(self.frame), np.nan)
        )
        self.latitude = pd.to_numeric(
            self.frame[latitude_col],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        self.longitude = pd.to_numeric(
            self.frame[longitude_col],
            errors="coerce",
        ).to_numpy(dtype=np.float64)

        if elevation_col is not None:
            self.elevation = pd.to_numeric(
                self.frame[elevation_col],
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            self.vertical_datum_status = (
                f"absolute cable elevation from geometry column {elevation_col!r}"
            )
        elif tvd_col is not None and wellhead_elevation_m is not None:
            self.elevation = (
                float(wellhead_elevation_m)
                - self.tvd
            )
            self.vertical_datum_status = (
                "absolute cable elevation = supplied wellhead elevation - TVD"
            )
        else:
            self.elevation = np.full(len(self.frame), np.nan)
            self.vertical_datum_status = (
                "vertical datum unavailable; true 3-D source-to-cable distance "
                "cannot be computed"
            )

        finite_horizontal = (
            np.isfinite(self.east)
            & np.isfinite(self.north)
            & np.isfinite(self.latitude)
            & np.isfinite(self.longitude)
        )
        if np.count_nonzero(finite_horizontal) < 2:
            raise ValueError(
                "Cable geometry has fewer than two finite horizontal points."
            )

        # Preserve physical channel/file order.
        self.valid = finite_horizontal
        self.east_valid = self.east[self.valid]
        self.north_valid = self.north[self.valid]
        self.latitude_valid = self.latitude[self.valid]
        self.longitude_valid = self.longitude[self.valid]
        self.channel_valid = self.channel[self.valid]
        self.md_valid = self.md[self.valid]
        self.tvd_valid = self.tvd[self.valid]
        self.elevation_valid = self.elevation[self.valid]

        # Wellhead = shallowest finite TVD, otherwise first valid cable point.
        if np.any(np.isfinite(self.tvd_valid)):
            self.wellhead_index = int(np.nanargmin(self.tvd_valid))
        else:
            self.wellhead_index = 0

        self.wellhead_east = float(self.east_valid[self.wellhead_index])
        self.wellhead_north = float(self.north_valid[self.wellhead_index])
        self.wellhead_latitude = float(
            self.latitude_valid[self.wellhead_index]
        )
        self.wellhead_longitude = float(
            self.longitude_valid[self.wellhead_index]
        )
        self.wellhead_elevation = (
            float(self.elevation_valid[self.wellhead_index])
            if np.isfinite(self.elevation_valid[self.wellhead_index])
            else np.nan
        )

        # Match scripts.safod.prepare_event exactly: local tangent-plane EN
        # coordinates around the wellhead.
        earth_radius_m = 6371000.0
        lat0_rad = np.deg2rad(self.wellhead_latitude)

        local_east = (
            np.deg2rad(
                self.longitude_valid - self.wellhead_longitude
            )
            * earth_radius_m
            * np.cos(lat0_rad)
        )
        local_north = (
            np.deg2rad(
                self.latitude_valid - self.wellhead_latitude
            )
            * earth_radius_m
        )

        # Use only the down-going pass to define the profile direction.
        if np.any(np.isfinite(self.tvd_valid)):
            turn_index = int(np.nanargmax(self.tvd_valid))
        else:
            turn_index = len(self.tvd_valid) - 1

        down_indices = np.arange(len(self.tvd_valid)) <= turn_index
        down_tvd = self.tvd_valid[down_indices]

        finite_down_tvd = down_tvd[np.isfinite(down_tvd)]
        if finite_down_tvd.size < 5:
            raise ValueError(
                "Too few finite down-leg TVD values to define the 2-D profile."
            )

        deep_threshold = float(
            np.nanpercentile(finite_down_tvd, 90.0)
        )
        deep_mask = (
            down_indices
            & np.isfinite(self.tvd_valid)
            & (self.tvd_valid > deep_threshold)
        )

        if np.count_nonzero(deep_mask) < 5:
            deep_threshold = float(
                np.nanpercentile(finite_down_tvd, 80.0)
            )
            deep_mask = (
                down_indices
                & np.isfinite(self.tvd_valid)
                & (self.tvd_valid > deep_threshold)
            )

        deep_east = float(
            np.nanmedian(local_east[deep_mask])
        )
        deep_north = float(
            np.nanmedian(local_north[deep_mask])
        )
        direction_norm = float(
            np.hypot(deep_east, deep_north)
        )

        if not np.isfinite(direction_norm) or direction_norm <= 0.0:
            raise RuntimeError(
                "Could not define SAFOD modelling-profile direction from "
                "the wellhead and deepest down-leg cable."
            )

        self.along_unit = np.array(
            [
                deep_east / direction_norm,
                deep_north / direction_norm,
            ],
            dtype=np.float64,
        )
        self.cross_unit = np.array(
            [
                -self.along_unit[1],
                self.along_unit[0],
            ],
            dtype=np.float64,
        )

        local_en = np.column_stack(
            [
                local_east,
                local_north,
            ]
        )
        self.along_m = local_en @ self.along_unit
        self.cross_m = local_en @ self.cross_unit

        # Retain these names for compatibility with existing code, but the
        # section origin is now the physical wellhead and cross-plane offset 0.
        self.xy_origin = np.array(
            [
                self.wellhead_east,
                self.wellhead_north,
            ],
            dtype=np.float64,
        )
        self.cross_plane_m = 0.0

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        wellhead_elevation_m: float | None,
        sheet_name: str | int | None = 0,
        min_tvd_m: float | None = 10.0,
    ) -> "CableGeometry3D":
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix in {".xlsx", ".xls"}:
            frame = pd.read_excel(path, sheet_name=sheet_name)
        elif suffix in {".csv", ".txt"}:
            frame = pd.read_csv(path)
        else:
            raise ValueError(
                f"Unsupported geometry format {suffix!r}; use Excel or CSV."
            )

        if min_tvd_m is not None:
            tvd_col = _resolve_column(
                frame.columns,
                cls.TVD_ALIASES,
                required=False,
                label="TVD",
            )
            if tvd_col is not None:
                tvd = pd.to_numeric(
                    frame[tvd_col],
                    errors="coerce",
                )
                keep = tvd >= float(min_tvd_m)
                if np.count_nonzero(keep) >= 2:
                    frame = frame.loc[keep].reset_index(drop=True)

        return cls(
            frame,
            wellhead_elevation_m=wellhead_elevation_m,
        )

    def _source_coordinates(
        self,
        longitude: float,
        latitude: float,
        depth_km: float,
    ) -> dict[str, float]:
        east, north = self.transformer.transform(
            float(longitude),
            float(latitude),
        )

        earth_radius_m = 6371000.0
        local_east = (
            np.deg2rad(
                float(longitude) - self.wellhead_longitude
            )
            * earth_radius_m
            * np.cos(
                np.deg2rad(self.wellhead_latitude)
            )
        )
        local_north = (
            np.deg2rad(
                float(latitude) - self.wellhead_latitude
            )
            * earth_radius_m
        )
        local_en = np.array(
            [
                local_east,
                local_north,
            ],
            dtype=np.float64,
        )

        along = float(local_en @ self.along_unit)
        cross = float(local_en @ self.cross_unit)

        # FDSN depth is positive downward from its catalog datum. With z-up
        # elevation, source elevation is -depth. This is valid only when cable
        # elevation uses the same geoid/sea-level datum.
        source_elevation = -1000.0 * float(depth_km)

        return {
            "source_east_m": float(east),
            "source_north_m": float(north),
            "source_elevation_m": source_elevation,
            "source_section_x_m": along,
            "source_crossline_m": cross,
        }

    def event_metrics(
        self,
        *,
        longitude: float,
        latitude: float,
        depth_km: float,
    ) -> dict[str, Any]:
        source = self._source_coordinates(
            longitude,
            latitude,
            depth_km,
        )

        horizontal_wellhead_m = float(
            np.hypot(
                source["source_east_m"] - self.wellhead_east,
                source["source_north_m"] - self.wellhead_north,
            )
        )

        metrics: dict[str, Any] = {
            **source,
            "wellhead_epicentral_distance_km": horizontal_wellhead_m / 1000.0,
            "vertical_datum_status": self.vertical_datum_status,
        }

        if np.isfinite(self.wellhead_elevation):
            wellhead_3d = np.array(
                [
                    self.wellhead_east,
                    self.wellhead_north,
                    self.wellhead_elevation,
                ]
            )
            source_3d = np.array(
                [
                    source["source_east_m"],
                    source["source_north_m"],
                    source["source_elevation_m"],
                ]
            )
            metrics["wellhead_hypocentral_distance_km"] = (
                float(np.linalg.norm(source_3d - wellhead_3d)) / 1000.0
            )
        else:
            metrics["wellhead_hypocentral_distance_km"] = np.nan

        if np.all(np.isfinite(self.elevation_valid)):
            cable_3d = np.column_stack(
                [
                    self.east_valid,
                    self.north_valid,
                    self.elevation_valid,
                ]
            )
            source_3d = np.array(
                [
                    source["source_east_m"],
                    source["source_north_m"],
                    source["source_elevation_m"],
                ]
            )
            distance_3d, segment, fraction, closest = (
                _point_to_polyline_distance(source_3d, cable_3d)
            )

            closest_channel = (
                (1.0 - fraction) * self.channel_valid[segment]
                + fraction * self.channel_valid[segment + 1]
            )
            closest_md = (
                (1.0 - fraction) * self.md_valid[segment]
                + fraction * self.md_valid[segment + 1]
                if np.all(
                    np.isfinite(
                        [self.md_valid[segment], self.md_valid[segment + 1]]
                    )
                )
                else np.nan
            )
            closest_tvd = (
                (1.0 - fraction) * self.tvd_valid[segment]
                + fraction * self.tvd_valid[segment + 1]
                if np.all(
                    np.isfinite(
                        [self.tvd_valid[segment], self.tvd_valid[segment + 1]]
                    )
                )
                else np.nan
            )

            cable_section = np.column_stack(
                [
                    self.along_m,
                    -self.elevation_valid,  # positive downward absolute z
                ]
            )
            source_section = np.array(
                [
                    source["source_section_x_m"],
                    -source["source_elevation_m"],
                ]
            )
            distance_2d, _, _, _ = _point_to_polyline_distance(
                source_section,
                cable_section,
            )

            crossline = abs(source["source_crossline_m"])
            out_of_plane_angle = math.degrees(
                math.atan2(crossline, max(distance_2d, 1.0e-12))
            )

            metrics.update(
                {
                    "min_3d_distance_to_cable_km": distance_3d / 1000.0,
                    "min_inplane_distance_to_cable_km": distance_2d / 1000.0,
                    "closest_cable_channel": float(closest_channel),
                    "closest_cable_md_m": float(closest_md),
                    "closest_cable_tvd_m": float(closest_tvd),
                    "closest_cable_east_m": float(closest[0]),
                    "closest_cable_north_m": float(closest[1]),
                    "closest_cable_elevation_m": float(closest[2]),
                    "out_of_plane_angle_deg": float(out_of_plane_angle),
                }
            )
        else:
            # Still compute horizontal source-to-cable and crossline geometry.
            cable_xy = np.column_stack([self.east_valid, self.north_valid])
            source_xy = np.array(
                [source["source_east_m"], source["source_north_m"]]
            )
            distance_horizontal, segment, fraction, closest = (
                _point_to_polyline_distance(source_xy, cable_xy)
            )
            metrics.update(
                {
                    "min_3d_distance_to_cable_km": np.nan,
                    "min_inplane_distance_to_cable_km": np.nan,
                    "closest_cable_channel": float(
                        (1.0 - fraction) * self.channel_valid[segment]
                        + fraction * self.channel_valid[segment + 1]
                    ),
                    "closest_cable_md_m": np.nan,
                    "closest_cable_tvd_m": np.nan,
                    "closest_cable_east_m": float(closest[0]),
                    "closest_cable_north_m": float(closest[1]),
                    "closest_cable_elevation_m": np.nan,
                    "out_of_plane_angle_deg": np.nan,
                    "min_horizontal_distance_to_cable_km": (
                        distance_horizontal / 1000.0
                    ),
                }
            )

        metrics.update(self.classify_2d(metrics))
        return metrics

    @staticmethod
    def classify_2d(metrics: dict[str, Any]) -> dict[str, Any]:
        """
        Conservative geometric classification.

        This is not a proof that Earth structure is 2-D. It only evaluates
        whether source and cable are approximately coplanar and whether a local
        source can reasonably be embedded in a practical 2-D model.
        """
        angle = metrics.get("out_of_plane_angle_deg", np.nan)
        crossline_km = abs(metrics["source_crossline_m"]) / 1000.0
        distance_km = metrics.get("min_3d_distance_to_cable_km", np.nan)

        if not np.isfinite(angle) or not np.isfinite(distance_km):
            return {
                "geometry_2d_class": "unknown",
                "recommended_2d_use": "unknown_vertical_datum",
                "suitable_for_direct_2d_source": False,
                "suitability_2d_reason": (
                    "Absolute cable elevation is unavailable, so the 3-D "
                    "source-to-cable and out-of-plane geometry cannot be "
                    "evaluated consistently."
                ),
                "amplitude_warning_2d": (
                    "Even a coplanar event is a 3-D point source, whereas a 2-D "
                    "solver represents a line source; absolute amplitudes require "
                    "a 2-D/3-D correction or calibration."
                ),
            }

        if angle <= 10.0 and crossline_km <= 0.5:
            geometry_class = "good"
        elif angle <= 20.0 and crossline_km <= 2.0:
            geometry_class = "borderline"
        else:
            geometry_class = "poor"

        if geometry_class == "good" and distance_km <= 10.0:
            recommended = "direct_2d_source"
            suitable = True
            reason = (
                f"Source is nearly in the cable section plane "
                f"(out-of-plane angle {angle:.1f}°, crossline "
                f"{crossline_km:.2f} km) and lies {distance_km:.2f} km from "
                "the cable. Suitable for a controlled local 2-D source test."
            )
        elif geometry_class == "good":
            recommended = "2d_incident_wave_or_larger_domain"
            suitable = False
            reason = (
                f"Source is nearly coplanar (angle {angle:.1f}°), but its "
                f"closest cable distance is {distance_km:.2f} km. A local "
                "source would require a much larger domain; a prescribed "
                "incident wave may be more efficient."
            )
        elif geometry_class == "borderline":
            recommended = "sensitivity_test_2d_vs_3d"
            suitable = False
            reason = (
                f"Moderate out-of-plane geometry (angle {angle:.1f}°, "
                f"crossline {crossline_km:.2f} km). Kinematics may be usable "
                "for an exploratory 2-D test, but waveform amplitudes and phase "
                "content require sensitivity analysis."
            )
        else:
            recommended = "3d_preferred"
            suitable = False
            reason = (
                f"Source is strongly out of the cable section plane "
                f"(angle {angle:.1f}°, crossline {crossline_km:.2f} km). "
                "A 2-D local point-source model would impose incorrect "
                "propagation geometry; 3-D modelling is preferred."
            )

        return {
            "geometry_2d_class": geometry_class,
            "recommended_2d_use": recommended,
            "suitable_for_direct_2d_source": suitable,
            "suitability_2d_reason": reason,
            "amplitude_warning_2d": (
                "A 2-D elastic solver represents a line source, not the "
                "geometrical spreading of a 3-D point earthquake. Do not use "
                "absolute amplitude agreement without correction/calibration."
            ),
        }


# ==============================================================================
# 5. COVERAGE MATCHING / CATALOG ASSEMBLY
# ==============================================================================

def _coverage_for_event(
    event_time: pd.Timestamp,
    manifest: pd.DataFrame,
    *,
    pre_s: float,
    post_s: float,
) -> dict[str, Any]:
    valid = manifest[
        manifest["start_time_utc"].notna()
        & manifest["end_time_utc"].notna()
        & manifest["error"].isna()
    ].copy()

    if valid.empty:
        return {
            "origin_covered": False,
            "window_covered": False,
            "matching_file_count": 0,
            "matching_recordings": "",
            "primary_recording": "",
            "primary_file": "",
            "event_window_start_utc": (
                event_time - pd.to_timedelta(pre_s, unit="s")
            ),
            "event_window_end_utc": (
                event_time + pd.to_timedelta(post_s, unit="s")
            ),
        }

    window_start = event_time - pd.to_timedelta(pre_s, unit="s")
    window_end = event_time + pd.to_timedelta(post_s, unit="s")

    origin_match = valid[
        (valid["start_time_utc"] <= event_time)
        & (valid["end_time_utc"] > event_time)
    ].copy()

    window_intersections = valid[
        (valid["end_time_utc"] > window_start)
        & (valid["start_time_utc"] < window_end)
    ].copy()

    # Union intervals to decide whether the complete requested window is covered.
    intervals = sorted(
        (
            max(row.start_time_utc, window_start),
            min(row.end_time_utc, window_end),
        )
        for row in window_intersections.itertuples()
    )

    covered_until = window_start
    fully_covered = False
    tolerance = pd.to_timedelta(2.0, unit="s")

    for start, end in intervals:
        if start > covered_until + tolerance:
            break
        if end > covered_until:
            covered_until = end
        if covered_until >= window_end:
            fully_covered = True
            break

    if not origin_match.empty:
        origin_match = origin_match.sort_values(
            ["recording_priority", "start_time_utc"],
            ascending=[False, False],
        )
        primary = origin_match.iloc[0]
        primary_recording = str(primary["recording_label"])
        primary_file = str(primary["file_path"])
    else:
        primary_recording = ""
        primary_file = ""

    matching_recordings = ";".join(
        sorted(set(window_intersections["recording_label"].astype(str)))
    )

    return {
        "origin_covered": bool(not origin_match.empty),
        "window_covered": bool(fully_covered),
        "matching_file_count": int(len(window_intersections)),
        "matching_recordings": matching_recordings,
        "primary_recording": primary_recording,
        "primary_file": primary_file,
        "event_window_start_utc": window_start,
        "event_window_end_utc": window_end,
    }


class SAFODCatalogBuilder:
    def __init__(
        self,
        *,
        cable: CableGeometry3D,
        manifest: pd.DataFrame,
        pre_s: float = 2.0,
        post_s: float = 12.0,
    ) -> None:
        self.cable = cable
        self.manifest = manifest.copy()
        self.pre_s = float(pre_s)
        self.post_s = float(post_s)

    def build(self, events: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []

        for event in events.itertuples(index=False):
            event_dict = event._asdict()
            origin_time = pd.Timestamp(event_dict["origin_time_utc"])

            geometry = self.cable.event_metrics(
                longitude=float(event_dict["longitude"]),
                latitude=float(event_dict["latitude"]),
                depth_km=float(event_dict["depth_km"]),
            )
            coverage = _coverage_for_event(
                origin_time,
                self.manifest,
                pre_s=self.pre_s,
                post_s=self.post_s,
            )

            rows.append(
                {
                    **event_dict,
                    **coverage,
                    **geometry,
                    "plot_status": "pending" if coverage["window_covered"] else (
                        "partial_window" if coverage["origin_covered"] else "not_recorded"
                    ),
                    "review_notes": "",
                }
            )

        catalog = pd.DataFrame.from_records(rows)
        if catalog.empty:
            return catalog

        return catalog.sort_values(
            ["origin_time_utc", "event_id"]
        ).reset_index(drop=True)


# ==============================================================================
# 6. OVERVIEW PLOTS / NOTES
# ==============================================================================

def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_catalog_overview(
    catalog: pd.DataFrame,
    cable: CableGeometry3D,
    out_dir: str | Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if catalog.empty:
        return

    recorded = catalog[catalog["origin_covered"]].copy()
    plot_frame = recorded if not recorded.empty else catalog

    # Map view.
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    scatter = ax.scatter(
        plot_frame["source_east_m"] / 1000.0,
        plot_frame["source_north_m"] / 1000.0,
        c=plot_frame["magnitude"],
        s=18.0 + 18.0 * np.maximum(plot_frame["magnitude"].fillna(0.0), 0.0),
        alpha=0.75,
    )
    ax.plot(
        cable.east_valid / 1000.0,
        cable.north_valid / 1000.0,
        linewidth=2.0,
        label="SAFOD cable",
    )
    ax.scatter(
        [cable.wellhead_east / 1000.0],
        [cable.wellhead_north / 1000.0],
        marker="^",
        s=70,
        label="wellhead",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("UTM easting [km]")
    ax.set_ylabel("UTM northing [km]")
    ax.set_title("NCEDC events during SAFOD deep-DAS recording")
    ax.legend(loc="best")
    fig.colorbar(scatter, ax=ax, label="Magnitude")
    _save_figure(fig, out_dir / "catalog_map.png")

    # Time-distance.
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    classes = ["good", "borderline", "poor", "unknown"]
    markers = {"good": "o", "borderline": "s", "poor": "x", "unknown": "."}
    for class_name in classes:
        subset = plot_frame[
            plot_frame["geometry_2d_class"].fillna("unknown") == class_name
        ]
        if subset.empty:
            continue
        distance = subset["min_3d_distance_to_cable_km"]
        if distance.isna().all():
            distance = subset["wellhead_epicentral_distance_km"]
        ax.scatter(
            subset["origin_time_utc"],
            distance,
            s=18.0 + 15.0 * np.maximum(subset["magnitude"].fillna(0.0), 0.0),
            marker=markers[class_name],
            alpha=0.75,
            label=class_name,
        )
    ax.set_xlabel("Origin time [UTC]")
    ax.set_ylabel("Minimum source–cable distance [km]")
    ax.set_title("Recorded catalog: distance, magnitude, and 2-D geometry")
    ax.legend(title="2-D geometry")
    ax.grid(True, alpha=0.25)
    _save_figure(fig, out_dir / "catalog_time_distance.png")

    # Crossline/in-plane diagnostic.
    finite = plot_frame[
        plot_frame["min_inplane_distance_to_cable_km"].notna()
        & plot_frame["source_crossline_m"].notna()
    ].copy()
    if not finite.empty:
        fig, ax = plt.subplots(figsize=(7.5, 6.2))
        scatter = ax.scatter(
            finite["min_inplane_distance_to_cable_km"],
            np.abs(finite["source_crossline_m"]) / 1000.0,
            c=finite["out_of_plane_angle_deg"],
            s=20.0 + 18.0 * np.maximum(finite["magnitude"].fillna(0.0), 0.0),
            alpha=0.8,
        )
        ax.set_xlabel("Minimum in-plane source–cable distance [km]")
        ax.set_ylabel("|Crossline offset| [km]")
        ax.set_title("Geometric suitability for the SAFOD modelling section")
        ax.grid(True, alpha=0.25)
        fig.colorbar(scatter, ax=ax, label="Out-of-plane angle [deg]")
        _save_figure(fig, out_dir / "catalog_2d_geometry.png")

        # Vertical-section projection.
        fig, ax = plt.subplots(figsize=(9.0, 6.5))
        ax.plot(
            cable.along_m / 1000.0,
            -cable.elevation_valid / 1000.0,
            linewidth=2.0,
            label="cable projection",
        )
        scatter = ax.scatter(
            finite["source_section_x_m"] / 1000.0,
            finite["depth_km"],
            c=np.abs(finite["source_crossline_m"]) / 1000.0,
            s=20.0 + 18.0 * np.maximum(finite["magnitude"].fillna(0.0), 0.0),
            alpha=0.8,
        )
        ax.invert_yaxis()
        ax.set_xlabel("Along-section coordinate [km]")
        ax.set_ylabel("Depth [km]")
        ax.set_title("Events projected into the SAFOD modelling section")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)
        fig.colorbar(scatter, ax=ax, label="|Crossline offset| [km]")
        _save_figure(fig, out_dir / "catalog_vertical_section.png")


def write_catalog_notes(
    path: str | Path,
    *,
    catalog: pd.DataFrame,
    manifest: pd.DataFrame,
    cable: CableGeometry3D,
    query_start: pd.Timestamp,
    query_end: pd.Timestamp,
    radius_km: float,
    min_magnitude: float | None,
) -> None:
    path = Path(path)

    recorded_count = int(catalog["origin_covered"].sum()) if not catalog.empty else 0
    window_count = int(catalog["window_covered"].sum()) if not catalog.empty else 0
    good_count = int(
        (
            catalog["geometry_2d_class"].eq("good")
            & catalog["origin_covered"]
        ).sum()
    ) if not catalog.empty else 0

    text = f"""# SAFOD deep-DAS event catalog

## Scope

- NCEDC query interval: `{query_start.isoformat()}` to `{query_end.isoformat()}`
- Query radius around the SAFOD wellhead: `{radius_km:.1f} km`
- Minimum magnitude: `{min_magnitude if min_magnitude is not None else "none"}`
- HDF5 files inventoried: `{len(manifest)}`
- Catalog events returned: `{len(catalog)}`
- Event origins covered by DAS data: `{recorded_count}`
- Full requested waveform windows covered: `{window_count}`
- Recorded events with `good` coplanar geometry: `{good_count}`

## Distance definitions

- `wellhead_epicentral_distance_km`: horizontal source-to-wellhead distance.
- `wellhead_hypocentral_distance_km`: 3-D source-to-wellhead distance.
- `min_3d_distance_to_cable_km`: shortest distance from the hypocenter to the
  piecewise-linear 3-D cable, not merely to the wellhead or nearest sampled
  channel.
- `min_inplane_distance_to_cable_km`: shortest distance after projecting source
  and cable into the SAFOD modelling section used by prepare_event.
- `source_crossline_m`: signed horizontal distance perpendicular to that
  section.
- `out_of_plane_angle_deg`: `atan(|crossline| / in-plane distance)`.

Vertical datum status:

> {cable.vertical_datum_status}

2-D profile definition:

- origin: physical wellhead / shallowest cable point;
- horizontal unit vector EN:
  `({cable.along_unit[0]:.8f}, {cable.along_unit[1]:.8f})`;
- direction: wellhead to median deepest down-leg cable;
- convention: identical to `scripts.safod.prepare_event`, not PCA.

The source depth and cable elevation must use the same absolute datum. TVD alone
is not enough unless the wellhead elevation is supplied.

## Meaning of the 2-D classification

The classification is a geometric screening tool, not proof that the geology is
two-dimensional.

- `good`: out-of-plane angle <= 10 degrees and crossline offset <= 0.5 km.
- `borderline`: angle <= 20 degrees and crossline offset <= 2 km.
- `poor`: larger out-of-plane geometry.
- `unknown`: consistent absolute cable elevation is unavailable.

`direct_2d_source` additionally requires the source to lie within 10 km of the
cable, so it can be embedded in a practical local model. More distant but
coplanar events may be better represented as prescribed incident waves.

## Fundamental amplitude limitation

A real earthquake is a 3-D point source. A 2-D elastic solver represents a line
source and therefore has different geometrical spreading. Coplanarity can make
travel times and phase geometry useful, but absolute amplitude agreement still
requires a 2-D/3-D correction or empirical calibration.

## Next processing stage

`catalog_recorded.csv` is the input table for event-window extraction and DAS
gather plotting. The plotter should use `recording_manifest.csv` rather than
directory names so that file boundaries, gaps, sampling-rate changes, and the
800/1000 Hz configurations are handled explicitly.
"""
    path.write_text(text, encoding="utf-8")


def load_roots_config(path: str | Path) -> list[RecordingRoot]:
    content = json.loads(Path(path).read_text(encoding="utf-8"))
    roots = content["recording_roots"] if isinstance(content, dict) else content
    return [
        RecordingRoot(
            label=str(item["label"]),
            path=str(item["path"]),
            priority=int(item.get("priority", 0)),
            notes=str(item.get("notes", "")),
        )
        for item in roots
    ]