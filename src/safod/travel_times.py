"""Model-based P/S guides, in metres and seconds, using EikoHypoTomo.

Native Zhang files store velocity as [z, y, x] in km/s. EikoHypoTomo's
3-D interface expects [component, y, x, z] in a single consistent unit
system and source/receiver columns [x, y, z]. We use SI throughout.
These are first-arrival predictions, not waveform picks.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from pyproj import Geod
from scipy.interpolate import RegularGridInterpolator

DEFAULT_NATIVE_GRID = Path(
    "data/safod/velocity_models/zhang_thurber_bedrosian_2009/processed/"
    "zhang2009_native_grid.npz"
)
DEFAULT_EIKO_ROOT = Path("/home/groups/ettore88/alina/packages/EikoHypoTomo")
DEFAULT_CACHE = Path("results/safod/travel_times_3d")
CACHE_SCHEMA = 1


class CoverageError(ValueError):
    """The supported physical model does not contain the requested geometry."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _axis(values, name):
    values = np.asarray(values, dtype=np.float64)
    if (values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values))
            or np.any(np.diff(values) <= 0)):
        raise ValueError(f"{name} must be a finite, strictly increasing axis.")
    return values


def _spacing(values):
    delta = np.diff(_axis(values, "Eikonal grid axis"))
    if not np.allclose(delta, delta[0], rtol=1e-9, atol=1e-8):
        raise ValueError("EikoHypoTomo requires a uniform computational grid.")
    return float(delta[0])


def _backend(root: Path, dimension: int):
    root = Path(root).expanduser().resolve()
    module_name = f"EikonalTomography{dimension}DMod"
    module_file = root / "python" / f"{module_name}.py"
    if not module_file.is_file():
        raise FileNotFoundError(
            f"Required EikoHypoTomo module not found: {module_file}. "
            "Use --eiko-root for the installed EikoHypoTomo checkout; "
            "no alternate travel-time method will be substituted."
        )
    os.environ.setdefault("DATAPATH", os.environ.get("TMPDIR", "/tmp"))
    for path in (root / "external/python-solver/GenericSolver/python", root / "python"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    module = importlib.import_module(module_name)
    if Path(module.__file__).resolve() != module_file:
        raise RuntimeError(f"{module_name} was already imported from another checkout.")
    vector = importlib.import_module("pyVector")
    pykonal = importlib.import_module("pykonal")
    return module, vector, {
        "backend": "EikoHypoTomo/pykonal",
        "backend_file": str(module_file),
        "backend_sha256": file_sha256(module_file),
        "pykonal_version": str(getattr(pykonal, "__version__", "unknown")),
    }


def require_backend(root=DEFAULT_EIKO_ROOT, dimension=3):
    """Fail early on missing/incompatible installations, before reading DAS."""
    module, _, provenance = _backend(root, dimension)
    if not callable(getattr(module, f"EikonalTT_{dimension}D", None)):
        raise ImportError(f"EikoHypoTomo does not expose EikonalTT_{dimension}D.")
    return provenance


def solve_eikonal(velocity, axes, source, receivers, *, eiko_root=DEFAULT_EIKO_ROOT):
    """Return origin-relative times; source/receiver columns follow axes order.

    2-D uses the existing project's grid-node source injection. 3-D uses
    PointSourceSolver accurate injection to retain subcell source coordinates.
    Receiver upper endpoints are excluded because the wrapper accesses i+1.
    """
    axes = tuple(_axis(a, "axis") for a in axes)
    dimension = len(axes)
    if dimension not in (2, 3):
        raise ValueError("Only 2-D and 3-D eikonal calculations are supported.")
    velocity = np.asarray(velocity, dtype=np.float64)
    if velocity.shape != tuple(a.size for a in axes):
        raise ValueError("Velocity shape does not match its physical axes.")
    if not np.all(np.isfinite(velocity)) or np.any(velocity <= 0):
        raise ValueError("Velocity must be finite and strictly positive.")
    source = np.asarray(source, dtype=np.float64)
    receivers = np.asarray(receivers, dtype=np.float64)
    if source.shape != (dimension,) or receivers.ndim != 2 or receivers.shape[1] != dimension:
        raise ValueError("Invalid source/receiver coordinate shapes.")
    if receivers.shape[0] == 0 or not np.all(np.isfinite(source)) or not np.all(np.isfinite(receivers)):
        raise ValueError("Source and nonempty receiver coordinates must be finite.")
    for i, axis in enumerate(axes):
        if (not axis[0] <= source[i] <= axis[-1]
                or np.any(receivers[:, i] < axis[0])
                or np.any(receivers[:, i] >= axis[-1])):
            raise CoverageError(f"Source/receivers outside axis {i}: {axis[0]} .. {axis[-1]} m.")
    spacing = tuple(_spacing(a) for a in axes)
    module, vector, provenance = _backend(eiko_root, dimension)
    # Public EikoHypoTomo 3-D signature: oy, ox, oz, dy, dx, dz;
    # arrays [component, y, x, z], coordinate columns [x, y, z].
    if dimension == 3:
        ordered = velocity.transpose(1, 0, 2)[None, ...]
        order = (1, 0, 2)
    else:
        ordered = velocity[None, ...]
        order = (0, 1)
    vv = vector.vectorIC(np.ascontiguousarray(ordered))
    tt = vector.vectorIC(np.zeros((1, len(receivers)), dtype=np.float64))
    operator = getattr(module, f"EikonalTT_{dimension}D")(
        vv, tt, *(float(axes[i][0]) for i in order),
        *(spacing[i] for i in order), source[None, :], receivers,
        verbose=False, allocateTT=False, Acc_Inj=(dimension == 3),
        computeRays=False, nthreads=1,
    )
    operator.forward(False, vv, tt)
    times = np.asarray(tt.getNdArray()[0], dtype=np.float64).copy()
    if times.shape != (len(receivers),) or not np.all(np.isfinite(times)) or np.any(times < 0):
        raise RuntimeError("EikoHypoTomo returned invalid travel times; no fallback used.")
    provenance.update(dimension=dimension, accurate_injection=(dimension == 3))
    if dimension == 2:
        provenance["injected_source_m"] = [
            float(a[0] + int((source[i] - a[0]) / spacing[i] + 0.5) * spacing[i])
            for i, a in enumerate(axes)
        ]
    return times, provenance


@dataclass(frozen=True)
class ZhangCoordinates:
    origin_lat: float
    origin_lon: float
    x_azimuth_deg: float
    y_azimuth_deg: float

    @classmethod
    def from_grid(cls, x_km, y_km, lat, lon):
        """Same origin and positive-axis azimuth convention as section extraction."""
        ix = int(np.argmin(abs(x_km)))
        iy = int(np.argmin(abs(y_km)))
        if abs(x_km[ix]) > 1e-9 or abs(y_km[iy]) > 1e-9:
            raise ValueError("Native grid must contain the documented Zhang X=Y=0 origin.")
        origin_lat, origin_lon = float(lat[iy, ix]), float(lon[iy, ix])
        px = np.flatnonzero(x_km > 0)[0]
        py = np.flatnonzero(y_km > 0)[0]
        geod = Geod(ellps="WGS84")
        ax = geod.inv(origin_lon, origin_lat, float(lon[iy, px]), float(lat[iy, px]))[0] % 360
        ay = geod.inv(origin_lon, origin_lat, float(lon[py, ix]), float(lat[py, ix]))[0] % 360
        if abs(np.cos(np.deg2rad(ax - ay))) > 0.01:
            raise ValueError("Native Zhang axes are not sufficiently orthogonal for the existing convention.")
        return cls(origin_lat, origin_lon, float(ax), float(ay))

    def xy_m(self, latitude, longitude):
        lat, lon = np.broadcast_arrays(np.asarray(latitude, float), np.asarray(longitude, float))
        if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
            raise ValueError("Geographic coordinates must be finite.")
        az, _, distance = Geod(ellps="WGS84").inv(
            np.full(lat.shape, self.origin_lon), np.full(lat.shape, self.origin_lat), lon, lat
        )
        east = distance * np.sin(np.deg2rad(az))
        north = distance * np.cos(np.deg2rad(az))
        # Preserve en_to_zhang_xy in extract_safod_section.py exactly.
        x = east * np.sin(np.deg2rad(self.x_azimuth_deg)) + north * np.cos(np.deg2rad(self.x_azimuth_deg))
        y = east * np.sin(np.deg2rad(self.y_azimuth_deg)) + north * np.cos(np.deg2rad(self.y_azimuth_deg))
        return np.asarray(x), np.asarray(y)


class Zhang3D:
    """Physical Zhang tomography; no artificial bounding nodes or lateral extrapolation."""

    def __init__(self, path=DEFAULT_NATIVE_GRID):
        self.path = Path(path)
        self.sha256 = file_sha256(self.path)
        with np.load(self.path, allow_pickle=False) as pkg:
            x, y, z = (_axis(pkg[k], k) for k in ("x_km", "y_km", "z_km"))
            lat, lon = np.asarray(pkg["lat_deg"]), np.asarray(pkg["lon_deg"])
            if lat.shape != (len(y), len(x)) or lon.shape != lat.shape:
                raise ValueError("Native latitude/longitude table shape mismatch.")
            self.coordinates = ZhangCoordinates.from_grid(x, y, lat, lon)
            # The established extractor excludes artificial z=-150,+340 km.
            # Also exclude the artificial +/-240 km horizontal bounding nodes.
            masks = ((x > -200) & (x < 200), (y > -200) & (y < 200), (z >= -0.5) & (z <= 10.0))
            self.axes = tuple(_axis(a[m] * 1000, "physical axis") for a, m in zip((x, y, z), masks))
            self.interpolators = []
            for key in ("vp_km_s", "vs_km_s"):
                cube = np.asarray(pkg[key], dtype=np.float64)
                if cube.shape != (len(z), len(y), len(x)):
                    raise ValueError(f"Invalid native [z,y,x] shape for {key}.")
                cube = cube[np.ix_(masks[2], masks[1], masks[0])] * 1000
                if not np.all(np.isfinite(cube)) or np.any(cube <= 0):
                    raise ValueError(f"Invalid physical tomography velocities in {key}.")
                self.interpolators.append(RegularGridInterpolator(
                    self.axes, cube.transpose(2, 1, 0), bounds_error=True
                ))
            if np.any(self.interpolators[0].values <= self.interpolators[1].values):
                raise ValueError("Physical native model must have Vp > Vs.")

    def predict(self, *, source_lat, source_lon, source_depth_km,
                receiver_lat, receiver_lon, receiver_tvd_m,
                surface_elevation_m=660.46, spacing_m=100.0, padding_m=3000.0,
                max_nodes=12_000_000, eiko_root=DEFAULT_EIKO_ROOT, cache_dir=DEFAULT_CACHE):
        """3-D origin-relative P/S times at physical receiver centres.

        A flat local ground surface is z=-surface_elevation in MSL depth.
        Above the physical top tomography node, extend its velocity constantly
        down to local ground, exactly as in the existing 2-D section preparation.
        No bottom or lateral extrapolation and no straight-ray fallback.
        """
        sx, sy = self.coordinates.xy_m(source_lat, source_lon)
        rx, ry = self.coordinates.xy_m(receiver_lat, receiver_lon)
        tvd = np.asarray(receiver_tvd_m, dtype=np.float64)
        if rx.ndim != 1 or tvd.shape != rx.shape or len(rx) == 0:
            raise ValueError("Receiver coordinates and TVD must be nonempty 1-D arrays of equal length.")
        source = np.array([float(sx), float(sy), float(source_depth_km) * 1000])
        receivers = np.column_stack((rx, ry, tvd - float(surface_elevation_m)))
        points = np.vstack((source, receivers))
        h, pad = float(spacing_m), float(padding_m)
        if (not np.isfinite(h) or h <= 0 or not np.isfinite(pad) or pad < h
                or not np.isfinite(surface_elevation_m) or not np.all(np.isfinite(points))):
            raise ValueError("Invalid coordinates, spacing, padding, or surface elevation.")
        if np.any(tvd < 0):
            raise CoverageError("Receiver TVD cannot be negative.")
        ground = -float(surface_elevation_m)
        axes = []
        for i, native in enumerate(self.axes):
            low = ground if i == 2 else float(native[0])
            high = float(native[-1])
            if np.min(points[:, i]) < low or np.max(points[:, i]) >= high:
                raise CoverageError(f"Geometry outside physical Zhang axis {i}: [{low}, {high}) m.")
            start = max(low, float(np.floor((np.min(points[:, i]) - pad) / h) * h))
            # Ending below the physical edge never extrapolates tomography.
            end = min(high, float(np.ceil((np.max(points[:, i]) + pad) / h) * h))
            count = int(np.floor((end - start) / h + 1e-9)) + 1
            if count > max_nodes:
                raise ValueError("A single grid axis exceeds --phase-max-nodes.")
            axis = start + h * np.arange(count)
            if axis.size < 3 or axis[-1] <= np.max(points[:, i]):
                raise CoverageError("Insufficient model support above a receiver interpolation endpoint.")
            axes.append(axis)
        shape = tuple(len(a) for a in axes)
        nodes = math.prod(shape)
        if nodes > max_nodes:
            raise ValueError(f"3-D grid {shape} has {nodes:,} nodes, exceeding --phase-max-nodes={max_nodes:,}.")
        metadata = {
            "schema": CACHE_SCHEMA, "model_name": "zhang2009_native_3d",
            "model_path": str(self.path), "model_sha256": self.sha256,
            "method": "3-D eikonal FMM (EikoHypoTomo/pykonal)",
            "source_xyz_m": source.tolist(), "spacing_m": h, "padding_m": pad,
            "receiver_bounds_xyz_m": np.column_stack((receivers.min(axis=0), receivers.max(axis=0))).tolist(),
            "n_receivers": len(receivers),
            "surface_elevation_m": float(surface_elevation_m),
            "vertical_datum": "depth below MSL, positive down; receiver z=TVD-surface_elevation",
            "top_extension": "constant top-node velocity up to local ground",
            "native_top_depth_m": float(self.axes[2][0]),
            "grid_shape_xyz": list(shape), "grid_bounds_xyz_m": [[float(a[0]), float(a[-1])] for a in axes],
            "estimated_working_memory_bytes": nodes * 160,
            "zhang_coordinates": self.coordinates.__dict__,
        }
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
        digest.update(receivers.tobytes())
        # Cache is invalidated when either installed wrapper or PyKonal changes.
        _, _, backend = _backend(eiko_root, 3)
        digest.update(json.dumps(backend, sort_keys=True).encode())
        metadata.update(backend)
        key = digest.hexdigest()
        metadata["cache_key"] = key
        cache = None if cache_dir is None else Path(cache_dir) / f"{key}.npz"
        if cache is not None and cache.is_file():
            with np.load(cache, allow_pickle=False) as pkg:
                p, s = pkg["p_time_s"].copy(), pkg["s_time_s"].copy()
                stored = json.loads(str(pkg["metadata"].item()))
            if p.shape != (len(receivers),) or s.shape != p.shape or not np.all(np.isfinite([p, s])):
                raise RuntimeError(f"Invalid travel-time cache: {cache}")
            return p, s, stored
        print(f"Zhang 3-D grid {shape}, h={h:g} m, bounds={metadata['grid_bounds_xyz_m']}; "
              f"estimated working memory {nodes * 160 / 2**30:.2f} GiB")
        times = []
        yz = np.stack(np.meshgrid(axes[1], axes[2], indexing="ij"), axis=-1).reshape(-1, 2)
        # Interpolate one x slab at a time to bound temporary memory.
        for interpolator in self.interpolators:
            velocity = np.empty(shape, dtype=np.float64)
            for ix, x in enumerate(axes[0]):
                query = np.column_stack((np.full(len(yz), x), yz))
                query[:, 2] = np.maximum(query[:, 2], self.axes[2][0])
                velocity[ix] = interpolator(query).reshape(shape[1:])
            tt, _ = solve_eikonal(velocity, axes, source, receivers, eiko_root=eiko_root)
            times.append(tt)
            del velocity
        p, s = times
        if np.any(s < p):
            raise RuntimeError("S predictions precede P predictions; check Vp/Vs and model support.")
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=cache.parent, suffix=".npz", delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream, p_time_s=p, s_time_s=s,
                                    receiver_xyz_m=receivers, metadata=json.dumps(metadata, sort_keys=True))
            temporary.replace(cache)
        return p, s, metadata


def add_travel_time_arguments(parser):
    parser.add_argument("--phase-native-grid", type=Path, default=DEFAULT_NATIVE_GRID)
    parser.add_argument("--eiko-root", type=Path, default=DEFAULT_EIKO_ROOT)
    parser.add_argument("--phase-cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--phase-3d-spacing-m", type=float, default=100.0)
    parser.add_argument("--phase-3d-padding-m", type=float, default=3000.0)
    parser.add_argument("--phase-max-nodes", type=int, default=12_000_000)


def prediction_options(args):
    return dict(spacing_m=args.phase_3d_spacing_m, padding_m=args.phase_3d_padding_m,
                max_nodes=args.phase_max_nodes, eiko_root=args.eiko_root, cache_dir=args.phase_cache_dir)


def receivers_from_channels(geometry_csv, channels):
    """Interpolate geography along immutable physical reference channels."""
    import pandas as pd
    table = pd.read_csv(geometry_csv)
    key = "ReferenceChannel" if "ReferenceChannel" in table else "Channel"
    columns = [key, "Lat_WGS84", "Lon_WGS84", "TVD_m"]
    data = table[columns].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if not np.all(np.isfinite(data)):
        raise ValueError("Channel geometry contains non-finite coordinates.")
    data = data[np.argsort(data[:, 0])]
    channels = np.asarray(channels, dtype=float)
    if np.any(np.diff(data[:, 0]) <= 0) or not np.all(np.isfinite(channels)):
        raise ValueError("Physical reference geometry must have unique finite channel numbers.")
    if channels.min() < data[0, 0] or channels.max() > data[-1, 0]:
        raise CoverageError("Observed channels exceed the available geographic geometry.")
    return tuple(np.interp(channels, data[:, 0], data[:, i]) for i in (1, 2, 3))
