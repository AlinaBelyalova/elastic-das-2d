"""Original separate Lin et al. (2010) Vp/Vs, with no inferred Vs or extrapolation.

The seven columns are longitude, latitude, MSL depth (km), native x/y (km),
velocity (km/s), and DWS. Native horizontal coordinates are model labels:
geographic quadrilaterals locate them without guessing the author's projection.
Eikonal propagation uses a local Cartesian east/north frame in metres.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from pyproj import Geod
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from src.safod.travel_times import (
    CoverageError, DEFAULT_LIN_VP, DEFAULT_LIN_VS, Zhang3D, _axis, file_sha256,
)


@dataclass(frozen=True)
class LinCoordinates:
    """WGS84 azimuthal-equidistant local EN, centred at the SAFOD wellhead."""

    origin_lat: float = 35.9742039
    origin_lon: float = -120.5521414

    def xy_m(self, latitude, longitude):
        lat, lon = np.broadcast_arrays(np.asarray(latitude, float), np.asarray(longitude, float))
        if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
            raise ValueError("Geographic coordinates must be finite.")
        azimuth, _, distance = Geod(ellps="WGS84").inv(
            np.full(lat.shape, self.origin_lon), np.full(lat.shape, self.origin_lat), lon, lat)
        angle = np.deg2rad(azimuth)
        return np.asarray(distance * np.sin(angle)), np.asarray(distance * np.cos(angle))

    def geographic(self, x_m, y_m):
        x, y = np.broadcast_arrays(np.asarray(x_m, float), np.asarray(y_m, float))
        lon, lat, _ = Geod(ellps="WGS84").fwd(
            np.full(x.shape, self.origin_lon), np.full(x.shape, self.origin_lat),
            np.rad2deg(np.arctan2(x, y)), np.hypot(x, y))
        return np.asarray(lat), np.asarray(lon)


class LinGrid:
    """One complete native velocity grid; geographic lookup is vectorized."""

    def __init__(self, path):
        self.path = Path(path)
        self.sha256 = file_sha256(self.path)
        raw = np.loadtxt(self.path)
        if raw.ndim != 2 or raw.shape[1] != 7 or not np.all(np.isfinite(raw)):
            raise ValueError(f"Lin input must contain seven finite columns: {self.path}")
        self.x, self.y, self.z = (_axis(np.unique(raw[:, i]), f"Lin column {i}")
                                  for i in (3, 4, 2))
        shape = (len(self.z), len(self.y), len(self.x))
        ix, iy, iz = (np.searchsorted(axis, raw[:, i])
                      for axis, i in ((self.x, 3), (self.y, 4), (self.z, 2)))
        linear = np.ravel_multi_index((iz, iy, ix), shape)
        if len(raw) != np.prod(shape) or len(np.unique(linear)) != len(raw):
            raise ValueError(f"Lin grid is incomplete or has duplicate nodes: {self.path}")
        self.velocity = np.empty(shape)
        self.dws = np.empty(shape)
        self.velocity[iz, iy, ix] = raw[:, 5]
        self.dws[iz, iy, ix] = raw[:, 6]
        if np.any(self.velocity <= 0) or np.any(self.dws < 0):
            raise ValueError(f"Lin velocities must be positive and DWS nonnegative: {self.path}")
        self.lon, self.lat = np.empty(shape[1:]), np.empty(shape[1:])
        top = iz == 0
        self.lon[iy[top], ix[top]], self.lat[iy[top], ix[top]] = raw[top, 0], raw[top, 1]
        if not (np.allclose(raw[:, 0], self.lon[iy, ix], rtol=0, atol=1e-10)
                and np.allclose(raw[:, 1], self.lat[iy, ix], rtol=0, atol=1e-10)):
            raise ValueError("Lin horizontal coordinates must be independent of depth.")
        self.lon_scale = float(np.cos(np.deg2rad(self.lat.mean())))
        self.tree = cKDTree(np.column_stack((self.lon.ravel() * self.lon_scale, self.lat.ravel())))
        self.interpolator = RegularGridInterpolator(
            (self.x, self.y, self.z), self.velocity.transpose(2, 1, 0), bounds_error=True)

    def native_xy_km(self, latitude, longitude):
        """Invert bilinear geographic cells, with no nearest-node substitution.

        Try the four cells adjacent to the nearest geographic node. Newton
        iterations solve both coordinates together; an outside point fails.
        Horizontal file precision is 0.0001 degree, so the geographic cell
        mapping (rather than an undocumented map projection) defines lookup.
        Fractions within 1e-10 of a cell edge are rounded to that edge only
        to absorb floating-point roundoff (3 micrometres at 30 km spacing).
        """
        lat, lon = np.broadcast_arrays(np.asarray(latitude, float), np.asarray(longitude, float))
        if not np.all(np.isfinite([lat, lon])):
            raise ValueError("Lin query coordinates must be finite.")
        target = np.column_stack((lon.ravel(), lat.ravel()))
        _, nearest = self.tree.query(target * [self.lon_scale, 1.0])
        jn, inn = np.unravel_index(nearest, self.lon.shape)
        found = np.zeros(len(target), dtype=bool)
        result = np.empty((len(target), 2))
        for dj, di in ((-1, -1), (-1, 0), (0, -1), (0, 0)):
            jj, ii = jn + dj, inn + di
            take = np.flatnonzero((~found) & (jj >= 0) & (jj < len(self.y)-1)
                                  & (ii >= 0) & (ii < len(self.x)-1))
            if len(take) == 0:
                continue
            j, i = jj[take], ii[take]
            a = np.column_stack((self.lon[j, i], self.lat[j, i]))
            b = np.column_stack((self.lon[j, i+1], self.lat[j, i+1])) - a
            d = np.column_stack((self.lon[j+1, i], self.lat[j+1, i])) - a
            cross = np.column_stack((self.lon[j+1, i+1], self.lat[j+1, i+1])) - a - b - d
            u, v = np.full(len(take), 0.5), np.full(len(take), 0.5)
            for _ in range(10):
                residual = a + u[:, None]*b + v[:, None]*d + (u*v)[:, None]*cross - target[take]
                du, dv = b + v[:, None]*cross, d + u[:, None]*cross
                determinant = du[:, 0]*dv[:, 1] - du[:, 1]*dv[:, 0]
                if np.any(abs(determinant) < 1e-14):
                    raise ValueError("Degenerate Lin geographic cell.")
                u -= (residual[:, 0]*dv[:, 1] - residual[:, 1]*dv[:, 0]) / determinant
                v -= (du[:, 0]*residual[:, 1] - du[:, 1]*residual[:, 0]) / determinant
            residual = a + u[:, None]*b + v[:, None]*d + (u*v)[:, None]*cross - target[take]
            u = np.where(abs(u) < 1e-10, 0., np.where(abs(u-1) < 1e-10, 1., u))
            v = np.where(abs(v) < 1e-10, 0., np.where(abs(v-1) < 1e-10, 1., v))
            inside = ((u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
                      & (np.max(abs(residual), axis=1) < 1e-10))
            accepted = take[inside]
            result[accepted, 0] = self.x[i[inside]] + u[inside]*np.diff(self.x)[i[inside]]
            result[accepted, 1] = self.y[j[inside]] + v[inside]*np.diff(self.y)[j[inside]]
            found[accepted] = True
        if not np.all(found):
            lon_bad, lat_bad = target[np.flatnonzero(~found)[0]]
            raise CoverageError(f"Outside original Lin geographic cells: lon={lon_bad}, lat={lat_bad}")
        return result[:, 0].reshape(lat.shape), result[:, 1].reshape(lat.shape)

    def sample(self, latitude, longitude, depth_km):
        lat, lon, depth = np.broadcast_arrays(latitude, longitude, depth_km)
        if not np.all(np.isfinite(depth)):
            raise ValueError("Lin depth must be finite.")
        if np.any(depth < self.z[0]) or np.any(depth > self.z[-1]):
            raise CoverageError(f"Outside Lin depth range [{self.z[0]}, {self.z[-1]}] km MSL")
        x, y = self.native_xy_km(lat, lon)
        return self.interpolator(np.column_stack((x.ravel(), y.ravel(), depth.ravel()))).reshape(lat.shape)

    def info(self):
        return dict(path=str(self.path), sha256=self.sha256,
                    shape_zyx=list(self.velocity.shape), native_x_km=self.x.tolist(),
                    native_y_km=self.y.tolist(), native_depth_km=self.z.tolist())


class Lin3D(Zhang3D):
    """Use the shared SI-unit eikonal/cache implementation on original Lin fields."""

    model_name = "lin2010_original_vp_vs_3d"
    model_label = "Lin 2010 3-D (original Vp/Vs)"
    coordinate_frame = "Local WGS84 EN at SAFOD wellhead; metres; z is MSL-referenced depth"
    top_extension = "none; original Lin support starts at -1 km MSL"

    def __init__(self, vp_path=DEFAULT_LIN_VP, vs_path=DEFAULT_LIN_VS):
        vp, vs = LinGrid(vp_path), LinGrid(vs_path)
        self.interpolators = [vp, vs]
        self.path = Path(vp_path)
        self.sha256 = hashlib.sha256((vp.sha256 + vs.sha256).encode()).hexdigest()
        self.coordinates = LinCoordinates()
        bounds = []
        for grid in self.interpolators:
            x, y = self.coordinates.xy_m(grid.lat, grid.lon)
            bounds.append([(float(x.min()), float(x.max())), (float(y.min()), float(y.max())),
                           (1000*float(grid.z[0]), 1000*float(grid.z[-1]))])
        self.axes = tuple(np.array([max(b[i][0] for b in bounds), min(b[i][1] for b in bounds)])
                          for i in range(3))
        # Both fields use the same native x/y labels; their node densities differ.
        z, y, x = np.meshgrid(vp.z, vp.y, vp.x, indexing="ij")
        if (vp.x[0] != vs.x[0] or vp.x[-1] != vs.x[-1]
                or vp.y[0] != vs.y[0] or vp.y[-1] != vs.y[-1]
                or not np.array_equal(vp.z, vs.z)):
            raise ValueError("Original Lin Vp/Vs must share native bounds and depth nodes.")
        sampled_vs = vs.interpolator(np.column_stack((x.ravel(), y.ravel(), z.ravel())))
        # Independently inverted coarse fields can be inconsistent far from
        # SAFOD. Record that fact; require Vp > Vs everywhere in the volume
        # actually used, rather than rejecting an unrelated distant node.
        self.native_vp_le_vs_nodes = int(np.sum(vp.velocity.ravel() <= sampled_vs))

    def _top_bound(self, ground):
        # Native Lin covers above-sea-level catalogue sources too. Do not move
        # a source to the local flat ground or impose Zhang's z=-0.5 km top.
        return float(self.axes[2][0])

    def _model_metadata(self):
        return dict(lin_coordinates=self.coordinates.__dict__,
                    lin_original_fields=[grid.info() for grid in self.interpolators],
                    horizontal_lookup="inverse bilinear published geographic cells; no extrapolation",
                    field_units="native km/s, converted to m/s for propagation",
                    native_vp_le_vs_nodes=self.native_vp_le_vs_nodes,
                    field_resolution_note="Vp 10 km, Vs 30 km horizontal nodes; numerical spacing adds no structure")

    def _sample_slab(self, grid, x, yz):
        # yz is a mesh flattened with z varying fastest. Map ny horizontal
        # locations once, rather than transforming ny*nz geographic points.
        nz = len(np.unique(yz[:, 1]))
        y = yz[::nz, 0]
        lat, lon = self.coordinates.geographic(np.full(len(y), x), y)
        nx, ny = grid.native_xy_km(lat, lon)
        query = np.column_stack((np.repeat(nx, nz), np.repeat(ny, nz), yz[:, 1]/1000))
        velocity = grid.interpolator(query) * 1000
        if not np.all(np.isfinite(velocity)) or np.any(velocity <= 0):
            raise ValueError("Invalid interpolated Lin velocity.")
        if grid is self.interpolators[1]:
            vp = self._sample_slab(self.interpolators[0], x, yz)
            if np.any(vp <= velocity):
                raise ValueError("Lin Vp <= Vs inside the requested computational volume.")
        return velocity

    def predict(self, **options):
        # Validate exact source/receiver locations before constructing a volume.
        for grid in self.interpolators:
            grid.sample(options["source_lat"], options["source_lon"], options["source_depth_km"])
            grid.sample(options["receiver_lat"], options["receiver_lon"],
                        (np.asarray(options["receiver_tvd_m"])-options.get("surface_elevation_m", 660.46))/1000)
        p, s, metadata = super().predict(**options)
        elevation = float(options.get("surface_elevation_m", 660.46))
        return p, s, dict(metadata, source_above_local_flat_ground=(1000*float(options["source_depth_km"]) < -elevation))
