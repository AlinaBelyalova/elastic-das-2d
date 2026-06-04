# from __future__ import annotations

# from dataclasses import dataclass
# from pathlib import Path

# import numpy as np
# import pandas as pd
# from scipy.ndimage import gaussian_filter

# from src.grid import Grid2D
# from src.model import ElasticModel2D


# @dataclass(frozen=True)
# class SafodBuildMetadata:
#     """
#     Metadata and auxiliary arrays produced when building the SAFOD-like model.
#     """
#     geom_file: str
#     build_initial_model: bool

#     x_tie_m: float
#     z_tie_m: float

#     fault_dip_deg: float
#     fault_dip_sign: float
#     fault_width_m: float | None

#     left_block_name: str
#     right_block_name: str

#     x_fault_line: np.ndarray
#     z_fault_line: np.ndarray
#     dist_to_fault: np.ndarray


# def _load_cable_geometry(geom_file: str) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Load projected 2D cable geometry from CSV.

#     Expected columns:
#         X_2D_m, Z_2D_m
#     """
#     path = Path(geom_file)
#     if not path.exists():
#         raise FileNotFoundError(f"Geometry file not found: {geom_file}")

#     df = pd.read_csv(path)

#     required = ["X_2D_m", "Z_2D_m"]
#     missing = [c for c in required if c not in df.columns]
#     if missing:
#         raise ValueError(f"Missing required columns in geometry file: {missing}")

#     x_cable = df["X_2D_m"].to_numpy(dtype=np.float64)
#     z_cable = df["Z_2D_m"].to_numpy(dtype=np.float64)

#     idx = np.argsort(z_cable)
#     x_cable = x_cable[idx]
#     z_cable = z_cable[idx]

#     if np.any(np.diff(z_cable) < 0.0):
#         raise ValueError("z_cable is not monotonic after sorting.")

#     return x_cable, z_cable


# def _vp_to_vs_rho(vp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Convert Vp to Vs and density using simple empirical relations.

#     Notes
#     -----
#     This is only a first-order approximation for the synthetic model.
#     It will fail if Vp <= 1360 m/s because the resulting Vs becomes non-positive.
#     """
#     vs = (vp - 1360.0) / 1.16
#     rho = 310.0 * (vp ** 0.25)

#     if np.any(vs <= 0.0):
#         raise ValueError(
#             "Non-positive Vs generated from empirical relation. "
#             "This likely means Vp <= 1360 m/s somewhere in the model."
#         )
#     if np.any(rho <= 0.0):
#         raise ValueError("Non-positive density generated from empirical relation.")

#     return vs, rho


# def _apply_gaussian_smoothing(
#     arr: np.ndarray,
#     sigma_x_m: float,
#     sigma_z_m: float,
#     dx: float,
#     dz: float,
# ) -> np.ndarray:
#     """
#     Apply Gaussian smoothing with sigma specified in meters.

#     This makes smoothing physically consistent even if dx/dz change.
#     """
#     sigma_x_cells = sigma_x_m / dx
#     sigma_z_cells = sigma_z_m / dz
#     return gaussian_filter(arr, sigma=(sigma_x_cells, sigma_z_cells))


# def build_safod_model(
#     geom_file: str,
#     build_initial_model: bool = False,
#     dx: float = 5.0,
#     dz: float = 5.0,
#     dt: float = 5.0e-4,
#     nt: int = 2000,
#     pad_x_m: float = 500.0,
#     pad_z_m: float = 500.0,
#     z_tie_m: float = 2100.0,
#     fault_dip_deg: float = 82.0,
#     fault_dip_sign: float = -1.0,
#     fault_width_m: float = 150.0,
#     velocity_drop_fraction: float = 0.30,
#     density_drop_fraction: float = 0.15,
#     left_block_name: str = "salinian",
#     right_block_name: str = "franciscan",
#     true_model_smoothing_m: float = 10.0,
#     initial_model_smoothing_m: float = 100.0,
# ) -> tuple[Grid2D, ElasticModel2D, np.ndarray, np.ndarray, SafodBuildMetadata]:
#     """
#     Build a SAFOD-like 2D isotropic elastic model tied to a projected cable geometry.

#     Returns
#     -------
#     grid :
#         Grid2D object.
#     model :
#         ElasticModel2D object.
#     x_cable :
#         1D projected cable x-coordinate array [m].
#     z_cable :
#         1D projected cable depth array [m].
#     metadata :
#         SafodBuildMetadata object with fault geometry and QA fields.
#     """
#     if dx <= 0.0 or dz <= 0.0 or dt <= 0.0:
#         raise ValueError("dx, dz, and dt must be positive.")
#     if nt <= 0:
#         raise ValueError("nt must be positive.")
#     if not (0.0 < fault_dip_deg < 90.0):
#         raise ValueError("fault_dip_deg must be between 0 and 90 degrees.")
#     if left_block_name == right_block_name:
#         raise ValueError("left_block_name and right_block_name must differ.")
#     if not (0.0 <= velocity_drop_fraction < 1.0):
#         raise ValueError("velocity_drop_fraction must be in [0, 1).")
#     if not (0.0 <= density_drop_fraction < 1.0):
#         raise ValueError("density_drop_fraction must be in [0, 1).")

#     # --------------------------------------------------------
#     # 1) Load cable geometry
#     # --------------------------------------------------------
#     x_cable, z_cable = _load_cable_geometry(geom_file)

#     if not (z_cable.min() <= z_tie_m <= z_cable.max()):
#         raise ValueError(
#             f"z_tie_m={z_tie_m:.2f} is outside cable depth range "
#             f"[{z_cable.min():.2f}, {z_cable.max():.2f}]"
#         )

#     x_tie_m = float(np.interp(z_tie_m, z_cable, x_cable))

#     # --------------------------------------------------------
#     # 2) Build grid through Grid2D
#     # --------------------------------------------------------
#     x_min = float(x_cable.min() - pad_x_m)
#     x_max = float(x_cable.max() + pad_x_m)
#     z_min = 0.0
#     z_max = float(z_cable.max() + pad_z_m)

#     nx = int(np.floor((x_max - x_min) / dx)) + 1
#     nz = int(np.floor((z_max - z_min) / dz)) + 1

#     grid = Grid2D(
#         nx=nx,
#         nz=nz,
#         dx=dx,
#         dz=dz,
#         nt=nt,
#         dt=dt,
#         x0=x_min,
#         z0=z_min,
#     )

#     X, Z = grid.meshgrid()

#     # --------------------------------------------------------
#     # 3) Background block properties
#     # --------------------------------------------------------
#     vp_salinian = 3500.0 + 0.8 * Z
#     vp_franciscan = 3000.0 + 0.6 * Z

#     vs_salinian, rho_salinian = _vp_to_vs_rho(vp_salinian)
#     vs_franciscan, rho_franciscan = _vp_to_vs_rho(vp_franciscan)

#     # --------------------------------------------------------
#     # 4) Fault geometry
#     # --------------------------------------------------------
#     fault_dip_rad = np.radians(fault_dip_deg)
#     m_fault = fault_dip_sign / np.tan(fault_dip_rad)

#     dist_to_fault = (
#         X - x_tie_m - m_fault * (Z - z_tie_m)
#     ) / np.sqrt(1.0 + m_fault**2)

#     x_fault_line = x_tie_m + m_fault * (grid.z - z_tie_m)

#     # --------------------------------------------------------
#     # 5) Assign left/right blocks
#     # --------------------------------------------------------
#     block_map = {
#         "salinian": (vp_salinian, vs_salinian, rho_salinian),
#         "franciscan": (vp_franciscan, vs_franciscan, rho_franciscan),
#     }

#     if left_block_name not in block_map:
#         raise ValueError(f"Unknown left_block_name: {left_block_name}")
#     if right_block_name not in block_map:
#         raise ValueError(f"Unknown right_block_name: {right_block_name}")

#     vp_left, vs_left, rho_left = block_map[left_block_name]
#     vp_right, vs_right, rho_right = block_map[right_block_name]

#     vp = np.empty(grid.shape, dtype=np.float64)
#     vs = np.empty(grid.shape, dtype=np.float64)
#     rho = np.empty(grid.shape, dtype=np.float64)

#     mask_left = dist_to_fault < 0.0
#     mask_right = ~mask_left

#     vp[mask_left] = vp_left[mask_left]
#     vs[mask_left] = vs_left[mask_left]
#     rho[mask_left] = rho_left[mask_left]

#     vp[mask_right] = vp_right[mask_right]
#     vs[mask_right] = vs_right[mask_right]
#     rho[mask_right] = rho_right[mask_right]

#     # --------------------------------------------------------
#     # 6) Fault damage zone for "true" model
#     # --------------------------------------------------------
#     if not build_initial_model:
#         gauss_profile = np.exp(
#             -(dist_to_fault**2) / (2.0 * (fault_width_m / 3.0) ** 2)
#         )

#         velocity_drop_factor = 1.0 - velocity_drop_fraction * gauss_profile
#         density_drop_factor = 1.0 - density_drop_fraction * gauss_profile

#         vp *= velocity_drop_factor
#         vs *= velocity_drop_factor
#         rho *= density_drop_factor

#     # --------------------------------------------------------
#     # 7) Smoothing in meters
#     # --------------------------------------------------------
#     sigma_m = initial_model_smoothing_m if build_initial_model else true_model_smoothing_m

#     vp = _apply_gaussian_smoothing(vp, sigma_m, sigma_m, dx, dz)
#     vs = _apply_gaussian_smoothing(vs, sigma_m, sigma_m, dx, dz)
#     rho = _apply_gaussian_smoothing(rho, sigma_m, sigma_m, dx, dz)

#     # --------------------------------------------------------
#     # 8) Build ElasticModel2D
#     # --------------------------------------------------------
#     model = ElasticModel2D(grid=grid, vp=vp, vs=vs, rho=rho)

#     metadata = SafodBuildMetadata(
#         geom_file=geom_file,
#         build_initial_model=build_initial_model,
#         x_tie_m=x_tie_m,
#         z_tie_m=z_tie_m,
#         fault_dip_deg=fault_dip_deg,
#         fault_dip_sign=fault_dip_sign,
#         fault_width_m=None if build_initial_model else fault_width_m,
#         left_block_name=left_block_name,
#         right_block_name=right_block_name,
#         x_fault_line=x_fault_line.copy(),
#         z_fault_line=grid.z.copy(),
#         dist_to_fault=dist_to_fault.copy(),
#     )

#     return grid, model, x_cable, z_cable, metadata

