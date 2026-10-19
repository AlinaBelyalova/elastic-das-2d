from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Grid2D:
    nx: int
    nz: int
    dx: float
    dz: float
    nt: int
    dt: float
    x0: float = 0.0
    z0: float = 0.0

    def __post_init__(self) -> None:
        if self.nx < 2 or self.nz < 2:
            raise ValueError(f"nx and nz must be >= 2, got nx={self.nx}, nz={self.nz}.")
        if self.nt < 2:
            raise ValueError(f"nt must be >= 2, got {self.nt}.")
        if self.dx <= 0.0 or self.dz <= 0.0 or self.dt <= 0.0:
            raise ValueError(
                f"dx, dz, dt must be positive, got dx={self.dx}, dz={self.dz}, dt={self.dt}."
            )

        x = self.x0 + np.arange(self.nx, dtype=np.float64) * self.dx
        z = self.z0 + np.arange(self.nz, dtype=np.float64) * self.dz

        x.flags.writeable = False
        z.flags.writeable = False

        object.__setattr__(self, "x", x)
        object.__setattr__(self, "z", z)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nx, self.nz)

    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.x, self.z, indexing="ij")

    def get_closest_node(self, x_m: float, z_m: float) -> tuple[int, int, float, float]:
        """
        Return the nearest grid node to the requested physical coordinates.

        Parameters
        ----------
        x_m, z_m : float
            Physical coordinates [m].

        Returns
        -------
        ix, iz, x_node, z_node
            Integer node indices and the snapped grid coordinates.
        """
        tol_x = 1.0e-5 * self.dx
        tol_z = 1.0e-5 * self.dz

        if not (self.x[0] - tol_x <= x_m <= self.x[-1] + tol_x):
            raise ValueError(
                f"x_m={x_m} is outside grid x range [{self.x[0]}, {self.x[-1]}] "
                f"within tolerance {tol_x}."
            )
        if not (self.z[0] - tol_z <= z_m <= self.z[-1] + tol_z):
            raise ValueError(
                f"z_m={z_m} is outside grid z range [{self.z[0]}, {self.z[-1]}] "
                f"within tolerance {tol_z}."
            )

        ix = int(round((x_m - self.x[0]) / self.dx))
        iz = int(round((z_m - self.z[0]) / self.dz))

        ix = max(0, min(ix, self.nx - 1))
        iz = max(0, min(iz, self.nz - 1))

        return ix, iz, float(self.x[ix]), float(self.z[iz])