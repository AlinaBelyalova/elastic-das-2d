from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from src.grid import Grid2D


@dataclass
class ElasticModel2D:
    """
    2D isotropic elastic model.

    Parameters
    ----------
    grid :
        Grid2D object defining the computational mesh.
    vp :
        P-wave velocity array [m/s], shape (nx, nz).
    vs :
        S-wave velocity array [m/s], shape (nx, nz).
    rho :
        Density array [kg/m^3], shape (nx, nz).

    Derived fields
    --------------
    lam :
        First Lamé parameter [Pa].
    mu :
        Shear modulus [Pa].
    """
    grid: Grid2D
    vp: np.ndarray
    vs: np.ndarray
    rho: np.ndarray

    lam: np.ndarray = field(init=False)
    mu: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.vp = np.asarray(self.vp, dtype=np.float64)
        self.vs = np.asarray(self.vs, dtype=np.float64)
        self.rho = np.asarray(self.rho, dtype=np.float64)

        expected_shape = self.grid.shape
        for name, arr in [("vp", self.vp), ("vs", self.vs), ("rho", self.rho)]:
            if arr.shape != expected_shape:
                raise ValueError(
                    f"{name} has shape {arr.shape}, expected {expected_shape}."
                )

        if np.any(self.vp <= 0.0):
            raise ValueError("vp contains non-positive values.")
        if np.any(self.vs <= 0.0):
            raise ValueError("vs contains non-positive values.")
        if np.any(self.rho <= 0.0):
            raise ValueError("rho contains non-positive values.")
        if np.any(self.vp <= self.vs):
            raise ValueError("vp must be strictly greater than vs everywhere.")

        self.mu = self.rho * self.vs**2
        self.lam = self.rho * self.vp**2 - 2.0 * self.mu

        if np.any(self.mu <= 0.0):
            raise ValueError("Computed shear modulus mu contains non-positive values.")
        if np.any(self.lam + 2.0 * self.mu <= 0.0):
            raise ValueError("Model is not physically valid: lambda + 2mu must be positive.")