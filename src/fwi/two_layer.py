# ==============================================================================
# src/fwi/two_layer.py
#
# FWI-oriented OOP layer around the production Numba elastic-DAS solver.
#
# Python classes own configuration, validation, caching, and experiment setup.
# The hot time-stepping path remains array/scalar based and is delegated to
# @njit kernels in solver_numba_fused.py. Python objects do not enter hot loops.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import numpy as np

from src.grid import Grid2D
from src.receivers import Receivers2D
from src.das import DASResult, compute_axial_strain_rate
from src.sampling import ReceiverSampling2D, build_receiver_sampling
from src.source import MomentTensor2D, build_rotated_double_couple_2d
from src.source_injection import StressSourceInjection
from src.source_spreading import build_stress_source_spreading
from src.solver_numpy import ElasticRunResult, max_stable_dt
from src.solver_numba_fused import run_elastic_solver_numba_fused


@dataclass(frozen=True)
class ElasticLayer:
    vp_m_s: float
    vs_m_s: float
    rho_kg_m3: float
    name: str = "layer"

    def __post_init__(self) -> None:
        for name, value in (
            ("vp_m_s", self.vp_m_s),
            ("vs_m_s", self.vs_m_s),
            ("rho_kg_m3", self.rho_kg_m3),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive; got {value}.")
        if self.vp_m_s <= self.vs_m_s:
            raise ValueError("vp must exceed vs.")
        if self.vs_m_s > self.vp_m_s / np.sqrt(2.0):
            raise ValueError("Current solver requires lambda >= 0: vs <= vp/sqrt(2).")


@dataclass(frozen=True)
class TwoLayerModelSpec:
    interface_depth_m: float
    top: ElasticLayer
    bottom: ElasticLayer
    require_grid_aligned_interface: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.interface_depth_m):
            raise ValueError("interface_depth_m must be finite.")


@dataclass(frozen=True)
class DomainSpec:
    width_m: float = 2000.0
    depth_m: float = 2000.0
    dx_m: float = 5.0
    dz_m: float = 5.0
    duration_s: float = 1.0
    x0_m: float = 0.0
    z0_m: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("width_m", self.width_m),
            ("depth_m", self.depth_m),
            ("dx_m", self.dx_m),
            ("dz_m", self.dz_m),
            ("duration_s", self.duration_s),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive; got {value}.")
        if not np.isclose(self.dx_m, self.dz_m):
            raise ValueError("Current production solver requires dx == dz.")


@dataclass(frozen=True)
class SolverSpec:
    half_order: int = 2
    cfl_safety: float = 0.80
    n_boundary: int = 60
    gamma_s: float = 100.0
    use_ts_sfd: bool = False
    free_surface: bool = False

    def __post_init__(self) -> None:
        if self.half_order not in (1, 2, 3, 4):
            raise ValueError("half_order must be 1, 2, 3, or 4.")
        if not 0.0 < self.cfl_safety <= 1.0:
            raise ValueError("cfl_safety must lie in (0, 1].")
        if self.n_boundary <= self.half_order:
            raise ValueError("n_boundary must exceed half_order.")
        if self.gamma_s < 0.0:
            raise ValueError("gamma_s must be non-negative.")


@dataclass(frozen=True)
class DASGeometrySpec:
    x_top_m: float = 1300.0
    z_top_m: float = 350.0
    x_bottom_m: float = 1300.0
    z_bottom_m: float = 1500.0
    channel_spacing_m: float = 5.0
    gauge_length_m: float = 20.0

    def __post_init__(self) -> None:
        if self.channel_spacing_m <= 0.0:
            raise ValueError("channel_spacing_m must be positive.")
        if self.gauge_length_m <= 0.0:
            raise ValueError("gauge_length_m must be positive.")
        if np.hypot(
            self.x_bottom_m - self.x_top_m,
            self.z_bottom_m - self.z_top_m,
        ) <= self.channel_spacing_m:
            raise ValueError("Cable is too short for the requested spacing.")


@dataclass(frozen=True)
class MomentTensorSourceSpec:
    x_m: float = 850.0
    z_m: float = 600.0
    theta_deg: float = 25.0
    scalar_moment_nm: float = 1.0e12
    f0_hz: float = 8.0

    def __post_init__(self) -> None:
        if self.scalar_moment_nm <= 0.0:
            raise ValueError("scalar_moment_nm must be positive.")
        if self.f0_hz <= 0.0:
            raise ValueError("f0_hz must be positive.")
        if not np.isfinite(self.x_m) or not np.isfinite(self.z_m):
            raise ValueError("Source coordinates must be finite.")


@dataclass(frozen=True)
class ElasticMedium2D:
    grid: Grid2D
    vp: np.ndarray
    vs: np.ndarray
    rho: np.ndarray

    def __post_init__(self) -> None:
        expected = (int(self.grid.nx), int(self.grid.nz))
        for name in ("vp", "vs", "rho"):
            arr = np.array(getattr(self, name), dtype=np.float64, copy=True, order="C")
            if arr.shape != expected:
                raise ValueError(f"{name} shape {arr.shape} does not match grid {expected}.")
            if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
                raise ValueError(f"{name} must be finite and positive everywhere.")
            arr.flags.writeable = False
            object.__setattr__(self, name, arr)
        if np.any(self.vp <= self.vs):
            raise ValueError("vp must exceed vs everywhere.")
        if np.any(self.vs > self.vp / np.sqrt(2.0)):
            raise ValueError("Current solver requires lambda >= 0 everywhere.")

    @property
    def shape(self) -> tuple[int, int]:
        return self.vp.shape


@dataclass(frozen=True)
class PreparedMomentTensorSource:
    spec: MomentTensorSourceSpec
    moment_tensor: MomentTensor2D
    source_ix: int
    source_iz: int
    stf_xx: np.ndarray
    stf_zz: np.ndarray
    stf_xz: np.ndarray
    injection: StressSourceInjection

    def __post_init__(self) -> None:
        nt = self.stf_xx.size
        for name in ("stf_xx", "stf_zz", "stf_xz"):
            arr = np.array(getattr(self, name), dtype=np.float64, copy=True, order="C")
            if arr.shape != (nt,):
                raise ValueError(f"{name} must have shape ({nt},); got {arr.shape}.")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains NaN or Inf.")
            arr.flags.writeable = False
            object.__setattr__(self, name, arr)


@dataclass(frozen=True)
class ForwardShotResult:
    wavefield: ElasticRunResult
    das: DASResult
    elapsed_s: float

    @property
    def time_steps_per_second(self) -> float:
        return float(self.wavefield.t_v.size / self.elapsed_s)


def _build_grid(domain: DomainSpec, solver: SolverSpec, vp_max_m_s: float) -> Grid2D:
    nx_float = domain.width_m / domain.dx_m
    nz_float = domain.depth_m / domain.dz_m
    if not np.isclose(nx_float, round(nx_float), atol=1.0e-10):
        raise ValueError("width_m must be an integer multiple of dx_m.")
    if not np.isclose(nz_float, round(nz_float), atol=1.0e-10):
        raise ValueError("depth_m must be an integer multiple of dz_m.")

    nx = int(round(nx_float)) + 1
    nz = int(round(nz_float)) + 1
    dt = max_stable_dt(
        vp_max=float(vp_max_m_s),
        dx=domain.dx_m,
        dz=domain.dz_m,
        half_order=solver.half_order,
        safety=solver.cfl_safety,
        use_ts_sfd=solver.use_ts_sfd,
    )
    nt = int(np.ceil(domain.duration_s / dt))

    return Grid2D(
        nx=nx,
        nz=nz,
        dx=domain.dx_m,
        dz=domain.dz_m,
        nt=nt,
        dt=dt,
        x0=domain.x0_m,
        z0=domain.z0_m,
    )


def _build_two_layer_medium(grid: Grid2D, spec: TwoLayerModelSpec) -> ElasticMedium2D:
    frac = (spec.interface_depth_m - float(grid.z0)) / float(grid.dz)
    if spec.require_grid_aligned_interface and not np.isclose(frac, round(frac), atol=1.0e-10):
        raise ValueError(
            "For the first controlled FWI test, interface_depth_m must lie exactly on a z grid row."
        )
    if not float(grid.z[0]) < spec.interface_depth_m < float(grid.z[-1]):
        raise ValueError("Interface must lie strictly inside the domain.")

    below_1d = np.asarray(grid.z, dtype=np.float64) >= float(spec.interface_depth_m)
    vp_1d = np.where(below_1d, spec.bottom.vp_m_s, spec.top.vp_m_s)
    vs_1d = np.where(below_1d, spec.bottom.vs_m_s, spec.top.vs_m_s)
    rho_1d = np.where(below_1d, spec.bottom.rho_kg_m3, spec.top.rho_kg_m3)

    vp = np.ascontiguousarray(np.broadcast_to(vp_1d[None, :], (grid.nx, grid.nz)))
    vs = np.ascontiguousarray(np.broadcast_to(vs_1d[None, :], (grid.nx, grid.nz)))
    rho = np.ascontiguousarray(np.broadcast_to(rho_1d[None, :], (grid.nx, grid.nz)))

    return ElasticMedium2D(grid=grid, vp=vp, vs=vs, rho=rho)


def _build_linear_das_cable(
    grid: Grid2D,
    spec: DASGeometrySpec,
    *,
    n_boundary: int,
) -> Receivers2D:
    dx_line = float(spec.x_bottom_m - spec.x_top_m)
    dz_line = float(spec.z_bottom_m - spec.z_top_m)
    length_m = float(np.hypot(dx_line, dz_line))

    s = np.arange(
        int(np.floor(length_m / spec.channel_spacing_m)) + 1,
        dtype=np.float64,
    ) * float(spec.channel_spacing_m)
    s = s[s <= length_m + 1.0e-10]
    if s.size < 3:
        raise ValueError("Need at least three receiver channels.")

    fraction = s / length_m
    x = spec.x_top_m + fraction * dx_line
    z = spec.z_top_m + fraction * dz_line
    tx = np.full(s.size, dx_line / length_m, dtype=np.float64)
    tz = np.full(s.size, dz_line / length_m, dtype=np.float64)

    ix = np.rint((x - float(grid.x0)) / float(grid.dx)).astype(np.int64)
    iz = np.rint((z - float(grid.z0)) / float(grid.dz)).astype(np.int64)

    if np.any(ix < n_boundary) or np.any(ix >= grid.nx - n_boundary):
        raise ValueError("Cable enters the side sponge.")
    if np.any(iz < n_boundary) or np.any(iz >= grid.nz - n_boundary):
        raise ValueError("Cable enters the top/bottom sponge.")

    return Receivers2D(x=x, z=z, ix=ix, iz=iz, tx=tx, tz=tz, s=s)


def _physical_moment_rate_factor(*, nt: int, dt: float, f0_hz: float) -> np.ndarray:
    """Return -dW/dt for a unit Ricker moment history, sampled at t=(n+1/2)dt."""
    t = (np.arange(nt, dtype=np.float64) + 0.5) * float(dt)
    t0 = 1.2 / float(f0_hz)
    a = np.pi * float(f0_hz)
    tau = t - t0
    arg2 = (a * tau) ** 2
    return 2.0 * a**2 * tau * (3.0 - 2.0 * arg2) * np.exp(-arg2)


def _prepare_moment_tensor_source(
    grid: Grid2D,
    spec: MomentTensorSourceSpec,
) -> PreparedMomentTensorSource:
    mt = build_rotated_double_couple_2d(
        theta_deg=spec.theta_deg,
        scalar_moment=spec.scalar_moment_nm,
    )
    factor = _physical_moment_rate_factor(nt=grid.nt, dt=grid.dt, f0_hz=spec.f0_hz)

    spreading = build_stress_source_spreading(grid=grid, x_s=spec.x_m, z_s=spec.z_m)
    injection = StressSourceInjection(
        normal_ix=spreading.sxx.ix,
        normal_iz=spreading.sxx.iz,
        normal_w=spreading.sxx.w,
        shear_ix=spreading.sxz.ix,
        shear_iz=spreading.sxz.iz,
        shear_w=spreading.sxz.w,
        mode="bilinear",
    )
    injection.validate_bounds(nx=int(grid.nx), nz=int(grid.nz))
    source_ix, source_iz, _, _ = grid.get_closest_node(spec.x_m, spec.z_m)

    return PreparedMomentTensorSource(
        spec=spec,
        moment_tensor=mt,
        source_ix=int(source_ix),
        source_iz=int(source_iz),
        stf_xx=factor * mt.Mxx,
        stf_zz=factor * mt.Mzz,
        stf_xz=factor * mt.Mxz,
        injection=injection,
    )


class NumbaElasticDASForward:
    """
    Cached production forward operator.

    OOP is used for orchestration/caching only. The hot loop still receives
    contiguous NumPy arrays and scalars, which is what Numba optimises well.
    """

    def __init__(
        self,
        *,
        grid: Grid2D,
        receivers: Receivers2D,
        gauge_length_m: float,
        solver: SolverSpec,
    ) -> None:
        self.grid = grid
        self.receivers = receivers
        self.gauge_length_m = float(gauge_length_m)
        self.solver = solver
        if self.gauge_length_m <= 0.0:
            raise ValueError("gauge_length_m must be positive.")
        self.receiver_sampling: ReceiverSampling2D = build_receiver_sampling(grid, receivers)

    def prepare_source(self, spec: MomentTensorSourceSpec) -> PreparedMomentTensorSource:
        return _prepare_moment_tensor_source(self.grid, spec)

    def run(
        self,
        medium: ElasticMedium2D,
        source: PreparedMomentTensorSource,
        *,
        snapshot_stride: int | None = None,
    ) -> ForwardShotResult:
        if medium.grid is not self.grid:
            raise ValueError(
                "medium.grid must be the same Grid2D instance used to build the cached engine."
            )

        start = perf_counter()
        wavefield = run_elastic_solver_numba_fused(
            vp=medium.vp,
            vs=medium.vs,
            rho=medium.rho,
            dx=self.grid.dx,
            dz=self.grid.dz,
            dt=self.grid.dt,
            nt=self.grid.nt,
            source_ix=source.source_ix,
            source_iz=source.source_iz,
            stf_xx=source.stf_xx,
            stf_zz=source.stf_zz,
            stf_xz=source.stf_xz,
            receiver_sampling=self.receiver_sampling,
            half_order=self.solver.half_order,
            use_ts_sfd=self.solver.use_ts_sfd,
            n_boundary=self.solver.n_boundary,
            gamma_s=self.solver.gamma_s,
            snapshot_stride=snapshot_stride,
            free_surface=self.solver.free_surface,
            source_injection=source.injection,
        )
        das = compute_axial_strain_rate(
            vx=wavefield.receiver_vx,
            vz=wavefield.receiver_vz,
            receivers=self.receivers,
            gauge_length_m=self.gauge_length_m,
        )
        return ForwardShotResult(
            wavefield=wavefield,
            das=das,
            elapsed_s=float(perf_counter() - start),
        )

    def run_many(
        self,
        medium: ElasticMedium2D,
        sources: Iterable[PreparedMomentTensorSource],
    ) -> list[ForwardShotResult]:
        # Do not create Python threads here: Numba kernels already use parallel
        # CPU regions. At scale, shot/event parallelism belongs at process/Slurm level.
        return [self.run(medium, source) for source in sources]


class TwoLayerFWIProblem:
    """Reproducible heterogeneous benchmark for adjoint/Taylor/FWI development."""

    def __init__(
        self,
        *,
        model: TwoLayerModelSpec,
        domain: DomainSpec = DomainSpec(),
        solver: SolverSpec = SolverSpec(),
        das: DASGeometrySpec = DASGeometrySpec(),
        source: MomentTensorSourceSpec = MomentTensorSourceSpec(),
    ) -> None:
        self.model_spec = model
        self.domain_spec = domain
        self.solver_spec = solver
        self.das_spec = das
        self.source_spec = source

        vp_max = max(model.top.vp_m_s, model.bottom.vp_m_s)
        self.grid = _build_grid(domain, solver, vp_max_m_s=vp_max)
        self.true_medium = _build_two_layer_medium(self.grid, model)
        self.receivers = _build_linear_das_cable(
            self.grid,
            das,
            n_boundary=solver.n_boundary,
        )
        self.engine = NumbaElasticDASForward(
            grid=self.grid,
            receivers=self.receivers,
            gauge_length_m=das.gauge_length_m,
            solver=solver,
        )
        self.source = self.engine.prepare_source(source)
        self._validate_geometry()

    def _validate_geometry(self) -> None:
        interface = self.model_spec.interface_depth_m
        if not self.das_spec.z_top_m < interface < self.das_spec.z_bottom_m:
            raise ValueError("The controlled DAS cable must cross the interface.")
        if self.source_spec.z_m >= interface:
            raise ValueError("The first controlled source should lie above the interface.")
        margin = self.solver_spec.n_boundary
        if not (
            margin <= self.source.source_ix < self.grid.nx - margin
            and margin <= self.source.source_iz < self.grid.nz - margin
        ):
            raise ValueError("Source lies inside or too near the sponge.")

    @classmethod
    def default(cls) -> "TwoLayerFWIProblem":
        return cls(
            model=TwoLayerModelSpec(
                interface_depth_m=950.0,
                top=ElasticLayer(3000.0, 1700.0, 2400.0, "upper"),
                bottom=ElasticLayer(4500.0, 2600.0, 2400.0, "lower"),
            )
        )

    def run_true(self, *, snapshot_stride: int | None = None) -> ForwardShotResult:
        return self.engine.run(
            self.true_medium,
            self.source,
            snapshot_stride=snapshot_stride,
        )

    def medium_with_velocities(
        self,
        *,
        vp: np.ndarray,
        vs: np.ndarray,
        rho: np.ndarray | None = None,
    ) -> ElasticMedium2D:
        if rho is None:
            rho = self.true_medium.rho
        return ElasticMedium2D(grid=self.grid, vp=vp, vs=vs, rho=rho)

    def smooth_initial_medium(
        self,
        *,
        transition_half_width_m: float = 100.0,
    ) -> ElasticMedium2D:
        if transition_half_width_m <= 0.0:
            raise ValueError("transition_half_width_m must be positive.")
        z = np.asarray(self.grid.z, dtype=np.float64)[None, :]
        w_bottom = 0.5 * (
            1.0
            + np.tanh(
                (z - self.model_spec.interface_depth_m) / transition_half_width_m
            )
        )
        top = self.model_spec.top
        bottom = self.model_spec.bottom
        vp_1d = top.vp_m_s + w_bottom * (bottom.vp_m_s - top.vp_m_s)
        vs_1d = top.vs_m_s + w_bottom * (bottom.vs_m_s - top.vs_m_s)
        vp = np.ascontiguousarray(np.broadcast_to(vp_1d, self.true_medium.shape))
        vs = np.ascontiguousarray(np.broadcast_to(vs_1d, self.true_medium.shape))
        return ElasticMedium2D(
            grid=self.grid,
            vp=vp,
            vs=vs,
            rho=self.true_medium.rho,
        )

    def summary(self) -> str:
        g = self.grid
        return (
            "TwoLayerFWIProblem\n"
            f"  grid              : {g.nx} x {g.nz}\n"
            f"  dx/dz             : {g.dx:.3f} / {g.dz:.3f} m\n"
            f"  dt / nt           : {g.dt:.6e} s / {g.nt}\n"
            f"  duration          : {g.nt * g.dt:.3f} s\n"
            f"  interface         : {self.model_spec.interface_depth_m:.1f} m\n"
            f"  upper Vp/Vs/rho   : {self.model_spec.top.vp_m_s:.0f} / "
            f"{self.model_spec.top.vs_m_s:.0f} / {self.model_spec.top.rho_kg_m3:.0f}\n"
            f"  lower Vp/Vs/rho   : {self.model_spec.bottom.vp_m_s:.0f} / "
            f"{self.model_spec.bottom.vs_m_s:.0f} / {self.model_spec.bottom.rho_kg_m3:.0f}\n"
            f"  source x/z        : {self.source_spec.x_m:.1f} / {self.source_spec.z_m:.1f} m\n"
            f"  source f0/theta   : {self.source_spec.f0_hz:.1f} Hz / "
            f"{self.source_spec.theta_deg:.1f} deg\n"
            f"  receivers         : {self.receivers.nrec}\n"
            f"  dCh / GL          : {self.das_spec.channel_spacing_m:.2f} / "
            f"{self.das_spec.gauge_length_m:.2f} m\n"
            f"  sponge            : {self.solver_spec.n_boundary} cells, "
            f"gamma={self.solver_spec.gamma_s:.1f}\n"
            "  backend           : numba_fused only"
        )
