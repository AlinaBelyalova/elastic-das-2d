# Staggered-grid elastic solver (NumPy / Numba)

In short: 2-D isotropic Virieux velocity-stress solver on a staggered grid
with leapfrog time stepping, NumPy and Numba backends, moment-tensor sources,
receiver sampling, a DAS operator and an FWI layer. SAFOD forward
(`scripts/safod/run_forward.py`, backend `numba_fused`) and the FWI checks
are built on it. The separate SBP-SAT prototype is described in
[docs/sbp_sat/method.md](../sbp_sat/method.md); the current stage and next
steps are in [docs/status.md](../status.md). The list of checks is in
[scripts/validation/README.md](../../scripts/validation/README.md).

This document restates the code. The source of truth is the headers and
docstrings of the modules listed in each section.

## Formulation and discretization

Code: `src/elastic/solver_numpy.py`, `src/elastic/solver_numba_fused.py`,
`src/common/grid.py` (`Grid2D`), `src/common/model.py` (`ElasticModel2D`).

- First-order velocity-stress equations in the (x, z) plane, fields
  `vx, vz, sxx, szz, sxz`, arrays `[ix, iz]` of shape `(nx, nz)`. Index `iz`
  increases with depth: the free surface is the top boundary.
- Virieux (1986) staggered grid, requires `dx == dz = h`
  (otherwise `ValueError`):

  | Field | Node position |
  | --- | --- |
  | `sxx`, `szz` | `(i*h, j*h)` |
  | `vx` | `((i+1/2)*h, j*h)` |
  | `vz` | `(i*h, (j+1/2)*h)` |
  | `sxz` | `((i+1/2)*h, (j+1/2)*h)` |

- Leapfrog, second order in time: `t_sigma[n] = n*dt`,
  `t_v[n] = (n+1/2)*dt`; step `sigma^n, v^(n-1/2) -> v^(n+1/2) -> sigma^(n+1)`.
- Spatial order `2M`, `M = half_order = 1..4` (orders 2, 4, 6, 8;
  default `half_order=2`). Coefficients: classical Taylor (C-SFD,
  default) or TS-SFD (`use_ts_sfd=True`, least-squares fit to dispersion;
  they depend on the representative Courant number `mean(vp)*dt/dx`).
- Materials: `mu = rho*vs²`, `lambda = rho*(vp² - 2*vs²)`; the solver requires
  `mu > 0`, `lambda >= 0` (i.e. `Vs <= Vp/sqrt(2)`) and `lambda + 2*mu > 0`.
  On staggered nodes: `1/rho` at `vx` and `vz` nodes is the inverse of the
  arithmetic mean of the two neighbouring `rho`; `mu_xz` is the harmonic mean
  of the four neighbours.
- Time step: the solver rejects `dt` above the bound
  `max_stable_dt(..., safety=1.0)` (conservative Courant limits for
  each order are set in the code; for TS-SFD the margin is further reduced).

Order of operations within one step (same in NumPy and Numba):

1. velocity update `sigma^n -> v^(n+1/2)`;
2. if `free_surface=True`, velocity extension above the surface (Graves FS1);
3. stress update `-> sigma^(n+1)`;
4. source injection into the updated stresses;
5. if `free_surface=True`, FS1 stress mirroring (imaging);
6. sponge for all five fields;
7. receiver sampling at time `t_v[n]`.

## Backends and run assembly

| Module | Role |
| --- | --- |
| `src/elastic/solver_numpy.py` | `run_elastic_solver_numpy`: baseline NumPy implementation; source only at node `(ix, iz)` |
| `src/elastic/solver_numba_fused.py` | `run_elastic_solver_numba_fused`: production Numba backend with the same API plus `source_injection`; must stay mathematically consistent with NumPy |
| `src/elastic/solver_numpy_pointforce.py` | checks only (`pointforce_2d`, `reciprocity`): vertical point force in the `vz` equation, orders 2–8, no free surface (sponge on all sides) |
| `src/elastic/simulator.py` | `run_forward_simulation(..., backend="numba_fused" \| "numpy", free_surface=False)`: sampling, injection, solver, then DAS; returns `(ElasticRunResult, DASResult)` |

Backend accuracy and speed comparison: `scripts.benchmarks.backends`,
`scripts.benchmarks.performance`.

## Free surface: Graves (1996) FS1

Code: `src/elastic/solver_numpy.py` (header, loop blocks 1b, 2 and 3b),
`src/elastic/solver_numba_fused.py` (`fill_velocity_ghosts_free_surface_numba`,
`update_stress_fused_free_surface_numba`, `mirror_stress_ghosts_free_surface_numba`).

- Planar FS1 for the 2-4 scheme. With `free_surface=True` and `half_order != 2`
  both backends raise `NotImplementedError`. Orders 2, 6, 8 are available only
  without a free surface.
- The physical surface is the normal-stress row `iz = M`; nodes
  `iz = 0..M-1` are ghost nodes (air). The top sponge is disabled.
- Velocities above the surface are built from the second-order boundary
  conditions `sigma_zz = 0`, `sigma_xz = 0` after the velocity update
  (at the new half step). At the surface, the vertical derivative in the
  equation for the horizontal normal stress is second order.
- After source injection: `szz[k] = 0`, `szz[k-1] = -szz[k+1]`,
  `sxz[k-1] = -sxz[k]`, `sxz[k-2] = -sxz[k+1]` (`k = M`).
- A moment-tensor source must lie at `iz >= M+1`, otherwise it
  ends up in `szz` at the surface.
- The adjoint operator and gradient have no free surface:
  `src/fwi/adjoint_numba.py` and `src/fwi/gradient_numba.py` reject
  `free_surface=True`.

The old README wording "flat free-surface boundary condition using
ghost nodes" referred to the earlier mixed implementation: ghost velocities
were filled before the velocity update, i.e. at the old half step. It was
replaced by Graves FS1 (ghost nodes remain; the formulas and step order changed);
reasons are in handoff 2026-10-08, §H–§J. Checks: `scripts.validation.elastic.free_surface_graves_fs1_qc`
(algebra and step order), `scripts.validation.elastic.free_surface_normal_incidence_qc`
(reflection at normal incidence).

## Absorbing boundaries: sponge

Code: `make_sponge_mask` in `src/elastic/solver_numpy.py`; the Numba backend
uses the same mask.

- There is no PML. A multiplicative Gaussian sponge is implemented: in a layer
  `n_boundary` cells wide the factor is `exp(-gamma_s*dt*w²)`, `w = (n_boundary - i)/n_boundary`
  (`i = 0` at the outer edge); in corners the factors are multiplied.
- The mask multiplies `vx, vz, sxx, szz, sxz` at the end of each step. Velocities
  and stresses belong to different times, so the header calls this an
  O(dt) approximation.
- With `free_surface=True` the top is not damped.
- `n_boundary > M` is required: the difference operators do not update the
  `M` outermost cells. Defaults in the solvers and `run_forward_simulation`:
  `n_boundary=40`, `gamma_s=300.0`.
- The source must lie outside the sponge: `ix` in `[n_boundary, nx - n_boundary)`,
  `iz` in `[n_boundary, nz - n_boundary)`; with `free_surface=True` the minimum
  `iz` is `M+1`.
- In some scripts and in `src/das/receivers.py` the sponge width is called
  `n_pml`; it is the same sponge, not a PML. In `receivers.py` the `n_pml`
  parameter (default 0) is used only to check that channels do not lie in a
  layer of that width at any of the four boundaries: this check does not know
  about `free_surface` and rejects channels with `iz < n_pml` at the top boundary too.

Tuning `n_boundary` and `gamma_s`: `scripts.validation.elastic.boundary_absorption`.

## Source

Code: `src/elastic/source.py`, `src/elastic/source_spreading.py`,
`src/elastic/source_injection.py`.

- Moment tensor `MomentTensor2D(Mxx, Mzz, Mxz)` following the
  Aki & Richards (2002), ch. 3 convention. Base double couple
  `M0 * [[0, 1], [1, 0]]` (normal `z`, slip `x`); `theta_deg` rotates the
  fault in the x–z plane (`build_rotated_double_couple_2d`). Isotropic
  tensor: `build_isotropic_source_tensor_2d`. Source construction:
  `build_source_2d`, `build_dc_source`, `build_isotropic_source`.
- Units: N·m in the sense of the 2-D model convention; matching 2-D and 3-D
  amplitudes (real earthquake moments) is marked in the code as a separate
  issue.
- Time function `SourceTimeFunction`: `values[n]` is multiplied by the tensor
  component (in `run_forward_simulation`) and by `dt/(dx*dz)` (in the solver) and
  added to the stresses at step `n`, i.e. to `sigma^(n+1)`. The mode is set by
  `source_time_mode`; with `source_time_mode=None` the choice is made by
  `derivative_order`: 0 — `legacy_ricker` (default), 1 —
  `legacy_ricker_derivative`. The `build_source_time_function` docstring names
  `ricker_moment` as the physical (recommended) mode.
  - `source_time_mode="ricker_moment"`: `values = -dW/dt` at time
    `(n+1/2)*dt`, where `W(t)` is the Ricker moment history (`S_M = -dM_phys/dt`).
    In this mode the receiver velocity is comparable to the analytical velocity
    for the moment history `W(t)`.
  - `legacy_ricker`, `legacy_ricker_derivative`: kept for
    reproducibility; these are generic stress-rate wavelets, not a
    physical Ricker moment history.
- Distribution over the grid (`spreading`):
  - `"nearest"` (default): `sxx`, `szz` go to a single node; `sxz` gets 0.25
    in each of the four surrounding `sxz` nodes, centred on the source node;
  - `"bilinear"`: bilinear weights from the physical source coordinates, each
    component on its own subgrid (`Mxx -> sxx`, `Mzz -> szz`, `Mxz -> sxz`).
    The `source.py` header designates this mode for SAFOD and future FWI.
- `source_injection.py` converts the source into six arrays of shape `(4,)`; a
  single injection kernel serves both modes. The bilinear mode exists only in
  `numba_fused`; `numpy` falls back to `"nearest"` with a `UserWarning`.

Checks: `scripts.validation.elastic.source_radiation`,
`moment_tensor_linearity`, `magnitude_scaling`, `moment_tensor_analytical`.

## Receivers and sampling

Code: `src/das/receivers.py`, `src/elastic/sampling.py`.

- `Receivers2D`: channel-centre coordinates `x, z`, unit tangent
  `tx, tz` in the x–z plane, uniform along-cable coordinate `s`
  (measured depth works for borehole geometry). The `ix, iz` fields
  are deprecated and kept for compatibility.
- `build_das_cable_from_waypoints`: synthetic uniform channels along a
  polyline; `build_receivers_from_channel_centres`: given field channel
  centres without resampling.
- `build_receiver_sampling` computes bilinear weights for `vx` and `vz`
  on their staggered subgrids once; only arrays are used inside the loop.
  `scatter_adjoint` is the exact algebraic transpose of the sampling.
  Receiver records correspond to times `t_v[n] = (n+1/2)*dt`.

## DAS operator

Code: `src/das/strain.py`.

- `compute_axial_strain_rate(vx, vz, receivers, gauge_length_m)` (from velocities)
  and `compute_axial_strain(ux, uz, ...)` (from displacements):

  ```text
  eps_dot_ss(s) = ( [vx(s+L/2) - vx(s-L/2)] * tx(s)
                  + [vz(s+L/2) - vz(s-L/2)] * tz(s) ) / L
  ```

  `tx, tz` are taken at the gauge centre.
- Rule: first difference the Cartesian components over gauge `L`, then project
  onto the tangent at the gauge centre. In the reverse order a rigid translation
  gives a nonzero response on a curved cable.
- `L` is a continuous parameter and need not be a multiple of the channel
  spacing: values at `s ± L/2` are obtained by linear interpolation along the
  uniform channel array. Only channels whose `s ± L/2` lies within the
  cable are output (`DASResult.channel_indices`).
- Preserved exactly: a rigid translation gives zero; a field linear in `s`
  is recovered exactly (checked in `_self_test`).
- `das_adjoint` is the exact transpose (dot-product test in `_self_test`).
- In `run_forward_simulation` DAS is computed after the run from `receiver_vx`,
  `receiver_vz` (strain rate).

## Analytical references

| Module | Contents | Checks |
| --- | --- | --- |
| `src/analytical/pointforce_2d.py` | 2-D Green's function `v_z` for a vertical point force in a homogeneous medium (plane strain, kernel after Kausel 2006) in the frequency domain; `dt/2` shift via the factor `exp(+i*omega*dt/2)` for comparison with leapfrog velocities | `scripts.validation.elastic.pointforce_2d` (orders 2, 4, 6, 8) |
| `src/analytical/moment_tensor.py` | homogeneous full space, symmetric moment tensor; the 2-D solution is the integral of the 3-D point solution along `y`; based on Haipeng Li's `AnalyticalSolution.py` (Aki & Richards, 2002) | `scripts.validation.elastic.moment_tensor_analytical` |

The `pointforce_2d.py` header calls the kernel a "practical analytical reference
candidate": if the convergence plot behaves unexpectedly, first
re-check the kernel itself: factors, signs and the near field.

## FWI layer (`src/fwi`)

| Module | Contents |
| --- | --- |
| `two_layer.py` | OOP layer over the Numba solver (`TwoLayerFWIProblem`, `NumbaElasticDASForward` and specifications of the model, domain, solver, DAS, source); Python objects stay out of hot loops |
| `adjoint_numba.py` | `NumbaElasticDASAdjoint`: exact discrete adjoint operator of the Numba solver for `free_surface=False`; C-SFD and TS-SFD, sponge, bilinear receivers, `nearest`/`bilinear` source; DAS transpose via `das_adjoint`. Backward step: receiver scatter, sponge, source gather, transpose of the stress and velocity updates |
| `gradient_numba.py` | `NumbaElasticDASGradient`: checkpointed gradient with respect to `lambda, mu`, converted to `vp, vs` and `log(vp), log(vs)`; `rho` is fixed; the transpose of the harmonic `mu_xz` is accounted for. `free_surface=True` and TS-SFD are not supported (`NotImplementedError`). Objective function `J = 0.5*dt*dCh*||d_syn - d_obs||²` |
| `misfit.py` | `l2_das_misfit`: `J = 0.5*dt*dCh*sum(w*(syn - obs)²)` and `dJ/d(syn)`; not yet used by the gradient |

Checks: `scripts.validation.fwi.two_layer_numba`,
`two_layer_adjoint_dot_product`, `two_layer_taylor`.

## Checks and context

- All checks, criteria, runtimes and result folders:
  [scripts/validation/README.md](../../scripts/validation/README.md).
- Environment and running on Sherlock: [docs/setup.md](../setup.md).
- Background (adjoint, gradient, time levels, free-surface
  replacement, Graves FS1 derivation and checks):
  `docs/history/FWI_SAFOD_FULL_HANDOFF_2026-10-08.md`,
  §E–§M. The `docs/history/` files are local and not in Git; the index is
  `docs/history/README.md`.
