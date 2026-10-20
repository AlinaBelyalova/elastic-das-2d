# SBP-SAT mathematical specification

In short: the mathematical specification of the NumPy SBP-SAT prototype in `src/sbp_sat/` (equations, grids, SBP operators, SAT, time levels, adjoint operator and gradient) and the mapping of formulas to functions and checks.
Check commands are in [validation.md](validation.md), the table of all checks is in [scripts/validation/README.md](../../scripts/validation/README.md), the current stage is in [docs/status.md](../status.md).

This is the specification of the existing NumPy reference in `src/sbp_sat/`.
The controlled problem is 2D isotropic velocity-stress elasticity, periodic
in x, with traction-free top and bottom boundaries.

## Equations and conventions

Depth z increases downward; the top is z=0. Arrays use `[ix, iz]` ordering.
Coordinates and grid spacing are in metres, time in seconds, velocity in m/s,
stress and elastic moduli in Pa, density in kg/m³. With no source:

```text
rho * vx_t  = sxx_x + sxz_z
rho * vz_t  = sxz_x + szz_z
      sxx_t = (lambda + 2 mu) * vx_x + lambda * vz_z
      szz_t = lambda * vx_x + (lambda + 2 mu) * vz_z
      sxz_t = mu * (vx_z + vz_x)
```

The free-surface condition is zero traction: `szz = sxz = 0` at both z ends.
It is imposed weakly through SAT; the implementation does not overwrite
boundary stresses with zero.

## Native grids and material coefficients

Let `n = n_intervals`, `h = dz`. The z N-grid has `n+1` points at `j*h`,
`j=0,...,n`; the M-grid has `n` points at `(j+1/2)*h`, `j=0,...,n-1`.
The constructor requires `n >= 8`. In periodic x, both grids have `nx` points:
`x_N = i*dx`, `x_M = (i+1/2)*dx`, with period `nx*dx` and no repeated endpoint.

| Quantity | Grid | Array shape |
| --- | --- | --- |
| `sxx`, `szz`, `vp`, `vs`, `rho` | N-N | `(nx, n+1)` |
| `vx`, `rho_vx` | M-N | `(nx, n+1)` |
| `vz`, `rho_vz` | N-M | `(nx, n)` |
| `sxz`, `mu_MM` | M-M | `(nx, n)` |

`prepare_model` computes `lambda = rho*(Vp² - 2*Vs²)`,
`l2m = rho*Vp²`, `mu = rho*Vs²` on N-N. It checks positive finite inputs,
`Vp > Vs` and the branch restriction `lambda >= 0`. Density is averaged
onto each velocity grid with two neighboring N-N values. Shear modulus is
mapped to M-M by a four-point arithmetic average, periodic in x and bounded
in z. `_average_NN_to_MM_transpose` implements the transpose of this map.
These are the prototype's material choices, including at discontinuities.

## SBP operator contract

`operators.py` stores `H_N` and `H_M` as vectors of diagonal weights. In the
following equations, H denotes the corresponding diagonal matrix.
`D_MN = D_M_to_N` has shape `(n+1,n)`; `D_NM = D_N_to_M` has shape `(n,n+1)`.
The pair must satisfy

$$
H_N D_{MN} + D_{NM}^{T} H_M = -e_L p_L^T + e_R p_R^T.
$$

`e_L`, `e_R` select the first/last N-grid value; `p_L`, `p_R` project M-grid
values to the physical boundary. Their nonzero coefficients are
`p_L[:3] = (15/8, -5/4, 3/8)`, with the reflected sequence at the right end.
The derivative has fourth-order interior and second-order boundary accuracy.
The norm weights scale with h, derivatives with `1/h`, projections are
dimensionless. The coefficient reference is
[Gao et al., equations 12–14](https://arxiv.org/pdf/1802.06123).

For periodic x, `H_x = dx*I` and `(D_MN_x).T = -D_NM_x`.
For arrays stored as `[ix,iz]`, a z derivative is applied as `f @ D.T`;
its Euclidean transpose is `q @ D`. This array-layout transpose is distinct
from the physical SBP integration-by-parts identity.

The constructor checks the SBP residual against `5e-13`. Polynomial accuracy,
positive norms, boundary traces, and several grid sizes also need independent
small tests; the existing operator QC checks one SBP grid and one 2D energy case.

## SAT terms and energy cancellation

For one fixed x index, write z profiles as column vectors. Before division
by density, the terms added by `velocity_rhs` are

$$
SAT_x = H_N^{-1}(e_L p_L^T - e_R p_R^T)\sigma_{xz},
$$
$$
SAT_z = H_M^{-1}(p_L e_L^T - p_R e_R^T)\sigma_{zz}.
$$

Thus the upper coefficient is `+1`, the lower is `-1`. This follows from the
left/right terms in our SBP identity with depth positive down. The elastic
SAT construction is described by [Gao and Keyes, equations 11–13](https://arxiv.org/pdf/1903.06795);
their top/right-endpoint convention must be converted to our top/left endpoint.

For the 1D shear subsystem, with R the positive diagonal density on N and
G the positive diagonal shear modulus on M,

$$
R\dot v = D_{MN}\tau + SAT_x,\qquad
\dot\tau = G D_{NM}v,
$$
$$
E_h = \tfrac12 v^T H_N R v + \tfrac12\tau^T H_M G^{-1}\tau.
$$

The derivative terms give
`dE/dt = -(e_L.T @ v)*(p_L.T @ tau) + (e_R.T @ v)*(p_R.T @ tau)`.
The SAT contribution is exactly its negative, so `dE/dt = 0` for arbitrary
fields in this source-free, closed semidiscrete problem. The normal pair
uses the same identity with the roles of N and M exchanged.

In 2D, the energy includes `dx*H_N` or `dx*H_M` on the native grids. The
normal-stress compliance is the inverse of
`[[l2m, lambda], [lambda, l2m]]`; the shear energy density is `sxz²/(2*mu_MM)`.
In block notation, `v_dot = A*sigma`, `sigma_dot = B*v`,
`M*A + B.T*W = 0`, where M contains density and quadrature and W contains
compliance and quadrature. `semidiscrete_energy_rate` evaluates the resulting
energy derivative. The 2D energy has units J/m of out-of-plane length.

## Time levels and source convention

The stored state is `(v^(n-1/2), sigma^n)`. Both `advance_one_step` and
`run_forward` implement this order:

```text
v^(n+1/2)   = v^(n-1/2) + dt * A(sigma^n)       # includes SAT
sample data = R_receiver(v^(n+1/2))              # run_forward only
sigma^(n+1) = sigma^n + dt * B(v^(n+1/2))
sxx[ix,iz] += dt * mxx * stf[n]
szz[ix,iz] += dt * mzz * stf[n]
```

`StressPointSource` injects normal stresses only, on one N-N node. The
products `mxx*stf`, `mzz*stf` have units Pa/s. No cell-volume or seismic-moment
normalization is supplied by this prototype. Source-to-SAFOD amplitude
mapping remains an integration task. Data samples precede the source
injection of the same iteration; the final source sample therefore affects
the final state but not the returned receiver history.

For the source-free stored state, the conserved leapfrog quadratic is

$$
E_{LF}=\tfrac12 v^TMv+\tfrac12\sigma^TW\sigma
       +\tfrac{\Delta t}{2}v^TMA\sigma.
$$

Ordinary energy evaluated using fields at different time levels is not this
invariant. The QC also evaluates physical energy with
`v_centred = v + (dt/2)*A*sigma`. The cross term and its sign follow from
the stored time levels above. Conservation alone does not establish a stable
choice of dt: positivity of the quadratic and a timestep bound must also be
checked. Existing experiments use conservative dt values; they do not prove
a sharp CFL bound for every model.

## Exact adjoint and model gradient

For fixed material and zero initial state, `F: stf -> receiver data` is linear.
`run_adjoint_to_source` implements its Euclidean transpose. A fixed nonzero
initial state adds an affine contribution to the forward data; F still
denotes the source-dependent linear part.

The reverse order is source transpose, stress-update transpose, receiver
scatter, velocity-update transpose. The last includes all SAT terms.
`scatter_receivers_adjoint` accumulates repeated receiver contributions.
This adjoint is specific to the SBP-SAT forward discretization.

The objective is `J = 0.5*dt*sum((d_syn-d_obs)**2)`. Thus its receiver-space
adjoint input is `dt*(d_syn-d_obs)`. Density and dt remain fixed. If Q maps
N-N material to M-M, the gradient first applies `g_mu_NN = Q.T*g_mu_MM`, then

```text
g_log_Vp = 2*rho*Vp² * (g_lambda + g_l2m)
g_log_Vs = -4*rho*Vs² * g_lambda + 2*rho*Vs² * g_mu_NN
```

These are gradients in the ordinary array inner product used by the Taylor
test. The reference stores every velocity half-step for this calculation;
checkpointing is deferred.

## Formula to code to check

All functions below except the first row currently live in `elastic2d.py`.
Validation module names have the prefix `scripts.validation.sbp_sat.`.

| Responsibility | Function | Existing QC |
| --- | --- | --- |
| D, H, boundary projections | `staggered_sbp_4_2`, `sbp_residual` | `operator_qc` |
| Material and grid transfer | `prepare_model`, `_average_NN_to_MM_transpose` | Covered within energy and Taylor checks |
| Divergence and SAT | `velocity_rhs` | `operator_qc` (energy), `normal_incidence_qc`, `oblique_qc` |
| Constitutive stress update | `stress_rhs` | `operator_qc` (energy), `normal_incidence_qc`, `oblique_qc` |
| Time stepping | `advance_one_step`, `run_forward` | `time_energy_qc`, `normal_incidence_qc`, `adjoint_dot_product` |
| Receiver and timestep transpose | `scatter_receivers_adjoint`, `run_adjoint_to_source` | `adjoint_dot_product` |
| Parameter derivatives | `objective_and_gradient_log_vp_vs` | `taylor` |
| Semidiscrete energy | `semidiscrete_energy_rate` | `operator_qc` |

## Checks

Run commands live only in [validation.md](validation.md), together with the
full oblique and time-energy acceptance criteria. The table of all checks,
with the pass criterion of each and runtimes where measured, is
[scripts/validation/README.md](../../scripts/validation/README.md); saved runs
and naming rules are in [results/README.md](../../results/README.md), dated
results and the next task in [docs/status.md](../status.md).

## Deferred integration

The reference has no absorbing boundaries, arbitrary off-grid sources,
production receiver geometry, finite-gauge DAS, Numba kernels, checkpointing,
or SAFOD loaders. Adding each operator requires a matching transpose and
new validation of the complete chain. Planned work and its order are
maintained in [docs/status.md](../status.md).
