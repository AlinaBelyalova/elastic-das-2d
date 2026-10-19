# SBP-SAT validation: checks and acceptance criteria

Кратко: единственное место с командами проверок SBP-SAT, правила записи результатов и критерии приёмки тестов oblique и time-energy.
Формулы — в [method.md](method.md), список всех проверок со временем — в [scripts/validation/README.md](../../scripts/validation/README.md), текущий этап — в [docs/status.md](../status.md).

## Provenance

The oblique and time-energy scripts were added as the QC v2 add-on to the
standalone NumPy SBP-SAT reference in `src/sbp_sat/`. They cover oblique
reflection and fully discrete energy conservation. The original add-on
introduced validation scripts only; it did not modify the solver or the four
original QC scripts (QC 1–4: `operator_qc`, `normal_incidence_qc`,
`adjoint_dot_product`, `taylor`).

Two archived runs are kept in `results/validation/sbp_sat/`:
`reference_v2/` (the add-on authoring-environment results) and
`sherlock_20261016/` (the local Sherlock run of 2026-10-16). Never write
into them. The historical ZIP/PATCH packages of the add-on are stored
outside the project, in
`/home/groups/ettore88/alina/project_archives/Elastic_das_project/cleanup_20261017T004124Z.tar.gz`
(`removed/sbp_sat_qc_addon_v2.zip`, `removed/sbp_sat_qc_addon_v2.patch`).
Older handoffs and other history: `docs/history/README.md`
(local only: `docs/history/` is excluded from Git).

## Run the checks

Run from the project root with the `fwi` environment active (see
[docs/setup.md](../setup.md)). On Sherlock, run inside a compute allocation,
not on a login node.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m scripts.validation.sbp_sat.operator_qc
python -m scripts.validation.sbp_sat.adjoint_dot_product
python -m scripts.validation.sbp_sat.taylor
python -m scripts.validation.sbp_sat.normal_incidence_qc
python -m scripts.validation.sbp_sat.time_energy_qc \
  --output-dir results/validation/sbp_sat/<label>_<YYYYMMDD>/time_energy --plot
python -m scripts.validation.sbp_sat.oblique_qc \
  --output-dir results/validation/sbp_sat/<label>_<YYYYMMDD>/oblique --plot
```

- `<label>_<YYYYMMDD>` is a placeholder for a new run folder (replace it
  before running): a label plus the run date, the same pattern as
  `sherlock_20261016`. Naming rules are in
  [results/README.md](../../results/README.md).
- `time_energy_qc` and `oblique_qc` refuse a non-empty `--output-dir` unless
  `--overwrite` is given. Never use `--overwrite` on `reference_v2/` or
  `sherlock_20261016/`.
- The first four checks print to stdout only. `time_energy_qc` and
  `oblique_qc` write `summary.json` and `traces.npz` only when `--output-dir`
  is supplied; `--plot` requires `--output-dir` and adds a PNG.
- Every script raises an error and exits nonzero on a QC failure.
- NumPy is required; Matplotlib only with `--plot`. Thread limits avoid
  oversubscription for these small dense reference operators. Runtime depends
  on the BLAS build and CPU; measured runtimes are in
  [scripts/validation/README.md](../../scripts/validation/README.md).

The normal-incidence QC gates strictly decreasing waveform error under
refinement (h = 10, 5, 2.5 m) for both P and SV; its printed peak and traction
diagnostics have no separate acceptance threshold. The gates of the other
three quick checks are listed in
[scripts/validation/README.md](../../scripts/validation/README.md).

## What the oblique test checks

Cases: incident P at 30 and 60 degrees; incident SV at 20 and 30 degrees.
Angles are measured from the vertical. Material properties are Vp=5000 m/s,
Vs=3000 m/s and rho=2600 kg/m^3. SV cases are below the critical angle
arcsin(Vs/Vp), approximately 36.87 degrees, so both reflected modes propagate.

The exact solution is a monochromatic incident wave plus reflected P and SV
waves. All waves have the same horizontal slowness and temporal frequency.
The horizontal domain contains one wavelength, making the field exactly
periodic in x, as required by this prototype. Frequencies depend on angle,
but are fixed across grid refinements within each case.

This is a **total-field initial-value evolution benchmark**. Both reflected
waves are included at t=0, and the solver then evolves the fields freely.
There is no boundary forcing, manufactured volume forcing, analytic field
resetting, or source injection after initialisation. This checks preservation
and convergence of an exact oblique free-surface solution. It does not test
generation of reflected wave packets from an incident-only initial condition.

Each component is evaluated on its actual staggered grid. Stresses start at
t=0 and velocities at t=-dt/2. Surface velocities are compared at their
half-time samples, and M-grid surface values use the SBP trace projection.

### Independent continuum reference

Use the velocity convention

    v = a exp[i omega (p x + q z - t)],

with z positive downward. The slowness is s=(p,q); upward incident waves have
q<0, and downward reflected waves have q>0. Polarisations are

    a_P  = Vp (p, q),
    a_SV = Vs (q, -p).

For unit particle-velocity amplitude, the corresponding stress amplitude is

    sigma = -lambda (s dot a) I - mu (s a^T + a s^T).

Two continuum traction equations solve for the reflected velocity amplitudes:

    [t_P_down, t_SV_down] [R_P, R_S]^T = -t_incident,
    t = (sigma_xz, sigma_zz).

This calculation uses no discrete derivative or SAT coefficients. The
reference is checked against closed-form expressions and reflected energy
flux balance. Signs of R depend on the stated polarisation convention;
at normal incidence both same-mode coefficients are -1 in this convention.
A velocity-amplitude conversion coefficient slightly exceeding 1 is allowed:
energy-flux ratios also contain the wave speed and normal direction cosine.

At late times, four continuum modes (up/down P/SV) are fitted to the numerical
velocity field over z=30..140 m. The fit uses the native grids, removes the
known temporal phase and separates the +kx Fourier mode. Reflection ratios
are relative to the fitted incident mode. Complex errors therefore include
phase as well as amplitude. The unwanted incident mode and fit residual are
checked separately.

### Boundaries and acceptance

The prototype has free surfaces at BOTH z ends. The comparison is near the
top and ends after two periods. An 800 m depth and a travel-time guard isolate
the lower boundary. A separate authoring check repeated the longest case at
1000 m depth and compared surface traces. The guard is an engineering buffer,
not a claim of strict discrete finite propagation speed.

Default h values are 10, 5, 2.5 and 1.25 m, with dt=0.15 h/Vp. Surface-waveform
and traction errors must decrease at every refinement. At the finest grid,
each of the following must be below 0.01:

- relative L2 error of the two-component surface waveform;
- maximum norm of the complex traction amplitude divided by rho*c_inc*V0;
- maximum absolute complex error of either reflected amplitude ratio;
- unwanted incident amplitude relative to the intended incident mode;
- relative modal-fit residual.

V0=1 m/s, so the traction diagnostic is explicitly dimensionless. It measures
a norm of Fourier amplitudes, not an assigned pointwise-zero stress. The 1%
gate is a chosen benchmark tolerance, not a universal SBP requirement.
The finest spacing is a validation choice for this 29--50 Hz benchmark,
not a prescribed production SAFOD grid spacing.
The first three spacings alone failed this traction gate for several cases;
1.25 m was added without relaxing the threshold. `--hs` permits other
refinements; at least three distinct spacings are required, and all acceptance
gates still apply. Do not infer a global fourth-order method from these results:
space and time are refined together and the boundary closures have lower order.

A negative control disabling the top SAT (in memory only) was also run to
verify that the oblique test detects a boundary implementation error.

## What the time-energy test checks

This test uses the unchanged leapfrog timestep in a smooth heterogeneous
32 by 41 N-grid model, periodic x and free top/bottom boundaries. Density,
Vp and Vs vary spatially. Random initial native-grid fields excite short-wave
and boundary modes. The default duration is 0.5 s, allowing repeated boundary
interactions, with no source or damping.

Write the source-free semidiscrete system as

    v_dot = A sigma,       sigma_dot = B v,
    M A + B^T W = 0,

where M is the density/quadrature kinetic weight and W the
compliance/quadrature stress weight. For the stored leapfrog state
(v^(n-1/2), sigma^n), the invariant is

    E_LF = 0.5 v^T M v + 0.5 sigma^T W sigma
           + 0.5 dt v^T M A sigma.

The cross term is essential. The ordinary continuum-form energy evaluated
from fields at different times is not the leapfrog invariant.

The script also evaluates physical energy with

    v_centred = v^(n-1/2) + (dt/2) A sigma^n.

That centred energy is allowed to vary. The QC requires finite positive
energies, invariant drift below 1e-9, and centred-energy variation below 5%.
It checks this timestep and model; it does not establish a sharp CFL bound
or validate sponge/PML integration.

## Scope after these checks

The source-to-point-receiver transpose and rho-fixed log(Vp)/log(Vs) gradient
remain covered by the original dot-product and Taylor tests. These additions
do not add DAS, Numba, arbitrary sources, absorbing boundaries, SAFOD geometry,
postcritical/evanescent SV-to-P reflection or a production FWI driver.

The current stage and next steps are maintained in
[docs/status.md](../status.md). Formulas and function mapping are maintained
in the [mathematical specification](method.md). Production integration must
validate the complete
source -> propagation -> receiver interpolation -> DAS observation chain.

## Reference

Sanchez-Sesma et al. (2023), Appendix 2, gives free-surface P/SV reflection
relations. This implementation derives its coefficients from the traction
equations and states its own velocity-polarisation convention explicitly:
https://doi.org/10.1186/s40623-023-01780-0
