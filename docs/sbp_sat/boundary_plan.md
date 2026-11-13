# Boundary audit and implementation plan

Current user-directed choice (2026-10-28): restore the original Gaussian
absorbing strips, keep SBP/free surface and one fully displayed grid.
See [sponge.md](sponge.md). PML development is deferred. The audit and PML
plan below are retained as the history of the earlier proposal, not the
current event configuration or a claim that a PML change is required.

Status: 2026-10-28. The current solver has an energy-stable local impedance
condition, not PML. This audit does not claim that the observed real-event
reflections have already been removed.

## Finding and scope

The transition to SBP-SAT replaced the legacy volume sponge with the local
law `sigma*n = -rho*Vp*v_normal*n - rho*Vs*v_tangent*t` at the three artificial
faces. The SBP adapter sets sponge width and gamma to zero. Numba preserves
the same numerical updates; it is not a different absorbing boundary.

The signs give outward energy loss. The free SAT force and constitutive
maps satisfy `QF + B.T W = 0`; the impedance operator has `QD` symmetric
positive semidefinite. The trapezoidal velocity damping and leapfrog stress
update satisfy the documented exact modified-energy dissipation identity.
The source/receiver transpose, model-dependent boundary VJP and full-state
checkpoint replay have independent tests. Fresh manufactured-model results:

| Check | Result |
| --- | --- |
| Maximum relative discrete energy-balance error | 2.19e-16 |
| Source-to-data dot-product relative error | 2.53e-15 |
| Checkpoint versus full-history objective/data/gradients | bitwise equal |
| Central finite-difference model derivative relative error | 1.12e-10 |
| Checkpoint model Taylor rates | 2.00113, 2.00057, 2.00028 |

This confirms consistency of the implemented discrete problem. It does not
prove that the chosen truncated problem approximates an unbounded medium
with sufficiently small error.

The local law is exact for a homogeneous outgoing plane P/SV wave at normal
incidence in the continuum, and generally reflects oblique waves. The new
`scripts.validation.sbp_sat.impedance_reflection` independently solves the
two traction equations for reflected P/SV amplitudes. It checks both boundary
residual and incident = reflected + absorbed energy flux, including an
evanescent converted P wave above the SV critical angle.

Example: Vp=5000 m/s, Vs=3000 m/s, rho=2600 kg/m3. Angles are measured from
the outward boundary normal; these are **energy fractions**, not amplitudes:

| Angle | Incident P reflected energy | Incident SV reflected energy |
| --- | --- | --- |
| 0 deg | 0 | 0 |
| 30 deg | 0.0678% | 3.663% |
| 60 deg | 3.402% | 11.458% |
| 75 deg | 19.620% | 35.377% |

The audit is a continuum boundary-law calculation, not a discrete wave-packet
test and not a measurement on the SAFOD model. It demonstrates why refining
the grid or compiling the same law cannot remove its oblique reflection.

```bash
"$FWI" -m scripts.validation.sbp_sat.impedance_reflection
```

Not every curved wavefront is a boundary reflection. The physical model can
reflect/convert waves at the free surface and at actual material variations.
The cable and fault-prior drawing do not impose mechanical interfaces.
Origin of a particular event branch must be determined from the full saved
wavefield and quantitative boundary tests, not assigned from one frame.

## One domain

For SBP, `SBP_X_MIN_M`, `SBP_X_MAX_M`, `SBP_Z_MAX_M` are the actual outer
faces. The top is z=0. The current default domain remains
`[-2100,4100] x [0,5600]` m at 5 m spacing, 1241x1121 N-N nodes.
No hidden extension or inner reference region is introduced. Both the model
figure and GIF show this entire grid. Native Vz samples remain at half-depth
nodes, 1241x1120; visualization does not move them to the normal-stress nodes.
The model builder receives exactly these bounds with zero extra bottom cells.
Existing saved NPZ/snapshots can be rendered without propagation.

For the planned PML, absorbing strips will occupy explicitly configured
widths **inside these bounds**, at left/right/bottom only. They remain part
of the same computational array, with modified equations in the strips.
There is no top layer. At a top-side corner the free-surface condition and
the side PML must coexist consistently. No internal rectangular reference
overlay is needed. A PML cannot make its own strip simultaneously act as
ordinary undamped physical material; receivers and the source, including
their interpolation/gauge supports, must remain outside the strips.
Reject unsuitable bounds/widths instead of moving or clipping geometry.

## Chosen approach and papers

Use an unsplit complex-frequency-shifted PML (CFS-PML), with auxiliary
memory variables and SAT boundary closure derived for this staggered grid.
The frequency-domain derivative stretch has the form

`d/dxi -> [1 / (kappa_xi + d_xi/(s + alpha_xi))] d/dxi`,

where `s` is Laplace frequency, kappa is dimensionless, and d/alpha have
units 1/s. Away from the strips kappa=1, d=0, and all zero-initialized memory
variables must leave the present elastic equations unchanged. The direction
of every derivative contribution must stay explicit; damping whole fields
after a step is not this formulation.

Primary references:

1. Komatitsch and Martin (2007), *An unsplit convolutional perfectly matched
   layer improved at grazing incidence for the seismic wave equation*,
   Geophysics 72, SM155-SM167, DOI
   [10.1190/1.2757586](https://doi.org/10.1190/1.2757586).
   Their elastic staggered-grid benchmarks motivate the frequency shift,
   particularly for grazing waves. They do not certify our SAT coupling.
2. Duru, Kozdon and Kreiss (2015), *Boundary conditions and stability of a
   perfectly matched layer for the elastic wave equation in first order
   form*, JCP 303, 372-395, DOI
   [10.1016/j.jcp.2015.09.048](https://doi.org/10.1016/j.jcp.2015.09.048).
   The SAT boundary procedure must extend to the auxiliary equations;
   dissipative termination and corner consistency matter for stability.
3. Duru and Kreiss (2022), *The perfectly matched layer (PML) for hyperbolic
   wave propagation problems: A review*,
   [arXiv:2201.03733](https://arxiv.org/abs/2201.03733).
   See the complex stretch, auxiliary equations, and discrete SAT extension
   in sections 3 and 5. Constant-coefficient estimates in the review do not
   automatically prove stability of arbitrary variable damping profiles,
   heterogeneous media, our staggered operators or our time integrator.

A finite discrete PML still has residual reflection. Width, smoothness,
frequency shift, timestep, grazing incidence and corner treatment must be
tested. Increasing a damping coefficient alone need not improve the result.

## Ordered implementation gates

1. **Done in this audit:** make one domain explicit, remove the reference
   overlay, preserve the current default physical grid, rerun energy,
   transpose and checkpoint/gradient checks, expose the continuum boundary
   error. This does not alter the absorbing PDE or remove event reflections.

2. **Next: implement and verify PML forward.** Split the present elastic
   spatial map into its x/z contributions, including the corresponding SAT
   lifts, density and constitutive placement. Derive the memory updates and
   compatible outer impedance termination, top free surface and corners.
   First write an independent small matrix/reference realization, then the
   Numba kernels. The old five-field energy/CFL certificate cannot simply be
   advertised as a proof for the enlarged PML system. Write the stability
   argument for the actual augmented spatial and time-discrete operators,
   stating coefficient/profile assumptions explicitly. A small-grid spectrum
   is a diagnostic, not a proof for arbitrary grids or variable media; keep
   theoretical scope separate from empirical validation. Check zero damping,
   accuracy order, spectrum/amplification and long-time behavior.
   Use smooth profiles sampled at each native grid location. Widths and
   profile parameters are explicit fixed numerical parameters; choose them
   from measured error, not a promise that an arbitrary cell count suffices.

3. **Forward acceptance before replacing event output.** Run discrete P and
   SV packets at 0/30/60/75 degrees into each side and the bottom, mode
   conversion, bottom corners, and a free-surface/Rayleigh-wave case. Test
   representative 2/5/10/20 Hz content and the actual Ricker band; zero Hz
   has no propagating harmonic-wave reflection coefficient to promise.
   Compare with analytic cases where available and a sufficiently large
   homogeneous benchmark whose return times are excluded from the test.
   Such a benchmark is a validation only, not an inner production domain.
   Measure reflected energy and receiver errors with a common fixed scale,
   no independent trace normalization. Initial target: reflected packet
   energy below 1e-4 for the specified resolved cases, plus convergence with
   layer/grid refinement. This is a project acceptance target, not a paper
   theorem or a result already achieved. Include a heterogeneous-model and
   long-time stability check; preserve the physical free-surface return.
   A single normal-P packet passing is insufficient.

4. **Complete derivative and restart chain.** All memory variables belong to
   the forward state and to its exact reverse. Derive their discrete
   transpose and material VJP, including impedance terms. Freeze PML profile
   parameters during model differentiation and serialize that policy; if a
   profile is recomputed from the model its derivative cannot be omitted.
   Update full-state checkpoints, replay and memory estimates for the extra
   state. Require source/DAS dot products, directional finite differences,
   second-order Taylor remainders and checkpoint/full-history parity again.
   The current 9.073 GiB event history estimate is for the impedance solver
   and must not be reused as a PML estimate.

5. **One event, then proceed to FWI.** Rerun NC75379261 with the same material
   arrays, source/STF, registered geometry, spacing, duration and DAS
   convention. Revalidate the timestep for the new scheme. Keep the existing
   forward/compare directory hierarchy, full-domain GIF and shared Zhang/Lin
   3-D guide overlays. Store a distinct boundary-scheme/version and the full
   PML profiles/widths in provenance so an old impedance package cannot be
   silently treated as PML. Quantify remaining boundary contamination in
   the relevant time window. Start real-data gradient work only after these
   forward and derivative gates pass.

## Files and dependency order

| Layer | Current files | PML work |
| --- | --- | --- |
| Geometry/material grid | `scripts/safod/settings.py`, `scripts/safod/run_forward.py` | one set of bounds; explicit strip widths; support checks |
| SBP pair, quadrature, closures | `src/sbp_sat/operators.py` | retain coefficients, derive directional SAT coupling |
| Reference/time/transpose/VJP | `src/sbp_sat/rectangular.py`, proposed `src/sbp_sat/pml.py` | add memory state and its actual discrete reverse |
| Compiled forward | `src/sbp_sat/numba_solver.py` | same extended equations as reference; parity checks |
| Model/source/DAS adapter | `src/sbp_sat/simulator.py` | profile policy, native support, new storage/provenance |
| Restart | `src/sbp_sat/checkpointing.py` | all auxiliary fields, replay and cost |
| Event sensitivity diagnostic | `scripts/validation/sbp_sat/event_boundary_control.py` | recognize new boundary scheme or reject it; never replay PML as impedance |
| Validation | `tests/test_sbp_sat_*.py`, `scripts/validation/sbp_sat/` | forward absorption first, then transpose/gradient/restart |

Keep 3-D travel-time guides, catalogue geodesy, observed data processing,
source interpretation and DAS units unchanged in this boundary work.
