# Restored Gaussian absorbing strips

The event driver again uses the legacy-style Gaussian sponge on the sides
and bottom. PML is deferred. The bounded SBP-SAT solver, weak free surface,
native grids, source/STF, physical receivers and DAS convention are retained.

## What changed during the original SBP integration

The old `src/elastic/solver_numpy.py:make_sponge_mask` and the fused solver
multiply all five fields by an exponential at the end of each step. The SBP
adapter originally removed that volume sponge and used only the local
impedance law at the outer faces. This changed the absorbing method, not
just the top boundary. The original normal-P check did not demonstrate low
reflection at oblique P/SV incidence. SBP also changed the interior/closure
operators and their native grids; restoring the absorber does not promise
bitwise equality with the legacy synthetic waveform.

Gradual amplitude reduction in boundary strips is the approach introduced
by Cerjan et al. (1985), *A nonreflecting boundary condition for discrete
acoustic and elastic wave equations*, DOI
[10.1190/1.1441945](https://doi.org/10.1190/1.1441945). This implementation
retains our existing code's time-scaled quadratic Gaussian profile; it does
not claim to reproduce every parameter choice in that paper or to be PML.

## One domain and the same historical physical extent

The legacy event calculation used x=-1500..3500 m and z=0..5000 m as its
display/interested window, with 120*5=600 m absorber buffers around its
sides/bottom. Its physical computational extent was therefore already
x=-2100..4100 m, z=0..5600 m. The new SBP settings specify these actual
outer faces directly. No new cells are appended:

| Setting | Value |
| --- | --- |
| x | -2100..4100 m |
| z (positive down) | 0..5600 m |
| dx, dz | 5 m, 5 m |
| Material/normal-stress grid | 1241 x 1121 |
| Native Vz grid | 1241 x 1120 |
| Side sponge strips | -2100..-1500 m and 3500..4100 m |
| Bottom sponge strip | 5000..5600 m |
| Top | Free surface; no top sponge |
| Damping rate gamma | 80 1/s |
| Appended material padding | zero |

The full domain and native Vz axes remain in the saved arrays. At the user's
display preference, the model plot/GIF exclude the absorbing strips: the
default viewport is x=-1500..3500 m, z=0..5000 m. There are no strip overlays,
inner rectangles or boundary-method captions. This changes only the viewport;
no grid cells, geometry or material interpolation rules are changed. Saved
rendering reads the actual run's widths/rate, not today's configuration.

## Native profile and time map

For width w and the chosen actual outer faces, define quadratic ramps

`r_L=max(0,(x_min+w-x)/w)^2`,
`r_R=max(0,(x-(x_max-w))/w)^2`,
`r_B=max(0,(z-(z_max-w))/w)^2`.

The rate is `sigma=gamma*(r_L+r_R+r_B)` in 1/s and the multiplier is
`m=exp(-dt*sigma)`. No r_top exists. At bottom corners the two directional
rates add, as in the old product of masks. The profile is sampled at each
field's actual native coordinates: M-N for vx, N-M for vz, N-N for both
normal stresses and M-M for shear. On N-N it matches the legacy mask at
equal spacing/width; it is not copied blindly onto staggered fields.

Let Phi be the existing complete SBP-SAT step, including source injection.
The implemented step is `y_next=S*Phi(y,a)`, with diagonal fixed S and
observation `d=O*y_next`. Numba and matrix execution implement this same map.
The exterior impedance SAT remains as a dissipative termination after the
wave has traversed the sponge. Setting the rate or both widths to zero recovers
the impedance-only reference.

This is the old **post-step first-order splitting in the absorbing strips**.
It is not Strang splitting, and is not advertised as second-order time
accuracy for the damped PDE. The undamped SBP update is unchanged. Taylor
rates of two below concern accuracy of the model derivative, not the time
order of the sponge. The sponge is an artificial numerical absorber, not a
physical intrinsic-attenuation/Q model.

## Energy and stability scope

With positive materials, S contracts the instantaneous physical elastic
energy: both normal stresses have the same multiplier, so their coupled
compliance block contracts as well. The old modified leapfrog energy has
a derivative cross term; its impedance-only exact dissipation identity must
not be copied unchanged onto the filtered step.

More precisely, let H be the physical-energy norm and E the positive
modified-energy norm of the core, and let `eta=dt/dt_limit<1`. The existing
core bound implies `(1-eta)H <= E <= (1+eta)H`. Thus, for
`C=sqrt((1+eta)/(1-eta))`,

`||S-I||_E <= C ||S-I||_H <= C*dt*sigma_max`,

and the core is nonexpansive in E. Repeated filtered steps consequently
have the finite-time bound `||T^n||_E <= exp(C*sigma_max*n*dt)` for fixed
bounded profiles and a fixed strict CFL margin. This is a uniform finite-time
stability estimate, not a monotone full-step energy identity or a claim of
zero reflection. Its constant is loose; separate spectra and long-run tests
remain useful diagnostics. The post-filter jump must be included when
checking the old energy balance.

The multiplier is required to be at least 0.25. This prevents an ill-conditioned
local division in the exact VJP and excessive attenuation in one step.
At the actual default gamma/dt, even a bottom corner has m about 0.975.

## Exact source transpose, model gradient and restart

The reverse step is `Phi_y^T S^T (adj_next+O^T q)`. Observation is after S,
so its transpose must enter before multiplication. Source transpose is
`Phi_a^T S^T adj_next`, not the old gather without that factor.

Widths, rate, grid, dt, rho, source and geometry are fixed during the
log-Vp/log-Vs derivative. The multiplier has no material derivative.
Constitutive VJP terms use pre-multiplier velocities and stress cotangents
already multiplied by S. Existing histories contain post-multiplier velocities;
only the same-step local division by the known, bounded multiplier is used
to obtain those velocities. No backwards PDE reconstruction is performed.
The boundary-impedance VJP uses the average of the old input velocity and
the pre-multiplier new velocity.

The state still has five fields; the existing full-state checkpoints/replay
remain complete. Static masks add approximately 42.4 MiB at the event grid,
but no extra time-history fields. Full-history and checkpoint gradients are
checked bitwise. The existing approximately 9.073 GiB history-only estimate
is unchanged; it excludes masks and other live allocations.

Low-level adapters default to the impedance-only reference for backward
compatibility. The event driver explicitly passes `GaussianSponge(600,600,80)`.
Forward and model-gradient callers must pass the same explicit sponge object
or reconstruct its parameters from saved provenance. The adapter validates
all nonzero source and receiver interpolation supports outside the strips;
it rejects overlap instead of moving/clipping coordinates.

New packages record `boundary_scheme_version=impedance_gaussian_sponge_v1`,
widths, rate and the post-step splitting. Legacy `n_boundary/gamma_s` now
describe the active strip (120 cells, 80 1/s); `material_padding_*` remain
zero in SBP. Historical impedance packages remain valid records of their old
calculation. The old boundary-extension diagnostic explicitly rejects new
sponge packages rather than replaying them with the wrong boundary scheme.

## Verification and event commands

Local targeted tests: 228 pass, including 23 new sponge tests. Checks include
native legacy-mask equality, source/DAS transpose (including algebraic data
inside strips), Numba/reference parity, fixed-profile model finite differences
and Taylor checks, all-history/checkpoint parity, small amplification spectra,
4000-step source-free decay, full driver dry-run/save and native snapshots.

The SI-scale manufactured source/DAS QC gives Numba/dense relative error
1.68e-15, dot-product relative error 8.22e-15 and model Taylor rates
2.000303, 2.000148, 2.000073. These are numerical consistency checks, not
a real-event reflection measurement. Run from the project root:

```bash
"$FWI" -m pytest -q tests/test_safod_forward_render.py tests/test_sbp_sat_*.py
"$FWI" -m scripts.validation.sbp_sat.pipeline_qc --sponge \
  --output-dir results/validation/sbp_sat/restored_sponge_pipeline
```

Independent Fourier P/SV packets compare this solver with a larger same-mesh
benchmark and an impedance-only control. The reported positive elastic
solution-error energy is on one fixed initial-energy scale; it is not an
exact reflected-energy coefficient. The packets have finite bandwidth and
angular spread. The larger benchmark is validation only, not a production
grid addition. Reproduce a demanding case on a compute node:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=4 "$FWI" \
  -m scripts.validation.sbp_sat.sponge_packets --wave SV --face bottom \
  --angle 75 --output-dir results/validation/sbp_sat/restored_sponge_SV75
```


Completed packet measurements (same mesh and fixed normalization):

| Face | Mode | Central angle | Sponge error energy | Impedance-only error energy |
| --- | --- | --- | --- | --- |
| bottom | P | 0 deg | 1.67e-05 | 0.000111 |
| bottom | SV | 0 deg | 0.00028 | 0.00275 |
| bottom | P | 30 deg | 0.000677 | 0.00394 |
| bottom | SV | 30 deg | 0.00127 | 0.0172 |
| bottom | P | 75 deg | 0.0527 | 0.0566 |
| bottom | SV | 75 deg | 0.0785 | 0.102 |
| left | P | 30 deg | 0.000424 | 0.00259 |
| left | SV | 30 deg | 0.0013 | 0.0112 |
| right | P | 30 deg | 0.000424 | 0.00259 |
| right | SV | 30 deg | 0.0013 | 0.0112 |

These manufactured packets use width=30 m, gamma=0.96 1/s and Vp/Vs=3/1.5 m/s. The dimensionless P attenuation gamma*width/Vp=9.6 matches width=600 m, gamma=80 1/s and Vp=5000 m/s; they do not reproduce the heterogeneous event or its entire frequency band. Left/right symmetry is also checked numerically.

The old Gaussian method also has residual reflections, especially at grazing
incidence. At 75 degrees the current packets give error fractions 0.0527 P
and 0.0785 SV (controls 0.0566 and 0.1022). Thus the PML reference's 1e-4
target is **not** achieved by this restored legacy method in these cases.
No damping parameter was changed to hide that result. The actual SAFOD
event must be rerun before claiming its visible boundary branches are gone.

The same event output paths and native-coordinate GIF are retained:

```bash
mkdir -p results/events/logs
SAFOD_EVENT_KEY=nc75379261 \
SAFOD_MODEL=hybrid_zhang2009_boness2006_bill_logs SAFOD_THETA_DEG=35 \
sbatch scripts/safod/run_forward_sbp_sat.sbatch --overwrite
```

The batch also runs compare_event. It uses the same Zhang/Lin 3-D predicted
arrival curves; changing the absorbing strips does not change those guides.
