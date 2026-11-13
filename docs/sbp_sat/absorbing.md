# Bounded SBP-SAT and dissipative impedance boundaries

The equations below describe the retained impedance-only core/reference.
The event driver now restores the legacy-style Gaussian sponge inside the
same domain; its additional step/transpose/energy scope is in [sponge.md](sponge.md).

This is an additive NumPy reference after the [DAS core](integration.md),
not a replacement of `scripts.safod.run_forward`. It uses the same isotropic
equations, SI units, positive-down z, arithmetic material transfer,
centre-tangent DAS observable and fixed-density log(Vp)/log(Vs) parameters.
The periodic-x reference and all its checks remain available unchanged.

## Bounded grids

Both coordinates use `staggered_sbp_4_2`. Write n_x/n_z for interval counts:

| Quantity | Shape |
| --- | --- |
| Normal stresses, Vp, Vs, rho, lambda, lambda+2mu | `(n_x+1, n_z+1)` |
| vx, rho_vx | `(n_x, n_z+1)` |
| vz, rho_vz | `(n_x+1, n_z)` |
| sxz, mu_MM | `(n_x, n_z)` |

There is no duplicated periodic endpoint or wrapping. The physical domain is
`[0,n_x*dx] x [0,n_z*dz]`. Both interval counts must be at least eight.
Density is averaged to each velocity grid from its two adjacent N-N nodes;
shear modulus uses four equal neighboring N-N weights. Model coefficient
arrays are read-only, so cached boundary operators cannot silently go stale.

`RectangularSBPSAT` defaults to a free top and absorbing left/right/bottom.
Each face can explicitly be `free` or `absorbing`. A nonzero shear grid
source needs all four surrounding shear nodes; this implementation requires
all sources in the interior. No source or receiver coordinates are clipped.

## Boundary law and spatial energy identity

With outward normal n and tangential direction t, the absorbing condition is

$$
\sigma n=-Z_p v_n n-Z_s v_t t,\qquad
Z_p=\rho V_p,\quad Z_s=\rho V_s.
$$

It is a local impedance condition. It matches outgoing plane P/S waves at
normal incidence in the continuum. Oblique incidence generally reflects
energy and can mix modes; this implementation is not PML.

The traction-free SATs are first applied at all four faces, canceling the
discrete integration-by-parts flux. The impedance contribution adds `-D v`
to force density, before density division. Let Q be the diagonal native
velocity quadrature, R the native density, M=QR, and W the stress compliance
quadrature. Write F for the force map including free SATs, and B for the
constitutive strain-rate map. Then

$$
QF+B^T W=0,\qquad QD=(QD)^T\succeq0,
$$
$$
R\dot v=F\sigma-Dv,\qquad \dot\sigma=Bv,\qquad
\dot E=-v^TQDv\le0.
$$

At a face whose normal velocity lives on the staggered M grid, the normal
trace is `p^T v`, and the impedance force is `H_M^-1 p Z_p p^T v`.
For its boundary-aligned tangential velocity, the trace is `e^T v`, and the
force is `H_N^-1 e Z_s e^T v`. Remaining-axis face quadrature supplies Q.
Their quadratic form is the nonnegative sum of `Z_p*(normal trace)^2` and
`Z_s*(tangential trace)^2` integrated along each active face. Corners receive
both face contributions; no overwrite or corner sign exception is needed.

Zp is sampled on the N-N boundary. Zs is the two-point arithmetic average of
rho*Vs along that boundary onto its staggered tangential grid. All face
impedances therefore remain nonnegative. This averaging is part of the
discretization, including its parameter derivatives.

For background on energy-based elastic boundary closure, see
[Duru, Kozdon and Kreiss (2015)](https://doi.org/10.1016/j.jcp.2015.09.048).
The exact staggered time-discrete calculation below specifies this code's
particular implementation; it is not a claim to implement that paper's PML.

## Time stepping and exact dissipation

The stored state remains `(v^(n-1/2), sigma^n)`. Let `v_minus=v^(n-1/2)`,
`v_plus=v^(n+1/2)` and `v_bar=(v_plus+v_minus)/2`:

$$
(R+\tfrac{dt}{2}D)v_+
=(R-\tfrac{dt}{2}D)v_-+dt F\sigma^n,
$$
$$
\sigma^{n+1}=\sigma^n+dt Bv_+.
$$

Receiver data sample v_plus; a grid stress-rate source is added to the
new stress afterward, as in the existing prototype. For zero source define

$$
E_{LF}(v,\sigma)=\tfrac12v^TMv+\tfrac12\sigma^TW\sigma
                 +\tfrac{dt}{2}v^TQF\sigma.
$$

In the change of E_LF, the terms quadratic in `dt Bv_plus` cancel by
`QF=-B^T W`. The remaining terms are

$$
\Delta E_{LF}=\tfrac12(v_++v_-)^TQ
  [R(v_+-v_-)-dt F\sigma^n]
=-dt\,\bar v^TQD\bar v\le0.
$$

Thus this is an exact discrete balance, rather than a claim based only on
a decaying plot. Source work is excluded from this zero-source identity.
Random-state tests need not lose all their energy: finite duration and
non-radiating/static stress modes can retain energy. The criteria are
positivity, non-increase and the balance residual, not zero final energy.

Positivity of E_LF has the same Schur-complement requirement as the closed
reference. `rectangular_timestep_bound` replaces the periodic x norm by the
two bounded x derivative norm bounds, and uses the two bounded z bounds:

$$
dt<\frac{2}{\sqrt{C_*/\rho_{min}}\sqrt{k_x^2+k_z^2}}.
$$

C_*, rho_min and the weighted induced-norm bounds are defined in
[integration](integration.md). This is sufficient, conservative, and checked
against an independently assembled small spectrum. Positive impedance adds
no extra timestep restriction to this energy argument. The code rejects
uncertified dt rather than adjusting it. Dispersion accuracy is separate.

## Boundary solve and exact transpose

`K=R+dt*D/2` consists of a positive diagonal and two disjoint rank-one
normal-trace contributions per velocity line. At least eight intervals
ensure the left/right (or top/bottom) three-point trace supports do not
overlap. Sherman-Morrison updates solve K exactly to rounding precision,
including corners with tangential damping. Preparation and each solve cost
O(number of grid nodes); no global 2-D factorization is constructed.

For an ordinary-array cotangent, the implicit solve transpose is `K^-T`.
Because `QK=K^T Q`, it is implemented as `K^-T q=Q K^-1(Q^-1 q)`.
The adjoint then applies `(R-dt*D/2)^T` to the previous velocity and
`dt F^T` to the previous stress. It also reverses the stress update and
receiver interpolation/DAS in their actual forward order. It is not obtained
by merely running a damped forward equation backward.

## Model derivatives include the boundary

Density, grid, dt, source and observation geometry stay fixed. The existing
bulk Lamé/average chain rule is retained, with the bounded four-cell average
and its transpose. Impedances are recomputed from each perturbed Vp/Vs model;
they are not silently frozen at an old model.

In addition to the bulk stress-update derivative, a velocity-step variation
contributes

$$
\delta J_{boundary}=-dt\,a^T(\delta D)\bar v,\qquad
a=K^{-T}(\text{velocity-step cotangent}).
$$

Normal impedances use `delta Zp=rho*delta Vp`; tangential impedances use the
same boundary average applied to `rho*delta Vs`. Their transpose scatters to
the same boundary model nodes, including both faces at corners. Log gradients
multiply the resulting Vp/Vs derivatives by Vp/Vs respectively. A negative
control omitting these impedance derivatives fails the directional-derivative
check even though a fixed-model source-to-data dot product still passes.

## Observation, scope and verification

`build_rectangular_das_observation` uses the same gauge-centre tangent,
non-integer gauge length and endpoint interpolation as the existing DAS
operator. It strictly requires the common support of both velocity grids:
`dx/2 <= x-x0 <= width-dx/2` and `dz/2 <= z <= depth-dz/2`.
There is no periodic sampling wrap or boundary extrapolation.

Tests cover each dissipative face, corners, free conservation, wrong-sign
absorption controls, native-grid and implicit-solve transposes, complete
source-to-DAS transpose, Vp/Vs and boundary-only Taylor directions, and an
independent timestep spectrum. Taylor steps .01 to .00125 enter the quadratic
regime; a boundary-only direction still has a visible cubic term at .02.
Its central finite-difference directional derivative agrees independently.
Acceptance tolerances are not relaxed to accommodate that coarse step.

The outgoing normal-P packet check runs the actual bounded 2-D engine in the
near-lambda=0 decoupled P limit, with free lateral faces, correct half-time
initial velocities and no analytic field resetting. The initial packet is
downward only. Its residual-energy ratio must decrease on h=1 to 0.5 m and
be below 1e-4 at h=1. A fully free lower-face control must retain its energy.
This is a normal-P absorption check, not a broadband oblique P/S benchmark.

Dense 1-D reference operators remain the default. The [SAFOD adapter](pipeline.md)
adds CSR execution with identical coefficients, quadrature-normalized off-grid
moment sources and the existing STF/DAS/forward/comparison interfaces.
Forward stores receiver histories and requested snapshots. The low-level
gradient retains the full-history reference and adds optional
[checkpoint/replay](checkpointing.md), used by default by the adapter.
Boundary receiver traces where needed, practical oblique-reflection assessment
and compiled reverse/VJP remain follow-up work.
Commands: [validation](validation.md).

The independent continuum impedance-reflection audit and the staged PML
implementation/acceptance plan are in [boundary_plan.md](boundary_plan.md).
The normal-P packet result above does not certify oblique P/S absorption.
