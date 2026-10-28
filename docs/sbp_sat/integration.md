# SBP-SAT foundation and DAS integration

This change preserves the periodic-x, free-top/free-bottom NumPy reference
and connects it to the project's existing finite-gauge DAS observable.
It does not replace the SAFOD production driver or the Graves FS1 backend.
Base equations, units and time levels: [method](method.md).
Commands and acceptance criteria: [validation](validation.md).

## Boundary algebra and negative controls

For depth increasing down, the upper boundary is the left endpoint of the
z SBP pair. The shear energy derivative before SAT is

$$
\dot E_h=-(e_L^T v)(p_L^T\tau)+(e_R^T v)(p_R^T\tau).
$$

Adding `H_N^-1(e_L p_L^T-e_R p_R^T) tau` to the force density cancels it.
The normal pair exchanges N and M and uses
`H_M^-1(p_L e_L^T-p_R e_R^T) sigma_zz`. The density division follows SAT.
`boundaries.py` implements both penalties and their ordinary array
transposes. A second H or density factor must not be added to the transpose.

Tests isolate both endpoints of both 1-D pairs. With unit traces the raw
flux is -15/8 at the top and +15/8 at the bottom. Correct SAT cancels it;
disabled SAT and reversed signs explicitly fail cancellation. Separate tests
remove or reverse the upper SAT inside the actual 2-D velocity RHS and verify
that its energy check detects the error. Stress rows are never set to zero.

Polynomial tests cover constants, linear and quadratic fields everywhere;
cubic/quartic fields on interior stencils; quadratic boundary projection;
positive quadrature and constant integration. Cases include the minimum
8 intervals, 9, 12 and 40 intervals and h=0.125 and 3.7 m.

## A sufficient timestep bound

Write the undamped source-free system as `v_dot=A sigma`, `sigma_dot=B v`,
with kinetic weight M and compliance weight W. The SBP/SAT identity is
`MA+B^T W=0`. Let `K=M^(1/2) A W^(-1/2)` and transform the stored state
to `u=M^(1/2)v`, `w=W^(1/2)sigma`. The leapfrog invariant has matrix

$$
E_{LF}=\frac12
\begin{bmatrix}u\\w\end{bmatrix}^{T}
\begin{bmatrix}I&\frac{dt}{2}K\\\frac{dt}{2}K^T&I\end{bmatrix}
\begin{bmatrix}u\\w\end{bmatrix}.
$$

Positivity requires `dt*||K||_2 < 2`. To bound this without assembling a
2-D matrix, define the minimum of both native velocity densities `rho_min`
and the stiffness bound

$$
C_* = \max\{\max(l2m+lambda),\max(l2m-lambda),2\max(mu_{MM})\}.
$$

For periodic x, `k_x=7/(3 dx)` bounds the fourth-order derivative norm.
For each native z derivative use

$$
T=H_{out}^{1/2}D H_{in}^{-1/2},\quad
k_D=\sqrt{\|T\|_1\|T\|_\infty},\quad k_z=\max(k_{MN},k_{NM}).
$$

The normal-strain energy is bounded by C_* times the squared normal strain
rates. For engineering shear, `|a+b|² <= 2(|a|²+|b|²)` and
`2 mu_MM <= C_*`. Combining both velocity components gives

$$
\|K\|_2^2\le\frac{C_*}{rho_{min}}(k_x^2+k_z^2),\qquad
dt_{sufficient}=\frac{2}{\sqrt{C_*/rho_{min}}\sqrt{k_x^2+k_z^2}}.
$$

`C_*/rho_min` has units m²/s², k has units 1/m and dt is in seconds.
This applies to the material interpolation and closed boundaries of this
reference. It is a sufficient bound, not the largest stable step and not a
dispersion criterion. No 2-D matrix, eigensolve or SVD is used in the bound.

`run_forward` and `run_adjoint_to_source` reject nonfinite or nonpositive
steps and require the strict inequality. Neither changes dt or nt.
The model-gradient call uses the same guard through its forward run.
During Taylor tests density, dt and observation geometry stay fixed.
`advance_one_step` remains a low-level primitive: its loop caller must
certify dt once. Existing controlled QC steps meet this condition.

An independent small test assembles the actual A and B by probing basis
vectors. It checks that the bound is below the sharp spectral limit and
that a timestep beyond the actual limit gives an unstable step matrix.
The guarantee must be rederived if damping, different boundaries or another
time integrator is introduced.

## DAS operator and transpose

`build_das_observation(model,op_z,dx,receivers,gauge_length_m)` accepts the
existing `Receivers2D`. It precomputes immutable weights in O(n_receivers):

1. bilinear sampling of each Cartesian velocity on its native staggered grid;
2. cable interpolation to `s +/- L/2`, with non-integer L/ds supported;
3. Cartesian differencing followed by projection on the gauge-centre tangent.

The finite-gauge observable is exactly the convention of `src/das/strain.py`:

$$
d(s)=\frac{[v_x(s+L/2)-v_x(s-L/2)]t_x(s)
             +[v_z(s+L/2)-v_z(s-L/2)]t_z(s)}{L}.
$$

The output is strain rate in 1/s for velocities in m/s. Valid original
channel indices are retained; coordinates are not resampled or moved.
Physical x must be in `[x0,x0+nx*dx)`. Native stencil indices wrap because
the propagation problem itself is periodic. z must be in the common support
of both velocity grids, `[dz/2,depth-dz/2]`. Boundary extrapolation is rejected.
Existing native point receivers on the free surface remain supported.

The transpose reverses these operations and uses accumulating scatter for
overlapping gauges and repeated native nodes. No time or quadrature weights
are added to this Euclidean transpose. For the existing objective
`J=0.5*dt*sum(residual²)`, the objective contributes dt exactly once.
Passing `DASObservation` as the receivers argument uses this full chain in
the existing forward, source transpose and log(Vp)/log(Vs) gradient routines.

The centre-tangent expression is a preserved physical approximation. A true
curved-fibre axial-strain average involves
`integral t(s)^T grad(v) t(s) ds / L`. The centre-tangent approximation is
intended for a gauge short relative to the bend radius. Algebraic transpose
and Taylor tests do not validate that approximation or mechanical coupling.
Changing the observable requires a separate scientific decision.

## Sources and limits

`StressPointSource` retains its previous defaults and adds optional mxz.
Normal components are injected on N-N. Shear uses four equal M-M weights,
whose positions surround that same physical N-N node. A nonzero shear
component requires all four nodes in z. Its source adjoint uses identical
weights, including the periodic x seam.

Coefficient times STF has units Pa/s. It is a generic grid stress-rate
forcing. No claim is made that these coefficients equal a catalogue 3-D
seismic moment. Source normalization, the physical STF convention and
off-grid source support are now matched by the separate
[SAFOD adapter](pipeline.md); the generic grid-source class remains unchanged.

An additive [bounded reference](absorbing.md) now implements energy-dissipative
impedance sides/bottom, their discrete transpose and parameter derivatives.
The [opt-in SAFOD backend](pipeline.md) adds moment-source normalization,
off-grid source support, CSR execution, forward products and comparison.
The bounded adapter now has [checkpoint/replay](checkpointing.md). Boundary
receiver traces where needed, practical oblique-reflection assessment and
compiled reverse/VJP remain follow-up work. A free lower boundary and periodic x cannot be
silently substituted for the production open-domain problem.
