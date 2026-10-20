# SAFOD Elastic-DAS Project Instructions

## Project context

Read in this order before substantive changes:

1. `docs/status.md` — current state, next steps and open questions;
2. `docs/project_map.md` — where modules, entry points, inputs and results live;
3. the topic document: `docs/setup.md` (environments, inputs, external paths,
   tests), `docs/sbp_sat/` (SBP-SAT method and validation),
   `docs/elastic/method.md` (staggered-grid solver, DAS operator, FWI),
   `docs/safod/workflow.md` (SAFOD and SAFOD-deep runbook).

Scientific conventions for SAFOD are recorded in
`docs/history/FWI_SAFOD_FULL_HANDOFF_2026-08-14.md`. The SBP-SAT and
free-surface background is in
`docs/history/FWI_SAFOD_FULL_HANDOFF_2026-10-08.md`.

Handoffs are historical records. Their file paths, repository/git
instructions and "next chat" instructions are superseded by the current
layout. Their scientific conventions remain the reference unless
`docs/status.md` says otherwise. Do not assume that older scripts, comments,
plots, or constants are more authoritative than these conventions without
checking consistency.

All computed outputs and Slurm logs go under `results/` (rules in
`results/README.md`); the web viewer build output stays in `web_safod/site`
and `web_safod/cache`.

Before editing code, inspect the relevant implementation and trace the
dependencies sufficiently to understand what the change affects.

---

## General working style

Work autonomously on routine implementation, debugging, refactoring,
documentation, plotting, testing, and code-quality improvements.

Be thorough and careful rather than making the first change that appears to
work.

Prefer the smallest coherent change that solves the actual problem.

Do not blindly preserve existing code if it appears mathematically,
physically, or numerically inconsistent. Investigate the inconsistency and
explain it.

Do not make unrelated cleanup changes unless they are necessary for the task.

When multiple interpretations are possible, determine which interpretation is
consistent with the existing scientific conventions, handoff documentation,
and surrounding implementation.

---

## Mathematical and scientific rigor

Treat mathematical correctness as a first-class requirement.

For any change involving equations, geometry, coordinate transformations,
wave propagation, source representation, DAS response, interpolation,
discretization, or physical quantities:

- verify the mathematical relationship rather than relying on variable names;
- check units and dimensional consistency explicitly;
- check sign conventions explicitly;
- check coordinate-system definitions and axis orientation explicitly;
- distinguish physical coordinates from array indices;
- distinguish depth, elevation, TVD, MD, profile distance, Zhang X/Y, and
  computational-grid coordinates;
- verify origin definitions before applying offsets or shifts;
- check whether intervals are inclusive or exclusive at their boundaries;
- check indexing and off-by-one behavior;
- verify assumptions about sampling interval, spatial spacing, and grid origin;
- distinguish quantities such as displacement, velocity, acceleration, strain,
  and strain rate;
- distinguish Hz from angular frequency where relevant;
- distinguish metres from kilometres and seconds from milliseconds;
- preserve numerical precision unless reduced precision is intentional and
  justified.

When implementing a formula, compare the implementation with the underlying
mathematical expression and explain any non-obvious discretization.

If a result is surprising, investigate it rather than immediately modifying
parameters to make the result look expected.

---

## Numerical modelling rigor

For changes affecting the elastic forward model or FWI preparation, check the
relevant numerical consequences.

Where applicable, verify:

- computational-domain bounds;
- grid spacing and resulting grid dimensions;
- source and receiver locations relative to the computational grid;
- interpolation and extrapolation behavior;
- padding and absorbing-boundary regions;
- CFL/stability constraints;
- spatial and temporal sampling;
- numerical dispersion considerations;
- consistency of Vp, Vs, density, and derived elastic parameters;
- array shapes and axis ordering;
- boundary indexing;
- coordinate-to-grid conversion;
- consistency between saved metadata and actual arrays.

Do not silently clip geometry, sources, receivers, or model values to make them
fit a domain. If clipping occurs, identify why.

Do not silently extrapolate a physical model beyond its supported region
without identifying the extrapolation rule.

For changes to model bounds or grid geometry, report the resulting physical
bounds, grid dimensions, and spacing explicitly.

---

## Computational efficiency

Treat computational efficiency as important, especially because this project
runs on Sherlock and may operate on large arrays.

Before introducing an expensive implementation:

- consider asymptotic cost where relevant;
- avoid unnecessary repeated I/O;
- avoid unnecessary copies of large arrays;
- avoid Python loops over large numerical arrays when a clear vectorized or
  compiled alternative exists;
- avoid loading substantially more data than the task requires;
- reuse computed quantities when appropriate;
- consider memory footprint as well as runtime;
- preserve readability when optimizing.

For operations involving potentially large arrays or model volumes, estimate
memory use when it is material to the implementation.

Do not optimize blindly. Prefer an implementation that is mathematically
correct, understandable, and sufficiently efficient.

If an optimization changes numerical behavior, quantify or test the effect.

---

## Scientific conventions that require approval

Ask before changing any established scientific interpretation or convention,
including:

- SAFOD profile origin or orientation;
- coordinate-system definitions;
- transformations between geographic, projected, Zhang, or profile
  coordinates;
- borehole geometry interpretation;
- channel registration convention;
- Vp/Vs model interpretation or parameterization;
- density assumptions;
- source mechanism or source parameterization;
- source-time-function convention;
- DAS observable or gauge-length formulation;
- strain versus strain-rate conventions;
- physical units;
- sign conventions;
- event origin-time convention;
- fundamental FWI parameterization;
- major workflow architecture.

Routine implementation needed to preserve an already established convention
does not require approval.

If a requested task appears to require changing a scientific convention in
order to succeed, stop and explain why before making that change.

---

## Destructive or high-impact actions

Ask before:

- deleting files or datasets;
- deleting substantial existing functionality;
- replacing a major workflow with a different one;
- making destructive Git operations;
- resetting or discarding uncommitted work;
- changing externally consumed file formats or interfaces;
- changing a scientific convention listed above.

Do not commit, push, force-push, reset, clean, or rewrite Git history unless
explicitly requested.

Normal edits to source files do not require approval.

---

## Verification

Do not consider a substantive task complete merely because the code runs.

Choose verification appropriate to the change.

This may include:

- targeted unit or syntax checks;
- running the smallest relevant project script;
- checking numerical bounds and dimensions;
- checking array shapes;
- checking units;
- comparing key values before and after the change;
- checking geometry against the model domain;
- inspecting generated metadata;
- examining numerical tolerances;
- checking plots or diagnostic outputs when they carry scientific meaning.

For mathematical changes, test at least one known, limiting, or independently
checkable case when practical.

For coordinate transformations, verify the forward relationship and, when
possible, reconstruction or inverse consistency.

For changes involving bounds, spacing, or dimensions, compute the expected
values independently and compare them with the program output.

Do not weaken tests or tolerances simply to make a failing result pass without
understanding the discrepancy.

---

## Self-review

Before finishing a substantive task:

1. Review `git diff`.
2. Check that every changed line is relevant to the requested task.
3. Re-check mathematical and physical assumptions.
4. Look specifically for unit, sign, indexing, coordinate, and boundary
   mistakes.
5. Check for duplicated constants or stale assumptions elsewhere in the
   workflow.
6. Run the appropriate verification.
7. Consider whether the implementation introduces unnecessary computational
   cost or memory use.

---

## Final report

At the end of a substantive task, report concisely:

1. what was changed;
2. which files were modified;
3. the mathematical/physical reasoning behind non-trivial changes;
4. what verification was performed;
5. the relevant numerical results, dimensions, bounds, or tolerances;
6. any assumptions made;
7. any unresolved ambiguity or scientific consequence that deserves review.

Do not claim that something is correct merely because execution completed
without an exception.
