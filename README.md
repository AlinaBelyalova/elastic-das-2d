# Elastic DAS Project

Validated and accelerated 2D elastic finite-difference modelling for synthetic DAS.

## Project question

**Can a 2D staggered-grid elastic finite-difference solver for synthetic DAS be validated against an analytical reference and accelerated with Numba while preserving numerical correctness in a free-surface configuration?**

## Overview

This project implements a 2D isotropic elastic wave solver in the $((x, z))$ plane using a staggered-grid velocity-stress formulation.

The main goals of the project are:

- to build a physically consistent forward-modelling workflow for synthetic DAS,
- to validate the numerical propagation engine against an analytical point-force reference,
- to accelerate the solver with Numba while preserving agreement with a NumPy baseline.

The project focuses on forward modelling, validation, and computational performance analysis rather than full inversion.

---

## Main components

### Solver
- 2D first-order elastic velocity-stress formulation
- Virieux-style staggered grid
- explicit leapfrog time staggering
- configurable spatial finite-difference order
- free-surface boundary condition
- NumPy baseline solver
- Numba fused accelerated backend

### Source and receivers
- double-couple moment-tensor source in 2D
- DAS cable / receiver geometry
- staggered-aware receiver sampling

### DAS operator
- axial strain-rate computation from receiver particle velocities
- gauge-length differencing
- projection onto the local cable tangent
- physically consistent “difference first, then project” formulation

### Validation and benchmarking
- analytical point-force validation in a homogeneous medium
- convergence study for spatial FD orders 2, 4, 6, 8
- NumPy vs Numba fused performance comparison
- runtime, speedup, and throughput scaling for `free_surface=True`

---

## Repository structure

```text
src/        core solver, source, DAS, model, sampling, and simulator modules
scripts/    validation and performance study scripts
notebooks/  demonstration notebook(s)