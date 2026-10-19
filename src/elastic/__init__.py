"""Staggered-grid velocity-stress FD solver stack.

solver_numpy and solver_numba_fused are the production backends and must stay
mathematically consistent. solver_numpy_pointforce is a validation-only solver
used by scripts/validation/elastic/{pointforce_2d,reciprocity}.py.
"""
