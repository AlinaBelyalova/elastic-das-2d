"""Elastic-DAS project library.

Packages
--------
common      Grid2D, ElasticModel2D and shared signal processing.
elastic     Staggered-grid velocity-stress FD solver (NumPy and Numba
            backends), sources, source spreading/injection, receiver sampling
            and the forward-modelling orchestrator.
das         Physical receiver/cable geometry and the finite-gauge DAS
            strain / strain-rate operator.
analytical  Analytical reference solutions used by validation scripts.
fwi         Adjoint, gradient and misfit on top of the Numba elastic stack.
sbp_sat     SBP-SAT elastic prototype (current work).
safod       SAFOD velocity models and SAFOD-specific plotting.
safod_deep  SAFOD-deep continuous DAS archive: recording inventory, NCEDC
            catalog and 3-D cable geometry.

Packages have no eager imports: import the module you need, e.g.
``from src.elastic.solver_numpy import ...``.
"""
