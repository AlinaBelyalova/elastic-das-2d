# 2D Elastic Forward Modelling for Borehole DAS

This project develops a workflow for 2D elastic forward modelling in a borehole DAS setting inspired by the SAFOD geometry.

The immediate goal is to build and validate the modelling infrastructure:
- define the computational grid,
- construct a SAFOD-like elastic model,
- represent the borehole DAS cable geometry,
- and prepare the project for later wave-equation solving, source implementation, and benchmarking.

The longer-term modelling target is elastic wave propagation from earthquake-like double-couple sources and the prediction of synthetic DAS responses along the borehole fibre.

---

## Current project status

At the current stage, the repository includes:
- a structured `Grid2D` class,
- an `ElasticModel2D` class that computes Lamé parameters from `Vp`, `Vs`, and `rho`,
- a SAFOD-like model builder based on projected cable geometry,
- plotting utilities for model inspection,
- a script that builds and saves the current synthetic model.

This is a **geologically informed simplified synthetic model**, not yet a fully realistic SAFOD truth model.

---

## Repository structure

```text
elastic_das_project/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── grid.py
│   ├── model.py
│   ├── safod_builder.py
│   └── plotting.py
├── scripts/
│   └── build_safod_model.py
└── notebooks/
    └── 01_build_safod_model.ipynb