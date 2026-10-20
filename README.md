# Elastic DAS Project

The project covers 2-D modelling of elastic waves and the DAS response: a
staggered-grid solver with a DAS observation operator, FWI on the Numba stack,
comparison with real SAFOD events, work with the SAFOD-deep
continuous-recording archive, and a static web viewer of its events. The
current stage is the SBP-SAT prototype; status and next steps are in
[docs/status.md](docs/status.md).

## Layout rule

| Where | What | How to use |
| --- | --- | --- |
| `src/` | library | import only: `from src.<package>.<module> import ...` |
| `scripts/` | entry points | `python -m scripts.<area>.<name>`, always from the project root |
| `docs/` | explanations | methods, runbook, status, project map |
| `data/`, `config/` | inputs | velocity models, SAFOD-deep archive roots |
| `results/` | everything computed | folder = script name without `_qc`; [rules](results/README.md) |
| `scripts/legacy/`, `results/legacy/`, `docs/history/` | old | old code, old results, earlier handoffs; for reference only (`docs/history/` exists only in the local copy, outside Git) |

The area names `sbp_sat`, `elastic`, `fwi`, `safod`, `safod_deep` repeat
across `src/`, `scripts/`, `results/` and `docs/`, so a single name leads to
the code, the run, the results and the documentation. For example:
`src/sbp_sat/` → `scripts/validation/sbp_sat/` → `results/validation/sbp_sat/`
→ `docs/sbp_sat/`. Exceptions: SAFOD event results are in `results/events/`,
FWI is described in `docs/elastic/method.md`, SAFOD-deep in
`docs/safod/workflow.md`.

## Structure

```text
src/          library: solvers, DAS, FWI, SBP-SAT, SAFOD models, SAFOD-deep catalog
scripts/      entry points: checks, examples, benchmarks, SAFOD, SAFOD-deep, legacy
tests/        fast automated tests (pytest.ini in the root)
docs/         status, project map, methods, SAFOD runbook, history
config/       SAFOD-deep archive roots on Oak
data/         local SAFOD velocity models; raw DAS recordings are on Oak
results/      all computed results and logs (local, outside Git)
notebooks/    demos and exploration
web_safod/    static viewer of SAFOD-deep events
```

The root also contains `requirements.txt`, `pytest.ini` and `AGENTS.md`
(instructions for AI agents).

## Where to start

| Task | Document | Code | Results |
| --- | --- | --- | --- |
| Current stage | [docs/status.md](docs/status.md) | `src/sbp_sat/` | `results/validation/sbp_sat/` |
| Environment, data, tests | [docs/setup.md](docs/setup.md) | `tests/`, `web_safod/tests/` | — |
| SBP-SAT | [method](docs/sbp_sat/method.md), [checks](docs/sbp_sat/validation.md) | `src/sbp_sat/`, `scripts/validation/sbp_sat/` | `results/validation/sbp_sat/` |
| Legacy solver and DAS | [docs/elastic/method.md](docs/elastic/method.md) | `src/elastic/`, `src/das/`, `src/common/`, `src/analytical/`, `scripts/validation/elastic/` | `results/validation/elastic/`, `results/examples/`, `results/benchmarks/` |
| FWI | [docs/elastic/method.md](docs/elastic/method.md) | `src/fwi/`, `scripts/validation/fwi/` | `results/validation/fwi/` |
| SAFOD event | [docs/safod/workflow.md](docs/safod/workflow.md) | `scripts/safod/`, `src/safod/` | `results/events/` |
| SAFOD event search | [docs/safod/workflow.md](docs/safod/workflow.md) | `scripts/safod/selection/` | `results/safod/` |
| SAFOD-deep archive and site | [docs/safod/workflow.md](docs/safod/workflow.md), [web_safod/README.md](web_safod/README.md) | `src/safod_deep/`, `scripts/safod_deep/`, `web_safod/` | `results/safod_deep/`, `web_safod/site/` |
| Velocity models | [docs/safod/workflow.md](docs/safod/workflow.md) | `scripts/safod/velocity_models/`, `src/safod/models/` | `data/safod/velocity_models/` |
| All checks | [scripts/validation/README.md](scripts/validation/README.md) | `scripts/validation/` | `results/validation/` |
| Where things are | [docs/project_map.md](docs/project_map.md) | — | [results/README.md](results/README.md) |
| History | `docs/history/README.md` | `scripts/legacy/` ([README](scripts/legacy/README.md)) | `results/legacy/` |

## Quick check

From the project root, in the `fwi` environment, on a compute node
([environment](docs/setup.md)):

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m scripts.validation.sbp_sat.operator_qc
```

Checks the SBP identity and the SAT energy balance, runs in a fraction of a
second and writes nothing. The other SBP-SAT checks, their commands and
criteria are in [docs/sbp_sat/validation.md](docs/sbp_sat/validation.md).

## Legacy solver

Velocity-stress staggered-grid scheme (Virieux), leapfrog in time, NumPy and
Numba backends; DAS: interpolation, difference over the gauge length and
projection onto the cable tangent. Details: [docs/elastic/method.md](docs/elastic/method.md).
