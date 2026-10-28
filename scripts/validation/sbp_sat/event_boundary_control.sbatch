#!/bin/bash
#SBATCH --job-name=safod_sbp_bc
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=results/events/logs/safod_sbp_bc_%j.out
#SBATCH --error=results/events/logs/safod_sbp_bc_%j.err

# Run from the project root after mkdir -p results/events/logs.
# Replays the saved baseline, then extends only the sides/bottom by 120 cells.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export SAFOD_EVENT_KEY="${SAFOD_EVENT_KEY:-nc75379261}"
CONTROL_MODEL="${SAFOD_MODEL:-hybrid_zhang2009_boness2006_bill_logs}"
CONTROL_THETA="${SAFOD_THETA_DEG:-35}"
CONTROL_PYTHON=/home/groups/ettore88/alina/.envs/fwi/bin/python

"$CONTROL_PYTHON" -u -m scripts.validation.sbp_sat.event_boundary_control \
  --theta-deg "$CONTROL_THETA" --initial-model "$CONTROL_MODEL" \
  --extra-padding-cells 120 "$@"
