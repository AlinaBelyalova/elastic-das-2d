#!/bin/bash
#SBATCH --job-name=safod_adaptive_qc
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --array=0-7%4
#SBATCH --output=safod_adaptive_qc_%A_%a.out
#SBATCH --error=safod_adaptive_qc_%A_%a.err

set -euo pipefail

PROJECT_ROOT="/home/groups/ettore88/alina/classes/geophys245/Elastic_das_project"
PYTHON="/home/groups/ettore88/alina/.envs/geo/bin/python"

cd "${PROJECT_ROOT}"

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

"${PYTHON}" -m scripts.safod_deep.plot_catalog_events \
    --num-shards 8 \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --overwrite
