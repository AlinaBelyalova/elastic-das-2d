#!/bin/bash
#SBATCH --job-name=safod_sbp_sat
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=results/events/logs/safod_sbp_sat_%j.out
#SBATCH --error=results/events/logs/safod_sbp_sat_%j.err

# Submit from the project root, after mkdir -p results/events/logs.
# GIF/snapshots follow the forward default (enabled); pass --no-gif to opt out.
# Pass --overwrite only to intentionally regenerate an existing forward NPZ.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the project root}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export SAFOD_EVENT_KEY="${SAFOD_EVENT_KEY:-nc75379261}"
SBP_MODEL="${SAFOD_MODEL:-hybrid_zhang2009_boness2006_bill_logs}"
case "${SAFOD_SOURCE_MECHANISM:-manual}" in
  manual) SOURCE_ARGS=(--theta-deg "${SAFOD_THETA_DEG:-35}") ;;
  comcat) SOURCE_ARGS=(--comcat-mechanism) ;;
  *) echo "SAFOD_SOURCE_MECHANISM must be manual or comcat." >&2; exit 2 ;;
esac
SBP_FWI_PYTHON=/home/groups/ettore88/alina/.envs/fwi/bin/python
SBP_GEO_PYTHON=/home/groups/ettore88/alina/.envs/geo/bin/python

"$SBP_FWI_PYTHON" -u -m scripts.safod.run_forward \
  --backend sbp_sat --sbp-execution numba "${SOURCE_ARGS[@]}" \
  --initial-model "$SBP_MODEL" --duration-s 3 "$@"
"$SBP_GEO_PYTHON" -u -m scripts.safod.compare_event \
  --backend sbp_sat "${SOURCE_ARGS[@]}" --initial-model "$SBP_MODEL"
