#!/bin/bash
# Submit from the project root: sbatch web_safod/scripts/build_site.sbatch  (Slurm logs -> web_safod/cache/)
#SBATCH --job-name=safod_web
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=web_safod/cache/safod_web_%j.out
#SBATCH --error=web_safod/cache/safod_web_%j.err

set -euo pipefail

REPO="/home/groups/ettore88/alina/classes/geophys245/Elastic_das_project"
PYTHON="/home/groups/ettore88/alina/.envs/fwi/bin/python"

cd "${REPO}"

"${PYTHON}" -m web_safod.scripts.update_event_explorer
