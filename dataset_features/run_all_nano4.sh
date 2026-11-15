#!/usr/bin/env bash
#SBATCH --job-name=weight_only_analysis
#SBATCH --partition=8gpus
#SBATCH --account=MST114495
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=weight_only_analysis_%j.out
#SBATCH --error=weight_only_analysis_%j.err

# Submit from quant_analysis; sbatch executes a spool copy of this script.
set -euo pipefail
REPO="${ANALYSIS_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[[ -f "$REPO/dataset_features/run_all.py" ]] || {
    echo 'Submit from updated quant_analysis or set ANALYSIS_PROJECT_ROOT.' >&2
    exit 2
}
PYTHON_BIN="${ANALYSIS_PYTHON:-$HOME/.conda/envs/${ANALYSIS_CONDA_ENV:-tgda}/bin/python}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS" NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
cd "$REPO"
COMMAND=("$PYTHON_BIN" -B -u -m dataset_features.run_all "$@")
printf '%q ' "${COMMAND[@]}"; printf '\n'
# The Python planner uses only the standard library and common file helpers.
DRY_RUN=0
for option in "$@"; do
    if [[ "$option" == --dry-run ]]; then
        DRY_RUN=1
    fi
done
if (( ! DRY_RUN )) && command -v ml >/dev/null 2>&1; then
    ml purge
    ml load miniconda3
fi
[[ "$PYTHON_BIN" = /* && -x "$PYTHON_BIN" ]] || {
    echo "Set ANALYSIS_PYTHON to an executable absolute path: $PYTHON_BIN" >&2
    exit 2
}
exec "${COMMAND[@]}"
