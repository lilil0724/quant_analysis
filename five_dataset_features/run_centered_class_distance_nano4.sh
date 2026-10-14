#!/usr/bin/env bash
#SBATCH --job-name=centered_distance
#SBATCH --partition=8gpus
#SBATCH --account=MST114495
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=centered_distance_%j.out
#SBATCH --error=centered_distance_%j.err
set -euo pipefail

# sbatch spools the script: submit from the quant_analysis repository root.
# For direct Bash execution, locate the repository relative to this script.
ANALYSIS_PROJECT_ROOT="${ANALYSIS_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
ANALYSIS_CONDA_ENV="${ANALYSIS_CONDA_ENV:-tgda}"
ANALYSIS_PYTHON="${ANALYSIS_PYTHON:-$HOME/.conda/envs/${ANALYSIS_CONDA_ENV}/bin/python}"
if command -v ml >/dev/null 2>&1; then
    ml purge
    ml load miniconda3
fi
[[ -f "$ANALYSIS_PROJECT_ROOT/five_dataset_features/centered_class_distance.py" ]] || {
    echo "Submit from quant_analysis or set ANALYSIS_PROJECT_ROOT." >&2
    exit 1
}
[[ -x "$ANALYSIS_PYTHON" ]] || {
    echo "Missing Python: $ANALYSIS_PYTHON; set ANALYSIS_PYTHON." >&2
    exit 1
}
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
cd "$ANALYSIS_PROJECT_ROOT"
"$ANALYSIS_PYTHON" -B -c 'import sys, numpy; print(sys.executable); print("NumPy", numpy.__version__)'
exec "$ANALYSIS_PYTHON" -B -u -m five_dataset_features.centered_class_distance "$@"
