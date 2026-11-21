#!/usr/bin/env bash
#SBATCH --job-name=dafb_w3_analysis
#SBATCH --partition=8gpus
#SBATCH --account=MST114495
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=dafb_w3_analysis_%j.out
#SBATCH --error=dafb_w3_analysis_%j.err

# Compatibility launcher for DAFB. The full pipeline also accepts --datasets dafb.
set -euo pipefail
repo="$(cd "${ANALYSIS_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}" && pwd)"
[[ -f "$repo/legacy/dataset_features/paired_inputs.py" ]] || {
    echo 'Submit from the quant_analysis root or set ANALYSIS_PROJECT_ROOT.' >&2
    exit 2
}
python_bin="${ANALYSIS_PYTHON:-$HOME/.conda/envs/${ANALYSIS_CONDA_ENV:-tgda}/bin/python}"
if command -v ml >/dev/null 2>&1; then
    ml purge
    ml load miniconda3
fi
[[ "$python_bin" = /* && -x "$python_bin" ]] || {
    echo "Set ANALYSIS_PYTHON to an executable absolute path: $python_bin" >&2
    exit 2
}
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS" MPLBACKEND=Agg
cd "$repo/legacy"
exec "$python_bin" -B -u -m dataset_features.dafb "$@"
