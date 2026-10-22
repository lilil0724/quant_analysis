#!/usr/bin/env bash
#SBATCH --job-name=margin_matching
#SBATCH --partition=8gpus
#SBATCH --account=MST114495
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=margin_matching_%j.out
#SBATCH --error=margin_matching_%j.err

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: sbatch hqq_confidence/run_margin_matching_nano4.sh --serial 20004 --input-id ID [options]
  --stage all|preflight|import|run  Default all: import a NEW snapshot, then analyze.
  --source-root PATH               Parent of SERIAL, or SERIAL directory.
  --workspace PATH                 Owns data/ID, result/RUN, reports/RUN.
  --input-id ID                    Required except for read-only preflight.
  --run-id ID                      New ID; default Taiwan timestamp + serial.
  --datasets CSV                   Import/preflight: default all 16 incl cotton/inat17.
  --models CSV                     Import/preflight: default ViT/Swin/BEiTv2.
  --modes CSV                      Import/preflight: default ft,fz,cal.
  --chunk-size N                   Default 256; bounded logit operations.
  --dry-run                        Print commands without loading modules or running.
  -h, --help

Soy+cotton = soyageing,soygene,soyglobal,soylocal,cotton; Other includes inat17.
Run stage uses the snapshot's full imported matrix. Use --stage run to reuse it.
All/import refuse an existing input ID; every analysis refuses an existing run ID.
CPU analysis, no inference. Queue reserves one GPU per repository policy.
Environment: ANALYSIS_PROJECT_ROOT, TGDA_CONDA_ENV (default tgda), TGDA_PYTHON.
Default Python: $HOME/.conda/envs/tgda/bin/python (same as TGDA launchers).
ANALYSIS_PYTHON, when set, overrides TGDA_PYTHON. No conda activation is required.
EOF
}
fail() { echo "Error: $*" >&2; exit 2; }
value() { [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "Missing value for $1"; }

STAGE=all
SERIAL=''
SOURCE_ROOT=/work/kyle0724/data/w3_g128_analysis
WORKSPACE=''
INPUT_ID=''
RUN_ID=''
CHUNK=256
DRY_RUN=0
SELECT=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage|--serial|--source-root|--input-root|--workspace|--input-id|--run-id|--chunk-size|--datasets|--models|--modes)
            value "$@"
            case "$1" in
                --stage) STAGE="$2" ;;
                --serial) SERIAL="$2" ;;
                --source-root|--input-root) SOURCE_ROOT="$2" ;;
                --workspace) WORKSPACE="$2" ;;
                --input-id) INPUT_ID="$2" ;;
                --run-id) RUN_ID="$2" ;;
                --chunk-size) CHUNK="$2" ;;
                *) SELECT+=("$1" "$2") ;;
            esac
            shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown option: $1" ;;
    esac
done
[[ "$STAGE" =~ ^(all|preflight|import|run)$ ]] || fail 'Invalid --stage'
[[ "$SERIAL" =~ ^[1-9][0-9]*$ ]] || fail 'Specify --serial N'
[[ "$CHUNK" =~ ^[1-9][0-9]*$ ]] || fail 'chunk-size must be positive'
if [[ "$STAGE" != preflight ]]; then
    [[ "$INPUT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || fail 'Specify a safe --input-id'
fi
[[ "$STAGE" != run || ${#SELECT[@]} -eq 0 ]] || fail '--stage run uses the imported selection; omit --datasets/--models/--modes'
REPO="${ANALYSIS_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[[ -f "$REPO/hqq_confidence/analyze_margin_matching.py" ]] || fail 'Submit from the updated quant_analysis checkout or set ANALYSIS_PROJECT_ROOT'
REPO="$(cd "$REPO" && pwd)"
WORKSPACE="${WORKSPACE:-$REPO/hqq_confidence}"
if [[ "$WORKSPACE" != /* ]]; then WORKSPACE="$PWD/$WORKSPACE"; fi
if [[ "$SOURCE_ROOT" != /* ]]; then SOURCE_ROOT="$PWD/$SOURCE_ROOT"; fi
RUN_ID="${RUN_ID:-$(TZ=Asia/Taipei date +%Y%m%d_%H%M%S)_margin_s$SERIAL}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || fail 'Invalid --run-id'
TGDA_CONDA_ENV="${TGDA_CONDA_ENV:-tgda}"
TGDA_PYTHON="${ANALYSIS_PYTHON:-${TGDA_PYTHON:-${HOME}/.conda/envs/${TGDA_CONDA_ENV}/bin/python}}"
export TGDA_PYTHON
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "$REPO"
PREFIX=("$TGDA_PYTHON" -B -u -m hqq_confidence.analyze_margin_matching)
execute() {
    printf '%q ' "$@"; printf '\n'
    if (( ! DRY_RUN )); then "$@"; fi
}
if (( ! DRY_RUN )); then
    ml purge
    ml load miniconda3
    [[ "$TGDA_PYTHON" = /* && -x "$TGDA_PYTHON" ]] || fail "Python not found or not executable: $TGDA_PYTHON; set TGDA_PYTHON to the absolute Python path in the tgda environment"
    echo "SLURM job ID: ${SLURM_JOB_ID:-local}"
    echo "Python: $TGDA_PYTHON"
    "$TGDA_PYTHON" -c 'import sys, numpy, pandas, matplotlib, scipy, sklearn, threadpoolctl; print("Interpreter:", sys.executable); print("NumPy:", numpy.__version__, "pandas:", pandas.__version__, "matplotlib:", matplotlib.__version__, "SciPy:", scipy.__version__, "sklearn:", sklearn.__version__)'
fi
if [[ "$STAGE" == preflight ]]; then
    execute "${PREFIX[@]}" preflight-server --source-root "$SOURCE_ROOT" --serial "$SERIAL" --chunk-size "$CHUNK" "${SELECT[@]}"
fi
if [[ "$STAGE" == all || "$STAGE" == import ]]; then
    execute "${PREFIX[@]}" import-server --source-root "$SOURCE_ROOT" --serial "$SERIAL" \
        --workspace "$WORKSPACE" --input-id "$INPUT_ID" --chunk-size "$CHUNK" "${SELECT[@]}"
fi
if [[ "$STAGE" == all || "$STAGE" == run ]]; then
    execute "${PREFIX[@]}" run --workspace "$WORKSPACE" --input-id "$INPUT_ID" \
        --run-id "$RUN_ID" --serial "$SERIAL" --chunk-size "$CHUNK"
fi
if (( DRY_RUN )); then echo 'Dry-run only; no files imported or analysis executed.'; fi
