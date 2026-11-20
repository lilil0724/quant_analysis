#!/usr/bin/env bash
#SBATCH --job-name=bias_competitors
#SBATCH --partition=8gpus
#SBATCH --account=MST114495
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=bias_competitors_%j.out
#SBATCH --error=bias_competitors_%j.err

# Queue defaults match the existing nano4 launcher; analysis itself uses CPUs.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: sbatch hqq_confidence/run_bias_competitors_nano4.sh --serial N [options]
       bash hqq_confidence/run_bias_competitors_nano4.sh --serial N --dry-run

  --serial N              Required source serial; no serials are merged.
  --input-root PATH       Parent w3_g128_analysis or selected serial directory.
  --output-dir PATH       New output directory; existing outputs are preserved.
  --datasets CSV          Default: all 16 datasets, including inat17/cotton.
  --models CSV            Default: vit_b16,swin_base_patch4_window7_224_in22k,beitv2_base_patch16_224.
  --modes CSV             Default: ft,fz,cal.
  --chunk-size N          Default: 256 rows of logits per chunk.
  --threads N             Default: SLURM_CPUS_PER_TASK or 8.
  --seed N                Default: 100004, controls folds and shuffled biases.
  --preflight-only        Read-only completion/checksum/sample/protocol checks.
  --save-sample-results   Also save derived per-sample features/predictions.
  --dry-run               Print command without modules or execution.

Submit from the quant_analysis checkout. Environment overrides:
ANALYSIS_PROJECT_ROOT, ANALYSIS_INPUT_ROOT, ANALYSIS_OUTPUT_ROOT,
ANALYSIS_CONDA_ENV (default opencode_env), ANALYSIS_PYTHON.
Uses existing paired logits; no inference/checkpoint loading/W&B/GPU computation.
The 8gpus queue reserves one GPU, matching the existing nano4 job configuration.
EOF
}
fail() { echo "Error: $*" >&2; exit 2; }
require_value() { [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "Missing value for $1"; }

SERIAL=''
INPUT_ROOT="${ANALYSIS_INPUT_ROOT:-}"
OUTPUT_DIR="${ANALYSIS_OUTPUT_ROOT:-}"
DRY_RUN=0
EXTRA=()
THREADS="${SLURM_CPUS_PER_TASK:-8}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --serial|--input-root|--output-dir|--datasets|--models|--modes|--chunk-size|--threads|--seed)
            require_value "$@"
            case "$1" in
                --serial) SERIAL="$2" ;;
                --input-root) INPUT_ROOT="$2" ;;
                --output-dir) OUTPUT_DIR="$2" ;;
                --threads) THREADS="$2" ;;
                *) EXTRA+=("$1" "$2") ;;
            esac
            shift 2 ;;
        --preflight-only|--save-sample-results) EXTRA+=("$1"); shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown option: $1" ;;
    esac
done
[[ "$SERIAL" =~ ^[1-9][0-9]*$ ]] || fail 'Specify --serial N'
[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || fail 'threads must be positive'
REPO="${ANALYSIS_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[[ -f "$REPO/hqq_confidence/server_bias_competitors.py" && -f "$REPO/hqq_confidence/bias_competitor_core.py" ]] || fail 'Submit from updated quant_analysis or set ANALYSIS_PROJECT_ROOT'
REPO="$(cd "$REPO" && pwd)"
INPUT_ROOT="${INPUT_ROOT:-$REPO/../../data/w3_g128_analysis}"
if [[ "$INPUT_ROOT" != /* ]]; then INPUT_ROOT="$PWD/$INPUT_ROOT"; fi
INPUT_ROOT="${INPUT_ROOT%/}"
if [[ "$INPUT_ROOT" == */"$SERIAL" ]]; then INPUT_ROOT="${INPUT_ROOT%/*}"; fi
OUTPUT_DIR="${OUTPUT_DIR:-$INPUT_ROOT/$SERIAL/analysis/bias_competitors_v1}"
if [[ "$OUTPUT_DIR" != /* ]]; then OUTPUT_DIR="$PWD/$OUTPUT_DIR"; fi
CONDA_ENV_NAME="${ANALYSIS_CONDA_ENV:-opencode_env}"
PYTHON_BIN="${ANALYSIS_PYTHON:-$HOME/.conda/envs/$CONDA_ENV_NAME/bin/python}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
cd "$REPO"
COMMAND=("$PYTHON_BIN" -B -u -m hqq_confidence.server_bias_competitors
    --input-root "$INPUT_ROOT" --serial "$SERIAL" --output-dir "$OUTPUT_DIR"
    --threads "$THREADS" "${EXTRA[@]}")
echo "Job: ${SLURM_JOB_ID:-preview}"
echo "Python: $PYTHON_BIN"
echo "Input: $INPUT_ROOT/$SERIAL"
echo "Output: $OUTPUT_DIR"
printf '%q ' "${COMMAND[@]}"; printf '\n'
if (( DRY_RUN )); then
    echo 'Dry-run only; no analysis executed.'
    exit 0
fi
ml purge
ml load miniconda3
[[ "$PYTHON_BIN" = /* && -x "$PYTHON_BIN" ]] || fail "Set ANALYSIS_PYTHON to an executable absolute path: $PYTHON_BIN"
"${COMMAND[@]}"
