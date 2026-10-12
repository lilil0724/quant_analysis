#!/usr/bin/env bash
#SBATCH --job-name=w3_paired_tsne
#SBATCH --partition=8gpus
#SBATCH --account=MST114495
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=w3_paired_tsne_%j.out
#SBATCH --error=w3_paired_tsne_%j.err

# Submit from the quant_analysis checkout on nano4.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: sbatch five_dataset_features/run_tsne_nano4.sh --serial 20004 [options]
       bash five_dataset_features/run_tsne_nano4.sh --serial 20004 --dry-run [options]

  --serial N          Required W3/g128 source serial.
  --input-root PATH   Parent w3_g128_analysis directory; default <repo>/../../data/w3_g128_analysis.
                      A selected serial directory is accepted too.
  --output-root PATH  Default: <input-root>/<serial>/analysis/tsne.
  --stage NAME        run (default) or preflight (no outputs).
  --datasets CSV      Default: all 16 datasets.
  --models CSV        Default: all 3 models.
  --modes CSV         Default: ft,fz,cal.
  --max-samples N     Paired sample budget per plot, default 2000.
  --perplexity X      Requested t-SNE perplexity, default 30.
  --iterations N      t-SNE iterations, default 750 (minimum 250).
  --seed N            Sampling/embedding seed, default 100.
  --overwrite         Replace generated output for a changed run.
  --dry-run           Show the exact Python command only.

Environment: ANALYSIS_PROJECT_ROOT, ANALYSIS_INPUT_ROOT, ANALYSIS_OUTPUT_ROOT,
             ANALYSIS_CONDA_ENV (tgda), ANALYSIS_PYTHON.
Requires NumPy, Matplotlib and scikit-learn in the selected Python environment.
All selected pairs are checked before any plot is written. FP and W3 share one fit.
EOF
}
fail() { echo "Error: $*" >&2; exit 2; }
require_value() { [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "Missing value for $1"; }
validate_names() {
    local label="$1" allowed="$2" name
    shift 2
    local -A seen=()
    for name in "$@"; do
        [[ "$allowed" == *" $name "* ]] || fail "Unknown $label: $name"
        [[ ! -v "seen[$name]" ]] || fail "Duplicate $label: $name"
        seen[$name]=1
    done
}
read_selection() {
    [[ "$2" != ,* && "$2" != *, && "$2" != *,,* ]] || fail "Empty name in $1"
}

SERIAL=''
STAGE=run
INPUT_ROOT="${ANALYSIS_INPUT_ROOT:-}"
OUTPUT_ROOT="${ANALYSIS_OUTPUT_ROOT:-}"
MAX_SAMPLES=2000
PERPLEXITY=30
ITERATIONS=750
SEED=100
OVERWRITE=0
DRY_RUN=0
DATASETS=(aircraft cars cub dogs flowers food inat17 moe nabirds pets soyageing soygene soyglobal soylocal vegfru cotton)
MODELS=(vit_b16 swin_base_patch4_window7_224_in22k beitv2_base_patch16_224)
MODES=(ft fz cal)
ALLOWED_DATASETS=" ${DATASETS[*]} "
ALLOWED_MODELS=" ${MODELS[*]} "
ALLOWED_MODES=" ${MODES[*]} "

while [[ $# -gt 0 ]]; do
    case "$1" in
        --serial|--input-root|--output-root|--stage|--datasets|--models|--modes|--max-samples|--perplexity|--iterations|--seed)
            require_value "$@"
            case "$1" in
                --serial) SERIAL="$2" ;;
                --input-root) INPUT_ROOT="$2" ;;
                --output-root) OUTPUT_ROOT="$2" ;;
                --stage) STAGE="$2" ;;
                --datasets) read_selection "$1" "$2"; IFS=',' read -r -a DATASETS <<< "$2" ;;
                --models) read_selection "$1" "$2"; IFS=',' read -r -a MODELS <<< "$2" ;;
                --modes) read_selection "$1" "$2"; IFS=',' read -r -a MODES <<< "$2" ;;
                --max-samples) MAX_SAMPLES="$2" ;;
                --perplexity) PERPLEXITY="$2" ;;
                --iterations) ITERATIONS="$2" ;;
                --seed) SEED="$2" ;;
            esac
            shift 2 ;;
        --overwrite) OVERWRITE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown option: $1" ;;
    esac
done

[[ "$SERIAL" =~ ^[1-9][0-9]*$ ]] || fail 'Specify --serial N'
[[ "$STAGE" == run || "$STAGE" == preflight ]] || fail 'Stage must be run or preflight'
[[ "$MAX_SAMPLES" =~ ^[1-9][0-9]*$ && "$MAX_SAMPLES" -ge 2 ]] || fail 'max-samples must be >= 2'
[[ "$ITERATIONS" =~ ^[1-9][0-9]*$ && "$ITERATIONS" -ge 250 ]] || fail 'iterations must be >= 250'
[[ "$SEED" =~ ^[0-9]+$ ]] || fail 'seed must be nonnegative'
validate_names dataset "$ALLOWED_DATASETS" "${DATASETS[@]}"
validate_names model "$ALLOWED_MODELS" "${MODELS[@]}"
validate_names mode "$ALLOWED_MODES" "${MODES[@]}"

ANALYSIS_PROJECT_ROOT="${ANALYSIS_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[[ -f "$ANALYSIS_PROJECT_ROOT/five_dataset_features/tsne_paired.py" ]] || fail 'Submit from updated quant_analysis or set ANALYSIS_PROJECT_ROOT'
REPO="$(cd "$ANALYSIS_PROJECT_ROOT" && pwd)"
INPUT_ROOT="${INPUT_ROOT:-$REPO/../../data/w3_g128_analysis}"
if [[ "$INPUT_ROOT" != /* ]]; then INPUT_ROOT="$PWD/$INPUT_ROOT"; fi
INPUT_ROOT="${INPUT_ROOT%/}"
if [[ "$INPUT_ROOT" == */"$SERIAL" ]]; then INPUT_ROOT="${INPUT_ROOT%/*}"; fi
OUTPUT_ROOT="${OUTPUT_ROOT:-$INPUT_ROOT/$SERIAL/analysis/tsne}"
if [[ "$OUTPUT_ROOT" != /* ]]; then OUTPUT_ROOT="$PWD/$OUTPUT_ROOT"; fi
ANALYSIS_CONDA_ENV="${ANALYSIS_CONDA_ENV:-tgda}"
ANALYSIS_PYTHON="${ANALYSIS_PYTHON:-${HOME}/.conda/envs/${ANALYSIS_CONDA_ENV}/bin/python}"
THREADS="${SLURM_CPUS_PER_TASK:-8}"
[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || fail 'SLURM_CPUS_PER_TASK must be positive'
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS" LOKY_MAX_CPU_COUNT="$THREADS"
cd "$REPO"
COMMAND=("$ANALYSIS_PYTHON" -B -u -m five_dataset_features.tsne_paired "$STAGE"
    --input-root "$INPUT_ROOT" --serial "$SERIAL" --output-root "$OUTPUT_ROOT"
    --datasets "${DATASETS[@]}" --models "${MODELS[@]}" --modes "${MODES[@]}"
    --max-samples "$MAX_SAMPLES" --perplexity "$PERPLEXITY"
    --iterations "$ITERATIONS" --seed "$SEED")
if (( OVERWRITE )); then COMMAND+=(--overwrite); fi

echo "Job: ${SLURM_JOB_ID:-preview}; stage: $STAGE"
echo "Python: $ANALYSIS_PYTHON"
echo "Input serial: $INPUT_ROOT/$SERIAL"
echo "Output: $OUTPUT_ROOT"
echo "Scope: ${#DATASETS[@]} datasets x ${#MODELS[@]} models x ${#MODES[@]} modes"
printf '%q ' "${COMMAND[@]}"; printf '\n'
if (( DRY_RUN )); then
    echo 'Dry-run finished; no analysis executed.'
    exit 0
fi

[[ -d "$INPUT_ROOT/$SERIAL" ]] || fail "Missing serial directory: $INPUT_ROOT/$SERIAL"
ml purge
ml load miniconda3
[[ "$ANALYSIS_PYTHON" = /* && -x "$ANALYSIS_PYTHON" ]] || fail "Set ANALYSIS_PYTHON to an executable absolute path: $ANALYSIS_PYTHON"
"$ANALYSIS_PYTHON" -c 'import numpy, matplotlib, sklearn' || fail 'Install NumPy, Matplotlib and scikit-learn in the selected Python environment'
"${COMMAND[@]}"
echo "Completed t-SNE stage: $STAGE"
