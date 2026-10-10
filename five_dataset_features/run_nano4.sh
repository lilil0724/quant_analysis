#!/usr/bin/env bash
#SBATCH --job-name=hqq_feature_analysis
#SBATCH --partition=8gpus
#SBATCH --account=MST114495
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=hqq_feature_analysis_%j.out
#SBATCH --error=hqq_feature_analysis_%j.err

# Submit from quant_analysis on nano4. Module/Python setup matches TGDA inference.
# The queue allocates one GPU; the NumPy analysis currently computes on CPUs.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: sbatch five_dataset_features/run_nano4.sh --serial N [options]
       bash five_dataset_features/run_nano4.sh --serial N --dry-run [options]

  --serial N          Required source experiment serial (for example 20005).
  --input-root PATH   Parent w3_g128_analysis directory (default: <repo>/../../data/w3_g128_analysis).
                      A selected serial directory is also accepted with the same --serial.
  --output-root PATH  Default: <input-root>/<serial>/analysis.
  --stage STAGE       all (default), preflight, analyze, report, or package.
  --datasets CSV      Default: all 16 datasets.
  --models CSV        Default: vit_b16,swin_base_patch4_window7_224_in22k,beitv2_base_patch16_224.
  --modes CSV         Default: ft,fz,cal.
  --dry-run           Print commands; do not load modules or execute analysis.
  -h, --help          Show this help.

Environment: ANALYSIS_PROJECT_ROOT, ANALYSIS_INPUT_ROOT, ANALYSIS_OUTPUT_ROOT,
             ANALYSIS_CONDA_ENV (tgda), ANALYSIS_PYTHON.
SBATCH defaults: partition=8gpus, GPUs/node=1, CPUs=8, memory=64G, time=24h.
Override resources before the script: sbatch --partition=dev --gpus-per-node=1 ...
Input: <serial>/<dataset>/<model>/<mode>/{manifest.json,samples.json,fp/,w3_g128/}.
Output: CKA, statistics, shared-FP PCA, aggregate report and w3_g128_results.zip.
EOF
}
fail() { echo "Error: $*" >&2; exit 2; }
require_value() { [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "Missing value for $1"; }
read_csv_selection() {
    [[ "$2" != ,* && "$2" != *, && "$2" != *,,* ]] || fail "Empty name in $1"
}
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

SERIAL=''
STAGE=all
INPUT_ROOT="${ANALYSIS_INPUT_ROOT:-}"
OUTPUT_ROOT="${ANALYSIS_OUTPUT_ROOT:-}"
DRY_RUN=0
DATASETS=(aircraft cars cub dogs flowers food inat17 moe nabirds pets soyageing soygene soyglobal soylocal vegfru cotton)
MODELS=(vit_b16 swin_base_patch4_window7_224_in22k beitv2_base_patch16_224)
MODES=(ft fz cal)
ALLOWED_DATASETS=" ${DATASETS[*]} "
ALLOWED_MODELS=" ${MODELS[*]} "
ALLOWED_MODES=" ${MODES[*]} "
while [[ $# -gt 0 ]]; do
    case "$1" in
        --serial|--input-root|--output-root|--stage|--datasets|--models|--modes)
            require_value "$@"
            case "$1" in
                --serial) SERIAL="$2" ;;
                --input-root) INPUT_ROOT="$2" ;;
                --output-root) OUTPUT_ROOT="$2" ;;
                --stage) STAGE="$2" ;;
                --datasets) read_csv_selection "$1" "$2"; IFS=',' read -r -a DATASETS <<< "$2" ;;
                --models) read_csv_selection "$1" "$2"; IFS=',' read -r -a MODELS <<< "$2" ;;
                --modes) read_csv_selection "$1" "$2"; IFS=',' read -r -a MODES <<< "$2" ;;
            esac
            shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown option: $1" ;;
    esac
done
[[ "$SERIAL" =~ ^[1-9][0-9]*$ ]] || fail 'Specify the source experiment with --serial N'
case "$STAGE" in all|preflight|analyze|report|package) ;; *) fail "Unknown paired-W3 stage: $STAGE" ;; esac
validate_names dataset "$ALLOWED_DATASETS" "${DATASETS[@]}"
validate_names model "$ALLOWED_MODELS" "${MODELS[@]}"
validate_names mode "$ALLOWED_MODES" "${MODES[@]}"

# Slurm runs a spool copy; use its submission directory, like TGDA inference.
ANALYSIS_PROJECT_ROOT="${ANALYSIS_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[[ -f "$ANALYSIS_PROJECT_ROOT/five_dataset_features/paired.py" ]] || fail 'Submit from updated quant_analysis or set ANALYSIS_PROJECT_ROOT'
REPO="$(cd "$ANALYSIS_PROJECT_ROOT" && pwd)"
INPUT_ROOT="${INPUT_ROOT:-$REPO/../../data/w3_g128_analysis}"
if [[ "$INPUT_ROOT" != /* ]]; then INPUT_ROOT="$PWD/$INPUT_ROOT"; fi
INPUT_ROOT="${INPUT_ROOT%/}"
if [[ "$INPUT_ROOT" == */"$SERIAL" ]]; then INPUT_ROOT="${INPUT_ROOT%/*}"; fi
OUTPUT_ROOT="${OUTPUT_ROOT:-$INPUT_ROOT/$SERIAL/analysis}"
if [[ "$OUTPUT_ROOT" != /* ]]; then OUTPUT_ROOT="$PWD/$OUTPUT_ROOT"; fi
ANALYSIS_CONDA_ENV="${ANALYSIS_CONDA_ENV:-tgda}"
ANALYSIS_PYTHON="${ANALYSIS_PYTHON:-${HOME}/.conda/envs/${ANALYSIS_CONDA_ENV}/bin/python}"
THREADS="${SLURM_CPUS_PER_TASK:-8}"
[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || fail 'SLURM_CPUS_PER_TASK must be positive'
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
cd "$REPO"
COMMAND=("$ANALYSIS_PYTHON" -B -u -m five_dataset_features.paired "$STAGE"
    --input-root "$INPUT_ROOT" --serial "$SERIAL" --output-root "$OUTPUT_ROOT"
    --datasets "${DATASETS[@]}" --models "${MODELS[@]}" --modes "${MODES[@]}")

echo "Job: ${SLURM_JOB_ID:-preview}; node: $(hostname); stage: $STAGE"
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
"${COMMAND[@]}"
echo "Completed stage: $STAGE"
