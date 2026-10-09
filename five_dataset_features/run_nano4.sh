#!/usr/bin/env bash
#SBATCH --job-name=hqq_feature_analysis
#SBATCH --account=MST114495
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=hqq_feature_analysis_%j.out
#SBATCH --error=hqq_feature_analysis_%j.err

# Submit from the quant_analysis checkout on nano4.
# Environment setup follows TGDA/scripts/run_exp_nano5.sh.
# Partition defaults to the cluster's configured queue.
# Analysis uses CPUs only. Override resources with sbatch options when needed.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: sbatch five_dataset_features/run_nano4.sh [options]
       bash five_dataset_features/run_nano4.sh --dry-run [options]

  --input-root PATH   Full inference output (default: <repo>/../../data/five_dataset_hqq).
  --stage STAGE       all (default), analyze, report, package, or smoke.
                      all = dataset analyses + report + ZIP; smoke is separate.
  --datasets CSV      Comma-separated subset; default: all 16 datasets.
  --dry-run           Print commands without loading modules or writing results.
  -h, --help          Show this help.

Environment:
  ANALYSIS_PROJECT_ROOT  nano4 checkout (default: SLURM_SUBMIT_DIR or current directory).
  ANALYSIS_INPUT_ROOT    Optional full inference output root.
  ANALYSIS_CONDA_ENV     Conda environment (default: opencode_env).
  ANALYSIS_PYTHON        Python (default: $HOME/.conda/envs/$ANALYSIS_CONDA_ENV/bin/python).

Outputs: <input-root>/analysis/ and <input-root>/analysis/five_dataset_hqq_results.zip.
Existing analysis outputs are regenerated. Full features are read without modification.
EOF
}

fail() { echo "Error: $*" >&2; exit 2; }
require_value() { [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "Missing value for $1"; }

STAGE=all
INPUT_ROOT="${ANALYSIS_INPUT_ROOT:-}"
DRY_RUN=0
DATASETS=(aircraft cars cub dogs flowers food inat17 moe nabirds pets soyageing soygene soyglobal soylocal vegfru cotton)
ALLOWED_DATASETS=" ${DATASETS[*]} "
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-root) require_value "$@"; INPUT_ROOT="$2"; shift 2 ;;
        --stage) require_value "$@"; STAGE="$2"; shift 2 ;;
        --datasets)
            require_value "$@"
            [[ "$2" != ,* && "$2" != *, && "$2" != *,,* ]] || fail 'Empty dataset in --datasets'
            IFS=',' read -r -a DATASETS <<< "$2"
            shift 2
            ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown option: $1" ;;
    esac
done
case "$STAGE" in all|analyze|report|package|smoke) ;; *) fail "Unknown stage: $STAGE" ;; esac
declare -A SEEN=()
for dataset in "${DATASETS[@]}"; do
    [[ "$ALLOWED_DATASETS" == *" $dataset "* ]] || fail "Unknown dataset: $dataset"
    [[ ! -v "SEEN[$dataset]" ]] || fail "Duplicate dataset: $dataset"
    SEEN[$dataset]=1
done

# Slurm executes a copied script from its spool directory: use the submit root.
ANALYSIS_PROJECT_ROOT="${ANALYSIS_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
REPO="$ANALYSIS_PROJECT_ROOT"
[[ -f "$REPO/five_dataset_features/analyze.py" ]] || fail 'Submit from quant_analysis or set ANALYSIS_PROJECT_ROOT'
REPO="$(cd "$REPO" && pwd)"
INPUT_ROOT="${INPUT_ROOT:-$REPO/../../data/five_dataset_hqq}"
# Relative input paths refer to the submission directory, before changing directory.
if [[ "$INPUT_ROOT" != /* ]]; then INPUT_ROOT="$PWD/$INPUT_ROOT"; fi
ANALYSIS_CONDA_ENV="${ANALYSIS_CONDA_ENV:-tgda}"
ANALYSIS_PYTHON="${ANALYSIS_PYTHON:-${HOME}/.conda/envs/${ANALYSIS_CONDA_ENV}/bin/python}"
cd "$REPO"
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
THREADS="${SLURM_CPUS_PER_TASK:-8}"
[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || fail 'SLURM_CPUS_PER_TASK must be a positive integer'
export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"

run_python() {
    if (( DRY_RUN )); then
        printf '  '; printf '%q ' "$ANALYSIS_PYTHON" -B -u "$@"; printf '\n'
    else
        "$ANALYSIS_PYTHON" -B -u "$@"
    fi
}

if (( ! DRY_RUN )); then
    [[ -d "$INPUT_ROOT" ]] || fail "Input root not found: $INPUT_ROOT"
    INPUT_ROOT="$(cd "$INPUT_ROOT" && pwd)"
    # Match the user's working TGDA inference environment initialization on nano4.
    ml purge
    ml load miniconda3
    [[ "$ANALYSIS_PYTHON" = /* && -x "$ANALYSIS_PYTHON" ]] || fail "Set ANALYSIS_PYTHON to an executable absolute path: $ANALYSIS_PYTHON"
    if [[ "$STAGE" == all || "$STAGE" == analyze || "$STAGE" == package ]]; then
        CONDITIONS=(fp)
        for w in 3 4; do
            for g in 8 128; do
                stem="w${w}_g${g}"
                CONDITIONS+=("$stem" "${stem}_a8" "${stem}_a8_qkv" "${stem}_a4" "${stem}_a4_qkv")
            done
        done
        # Check every requested dataset before starting; the analyzer verifies hashes and pairing.
        for dataset in "${DATASETS[@]}"; do
            [[ -f "$INPUT_ROOT/$dataset/dataset.json" ]] || fail "Missing $dataset/dataset.json"
            for condition in "${CONDITIONS[@]}"; do
                for filename in manifest.json features.npy predictions.csv precision_probe.npy; do
                    [[ -f "$INPUT_ROOT/$dataset/$condition/$filename" ]] || fail "Missing $dataset/$condition/$filename"
                done
            done
            if [[ "$STAGE" == all || "$STAGE" == package ]]; then
                [[ -f "$INPUT_ROOT/$dataset/exemplars/manifest.json" ]] || fail "Missing $dataset/exemplars/manifest.json for ZIP"
            fi
        done
    fi
    run_python -c 'import sys, numpy, matplotlib; print("Python:", sys.executable); print("NumPy:", numpy.__version__); print("Matplotlib:", matplotlib.__version__)'
fi

echo "Stage: $STAGE"
echo "SLURM job ID: ${SLURM_JOB_ID:-preview}"
echo "Node: $(hostname)"
echo "Python: $ANALYSIS_PYTHON"
echo "Repository: $REPO"
echo "Input: $INPUT_ROOT"
echo "Output: $INPUT_ROOT/analysis"
echo "Datasets (${#DATASETS[@]}): ${DATASETS[*]}"
if [[ "$STAGE" == smoke ]]; then
    run_python -m five_dataset_features.smoke --root "$INPUT_ROOT"
else
    if [[ "$STAGE" == all || "$STAGE" == analyze ]]; then
        for dataset in "${DATASETS[@]}"; do
            echo "Analyzing: $dataset"
            run_python -m five_dataset_features.analyze dataset --root "$INPUT_ROOT" --dataset "$dataset"
        done
    fi
    if [[ "$STAGE" == all || "$STAGE" == analyze || "$STAGE" == report ]]; then
        run_python -m five_dataset_features.analyze report --root "$INPUT_ROOT" --datasets "${DATASETS[@]}"
    fi
    if [[ "$STAGE" == all || "$STAGE" == package ]]; then
        run_python -m five_dataset_features.analyze package --root "$INPUT_ROOT" --datasets "${DATASETS[@]}"
    fi
fi
if (( DRY_RUN )); then echo 'Dry-run finished; no analysis executed.'; else echo "Analysis stage completed: $STAGE"; fi
