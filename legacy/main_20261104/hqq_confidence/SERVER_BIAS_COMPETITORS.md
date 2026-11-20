# nano4: multi-dataset / multi-setting W3/G128 diagnostics

This portable workflow runs the previously discussed class-bias and competitor
tests directly on TGDA paired exports. It needs NumPy, pandas, SciPy,
scikit-learn and threadpoolctl, but no TGDA/HQQ/torch/W&B installation or GPU
computation. Full layer features and checkpoints are not loaded.

Expected input:

```text
/work/kyle0724/data/w3_g128_analysis/<serial>/<dataset>/<model>/<ft|fz|cal>/
  manifest.json                 # status=complete, identity and checksums
  samples.json                  # ordered sample IDs and targets
  fp/logits.npy                 # full [N,C] float32 logits
  fp/predictions.csv
  w3_g128/logits.npy
  w3_g128/predictions.csv
```

Use the source directory written by `tools/features_analysis/w3_g128_analysis.py`.
A compact `all_pairs.csv`/feature-summary download alone lacks the full class
logits required here. Serials remain separate. Dataset/model/mode selections
must form a complete matrix; missing pairs are errors, never silently dropped.

## Install and check

Synchronize these files into the server's `quant_analysis/hqq_confidence/`:
`server_bias_competitors.py`, `bias_competitor_core.py`,
`run_bias_competitors_nano4.sh`, `requirements_bias_competitors.txt` and this guide.
The downloadable ZIP contains those files and the package's `__init__.py`.

```bash
cd /work/kyle0724/project/quant_analysis
ml purge
ml load miniconda3
PYTHON="$HOME/.conda/envs/opencode_env/bin/python"
"$PYTHON" -m pip install -r hqq_confidence/requirements_bias_competitors.txt

# Example serial only: select the serial whose source exports you intend to test.
bash hqq_confidence/run_bias_competitors_nano4.sh --serial 20004 --dry-run

# Read-only validation of every selected pair; can be submitted via sbatch too.
bash hqq_confidence/run_bias_competitors_nano4.sh --serial 20004 --preflight-only
```

Preflight hashes the manifest, samples and both logits/prediction CSVs, checks
sample/target/winner/margin consistency, and verifies weight-only W3/G128.
Within a model/mode, export code, HQQ source and quantization/precision protocols
must agree across selected datasets. Pooling and preprocessing are recorded per
dataset because they can legitimately vary with checkpoint configuration.
Only the files used by these diagnostics are hashed; unused layer arrays are
outside this stage's validation scope.

## Run

```bash
# Defaults: all 16 datasets x 3 models x ft/fz/cal = 144 independent paired analyses.
sbatch hqq_confidence/run_bias_competitors_nano4.sh --serial 20004

# BEiTv2 across all datasets and all three checkpoint modes.
sbatch hqq_confidence/run_bias_competitors_nano4.sh --serial 20004 \
  --models beitv2_base_patch16_224 --modes ft,fz,cal

# A smaller complete-test pilot; use a distinct output to avoid collisions.
sbatch hqq_confidence/run_bias_competitors_nano4.sh --serial 20004 \
  --datasets cub,pets,cars --models beitv2_base_patch16_224 --modes ft \
  --output-dir /work/kyle0724/data/w3_g128_analysis/20004/analysis/bias_competitors_pilot
```

Override input/output, environment or Python when needed:

```bash
ANALYSIS_PYTHON="$HOME/.conda/envs/tgda/bin/python" \
sbatch hqq_confidence/run_bias_competitors_nano4.sh --serial 20005 \
  --input-root /work/kyle0724/data/w3_g128_analysis \
  --output-dir /work/kyle0724/data/w3_g128_analysis/20005/analysis/bias_competitors_v1
```

That Python must have the listed analysis dependencies. The launcher defaults to
`opencode_env`. It uses the existing nano4 Slurm defaults: account MST114495,
partition 8gpus, one allocated GPU, eight CPUs, 64 GB RAM and 24 hours. The
diagnostics compute on CPUs. Override Slurm resources before the script filename
if using a different approved partition. Logs are `bias_competitors_<job>.out/.err`.

You can also run directly on a suitable server compute node:

```bash
"$PYTHON" -B -u -m hqq_confidence.server_bias_competitors \
  --input-root /work/kyle0724/data/w3_g128_analysis --serial 20004 \
  --datasets cub,pets,cars --models beitv2_base_patch16_224 --modes ft \
  --output-dir /work/kyle0724/data/w3_g128_analysis/20004/analysis/bias_competitors_direct
```

## Outputs and meaning

Default output is `<input-root>/<serial>/analysis/bias_competitors_v1/`.
An existing output directory is preserved; choose another `--output-dir` to rerun.

| File | Meaning |
|---|---|
| `REPORT.md`, `setting_summary.csv` | Per-model/mode results, with datasets weighted equally |
| `bias_results.csv` | FP, W3 and cross-fit corrected accuracy; gains, rescued/harmed counts and shuffled controls |
| `fold_audit.csv`, per-dataset `bias_vectors.npz` | Fold sizes/coverage/gains and estimated class bias |
| `winner_ranks.csv` | Tie-aware fraction of swaps/damage involving FP rank >=3 |
| `heldout_prediction_comparisons.csv` | Competitors versus nonlinear-margin comparator on heldout datasets |
| `heldout_prediction_scores.csv`, per-setting `model_fits.json` | Individual scores, coefficients and train/test dataset identities |
| `prediction_status.csv` | Explicit incomplete/not-applicable diagnostics |
| `coverage.csv`, `source_audit.csv`, `protocol.json` | Precision/pooling/preprocessing/identity, used-source hashes and fixed analysis protocol |
| `completion.json` | Workflow completion plus output checksums; written only after used-source hashes are rechecked |

`gain_pp > 0` favors bias correction. `delta_brier < 0` favors additional
competitor information. Brier changes are not accuracy percentage points.
Models and checkpoint modes are analyzed separately; cells are repeated
measurements rather than independent datasets.

The five-fold bias test uses four folds to estimate centered quant-FP class
offsets and corrects the remaining fold, with fixed strength 1 and five shuffled
class controls. Folds use targets for stratification; estimated biases use logits.
This is retrospective cross-fitting on test-loader samples, not a separately
calibrated benchmark. The original checkpoints are not modified.

The competitor test uses nearby-class counts at 0.25/0.5/1 FP logit std, with a
fixed nonlinear normalized-margin comparator and log class-count control.
LODO fits separately within each model/mode, giving each training dataset total
weight 1. At least three datasets are required; empty populations, single-class
training outcomes and failed convergence are explicitly incomplete. No primary
LODO average is reported when any selected fold is incomplete. A single-class
test outcome still has valid Brier/log-loss but undefined AUROC. Soybean-family
holdout is exploratory and requires at least two datasets in each family group.

CAL logits retain the exporter's crop/flip aggregation. Checkpoint and test-CSV
identities are manifest declarations; this script does not reopen the original
weights/images or certify the original evaluation implementation. Dataset and
checkpoint-training effects remain confounded. These diagnostics make no causal
or statistical-significance claims.

Use `--save-sample-results` to retain derived features/labels/predictions for
further verification. Raw logits are not copied. Logit operations use read-only
memory mapping and bounded chunks; lowering `--chunk-size` reduces working
memory. Model-fitting memory depends on the combined sample count in one
model/mode. Large exports, especially iNat17, also require substantial sequential
disk reads for checksums and diagnostics.
