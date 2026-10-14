# Centered FP backbone class distance from Serial 20004

This module reads existing paired FP/W3 exports. It uses CPU NumPy
only, performs no model inference, and does not load checkpoints or images.
Requirements: Python >= 3.9 and NumPy. Run from the `quant_analysis` repository
root with `python -m five_dataset_features.centered_class_distance`. The dedicated
Slurm launcher is `five_dataset_features/run_centered_class_distance_nano4.sh`.

The input schema is the original paired export:

```text
<input-root>/20004/<dataset>/<model>/<ft|fz|cal>/
├── manifest.json
├── samples.json
├── fp/final.npy
├── fp/predictions.csv
└── w3_g128/predictions.csv
```

Source manifests must be complete, have matching serial/dataset/model/mode,
declare weight-only W3/G128, and record the consumed files' SHA-256 values.
The script verifies those files, sample IDs/targets, prediction alignment, array
shape and dtype. It rechecks file hashes after computing a context. It does not
verify unused layers or logits, and it does not accept the newer per-condition
`features/final.npy` schema as if it were Serial 20004's paired schema.

## Computation

For each dataset/model/mode independently, compute the sample-weighted mean of
all saved FP final features in float64. Compare these two cosine geometries:

```text
raw:      x / ||x||
centered: (x - mean_FP) / ||x - mean_FP||
```

Centering occurs before L2 normalization. Classes are not individually centered.
Inter distance is the equally weighted average over unordered class pairs of
their cross-class sample-pair cosine distances. Intra distance is equally
weighted across classes with at least two samples, using distinct sample pairs.
Both computations use class sums; no N-by-N distance matrix is allocated.
Arrays are memory-mapped and processed in chunks, using O(chunk_size*D + C*D)
memory rather than loading the full feature array.

FP/W3 accuracy and relative loss are recomputed from the existing, verified
prediction CSVs. No new accuracy evaluation is performed.

If centering produces any exactly zero-norm sample, the centered distances for
that context are left undefined rather than silently dropping samples.
`centered_zero_norm_count` and `centered_status` explain those rows. Single-class
inter distance and undefined ratios are empty CSV fields. `centered_min_norm`
helps identify near-zero centered vectors whose directions may be unstable.

`mean_energy_fraction = ||mean_FP||^2 / mean_i(||x_i||^2)` describes how much raw
feature energy belongs to the shared mean. It is a diagnostic, not a causal
measure. Centering changes the geometry; a larger inter distance alone does not
establish better class separability or quantization robustness. Compare intra,
gap and normalized separation as well. CAL measurements remain backbone features
before attention pooling, using the original raw-image feature view.

## Server commands

Sync the module and launcher into the server's `quant_analysis` checkout.
Submit from the repository root. These commands explicitly keep results inside
the owning analysis module; source raw exports remain in the data directory:

```bash
cd /work/kyle0724/project/quant_analysis
bash -n five_dataset_features/run_centered_class_distance_nano4.sh

# Plan 16 datasets x 3 models x CAL = 48 contexts. No files are written.
bash five_dataset_features/run_centered_class_distance_nano4.sh \
  --input-root /work/kyle0724/data/w3_g128_analysis \
  --serial 20004 --modes cal \
  --output-root five_dataset_features/results/20004/centered_fp_distance_cal --dry-run

# Compute the 48 CAL contexts from existing raw features.
sbatch five_dataset_features/run_centered_class_distance_nano4.sh \
  --input-root /work/kyle0724/data/w3_g128_analysis \
  --serial 20004 --modes cal \
  --output-root five_dataset_features/results/20004/centered_fp_distance_cal
```

For all 144 FT/FZ/CAL contexts, omit `--modes cal`:

```bash
sbatch five_dataset_features/run_centered_class_distance_nano4.sh \
  --input-root /work/kyle0724/data/w3_g128_analysis --serial 20004 \
  --output-root five_dataset_features/results/20004/centered_fp_distance_all
```

To pilot one context before processing the whole selection:

```bash
sbatch five_dataset_features/run_centered_class_distance_nano4.sh \
  --input-root /work/kyle0724/data/w3_g128_analysis --serial 20004 \
  --datasets cub --models beitv2_base_patch16_224 --modes cal \
  --output-root five_dataset_features/results/20004/centered_fp_distance_pilot
```

The launcher requests 4 CPU cores, 8 GB RAM and 6 hours in `normal`, with no GPU.
Override scheduling resources through `sbatch` if required by the actual cluster.
It defaults to `$HOME/.conda/envs/tgda/bin/python`, invoking that executable
directly. Use another existing NumPy environment with:

```bash
export ANALYSIS_CONDA_ENV=opencode_env
# Or point to an existing Python executable:
# export ANALYSIS_PYTHON="$HOME/.conda/envs/opencode_env/bin/python"
```

`sbatch` copies its script into a spool directory. Submitting from the repository
root lets the launcher use `SLURM_SUBMIT_DIR`. If submitting from another
directory, export `ANALYSIS_PROJECT_ROOT` to the actual `quant_analysis` checkout.
The Python CLI accepts space-separated names for `--datasets`, `--models` and
`--modes`, for example `--modes ft fz cal`; comma-separated lists are not accepted.

## Outputs

The CAL submission above writes:

```text
/work/kyle0724/project/quant_analysis/five_dataset_features/results/20004/centered_fp_distance_cal/
├── centered_class_distances.csv
├── run_config.json
└── means/<dataset>/<model>/<mode>/fp_mean.npy
```

For compatibility with the standalone version, omitting `--output-root` keeps
its original default: `<selected-input-serial>/analysis/centered_fp_distance/`.
Use the explicit module output path shown above for this repository workflow.

The CSV has one row per context, retaining identity, N/classes/dimension,
`fp_accuracy`, `w3_accuracy`, `relative_loss_percent`, `raw_d_inter`,
`raw_d_intra`, `raw_distance_gap`, `raw_distance_ratio`,
`raw_normalized_separation`, the corresponding `centered_*` metrics, and shared
mean/zero-norm diagnostics. `run_config.json` records selection, source hashes,
code hash, calculation definition and output hashes. Saved means are float64.

Existing output directories are preserved: use another `--output-root` to rerun.
Selected missing or incompatible contexts fail; no silent skipping is performed.
Failures after computation starts preserve an adjacent `.partial` directory with
a failed run record. Input arrays and the original Serial 20004 results remain
unchanged. Use an unoccupied output directory for a retry.

Logs are written to the submission directory:
`centered_distance_<JOBID>.out` and `centered_distance_<JOBID>.err`.
Check the final `Complete:` line and `run_config.json` status. For the CAL-only
selection, expect 48 CSV rows; for the full selection, expect 144. A dry-run
prints the intended selection; it does not establish raw file availability.

## Verification boundary

Local verification used NumPy and five synthetic tests covering explicit
sample-pair parity, unequal class weights, global centering, zero/invalid
features, original paired-schema hashes/alignment, existing-output protection,
and source files remaining unchanged:

```bash
conda run -n opencode_env python -B -m unittest five_dataset_features.tests.test_centered_class_distance -v
```

All five tests passed. No server data run or Slurm submission was performed.
On the development Windows computer, `conda run` could not execute its wrapper;
the same suite passed using the environment's Python executable directly:

```powershell
& 'C:/Users/kyle0/miniconda3/envs/opencode_env/python.exe' -B -m unittest five_dataset_features.tests.test_centered_class_distance -v
```

Local Windows Bash validation was blocked by the sandbox's MSYS initialization
restriction; run the provided `bash -n` check on the server before submitting.
