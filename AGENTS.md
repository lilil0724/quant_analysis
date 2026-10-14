# AGENTS.md — quant_analysis

This repository holds independent HQQ analyses. Each analysis owns its code, `data/`, `results/`, tests and operating README. Use `conda run -n opencode_env python ...` for all Python commands.

## Choose the analysis first

- Legacy accuracy-ratio work: read `legacy_hqq/README.md`. Its baseline is exact checkpoint-and-seed FP32. The four scripts are inside `legacy_hqq/`.
- Dataset ranking stability: read `dataset_robustness/README.md`. It consumes a fixed, baseline-matched legacy summary in its own `data/upstream/`.
- FGIR / Ultra-FGIR contrasts: read `granularity_robustness/README.md`. Its group mapping is in `data/source/`.
- Matched-sample confidence: read `hqq_confidence/README.md`. Its baseline is matched AMP unquantized, with separate data and outputs. Keep its analysis changes inside that module.
- Five-dataset feature geometry: read `five_dataset_features/README.md`. Its full inference arrays are large and may live in an external input root during the remaining layout migration.

## Work sequence

1. Inspect `git status` and the relevant module README. Preserve existing uncommitted work and archived outputs.
2. Change the owning module and its tests. Give every new input a recorded source and SHA-256. Keep stage output inside the owning module.
3. Run its tests in `opencode_env`, then run the changed downstream stages against existing local inputs. Verify the saved artifacts using the module's verifier where available.
4. Update the module README and the root index when commands or paths change.

## Required Slurm script header

Every new or modified server Bash script intended for `sbatch` submission must
start with the following header, before any executable commands. Keep
`--partition=8gpus` and `--account=MST114495` fixed. Fill every other field with
a concrete value appropriate to the job; the empty values below are template
placeholders. This requirement applies to Slurm launchers, including CPU-based
analysis jobs; Python modules use their Bash launcher for these directives.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=
#SBATCH --partition=8gpus
#SBATCH --account=MST114495
#SBATCH --nodes=
#SBATCH --gpus-per-node=
#SBATCH --cpus-per-task=
#SBATCH --mem=
#SBATCH --time=
#SBATCH --output=
#SBATCH --error=
```

Before delivering a script, verify that all listed directives are present,
all values are filled, and output/error filenames distinguish jobs (for example,
`analysis_%j.out` and `analysis_%j.err`). Additional Slurm directives may be
included as required by the job.

<<<<<<< HEAD
`legacy_hqq/run.py` and `hqq_confidence/run.py` expose `acquire → prepare → analyze → plot`. Dataset and granularity robustness currently combine calculation and plotting; their split is recorded in `docs/repository_layout_plan.md`. Do not manually edit generated CSV, XLSX, Markdown or PNG outputs; rerun their owning stage. W&B downloading is only needed when local inputs are missing or an explicit refresh is requested.
=======
## Environment

Operational inference data and feature-analysis results reside on nano4, not
on the development Windows computer. Server checkouts are
`/work/kyle0724/project/TGDA` and `/work/kyle0724/project/quant_analysis`;
data reside under `/work/kyle0724/data/`.

Legacy CSV/confidence workflows use `opencode_env`. The nano4 feature-analysis
launcher currently defaults to the user's selected `tgda` environment, matching
inference; override it with `ANALYSIS_CONDA_ENV=opencode_env` if desired. Initialize
the module like TGDA's inference script and invoke the selected Python directly:

```bash
ml purge
ml load miniconda3
"$HOME/.conda/envs/opencode_env/bin/python" -m pip install -r requirements.txt
```

Downloading requires `WANDB_API_KEY` or `wandb login`.

## nano4 feature analysis

- Submit `sbatch five_dataset_features/run_nano4.sh --serial N` from the nano4
  quant_analysis checkout. Use Bash with `--dry-run` to inspect commands.
- `--input-root` names the parent `<checkout>/../../data/w3_g128_analysis` directory.
  Selected input is `<input-root>/<serial>`, organized as dataset/model/checkpoint
  mode, using TGDA paired FP/W3 manifests and layer files.
  Output defaults to `<input-root>/<serial>/analysis`; preserve all source files.
- The launcher calls `five_dataset_features.paired` and requires the complete
  selected 16-dataset × 3-model × ft/fz/cal matrix by default. Scope can be narrowed
  explicitly with `--datasets`, `--models`, and `--modes`; never silently skip pairs.
- Resources are partition 8gpus and one GPU per node. Module loading and direct
  Python execution match TGDA inference; default environment is tgda. Analysis
  metrics/PCA use NumPy CPUs despite GPU allocation.
- Validate source completion, identity, checksums, sample pairing and variable
  layer shapes before writing analysis output. Preserve source provenance in
  aggregate reports; package only a whitelist of compact generated results.
- The older `five_dataset_hqq` 21-condition schema remains supported by the
  separate analyze.py/smoke.py/run_local.ps1 entrypoints; do not mix its data with
  paired W3 exports or change the legacy CSV/confidence scripts.
- Environment overrides: `ANALYSIS_PROJECT_ROOT`, `ANALYSIS_INPUT_ROOT`,
  `ANALYSIS_OUTPUT_ROOT`, `ANALYSIS_CONDA_ENV`, `ANALYSIS_PYTHON`.
- Launcher checks on the development computer do not establish nano4 completion,
  resource sufficiency, environment availability or source data validity.

## Pipeline

Run stages from the repository root in this order:

```powershell
python download_save_wandb_data.py
python summarize_quant.py
python corr_quant.py
python plot_quant_acc_bit_group.py
```

All stages write `run_config.json` next to their output. The summary stage
reads `data/backbones_quant.csv`; correlation reads
`results_all/quant/quant_summary.csv`; plotting reads correlation outputs.

## CLI conventions

The pipeline uses command-line arguments only; there is no project config file.
Every stage keeps defaults that connect to the next stage, while accepting full
paths for reusable runs:

| Stage | `--input-file` | `--output-file` | `--results-dir` |
|---|---|---|---|
| download | N/A | downloaded raw CSV | downloader `run_config.json` |
| summarize | raw WandB CSV | primary summary CSV | validation CSVs and XLSX files |
| corr | summary CSV | primary factor-ablation CSV | correlations, factor models, and Markdown |
| plot | prepared accuracy-cell CSV | primary overview PNG | all generated PNGs |

`--serials` accepts explicit integer lists only; it does not expand ranges. It
is optional in the analysis stages: summary uses it to narrow raw rows and
correlation matches it against HQQ source-serial provenance. When subsetting
summary input, include any exact checkpoint-and-seed FP32 baseline rows needed
for matching.

Example with custom paths in PowerShell:

```powershell
python summarize_quant.py --input-file D:\runs\raw.csv `
  --output-file D:\analysis\summary.csv `
  --results-dir D:\analysis\summary
python corr_quant.py --input-file D:\analysis\summary.csv `
  --output-file D:\analysis\corr\factor_ablation_accuracy.csv `
  --results-dir D:\analysis\corr
python plot_quant_acc_bit_group.py `
  --input-file D:\analysis\corr\prepared_accuracy_cells.csv `
  --ablation-file D:\analysis\corr\factor_ablation_accuracy.csv `
  --corr-dir D:\analysis\corr --results-dir D:\analysis\plots `
  --output-file D:\analysis\plots\overview.png
```

## Condition and baseline rules

- Analysis conditions come from raw `hqq_nbits` and `hqq_group_size`; serial is
  provenance and an optional manual filter only.
- A baseline only matches rows with the same dataset, normalized model name,
  complete checkpoint path, and seed. Same-identity FP32 repeats use median
  top-1.
- Repeated HQQ rows from multiple serials are combined within a seed. Seeds are
  paired before aggregation; a condition with any unmatched seed remains an
  audit row and does not enter correlation or figures.
- `completeness.csv` reports observed nbits/group sizes, seed coverage, and
  unmatched conditions without assuming a serial grid.
- Checkpoint paths must include either `/cal_ckpts/`, `/ft_ckpts/`,
  `/fz_ckpts/`, or `/ckpt/cal/`, `/ckpt/ft/`, `/ckpt/fz/`.

## Analysis scope

The sole analysis outcome is:

```text
accuracy_ratio = hqq_top1 / fp32_top1
```

The factors are `hqq_nbits`, `hqq_group_size`, checkpoint kind, dataset, and
model. Checkpoint path is retained as context/provenance. Resource, timing,
loss, and top-k count outputs are intentionally out of scope. The summary
retains legacy HQQ fields because they are the schema currently logged by the
TiT project.

`corr_quant.py` produces global and within-context correlation supplements,
trend/levels factor models, and delta R2 ablation tables.

## Verification

After code changes, run the complete downstream pipeline against the checked-in
CSV before downloading new WandB data. Confirm that:

1. `completeness.csv` reports all observed contexts and identifies unmatched conditions.
2. `configuration_quality.csv` identifies any non-fixed HQQ setup fields.
3. Every dataset has an accuracy-only XLSX workbook.
4. `results_all/quant/corr/` contains no PNG files.
5. `results_all/quant/plots/` contains the overview, heatmaps, line plots, and
   factor-contribution plot.
>>>>>>> 8dea61bfc1447b4788acbe69d706d79f78a728bb
