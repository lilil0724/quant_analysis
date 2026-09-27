# AGENTS.md - quant_analysis

This repository contains a standalone four-stage analysis pipeline for legacy
HQQ quantization experiments from Weights & Biases. The legacy workflow is not
a Python package and has no lint, type-check, build, or CI configuration. Its
serial-independent summary behaviour is covered by a small `unittest` suite.

A separate matched-sample confidence workflow, including its own tests and
operating contract, is documented in `hqq_confidence/README.md`. Keep its data,
outputs, and changes isolated under `hqq_confidence/`; do not change the four
legacy scripts when working on it.

## Layout

```text
quant_analysis/
|-- download_save_wandb_data.py       # Download finished TiT legacy HQQ runs
|-- summarize_quant.py                # Validate, baseline-match, summarize, XLSX export
|-- corr_quant.py                     # Accuracy-ratio statistics and inference
|-- plot_quant_acc_bit_group.py       # Heatmaps, line plots, and factor figures
|-- requirements.txt
|-- tests/                            # Legacy serial-independent regression tests
|-- hqq_confidence/                    # Parallel artifact/confidence workflow
|   `-- README.md                      # Its CLI, schemas, metrics, and tests
|-- data/
|   |-- backbones_quant.csv            # Downloaded raw legacy HQQ schema
`-- results_all/quant/
    |-- quant_summary.csv              # Input to corr_quant.py
    |-- completeness.csv               # Observed-condition validation
    |-- configuration_quality.csv      # Fixed-config validation
    |-- quant_<dataset>.xlsx           # One accuracy workbook per dataset
    |-- corr/                          # Statistical tables, Markdown, model metadata
    `-- plots/                         # All generated PNG figures
```

Generated files under `results_all/quant/` are reproducible outputs. Do not
edit them manually; rerun their owning stage instead.

## Environment

All Python commands must use `opencode_env`:

```powershell
conda activate opencode_env
pip install -r requirements.txt
```

Downloading requires `WANDB_API_KEY` or `wandb login`.

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
