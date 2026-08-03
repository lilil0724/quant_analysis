# AGENTS.md - quant_analysis

This repository is a standalone, four-stage analysis pipeline for legacy HQQ
quantization experiments from Weights & Biases. It is not a Python package and
has no automated test, lint, type-check, build, or CI configuration.

## Layout

```text
quant_analysis/
|-- download_save_wandb_data.py       # Download finished TiT legacy HQQ runs
|-- summarize_quant.py                # Validate, baseline-match, summarize, XLSX export
|-- corr_quant.py                     # Accuracy-ratio statistics and inference
|-- plot_quant_acc_bit_group.py       # Heatmaps, line plots, and factor figures
|-- requirements.txt
|-- data/
|   |-- backbones_quant.csv            # Downloaded raw legacy HQQ schema
|   `-- quant_fp32_baseline_overrides.csv
`-- results_all/quant/
    |-- quant_summary.csv              # Input to corr_quant.py
    |-- completeness.csv               # Serial-grid validation
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

`--serials` accepts explicit integer lists only; it does not expand ranges.
When overriding the defaults, include the relevant FP32 baseline (`3999` or
`4999`) yourself. Summary and correlation can use the same explicit list to
reproduce a subset analysis.

Example with custom paths in PowerShell:

```powershell
$serials = 4000,4001,4002,4003,4999
python summarize_quant.py --input-file D:\runs\raw.csv `
  --output-file D:\analysis\summary.csv `
  --results-dir D:\analysis\summary --serials $serials
python corr_quant.py --input-file D:\analysis\summary.csv `
  --output-file D:\analysis\corr\factor_ablation_accuracy.csv `
  --results-dir D:\analysis\corr --serials $serials
python plot_quant_acc_bit_group.py `
  --input-file D:\analysis\corr\prepared_accuracy_cells.csv `
  --ablation-file D:\analysis\corr\factor_ablation_accuracy.csv `
  --corr-dir D:\analysis\corr --results-dir D:\analysis\plots `
  --output-file D:\analysis\plots\overview.png
```

## Serial and baseline rules

- The 3X0X scan comprises `3000..3003` through `3700..3703`; serial `3999` is
  its FP32 baseline.
- The 4X0X scan comprises `4000..4003` through `4700..4703`; serial `4999` is
  its FP32 baseline.
- A baseline only matches rows from the same series, dataset, model, and
  checkpoint kind. Never cross-match `3999` and `4999`.
- `summarize_quant.py` checks every expected 32-cell scan grid. It reads nbits
  and group size directly from the raw CSV rather than deriving them from serial.
- Checkpoint paths must include `/cal_ckpts/`, `/ft_ckpts/`, or `/fz_ckpts/`.

## Analysis scope

The sole analysis outcome is:

```text
accuracy_ratio = hqq_top1 / fp32_top1
```

The factors are `hqq_nbits`, `hqq_group_size`, checkpoint kind, dataset, and
model. Resource, timing, loss, and top-k count outputs are intentionally out of
scope. The summary retains legacy HQQ fields because they are the schema
currently logged by the TiT project.

`corr_quant.py` produces global and within-context correlation supplements,
trend/levels factor models, and delta R2 ablation tables.

## Verification

After code changes, run the complete downstream pipeline against the checked-in
CSV before downloading new WandB data. Confirm that:

1. `completeness.csv` lists no missing cells for expected contexts.
2. `configuration_quality.csv` identifies any non-fixed HQQ setup fields.
3. Every dataset has an accuracy-only XLSX workbook.
4. `results_all/quant/corr/` contains no PNG files.
5. `results_all/quant/plots/` contains the overview, heatmaps, line plots, and
   factor-contribution plot.
