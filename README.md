# HQQ Quantization Analysis

Reusable four-stage analysis flow for legacy HQQ quantization experiments
stored in Weights & Biases. The pipeline validates experiment coverage, pairs
each HQQ result with the correct FP32 baseline, analyzes accuracy retention,
and produces statistical tables plus PNG figures.

The only primary outcome is:

```text
accuracy_ratio = hqq_top1 / fp32_top1
```

Resource, timing, loss, and top-k analyses are intentionally out of scope.

## Pipeline

```text
W&B TiT runs
  -> download_save_wandb_data.py
  -> data/backbones_quant.csv
  -> summarize_quant.py
  -> results_all/quant/quant_summary.csv + XLSX workbooks
  -> corr_quant.py
  -> results_all/quant/corr/ statistical tables and Markdown
  -> plot_quant_acc_bit_group.py
  -> results_all/quant/plots/ PNG figures
```

| Stage | Script | Purpose |
|---|---|---|
| Download | `download_save_wandb_data.py` | Downloads finished legacy HQQ runs from `nycu_pcs/TiT`. |
| Summarize | `summarize_quant.py` | Validates serial grids, pairs FP32 baselines, averages repeated experiments, and exports per-dataset XLSX files. |
| Correlate | `corr_quant.py` | Calculates correlations, trend/levels models, delta R2 ablations, bootstrap summaries, and robust-model diagnostics. |
| Plot | `plot_quant_acc_bit_group.py` | Generates heatmaps, line plots, factor plots, and coefficient figures as PNG files. |

## Requirements

Use the `opencode_env` Conda environment:

```powershell
conda activate opencode_env
pip install -r requirements.txt
```

Downloading from W&B requires authentication. Set `WANDB_API_KEY` or run:

```powershell
wandb login
```

## Quick Start

From the repository root, run the default complete pipeline:

```powershell
python download_save_wandb_data.py
python summarize_quant.py
python corr_quant.py
python plot_quant_acc_bit_group.py
```

If `data/backbones_quant.csv` already exists, skip download and run the last
three commands.

## Serial Rules and Baselines

Each scan has 32 HQQ cells: eight nbits codes times four group-size codes.

| Scan | HQQ serials | FP32 baseline |
|---|---|---|
| 3X0X | `3000..3003` through `3700..3703` | `3999` |
| 4X0X | `4000..4003` through `4700..4703` | `4999` |

FP32 baselines are only paired within the same series, dataset, model, and
checkpoint kind. The pipeline never cross-matches `3999` with 4X0X or `4999`
with 3X0X.

Checkpoint paths must contain one of:

```text
/cal_ckpts/
/ft_ckpts/
/fz_ckpts/
```

## Outputs

### Summary

`results_all/quant/` contains:

- `quant_summary.csv`: one row per aggregated HQQ experiment cell.
- `completeness.csv`: expected versus observed serial grids and baseline status.
- `configuration_quality.csv`: checks that non-factor HQQ settings are fixed.
- `serial_mapping.csv`: serial-code mapping to actual nbits and group size.
- `quant_<dataset>.xlsx`: accuracy-only workbook for each dataset.

Repeated runs with the same series, dataset, model, checkpoint, serial, nbits,
and group size are combined into one experiment. `hqq_top1` is their mean; the
summary also retains median, standard deviation, run count, and seed count.

### Statistics

`results_all/quant/corr/` contains:

- `correlations_accuracy.csv`: global and within-context correlation results.
- `factor_effects_accuracy_trend.csv`: continuous trend-model coefficients.
- `factor_effects_accuracy_levels.csv`: categorical-level model coefficients.
- `factor_ablation_accuracy.csv`: legacy descriptive factor contribution,
  measured as delta R2.
- `factor_summary.csv`: context-aware bootstrap factor summaries.
- `model_summary.csv`: HC3 robust fixed-effects coefficients and confidence
  intervals.
- `model_metadata.json`: model fit information and mixed-effects diagnostic.
- `data_quality.md` and `report.md`: concise analysis reports.

The original correlation, trend/levels, and delta R2 ablation analysis remains
the primary descriptive method. Bootstrap confidence intervals, HC3 robust
fixed effects, and the mixed-effects diagnostic are supplementary robustness
checks.

### Figures

`results_all/quant/plots/` contains PNG figures only, including:

- `quant_accuracy_overview.png`
- checkpoint, dataset, and model heatmaps
- nbits/group-size and nbits/checkpoint interaction lines
- per-checkpoint Top-1 accuracy lines with 90% FP32 references
- factor-contribution and factor-relationship plots
- bootstrap nbits/group-size effect plots
- HC3 fixed-effects forest plot

## Reusable CLI

Every stage has defaults that connect to the next stage, but can be redirected
with full paths:

| Stage | Primary input | Primary output | Supporting output directory |
|---|---|---|---|
| Download | W&B | `--output-file` raw CSV | `--results-dir` |
| Summarize | `--input-file` raw CSV | `--output-file` summary CSV | `--results-dir` |
| Correlate | `--input-file` summary CSV | `--output-file` model CSV | `--results-dir` |
| Plot | `--input-file` prepared cells CSV | `--output-file` overview PNG | `--results-dir` |

`--serials` accepts explicit integer lists only. It does not expand ranges. If
you select a subset, include its matching FP32 baseline yourself.

Example: run only the 4X0X group-size code 0 cells plus their baseline.

```powershell
$serials = 4000,4100,4200,4300,4400,4500,4600,4700,4999

python summarize_quant.py `
  --input-file D:\runs\raw.csv `
  --output-file D:\analysis\summary\quant_summary.csv `
  --results-dir D:\analysis\summary `
  --serials $serials

python corr_quant.py `
  --input-file D:\analysis\summary\quant_summary.csv `
  --output-file D:\analysis\corr\model_summary.csv `
  --results-dir D:\analysis\corr `
  --serials $serials

python plot_quant_acc_bit_group.py `
  --input-file D:\analysis\corr\prepared_accuracy_cells.csv `
  --model-file D:\analysis\corr\model_summary.csv `
  --factor-summary-file D:\analysis\corr\factor_summary.csv `
  --ablation-file D:\analysis\corr\factor_ablation_accuracy.csv `
  --corr-dir D:\analysis\corr `
  --output-file D:\analysis\plots\overview.png `
  --results-dir D:\analysis\plots
```

Use `--help` on any script for the full argument list:

```powershell
python summarize_quant.py --help
```

## Verification

After changing the code, rerun the downstream stages against the checked-in raw
CSV and confirm:

1. `completeness.csv` identifies every missing context or baseline.
2. `configuration_quality.csv` reports unexpected non-fixed setup values.
3. A workbook exists for every dataset.
4. `results_all/quant/corr/` contains no PNG files.
5. `results_all/quant/plots/` contains all expected PNG figures.
