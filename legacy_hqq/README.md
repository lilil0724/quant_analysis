# Legacy HQQ accuracy-ratio analysis

The current main run uses the corrected BEiTv2 cohort. See
[the promotion record](../docs/baseline_promotion_20261021.md) for source,
downstream entrypoints and rollback. Refresh baseline-based runs through
`python -m legacy_hqq.recover_baselines`, then import its verified curated
15-dataset CSV; a raw W&B download alone does not select the corrected cohort.

This standalone workflow downloads finished TiT HQQ runs, matches each HQQ row to the FP32 baseline with the same dataset, normalized model, complete checkpoint path and seed, computes `accuracy_ratio = hqq_top1 / fp32_top1`, and draws the corresponding figures. Serials are provenance and optional filters; they do not define quantization conditions.

## Run

From the repository root, using `opencode_env`:

```powershell
conda run -n opencode_env python -m legacy_hqq.run --stage all --run-name main
```

Run one stage with `--stage acquire`, `prepare`, `analyze` or `plot`. `acquire` reuses `legacy_hqq/data/raw/backbones_quant.csv` when present. `--source-file PATH` imports a local raw CSV; `--refresh` explicitly replaces existing raw data. For a serial subset, pass an explicit `--serials` list that includes the required FP32 baseline runs. `--run-name` separates processed data and results, for example `serial_10000_10014`.

The stage scripts remain callable individually with `--input-file`, `--output-file`, `--results-dir` and their other documented options:

```powershell
conda run -n opencode_env python -m legacy_hqq.summarize_quant --help
conda run -n opencode_env python -m legacy_hqq.corr_quant --help
```

## Paths and stage products

| Stage | Input | Output |
|---|---|---|
| acquire | W&B or `--source-file` | `data/raw/backbones_quant.csv`, provenance config |
| prepare | raw CSV | `data/processed/<run-name>/quant_summary.csv`; `results/<run-name>/summary/` audit CSVs and one XLSX per dataset |
| analyze | matched summary | `data/processed/<run-name>/prepared_accuracy_cells.csv`; `results/<run-name>/corr/` statistics and reports |
| plot | prepared cells and ablation table | `results/<run-name>/plots/` PNG figures |

Paths in the table are relative to `legacy_hqq/`. The checked-in raw CSV was moved without changing its SHA-256. Previous `results_all/quant*` outputs are preserved under `results/archived_*` and are not the current run.

## Verification

```powershell
conda run -n opencode_env python -m unittest discover -s legacy_hqq/tests -v
```

Check `results/main/summary/completeness.csv` for unmatched conditions, `configuration_quality.csv` for non-fixed settings, one `quant_<dataset>.xlsx` per dataset, and the overview, heatmaps, line plots and factor contribution figure under `results/main/plots/`. Cells with any unmatched seed remain in the summary for audit and are excluded from correlation and plots.
