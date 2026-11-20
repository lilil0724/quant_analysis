# Legacy HQQ accuracy-ratio analysis

## 2026-10-21 規範採用狀態與 codebase

新規範見根目錄 [AGENTS.md](../AGENTS.md)。此模組仍在待遷移批次：現行輸入為 data/，
prepared outputs 仍在 data/processed/，完整產物仍在 results/；目前命令維持
可用，尚未宣稱符合新 result/run_id 與唯一報告規則。現行 baseline 是
修正版 unquantized AMP；下方歷史 FP32 名稱不得解讀為純 FP32 評估。

| 程式 | 職責 |
|---|---|
| run.py | acquire → prepare → analyze → plot 的順序與本機輸入重用。 |
| recover_baselines.py | 按完整身分與修正版 cohort 政策選取 baseline/HQQ，產生配對稽核。 |
| download_save_wandb_data.py | W&B CSV 取得，僅在明確刷新或缺少輸入時執行。 |
| summarize_quant.py | checkpoint/seed 配對、summary、完整性／設定稽核與 XLSX。 |
| corr_quant.py | matched accuracy ratio 的 correlation、factor model 與 ablation。 |
| plot_quant_acc_bit_group.py | 從前階段統計表產生圖表。 |

功能測試已封存至 outputs/layout_adoption_20261021/development_checks/，正式
codebase 僅保留分析流程。暫存測試在 repo 外執行；真實資料的配對、完整性與
產物驗證仍是必須的步驟。歷史 results/ 與 receipts 保持原狀以供遷移比較。


The current main run uses the corrected BEiTv2 cohort. See
[the promotion record](../docs/baseline_promotion_20261021.md) for source,
downstream entrypoints and rollback. Refresh baseline-based runs through
`python -m legacy_hqq.recover_baselines`, then import its verified curated
15-dataset CSV; a raw W&B download alone does not select the corrected cohort.

This standalone workflow downloads finished TiT HQQ runs, matches each HQQ row to the compatible unquantized AMP baseline with the same dataset, normalized model, complete checkpoint path and seed, computes `accuracy_ratio = hqq_top1 / fp32_top1`, and draws the corresponding figures. Serials are provenance and optional filters; they do not define quantization conditions.

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

Check `results/main/summary/completeness.csv` for unmatched conditions, `configuration_quality.csv` for non-fixed settings, one `quant_<dataset>.xlsx` per dataset, and the overview, heatmaps, line plots and factor contribution figure under `results/main/plots/`. Cells with any unmatched seed remain in the summary for audit and are excluded from correlation and plots.
