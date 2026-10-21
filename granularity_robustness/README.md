# Granularity robustness analysis

## 2026-10-21 規範採用狀態與 codebase

新規範見根目錄 [AGENTS.md](../AGENTS.md)。此模組仍待 result/reports 遷移；目前繼續使用
本文的 data/source/、data/upstream/ 與 results/ 路徑，保留原本 cohort 與數值。

| 程式 | 職責 |
|---|---|
| analyze.py | 用固定分組 mapping 與 matched summary 計算 FGIR/UFGIR contrasts、sensitivities、leave-one-out 及圖表。 |
| plot_retention_condition.py | 從保存的 retention table 畫指定 bits/group size 的圖。 |
| verify.py | 核對來源 hash、分組、strata、統計公式與輸出完整性。 |

功能測試已封存至 outputs/layout_adoption_20261021/development_checks/；
verify.py 為真實結果驗證而保留。新的功能測試使用 repo 外暫存目錄。


This module implements `dataset_robustness/granularity_analysis_proposal.md`
without changing the four legacy pipeline scripts. It compares externally
defined FGIR and Ultra-FGIR dataset groups within matched model, checkpoint and
HQQ group-size strata.

## Contract

- Dataset groups come only from `dataset_granularity_mapping.csv`.
- Primary groups are 8 FGIR and 4 Ultra-FGIR datasets; `food`, `inat17` and
  `moe` are classification-boundary datasets.
- Primary scope is 2/3/4-bit × group sizes 8/32/128/512.
- Each dataset is weighted once per matched stratum. Cells are not independent
  training repeats.
- Primary estimand is the Ultra-FGIR minus FGIR median 4→2-bit degradation.
- A co-reported matched-condition level contrast is the Ultra-FGIR minus FGIR
  median `accuracy_ratio` at each bit-width and group size; negative values mean
  lower Ultra-FGIR retention under the same quantization condition.
- Every pre-specified grouping, baseline, metric, aggregation, bit-scope and
  leave-one-dataset-out sensitivity is retained.
- Strata with fewer than two datasets in either group are marked incomparable.
- Results are descriptive associations: no p-values or seed-level confidence
  intervals are produced.

## Reproduce

Use `opencode_env` from the repository root:

```powershell
conda run -n opencode_env python -m granularity_robustness.analyze
conda run -n opencode_env python -m granularity_robustness.verify
```

The default input is
`granularity_robustness/data/upstream/quant_summary.csv`. The fixed dataset
mapping is `granularity_robustness/data/source/dataset_granularity_mapping.csv`.
Outputs are isolated under `granularity_robustness/results/main/`. Start with `report.md`; use
`dataset_granularity_mapping.csv`, `granularity_coverage.csv` and
`run_config.json` to audit inclusion and provenance.

For a compact heatmap of one quantization condition, run
`conda run -n opencode_env python -m granularity_robustness.plot_retention_condition`
from the repository root. By default it shows 3-bit, group size 128 across the
three models and checkpoint kinds. Use `--bits` and `--group-size` to select
other conditions. The plot and a `.source.json` file recording the source table
and its SHA-256 are written under `results/main/plots/`.
