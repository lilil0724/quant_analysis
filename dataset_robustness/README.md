# Dataset robustness stability audit

## 2026-10-21 規範採用狀態與 codebase

新規範見根目錄 [AGENTS.md](../AGENTS.md)。此模組的 result/reports 遷移尚未執行；目前仍
使用本文的 data/upstream/ 與 results/ 命令，保留歷史 snapshot/receipts。

| 程式 | 職責 |
|---|---|
| analyze.py | 讀固定 upstream summary，計算 context 內／間的 ranking stability、coverage、排除稽核及圖表。 |
| pca_profiles.py | 由驗證後的 loss profiles 做 NumPy SVD/PCA 與 sensitivity。 |
| verify.py | 核對真實輸入、統計表、配對／排名公式及完整產物。 |
| verify_pca.py | 核對 PCA 輸入／輸出 hash、特徵值、正交性及重建。 |

功能測試已封存至 outputs/layout_adoption_20261021/development_checks/；
上述真實資料 verifier 屬於正式分析流程並保留。新的功能測試使用 repo 外暫存目錄。


Descriptive comparisons of the legacy HQQ experiments. The main question is
whether dataset accuracy-retention rankings persist across quantization settings
and model/checkpoint contexts. This is separate from `hqq_confidence` and does not
modify the four legacy scripts.

## Contract

- Primary: 2, 3, 4 bits, all observed group sizes, compared separately.
- Sensitivities: 3/4 bits only and the complete observed bit grid.
- Primary metric: upstream matched-seed `accuracy_ratio`; supplement with
  `fp32_top1 - hqq_top1` in percentage points and baseline accuracy.
- No clipping of ratio > 1 or negative accuracy drops.
- One checkpoint path per dataset/model/checkpoint kind. Ambiguous or changing
  checkpoint identities fail rather than being silently averaged.
- Non-matched, nonfinite or invalid cells remain in an exclusion audit.
- Within-context comparisons use a fixed dataset cohort complete over the
  scope's bit × group grid. Cross-context comparisons use the common dataset
  cohort at each quantization condition. Coverage records exclusions explicitly.
- Rank 1 is highest retention; ties use average rank. Constant rankings and
  comparisons with fewer than three datasets have undefined Spearman rho.
- Pair reversal means both directions occur across settings. A `1e-12`
  numerical equality tolerance is not a practical equivalence threshold.
- Every condition has equal descriptive weight within a scope. There is no
  universal dataset ranking, p-value or seed-level confidence interval.
- Dataset-specific checkpoints confound dataset and training outcome. A
  one-seed experiment cannot establish training-seed reproducibility.

## Reproduce

Use `opencode_env` from the repository root. The fixed upstream summary and
legacy validation snapshot are in `dataset_robustness/data/upstream/legacy/`.
Downloading is not required.

```powershell
conda run -n opencode_env python -m dataset_robustness.analyze
conda run -n opencode_env python -m dataset_robustness.verify
```

The analyzer accepts `--input-file` and `--results-dir`. The explicit
`--primary-bits` option validates the agreed `2 3 4` contract rather than silently
changing report interpretation. It reads `configuration_quality.csv` beside the
summary when available. Source summary and analyzer SHA-256 hashes are recorded
in `run_config.json`.

## PCA of dataset loss profiles

After the main analysis and its verifier pass, run the exploratory PCA stage:

```powershell
conda run -n opencode_env python -m dataset_robustness.pca_profiles
conda run -n opencode_env python -m dataset_robustness.verify_pca
```

Start with `dataset_robustness/results/pca_phase1/report.md`. Each fit uses one
dataset per row and the complete bit × group grid per column within a fixed
model/checkpoint-kind context. The primary outcome is `1 - accuracy_ratio` for
2/3/4-bit conditions; separate fits use absolute accuracy drop and 3/4-bit
conditions. The script uses NumPy SVD, so no scikit-learn installation is
required. Each condition is centered across datasets, with no variance scaling
or missing-value imputation. `coverage.csv` records cohort inclusion.

The output contains PCA scores, loadings, explained variance, a full-profile
distance sensitivity table, plots, source/code/output SHA-256 hashes, and a
report. Different PCA fits do not share coordinate axes. Quantization settings
are repeated observations of dataset-specific checkpoints, not independent
training seeds; PCA is descriptive and does not identify a dataset-only effect.

Start with `dataset_robustness/results/main/report.md`. The report links the main
figures; `plots_manifest.csv` maps all plots to source tables. CSVs preserve the
full per-condition results and all pairwise stability comparisons. Reported
rank summaries are descriptive of equally weighted, observed conditions; their
variation is not an uncertainty interval.
