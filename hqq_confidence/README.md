# HQQ Confidence Fragility Analysis

## 2026-10-21 規範採用狀態與 codebase

新規範見根目錄 [AGENTS.md](../AGENTS.md)。此模組的完整 data/result/reports 遷移仍待執行；
現行命令與 data/processed/、results/、analysis/ 路徑保持相容。Baseline
維持 matched-sample unquantized AMP，不與其他 checkpoint/seed 混配。

| 程式 | 職責 |
|---|---|
| run.py | 協調 acquire → prepare → analyze → plot；重用既有 artifacts。 |
| download_artifacts.py | 取得有明確 run/artifact 版本的 prediction artifacts。 |
| prepare_pairs.py、common.py | schema、logits、sample ID、target、checkpoint/precision 驗證與 matched pairs。 |
| analyze_fragility.py、plot_fragility.py | damage/rescue、confidence bins、bootstrap 與圖表。 |
| analyze_overconfidence.py、analyze_dataset_difficulty.py、analyze_sample_margin_mechanism.py | 分別回答 calibration、dataset difficulty 與 sample margin 問題。 |
| analyze_bias_competitors.py、bias_competitor_core.py、server_bias_competitors.py | cross-fit bias／competitor 機制分析與 server adapter。 |

功能測試已封存至 outputs/layout_adoption_20261021/development_checks/；
流程內的 provenance／sample pairing 驗證保留。Serial 100004 的 margin-matching
未提交程式仍在 worktree archive，未在本批偷偷恢復或套用舊結果。


This directory contains a reproducible four-stage workflow for matched
unquantized/HQQ prediction artifacts. It is independent of the repository's
legacy accuracy-ratio pipeline.

The baseline condition is `fp_unquantized`: unquantized weights evaluated with
the same AMP FP16 setting as HQQ. It is not an FP32 baseline.

## Data flow

Run from the repository root with the `opencode_env` Conda environment:

```powershell
conda run -n opencode_env python -m hqq_confidence.run --stage all --run-name main
```

The stage runner reuses `hqq_confidence/data/selected_runs.csv` when present and
runs the download, prepare, fragility statistics and plot stages in order. Use
`--stage` to run one stage; `--refresh` explicitly downloads again. The
additional overconfidence, dataset-difficulty and sample-margin analyses retain
their separate commands:

```powershell
conda run -n opencode_env python -m hqq_confidence.analyze_overconfidence
conda run -n opencode_env python -m hqq_confidence.analyze_dataset_difficulty
conda run -n opencode_env python -m hqq_confidence.analyze_sample_margin_mechanism
```

The defaults connect these stages:

| Stage | Default input | Default output |
|---|---|---|
| download | finished `nycu_pcs/TiT` W&B runs | `hqq_confidence/data/selected_runs.csv` and `data/artifacts/<run-id>/<version>/` |
| prepare | selected-runs manifest | `hqq_confidence/data/processed/main/` |
| analyze | prepared summaries and matched pairs | `hqq_confidence/results/stats/` |
| plot | prepared and statistical tables | `hqq_confidence/results/plots/` |

Each command accepts absolute path overrides; use `--help` for its complete
interface. Each stage writes `run_config.json` in its output directory.
Downloading requires `WANDB_API_KEY` or `wandb login`. For an ambiguous rerun,
create a CSV containing one `run_id` per selected condition and pass it with
`--run-ids-file`; the downloader never resolves ambiguity by taking `:latest`.

## Sweep and validation contract

A sweep is identified by project, dataset, model, checkpoint SHA-256, serial,
and seed. It contains exactly these 13 runs:

```text
fp_unquantized
hqq_n1_g8    hqq_n1_g128
hqq_n1.58_g8 hqq_n1.58_g128
hqq_n2_g8    hqq_n2_g128
hqq_n3_g8    hqq_n3_g128
hqq_n4_g8    hqq_n4_g128
hqq_n8_g8    hqq_n8_g128
```

Only finished runs with `job_type=hqq-confidence-eval` are downloaded. The
prepare stage reports incomplete and provenance-incompatible sweeps without
including them in analysis. It checks sample-ID sets and targets, N/C, logits
and derived arrays, recomputed accuracy, matched AMP/dtype/Git provenance, and
quantized layer counts. Legacy artifacts may omit `metadata_amp_fp16`, and
older HQQ summaries may use `top1` instead of the canonical `hqq_top1`;
capability checks are used instead of
assuming that schema version alone determines available fields.

Unknown AMP, compute-dtype, Git, or quantized-layer provenance is excluded by
default. `--allow-unknown-provenance` permits only those unknown states; it does
not permit mismatches, debugging runs, invalid layer counts, or incomplete
sweeps.

## Outputs

`data/selected_runs.csv` records the W&B run/config/summary, chosen artifact
name, version and digest, and the cached NPZ path. Prepared outputs are:

- `completeness.csv` and `configuration_quality.csv`: one row per candidate sweep.
- `condition_summary.csv`: one row per valid sweep condition, including raw
  uncalibrated NLL, Brier score, ECE, accuracy, confidence, margin, and entropy.
- `transition_summary.csv`: one row per HQQ/baseline pair with accuracy ratio,
  accuracy drop, damage/rescue, agreement, and mean paired deltas.
- `pairing_quality.csv`: W&B-versus-artifact accuracy audits.
- `reliability_bins.csv`: equal-count raw-confidence reliability bins.
- `pairs/<sweep-id>/<condition>.npz`: compressed, reduced matched-sample arrays;
  full logits are not duplicated.

Statistical outputs include `confidence_bins_by_sweep.csv`, aggregated
`confidence_bins.csv`, `condition_statistics.csv`,
`condition_statistics_by_context.csv`, `bootstrap_metadata.json`,
`data_quality.md`, and `report.md`. Figures and their input-table mapping are
listed in `plots/plots_manifest.csv`.

## Metric definitions

All accuracies and accuracy drops are percentage points. Rates and ratios are
unitless fractions.

```text
damage       = baseline correct and HQQ wrong
rescue       = baseline wrong and HQQ correct
damage_rate  = damage_count / baseline_correct_count
rescue_rate  = rescue_count / baseline_wrong_count
accuracy_drop_points = baseline_accuracy - hqq_top1
accuracy_ratio       = hqq_top1 / baseline_accuracy
delta metric         = HQQ metric - baseline metric
```

The primary analysis uses only baseline-correct samples and relates damage to
the within-sweep percentile of the baseline top-1 minus top-2 logit margin.
Confidence intervals use a fixed-seed checkpoint-cluster bootstrap; serial
reruns and individual images are not treated as independent replicates. No
temperature scaling is fitted because only the test split is available. The
results are exploratory associations, not causal or confirmatory claims.

## Dataset–model overconfidence hypothesis

`analyze_overconfidence` combines the paper's prediction-swap decomposition with
the matched-sample margin analysis. It treats the unquantized AMP-FP16 condition
as the baseline, computes 15-bin overconfidence ECE (OECE), and tests whether
higher OECE predicts lower net accuracy damage per prediction swap:

```text
swap_rate = changed_top1_count / n_samples
net_damage_per_swap = (correct_to_wrong - wrong_to_correct) / changed_top1_count
```

The primary statistic is the median condition-wise Spearman correlation across
the HQQ grid. Its one-sided p-value permutes dataset labels as complete blocks,
so the 12 HQQ rows for a dataset are not treated as independent. Nested
leave-one-dataset-out models add baseline accuracy, log class count, margin, and
finally OECE. Sensitivity analyses cover alternate predictors and outcomes,
extreme HQQ conditions, and a post-hoc soy-dataset exclusion.

For serial 100004 without inat17, run against the prepared artifacts instead of
downloading logits again:

```powershell
python -m hqq_confidence.analyze_overconfidence `
  --prepared-dir D:\analysis\results_no_inat17\prepared `
  --results-dir D:\analysis\results_no_inat17\overconfidence_hypothesis `
  --serial 100004 --exclude-datasets inat17 --expected-datasets 14
```

The command writes dataset calibration and dataset-by-condition robustness
tables, mechanism bins, condition correlations, nested-model and sensitivity
results, three figures, and `report.md`. Raw test probabilities are never
temperature-scaled, and conclusions remain associative rather than causal.

## Exploratory dataset quantization difficulty

`analyze_dataset_difficulty` screens whether dataset-specific fine-tuned
ViT-B/16 checkpoints have a stable relative HQQ difficulty across 3-bit and
4-bit at group sizes 8 and 128. Difficulty is the mean within-condition
percentile rank of `1 - accuracy_ratio`; Kendall's W checks whether the four
rankings are concordant before treating this as one score.

The prespecified predictor is the 10th percentile of the baseline top-1/top-2
logit gap normalized by the within-sample standard deviation across class
logits. The analysis tests Q10 against difficulty, compares leave-one-dataset-out
models with baseline accuracy and log class count, and checks the mechanism
`Q10 -> swap rate -> relative accuracy loss`. Two-bit and eight-bit results are
boundary sensitivities; 1-bit and 1.58-bit do not enter inference.

```powershell
python -m hqq_confidence.analyze_dataset_difficulty `
  --prepared-dir D:\analysis\results_no_inat17\prepared `
  --manifest-file D:\analysis\data\selected_runs.csv `
  --results-dir D:\analysis\results_no_inat17\dataset_difficulty_3_4bit `
  --serial 100004 --exclude-datasets inat17 --expected-datasets 14
```

Only baseline artifacts are opened for full-logit feature extraction, one at a
time in bounded row chunks. Quantized conditions reuse compact prepared pair
vectors; no logits are downloaded again.

## Sample-level normalized-margin mechanism

\`analyze_sample_margin_mechanism\` tests the within-dataset mechanism behind
the dataset-level Q10 result. For each dataset and primary 3/4-bit condition,
it computes:

\`\`\`text
AUROC(-baseline_sample_normalized_margin, prediction_swap)
\`\`\`

The primary population contains all matched samples; baseline-correct samples
are a sensitivity population. Four condition AUROCs are averaged within each
dataset, and the 14 dataset scores receive equal weight. The prespecified gate
requires a median dataset AUROC of at least 0.60, a one-sided exact sign-test p
below 0.05 against chance, and a condition-specific median above 0.50 in all
four conditions. A single-class outcome in any primary condition makes the
gate \`INCOMPLETE\`, not \`FAIL\`.

Normalized-margin deciles are assigned within dataset without splitting ties.
The main curve first averages conditions within dataset and then averages
datasets equally; its interval is a fixed-seed dataset bootstrap.

\`\`\`powershell
python -m hqq_confidence.analyze_sample_margin_mechanism \`
  --prepared-dir D:\analysis\results_no_inat17\prepared \`
  --manifest-file D:\analysis\data\selected_runs.csv \`
  --results-dir D:\analysis\results_no_inat17\sample_margin_mechanism \`
  --serial 100004 --exclude-datasets inat17 --expected-datasets 14
\`\`\`

The stage reads each full baseline-logit artifact once, aligns normalized
margins to the existing prepared pairs by sample ID, and does not modify or
redownload prepared artifacts. Outputs include condition and dataset AUROCs,
baseline-correct sensitivity results, dataset/condition/aggregate decile
tables, four figures, a decision JSON, and a Markdown report.

## Verification

### Cross-fit class bias and competitor spectrum

For nano4's multi-dataset/model/ft-fz-cal paired-W3 exports, use the portable
`server_bias_competitors` entrypoint and `run_bias_competitors_nano4.sh` launcher.
See [server commands and output definitions](SERVER_BIAS_COMPETITORS.md).
This server stage accepts a selected serial, preserves the complete requested
matrix and analyzes each model/mode separately. It reads full logit NPY files
with memory mapping and does not require the Windows research snapshot.

The independent `analyze_bias_competitors` stage reads the already audited
serial-100004 raw-logit artifacts. It performs five-fold class bias correction
with five permuted-class controls and tests additional nearby-competitor counts
against a nonlinear top-two-margin model using leave-one-dataset-out prediction.
Soybean-family holdouts are exploratory. A protocol JSON is saved before analysis;
all 70 artifact hashes must match the supplied prior audit configuration.

```powershell
python -B -m hqq_confidence.analyze_bias_competitors --reference-config hqq_confidence/analysis/dataset_relationship_20261010/raw_logit_run_config.json --output-dir hqq_confidence/analysis/bias_competitors_20261011
```

Use a new output directory for each run. The stage retains source identity audits,
folds and bias vectors, original/corrected predictions, heldout metrics, fitted
model coefficients, a figure and a Markdown report. Existing test logits are used
for cross-fitting; corrected accuracy is a retrospective mechanism diagnostic,
not an independently calibrated benchmark. The baseline uses AMP FP16 and the
checkpoint family is ViT-B/16 FT; do not infer BEiTv2 results from this analysis.

Development checks use a temporary directory outside the production tree.
Input provenance, matched-sample alignment and saved-result validation remain
part of the production analysis flow.

Run the legacy scripts separately against `legacy_hqq/data/raw/backbones_quant.csv` when
changing shared repository dependencies. This workflow does not import or
modify any of the four legacy stages.
