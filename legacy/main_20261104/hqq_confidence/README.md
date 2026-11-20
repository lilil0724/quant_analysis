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
| margin_matching.py、analyze_margin_matching.py | baseline normalized-margin overlap matching；100004 NPZ 與 server 20004 FP/W3-G128 輸入。 |
| run_margin_matching_nano4.sh | serial-specific preflight、快照匯入與分析的 Slurm launcher。 |

功能測試已封存至 outputs/layout_adoption_20261021/development_checks/；
流程內的 provenance／sample pairing 驗證保留。2026-10-22 已從 worktree archive
恢復並整理 Serial 100004 margin matching，新增 server 20004 入口；原封存保持原狀。
這個新入口採用 `data/<input_id>/`、`result/<run_id>/` 與 `reports/<run_id>/REPORT.md`。
其他 confidence 流程仍沿用下述舊路徑，尚未完成整體遷移。


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

## Margin matching: Soy + cotton versus Other, server serial 20004

這個入口回答：將兩組 baseline normalized-margin 分布標準化後，FP→Q accuracy
retention 的組間差距如何改變？它不執行 bias correction、competitor fitting、
推論或 W&B 下載。只有 `analyze_margin_matching` 使用下列新目錄契約。

預設分組（依本次需求，不是物種或 FGIR 分類）：

| Group label | Datasets |
|---|---|
| `soy_cotton`（5） | soyageing, soygene, soyglobal, soylocal, cotton |
| `other`（11） | aircraft, cars, cub, dogs, flowers, food, inat17, moe, nabirds, pets, vegfru |

原始組內每個 dataset 等權。使用 FP 的 `(top1_logit-top2_logit)/std(class_logits,ddof=0)`，
主分析 bin width 0.05，敏感度 0.025／0.10。以兩組 histogram 的逐 bin minimum
正規化為共同目標 q；相同 sample weights 同時套用到 FP 與 Q correctness。
Ratio 是 **加權 Q accuracy / 加權 FP accuracy**，不是 dataset ratios 的算術平均。
匹配後 dataset mass 可能改變，所有樣本（包含 baseline 錯誤）都納入。

### Input contract and validation

Server source 預期為既有 paired W3/G128 exporter 的格式：

```text
/work/kyle0724/data/w3_g128_analysis/20004/<dataset>/<model>/<ft|fz|cal>/
  manifest.json          # complete; top-level serial/dataset/model/checkpoint_mode
  samples.json           # ordered sample_ids and targets
  fp/logits.npy          # float32 [N,C], unquantized matched precision
  fp/predictions.csv
  w3_g128/logits.npy      # weight-only W3/G128, activations unquantized
  w3_g128/predictions.csv
```

完整預設矩陣為 **16 datasets × 3 models × 3 modes = 144 pairs**，各 model/mode
分開產生 9 組比較。Models 為 `vit_b16`、`swin_base_patch4_window7_224_in22k`、
`beitv2_base_patch16_224`。任何指定 cell 缺少都失敗，不靜默縮減矩陣。
只具備 aggregate summary／feature arrays 不足以執行本分析。
新的 unified exporter schema 2（每個 condition 獨立 identity/manifest）不會被當成
舊 paired schema；需提供上述 serial 20004 原始配對 exports。

沿用 `server_bias_competitors.check_pair/preflight`，檢查 source status、serial、
dataset/model/mode、checkpoint/test-CSV/config/code/HQQ hashes、seed、quantization、
precision、sample/target digest、CSV/logit winner/margin/true-margin 一致性。
同 model/mode 內 export/precision/quantization protocol 必須相同；每個 dataset 的
checkpoint、pooling、preprocessing 保留在 coverage，允許依訓練設定不同。
Tie winner 沿用已驗證的 stored winner，margin 仍由 logits 重算。
這些是 exporter 宣告的 checkpoint bytes identity，分析不重新讀取權重／圖片。

`import-server` 只複製 manifest、samples、兩組 logits/predictions 到本分析
`data/<input_id>/`，不複製 layer features。原始 manifest 保留原樣；
`source_manifest.json` 記錄原路徑、匯入相對路徑、時間、檔案大小、SHA-256、
selection 與 status，作為 relocation receipt。匯入前後核對 hash，來源與快照
分開保存；**需另有容納所選 logits 副本的磁碟空間**，inat17 可能很大。
新輸入 ID 拒絕覆寫，失敗匯入標記 failed，改用新的 ID。

### Run on nano4

先同步目前分支程式到 server 的 quant_analysis checkout；可用這次交付的
`dist/margin_matching_server_20261022_tgda.zip`，在新的目錄解壓縮後使用。
ZIP 包含分析及其 validator 相依程式、README、launcher、requirements，不含資料或歷史結果。
需要 NumPy、pandas、matplotlib、SciPy、scikit-learn、threadpoolctl，
不需要 torch／TGDA／HQQ／W&B 的分析功能。Server launcher 比照 TGDA 的
`scripts/run_exp_nano5.sh`／`scripts/run_feature_analysis.sh`，預設直接執行
`$HOME/.conda/envs/tgda/bin/python`，不依賴 shell activation 或 `conda run`
選擇 interpreter。本機開發與上方其他分析命令仍使用 `opencode_env`。
先確認 tgda 有所需分析依賴：

```bash
cd /work/kyle0724/project/quant_analysis
ml purge
ml load miniconda3
TGDA_PYTHON="${TGDA_PYTHON:-$HOME/.conda/envs/${TGDA_CONDA_ENV:-tgda}/bin/python}"
"$TGDA_PYTHON" -m pip install -r hqq_confidence/requirements_margin_matching.txt

# Print commands only: no modules, import, or computation.
bash hqq_confidence/run_margin_matching_nano4.sh \
  --serial 20004 --input-id server20004_v1 --run-id margin20004_v1 --dry-run

# Read-only check of the complete selection. Submit large hash scans to Slurm.
sbatch hqq_confidence/run_margin_matching_nano4.sh --serial 20004 --stage preflight

# Import a NEW input snapshot and calculate all 9 model/mode comparisons.
sbatch hqq_confidence/run_margin_matching_nano4.sh \
  --serial 20004 --input-id server20004_v1 --run-id margin20004_v1
```

Launcher header 固定 `8gpus`／`MST114495`，1 node、1 GPU、8 CPUs、64G、24h；
計算在 CPU 上進行，log 為 `margin_matching_<job>.out/.err`。資源額度是提交設定，
本機測試不代表 20004 exports 已完成、server 可用或資源已實測足夠。

若先 pilot，用兩組都有的 dataset selection；pilot 與完整矩陣使用不同 input/run IDs：

```bash
sbatch hqq_confidence/run_margin_matching_nano4.sh --serial 20004 \
  --datasets cotton,soylocal,cub,inat17 --models vit_b16 --modes ft \
  --input-id server20004_pilot_v1 --run-id margin20004_pilot_v1

# Reuse an existing completed snapshot. Dataset/model/mode selection comes from it.
sbatch hqq_confidence/run_margin_matching_nano4.sh --serial 20004 --stage run \
  --input-id server20004_v1 --run-id margin20004_v2
```

使用 `--source-root` 改 upstream（亦接受 `--input-root`），`--workspace` 改 owning
分析資料夾。例如磁碟空間位於 `/work/kyle0724/data/margin_matching` 時，將該路徑
傳給 `--workspace`；其下仍維持 data/result/reports 契約。外部 workspace 的 reports
要另行同步保存，才會出現在本機 checkout。`ANALYSIS_PROJECT_ROOT` 指向程式 checkout，
`TGDA_CONDA_ENV` 預設 `tgda`，用來組合 `$HOME/.conda/envs/<env>/bin/python`。
若環境放在其他路徑，設定 `TGDA_PYTHON`；`ANALYSIS_PYTHON` 有更高覆寫優先序。
真正執行前會載入 miniconda3、核對 interpreter 可執行性與所需套件，並在 job log
列出實際 `sys.executable`／套件版本。`run.json` 與報告的重現命令使用這次真正的
`sys.executable`，不再寫死 `opencode_env`。例如：

```bash
TGDA_PYTHON=/home/kyle0724/.conda/envs/tgda/bin/python \
sbatch hqq_confidence/run_margin_matching_nano4.sh --serial 20004 \
  --input-id server20004_v1 --run-id margin20004_v1
```

也可在適合的 compute node 直接執行：

```bash
TGDA_PYTHON="${TGDA_PYTHON:-$HOME/.conda/envs/${TGDA_CONDA_ENV:-tgda}/bin/python}"
"$TGDA_PYTHON" -m hqq_confidence.analyze_margin_matching import-server \
  --source-root /work/kyle0724/data/w3_g128_analysis --serial 20004 --input-id server20004_v1
"$TGDA_PYTHON" -m hqq_confidence.analyze_margin_matching run \
  --input-id server20004_v1 --serial 20004 --run-id margin20004_direct_v1
```

### Outputs and interpretation

| Path under `result/<run_id>/` | Meaning |
|---|---|
| `tables/group_metrics.csv` | Per-model/mode/condition/group/bin-width raw and matched FP/Q accuracy, ratio, drop and status |
| `tables/gaps.csv` | Other minus focus ratio gaps, raw/matched changes; zero original gap reduction fraction = N/A |
| `tables/balance.csv` | Original, target, matched bin proportions and balance error |
| `tables/diagnostics.csv` | ESS, common support, overlap coefficient, weighted KS and weight amplification |
| `tables/dataset_weights.csv`, `dataset_metrics.csv` | Dataset composition change and primary-width within-dataset metrics |
| `tables/coverage.csv` | Dataset/model/mode/checkpoint/seed/precision and source identity |
| `prepared/<model>/<mode>/sample_weights.npz` | IDs, targets, baseline correctness, margin, base and three matching weights; pickle-free |
| `figures/` | Actual matching distributions and accuracy ratios per setting |
| `provenance/source_manifest.json`, `provenance/source/` | Input receipt and code snapshot including uncommitted source |
| `run.json` | Atomic status, commands/timezone/environment, input/output/report hashes, validation and error |

唯一報告在 `reports/<run_id>/REPORT.md`，引用圖片與 result/figures 副本 hash 相同。
`data/`、`result/` ignored，reports 可追蹤。同 run/report ID 拒絕覆寫；修正後建立新 run。
所有 accuracy/drop/damage/rescue/ratio 欄位是 fraction；乘以 100 才是百分比或 pp。
此入口的 `damage_rate/rescue_rate` 沿用歷史 matching 欄名，表示**無條件加權 mass**，
不同於上方 legacy fragility 的 conditional rates；驗證 `Q = FP - damage + rescue`。

缺少 common margin-bin support 的 matched estimates 保留 N/A 與
`no_common_support`；weighted FP accuracy 為零則 ratio 為 N/A。
這些未定義估計不等同 I/O／配對失敗，完整 workflow 可完成但在 run/report 明列數量。
匹配只統一 bin proportions，不能統一 baseline correctness、類別數、checkpoint 訓練、
擾動或 rescue/damage；dataset 組成與 ESS 會改變。結果是共同支撐區域的描述性比較，
不做因果歸因或顯著性宣稱；不可把 cotton 當作 soybean 或推論跨 model/mode 的獨立重複。

### Reproduce the archived serial 100004 analysis

原封存的 selected_runs.csv 含歷史絕對 npz_path，保持原樣。
明確給定 `--source-root` 後，以 `artifacts/<run_id>/<artifact_version>/predictions.npz`
定位並核對每個 SHA-256；不以 basename 尋找檔案，也不依賴舊 prepared cache。
重新驗證完整 13 條件、logit-derived fields、sample IDs/targets、checkpoint full path/hash、
seed、classifier/evaluation config 與 provenance；量化正確性從原始 NPZ 取得。

```powershell
conda run -n opencode_env python -m hqq_confidence.analyze_margin_matching import-confidence `
  --source-root outputs/worktree_archive/20261021-154622/serial-100004-logits-analysis/files/hqq_confidence/data/serial_100004 `
  --serial 100004 --input-id serial100004_archive_20261022
conda run -n opencode_env python -m hqq_confidence.analyze_margin_matching run `
  --input-id serial100004_archive_20261022 --serial 100004 `
  --run-id 20261022_margin_s100004_verified --grouping legacy-soy
```

`legacy-soy` 明確保留歷史 soy4/other10（該快照沒有 cotton/inat17），只用於歷史重現。
新的 server 20004 預設為本次要求的 soy+cotton / other-including-inat17。

2026-10-22 本機驗證：已匯入 182 份原始 NPZ（14 datasets × 13 conditions），
用新版入口重算 227,830 筆樣本；三種 bin width 的全部 sample IDs/weights 與封存一致。
144 列 group metrics 的 FP/Q accuracy、ratio、absolute drop、damage/rescue mass、ESS
七個欄位與原 CSV 最大差異皆為 0（驗收容差 rtol/atol 各 1e-12）。
唯一報告見 [100004 歷史重現驗證](reports/20261022_margin_s100004_verified/REPORT.md)。
完整 output/report hashes、唯一 Markdown 與引用圖片 hash 均已重新核對。
暫存開發驗證包含 93 項 confidence 回歸測試，以及最後修正後 23 項 matching 相關檢查，
另通過 Bash syntax 與 launcher dry-run；測試／合成資料留在 repo 外暫存位置並於交付清理。
這些本機證據不代表 serial 20004 在 nano4 已執行或全矩陣可用。
上述真實資料已在本機存在，照抄 input/run ID 會拒絕覆寫；重跑必須換新 run ID，
已有完整 snapshot 時直接使用 `run`，不要重新 `import-confidence`。
