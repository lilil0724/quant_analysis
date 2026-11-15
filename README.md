# Quantization analysis workspace

2026-10-21 起採用 [分析目錄與資料規範](docs/analysis_conventions_proposal.md)，
操作規則見 [AGENTS.md](AGENTS.md)，實作進度見
[第一批遷移計畫](docs/analysis_layout_implementation_20261021.md)。
Activation 分析已集中於 `activation_quantization/`，包含 serial 20007 W6A4
與 serial 20008 W/A grid，共用 `data/<input_id>/`、`result/<run_id>/` 與唯一
`reports/<run_id>/REPORT.md`。其餘模組保留下表的現行路徑，待逐模組驗證遷移。

目前預設結果已採用修正版 baseline。入口、來源與回復方式見
[正式替換紀錄](docs/baseline_promotion_20261021.md)；操作契約見 [AGENTS.md](AGENTS.md)。
原 worktree 重跑紀錄保留於 [2026-10-21 報告](docs/baseline_recovery_20007_20261021.md)。

每個分析各自管理程式、資料與結果。使用 `opencode_env`，從此目錄執行 Python module。
Margin matching 的 server launcher 依 TGDA 設定使用 `tgda` 環境的 Python，
路徑覆寫與提交命令見 [server 說明](hqq_confidence/README.md#run-on-nano4)。

| 分析 | 入口與說明 | 資料 | 結果 |
|---|---|---|---|
| Activation quantization：20007 W6A4／20008 W/A grid | [統一入口、設定與配對](activation_quantization/README.md) | `activation_quantization/data/<input_id>/`（20007 原始輸入需從備份恢復） | [Baseline acc_ratio 曲線／dataset heatmaps](activation_quantization/reports/20261102_162715_226338_wa_s20008/REPORT.md)；[accuracy／A6 圖集](activation_quantization/reports/20261102_142515_734861_wa_s20008/REPORT.md)；[其他報告](activation_quantization/README.md#目前結果與歷史遷移) |
| Legacy HQQ accuracy ratio | [legacy_hqq/README.md](legacy_hqq/README.md) | `legacy_hqq/data/` | `legacy_hqq/results/` |
| Dataset robustness | [dataset_robustness/README.md](dataset_robustness/README.md) | `dataset_robustness/data/` | `dataset_robustness/results/` |
| Granularity robustness | [granularity_robustness/README.md](granularity_robustness/README.md) | `granularity_robustness/data/` | `granularity_robustness/results/` |
| Serial 20007 corrected baseline（歷史比對） | [promotion record](docs/baseline_promotion_20261021.md) | `data/baseline_recovery/` | `outputs/serial_20007/results/main/`（保留，不覆寫） |
| Matched-sample HQQ confidence | [hqq_confidence/README.md](hqq_confidence/README.md) | `hqq_confidence/data/` | `hqq_confidence/results/` |
| Soy+cotton / Other normalized-margin matching (20004 server; 100004 reproduction) | [方法、資料契約與 Slurm 命令](hqq_confidence/README.md#margin-matching-soy--cotton-versus-other-server-serial-20004) | `hqq_confidence/data/<input_id>/` | `hqq_confidence/result/<run_id>/`；唯一報告在 `hqq_confidence/reports/<run_id>/REPORT.md` |
| 16-dataset paired FP/W3 features | [用途與 codebase](dataset_features/README.md#paired-fpw3nano4-主分析) | external paired raw exports; compact copy in `dataset_features/data/w3_g128_results/` | `<serial>/analysis/`; local historical summaries in `dataset_features/analysis/` |
| Paired FP/W3 t-SNE | [sampling and nano4 commands](dataset_features/README.md#paired-fpw3-t-sne-on-nano4) | external paired raw exports | `<serial>/analysis/tsne_classes/` (10% classes, 5-20, all their samples) |
| Centered FP backbone class distances (20004) | [dataset_features/CENTERED_CLASS_DISTANCE.md](dataset_features/CENTERED_CLASS_DISTANCE.md) | external paired FP/W3 raw exports | `dataset_features/results/20004/centered_fp_distance_cal/` (documented CAL submission) |
| 16-dataset all-layer features (20004) | [dataset_features/README.md](dataset_features/README.md#16-dataset-all-layer-workflow-20004) | `data/20004/` | `dataset_features/results/20004/` |
| Legacy 21-condition / five-dataset feature code（封存） | [清單、SHA-256 與還原說明](archive/five_dataset_features_legacy_20261101/README.md) | historical inputs remain in place | historical outputs remain in place; archived code is not a relocated executable package |

## Common flow

Nano4 單一提交入口：[`dataset_features/run_all_nano4.sh`](dataset_features/run_all_nano4.sh)，
可選擇或一次執行既有 W3/G128 paired、centered、bias、margin、t-SNE；原分析與 plot
不變。乾跑、preflight、pilot、完整提交及輸出位置見
[統一 launcher 說明](dataset_features/README.md#一次提交既有分析run_all_nano4sh)。

Paired W3/G128 feature、t-SNE、centered-distance、bias/competitor 與 margin-matching
分析現在同時接受 legacy paired 或新版 `feature_analysis` schema 2，沿用原本統計與
plot。來源格式、必要檔案、明確跨 serial 選取及 Nano4 命令見
[dual-input 說明](dataset_features/README.md#舊-paired-與新版-schema-2-共用原分析)；
[margin/bias 輸入](hqq_confidence/README.md#兩種輸入格式2026-11-15)。

```text
acquire (read/download) → prepare (validate/pair) → analyze (compute/report) → plot
```

The legacy and HQQ confidence workflows now have stage entry points:

```powershell
conda run -n opencode_env python -m legacy_hqq.run --stage all --run-name main
conda run -n opencode_env python -m hqq_confidence.run --stage all --run-name main
```

Both reuse existing raw data by default. A new W&B download requires missing input or an explicit `--refresh`. Dataset and granularity robustness currently retain their existing combined analysis command; their data and results have been separated, and the stage split is the next migration step.

Feature analysis now exposes the 16-dataset config-driven, paired FP/W3, paired t-SNE and centered-FP workflows described in [the module README](dataset_features/README.md). The independent schema-2 reader handles one FP/quant pair and does not yet provide a 16-dataset orchestrator. The old 21-condition analyzer, geometry/distance/PCA tools and five-dataset slide scripts are [archived with original hashes](archive/five_dataset_features_legacy_20261101/README.md); their former module commands require source restoration. Existing data, reports and result manifests were preserved. This source cleanup does not complete the feature module's pending data/result layout migration.

The exploratory dataset loss-profile PCA runs with `conda run -n opencode_env python -m dataset_robustness.pca_profiles`. Its report and figures are in `dataset_robustness/results/pca_phase1/`.

Current paired FP/W3 analysis retains its shared-FP-basis final-feature PCA; the config-driven pipeline retains its per-position PCA. The separate historical FP/W3/W4 final-CLS study is preserved in the source archive above.

See [docs/repository_layout_plan.md](docs/repository_layout_plan.md) for the target layout and remaining steps. Archived results from the old `results_all/` layout are kept in each analysis's `results/archived_*` directories. They are snapshots, while `results/main/` is the newly generated run.
