# Quantization analysis workspace

2026-10-21 起採用 [分析目錄與資料規範](docs/analysis_conventions_proposal.md)，
操作規則見 [AGENTS.md](AGENTS.md)，實作進度見
[第一批遷移計畫](docs/analysis_layout_implementation_20261021.md)。
W6A4 已採用自己的 `data/<input_id>/`、`result/<run_id>/` 與唯一
`reports/<run_id>/REPORT.md`。其餘模組保留下表的現行路徑，待逐模組驗證遷移。

目前預設結果已採用修正版 baseline。入口、來源與回復方式見
[正式替換紀錄](docs/baseline_promotion_20261021.md)；操作契約見 [AGENTS.md](AGENTS.md)。
原 worktree 重跑紀錄保留於 [2026-10-21 報告](docs/baseline_recovery_20007_20261021.md)。

每個分析各自管理程式、資料與結果。使用 `opencode_env`，從此目錄執行 Python module。
Margin matching 的 server launcher 依 TGDA 設定使用 `tgda` 環境的 Python，
路徑覆寫與提交命令見 [server 說明](hqq_confidence/README.md#run-on-nano4)。

| 分析 | 入口與說明 | 資料 | 結果 |
|---|---|---|---|
| Weight／activation accuracy（新規範；含歷史 W6A4） | [用途、W&B 匯入與篩選命令](activation_quant/README.md) | `activation_quant/data/<input_id>/`；含 `baseline_20261021`、`wandb_s20008_20261104` | [Serial 20008 通用流程驗證](activation_quant/reports/20261104_s20008_general_validated/REPORT.md)；[W3／4／5 dataset×A 圖](activation_quant/reports/20261104_s20008_dataset_activation/REPORT.md)；[歷史 W6A4 報告](activation_quant/reports/20261021_163522_084281_w6a4_s20007/REPORT.md)；完整產物在 `activation_quant/result/` |
| Legacy HQQ accuracy ratio | [legacy_hqq/README.md](legacy_hqq/README.md) | `legacy_hqq/data/` | `legacy_hqq/results/` |
| Dataset ranking／PCA | [dataset_ranking/README.md](dataset_ranking/README.md) | `dataset_ranking/data/` | `dataset_ranking/results/` |
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

W3／4／5 的 [W+A minus weight-only delta 圖](activation_quant/reports/20261104_s20008_weight_delta/REPORT.md)
使用 Top-1 百分點差。兩批資料的 scope/head 設定不同，因此為描述性比較；方法與
`--weight-delta` 命令見 [模組說明](activation_quant/README.md)。
同配對的 [acc_ratio delta 圖](activation_quant/reports/20261104_s20008_weight_delta_acc_ratio/REPORT.md)
使用 `--weight-delta --delta-metric acc_ratio`，先各自除以共同 AMP baseline 再相減。
另有 [acc_ratio(W+A) / acc_ratio(W-only) 圖](activation_quant/reports/20261104_s20008_weight_acc_ratio_quotient/REPORT.md)，
使用 `--delta-metric acc_ratio_quotient`，每個配對先相除再取 context 平均，中性值為 1。

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

Dataset ranking and exploratory loss-profile PCA now use `dataset_ranking`.
Use the [module commands](dataset_ranking/README.md#reproduce) with new output
directories; existing `results/main/` and its provenance remain historical.
The local checkout currently has no saved PCA run. The full result/reports
layout migration remains pending.

Current paired FP/W3 analysis retains its shared-FP-basis final-feature PCA; the config-driven pipeline retains its per-position PCA. The separate historical FP/W3/W4 final-CLS study is preserved in the source archive above.

2026-11-04: feature code, local inputs and historical outputs are consolidated in
`dataset_features/`. The former `five_dataset_features/` tree has been removed
after per-file size/SHA-256 verification; existing destination artifacts and
historical manifests are unchanged. See the [relocation receipt and path mapping](dataset_features/relocations/20261104_module_consolidation/README.md).
Historical paths in saved reports/manifests resolve by replacing that module
prefix. This consolidation retains `analysis/` and `results/`; the full
`result/<run_id>/reports` contract and unavailable server raw-array verification
remain pending. Back up the ignored local data/results separately from Git.

See [docs/repository_layout_plan.md](docs/repository_layout_plan.md) for the target layout and remaining steps. Archived results from the old `results_all/` layout are kept in each analysis's `results/archived_*` directories. They are snapshots, while `results/main/` is the newly generated run.
