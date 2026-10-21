# 20007 baseline 修復：2026-10-21 完整重跑

本頁保留原 worktree 重跑時的歷史紀錄。主 checkout 的正式結果與驗證入口以
[正式替換紀錄](baseline_promotion_20261021.md) 為準；以下 dated receipts 的 worktree 路徑屬歷史來源。

本次已完成下載、配對、重新分析與驗證。使用 worktree
`C:/project/quant_analysis/outputs/worktrees/baseline-recovery-20007`，
分支 `codex/baseline-recovery-20007`，起點 commit
`6409f800b755d6f4c9dd870811a163ce85c86486`。本次沒有執行新 GPU 推論、訓練、push 或 merge。

## 來源與配對

- W&B `nycu_pcs/TiT` 的 10000–10015 快照於 2026-10-21 09:30（Asia/Taipei）重新下載，共 3,762 個 finished runs。
- dogs、flowers、moe、pets、vegfru 的 BEiTv2，FT/FZ/CAL 共 15 contexts，採用完整新評估：15 個未量化 baseline + 480 個 HQQ runs。
- 其餘被目前 W&B 移除的 990 個未受影響 runs，來自已驗證 SHA-256 的舊快照；沒有將它們誤報為本次重新下載。
- 合併後為 16 datasets × 3 models × FT/FZ/CAL：144 個 baseline + 4,608 個 HQQ 條件，共 4,752 runs。每 context 均有 32 個量化條件，無缺格、重複或拒絕配對。
- Bits 為 `1, 1.58, 2, 3, 4, 5, 6, 8`，group sizes 為 `8, 32, 128, 512`。
- 受影響 contexts 整組替換 FP 與量化 runs。FT/FZ 記錄的 classifier 為 `cls`，CAL 為 `pool`；不把新 baseline 與舊量化結果混合平均。
- baseline 語義為 **unquantized AMP**；`fp32_top1` 只是保留的 legacy 欄位名稱。完整 checkpoint 路徑、seed、classifier、precision 與已記錄的評估設定均核對；路徑 alias 僅為 `../../data/ → /work/kyle0724/data/`。
- batch size 差異與共同未記錄欄位保留於配對表。checkpoint bytes 缺少共同 hash 時仍無法證明跨時間權重完全相同。

原本 legacy／dataset／granularity 分析維持 **15 datasets，排除 cotton**，以保留比較範圍；20007 heatmap 則涵蓋完整 16 datasets。16-dataset curated 原始資料同時保存。

## 結果入口

修復前後逐格差異與結論比較：[完整比較報告](../outputs/baseline_recovery/comparison_20261021/report.md)。

| 分析 | 本次範圍與驗證 | 報告 |
|---|---|---|
| Serial 20007 W6A4 acc_ratio | 144/144 可配對，無 N/A，逐格公式驗證 | [報告與 heatmap](../outputs/serial_20007/results_baseline_recovery_20261021/report.md) |
| Legacy HQQ | 4,320 matched cells、15 個 XLSX、28 張圖 | [correlation 報告](../legacy_hqq/results/baseline_recovery_20261021/corr/report.md) |
| Dataset robustness | 4,320 cells；5,310 within-context correlations、1,152 cross-context correlations、2,835 pairwise checks、29 張圖通過驗證 | [報告](../dataset_robustness/results/baseline_recovery_20261021/report.md) |
| Loss-profile PCA | 9 contexts、36 fits、5 張圖；來源雜湊、特徵值、正交性與重建誤差通過驗證 | [報告](../dataset_robustness/results/baseline_recovery_20261021_pca/report.md) |
| FGIR / UFGIR | 36 primary strata、8 sensitivities、12 leave-one-out、6 張主圖通過驗證，另重畫 W3/G128 圖 | [報告](../granularity_robustness/results/baseline_recovery_20261021/report.md) |

原本使用不相容 baseline 的 flowers / BEiTv2 / FZ 對照，20007 accuracy 為 98.98%，修正版 baseline 為 99.48%，本次 ratio 為 **0.994974**。舊 baseline 48.09% 導致的約 2.058 ratio 不應用來主張量化改善。

本次重新計算 480 個受影響的 legacy 條件；另外 3,840 個未受影響條件的 FP accuracy、HQQ accuracy、ratio 與原始主分析均在 `1e-12` 容差內相同。新舊評估亦有 batch size 等差異，因此不把所有數值變化單獨歸因於 classifier。

FGIR/UFGIR 的總體 evidence grade 仍為 `context-dependent association`。相同條件的 retention 差異與 4→2-bit degradation slope 是不同問題；詳細數值與敏感度保留於各自報告，不能作 dataset 因果推論。

## 稽核產物

- [整體驗證 receipt](../outputs/baseline_recovery/completed_20261021/verification.json)：全部 raw pairing、未受影響值、來源雜湊及各分析 verifier 的核對結果。
- [cohort manifest](../data/baseline_recovery/selected_20261021/manifest.json)：狀態 `complete`，144/144 contexts、0 missing conditions。
- [逐格配對](../data/baseline_recovery/selected_20261021/pair_audit.csv)、[coverage](../data/baseline_recovery/selected_20261021/context_coverage.csv)、[完整選取 runs](../data/baseline_recovery/selected_20261021/selected_runs.jsonl)。
- [15 個新舊 baseline](../outputs/baseline_recovery/completed_20261021/baseline_before_after.csv)、[480 個新舊條件比較](../outputs/baseline_recovery/completed_20261021/affected_condition_before_after.csv)。
- [最新 W&B grid 下載 manifest](../data/baseline_recovery/snapshots/20261021/full_grid_10000_10015/source_manifest.json)。
- [20007 來源 manifest](../outputs/serial_20007/data/snapshot_20261021/source_manifest.json)。20007 本身仍是同一批 144 個 W6A4 runs，本次更新其 baseline 對照。

## 重現現有快照

以下命令從本 worktree 根目錄執行，均使用 `opencode_env`。原始資料與產物保留於本機；`data/`、`outputs/` 及各分析 results 為 ignored 產物，轉移或封存 worktree 時需一起保存。

```powershell
conda run --no-capture-output -n opencode_env python -m legacy_hqq.recover_baselines --current-jsonl data/baseline_recovery/snapshots/20261021/full_grid_10000_10015/runs.jsonl --output-dir data/baseline_recovery/selected_20261021
conda run --no-capture-output -n opencode_env python -m legacy_hqq.run --stage all --run-name baseline_recovery_20261021 --source-file data/baseline_recovery/selected_20261021/curated_15_datasets.csv --refresh
```

下游來源是本次 legacy summary 的固定副本，兩份 SHA-256 均為
`5fcd239b9bb2c3cf68ce80346bf5ac7f03654dd819c5abae900daf1fe003f71b`。

```powershell
conda run --no-capture-output -n opencode_env python -m dataset_robustness.analyze --input-file dataset_robustness/data/upstream/baseline_recovery_20261021/legacy/quant_summary.csv --results-dir dataset_robustness/results/baseline_recovery_20261021
conda run --no-capture-output -n opencode_env python -m dataset_robustness.pca_profiles --main-results dataset_robustness/results/baseline_recovery_20261021 --output-dir dataset_robustness/results/baseline_recovery_20261021_pca
conda run --no-capture-output -n opencode_env python -m granularity_robustness.analyze --input-file granularity_robustness/data/upstream/baseline_recovery_20261021/quant_summary.csv --results-dir granularity_robustness/results/baseline_recovery_20261021
conda run --no-capture-output -n opencode_env python -m granularity_robustness.plot_retention_condition --table granularity_robustness/results/baseline_recovery_20261021/retention_level_contrasts.csv --bits 3 --group-size 128 --output-dir granularity_robustness/results/baseline_recovery_20261021/plots
conda run --no-capture-output -n opencode_env python outputs/baseline_recovery/replot_20007.py --selected-file data/baseline_recovery/selected_20261021/selected_runs.jsonl --target-file outputs/serial_20007/data/snapshot_20261021/serial_20007_runs.json --output-dir outputs/serial_20007/results_baseline_recovery_20261021
```

本次 30 個測試通過（legacy/recovery 14、dataset/PCA 10、granularity 6）。驗證現有輸出：

```powershell
conda run --no-capture-output -n opencode_env python -m dataset_robustness.verify --results-dir dataset_robustness/results/baseline_recovery_20261021
conda run --no-capture-output -n opencode_env python -m dataset_robustness.verify_pca --output-dir dataset_robustness/results/baseline_recovery_20261021_pca
conda run --no-capture-output -n opencode_env python -m granularity_robustness.verify --results-dir granularity_robustness/results/baseline_recovery_20261021
conda run --no-capture-output -n opencode_env python outputs/baseline_recovery/verify_completed_run.py
```

重新下載時需用新的 snapshot 目錄，以保留本次 receipt 依賴的來源。下載工具位於
`outputs/baseline_recovery/acquire_backbone_baseline.py`（`--data-dir`）及
`outputs/serial_20007/download.py`（`--raw-dir`）；W&B 只讀取 config/summary，不執行遠端推論。
