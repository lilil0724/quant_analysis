# 分析歷史紀錄 — 2026-10-21

此文件保存已完成工作的來源、結果與回復資訊。日常操作規範以根目錄
[AGENTS.md](../AGENTS.md) 為準；歷史路徑與當時的驗證結果不代表新一次執行已完成。

## 分析目錄規範採用

- 使用者於 2026-10-21 採用 [分析目錄規範](analysis_conventions_proposal.md)。
- [第一批實作紀錄](analysis_layout_implementation_20261021.md)：W6A4 已遷移；
  legacy、dataset、granularity、confidence、feature 的完整結果目錄遷移仍待執行。
- W6A4 正式入口：[模組說明](../w6a4_analysis/README.md)。本批完成的 run 為
  [20261021_163522_084281_w6a4_s20007](../w6a4_analysis/reports/20261021_163522_084281_w6a4_s20007/REPORT.md)。
- 第一批驗證紀錄：[verification.json](../outputs/layout_adoption_20261021/verification.json)。
  10 項外部暫存功能檢查通過；144 配對、6 個輸入及 15 個產物雜湊核對通過。
- 功能測試原始碼已在核對後移出正式 codebase，封存在
  `outputs/layout_adoption_20261021/development_checks/`；真實資料驗證程式保留。
- 原有 2,301 個輸入／輸出與 74 個正式 Python 檔案在該批驗證時保持不變。

## Worktree 清理與文件整併

- 4 個額外 worktree 已在完整封存、核對全部 1,541 個檔案後移除；兩個主 checkout
  與分支保留。封存位於 `outputs/worktree_archive/20261021-154622/`。
- 本次將原 `agent.md` 的操作規則合併到根目錄 `AGENTS.md`，歷史部分保存於本文件。
  原兩份文件的整併前副本與 SHA-256 位於
  `outputs/agent_docs_consolidation_20261021/backup_manifest.json`。
- Git 忽略的備份／資料／完整結果不會隨 branch 傳送，仍需另外保存。

## Corrected baseline promotion (2026-10-21)

At the original promotion, the local result entrypoints were `legacy_hqq/results/main/`,
`dataset_robustness/results/main/`, `dataset_robustness/results/pca_phase1/`,
`granularity_robustness/results/main/`, and the historical `outputs/serial_20007/results/main/`.
Read [the promotion record](baseline_promotion_20261021.md) for provenance,
verification and rollback. The archived recovery snapshots are historical inputs,
not dependencies of the formal outputs.

Promotion verification passed on 2026-10-21: 144 baselines, 4,608 full-grid pairs,
4,320 matched legacy cells, 144 usable W6A4 ratios and no missing/unmatched cells.
All formal numeric results match the completed recovery; 3,840 unaffected legacy
cells match the archived original. The 35 regression tests passed after local
integration. Receipt: `outputs/baseline_recovery/promotion_20261021/verification.json`.
Backup: `outputs/backups/baseline-promotion-20261021/manifest.json`, with Git ref
`codex/backup-before-baseline-promotion-20261021`. Worktrees were subsequently
archived as described above; no push.

For baseline-based reruns, use the verified cohort in
`data/baseline_recovery/selected_20261021/`. It contains 144 unquantized AMP
baselines plus 4,608 HQQ conditions. The legacy/dataset/granularity main cohort
remains 15 datasets (4,320 conditions); the 20007 comparison includes all 16.

The five affected datasets are dogs, flowers, moe, pets and vegfru. Their
BEiTv2 FT/FZ readout is cls and CAL is pool; replace baseline and weight-only
HQQ runs together. Match complete checkpoint path, seed and evaluation config.
The only accepted path alias is `../../data/` to `/work/kyle0724/data/`.
`fp32_top1` is a legacy field name; the stored baseline uses AMP, and paths
alone do not establish checkpoint byte identity.

Rebuild the cohort with `python -m legacy_hqq.recover_baselines` in opencode_env.
Use its curated 15-dataset CSV for legacy acquisition; raw W&B repeats must pass
the same cohort/evaluation audit before entering the default pipeline. Preserve
the snapshot/manifest together when changing inputs. Generated data and results
are ignored by Git and require separate transfer or backup.
