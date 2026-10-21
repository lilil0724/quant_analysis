# AGENTS.md — quant_analysis

本檔是本 repository 的操作規範入口。開始修改或執行分析前，先讀對應模組的
README.md；新增或遷移分析時，再讀 [完整目錄規範](docs/analysis_conventions_proposal.md)。

## 工作順序

1. 檢查 git status 與模組 README，保留未提交修改、輸入及既有結果。
2. 在 owning module 完成修改，使用既有本機資料驗證來源、配對、coverage、
   公式與保存產物；完成條件以實際命令與輸出為證據。
3. 命令、資料路徑或行為改變時，同步更新模組 README 與根 README 索引。

所有 Python 命令使用 `conda run -n opencode_env python ...`。

## 分析資料與產物的七項規則

1. **輸入**：放在該分析的 `data/<input_id>/`。每次匯入記錄來源身分、取得時間、
   檔案大小與 SHA-256；保留原始 manifest。新增輸入版本使用新 ID。
2. **輸出**：本次整理後資料、統計表、完整圖集、log 與 provenance 全部放在
   `result/<run_id>/`，使用單數 result。run.json 記錄時間／時區、命令、參數、
   輸入與輸出 hash、程式／環境、驗證及狀態；失敗 run 明確標示 failed。
3. **模組說明**：每個分析根目錄保留一份 README.md，說明用途、codebase 分工、
   資料契約、指標／baseline、執行命令、依賴、輸出與驗證方式。
4. **唯一報告**：每次 run 只產生一份 `reports/<run_id>/REPORT.md`，包含分析時間、
   輸入來源、方法、結果、限制與重現命令。`reports/<run_id>/figures/` 只放實際
   引用的圖片，並核對與 `result/<run_id>/figures/` 中對應圖片的 SHA-256 相同。
5. **正式 codebase**：保留分析流程與真實資料驗證。功能測試、mock／synthetic
   fixtures、probe 和一次性 debug code 在 repository 外的暫存位置執行並清理。
   既有獨有程式先核對封存，再移出正式目錄。
6. **每次獨立**：重跑使用新的 run ID；已有 input/run/report 目錄時拒絕覆寫。
   完成的 run 與歷史 manifest 固定保存，路徑遷移使用 relocation receipt 記錄。
7. **Git 與備份**：所有 `data/` 和 `result/` 內容 gitignore；分析程式、README、
   唯一報告及其引用圖片保持可追蹤。忽略資料另行備份，branch 不是這些檔案的備份。

重用已驗證輸入；僅在缺少輸入或使用者明確要求 refresh 時下載。W&B 下載需要
WANDB_API_KEY 或 wandb login。產物由 owning stage 重算，修改公式或資料後產生新 run。

## 選擇分析與目前遷移狀態

| 分析 | 先讀 | 狀態與 baseline |
|---|---|---|
| W6A4 serial 20007 | [w6a4_analysis/README.md](w6a4_analysis/README.md) | 已遷移；matched unquantized AMP。 |
| Legacy ratios | [legacy_hqq/README.md](legacy_hqq/README.md) | 待遷移；修正版 baseline 為 unquantized AMP。 |
| Dataset ranking／PCA | [dataset_robustness/README.md](dataset_robustness/README.md) | 待遷移；使用固定 upstream legacy snapshot。 |
| FGIR／UFGIR | [granularity_robustness/README.md](granularity_robustness/README.md) | 待遷移；保留原分組 mapping 與 matched summary。 |
| Confidence | [hqq_confidence/README.md](hqq_confidence/README.md) | 待遷移；matched-sample unquantized AMP。 |
| Feature geometry | [five_dataset_features/README.md](five_dataset_features/README.md) | 待遷移；部分完整 inference arrays 仍在 nano4。 |

新分析與已遷移模組遵守上述目錄契約。待遷移模組先沿用其 README 的現行
`results/`、`analysis/`、`data/processed/` 或 server input-root，完成實際驗證後再遷移；
分階段修改時明列尚未完成項目。

Baseline 配對核對完整 checkpoint 路徑、dataset、model、mode、seed、classifier、
precision 與已記錄評估設定。`fp32_top1` 是 legacy 欄位名稱，不代表純 FP32。
缺少可比資料保留 N/A；checkpoint 路徑／config 相同不能證明權重 bytes 相同。

需要重新規劃目錄遷移時，讀 [實作與進度紀錄](docs/analysis_layout_implementation_20261021.md)。
需要 baseline provenance、舊結果或回復資訊時，讀 [分析歷史紀錄](docs/analysis_history_20261021.md)。
既有 worktree 封存保持原狀；本次目錄整理沿用主 checkout。

## Slurm launcher 必要 header

新增或修改供 sbatch 提交的 Bash script，須在任何可執行命令前包含以下 header。
partition 固定 8gpus、account 固定 MST114495；其餘欄位填入該工作的具體值。
output/error 檔名要能區分 job，例如 analysis_%j.out 與 analysis_%j.err。

```bash
#!/usr/bin/env bash
#SBATCH --job-name=
#SBATCH --partition=8gpus
#SBATCH --account=MST114495
#SBATCH --nodes=
#SBATCH --gpus-per-node=
#SBATCH --cpus-per-task=
#SBATCH --mem=
#SBATCH --time=
#SBATCH --output=
#SBATCH --error=
```

交付前核對所有 directives 與填值。保留 server inference 來源與 provenance；
本機 launcher／合成檢查不代表 nano4 已完成、資源足夠或缺少的 raw arrays 已通過驗證。
