# 分析目錄重整提案

> 歷史規劃：2026-10-21 起，新的分析與遷移採用
> [新版規範](analysis_conventions_proposal.md)。下方 `results/`、固定 `main`
> run、多份階段報告、保留 tests/ 等目標已被新版的 `result/<run_id>/`、
> 唯一 REPORT.md 與外部暫存測試規則取代。未遷移模組仍依各 README 的現行命令執行。

狀態：**分階段實作中**。本文件原以 2026-09-28 的工作樹提出目標架構。第一階段已將 `other_test/` 的四個分析目錄移回頂層，將 legacy 程式、資料與舊結果歸入 `legacy_hqq/`，並把 dataset/granularity 的輸入及結果歸入各自目錄；舊 snapshot 已保留。下方的目標樹仍包含尚未完成的階段拆分與 five-dataset feature 資料搬移。

## 目標

每個研究問題有自己的目錄，並在該目錄內管理程式、輸入資料、中間資料、結果、測試及說明。所有分析都提供同一種使用順序：

```text
acquire（讀取／下載） → prepare（檢查、配對、整理）
                   → analyze（計算、統計、報告） → plot（圖表）
```

`acquire` 必須同時支援使用已有資料；只有來源需要更新時才連接 W&B 或匯入外部檔案。沒有圖表的分析可以讓 `plot` 明確回報「無圖表」，不要假造空白圖片。封裝 ZIP 等額外工作可有自己的 `package` 階段，不能取代四個共同階段。

## 目標目錄

```text
quant_analysis/
├── AGENTS.md
├── README.md                         # 分析索引及共同執行約定
├── CONTEXT.md                        # 跨分析的研究術語
├── requirements.txt
├── docs/                              # 跨分析文件、ADR、遷移說明
├── legacy_hqq/                        # 現有四階段 accuracy-ratio 分析
│   ├── README.md
│   ├── run.py                          # 統一階段入口
│   ├── download_save_wandb_data.py    # 第一輪保留既有檔名
│   ├── summarize_quant.py
│   ├── corr_quant.py
│   ├── plot_quant_acc_bit_group.py
│   ├── data/
│   │   ├── raw/                     # W&B CSV 與來源 manifest
│   │   └── processed/<run-name>/    # baseline-matched summary、畫圖輸入
│   ├── results/<run-name>/
│   │   ├── audit/                   # completeness、configuration quality
│   │   ├── tables/                  # 統計 CSV、各 dataset XLSX
│   │   ├── plots/                   # PNG
│   │   ├── reports/                 # Markdown
│   │   └── run_config.json
│   └── tests/
├── dataset_robustness/
│   ├── README.md
│   ├── run.py
│   ├── analyze.py                    # 可逐步拆出 prepare / plot
│   ├── data/{upstream,processed}/
│   ├── results/<run-name>/{audit,tables,plots,reports}/
│   └── tests/
├── granularity_robustness/
│   ├── README.md
│   ├── run.py
│   ├── analyze.py
│   ├── data/{source,upstream,processed}/  # 含 dataset_granularity_mapping.csv
│   ├── results/<run-name>/{audit,tables,plots,reports}/
│   └── tests/
├── hqq_confidence/
│   ├── README.md
│   ├── run.py
│   ├── download_artifacts.py         # 其餘現有程式先保留檔名
│   ├── data/{raw,processed}/          # selected_runs、NPZ cache、matched pairs
│   ├── results/<run-name>/{audit,tables,plots,reports}/
│   └── tests/
└── five_dataset_features/
    ├── README.md
    ├── run.py                        # 可沿用 run_local.ps1 作 PowerShell 入口
    ├── analyze.py                    # 其餘現有程式先保留檔名
    ├── data/{raw,processed}/          # raw 為從 nano5 匯入的大型檔案
    ├── results/<run-name>/{audit,tables,plots,reports,packages}/
    └── tests/
```

上圖的 `{...}` 與 `<run-name>` 表示目錄樣板，不是要建立帶大括號的實際目錄。各分析只建立自己需要的子目錄。`run-name` 預設為 `main`；敏感度分析、serial 子集及其他變體使用明確名稱，例如 `exclude_low11`、`serial_100003`，避免覆蓋主分析。路徑由各分析目錄定位，不依賴啟動 PowerShell 時的工作目錄。

`data/raw/` 放外部取得的原始輸入，`data/upstream/` 放其他分析匯出的固定版本輸入，`data/processed/` 放可重算的中間表與配對陣列；`results/` 只放分析輸出。每次匯入應記錄來源路徑、檔案 SHA-256、取得時間與使用的上游 run name。大型 feature arrays 與 W&B artifacts 可以留在本機各分析的 `data/raw/`，由 `.gitignore` 排除；不應為了目錄整齊而提交數 GB 原始資料。

## 五個分析的資料契約

| 分析 | acquire 來源 | prepare 的主要輸出 | analyze / plot 的主要輸出 |
|---|---|---|---|
| `legacy_hqq` | W&B 或現有 `data/backbones_quant.csv` | 精確 checkpoint/seed 配對後的 `quant_summary.csv`；畫圖輸入 | accuracy-ratio 統計、XLSX、audit、heatmaps 等 |
| `dataset_robustness` | 從指定 `legacy_hqq` run 匯入 `quant_summary.csv` 及 configuration quality | 排除稽核與完整 dataset cohort 的分析 cells | 條件間排名穩定性表、報告、圖 |
| `granularity_robustness` | 匯入指定的 matched summary 與版本固定的 dataset 分組 CSV | FGIR/Ultra-FGIR 對照 strata 及 coverage | contrast/sensitivity 表、報告、圖 |
| `hqq_confidence` | W&B prediction artifacts 或既有 cache | 完整 sweep 檢查與 matched-sample pairs | fragility、overconfidence、difficulty、margin 統計及圖 |
| `five_dataset_features` | nano5 匯出的 inference files 或既有本機副本 | checksum、shape、sample-ID 驗證後的特徵與條件表 | CKA/PCA/geometry 圖、報告；`package` 另產 ZIP |

`dataset_robustness` 與 `granularity_robustness` 不重新實作 FP32 baseline 配對。它們只讀 `legacy_hqq` 已完成配對的匯出，並在各自 `data/upstream/` 保存使用的版本與雜湊。`hqq_confidence` 的 baseline 是 matched AMP unquantized，不可與 legacy 的 FP32 baseline 混用。

## 統一執行介面

從 repo 根目錄使用 `opencode_env`。Legacy 與 confidence 的共同階段入口已可執行：

```powershell
conda run -n opencode_env python -m legacy_hqq.run --stage all --run-name main
conda run -n opencode_env python -m hqq_confidence.run --stage all --run-name main
```

Dataset 與 granularity 目前仍用各自的 `analyze` 命令一次計算並畫圖；預計拆成：

```powershell
conda run -n opencode_env python -m dataset_robustness.run --stage prepare --run-name main
conda run -n opencode_env python -m dataset_robustness.run --stage analyze --run-name main
conda run -n opencode_env python -m dataset_robustness.run --stage plot --run-name main
```

第二段是**尚未實作的目標介面**。共同參數為 `--stage acquire|prepare|analyze|plot|all` 與 `--run-name`。各分析在自己的 `README.md` 記錄來源與專屬參數，例如 W&B serial、上游 summary、granularity mapping 或 feature dataset。`all` 依序執行階段；已有原始資料時沿用並驗證，不默默重新下載。每一階段產生自己的 `run_config.json` 或在 run manifest 中記錄命令、輸入雜湊、程式版本、時間及輸出路徑；缺少前一階段的有效輸入時應停止並指出缺少的檔案。

共同介面由各分析自己的 `run.py` 實作；先不新增跨分析的通用執行框架。數據驗證、baseline 定義和統計方法仍留在各自分析，避免看似通用的程式把不同研究契約混在一起。

## 現況到目標的搬移對照

| 目前位置 | 目標位置或處理方式 |
|---|---|
| repo 根目錄四支 legacy `.py` 與 `tests/` | `legacy_hqq/` 的程式與測試 |
| `data/backbones_quant*.csv` | `legacy_hqq/data/raw/`；先保留原檔與校驗值 |
| `results_all/quant*` | 由 legacy 階段重產到 `legacy_hqq/results/<run-name>/`；舊 snapshot 在驗證完成前保留 |
| `other_test/dataset_robustness/` | `dataset_robustness/`；現有 root `results_all/dataset_robustness*` 對應不同 run name |
| `other_test/granularity_robustness/` | `granularity_robustness/`；mapping CSV 放 `data/source/` |
| `other_test/hqq_confidence/` | `hqq_confidence/`；既有 `data/` 與 `results/` 的內容先盤點雜湊再遷移 |
| `other_test/five_dataset_features/` | `five_dataset_features/`；目前直接位於 dataset 子目錄的資料歸 `data/raw/`，`analysis/` 產物歸 `data/processed/` 或 `results/` |
| `outputs/<thread-id>/` | 先辨識來源與用途；只有可追溯到某分析的檔案才歸入該分析，否則保留為獨立歷史輸出 |

## 實作順序與驗證

1. 先記錄目前 `git status`、重要輸入的 SHA-256 與現有可執行命令；不要清除 `other_test/` 或覆蓋未提交檔案。
2. 搬回四個現有 Python module 的程式與測試，修正模組 import、`__file__` 相對路徑、PowerShell wrapper 與 README。暫不移動大型資料。
3. 建立各自的 `data/`、`results/` 與階段入口；先讓同一份原始資料在新路徑產生與舊路徑等價的核心 CSV 與圖。再把現有資料和 snapshot 對應到各 run name。
4. 將 `dataset_robustness`、`granularity_robustness` 和 `five_dataset_features` 內混合計算與畫圖的工作拆到 `analyze` / `plot` 階段；每段可獨立重跑，不會修改前一段輸入。
5. 更新根目錄 `README.md`、`AGENTS.md`、`.gitignore` 和各分析 README；搜尋並清除舊絕對路徑與 `results_all/` 依賴。
6. 以 `opencode_env` 跑現有測試，以及每個分析的 `prepare → analyze → plot`。legacy 要核對 baseline matching、audit、每 dataset XLSX 與圖；confidence 要核對 sweep/pairing；feature 要核對 snapshot/checksum。只有驗證成功後才處理舊目錄的去留。

這次規劃不會改變任何分析定義。目錄遷移的驗收重點是相同輸入得到相同可比較結果，並能只重跑某個分析的單一階段。

## 第一階段實作紀錄（2026-09-28）

- `legacy_hqq/` 已接管四支腳本、原始 CSV、測試與舊結果。`python -m legacy_hqq.run --stage all --run-name main` 已從本地 CSV 完成四段，產生 4,320 個配對條件、0 個 unmatched baseline；原始 CSV 搬移前後 SHA-256 相同。
- `dataset_robustness/`、`granularity_robustness/` 已在各自 `data/` 放固定上游輸入，並在各自 `results/main/` 重產；兩個 verifier 均通過。現有程式仍一次執行計算與畫圖。
- `hqq_confidence/` 的新四階段入口已從既有 cache 完成一次執行；主要 prepared pairs 已移至 `data/processed/main/`。其額外三個研究問題仍使用各自原有命令。
- `five_dataset_features/` 的程式已回到頂層 module；`analysis/` 的既有 snapshot 暫時保留原位。這個 snapshot 依賴分析程式與來源檔雜湊，目前 workspace 缺少完整五個 dataset 的 raw inference arrays，須取得原始檔後才能搬移並重新驗證。
- 根目錄舊 `data/`、`results_all/`、`outputs/` 已歸入對應分析；空的 `other_test/` 與 `results_all/` 目錄已移除。既有使用者修改（例如 `CONTEXT.md` 和 feature 分析的新檔）保留。

接續工作：為 dataset、granularity 與 feature 分析建立真正可分開重跑的 `acquire → prepare → analyze → plot`；使各分析能從已追蹤來源重新建立自己的 `data/upstream/`，再完成 feature 大型資料與 snapshot 遷移。完成前應以各 module README 的現行命令為準，不能執行上方標為目標介面的範例。
