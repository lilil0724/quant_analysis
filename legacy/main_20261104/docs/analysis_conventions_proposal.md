# 分析程式、資料與結果管理規範

規劃日期：2026-10-21（Asia/Taipei）。適用範圍：quant_analysis 內各個統計／分析資料夾。
狀態：使用者已於 2026-10-21 採用並要求執行。第一批遷移範圍與進度見 [實作計畫](analysis_layout_implementation_20261021.md)；尚未遷移的模組保留現有文件指定路徑，不能視為已符合新規範。

## 1. 目的與決策

每個研究問題由自己的分析資料夾管理程式、輸入與輸出。Git 保存分析流程、用途說明，以及每次分析的精簡報告與報告引用圖片；大型資料與完整產物另行備份。

| 使用者要求 | 規範 |
|---|---|
| 輸入集中 | 所有執行時讀取的分析資料放在該分析的 `data/`。 |
| 輸出集中 | 中間資料、統計表、完整圖表、log 與執行紀錄放在 `result/<run_id>/`。統一使用單數 `result`。 |
| 資料夾說明 | 每個分析資料夾以一份 `README.md` 說明用途、程式分工與執行方式。 |
| 每次一份報告 | 每個 run 只產生一份 Markdown：`reports/<run_id>/REPORT.md`。 |
| 保留分析流程 code | 正式 codebase 保留分析所需的 Python／launcher；功能測試、synthetic fixtures、probe 與臨時 debug code 使用 repo 外的暫存目錄，驗證後清理。 |
| 每次獨立目錄 | 每次重跑取得新 `run_id`；完整輸出和精簡報告使用相同 ID 對應。 |
| 忽略輸入輸出 | `data/`、`result/` 全部忽略；`reports/` 是明列的 Git 例外，僅保存唯一報告與實際引用的圖片。 |

第七點採用的設計假設：允許 Git 保留精簡的結果報告及其引用圖片。其餘生成檔案仍全部忽略。若報告也要忽略，可將唯一 `REPORT.md` 放回 `result/<run_id>/`；兩種方案擇一，不同時維護兩份報告。

## 2. 標準目錄

```text
quant_analysis/
├── README.md                         # 分析索引
├── AGENTS.md                         # 執行規則，指向本規範
├── .gitignore
├── requirements.txt
└── analysis_name/
    ├── README.md                     # 用途、codebase、命令與資料契約
    ├── __init__.py                   # Python module 需要時保留
    ├── run.py                        # 分析入口
    ├── acquire.py                    # 匯入／下載，需要時才拆檔
    ├── prepare.py                    # 整理與輸入驗證，需要時才拆檔
    ├── analyze.py                    # 統計計算
    ├── plot.py                       # 畫圖，需要時才拆檔
    ├── data/                         # 全部 gitignore
    │   └── <input_id>/
    │       ├── source_manifest.json
    │       └── ...                   # 原始資料或固定的上游輸入
    ├── result/                       # 全部 gitignore
    │   └── <run_id>/
    │       ├── run.json
    │       ├── prepared/             # 本次產生的整理後資料
    │       ├── tables/
    │       ├── figures/              # 完整圖表
    │       ├── audit/                # 配對、缺值、排除與驗證紀錄
    │       ├── provenance/           # 程式版本等重現資訊
    │       └── logs/
    └── reports/                      # 可由 Git 追蹤
        └── <run_id>/
            ├── REPORT.md            # 本次唯一的 Markdown 報告
            └── figures/             # 只保留 REPORT.md 實際引用的圖片
```

只建立分析實際需要的檔案與子目錄。小型分析可以只用 `run.py`；不為了符合圖示拆出空模組。`reports/` 內不另放該次 README、summary.md 或第二份結論報告。

## 3. 輸入資料契約

- `data/<input_id>/` 為唯讀的輸入快照；刷新來源時建立新的 input ID，保留已被分析引用的版本。
- 外部資料先匯入／下載到本分析的 `data/`，核對後才執行統計。一般分析與畫圖不隱含下載或刷新。
- 上游分析產出的檔案若成為本分析輸入，匯入本分析 `data/`，記錄上游分析、run ID、來源路徑與 SHA-256。下游不依賴另一分析會被覆寫的預設輸出路徑。
- 大型資料在伺服器也遵守相同的分析目錄與 `data/` 層級；本機缺少原始資料時，明確記錄缺少項目，保留舊快照直到可以驗證遷移。
- 路徑由分析模組位置定位，例如 `Path(__file__).resolve().parent`；執行時與呼叫 PowerShell 的目錄無關。資料參數指向本分析 `data/` 內的選定快照。
- `source_manifest.json` 至少包含：input ID、取得時間與時區、來源 URI／檔案路徑／W&B project 與 run/artifact 身分、每個檔案的相對路徑／大小／SHA-256，以及 schema 和樣本／條件範圍。
- 來源資料的 hash 與 checkpoint 權重檔 hash 分開記錄；未取得權重檔 hash 時保留 unknown，不能以 config 或路徑相同宣稱權重 bytes 相同。

所有本次新產生的 prepared tables、pair arrays 與 cache 都屬於輸出，寫入 `result/<run_id>/`，不回寫 `data/`。

## 4. 每次執行與輸出契約

### Run 身分

Run ID 使用 Asia/Taipei 時間及簡短分析名稱，例如 `20261021_161530_w6a4_s20007`。同名目錄已存在時停止；呼叫者指定新名稱或由入口加上不重複後綴。新的結果不覆寫舊 run。

一個 run 是一組輸入、參數與程式版本所對應的分析執行。`prepare → analyze → plot → report` 共用同一 ID，構成一次輸出；分階段執行不各自產生 Markdown。

### 執行紀錄

`result/<run_id>/run.json` 為本次機器可讀的完整紀錄，至少包含：

- `run_id`、分析名稱、狀態 `running / complete / failed`。
- 開始／結束時間（ISO 8601，含 UTC offset）、時區與耗時。
- 實際執行命令、參數、Python／環境與主要套件版本。
- Git commit、dirty 狀態，以及本次使用程式的 SHA-256。若使用未提交程式，將相關來源快照保存在本次 `provenance/source/`，包含新檔。
- Input ID、完整輸入清單、來源與 hash；模型／dataset／mode／seed／checkpoint／precision／量化設定等相關分析身分。
- 輸出相對路徑及 SHA-256、報告相對路徑、驗證結果與缺值／排除原因。
- 失敗時的錯誤與已完成階段。

完成所有分析與輸出核對後才標為 `complete`。失敗 run 保留資料與同一份 `REPORT.md`，狀態標示失敗，不能列作完整結果。完成後的 run 固定保存；修正數字或參數須建立新 run。

### 唯一報告

`reports/<run_id>/REPORT.md` 由正式分析流程生成，至少包含：

1. 研究問題、run ID、完成／失敗狀態。
2. 分析開始／結束時間、時區與耗時。
3. 輸入資料位置、來源、input ID、SHA-256、樣本／條件數。大量清單可引用 `run.json`，報告保留來源摘要與其 hash。
4. 分析方法、重要參數、baseline 與單位定義。
5. 主要結果、圖表與由數據支持的解讀。
6. 缺值、排除條件、驗證結果與結論限制。
7. 重現命令、Git commit／dirty 狀態、完整輸出位置。

完整圖表先寫入 `result/<run_id>/figures/`；報告流程只將實際引用的 PNG／SVG 複製至 `reports/<run_id>/figures/`，核對兩份 hash 相同。報告使用 `figures/name.png` 相對連結，讓 Git clone 後仍可閱讀。CSV、NPZ、log、完整圖集與未引用圖片留在 ignored 的 `result/`。

報告可列精簡數值表；原始樣本與完整統計表保留於 `result/`。所有新生成 Markdown 報告集中到這一份，不再由各階段另寫 report.md／data_quality.md／summary.md。

## 5. Codebase 與資料夾 README

正式分析程式可包含匯入、整理、統計、畫圖、報告、必要的 schema／配對／雜湊／公式檢查，以及伺服器 launcher。驗證是分析流程的一部分，失敗時停止並留下原因。

功能測試、`test_*.py`、synthetic fixtures、一次性 probe、debug script 和 mock-data generator 不留在正式 codebase。開發時在 repo 外暫存目錄執行必要檢查，將檢查命令與結論留於開發紀錄；分析成功仍須有真實輸入及產物的核對證據。

每個分析根目錄的 `README.md` 至少說明：

- 本分析回答的問題、適用範圍與指標定義。
- 各個正式程式的職責、主要呼叫／資料流與依賴。
- 輸入 schema、來源、取得方式與 baseline 配對規則。
- 從 repo root 執行的命令、環境與必要參數。
- `data/`、`result/`、`reports/` 的位置、命名與輸出 schema。
- 內建資料驗證、完成條件與如何閱讀結果。

多個獨立研究問題若共用同一個大模組，由 README 解釋分工；只有需要各自執行、各自擁有資料與報告的分析才建立自己的子資料夾，並遵守同一套規範。

## 6. Git 與備份

建議新增的共同忽略規則如下；遷移期間保留舊路徑的忽略規則：

```gitignore
# Runtime input snapshots and complete generated output
**/data/
**/result/

# Legacy output locations, retained during migration
**/results/
/outputs/
/.codex_tmp/

# Python runtime cache
__pycache__/
*.py[cod]
```

`reports/<run_id>/REPORT.md` 與引用圖片不放在上述 ignored 目錄，納入 Git。報告以外的輸出不使用 `git add -f` 提交。模組程式與 README 保持可追蹤，避免再次把正式分析程式放進 `outputs/` 等忽略目錄。

已受追蹤的輸入／輸出不會因新增 `.gitignore` 自動消失；遷移時先備份與核對，再使用僅取消追蹤的方式處理，保留本機檔案。這不會清除 Git 歷史內原有的資料。

Git 不負責 `data/` 與 `result/` 的保存；資料備份需包含輸入 manifest、完整 run 目錄與必要程式快照。刪除任何舊目錄前，先確認新副本存在且 SHA-256 一致。

## 7. 目前 repo 的採用順序

1. 將本草案定為正式規範後，更新根 README 與 AGENTS.md 的入口；統一 `result/` 與每次唯一報告的規則。舊 layout 文件註明被取代的規則，保留歷史紀錄。
2. 以 W6A4 作為第一個遷移分析：將正式 `tools/baseline_recovery/replot_20007.py` 的職責收回獨立分析資料夾，輸入歸其 `data/`，輸出與報告依新契約產生。舊 W6A4 worktree 的封存是歷史參考來源，不直接把過時腳本當作正式版。
3. 再遷移 legacy、dataset robustness、granularity robustness、confidence 與 feature 分析。每次只處理一個分析，保留統計公式、baseline 意義及 cohort。
4. 如需恢復 serial-100004 margin matching，從 worktree 封存取回三個未提交檔案，依新規範保留兩個正式分析檔；功能測試移至暫存驗證流程。相關配對驗證缺口應另列修復工作，不混入純路徑遷移。
5. 對照原始輸入與來源 manifest 搬移／複製，產生 relocation receipt；舊 manifest 的歷史路徑不直接竄改。沒有原始資料的 feature 歷史結果先保持封存。
6. 更新分析入口與下游輸入匯入路徑；以新 run 重算並核對核心數值及引用圖片。驗證完成後再取消舊檔追蹤、處理 tests／臨時碼及舊目錄。

重現命令一律使用 `conda run -n opencode_env python ...`。本規範不新增通用分析框架；先調整各分析自己的入口。新的 CLI 命令由各 README 在實作完成後記錄，草案中的目標不當成已可執行功能。

## 8. 驗收條件

- 每個分析有自己的 README；正式 code 不藏在 ignored 輸出目錄。
- 所有原始／上游輸入位於本分析 `data/`，匯入前後 hash 一致。
- 本次產生的檔案全部位於 `result/<run_id>/`，只有唯一報告與引用圖片位於同 ID 的 `reports/`。
- 每個 run 的 reports 子樹恰有一個 `.md`；每個被引用的圖片都存在，與完整輸出圖片 hash 相同。
- 重跑會建立新 ID 或拒絕同名；不會覆寫舊 input/run，也不會悄悄重下載。
- 暫存的功能測試碼不留在正式 codebase；分析保有必要的輸入與輸出驗證。
- `git check-ignore` 確認 data/result 檔案被忽略，`git ls-files` 確認不再追蹤；程式、README 與精簡報告圖片可追蹤。
- 使用同一份輸入，純目錄遷移後的核心數值與原結果一致；浮點容差明列於遷移核對紀錄。W6A4 pilot 保持 144 個 context、目前全可配對的範圍及相同 ratio。
- Git commit 與資料備份分別可用；現有 worktree archive 在正式分析與資料恢復驗證完成前保留。
