# W6A4 serial 20007 分析

此分析將既有 W6A4 評估與修正版未量化 AMP baseline 配對，計算
`accuracy_ratio = w6a4_top1 / baseline_top1`，比較 16 datasets × 3 models ×
FT/FZ/CAL，共 144 個 context。這裡只分析保存的 W&B config/summary，沒有新推論。

## Codebase 與流程

| 程式 | 職責 |
|---|---|
| `acquire.py` | 匯入既有快照與原始 receipts，核對來源 hash，再一次發布新的 data/input_id；拒絕覆寫。 |
| `analyze.py` | 檢查完整 context、固定量化設定與 baseline，配對、核對公式／參考值、產生 heatmap。 |
| `run.py` | 建立獨立 run、記錄輸入／程式／環境、執行分析、寫唯一報告，以及唯讀驗證既有 run。 |

```text
匯入快照 → data/<input_id>/（唯讀）
         → 配對／統計／畫圖 → result/<run_id>/
         → 唯一報告與引用圖片 → reports/<run_id>/
```

## 資料與 baseline 契約

目前輸入為 `data/baseline_20261021/`，所有檔案及原來源路徑、取得時間與
SHA-256 記於 `source_manifest.json`。保存六個來源檔：cohort JSONL 與其原始
receipt、144 個 W6A4 runs 與其原始 receipt、遷移前配對表與其 verification
receipt。原 receipts 保留原始 bytes；其中的舊路徑屬於歷史 provenance。

- 每個 baseline context 唯一；兩側都必須有完整 144-context 矩陣。
- 固定設定：serial 20007、W6A4、weight/activation G32、extended、所有 heads
  納入（exclude 為空）、compute float16、quant_zero/quant_scale=True、
  debugging=False、AMP fp16=True；不相容時停止。
- 比對完整 checkpoint 路徑、seed、classifier、precision 與已記錄的評估設定。
  唯一接受的路徑 alias 是 `../../data/` → `/work/kyle0724/data/`。
- Baseline 為 unquantized AMP；`fp32_top1` 只是其他 legacy 流程的舊欄位名稱。
- 保留 batch-size 差異與 checkpoint bytes 未驗證狀態；ratio 不截斷。
- 新結果與固定的遷移前參考表比較 run IDs 與 top1、ratio、drop，容差 1e-12。
  更換 cohort 或 serial 需要另外建立分析契約與參考，不能覆寫此 input ID。

## 執行

從 repository root 使用 `opencode_env`；相對資料路徑由模組定位。

```powershell
# 沿用本機已匯入資料，自動產生含時間及微秒的新 run ID
conda run -n opencode_env python -m w6a4_analysis.run --input-id baseline_20261021

# 明確命名；相同 ID 再次執行會拒絕，既有報告也會阻止覆寫
conda run -n opencode_env python -m w6a4_analysis.run --input-id baseline_20261021 --run-id MY_NEW_RUN

# 只讀驗證，不產生新 run、不改寫舊 receipt
conda run -n opencode_env python -m w6a4_analysis.run --verify-run 20261021_163522_084281_w6a4_s20007
```

首次匯入另一份已驗證的同契約本機快照時使用以下入口；default sources 是遷移前
保存的位置，不是一般分析執行時的依賴。既有 input ID 會拒絕匯入：

```powershell
conda run -n opencode_env python -m w6a4_analysis.acquire --input-id NEW_INPUT_ID --selected-file PATH_TO_SELECTED_RUNS --target-file PATH_TO_20007_RUNS --reference-file PATH_TO_VERIFIED_PAIRS
```

## 輸出與驗證

- `result/<run_id>/tables/`：配對表與 heatmap 矩陣。
- `result/<run_id>/figures/`：PNG／SVG／PDF 完整輸出。
- `result/<run_id>/audit/verification.json`：144 格參考／公式核對。
- `result/<run_id>/provenance/source/`：本次使用的相關分析程式與 requirements。
- `result/<run_id>/run.json`：時間、命令、環境、輸入與輸出 hash、code identity、狀態。
- `reports/<run_id>/REPORT.md`：唯一報告；其 `figures/` 僅保存報告引用的 PNG／SVG。

失敗 run 保留 error.log 與唯一失敗報告，run.json 標示 failed。所有輸入與完整
輸出被 Git 忽略，必須另外備份；精簡報告與引用圖片可追蹤。

目前正式結果：[REPORT.md](reports/20261021_163522_084281_w6a4_s20007/REPORT.md)。
舊 `outputs/serial_20007/` 保留為歷史比對，不是此模組日常分析入口。
