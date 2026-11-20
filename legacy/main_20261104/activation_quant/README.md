# Weight／activation quantization accuracy 分析

此分析將不同 weight／activation 設定的評估與修正版未量化 AMP baseline 配對，
計算 `accuracy_ratio = quant_top1 / baseline_top1` 與 accuracy drop（percentage
points）。標準 cohort 為 16 datasets × 3 models × FT/FZ/CAL，共 144 個 context。
只分析 W&B config/summary，不執行推論或修改 TGDA。模組於 2026-11-04 由
`w6a4_analysis` 改名為 `activation_quant`，
歷史 serial 20007 使用原 schema-1 相容流程；新輸入使用 schema 2。

資料、完整結果與報告隨目錄一起移動，歷史 manifests、run.json、程式快照與報告
保持原 bytes。歷史命令中的 `python -m w6a4_analysis...` 改用
`python -m activation_quant...`；歷史絕對路徑的模組 prefix 同樣替換。
逐檔 SHA-256 與移動核對見 [relocation receipt](relocations/20261104_module_rename/README.md)。

## Codebase 與流程

| 程式 | 職責 |
|---|---|
| `acquire.py` | 明確指定 serial 下載 W&B 或匯入 JSONL；核對並複製既有 baseline 與 receipts，發布不可覆寫的輸入。保留原 schema-1 匯入。 |
| `quant.py` | 通用條件身分、完整評估身分配對、篩選、缺值／歧義／重複稽核、coverage 與公式驗證。 |
| `delta.py` | W+A 與已驗證歷史 weight-only cohort 配對，保留設定差異，計算描述性的 Top-1 delta（百分點）及 normalized ratio delta。 |
| `plot.py` | 各條件 heatmap、固定其餘設定且取共同評估集合的 W×A 比較圖，以及每個 W 的 dataset×A heatmap。 |
| `workflow.py` | schema-2 run、來源快照、唯一報告、hash 與唯讀驗證。 |
| `analyze.py` | schema-1 W6A4 專用的歷史配對／參考值驗證與圖表。 |
| `run.py` | 統一 CLI；依 input/run schema 分派通用或歷史流程，保留既有命令。 |

```text
匯入快照 → data/<input_id>/（唯讀）
         → 配對／統計／畫圖 → result/<run_id>/
         → 唯一報告與引用圖片 → reports/<run_id>/
```

## 通用輸入與命令（schema 2）

依賴根 `requirements.txt` 的 wandb、numpy、pandas、matplotlib，Python 使用
`opencode_env`。W&B 下載需要 `WANDB_API_KEY` 或已有 wandb login；不輸出 key。
正常分析只讀本機快照，不隱含連網或 refresh。新 serial 或 refresh 使用新 input ID。

```powershell
# 首次取得；相同 input ID 拒絕覆寫。數字與字串 serial 都會查詢，保存所有 run states。
conda run -n opencode_env python -m activation_quant.acquire --input-id NEW_S20008_INPUT --serials 20008 --baseline-input-id baseline_20261021

# 分析已下載的所有設定（不重新下載）
conda run -n opencode_env python -m activation_quant.run --input-id wandb_s20008_20261104

# 選擇設定；activation bits 支援小數，以及 off／unrecorded
conda run -n opencode_env python -m activation_quant.run --input-id wandb_s20008_20261104 --weight-bits 3 4 --activation-bits 1.58 4 --weight-group-sizes 32 --activation-group-sizes 32 --scopes extended --serials 20008

# 若來源 activation group size 明確為 null（全 token/head vector），使用 full 篩選
# conda run -n opencode_env python -m activation_quant.run --input-id INPUT_WITH_FULL_GROUP --activation-group-sizes full

# 唯讀重算配對、核對輸入／輸出／報告 hash 與引用圖片
conda run -n opencode_env python -m activation_quant.run --verify-run RUN_ID

# W3/4/5 分別畫 dataset×A 的 W+A minus weight-only delta（不重下載）
conda run -n opencode_env python -m activation_quant.run --input-id wandb_s20008_20261104 --weight-delta

# 改畫 acc_ratio 差（無單位），使用同一個 matched AMP baseline
conda run -n opencode_env python -m activation_quant.run --input-id wandb_s20008_20261104 --weight-delta --delta-metric acc_ratio

# 改畫 acc_ratio(W+A) / acc_ratio(W-only)，中性值為 1
conda run -n opencode_env python -m activation_quant.run --input-id wandb_s20008_20261104 --weight-delta --delta-metric acc_ratio_quotient

# 本機 JSONL 匯入，仍需明列 serial，且所有列必須屬於指定 serial
conda run -n opencode_env python -m activation_quant.acquire --input-id NEW_LOCAL_INPUT --serials 20008 --target-jsonl PATH_TO_RUNS_JSONL
```

W&B project 預設 `nycu_pcs/TiT`，可用 `--project` 指定。JSONL 每列保存
`id/state/url/created_at/config/summary`；量化讀 `summary.hqq_top1`，baseline 讀
`summary.top1`，單位均為百分比。`config` 需有 dataset/model、完整 checkpoint、
seed、classifier、precision 與評估設定；具體 baseline 評估欄位沿用
`legacy_hqq.recover_baselines.differences`，不以 basename 或 serial 代替配對。

新 input 保存 `target_runs.jsonl`、原 baseline cohort JSONL 與原始 receipts。
`source_manifest.json` 記錄 schema、取得時間／時區、W&B filter、project、serial、
run IDs／states、宣告的預期 context 矩陣，以及檔案大小／SHA-256。原始 receipts
逐 bytes 保留；不可變 input 不寫入 prepared tables。下載失敗不發布 input。

- 條件由 W/A bits、兩側 group size、scope、exclude、compute dtype、quant flags、
  fp16、quant_cfg_path 及其他已記錄 `hqq_*` 設定共同決定。僅排除
  `hqq_verbose`、`hqq_report_path`、`hqq_compare_fp32` 等輸出／診斷欄位。
- 不固定 bits 清單；支援正數與非整數 bits。activation 明確 null 為 off；缺少欄位
  為 unrecorded，不能當作 off。activation group size 明確 null 是合法的全向量分組，
  與缺少 group size 不同。未記錄的 scope／其他選項也不猜成目前預設值。
- baseline 必須是相容、finished、非 debugging 的未量化 AMP fp16。缺少或多個
  相容 baseline、重複量化身分、無效 metrics／設定或 unfinished run 都保留 N/A
  及原因；不自動選最新 run，也不截斷 ratio。
- 每個條件對 manifest 宣告的 144-context 矩陣獨立核對。多個 seed 不增加 context
  coverage；整個未出現的量化條件不宣稱已有資料。完成執行與 coverage complete
  分開記錄，部分資料為 `coverage_status=partial`。
- condition ID 含完整設定 hash；同為 W3A4 但 scope/group/exclude 不同不合併。
  各條件圖顯示每個 context 的 accepted seed/evaluation ratio 平均；W×A 圖固定
  非 bits 設定並取完整評估身分與 baseline ID 的交集，先平均 context，再等權平均
  datasets。共同集合及各條件原始集合數量保存於比較 audit。
- 每個固定 W 另產生 `dataset_activation_w<bit>_<id>` 圖，橫軸為 A、縱軸為
  dataset；跨 A 取完整評估身分與 baseline ID 的交集，各 model/mode 的 ratio
  在 seed/evaluation 內平均後再等權平均。20008 每個 dataset 都有 9/9 個 context。
  三張圖固定 0–1.6 色階、中心 1、格內顯示三位小數。FGIR/UFGIR 顯示順序依
  使用者參考圖，food/inat17/moe 列 FGIR、cotton 列 UFGIR；不變更 granularity
  模組的研究分組。完整數值及共同集合保存在 tables 與
  `audit/dataset_activation_populations.csv`，不同非 A 設定不混合。
- 舊 weight-only 資料因未記錄欄位與現代顯式設定不同，可能拆成更多條件；不靠
  默認補值把歷史條件合併。Config 路徑／設定不代表 checkpoint bytes 已驗證。

`--weight-delta` 模式只繪製各 W 的 dataset×A delta 圖：
`delta_top1_pp = W+A top1_pct - weight_only top1_pct`，負值表示 W+A 較低，單位為
百分點。也保存 `delta_accuracy_ratio = delta_top1_pp / matched AMP top1_pct`。
加上 `--delta-metric acc_ratio` 時，圖改為
`Δ acc_ratio = (W+A top1 / AMP top1) - (weight-only top1 / 同一 AMP top1)`。
先計算每個配對的 ratio 差，再依相同 context 平均；不以 dataset 平均值相除。
它沒有百分比／百分點單位，三張圖共用對稱、中心 0 的色階，至少 -1 到 +1，
需要時共同擴大以容納所有顯示值。預設 `--delta-metric top1_pp` 保留原百分點圖。
`--delta-metric acc_ratio_quotient` 則計算每個配對的
`(W+A top1 / AMP top1) / (weight-only top1 / 同一 AMP top1)`，等價於
`W+A top1 / weight-only top1`，再依相同 model/mode context 平均。
1 為相同、低於 1 表示 W+A 較低。分母為 0／無效或配對失敗時保留 N/A，
原因見 `weight_ratio_reason`；不以 0 或無限大代替。圖中三張共用至少 0–1.6、
中心 1 的色階，必要時共同擴大。這是配對 quotient 的平均，不是 dataset 均值相除。
此模式額外保存 `wa_over_weight_acc_ratio`、`weight_only_acc_ratio` 及其接受旗標，
不改寫既有 delta run 的表格 schema。
使用相同 W/group、compute dtype、quant flags、precision、完整評估身分與共同
AMP baseline ID 配對；歧義／缺值／已知 checkpoint hash 不同的比較保留 N/A。
跨 A 取相同評估身分與 weight-only run ID 的交集，再在 seed/evaluation 內平均、
最後等權平均 model/mode。色階固定 -100 到 +100、中心 0，各 W 保持相同尺度。

**這批 delta 不是 isolated activation effect**：20008 W+A 是 extended、包含 head；
保存的 weight-only 是 legacy linear、排除 head。W&B 相同 W/G32/extended/包含
head/activation-off 的查詢未找到對照。圖與唯一報告直接標示 scope/head 差異，
不得將差值只歸因於 activation quantization。歷史缺少 A/scope 欄位只在已驗證的
`recover_baselines` selected cohort 契約下解讀為 legacy weight-only linear，
不套用於任意新來源。其他量化差異逐格保存在 `quant_setting_differences`。

Delta 模式額外保存 `tables/weight_delta_pairs.csv`、`weight_only_pairs.csv`、各 W
的 mean/context CSV，以及 `audit/weight_delta_populations.csv`。唯讀驗證會重算
weight-only 配對及 delta 公式，並核對保存表格與所有圖片的 hash。

新 run 產生 `tables/acc_ratio_pairs.csv`、各條件 heatmap matrix、W×A context
means、`audit/condition_coverage.csv`、`missing_contexts.csv`、
`comparison_populations.csv`、`verification.json`、完整 PNG/SVG/PDF、程式來源與
`run.json`。唯一 `reports/<run_id>/REPORT.md` 引用的 PNG/SVG 複製後逐檔驗證 hash。
既有 input/run/report ID 拒絕覆寫；失敗 run 保存 error.log 與唯一失敗報告。

## 歷史 W6A4 資料與 baseline 契約（schema 1）

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

## 歷史執行

從 repository root 使用 `opencode_env`；相對資料路徑由模組定位。

```powershell
# 沿用本機已匯入資料，自動產生含時間及微秒的新 run ID
conda run -n opencode_env python -m activation_quant.run --input-id baseline_20261021

# 明確命名；相同 ID 再次執行會拒絕，既有報告也會阻止覆寫
conda run -n opencode_env python -m activation_quant.run --input-id baseline_20261021 --run-id MY_NEW_RUN

# 只讀驗證，不產生新 run、不改寫舊 receipt
conda run -n opencode_env python -m activation_quant.run --verify-run 20261021_163522_084281_w6a4_s20007
```

首次匯入另一份已驗證的同契約本機快照時使用以下入口；default sources 是遷移前
保存的位置，不是一般分析執行時的依賴。既有 input ID 會拒絕匯入：

```powershell
conda run -n opencode_env python -m activation_quant.acquire --input-id NEW_INPUT_ID --selected-file PATH_TO_SELECTED_RUNS --target-file PATH_TO_20007_RUNS --reference-file PATH_TO_VERIFIED_PAIRS
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

目前通用流程的 serial 20008 驗證結果：
[REPORT.md](reports/20261104_s20008_general_validated/REPORT.md)。
本機輸入 `wandb_s20008_20261104` 含 W3/4/5 × A1/1.58/2/3/4/5/6，共 21 種設定。
依參考圖分開繪製的 W3／W4／W5 dataset×activation heatmaps：
[REPORT.md](reports/20261104_s20008_dataset_activation/REPORT.md)。
重現命令與一般 schema-2 分析相同，使用新 run ID；未重新下載資料。
W+A minus weight-only 描述性 delta：
[REPORT.md](reports/20261104_s20008_weight_delta/REPORT.md)。
acc_ratio 差版本：
[REPORT.md](reports/20261104_s20008_weight_delta_acc_ratio/REPORT.md)。
acc_ratio(W+A) / acc_ratio(W-only) 版本：
[REPORT.md](reports/20261104_s20008_weight_acc_ratio_quotient/REPORT.md)。
歷史 W6A4 正式結果：
[REPORT.md](reports/20261021_163522_084281_w6a4_s20007/REPORT.md)。
舊 `outputs/serial_20007/` 保留為歷史比對，不是此模組日常分析入口。
