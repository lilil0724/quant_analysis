# Activation quantization 分析

本目錄統一管理 activation quantization 的快照匯入、baseline 配對、統計、
W6A4 heatmap 與報告。從 repository root 使用同一個 `run` 入口，依 input
manifest 的 serial 選擇分析契約；`--serial` 可額外核對，填錯時停止。
本流程只分析已保存的 W&B config/summary，不訓練模型或執行量化推論。

## 支援的分析與 baseline

| Serial | 條件與矩陣 | Baseline／完成條件 |
|---|---|---|
| 20007 | W6A4，16 datasets × 3 models × FT/FZ/CAL，共 144 格 | 完整修正版未量化 AMP cohort；逐格比對歷史 reference 的 run IDs、top1、ratio、drop，容差 1e-12 |
| 20008 | W3/4/5 × A1/1.58/2/3/4/5/6，144 contexts × 21，共 3,024 格 | 僅唯一且相容的未量化 AMP baseline 計算 ratio/drop；缺失或多義保留 N/A；W/A grid 本身必須完整 |

兩者均固定 weight/activation G32、extended scope、head 納入、compute float16、
quant_zero/quant_scale=True、debugging=False、AMP fp16=True。
**Serial 是來源／契約識別，不能只因 scope 相同就混合兩批的結果。**
20008 的現行 TGDA launcher 描述 post-softmax、排除 patch embedding 的新流程，
但目前 W&B 快照缺少 server `evaluation.json` 的 coverage.protocol，尚未逐 run 核對。
20007 保留歷史 W6A4 契約，不以它當作 20008 的單變因控制組。

Baseline 比對完整 checkpoint 路徑、dataset、正規化 model、checkpoint mode、
seed、classifier、precision 與已記錄的 evaluation settings。唯一的路徑 alias 是
`../../data/` → `/work/kyle0724/data/`。缺少共同 checkpoint SHA-256 時，權重 bytes
身分仍未驗證；batch size 差異另外記錄。Baseline 是未量化 AMP，並非純 FP32。
`accuracy_ratio = quant_top1 / baseline_top1`，`drop_pp = baseline_top1 - quant_top1`；
ratio 不截斷。20008 的 W/A、model/mode 平均為 context 等權，不按測試樣本數加權。

## Codebase 與資料流

| 檔案 | 職責 |
|---|---|
| `run.py` | 統一 CLI、從 manifest 選 serial、核對 serial、唯讀驗證入口 |
| `acquire.py` | 匯入兩種已驗證本機快照，保留原 receipts，拒絕覆寫 input ID |
| `storage.py` | 共用 ID、路徑邊界、檔案大小／SHA-256 驗證 |
| `contracts.py` | 本模組的 dataset/model/mode 與 baseline identity／evaluation 契約 |
| `fixed.py`、`w6a4.py` | 20007 執行／provenance／驗證、144 格配對／reference 比較／heatmap |
| `grid.py` | 20008 完整 grid、baseline audit、W/A 平均、成對 activation 增量、報告與驗證 |
| `plot.py` | 20008 accuracy–activation 折線圖、同 checkpoint／W 的 A6 損傷 heatmaps、繪圖資料與公式核對 |
| `group_curves.py` | 依輸入快照中的使用者分組，計算 FGIR／UFGIR 的共同 strata 平均 acc_ratio 及兩條曲線 |
| `baseline_ratio.py` | 相對未量化 AMP 的 16-dataset ratio 總覽、W3/W4/W5 詳細 heatmaps、N/A 與資料量核對 |

```text
已保存的 W&B 快照／receipts
  → acquire → data/<input_id>/（不可覆寫）
  → run（由 manifest 選 20007 或 20008）
  → result/<run_id>/（完整表格、圖、provenance、run.json）
  → reports/<run_id>/REPORT.md（唯一報告與實際引用圖片）
```

配對規則已收回本目錄，不需要呼叫 `quant_acc` 或原 `w6a4_analysis`。
新增 serial／不同矩陣須先另訂契約，不會自動套用目前公式與 coverage 假設。

## 執行與驗證

依 Windows host 規則，使用 `opencode_env` 的完整 Python 路徑，無需 shell activation：

```powershell
$python = 'C:\Users\Public\miniconda3\envs\opencode_env\python.exe'

# 直接重用本機已驗證的 20008 input；自動產生 Asia/Taipei 的新 run ID
& $python -B -m activation_quantization.run --input-id wandb_s20008_20261102_135933_037365

# 可明確核對 serial 與命名；已有 run/result/report 時拒絕覆寫
& $python -B -m activation_quantization.run --serial 20008 --input-id INPUT_ID --run-id NEW_RUN_ID

# 20007：先恢復／匯入完整原始快照，不能只用歷史報告重建輸入
& $python -B -m activation_quantization.run --serial 20007 --input-id baseline_20261021

# 兩種分析共用唯讀驗證命令；支援整合前已完成的 20008 run
& $python -B -m activation_quantization.run --verify-run 20261102_135946_317954_s20008_brief
& $python -B -m activation_quantization.run --verify-run RUN_ID
```

`--input-id` 與 `--verify-run` 互斥。ID 僅允許英數字、底線與連字號。
現有 20008 的 `--input-id ... --run-id ...` 命令保持可用，`--serial` 非必要。
原 `python -m w6a4_analysis.run` 改用上述統一入口；原資料／報告中的命令保留作歷史紀錄。

## 匯入資料與來源保存

不自動下載或刷新；重用本機已驗證資料。`acquire` 是離線快照匯入入口，
不需要 W&B 登入。上游另行下載 W&B 時才需要 wandb 與 WANDB_API_KEY／login。

20007 匯入需要原 corrected cohort、20007 W&B snapshot、已驗證 reference pairs，
以及各自相鄰的原 receipts；核對後保留六個來源檔及新 source manifest：

```powershell
& $python -B -m activation_quantization.acquire --serial 20007 --input-id NEW_20007_INPUT --selected-file PATH_TO_SELECTED_RUNS_JSONL --target-file PATH_TO_20007_RUNS_JSON --reference-file PATH_TO_VERIFIED_PAIRS_CSV
```

20008 匯入現有完整 snapshot directory，其中必須有 `target.json`、`baselines.json`、
`filters.json` 與記錄大小／SHA-256 的 `source_manifest.json`，manifest.serial=20008：

```powershell
& $python -B -m activation_quantization.acquire --serial 20008 --input-id NEW_20008_INPUT --source-dir PATH_TO_SNAPSHOT_DIRECTORY
```

20008 新 input 另外保存原 `original_source_manifest.json`；重複匯入時保留所有
前代檔案／receipts，並用含 hash 的新名稱保存上一代主 manifest，實際名稱記於
`original_manifest_file`。原文件 bytes 不變。
所有 input manifest 記錄來源身分、取得時間、檔案大小與 SHA-256；既有 ID 拒絕覆寫。
20008 現有快照來源為 Public API `wandb.Api().runs` 的全量 config/summary，
target filter 為 `config.serial=20008`；baseline query 與取得時間保存在 filters／manifest。

## 輸出與完成條件

依賴為現有 opencode_env 的 pandas、numpy、matplotlib 與 Python standard library。
每個 run 保存 input hashes、程式快照／SHA-256、Git identity、實際命令、環境與狀態；
失敗 run 標為 failed，保留唯一失敗報告，不能當成完整分析。

- 20007：`tables/acc_ratio_pairs.csv`、`heatmap_values.csv`、PNG/SVG/PDF heatmap、
  `audit/verification.json`。報告僅引用 PNG/SVG，核對與 result 圖片 hash 相同。
- 20008：`runs_and_baseline_audit.csv`、W/A／model／mode 平均、baseline 摘要、
  同 checkpoint 同 weight 的 activation 成對增量，及 `validation.json`。新 run 亦自動產生下述圖集。
- 共用：`result/<run_id>/run.json`、`provenance/source/`、`reports/<run_id>/REPORT.md`。

執行時檢查 input manifest 大小／hash、路徑邊界、完整矩陣與唯一性、固定量化設定、
同 checkpoint 的 evaluation settings、保存後公式與 N/A。唯讀驗證另核對所有
receipt 所列輸出／報告 hashes，20007 重新比對 reference、報告圖片和原始配對。
所有 data/result 均 gitignore，需另行備份；Git 保存正式程式、README、報告與引用圖片。

## Activation 觀察圖

沿用上面的 `activation_quantization.run --input-id ...` 命令，新的 20008 run
會自動重算統計並產生兩類圖；不改寫先前的 run。

1. `accuracy_vs_activation`：model × FT/FZ/CAL 共九個面板，W3／W4／W5 各一色。
   淡線是個別 dataset，粗線是相同 16 datasets 的等權平均，Y 軸共用 0–100%。
   X 軸的 A1／A1.58／A2／A3／A4／A5／A6 是離散評估設定；淡線不是信賴區間。
2. `activation_loss_vs_a6_w3`／`w4`／`w5`：每張九個面板，各為 16 datasets × 七個 A。
   `loss_vs_a6_pp = Acc(同 checkpoint, W, A6) - Acc(同 checkpoint, W, A)`。
   紅色為比 A6 差，藍色為比 A6 好；保留負值，所有 W／model／mode 共用 dataset
   順序及零中心對稱色階。**A6 仍量化 activation，此圖不是完整的 activation 量化損傷。**

完整 PNG／可編輯 SVG、四頁 `activation_observation_atlas.pdf` 在
`result/<run_id>/figures/`；唯一報告引用並保存四張相同 hash 的 PNG。
`tables/accuracy_curve_means.csv` 保存 189 個平均點及各點的 dataset 數；
`tables/activation_loss_vs_a6.csv` 保存 3,024 個點、checkpoint／seed／classifier、
對應 A6 run ID、accuracy 與配對損失。完成前和唯讀驗證時均重新計算兩份表的公式，
並核對報告圖片。既有不含圖集的歷史 run 保持原驗證方式。

## 目前結果與歷史遷移

- [20008：Baseline acc_ratio 兩組曲線與全部 dataset heatmaps](reports/20261102_162715_226338_wa_s20008/REPORT.md)：
  同一個未量化 AMP 分母；16-dataset 總覽、W3/W4/W5 完整分層，保留 735 個 N/A 與 51 個 ratio>1。
- [20008：soy+cotton UFGIR／其餘 FGIR 的 acc_ratio 曲線](reports/20261102_153951_152893_wa_s20008/REPORT.md)：
  本次使用者指定 5 個 UFGIR、11 個 FGIR；兩條線各平均同一組 24 個 model／mode／W strata。
- [20008：Activation 折線圖與損傷 heatmaps](reports/20261102_142515_734861_wa_s20008/REPORT.md)：
  全部 3,024 格、189 個均值，每個均值 16 datasets；保留 127 個負損失格。
- [20008：整合後重算](reports/20261102_140733_229106_wa_s20008/REPORT.md)：3,024 格完整，
  與整合前六份 CSV 逐檔比對且 bytes 完全一致；baseline 109/144 contexts 唯一可比，25 缺失、10 多義。
- [20008：整合前原報告](reports/20261102_135946_317954_s20008_brief/REPORT.md)：原 input/result/report 保持原狀。
- [20007：正式歷史報告](reports/20261021_163522_084281_w6a4_s20007/REPORT.md)：
  原兩份報告及四張圖片均已移入本目錄，逐檔 SHA-256 不變。
- 20007 的 `data/baseline_20261021/` 和完整 `result/` 在本次開始時不在 checkout；
  本次沒有宣稱完成其真實資料重算。需恢復原備份後再使用 run／verify-run。
- [Relocation receipt](migrations/20261102_consolidation/relocation.json) 記錄原路徑、
  新路徑、檔案大小與 hashes；歷史 report／manifest 內容與 provenance 不改寫。

## FGIR／UFGIR 平均 acc_ratio 曲線

使用者本次指定 **UFGIR = soyageing、soygene、soyglobal、soylocal、cotton（5 個）**；
**FGIR = 其餘 11 個 datasets**，包含 food／inat17／moe。此為本次操作性分組，
不修改 `dataset_static_feature` 的歷史 8/4 core 規則。

新 input `wandb_s20008_soy_cotton_20261102_153946_160335` 重用原 W&B bytes，另保存
`dataset_granularity_mapping.csv`、`grouping_definition.json` 與完整前代 manifests。
同一個 run 入口偵測已列入 input manifest 的 grouping CSV 後，自動產生本圖：

```powershell
& $python -B -m activation_quantization.run --input-id wandb_s20008_soy_cotton_20261102_153946_160335
```

每筆 `acc_ratio = quant_top1 / uniquely matched unquantized AMP top1`，不是相對 A6。
先在每個 model／mode／W 內，對各組可用 dataset ratios 取算術平均，再對兩組共有的
strata 等權平均；不採 group accuracy 均值的比值、不依影像數加權。每個 stratum
至少各有兩個 datasets，七種 A 的成員集合固定。此批共保留 24 個 strata，FGIR
每層可用 6–11 個 datasets，UFGIR 每層 5 個；ViT-B/16 FT 僅 cotton 有可比 UFGIR
baseline，故其三個 W strata 排除。圖中不補缺失 baseline、不作因果或顯著性推論。

`figures/fgir_ufgir_mean_acc_ratio.png/.svg/.pdf` 為兩條線圖；
`tables/granularity_ratio_curves.csv` 保存 14 個曲線點，
`granularity_ratio_strata.csv` 保存各層 mean 與數量，`granularity_ratio_coverage.csv`
保存保留／排除原因及完整 dataset 成員。完成與唯讀驗證時均核對分組、ratio 公式、
相同 strata、平均值及報告圖片 hash。未帶 grouping CSV 的歷史 input 保持原行為。

## Baseline acc_ratio heatmaps

使用相同含 grouping CSV 的 input 執行 `run`，除分組曲線外，會產生
`dataset_mean_baseline_acc_ratio` 總覽及 `baseline_acc_ratio_w3/w4/w5` 詳細 heatmaps。
公式與分組曲線相同：`quant_top1 / uniquely matched unquantized AMP top1`，不是相對 A6。
所有圖共用以 ratio=1 為中心的色階，ratio>1 不截斷。

總覽包含全部 16 datasets，各列在分組曲線使用的共同 model/mode/W strata 內，
對該 dataset 的有效 ratios 取平均；列名 n 標示可用條件數（目前為 9/24 或 24/24）。
因為不同 dataset 的條件數不同，總覽的組內列平均不等同於分組曲線的共同 strata
等權平均。分層 W3/W4/W5 圖顯示全部 27 個 model/mode/W 條件：缺失或多義
baseline 格子保留灰色 N/A，並非 ratio=0。

本批共 3,024 格，2,289 個有效 ratio、735 個 N/A、51 個 ratio>1（最大約 1.5685）；
目前色階共用 0/1/1.6。總覽 112 個均值的公式及各列 n 在完成與唯讀驗證時重算。
原始格表為 `tables/baseline_ratio_cells.csv`，總覽為 `dataset_mean_baseline_ratio.csv`；
PNG／SVG 與四頁 `figures/baseline_acc_ratio_heatmaps.pdf` 保存於新 run。
