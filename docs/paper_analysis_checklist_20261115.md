# Dataset 與 quantization accuracy：資料及分析 checklist

更新日期：2026-11-15（Asia/Taipei）。依本機現行程式、保存的 audit 與使用者提供的 server 目錄結構整理；本次未登入 Nano4、未下載資料、未重新執行 inference 或分析。

## 狀態與分工

- **已確認缺口**：本機 audit 或 owning source 已能證實的資料／功能問題。
- **待驗證**：尚未核對 server 實際檔案或新版結果；不代表檔案不存在或必須重新 inference。
- **建議擴充**：論文可採用的新增分析，不是現行 runner 已有功能。
- `[x]` 只表示該項所描述的程式能力或分工已確認；不代表 Nano4 實驗已完成。

| 執行位置 | Folder | 工作 |
|---|---|---|
| 本機 | `quant_acc/` | Full weight grid 的執行／彙整與 accuracy analysis |
| 本機 | `dataset_robustness/` | Dataset ranking、stability、accuracy loss-profile PCA |
| 本機 | `dataset_static_feature/` | Dataset 分組與靜態屬性分析 |
| Nano4 | `dataset_features/` | Feature PCA、CKA、t-SNE、centered geometry |
| Nano4 | `dataset_confidence/` | Logit bias／competitors、margin matching 與後續 margin 分析 |
| Nano4，需要重新匯出時 | TGDA repository | 載入 checkpoints／test data，匯出 features、logits、predictions |

## P0：資料與 protocol，分析前完成

### A. 共同實驗身分【待驗證】

- [ ] 固定本次論文的 dataset／model／FT-FZ-CAL 清單；採完整預設時為 16 × 3 × 3 = 144 contexts。
- [ ] 固定 weight-only、量化 module 範圍、classifier exclusions、HQQ 版本、group size、seed 與 precision。
- [ ] 核對本機 full grid 的 W3/G128 與 server W3/G128 的 checkpoint、samples、preprocessing、classifier 和評估設定可比。
- [ ] 每組 FP/Q 核對 checkpoint 完整路徑及可用的權重 hash、sample IDs、targets、class dimension、execution metadata。
- [ ] 對 CAL 核對實際量化 coverage，確認 classifier 沒有被量化；不以資料夾名稱或 serial 單獨推定。
- [ ] 保留 CAL raw-image backbone features 與 crop/flip aggregation logits 的差異說明。
- [ ] 明確記錄 FP 的實際 precision；`fp32_top1` 舊欄名不作為純 FP32 證據。
- [ ] 舊 FT/FZ 與新 CAL 若使用不同 serial，建立逐 context 的 source map；搬入同一個父目錄不會改變 manifest 身分。

### B. 本機 full grid【已有不完整 audit，需刷新驗收】

依 `quant_acc/result/20261115_weight_only_s10000_10016_audit/run.json`，當次保存結果為：
141/144 baselines、4,490/4,608 quantized conditions、140/144 complete contexts，缺 118 個 quantized conditions；新版 CAL serial 10016 當次查詢取得 0 筆。
這些是歷史 query／snapshot 的結果，不是目前 W&B 或 Nano4 的即時狀態；140 個 complete contexts 也不代表已符合新版 backbone-only protocol。

| 當次不完整 context | Baseline | 當次已有 Q／預期 Q | 缺 Q |
|---|---|---:|---:|
| vegfru / ViT-B16 / CAL | 有 | 10 / 32 | 22 |
| vegfru / Swin-B / FT | 無 | 0 / 32 | 32 |
| vegfru / Swin-B / FZ | 無 | 0 / 32 | 32 |
| vegfru / Swin-B / CAL | 無 | 0 / 32 | 32 |

- [ ] 取得並驗證你本機重跑後的新版 CAL full-grid summary 與相容 FP baseline；舊 audit 尚未納入它們。
- [ ] 核對上述 4 contexts，優先重用已完成且可驗證的資料；僅對確實缺少／不相容的條件重新執行。
- [ ] 保存每個條件的 Q/FP accuracy、checkpoint、seed、bits、group、precision、classifier exclusion、來源 serial/run ID。
- [ ] 新 run 重新輸出 completeness／pair audit／missing conditions；缺格保留 N/A，不沿用不相容的舊 CAL。
- [ ] 若採單 seed、8 個 bits × 4 個 groups 的完整契約，驗收 144 baselines＋4,608 Q conditions；若更改 grid，另記新的預期數量。

### C. Nano4 feature／logit 原始資料【待 server preflight】

- [x] 現行 reader 支援逐 mode 混合舊 FT/FZ paired 格式與新版 CAL schema 2。
- [ ] 完整列出各 context 的實際 source folder、serial、格式及是否 complete；目前貼出的 soyageing 目錄不能證明全矩陣完整。
- [ ] 新 CAL 的 `fp/` 與 `w3_g128_linear/` 各自具有原始 manifest、samples、metrics、predictions 與完整 logits。
- [ ] Feature 分析需要的 FP/Q layer arrays 都存在；至少有 final，要跑逐層 CKA 則保存所選全部 layers。
- [ ] 舊 FT/FZ 的根 manifest、samples、summary、layer metrics 與 manifest 宣告的 FP/Q arrays 均存在。
- [ ] 核對 array shape/dtype、finite values、SHA-256、sample/target alignment、logits 與 predictions 一致性。
- [ ] 新 CAL 根目錄沒有殘留舊 paired manifest，避免與 `fp/manifest.json` 造成格式歧義。
- [ ] 若需重新匯出，開啟 `save_features`、`save_predictions`、`save_logits` 與 `extract_layers=all`；example config 目前三個 save 預設為 false，且含 W6A4，需選定本次 W-only conditions。
- [ ] 保存完整 raw arrays；compact ZIP／summary CSV 不足以重新計算全部分析。
- [ ] 為 margin import 的 FP/Q logits 快照及 PCA coordinates 預留額外磁碟空間。
- [ ] 先執行 feature pilot，再以涵蓋 soy/cotton 和 other 的 selection 驗證 bias／margin；最後驗證完整選定矩陣。

## P1：核心分析與交付產物

### D. 本機 accuracy 與 dataset ranking【程式已有，新資料結果待驗收】

- [ ] 重產 full-grid summary、absolute accuracy drop、accuracy retention ratio、bit/group 曲線與 heatmaps。
- [ ] 重產 dataset within-context／cross-context ranking stability、pairwise reversals、coverage 與 baseline audit。
- [ ] 重產 accuracy loss-profile PCA 的 scores、loadings、explained variance 與 sensitivity。
- [ ] 對本次新版輸入執行對應 verifier；不能將舊輸入的通過紀錄當成本次驗證。

### E. Dataset static feature【已確認資料契約與分析缺口】

- [ ] 匯入本次 quant_summary 與版本固定的 dataset mapping；目前模組預設 input paths 在本機不存在，需明確指定或建立新 input snapshot。
- [ ] 重用並核對 audit provenance 內已有的 15-dataset mapping；不需從零重建既有分組。
- [ ] 決定 16 datasets 中 cotton 的分組與納入政策，不能只由名稱或 margin 分組推定 FGIR/UFGIR。
- [ ] 更新 16-dataset 的 mapping／驗證契約；目前 verifier 固定檢查 15 datasets、4,320 rows，analyzer 固定要求 core 8 FGIR＋4 Ultra-FGIR。
- [ ] 排查舊 audit 的中止點：`Baseline changes within context: vegfru, swin_base_patch4_window7_224_in22k, cal`。先區分缺 context 與真正 baseline 不一致；現行程式對空 context 也可能進入同一錯誤分支。
- [ ] 用新版輸入完成 retention contrasts、4→2-bit degradation、coverage、leave-one-dataset-out、sensitivity 和圖表。
- [ ] 若論文的 static feature 包含連續屬性，新增具來源與 split 定義的 class count、train/test size、samples/class 資料表及關聯分析；目前模組只做固定粒度分組比較。

### F. Nano4 feature PCA／CKA／t-SNE【程式已有，新來源結果待驗收】

- [ ] Paired：取得新版 `all_pairs.csv`、`coverage.csv`、每組 `layer_metrics.csv`、`summary.json`。
- [ ] PCA：取得 FP basis、explained variance、FP/Q coordinates 與 `pca.png`；FP/Q 共用 FP 擬合的基底。
- [ ] CKA：取得每個匯出 layer 的 FP→Q centered linear CKA 與 undefined 狀態。
- [ ] 補逐層 CKA 曲線／heatmap 的 plot 步驟；現行 paired runner 輸出逐層 CSV，但不自動生成此類圖。
- [ ] Geometry：取得 class-equal intra/inter distance、separability、norm、final H-score，以及 centered distance CSV／FP means。
- [ ] t-SNE：取得圖、coordinates、method；固定 class IDs、seed 與設定，明列抽樣範圍。
- [ ] 跨 dataset 比較 sample-size 差異時，增加 matched-N CKA 敏感度；既有 config-driven 流程有相關程式，但不是目前 schema-2 paired runner 的自動產物，不能直接當成已完成。

### G. Nano4 logits／margin【部分程式已有，部分需接新輸入】

- [ ] Paired：取得 FP/Q accuracy、swap、damage/rescue、net damage 與 baseline normalized-margin Q10。
- [ ] Bias/competitor：取得 bias、winner rank、cross-fit folds、held-out prediction 比較、模型參數與 completion receipt。
- [ ] Margin matching：取得 raw/matched group metrics、gaps、balance、ESS/support/KS、dataset weights、sample weights、圖與報告。
- [ ] 分別保留 FP-correct damage 與 FP-wrong rescue；標明 conditional rates 與 unconditional mass 的分母。
- [ ] 若要保留舊 sample-margin 機制分析，將新版／混合 exports 接到 decile／AUROC 分析；現行腳本依賴舊 confidence NPZ 和 prepared inputs，尚未接入統一 runner。
- [ ] 決定 sample-margin 機制分析的條件範圍：沿用原 W3/W4 × G8/G128 需有 FP＋4 個 Q conditions；改成 W3/G128 單條件則明記分析範圍的改變。
- [ ] 產出 margin decile swap/damage 曲線、AUROC 與 FP-correct sensitivity，並保存 undefined 狀態；目前 runner 不會自動產生這套舊分析產物。

## P2：論文完整性與可重現性

- [ ] 合併 dataset-level 主表：dataset/model/mode/seed/quantization identity、FP/Q accuracy、drop、ratio、static attributes、FP margin/geometry、FP→Q CKA；不同來源不可只靠 dataset 名稱合併。
- [ ] 將量化前 FP 屬性與量化後 representation 變化分開解讀；後者是事後診斷，不直接當成量化前預測能力。
- [ ] 固定主要 outcome、主要量化條件、主要分組及 sensitivity 範圍；避免看完圖才挑 dataset、層或條件。
- [ ] 若要推廣 W3/G128 機制到整個 weight grid，選取額外代表性條件並增加 adapter／runner 支援；目前 feature/logit runner 固定 W3/G128。
- [ ] 明確限制 dataset、checkpoint、training recipe 與 seed 的混雜；不同 bits/groups 不當成獨立訓練重複。
- [ ] 若要主張跨訓練重現性，補獨立訓練 seeds；重跑同 checkpoint 主要驗證流程重現性。
- [ ] 各 run 保存 command、環境、原始 manifests、source/output hashes、coverage、failure/undefined 狀態與唯一報告。
- [ ] 備份被 Git 忽略的 `data/` 與 `result/`；報告及實際引用圖片另放可追蹤的 `reports/`。

## 建議執行順序

1. A–C：固定 protocol，核對／補齊資料，完成 server preflight。
2. D–E：本機重跑 accuracy、ranking 與 static-feature 分析。
3. F–G：Nano4 重算既有 feature/logit 流程；個別處理尚未接入的 CKA plots、matched-N 與 sample-margin 機制分析。
4. P2：合併主表、整理論文主圖與 sensitivity、完成 provenance 和備份。

證據入口：`quant_acc/result/20261115_weight_only_s10000_10016_audit/{run.json,audit/,logs/legacy15_granularity.log,provenance/legacy_dataset_granularity_mapping.csv}`；各 owning module README、`dataset_features/run_all.py`、`dataset_static_feature/{analyze,verify}.py`、`dataset_confidence/analyze_sample_margin_mechanism.py`。歷史記錄的舊名稱不改寫，使用既有 relocation receipts 對照現行路徑。
