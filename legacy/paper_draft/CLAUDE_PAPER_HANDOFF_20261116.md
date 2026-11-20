# 論文初稿交接：Dataset 與 weight-only quantization accuracy 的關係

更新日期：2026-11-16（Asia/Taipei）。用途：交給 Claude 撰寫第一版論文。本文是研究進度與證據交接，不是已完成投稿稿件。

## 0. 給 Claude 的閱讀順序與工作要求

1. 先讀本文件的進度、protocol、結果與限制，再撰寫正文。不要將「有程式」「server 記錄通過」「本機獨立核對」「尚待實驗」混為一談。
2. 研究主題是 **dataset 與量化 accuracy robustness 的關係**。作者目前只研究 **weight-only**；完整 accuracy grid 在本機執行，Nano4 負責 feature/logit 相關 inference exports 與分析。
3. 可依已驗證的 W3/G128 結果撰寫 Methods、Results、Discussion。完整 weight grid、dataset static 屬性／FGIR-UFGIR 對照的最終新版結果尚未交付驗收，請用 `[TBD: ...]` 保留，不要補出數字或正向結論。
4. 建議先產出英文學術初稿，沿用附錄的既有英文 Introduction；另附中文的缺資料／作者待確認清單。語言若另有指定，以作者要求為準。
5. 所有結果句標明對應的資料表／欄位／條件。新的計算另列方法與產物，不要說成已在本次 audit 完成的結果。
6. 若你沒有本機檔案權限，本文的摘要表可供初稿使用；要檢查逐 dataset 結果與製圖，請作者另提供對應 CSV/PNG，不能假設可開啟本文的 `C:/...` 路徑。
7. 不要自動提交、投稿、覆寫現有 Word 草稿、改寫原始結果或刪除資料。先輸出新的 Markdown 初稿與 evidence map。

## 1. 研究定位

### 主題與候選題目

核心問題：在一致且可配對的 weight-only PTQ 設定下，為何不同 datasets 的 accuracy retention／掉分不同？這些差異在 architecture、checkpoint regime、bit width/group size 改變時是否仍存在？Representation geometry 與 classifier logits 能提供哪些描述性解釋？

較廣義的候選題目：**Dataset-Dependent Accuracy Robustness under Weight-Only Post-Training Quantization**。

既有 Introduction 使用的候選題目：**Dataset-Dependent Robustness of Low-Bit Post-Training Quantization in Fine-Grained and Ultra-Fine-Grained Recognition**。題目尚未定稿；FGIR/UFGIR 是研究動機與比較角度，不預設語意粒度是唯一原因。

目前工作主要是以既有 HQQ 量化方法進行系統性評估與機制診斷。沒有提出或驗證新的 quantizer，不應改寫成新量化演算法論文。

### 研究問題

| RQ | 問題 | 目前可回答程度 |
|---|---|---|
| RQ1 | Dataset 的 accuracy retention 如何隨 bit width/group size 改變？排名是否跨 model/mode 穩定？ | 需要本機 full-grid 最終新版結果；W3/G128 單條件結果已有 |
| RQ2 | FGIR/UFGIR 分組、類別數、資料量等 static properties 與掉分是否相關？ | Static 分組程式已有；最終 mapping、連續屬性表與新版結果待補 |
| RQ3 | FP feature geometry、H-score、norm、margin，以及 FP→Q CKA 與掉分如何關聯？ | 144 組 W3/G128 指標已有；可做後續關聯分析，不能預設所有 predictor 都有效 |
| RQ4 | Margin 分布、class-dependent logit bias、competitor structure 能描述多少 prediction changes 與 accuracy loss？ | W3/G128 bias/competitor、margin matching 已完成；舊 decile/AUROC 分析尚未接新版輸入 |

### Robustness 的操作定義

本文的 quantization robustness 指相對於 matched unquantized baseline 的任務表現保留，不是 adversarial robustness 或 corruption robustness。Dataset 也綁定其訓練所得 checkpoint；目前跨 dataset 比較不能隔離出純 dataset 的因果效果。

## 2. 實驗與論文進度快照

| 工作 | 狀態 | 寫稿時如何使用 |
|---|---|---|
| Weight-only full grid，本機 | 作者安排本機執行；本次尚未取得完整新版完成證據 | Experimental design 可描述；Results 留 `[TBD]` |
| Dataset ranking／accuracy loss-profile PCA | 程式存在，最終新版 full-grid 輸入尚待驗收 | 不把舊版描述性結果充作最終新版結果 |
| Dataset static features | 現有程式主要做 FGIR/UFGIR 分組；16-dataset 契約與資料尚有待補 | 不宣稱已完成 class-count/training-size 連續屬性研究 |
| Nano4 `run_all`：paired、centered、bias、margin、t-SNE | **已完成**，server run status=complete，下載包本機 audit 通過 | 可撰寫 W3/G128 Methods/Results，保留限制 |
| 整合 W3/G128 dataset 主表 | **已完成**，144 列、一列一個 dataset/model/mode | 後續接 full grid/static data 的入口 |
| 初稿 Introduction、文獻定位與引用對照 | **已存在**，主要是研究背景及預定 contributions | 本文附錄收錄英文 Introduction；須依進度改寫完成式主張 |
| 完整論文、定稿主圖、最終統計分析、投稿 | **尚未完成** | 不把資料整理完成當成 paper 完稿 |

本文件比 `docs/paper_analysis_checklist_20261115.md` 更新。舊 checklist 中「等待 server 產物」的部分已因本次交付而有進展；其他待辦仍需逐項確認。

### 已驗證的 run 身分

- Source run：`20261115_211022_759540_w3_s20004`。
- Source serial：`20004`。Serial 是來源身分，不表示量化條件，也不能單獨證明 classifier exclusion。
- Server input root：`/work/kyle0724/data/w3_g128_analysis`。
- Server run 開始：2026-11-15 21:10:23 +08:00；完成：2026-11-15 23:59:14 +08:00。
- 分析範圍：16 datasets × 3 models × 3 modes = **144 FP/Q contexts**，FP 對 W3/G128。
- 96 組 FT/FZ 使用舊 paired schema，48 組 CAL 使用新版 schema 2；均有來源及 hash 記錄。
- 本機 audit：`20261116_140853_554056_runall_s20004_audit`。
- 下載包 1,706 檔、約 255.7 MB 解壓容量；audit 驗證 1,695 個相異產物／code snapshot hashes，0 mismatch，所有匯入檔案另有 inventory hash。
- 五個 stage 共 11 個 preflight/analysis 紀錄 complete、exit code 0；306 張 PNG 通過圖片完整性檢查。
- 原始壓縮檔 SHA-256：`99c61da30c96b21648409763f26875fab187da46febede8e752e41ec6af3edb4`。

## 3. Dataset、模型與比較分組

### 已驗證 W3/G128 export 的 test rows／classifier classes

下表來自整合主表與 bias coverage。它描述本次 evaluation，不是原始資料集全部 train/test 規模；training size 仍需另有來源。所有 model/mode 的 N、C 已核對一致。

| Dataset | Test N | Classifier C |
| --- | --- | --- |
| aircraft | 3333 | 100 |
| cars | 8041 | 196 |
| cub | 5794 | 200 |
| dogs | 8580 | 120 |
| flowers | 6149 | 102 |
| food | 25250 | 101 |
| inat17 | 95986 | 5089 |
| moe | 2943 | 173 |
| nabirds | 24633 | 555 |
| pets | 3669 | 37 |
| soyageing | 4950 | 198 |
| soygene | 11143 | 1110 |
| soyglobal | 5814 | 1938 |
| soylocal | 600 | 200 |
| vegfru | 116931 | 292 |
| cotton | 240 | 80 |

每個 model/mode family 合計 324,056 個 test rows。不同 models/modes 重用這些資料，不可當成彼此獨立的訓練重複或新的獨立 datasets。

### 模型與 regime

| 名稱 | 檔案中的 model key | 已匯出 feature positions | Final feature dimension |
|---|---|---:|---:|
| ViT-B/16 | `vit_b16` | 13（12 blocks＋final） | 768 |
| Swin-B | `swin_base_patch4_window7_224_in22k` | 25（24 blocks＋final） | 1024 |
| BEiTv2 | `beitv2_base_patch16_224` | 13（12 blocks＋final） | 768 |

- Modes 為 `ft`、`fz`、`cal`，是三種 checkpoint regimes。FT/FZ 的實際訓練／凍結政策、pretrained weights、超參數及 CAL 版本要由 TGDA 訓練設定補寫；不能只由名稱推定完整流程。
- **CAL 是模型／checkpoint regime，不是本文件中的 probability calibration。** CAL predictions/logits 保留 exporter 的 crop/flip aggregation；features 來自 raw-image backbone call，兩者不是同一個 view representation。
- Feature positions 數量不等於 HQQ 量化 Linear layers 數量。各 architecture 的實際 quantized sites/counts 應取自原始 coverage manifest，不要寫成 13/25 個量化層。

### 三種分組不可混用

| 分析 | Group 定義 | 邊界 |
|---|---|---|
| Static granularity 的歷史 core mapping | FGIR 8：aircraft/cars/cub/dogs/flowers/nabirds/pets/vegfru；Ultra-FGIR 4：soyageing/soygene/soyglobal/soylocal | food/inat17/moe 是 core boundary；cotton 的正式納入政策待確認 |
| 本次 margin matching | `soy_cotton`＝四個 soybean datasets＋cotton，共 5；`other`＝其餘 11 | 是本次比較分組，不能直接改稱 semantic UFGIR vs FGIR |
| Competitor 的探索性 family holdout | soybean 4 vs non-soybean 12，cotton 在 non-soybean 側 | 與 margin 的 5/11 分組不同；primary prediction validation 是 LODO |

## 4. Quantization protocol 與 baseline

### 本次已完成的機制分析

- 量化設定：**weight-only Linear W3、weight group size=128**；activation bits=null。
- 研究意圖是量化選定 backbone Linear weights、排除 classifier。未獨立量化 Conv、attention matmul/probabilities 等運算；不將「weight-only」直接等同所有 backbone module 均被量化。
- FP baseline 是相同 checkpoint、test samples 與評估路徑下不使用 HQQ 的參考。Coverage 記錄 inference seed=100、AMP FP16、HQQ compute dtype=float16。它不是純 FP32 baseline。
- Inference seed=100、t-SNE seed=100、bias/competitor diagnostic seed=100004 具有不同用途，不是多個獨立訓練 seeds。
- Server preflight 已記錄原始檔身分、配對與 coverage 檢查通過；下載包沒有原始 condition manifests 內容，因此本機無法獨立重新驗證 CAL quantized sites。可報告 server 的檢查紀錄，勿升級成重新讀取 weights/images 的獨立驗證。
- 提交前仍需補 HQQ 版本／commit、量化設定檔、實際 exclusions/sites 與 checkpoint training provenance。保存的 source hashes／code snapshots 可協助定位；不要從一般預設值補造未記錄參數。
- 不以本實驗宣稱 integer-kernel throughput、latency 或 energy improvement；本次沒有這些量測。

### Full weight grid 的計畫與現況

既有契約為 bits `{1, 1.58, 2, 3, 4, 5, 6, 8}` × group sizes `{8, 32, 128, 512}`，每 context 32 個 Q conditions。若最終仍採 144 contexts／單 seed，預期為 4,608 Q conditions＋144 FP baselines。最終 grid 以作者本機重跑設定為準；1.58 是量化設定標籤，不應推論對應硬體資料型態或效能。

目前可查到的歷史 full-grid audit `20261115_weight_only_s10000_10016_audit`：141 baselines、4,490 Q conditions、140/144 complete contexts，缺 118 個 Q conditions；當次 query 沒取得新版 CAL 10016。這是舊 snapshot 的紀錄，**不是目前本機執行的即時狀態**。

| 歷史 audit 缺口 | 缺 baseline | 缺 Q conditions |
|---|---:|---:|
| vegfru / ViT-B16 / CAL | 0 | 22 |
| vegfru / Swin-B / FT | 1 | 32 |
| vegfru / Swin-B / FZ | 1 | 32 |
| vegfru / Swin-B / CAL | 1 | 32 |

在新 full-grid 結果驗收前，不聲稱這些缺口仍存在，也不聲稱已補完。Server W3/G128 的完整 144 contexts 不能取代完整 accuracy grid 的驗收。

## 5. 指標與分析方法：照實作撰寫

### 5.1 Accuracy、swaps、damage/rescue

令 `a_FP`、`a_Q` 為 accuracy fractions：

```text
accuracy retention ratio = a_Q / a_FP
relative loss            = 1 - a_Q / a_FP
absolute accuracy drop   = a_FP - a_Q
swap                     = FP predicted label != Q predicted label
damage                   = FP correct AND Q wrong
rescue                   = FP wrong AND Q correct
net damage R             = (damage_count - rescue_count) / swap_count
a_Q                      = a_FP - damage_count/N + rescue_count/N
```

Paired 與 margin matching 的 `damage_rate/rescue_rate` 是無條件 mass（分母為全部樣本或群組權重）；舊 confidence fragility 的 conditional rates 分母不同。Damage prediction model 則只在 FP-correct 子集評估。若分母為 0，保留 undefined，不補 0。

單位：paired accuracy、drop、rates 是 fractions；乘 100 後是百分比或 accuracy percentage points。`model_mode_summary.csv` 的 accuracy 已是百分比；bias `gain_pp` 已是 pp。

### 5.2 Feature CKA、geometry、norm 與 H-score

- 同一 context/layer 的 FP/Q matched features 使用 centered linear CKA：

  `CKA = ||Xc.T @ Yc||_F² / (||Xc.T @ Xc||_F × ||Yc.T @ Yc||_F)`。

- Geometry 先做每個 feature vector 的 L2 normalization。`D_intra` 在每個 eligible class 內取不同樣本對的平均 cosine distance，再對 classes 等權；`D_inter` 對不同 class pairs 等權。`separability = D_inter - D_intra`。
- Norm 指標：P50/P95/P99 與 P99/P50。
- Final regularized H-score 實作：`tr(S_b @ inverse(S_w + λI))`，`λ=0.01×tr(S_w)/D`。這裡使用 within-class scatter 定義，撰寫時不要自動替換成其他 H-score 公式。
- Full-dimensional 指標使用全部配對樣本；圖形 display subset 不取代這些數值。
- FP margin/geometry 是量化前屬性；FP→Q CKA 是量化後診斷。後者不能直接當成量化前的預測器。

### 5.3 PCA 與 centered geometry

- Feature PCA：每個 dataset/model/mode 在 **FP final raw features** 擬合兩個 PCs，FP/Q 共用同一 FP mean/basis；不各自 refit，不是 loss-profile PCA，也不是預先 L2-normalized features 的 PCA。
- 保存 FP basis、兩個 explained-variance ratios、全部 FP/Q 2D coordinates；圖片僅是 display。不同 contexts 的 PCA axes 不共用。
- Centered geometry 比較 `x/||x||` 與 `(x-mean_FP)/||x-mean_FP||`；mean 使用該 context 全部 FP samples 的 sample-weighted mean。不是逐 class centering。
- 跨 dataset 的量化 loss-profile PCA 在本機 full-grid 路徑，資料矩陣是一個 dataset 一列、量化條件一欄；目前最終新版結果待補。

### 5.4 t-SNE

- 每個 dataset 選取 `min(C,20,max(5,ceil(0.1*C)))` 個 classes，保留這些 classes 的全部 test samples；同 dataset 在不同 models/modes 使用相同 class IDs。
- Per-sample L2 normalization；需要時共用 PCA-50；FP/Q 拼接後 joint fit 一次 t-SNE，再分面顯示。
- 此 run：seed=100、perplexity=30、iterations=750。Method receipts 保存實際 sampling、selected IDs、counts、軟體版本與 hashes。
- t-SNE 描述選定類別的局部結構。不同 contexts 的獨立 embedding 距離／群大小不作為 robustness 排名或因果證據。

### 5.5 Logit margin

`z` 是 classifier 最終輸出的 class logits；CAL 沿用匯出的 view aggregation。Normalized decision margin：

```text
m(x) = (largest_logit - second_largest_logit) / std(all_class_logits, ddof=0)
```

Baseline normalized-margin Q10 是 FP samples 的第 10 百分位。它不是 softmax probability margin，也不直接等於既有文獻使用的所有 margin 定義。Zero/nonfinite spread 不當成有效零 margin。

### 5.6 Bias correction 與 competitors

- Bias diagnostic：依 target 分層的 five-fold crossfit，使用另外四折估計 row-centered `Q logits - FP logits` 的 class offsets；correction strength 固定 1；另有五組 shuffled-class controls。
- 此流程使用 paired test logits 的回顧性 crossfit，不能宣稱是未接觸 test data 的部署修正或 train-free generalization benchmark。
- Competitor features：FP winner 附近、落在 0.25/0.5/1.0 個 class-logit std 內的 competing-class counts。
- 比較 `margin_only` 與 `margin_plus_competitors` logistic models。兩者均有 nonlinear margin basis 與 log class count；training-only standardization、L2 C=100、train datasets 等權。
- Primary validation：每個 model/mode 分開做 leave-one-dataset-out。Damage 在 FP-correct 子集評估；swap 模型保留原本的 FP correctness control。另有探索性的 soybean-family transfers。
- Brier/log-loss 越小越好；AUROC 越大越好。`delta_* = competitors - margin_only`，因此改善的 Brier/log-loss delta 為負，AUROC delta 為正。
- 保留 stored maximal tie winners、AUROC undefined 狀態；不以 model/condition 次數當成獨立訓練重複。

### 5.7 Baseline-only margin matching

- 比較 focus 5 datasets（soybean 四組＋cotton）與 other 11，原始組內每個 dataset 等權。
- 主 bin width=0.05，sensitivities=0.025/0.10。
- 兩組 histogram 每個 bin 取 minimum，正規化為共同目標分布；只依 FP margin 建權重。
- 同一組 weights 同時評估 FP 與 Q correctness，包含 FP 原本錯誤的樣本。
- Retention 是 **weighted Q accuracy / weighted FP accuracy**，不是 dataset retention ratios 的簡單平均。
- Gap=`ratio_other - ratio_soy_cotton`。Absolute gap reduction fraction=`(|raw gap|-|matched gap|)/|raw gap|`；0 raw gap 時保留 undefined。
- 必須一起報告 ESS、raw ESS、overlap/support、weights 與 dataset composition changes。Matching 後比較的是共同支撐區域；不能當成隨機實驗或 causal adjustment 完成。

## 6. 已驗證結果：可寫入初稿的數字

### 6.1 Model/mode 摘要

以下平均值對 16 datasets 等權；relative loss 與 final CKA 為 dataset 中位數。僅適用本次 W3/G128。

| Model | Mode | Mean FP acc (%) | Mean Q acc (%) | Median rel. loss (%) | Median final CKA | Mean bias gain (pp) | Mean damage ΔBrier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-B16 | ft | 70.75 | 65.52 | 6.10 | 0.881 | +0.641 | +0.003710 |
| ViT-B16 | fz | 62.81 | 56.14 | 8.49 | 0.913 | +1.524 | -0.004579 |
| ViT-B16 | cal | 72.04 | 67.28 | 3.53 | 0.952 | +1.294 | -0.006072 |
| Swin-B | ft | 72.52 | 62.62 | 9.26 | 0.866 | +1.914 | -0.002342 |
| Swin-B | fz | 69.65 | 59.53 | 10.60 | 0.913 | +4.150 | -0.009322 |
| Swin-B | cal | 79.28 | 70.73 | 5.13 | 0.903 | +2.466 | -0.002221 |
| BEiTv2 | ft | 64.97 | 50.38 | 25.67 | 0.778 | +2.783 | +0.000012 |
| BEiTv2 | fz | 62.92 | 12.28 | 89.42 | 0.160 | +2.060 | -0.000193 |
| BEiTv2 | cal | 76.36 | 50.10 | 44.76 | 0.857 | +11.313 | -0.002387 |

來源：`data/derived/20261116_140853_554056_runall_s20004_audit/model_mode_summary.csv`。Delta Brier 是 primary LODO damage 結果。

可支持的敘述：在此量化條件下，退化隨 model/mode 差異很大；BEiTv2/FZ 尤其明顯。不能推廣成 BEiTv2 在所有 quantization settings 都不穩定，也不能以此單一表建立普遍 dataset ranking。

### 6.2 主 margin matching 結果

以下 gap 是 ratio 的差，**不是 accuracy pp**。Reduction 百分比是 gap 的相對縮小幅度，不是 Q accuracy 改善百分比。

| Model | Mode | Raw gap (ratio) | Matched gap (ratio) | Absolute gap reduction (%) |
| --- | --- | --- | --- | --- |
| ViT-B16 | ft | 0.2119 | 0.0425 | 80.0 |
| ViT-B16 | fz | 0.3097 | 0.1883 | 39.2 |
| ViT-B16 | cal | 0.2683 | 0.1184 | 55.9 |
| Swin-B | ft | 0.4984 | 0.2480 | 50.2 |
| Swin-B | fz | 0.5688 | 0.4117 | 27.6 |
| Swin-B | cal | 0.3409 | 0.1120 | 67.1 |
| BEiTv2 | ft | 0.4077 | 0.1881 | 53.9 |
| BEiTv2 | fz | 0.1839 | 0.0783 | 57.4 |
| BEiTv2 | cal | 0.3248 | 0.0608 | 81.3 |

可支持的敘述：在這 9 個 model/mode 設定、bin width=0.05 下，共同 margin 分布標準化後的組間 retention gap 皆縮小，約 27.6%–81.3%。

不支持的敘述：margin 因果解釋了 27.6%–81.3% 的所有 dataset effects；matched 結果等同實際部署 accuracy 被改善；剩餘差異只由 semantic granularity 決定。

### 6.3 需要一併呈現的負面／未定義結果

- Competitor models 的 mean damage ΔBrier 在 7/9 設定為負，但 ViT/FT 為 **+0.003710**，BEiTv2/FT 約 **+0.000012**。不能寫成 competitors 在所有設定都提升 prediction quality，且目前沒有統計顯著性主張。
- 1,152 列 score 中有 **8 列 undefined AUROC**：BEiTv2/FZ、cotton/soylocal 的部分 damage tests，因 damage outcome 全為 1。這是四組 evaluation contexts × 兩種 risk models；不是 dataset 只剩一個分類 label。保留 NA，不補成 0 或 0.5。
- 三種 bin widths 中，最低 matched/raw ESS 約 **20.8%**；最高 weight multiplier 約 **5.20**。ESS/N 最低約 4.05%，但 raw dataset-equal weighting 本身已降低 ESS，不能解讀為只保留 4.05% 實際樣本。
- Matching 最大 histogram balance error 約 **7.88e-14**；108 列 group estimates 均有定義。
- **38/144** contexts 的 PCA 前兩軸合計 variance<10%；最低為 inat17/Swin/FT 的 **1.50%**。這些圖不能代表全部 representation geometry。
- BEiTv2/FZ mean FP/Q accuracy 為 **62.92%/12.28%**，median relative loss **89.42%**、median final CKA 約 **0.160**。跨表格與 count 核對一致，但仍應與 full grid 同 checkpoint/protocol 的結果對照。

### 6.4 尚未形成的結果，不要由圖推論補齊

- Full-grid dataset ranking 與 rank reversal 的最終新版結論。
- FGIR/UFGIR 或 continuous static attributes 的最終關聯、效應量與顯著性。
- CKA／H-score／margin 作為「量化前 predictor」的完整統計比較。主表已有數值不代表已完成 predictive validation。
- 跨訓練 seeds 的不確定性、因果效應、普遍性結論。
- 舊 sample-margin decile/AUROC 與 matched-N CKA 的新版完整結果。

## 7. 資料與稿件位置

### Paper root（結果已複製，原始下載檔保留）

```text
C:/iCloudDrive/project_2/paper/quant-dataset robustness/
├── README_DATA.md
├── TRANSFER_MANIFEST.json
├── data/
│   ├── run_all/20261115_211022_759540_w3_s20004/
│   │   ├── run.json
│   │   ├── logs/  provenance/
│   │   ├── paired/       # layer metrics、PCA basis/full coords、summary
│   │   ├── centered/     # raw/centered geometry 與 FP means
│   │   ├── tsne/         # 144 組 PNG/coordinates/method
│   │   ├── bias/         # 144 組 diagnostics、LODO/family-transfer tables
│   │   └── margin/
│   │       ├── result/analysis/  # tables、sample weights、figures、provenance
│   │       └── reports/analysis/
│   ├── derived/20261116_140853_554056_runall_s20004_audit/
│   │   ├── dataset_master.csv
│   │   ├── model_mode_summary.csv
│   │   ├── margin_primary_summary.csv
│   │   ├── matching_support_summary.csv
│   │   ├── undefined_auroc.csv
│   │   └── source_code_comparison.csv
│   └── audit/20261116_140853_554056_runall_s20004_audit/
│       ├── verification.json  hash_verification.csv
│       ├── input_source_manifest.json  run.json
│       ├── COPY_RECEIPT.json
│       └── provenance/
└── reports/20261116_140853_554056_runall_s20004_audit/
    ├── REPORT.md
    └── figures/
```

`dataset_master.csv` 有 144 列，以 dataset/model/mode 一對一合併 paired、PCA variance、centered geometry、bias、class count、t-SNE sampling metadata。合併 full grid 前，仍需比對 checkpoint、seed、precision、classifier 與 model alias，不能只按 dataset 名稱合併。

本機 paper bundle 未含原始逐層 features、full logits、predictions/samples snapshot 或 CAL 原始 condition manifests。完整 raw data 仍在 server；目前副本可重畫現有 PCA/t-SNE coordinates、畫逐層 CKA 曲線，但不能完整重算 representation metrics 或舊 sample-level margin outcomes。

### Code 與既有文稿

| 位置 | 用途／狀態 |
|---|---|
| `C:/project/quant_analysis/quant_acc/` | Accuracy grid 分析；原 `legacy_hqq/` |
| `C:/project/quant_analysis/dataset_ranking_robustness/` | 目前的 ranking/loss-profile PCA folder；部分 README 還保留舊 `dataset_robustness` 名稱，執行前需核對實際命令 |
| `C:/project/quant_analysis/dataset_static_feature/` | 原 granularity 分組分析 |
| `C:/project/quant_analysis/dataset_features/` | Feature readers/metrics、統一 Nano4 runner |
| `C:/project/quant_analysis/dataset_confidence/` | Margin、bias、competitor methods |
| `C:/project/TGDA/` | Training／quantized evaluation／feature exports 的 owning repository |
| `C:/project/quant_analysis/paper_draft/INTRODUCTION.md` | 既有詳細英文 intro、中文說明、citation map |
| `C:/project/quant_analysis/paper_draft/ptq_fgir_intro_gap_check_20261116.md` | 文獻定位與 novelty 邊界 |
| `C:/project/quant_analysis/paper_draft/references.json`、`pdfs/` | 引用身分與本機 PDF；引用編號不等於 PDF filename prefix |

原始 source/provenance 中的舊名稱、Linux 絕對路徑及 manifests 保留原文；用 relocation receipts 與本機相對路徑定位，不修改歷史 hash 紀錄。

## 8. 還缺什麼：作者待確認／待補

### 投稿必要資訊

- [ ] 新 full-grid run 的完整設定與完成／缺格紀錄，尤其新版 CAL 及相容 baseline。
- [ ] FT/FZ/CAL 的正式方法名稱、training regime、pretrained checkpoints、seed、split、preprocessing 及 CAL view aggregation。
- [ ] HQQ 版本、原始方法引用、設定檔、actual quantized sites/counts 與 classifier exclusions。
- [ ] 同一 W3/G128 在本機 grid 與 server exports 的 checkpoint/protocol compatibility。
- [ ] 完整 static attributes 資料表與其來源；training sample counts 不能從 test N 推算。
- [ ] 最終 FGIR/UFGIR mapping、cotton 政策及 boundary/sensitivity 設計。
- [ ] 更新 static verifier 的舊 15-dataset/4,320-row 固定檢查；處理舊 audit 的 vegfru/Swin/CAL 缺 context／baseline 錯誤，不能僅憑錯誤字串宣稱 baseline 確實變動。

### 目前未接入新 runner 的分析

- [ ] 逐層 CKA curves／heatmaps：數據已在 `layer_metrics.csv`，圖尚需另外製作。
- [ ] Matched-N CKA：既有另一套 config-driven workflow 有程式，尚未接目前 schema-2 paired runner。
- [ ] 舊 sample-margin decile／AUROC：需要 adapter 與 per-sample outcomes；舊版主要條件為 W3/W4 × G8/G128，不能把這次單條件 competitor AUROC 當成已完成該分析。
- [ ] 若主張機制適用於多 bit/group，補代表性條件的 exports 及相同分析；目前機制證據僅 W3/G128。
- [ ] 若主張跨訓練重現性，補獨立訓練 seeds；重跑同 checkpoint 不等於新的 training replicate。

## 9. 初稿架構與建議圖表

| Section | 現在可寫 | 待補處理 |
|---|---|---|
| Abstract | 研究問題、matched protocol、已完成 W3/G128 診斷與保守結論 | 不寫 full-grid/static 的虛構結果；保留 TBD 或明確 preliminary 範圍 |
| Introduction | 附錄既有 intro、dataset heterogeneity、FGIR/UFGIR 動機 | 將 planned contributions 與已完成證據分開 |
| Related Work | 依已整理 citation map；quantization、heterogeneous degradation、granularity、representations | HQQ 方法原始引用待補；新 novelty claim 需另查原始文獻 |
| Experimental Setup | 16 datasets、3 backbones、3 regimes、baseline、W-only scope、配對與 provenance | Training/pretrained/HQQ 精確版本欄位 TBD |
| Dataset-level Results | W3/G128 144-row evidence | Full-grid curves、ranking、static associations TBD |
| Representation Analysis | CKA/geometry/H-score/PCA/t-SNE 的方法與現有例圖 | 全面關聯檢定／matched-N sensitivities 未完成者列 TODO |
| Decision-level Analysis | Damage/rescue、bias、LODO competitors、margin matching、ESS/NA | 不將回顧性修正當成部署方法 |
| Discussion/Limitations | Confounding、單 seed、grouping/domain、support、低 PCA variance、W3-only 範圍 | 不編造 confidence intervals 或 causal conclusions |
| Conclusion | 本次可證實的 setting-dependent degradation 與 decision diagnostics | 未完成研究問題以未來工作／TBD 清楚標示 |

建議圖表（編號僅是初稿安排，不代表圖已全部完成）：

- Table 1：dataset/test N/classes；train N 待補。
- Table 2：本文件的 model/mode W3/G128 摘要。
- Figure A：full weight-grid curves／heatmaps（待新版資料）。
- Figure B：dataset ranking stability、static group contrasts（待新版資料／驗收）。
- Figure C：代表性 layer-wise CKA curves，加上 PCA/t-SNE 輔助圖；明示 layer mapping、sampling、explained variance。
- Figure D：FP margin distributions 與 overlap matching；Table 3 報 9 組 gaps、ESS/support。
- Figure E／Table 4：bias gain、shuffled controls 與 LODO competitors，保留非改善設定與 NA。
- Appendix：完整 144 contexts、bin-width sensitivities、dataset mass shifts、undefined scores、provenance。

## 10. 初稿主張的界線

可以寫：

- 本研究在 matched W3/G128 contexts 中觀察到不同 dataset/model/regime 的差異，並以 representation 與 decision-level diagnostics 加以描述。
- 此資料中，baseline-margin overlap standardization 後，9 組 model/mode 的特定兩組 retention gaps 均縮小；其程度與 support/weights 一起報告。
- Bias 與 competitor 訊號在部分設定有額外描述能力，效果並非普遍；所有結論受 checkpoint、training、domain、sample size 與 seed 限制。

不能直接寫：

- 「UFGIR 天生 margin 較小，因此量化一定更差」或「margin 是 dataset effect 的因果原因」。
- 「第一次研究 dataset-dependent quantization」或「從來沒有研究 FGIR quantization」；既有文獻已有相關 heterogeneous degradation。
- 「Full grid／static analysis 全部完成」「多 training seeds」「statistically significant」「普遍 predictor」而沒有相應證據。
- 「CKA 高所以 accuracy 一定保留」「t-SNE 群距代表分類難度」「PCA 二維圖證明整個 feature space collapse」。
- 「校正不使用 test data」或「matched accuracy 是真實部署模型重新評估所得」。
- 「AMP-FP16 baseline 是純 FP32」「所有 backbone layers 都被量化」「已量測整數運算加速」。

## 11. 可直接交給 Claude 的任務

> 請依這份交接文件與已提供的 CSV/圖，撰寫一篇英文論文第一版，研究 dataset 與 weight-only PTQ accuracy robustness 的關係。沿用附錄的 Introduction 與既有引用定位，但依本文件更新完成狀態。Methods 必須符合實際實作，Results 只能使用已驗證 W3/G128 證據；full-grid/static/尚未完成的分析以具體 `[TBD: ...]` 標記。請同時輸出：完整 Markdown draft、claim→evidence 對照表、每張圖表所需的資料檔與欄位、中文的作者待補清單。不要新增不存在的數字、文獻、p-values、training seeds 或因果結論；若要新增計算，請列明方法並保存產物。保留 negative/undefined results，特別是 competitor 非全面改善、8 列 undefined AUROC、ESS 降低、低 PCA variance 及 CAL features/logits views 差異。

## 附錄 A：既有英文 Introduction（供初稿沿用並更新）

以下保留既有稿內容。其 research overview/contributions 有些描述的是預定完整研究範圍；**本文件第 2、6、8 節的狀態優先**，不要把此附錄的完成式語氣當成 full-grid/static 已驗收。

Quantization offers a practical approach to reducing the memory footprint and computational requirements of neural networks by representing weights and activations at lower numerical precision. Two common approaches are quantization-aware training (QAT) and post-training quantization (PTQ). QAT incorporates quantization effects into training or fine-tuning, allowing model parameters to adapt to reduced precision, but requires additional optimization and suitable training data. PTQ instead converts an already trained model without requiring a full quantization-aware training stage. This lower adaptation cost makes PTQ attractive when training resources or access to the original training data are limited, although calibration and, in some methods, local optimization are still necessary [1].

Substantial research has improved the accuracy of quantized models. For vision transformers (ViTs), PTQ4ViT addresses challenging activation distributions through twin uniform quantization and Hessian-guided calibration [2]. Mixed-precision approaches such as LampQ allocate different bit widths according to component sensitivity, balancing accuracy and compression under a precision budget [3]. Related data-limited quantization research improves the information available during adaptation: FDDA aligns synthetic data with class-conditional statistics [4], while SynQ improves synthesis-aware fine-tuning through noise reduction, attention to relevant image regions, and selective use of soft labels [13]. These advances address complementary sources of quantization error, with different requirements for data and optimization. However, improvements to a quantization method do not by themselves explain why its performance may vary between recognition datasets.

Complementary studies have demonstrated that quantization degradation is heterogeneous. Evaluations in medical imaging report variations across datasets and backbone architectures [5], while fairness-oriented research shows that classes or population groups can experience unequal degradation [6]. Reliability benchmarks further examine the influence of calibration distributions and worst-case group performance [14]. At the sample level, quantization-induced prediction disagreements have been associated with distribution shift and small output margins [7]. These findings motivate examining not only the quantization procedure, but also the properties of the data and learned model on which that procedure operates.

Fine-grained image recognition (FGIR) provides a useful setting for this investigation because successful classification requires distinguishing visually similar subcategories. Ultra-fine-grained image recognition (UFGIR) extends this challenge to distinctions such as cultivar identity, where visual differences can be difficult even for human experts to identify [8]. Research on UFGIR consequently emphasizes learning subcategory-specific discriminative information, including under limited training samples [12]. These task characteristics motivate asking how well low-precision computation preserves the distinctions needed for recognition. They do not, however, establish that UFGIR models necessarily have smaller prediction margins or greater quantization sensitivity. Both possibilities require empirical examination.

Despite existing evidence of heterogeneous quantization effects, the relationship between recognition granularity and accuracy retention under aggressive low-bit PTQ remains insufficiently characterized, particularly across FGIR and UFGIR datasets. Comparing final quantized accuracies alone is insufficient because datasets differ in their unquantized performance, and their relative robustness may also depend on architecture and checkpoint regime. We therefore ask whether FGIR and UFGIR datasets exhibit different degradation patterns, whether dataset robustness rankings persist across model contexts, and which representation- and decision-level characteristics are associated with those patterns. Our focus is a systematic characterization of these relationships, with recognition granularity treated as a motivating task property rather than an assumed cause.

To address these questions, we organize the analysis at three complementary levels. First, we characterize dataset-level degradation under a common weight-only PTQ protocol. Each quantized model is compared with its matched unquantized baseline, and both accuracy retention and absolute accuracy loss are examined across bit widths and quantization group sizes. We compare FGIR and UFGIR groups and evaluate the stability of dataset robustness rankings across architectures and checkpoint types. This distinguishes patterns that recur across evaluated contexts from those that depend on a particular model or quantization configuration.

Second, we investigate learned representations through paired unquantized and quantized features. The broader accuracy sweep is complemented by a focused W3/G128 analysis, using 3-bit weights and a quantization group size of 128. We measure layer-wise representation similarity using centered linear CKA [9] and examine intra- and inter-class feature geometry. Unquantized geometry describes the initial representation structure, whereas changes after quantization diagnose the perturbation introduced by reduced precision. PCA and t-SNE [11] provide complementary visualizations of the paired features. Together, these analyses assess whether representation-level measurements are associated with accuracy degradation, without equating feature similarity or visual cluster separation with preserved classification performance.

Third, we study classification decisions using paired logits and predictions. Prior work distinguishes the susceptibility of a prediction to quantization from the accuracy consequence of changing that prediction [10], motivating an analysis beyond the total prediction-swap rate. We examine normalized logit-margin distributions, class-dependent logit offsets, and competing-class structure. We separately measure damage to initially correct predictions and rescue of initially incorrect predictions, relating these transitions to net accuracy loss. This connects dataset-level degradation with sample-level decision behavior while testing candidate explanations rather than assuming that margin or feature geometry alone determines robustness.

The contributions of this work are summarized as follows:

- **Dataset-level characterization:** We characterize accuracy-retention and degradation patterns across FGIR and UFGIR datasets under matched model and quantization settings.
- **Cross-context comparison:** We examine the stability and reversals of dataset robustness rankings across architectures and checkpoint types, assessing the extent to which robustness patterns generalize across the evaluated contexts.
- **Representation and decision analysis:** We relate dataset-level degradation to layer-wise representation similarity, class geometry, logit margins, and prediction transitions, providing complementary diagnostics of robustness under aggressive weight-only PTQ.

## 附錄 B：既有 citation map

以下沿用 `paper_draft/INTRODUCTION.md` 與 `references.json` 的整理，不是本次新增全面文獻檢索。原始 PDF 在 `C:/project/quant_analysis/paper_draft/pdfs/`；若 Claude 沒有本機存取權，應請作者附上要引用的 PDFs。BibTeX、頁碼／公式及 publication version 仍需投稿前逐項核對。

編號 [1]–[8] 延續原 intro；[9]–[14] 是本詳細版新增的引用編號。**引用編號不等於 PDF 檔名前綴**。

| 引用 | 論文與來源 | 本機 PDF | 引用用途 |
|---|---|---|---|
| [1] | Gholami et al., [A Survey of Quantization Methods for Efficient Neural Network Inference](https://arxiv.org/abs/2103.13630), 2021 preprint / book chapter | [PDF](C:/project/quant_analysis/paper_draft/pdfs/01_Gholami_2021_Quantization_Survey.pdf) | QAT／PTQ 定義與成本。 |
| [2] | Yuan et al., [PTQ4ViT: Post-Training Quantization for Vision Transformers with Twin Uniform Quantization](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/7856_ECCV_2022_paper.php), ECCV 2022 | [PDF](C:/project/quant_analysis/paper_draft/pdfs/02_Yuan_2022_PTQ4ViT.pdf) | ViT quantizer 設計與 calibration。 |
| [3] | Kim et al., [LampQ: Towards Accurate Layer-wise Mixed Precision Quantization for Vision Transformers](https://arxiv.org/abs/2511.10004), AAAI 2026 / 2025 preprint | [PDF](C:/project/quant_analysis/paper_draft/pdfs/03_Kim_LampQ.pdf) | Sensitivity-aware mixed precision。 |
| [4] | Zhong et al., [Fine-grained Data Distribution Alignment for Post-Training Quantization](https://doi.org/10.1007/978-3-031-20083-0_5), ECCV 2022 | [PDF](C:/project/quant_analysis/paper_draft/pdfs/04_Zhong_2022_FDDA.pdf) | Class-conditional synthetic distribution；不是 FGIR／UFGIR 的實驗證據，也不是本專案 cosine distance 公式的來源。 |
| [5] | Ozdemir and Li, [Bridging the Reliability Gap: INT8 Quantization Effects on Discrimination and Calibration in Medical Imaging](https://proceedings.mlr.press/v333/ozdemir26a.html), CHIL 2026 | [PDF](C:/project/quant_analysis/paper_draft/pdfs/05_Ozdemir_2026_Bridging_Reliability_Gap.pdf) | 不同 dataset／backbone 的量化差異。 |
| [6] | Liu et al., [Understanding the Unfairness in Network Quantization](https://proceedings.mlr.press/v267/liu25au.html), ICML 2025 | [PDF](C:/project/quant_analysis/paper_draft/pdfs/06_Liu_2025_Unfairness_Network_Quantization.pdf) | 類別／群體的不均等退化。 |
| [7] | Hu et al., [Towards Understanding Model Quantization for Reliable Deep Neural Network Deployment](https://doi.org/10.1109/CAIN58948.2023.00015), CAIN 2023 | [PDF](C:/project/quant_analysis/paper_draft/pdfs/07_Hu_2023_Reliable_Quantization_Deployment.pdf) | Distribution shift、output margin 與 disagreements。原文的機率 margin 不等於本研究的 normalized logit margin。 |
| [8] | Yu et al., [Benchmark Platform for Ultra-Fine-Grained Visual Categorization Beyond Human Performance](https://openaccess.thecvf.com/content/ICCV2021/papers/Yu_Benchmark_Platform_for_Ultra-Fine-Grained_Visual_Categorization_Beyond_Human_Performance_ICCV_2021_paper.pdf), ICCV 2021 | [PDF](C:/project/quant_analysis/paper_draft/pdfs/08_Yu_2021_Ultra_Fine_Grained_Benchmark.pdf) | UFGIR 定義與 cultivar-level 判別挑戰。 |
| [9] | Kornblith et al., [Similarity of Neural Network Representations Revisited](https://proceedings.mlr.press/v97/kornblith19a.html), ICML 2019 | [PDF](C:/project/quant_analysis/paper_draft/pdfs/15_Kornblith_2019_CKA.pdf) | Centered linear CKA 的方法依據。 |
| [10] | Xia et al., [An Underexplored Dilemma between Confidence and Calibration in Quantized Neural Networks](https://arxiv.org/abs/2111.08163), NeurIPS 2021 workshop | [PDF](C:/project/quant_analysis/paper_draft/pdfs/10_Xia_2021_Confidence_Calibration_Quantization.pdf) | 區分 prediction swap 與 accuracy consequence 的動機。 |
| [11] | van der Maaten and Hinton, [Visualizing Data using t-SNE](https://www.jmlr.org/papers/v9/vandermaaten08a.html), JMLR 2008 | [PDF](C:/project/quant_analysis/paper_draft/pdfs/16_van_der_Maaten_2008_tSNE.pdf) | t-SNE 方法；可移至 Methods／圖說。 |
| [12] | Fang et al., [Learning Contrastive Self-Distillation for Ultra-Fine-Grained Visual Categorization Targeting Limited Samples](https://arxiv.org/abs/2311.06056), TCSVT 2024 | [PDF](C:/project/quant_analysis/paper_draft/pdfs/09_Fang_2024_CSDNet_Ultra_FGVC.pdf) | Subcategory-specific discriminative information 與 limited samples。 |
| [13] | Kim et al., SynQ: Accurate Zero-shot Quantization by Synthesis-aware Fine-tuning, ICLR 2025 | [PDF](C:/project/quant_analysis/paper_draft/pdfs/11_Kim_2025_SynQ.pdf) | 合成資料與量化後微調；不能支持完全不需訓練。 |
| [14] | Yuan et al., [Benchmarking the Reliability of Post-training Quantization: a Particular Focus on Worst-case Performance](https://arxiv.org/abs/2303.13003), 2023 preprint | [PDF](C:/project/quant_analysis/paper_draft/pdfs/14_Yuan_2023_PTQ_Worst_Case_Reliability.pdf) | Calibration distribution 與 worst-case group reliability。 |

特別注意：[4] FDDA 的 fine-grained 指 synthetic distribution alignment，不能直接當成 FGIR/UFGIR 實驗證據；Hu 等文獻的 probability/output margin 不自動等於本研究 normalized logit margin；CKA 是 representation 方法依據；t-SNE 不取代定量驗證。不要將 SynQ 等帶 fine-tuning 的方法當成「PTQ 一律不需要任何訓練／優化」的證據。HQQ 的實際 method/source citation 仍需補齊。

## 附錄 C：Evidence ledger 與本文件來源

| ID | Source file | SHA-256 | 用途 |
| --- | --- | --- | --- |
| E01 | `C:/iCloudDrive/project_2/paper/quant-dataset robustness/data/audit/20261116_140853_554056_runall_s20004_audit/verification.json` | `f49f11b1d41eb9a6a62a00e6dfd6930d1913d487f200cdb97786451c93c95a0d` | 下載結果完整性／數值 audit |
| E02 | `C:/iCloudDrive/project_2/paper/quant-dataset robustness/data/derived/20261116_140853_554056_runall_s20004_audit/dataset_master.csv` | `efea423202a1b20d4d06f45d273e111a53632e51d9c559ae2f226de667248807` | 144 列整合指標 |
| E03 | `C:/iCloudDrive/project_2/paper/quant-dataset robustness/data/derived/20261116_140853_554056_runall_s20004_audit/model_mode_summary.csv` | `f671a0c67b8fb3c87236b6c377a76f3359f573d5a6c196a55bcd3c6a90aca181` | 9 組 model/mode 描述性摘要 |
| E04 | `C:/iCloudDrive/project_2/paper/quant-dataset robustness/data/derived/20261116_140853_554056_runall_s20004_audit/margin_primary_summary.csv` | `ca5451bf6aad233d430657eded57ddf4ef01c85f0c09f4ee6fb57dc8663559fc` | 主 bin width matching gaps |
| E05 | `C:/iCloudDrive/project_2/paper/quant-dataset robustness/data/derived/20261116_140853_554056_runall_s20004_audit/matching_support_summary.csv` | `ba31c519a3d18451b99ff022f626fb717f722bcf60f876506c720bbc54e74d4f` | ESS、support 與 weights |
| E06 | `C:/iCloudDrive/project_2/paper/quant-dataset robustness/data/derived/20261116_140853_554056_runall_s20004_audit/undefined_auroc.csv` | `e1d8ba61428025b5e2e37eff0ead4e746d9fe7d67254bbf4f649fb29be990bef` | 8 列 undefined scores |
| E07 | `C:/iCloudDrive/project_2/paper/quant-dataset robustness/data/run_all/20261115_211022_759540_w3_s20004/run.json` | `08f08aa752c85d338e56554a834140c8c1e538d50dde8706bd030ee3933ee988` | Server scope、stages、時間與來源 |
| E08 | `C:/iCloudDrive/project_2/paper/quant-dataset robustness/data/run_all/20261115_211022_759540_w3_s20004/paired/analysis_manifest.json` | `6467dfdb3e3f0f6a804de5a822aa3edf6dd015112996dc00e2925dd3859460c6` | Paired source schema/provenance |
| E09 | `C:/project/quant_analysis/quant_acc/result/20261115_weight_only_s10000_10016_audit/run.json` | `382f6c876d1e9cd238b426a60cfa419d6d0fd1c20936e0f13ef246fc26a5187d` | 歷史 full-grid audit，不是即時狀態 |
| E10 | `C:/project/quant_analysis/dataset_static_feature/verify.py` | `1639e8a696a36ff64ef5ef193b0943d5271503bab79cfd799defaa5435cfa0ab` | Static 舊固定資料契約 |
| E11 | `C:/project/quant_analysis/paper_draft/references.json` | `5a9642bdbd15c268e16861733daabe8ff03a3819c42fd6ddd1087b150d25eb1d` | 既有 bibliography／PDF 身分 |
| E12 | `C:/project/quant_analysis/paper_draft/INTRODUCTION.md` | `9da94e8a5351ddf69a88bda4b54541b380593644b03ba02bd62065565cad9cf5` | 既有英文 intro 與 citation map |
| E13 | `C:/iCloudDrive/project_2/paper/quant-dataset robustness/data/audit/20261116_140853_554056_runall_s20004_audit/COPY_RECEIPT.json` | `9fe182fe4a5e8de6adcf78c1a3201e58c2400ccde02da220428c5f3fecc077ea` | 論文資料複製驗收 |

歷史進度與本次已完成產物衝突時，以原始 run/audit receipts 與實際資料為準。整份交接不修改原始實驗數據，也未新增 inference、full-grid 或 static-feature 實驗。

本次交接另核對 references.json 列出的 16 份本機 PDFs：檔案均存在且 SHA-256 相符；此檢查不取代逐項文獻內容與書目核對。
