# 20004：16-dataset 全層 feature 分析計畫

規劃日期：2026-10-07。這是執行前的分析設計，尚未對 20004 計算結果。

## 目前可確認的範圍

- 使用者已完成 16 個 datasets 的 inference，預計把資料放在 `C:/project/quant_analysis/data/20004`；規劃時本機尚無此目錄。
- 尚未確認 16 個 dataset 名稱、model/checkpoint 種類、seed、完整 quantization grid、feature shape、匯出格式。以下以既有 manifest 格式作為暫定輸入契約，不能視為已驗證。
- 本機 `five_dataset_features/analyze.py` 已逐層計算 centered linear CKA、class-equal cosine distance、feature norm；PCA 只處理最後位置。
- 舊程式固定 5 datasets、21 conditions、13 positions、768 dimensions；既有 distance/PCA/report/verifier 也依賴這些設定。不能只換 input path 就宣稱支援 16 datasets。
- 舊 TGDA `tools/five_dataset_hqq/run.py::feature_hooks` 擷取 `output[:, 0, :]`：index 0–11 是 block 1–12 的 CLS，index 12 是 `encoder_norm` 後的 CLS。不是 patch tokens、attention weights 或 embedding output。20004 必須依自己的 exporter 再確認一次。
- 本計畫延伸現有 owning module；原始輸入依使用者指定放在 repository 的 `data/20004`，新結果放在 `five_dataset_features/results/20004`。既有工作目錄有多項未提交變更，實作時需保留。

## 研究問題與主要比較

| 圖片中的需求 | 可檢驗的問題 | 主要結果 |
| --- | --- | --- |
| FP / quantized inter、intra 差異 | 同一 dataset/checkpoint 的量化是否伴隨類間分離下降？由 inter 還是 intra 改變？ | 每層 FP/Q distance、signed delta、gap deterioration |
| dataset 越難差異越大？ | 預先定義的 FP 任務難度，是否與固定條件的 geometry deterioration 有關？ | 16 點 scatter、Spearman、逐 dataset 排除敏感性 |
| UFGIR 是否有較高 CKA？ | 固定條件下，UFGIR 與 FGIR 的 FP→Q 同層 CKA 是否不同？ | dataset 曲線、group 差異與各組 dataset 數 |
| CKA 與 acc ratio 關係很小？ | CKA 和 accuracy retention 的關聯方向、大小與不確定性是多少？ | 固定條件相關、CI、對 absolute accuracy drop 的敏感性 |
| 哪一層 CKA drop 最大？ | 哪個輸出位置的 FP→Q 差異最大？哪一步的 CKA 曲線下降最大？ | 兩種不同定義的 layer 排名 |
| 每層 PCA | FP/Q 在共同投影下如何移動？投影捕捉多少 variance 與位移？ | 全層 atlas、explained variance、全維度位移 |

先列出假設再看結果。不能把「CKA 與 accuracy 無關」作為一定要得到的結論。

### 比較契約

1. 比較單位是 `dataset × model/checkpoint × training seed × quantization condition × feature position`；FP/Q 必須共用同一 checkpoint、test split、preprocessing、樣本與 inference precision 設定。
2. `fp` 的實際意思依 metadata 記錄，可能是 AMP-FP16 unquantized，不能自動寫成 FP32。
3. 跨 dataset 先固定 model/head context 與 quantization condition。若有多個 models、ft/fz/cal 或 seeds，先分開報告；不要把所有列當作獨立 datasets。
4. 若 20004 包含 W3/G128 weight-only，建議用它作主要條件，W4/G128 weight-only 作敏感性；W3/W4 × G8/G128 若齊全再檢查方向一致性。不存在的條件不補值，正式 config 在看分析結果前固定主要條件。
5. Linear A8、Linear A4、Linear+QKV 分 family 報告。所有既有條件都可計算，但避免讓嚴重 A4 degradation 主導跨條件相關。
6. 每個固定條件最多有 16 個 dataset-context 觀察；數百個 condition/layer rows 是重複測量。各 dataset 等權，不按圖片數加權。
7. 若使用原有 21 conditions 與 13 positions，單一 model/seed 會有 336 個 condition summaries、4,368 個 layer rows；這只是條件成立時的 coverage 檢查，實際以 20004 inventory 為準。

## 第一階段：inventory 與輸入驗證

暫定可接受的原始結構：

```text
C:/project/quant_analysis/data/20004/
  <dataset>/
    dataset.json
    fp/
      features.npy
      predictions.csv
      manifest.json
      precision_probe.npy       # 若原始匯出有提供
    <quantization_condition>/
      features.npy
      predictions.csv
      manifest.json
      precision_probe.npy
```

若是 `features_dict_test.pth` 或有 model/head 子目錄，先寫 adapter 轉成一致的分析介面並記錄來源；不改寫原檔、不猜測 layer 對應。沒有 FP 或無法配對的資料，不能做主要 FP/Q 比較。

輸出 `inventory.csv`、`coverage.csv`、`validation.json`，驗證：

- dataset/model/head/seed/checkpoint SHA-256、condition 的實際 quantization 設定與 exporter 版本。
- FP/Q 的 sample IDs 唯一、集合與 targets 一致；順序不同只能以明確 IDs 重新對齊並記錄，不能按列號硬配。
- feature shape、dtype、finite values、zero-norm counts、每個位置的 tensor 語義。以 manifest/adapter 定義 shape，不硬編碼 13/768。
- 依 manifest 核對 raw file SHA-256；checkpoint hash 使用匯出紀錄，不能把「有 hash」寫成已重新讀取 checkpoint 驗證。
- `correct` 與 `prediction == target` 一致；從 matched predictions 重算 accuracy。檢查 class counts、singleton classes、缺失 conditions。
- 如有 precision probe，比較 FP32-valued activation 與 FP16 serialization 的 metric 差異；此 probe 不等於 FP32 inference baseline。沒有 probe 就記錄 storage sensitivity 未驗證。
- 缺失或不相容條件列入 exclusion audit；正式跨 dataset 比較採明確的 common cohort，報告每個比較的有效 dataset 數。
- 沿用 FGIR/UFGIR mapping 時核對 16 個名稱與分組來源；既有 `granularity_robustness/data/source/dataset_granularity_mapping.csv` 只有 15 個名稱，且有 boundary datasets，不能直接假定覆蓋 20004 或全部屬於兩個核心組別。

## 第二階段：accuracy 與共用的每層表

`condition_summary.csv` 保留 FP accuracy、Q accuracy、accuracy ratio、absolute drop、swap rate、C→W、W→C：

```text
accuracy_ratio = Q_accuracy / FP_accuracy
relative_loss = 1 - accuracy_ratio
absolute_drop_pp = 100 * (FP_accuracy - Q_accuracy)
```

FP accuracy 為零時 ratio 未定義；Q 比 FP 好時保留 ratio > 1 和負 drop，不 clipping。

`layer_metrics.csv` 的唯一鍵：

```text
dataset, model, checkpoint_id, seed, condition, feature_position
```

記錄 `position_name`、N、D、CKA、inter、intra、gap、FP/Q deltas、norm、class coverage，以及結果所屬 snapshot。Accuracy 是 condition 層級量，join 至每層只為畫圖，不代表有 per-layer classifier accuracy。

## 第三階段：inter / intra class distance

### 定義與 FP/Q 差異

主分析使用 ground-truth classes、每個樣本 L2-normalized CLS、cosine distance `1 - cosine_similarity`。

- `D_intra`：每個 class 的不同樣本 pair 先平均，再對 classes 等權平均。至少兩個樣本才可計算；保留有效 class 數。
- `D_inter`：每對不同 classes 的 cross-sample distances 先平均，再對 class pairs 等權平均。這不是 normalized-centroid distance。
- `gap = D_inter - D_intra`：類間與類內 distance 的差距，主要 geometry 指標。

各層保留：

```text
delta_inter = Q_inter - FP_inter
delta_intra = Q_intra - FP_intra
delta_gap = Q_gap - FP_gap
gap_deterioration = FP_gap - Q_gap
gap_drop_fraction = 1 - Q_gap / FP_gap
distance_ratio = D_inter / D_intra
```

主要用 signed gap deterioration 區分惡化與改善；`abs(delta_inter/intra/gap)` 回答「變動有多大」作補充，不能混為同一個方向假設。

FP gap 非正或接近零時，relative gap drop 不作主要解讀，保留 absolute change 與 missing reason；實作須記錄除法的數值容差。比率不得因結果難看而 clipping。Distance ratio 是定義敏感性，不取代 gap 主結果。

### 「dataset 越難」的定義

主 difficulty proxy 暫定 `1 - FP_accuracy`，表示這個 dataset-checkpoint 在此 test split 上的基線任務難度。FGIR/UFGIR 表示標籤粒度類型，是另一個分組問題，不能直接等同難度高低。

主要比較：固定主要 quantization condition，以 16 個 dataset-context 的 FP difficulty 對 final-position gap deterioration 作 Spearman。另看 relative gap drop、inter/intra 各自改變，以及 group 的差異。

補充比較：

- FP gap 對 absolute/relative accuracy loss，回答「量化前 separability 是否與 accuracy retention 有關」。
- Gap deterioration 對 accuracy loss，回答「幾何變化是否伴隨效能下降」。
- 全層相關曲線作探索；最後位置事先定為主要結果，避免事後只挑最好看的 layer。

Difficulty 與 accuracy ratio 共用 FP accuracy，可能有代數上的耦合；必須並列 Q accuracy、FP accuracy、absolute drop。不同 checkpoint、class count、sample size 也限制跨 dataset 解釋。

主要圖：每個 dataset 的 FP/Q distance layer curves、16-dataset final-gap paired plot、difficulty 對 gap deterioration scatter、gap deterioration 對 accuracy loss scatter。

## 第四階段：CKA

### 同層 FP→Q 比較

每個 dataset、condition、position，對 matched samples 的 FP CLS 與 Q CLS 計算 centered linear CKA。主分析延續現有實作：center 原始 feature vectors，不先做逐樣本 L2 normalization。

CKA 描述樣本表徵結構相似度；不直接等於 feature L2 difference 或 classifier decision retention。[CKA 原始論文](https://proceedings.mlr.press/v97/kornblith19a.html) 提供指標定義與表示比較背景。

輸出逐 dataset 的 layer curves，以及每個主要條件的 `dataset × feature position` heatmap。Final encoder_norm CKA 作主要摘要；block outputs 的 depth mean 為次要摘要，不把最後 norm 當額外 block 加入 block mean。

### UFGIR vs FGIR

- 固定 condition/model/head，先畫所有 datasets 的曲線與 final CKA，分組後報告 median difference、各組 N 和分布。
- Core mapping 作主要 group 比較；boundary datasets 另外呈現，expanded mapping 作敏感性。
- 做跨 dataset common-N、matched class-balanced subsampling 的 CKA 敏感性；FP/Q 使用同一批 indices，重複固定 seeds，報告實際 N 與抽樣 variation。小樣本資料限制也需列出。
- 如有多個 soybean datasets，增加整個 soybean family 排除或 family-level aggregation 的敏感性。它們不應被當作完全獨立 benchmark 的強力重複驗證。
- 若做 group-label permutation，需說明在當前 dataset selection 下的 exchangeability 假設；分組是 benchmark 屬性，結果屬於有限集合的探索性比較。

### CKA vs accuracy ratio

1. **主要跨 dataset 比較**：固定主要 condition，16 點 final CKA 對 accuracy ratio，報告 rho、有效 N、scatter、dataset bootstrap CI。
2. **敏感性**：換成 absolute drop、其他 weight-only conditions、leave-one-dataset-out。選擇性加 FP accuracy 與 log class count 的簡單調整只能當探索；16 datasets 不適合大量 covariates。
3. **同 dataset 跨 conditions**：另外列出，依 weight-only/A8/A4/QKV family 分開。不能把各 families 混在一起的高相關寫成跨 dataset 預測能力。
4. 若 formal correlation test，採 dataset-unit permutation，置換重複條件時整個 dataset block 同步；多 layer/condition tests 作預先定義的 FDR family。SciPy 明確建議小樣本採 permutation 而非只依賴 `spearmanr` 的 asymptotic p-value。[SciPy 文件](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)
5. `p > .05` 只能說證據不足以拒絕零關聯，不能證明關係小。若要支持「實務上很小」，先固定可接受的 rho 範圍（例如 ±0.30，僅是待約定例子），再看 CI 是否足以排除更大關係；CI 很寬時回報 inconclusive。

### 哪一層 drop 最大

分別計算，不能混稱為同一種 drop：

```text
representation_loss[position] = 1 - CKA(FP_position, Q_position)
largest_FP_Q_difference = argmax representation_loss

curve_step_drop[position] = CKA[position-1] - CKA[position]
largest_curve_drop = argmax curve_step_drop among positive values
```

第一個指出哪個位置的 FP/Q 差異最大；第二個指出曲線在哪一步下降最大。沒有正下降就報告 none；相同值保留 ties。Block12→encoder_norm 另列。

量化是對多層同時套用，這兩個指標都包含上游累積變化，不能證明某個 block 是 accuracy loss 的原因。做因果 layer sensitivity 需要另行 layer-isolated quantization inference，不屬於這批既有 features 可直接回答的範圍。

## 第五階段：每層 paired PCA

對每個 `dataset × model/checkpoint × position`，用完整 FP test features fit 一個 PCA basis，保存 FP mean、components、explained variance。FP 與所有 Q conditions 共用它：

```text
Z_FP = (X_FP - mean_FP) @ components_FP
Z_Q = (X_Q - mean_FP) @ components_FP
```

主分析使用 centered raw CLS features，與舊 PCA 一致；normalized-feature PCA 可作 distance-analysis 的補充但需另外標示。

- 同一 dataset/position 的各 conditions 共用色彩、選樣 IDs 與 axis limits；不能各自 fit PCA、各自 center Q 後再直接比較位置。
- 不同 layers/datasets 的 PCA basis 不同，PC1/PC2 方向與距離不能直接橫向等同。
- 全測試集用於計算。類別很多時圖用固定、class-balanced representative subset；caption 標示顯示 N，避免用代表圖片替代統計母體。
- 優先輸出 FP、主要 weight-only、weight-only sensitivity 的全層 atlas；其他 conditions 也保存全層 coordinates/bases，可由 condition selector 或分頁 atlas 查閱，避免一次產生數千張單圖。
- 類別色彩用於看 class structure；另做 C→C、C→W、W→C、W→W 配色，以及固定樣本的 FP→Q arrows。誤差群體的 layer comparison 是回溯描述，仍非原因定位。

每張圖標示 PC1、PC2 explained variance；同步輸出：

```text
full_displacement = ||X_Q - X_FP||_2
relative_displacement = ||X_Q - X_FP||_2 / ||X_FP||_2
projected_energy_fraction = ||(X_Q-X_FP) @ components_FP||_2^2
                            / ||X_Q-X_FP||_2^2
```

報告各 outcome 群的 median/P90 relative displacement。分母為零時保留 NA 與 reason。PCA 是視覺摘要；「2D 看起來差不多」不能替代 full-dimensional 指標，也不能證明分類資訊沒有改變。

## 不確定性、限制與驗收

- Dataset-level bootstrap/permutation/leave-one-out 以 dataset 為單位，重複條件一起移動。這些只描述目前 benchmark set；若 family dependence 明顯，另用 family 敏感性並降低推論強度。
- 單一 dataset 的 FP/Q metric CI，若需要，採 class-stratified paired sample bootstrap：同一 replicate 的 FP/Q、所有 positions、predictions 使用相同 IDs/indices。這是固定 checkpoint 的 test-sampling uncertainty，不是 training-seed variation。
- Spearman ties、constant inputs、singleton class、zero-norm、missing condition、FP accuracy/gap 近零等皆明確標示；未定義值不能填 0。
- Per-layer 探索要保留全部結果，不能只報顯著位置。正式主結果先固定，追加分析在 report 標成 exploratory。
- 只有 CLS features 可回答 CLS geometry；不能因此宣稱分析過 patch tokens、Q/K/V 或 attention matrices。若 exporter 提供其他 tensor，另建立匹配的分析契約。
- 無需重新 training/inference；不下載 W&B。若必要輸入缺失，先回報 coverage 與可完成部分，是否新增 inference 是後續決策。

驗收以預期 inventory 產生的完整 coverage 為準：所有有效 dataset/condition/position 恰好一列；FP self-CKA 在非退化輸入接近 1；FP distance deltas 為 0；accuracy summaries 能從 predictions 重算；同一 position 的 FP/Q 共用 PCA basis；hash、來源與 code version 能重現結果。

## 實作與執行順序

建議延伸 `five_dataset_features`，將舊的 5-dataset defaults 保留，為 20004 新增 config 驅動的 cohort、conditions、layer map、input/output root；不要只改 `DATASETS` 常數而遺漏其他 report/verifier 的假設。

規劃的 stage interface（尚未實作，不是目前可直接執行的命令）：

```text
inventory → validate → metrics → comparisons → pca → plot → verify/report
```

| Stage | 工作 | 驗收輸出 |
| --- | --- | --- |
| inventory | 掃描 20004，再凍結預期 dataset/context/condition 集合 | inventory.csv、coverage.csv、run_config.json |
| validate | hashes、schema、sample pairing、feature positions | validation.json、exclusions.csv |
| metrics | 完整每層 CKA/distance 與 condition predictions | layer_metrics.csv、condition_summary.csv |
| comparisons | 固定條件 distance/difficulty、groups、CKA/accuracy、layer drop | comparisons.csv、group_comparisons.csv、layer_drop_summary.csv |
| pca | 每個 dataset/position 的 FP basis、全條件配對投影 | pca/bases、pca/coords、pca_summary.csv |
| plot | 先主圖，再全層 atlas 與 conditions 補充圖 | figures/*.png、*.svg、pca atlas |
| verify/report | coverage、output hashes、snapshot、中文結果解釋 | analysis_snapshot.json、REPORT.md |

結果目錄：

```text
C:/project/quant_analysis/five_dataset_features/results/20004/
  run_config.json
  inventory.csv
  coverage.csv
  validation.json
  exclusions.csv
  condition_summary.csv
  layer_metrics.csv
  comparisons.csv
  group_comparisons.csv
  layer_drop_summary.csv
  pca_summary.csv
  pca/
  figures/
  analysis_snapshot.json
  REPORT.md
```

初次 validation 對 raw inputs 計算 hashes 建立 snapshot；後續 stages 檢查 source/code/derived identities，正式 verify 再核對 raw hashes，不能只從未變的 manifest 推定原始 arrays 未變。來源或分析規則變更時失效並列出需要重跑的 stages，不能混用舊五-dataset CSV 與新 20004 features。既有 snapshot 格式對 code/path 有假設，實作需一併參數化。

用 memory-mapped features 與 chunked statistics，每次處理一個 dataset/condition；CKA、PCA 使用 D×D accumulators，不建立 N×N sample matrix。PCA basis 每個 dataset/position 只 fit 一次，再投影全部 conditions。大型 class counts 的 inter distance 也可用 sums 避免完整 class-pair matrix。時間與磁碟估算等 inventory 得到真實 N/D/conditions 後再提供。

實作時所有 Python 命令使用 `conda run -n opencode_env python ...`。必要驗證測試涵蓋 schema/pairing、非等量 classes 的 distance、duplicate/missing rows、layer names、共用 PCA projection、重複測量 inference 與 snapshot rejection；低影響排版調整不另加測試。

先完成 inventory/validate，再以一個 dataset 跑通 metrics/PCA/verify，確認正式 config 後跑全部 16 datasets。最終報告對每個假設列出 supported / unsupported / inconclusive，附有效 cohort、效果量與敏感性，不把探索關聯寫成原因。
