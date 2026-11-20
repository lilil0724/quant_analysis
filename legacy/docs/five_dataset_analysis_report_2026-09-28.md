# Five-dataset HQQ 特徵與預測分析報告

分析日期：2026-09-28。此報告解讀本機 five_dataset_features 全測試集輸出；數字取自當日已驗證的分析檔。它是描述性實驗分析，不包含新模型訓練、速度測試或因果推論。

## 1. 實驗範圍

五個資料集各有一個資料集專屬的 fine-tuned ViT-B/16 checkpoint。每組含 1 個未量化的 AMP-FP16 基準，以及四種 weight 設定（W3/W4 × group 8/128）各自的五個條件：weight-only、加 Linear A8、加 Linear A8 與 QKV A8、加 Linear A4、加 Linear A4 與 QKV A4。因此每組 21 個條件，總計 105 個。Linear 與 QKV activation 使用 fake quantization；classification head 不在 HQQ weight quantization 範圍內。FP 基準是 AMP-FP16，不是 FP32。

| Dataset | 測試影像數 | 類別數 | FP accuracy |
|---|---:|---:|---:|
| cotton | 240 | 80 | 50.42% |
| soyageing | 4,950 | 198 | 37.25% |
| soyglobal | 5,814 | 1,938 | 18.70% |
| cub | 5,794 | 200 | 87.83% |
| pets | 3,669 | 37 | 93.84% |

五個 checkpoint、資料集與類別數均不同；跨資料集統計的單位是五個 dataset–model pair。每組的 20 個量化條件是重複量測，不能視為 100 個獨立樣本。

## 2. 資料驗證與指標

既有分析器對五組各 21 個條件逐一驗證 completion manifest、identity、檔案 SHA-256、特徵陣列的 [N, 13, 768] 形狀與 dtype，以及預測的 sample ID 順序和 target。五組均完成分析。13 個位置是 12 個 Transformer block 輸出與 final normalized CLS feature。13 個 five_dataset_features 單元測試通過。精簡 ZIP 有 278 個項目；其中 277 個內容檔的 checksum 均與 ZIP 內清單吻合，ZIP 測試未發現損壞，且不含完整 features.npy。

預測指標的定義如下：

- Accuracy drop = FP accuracy − 條件 accuracy；本報告將其乘以 100 表示為百分點。
- Relative loss = 1 − 條件 accuracy / FP accuracy。負值表示條件準確率略高於 FP。
- Swap rate = 與 FP 的 top-1 預測標籤不同的影像比例。
- Damage / rescue = FP 正確而量化後錯誤 / FP 錯誤而量化後正確的影像數。
- Net damage R = (damage − rescue) / swap 數；沒有 swap 時不定義。
- Normalized margin Q10 = 每張影像的 top-1 與 top-2 logit 差除以該張影像的 logit 標準差，再取第 10 百分位。

這些直接預測結果見[105 條件彙總表](../five_dataset_features/analysis/all_conditions.csv)。下表的家族平均是在每個資料集內對四種 W/G 設定等權平均，並非將不同影像數的資料集混成一個樣本池。

## 3. 準確率與逐步加入量化

各數字為相對 FP 的平均 relative loss：

| Dataset | Weight-only | Linear A8 | Linear A8+QKV | Linear A4 | Linear A4+QKV |
|---|---:|---:|---:|---:|---:|
| cotton | 7.02% | 5.99% | 5.58% | 73.55% | 73.55% |
| soyageing | 9.46% | 9.80% | 9.96% | 84.77% | 84.77% |
| soyglobal | 9.50% | 9.38% | 9.80% | 68.03% | 67.32% |
| cub | 1.44% | 1.60% | 1.58% | 83.85% | 84.13% |
| pets | 0.53% | 0.55% | 0.55% | 56.61% | 57.13% |

為了分開量化步驟，下一表在相同 W/G 設定內計算後一條件減前一條件的 accuracy，再對四種 W/G 等權平均。單位為百分點；正值代表準確率上升。

| Dataset | Weight → A8 | A8 → A8+QKV | Weight → A4 | A4 → A4+QKV |
|---|---:|---:|---:|---:|
| cotton | +0.52 | +0.21 | −33.54 | 0.00 |
| soyageing | −0.13 | −0.06 | −28.06 | 0.00 |
| soyglobal | +0.02 | −0.08 | −10.94 | +0.13 |
| cub | −0.14 | +0.01 | −72.39 | −0.25 |
| pets | −0.02 | 0.00 | −52.62 | −0.49 |

最穩定的結果是 Linear A4 在五組、四種 W/G 設定下都大幅降低準確率。A8 的額外效果小得多；QKV 的額外準確率效果沒有一致方向。Weight-only W3/group 128 在五組都比 W3/group 8 低 1.20–10.00 個百分點；W4 的對應差距只有 0.05–1.84 個百分點。這些是此批 checkpoint 與測試集上的成對觀察，不是所有模型的通則。

Swap 與 damage/rescue 有助於辨別「相同 accuracy 但預測改變」的情況。例如 cub 的 W3/group 128：weight-only accuracy 為 84.41%，swap rate 為 10.03%，damage/rescue 為 316/118；加入 Linear A4 後，accuracy 為 7.44%，swap rate 為 92.37%，damage/rescue 為 4,674/16，R 從 0.341 升到 0.870。相較之下，soyglobal FP accuracy 僅 18.70%，相對損失與絕對百分點降幅必須並列解讀，不能只看其中一項。

## 4. 特徵表示：CKA、類別分離、H-score 與 norm

CKA 在每個資料集內，以同一批、同順序影像的 FP 特徵 X 和量化特徵 Y，逐層計算 centered linear CKA。它比較整批特徵的線性幾何相似性，接近 1 表示相似；本分析沒有直接計算不同資料集影像彼此的 CKA。CKA 對等比例縮放和特徵維度的正交旋轉不敏感，固定的 classification head 卻可能改變輸出，因此高 CKA 不保證高 accuracy。

下表為相對各資料集自己 FP 的 final CLS CKA，對四種 W/G 等權平均；最後兩欄是基於真實類別的 class-equal separation。

| Dataset | Weight CKA | A8 CKA | A4 CKA | FP separation | A4 separation |
|---|---:|---:|---:|---:|---:|
| cotton | 0.959 | 0.958 | 0.577 | 0.277 | 0.062 |
| soyageing | 0.956 | 0.953 | 0.705 | 0.428 | 0.126 |
| soyglobal | 0.966 | 0.965 | 0.747 | 0.537 | 0.390 |
| cub | 0.919 | 0.916 | 0.083 | 0.681 | 0.051 |
| pets | 0.982 | 0.981 | 0.329 | 0.758 | 0.082 |

Class-equal separation 定義為類間平均 cosine distance 減類內平均 cosine distance。特徵先做 L2 normalization，再對每個類別及每對類別等權平均，避免大類別完全主導結果。它描述標籤在特徵空間的分離程度，不等於分類器 accuracy。

A8 的 final CKA 幾乎等於 weight-only，且 class separation 變化小。A4 的 CKA 與 separation 在五組皆下降，但兩者不是跨資料集的準確率預測器。例如 cub 的 final CKA 降至 0.083，並伴隨嚴重預測集中：四個 Linear A4 條件各有 39.6%–66.6% 的影像被預測成單一類別。soyglobal 的 A4 final CKA 仍為 0.747，accuracy 卻只剩約 5.40%–6.64%（FP 為 18.70%）。因此 CKA 要與預測轉換、separation 和 accuracy 合看。逐層數字與圖見[cub layer metrics](../five_dataset_features/analysis/cub/layer_metrics.csv)、[cub CKA heatmap](../five_dataset_features/analysis/cub/cka_heatmap.png)及[量化步驟圖](../five_dataset_features/analysis/cub/activation_transition.png)；另外四組在各自同名資料夾。

![CUB 的 accuracy、final CKA 與 class separation 隨量化步驟的變化](../five_dataset_features/analysis/cub/activation_transition.png)

Regularized H-score 在 final feature 的每個條件均有計算；逐層 H-score 只對 FP 計算。它使用 within-class scatter 的平均變異量之 1% 作 ridge。每層另外儲存未正規化特徵長度的 P50、P95、P99 與 P99/P50。這些量能協助檢查表徵改變，但各資料集的類別數、樣本數與 FP 品質相差甚大；尤其 cotton 只有 240 張、80 類且特徵寬度為 768，H-score 絕對值高度依賴 ridge，不宜直接把跨資料集 H-score 大小解讀為難度排名。

## 5. PCA 視覺化與 FP16 儲存 probe

每個資料集只用 FP final feature 擬合一次 2D PCA，然後將該資料集全部 21 條件投到同一基底；這讓同資料集的位移可視化，但 PCA 是投影圖，不能替代原始 768 維指標。FP 前兩主成分合計解釋變異約為 cotton 16.0%、soyageing 30.1%、soyglobal 36.6%、cub 25.4%、pets 13.0%。完整測試影像的座標保存在各資料集的 coords 檔；ZIP 只保留每組 20 個代表樣本的座標與影像。參見[cub PCA 圖](../five_dataset_features/analysis/cub/pca_distribution_correct_swap.png)。

每組第一個真實 batch 的 32 張影像另存 FP32 precision probe。依 smoke 階段同一判準，對四種 W/G 的 Weight → A8、A8 → A8+QKV、Weight → A4、A4 → A4+QKV，每步的 13 個位置比較「activation 引起的 CKA 改變」與「同一對特徵轉存 FP16 引起的 CKA 差異」。共 5 × 4 × 4 × 13 = 1,040 個增量檢查，全部超過 10 倍；A8 相關比較的最小比值是 soyglobal 的 57.5 倍。這支持 FP16 儲存足以在該批 probe 分辨這些 activation 效果，但不是完整測試集上每一項指標的誤差上界。各組 FP 對條件的 probe 結果另見各資料集的 precision_probe.csv。

## 6. 跨資料集候選指標

跨資料集步驟先從每組未量化 FP final feature 取候選指標：FP accuracy、class-equal separation、類內與類間距離、regularized H-score、norm P50/P95/P99/P99/P50，以及 normalized margin Q10。對每個量化家族，先在資料集內平均四種 W/G 的 relative loss，再用五個 dataset–model pair 計算描述性 Spearman ρ。量化後才會產生的 CKA 不在這張「量化前候選 predictor」表內。

| FP 候選指標 vs relative loss | Weight-only ρ | Linear A8 ρ | Linear A4 ρ |
|---|---:|---:|---:|
| FP accuracy | −1.0 | −0.9 | −0.3 |
| FP normalized margin Q10 | −1.0 | −0.9 | −0.3 |
| FP separation | −0.6 | −0.7 | −0.5 |
| FP H-score | −0.5 | −0.6 | −0.2 |
| FP norm P99/P50 | +0.2 | +0.1 | +0.3 |

QKV A8/A4 的對應係數和 Linear A8/A4 非常接近；其餘指標與全部 50 個係數見[完整 Spearman 表](../five_dataset_features/analysis/descriptive_spearman.csv)，FP 數值見[FP indicators](../five_dataset_features/analysis/fp_indicators.csv)及[五點散佈圖](../five_dataset_features/analysis/fp_predictors_vs_loss.png)。

Weight-only 的 −1.0 是五點完全反向排序，不是可泛化的效應量。FP accuracy 與 Q10 在此五組恰好完全同序，無法分辨兩者的獨立作用；dataset、checkpoint、類別數與 FP 表現也同時改變。A4 下，這些排序關係明顯變弱。流程未把四種 W/G 當作獨立資料點，也未宣稱 p 值、統計顯著性、外部驗證或因果機制。

## 7. 目前結論與未解限制

1. **直接實測：**Linear A4 對五組的 accuracy、swap 與 class separation 造成大幅影響；A8 的額外 accuracy 效果小。W3/group 128 比 W3/group 8 更脆弱的方向在五組一致。
2. **表徵線索：**A4 通常伴隨較低的 CKA 與 separation；但跨資料集 CKA 高低沒有穩定對應到 relative loss，高 CKA 不能替代 accuracy。
3. **候選關聯：**FP accuracy 與 margin Q10 能完美排序這五組的 weight-only relative loss，卻因兩者同序與資料集／checkpoint 混雜，不能判定哪個指標獨立解釋脆弱性。
4. **精度 probe：**首批 32 張影像的量化特徵改變明顯大於 FP16 儲存造成的 CKA 差異；完整資料的數值精度充分性仍需以目標指標判斷。
5. **Provenance：**五組 dataset.json 目前都記錄 quant_zero=true 與 quant_scale=true；先前的 [CUB smoke 報告](../five_dataset_features/analysis/cub_smoke_report.md)指出 runner 應記錄修正後的有效設定為 false。應向 nano5 的實際 runner／HQQ 記錄核對後，再將此批結果用於正式論文主張。此 metadata 疑點本身不等於已證明數值無效。
6. **設計邊界：**每組只有一個 dataset-specific checkpoint 與 seed 100；cotton 測試影像較少，soyglobal 的 FP accuracy 較低。Activation fake quantization 不證明整數 kernel、延遲或吞吐量改善。

## 8. 檔案索引

- [分析方法與 CLI](../five_dataset_features/README.md)
- [全部 105 條件的預測指標](../five_dataset_features/analysis/all_conditions.csv)
- [量化前 FP 指標](../five_dataset_features/analysis/fp_indicators.csv)
- [跨資料集 Spearman](../five_dataset_features/analysis/descriptive_spearman.csv)
- [分析程式版本 checksum](../five_dataset_features/analysis/analysis_version.json)
- [五組分析來源與報告快照](../five_dataset_features/analysis/analysis_snapshot.json)
- [精簡、含 checksum 的 10.03 MB 結果 ZIP](../five_dataset_features/analysis/five_dataset_hqq_results.zip)

各資料集的 summary.csv、layer_metrics.csv、precision_probe.csv、四張圖與 PCA 座標，分別存於 five_dataset_features/analysis/cotton、soyageing、soyglobal、cub、pets。
