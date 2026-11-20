# Serial 20004：用 inter／intra-class distance 看量化損失

分析日期：2026-10-10。沿用完整 144 組 FP／weight-only W3/g128 的已下載結果，分 9 個 model/checkpoint 組、每組 16 datasets 計算。所有 144 個 layer_metrics.csv 的來源 checksum 已重新核對；未重算原始 features。

這批結果的重點：**FP 本身的類別可分離度比 Final CKA 更貼近量化損失，但 distance 的變化方向依模型／checkpoint 而異；不能假設一定是「intra 變大、inter 變小」，也不能把 separation 下降幅度當成 accuracy loss 的替代品。**

## 距離與 separation 的定義

- 距離為 `1 − cosine similarity`，每個 sample feature 先做 L2 normalization，並非未正規化的 Euclidean distance。
- `D_intra`：各類內不同 sample pair 的平均距離，再對有至少 2 個 samples 的類別等權平均。
- `D_inter`：每一對不同類別的所有 cross-class sample pairs 平均距離，再對不同 class pairs 等權平均。它不是「正規化後 class centroid 之間的距離」，也不是只看 nearest/hardest negative class。
- 既有 gap：`G = D_inter − D_intra`。較大的 FP gap 描述同類相對緊密、異類相對分離。
- 本次補算正規化 separation：`S = (D_inter − D_intra)/(D_inter + D_intra)`。它除去兩種距離同時按比例改變的影響，不能消除訓練／baseline 混雜。
- 也保留 `R = D_inter/D_intra`；當 intra>0 時，`S=(R−1)/(R+1)`，兩者 rank 相同，因此不是兩個獨立證據。此次所有 final distances 的 denominator 都有效，FP gap 最小約 0.000797，沒有把接近 0 的量直接當成可靠相對變化尺度。

## FP 可分離度能否描述量化敏感度？

下表為「FP feature 指標」與「相對準確率損失 1 − W3 acc / FP acc」的 Spearman。負值表示 FP 可分離度較大的 dataset，通常量化損失較小。所有比較均固定 model/mode，每組 n=16；不是把 144 cells 當成獨立樣本。

| Model | Mode | FP gap vs loss | FP S vs loss | FP S adjusted* | Final CKA vs loss |
|---|---|---|---|---|---|
| ViT-B/16 | ft | -0.838 | -0.838 | -0.281 | -0.171 |
| ViT-B/16 | fz | -0.847 | -0.788 | -0.725 | -0.538 |
| ViT-B/16 | cal | -0.085 | -0.468 | 0.420 | -0.182 |
| Swin-B | ft | -0.862 | -0.871 | -0.506 | -0.403 |
| Swin-B | fz | -0.821 | -0.665 | -0.444 | -0.265 |
| Swin-B | cal | 0.053 | -0.744 | -0.028 | -0.132 |
| BEiT-v2-B | ft | -0.565 | -0.515 | 0.104 | -0.282 |
| BEiT-v2-B | fz | -0.844 | -0.888 | -0.712 | -0.176 |
| BEiT-v2-B | cal | 0.091 | -0.671 | -0.346 | 0.009 |


*adjusted：將 FP S rank 與 relative-loss rank 各自對 intercept、FP accuracy rank 回歸，再求殘差相關。這裡只控制 FP accuracy；不和主報告控制 accuracy/classes/samples 的 Q10 partial correlation 直接當同一規格比較。

1. 在 FT／FZ，raw FP gap 多數有強烈負相關，例如 ViT FT −0.838、ViT FZ −0.847、Swin FT −0.862、BEiT FZ −0.844。
2. CAL 的 raw gap 相關接近 0，但正規化後 S 的相關更清楚：ViT −0.468、Swin −0.744、BEiT −0.671。這提示 raw gap 的絕對尺度可能掩蓋部分相對類別分離關係；S／R 是這批資料值得補看的描述性指標。這是同批資料上探索到的結果，沒有 held-out 證據。
3. 9 個組內，FP S 與 loss 的相關絕對值都高於 Final CKA。不過 FP S 是量化前的分類結構，CKA 是量化前後表示的相似度，兩者測量不同；不是驗證某個已訓練預測模型更好。
4. 控制 FP accuracy 後並非全部關係都維持。ViT CAL 的 FP S 相關從 −0.468 變 +0.420、Swin CAL 由 −0.744 變 −0.028，BEiT FT 由 −0.515 變 +0.104；FZ 則仍保留負相關。S 仍然受 baseline 混雜，不能直接宣稱是獨立量化難度因子。

![距離相關](C:/project/quant_analysis/five_dataset_features/analysis/serial_20004_20261010/class_distances/distance_correlations.png)

## FP 到 W3 的距離實際怎麼變？

以下是 16 datasets 等權平均的 final distances。S 是先對每個 dataset 計算，再平均，不是把表內平均 inter／intra 代回公式。

| Model | Mode | Intra FP → W3 | Inter FP → W3 | Gap FP → W3 | S FP → W3 | Gap 較小 datasets | S 較小 datasets |
|---|---|---|---|---|---|---|---|
| ViT-B/16 | ft | 0.4140 → 0.4478 | 0.9858 → 0.9397 | 0.5718 → 0.4920 | 0.422 → 0.362 | 16/16 | 16/16 |
| ViT-B/16 | fz | 0.4022 → 0.4200 | 0.6952 → 0.6788 | 0.2930 → 0.2588 | 0.251 → 0.221 | 15/16 | 14/16 |
| ViT-B/16 | cal | 0.1061 → 0.1095 | 0.2951 → 0.2865 | 0.1891 → 0.1770 | 0.404 → 0.374 | 11/16 | 15/16 |
| Swin-B | ft | 0.4775 → 0.4603 | 0.9569 → 0.8542 | 0.4794 → 0.3939 | 0.346 → 0.302 | 16/16 | 14/16 |
| Swin-B | fz | 0.3391 → 0.3567 | 0.5891 → 0.5911 | 0.2499 → 0.2344 | 0.251 → 0.232 | 15/16 | 14/16 |
| Swin-B | cal | 0.0747 → 0.0772 | 0.2011 → 0.1947 | 0.1264 → 0.1174 | 0.446 → 0.405 | 10/16 | 15/16 |
| BEiT-v2-B | ft | 0.4675 → 0.4846 | 0.9435 → 0.8181 | 0.4760 → 0.3335 | 0.344 → 0.261 | 14/16 | 12/16 |
| BEiT-v2-B | fz | 0.1599 → 0.3321 | 0.3138 → 0.3961 | 0.1539 → 0.0640 | 0.276 → 0.098 | 11/16 | 12/16 |
| BEiT-v2-B | cal | 0.0107 → 0.0170 | 0.0218 → 0.0291 | 0.0111 → 0.0122 | 0.369 → 0.288 | 5/16 | 14/16 |


- **ViT FT**：intra 平均 0.414→0.448、inter 0.986→0.940，gap 在 16/16 datasets 變小，符合典型的「類內散開、類間靠近」方向。
- **Swin FT**：intra 平均反而縮小（0.477→0.460），但 inter 縮得更多（0.957→0.854），因此 gap 仍在 16/16 datasets 變小。intra 變小本身不能當成量化變好的證據。
- **BEiT FZ**：16/16 datasets 的 intra 與 inter 都變大；intra 平均 0.160→0.332，inter 0.314→0.396，平均 gap 由 0.154→0.064、S 由 0.276→0.098。在 11/16 datasets gap 下降、12/16 S 下降。平均類內擴散相對更大，而不是平均類間距離縮小。
- **BEiT CAL**：raw gap 平均略增（0.0111→0.0122），但平均 S 降低（0.369→0.288），14/16 datasets 的 S 下降。這正是只看 raw gap 容易忽略尺度變化的情況。

![量化前後距離](C:/project/quant_analysis/five_dataset_features/analysis/serial_20004_20261010/class_distances/inter_intra_before_after.png)

## Separation 變化仍不能代替 accuracy

這批的 W3 accuracy 全部低於自己的 FP baseline，但 31/144 組 raw gap 並未下降、18/144 組 S 並未下降。若直接把「gap／S 是否下降」當成分類品質判斷，就會漏掉這些情況。

| Dataset | Model | Mode | Gap FP → W3 | S FP → W3 | 相對準確率損失 |
|---|---|---|---|---|---|
| cub | ViT-B/16 | ft | 0.6809 → 0.5969 | 0.527 → 0.451 | 3.91% |
| soyageing | ViT-B/16 | cal | 0.1574 → 0.1315 | 0.319 → 0.325 | 56.59% |
| soygene | BEiT-v2-B | fz | 0.0321 → 0.0649 | 0.129 → 0.189 | 99.04% |
| soyageing | BEiT-v2-B | fz | 0.0212 → 0.0460 | 0.089 → 0.142 | 97.54% |


例如 soygene／BEiT FZ 的 gap 由 0.0321 增至 0.0649、S 由 0.129 增至 0.189，準確率卻由 15.88% 降至 0.15%，相對損失 99.04%。soyageing／ViT CAL 的 S 也略升（0.319→0.325），仍損失 56.59%。全體 class-pair 平均幾何不是最終分類器的 decision margin；可能涉及局部難分類類別或 head 與 feature 的對齊，但 compact 結果沒有逐筆資料，無法判定原因。

另一方面，gap-loss（FP gap − W3 gap）與 relative-loss 的組內相關方向不一致；FZ 甚至為負（ViT −0.703、Swin −0.503、BEiT −0.835）。不應把這讀成「掉越多 gap 反而越耐量化」；這個方向可能受 baseline 與 floor effect 影響：原本 gap 大的 datasets 有較大的下降空間，原本 gap 小的 datasets 則接近下界，baseline 也不同。

## 適合用在目前研究問題的方式

先用 **FP 的 S／R、FP accuracy、Q10** 一起描述 dataset 的原始分類結構，再以 **Δinter、Δintra、ΔS** 描述量化造成的幾何變化。不要只用 raw gap 或其 retention 作單一難度分數。S 是本次探索性補充，若要選定為預測指標，仍需固定 baseline、dataset-held-out 比較與額外 seeds／quantization settings。

CAL 還有 feature-view 限制：保存的 features 來自第一個 raw-image backbone call，而 accuracy 使用 crop/flip aggregation logits；現在的 inter／intra 不等於最終 CAL decision representation。若下一步取得原始 features／predictions，可檢查 per-class、nearest-negative class distance 或 sample-level margin，直接對應 damage／rescue；這些尚未在此執行。

## 輸出與重現方法

`final_class_distances.csv` 保存 144 組 final inter/intra、gap、ratio、S、變化量與配對 accuracy；`group_class_distances.csv` 保存等權平均及方向計數；`layer_class_distances.csv` 保存各層等權摘要；`distance_correlations.csv` 保存所有組內 Spearman、absolute accuracy-drop 關係與 FP-accuracy-adjusted exploratory correlations；`distance_method.json` 保存公式與來源 checksum 記錄。

所有結果只適用 serial 20004 的 weight-only W3/g128。資料來源為每組原始 `layer_metrics.csv`、C:/project/quant_analysis/five_dataset_features/data/w3_g128_results/all_pairs.csv 與 `analysis_manifest.json`；原檔未修改。
