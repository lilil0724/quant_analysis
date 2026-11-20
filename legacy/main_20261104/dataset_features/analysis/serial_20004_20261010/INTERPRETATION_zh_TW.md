# Serial 20004：FP 與 W3/g128 特徵分析

分析日期：2026-10-10。範圍為 HQQ weight-only W3、group size 128，沒有 activation quantization；不能延伸成 W6A4、其他 bit-width 或多 seed 的結論。

這批結果最值得優先追查的是 BEiT-v2-B 的 fz：平均準確率從 62.92% 降到 12.28%，而且各層 CKA 在早期就開始下降。ViT 與 Swin 的 soybean／cotton 類資料也較敏感；排除 BEiT fz 後仍可看到這個模式。高 CKA 不保證準確率保留，Q10 雖然與量化損失有強烈相關，卻和 FP baseline 高度重疊。

## 資料完整性與分析邊界

- 已讀取 16 datasets × 3 models × ft/fz/cal = **144 組配對結果**，全部屬於 serial 20004。每個 model/mode 都有完整 16 datasets。
- 580 個下載檔案包括 579 個有 manifest checksum 的 compact 結果檔與 manifest 本身；579 個 checksum 全部吻合。
- coverage、個別 summary、總表的身份／樣本數一致；4,896 列 layer metrics 的 FP/W3 layer 配對與 final 指標一致。準確率差 = (damage − rescue)/N 等計數關係也吻合。
- manifest 列出的 288 個完整 PCA coordinate 檔依 ZIP 規格未下載，並非 compact 結果遺漏。原始 features、logits、逐筆 predictions 也未提供，這次無法獨立重算 CKA／PCA／H-score 或重新校驗原始推論檔；核對的是下載結果的完整性與內部一致性。
- 未重跑 nano4 推論。來源 manifest 記錄 analysis execution=CPU、status=complete；沒有提供 scheduler log，所以不推斷實際 job runtime。

## 模型與 checkpoint 的整體差異

下表先對每個 dataset 計算指標，再對 16 datasets 等權平均。樣本數大的 vegfru／inat17 不會主導平均。相對損失 = 1 − W3 accuracy / FP accuracy；準確率下降用百分點（pp）表示，兩者分開看。平均相對損失不是 1 − 平均 W3 accuracy / 平均 FP accuracy。

| Model | Mode | FP acc (%) | W3 acc (%) | 下降 (pp) | 相對損失 (%) | Final CKA |
|---|---|---|---|---|---|---|
| ViT-B/16 | ft | 70.75 | 65.52 | 5.23 | 11.21 | 0.873 |
| ViT-B/16 | fz | 62.81 | 56.14 | 6.67 | 16.47 | 0.905 |
| ViT-B/16 | cal | 72.03 | 67.28 | 4.75 | 12.06 | 0.944 |
| Swin-B | ft | 72.52 | 62.62 | 9.91 | 22.12 | 0.873 |
| Swin-B | fz | 69.65 | 59.53 | 10.12 | 24.24 | 0.913 |
| Swin-B | cal | 79.28 | 70.73 | 8.55 | 15.17 | 0.875 |
| BEiT-v2-B | ft | 64.97 | 50.38 | 14.59 | 27.23 | 0.750 |
| BEiT-v2-B | fz | 62.92 | 12.28 | 50.64 | 85.61 | 0.188 |
| BEiT-v2-B | cal | 76.36 | 50.11 | 26.25 | 38.58 | 0.809 |


1. **BEiT fz 是獨立值得追查的重度退化情境。** 16 datasets 平均相對損失 85.61%、準確率下降 50.64 pp、Final CKA 0.188、prediction swap rate 87.47%。同模型 ft／cal 的損失與 CKA 不同，顯示這不是所有 BEiT 設定都相同的表現。這些結果不區分量化實作問題、特定層敏感度或 checkpoint 特性；需要受控實驗判定。
2. **CAL 並非對每種模型都較耐量化。** Swin CAL 平均相對損失 15.17%，低於 FT 的 22.12%；13/16 datasets 的 CAL 損失較低。ViT CAL 有 10/16 datasets 損失比 FT 低，但平均是 12.06%，略高於 FT 的 11.21%。BEiT CAL 損失 38.58%，高於 FT 的 27.23%，而 W3 平均準確率幾乎相同（50.11% 對 50.38%）。checkpoint 與 FP baseline 不同，這是配對描述，不是 CAL 的因果效果。
3. **這 144 組的 W3 accuracy 都低於自己的 FP baseline。** 但嚴重程度不同，不能用單一平均涵蓋所有模型／checkpoint。

![量化損失矩陣](C:/project/quant_analysis/five_dataset_features/analysis/serial_20004_20261010/relative_loss_heatmap.png)

圖上以百分比呈現每個 dataset 的相對損失，排序依排除 BEiT fz 後的 dataset 平均；同一列有 9 個重複設定，不是 9 個獨立 datasets。

## 哪些 datasets 較敏感？

soy／cotton 的整體相對損失較高。為避免 BEiT fz 的廣泛退化主導排序，另列排除該設定後、每個 dataset 剩餘 8 組設定的等權平均。這只是對本次完整矩陣的摘要。

| Dataset | N | Test classes | 9 設定平均相對損失 (%) | 排除 BEiT fz 後 (%) | 9 設定平均下降 (pp) |
|---|---|---|---|---|---|
| soyageing | 4950 | 198 | 63.46 | 59.20 | 27.89 |
| soylocal | 600 | 200 | 50.86 | 45.18 | 13.72 |
| soygene | 11143 | 1110 | 47.51 | 41.07 | 15.66 |
| cotton | 240 | 80 | 45.09 | 38.75 | 14.35 |
| soyglobal | 5814 | 1938 | 45.32 | 38.71 | 8.49 |
| dogs | 8580 | 120 | 22.35 | 14.62 | 19.64 |
| cars | 8041 | 196 | 22.91 | 14.23 | 17.26 |
| pets | 3669 | 37 | 20.06 | 14.07 | 17.69 |
| aircraft | 3333 | 100 | 20.86 | 12.54 | 14.01 |
| inat17 | 95986 | 5089 | 20.68 | 11.85 | 14.07 |
| vegfru | 116931 | 292 | 17.67 | 11.06 | 16.42 |
| cub | 5794 | 200 | 16.32 | 9.08 | 14.60 |
| food | 25250 | 101 | 16.35 | 7.98 | 14.80 |
| nabirds | 24633 | 555 | 15.82 | 7.38 | 13.83 |
| flowers | 6149 | 102 | 10.81 | 5.76 | 10.51 |
| moe | 2943 | 173 | 13.16 | 2.69 | 10.08 |


排除 BEiT fz 後，soyageing、soylocal、soygene、cotton、soyglobal 仍排在最高損失群。soyglobal 的 FP 準確率本來偏低，因此 45.32% 相對損失對應的平均絕對下降為 8.49 pp；必須同時讀 baseline 與 pp。soyageing 則平均下降 27.89 pp，不只是小分母帶來的相對損失。cotton 的 N=240、soylocal 的 N=600，單次 test 結果的不確定性也需納入後續驗證。

目前不能把這個模式解釋成「農業影像本質上比較怕量化」：類別數、FP accuracy、樣本量、訓練與 checkpoint 都同時變動，且只有本批 W3/g128 結果。

## CKA 與準確率並不等價

Final CKA 與相對損失的組內 Spearman 在 −0.538 到 +0.009 之間，沒有一致的強關係。兩個 ViT 例子更直接：

| Dataset | Model | Mode | FP acc (%) | W3 acc (%) | 相對損失 (%) | Final CKA |
|---|---|---|---|---|---|---|
| soyageing | ViT-B/16 | cal | 36.63 | 15.90 | 56.59 | 0.968 |
| cub | ViT-B/16 | ft | 87.83 | 84.40 | 3.91 | 0.782 |
| soygene | ViT-B/16 | cal | 51.90 | 49.38 | 4.86 | 0.975 |
| soygene | BEiT-v2-B | fz | 15.88 | 0.15 | 99.04 | 0.233 |


soyageing／ViT CAL 的 CKA=0.968，仍掉 56.59% 相對準確率；CUB／ViT FT 的 CKA=0.782，卻只掉 3.91%。CKA 描述整體表示幾何相似度，不能直接替代分類 margin 或錯誤率。這些案例不同 checkpoint／dataset，因此用來反駁「高 CKA 保證保留準確率」，不作單一機制的因果比較。

各層平均 CKA 顯示 BEiT fz 的 block_00≈0.975、block_01≈0.741、block_11≈0.132、final≈0.188；ViT／Swin 也有中後段下降及 final 回升，不能假設 CKA 隨深度單調下降。每個觀測層包含前面量化的累積影響，低 CKA 的層不等於該層單獨造成損失。

![各層 CKA](C:/project/quant_analysis/five_dataset_features/analysis/serial_20004_20261010/layer_cka_profiles.png)

## Q10 可以作為篩查訊號，但還不是獨立預測器

以下每個相關係數都只在固定 model/mode 的 16 datasets 內計算，不把 144 個 cells 當成獨立樣本。Q10 指 FP normalized-margin 的第 10 百分位；數值小表示低 margin 尾端較弱。

| Model | Mode | Q10 vs loss | FP acc vs loss | Q10 vs FP acc | Q10 adjusted* | CKA vs loss |
|---|---|---|---|---|---|---|
| ViT-B/16 | ft | -0.935 | -0.935 | 0.979 | -0.301 | -0.171 |
| ViT-B/16 | fz | -0.926 | -0.874 | 0.953 | -0.669 | -0.538 |
| ViT-B/16 | cal | -0.888 | -0.921 | 0.974 | -0.046 | -0.182 |
| Swin-B | ft | -0.935 | -0.912 | 0.974 | -0.606 | -0.403 |
| Swin-B | fz | -0.982 | -0.965 | 0.968 | -0.617 | -0.265 |
| Swin-B | cal | -0.812 | -0.832 | 0.959 | -0.256 | -0.132 |
| BEiT-v2-B | ft | -0.588 | -0.612 | 0.947 | -0.173 | -0.282 |
| BEiT-v2-B | fz | -0.906 | -0.926 | 0.982 | 0.033 | -0.176 |
| BEiT-v2-B | cal | -0.776 | -0.641 | 0.953 | -0.756 | 0.009 |


*adjusted：分別將 Q10 與相對損失的 rank 對 intercept、FP accuracy rank、class-count rank、sample-count rank 回歸，再計算殘差相關。使用 average tied ranks，每組 n=16；這是探索性 partial rank correlation，沒有推論檢定或 held-out 預測驗證。

Q10 與相對損失原始相關約 −0.588～−0.982，但 Q10 與 FP accuracy 的相關也高達 +0.947～+0.982。控制 baseline／類別數／樣本數後，ViT CAL 的相關由 −0.888 降到 −0.046，BEiT fz 由 −0.906 變為 +0.033；其他組仍保留較大相關，例如 BEiT CAL −0.756。結論是 Q10 有篩查價值，但其增量資訊依模型／checkpoint 而異，這批結果不足以說它能獨立預測量化難度。相對損失本身還以 FP accuracy 作分母，因此應搭配絕對下降的結果一起核對；CSV 也提供該相關。

![特徵相關](C:/project/quant_analysis/five_dataset_features/analysis/serial_20004_20261010/feature_correlations.png)

## H-score、separability 與 PCA 的解讀

H-score 只在 final features 上計算；以各 dataset 的 W3/FP 比值再取組內中位數，BEiT fz 為 0.228，其他組約 0.674～0.867。這支持 BEiT fz 的表示／類別結構變化值得追查，卻不代表 H-score 的變化就是造成 accuracy loss 的原因。

FP separability 與 loss 的相關在 FT／FZ 通常為負，在三個 CAL 組卻接近 0（ViT −0.085、Swin +0.053、BEiT +0.091）。CAL 的 feature 來自第一個 raw-image backbone call，prediction logits 保留 crop/flip aggregation，因此這些 feature metrics 並未完整描述 CAL 的最終決策表示。不同模型或 checkpoint 的 feature scale／pooling 也不同，不宜直接比較 H-score 或距離的絕對值。

已視覺檢查兩張來源 PCA 圖：
- [soygene／BEiT fz PCA](C:/project/quant_analysis/five_dataset_features/data/w3_g128_results/soygene/beitv2_base_patch16_224/fz/pca.png)：W3 投影相較 FP 明顯收縮／偏移；FP PC1+PC2 只解釋 37.2%+12.3%=49.5% 的 FP variance。這是固定 FP basis 中的投影變化，不能單靠此圖斷言完整空間 rank collapse 或量化機制。
- [soyageing／ViT CAL PCA](C:/project/quant_analysis/five_dataset_features/data/w3_g128_results/soyageing/vit_b16/cal/pca.png)：兩種條件的粗略形狀仍相近，然而準確率損失很大；前兩個 FP PCs 為 43.2%+6.7%=49.9%。2D PCA 無法代替 label-aware margin 或高維判別分析，198 類顏色也不能清楚分辨所有類別。

## 建議下一個受控實驗

1. 先確認 BEiT fz 的 source checkpoint、quantizer scope、group size、classifier head 與 loading 都符合預期。再在同一 checkpoint／test split 做較高 weight-bit 或選定模組保留 FP 的對照，判定退化是否隨量化配置改變；目前沒有權重／逐筆 logits，不能從 compact ZIP 判定實作錯誤。
2. 選 soyageing 與一個較穩定的對照 dataset，在固定 model／checkpoint 下檢查 sample-level margin、damage/rescue 與 CKA 變化；不以跨 checkpoint 的 PCA 直接當因果證據。
3. 若要宣稱 Q10 有獨立預測能力，先與 FP accuracy-only baseline 做 dataset-held-out 比較，再增加 seed／bit-width。這次沒有執行這些驗證。

## 計算方法與檔案

- `group_summary.csv`：每個 model/mode 的 16-dataset 等權均值；H-score ratio 取中位數。
- `dataset_summary.csv`／`dataset_summary_without_beit_fz.csv`：每 dataset 的 9／8 設定描述摘要，不跨設定推論獨立性。
- `paired_derived_metrics.csv`：來源 144 rows，附類別數、H-score ratio、Q10 ratio、separability delta、norm-tail delta。類別數由 n_inter_class_pairs=C(C−1)/2 反解，並核對不同設定一致。
- `within_group_correlations.csv`：描述性 Spearman、absolute-drop 關係與 exploratory partial rank correlations。Spearman 使用 average tied ranks，不作多重檢定或顯著性宣稱。
- `mode_comparison.csv`：相同 dataset/model 的 checkpoint 描述差異；與各自 FP baseline 配對。
- `layer_summary.csv`：每個模型觀測 boundary 的 16-dataset 等權摘要，保留不同架構的 layer 名稱。
- `integrity.json`：下載結果校驗、來源 manifest hash、來源 code digest、缺少原始資料的邊界。

原始來源：[C:/project/quant_analysis/five_dataset_features/data/w3_g128_results/all_pairs.csv](C:/project/quant_analysis/five_dataset_features/data/w3_g128_results/all_pairs.csv) 與 [C:/project/quant_analysis/five_dataset_features/data/w3_g128_results/analysis_manifest.json](C:/project/quant_analysis/five_dataset_features/data/w3_g128_results/analysis_manifest.json)；原檔未修改。

Source manifest SHA-256: `4bf50637c0602e38887580ca3bdf4253abaa87966053ba1286cea8d32f9f73ec`。來源 analysis code digest: `a543431486848653d9e5ef46171f43cffa5dd79403292bf6e33bc0b219245bcf`；這是來源版本記錄，沒有把目前本機 code 版本視為產生這批結果的版本。
