# acc_ratio 與 FP／量化後 inter-intra class distance

## 結論

有關係，但方向與強度取決於 model/mode 與距離定義，不能用單一距離或其保留率解釋全部 accuracy drop。
FT/FZ 的 FP raw gap 與 acc_ratio 多數呈明顯正相關；CAL 的 raw gap 接近無相關。
量化後 raw gap 在多數 FT/FZ 仍正相關；BEiTv2 FZ 的關係弱於 FP，且量化後正規化 S 的相關估計為弱負值，bootstrap 區間跨過零，方向仍不確定。
gap 的保留率不具普遍一致的關係，BEiTv2 FZ 明顯反向。

## 定義與分析單位

- acc_ratio = quantized accuracy / FP accuracy；越高表示準確率保留越好。
- G = D_inter - D_intra，亦即既有 inter-intra gap。
- S = (D_inter - D_intra)/(D_inter + D_intra)，補充相對距離尺度。
- G_quant/G_FP 表示 gap 保留率；若問題中的 FP/quantized 指的是兩者相除，G_FP/G_quant 為其倒數，Spearman 符號恰好相反，並不是另一個獨立證據。
- 距離為 L2-normalized features 上的 1-cosine；intra 對類別等權平均，inter 對不同類別 pair 等權平均。
- 僅分析 final backbone representation；serial 20004、W3/G128，3 models × 3 modes，每組 16 datasets。
- 重新從 144 個原始 compact layer_metrics.csv 取 final FP/W3 distances，與 all_pairs.csv 的 separability 核對一致；原始 features 不在此 ZIP，未重新計算所有 sample-pair distances。

## 組內 Spearman correlation

正值表示該距離量較大的 dataset 通常有較高 acc_ratio；負值表示相反。

| Model | Mode | FP G | Quant G | G_quant/G_FP | FP S | Quant S |
|---|---|---:|---:|---:|---:|---:|
| ViT-B/16 | FT | +0.838 | +0.818 | +0.341 | +0.838 | +0.803 |
| ViT-B/16 | FZ | +0.847 | +0.859 | -0.121 | +0.788 | +0.774 |
| ViT-B/16 | CAL | +0.085 | +0.106 | +0.647 | +0.468 | +0.471 |
| Swin-B | FT | +0.862 | +0.879 | +0.400 | +0.871 | +0.809 |
| Swin-B | FZ | +0.821 | +0.897 | -0.044 | +0.665 | +0.626 |
| Swin-B | CAL | -0.053 | -0.062 | +0.415 | +0.744 | +0.797 |
| BEiTv2-B | FT | +0.565 | +0.841 | +0.409 | +0.515 | +0.609 |
| BEiTv2-B | FZ | +0.844 | +0.474 | -0.706 | +0.888 | -0.253 |
| BEiTv2-B | CAL | -0.091 | -0.229 | -0.274 | +0.671 | +0.559 |

這裡沒有把 144 個 model/mode/dataset cells 當成獨立樣本，沒有宣稱顯著性或因果。

## BEiTv2 的不確定性與 baseline 控制

下表使用 5,000 次 dataset bootstrap 的 pointwise 95% percentile interval；區間不是多重比較校正後的 simultaneous interval。
Partial rank 是將 predictor/outcome 的 rank 分別對 controls 回歸，再求殘差相關。

| Mode | 指標 | rho | bootstrap 95% CI | 控制 FP acc | 控制 FP acc + log(classes) |
|---|---|---:|---|---:|---:|
| FT | fp_gap | +0.565 | [+0.084, +0.854] | -0.041 | -0.071 |
| FT | quant_gap | +0.841 | [+0.503, +0.970] | +0.734 | +0.663 |
| FT | gap_retention | +0.409 | [-0.088, +0.837] | +0.637 | +0.521 |
| FT | fp_s_norm | +0.515 | [+0.001, +0.851] | -0.104 | +0.024 |
| FT | quant_s_norm | +0.609 | [+0.074, +0.911] | +0.446 | +0.443 |
| FZ | fp_gap | +0.844 | [+0.586, +0.943] | +0.517 | +0.704 |
| FZ | quant_gap | +0.474 | [-0.092, +0.830] | +0.065 | +0.195 |
| FZ | gap_retention | -0.706 | [-0.895, -0.246] | -0.420 | -0.417 |
| FZ | fp_s_norm | +0.888 | [+0.674, +0.964] | +0.712 | +0.794 |
| FZ | quant_s_norm | -0.253 | [-0.772, +0.380] | -0.220 | -0.189 |
| CAL | fp_gap | -0.091 | [-0.636, +0.473] | -0.031 | +0.017 |
| CAL | quant_gap | -0.229 | [-0.703, +0.330] | -0.221 | -0.188 |
| CAL | gap_retention | -0.274 | [-0.737, +0.296] | -0.458 | -0.447 |
| CAL | fp_s_norm | +0.671 | [+0.190, +0.920] | +0.346 | +0.334 |
| CAL | quant_s_norm | +0.559 | [+0.064, +0.838] | +0.274 | +0.269 |

每組只有 16 datasets，控制分析仍屬探索性，不能代替 dataset-held-out prediction。

## 為何 gap 變大仍可能 accuracy 變差

BEiTv2 CAL／Cars：FP gap 約 0.00110，quant gap 約 0.00267，gap 增加約 2.42 倍，
但 accuracy 92.14% → 43.85%，acc_ratio 47.59%；正規化 S 則由 0.276 → 0.183。
代表 raw inter/intra 同時改變尺度時，raw gap 增大不能直接表示分類變好。

BEiTv2 FZ／SoyGene：FP gap 約 0.0321 → quant gap 0.0649，正規化 S 約 0.129 → 0.189，
兩者都增加，但 accuracy 15.88% → 0.15%。即使正規化也不是分類效能的充分指標。
類別平均幾何沒有量測 feature 與已訓練 classifier head 的對齊，也不直接量測 hardest class pair 或 decision margin。
這些是量測限制，尚未透過本次 aggregate outputs 判定實際主導機制。

## 研究上可如何使用

FP separation 可作量化前候選指標；量化後 separation/保留率屬結果描述，不能充當尚未量化時的預測量。
目前 FP S 比 raw FP G 更能描述 CAL 的跨 dataset 關係，但控制 baseline 後會改變，不宜直接當成獨立量化難度因子。
FP/quant gap ratio 在 gap 很小時也會放大尺度差異；目前所有 gap 正值，但 FP 最小約 0.000797。
CAL 保存的 feature 是第一個 raw-image backbone call，accuracy 卻來自完整 crop/flip logits，兩者 representation 邊界不同。

![Correlation heatmap](C:/project/quant_analysis/five_dataset_features/analysis/serial_20004_20261010/class_distances/acc_ratio_relationship/correlation_heatmap.png)

![FP/quant gaps vs acc_ratio](C:/project/quant_analysis/five_dataset_features/analysis/serial_20004_20261010/class_distances/acc_ratio_relationship/fp_quant_gap_vs_acc_ratio.png)

## 輸出

distance_accuracy_pairs.csv 保存 144 組配對指標；correlations.csv 保存 81 個組內分析、Pearson sensitivity、bootstrap interval、leave-one-dataset-out 範圍與 partial rank；
correlation_summary.csv 為組內 Spearman 表；method.json 保存定義、限制、seed、所有來源的 SHA256。
所有來源在分析前後 checksum 相同；沒有改動來源資料或 inference code，沒有重新執行 nano4 GPU 推論。
