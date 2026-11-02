# Dataset 與量化 accuracy retention：目前資料的證據與原因

日期：2026-10-10。結論是 **有明顯關係，但目前識別的是 dataset–model–checkpoint 組合的量化敏感度，不能把它全部歸因於 dataset 本身。** 最符合現有證據的解釋是分類餘裕、資料經模型後的類別幾何、量化擾動與分類方向的關係，以及正確／錯誤預測的轉換。樣本數或單一 CKA 指標不足以解釋。

這次重新分析本機既有結果；未訓練新模型、未重跑 nano4/nano5 推論。144 組資料的 579 個 compact output SHA-256 全部通過；每組 summary 的身份、樣本數、accuracy、damage/rescue 和總表一致。完整 features、逐筆 logits/predictions 與 checkpoint 不包含在 serial 20004 compact 結果中，因此不能獨立重算其原始 feature 指標或判定哪個模組造成錯誤。

**1. 使用哪些資料，以及 acc_ratio 代表什麼**

| 資料 | 可用範圍 | 本報告用途 |
|---|---|---|
| `data/backbones_quant.csv` | 15 datasets × 3 models × 3 checkpoint modes × 8 bits × 4 group sizes；4,320 個 exact checkpoint/seed baseline-matched cells | 重算低 bit 差異、跨量化設定與跨模型的排序 |
| `five_dataset_features/data/w3_g128_results`，serial 20004 | 16 datasets × 3 models × FT/FZ/CAL；144 組 FP/W3-g128 配對 | margin、class distances、CKA、damage/rescue、held-out 模型 |
| serial 100004 舊 artifacts／分析 | 排除 inat17 的 14 個 ViT-B/16 dataset–checkpoint pairs；3/4-bit × g8/g128 | 樣本內 margin 機制及先前 Q10-only held-out 結果；與新 144 組分開解讀 |

`acc_ratio = accuracy_Q / accuracy_FP`；`1 − acc_ratio` 是相對 accuracy loss。baseline 必須與量化結果使用相同 dataset、checkpoint、seed 和 test protocol；舊 confidence workflow 的 baseline 是 **unquantized AMP-FP16**，不能一概叫純 FP32。比例、FP accuracy 和絕對下降（percentage points, pp）應一起報。

本次模型分析是 **weight-only W3/g128**。目前 TGDA exporter 預設不量化 activation，並排除指定 classifier heads；這是現有 code contract，來源 compact manifest 只保存 source hash，不能用目前程式版本替代來源 checkpoint/module coverage 的逐項證明。legacy CSV 的 quant-zero/scale 等設定與新 exporter 預設也不完全相同，不把兩批數值差異全部當作 dataset 效果。

**2. 關係相當穩定，但不是跨所有條件不變的排序**

重算 legacy 資料，固定 model/checkpoint mode、比較 quantization conditions 時，dataset retention 排序的 median Spearman 為 **0.834**；限 3/4-bit 為 **0.904**。跨 model/checkpoint contexts、固定量化條件的 primary median 約 **0.562**。完整 1–8-bit grid 的排序一致性降到 **0.314**：極低 bit 接近 chance floor、高 bit 接近 1，都使排序容易被小差異主導。

| Legacy bit | 四個 soybean datasets 的等權 mean ratio | 其餘 11 datasets 的等權 mean ratio |
|---|---:|---:|
| 3 | 0.584 | 0.871 |
| 4 | 0.859 | 0.983 |
| 6 | 0.992 | 0.999 |

先平均每個 dataset 的 36 個 model/mode/group 設定，再對 datasets 等權；4,320 cells 不是 4,320 個獨立資料集。此表的 comparator 是「其餘 11 datasets」，不是全數 conventional FGIR。3-bit soybean 組的 mean absolute drop 為 13.18 pp，其餘組 10.09 pp；4-bit 為 4.24 vs 1.37 pp。因此部分差異也存在於絕對下降，但比例確實放大低 baseline 的影響。

新 W3/g128 的 9 contexts，dataset 排序 Kendall’s W=**0.692**、pairwise Spearman median=**0.610**。BEiT FZ 是各 dataset 普遍嚴重退化的情境；完整 cohort 保留，另將它排除作敏感度分析。剩餘 8 contexts 的 W=**0.701**、median rho=**0.604**，不是刪掉這個情境才產生 dataset 差異。

| Dataset（8 contexts 等權平均） | mean acc_ratio | mean FP accuracy | mean absolute drop |
|---|---:|---:|---:|
| soyageing | 0.408 | 46.44% | 28.17 pp |
| soylocal | 0.548 | 27.62% | 12.73 pp |
| soygene | 0.589 | 40.51% | 15.65 pp |
| cotton | 0.613 | 33.23% | 12.60 pp |
| soyglobal | 0.613 | 22.14% | 8.74 pp |
| cub | 0.909 | 88.92% | 8.03 pp |
| flowers | 0.942 | 98.66% | 5.47 pp |
| moe | 0.973 | 93.63% | 2.39 pp |

全部 16 datasets 的表在 `dataset_profiles.csv`。mean ratio 是先算各 cell 的 ratio 再平均，不能由表中的 mean accuracies 相除重建。

balanced additive sums-of-squares 的描述性分解也顯示：完整 9-context ratio matrix 的 dataset／context／剩餘交互作用與未量測誤差份額為 28.7%／54.8%／16.5%；排除 BEiT FZ 後為 56.9%／14.8%／28.3%。這不是因果解釋率，也不提供 seed-level uncertainty。換成 absolute drop，排除後 dataset 份額只有 21.7%、Kendall W=0.426，說明「retention 的 dataset 排序」與「掉最多 accuracy 百分點的排序」有明顯差別。

**3. 最有力的機制：margin 小，量化更容易改變預測**

令 FP top-1 與 top-2 logits 的差為 `m`。若每一個 logit 的擾動大小不超過 `ε`，則 `m > 2ε` 足以保證 top-1 不變。這是直接的數學充分條件；不滿足不代表必然翻轉。分類餘裕要和**同尺度的量化擾動**一起看。

Q10 是每個 dataset 的 FP normalized decision margin 第 10 百分位：`(top1−top2)/std(all class logits)`。它描述脆弱尾端，沒有提供量化誤差界，因此不是 robustness certificate，也不是 dataset intrinsic property。

在新資料的固定 model/mode 內，排除 BEiT FZ 後：

| 關係 | 8 contexts 的 rho 平均 | 組內 rho 範圍 |
|---|---:|---:|
| FP Q10 → acc_ratio | +0.856 | +0.588 到 +0.982 |
| FP Q10 → prediction swap | −0.917 | −0.985 到 −0.809 |
| FP accuracy → acc_ratio | +0.836 | +0.612 到 +0.965 |
| swap → acc_ratio | −0.961 | −0.985 到 −0.888 |

再去掉四個 soybean 與 cotton，保留 11 datasets：Q10→ratio 在 **8/8 contexts 仍正向**，rho=+0.509～+0.973，平均 +0.728；Q10→swap 仍全部負向，平均 −0.805。因此關係不全由 soybean 群造成。這些是探索性相關，沒有把 cells 當獨立樣本作顯著性檢定。

先前 serial 100004 的 **within-dataset** 結果更能補足跨 dataset 混淆：14 個 dataset 的平均四條件 AUROC 中位數 **0.940**，低 normalized margin 較能辨識 prediction swaps。dataset 等權的最低 margin decile 平均 swap=**39.04%**，最高 decile=**2.17%**。這說明同一 dataset 內也有 margin–swap 關係；不能從 AUROC 推出所有 loss 都由 Q10 決定。

FP accuracy 與 Q10 高度重疊。控制 FP accuracy 後，Q10→ratio 的 mean partial rank correlation 約 **+0.362**，各 context 從 −0.089 到 +0.735；再控制 class count／test n，mean 約 +0.428，仍是 n=16 的探索性殘差相關。控制變數不同會改變結果，不能把它當作獨立因果效果。

**3a. 直接重讀原始 logits：脆弱 margin 的分布比單看擾動總量更有解釋力**

另以 serial 100004 的 70 份既有 artifacts 重算 14 datasets × 4 conditions 的 56 個 paired cells；這批有原始 logits，不能和沒有原始 arrays 的 serial 20004 混淆。計算 `δz=z_Q−z_FP`，扣除每個樣本跨類別的共同位移後，除以該樣本 baseline logit std。這使 margin 與 noise 在同一 normalized 尺度比較，共同位移不影響 argmax。

四個 soybean 的 baseline Q10 dataset 中位數為 **0.0302**，其餘十個 datasets 為 **0.9056**。更直接地，normalized margin <0.25 的樣本比例，先在各 dataset 算比例再等權平均，為 **64.31% vs 4.16%**。soybean 有大量樣本原本就貼近決策邊界，這比只稱其「困難」更具體。

| Condition | soy／other 的 normalized RMS median 比值* | soy／other 的 swap-rate 比值* | Q10→swap rho | normalized RMS median→swap rho |
|---|---:|---:|---:|---:|
| W3/g8 | 1.28 | 8.68 | −0.974 | +0.723 |
| W3/g128 | 1.27 | 6.66 | −0.978 | +0.714 |
| W4/g8 | 1.35 | 9.93 | −0.930 | +0.829 |
| W4/g128 | 1.41 | 9.30 | −0.960 | +0.763 |

*先在各 dataset 取得 median RMS／swap rate，再按 group 等權平均後相除；不是 pooled images 或個別 sample 的比值。normalized noise 本身也和 swaps 有關，不是可以完全忽略。不同條件的 RMS 絕對值不可在表中由 group ratio 推回。

擾動總量只有約 1.3–1.4 倍，而 swaps 約 6.7–9.9 倍，配合 margin 分布差異，更支持「同樣量級的誤差作用在大量小 margin 樣本上」這個解釋。這不是完成的 causal mediation，群組平均比值也不是因果效果的分攤。

在共同的 normalized margin bin [0.5,1) 內，四條件 soy4 的 dataset-equal swap 依序為 **2.57%、31.38%、0.185%、5.18%**，other10 為 **12.42%、33.59%、2.22%、11.00%**，soy 在這個 bin 不再較高。每條件都有全部 4／10 datasets 貢獻，該 bin 合计樣本數為 2,219／10,864。其他 bins 與 counts 全部公開；此 bin 是解讀用的事後例子，寬 bins 仍有分布差異，不是完全 margin-matched 調整，更不能挑一個 bin 宣稱排除了全部 dataset 效果。

原始 logits 也驗證 `m > 2×centered normalized Linf` 的事後充分條件：667,387 個符合嚴格條件的 **pair-samples** 中，prediction flip=0。同一樣本可在多個條件重複，不能把這個數當獨立樣本數或事前 robustness certificate。FP16 ties 的 winner 可能與 NumPy argmax tie rule 不同；驗證保留原 artifact 的 top-1，確認其 logit 達到最大值，沒有重新定義 prediction。

四個 Q10→swap 係數的 median 為 −0.967033，與先前 Step1 的 summary 一致；逐條件值與 median 是不同摘要，不是資料變化或獨立重複。完整原始量測見 [raw-logit 報告](C:/project/quant_analysis/hqq_confidence/analysis/dataset_relationship_20261010/raw_logit_mechanism_report.md)。

![原始 logits 的 margin 分布與擾動對照](C:/project/quant_analysis/hqq_confidence/analysis/dataset_relationship_20261010/raw_logit_mechanism.png)

**4. 類別幾何支持 margin 解釋，但平均距離不是分類邊界**

以 class-equal cosine distances 定義 `S=(D_inter−D_intra)/(D_inter+D_intra)`。本次從每組原始 compact `layer_metrics.csv` 重建 S，再核對既有 distance 表。

FP S→ratio 在 8/8 contexts 正向，平均 rho=**+0.695**；去掉 soybean/cotton 後仍 8/8 正向，平均 **+0.591**。FP S→swap 平均 rho=**−0.738**。類別相對分離較清楚時，決策較有餘裕，是和 margin 路徑相容的證據。

但控制 FP accuracy 後，FP S→ratio 的 mean partial rho 只約 **+0.226**。另外，量化後 S 的絕對大小和 ratio 正相關，不代表 S 的下降量解釋 loss：`ΔS` 的相關方向與 baseline 起點重疊，gap retention 的方向也不一致。quant S 是事後指標，不能作量化前的獨立預測證据。

平均 inter/intra distances 混合全部 class pairs，不針對最近的混淆類別；固定 linear head 下，真正改變類別 a 和 j logit gap 的是 `(w_a−w_j)^T δh`。即使全局 CKA 高、總 feature drift 小，若 drift 對準敏感的分類方向仍可造成錯誤。CAL features 來自第一個 raw-image backbone view，accuracy logits 則包含 crop/flip 聚合，兩者也不是完整一致的決策邊界。

**5. acc_ratio 還取決於 damage/rescue 和分母**

匹配同一批樣本時，以下是**精確恆等式**，已在全部 144 cells 驗證：

```text
1 − acc_ratio
  = (damage − rescue) / (N × accuracy_FP)
  = swap_rate × net_damage_per_swap / accuracy_FP
```

swap 不必然使 accuracy 下降；原本錯誤→另一個錯誤，accuracy 沒變；錯→對則是 rescue。低 baseline 會把同樣 absolute drop 轉成更大的相對 loss。

soy/cotton 在保留的全部 8 contexts 都比其餘 datasets 有較高 swap rate、較低 retention，但平均 `net_damage_per_swap` 反而全部較低。所以不能簡化成「每次 swap 更有破壞力」或「overconfidence 更大」；大量 swaps、較低 baseline 和 damage/rescue 組成共同决定比例。

例：soyageing／ViT CAL，FP=36.63%、Q=15.90%，ratio=0.434，drop=20.73 pp；swap=81.07%，net damage/swap=0.256。代入 `0.8107 × 0.2557 / 0.3663 ≈ 0.5659`。原本正確樣本約 70.44% 被破壞，而原本錯誤樣本約 8.00% 被救回，不能只說分母小。

另一個反例更重要：BEiT CAL 的 aircraft FP=91.15%、ratio=0.952，cars FP=92.14%、ratio=0.476。baseline 幾乎相同，retention 仍差很多；baseline 能預測整體群集，卻不是充分原因。兩者也不是相同 checkpoint，仍需量化擾動和 decision direction 分析。

**6. 新補的 held-out：不能把高相關直接當作可預測的原因**

針對每個 model/mode，16 datasets 各自做 OLS LODO：每次只用其餘 15 datasets 擬合。固定測試 constant、FP accuracy、baseline controls、raw Q10、log Q10、FP S，以及 FP accuracy 加這些指標。training-fold-only standardization，沒有 outcome clipping、模型調參或測試集 outcome leakage。所有 LODO 結果與獨立 PRESS identity 核對；其他 family folds 直接重擬合。

log Q10 是**新增事後探索的固定變換**，不能拿它替代先前 raw-Q10 主要模型或宣布前次失敗被推翻。報告保留全部模型，而非只挑誤差最小者。

| Context | FP accuracy-only MAE | 加 log Q10 MAE | 加 FP S MAE |
|---|---:|---:|---:|
| ViT FT | 3.13 | 3.19 | 3.16 |
| ViT FZ | 6.55 | 7.93 | 7.01 |
| ViT CAL | 6.13 | 7.04 | 6.44 |
| Swin FT | 7.48 | 9.02 | 8.03 |
| Swin FZ | 6.63 | 7.13 | 7.06 |
| Swin CAL | 7.72 | 9.50 | 8.16 |
| BEiT FT | 19.90 | 17.90 | 20.15 |
| BEiT CAL | 18.50 | 13.28 | 16.07 |

單位是 **accuracy-ratio percentage points**，不是 task accuracy-drop pp。排除 BEiT FZ 的 8-context 等權 MAE：constant=17.99、FP-only=9.51、FP+logQ10=9.37、FP+S=9.51。平均增益只有約 0.13，主要來自 BEiT；ViT/Swin 的 6/6 contexts 加 log Q10 都變差。

再把四個 soybean 視為同一 holdout block，確保預測 soybean 時訓練中沒有其他 soybean：FP-only／FP+logQ10 的 mean MAE=9.34／9.17；仍沒有跨 contexts 一致改善。進一步將 soy+cotton 五個 datasets 一起留出，mean MAE=13.64／13.36，且 BEiT CAL 的 MAE 從 FP-only 21.85 增至 28.19。這是更強的 extrapolation stress test，不代表 soybean/cotton 是同一來源 family；各 fold datasets 等權，沒有把 block 當成多次獨立實驗。

先前 14-dataset raw-Q10-only 線性分析也已在現有檔案中核對：mean relative-loss MAE=4.75，而 constant=3.84、accuracy-only=0.89。它只是特定 raw-linear magnitude predictor 的不足；不能推論 margin 沒有樣本層級機制。

**7. 哪些候選原因目前較弱**

- **CKA**：8 contexts 的 CKA→ratio 平均 rho=+0.246，範圍 −0.009～+0.538；去掉 soy/cotton 後升至平均 +0.562，表示 CKA 並非完全沒訊號，但其關係受 cohort 影響。soyageing／ViT CAL 的 CKA=0.968 卻只保留 43.4% accuracy，是高 CKA 不保證 retention 的反例。
- **H-score retention**：方向與 dataset 組成相當敏感；不能由一個 representation ratio 推出 accuracy ratio 的因果機制。
- **Class count**：8 contexts 的 class-count→ratio 平均 rho=−0.247；去掉 soybean 後為 +0.067，缺乏一致方向。inat17 類別很多卻比多個 soybean datasets 更穩；「類別多就一定不耐量化」不成立。
- **Test sample count**：平均 rho=+0.163，去掉 soy/cotton 後為 −0.269。test n 主要關乎估計精度，不能把它當訓練樣本量的效應。
- **Logged training sample count**：舊 14-dataset `selected_runs.csv` 中可核對 `num_images_train` 的設定記錄；四條件與 ratio 的 rho 為 +0.112、+0.015、+0.191、+0.138，與 swap 為 −0.051、−0.037、−0.086、−0.125，沒有強烈一致的大小關係。這是 recorded count，不是從原始 training split 獨立核對的因果樣本量實驗。
- **Final feature norm tail**：平均 rho=+0.185、方向從 −0.359 到 +0.929；它是 across-image feature norm 的尾部，不是 channel outlier/clipping 測量。
- **Activation clipping/outliers**：本批是 weight-only，不能用 W+A 文獻的 activation clipping 解釋已發生的損失。浮點 activation 的方向/能量仍可影響 weight error 的傳播，但這尚需原始 features/weights 驗證。
- **Overconfidence**：既有 OECE 結果不支持統一的損失機制；現有 decomposition 也反駁 soybean 每個 swap 都比較傷的簡化說法。

**8. 為什麼 weight-only 也可能與資料有關**

單一 linear layer 對相同 FP input h 的量化誤差為 `δy=ΔW h`。因此 `E||δy||² = tr(ΔW E[hh^T] ΔW^T)`：相同 `ΔW` 在不同特徵方向/能量上能产生不同輸出誤差。這是直接數學推導，與 [Optimal Brain Compression](https://arxiv.org/abs/2208.11580) 的 data-dependent layer reconstruction 思路一致；HQQ 並不是 OBQ。

多層網路還有 upstream error 和 nonlinear propagation。最終傷害取決於擾動是否沿著敏感的 head/class-boundary 方向，而非誤差總量而已。[PTQ4ViT](https://arxiv.org/abs/2111.12293) 的 Hessian-guided metric 支持任務敏感度重要、局部 MSE/cosine 不充分；該文含 activation quantization，其方法消融不能證明本資料的因果。

所以比較合理的待驗證鏈條是：

```text
dataset 的輸入分布、標籤定義與訓練結果
    → FP 特徵分布、class geometry、decision margin
量化配置和該 checkpoint 的權重
    → 輸出擾動的大小及方向
margin 相對於這個擾動
    → swaps → damage/rescue → accuracy_ratio
```

「葉片纹理類似、品種標籤細、pretraining domain mismatch、每類訓練不足」都是可疑上游因素，**本資料沒有隔離它們，不能列為已查明的根因**。soybean family、葉片影像、低 baseline、label space 和 checkpoint history 同時變動。

**9. 要把原因推進到可驗證的機制，下一步測什麼**

| 待區分的解釋 | 最直接的檢驗 | 支持結果應該長什麼樣 |
|---|---|---|
| 新 20004 cohort 的 margin 小 vs perturbation 大 | 取得新 16-dataset/3-model paired logits，延續舊批次已完成的同 margin bin／noise 分析 | 若近似 margin 下仍有 loss 差異，擾動量/方向需額外解釋 |
| 全局 geometry vs decision direction | classifier rival 方向上的 feature drift，配對 true-class margin；CAL 保存聚合 view | head 方向的 drift 比 CKA/平均 inter-intra 更能對應 damage |
| checkpoint/quantizer 異常 | 先核對 BEiT FZ 原始 checkpoint、模組 coverage；bits sweep／保留選定模組 FP | 配置介入恢復 accuracy，且同樣本損失的變化對應特定模組 |
| dataset 訊號 vs training outcome | 每 dataset 多個 seeds；matched-accuracy 不同 margin checkpoints | 同 dataset、近似 FP accuracy 下 margin 差仍對應量化損失 |
| label granularity | 同一批影像/固定 backbone，改 label hierarchy 或作預定義 task 比較 | 幾何、margin、damages 隨受控標籤變化一致移動 |
| 可泛化預測 | 凍結分析後加入新 family/model/bit/quantizer | 對新資料仍優於 FP-accuracy baseline，而非只在已看過的16組內改善 |

這些是後續實驗要求，不是本次已完成的訓練或伺服器執行。現有資料已足以判定有 dataset-associated retention pattern，也支持 margin–swap 路徑；未足以唯一識別 dataset 的因果來源。

**可重現輸出**

- [本次分析腳本](C:/project/quant_analysis/docs/research/analyze_dataset_relationship_20261010.py)：來源 hash、pair algebra、geometry 重建、cohort sensitivity、LODO/family holdout、legacy 排序與圖。
- `paired_inputs.csv`、`dataset_profiles.csv`、`rank_stability_variance.csv`、`within_context_correlations.csv`、`soy_cotton_contrasts.csv`：重算資料與所有組內／敏感度結果。
- `heldout_predictions.csv`、`heldout_summary.csv`：每 dataset、每 context、每模型的完整 prediction/error；未以事後最佳模型取代 comparator。
- `legacy_*`：舊 CSV 的 matched summaries、配置、排序與 soybean bit profiles。
- `run_config.json`：輸入／腳本 SHA-256、coverage、方法與限制。
- [文獻核對筆記](C:/project/quant_analysis/docs/research/dataset-quantization-mechanisms-sources-20261010.md)：五篇 primary-source 研究及適用邊界。

![Dataset retention 與 held-out 預測](C:/project/quant_analysis/five_dataset_features/analysis/dataset_relationship_20261010/dataset_relationship.png)

圖的 heatmap 是同 dataset 的重複 contexts；右圖 MAE 越低越好。BEiT FZ 的排除僅為明列的敏感度 cohort，完整結果另存；沒有視覺化或模型上的獨立驗證宣稱。
