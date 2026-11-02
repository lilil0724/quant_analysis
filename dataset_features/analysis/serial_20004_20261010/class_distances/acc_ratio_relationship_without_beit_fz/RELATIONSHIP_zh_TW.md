# 排除 BEiTv2 FZ 後，acc_ratio 與 class separation 的關係

## 結論

排除 BEiTv2 FZ 的 16 組結果後，保留 128 cells、8 個 model/mode、16 個 dataset names。
FP S 與 quant S 在全部 8 組均為正相關；各組 rho 的等權平均分別約 +0.695、+0.681。
但 raw inter-intra gap 的 CAL 關係仍弱，gap 保留率仍有不同方向，不能直接以 gap 大小／保留率判斷 accuracy。
這是同批資料的敏感度檢查，排除是使用者指定的事後選擇，未構成獨立驗證。

## 定義與方法

acc_ratio = quant accuracy / FP accuracy；G = inter-intra；S = (inter-intra)/(inter+intra)。
每個保留的 model/mode 內，使用 16 個 dataset 計算 average-tied-rank Spearman。
所有保留組別的 rho 已重新計算並確認與先前相同：因原本就是組內分析，刪除另一個 model/mode 不會改變這些組內數值。
另比較 9 組與 8 組的等權 rho 平均。這個平均不是把 128 cells 混在一起計算的 pooled correlation。
5,000 次 bootstrap 在 8 組間使用相同 dataset 抽樣索引，以保留同一 dataset 的重複量測關係；區間為 pointwise exploratory interval。

## 組內結果

| Model | Mode | FP G | Quant G | G_quant/G_FP | FP S | Quant S |
|---|---|---:|---:|---:|---:|---:|
| ViT-B/16 | FT | +0.838 | +0.818 | +0.341 | +0.838 | +0.803 |
| ViT-B/16 | FZ | +0.847 | +0.859 | -0.121 | +0.788 | +0.774 |
| ViT-B/16 | CAL | +0.085 | +0.106 | +0.647 | +0.468 | +0.471 |
| Swin-B | FT | +0.862 | +0.879 | +0.400 | +0.871 | +0.809 |
| Swin-B | FZ | +0.821 | +0.897 | -0.044 | +0.665 | +0.626 |
| Swin-B | CAL | -0.053 | -0.062 | +0.415 | +0.744 | +0.797 |
| BEiTv2-B | FT | +0.565 | +0.841 | +0.409 | +0.515 | +0.609 |
| BEiTv2-B | CAL | -0.091 | -0.229 | -0.274 | +0.671 | +0.559 |

## 排除前後的描述性摘要

| 指標 | 原 9 組 rho 平均 | 排除後 8 組 rho 平均 | 正向組數 | 控制 FP accuracy 後的 rho 平均 |
|---|---:|---:|---:|---:|
| fp_gap | +0.524 | +0.484 | 6/8 | +0.197 |
| quant_gap | +0.509 | +0.514 | 6/8 | +0.312 |
| gap_retention | +0.119 | +0.222 | 5/8 | +0.212 |
| fp_s_norm | +0.716 | +0.695 | 8/8 | +0.226 |
| quant_s_norm | +0.577 | +0.681 | 8/8 | +0.317 |

## 解讀

- FT/FZ 的 raw FP 與 quant gap，在保留的 5 組皆正向；CAL 的 3 組 raw gap 仍接近零或弱負相關。
- normalized S 避免只看 raw gap 的相對尺度問題，保留的 8 組 FP S/quant S 相關均正向。
- 排除 BEiTv2 FZ 後，quant S 的等權 rho 平均由 +0.577 上升至 +0.681；FP S 則由 +0.716 降至 +0.695。
- 控制 FP accuracy 後，FP S/quant S 的組間 rho 平均下降至約 +0.226/+0.317。
  ViT CAL 的調整後方向為負，Swin CAL 的 FP S 關係接近零，BEiTv2 CAL 的 FP/quant S 為約 +0.346/+0.274。
  所以 8/8 正向不代表 separation 已被證明是獨立於 baseline 的量化難度因子。
- gap 保留率只有 5/8 組正向；BEiTv2 CAL、ViT FZ、Swin FZ 仍負向。normalized S 保留率也非一致正向。
- 若看 ΔS = S_quant - S_FP，與 acc_ratio 在 8/8 組皆負向，平均 rho 約 -0.407；控制 FP accuracy 後平均變成約 +0.097。
  所以「S 的絕對大小與 accuracy retention 正相關」不能換成「S 的下降量直接決定 accuracy loss」；變化量也受 baseline 起點與尺度影響。
- Quant separation 是量化後觀察到的結果指標；FP separation 才是量化前可取得的候選預測指標。
- 若 FP/quant 指 G_FP/G_quant 的除法，它與表中 gap retention 互為倒數，Spearman 符號相反，不是新的獨立證據。

## 限制與輸出

同一 dataset 在模型／模式間重複，128 cells 不能視為 128 個獨立 dataset。
仍只有 16 個 dataset names；未宣稱多重比較校正後顯著性或因果。
CAL features 為第一個 raw-image backbone view，accuracy 使用 crop/flip 聚合，representation 邊界不同。
本次只用既有 compact distances，沒有重新執行 nano4 inference；來源 SHA256 在分析前後一致。
原始完整分析及來源沒有修改，本次輸出另外存放。

correlations.csv：72 個重新計算的組內 correlations、bootstrap interval、leave-one-dataset-out、partial-rank sensitivity。
group_relationship_summary.csv：各指標的排除前後 group-mean rho 與 dataset-block bootstrap interval。
distance_accuracy_pairs.csv：只包含保留的 128 組資料；method.json：排除範圍、方法、來源 SHA256。

![Correlation heatmap](C:/project/quant_analysis/five_dataset_features/analysis/serial_20004_20261010/class_distances/acc_ratio_relationship_without_beit_fz/correlation_heatmap.png)

![FP/quant gap scatter](C:/project/quant_analysis/five_dataset_features/analysis/serial_20004_20261010/class_distances/acc_ratio_relationship_without_beit_fz/fp_quant_gap_vs_acc_ratio.png)
