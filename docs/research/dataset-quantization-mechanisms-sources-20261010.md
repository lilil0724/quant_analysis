# Dataset-dependent quantization mechanisms: primary-source checks

核對日期：2026-10-10。本文只整理原始文獻及可檢驗推論；沒有讀取或分析本次實驗數據，也沒有執行推論或伺服器工作。

## 可以支持的結論

量化後的表現同時取決於擾動、資料經模型後的特徵方向，以及分類器對這些方向的敏感度。即使固定同一組量化權重，不同輸入分布也可能產生不同輸出誤差。這支持研究 dataset dependence 的合理性，不能直接證明本資料中的 dataset 效應來自哪個機制。

必須分開 weight-only 與 weight＋activation。前者仍會受浮點 activation 的大小與方向影響；後者另外會受到量化範圍、clipping、通道尺度及 attention 機率分布影響。不能用後者的文獻消融直接解釋沒有 activation quantization 的 HQQ 條件。

## 五篇最相關的原始研究

### 1. Optimal Brain Compression — Frantar, Singh, Alistarh, NeurIPS 2022

- 原始來源：[會議 PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/1caf09c9f4e6b0150b06a07e77f2710c-Paper-Conference.pdf)、[arXiv](https://arxiv.org/abs/2208.11580)。核對位置：§3 Eq. (2)、§4 的 `H = 2XXᵀ`、§5 的 OBQ。
- 範圍：OBQ 的核心是 weight quantization；本文也涵蓋 pruning，不應將兩者實驗混為同一證據。
- 證據：線性層重建目標為 `||WX − WqX||²`，其 Hessian 由 calibration inputs 的 `XXᵀ` 決定。它明確顯示：相同的權重差 `ΔW`，在不同特徵二階矩上會有不同的輸出重建成本。
- 適用限制：這是 layer reconstruction 的數學關係及演算法證據，並非跨資料集的 accuracy_ratio 因果試驗；HQQ 不等於 OBQ。由此推得本資料可能存在特徵方向／能量差異，是待驗證假說。

### 2. PTQ4ViT — Yuan et al., ECCV 2022

- 原始來源：[會議 PDF](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720190.pdf)、[arXiv](https://arxiv.org/abs/2111.12293)。核對位置：§3.3、Fig. 5、Eq. (6–7)、§4.3 Table 3。
- 範圍：ViT PTQ，量化 weights 與 activations，含 post-Softmax／post-GELU 的 twin uniform quantization。
- 證據：局部 MSE、cosine 等誤差未必選到使任務損失最小的 scale；作者用梯度平方加權的 Hessian-guided proxy，並以消融顯示改善。這支持「相同大小的特徵誤差，任務後果可以不同」。
- 適用限制：主要驗證為 ImageNet 的特定 ViT／bit 條件。二階 Taylor 與 Fisher 對角替代都是近似，低 bit 的大擾動可能違反小擾動假設；proxy 亦非真實 accuracy。這些消融支持該方法的局部介入效果，沒有證明本資料的 dataset 因果效應。

### 3. FQ-ViT — Lin et al., IJCAI 2022

- 原始來源：[會議頁](https://www.ijcai.org/proceedings/2022/164)、[會議 PDF](https://www.ijcai.org/proceedings/2022/0164.pdf)。核對位置：§3.2–3.3、Fig. 2–3、§4。正式會議為 IJCAI 2022。
- 範圍：fully quantized ViT，主要設定為 8-bit weights／activations，4-bit attention maps。
- 證據：LayerNorm 輸入存在嚴重通道 range 差異，共用 layer-wise scale 會犧牲小 range 通道解析度；attention 值高度非均勻，uniform quantization 對小值不利。PTF 與 Log-Int-Softmax 分別處理這些問題。
- 適用限制：其瓶頸包含 activation 及 LayerNorm／Softmax 量化；若本實驗只有 Linear weights 被量化，不能宣稱同一 activation clipping 機制已發生。模型內的分布診斷及方法消融，也不等於不同 dataset 之間的機制證明。

### 4. RepQ-ViT — Li et al., ICCV 2023

- 原始來源：[arXiv 全文](https://arxiv.org/html/2212.08254)、[arXiv 摘要](https://arxiv.org/abs/2212.08254)、[會議 PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_RepQ-ViT_Scale_Reparameterization_for_Post-Training_Quantization_of_Vision_Transformers_ICCV_2023_paper.pdf)。CVF 直接開啟失敗；本次核對使用 arXiv 全文。位置：§3、§4.4 Tables 3–4。
- 範圍：weight＋activation PTQ；特別處理 post-LayerNorm 與 post-Softmax 分布。
- 證據：post-LayerNorm 通道差異、post-Softmax power-law 分布會限制低 bit 表示。W4A4 消融中，DeiT-S 的 LayerNorm layer-wise／channel-wise／reparameterized 方法分別為 33.17／70.28／69.03% Top-1。
- 適用限制：這是改變量化表示的介入證據，支持該設定的 activation 分布瓶頸，不能推出本資料有同等數值或相同根因。FQ-ViT 的 LayerNorm「輸入」與本篇的「輸出」是不同位置，不應合併成一個未定位的 outlier 指標。

### 5. Certifiably Quantisation-Robust training and inference of Neural Networks — Dang et al., AISTATS 2025

- 原始來源：[PMLR 會議頁](https://proceedings.mlr.press/v258/dang25a.html)、[PMLR 提供的 PDF](https://raw.githubusercontent.com/mlresearch/v258/main/assets/dang25a/dang25a.pdf)。本次核對 publisher abstract 及搜尋索引的原文段落；PDF 直接讀取被工具 content-type 限制，未逐頁核對全部證明。
- 範圍：inputs、parameters、activations 的有界量化擾動；含 robust training，並非 ViT HQQ PTQ 專屬研究。
- 證據：將量化 robustness 定義為在允許量化族內限制最壞輸出偏差，透過 bilinear optimization 與 interval bounds 提供保證；測試含 MNIST、F-MNIST、CIFAR10。它支持區分「某次量化觀測準確率」與「有證明的擾動容忍度」。
- 適用限制：沒有提供本資料的 certificate，也不能用一般 worst-case bound 代替 ViT 的實際 paired flips。此來源只作 robustness 定義及證據等級參照。

## Margin 與幾何：本報告的直接推導，非上述論文的資料結論

令 FP logits 為 `z(x)`，量化 logits 為 `zq(x) = z(x) + δz(x)`，FP top-1 類別為 `a`。定義原始 top-1 margin：

```text
m(x) = z_a(x) − max_{j≠a} z_j(x)
```

若 `||δz(x)||∞ ≤ ε(x)` 且 `m(x) > 2ε(x)`，則量化後 top-1 不變，因為每個 pairwise logit gap 最多下降 `2ε(x)`。這是充分條件；不滿足不代表一定翻轉。`m / std(logits)` 的 Q10 不是這個 certificate：若使用 normalized margin，擾動大小也須在同一尺度比較，且仍缺少每筆資料的擾動界。

最後線性 head 若保持固定，feature shift `δh` 對類別 `a` 與 `j` 的 gap 影響為 `(w_a − w_j)ᵀδh`。因此 feature L2、cosine drift、CKA 只描述總體變動，無法單獨判定變動是否對準分類邊界。若 head 也量化，還必須加入 head weight 與 bias 的擾動項。

accuracy 的下降亦不是 prediction swap rate：原本正確變錯是 damage，原本錯誤變對是 rescue。配對樣本完全一致時，`accuracy_FP − accuracy_Q = damage_rate − rescue_rate`；`accuracy_ratio` 另除以 FP accuracy，所以同一絕對下降可能呈現不同 retention。

## 對後續分析的優先順序

1. 先在固定 model／checkpoint kind／bit／group 的 matched cells 比較 dataset；同時報 baseline accuracy、絕對下降與 ratio。dataset 常與 label space、checkpoint／fine-tuning history 綁定，不能稱為 dataset intrinsic effect。
2. 若有同樣本 logits，先檢驗 FP margin 與 paired damage／rescue，並比較實際 logit perturbation 與 margin。低 margin 本身只代表較小的容忍空間；擾動的方向與大小仍決定是否翻轉。
3. 若有 paired features，檢驗 class-separation／head 方向上的 drift 是否比總體 CKA 或 L2 更接近 damage。可分解 input second moment 與 `ΔW` 的方向關係，但不能把 covariance association 直接命名為因果。
4. activation outliers／clipping 只在已確認有 activation quantization 的條件檢驗，並固定量化位置與 scale granularity。weight-only → activation 的比較需使用相同樣本、checkpoint 及 weight 設定。
5. 相關性、樣本內消融與跨 dataset 預測是不同證據；因果機制需介入，獨立預測需 held-out dataset／family 驗證。以上是待做檢驗，不是本次已取得的結果。
