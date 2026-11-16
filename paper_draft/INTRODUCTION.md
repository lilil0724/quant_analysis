# Detailed introduction draft

更新：2026-11-16，Asia/Taipei。

本稿整合研究背景、文獻定位、FGIR／UFGIR 切入點與具體分析。英文中的研究概述與 contributions 描述論文預定涵蓋的工作；新版 CAL、完整跨資料集矩陣及個別分析仍須依實際產物驗收。本文未加入尚未確認的結果數字，也未宣稱已證實因果機制。

暫定題目：**Dataset-Dependent Robustness of Low-Bit Post-Training Quantization in Fine-Grained and Ultra-Fine-Grained Recognition**

## English introduction

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

## 中文說明：各段的作用

| 段落 | 內容 | 為什麼放在這裡 |
|---|---|---|
| 1 | QAT／PTQ 與資源成本 | 先說明選擇 PTQ 的實務動機，不把 PTQ 說成零校準或零優化。 |
| 2 | Quantizer、mixed precision、合成資料 | 交代提升量化準確率的既有方法，並明確把需要微調的 SynQ 放在較廣義的量化研究中。 |
| 3 | Dataset／architecture／group／sample 差異 | 承認已有不均等退化與資料相依研究，讓 gap 建立在既有成果上。 |
| 4 | FGIR／UFGIR 動機 | 從「細微判別資訊能否保留」切入，不先假設 UFGIR 一定較差。 |
| 5 | 研究缺口與問題 | 明確提出退化曲線、排名一致性、可測量解釋三個問題。 |
| 6 | 整體 accuracy 與排名分析 | 交代 baseline 配對、retention、absolute loss、bit／group sweep 與 FGIR／UFGIR contrasts。 |
| 7 | 表徵分析 | 區分量化前的 feature geometry 與量化後的 CKA／geometry 變化，並交代 W3/G128 的範圍。 |
| 8 | 決策分析 | 串起 normalized margin、logit offsets、competitors、swaps，以及 damage／rescue。 |
| Contributions | 三項研究工作 | 總結分析層次；不把既有 CKA 或 t-SNE 宣稱為新提出的方法。 |

本稿適合作為較完整的初稿。篇幅有限時，可把第二段的具體方法細節、PCA／t-SNE，以及部分 protocol 細節移到 Related Work／Methods。

## Citation map

編號 [1]–[8] 延續原 intro；[9]–[14] 是本詳細版新增的引用編號。**引用編號不等於 PDF 檔名前綴**。

| 引用 | 論文與來源 | 本機 PDF | 引用用途 |
|---|---|---|---|
| [1] | Gholami et al., [A Survey of Quantization Methods for Efficient Neural Network Inference](https://arxiv.org/abs/2103.13630), 2021 preprint / book chapter | [PDF](pdfs/01_Gholami_2021_Quantization_Survey.pdf) | QAT／PTQ 定義與成本。 |
| [2] | Yuan et al., [PTQ4ViT: Post-Training Quantization for Vision Transformers with Twin Uniform Quantization](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/7856_ECCV_2022_paper.php), ECCV 2022 | [PDF](pdfs/02_Yuan_2022_PTQ4ViT.pdf) | ViT quantizer 設計與 calibration。 |
| [3] | Kim et al., [LampQ: Towards Accurate Layer-wise Mixed Precision Quantization for Vision Transformers](https://arxiv.org/abs/2511.10004), AAAI 2026 / 2025 preprint | [PDF](pdfs/03_Kim_LampQ.pdf) | Sensitivity-aware mixed precision。 |
| [4] | Zhong et al., [Fine-grained Data Distribution Alignment for Post-Training Quantization](https://doi.org/10.1007/978-3-031-20083-0_5), ECCV 2022 | [PDF](pdfs/04_Zhong_2022_FDDA.pdf) | Class-conditional synthetic distribution；不是 FGIR／UFGIR 的實驗證據，也不是本專案 cosine distance 公式的來源。 |
| [5] | Ozdemir and Li, [Bridging the Reliability Gap: INT8 Quantization Effects on Discrimination and Calibration in Medical Imaging](https://proceedings.mlr.press/v333/ozdemir26a.html), CHIL 2026 | [PDF](pdfs/05_Ozdemir_2026_Bridging_Reliability_Gap.pdf) | 不同 dataset／backbone 的量化差異。 |
| [6] | Liu et al., [Understanding the Unfairness in Network Quantization](https://proceedings.mlr.press/v267/liu25au.html), ICML 2025 | [PDF](pdfs/06_Liu_2025_Unfairness_Network_Quantization.pdf) | 類別／群體的不均等退化。 |
| [7] | Hu et al., [Towards Understanding Model Quantization for Reliable Deep Neural Network Deployment](https://doi.org/10.1109/CAIN58948.2023.00015), CAIN 2023 | [PDF](pdfs/07_Hu_2023_Reliable_Quantization_Deployment.pdf) | Distribution shift、output margin 與 disagreements。原文的機率 margin 不等於本研究的 normalized logit margin。 |
| [8] | Yu et al., [Benchmark Platform for Ultra-Fine-Grained Visual Categorization Beyond Human Performance](https://openaccess.thecvf.com/content/ICCV2021/papers/Yu_Benchmark_Platform_for_Ultra-Fine-Grained_Visual_Categorization_Beyond_Human_Performance_ICCV_2021_paper.pdf), ICCV 2021 | [PDF](pdfs/08_Yu_2021_Ultra_Fine_Grained_Benchmark.pdf) | UFGIR 定義與 cultivar-level 判別挑戰。 |
| [9] | Kornblith et al., [Similarity of Neural Network Representations Revisited](https://proceedings.mlr.press/v97/kornblith19a.html), ICML 2019 | [PDF](pdfs/15_Kornblith_2019_CKA.pdf) | Centered linear CKA 的方法依據。 |
| [10] | Xia et al., [An Underexplored Dilemma between Confidence and Calibration in Quantized Neural Networks](https://arxiv.org/abs/2111.08163), NeurIPS 2021 workshop | [PDF](pdfs/10_Xia_2021_Confidence_Calibration_Quantization.pdf) | 區分 prediction swap 與 accuracy consequence 的動機。 |
| [11] | van der Maaten and Hinton, [Visualizing Data using t-SNE](https://www.jmlr.org/papers/v9/vandermaaten08a.html), JMLR 2008 | [PDF](pdfs/16_van_der_Maaten_2008_tSNE.pdf) | t-SNE 方法；可移至 Methods／圖說。 |
| [12] | Fang et al., [Learning Contrastive Self-Distillation for Ultra-Fine-Grained Visual Categorization Targeting Limited Samples](https://arxiv.org/abs/2311.06056), TCSVT 2024 | [PDF](pdfs/09_Fang_2024_CSDNet_Ultra_FGVC.pdf) | Subcategory-specific discriminative information 與 limited samples。 |
| [13] | Kim et al., SynQ: Accurate Zero-shot Quantization by Synthesis-aware Fine-tuning, ICLR 2025 | [PDF](pdfs/11_Kim_2025_SynQ.pdf) | 合成資料與量化後微調；不能支持完全不需訓練。 |
| [14] | Yuan et al., [Benchmarking the Reliability of Post-training Quantization: a Particular Focus on Worst-case Performance](https://arxiv.org/abs/2303.13003), 2023 preprint | [PDF](pdfs/14_Yuan_2023_PTQ_Worst_Case_Reliability.pdf) | Calibration distribution 與 worst-case group reliability。 |

## Optional paragraph: margin matching

下句僅在新版 matching 產物驗收、確認 overlap／balance、並核對分組後加入決策分析段末：

> We further assess whether between-group differences in accuracy retention persist after aligning their baseline normalized-margin distributions over a common support region.

現行 matching 為 soy＋cotton／other，不可未經核對就改稱 FGIR／UFGIR。這種比較也不會單獨建立 margin 或 granularity 的因果效果。

## Before submission

- 依最終 coverage 填入 dataset 數、模型名稱、FT／FZ／CAL 定義、bit／group grid 與 seed 範圍；目前不把 16 × 3 × 3 寫成已完成矩陣。
- 將完整 accuracy sweep 與固定 W3/G128 feature／logit 分析分開。若 protocol 變動，同步修訂第七段。
- Retention 使用實際配對的 unquantized AMP baseline，不因 legacy 欄名而寫成 FP32。
- CKA／class geometry／PCA／t-SNE 的量化前屬性與量化後變化分開解讀，不能把事後變化當成已驗證的量化前預測器。
- 不同 datasets 的權重、domain、類別數、樣本數與 pretraining alignment 可能同時不同。跨資料集 association 不等於 label-granularity causal effect。
- 完整驗收後，在 contributions 前加入 1–2 句最重要的實證發現，並保留條件與範圍。不要在結果未確認前加入 universal ranking、UFGIR 必然較差或 margin 已解釋全部差異等結論。
- 本稿沒有引回使用者先前要求排除的論文；PDF 仍僅保存在 Git 忽略的本機資料夾。

## Evidence and source resolution

原有 14 份本機 PDF 的來源與 SHA-256 見 [references.json](references.json)。新增 CKA／t-SNE 時，唯讀查詢 Zotero 個人文庫的 66 個 top-level items 未找到匹配標題，故使用 PMLR／JMLR 官方 PDF；沒有修改 Zotero。

Own-analysis scope was checked against [analysis checklist](../docs/paper_analysis_checklist_20261115.md), [accuracy README](../quant_acc/README.md), [feature README](../dataset_features/README.md), [confidence README](../dataset_confidence/README.md), [FGIR/UFGIR group README](../dataset_static_feature/README.md), and [ranking README](../dataset_ranking_robustness/README.md). These establish code capabilities and intended contracts, not completion of new Nano4 experiments.
