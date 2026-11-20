# PTQ / FGIR / UFGIR introduction draft

較完整的研究概述、分析主線與更新引用請見 [INTRODUCTION.md](INTRODUCTION.md)；本檔保留早期精簡稿與核對筆記。

Prepared: 2026-11-16, Asia/Taipei. Writing and targeted literature synthesis only. No experiments were run and no existing analysis results were changed. Numbered references below are local to this draft.

## Suggested framing

Question: **How well does aggressive post-training quantization preserve the discriminative information needed for fine-grained and ultra-fine-grained recognition?**

The contribution is a systematic characterization of dataset-associated degradation and candidate explanations, rather than a claim that dataset effects or FGIR quantization have never been studied. Recognition granularity here means semantic/visual label distinctions, not per-tensor/per-channel quantization granularity.

## English introduction

Quantization reduces the memory and computational requirements of neural networks by representing weights and activations at lower precision. Quantization-aware training (QAT) incorporates quantization into training or fine-tuning, whereas post-training quantization (PTQ) converts an already trained model without requiring a full quantization-aware training stage. PTQ therefore offers a practical deployment option when training resources or access to the original training data are limited, although calibration and, in some methods, local optimization are still required [1].

Substantial progress has improved quantized accuracy through quantizer design, such as twin uniform quantization in PTQ4ViT [2], and sensitivity-aware mixed-precision allocation, such as LampQ [3]. Related work also improves synthetic data for data-limited quantization [4], although such pipelines can include model fine-tuning. Complementary studies show that quantization effects vary across architectures and tasks [5], can disproportionately affect particular classes or population groups [6], and are related to distribution shift and prediction uncertainty [7]. These findings motivate examining the data conditions under which a given quantization configuration preserves task performance.

Fine-grained image recognition (FGIR) provides a useful setting for this question because successful classification depends on distinguishing visually similar subcategories. Ultra-fine-grained image recognition (UFGIR) extends this challenge to distinctions, such as cultivar identity, that can be difficult even for human experts [8]. However, the relationship between recognition granularity and accuracy retention under aggressive low-bit PTQ remains insufficiently characterized, particularly across FGIR and UFGIR datasets.

In this work, we investigate quantization robustness across FGIR and UFGIR datasets under matched architectures, checkpoint types, and quantization configurations. We examine how accuracy retention changes with bit width and quantization group size, and whether differences in prediction-margin distributions and feature geometry help account for the observed degradation patterns. This analysis aims to clarify when aggressive PTQ preserves the distinctions required by a recognition task and when further adaptation may be necessary.

The last paragraph describes the proposed study scope, not completed or validated results. Retain only analyses that are actually included in the eventual paper.

## Concise slide version

- **PTQ motivation:** lower model memory and computation costs without a full QAT stage.
- **Method progress:** improved quantizers and sensitivity-aware precision allocation.
- **Uneven degradation:** quantization effects vary across models, tasks, classes and data distributions.
- **FGIR / UFGIR motivation:** recognition depends on subtle visual distinctions whose preservation under aggressive PTQ deserves evaluation.
- **Research question:** do FGIR and UFGIR datasets exhibit different degradation patterns, and which measurable properties help explain them?

Suggested gap sentence:

> The relationship between recognition granularity and accuracy retention under aggressive post-training quantization remains insufficiently characterized, particularly across fine-grained and ultra-fine-grained recognition datasets.

This is a synthesis from a targeted review, not an exhaustive priority or novelty claim.

## Citation map

| ID | Paper / verified publication | Zotero status | What it supports | What it does not establish |
|---|---|---|---|---|
| [1] | Gholami et al., [A Survey of Quantization Methods for Efficient Neural Network Inference](https://arxiv.org/abs/2103.13630), 2021 preprint / book chapter | Not present in queried personal-library top-item inventory | QAT/PTQ concepts and resource tradeoffs | That PTQ has zero optimization or universally simple hardware deployment |
| [2] | Yuan et al., [PTQ4ViT: Post-Training Quantization for Vision Transformers with Twin Uniform Quantization](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/7856_ECCV_2022_paper.php), ECCV 2022 | Present, KPDTMCU5; local PDF inspected | ViT-specific quantizer design and Hessian-guided calibration | Cross-dataset FGIR/UFGIR sensitivity; local Zotero date 2024 is the stored preprint/version date, not the conference year |
| [3] | Kim et al., [LampQ: Towards Accurate Layer-wise Mixed Precision Quantization for Vision Transformers](https://arxiv.org/abs/2511.10004), AAAI 2026 / 2025 preprint | Present, 9S27BCTM; local PDF inspected | Fisher-based sensitivity and precision allocation | That allocation granularity is the same concept as recognition granularity |
| [4] | Zhong et al., [Fine-grained Data Distribution Alignment for Post-Training Quantization](https://doi.org/10.1007/978-3-031-20083-0_5), ECCV 2022 | Present, PJUUSX4U; local PDF methods inspected | Class-wise synthetic distribution alignment and within-class diversity | FGIR/UFGIR task analysis; strictly training-free quantization. Sections 3.4–3.5 explicitly update the quantized model and generator |
| [5] | Ozdemir and Li, [Bridging the Reliability Gap: INT8 Quantization Effects on Discrimination and Calibration in Medical Imaging](https://proceedings.mlr.press/v333/ozdemir26a.html), CHIL 2026 | Not present in queried personal-library top-item inventory | Three datasets / eight backbones and task-architecture-dependent behavior | Isolation of recognition granularity |
| [6] | Liu et al., [Understanding the Unfairness in Network Quantization](https://proceedings.mlr.press/v267/liu25au.html), ICML 2025 | Not present in queried personal-library top-item inventory | Uneven degradation across classes / attribute groups | FGIR-versus-UFGIR dataset contrasts |
| [7] | Hu et al., [Towards Understanding Model Quantization for Reliable Deep Neural Network Deployment](https://mpapad.github.io/publications/pdfs/CAIN2023_quantization.pdf), CAIN 2023 | Present, GNFFMGUE / MA624YWG; one local duplicate PDF inspected | Distribution-shift analysis and small output-probability margins characterizing prediction disagreements | That small margins are unique to UFGIR or that disagreements necessarily cause an accuracy decrease |
| [8] | Yu et al., [Benchmark Platform for Ultra-Fine-Grained Visual Categorization Beyond Human Performance](https://openaccess.thecvf.com/content/ICCV2021/papers/Yu_Benchmark_Platform_for_Ultra-Fine-Grained_Visual_Categorization_Beyond_Human_Performance_ICCV_2021_paper.pdf), ICCV 2021 | Present, Y4XNZMXR; local PDF text extraction malformed, verified against the primary CVF paper | UFG definition and cultivar-level distinctions difficult for human experts | Lower logit margins or worse quantization robustness |

## Useful additional Zotero papers

- **Fang et al., Learning Contrastive Self-Distillation for Ultra-Fine-Grained Visual Categorization Targeting Limited Samples**, TCSVT 2024, [arXiv](https://arxiv.org/abs/2311.06056), Zotero CL7EGYPA. CSDNet supports motivation around subcategory-specific discrepancies and limited samples. It is not quantization evidence.
- **Xia et al., An Underexplored Dilemma between Confidence and Calibration in Quantized Neural Networks**, NeurIPS 2021 workshop, [arXiv](https://arxiv.org/abs/2111.08163), Zotero YWQQMHYK. Distinguishes how easily predictions swap from whether a swap harms accuracy. Useful for motivating margin and damage/rescue analysis. Its probability calibration is different from the calibration step used to fit quantizers.
- **Kim et al., SynQ: Accurate Zero-shot Quantization by Synthesis-aware Fine-tuning**, ICLR 2025, Zotero GWLL4N65. Supports synthetic-data quality, CAM alignment and soft-label fine-tuning, not a claim of no training. Sections 3–5 and appendix C.1 explicitly fine-tune the quantized network; standard experiments use 100 epochs.
- **Chang et al., Why Do Some Inputs Break Low-Bit LLM Quantization?**, [arXiv](https://arxiv.org/abs/2506.12044), Zotero KDP5B8AS. Optional related-work evidence for input-dependent error amplification. It does not directly validate a vision mechanism.
- **Marchisio et al., How Does Quantization Affect Multilingual LLMs?**, [arXiv](https://arxiv.org/abs/2407.03211), Zotero FALX52GA. Optional cross-domain evidence that aggregate quantization quality can mask differences between languages/tasks. Avoid letting LLM work dominate a vision introduction.

## FGIR / UFGIR angle and evidentiary limits

Recommended chain: subtle class distinctions motivate inspecting representation preservation; quantization perturbs model computation; margin and feature geometry are measurable candidate links; the relationship is then tested across FGIR/UFGIR datasets. The arrows are hypotheses, not established causal effects.

Do not infer that UFGIR necessarily has smaller margins from the definition of UFGIR. Visual difficulty for people, representation separability, calibrated probabilities and model logit margins are different quantities.

Distinguish prediction disagreement from damage to an initially correct prediction and rescue of an initially incorrect prediction. More swaps alone do not imply greater net accuracy loss.

For this repository, use the matched **unquantized AMP baseline**, not the label "FP32" unless the underlying inference actually used FP32. Accuracy retention is A_quantized / A_unquantized for the same dataset and compatible checkpoint/evaluation settings. Also report absolute accuracy loss to make differing baseline accuracies visible. Quantization robustness here means task-performance preservation under quantization, not corruption/adversarial robustness.

The same architecture or checkpoint type across datasets does not imply identical learned weights. Comparing different bird/vehicle/pet datasets with cultivar datasets also changes domain, acquisition, sample count, class count, and possibly pretraining alignment. Current fixed-group contrasts support associations. A causal test of label granularity would need a verified label hierarchy on shared images and carefully controlled training/evaluation, including sample and class-count considerations.

The strongest central claim available without such controls is **dataset-conditioned quantization robustness in FGIR and UFGIR**, with recognition granularity as a motivating factor and candidate correlate.

## Local sources actually inspected

Stored filenames are preserved below. PDF extraction was read selectively for relevant methods and experiments, not as a claim that every appendix was fully reviewed.

- PTQ4ViT: `C:\Users\kyle\Zotero\storage\HDBMF3FS\Yuan …等 - 2024 - PTQ4ViT Post-training quantization for vision transformers with twin uniform quantization.pdf`, methods / calibration passages.
- LampQ: `C:\Users\kyle\Zotero\storage\TEXKDIAT\arxiv_2511.10004.pdf`, sensitivity / allocation passages.
- FDDA: `C:\Users\kyle\Zotero\storage\LY2YQXYI\Zhong …等 - 2022 - Fine-grained Data Distribution Alignment for Post-Training Quantization.pdf`, especially printed pp. 77–78, sections 3.3–3.5.
- CAIN: `C:\Users\kyle\Zotero\storage\8RYETWZR\Hu …等 - 2023 - Towards Understanding Model Quantization for Reliable Deep Neural Network Deployment.pdf`, study design and RQ3, PDF pp. 8–9.
- UFG benchmark: `C:\Users\kyle\Zotero\storage\HA65F7U2\Yu …等 - 2021 - Benchmark Platform for Ultra-Fine-Grained Visual Categorization Beyond Human Performance.pdf`; malformed local text extraction, primary CVF PDF used to verify definition.
- CSDNet: `C:\Users\kyle\Zotero\storage\QMB35H9P\Fang …等 - 2024 - Learning Contrastive Self-Distillation for Ultra-Fine-Grained Visual Categorization Targeting Limite.pdf`, introduction and SSDP method.
- Confidence/calibration: `C:\Users\kyle\Zotero\storage\EB8E8GAT\Xia …等 - 2021 - An Underexplored Dilemma between Confidence and Calibration in Quantized Neural Networks.pdf`, PDF pp. 2–3.
- SynQ: `C:\Users\kyle\Zotero\storage\LQ2TF3N8\Kim …等 - 2025 - SYNQ ACCURATE ZERO-SHOT QUANTIZATION BY SYNTHESIS-AWARE FINE-TUNING.pdf`, sections 3–5 and appendix C.1.
- Input-dependent LLM error: `C:\Users\kyle\Zotero\storage\X5RRBSAC\Chang …等 - 2025 - Why Do Some Inputs Break Low-Bit LLM Quantization.pdf`, setup and residual-state analysis.
- Multilingual LLMs: `C:\Users\kyle\Zotero\storage\HKGFEE8C\Marchisio …等 - 2024 - How Does Quantization Affect Multilingual LLMs.pdf`, evaluation and language-dependent results.

Primary external sources were used when the requested paper was absent from the queried personal Zotero library or to verify publication metadata. No Zotero items were created or modified. An absent personal-library entry is not a claim about unqueried group libraries.

Further overlap checks: [research-gap memo](ptq_fgir_intro_gap_check_20261116.md).
