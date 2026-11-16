# PTQ / FGIR introduction: research-gap check

Checked: 2026-11-16 (Asia/Taipei). This is a targeted literature check for introduction wording, not an exhaustive novelty review or an experiment report. Local Zotero papers are being inspected separately in the parent task; this note checks primary-source overlap and source scope. No analysis inputs or results were changed.

## Recommended positioning

Lead with the unresolved **relationship between downstream recognition granularity and accuracy retention under aggressive PTQ**, particularly for ViTs. Avoid claiming that dataset effects, quantization on FGIR, or unequal degradation have never been studied.

Suggested English transition:

> Prior studies have documented heterogeneous quantization effects across architectures, data distributions, and population groups. However, the relationship between recognition granularity and accuracy retention under aggressive post-training quantization remains insufficiently characterized, particularly across fine-grained and ultra-fine-grained recognition datasets. We therefore examine whether degradation patterns differ systematically between these settings under matched model and quantization configurations.

“Insufficiently characterized” is a synthesis of this targeted review, not a fact established by any one citation. “We examine whether” states a question rather than assuming that UFGIR must be less robust.

## Relevant primary sources

| Source | Verified scope | Role in introduction / boundary |
|---|---|---|
| [Hu et al., Towards Understanding Model Quantization for Reliable Deep Neural Network Deployment (CAIN 2023)](https://mpapad.github.io/publications/pdfs/CAIN2023_quantization.pdf) | Four datasets, eight architectures and 42 shifted sets. Studies quantization disagreements and distribution shift; reports that smaller output margins characterize disagreements. | Strong precedent for data-dependent behavior and a margin-based motivation. Prevents a broad “dataset effects are ignored” claim. Does not directly establish FGIR/UFGIR granularity effects. |
| [Liu et al., Understanding the Unfairness in Network Quantization (ICML 2025)](https://proceedings.mlr.press/v267/liu25au.html) | PTQ/QAT disparity across attribute groups/classes, with imbalanced and balanced datasets and ResNet/VGG models. | Supports the distinction between aggregate performance and uneven group degradation. Formal publication year is 2025. Group disparity is not equivalent to cross-dataset recognition granularity. |
| [Yuan et al., Benchmarking the Reliability of Post-training Quantization: a Particular Focus on Worst-case Performance (2023)](https://arxiv.org/abs/2303.13003) | Examines calibration distribution, calibration paradigm, augmentation/sampling and worst-case group reliability under shifts and noise. | Cite for calibration/distribution effects and the inadequacy of average accuracy alone. Do not describe it as exclusively a class-drop paper. |
| [Ozdemir and Li, Bridging the Reliability Gap: INT8 Quantization Effects on Discrimination and Calibration in Medical Imaging (CHIL 2026)](https://proceedings.mlr.press/v333/ozdemir26a.html) | Three medical imaging datasets and eight pretrained backbones under FP32, FP16, INT8 PTQ and INT8 QAT. Evaluates discrimination, calibration and efficiency. | Direct evidence that cross-dataset/architecture comparisons already exist. It does not isolate recognition granularity. |
| [Yu et al., Benchmark Platform for Ultra-Fine-Grained Visual Categorization Beyond Human Performance (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Yu_Benchmark_Platform_for_Ultra-Fine-Grained_Visual_Categorization_Beyond_Human_Performance_ICCV_2021_paper.pdf) | Defines UFG visual categorization through cultivar-level distinctions that can be difficult even for human experts, and supplies an evaluation benchmark. | Supports the FGIR-to-UFGIR motivation and task definition. Does not prove that UFGIR has smaller model margins or higher quantization sensitivity. |
| [Chang et al., Why Do Some Inputs Break Low-Bit LLM Quantization? (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.168/) | Input-dependent error under 3–4-bit LLM weight quantization; examines residual activations, error amplification and MLP gates. | Optional cross-domain evidence that vulnerability is not uniform across inputs. LLM mechanisms cannot be assumed to transfer to ViTs. |
| [Marchisio et al., How Does Quantization Affect Multilingual LLMs? (Findings of EMNLP 2024)](https://aclanthology.org/2024.findings-emnlp.935/) | Different languages and tasks suffer different degradation; human evaluation can reveal damage that automatic metrics underestimate. | Optional precedent for task/data-dependent degradation. Not a primary citation for visual granularity. |

## What the FGIR / UFGIR angle can claim

- **Motivation:** preserving distinctions among visually similar subclasses is relevant to low-bit deployment; UFGIR offers a demanding setting to investigate which distinctions survive quantization. The cited UFG benchmark establishes subtle distinctions, not quantization sensitivity.
- **Descriptive question:** do FGIR and UFGIR datasets show different accuracy-retention curves when model family, checkpoint regime and quantization configuration are matched?
- **Mechanism question:** do differences in full-precision prediction margins, class geometry or competitor structure help account for those patterns? Treat each as a testable explanation, not an established consequence of semantic granularity.
- **Stronger causal design:** use the same images with verified coarse/fine label hierarchies, while controlling training/evaluation protocol and sample counts. Comparing bird/dog/vehicle datasets against soybean cultivar datasets entangles domain, labels, sample counts, acquisition conditions and granularity.

## Wording to avoid

- “PTQ requires no training” supported by synthesis-aware fine-tuning methods. The parent task's local-paper check confirms that SynQ fine-tunes the quantized model, while FDDA jointly updates its quantized model and generator. Keep those methods in a broader synthetic-data quantization discussion; do not use them as proof of a strictly training-free PTQ pipeline. Distinguish avoiding full QAT/retraining from avoiding all optimization.
- “No previous work has studied dataset-dependent quantization.” The sources above directly contradict that breadth.
- “Quantization has not been studied for FGIR.” This targeted review does not establish the absence of prior FGIR quantization work; avoid this blanket claim.
- “UFGIR inherently has lower logit margins, therefore it is more difficult to quantize.” Neither step follows from the benchmark definition; both need empirical verification, and cross-dataset association does not isolate causality.
- “Robustness” without definition. Define **quantization robustness** as preservation of task performance relative to the matched unquantized baseline, so it is not confused with adversarial or corruption robustness. Check baseline precision and evaluation settings explicitly.

## Review limitations

Searches were targeted to FGVC/FGIR/UFGIR quantization, dataset/granularity effects, and closely related reliability work. A search result for an unidentified medical-segmentation appendix mentioned dataset scale and class-count comparisons, but the primary PDF was blocked by an OpenReview browser challenge; it is excluded from the evidence above. Search absence is not evidence of absence. No server experiments or new causal analyses were performed.
