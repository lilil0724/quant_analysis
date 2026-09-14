# Related work: sample margin and quantization-induced prediction swaps

Research date: 2026-09-13

## Bottom line

There is a direct published precedent for the central sample-level mechanism.
Hu et al. (CAIN 2023) define a quantization *disagreement* as a changed
prediction, use the full-precision model's top-1 minus top-2 probability margin,
and show that this margin discriminates disagreement from normal inputs with
AUC-ROC values from 0.63 to 0.97. Therefore, this project must **not** claim to
be the first to show that low-margin samples are more likely to change prediction
after quantization.

The defensible contribution is narrower: a replication and extension to
fine-tuned ViT image classifiers under HQQ at 3/4 bits, using a scale-normalized
logit margin, within-dataset AUROC, dataset-equal aggregation, margin-decile
curves, and a separate dataset-level Q10 screening analysis. I did not find a
paper that combines all of those elements, but this scoped search is not enough
to establish novelty.

## Direct precedent

### Hu et al. (CAIN 2023): the closest prior experiment

[Towards Understanding Model Quantization for Reliable Deep Neural Network
Deployment](https://doi.org/10.1109/CAIN58948.2023.00015), Qiang Hu, Yuejun
Guo, Maxime Cordy, Xiaofei Xie, Wei Ma, and Mike Papadakis, 2nd IEEE/ACM
International Conference on AI Engineering, 2023. An accessible
[author-hosted paper](https://mpapad.github.io/publications/pdfs/CAIN2023_quantization.pdf)
is available.

- The paper calls an input a *disagreement* when the compressed/quantized model
  and original model predict different labels. This is the same endpoint as our
  `prediction_swap`.
- It explicitly proposes the mechanism that quantization moves the decision
  boundary and that inputs near the original boundary may cross it.
- Its margin is the difference between the original model's top-1 and top-2
  predicted probabilities. It trains one-variable logistic regressions from
  uncertainty scores and evaluates their ability to distinguish disagreements
  with AUC-ROC. Margin is best in 27 of 30 cases; the paper summarizes its
  margin-based AUC-ROC range as 0.63--0.97.
- It covers MNIST/LeNet, CIFAR-10/ResNet20 and NiN, IMDb/LSTM and GRU, and
  iWildCam/DenseNet and ResNet50, including in- and out-of-distribution data,
  TensorFlow Lite/Core ML quantization, and several training strategies.

**Difference from this project.** Hu et al. use probability margin rather than
our `(top1_logit - top2_logit) / std(all_class_logits)`, include mostly 8/16-bit
deployment quantization rather than HQQ 3/4-bit conditions, and do not construct
our 14-dataset, dataset-equal condition aggregation or dataset-level Q10
difficulty screen. These differences justify a replication/extension claim,
not a first-demonstration claim.

### Wu, Dhiman, and Koshiyama (2026): highly direct, but recent and non-archival

[Which Decisions Low-Bit Quantization Breaks, and How to Predict
Them](https://arxiv.org/abs/2608.06564), arXiv:2608.06564, v3, 2026.

- It tracks the selected option's score minus its best alternative before and
  after 8-to-2-bit quantization, calls changes *flips*, and predicts held-out
  flip rates from paired margins across multiple LLM families and quantizers.
- It reports bit-dependent multiplicative *margin shrinkage*, strengthening the
  general decision-margin interpretation of quantization damage.
- It is an LLM decision study, not vision classification; it requires paired
  pre/post margins, whereas our predictor uses only baseline normalized margin.
- The arXiv page states that it is under review at a non-archival EMNLP 2026
  workshop. It is relevant evidence but should be labeled a recent preprint.

### Xia et al. (2021): confidence, swaps, and accuracy damage

[An Underexplored Dilemma between Confidence and Calibration in Quantized
Neural Networks](https://arxiv.org/abs/2111.08163), Guoxuan Xia, Sangwon Ha,
Tiago Azevedo, and Partha Maji, ICBINB workshop at NeurIPS 2021.

- The paper separates whether a prediction changes after PTQ from whether that
  change alters accuracy, and argues that low-confidence decisions change more
  readily while also being less likely to have been correct initially.
- Experiments cover multiple CNNs on CIFAR-100 and ImageNet. This is a direct
  vision precedent for our `swap -> relative accuracy loss` interpretation.
- It studies confidence/calibration and aggregate behavior, not normalized
  top-1/top-2 logit margin as a per-sample swap AUROC or cross-dataset Q10.

### Kiselev (2026): decision-boundary geometry, but an unreviewed preprint

[Boundary-Aware Quantization: Finite-Scale Decision Geometry of Neural
Classifiers](https://arxiv.org/abs/2607.01478), O. M. Kiselev,
arXiv:2607.01478, 2026.

The paper measures local logit-margin radii, boundary displacement, prediction
flips, and low-margin boundary-band flips under PTQ, including CIFAR-10. It is
highly aligned with the proposed mechanism and shows that accuracy can conceal
substantial decision-boundary changes. However, it appeared in July 2026 and
the arXiv record does not identify a peer-reviewed venue, so it must be cited as
an unreviewed preprint rather than established evidence.

## Close empirical work

### Confidence-sensitive quantization

[When Quantization Affects Confidence of Large Language
Models?](https://aclanthology.org/2024.findings-naacl.124/), Irina Proskurina,
Luc Brun, Guillaume Metzler, and Julien Velcin, Findings of NAACL 2024.

The authors find that GPTQ 4-bit quantization disproportionately affects samples
on which the full model initially has low confidence. This independently
supports sample heterogeneity and the direction of our result, but it studies
LLM confidence/calibration rather than image-level argmax-swap AUROC or HQQ.

### Prediction-level PTQ objectives

[PD-Quant: Post-Training Quantization based on Prediction Difference
Metric](https://arxiv.org/abs/2212.07048), Jiawei Liu et al., CVPR 2023.

PD-Quant optimizes quantization parameters using KL divergence between
full-precision and quantized predictions and shows benefits over local
feature-error objectives at low bit widths. It supports treating output-level
prediction change as scientifically meaningful, but does not show that a
baseline margin predicts which samples flip.

### Hard samples in vision quantization

[Hard Sample Matters a Lot in Zero-Shot
Quantization](https://arxiv.org/abs/2303.13826), Huantong Li et al., CVPR 2023.

HAST reports that zero-shot quantized models degrade most on hard samples and
uses `1 - p(true label)` to define sample difficulty. This is related to our
focus on vulnerable samples, but it concerns synthetic calibration/fine-tuning
for zero-shot quantization, not baseline top-1/top-2 margin or prediction swaps.

### Formal classification consistency

[Quantization with Guaranteed Floating-Point Neural Network
Classifications](https://doi.org/10.1145/3763118), Anan Kabaha and Dana
Drachsler-Cohen, PACMPL/OOPSLA 2025; see the
[author-hosted paper](https://ddana.net.technion.ac.il/files/2025/10/oopslab25main-p551-p-3c52fc33af-162586-final.pdf).

This paper formally defines a quantized classifier as consistent when its
argmax equals the floating-point classifier's argmax, then verifies and corrects
inconsistent inputs. The endpoint is identical to our swap definition, but its
goal is formal verification/correction rather than an empirical risk predictor.

## Work on heterogeneous quantization difficulty

[Benchmarking the Reliability of Post-training Quantization: a Particular Focus
on Worst-case Performance](https://arxiv.org/abs/2303.13003), Zhihang Yuan et
al., ICML AdvML Frontiers Workshop 2023.

This study finds that PTQ reliability varies across tasks, calibration
distributions, and worst-case groups under distribution shift. It supports the
need to avoid reporting only aggregate accuracy, but it does not propose Q10 or
test whether a dataset's margin distribution ranks quantization difficulty.

[Intriguing Properties of Quantization at
Scale](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6c0ff499edc529c7d8c9f05c7c0ccb82-Abstract-Conference.html),
Arash Ahmadian et al., NeurIPS 2023.

In controlled LLM training, PTQ sensitivity changes with weight decay, dropout,
gradient clipping, and training precision even when pre-quantization quality is
similar. This is important for interpretation: quantization difficulty need not
be an intrinsic property of a dataset. It motivates our planned checkpoint/seed
replication.

[Training Dynamics Impact Post-Training Quantization
Robustness](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6937a7c60361d05f5b6cfa04d2c27a5b-Abstract-Conference.html),
Albert Catalan-Tatjer, Niccolò Ajroldi, and Jonas Geiping, ICLR 2026.

Across LLM training trajectories, validation loss and quantization error can
diverge after learning-rate decay. This again argues against treating baseline
accuracy as a sufficient proxy or quantization difficulty as dataset-intrinsic.

## Theoretical context only

- [Robustness of Neural Networks to Parameter
  Quantization](https://arxiv.org/abs/1903.10672), Abhishek Murthy, Himel Das,
  and Md Ariful Islam (2019), defines local/global robustness under parameter
  quantization and explicitly anticipates label changes near decision
  boundaries. Its experiments are small binary MLPs and it does not validate
  margin AUROC.
- [Gradient $\ell_1$ Regularization for Quantization
  Robustness](https://arxiv.org/abs/2002.07520), Milad Alizadeh et al. (ICLR
  2020), models quantization as bounded weight/activation perturbation and
  controls first-order output/loss change. It supports the perturbation picture,
  not our specific sample-margin estimator.
- [Certifiably Quantisation-Robust Training and Inference of Neural
  Networks](https://proceedings.mlr.press/v258/dang25a.html), Hue Dang et al.
  (AISTATS 2025), bounds discrepancies over families of quantized networks and
  finite datasets. It is formal support for quantization robustness as a
  per-input property, not evidence for Q10 or normalized margin.

## Safe positioning for a report or paper

Suggested wording:

> Prior work has shown that full-precision top-1/top-2 probability margin can
> distinguish inputs whose predictions disagree after model quantization. We
> test whether that mechanism generalizes to fine-tuned ViT classifiers under
> low-bit HQQ, using a logit-scale-normalized margin, within-dataset AUROC,
> dataset-equal aggregation, and a separate cross-dataset Q10 screen.

Avoid these claims:

- "We are the first to show that low-margin inputs flip under quantization."
- "Margin is a novel predictor of quantization disagreement."
- "Q10 is an intrinsic dataset property" or "margin causally determines
  quantization loss."

Potentially defensible after a systematic review and stronger replication:

- first evaluation of this mechanism for HQQ-quantized, fine-tuned ViTs;
- first use of scale-normalized baseline logit margin for HQQ swap prediction;
- first dataset-equal cross-dataset test connecting a low-tail margin statistic
  to relative HQQ accuracy loss.

These are currently search findings, not established novelty claims.
