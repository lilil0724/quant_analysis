# Quantization Robustness Analysis

This context defines the distinct quantities used to explain accuracy retention after quantization. It prevents decision stability, calibration, and final accuracy from being conflated.

## Language

**Decision stability**:
Resistance of a baseline top-1 prediction to changing after quantization, measured by prediction swap rate.
_Avoid_: Quantization robustness when referring only to whether predictions change

**Accuracy robustness**:
Retention of baseline task accuracy after quantization, measured by accuracy ratio (`quantized accuracy / baseline accuracy`).
_Avoid_: Stability, swap robustness

**Swap net damage**:
Net accuracy loss per changed prediction, defined as `(Correct-to-Wrong - Wrong-to-Correct) / prediction swaps`.
_Avoid_: Damage rate, accuracy drop

**Normalized decision margin**:
The baseline top-1 minus top-2 logit gap divided by the within-sample standard deviation across class logits. It is the primary cross-dataset margin measure and a candidate explanation for decision stability, not a calibration measure.
_Avoid_: Confidence, overconfidence, raw margin

**Raw decision margin**:
The unnormalized baseline top-1 minus top-2 logit gap. It is a sensitivity measure because independently fine-tuned checkpoints may use different logit scales.
_Avoid_: Primary margin

**Vulnerable-tail margin**:
The 10th percentile of baseline normalized decision margins within a dataset-model pair. It is the prespecified primary logit predictor of exploratory dataset quantization difficulty; lower values denote a more exposed decision-boundary tail.
_Avoid_: Mean margin, post-hoc best-performing margin statistic

**Promising difficulty signal**:
An exploratory result with Kendall's W of at least 0.6 across the four primary HQQ conditions, Spearman rho at most -0.5 between vulnerable-tail margin and difficulty with one-sided dataset permutation p below 0.05, negative direction in at least three conditions, and a negative association after every leave-one-dataset-out deletion.
_Avoid_: Statistical significance alone, selecting the best sensitivity predictor after inspection

**Primary precision range**:
The 3-bit and 4-bit HQQ conditions at group sizes 8 and 128 used to define exploratory dataset quantization difficulty. The 2-bit and 8-bit conditions are boundary stress tests, while 1-bit and 1.58-bit conditions are excluded from inference because saturation can erase relative dataset differences.
_Avoid_: Treating all bit widths as exchangeable, calling 2-bit or 8-bit independent validation

**Baseline difficulty controls**:
Baseline accuracy and the logarithm of class count. They are the only prespecified controls used to test whether vulnerable-tail margin improves leave-one-dataset-out prediction of exploratory dataset quantization difficulty.
_Avoid_: Expanding the control set after seeing results, training-fit improvement without held-out improvement

**Margin mechanism consistency**:
The evidence pattern in which lower vulnerable-tail margin predicts higher prediction swap rate with median condition-wise rho at most -0.5, and higher swap rate predicts greater relative accuracy loss with median rho at least 0.5. Each path requires the expected direction in at least three of four primary conditions and a one-sided dataset-block permutation p below 0.05; without both paths, vulnerable-tail margin is not described as an explanation of accuracy difficulty.
_Avoid_: Causal mediation, inferring mechanism from margin-to-accuracy correlation alone

**Sample-level margin discrimination**:
Within a dataset–model pair and quantization condition, the ability of negative baseline normalized decision margin to distinguish samples whose top-1 prediction swaps after quantization. The primary population is all matched samples; baseline-correct samples form a sensitivity population, and AUROC is undefined when the swap outcome contains only one class.
_Avoid_: Dataset-level margin association, restricting the primary population to baseline-correct samples, assigning a numeric AUROC to a single-class outcome

**Dataset-level sample mechanism score**:
The unweighted mean sample-level margin discrimination across the four primary HQQ conditions for one dataset–model pair. Cross-dataset inference gives each dataset one equal-weight observation; condition rows remain diagnostic repeated measurements rather than independent replications.
_Avoid_: Pooling images across datasets, weighting datasets by sample or swap count, treating dataset–condition rows as independent

**Complete dataset-level sample mechanism score**:
A dataset-level sample mechanism score for which AUROC is identifiable in all four primary HQQ conditions. If any primary condition has a single-class swap outcome, the dataset score is missing and the primary gate is incomplete rather than failed; missing sensitivity results do not invalidate a complete primary score.
_Avoid_: Averaging across only the identifiable primary conditions, interpreting unavailable evidence as evidence against the mechanism

**Supported sample-level margin mechanism**:
A primary result whose median dataset-level sample mechanism score is at least 0.60, whose one-sided exact sign test against chance has p below 0.05, and whose condition-specific cross-dataset median AUROC exceeds 0.50 in all four primary HQQ conditions. Baseline-correct results are sensitivity evidence and do not determine this gate.
_Avoid_: Declaring support from pooled-image significance, a single condition, or sensitivity results alone

**Margin decile**:
A within-dataset percentile bin of baseline normalized decision margin, numbered from 1 for the lowest-margin samples to 10 for the highest. Equal margin values remain in the same bin, so bins may be unequal or empty and must carry their observed sample counts.
_Avoid_: Splitting tied margins by sample order, implying every decile contains exactly ten percent of samples

**Dataset-equal decile curve**:
The primary visualization of swap rate by margin decile, formed by averaging the four primary conditions within each dataset and then averaging datasets with equal weight. Empty deciles are omitted rather than filled with zero, and uncertainty is a dataset bootstrap with the contributing-dataset count reported.
_Avoid_: Pooling samples across datasets, sample-count weighting, zero-filling empty deciles

**Overconfidence**:
The positive part of the gap between baseline confidence and empirical accuracy, summarized across confidence bins by OECE. It is a property of a dataset-model pair, not an intrinsic dataset property.
_Avoid_: Confidence, logit margin, dataset overconfidence

**Dual-mechanism model**:
The research model in which decision margin explains whether quantization causes a prediction swap, while overconfidence explains the net accuracy damage conditional on swaps; together these mechanisms explain accuracy robustness.
_Avoid_: A single margin-to-accuracy or OECE-to-accuracy mechanism

**Exploratory dataset quantization difficulty**:
The mean within-condition percentile rank of relative accuracy loss across 3-bit and 4-bit HQQ with group sizes 8 and 128. Median relative accuracy loss is reported as a secondary effect-size outcome; the construct is valid as a common difficulty score only when condition rankings are sufficiently concordant.
_Avoid_: Universal dataset difficulty, causal dataset effect
