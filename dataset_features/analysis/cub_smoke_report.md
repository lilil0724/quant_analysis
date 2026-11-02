# CUB HQQ activation smoke analysis

## Scope and validation

This is a diagnostic analysis of **64 ordered test images**, not the 5,794-image CUB test set. The selected images cover only three of 200 classes (30, 30, and 4 images). All six available conditions have 64 aligned predictions and `[64, 13, 768]` feature arrays. The local smoke validator checked completion manifests, file checksums, sample IDs, targets, and feature shapes. The conditions all use the same checkpoint and test-image selection recorded in `cub/dataset.json`.

The 13 feature positions are the outputs of Transformer blocks 1–12 and the final normalized CLS feature. Accuracy uses all 64 images; the FP16 precision check uses the first 32 images for which FP32 probe features were saved.

## Prediction and final-feature results

| Condition | Correct / 64 | Accuracy | Drop vs FP | Swaps vs FP | Damage / rescue | Final CKA to FP | Final class-equal separation |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP | 54 | 84.38% | 0 pp | 0 | 0 / 0 | 1.000 | 0.1943 |
| W3, group 128 | 51 | 79.69% | 4.69 pp | 7 | 4 / 1 | 0.871 | 0.1515 |
| W3 + Linear A8 | 51 | 79.69% | 4.69 pp | 7 | 4 / 1 | 0.873 | 0.1507 |
| W3 + Linear A8 + QKV A8 | 51 | 79.69% | 4.69 pp | 7 | 4 / 1 | 0.873 | 0.1519 |
| W3 + Linear A4 | 1 | 1.56% | 82.81 pp | 63 | 53 / 0 | 0.289 | -0.0022 |
| W3 + Linear A4 + QKV A4 | 1 | 1.56% | 82.81 pp | 63 | 53 / 0 | 0.177 | 0.0044 |

`Drop vs FP` is an absolute accuracy difference in percentage points. The relative loss `1 - accuracy / FP accuracy` is 5.56% for W3/A8 and 98.15% for both A4 conditions. Swaps count different predicted labels relative to FP; damage is FP-correct to quantized-wrong, and rescue is FP-wrong to quantized-correct. Class-equal separation is `D_inter - D_intra` after L2 normalization and cosine distance, using the three present classes only.

Adding Linear A8 to W3 changes **zero** predicted labels on these 64 images. Adding QKV A8 changes **zero** more. The A8 features do change, so identical predictions do not mean the activation quantizer is inactive. Linear A4 changes 63 labels relative to W3; adding QKV A4 changes 17 labels relative to Linear A4 but leaves the number correct at one. Linear A4 predicts class 8 for 52 of 64 images, consistent with a severe prediction collapse in this smoke sample.

The [CKA plot](cub_smoke_cka.png) shows that W3 and A8 remain close to one another across the 13 positions. Both A4 conditions diverge sharply, especially near the last blocks and final feature. CKA describes representation similarity; accuracy and prediction comparisons above are the direct performance observations.

The FP-only PCA basis explains **41.8%** and **16.1%** of FP final-feature variance in its first two components. The [shared-basis PCA plot](cub_smoke_pca.png) places all six conditions on the same axes: the A8 class groups remain separated in this view, while the two A4 groups overlap in a compact region and almost all markers are incorrect. The [2D coordinates](cub_smoke_pca_coords.npz) include the sample IDs, true labels, and explained-variance ratios. PCA is a visualization of this selected sample, not a substitute for the original-space distances above.

The FP final-feature regularized H-score is **4,336**, compared with **4,227** for W3 and **1,645** for Linear A4. FP normalized-margin Q10 is **0.54**, compared with **0.43** for W3 and about **0.02** for both A4 conditions. These supplementary values are in the JSON report. With 64 samples and 768 dimensions, the absolute H-score is especially sensitive to the ridge and this three-class selection; interpret only as a smoke diagnostic.

## FP16 storage check

For each incremental pair, the 32-image FP32 probe compared the CKA change caused by activation quantization with the CKA change caused by saving the same features in FP16. All 13 positions in every pair pass the smoke criterion `activation change > 10 × FP16 storage delta`.

| Incremental pair | Smallest change / storage-error ratio across 13 positions | Final-position CKA change | Final-position FP16 storage delta |
|---|---:|---:|---:|
| W3 → Linear A8 | 340× | 0.003273 | 0.00000262 |
| Linear A8 → Linear + QKV A8 | 563× | 0.001080 | 0.00000101 |
| W3 → Linear A4 | 3,559× | 0.465700 | 0.00011376 |
| Linear A4 → Linear + QKV A4 | 6,239× | 0.388475 | 0.00006226 |

This supports FP16 feature storage for distinguishing these activation effects in this probe. It does not establish precision adequacy for every later dataset or metric; repeat the check on the full-run data.

## Interpretation and next step

The dominant smoke result is the A4 collapse under W3/group-128. A8 preserves the 64-image accuracy even though it measurably moves the features. The present sample is highly class-skewed, so neither the accuracies nor the three-class separation should be reported as CUB-200 performance or used for cross-dataset correlation. The other 15 CUB conditions and the other four datasets are absent, so the planned full analysis cannot yet run.

The copied `dataset.json` records `quant_zero=true` and `quant_scale=true`. The pinned HQQ fork emits a warning and clears these deprecated metadata-quantization settings in `HQQLinear.initialize()`. The local TGDA runner now sets both to `false` and records the effective values. Sync that runner to nano5 and rerun the smoke before launching the full jobs, so the server manifest matches the effective quantization configuration. The rerun must be copied back before relying on this report as the final smoke gate.

Machine-readable details, including per-layer CKA, distances, norm percentiles, prediction summaries, and the precision probe, are in [smoke_precision.json](smoke_precision.json). Per-sample predictions remain in the copied condition folders. No W&B run or artifact was created by the local analysis.
