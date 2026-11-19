# Evidence map

Each row is one result statement in the paper, the macro that prints its number and the input it is derived from. Inputs are in `inputs/`; `grid.csv` is the full weight-only grid and `dataset_master.csv` the 144-row W3/G128 table.

| Section | Statement | Macros | Input and columns |
|---|---|---|---|
| Abstract, 4.1 | Group retention at W3/G128 | `retSc_w3g128`, `retOther_w3g128`, `ret_w3g128`, `gap_w3g128` | `grid.csv`: `hqq_top1 / fp32_top1`, 15 complete datasets, dataset-equal then setting mean |
| 1, Fig. 1 | Every soy+cotton dataset below every other dataset at W3/G128 | `maxSc_w3g128`, `minOther_w3g128` | `grid.csv`, mean over the 9 settings per dataset; asserted in `compute.py` |
| 3 | Baseline top-1 by group | `fpAccSc`, `fpAccOther` | `grid.csv`: `fp32_top1`, mean per dataset, then per group |
| 3 | Grid size | `cells`, `contexts`, `gridDatasets` | `grid.csv` rows and contexts |
| 4.1, Fig. 2 | Saturation at 1 and 8 bits; gap counts at 3 and 4 bits; W2 gaps | `ret_w1_max`, `ret_w8_min`, `gapPos_w3`, `gapPos_w4`, `gap_w2g8`, `gap_w2g128`, `ret_w2g128` | `grid.csv` |
| 4.1 | Negative 1-bit gap for BEiTv2 FT; low-baseline contexts all soy+cotton | `lowBaseline` | `grid.csv`; both asserted in `compute.py` |
| 4.2, Tab. 2, Fig. 3 | Kendall's W; agreement between settings; soy+cotton among the five hardest | `wMin`, `wMax`, `agreeMedian`, `agreeMin`, `agreeMax`, `scHardestMin`, `scHardestAll` | `grid.csv`, primary conditions W3/W4 x G8/G128 |
| 4.3, Tab. 2 | Margin signal, accuracy correlation, held-out error | `gatePass`, `gatePPass`, `rhoMarginMedian`, `rhoMarginBeitFt`, `rhoAccMedian`, `marginHelps` | `grid.csv` with `dataset_master.csv`: `fp_normalized_margin_q10`, `fp_accuracy`, `num_classes` |
| 4.3, Tab. 3, Fig. 4 | Margin matching | `reductionMin`, `reductionMax`, `matchedPositive`, `overlapMin`, `overlapMax`, `essRatioMin`, `weightMax` | `margin_gaps.csv`, `matching_support.csv`, `margin_balance.csv` |
| 4.1, robustness table | Gap under chance correction, without baselines below 15%, leave-one-out, soybean as one unit, cotton alone; binomial error; dataset bootstrap | `gapChance_w3g128`, `gapNoLow_w3g128`, `looMinGap`, `seMax`, `seOtherMax`, `bootPositive`, `bootLowBeitFt` | `grid.csv` at W3/G128; `margin_coverage.csv`: `num_classes`, `n_samples` |
| 4.3, accuracy figure | Baseline accuracy against retention; pairs at equal baseline accuracy; partial correlation | `pooledN`, `pooledR2`, `pairsClose`, `pairsDiffer`, `pairsMaxDiff`, `pairA*`, `pairB*`, `pairC*`, `partialNeg`, `partialMedian` | `grid.csv` at W3/G128 and the primary conditions; `dataset_master.csv`: `fp_normalized_margin_q10`, `fp_accuracy` |
| 4.4, Fig. 5 | Mid-depth CKA by group | `midCkaSc`, `midCkaOther` | `layer_metrics.csv`: `cka_vs_fp`, condition `w3_g128` |
| 4.4, Tab. 5 | Property correlations | `rhoCkaMedian`, `rhoSepMedian`, `rhoHscoreMedian`, `rhoNormMedian` | `dataset_master.csv` with the difficulty score |
| 4.4, Tab. 4 | BEiTv2 FZ collapse | `beitFzCka`, `beitFzFp`, `beitFzQ` | `dataset_master.csv`: `final_cka`, `fp_accuracy`, `accuracy` |
| 4.5, Tab. 4 | Damage over rescue; swap by group | `damageOverRescueMin`, `swapHigherSc` | `dataset_master.csv`: `damage_rate`, `rescue_rate`, `swap_rate` |
| 4.5, Tab. 4 | Offset correction and shuffled controls | `biasGainMin`, `biasGainMax`, `shuffleLossMin`, `shuffleLossMax` | `bias_results.csv`: `gain_pp`, `shuffle_mean_gain_pp` |
| 4.5, Tab. 4 | Competitor features; undefined AUROC | `competitorBetter`, `aurocUndefined`, `aurocRows` | `competitor_comparisons.csv` (scope `lodo`, outcome `damage`), `competitor_scores.csv` |
| A.1 | Agreement of grid and paired runs | `consistContexts`, `consistFp`, `consistQ` | `grid.csv` at W3/G128 against `dataset_master.csv` |
| A.2 | Bin-width sensitivity | `reductionMinAll`, `reductionMaxAll` | `margin_gaps.csv`, all bin widths |
| Fig. 6 | Swap rate by margin decile | none | Synthetic placeholder |
| Tab. 1 | Training-set sizes | none | Dummy placeholder |
