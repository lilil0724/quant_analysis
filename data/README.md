# Data

The inputs `plotter.sh` reads. `codebase/download/snapshot.py` copies them from the run bundles under
`legacy/`; `MANIFEST.json` records each file's source and SHA-256.

| File | Content | Source |
|---|---|---|
| `grid.csv` | Weight-only HQQ grid: baseline and quantized top-1 for 144 contexts × 32 conditions (4,608 cells) | W&B `nycu_pcs/TiT` serials 10000–10016, selected by run `20261117_legacy_hqq_s10000_10016_cal10016_v2` |
| `dataset_master.csv` | One row per context at W3/G128: accuracies, swap/damage/rescue rates, margins, feature properties | serial 20004 exports, audit `20261116_140853_554056_runall_s20004_audit` |
| `margin_gaps.csv`, `margin_diagnostics.csv`, `margin_balance.csv`, `margin_coverage.csv` | Margin matching: raw and matched gaps, ESS, histograms, class counts | serial 20004, run `20261115_211022_759540_w3_s20004` |
| `matching_support.csv` | Overlap and ESS summary of the matching | serial 20004 audit |
| `bias_results.csv` | Class-offset correction and shuffled controls | serial 20004 run |
| `winner_ranks.csv` | Baseline rank of the quantized winner for every swap | serial 20004 run |
| `competitor_comparisons.csv`, `competitor_scores.csv` | Held-out damage prediction with and without competitor counts | serial 20004 run |
| `layer_metrics.csv` | Layer-wise CKA between baseline and W3/G128 features, 144 contexts | serial 20004 run (`paired/`) |
| `tsne_cub_vit_b16_ft.csv` | t-SNE coordinates for CUB, ViT-B/16 FT | serial 20004 run |
