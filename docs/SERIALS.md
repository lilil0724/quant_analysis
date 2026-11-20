# Serials

W&B project `nycu_pcs/TiT`. Runs were trained and evaluated outside this repo (TGDA on NCHC nano4).

| Serial | Content | Used by | Processed by |
|---|---|---|---|
| 10000–10016 | Weight-only HQQ grid: unquantized AMP baselines and HQQ runs, 16 datasets × 3 backbones × FT/FZ, 8 bit widths × 4 group sizes | `data/grid.csv` | `legacy/quant_acc/`, run `20261117_legacy_hqq_s10000_10016_cal10016_v2` |
| 10016 | The CAL baselines and HQQ runs (all CAL cells of the grid come from this serial only) | `data/grid.csv` | same run |
| 20004 | W3/G128 paired exports on nano4: features after every block and final logits of baseline and quantized models, 144 contexts | every other file in `data/` | `legacy/dataset_features/`, `legacy/dataset_confidence/`; run `20261115_211022_759540_w3_s20004`, audit `20261116_140853_554056_runall_s20004_audit` |
| 20007 | W6A4 weight and activation quantization | not in the paper | `legacy/activation_quantization/` |
| 20008 | Weight and activation (W+A) grid | not in the paper | `legacy/activation_quantization/` |

Selection rule for the grid: for FT and FZ, the newest finished positive AMP baseline per context and
the newest compatible HQQ run per condition; for CAL, serial 10016 only. Older duplicates are kept
out, not averaged; `legacy/quant_acc/20261117_legacy_hqq_s10000_10016_cal10016_v2/audit/` lists every
selected and excluded run.
