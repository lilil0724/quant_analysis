# Current analysis record

## Corrected baseline promotion (2026-10-21)

The formal local result entrypoints are `legacy_hqq/results/main/`,
`dataset_robustness/results/main/`, `dataset_robustness/results/pca_phase1/`,
`granularity_robustness/results/main/`, and `outputs/serial_20007/results/main/`.
Read [the promotion record](docs/baseline_promotion_20261021.md) for provenance,
verification and rollback. The preserved recovery worktrees are historical inputs,
not dependencies of the formal outputs.

For baseline-based reruns, use the verified cohort in
`data/baseline_recovery/selected_20261021/`. It contains 144 unquantized AMP
baselines plus 4,608 HQQ conditions. The legacy/dataset/granularity main cohort
remains 15 datasets (4,320 conditions); the 20007 comparison includes all 16.

The five affected datasets are dogs, flowers, moe, pets and vegfru. Their
BEiTv2 FT/FZ readout is cls and CAL is pool; replace baseline and weight-only
HQQ runs together. Match complete checkpoint path, seed and evaluation config.
The only accepted path alias is `../../data/` to `/work/kyle0724/data/`.
`fp32_top1` is a legacy field name; the stored baseline uses AMP, and paths
alone do not establish checkpoint byte identity.

Rebuild the cohort with `python -m legacy_hqq.recover_baselines` in opencode_env.
Use its curated 15-dataset CSV for legacy acquisition; raw W&B repeats must pass
the same cohort/evaluation audit before entering the default pipeline. Preserve
the snapshot/manifest together when changing inputs. Generated data and results
are ignored by Git and require separate transfer or backup.
