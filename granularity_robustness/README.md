# Granularity robustness analysis

This module implements `dataset_robustness/granularity_analysis_proposal.md`
without changing the four legacy pipeline scripts. It compares externally
defined FGIR and Ultra-FGIR dataset groups within matched model, checkpoint and
HQQ group-size strata.

## Contract

- Dataset groups come only from `dataset_granularity_mapping.csv`.
- Primary groups are 8 FGIR and 4 Ultra-FGIR datasets; `food`, `inat17` and
  `moe` are classification-boundary datasets.
- Primary scope is 2/3/4-bit × group sizes 8/32/128/512.
- Each dataset is weighted once per matched stratum. Cells are not independent
  training repeats.
- Primary estimand is the Ultra-FGIR minus FGIR median 4→2-bit degradation.
- A co-reported matched-condition level contrast is the Ultra-FGIR minus FGIR
  median `accuracy_ratio` at each bit-width and group size; negative values mean
  lower Ultra-FGIR retention under the same quantization condition.
- Every pre-specified grouping, baseline, metric, aggregation, bit-scope and
  leave-one-dataset-out sensitivity is retained.
- Strata with fewer than two datasets in either group are marked incomparable.
- Results are descriptive associations: no p-values or seed-level confidence
  intervals are produced.

## Reproduce

Use `opencode_env` from the repository root:

```powershell
conda run -n opencode_env python -m granularity_robustness.analyze
conda run -n opencode_env python -m unittest granularity_robustness.test_analysis
conda run -n opencode_env python -m granularity_robustness.verify
```

The default input is
`results_all/dataset_robustness/legacy/quant_summary.csv`. Outputs are isolated
under `results_all/granularity_robustness/`. Start with `report.md`; use
`dataset_granularity_mapping.csv`, `granularity_coverage.csv` and
`run_config.json` to audit inclusion and provenance.
