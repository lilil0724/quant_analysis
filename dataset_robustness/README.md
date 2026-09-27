# Dataset robustness stability audit

Descriptive comparisons of the legacy HQQ experiments. The main question is
whether dataset accuracy-retention rankings persist across quantization settings
and model/checkpoint contexts. This is separate from `hqq_confidence` and does not
modify the four legacy scripts.

## Contract

- Primary: 2, 3, 4 bits, all observed group sizes, compared separately.
- Sensitivities: 3/4 bits only and the complete observed bit grid.
- Primary metric: upstream matched-seed `accuracy_ratio`; supplement with
  `fp32_top1 - hqq_top1` in percentage points and baseline accuracy.
- No clipping of ratio > 1 or negative accuracy drops.
- One checkpoint path per dataset/model/checkpoint kind. Ambiguous or changing
  checkpoint identities fail rather than being silently averaged.
- Non-matched, nonfinite or invalid cells remain in an exclusion audit.
- Within-context comparisons use a fixed dataset cohort complete over the
  scope's bit × group grid. Cross-context comparisons use the common dataset
  cohort at each quantization condition. Coverage records exclusions explicitly.
- Rank 1 is highest retention; ties use average rank. Constant rankings and
  comparisons with fewer than three datasets have undefined Spearman rho.
- Pair reversal means both directions occur across settings. A `1e-12`
  numerical equality tolerance is not a practical equivalence threshold.
- Every condition has equal descriptive weight within a scope. There is no
  universal dataset ranking, p-value or seed-level confidence interval.
- Dataset-specific checkpoints confound dataset and training outcome. A
  one-seed experiment cannot establish training-seed reproducibility.

## Reproduce

Use `opencode_env` from the repository root. All outputs below are isolated from
the existing `results_all/quant` snapshots; downloading is not required.

```powershell
conda activate opencode_env
python summarize_quant.py --input-file data/backbones_quant.csv --output-file results_all/dataset_robustness/legacy/quant_summary.csv --results-dir results_all/dataset_robustness/legacy
python corr_quant.py --input-file results_all/dataset_robustness/legacy/quant_summary.csv --output-file results_all/dataset_robustness/legacy/corr/factor_ablation_accuracy.csv --results-dir results_all/dataset_robustness/legacy/corr
python plot_quant_acc_bit_group.py --input-file results_all/dataset_robustness/legacy/corr/prepared_accuracy_cells.csv --corr-dir results_all/dataset_robustness/legacy/corr --ablation-file results_all/dataset_robustness/legacy/corr/factor_ablation_accuracy.csv --results-dir results_all/dataset_robustness/legacy/plots --output-file results_all/dataset_robustness/legacy/plots/overview.png
python -m dataset_robustness.analyze
python -m unittest discover -s tests
python -m unittest dataset_robustness.test_analysis
python -m dataset_robustness.verify
```

The analyzer accepts `--input-file` and `--results-dir`. The explicit
`--primary-bits` option validates the agreed `2 3 4` contract rather than silently
changing report interpretation. It reads `configuration_quality.csv` beside the
summary when available. Source summary and analyzer SHA-256 hashes are recorded
in `run_config.json`.

Start with `results_all/dataset_robustness/report.md`. The report links the main
figures; `plots_manifest.csv` maps all plots to source tables. CSVs preserve the
full per-condition results and all pairwise stability comparisons. Reported
rank summaries are descriptive of equally weighted, observed conditions; their
variation is not an uncertainty interval.
