# HQQ Confidence Fragility Analysis

This directory contains a reproducible four-stage workflow for matched
unquantized/HQQ prediction artifacts. It is independent of the repository's
legacy accuracy-ratio pipeline.

The baseline condition is `fp_unquantized`: unquantized weights evaluated with
the same AMP FP16 setting as HQQ. It is not an FP32 baseline.

## Data flow

Run from the repository root with the `opencode_env` Conda environment:

```powershell
python -m hqq_confidence.download_artifacts
python -m hqq_confidence.prepare_pairs
python -m hqq_confidence.analyze_fragility
python -m hqq_confidence.plot_fragility
```

The defaults connect these stages:

| Stage | Default input | Default output |
|---|---|---|
| download | finished `nycu_pcs/TiT` W&B runs | `hqq_confidence/data/selected_runs.csv` and `data/artifacts/<run-id>/<version>/` |
| prepare | selected-runs manifest | `hqq_confidence/results/prepared/` |
| analyze | prepared summaries and matched pairs | `hqq_confidence/results/stats/` |
| plot | prepared and statistical tables | `hqq_confidence/results/plots/` |

Each command accepts absolute path overrides; use `--help` for its complete
interface. Each stage writes `run_config.json` in its output directory.
Downloading requires `WANDB_API_KEY` or `wandb login`. For an ambiguous rerun,
create a CSV containing one `run_id` per selected condition and pass it with
`--run-ids-file`; the downloader never resolves ambiguity by taking `:latest`.

## Sweep and validation contract

A sweep is identified by project, dataset, model, checkpoint SHA-256, serial,
and seed. It contains exactly these 13 runs:

```text
fp_unquantized
hqq_n1_g8    hqq_n1_g128
hqq_n1.58_g8 hqq_n1.58_g128
hqq_n2_g8    hqq_n2_g128
hqq_n3_g8    hqq_n3_g128
hqq_n4_g8    hqq_n4_g128
hqq_n8_g8    hqq_n8_g128
```

Only finished runs with `job_type=hqq-confidence-eval` are downloaded. The
prepare stage reports incomplete and provenance-incompatible sweeps without
including them in analysis. It checks sample-ID sets and targets, N/C, logits
and derived arrays, recomputed accuracy, matched AMP/dtype/Git provenance, and
quantized layer counts. Legacy artifacts may omit `metadata_amp_fp16`, and
older HQQ summaries may use `top1` instead of the canonical `hqq_top1`;
capability checks are used instead of
assuming that schema version alone determines available fields.

Unknown AMP, compute-dtype, Git, or quantized-layer provenance is excluded by
default. `--allow-unknown-provenance` permits only those unknown states; it does
not permit mismatches, debugging runs, invalid layer counts, or incomplete
sweeps.

## Outputs

`data/selected_runs.csv` records the W&B run/config/summary, chosen artifact
name, version and digest, and the cached NPZ path. Prepared outputs are:

- `completeness.csv` and `configuration_quality.csv`: one row per candidate sweep.
- `condition_summary.csv`: one row per valid sweep condition, including raw
  uncalibrated NLL, Brier score, ECE, accuracy, confidence, margin, and entropy.
- `transition_summary.csv`: one row per HQQ/baseline pair with accuracy ratio,
  accuracy drop, damage/rescue, agreement, and mean paired deltas.
- `pairing_quality.csv`: W&B-versus-artifact accuracy audits.
- `reliability_bins.csv`: equal-count raw-confidence reliability bins.
- `pairs/<sweep-id>/<condition>.npz`: compressed, reduced matched-sample arrays;
  full logits are not duplicated.

Statistical outputs include `confidence_bins_by_sweep.csv`, aggregated
`confidence_bins.csv`, `condition_statistics.csv`,
`condition_statistics_by_context.csv`, `bootstrap_metadata.json`,
`data_quality.md`, and `report.md`. Figures and their input-table mapping are
listed in `plots/plots_manifest.csv`.

## Metric definitions

All accuracies and accuracy drops are percentage points. Rates and ratios are
unitless fractions.

```text
damage       = baseline correct and HQQ wrong
rescue       = baseline wrong and HQQ correct
damage_rate  = damage_count / baseline_correct_count
rescue_rate  = rescue_count / baseline_wrong_count
accuracy_drop_points = baseline_accuracy - hqq_top1
accuracy_ratio       = hqq_top1 / baseline_accuracy
delta metric         = HQQ metric - baseline metric
```

The primary analysis uses only baseline-correct samples and relates damage to
the within-sweep percentile of the baseline top-1 minus top-2 logit margin.
Confidence intervals use a fixed-seed checkpoint-cluster bootstrap; serial
reruns and individual images are not treated as independent replicates. No
temperature scaling is fitted because only the test split is available. The
results are exploratory associations, not causal or confirmatory claims.

## Verification

CPU-only synthetic tests do not require W&B credentials:

```powershell
$env:MPLCONFIGDIR = Join-Path $env:TEMP 'quant-analysis-mpl'
python -m unittest discover -s hqq_confidence/tests -p "test_*.py"
```

Run the legacy scripts separately against `data/backbones_quant.csv` when
changing shared repository dependencies. This workflow does not import or
modify any of the four legacy stages.
