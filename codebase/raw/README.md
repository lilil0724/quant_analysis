# TGDA raw export analysis

This stage reads DAFB's matched unquantized AMP and Linear weight-only W3/G128
inference exports on Nano4. It validates provenance and sample order, computes
feature and logit diagnostics, and writes CSVs with the same schemas as the
corresponding paper inputs.

## Run on Nano4

From the repository root, run:

    sbatch codebase/raw/run_nano4.sh --serial 20004 \
      --input-root /work/kyle0724/data/w3_g128_analysis

The input root contains the serial directory. Each DAFB model/mode context needs
the completed schema-2 FP and W3/G128 directories with matching manifests,
sample IDs, labels, features, predictions, logits, and metrics. The stage reads
the files in place using the path available on the execution host.

The default scope covers ViT-B/16, Swin-B, and BEiTv2-B with FT, FZ, and CAL.
Use --preflight-only to verify the selected files and their checksums before
analysis, or add --models vit_b16 --modes ft for a single pair. The full run
uses NumPy, pandas, SciPy, Matplotlib, scikit-learn, and threadpoolctl from the
Nano4 tgda environment. Set ANALYSIS_PYTHON to use another environment.

## Source and outputs

- paired_inputs.py validates TGDA manifests, quantization coverage, and the
  pairing between features, predictions, and logits.
- paired.py, metrics.py, centered_class_distance.py, and tsne_paired.py
  compute geometry from paired features. The existing formulas are shared
  with the legacy entry points.
- server_bias_competitors.py and margins.py provide within-DAFB logit diagnostics.
- dafb.py coordinates the run. prepared.py joins the feature and prediction
  summaries by model and checkpoint mode.

Each run creates codebase/raw/result/<run_id>/ with detailed CSVs, figures,
and input/output checksums in run.json. Its prepared/ directory includes
dataset_master.csv, layer_metrics.csv, bias_results.csv, and winner_ranks.csv
when their corresponding analyses are selected. The master table uses one row
per model/mode and leaves fields blank when an optional analysis was omitted.
The W&B accuracy grid and cross-dataset margin matching require their own inputs.

The run directory is ignored by Git. The single report at
codebase/raw/reports/<run_id>/REPORT.md can be tracked. The paper build
continues reading its committed snapshot under the repository's root data/
directory. Use the prepared tables as new inputs after choosing the dataset
cohort and checking agreement with any external accuracy grid.
