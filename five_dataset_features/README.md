> Current scope: 16 datasets and 336 conditions. Inference and analysis run on nano4. See the TGDA checkout's `tools/five_dataset_hqq/ALL_DATASETS.md` for inference setup and storage planning.

# Five-dataset feature analysis

This is independent of the legacy 13-condition quantization analyzer and the W&B artifact workflow. Analyze the existing nano4 inference output directly on nano4. Analysis needs the full `[N,13,768]` feature arrays for all conditions, not just the final small ZIP. Raw datasets and checkpoints remain in the server's `data/` tree.

## nano4 CPU analysis

`run_nano4.sh` reuses the existing analysis module. It defaults to all 16 datasets,
processed sequentially in one CPU job, followed by the cross-dataset report and
ZIP. It does not perform inference. The `all` stage does not require the separate
CUB smoke output; use `--stage smoke` to validate it explicitly.

Place the checkouts and data as follows:

```text
/work/kyle0724/
├── project/
│   ├── TGDA/                 # inference checkout
│   └── quant_analysis/       # analysis checkout
└── data/five_dataset_hqq/
    ├── aircraft/              # dataset.json, exemplars/, all 21 condition folders
    ├── cub/
    ├── ...                   # the other configured datasets
    ├── smoke/cub/            # only needed for --stage smoke
    └── analysis/             # generated here
```

Submit from the `quant_analysis` checkout on nano4:

```bash
cd /work/kyle0724/project/quant_analysis
bash five_dataset_features/run_nano4.sh --dry-run
sbatch five_dataset_features/run_nano4.sh
```

The input defaults to `<repo>/../../data/five_dataset_hqq`; override it with
`--input-root /absolute/path/to/five_dataset_hqq`. Outputs are always under that
input root's `analysis/`. The script checks all selected datasets' required files
before starting; the analyzer then validates checksums, condition identity,
sample pairing and shapes. ZIP creation also needs each dataset's exported
`exemplars/` files. Re-running regenerates existing analysis outputs, including
the ZIP; do not run concurrent jobs against the same dataset/output directory.

Environment setup follows TGDA's `scripts/run_exp_nano5.sh`: `ml purge`,
`ml load miniconda3`, then direct execution with
`$HOME/.conda/envs/opencode_env/bin/python`. No `conda run` or shell activation
is used. The environment must already contain the repository requirements.
Set `ANALYSIS_CONDA_ENV` to change the environment name, or set
`ANALYSIS_PYTHON=/absolute/path/to/python` for another installation location.
Module setup still runs when Python is overridden, matching the inference script.
Set `ANALYSIS_PROJECT_ROOT` if submitting from another directory;
`ANALYSIS_INPUT_ROOT` overrides the default input (the CLI flag takes precedence).
Dry-run does not load modules, execute Python, or require inference files.

The job requests 8 CPUs, 64 GB RAM and 24 hours, with no GPU request. No partition
is hardcoded: it uses the cluster default. If nano4 requires an explicit CPU
queue, pass its actual name with `sbatch --partition=<cpu-queue>`. Resources and
queue eligibility have not been validated on nano4. The largest datasets may
require adjusted memory/time limits after observing a first run.

```bash
# Small full-test pilot; report and ZIP cover this subset only.
sbatch five_dataset_features/run_nano4.sh --datasets cub,pets

# Recompute statistics and report without making the ZIP.
sbatch five_dataset_features/run_nano4.sh --stage analyze

# Repackage already generated analyses.
sbatch five_dataset_features/run_nano4.sh --stage package

# Separate six-condition CUB smoke analysis.
sbatch five_dataset_features/run_nano4.sh --stage smoke
```

Logs are `hqq_feature_analysis_<job-id>.out` and `.err` in the submission
directory. The final archive is `<input-root>/analysis/five_dataset_hqq_results.zip`.

## Server smoke and storage

After CUB smoke inference, run `--stage smoke` on nano4 and inspect
`/work/kyle0724/data/five_dataset_hqq/analysis/smoke_precision.json` before
submitting full inference. The report includes sample pairing, checksums,
prediction changes and FP16 storage checks against the incremental W3-to-A8 and
A8-to-QKV changes. No feature transfer to a Windows computer is required.

For all 16 datasets, use TGDA preflight's actual test counts and disk estimate.
The original five-dataset estimate of 8.48 GB plus about 0.419 GB per 1,000
cotton test images does not cover the expanded scope. Allow extra server space
for analysis figures, coordinates and temporary ZIP staging. The final ZIP
excludes full-test feature arrays; it is a sharing output, not analysis input.

## Analysis definitions

The dataset stage rejects missing or corrupt conditions and verifies all sample IDs and targets. CKA is centered linear CKA using feature cross-products, never an N-by-N sample matrix. Cosine intra-class distances average distinct sample pairs within each class, then classes equally; inter-class distances average class pairs equally. The final `D_inter - D_intra` is the primary feature separability score. H-score uses within-class scatter regularization of 1% of mean within-feature variance. Norm P50/P95/P99 and P99/P50 use unnormalized features. The final FP feature measures are candidate predictors; per-layer results are supplementary.

FP final features fit one PCA basis per dataset, and every condition uses that basis. Server dataset output includes full-test 2D coordinates for each condition; the small ZIP only includes up to 20 representative samples per dataset. The first real batch's FP32 activations are compared with their FP16 roundtrip in `precision_probe.csv`. Inspect A8 rows before accepting FP16 feature storage; if storage error is material, set `feature_dtype` to `float32` for the affected dataset in the server experiment JSON, rerun its inference, and re-evaluate disk capacity.

`report` computes mean loss by activation family within each selected dataset, then descriptive Spearman correlations across the selected dataset-model pairs (16 by default). Quantized conditions are repeated measurements, not independent datasets. Empty CSV cells mean undefined. No significance claim is made. Inference on nano4 chooses up to ten classes with at least two test images using seed 100, saves two 224px display images per class, and records a checksum manifest. Server packaging verifies those files, then builds an explicit whitelist ZIP and prints its size. The 50 MB limit is a target; inspect the printed size. The package excludes all full-test feature arrays, full images, and checkpoints.
