# nano4 paired FP/W3 feature analysis

The current `run_nano4.sh` analyzes TGDA's existing `w3_g128_analysis` exports on
nano4. It reads the source manifest and per-layer FP/W3 arrays through
`five_dataset_features.paired`, reusing the established feature metrics. It does
not repeat inference or modify the source files.

## Server layout and submission

```text
<server-root>/
├── project/
│   ├── TGDA/
│   └── quant_analysis/
└── data/w3_g128_analysis/
    └── <serial>/
        ├── <dataset>/<model>/<ft|fz|cal>/
        │   ├── manifest.json
        │   ├── samples.json
        │   ├── fp/                 # per-layer .npy, logits.npy, predictions.csv
        │   ├── w3_g128/            # same matched test samples
        │   ├── layer_metrics.csv   # existing TGDA metrics
        │   └── summary.json        # existing TGDA summary
        └── analysis/               # new analysis outputs
```

Select one source serial explicitly; the example below uses the listed `20005`
directory, without assuming it is complete:

```bash
cd /work/kyle0724/project/quant_analysis
bash five_dataset_features/run_nano4.sh --serial 20005 --dry-run
sbatch five_dataset_features/run_nano4.sh --serial 20005
```

Defaults are all 16 datasets, the three model directory names `vit_b16`,
`swin_base_patch4_window7_224_in22k`, and `beitv2_base_patch16_224`, and all three
checkpoint modes `ft,fz,cal`: 144 selected pairs, each containing FP and W3.
Preflight requires every selected pair to be complete and compatible. It never
silently reduces this matrix or merges experiments from different serials.

The default input parent is `<analysis-checkout>/../../data/w3_g128_analysis`.
Override it using `--input-root`; a selected serial directory is also accepted
when its name matches `--serial`. Output defaults to `<input-root>/<serial>/analysis`;
`--output-root` can select another directory. Original pair directories are
preserved. Within the input parent, outputs must stay under the selected serial's
`analysis/` directory, protecting source exports from other experiments too.

```bash
# One dataset/model/mode pilot, still using the complete test split.
sbatch five_dataset_features/run_nano4.sh --serial 20005 \
  --datasets cub --models vit_b16 --modes ft

# Read-only input validation; no analysis outputs.
sbatch five_dataset_features/run_nano4.sh --serial 20005 --stage preflight

# Analyze and report, without ZIP.
sbatch five_dataset_features/run_nano4.sh --serial 20005 --stage analyze

# Repackage the matching selected analysis scope.
sbatch five_dataset_features/run_nano4.sh --serial 20005 --stage package
```

Use the same dataset/model/mode options for subsequent report/package stages.
Stages are `all`, `preflight`, `analyze`, `report`, and `package`. There is no
legacy CUB smoke stage for this paired export format.

## Environment and resources

Like TGDA's inference launcher, the script runs `ml purge`, `ml load miniconda3`,
and directly executes `$HOME/.conda/envs/tgda/bin/python`. Override using
`ANALYSIS_CONDA_ENV` or `ANALYSIS_PYTHON`. The environment must already contain
NumPy and Matplotlib. No package installation or Conda activation occurs in jobs.
`ANALYSIS_PROJECT_ROOT` changes the checkout path; `ANALYSIS_INPUT_ROOT` and
`ANALYSIS_OUTPUT_ROOT` set data/output defaults, with CLI arguments taking precedence.

SBATCH defaults are partition `8gpus`, one GPU per node, one task, 8 CPUs,
64 GB RAM and 24 hours, account `MST114495`. Resources can be overridden before
the script name, for example `sbatch --partition=dev --time=00:30:00 ...`.
The queue allocates a GPU; the current NumPy metrics/PCA computations use CPUs.
Resource sufficiency and nano4 execution have not been measured by local tests.
Logs are `hqq_feature_analysis_<job-id>.out/.err` in the submission directory.

## Outputs and scientific scope

Each selected dataset/model/mode receives paired prediction statistics, per-layer
CKA, class-equal cosine distances/separation and norm summaries, plus final-layer
H-score and normalized-margin Q10. PCA fits the FP final-feature basis once per
pair and projects both conditions into it; dimensions and layer names come from
the source manifest, accommodating ViT, BEiT and Swin.

Outputs include per-pair `summary.json`, `layer_metrics.csv`, `fp_pca.npz`,
`coords_fp.npy`, `coords_w3_g128.npy`, and `pca.png`. Aggregate outputs include
`coverage.csv`, `all_pairs.csv`, `REPORT.md`, `analysis_manifest.json`, and
`w3_g128_results.zip`. The ZIP includes compact results and figures, excluding
full source features, logits, checkpoints and full-test PCA coordinates.

Dataset/model/checkpoint combinations are repeated measurements. Report any
cross-dataset association separately within a model/checkpoint scope, with
coverage metadata; do not treat all 144 pairs as independent datasets. CAL
features describe the first raw-image backbone call while logits retain the
normal crop/flip aggregation recorded by TGDA. PCA/CKA do not establish causality.

## Legacy 21-condition workflow

`analyze.py`, `smoke.py`, and `run_local.ps1` retain support for the older
`data/five_dataset_hqq` schema: direct dataset folders, `dataset.json`, and 21
conditions with `[N,13,768]` feature arrays. The supplied tree lists only cotton,
cub, pets, soyageing and soyglobal in that directory. This legacy schema is
separate from the current paired-W3 launcher; no path-only conversion is used.
The user-provided tree also lists serials 20001–20005 and job 520062 logs.
Directory and log filenames alone do not establish successful completion.
