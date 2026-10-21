# nano4 paired FP/W3 feature analysis

## 2026-10-21 規範採用狀態與 codebase

新規範見根目錄 [AGENTS.md](../AGENTS.md)。此模組有多種歷史輸出 schema，完整原始 inference
arrays 未全部位於本機，故本批保留現行 data/、analysis/、results/ 與 server
輸入根目錄，不以缺少 raw arrays 的情況宣稱已完成遷移或重算。

| 程式群 | 職責 |
|---|---|
| run.py、run_config.py、run_snapshot.py、unified_inputs.py | 16-dataset all-layer 分析的選取、身分／manifest 驗證與階段協調。 |
| layer_analysis.py、layer_pca.py、plot_run.py、report_run.py | 各層指標、PCA、圖表與報告。 |
| analyze.py、inputs.py、metrics.py、analysis_snapshot.py | 舊 21-condition schema 的讀取、統計與 snapshot 核對。 |
| paired.py、tsne_paired.py、paired_feature_pca.py | 成對 FP/W3 指標與共同取樣、t-SNE/PCA。 |
| centered_class_distance.py、quantized_class_distance.py、distance_hypothesis.py、distance_accuracy.py | class-distance 與 accuracy 關係分析。 |
| verify_paired_feature_pca.py、smoke.py | 驗證已取得的真實 artifact／real-batch 資料，不產生 synthetic fixture。 |

功能測試已封存至 outputs/layout_adoption_20261021/development_checks/。
smoke.py 雖名為 smoke，實際讀取與驗證真實推論輸出，屬正式資料驗證流程而保留。
新增功能測試使用 repo 外暫存目錄；歷史 source/hash receipts 原樣保存。


For global-mean-centered FP backbone class distances from existing Serial 20004
exports, use `python -m five_dataset_features.centered_class_distance` or the CPU
launcher `run_centered_class_distance_nano4.sh`. See
[CENTERED_CLASS_DISTANCE.md](CENTERED_CLASS_DISTANCE.md) for input requirements,
CAL-only/full-matrix submission commands, metric definitions and output paths.

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

<<<<<<< HEAD
## Analysis snapshot and targeted reruns

Each dataset analysis writes `analysis/<dataset>/analysis_snapshot.json`. It records the checked `dataset.json` and 21 condition-manifest hashes, the analysis-code hashes, and the derived-file hashes. The `report` stage verifies all five dataset snapshots before reading their results and records their identities in `analysis/analysis_snapshot.json`. The `package` stage verifies these snapshots and then repeats the existing full raw-file checksum validation before writing the ZIP. Both snapshot files are included in the ZIP.

An older analysis without snapshots, a changed input manifest, changed analysis code, or a changed derived file stops the next stage with a rerun instruction. `report` checks manifest and snapshot hashes without rereading all feature arrays; `package` validates every raw file against its condition manifest.

If one dataset's inference output changes, rerun that dataset, then the aggregate report and package:

```powershell
python -m five_dataset_features.analyze dataset --root $localRoot --dataset cub
python -m five_dataset_features.analyze report --root $localRoot
python -m five_dataset_features.analyze package --root $localRoot
```

If `analyze.py`, `metrics.py`, or `analysis_snapshot.py` changes, rerun all five dataset analyses and the report with `-Stage analyze`, then run `-Stage package`. Existing outputs created before snapshot support also require this full analysis rerun.

## Analysis definitions
=======
```bash
# One dataset/model/mode pilot, still using the complete test split.
sbatch five_dataset_features/run_nano4.sh --serial 20005 \
  --datasets cub --models vit_b16 --modes ft
>>>>>>> 8dea61bfc1447b4788acbe69d706d79f78a728bb

# Read-only input validation; no analysis outputs.
sbatch five_dataset_features/run_nano4.sh --serial 20005 --stage preflight

# Analyze and report, without ZIP.
sbatch five_dataset_features/run_nano4.sh --serial 20005 --stage analyze

<<<<<<< HEAD
`report` computes mean loss by activation family within each of the five datasets, then descriptive Spearman correlations across the five dataset-model pairs. It does not count the 20 quantized conditions as 100 independent samples. Empty CSV cells mean undefined. No significance claim is made. Inference on nano5 chooses ten classes with at least two test images using seed 100, saves two 224px display images per class, and records a checksum manifest. Local packaging verifies those copied files, then builds an explicit whitelist ZIP and prints its size. The 50 MB limit is a target; inspect the printed size. The package excludes all full-test feature arrays, full images, and checkpoints.

## W3/G128 presentation figures

From the parent directory of this module, regenerate three 16:9 figures from the
saved `analysis/` CSVs:

```powershell
conda run -n opencode_env python -m five_dataset_features.plot_presentation --root .\five_dataset_features
```

The default output is `analysis/slides_w3_g128/`, with PNG and editable SVG
versions of each figure. `w3_g128_layer_cka` compares the five dataset-specific
CKA curves for weight-only, Linear A8, and Linear A4. `w3_g128_a4_cka_vs_loss`
shows the fixed-condition final CKA and relative accuracy loss, alongside
accuracy, class separation, prediction swaps, and damage/rescue counts.
`w3_g128_layer_cka_qkv` is the matching slide with weight-only, Linear A8 +
QKV A8, and Linear A4 + QKV A4. Its A8/A4 curves use the `_qkv` conditions;
the other two slides continue to use conditions without QKV quantization.
These are descriptive full-test-set comparisons, not five independent model
replications. The script checks that every input CSV matches its recorded
analysis snapshot hash. It does not require the original inference arrays or
reverify their manifests; use the full analysis workflow when those inputs
are available and need to be revalidated.

## Feature geometry and quantization damage

Run the descriptive nearest-rival geometry analysis against the full copied
inference output. From the repository root:

```powershell
conda run -n opencode_env python five_dataset_features/geometry_robustness.py `
  --input-root five_dataset_features/data
```

The script verifies the saved analysis snapshot, then verifies the original
dataset manifests and SHA-256 hashes for FP and the four weight-only conditions.
It pairs sample IDs and targets before comparing normalized final CLS features.
Outputs stay in `five_dataset_features/analysis/geometry_robustness/`: CSVs,
`RESULTS.md`, `run_config.json`, and 16:9 PNG/SVG presentation figures. Without
`--input-root`, only the aggregate figures can be regenerated from saved CSVs.

The sample test uses leave-one-out test-set class centroids and is descriptive.
It reports C→W rates for low/high geometric-gap quintiles, repeats the comparison
within FP normalized-logit-margin quintiles, and reports damage AUC for both
scores. Each dataset has its own checkpoint and one seed; repeated quantization
conditions are not independent model replications.

## Inter/intra-class distance hypothesis

To test whether closer FP class distances accompany larger weight-only accuracy
loss, run from the repository root:

```powershell
conda run -n opencode_env python -m five_dataset_features.distance_hypothesis
```

This stage verifies the saved aggregate analysis snapshot and reads its
`fp_indicators.csv` and `all_conditions.csv`; no raw feature arrays or new W&B
download are needed. It writes `analysis/distance_hypothesis/` with a dataset
summary, all four fixed-condition losses, Spearman correlations, a concise
`RESULTS.md`, and three 16:9 PNG/SVG figures. `run_config.json` records the
source paths, SHA-256 hashes, snapshot hash, reference precision, and conditions.

The FP distance gap is `D_inter - D_intra`, and the ratio is
`D_inter / D_intra`; smaller values mean closer distances. The two outcome
measures are relative accuracy loss `(FP - Q) / FP` and absolute accuracy drop
`FP - Q`. Each of the five observations combines one dataset with its own
fine-tuned checkpoint and one seed, so these associations cannot identify a
dataset-only causal effect. The four conditions are checked separately and are
not treated as independent dataset replications.

## Quantized inter/intra-class distances

Summarize final normalized CLS (layer 12, 768D) cosine distances for all five
datasets and all 20 quantized conditions, with each dataset's matched AMP-FP16
reference:

```powershell
conda run -n opencode_env python -m five_dataset_features.quantized_class_distance
conda run -n opencode_env python -m unittest five_dataset_features.tests.test_quantized_class_distance -v
```

This stage verifies the saved aggregate/dataset snapshots, analysis-code hashes,
and input CSV hashes. It reuses the full-test-set distances already computed in
`layer_metrics.csv`; it does not reread or revalidate the raw inference arrays.
Outputs stay in `analysis/quantized_class_distance/`: `final_class_distances.csv`
(105 rows including FP), Traditional Chinese `REPORT.md`, PNG/SVG gap figures,
and `run_config.json` with input/output SHA-256 and saved-table verification.
The table includes inter, intra, gap, ratio, Q-minus-FP changes, gap retention,
accuracy ratio, relative loss, and absolute accuracy drop (accuracy units 0–1).
Classes use ground-truth labels, with equal class/class-pair weights; inter is
the mean cross-class sample-pair distance, not normalized-centroid distance.
For another saved analysis directory, pass `--analysis-root <path>`; the output
is its `quantized_class_distance/` subdirectory.

## Distance drop versus accuracy

Compare FP-to-quantized proportional drops in inter, intra, gap, and inter/intra
ratio with matched accuracy, relative accuracy loss, and absolute accuracy drop:

```powershell
conda run -n opencode_env python -m five_dataset_features.distance_accuracy
conda run -n opencode_env python -m unittest five_dataset_features.tests.test_distance_accuracy -v
```

This reuses the verified saved full-test final CLS distances and accuracy rows;
it does not reread raw inference arrays. Outputs are in `analysis/distance_accuracy/`:
100 paired rows, fixed-condition and within-dataset correlation CSVs,
Traditional Chinese `REPORT.md`, PNG/SVG figures, and input/code/output SHA-256
with saved-table verification in `run_config.json`. Drop fraction is `1 - Q/FP`;
absolute accuracy drop is also reported in percentage points. Undefined distance
denominators stay empty. Intra drop below zero means increased intra distance.
Fixed-condition correlations use five dataset-checkpoint pairs and include
leave-one-dataset-out sensitivity. Within-dataset results use 20 repeated
conditions, with A4-excluded (12) and weight-only (4) checks. They are descriptive,
with no pooled independent-sample significance or causal claim.

## Paired final CLS feature PCA

From the repository root, analyze the matched full-test-set final CLS features
for the AMP-FP16 reference and two fixed weight-only HQQ conditions:

```powershell
conda run -n opencode_env python -m five_dataset_features.paired_feature_pca
conda run -n opencode_env python -m five_dataset_features.verify_paired_feature_pca
conda run -n opencode_env python -m unittest five_dataset_features.tests.test_paired_feature_pca -v
```

The input is the current local copy in `five_dataset_features/data/`; existing
PCA bases and analysis snapshots are in `five_dataset_features/analysis/`.
The analyzer validates the original source manifests, file hashes, matched
sample identities, prediction rows, saved PCA basis, and analysis-code hashes
across those two roots. Its separate outputs are in
`five_dataset_features/analysis/paired_feature_pca/`: `REPORT.md`, dataset
figures, `summary.csv`, `samples.csv`, `cross_dataset_spearman.csv`, and
`run_config.json`. The verifier writes `verification.json` after independently
checking output hashes and key sample-level calculations.

For each dataset, the saved PCA basis was fit on its FP final CLS features.
FP, W3/G128, and W4/G128 use that same basis within the dataset. The paired
arrows use the true 2D scale, while the tables use every matched image and
measure full 768D feature drift, prediction transitions, and the fraction of
quantization drift energy visible in the first two PCs. PCA axes differ across
datasets. The W3 condition is the main comparison and W4 is a sensitivity
check. Correlations across the five dataset-checkpoint pairs are descriptive;
the distinct checkpoints, classes, and single training seed confound any
dataset-only causal reading.

## 16-dataset, all-layer workflow (20004)

This config-driven workflow analyzes a separate completed inference export. It
does not change the existing five-dataset commands or snapshots. Raw inputs
belong under the repository's `data/20004/`; outputs go to
`five_dataset_features/results/20004/`.

The reader currently accepts the manifest-backed `tgda_manifest_v1` layout:
one `<dataset>/dataset.json` with one child directory per condition, each with
`manifest.json`, `features.npy`, and `predictions.csv`. It checks that declared
file hashes match and aligns both predictions and features to canonical IDs
from `dataset.json`. Feature rows follow the parallel `predictions.csv` rows;
if the CSV was sorted independently, its manifest must explicitly record
`feature_sample_ids` in physical array-row order. Reordering uses a lazy mmap
view and materializes only requested chunks. It refuses unknown
feature-position semantics or schema: `feature_shape` and `position_names`
must be declared in metadata or the config, with conflicts rejected. Hash
field presence alone does not establish an exporter version. For a legacy
ViT-B/16 export without these declarations, inspect its exporter and explicitly
set `[13,768]` plus `block1` through `block12`, followed by `encoder_norm`;
there is no automatic layer-mapping fallback. The exporter’s `exemplars/` metadata and hidden/incompatible
backup directories are excluded from condition discovery. Inventory must find
the configured 16 datasets and a complete condition grid before validation succeeds.

Create `five_dataset_features/config/20004.json` from
`config/20004.example.json`, then set the primary condition and any conditions
present in the inventory. Leave the group mapping empty until a verified
16-dataset file with columns `dataset_name,core_group,expanded_group` is ready;
core/expanded labels must be `FGIR`, `UFGIR` (or `Ultra-FGIR`), or blank for a
boundary dataset. The older 15-dataset mapping is not silently reused.

Run from the repository root, first to inspect source coverage and then to
validate every raw file:

```powershell
conda run -n opencode_env python -m five_dataset_features.run --config five_dataset_features/config/20004.json --stage inventory
conda run -n opencode_env python -m five_dataset_features.run --config five_dataset_features/config/20004.json --stage validate
```

`validate` checks and hashes all feature arrays, predictions, probes, manifests,
sample identities, labels, feature dimensions, and finite values. It freezes
the dataset/context/condition/position cohort in `results/20004/state/`.
It also writes the currently resolved settings to
`results/20004/state/run_config.json`, including Python, NumPy, SciPy, and
Matplotlib versions; stage fingerprints include those versions. A standalone metrics/PCA/comparison/plot/
report stage rechecks the validated source hashes in its scope before using
cached or raw inputs; `--stage all` performs full validation again.
After validation, run one dataset through metrics and PCA before the complete
pipeline:

```powershell
conda run -n opencode_env python -m five_dataset_features.run --config five_dataset_features/config/20004.json --stage metrics --dataset <dataset_name>
conda run -n opencode_env python -m five_dataset_features.run --config five_dataset_features/config/20004.json --stage pca --dataset <dataset_name>
conda run -n opencode_env python -m five_dataset_features.run --config five_dataset_features/config/20004.json --stage all
```

The available stages are `inventory`, `validate`, `metrics`, `pca`,
`comparisons`, `plot`, `report`, `verify`, and `all`. `all` rechecks the full
input hashes and reuses only stages whose code, config, inputs, receipts, and
output hashes still match. A partial `--dataset` metrics/PCA run cannot produce
cross-dataset summaries. `verify` rehashes the original arrays again before it
records `state/verification.json`. To intentionally change a locked cohort,
inspect the new inventory and rerun validation with `--rebuild`; the old lock
is archived by hash.

The workflow produces per-position CKA, class-equal inter/intra cosine distance
and signed FP/Q changes, accuracy ratios and absolute drops, fixed-condition
dataset comparisons with bootstrap/permutation summaries, matched-N CKA
sensitivity, and one FP PCA basis per dataset/position shared by all conditions.
Per-context metrics also include `metric_status.csv`; `comparisons.csv` includes
leave-one-dataset-out correlation ranges so a single dataset's influence is
visible. Inference-position metrics and full-dimensional PCA displacement use
all paired samples, while PCA figures use a deterministic display subset.
The final report and figures stay under `five_dataset_features/results/20004/`.
Unknown checkpoint/model/seed metadata is labeled as a limitation and cannot
support formal cross-dataset permutation conclusions.

Cross-dataset strata also include the recorded `execution` settings and
preprocessing, even if `inference_profile` is present; AMP on/off are separate
cohorts. Missing execution metadata restricts results to descriptive output.
Undefined CKA/accuracy ratios are filtered per comparison and retained with
coverage and reasons, including undefined group CKA. Within-dataset associations
are separated into `weight_only`, `A8`, `A4`, and `QKV` using manifest settings
(not condition-name guessing); unknown families stay undefined. These rows are
available in `comparisons.csv` and the report, including for a single dataset.

Functional regression sources have been archived with hashes under
`outputs/layout_adoption_20261021/development_checks/`. Run development checks
from a temporary location outside the production tree. Production validation
still checks sample pairing, feature semantics, coverage and source corruption.

=======
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

## Paired FP/W3 t-SNE on nano4

`run_tsne_nano4.sh` reads the original serial's `samples.json`,
`fp/final.npy`, and `w3_g128/final.npy` files. The compact local download does
not include these arrays, so generate the plots where the source exports live.
The Python environment needs NumPy, Matplotlib, and scikit-learn; the launcher
defaults to `tgda`, matching the existing nano4 feature-analysis launcher, and
accepts `ANALYSIS_CONDA_ENV` or `ANALYSIS_PYTHON` as an override.

```bash
cd /work/kyle0724/project/quant_analysis
bash five_dataset_features/run_tsne_nano4.sh --serial 20004 --dry-run
sbatch five_dataset_features/run_tsne_nano4.sh --serial 20004 --stage preflight
sbatch five_dataset_features/run_tsne_nano4.sh --serial 20004 \
  --datasets cub --models vit_b16 --modes ft
sbatch five_dataset_features/run_tsne_nano4.sh --serial 20004
```

The default final job selects all 16 datasets, three models, and three modes
(144 plots). The default `--sampling classes --class-fraction 0.1` selects
`min(C, 20, max(5, ceil(0.1 * C)))` classes, where C is the number of observed
test classes. It randomly chooses class IDs using `--seed`, then keeps **all
test samples in those classes**, with no 2,000-point truncation. For example,
Pets selects 5 classes, Cotton 8, and CUB/SoyGlobal/iNat17 20 each. Class IDs
are selected from sorted labels, independently of sample order/populations,
so the same dataset uses the same IDs across models/modes; preflight rejects
a selected-class mismatch before writing any plots. A dataset with fewer than
5 classes keeps all its classes. `--class-fraction` can change the fraction;
the 5-20 bounds still apply.

Each selected class uses its own discrete color, with every class ID and its
sample count labeled on the color key. Source arrays remain unchanged.
Preflight prints selected classes, paired samples and joint fit points for
each pair. Keeping all selected-class samples can use more time/memory than
the old budget, especially for populous classes. Preflight does not measure
the eventual t-SNE runtime or memory use.

To reproduce the previous sampling policy, use `--sampling samples
--max-samples 2000`: this covers every class when the point budget permits;
if there are more classes than the budget, it chooses fewer classes with
repeated examples where possible. `--max-samples` is ignored in classes mode.
It L2-normalizes final features,
optionally reduces more than 50 dimensions with one shared PCA-50 fit, and fits
one t-SNE to the concatenated FP/W3 samples. Both panels use the same resulting
coordinates and class colors. `--class-fraction`, `--perplexity`, `--iterations`,
and `--seed` control cost and reproducibility. Each pair writes `tsne.png`,
`coordinates.csv` (the selected original row indices and paired coordinates),
and `method.json` under `<input-root>/20004/analysis/tsne_classes/<dataset>/<model>/<mode>/`.
Legacy samples mode defaults to `analysis/tsne/`. `--output-root` overrides
either destination. Method metadata records sampling parameters, selected
class IDs, and per-class sample counts along with source/output hashes.
The source export remains untouched. Matching completed outputs are skipped on
rerun; `--overwrite` recomputes generated plots when parameters change.

Preflight requires every selected source pair and checks its identity, W3/g128
configuration, sample pairing, final feature shapes, and source hashes before
plotting. Start with ViT-B/16 and inspect the complete selected populations:

```bash
sbatch five_dataset_features/run_tsne_nano4.sh --serial 20004 \
  --models vit_b16 --stage preflight
sbatch five_dataset_features/run_tsne_nano4.sh --serial 20004 \
  --models vit_b16
```

A full job may take substantial CPU time; start with one pair to
measure runtime on nano4. t-SNE axes are arbitrary, plots fitted for different
pairs have unrelated coordinates, and dense or sparse regions are exploratory
visuals rather than accuracy or causal metrics. The class subset is selected
**before** PCA/t-SNE fitting: it shows local structure among those classes,
not the full dataset's class separation. Removing other classes can change
the neighborhoods. Repeat sampling seeds to assess subset dependence rather
than choosing the most separated plot. CAL features are from the first
raw-image backbone call, whereas CAL accuracy uses crop/flip aggregation.

## Legacy 21-condition workflow

`analyze.py`, `smoke.py`, and `run_local.ps1` retain support for the older
`data/five_dataset_hqq` schema: direct dataset folders, `dataset.json`, and 21
conditions with `[N,13,768]` feature arrays. The supplied tree lists only cotton,
cub, pets, soyageing and soyglobal in that directory. This legacy schema is
separate from the current paired-W3 launcher; no path-only conversion is used.
The user-provided tree also lists serials 20001–20005 and job 520062 logs.
Directory and log filenames alone do not establish successful completion.
>>>>>>> 8dea61bfc1447b4788acbe69d706d79f78a728bb
