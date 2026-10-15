# Quantization analysis workspace

每個分析各自管理程式、資料與結果。使用 `opencode_env`，從此目錄執行 Python module。

| 分析 | 入口與說明 | 資料 | 結果 |
|---|---|---|---|
| Legacy HQQ accuracy ratio | [legacy_hqq/README.md](legacy_hqq/README.md) | `legacy_hqq/data/` | `legacy_hqq/results/` |
| Dataset robustness | [dataset_robustness/README.md](dataset_robustness/README.md) | `dataset_robustness/data/` | `dataset_robustness/results/` |
| Granularity robustness | [granularity_robustness/README.md](granularity_robustness/README.md) | `granularity_robustness/data/` | `granularity_robustness/results/` |
| Matched-sample HQQ confidence | [hqq_confidence/README.md](hqq_confidence/README.md) | `hqq_confidence/data/` | `hqq_confidence/results/` |
| Five-dataset features | [five_dataset_features/README.md](five_dataset_features/README.md) | `five_dataset_features/data/` (current local copy) | `five_dataset_features/analysis/` |
| Paired FP/W3 t-SNE | [sampling and nano4 commands](five_dataset_features/README.md#paired-fpw3-t-sne-on-nano4) | external paired raw exports | `<serial>/analysis/tsne_classes/` (10% classes, 5-20, all their samples) |
| Centered FP backbone class distances (20004) | [five_dataset_features/CENTERED_CLASS_DISTANCE.md](five_dataset_features/CENTERED_CLASS_DISTANCE.md) | external paired FP/W3 raw exports | `five_dataset_features/results/20004/centered_fp_distance_cal/` (documented CAL submission) |
| 16-dataset all-layer features (20004) | [five_dataset_features/README.md](five_dataset_features/README.md#16-dataset-all-layer-workflow-20004) | `data/20004/` | `five_dataset_features/results/20004/` |

## Common flow

```text
acquire (read/download) → prepare (validate/pair) → analyze (compute/report) → plot
```

The legacy and HQQ confidence workflows now have stage entry points:

```powershell
conda run -n opencode_env python -m legacy_hqq.run --stage all --run-name main
conda run -n opencode_env python -m hqq_confidence.run --stage all --run-name main
```

Both reuse existing raw data by default. A new W&B download requires missing input or an explicit `--refresh`. Dataset and granularity robustness currently retain their existing combined analysis command; their data and results have been separated, and the stage split is the next migration step. The five-dataset feature geometry test reads the current local copy under `five_dataset_features/data/` and writes separate outputs under `five_dataset_features/analysis/geometry_robustness/`.

The five-dataset inter/intra-class distance test uses the verified saved aggregate snapshot: `conda run -n opencode_env python -m five_dataset_features.distance_hypothesis`. Its tables and slide figures are in `five_dataset_features/analysis/distance_hypothesis/`.

Quantized inter/intra-class distances for all 100 quantized dataset-condition pairs run with `conda run -n opencode_env python -m five_dataset_features.quantized_class_distance`. The final CLS distances, matched FP changes, verification hashes, report, and PNG/SVG figures are in `five_dataset_features/analysis/quantized_class_distance/`.

FP-to-quantized distance drop fractions versus accuracy run with `conda run -n opencode_env python -m five_dataset_features.distance_accuracy`. Matched ratios, fixed-condition correlations, within-dataset A4-exclusion checks, reports, and figures are in `five_dataset_features/analysis/distance_accuracy/`.

The exploratory dataset loss-profile PCA runs with `conda run -n opencode_env python -m dataset_robustness.pca_profiles`. Its report and figures are in `dataset_robustness/results/pca_phase1/`.

The paired model-feature PCA runs with `conda run -n opencode_env python -m five_dataset_features.paired_feature_pca`, followed by `conda run -n opencode_env python -m five_dataset_features.verify_paired_feature_pca`. It compares full-test final CLS features in each dataset's FP PCA basis and writes its report, sample table, and figures to `five_dataset_features/analysis/paired_feature_pca/`.

See [docs/repository_layout_plan.md](docs/repository_layout_plan.md) for the target layout and remaining steps. Archived results from the old `results_all/` layout are kept in each analysis's `results/archived_*` directories. They are snapshots, while `results/main/` is the newly generated run.
