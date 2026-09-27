# Five-dataset feature analysis

This is independent of the legacy 13-condition quantization analyzer and the W&B artifact workflow. Run it **on the local Windows computer** after copying the file-only TGDA inference output from nano5. The server runs dataset inference only. Local analysis needs the full `[N,13,768]` feature arrays for all conditions, not just the final small ZIP.

## Transfer and run

On local PowerShell, replace the remote host and the server config's absolute `output_root`:

```powershell
$remote = 'user@nano5'
$remoteRoot = '/absolute/path/from/experiment.nano5.json/output_root'
$localRoot = 'D:\five_dataset_hqq'
New-Item -ItemType Directory -Force -Path (Join-Path $localRoot 'smoke') | Out-Null
scp -r "${remote}:${remoteRoot}/smoke/cub" (Join-Path $localRoot 'smoke')
& 'C:\project\quant_analysis\five_dataset_features\run_local.ps1' -Stage smoke -InputRoot $localRoot
```

Inspect `analysis/smoke_precision.json` locally before submitting the full nano5 inference array. For a standalone smoke copy placed directly at `<InputRoot>/cub`, the smoke stage also accepts that layout. Its report includes sample pairing and checksum validation, prediction changes, final-feature metrics for the 64 selected images, and FP16 storage checks against the incremental W3-to-A8 and A8-to-QKV changes.

When all five server inference jobs finish, copy their outputs to the same local root and run the complete local workflow:

```powershell
foreach ($dataset in @('cotton', 'soyageing', 'soyglobal', 'cub', 'pets')) {
    scp -r "${remote}:${remoteRoot}/$dataset" $localRoot
}
& 'C:\project\quant_analysis\five_dataset_features\run_local.ps1' -Stage all -InputRoot $localRoot
```

The script uses `C:\Users\Public\miniconda3\envs\opencode_env\python.exe` by default. Pass `-PythonExe` only if that environment is installed at another path. For targeted reruns, use `-Stage analyze` (five datasets and report) or `-Stage package` (small ZIP only). All analysis and packaging outputs go under `$localRoot\analysis`; the final ZIP is `$localRoot\analysis\five_dataset_hqq_results.zip`.

The FP16 features alone are estimated at **8.48 GB plus about 0.419 GB per 1,000 cotton test images**. Allow additional local disk for predictions, downloaded folders, figures, and temporary package staging. `scp` copies full dataset directories; check whether earlier `.invalid.*` backups exist if the transfer is unexpectedly large. Dataset analysis validates all 21 condition checksums, shapes, sample IDs, and targets before using the results.

## Analysis definitions

The dataset stage rejects missing or corrupt conditions and verifies all sample IDs and targets. CKA is centered linear CKA using feature cross-products, never an N-by-N sample matrix. Cosine intra-class distances average distinct sample pairs within each class, then classes equally; inter-class distances average class pairs equally. The final `D_inter - D_intra` is the primary feature separability score. H-score uses within-class scatter regularization of 1% of mean within-feature variance. Norm P50/P95/P99 and P99/P50 use unnormalized features. The final FP feature measures are candidate predictors; per-layer results are supplementary.

FP final features fit one PCA basis per dataset, and every condition uses that basis. Local dataset output includes full-test 2D coordinates for each condition; the small ZIP only includes the 20 representative samples per dataset. The first real batch's FP32 activations are compared with their FP16 roundtrip in `precision_probe.csv`. Inspect A8 rows before accepting FP16 feature storage; if storage error is material, set `feature_dtype` to `float32` for the affected dataset in the server experiment JSON, rerun its inference, copy the new output, and re-evaluate disk capacity.

`report` computes mean loss by activation family within each of the five datasets, then descriptive Spearman correlations across the five dataset-model pairs. It does not count the 20 quantized conditions as 100 independent samples. Empty CSV cells mean undefined. No significance claim is made. Inference on nano5 chooses ten classes with at least two test images using seed 100, saves two 224px display images per class, and records a checksum manifest. Local packaging verifies those copied files, then builds an explicit whitelist ZIP and prints its size. The 50 MB limit is a target; inspect the printed size. The package excludes all full-test feature arrays, full images, and checkpoints.
