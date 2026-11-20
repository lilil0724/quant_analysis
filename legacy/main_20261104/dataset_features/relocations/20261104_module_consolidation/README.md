# Feature module relocation — 2026-11-04

This is a filesystem relocation record, not a new inference or analysis run.
The user requested consolidating local inputs and results under the current
`dataset_features` module and fixing indexes and operational paths.

## Files and path mapping

- [receipt.json](receipt.json): timestamps/timezone, checkout identity, counts,
  status, manifest hashes and verification boundary.
- [files.csv](files.csv): every relocated file's old path, new path, byte size,
  SHA-256 and original last-write time in UTC.
- [existing_artifacts.csv](existing_artifacts.csv): the 70 pre-existing historical
  destination artifacts; all were verified unchanged after the merge.
- [directories.json](directories.json): relative directory inventory, including
  empty directories retained from the old tree.

The old module contained 1,418 files totaling 218,255,389 bytes. All were moved
without overwriting destinations, then checked against their pre-move size and
SHA-256. The empty source tree was removed. The moved tree includes existing
Python caches and historical hidden scratch directories; nothing was silently
discarded or promoted to production code.

For paths inside historical receipts or reports, replace the module prefix:

```text
five_dataset_features/data/...     -> dataset_features/data/...
five_dataset_features/analysis/... -> dataset_features/analysis/...
five_dataset_features/results/...  -> dataset_features/results/...
```

This also applies to absolute paths under the same repository. Use `files.csv`
for exact moved-file mappings; existing tracked `dataset_features/analysis/`
files were already in the destination. Historical manifests, ZIP contents and
completed reports retain their original bytes and provenance paths.

## Operational changes and boundaries

Live module CLI paths, three Slurm launchers, config defaults/example, root/module
indexes and the dependent TGDA paired-baseline CSV path now use `dataset_features`.
Archive source snapshots and dated plans/analysis reports remain historical;
the archive's navigation README points to the current module.

This keeps the current `data/`, `analysis/` and `results/` layouts. It does not
complete the future per-run `result/` and unique `reports/` migration, rewrite
saved code-identity receipts, or establish availability of the nano4 raw arrays.
Existing run resume checks must still respect their original source/code hashes;
new computation must use a new output location.

New runtime artifacts under data/analysis/results are ignored by Git. Existing
tracked historical summaries retain their tracking. Back up the ignored trees
separately: this receipt and a Git branch are not copies of the moved data.

Verification evidence is recorded in [validation.json](validation.json):

- 1,418 relocated files and 70 existing artifacts retain their size/SHA-256.
- Real Serial 20004 compact data retains all 144 dataset/model/mode pairs,
  4,896 layer rows and 579 available saved-output hashes. Summary/table values,
  accuracy drop, relative loss, damage/rescue rates, class-distance gaps and norm
  ratios agree within absolute/relative tolerance `1e-12`.
- 288 full PCA coordinate files are absent from this compact copy as expected;
  upstream raw-source hashes cannot be revalidated from recomputed output files
  that happen to share names such as `summary.json` or `layer_metrics.csv`.
- 14 archived-source hashes, 17 current Python sources, three Slurm headers and
  51 local navigation links/anchors pass checks. All 1,393 moved runtime files
  outside Python caches are ignored; no data/results files are Git-tracked.
- All five current module CLIs pass `--help`; three launchers pass `bash -n`;
  paired/t-SNE launcher dry-runs and the centered-distance CLI dry-run retain
  144 pairs and use the new module paths. The dry-runs create no result files.
- A focused independent review found no actionable issues. Final diffs pass
  whitespace checks in both repositories.

Commands include `conda run -n opencode_env python -B -m dataset_features.run
--help`, the analogous `paired`, `tsne_paired`, `centered_class_distance` and
`unified_inputs` help commands, and `bash dataset_features/run_nano4.sh --serial
20004 --dry-run` / `bash dataset_features/run_tsne_nano4.sh --serial 20004 --dry-run`.
Temporary verification scripts were run outside the repository and removed.
File relocation proves byte preservation; compact-data checks cannot replace
full raw-array pairing and feature-formula validation on nano4.
