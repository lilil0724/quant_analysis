# Corrected baseline promotion — 2026-10-21

This record defines the approved promotion from codex/baseline-recovery-20007
to the local quant_analysis main checkout. The file archive is
`outputs/backups/baseline-promotion-20261021/`; its manifest records 215 original
files, their SHA-256 hashes, absent destinations and the pre-change commits.
Git ref `codex/backup-before-baseline-promotion-20261021` preserves 6409f80.

## Formal outputs

| Analysis | Current report |
|---|---|
| Legacy accuracy ratio | [report](../legacy_hqq/results/main/corr/report.md) |
| Dataset robustness | [report](../dataset_robustness/results/main/report.md) |
| Loss-profile PCA | [report](../dataset_robustness/results/pca_phase1/report.md) |
| FGIR / UFGIR | [report](../granularity_robustness/results/main/report.md) |
| Serial 20007 W6A4 | [report](../outputs/serial_20007/results/main/report.md) |

The owning stages regenerate these outputs at their formal locations. Runtime
receipts and source paths refer to this checkout, not the source worktree.
The previous dated recovery and comparison remain historical reference runs.
Historical comparison source paths that named the old main files are resolved
by [the relocation receipt](../outputs/baseline_recovery/comparison_20261021/source_relocation.json)
to hash-matching files in the pre-promotion backup. Use that backup when doing a
new before/after comparison; main now contains the corrected values.

## Provenance and contract

- `data/baseline_recovery/source/`: verified historical raw snapshot and manifest.
- `data/baseline_recovery/snapshots/20261021/`: the 3,762 finished runs downloaded
  from W&B on 2026-10-21, including all 495 corrected BEiTv2 evaluations.
- `data/baseline_recovery/selected_20261021/`: regenerated main-local receipt,
  selected runs, pair audit, coverage, 15/16-dataset curated CSVs.
- 990 unaffected archived runs were restored because they were removed from the
  current W&B export. The selected source is 144 baselines + 4,608 HQQ conditions.
- Legacy/dataset/granularity preserve the 15-dataset cohort; 20007 includes 16.
- `data/baseline_recovery/reference_20261021/` freezes the completed worktree's
  expected tables and their hashes for post-promotion numerical comparison.

Baseline means unquantized AMP. The five datasets dogs/flowers/moe/pets/vegfru
use corrected BEiTv2 FT/FZ cls and CAL pool, replacing FP and HQQ together.
No new training or inference is needed to regenerate these saved-result analyses.

## Reproduce

Run with `conda run --no-capture-output -n opencode_env` from the repository root:

```powershell
python -m legacy_hqq.recover_baselines
python -m legacy_hqq.run --stage all --run-name main --source-file data/baseline_recovery/selected_20261021/curated_15_datasets.csv --refresh
python -m dataset_robustness.analyze
python -m dataset_robustness.pca_profiles
python -m granularity_robustness.analyze
python -m tools.baseline_recovery.replot_20007
```

Before downstream reruns after a changed legacy source, refresh their fixed
`data/upstream/` snapshots, preserving source hashes and the legacy audit/XLSX/
plot companions consumed by the dataset verifier. Their summary bytes must match
`legacy_hqq/data/processed/main/quant_summary.csv`.

The versioned acquisition tools are `tools.baseline_recovery.acquire_grid`
(`--data-dir`) and `tools.baseline_recovery.download_20007` (`--raw-dir`).
Download into a fresh snapshot directory, preserving the published input hashes.

## Verification and recovery

Run the three owning verifiers followed by the promotion verifier:

```powershell
python -m dataset_robustness.verify
python -m dataset_robustness.verify_pca
python -m granularity_robustness.verify
python -m tools.baseline_recovery.verify_main
```

The final receipt is [promotion verification](../outputs/baseline_recovery/promotion_20261021/verification.json).
Required counts are 144 baselines, 4,608 full-grid pairs, 4,320 matched legacy
cells, 144 W6A4 ratios, no missing/unmatched cells, and 3,840 unaffected legacy
cells equal to the archived original. The regression suite has 35 tests,
including five added source/promotion-integrity guards. Record actual completion below.

Rollback is local: preserve any new defaults, restore only the manifest's listed
paths from `outputs/backups/baseline-promotion-20261021/files/`, remove only paths
listed as originally absent if rollback requires it, and restore source from the
saved Git ref. Revalidate restored hashes. Keep both worktrees and all other
analysis directories intact. Git alone does not restore ignored data/results.

## Execution

Promotion completed on 2026-10-21. Source commit `195c485` was fast-forwarded
locally into main. All default results were rebuilt at their formal main paths.
The final promotion gate passed at 10:09 Asia/Taipei: 144 baselines, 4,608 full-grid
pairs, 4,320 matched legacy cells, 144 W6A4 ratios, zero missing/unmatched cells,
all values equal to the completed recovery, and 3,840 unaffected legacy values
unchanged. The 35 regression tests passed in main. The gate reran the three owning
verifiers and checked raw/import-receipt integrity and W6A4 artifact hashes.

Visual checks covered the formal 20007 heatmap and the default PCA score plot.
No new inference, training, remote push, worktree removal or unrelated result
replacement occurred. The existing TGDA/agent.md also records the promotion.
