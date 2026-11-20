# Promote the corrected baseline results

Approved in this chat on 2026-10-21: replace the main checkout's default results
with the completed recovery, retain a reversible archive, commit locally, and
record the outcome in the analysis history (now docs/analysis_history_20261021.md). Source is codex/baseline-recovery-20007;
destination is C:/project/quant_analysis on main. Both start at 6409f80.

## Contract

- Keep the 15-dataset legacy/dataset/granularity cohort and 16-dataset 20007 cohort.
- Import immutable snapshots; regenerate manifests and results at their main paths.
- Preserve both existing worktrees, other analyses, archived results and raw features.
- Version the recovery and reproduction scripts; keep generated data/results ignored.
- Acceptance: 144 baselines, 4,608 full-grid pairs, 4,320 legacy cells, zero missing
  or unmatched, 144 W6A4 ratios, unchanged unaffected 3,840 cells, 30 regression
  tests plus all owning verifiers passing. Runtime inputs must resolve outside the
  source worktree; the main artifacts must match the completed recovery numerically.
- Back up the whitelist and its hashes before changing any default paths.
- Do not push or delete a worktree.

## Tasks

- [x] 1. Verify clean destinations; create Git backup refs and a file archive.
- [x] 2. Version the recovery/reproduction source and run its regression tests;
  obtain a fresh code review before integration.
- [x] 3. Commit the reviewed source and fast-forward quant_analysis main locally.
- [x] 4. Import snapshots, archive the old active directories, regenerate legacy
  main, downstream main/PCA, and outputs/serial_20007/results/main.
- [x] 5. Run all verifiers; compare with immutable reference tables, verify source
  hashes and absence of active worktree dependencies, and record the receipt.
- [x] 6. Update quant_analysis/docs/analysis_history_20261021.md, main report pointers and the existing
  TGDA/agent.md cross-project log; commit locally and verify the final state.

## Recovery

Archive: outputs/backups/baseline-promotion-20261021. Its manifest records the
exact pre-change files and absent paths. Restore only this whitelist, first
preserving any newly generated defaults; use the saved Git ref for source rollback.
Never recursively copy or remove outputs/worktrees as part of rollback.
