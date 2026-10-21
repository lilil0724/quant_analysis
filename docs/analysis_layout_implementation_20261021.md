# Analysis layout adoption — Implementation Plan

> For agentic workers: use superpowers:executing-plans inline. User approved the specification and requested execution in this checkout; do not recreate worktrees, commit or push implicitly.

**Goal:** Adopt the approved conventions and complete their first migration batch: repository inventory, operational rules, W6A4 pilot and development-test archival.

**Architecture:** Each analysis retains its statistical implementation. W6A4 becomes a self-contained module with immutable local inputs, per-run result directories and one tracked report. Other analyses keep working paths until their own verified migration; historical data and receipts stay intact.

**Tech Stack:** Python in opencode_env, pandas, NumPy, Matplotlib, PowerShell for verified local archival.

**Spec:** [analysis_conventions_proposal.md](analysis_conventions_proposal.md), approved 2026-10-21.

## Global constraints

- Inputs in the owning analysis data/; generated files in result/<run_id>/; one REPORT.md and referenced figures in reports/<run_id>/.
- Same-input W6A4 migration preserves all 144 contexts, selected run identities and ratios.
- Preserve source manifests and original artifacts; track relocation rather than editing historical receipts.
- Functional test code belongs outside the production codebase. Keep actual data validation in the analysis flow.
- No new downloads, GPU inference, worktrees, commits or pushes.
- New rules apply to migrated analyses; label pending legacy paths honestly.

## Review focus

- Input tampering, wrong fixed W6A4 settings and duplicate/incomplete identities must fail before a complete result is published.
- A reused input ID or run ID must not overwrite existing files, including after failure.
- Failure reports must identify the error and input/code identity; successful run and report hashes must agree.
- Every report image must survive without ignored data and match the full output image.
- Historical code/data and tests must remain recoverable after archival; other module entrypoints must remain valid.

## Task 1: Preservation and adopted rules

- [x] Save Git status, tracked source copies, all module data/output inventories and SHA-256 into an ignored maintenance backup.
- [x] Adopt the approved spec; consolidate operating rules in AGENTS.md and preserve the historical record in docs/analysis_history_20261021.md, preserving Slurm requirements.
- [x] Update ignore rules without removing local data. Inspect tracked inputs/outputs before changing the index.

## Task 2: W6A4 pilot

- [x] Create w6a4_analysis/{README.md,__init__.py,acquire.py,analyze.py,run.py}.
- [x] Import existing cohort/target snapshots and a reference pair table into data/baseline_20261021/; record original receipts, paths and hashes.
- [x] Use temporary tests outside the repo to check immutable IDs, fixed settings, full coverage, hash rejection and source identity; observe failures before implementation.
- [x] Generate a new real-data run, compare IDs and numeric values with all 144 existing reference rows, and create exactly one Markdown report with referenced figures.
- [x] Expose read-only --verify-run for input/output/image/formula/reference checks.

## Task 3: Clean production surface

- [x] Archive all module tests/ directories with hashes, then remove those directories from the production tree. Retain production data validators.
- [x] Update module READMEs to describe the codebase and current migration status; remove stale test commands and update the root index to the new W6A4 entrypoint.
- [x] Mark the prior repository_layout_plan.md as historical where conventions differ; retain its history.

## Task 4: Verification and handoff

- [x] Run temporary checks and the new W6A4 real-data verifier; inspect the saved report and heatmap.
- [x] Confirm unchanged historical main input/output hashes, no tracked runtime data, no production tests/, correct ignore behavior and no new worktrees.
- [x] Obtain a read-only fresh review of the final change, fix material findings, and record results in the maintenance receipt.

## Execution record

Execution is local and uncommitted. Maintenance backup and verification receipt will be recorded here after verification. Remaining modules' full result/report migration is a subsequent batch; this plan does not claim those paths already comply.

First batch completed locally on 2026-10-21. Backup: outputs/layout_adoption_20261021/before/manifest.json (2431 files, 591913564 bytes). Archived102development files; preserved2301historical artifacts and74existing production Python files unchanged. External functional checks:10passed. Final W6A4 run: `20261021_163522_084281_w6a4_s20007`;144matched pairs,6input hashes,15artifact hashes,1Markdown report. Fresh review found1material finalization-I/O issue; fault tests failed before the atomic-publication fix and passed afterward. Full receipt: outputs/layout_adoption_20261021/verification.json. All changes remain uncommitted; no push. Other module migrations remain pending.
