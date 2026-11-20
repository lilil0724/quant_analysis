# Nano4 existing-analysis runner implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline. Existing approved dual-input analysis methods and plots remain unchanged.

**Goal:** Provide one sbatch entrypoint for selected existing W3/G128 analysis stages on Nano4.

**Architecture:** `dataset_features/run_all_nano4.sh` sets the existing environment/resource defaults and invokes `dataset_features.run_all`. The Python coordinator plans and preflights selected original stages, executes their original CLIs, and records commands/status/logs in a fresh output directory. It does not implement statistical or plotting methods.

**Tech stack:** Bash/Slurm and existing Python analysis dependencies.

**Spec:** User asks for a new server analysis script, following the previously approved dual-input/input-only implementation.

## Constraints

- Fixed partition 8gpus/account MST114495; explicit resource and separate job logs.
- Default full 16-dataset, 3-model, FT/FZ/CAL matrix; allow CSV scope/stage selection and explicit source map.
- Reuse original default seeds, sampling, statistical methods and plots.
- Dry-run writes nothing and does not inspect input arrays. Preflight-only validates selected inputs without creating analysis outputs.
- Full runs refuse existing output directories, preflight all selected stages before analysis, and record failures.
- Margin matching uses its existing immutable input snapshot import; original source arrays are not changed.
- Tests/fixtures stay outside the repository and are cleaned; real Nano4 execution is not implied by local checks.

## Tasks

- [x] Add failing command-contract, output-protection and lifecycle checks in a temporary external directory.
- [x] Implement the coordinator and Slurm launcher.
- [x] Verify CLI forwarding, preflight/dry-run/failure behavior, Bash syntax and headers.
- [x] Update module/root README with copyable Nano4 commands, source-map and output locations; clean task temporaries.

## Verification and decisions

- 11 temporary checks passed with the explicit opencode_env Python in the sandbox, covering dry-run, existing-output protection, CSV versus space-list forwarding, invalid selections, failed-preflight stop/status, no-output preflight-only, success receipts, and input/output overlap.
- A real local subprocess integration using small temporary schema-2 fixtures completed all five original analyses, 11 coordinator operations and six original completion receipts. A separate explicit source-map preflight with a different cohort label passed without creating outputs. These are development checks, not server experiments.
- `bash -n`, Python syntax/import, required Slurm header checks and launcher-to-Python dry-run passed. Git Bash used the already justified narrowly scoped sandbox exception after its sandbox startup failure; analysis checks used the sandbox.
- Independent read-only review found no important issues. Centered preflight uses the original in-process validator and records its result in run.json/Slurm output rather than a separate subprocess log.
- Existing analysis functions, defaults and plots were not edited. The coordinator keeps each subflow's original reports and output contract; it does not generate a new statistical report.
- Margin matching intentionally uses its original copy/import workflow, retaining a separate immutable logits snapshot under the orchestration result tree. README documents the extra disk use and files normally excluded from local downloads.
- Raw Nano4 availability, full-matrix execution time/memory and Slurm submission remain unverified. No server job or new inference was executed.
- Test scripts, synthetic arrays, plots and intermediate logs were created outside the repository and removed after checks. Final deliverables are the new coordinator/launcher, README updates and this implementation record.
