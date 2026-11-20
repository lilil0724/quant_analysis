# Feature analysis code cleanup implementation plan

> Execute inline in the current checkout, as required by AGENTS.md. No commit,
> push, inference, download, input migration or historical-result rewrite.

**Goal:** Keep the 16-dataset analysis surface clear, archive obsolete workflows
with verified original bytes, and preserve the current calculations and plots.

**Design and scope:** The user authorized archival where possible and minimal
dependency edits where necessary. Extract only the dataset list, CSV helpers and
SHA-256 helper used by paired analysis; do not redesign stages, formulas,
sampling, figures, source formats or output locations. Keep the config-driven,
paired-W3, paired t-SNE, centered-FP and schema-2 entrypoints. Schema-2 remains a
single-pair adapter, not a claimed 16-dataset orchestrator.

**Architecture:** `five_dataset_features/common.py` replaces active imports of
the legacy analyzer. Thirteen legacy files and the original conflicted README
are preserved under `archive/five_dataset_features_legacy_20261101/`, with a
SHA-256 relocation manifest and restoration instructions. The archive is source
preservation, not an executable relocated package. Historical data, results and
manifests remain at their original paths.

**Constraints:** Explicit opencode_env Python executable; sandbox execution;
functional checks and baseline copies outside the repository; no new worktree
for this layout cleanup. Metrics and all calculation/plot function bodies stay
unchanged. Existing provenance checks must remain strict; a source change must
not be hidden by modifying historical manifests.

## Tasks

- [x] Capture source/artifact baselines and add temporary regression checks.
      Observe the missing common-module/remaining legacy-dependency check fail.
- [x] Add common helpers, redirect paired/t-SNE imports and update paired code
      dependency hashing. Check 16 datasets and unchanged metric/plot behavior.
- [x] Copy and hash-verify the 13 legacy sources and original README; record
      original/archive paths, bytes, hashes and retained dependencies. Only then
      remove the verified original legacy sources from the active module.
- [x] Rewrite module README and root feature index with current entrypoints,
      coverage, data limitations, provenance consequences and archive links.
- [x] Verify all active imports/CLI help, launcher syntax and 16-dataset scope;
      compare old/new report and PCA plot on identical inputs, rehash historical
      artifacts, obtain an independent review, and remove this task's temporary
      files. Record commands, evidence and remaining limitations here.

## Validation and execution record

Execution date: 2026-11-01 (Asia/Taipei). Starting commit:
`a4a0136b34336e4c99447f93252965fdaf9bbd3c`; starting checkout was clean on
`codex/logit_analysis`. Work remains uncommitted in that checkout.

### Preservation and implementation

- Copied and SHA-256 verified 13 legacy sources plus the original conflicted
  README before removing the legacy originals. Paths, sizes, hashes and retained
  metrics dependency are recorded in
  `archive/five_dataset_features_legacy_20261101/manifest.json`.
- Scoped archive `.gitattributes` uses `* -text`: `core.autocrlf=true` must not
  change original source bytes on future checkouts. `git check-attr text`
  confirms this for archived Python, PowerShell and original README files.
- `common.py` copies the original dataset tuple and CSV/hash implementations.
  Paired/t-SNE imports redirect to it. Paired `code_digest` now includes this
  actual dependency instead of the removed analyzer. No backward hash bypass
  was added; historical manifests remain unchanged.
- All functions in `tsne_paired.py` are AST-identical. All functions in
  `paired.py` except `code_digest` are AST-identical. Thirteen retained core
  files, including `metrics.py`, config-driven stages, centered distances and
  schema-2 reader, are byte-identical to the captured baseline.
- Existing data/analysis trees contain 651 files; every size and SHA-256 was
  compared with a pre-edit baseline and remains unchanged.
- Module Python surface shrank from 28 to 17 files. The archive contains the 12
  removed Python files plus the old PowerShell launcher; it is not presented as
  an executable relocated package. The original README is a separate snapshot.

### Commands and observed results

Python commands used the explicit sandbox executable
`C:\Users\Public\miniconda3\envs\opencode_env\python.exe -B`.
The temporary regression suite was stored outside the repository, under this
chat's writable visualization scratch directory. `MPLCONFIGDIR` was also placed
there to avoid writing to the host font cache.

1. Before implementation, the common-module and legacy-dependency checks both
   failed for their expected reasons (missing common module; paired imported
   the legacy analyzer). After extraction, a function-AST check caught an
   accidental `Path.open` substitution; the helper was restored to the exact
   original `open(path, ...)` implementation.
2. `python -B <task-temp>/check_cleanup.py`: **8 checks passed**:
   independent active imports; archive relocation; unchanged calculations/plots;
   shared helper CSV/hash parity; all 651 historical artifacts unchanged;
   byte-identical PCA PNG and unchanged sampling on identical temporary inputs;
   byte-identical report built from the real 144 saved pair summaries; complete
   16-dataset constants/config/launcher selection. Temporary synthetic inputs
   are regression evidence only, not a new experiment.
3. `python -B -m five_dataset_features.<entrypoint> --help`: **5/5 passed** for
   `run`, `paired`, `tsne_paired`, `centered_class_distance`, `unified_inputs`.
4. Parsed and imported **17/17 active Python modules**. Compared all three shell
   launchers with Git HEAD (normalizing newline encoding only): **unchanged**.
   Verified all required SBATCH directives are populated before execution,
   partition is `8gpus`, account is `MST114495`, and output/error include `%j`.
5. Verified local links/anchors in both current READMEs, centered-distance docs
   and archive README; no conflict markers remain in current documentation.
   `git diff --check` passed.
6. A fresh read-only reviewer independently verified all 14 archive entries,
   Git byte preservation, absence of active imports of removed modules, scope
   and provenance documentation. **No actionable findings**.

### Limits and finalization

Git Bash `-n` was attempted inside the sandbox and blocked before parsing by
MSYS `NtCreateDirectoryObject ... 0xC0000022`. No Bash syntax-pass claim is made.
The launchers are unchanged, with their headers statically verified; no
unsandboxed execution was needed for this source cleanup.

Full raw inference arrays are absent locally. This is not a new nano4
experiment, complete raw-data validation or data/result layout migration.
Schema-2 remains a single-pair reader, not a complete 16-dataset orchestrator.
Old paired report/package stage receipts correctly require reanalysis after
the source identity change; use a fresh output root when doing that analysis.

Final regression rerun: **8/8 passed**. Task-created baseline copies, artifact
hash inventory, test script, fixture plots/reports and Matplotlib cache were
removed from the external scratch directory after the passing run. The initial
per-command sandbox-temp copy no longer existed at cleanup. No pre-existing
files or historical artifacts were deleted. Final `git diff --check` passed.
