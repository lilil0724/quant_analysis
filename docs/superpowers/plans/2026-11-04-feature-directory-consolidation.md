# Feature directory consolidation implementation plan

**Goal:** Complete the user's requested move from `five_dataset_features/` to
`dataset_features/`, including operational paths and indexes.

**Architecture:** Preserve every existing input, result and manifest byte;
merge the old module's contents into the current module only after checking
that destinations do not exist. Record every old/new file path, size and
SHA-256 in a tracked relocation receipt. Keep historical `analysis/` and
`results/` layouts: this is a module-name consolidation, not a new analysis run
or completion of the full `result/<run_id>/reports` migration.

**Spec:** User request in this chat; repository `AGENTS.md` and
`docs/analysis_conventions_proposal.md`.

## Constraints and decisions

- Use the existing main checkout, retaining the current branch and changes.
- Do not download, run inference, overwrite outputs, commit, or push.
- Preserve archived source snapshots and historical provenance paths.
- Keep new generated data/results ignored; retain existing tracked history.
- Use `conda run -n opencode_env python` for local Python checks.
- Execute inline; temporary migration/validation scripts live outside the repo.

## Tasks

- [x] Inventory both trees, reject conflicting destinations, hash all source
  files and record existing destination artifacts before moving.
- [x] Move the old tree into `dataset_features/`; verify all bytes and the
  pre-existing destination artifacts. Preserve empty directories and caches.
- [x] Update live CLI/launcher/config paths, root and module documentation,
  ignore rules, and dependent TGDA references. Keep historical snapshots intact.
- [x] Verify Python CLI help, Bash syntax/dry-runs and Slurm headers, real
  compact data coverage/formulas, archive hashes, links and remaining old paths.
- [x] Record evidence and remaining raw-array/layout limitations in the receipt
  and module README; inspect the final diff and obtain a focused review.

## Progress

- Initial checkout clean on `codex/activation_quant`; 1,418 source files,
  218,255,389 bytes; no destination file collisions or source reparse points.
- Existing `dataset_features/analysis/` contains 70 tracked historical artifacts;
  preserve these bytes and their existing Git tracking.
- Complete: all 1,418 relocated files and 70 pre-existing artifacts unchanged;
  144 real pairs, 4,896 layer rows, 579 output hashes and 14 archive hashes checked.
- Five CLI help checks, three Bash syntax checks, two launcher dry-runs and the
  centered-distance CLI dry-run passed. No result directory was created by the
  centered preview. Slurm directives and 51 live document links/anchors checked.
- Focused independent review found no actionable issues. Navigation-only TGDA
  help/docstring references were also updated; dated status logs stay historical.
- Git Bash and conda required sandbox escalation for their local execution;
  no server computation was performed. Full raw-array and result/report-layout
  migration remains pending, as documented in the relocation README.
