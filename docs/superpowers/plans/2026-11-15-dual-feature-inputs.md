# Dual feature-analysis inputs implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan inline, with temporary out-of-repository parity fixtures and a final review.

**Goal:** Let existing paired features, t-SNE, centered-distance, bias/competitor and margin-matching analyses consume legacy paired exports or schema-2 exports without changing their metrics, sampling or plots.

**Architecture:** A read-only input adapter resolves condition directories, validates provenance/sample identity, and supplies legacy-shaped in-memory metadata and prediction rows. Existing legacy validators remain in use for old exports. Each owning reader branches to the adapter for schema 2; original mathematical and plotting functions remain unchanged. Explicit source selection preserves actual serials in provenance.

**Tech stack:** Existing Python/NumPy/pandas analysis environment. No new dependency or inference.

**Spec:** User-approved conversation: both input formats, input-only changes, all original analysis methods and plots retained; begin implementation.

## Global constraints

- Work in the existing checkout and preserve prior changes, historical inputs and results.
- Use the explicit opencode_env Python executable. Tests/fixtures are temporary outside the repository and must be cleaned.
- Never infer missing historical hashes or represent backbone features as CAL classifier features.
- Keep W3/G128 scope of the existing analyses; accepting arbitrary quantization grids is outside this input-only change.
- Missing selected contexts, required arrays, ambiguous conditions, unsafe paths and identity mismatches fail explicitly.
- Nano4 scripts use partition 8gpus, account MST114495 and complete resource/log headers.

## Review focus

- Source hash, sample order/labels and checkpoint/execution mismatches must reject.
- New CSV margins must be derived with the original float64 formulas; ties and zero-spread behavior remain owning-stage policy.
- Final-only readers must not require unused layer arrays; logit-only readers must not require features.
- Input adapter changes must participate in stage code/provenance identities and copied snapshot validation.
- Identical old/new payloads must yield equal metrics, sampling, plotting inputs and rendered plots.

## Tasks

- [x] 1. Record original function hashes and create failing parity/invalid-input checks outside the repository.
- [x] 2. Implement a schema-2 paired adapter and safe file/condition resolution, retaining original manifests.
- [x] 3. Connect paired, t-SNE and centered-distance readers, preserving old validation and all formulas/plots.
- [x] 4. Connect bias/competitor and margin snapshot/import readers and provenance source copies.
- [x] 5. Add explicit source-map selection where needed; correct stale five_dataset_features launcher paths.
- [x] 6. Run temporary end-to-end parity and corruption/identity/output-protection checks; validate real local metadata where available.
- [x] 7. Update owning README and root index, record verification/limitations, review the diff, clean only task-created temporary files.

## Execution ledger

- 2026-11-15: Approved scope is input compatibility only. Existing W3/G128 analysis contracts remain fixed; no new analysis/plot design. Raw server arrays are not locally available, so fixture parity and real compact-source audits cannot establish Nano4 completion.
- Ruling: Preserve original within-model/mode export/protocol equality checks. An explicit source map can select old FT/FZ and new CAL, but does not assert that unrecorded old/new quantization settings are equivalent within one setting.
- Ruling: Preserve each stage's original file needs. Schema-2 final-feature readers do not hash unused arrays; logit-only snapshot imports retain original manifests without copying unused features.
- Ruling: Source-map serial identifies the actual selected export; the CLI serial labels the analysis cohort. Record both in provenance. No implicit latest-serial discovery.

## Verification evidence

- 24 temporary checks passed using `C:\Users\Public\miniconda3\envs\opencode_env\python.exe -B -m pytest ... -p no:cacheprovider`; final run: 24 passed in 10.57 s. Test sources/fixtures were outside the repository and cleaned after verification.
- RED checks first reproduced unsupported schema-2 inputs, unsupported mixed-serial mapping, and call-count-sensitive protocol comparison; each was followed by passing checks after implementation.
- Old/new payload parity checked paired summaries, layer CSVs, PCA coordinates and rendered PNG bytes; t-SNE coordinates and PNG bytes; centered-distance CSVs and FP means; bias computations; and margin snapshot import, tables and rendered PNG bytes.
- Rejection checks covered source corruption, missing required logits, reordered samples, mismatched checkpoint/execution, activation quantization, classifier coverage, ambiguous W3 conditions, unsafe feature paths, malformed source-map contexts, and changed source-root resume identity. Partial feature/logit capabilities were checked separately.
- `metrics.py`, `margin_matching.py`, `bias_competitor_core.py` remained byte-identical to the pre-task copies. AST comparisons confirmed original paired plot/report, t-SNE sampling/embed/plot, and centered-distance mathematical functions unchanged. Margin plotting/report functions remain unchanged in the diff.
- Five CLI help/import checks passed. Five Slurm header/syntax checks and four launcher dry-runs passed; no Slurm submission. Git Bash required narrowly scoped execution outside the sandbox after reproducible `0xC0000022` sandbox startup failure; Python analysis checks ran in the sandbox.
- Read-only audit found the existing real serial-20004 compact manifest still covers 144 contexts. Full raw features/logits are absent locally, so real-data numeric recomputation and Nano4 full-matrix execution remain unverified.
- Independent review identified stale t-SNE source-root provenance on resume. A failing relocation regression reproduced it; resume now checks resolved source folder, actual serial and source-map receipt. Targeted reviewer recheck reported it resolved, with no remaining findings.
