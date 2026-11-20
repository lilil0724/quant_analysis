# Change log

## 2026-11-20 — repo restructure

- Moved the six analysis packages (`activation_quantization`, `quant_acc`, `dataset_confidence`,
  `dataset_features`, `dataset_ranking_robustness`, `dataset_static_feature`), `archive/`, the old
  `docs/`, `AGENTS.md`, `CONTEXT.md`, the old README, requirements and CLAUDE.md, and the paper's
  working notes (`paper_draft/`) into `legacy/`. Tag `pre-restructure-20261120` marks the previous
  layout; `legacy/RELOCATION.json` lists every untracked file (path, size, SHA-256) before the move.
- The nano4 launchers are now submitted from `legacy/`; their default input root
  (`../../data/w3_g128_analysis` from the repo root) is unchanged on disk.
- Lifted the paper pipeline from `paper_draft/paper/` to `codebase/`, `data/`, `results/`, `paper/`,
  `drafts/`; `plotter.sh` replaces `build.sh` and adds a fourth LaTeX pass so the reference
  back-links resolve from a clean checkout. The output equals v5.

## 2026-11-19 — paper folder rebuilt

`paper_draft/paper/` was deleted outside the session and rebuilt by replaying the session's commands
from its log; the rebuilt v5 matched the earlier build and the checker. First commit of the paper
(26d1d6e).

## 2026-11-16 to 2026-11-19 — paper versions

v1 (8-page and 4-page drafts on the 141-context grid), v2 (new 144-context grid, structure from the
author's guideline), v3–v5 (three review rounds). Sources of v1–v4 are in `legacy/paper_versions/`.
