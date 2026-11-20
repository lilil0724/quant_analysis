# CLAUDE.md

Analysis repo for the WACV 2027 paper on dataset-dependent accuracy retention under weight-only HQQ.
The layout follows AAAAnalysis and TGDA_Analysis; `README.md` has the layout table.

## Running

- `bash plotter.sh` rebuilds everything from the committed `data/` in the `quant_analysis` conda env
  (Python 3.11; numpy, pandas, scipy, matplotlib pinned in `requirements.txt`) and TeX Live 2026.
- Pipeline modules run from the repo root as `python -m codebase.<stage>.<module>`.
- `bash plotter.sh --download` re-copies `data/` from the untracked run bundles in
  `legacy/paper_draft/data/` and `legacy/quant_acc/20261117_..._v2/`; they exist only on server-3090.

## Paper numbers

- Every printed number is a macro `\V{key}` from `results/numbers.csv` via `paper/tables/macros.tex`;
  never type a number into `paper/`.
- `codebase/verify/check.py` re-derives every macro and table cell from `data/` with its own code. It
  must not import from `codebase/`. A new number needs a matching derivation there.
- `paper/tables/` and `paper/figures/` are generated; edit their generators in `codebase/`.
- `paper/short.tex` (4-page, `sec_short/`) is the deliverable; `paper/main.tex` (8-page, `sec/`) shares
  the preamble, bibliography, tables and figures.
- A new paper version: copy `paper/short.pdf` to `drafts/short_vN.pdf` and add a row to `drafts/README.md`.

## Legacy

- `legacy/` holds the earlier analysis packages, their docs (`legacy/AGENTS.md`, `legacy/CONTEXT.md`,
  `legacy/docs/`) and the paper's working notes. `legacy/README.md` lists each item.
- The packages run with `legacy/` as the working directory: `cd legacy && python -m quant_acc.run ...`.
  Their nano4 launchers (`legacy/dataset_*/run_*_nano4.sh`) are submitted from `legacy/`.
- `legacy/RELOCATION.json` and tag `pre-restructure-20261120` record the layout before the move.
