# WACV 2027 drafts: dataset-dependent accuracy retention under weight-only PTQ

`main.pdf` is the 8-page paper, built from `main.tex` and `sec/`. `short.pdf` is the 4-page workshop paper, built from `short.tex` and `sec_short/`, with the cut material in its appendix. Both use the generated `tables/` and `figures/`, `main.bib` and the WACV 2027 author kit (`wacv.sty`, `ieeenat_fullname.bst`).

## Build

```bash
bash paper_draft/paper/build.sh
```

The script runs in the `quant_analysis` conda env and TeX Live 2026. It computes every number from `inputs/`, writes `tables/` and `figures/`, builds the PDF with pdflatex and bibtex, and runs the checker. The checker re-derives every printed number and table cell from `inputs/` with separate code and exits non-zero on any difference.

`inputs/` holds small copies of the tables the paper reads, listed with their sources and SHA-256 in `inputs/MANIFEST.json`. `codebase/snapshot.py` refreshes them where the run bundles in `paper_draft/data/` and `quant_acc/legacy/` exist.

## Placeholders

Placeholders print in red: a dagger marks a dummy number and `[TBD: ...]` marks missing text. Figure 6 is drawn from synthetic values and carries a PLACEHOLDER watermark. Set `\showplaceholdersfalse` in `preamble.tex` to hide the markers. `TBD_zh.md` lists every placeholder and what it needs.

## Files

| Path | Content |
|---|---|
| `codebase/registry.py` | Datasets, groups, models, regimes and conditions in print order |
| `codebase/compute.py` | All analyses; writes `build/*.csv` and `build/numbers.csv` |
| `codebase/tables.py` | `tables/macros.tex` and the LaTeX tables |
| `codebase/figures.py` | `figures/*.pdf` |
| `codebase/check.py` | Independent re-derivation of every printed value |
| `EVIDENCE.md` | Each result statement with its macro and source column |
| `TBD_zh.md` | Placeholders and decisions left to the author |
