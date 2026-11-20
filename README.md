# quant_analysis

Analysis repo for the WACV 2027 paper on dataset-dependent accuracy retention under weight-only
post-training quantization. Three backbones (ViT-B/16, Swin-B, BEiTv2-B), each trained with three
strategies (FT, FZ, CAL) on 16 fine-grained datasets, are quantized with HQQ at 8 bit widths and 4
group sizes. The paper asks whether the dataset affects quantization robustness, and how: five
ultra-fine-grained cultivar datasets (UFG) lose more than the eleven fine-grained datasets (FG), and
the baseline logit margin accounts for part of that gap. This repo trains nothing. It turns the W&B
runs and the nano4 feature exports into the paper's tables, figures and numbers.

## Status

The 4-page workshop version (`paper/short.pdf`) is the current deliverable; its versions are in
`drafts/`. The open items are in `docs/NEXT_STEPS.md`.

## Running it

    bash plotter.sh

from the repo root. It activates the `quant_analysis` conda env, computes every result from the
committed `data/`, writes the paper's tables and figures, builds `paper/main.pdf` and
`paper/short.pdf` with pdflatex, and runs the checker. `bash plotter.sh --download` first re-copies
`data/` from the run bundles under `legacy/`, which exist only on server-3090. `REPRODUCE.md` gives
the expected output.

## Layout

| Path | Holds |
|---|---|
| `plotter.sh` | the driver: one command rebuilds every CSV, table, figure, both PDFs and the checks |
| `requirements.txt` | the pinned Python dependencies, matching the `quant_analysis` conda env |
| `codebase/` | the pipeline: `registry.py` (datasets, groups, settings, conditions in print order), `download/`, `analyses/`, `build/`, `plots/`, `verify/` |
| `data/` | the committed inputs: the weight-only grid and the W3/G128 export tables; `data/README.md` lists each |
| `results/` | the CSVs `plotter.sh` writes; `results/numbers.csv` holds every printed number |
| `paper/` | `short.tex` (4-page) and `main.tex` (8-page) with `sec_short/` and `sec/`, `preamble.tex`, `main.bib`, the WACV 2027 style files, and the generated `tables/` and `figures/` |
| `drafts/` | one PDF per version of the paper; `drafts/README.md` lists them |
| `docs/` | `FINDINGS.md` (what each `plotter.sh` step shows), `SERIALS.md` (every serial), `NEXT_STEPS.md` (open items), `CLAUDE_CHANGES.md` (change log) |
| `legacy/` | the earlier analysis packages, their docs and the paper's working notes; `legacy/README.md` lists each |
| `CLAUDE.md` | the repo conventions for agents |
| `REPRODUCE.md` | the expected output of `plotter.sh` |
