# Reproducing the paper

On server-3090, from the repo root:

    bash plotter.sh

The command needs the `quant_analysis` conda env and TeX Live 2026 at
`/mnt/hdd10tb/edwin/texlive/2026`. It runs offline from the committed `data/` and takes about a minute.

Expected output, last line:

    checked 120 macros and the cells of 10 tables; placeholders not checked: 6 TBD notes, synthetic figures: 1

- `paper/short.pdf`: 7 pages (4 body pages, references, appendix), no overfull boxes in `paper/short.log`.
- `paper/main.pdf`: no overfull boxes in `paper/main.log`.
- `git status` shows no change to `results/`, `paper/tables/` or `paper/figures/`.

The "6 TBD notes" and the synthetic figure are placeholders in the 8-page version (`paper/sec/`);
`docs/NEXT_STEPS.md` lists what each needs.

`bash plotter.sh --download` first re-copies `data/` from the run bundles in
`legacy/paper_draft/data/` and `legacy/quant_acc/20261117_legacy_hqq_s10000_10016_cal10016_v2/`,
which are not committed. `data/MANIFEST.json` records the source and SHA-256 of each copied file.
