#!/usr/bin/env bash
# Rebuild results/, the paper's tables and figures, both PDFs and the checks from the committed data/.
# Run from the repo root on server-3090: bash plotter.sh (offline) or bash plotter.sh --download.
set -euo pipefail
cd "$(dirname "$0")"
source /mnt/ssd2tb/miniconda3/etc/profile.d/conda.sh && conda activate quant_analysis
export PATH=/mnt/hdd10tb/edwin/texlive/2026/bin/x86_64-linux:$PATH
export MPLBACKEND=Agg

# --download re-copies data/ from the run bundles under legacy/ (they are not committed)
if [ "${1:-}" = "--download" ]; then
    python -m codebase.download.snapshot
fi

mkdir -p results paper/tables paper/figures
python -m codebase.analyses.compute
python -m codebase.build.tables
python -m codebase.plots.figures

# the 8-page paper (main) and the 4-page workshop paper (short); LaTeX output goes to paper/*.log;
# from a clean checkout the reference back-links need the fourth pass
cd paper
for doc in main short; do
    pdflatex -interaction=nonstopmode -halt-on-error $doc.tex > /dev/null
    bibtex $doc > /dev/null
    for pass in 1 2 3; do
        pdflatex -interaction=nonstopmode -halt-on-error $doc.tex > /dev/null
    done
done
cd ..

python -m codebase.verify.check
