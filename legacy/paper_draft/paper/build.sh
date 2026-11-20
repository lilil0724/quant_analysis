#!/usr/bin/env bash
# Rebuild numbers, tables, figures, the PDF and the checks from inputs/; run from any directory on server-3090.
set -eu
cd "$(dirname "$0")"
source /mnt/ssd2tb/miniconda3/etc/profile.d/conda.sh && conda activate quant_analysis
export PATH=/mnt/hdd10tb/edwin/texlive/2026/bin/x86_64-linux:$PATH
python codebase/compute.py
python codebase/tables.py
python codebase/figures.py
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build/latex1.log
bibtex main > build/bibtex.log
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build/latex2.log
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build/latex3.log
pdflatex -interaction=nonstopmode -halt-on-error short.tex > build/short1.log
bibtex short > build/short_bibtex.log
pdflatex -interaction=nonstopmode -halt-on-error short.tex > build/short2.log
pdflatex -interaction=nonstopmode -halt-on-error short.tex > build/short3.log
python codebase/check.py
