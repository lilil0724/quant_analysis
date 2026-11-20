# Findings

One entry per `plotter.sh` step, in order. Numbers are from `results/numbers.csv`. UFG: the five
soybean and cotton cultivar datasets; FG: the other eleven. A setting is one backbone with one
training strategy (nine settings); a context is one dataset in one setting (144 contexts).

## `codebase.download.snapshot` (only with `--download`)

Copies the 14 input tables into `data/` and records their sources and SHA-256 in `data/MANIFEST.json`.

## `codebase.analyses.compute`

**Grid (`grid_analysis`).** The grid is complete: 144 contexts × 32 conditions = 4,608 cells.
Bit width and group size set the level of retention; the groups separate in between. At W3/G128 the
UFG group retains 0.49 and the FG group 0.82, a gap of 0.33. The gap is positive in 36 of 36
setting × group-size combinations at 3 bits and 34 of 36 at 4 bits. Smaller groups shift the band to
fewer bits: the gap is 0.32 at W2/G8 and 0.07 at W2/G128. Every UFG dataset keeps at most 0.55 at
W3/G128 (mean over settings) and every FG dataset at least 0.77. Baseline top-1 averages 32.5% for
UFG and 87.3% for FG.

**Robustness (`robustness_analysis`).** The W3/G128 gap stays positive in all nine settings with
chance-corrected retention (0.34), without the baselines below 15% (0.36), and without any one UFG
dataset (at least 0.15). A dataset bootstrap puts the 95% interval above zero in 8 of the 8 settings
other than BEiTv2 FT, whose lower bound is 0.002.

**Difficulty ranking (`difficulty_analysis`).** Within a setting, the four primary conditions (W3/W4
× G8/G128) agree on the ranking (Kendall's W 0.73–0.95). Across settings the median Spearman between
two settings' scores is 0.61, and among FG datasets alone 0.30. A UFG dataset is harder than an FG
dataset in at least 0.69 of the 55 UFG–FG pairs in every setting, and in all of them in 5 settings.

**Margin (`difficulty_analysis`, `accuracy_analysis`).** The prespecified margin signal (q10) holds in
9 of 9 settings, median ρ −0.87; baseline top-1 correlates almost as strongly (median ρ −0.84).
Controlling for top-1, the partial correlation of q10 stays negative in 7 of 9 settings (median
−0.35). Adding q10 to the controls lowers the leave-one-dataset-out error in only 3 of 9. Of 55
within-setting pairs with baseline top-1 within 1.5 points, 9 differ in retention by more than 0.2,
and in 6 of these the dataset with the smaller q10 retains less.

**Margin matching (`margin_analysis`).** Matching the two groups' margin distributions at W3/G128
reduces the gap by 28–81% (median 56%); the matched gap stays positive in 9 of 9 settings. The
shared histogram mass is 0.19–0.40, and the matched ESS falls to as little as 21% of the raw ESS.
A median 87% of UFG swaps (46% for FG) end on a class the baseline ranked below second.

**Decisions and features at W3/G128 (`w3_analysis`, `layer_analysis`).** A median 72.0% of UFG test
images change their prediction, against 14.4% of FG images; the UFG group swaps more in 9 of 9
settings. Only 33.1% of UFG images are correct at baseline, so at least 54% of UFG swaps move
between two wrong classes. Mid-depth CKA is 0.99 (UFG) and 0.97 (FG); final-layer CKA correlates
with the difficulty score at a median ρ of only −0.20.

**Not printed in the 4-page paper.** BEiTv2 FZ collapses at W3/G128 for every dataset (FG 0.20,
UFG 0.03); its transition is one bit higher. The class-offset correction, the competitor-count
models and the t-SNE example appear only in the 8-page version.

## `codebase.build.tables`

Writes `paper/tables/macros.tex` (every printed number as `\V{key}`) and the LaTeX tables.

## `codebase.plots.figures`

Writes `paper/figures/*.pdf`. `swap_by_decile_placeholder.pdf` is drawn from synthetic values and is
used only by the 8-page version.

## `codebase.verify.check`

Re-derives every printed number and every cell of 10 tables from `data/` with separate code and
fails on any difference, on a computed number that is never printed, and on a printed number it
cannot derive.
