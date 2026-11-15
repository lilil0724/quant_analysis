# Activation plots implementation plan

> **For agentic workers:** Use superpowers:executing-plans for this bounded plotting task.

**Goal:** Draw the two plot types selected by the user from the preceding proposal.
**Architecture:** Add plot.py to the owning activation_quantization module and call
it in new serial-20008 runs. Reuse the verified immutable input and existing run
receipt/report conventions. Historical runs remain unchanged.
**Tech Stack:** Existing opencode_env, pandas, numpy, matplotlib.
**Spec:** User selected plots 1 and 2: accuracy-versus-activation curves and
dataset heatmaps of loss relative to the same checkpoint/weight at A6.

## Constraints and design

- Line overview: model rows × FT/FZ/CAL columns, three W colors; each dataset
  is a faint line, the equal-dataset mean a bold line, shared 0–100% y axis.
- Three heatmaps, one per W: model rows × modes columns; each panel contains
  the same ordered 16 datasets × seven A settings. Loss = Acc(W,A6)-Acc(W,A).
- Preserve negative loss, use one symmetric zero-centered scale for all heatmaps,
  and explicitly distinguish the A6 reference from an unquantized baseline.
- Save PNG/SVG plus a four-page PDF atlas, plot source tables and validation in
  result/new_run; reference the four PNGs in the sole report and verify hashes.
- No network, inference, environment install, historical-output edits, or permanent
  synthetic tests. Work in the main checkout per AGENTS.md.

## Steps

1. [x] Check actual-data formula expectations before adding the plotting API.
2. [x] Add plotting/data validation and integrate new figures into run/report/verify.
3. [x] Execute a new actual-data run and inspect all four rendered figures.
4. [x] Verify formulas/coverage/source hashes, obtain a focused independent review,
   update README/index and remove task-created temporary files.

## Review focus

Correct A6 pairing within checkpoint and W; equal dataset counts in every mean;
negative values and consistent color scales; readable axes/legends; report image
hashes and unchanged historical runs.

## Evidence

- Plotting API availability check failed before implementation; actual-data formula
  checks then passed for 3,024 pairs, 432 zero A6 references and 189 means of 16 datasets.
- Actual run: `20261102_142515_734861_wa_s20008`; four PNGs, four editable SVGs,
  and a PDF independently counted as four pages with pypdf.
- Viewed all four PNGs; axes, legends, annotations, dataset order and captions readable.
- Loss range -12.98 to 99.27 pp; 127 negative cells preserved. Heatmap scale fixed
  at [-100,100] pp, symmetric around zero, across all 27 heatmap panels.
- New run and prior completed run pass read-only verification. Original six CSVs
  byte-identical; 49 pre-existing data/result/report files have unchanged hashes.
- Invalid actual-data copies with a missing point, duplicate point, or changed
  checkpoint were rejected. Report images equal result images by SHA-256.
- Independent code review: no material findings. `git diff --check` passed.
- Task-created availability-check script, hash inventory and matplotlib cache removed.
  No downloads, server inference, commits, or permanent test fixtures.
