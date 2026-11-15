# Baseline accuracy-ratio plots

**Goal:** User requests a two-group line plot and all-dataset heatmaps using
baseline accuracy ratios. Preserve soy+cotton UFGIR (5), other FGIR (11).

**Definition:** ratio = quantized top1 / uniquely matched unquantized AMP top1.
The preceding group plot already uses this denominator; redraw without changing
its 24-shared-stratum aggregation. This is not an A6 normalization.

**Outputs:** Two-line group curve, one 16×7 dataset mean heatmap using valid
ratios within the same shared strata (show each dataset's context count), and
three W-specific model×mode heatmap atlases preserving missing/ambiguous N/A.
All ratio heatmaps share a scale centered on 1; preserve values above 1.
The summary heatmap's dataset means have different weights from the group
curves' equal-stratum means; document this explicitly. Save PNG/SVG and PDF,
source CSVs, formula/coverage checks and the sole report in a new immutable run.

**Steps:**
- [x] Add baseline-ratio views and rendering in the owning module; integrate
  generation/report/read-only verification for grouped inputs.
- [x] Run the existing verified input with a new ID; inspect all new plots.
- [x] Verify ratios, summary counts, shared strata, N/A, report hashes, prior
  artifacts; independent focused review; README/index and temporary cleanup.

**Evidence:**

- API availability check failed before implementation. Actual data:3,024 cells,
  2,289 matched ratios,735 N/A,51 above1,max1.568493;112 dataset means verified
  independently against quantized/baseline Top-1 values, with n9or24 per row.
- CSV roundtrip initially failed on empty baseline_run_id versus NaN. Reproduced
  with the actual dataframe, normalized only the identifier column, and verified
  passing afterward. Initial run162307 remains marked failed.
- Final run `20261102_162715_226338_wa_s20008` passes read-only verification.
  The previous group curve PNG and all previous CSV tables are byte-identical;
  131 pre-existing input/result/report files retain original hashes.
- Visually inspected the dataset summary and three detailed W heatmaps; adjusted
  footer anchoring to prevent overlap with the X label and generated a fresh run.
  Four-page baseline heatmap PDF confirmed with pypdf. All report image hashes match.
- Independent review: no remaining material findings after the identifier fix.
  Color scale0/1/1.6 retains >1 and separates gray missing cells.
- README/index updated; git diff --check passed. Task temporary check script,
  hash inventory and matplotlib cache removed. No network or inference.
