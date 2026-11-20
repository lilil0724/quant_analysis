# FGIR/UFGIR activation ratio curves

**Goal:** The user's two-line plot: X activation bits, Y mean accuracy_ratio,
one FGIR and one UFGIR line, using the user's updated soy+cotton grouping.

**Design:** User explicitly revised the groups: UFGIR = soyageing, soygene,
soyglobal, soylocal, cotton (5 datasets); FGIR = the other 11 datasets.
Preserve this explicit mapping and user-instruction provenance in a new input
snapshot beside unchanged W&B data. It supersedes the earlier 8/4 core grouping
for this task only; the granularity_robustness module is unchanged.

Use only uniquely matched unquantized AMP ratios. Within each model/mode/W,
average datasets once per group. Retain common strata with at least two datasets
per group and the same member sets across all A values; average those strata
equally at each A. This follows the existing granularity matched-stratum contract
while using means as the user requested. Keep missing/ambiguous cells as N/A.
Record member sets/counts and exclusions; do not infer a causal granularity effect.

**Steps:**
- [x] Verify the grouping source and actual baseline coverage.
- [x] Add grouping aggregation/rendering to activation_quantization for inputs
  containing the explicit grouping CSV; preserve historical-run verification.
- [x] Run a new immutable snapshot/run, inspect the two-line figure and verify
  raw ratios, means, shared strata, hashes and unchanged historical artifacts.
- [x] Independent focused review, README/index updates and temporary cleanup.

**Evidence:**

- Plot API availability check failed before implementation; actual-data checks passed afterward.
- Input `wandb_s20008_soy_cotton_20261102_153946_160335` preserves unchanged
  W&B files, explicit 5/11 mapping, user-instruction provenance and predecessor manifests.
- Completed run `20261102_153951_152893_wa_s20008`; viewed two-line PNG and
  confirmed its PDF has one page. SVG is editable; report image hash equals result image.
- Independently recomputed all 14 means from original target/baseline summaries,
  covering 2,163 included ratio points, 24 shared strata. FGIR:189 cells/A;
  UFGIR:120 cells/A. Independent reviewer agreed to floating-point precision.
- ViT FT's three W strata excluded because UFGIR has only one matched dataset.
  Dataset membership stays fixed over all A; incorrect cotton grouping and
  changing membership were rejected in checks on in-memory actual-data copies.
- Current and prior runs pass read-only verification; all 82 prior data/result/report
  files unchanged by hash. Prior eight CSV tables remain byte-identical.
- Independent code review: no material findings; git diff --check passed.
- Temporary source staging, check script, hash inventory and matplotlib cache removed.
  No downloads, inference, commits or permanent test fixtures.
