# Next steps

Open items only. The 4-page version (`paper/short.tex`) is the deliverable; items marked "8-page"
affect only `paper/main.tex`.

## Missing text and data

1. **Moe citation.** Moe (anime characters) has no source citation (`\TBD` in both versions).
2. **Quantized layers per backbone.** The setup should state how many linear layers HQQ quantizes in
   each backbone. Needed from the HQQ run configuration.
3. **8-page: per-image W3/G128 predictions.** The swap-by-margin-decile figure is a placeholder drawn
   from synthetic values. It needs each test image's baseline and quantized prediction, which exist
   only in the nano4 exports. Options: fetch them and draw the figure, or drop the figure.
4. **8-page: text not revised since v2.** It still says "linear probing (LP)", carries the TBDs for
   the backbone-scaling citation and the HQQ version, and has not had the v3–v5 changes. Recommendation:
   revise it only if the 8-page paper is submitted.

## Unverified references

5. `paper/main.bib`: LampQ (`kim2026lampq`) lists only its first author ("and others"); check it and
   Fang et al. (TCSVT 2024) against their published versions.

## Decisions for the author

6. **Submission track.** Set to the Algorithms Track; the analysis could also fit the Applications Track.
7. **Title.** "Dataset-Dependent Accuracy Retention under Weight-Only Post-Training Quantization";
   two alternatives are in comments in `paper/main.tex`.
8. **Strip plot of the difficulty score (optional).** One column per setting, UFG and FG colored,
   would show the stable group separation and the unstable FG order. Recommendation: add it to the
   appendix only if a reviewer questions the ranking claim.
9. **Table 3 medians.** Each column of the W3/G128 group table is a separate median over settings,
   so FP − Q need not equal Damage − Rescue in a row. Options: add "medians per column" to the
   caption, or print means. Recommendation: the caption clause.
