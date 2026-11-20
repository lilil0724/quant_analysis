---
status: accepted
---

# Screen dataset–model quantization difficulty before broadening experiments

For serial 100004, define exploratory difficulty from the mean within-condition percentile rank of relative accuracy loss across 3/4-bit HQQ at group sizes 8 and 128. Use normalized-margin Q10 as the sole primary logit predictor, baseline accuracy and log class count as controls, and require both a prespecified promising gate and `Q10 → swap rate → relative loss` mechanism consistency; treat 2/8-bit as boundary sensitivities and exclude saturated 1/1.58-bit results. This favors a stable, scale-resistant screening signal over averaging incomparable condition severities or selecting whichever post-hoc logit statistic looks strongest.

## Consequences

The current result can justify testing additional seeds, models, and settings, but cannot be called an intrinsic dataset effect or an independent margin effect unless it generalizes and improves held-out prediction beyond the baseline controls.
