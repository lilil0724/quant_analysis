---
status: accepted
---

# Test the margin mechanism within dataset–model pairs

For serial 100004, test whether lower baseline normalized decision margin
discriminates samples whose top-1 prediction swaps under each of the four
primary 3/4-bit HQQ conditions. Use all matched samples as the primary
population and baseline-correct samples as a sensitivity population. Average
the four identifiable condition AUROCs within each dataset, give every dataset
equal weight, and call the mechanism supported only when the median dataset
score is at least 0.60, a one-sided exact sign test against 0.50 has p below
0.05, and every condition-specific cross-dataset median AUROC exceeds 0.50.

Require all four primary AUROCs to identify a dataset score. If any primary
condition has only one swap outcome class, report the primary gate as
INCOMPLETE rather than averaging fewer conditions or calling the mechanism
unsupported. Construct normalized-margin deciles within each dataset without
splitting tied margins, and summarize decile swap rates with dataset-equal
aggregation.

Implement this as a standalone analysis stage that reads the existing prepared
pairs and baseline artifact manifest. Compute per-sample normalized margins
from each baseline artifact once and align them to pairs by sample ID; do not
change or rewrite the prepared-pair artifact contract.

## Consequences

The analysis directly tests the proposed decision-boundary mechanism within
datasets and avoids treating images, conditions, or large datasets as
independent replication. It requires local access to baseline logits and may
return an incomplete result when a condition contains no outcome variation.
The test remains associative and does not establish a causal margin effect.
