"""One numeric contract for fixed-condition tables and their plotted points."""

from typing import Any

import numpy as np

from .inputs import Context


def nullable_number(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    return np.nan if value in (None, "") else float(value)


def fixed_condition_values(context: Context, condition: str, baseline: str,
                           layers: list[dict[str, Any]],
                           summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Read final-position FP/Q values; undefined numbers stay NaN for pairwise filtering."""
    final = len(context.position_names) - 1
    layer = next(row for row in layers if row["condition"] == condition and
                 int(row["position_id"]) == final)
    fp = next(row for row in layers if row["condition"] == baseline and
              int(row["position_id"]) == final)
    summary = summaries[condition]
    fp_accuracy = nullable_number(summary, "fp_accuracy")
    return {"dataset": context.dataset, "cka": nullable_number(layer, "cka"),
            "accuracy_ratio": (nullable_number(summary, "accuracy") / fp_accuracy
                               if fp_accuracy > 0 else np.nan),
            "accuracy_drop": nullable_number(summary, "accuracy_drop"),
            "difficulty": 1 - fp_accuracy,
            "gap_deterioration": nullable_number(layer, "gap_deterioration"),
            "inter_distance_change": nullable_number(layer, "delta_inter"),
            "intra_distance_change": nullable_number(layer, "delta_intra"),
            "fp_gap": nullable_number(fp, "separability")}
