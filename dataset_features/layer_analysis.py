"""Calculate per-context prediction and layer geometry tables."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .inputs import Context, open_features, predictions_for
from .metrics import (centered_linear_cka, class_equal_cosine_distances,
                      norm_quantiles, prediction_metrics)
from .run_config import RunConfig


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    _csv(path, [{key: _clean(value) for key, value in row.items()} for row in rows])


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def analyze_context(context: Context, config: RunConfig, output: Path) -> dict[str, Any]:
    """Write a complete set of accuracy and layer metrics for one context."""
    output.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(context.metadata["targets"], dtype=np.int64)
    baseline_name = config.baseline_condition
    fp = open_features(context, baseline_name)
    fp_predictions = predictions_for(context, baseline_name)
    summaries: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    fp_layer: dict[int, dict[str, Any]] = {}
    raw_layer: dict[tuple[str, int], dict[str, Any]] = {}

    conditions = [baseline_name, *[name for name in context.conditions
                                   if name != baseline_name]]
    for condition in conditions:
        features = open_features(context, condition)
        predictions = predictions_for(context, condition)
        outcome = prediction_metrics(fp_predictions, predictions)
        outcome["accuracy_ratio"] = (outcome["accuracy"] / outcome["fp_accuracy"]
                                      if outcome["fp_accuracy"] > 0 else None)
        outcome["absolute_drop_pp"] = 100 * outcome["accuracy_drop"]
        outcome["accuracy_ratio_status"] = ("complete" if outcome["fp_accuracy"] > 0 else
                                             "undefined_fp_accuracy_zero")
        summaries.append({"dataset": context.dataset, "context_id": context.context_id,
                          "condition": condition, "n_samples": len(labels),
                          **outcome})
        for index, position_name in enumerate(context.position_names):
            q_feature = features[:, index, :]
            cka = centered_linear_cka(fp[:, index, :], q_feature,
                                      chunk_size=config.chunk_size)
            geometry = class_equal_cosine_distances(q_feature, labels,
                                                    chunk_size=config.chunk_size)
            row = {
                "dataset": context.dataset, "context_id": context.context_id,
                "condition": condition, "position_id": index,
                "position_name": position_name, "n_samples": len(labels),
                "feature_width": context.feature_shape[-1],
                "cka": cka,
                **geometry,
                **norm_quantiles(q_feature, chunk_size=config.chunk_size),
            }
            row["d_inter"] = _clean(row["d_inter"])
            row["d_intra"] = _clean(row["d_intra"])
            row["separability"] = _clean(row["separability"])
            row["gap"] = row["separability"]
            _, class_counts = np.unique(labels, return_counts=True)
            zero_norm_failure = (
                (geometry["n_inter_class_pairs"] == 0 and len(class_counts) >= 2) or
                (geometry["n_intra_classes"] == 0 and np.any(class_counts >= 2)))
            cka_status = "complete" if cka is not None else "undefined_zero_centered_variance"
            inter_status = ("complete" if row["d_inter"] is not None else
                            "undefined_zero_norm_sample" if zero_norm_failure else
                            "undefined_fewer_than_two_classes")
            intra_status = ("complete" if row["d_intra"] is not None else
                            "undefined_zero_norm_sample" if zero_norm_failure else
                            "undefined_no_class_with_two_samples")
            gap_status = ("complete" if row["separability"] is not None else
                          "undefined_inter_or_intra_distance")
            row.update({"cka_status": cka_status, "d_inter_status": inter_status,
                        "d_intra_status": intra_status, "gap_status": gap_status})
            for metric, status in (("cka", cka_status), ("d_inter", inter_status),
                                   ("d_intra", intra_status), ("separability", gap_status)):
                status_rows.append({"dataset": context.dataset,
                                    "context_id": context.context_id,
                                    "condition": condition, "position_id": index,
                                    "position_name": position_name,
                                    "metric": metric, "status": status})
            if condition == baseline_name:
                row["delta_inter"] = 0.0
                row["abs_delta_inter"] = 0.0
                row["delta_intra"] = 0.0
                row["abs_delta_intra"] = 0.0
                row["delta_gap"] = 0.0
                row["abs_delta_gap"] = 0.0
                row["delta_separability"] = 0.0
                row["abs_delta_separability"] = 0.0
                row["gap_deterioration"] = 0.0
                row["gap_drop_fraction"] = (0.0 if row["separability"] is not None and
                                             row["separability"] > config.division_epsilon
                                             else None)
                row["distance_ratio"] = (row["d_inter"] / row["d_intra"]
                                          if row["d_inter"] is not None and
                                          row["d_intra"] not in (None, 0) else None)
            else:
                baseline = fp_layer[index]
                for key in ("d_inter", "d_intra", "separability"):
                    row[f"delta_{key}"] = (None if row[key] is None or baseline[key] is None
                                            else row[key] - baseline[key])
                    row[f"abs_delta_{key}"] = (None if row[f"delta_{key}"] is None else
                                                abs(row[f"delta_{key}"]))
                row["delta_gap"] = row["delta_separability"]
                row["abs_delta_gap"] = row["abs_delta_separability"]
                row["delta_inter"] = row["delta_d_inter"]
                row["abs_delta_inter"] = row["abs_delta_d_inter"]
                row["delta_intra"] = row["delta_d_intra"]
                row["abs_delta_intra"] = row["abs_delta_d_intra"]
                row["gap_deterioration"] = (None if row["delta_separability"] is None else
                                             -row["delta_separability"])
                baseline_gap = baseline["separability"]
                if baseline_gap is None or baseline_gap <= config.division_epsilon:
                    row["gap_drop_fraction"] = None
                else:
                    row["gap_drop_fraction"] = (row["gap_deterioration"] / baseline_gap
                                                 if row["gap_deterioration"] is not None else None)
                row["distance_ratio"] = (row["d_inter"] / row["d_intra"]
                                          if row["d_inter"] is not None and
                                          row["d_intra"] not in (None, 0) else None)
            raw_layer[(condition, index)] = row
            layer_rows.append(row)
            if condition == baseline_name:
                fp_layer[index] = row
        declared = context.conditions[condition]["manifest"].get("files", {})
        if "precision_probe.npy" in declared:
            probe = np.load(Path(context.conditions[condition]["path"]) / "precision_probe.npy",
                            mmap_mode="r", allow_pickle=False)
            if probe.ndim == 2:
                probe = np.asarray(probe)[None, ...]
            fp_probe_path = (Path(context.conditions[baseline_name]["path"]) /
                             "precision_probe.npy")
            fp_probe_declared = "precision_probe.npy" in context.conditions[
                baseline_name]["manifest"].get("files", {})
            if (fp_probe_declared and fp_probe_path.is_file() and probe.ndim == 3 and
                    probe.shape[1:] == context.feature_shape[1:]):
                fp_probe = np.load(fp_probe_path, mmap_mode="r", allow_pickle=False)
                if fp_probe.ndim == 2:
                    fp_probe = np.asarray(fp_probe)[None, ...]
                if fp_probe.ndim == 3 and fp_probe.shape == probe.shape:
                    for index, position_name in enumerate(context.position_names):
                        if len(fp_probe) < 2:
                            full_cka = roundtrip_cka = None
                            status = "insufficient_probe_samples"
                        else:
                            full_cka = centered_linear_cka(
                                fp_probe[:, index, :], probe[:, index, :],
                                chunk_size=config.chunk_size)
                            fp_roundtrip = fp_probe[:, index, :].astype(np.float16).astype(np.float32)
                            q_roundtrip = probe[:, index, :].astype(np.float16).astype(np.float32)
                            roundtrip_cka = centered_linear_cka(
                                fp_roundtrip, q_roundtrip, chunk_size=config.chunk_size)
                            status = "complete"
                        probe_rows.append({
                            "dataset": context.dataset, "context_id": context.context_id,
                            "condition": condition, "position_id": index,
                            "position_name": position_name, "n_probe_samples": len(fp_probe),
                            "cka_original_probe_values": full_cka,
                            "cka_fp16_serialization_roundtrip": roundtrip_cka,
                            "absolute_storage_cka_delta": (None if full_cka is None or
                                                            roundtrip_cka is None else
                                                            abs(full_cka - roundtrip_cka)),
                            "status": status})
                else:
                    probe_rows.append({"dataset": context.dataset,
                                       "context_id": context.context_id,
                                       "condition": condition, "status": "incompatible_fp_probe"})
            else:
                probe_rows.append({"dataset": context.dataset, "context_id": context.context_id,
                                   "condition": condition, "status": "incompatible_probe_shape"})
        print(f"METRICS {context.dataset}/{condition}", flush=True)

    drop_rows: list[dict[str, Any]] = []
    for condition in context.conditions:
        scores: list[tuple[int, float]] = []
        steps: list[tuple[int, float]] = []
        for index in range(len(context.position_names)):
            value = raw_layer[(condition, index)]["cka"]
            if value is not None:
                scores.append((index, 1 - float(value)))
            if index > 0:
                prev = raw_layer[(condition, index - 1)]["cka"]
                curr = value
                if prev is not None and curr is not None and float(prev) > float(curr):
                    steps.append((index, float(prev) - float(curr)))
        max_loss = max((value for _, value in scores), default=None)
        max_step = max((value for _, value in steps), default=None)
        for index, position_name in enumerate(context.position_names):
            loss = next((v for i, v in scores if i == index), None)
            step = next((v for i, v in steps if i == index), None)
            drop_rows.append({
                "dataset": context.dataset, "context_id": context.context_id,
                "condition": condition, "position_id": index,
                "position_name": position_name,
                "representation_loss_1_minus_cka": loss,
                "is_largest_representation_loss": loss is not None and loss == max_loss,
                "cka_curve_step_drop": step,
                "is_largest_positive_curve_step_drop": step is not None and step == max_step,
            })
    write_table(output / "condition_summary.csv", summaries)
    write_table(output / "layer_metrics.csv", layer_rows)
    write_table(output / "layer_drop_summary.csv", drop_rows)
    write_table(output / "metric_status.csv", status_rows)
    write_table(output / "precision_probe.csv", probe_rows or [{
        "dataset": context.dataset, "context_id": context.context_id,
        "status": "probe_not_exported"}])
    write_json(output / "context.json", {
        "dataset": context.dataset, "context_id": context.context_id,
        "metadata": context.metadata,
        "feature_shape": context.feature_shape,
        "position_names": context.position_names,
        "conditions": list(context.conditions),
        "source_hashes": context.source_hashes,
    })
    return {"dataset": context.dataset, "context_id": context.context_id,
            "conditions": len(context.conditions), "positions": len(context.position_names),
            "status": "complete"}
