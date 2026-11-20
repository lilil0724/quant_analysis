"""Project paired features into a single FP PCA basis at every position."""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np

from .inputs import Context, open_features, predictions_for
from .metrics import pca_fit, pca_project
from .run_config import RunConfig


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return value or "position"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _quantiles(values: np.ndarray) -> tuple[float | None, float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return None, None
    median, p90 = np.quantile(values, [0.5, 0.9])
    return float(median), float(p90)


def analyze_context(context: Context, config: RunConfig, output: Path) -> dict[str, Any]:
    """Save one FP basis and chunked, matched projections for every condition."""
    basis_dir, coordinate_dir = output / "bases", output / "coords"
    basis_dir.mkdir(parents=True, exist_ok=True)
    coordinate_dir.mkdir(parents=True, exist_ok=True)
    fp = open_features(context, config.baseline_condition)
    labels = np.asarray(context.metadata["targets"], dtype=np.int64)
    fp_rows = predictions_for(context, config.baseline_condition)
    fp_correct = np.fromiter((int(row["correct"]) for row in fp_rows), bool, len(fp_rows))
    summary_rows: list[dict[str, Any]] = []
    sample_order_hash = hashlib.sha256(
        "\n".join(map(str, context.metadata["sample_ids"])).encode("utf-8")).hexdigest()
    for position_id, position_name in enumerate(context.position_names):
        mean, components, explained = pca_fit(fp[:, position_id, :], chunk_size=config.chunk_size)
        pca_status = ("degenerate_zero_variance" if float(np.sum(explained)) <= 1e-12 else
                      "rank_below_two" if float(explained[1]) <= 1e-12 else "complete")
        position_slug = f"{position_id:03d}_{_slug(position_name)}"
        np.savez_compressed(basis_dir / f"{position_slug}.npz", mean=mean,
                            components=components, explained=explained)
        fp_coords = pca_project(fp[:, position_id, :], mean, components,
                                chunk_size=config.chunk_size)
        base_norm = np.empty(len(labels), dtype=np.float64)
        for start in range(0, len(labels), config.chunk_size):
            sl = slice(start, min(start + config.chunk_size, len(labels)))
            base_norm[sl] = np.linalg.norm(np.asarray(fp[sl, position_id, :], dtype=np.float64), axis=1)
        for condition in context.conditions:
            features = open_features(context, condition)
            predictions = predictions_for(context, condition)
            pred_correct = np.fromiter((int(row["correct"]) for row in predictions),
                                       bool, len(predictions))
            if condition == config.baseline_condition:
                q_coords = fp_coords
                displacement = np.zeros(len(labels), dtype=np.float64)
                energy = np.full(len(labels), np.nan, dtype=np.float64)
                relative = np.zeros(len(labels), dtype=np.float64)
            else:
                q_coords = np.empty_like(fp_coords)
                displacement = np.empty(len(labels), dtype=np.float64)
                energy = np.empty(len(labels), dtype=np.float64)
                for start in range(0, len(labels), config.chunk_size):
                    stop = min(start + config.chunk_size, len(labels))
                    sl = slice(start, stop)
                    x = np.asarray(fp[sl, position_id, :], dtype=np.float64)
                    q = np.asarray(features[sl, position_id, :], dtype=np.float64)
                    delta = q - x
                    norm_delta = np.linalg.norm(delta, axis=1)
                    displacement[sl] = norm_delta
                    denom = np.square(norm_delta)
                    coords = ((x - mean) @ components).astype(np.float32)
                    q_part = ((q - mean) @ components).astype(np.float32)
                    q_coords[sl] = q_part
                    proj_energy = np.square(q_part - coords).sum(axis=1)
                    energy[sl] = np.divide(proj_energy, denom,
                                           out=np.full_like(denom, np.nan), where=denom > 0)
                relative = np.divide(displacement, base_norm,
                                     out=np.full_like(displacement, np.nan), where=base_norm > 0)
            coord_path = coordinate_dir / f"{condition}_{position_slug}.npy"
            np.save(coord_path, q_coords)
            outcome = np.full(len(labels), "W_to_W", dtype="U8")
            outcome[fp_correct & pred_correct] = "C_to_C"
            outcome[fp_correct & ~pred_correct] = "C_to_W"
            outcome[~fp_correct & pred_correct] = "W_to_C"
            for group in ("all", "C_to_C", "C_to_W", "W_to_C", "W_to_W"):
                mask = np.ones(len(labels), dtype=bool) if group == "all" else outcome == group
                med, p90 = _quantiles(relative[mask]) if group != "all" or condition != config.baseline_condition else (0.0, 0.0)
                disp_med, disp_p90 = _quantiles(displacement[mask])
                energy_med, energy_p90 = _quantiles(energy[mask])
                summary_rows.append({
                    "dataset": context.dataset, "context_id": context.context_id,
                    "condition": condition, "position_id": position_id,
                    "position_name": position_name, "outcome": group,
                    "n_samples": int(mask.sum()),
                    "pc1_explained_variance": float(explained[0]),
                    "pc2_explained_variance": float(explained[1]),
                    "pca_status": pca_status,
                    "relative_displacement_median": med,
                    "relative_displacement_p90": p90,
                    "full_displacement_median": disp_med,
                    "full_displacement_p90": disp_p90,
                    "projected_energy_fraction_median": energy_med,
                    "projected_energy_fraction_p90": energy_p90,
                    "sample_order_sha256": sample_order_hash,
                    "basis_file": f"bases/{position_slug}.npz",
                    "coordinates_file": f"coords/{condition}_{position_slug}.npy",
                })
        print(f"PCA {context.dataset}/{position_name}", flush=True)
    _write_csv(output / "pca_summary.csv", summary_rows)
    return {"dataset": context.dataset, "context_id": context.context_id,
            "positions": len(context.position_names), "conditions": len(context.conditions),
            "status": "complete"}
