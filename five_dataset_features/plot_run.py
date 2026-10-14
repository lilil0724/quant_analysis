"""Render 20004 figures from verified stage tables and PCA coordinates."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .inputs import Context, predictions_for
from .comparison_values import fixed_condition_values, nullable_number
from .run_config import RunConfig


OUTCOME_COLORS = {"C_to_C": "#4177a8", "C_to_W": "#c43c39",
                  "W_to_C": "#29936b", "W_to_W": "#8a70a0"}


def _save(fig: Any, output: Path, name: str) -> None:
    fig.savefig(output / f"{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _nullable(row: dict[str, str], key: str) -> float:
    return nullable_number(row, key)


def _primary_figures(config: RunConfig, contexts: list[Context],
                     metrics_artifact: Path, comparisons_artifact: Path,
                     output: Path) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    primary = config.primary_condition
    if not primary:
        raise ValueError("Set primary_condition before plotting 20004")
    datasets = [context.dataset for context in sorted(contexts, key=lambda c: c.dataset)]
    cmap = plt.get_cmap("tab20")
    dataset_colors = {dataset: cmap(index % 20) for index, dataset in enumerate(datasets)}
    heat = np.full((len(contexts), len(contexts[0].position_names)), np.nan)
    contexts_sorted = sorted(contexts, key=lambda c: c.dataset)
    figures: list[str] = []
    for d_index, context in enumerate(contexts_sorted):
        folder = metrics_artifact / context.context_id
        layer_rows = _read_rows(folder / "layer_metrics.csv")
        conditions = [config.baseline_condition, primary,
                      *[name for name in config.sensitivity_conditions if name != primary]]
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.7), constrained_layout=True)
        for condition in conditions:
            subset = [row for row in layer_rows if row["condition"] == condition]
            x = [int(row["position_id"]) for row in subset]
            color = "#222222" if condition == config.baseline_condition else (
                "#d55e00" if condition == primary else "#0072b2")
            label = f"{condition} (baseline)" if condition == config.baseline_condition else condition
            axes[0].plot(x, [_nullable(row, "d_inter") for row in subset], marker=".",
                         color=color, label=label + " inter")
            axes[0].plot(x, [_nullable(row, "d_intra") for row in subset], marker=".",
                         linestyle="--", color=color, alpha=.8, label=label + " intra")
            axes[1].plot(x, [_nullable(row, "separability") for row in subset], marker="o",
                         color=color, label=label)
            if condition == primary:
                for row in subset:
                    heat[d_index, int(row["position_id"])] = _nullable(row, "cka")
        axes[0].set(title="Class-equal cosine distances", ylabel="Distance")
        axes[1].set(title="Inter − intra distance gap", ylabel="Gap")
        for ax in axes[:2]:
            ax.set_xlabel("Feature position")
            ax.grid(alpha=.25)
            ax.legend(fontsize=7, ncol=2)
        fp_rows = [row for row in layer_rows if row["condition"] == config.baseline_condition]
        q_rows = [row for row in layer_rows if row["condition"] == primary]
        axes[2].plot([int(row["position_id"]) for row in fp_rows],
                     [_nullable(row, "cka") for row in q_rows], marker="o", color="#6a3d9a")
        axes[2].set(title=f"CKA: {primary} vs FP", ylabel="Centered linear CKA",
                    xlabel="Feature position", ylim=(0, 1.02))
        axes[2].grid(alpha=.25)
        axes[2].set_xticks(range(len(context.position_names)))
        axes[2].set_xticklabels(context.position_names, rotation=75, ha="right", fontsize=7)
        for ax in axes[:2]:
            ax.set_xticks(range(len(context.position_names)))
            ax.set_xticklabels(context.position_names, rotation=75, ha="right", fontsize=7)
        fig.suptitle(f"{context.dataset}: layer metrics; primary={primary}")
        layer_figure = f"layers_{context.dataset}"
        _save(fig, output, layer_figure)
        figures.append(layer_figure)

    fig, ax = plt.subplots(figsize=(max(12, len(contexts) * .75), 6), constrained_layout=True)
    image = ax.imshow(heat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set(yticks=np.arange(len(contexts)), yticklabels=[c.dataset for c in contexts_sorted],
           xticks=np.arange(len(contexts[0].position_names)),
           xticklabels=contexts_sorted[0].position_names,
           title=f"{primary} CKA against matched FP features", xlabel="Feature position")
    plt.setp(ax.get_xticklabels(), rotation=55, ha="right")
    fig.colorbar(image, ax=ax, label="Centered linear CKA")
    _save(fig, output, "cka_heatmap_primary")

    comparison_rows = _read_rows(comparisons_artifact / "comparisons.csv")
    wanted = ["fp_difficulty_vs_gap_deterioration",
              "fp_difficulty_vs_inter_distance_change",
              "fp_difficulty_vs_intra_distance_change", "cka_vs_accuracy_ratio",
              "cka_vs_absolute_accuracy_drop"]
    figures.append("cka_heatmap_primary")
    for analysis in wanted:
        rows = [row for row in comparison_rows if row.get("condition") == primary and
                row.get("analysis") == analysis and row.get("dataset") in (None, "")]
        if not rows:
            continue
        for row in rows:
            x_key, y_key = row["x"], row["y"]
            cohort = set(row.get("datasets", "").split(";"))
            points: list[dict[str, float | str]] = []
            for context in contexts_sorted:
                if context.dataset not in cohort:
                    continue
                context_rows = _read_rows(metrics_artifact / context.context_id /
                                          "layer_metrics.csv")
                summaries = {r["condition"]: r for r in _read_rows(
                    metrics_artifact / context.context_id / "condition_summary.csv")}
                values = fixed_condition_values(context, primary, config.baseline_condition,
                                                context_rows, summaries)
                if np.isfinite(values[x_key]) and np.isfinite(values[y_key]):
                    points.append({"dataset": context.dataset, "x": values[x_key], "y": values[y_key]})
            fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
            for point in points:
                ax.scatter(point["x"], point["y"], color=dataset_colors[point["dataset"]], s=45)
                ax.annotate(point["dataset"], (point["x"], point["y"]), xytext=(4, 4),
                            textcoords="offset points", fontsize=8)
            ax.set(xlabel=x_key.replace("_", " "), ylabel=y_key.replace("_", " "),
                   title=f"{analysis.replace('_', ' ')} ({primary}, n={len(points)})")
            ax.grid(alpha=.25)
            safe = analysis.replace("_", "-")
            stratum_id = hashlib.sha256(row.get("stratum", "").encode()).hexdigest()[:8]
            name = f"scatter_{safe}_{stratum_id}"
            _save(fig, output, name)
            figures.append(name)
    return figures


def _pca_atlases(config: RunConfig, contexts: list[Context],
                  pca_contexts: dict[str, Path],
                  output: Path) -> list[str]:
    atlas_dir = output / "pca_atlas"
    atlas_dir.mkdir(parents=True, exist_ok=True)
    conditions = [config.primary_condition, *config.sensitivity_conditions]
    conditions = list(dict.fromkeys(name for name in conditions if name))
    figures = []
    for context in sorted(contexts, key=lambda c: c.dataset):
        context_artifact = pca_contexts[context.context_id]
        rows = _read_rows(context_artifact / "pca_summary.csv")
        n_pos = len(context.position_names)
        cols = len(conditions)
        fig, axes = plt.subplots(n_pos, cols, figsize=(5.1 * cols, 3.1 * n_pos),
                                 squeeze=False, constrained_layout=True)
        n = context.feature_shape[0]
        seed = int(hashlib.sha256(f"{config.run_id}/{context.context_id}/display".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(n, min(n, 400), replace=False))
        targets = np.asarray(context.metadata["targets"], dtype=np.int64)
        correct = {}
        for condition in [config.baseline_condition, *conditions]:
            predictions = predictions_for(context, condition)
            correct[condition] = np.fromiter((int(x["correct"]) for x in predictions),
                                             dtype=bool, count=n)
        classes, counts = np.unique(targets, return_counts=True)
        displayed_classes = set(classes[np.argsort(-counts, kind="stable")[:20]].tolist())
        class_colors = {int(label): plt.get_cmap("tab20")(index)
                        for index, label in enumerate(sorted(displayed_classes))}
        palette = [class_colors.get(int(targets[i]), (.72, .72, .72, .25)) for i in chosen]
        for pos, position_name in enumerate(context.position_names):
            rows_by_condition = {
                condition: next(r for r in rows if r["condition"] == condition and
                                int(r["position_id"]) == pos and r["outcome"] == "all")
                for condition in [config.baseline_condition, *conditions]
            }
            coordinates_by_condition = {
                condition: np.load(context_artifact / row["coordinates_file"],
                                   mmap_mode="r", allow_pickle=False)
                for condition, row in rows_by_condition.items()
            }
            displayed = np.concatenate(
                [coordinates_by_condition[condition][chosen]
                 for condition in [config.baseline_condition, *conditions]], axis=0)
            lower = displayed.min(axis=0)
            upper = displayed.max(axis=0)
            padding = np.maximum((upper - lower) * .05, 1e-5)
            for col, condition in enumerate(conditions):
                row = rows_by_condition[condition]
                coords = coordinates_by_condition[condition]
                fp_coords = coordinates_by_condition[config.baseline_condition]
                ax = axes[pos, col]
                ax.scatter(fp_coords[chosen, 0], fp_coords[chosen, 1], c=palette,
                           s=9, alpha=.30, linewidths=0)
                wrong = correct[config.baseline_condition][chosen] & ~correct[condition][chosen]
                ax.scatter(coords[chosen, 0], coords[chosen, 1], c=palette,
                           s=10, alpha=.6, marker="x", linewidths=.45)
                if wrong.any():
                    ax.scatter(coords[chosen[wrong], 0], coords[chosen[wrong], 1],
                               marker="x", color="#c43c39", s=18, linewidths=.7,
                               label="FP correct → Q wrong")
                ax.set(title=f"{position_name} | {condition} | "
                       f"EVR {_nullable(row, 'pc1_explained_variance'):.1%}/"
                       f"{_nullable(row, 'pc2_explained_variance'):.1%}", xlabel="FP-PC1",
                       ylabel="FP-PC2")
                ax.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
                ax.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
                ax.grid(alpha=.15)
        fig.suptitle(f"{context.dataset}: FP and quantized CLS projected into each layer's FP basis; "
                     f"{len(chosen)} shared display samples. Top 20 classes colored; others gray.")
        name = f"pca_atlas_{context.dataset}"
        _save(fig, atlas_dir, name)
        figures.append(f"pca_atlas/{name}")
    return figures


def plot(config: RunConfig, contexts: list[Context], metrics_artifact: Path,
         comparisons_artifact: Path, pca_artifact: Path,
         pca_contexts: dict[str, Path], output: Path) -> dict[str, Any]:
    """Create all figure files after their numeric sources are complete."""
    figures = _primary_figures(config, contexts, metrics_artifact,
                               comparisons_artifact, output)
    figures.extend(_pca_atlases(config, contexts, pca_contexts, output))
    return {"status": "complete", "figures": figures, "count": len(figures),
            "primary_condition": config.primary_condition}
