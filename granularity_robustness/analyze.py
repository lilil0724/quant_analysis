"""Analyze the association between dataset granularity and HQQ degradation."""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("FGIR", "Ultra-FGIR")
KEY = ["dataset_name", "model_name", "ckpt_kind", "hqq_nbits", "hqq_group_size"]
STRATUM = ["model_name", "ckpt_kind", "hqq_group_size"]
PRIMARY_BITS = [2.0, 3.0, 4.0]
EXPECTED_GROUP_SIZES = [8, 32, 128, 512]
SCENARIOS = (
    dict(name="core_primary_ratio_median", group_col="core_group", bits=PRIMARY_BITS,
         low=2.0, high=4.0, metric="ratio_endpoint_degradation", aggregation="median"),
    dict(name="expanded_primary_ratio_median", group_col="expanded_group", bits=PRIMARY_BITS,
         low=2.0, high=4.0, metric="ratio_endpoint_degradation", aggregation="median"),
    dict(name="broad_primary_ratio_median", group_col="broad_group", bits=PRIMARY_BITS,
         low=2.0, high=4.0, metric="ratio_endpoint_degradation", aggregation="median"),
    dict(name="core_baseline_ge30_ratio_median", group_col="core_group", bits=PRIMARY_BITS,
         low=2.0, high=4.0, metric="ratio_endpoint_degradation", aggregation="median",
         baseline_min=30.0),
    dict(name="core_primary_absolute_median", group_col="core_group", bits=PRIMARY_BITS,
         low=2.0, high=4.0, metric="absolute_endpoint_degradation", aggregation="median"),
    dict(name="core_primary_ratio_mean", group_col="core_group", bits=PRIMARY_BITS,
         low=2.0, high=4.0, metric="ratio_endpoint_degradation", aggregation="mean"),
    dict(name="core_without_2bit_ratio_median", group_col="core_group", bits=[3.0, 4.0],
         low=3.0, high=4.0, metric="ratio_endpoint_degradation", aggregation="median"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save a figure, tolerating a Windows viewer lock on an existing artifact."""
    try:
        fig.savefig(path, dpi=180, bbox_inches="tight")
    except OSError as error:
        if error.errno == 22 and path.is_file() and path.stat().st_size > 0:
            warnings.warn(f"Kept existing plot because Windows has it open: {path}")
        else:
            raise


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def prepare_cells(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = set(KEY) | {
        "ckpt_path", "hqq_top1", "fp32_top1", "accuracy_ratio",
        "baseline_matched", "n_seeds", "matched_seed_count",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    data = summary.copy()
    numeric = ["hqq_nbits", "hqq_group_size", "hqq_top1", "fp32_top1",
               "accuracy_ratio", "n_seeds", "matched_seed_count"]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    matched = _truthy(data["baseline_matched"])
    finite = np.isfinite(data[["hqq_top1", "fp32_top1", "accuracy_ratio"]]).all(axis=1)
    valid_accuracy = data["fp32_top1"].gt(0)
    reasons = np.where(~matched, "unmatched_baseline", "")
    reasons = np.where(~finite, np.where(reasons == "", "nonfinite_value",
                                         reasons + ";nonfinite_value"), reasons)
    reasons = np.where(~valid_accuracy, np.where(reasons == "", "invalid_baseline",
                                                 reasons + ";invalid_baseline"), reasons)
    data["exclusion_reason"] = reasons
    excluded = data.loc[data.exclusion_reason.ne("")].copy()
    cells = data.loc[data.exclusion_reason.eq("")].drop(columns="exclusion_reason").copy()
    if cells.duplicated(KEY).any():
        duplicates = cells.loc[cells.duplicated(KEY, keep=False), KEY]
        raise ValueError(f"Duplicate analysis cells:\n{duplicates.to_string(index=False)}")
    identity = ["dataset_name", "model_name", "ckpt_kind"]
    ambiguous = cells.groupby(identity, dropna=False).ckpt_path.nunique().gt(1)
    if ambiguous.any():
        raise ValueError(f"Checkpoint identity changes within contexts: {ambiguous[ambiguous].index.tolist()}")
    single_seed = cells.n_seeds.eq(1) & cells.matched_seed_count.eq(1)
    expected_ratio = cells.hqq_top1 / cells.fp32_top1
    if not np.allclose(cells.loc[single_seed, "accuracy_ratio"],
                       expected_ratio.loc[single_seed], rtol=1e-9, atol=1e-12):
        raise ValueError("Single-seed accuracy_ratio is inconsistent with HQQ/FP32 accuracy.")
    cells["absolute_drop_pp"] = cells.fp32_top1 - cells.hqq_top1
    return cells.sort_values(KEY).reset_index(drop=True), excluded.reset_index(drop=True)


def load_mapping(path: Path, dataset_names) -> pd.DataFrame:
    mapping = pd.read_csv(path, keep_default_na=False)
    required = {"dataset_name", "core_group", "expanded_group", "broad_group",
                "label_level", "primary_included", "source_url", "notes"}
    missing = sorted(required - set(mapping.columns))
    if missing:
        raise ValueError(f"Mapping is missing columns: {missing}")
    if mapping.dataset_name.duplicated().any():
        raise ValueError("Mapping contains duplicate dataset_name values.")
    observed = set(dataset_names)
    mapped = set(mapping.dataset_name)
    if observed != mapped:
        raise ValueError(f"Mapping/input dataset mismatch; missing={sorted(observed - mapped)}, "
                         f"extra={sorted(mapped - observed)}")
    for column in ["core_group", "expanded_group", "broad_group"]:
        invalid = set(mapping[column]) - {"", *GROUPS}
        if invalid:
            raise ValueError(f"Invalid {column} values: {sorted(invalid)}")
    core = mapping[mapping.core_group.isin(GROUPS)]
    if core.groupby("core_group").size().to_dict() != {"FGIR": 8, "Ultra-FGIR": 4}:
        raise ValueError("Core mapping must contain 8 FGIR and 4 Ultra-FGIR datasets.")
    return mapping.sort_values("dataset_name").reset_index(drop=True)


def _bits_label(bits) -> str:
    return ",".join(f"{value:g}" for value in bits)


def build_effects(cells: pd.DataFrame, mapping: pd.DataFrame, *, scenario: str,
                  group_col: str, bits, low: float, high: float,
                  baseline_min: float | None = None):
    bits = sorted(float(value) for value in bits)
    selected_mapping = mapping[mapping[group_col].isin(GROUPS)][["dataset_name", group_col]]
    selected_mapping = selected_mapping.rename(columns={group_col: "granularity_group"})
    joined = cells.merge(selected_mapping, on="dataset_name", how="inner", validate="many_to_one")
    contexts = cells[["model_name", "ckpt_kind"]].drop_duplicates().sort_values(
        ["model_name", "ckpt_kind"])
    groups = sorted(int(value) for value in cells.hqq_group_size.unique())
    coverage_rows, effect_rows = [], []
    for mapped in selected_mapping.itertuples(index=False):
        for context in contexts.itertuples(index=False):
            context_frame = joined[
                joined.dataset_name.eq(mapped.dataset_name)
                & joined.model_name.eq(context.model_name)
                & joined.ckpt_kind.eq(context.ckpt_kind)
            ]
            baseline_values = context_frame.fp32_top1.drop_duplicates()
            if len(baseline_values) != 1:
                raise ValueError(f"Baseline changes within context: {mapped.dataset_name}, "
                                 f"{context.model_name}, {context.ckpt_kind}")
            baseline = float(baseline_values.iloc[0])
            for group_size in groups:
                frame = context_frame[context_frame.hqq_group_size.eq(group_size)]
                observed = sorted(float(value) for value in frame.hqq_nbits.unique()
                                  if float(value) in bits)
                reason = ""
                if baseline_min is not None and baseline < baseline_min:
                    reason = "baseline_below_threshold"
                elif observed != bits:
                    reason = "incomplete_bit_panel"
                included = reason == ""
                coverage_rows.append(dict(
                    scenario=scenario, dataset_name=mapped.dataset_name,
                    granularity_group=mapped.granularity_group,
                    model_name=context.model_name, ckpt_kind=context.ckpt_kind,
                    hqq_group_size=group_size, baseline_accuracy=baseline,
                    baseline_min=baseline_min if baseline_min is not None else np.nan,
                    required_bits=_bits_label(bits), observed_bits=_bits_label(observed),
                    n_required_bits=len(bits), n_observed_bits=len(observed),
                    included=included, exclusion_reason=reason,
                ))
                if not included:
                    continue
                panel = frame[frame.hqq_nbits.isin(bits)].set_index("hqq_nbits")
                low_row, high_row = panel.loc[low], panel.loc[high]
                row = dict(
                    scenario=scenario, dataset_name=mapped.dataset_name,
                    granularity_group=mapped.granularity_group,
                    model_name=context.model_name, ckpt_kind=context.ckpt_kind,
                    hqq_group_size=group_size, baseline_accuracy=baseline,
                    low_bit=low, high_bit=high,
                    ratio_low=float(low_row.accuracy_ratio),
                    ratio_high=float(high_row.accuracy_ratio),
                    hqq_top1_low=float(low_row.hqq_top1),
                    hqq_top1_high=float(high_row.hqq_top1),
                    ratio_endpoint_degradation=float(high_row.accuracy_ratio - low_row.accuracy_ratio),
                    absolute_endpoint_degradation=float(low_row.absolute_drop_pp - high_row.absolute_drop_pp),
                )
                if set(PRIMARY_BITS).issubset(panel.index):
                    row.update(
                        ratio_d_4to2=float(panel.loc[4.0].accuracy_ratio - panel.loc[2.0].accuracy_ratio),
                        ratio_d_3to2=float(panel.loc[3.0].accuracy_ratio - panel.loc[2.0].accuracy_ratio),
                        ratio_d_4to3=float(panel.loc[4.0].accuracy_ratio - panel.loc[3.0].accuracy_ratio),
                        absolute_d_4to2=float(panel.loc[2.0].absolute_drop_pp - panel.loc[4.0].absolute_drop_pp),
                    )
                effect_rows.append(row)
    return pd.DataFrame(effect_rows), pd.DataFrame(coverage_rows)


def contrast_effects(effects: pd.DataFrame, *, scenario: str, metric: str,
                     aggregation: str) -> pd.DataFrame:
    rows = []
    for stratum, frame in effects.groupby(STRATUM, sort=True):
        grouped = {name: frame.loc[frame.granularity_group.eq(name), metric] for name in GROUPS}
        counts = {name: len(values) for name, values in grouped.items()}
        comparable = all(value >= 2 for value in counts.values())
        if aggregation == "median":
            center = {name: values.median() for name, values in grouped.items()}
        elif aggregation == "mean":
            center = {name: values.mean() for name, values in grouped.items()}
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        rows.append(dict(
            scenario=scenario, model_name=stratum[0], ckpt_kind=stratum[1],
            hqq_group_size=stratum[2], metric=metric, aggregation=aggregation,
            n_fgir=counts["FGIR"], n_ultra_fgir=counts["Ultra-FGIR"],
            fgir_center=center["FGIR"], ultra_fgir_center=center["Ultra-FGIR"],
            fgir_min=grouped["FGIR"].min(), fgir_max=grouped["FGIR"].max(),
            ultra_fgir_min=grouped["Ultra-FGIR"].min(),
            ultra_fgir_max=grouped["Ultra-FGIR"].max(),
            group_difference=(center["Ultra-FGIR"] - center["FGIR"]
                              if comparable else np.nan),
            comparable=comparable,
            status="ok" if comparable else "fewer_than_two_datasets_in_group",
        ))
    return pd.DataFrame(rows)


def retention_level_contrasts(curves: pd.DataFrame) -> pd.DataFrame:
    """Compare accuracy-retention levels within each fully matched condition."""
    rows = []
    level_stratum = [*STRATUM, "hqq_nbits"]
    for stratum, frame in curves.groupby(level_stratum, sort=True):
        grouped = {
            name: frame.loc[frame.granularity_group.eq(name), "accuracy_ratio"]
            for name in GROUPS
        }
        counts = {name: len(values) for name, values in grouped.items()}
        comparable = all(value >= 2 for value in counts.values())
        centers = {name: values.median() for name, values in grouped.items()}
        rows.append(dict(
            model_name=stratum[0], ckpt_kind=stratum[1],
            hqq_group_size=stratum[2], hqq_nbits=stratum[3],
            n_fgir=counts["FGIR"], n_ultra_fgir=counts["Ultra-FGIR"],
            fgir_median_accuracy_ratio=centers["FGIR"],
            ultra_fgir_median_accuracy_ratio=centers["Ultra-FGIR"],
            retention_gap=(centers["Ultra-FGIR"] - centers["FGIR"]
                           if comparable else np.nan),
            comparable=comparable,
            status="ok" if comparable else "fewer_than_two_datasets_in_group",
        ))
    return pd.DataFrame(rows)


def summarize_contrasts(contrasts: pd.DataFrame) -> dict:
    valid = contrasts[contrasts.comparable & contrasts.group_difference.notna()]
    values = valid.group_difference
    return dict(
        n_total_strata=len(contrasts), n_comparable_strata=len(valid),
        n_positive=int(values.gt(0).sum()), n_negative=int(values.lt(0).sum()),
        n_zero=int(values.eq(0).sum()),
        positive_fraction=float(values.gt(0).mean()) if len(values) else np.nan,
        median_group_difference=float(values.median()) if len(values) else np.nan,
        min_group_difference=float(values.min()) if len(values) else np.nan,
        max_group_difference=float(values.max()) if len(values) else np.nan,
    )


def leave_one_out(primary_effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in sorted(primary_effects.dataset_name.unique()):
        group = primary_effects.loc[
            primary_effects.dataset_name.eq(dataset), "granularity_group"].iloc[0]
        contrasts = contrast_effects(
            primary_effects[~primary_effects.dataset_name.eq(dataset)],
            scenario=f"leave_out_{dataset}", metric="ratio_d_4to2", aggregation="median")
        rows.append(dict(left_out_dataset=dataset, left_out_group=group,
                         **summarize_contrasts(contrasts)))
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=["float"]).columns:
        display[column] = display[column].map(
            lambda value: "NA" if pd.isna(value) else f"{value:.{digits}f}")
    columns = list(display.columns)
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in display.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def make_plots(curves, level_contrasts, baseline, primary_effects, primary_contrasts,
               sensitivity, leave_out, coverage, output: Path) -> pd.DataFrame:
    sns.set_theme(style="whitegrid", context="notebook")
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    contexts = curves[["model_name", "ckpt_kind"]].drop_duplicates().sort_values(
        ["model_name", "ckpt_kind"])
    colors = {"FGIR": "#3264a8", "Ultra-FGIR": "#c84a3a"}
    styles = {8: "-", 32: "--", 128: "-.", 512: ":"}

    limit = float(np.nanmax(np.abs(level_contrasts.retention_gap)))
    model_labels = {
        "beitv2_base_patch16_224_in22k": "BEiTv2",
        "swin_base_patch4_window7_224_in22k": "Swin",
        "vit_b16": "ViT",
    }
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True,
                             layout="constrained")
    heatmap = None
    for axis, context in zip(axes.flat, contexts.itertuples(index=False)):
        frame = level_contrasts[
            (level_contrasts.model_name == context.model_name)
            & (level_contrasts.ckpt_kind == context.ckpt_kind)
        ]
        matrix = frame.pivot(index="hqq_nbits", columns="hqq_group_size",
                             values="retention_gap").sort_index(ascending=False)
        heatmap = sns.heatmap(
            matrix, annot=True, fmt=".2f", center=0, cmap="RdBu_r",
            vmin=-limit, vmax=limit, cbar=False, linewidths=.5,
            linecolor="white", ax=axis,
        )
        axis.set_title(f"{model_labels.get(context.model_name, context.model_name)} / "
                       f"{context.ckpt_kind}")
        axis.set_xlabel("group size")
        axis.set_ylabel("HQQ bits")
    colorbar = fig.colorbar(
        heatmap.collections[0], ax=axes, fraction=.025, pad=.02, shrink=.78,
        label="Median accuracy_ratio gap (Ultra-FGIR − FGIR)",
    )
    colorbar.ax.axhline(0, color="black", linewidth=.8)
    fig.suptitle(
        "Matched-condition accuracy retention: Ultra-FGIR minus FGIR\n"
        "Blue/negative cells mean lower Ultra-FGIR retention",
    )
    path = plot_dir / "granularity_retention_level_gaps.png"
    save_figure(fig, path)
    plt.close(fig)
    manifest.append(dict(
        file=str(path.relative_to(output)), source_table="retention_level_contrasts.csv",
        description="Matched-condition Ultra-FGIR minus FGIR median accuracy-ratio levels",
    ))

    fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharex=True)
    for axis, context in zip(axes.flat, contexts.itertuples(index=False)):
        frame = curves[(curves.model_name == context.model_name)
                       & (curves.ckpt_kind == context.ckpt_kind)]
        for (group, group_size, dataset), values in frame.groupby(
                ["granularity_group", "hqq_group_size", "dataset_name"]):
            values = values.sort_values("hqq_nbits")
            axis.plot(values.hqq_nbits, values.accuracy_ratio,
                      color=colors[group], linestyle=styles[int(group_size)],
                      alpha=.12, linewidth=.8)
        medians = frame.groupby(["granularity_group", "hqq_group_size", "hqq_nbits"],
                                as_index=False).accuracy_ratio.median()
        for (group, group_size), values in medians.groupby(
                ["granularity_group", "hqq_group_size"]):
            values = values.sort_values("hqq_nbits")
            axis.plot(values.hqq_nbits, values.accuracy_ratio,
                      color=colors[group], linestyle=styles[int(group_size)], linewidth=2,
                      marker="o", label=f"{group}, g={int(group_size)}")
        axis.set_title(f"{context.model_name} / {context.ckpt_kind}")
        axis.set_xticks(PRIMARY_BITS)
        axis.axhline(1.0, color="black", linewidth=.7, alpha=.4)
    for axis in axes[-1]:
        axis.set_xlabel("HQQ bits")
    for axis in axes[:, 0]:
        axis.set_ylabel("accuracy ratio")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle("Granularity groups: dataset trajectories and medians", y=.995)
    fig.tight_layout(rect=[0, .07, 1, .98])
    path = plot_dir / "granularity_retention_curves.png"
    save_figure(fig, path)
    plt.close(fig)
    manifest.append(dict(file=str(path.relative_to(output)), source_table="primary_curve_cells.csv",
                         description="Primary 2/3/4-bit retention curves by context and group size"))

    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    for axis, context in zip(axes.flat, contexts.itertuples(index=False)):
        frame = primary_contrasts[(primary_contrasts.model_name == context.model_name)
                                  & (primary_contrasts.ckpt_kind == context.ckpt_kind)]
        matrix = frame.set_index("hqq_group_size")[["group_difference"]].T
        sns.heatmap(matrix, annot=True, fmt=".3f", center=0, cmap="coolwarm",
                    cbar=False, ax=axis, vmin=primary_contrasts.group_difference.min(),
                    vmax=primary_contrasts.group_difference.max())
        axis.set_title(f"{context.model_name} / {context.ckpt_kind}")
        axis.set_xlabel("group size")
        axis.set_ylabel("")
    fig.suptitle("Ultra-FGIR minus FGIR median 4→2-bit degradation", y=1.01)
    fig.tight_layout()
    path = plot_dir / "granularity_stratum_contrasts.png"
    save_figure(fig, path)
    plt.close(fig)
    manifest.append(dict(file=str(path.relative_to(output)), source_table="stratum_group_contrasts.csv",
                         description="Primary matched-stratum group differences"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.stripplot(data=baseline, x="display_group", y="fp32_top1", hue="display_group",
                  legend=False, jitter=.25, alpha=.7, ax=axes[0])
    axes[0].axhline(30, color="#b22222", linestyle="--", label="30% threshold")
    axes[0].set(xlabel="dataset group", ylabel="FP32 top-1 (%)", title="Baseline distribution")
    axes[0].legend()
    cov = coverage[(coverage.scenario == "core_baseline_ge30_ratio_median")]
    counts = cov[cov.included].groupby(
        ["model_name", "ckpt_kind", "granularity_group"]).dataset_name.nunique().reset_index()
    counts["context"] = counts.model_name + "/" + counts.ckpt_kind
    sns.barplot(data=counts, y="context", x="dataset_name", hue="granularity_group",
                palette=colors, ax=axes[1])
    axes[1].axvline(2, color="black", linestyle=":", linewidth=1)
    axes[1].set(xlabel="datasets retained at baseline ≥ 30%", ylabel="",
                title="Sensitivity-analysis coverage")
    fig.tight_layout()
    path = plot_dir / "granularity_baseline_coverage.png"
    save_figure(fig, path)
    plt.close(fig)
    manifest.append(dict(file=str(path.relative_to(output)), source_table="baseline_contexts.csv",
                         description="Baseline imbalance and retained dataset coverage"))

    ordered = leave_out.sort_values(["left_out_group", "left_out_dataset"])
    fig, axis = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=ordered, x="median_group_difference", y="left_out_dataset",
                    hue="left_out_group", palette=colors, s=90, ax=axis)
    primary_median = primary_contrasts.group_difference.median()
    axis.axvline(primary_median, color="black", linestyle="--", label="all datasets")
    axis.axvline(0, color="grey", linewidth=.8)
    axis.set(xlabel="Median Ultra-FGIR − FGIR degradation after leave-one-out",
             ylabel="dataset removed", title="Leave-one-dataset-out sensitivity")
    axis.legend()
    fig.tight_layout()
    path = plot_dir / "granularity_leave_one_out.png"
    save_figure(fig, path)
    plt.close(fig)
    manifest.append(dict(file=str(path.relative_to(output)), source_table="leave_one_dataset_out.csv",
                         description="Influence of each core dataset"))

    ratio = sensitivity[sensitivity.metric.eq("ratio_endpoint_degradation")].sort_values(
        "median_group_difference")
    absolute = sensitivity[sensitivity.metric.eq("absolute_endpoint_degradation")].sort_values(
        "median_group_difference")
    fig, axes = plt.subplots(1, 2, figsize=(17, 6), gridspec_kw={"width_ratios": [2.2, 1]})
    for axis, frame, xlabel in [
        (axes[0], ratio, "Accuracy-ratio group difference"),
        (axes[1], absolute, "Absolute-drop difference (percentage points)"),
    ]:
        positions = np.arange(len(frame))
        axis.errorbar(frame.median_group_difference, positions,
                      xerr=[frame.median_group_difference - frame.min_group_difference,
                            frame.max_group_difference - frame.median_group_difference],
                      fmt="o", capsize=3)
        axis.axvline(0, color="grey", linewidth=.8)
        axis.set_yticks(positions, frame.scenario)
        axis.set_xlabel(xlabel)
        for position, row in enumerate(frame.itertuples()):
            axis.annotate(f"{row.n_comparable_strata}/{row.n_total_strata}",
                          (row.max_group_difference, position), xytext=(5, 0),
                          textcoords="offset points", va="center", fontsize=8)
    axes[0].set_title("Ratio sensitivities")
    axes[1].set_title("Absolute-drop sensitivity")
    fig.suptitle("Pre-specified sensitivity analyses")
    fig.tight_layout()
    path = plot_dir / "granularity_sensitivity_summary.png"
    save_figure(fig, path)
    plt.close(fig)
    manifest.append(dict(file=str(path.relative_to(output)), source_table="granularity_sensitivity.csv",
                         description="Sensitivity effect ranges and comparable-stratum counts"))
    return pd.DataFrame(manifest)


def evidence_grade(primary, sensitivity, leave_out) -> str:
    valid = primary[primary.comparable].group_difference.dropna()
    key = sensitivity[sensitivity.scenario.isin({
        "expanded_primary_ratio_median", "broad_primary_ratio_median",
        "core_baseline_ge30_ratio_median", "core_primary_absolute_median",
    })].dropna(subset=["median_group_difference"])
    if valid.empty:
        return "insufficient evidence"
    primary_signs = set(np.sign(valid[valid.ne(0)]))
    reference_sign = np.sign(valid.median())
    sensitivity_same = (np.sign(key.median_group_difference).eq(reference_sign).all()
                        if reference_sign != 0 and len(key) else False)
    loo_same = (np.sign(leave_out.median_group_difference).eq(reference_sign).all()
                if reference_sign != 0 else False)
    if len(primary_signs) == 1 and sensitivity_same and loo_same:
        return "consistent association"
    if len(primary_signs) > 1:
        return "context-dependent association"
    return "insufficient evidence"


def make_report(output, config, mapping, primary, level_contrasts, sensitivity, leave_out,
                context_summary):
    main = summarize_contrasts(primary)
    grade = evidence_grade(primary, sensitivity, leave_out)
    low = sensitivity[sensitivity.scenario.eq("core_baseline_ge30_ratio_median")].iloc[0]
    abs_row = sensitivity[sensitivity.scenario.eq("core_primary_absolute_median")].iloc[0]
    grouping = sensitivity[sensitivity.scenario.isin(
        ["core_primary_ratio_median", "expanded_primary_ratio_median",
         "broad_primary_ratio_median"])]
    level_summary = level_contrasts.groupby("hqq_nbits").agg(
        n_strata=("retention_gap", "size"),
        n_ultra_lower=("retention_gap", lambda values: int(values.lt(0).sum())),
        median_retention_gap=("retention_gap", "median"),
        min_retention_gap=("retention_gap", "min"),
        max_retention_gap=("retention_gap", "max"),
    ).reset_index()
    lines = [
        "# Dataset granularity 與 HQQ 量化退化分析", "",
        "## 結論", "",
        f"本次 evidence grade 為 **{grade}**。這是目前 benchmark 內的描述性 association，"
        "不能解讀為辨識粒度的因果效果。", "",
        f"核心 8 FGIR vs 4 Ultra-FGIR 分析共有 {main['n_comparable_strata']}/"
        f"{main['n_total_strata']} 個可比較 strata；Ultra-FGIR − FGIR 的 4→2-bit degradation "
        f"中位差為 **{main['median_group_difference']:.4f}**。正向、負向與零差異 strata "
        f"分別為 {main['n_positive']}、{main['n_negative']}、{main['n_zero']}。", "",
        "正差值表示 Ultra-FGIR 的 retention 從 4-bit 降到 2-bit 時下降更多。", "",
        "## 固定量化條件下的 retention level", "",
        "這項比較直接對應『在相同量化條件下誰保留較多 FP32 accuracy』。負值表示 "
        "Ultra-FGIR 的 median accuracy_ratio 較低；它與 4→2-bit degradation slope 是不同問題。",
        "", markdown_table(level_summary), "",
        "![Matched retention gaps](plots/granularity_retention_level_gaps.png)", "",
        "## Model/checkpoint 背景", "", markdown_table(context_summary), "",
        "## Dataset 分類邊界敏感度", "", markdown_table(grouping[[
            "scenario", "n_comparable_strata", "n_positive", "n_negative",
            "positive_fraction", "median_group_difference", "min_group_difference",
            "max_group_difference"]]), "",
        "## Baseline 與 outcome 敏感度", "",
        f"排除 FP32 baseline <30% 後，可比較 strata 為 {int(low.n_comparable_strata)}/"
        f"{int(low.n_total_strata)}，group difference 中位數為 {low.median_group_difference:.4f}。"
        "缺少 Ultra-FGIR coverage 的 strata 保留為不可比較，沒有補值。", "",
        f"改用 absolute accuracy-drop outcome 時，中位組間差為 "
        f"{abs_row.median_group_difference:.4f} percentage points。Ratio 與 absolute-drop "
        "的數值單位不同，只比較方向與背景一致性。", "",
        "完整預先指定 sensitivity：", "", markdown_table(sensitivity[[
            "scenario", "metric", "aggregation", "n_comparable_strata", "n_positive",
            "n_negative", "positive_fraction", "median_group_difference",
            "min_group_difference", "max_group_difference"]]), "",
        "![Baseline coverage](plots/granularity_baseline_coverage.png)", "",
        "## 主要圖表", "",
        "![Retention curves](plots/granularity_retention_curves.png)", "",
        "![Stratum contrasts](plots/granularity_stratum_contrasts.png)", "",
        "![Leave-one-out](plots/granularity_leave_one_out.png)", "",
        "![Sensitivity summary](plots/granularity_sensitivity_summary.png)", "",
        "## 方法", "",
        "- Primary scope 是 2/3/4-bit 與 group sizes 8/32/128/512。",
        "- 每個 dataset 在相同 model/checkpoint/group-size 內計算 4→2-bit degradation。",
        "- 每個 dataset 等權；group median difference 是 primary estimand。",
        "- 每組少於 2 個 datasets 的 sensitivity stratum 標記為不可比較。",
        "- 沒有把 cells 當成獨立 training seeds，也沒有產生 p-value 或 seed-level CI。", "",
        "## 主要限制", "",
        "- Ultra-FGIR 組只有四個 soybean datasets，granularity 與 leaf domain、資料來源及"
        " baseline 品質高度共線。",
        "- 每個 dataset/model/checkpoint context 只有一個 training seed。",
        "- Dataset-specific checkpoints 使 dataset 與 training outcome 無法完全分離。",
        "- 低 baseline filtering 不成比例地移除 Ultra-FGIR contexts；coverage loss 是結果的一部分。",
        "- 目前只有 aggregate accuracy，不能分析 per-sample damage/rescue 或 calibration。", "",
        "## 可重現性", "",
        f"- Input：`{config['input_file']}`",
        f"- Input SHA-256：`{config['input_sha256']}`",
        f"- Mapping SHA-256：`{config['mapping_sha256']}`",
        f"- Core primary cells：{config['n_core_primary_cells']}",
        f"- Generated at：{config['created_at']}", "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(input_file: Path, mapping_file: Path, output: Path):
    cells, excluded = prepare_cells(pd.read_csv(input_file))
    mapping = load_mapping(mapping_file, cells.dataset_name.unique())
    full_bits = sorted(float(value) for value in cells.hqq_nbits.unique())
    scenarios = list(SCENARIOS) + [dict(
        name="core_full_grid_ratio_median", group_col="core_group", bits=full_bits,
        low=min(full_bits), high=max(full_bits), metric="ratio_endpoint_degradation",
        aggregation="median")]
    output.mkdir(parents=True, exist_ok=True)
    all_effects, all_coverage, all_contrasts, summaries = [], [], [], []
    effects_by_name = {}
    for definition in scenarios:
        definition = definition.copy()
        name = definition.pop("name")
        metric = definition.pop("metric")
        aggregation = definition.pop("aggregation")
        effects, coverage = build_effects(cells, mapping, scenario=name, **definition)
        contrasts = contrast_effects(effects, scenario=name, metric=metric,
                                     aggregation=aggregation)
        effects_by_name[name] = effects
        all_effects.append(effects)
        all_coverage.append(coverage)
        all_contrasts.append(contrasts)
        summaries.append(dict(scenario=name, metric=metric, aggregation=aggregation,
                              **summarize_contrasts(contrasts)))
    primary_name = "core_primary_ratio_median"
    primary_effects = effects_by_name[primary_name].copy()
    primary_contrasts = all_contrasts[0].copy()
    sensitivity = pd.DataFrame(summaries)
    coverage = pd.concat(all_coverage, ignore_index=True)
    sensitivity_contrasts = pd.concat(all_contrasts, ignore_index=True)
    leave_out = leave_one_out(primary_effects)
    context_summary = primary_contrasts.groupby(
        ["model_name", "ckpt_kind"], as_index=False).agg(
            n_strata=("group_difference", "size"),
            n_positive=("group_difference", lambda values: int(values.gt(0).sum())),
            n_negative=("group_difference", lambda values: int(values.lt(0).sum())),
            median_group_difference=("group_difference", "median"),
            min_group_difference=("group_difference", "min"),
            max_group_difference=("group_difference", "max"))
    dataset_summary = primary_effects.groupby(
        ["dataset_name", "granularity_group"], as_index=False).agg(
            n_strata=("ratio_d_4to2", "size"),
            median_ratio_d_4to2=("ratio_d_4to2", "median"),
            min_ratio_d_4to2=("ratio_d_4to2", "min"),
            max_ratio_d_4to2=("ratio_d_4to2", "max"),
            median_absolute_d_4to2=("absolute_d_4to2", "median"))
    core = mapping[mapping.core_group.isin(GROUPS)][["dataset_name", "core_group"]]
    primary_curves = cells.merge(core, on="dataset_name", how="inner").rename(
        columns={"core_group": "granularity_group"})
    primary_curves = primary_curves[primary_curves.hqq_nbits.isin(PRIMARY_BITS)].copy()
    level_contrasts = retention_level_contrasts(primary_curves)
    baseline_contexts = cells[["dataset_name", "model_name", "ckpt_kind", "fp32_top1"]].drop_duplicates()
    baseline_contexts = baseline_contexts.merge(
        mapping[["dataset_name", "core_group"]], on="dataset_name", validate="many_to_one")
    baseline_contexts["display_group"] = baseline_contexts.core_group.replace("", "Boundary")
    primary_effects.to_csv(output / "dataset_degradation_effects.csv", index=False)
    primary_contrasts.to_csv(output / "stratum_group_contrasts.csv", index=False)
    sensitivity_contrasts.to_csv(output / "sensitivity_stratum_contrasts.csv", index=False)
    sensitivity.to_csv(output / "granularity_sensitivity.csv", index=False)
    coverage.to_csv(output / "granularity_coverage.csv", index=False)
    leave_out.to_csv(output / "leave_one_dataset_out.csv", index=False)
    context_summary.to_csv(output / "context_direction_summary.csv", index=False)
    dataset_summary.to_csv(output / "dataset_effect_summary.csv", index=False)
    primary_curves.to_csv(output / "primary_curve_cells.csv", index=False)
    level_contrasts.to_csv(output / "retention_level_contrasts.csv", index=False)
    baseline_contexts.to_csv(output / "baseline_contexts.csv", index=False)
    excluded.to_csv(output / "excluded_cells.csv", index=False)
    mapping.to_csv(output / "dataset_granularity_mapping.csv", index=False)
    manifest = make_plots(primary_curves, level_contrasts, baseline_contexts, primary_effects,
                          primary_contrasts, sensitivity, leave_out, coverage, output)
    manifest.to_csv(output / "plots_manifest.csv", index=False)
    config = dict(
        input_file=str(input_file), mapping_file=str(mapping_file), results_dir=str(output),
        input_sha256=sha256(input_file), mapping_sha256=sha256(mapping_file),
        script_sha256=sha256(Path(__file__)), created_at=datetime.now(timezone.utc).isoformat(),
        primary_bits=PRIMARY_BITS, group_sizes=EXPECTED_GROUP_SIZES,
        grouping_counts={"FGIR": 8, "Ultra-FGIR": 4, "boundary": 3},
        n_input_cells=len(cells) + len(excluded), n_valid_cells=len(cells),
        n_excluded_cells=len(excluded), n_core_primary_cells=int(
            cells.dataset_name.isin(mapping[mapping.core_group.isin(GROUPS)].dataset_name).mul(
                cells.hqq_nbits.isin(PRIMARY_BITS)).sum()),
        inference="descriptive_only_no_p_values_no_seed_CIs",
        weighting="equal_dataset_weight_within_matched_stratum",
        evidence_grade=evidence_grade(primary_contrasts, sensitivity, leave_out),
        scenarios=[definition["name"] for definition in scenarios],
    )
    (output / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    make_report(output, config, mapping, primary_contrasts, level_contrasts, sensitivity,
                leave_out, context_summary)
    return config, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path,
                        default=ROOT / "results_all/dataset_robustness/legacy/quant_summary.csv")
    parser.add_argument("--mapping-file", type=Path,
                        default=Path(__file__).with_name("dataset_granularity_mapping.csv"))
    parser.add_argument("--results-dir", type=Path,
                        default=ROOT / "results_all/granularity_robustness")
    args = parser.parse_args()
    config, manifest = run(args.input_file.resolve(), args.mapping_file.resolve(),
                           args.results_dir.resolve())
    print(f"Analyzed {config['n_valid_cells']} cells; generated {len(manifest)} figures. "
          f"Evidence grade: {config['evidence_grade']}. Report: {args.results_dir / 'report.md'}")


if __name__ == "__main__":
    main()
