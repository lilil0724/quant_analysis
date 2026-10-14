"""Dataset-unit associations, group contrasts, and matched-N CKA checks."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .inputs import Context, open_features, read_csv
from .comparison_values import fixed_condition_values, nullable_number
from .metrics import centered_linear_cka, spearman_description
from .run_config import RunConfig


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _stratum(context: Context) -> tuple[str, bool]:
    metadata = context.metadata
    model = metadata.get("model", metadata.get("model_name"))
    head = metadata.get("head", metadata.get("head_name"))
    seed = metadata.get("training_seed", metadata.get("seed"))
    profile = metadata.get("inference_profile", metadata.get("preprocess"))
    execution = metadata.get("execution")
    missing = (any(value in (None, "", "unknown") for value in (model, head, seed)) or
               profile in (None, "", {}) or execution in (None, "", {}))
    # Preserve every known context field even when another field is absent.
    # Recorded execution precision must match; preprocessing alone cannot replace it.
    key = [model, head, seed, profile, metadata.get("preprocess"), execution]
    return json.dumps(key, sort_keys=True, ensure_ascii=False), not missing


def _condition_family(spec: dict[str, Any]) -> str:
    if ("act_bits" not in spec or "qkv" not in spec or
            spec.get("weight_bits") is None):
        return "unknown"
    if spec["qkv"] is True:
        return "QKV"
    if spec["qkv"] is not False:
        return "unknown"
    if spec["act_bits"] is None:
        return "weight_only"
    if spec["act_bits"] in (4, 8):
        return f"A{spec['act_bits']}"
    return "unknown"


def _corr_ci(x: np.ndarray, y: np.ndarray, config: RunConfig, seed: int,
             include_permutation: bool = True
             ) -> tuple[float | None, float | None, int, float | None]:
    rho = spearman_description(x, y)
    n = len(x)
    if rho is None:
        return None, None, 0, None
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(config.bootstrap_repeats):
        indices = rng.integers(0, n, n)
        estimate = spearman_description(x[indices], y[indices])
        if estimate is not None:
            samples.append(estimate)
    lo, hi = (map(float, np.quantile(samples, [0.025, 0.975]))
              if len(samples) >= max(50, config.bootstrap_repeats // 4) else (None, None))
    p = None
    if include_permutation:
        # For n=16 this is a direct Monte Carlo randomization at the dataset level.
        extreme = 0
        valid = 0
        for _ in range(config.permutation_repeats):
            estimate = spearman_description(x, rng.permutation(y))
            if estimate is not None:
                valid += 1
                extreme += abs(estimate) >= abs(rho)
        if valid:
            p = (1 + extreme) / (valid + 1)
    return lo, hi, len(samples), p


def _leave_one_out(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    values = [spearman_description(np.delete(x, index), np.delete(y, index))
              for index in range(len(x))]
    defined = [value for value in values if value is not None]
    if not defined:
        return {"loo_defined_repeats": 0, "loo_rho_min": None,
                "loo_rho_max": None, "loo_sign_stable": None}
    signs = {int(np.sign(value)) for value in defined}
    return {"loo_defined_repeats": len(defined), "loo_rho_min": min(defined),
            "loo_rho_max": max(defined), "loo_sign_stable": len(signs) == 1}


def _bh(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row.get("dataset_permutation_p_value") not in (None, ""):
            groups[(row["stratum"], row["condition"])].append(index)
    for indices in groups.values():
        ordered = sorted(indices, key=lambda i: rows[i]["dataset_permutation_p_value"])
        q = 1.0
        m = len(ordered)
        for rank in range(m, 0, -1):
            index = ordered[rank - 1]
            p = rows[index]["dataset_permutation_p_value"]
            q = min(q, float(p) * m / rank)
            rows[index]["fdr_q_value"] = min(1.0, q)


def _group_mapping(config: RunConfig) -> tuple[dict[str, str], dict[str, str], str | None]:
    if not config.group_mapping_file:
        return {}, {}, None
    if not config.group_mapping_file.is_file():
        raise FileNotFoundError(f"Group mapping does not exist: {config.group_mapping_file}")
    rows = read_csv(config.group_mapping_file)
    if not rows:
        raise ValueError("Group mapping file is empty")
    core: dict[str, str] = {}
    broad: dict[str, str] = {}
    seen: set[str] = set()
    for row in rows:
        if not {"dataset_name", "core_group", "expanded_group"}.issubset(row):
            raise ValueError("Group mapping requires dataset_name, core_group, expanded_group columns")
        dataset = row.get("dataset_name", row.get("dataset", "")).strip()
        if not dataset:
            raise ValueError("Group mapping has a row with no dataset_name")
        if dataset in seen:
            raise ValueError(f"Duplicate dataset in group mapping: {dataset}")
        seen.add(dataset)
        def normalize(value: str) -> str:
            cleaned = value.strip().casefold()
            if cleaned in {"ufgir", "ultra-fgir"}:
                return "UFGIR"
            if cleaned == "fgir":
                return "FGIR"
            return value.strip()

        core_value = normalize(row.get("core_group", row.get("group", "")))
        broad_value = normalize(row.get("expanded_group", row.get("broad_group", "")))
        allowed = {"", "FGIR", "UFGIR"}
        if core_value not in allowed or broad_value not in allowed:
            raise ValueError(f"Unsupported FGIR/UFGIR mapping value for {dataset!r}")
        if core_value:
            core[dataset] = core_value
        if broad_value:
            broad[dataset] = broad_value
    return core, broad, hashlib.sha256(config.group_mapping_file.read_bytes()).hexdigest()


def _group_difference(values: list[tuple[str, float]], group_map: dict[str, str],
                      config: RunConfig, seed: int, scope: str,
                      formal_inference: bool) -> list[dict[str, Any]]:
    selected = [(name, value, group_map.get(name)) for name, value in values
                if group_map.get(name) in {"FGIR", "UFGIR"} and np.isfinite(value)]
    left = np.asarray([v for _, v, group in selected if group == "UFGIR"], dtype=float)
    right = np.asarray([v for _, v, group in selected if group == "FGIR"], dtype=float)
    base = {"scope": scope, "group_a": "UFGIR", "group_b": "FGIR",
            "n_ufgir": len(left), "n_fgir": len(right),
            "datasets_ufgir": ";".join(n for n, _, g in selected if g == "UFGIR"),
            "datasets_fgir": ";".join(n for n, _, g in selected if g == "FGIR"),
            "excluded_undefined_datasets": ";".join(name for name, value in values
                                                    if not np.isfinite(value))}
    if min(len(left), len(right)) < 2:
        return [{**base, "status": "insufficient_group_coverage", "median_difference": None,
                 "ci95_low": None, "ci95_high": None, "permutation_p_value": None}]
    observed = float(np.median(left) - np.median(right))
    rng = np.random.default_rng(seed)
    boot = np.empty(config.bootstrap_repeats, dtype=np.float64)
    for i in range(len(boot)):
        boot[i] = (np.median(rng.choice(left, len(left), replace=True)) -
                   np.median(rng.choice(right, len(right), replace=True)))
    lo, hi = map(float, np.quantile(boot, [0.025, 0.975]))
    p_value = None
    if formal_inference:
        pooled = np.concatenate((left, right))
        n_left = len(left)
        extreme = 0
        for _ in range(config.permutation_repeats):
            perm = rng.permutation(pooled)
            diff = np.median(perm[:n_left]) - np.median(perm[n_left:])
            extreme += abs(diff) >= abs(observed)
        p_value = (1 + extreme) / (config.permutation_repeats + 1)
    return [{**base, "status": ("dataset_group_permutation" if formal_inference else
                                "descriptive_only_unverified_stratum"),
             "median_ufgir": float(np.median(left)), "median_fgir": float(np.median(right)),
             "median_difference": observed, "ci95_low": lo, "ci95_high": hi,
             "permutation_p_value": p_value}]


def _balanced_plan(labels: np.ndarray, n: int
                   ) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    classes, counts = np.unique(labels, return_counts=True)
    count_by_class = dict(zip(classes.tolist(), counts.tolist()))
    for size in range(min(len(classes), n), 1, -1):
        base, remainder = divmod(n, size)
        base_classes = np.asarray([label for label in classes
                                   if count_by_class[label] >= base])
        extra_classes = np.asarray([label for label in classes
                                    if count_by_class[label] >= base + 1])
        if len(base_classes) >= size and len(extra_classes) >= remainder:
            return base_classes, extra_classes, size, remainder
    return None


def _balanced_capacity(labels: np.ndarray, requested_n: int) -> int:
    for n in range(requested_n, 2, -1):
        if _balanced_plan(labels, n) is not None:
            return n
    return 0


def _balanced_indices(labels: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    plan = _balanced_plan(labels, n)
    if plan is None:
        raise ValueError(f"Cannot draw {n} distinct class-balanced samples")
    base_classes, extra_classes, size, remainder = plan
    extras = (rng.choice(extra_classes, remainder, replace=False)
              if remainder else np.empty(0, dtype=labels.dtype))
    extra_set = set(extras.tolist())
    remaining_classes = np.asarray([label for label in base_classes
                                    if label not in extra_set])
    selected = [*extras.tolist(),
                *rng.choice(remaining_classes, size - remainder, replace=False).tolist()]
    chosen: list[np.ndarray] = []
    base = n // size
    for label in selected:
        count = base + int(label in extra_set)
        eligible = np.flatnonzero(labels == label)
        chosen.append(rng.choice(eligible, count, replace=False))
    return np.sort(np.concatenate(chosen).astype(np.int64))


def _cka_sampling(contexts: list[Context], config: RunConfig,
                  primary: str, stratum: str) -> list[dict[str, Any]]:
    capacities = [_balanced_capacity(np.asarray(context.metadata["targets"], dtype=np.int64),
                                     min(context.feature_shape[0], config.cka_sampling_n_max))
                  for context in contexts]
    common_n = min(capacities)
    if common_n < 3:
        return [{"stratum": stratum, "condition": primary, "n_datasets": len(contexts),
                 "common_n": common_n, "sampling_repeats": 0,
                 "status": "insufficient_samples", "median_cka": None,
                 "p10_cka": None, "p90_cka": None}]
    values: list[float] = []
    detail_rows: list[dict[str, Any]] = []
    cka_by_dataset: dict[str, list[float]] = defaultdict(list)
    for context in contexts:
        labels = np.asarray(context.metadata["targets"], dtype=np.int64)
        fp = open_features(context, config.baseline_condition)
        q = open_features(context, primary)
        seed_base = int(hashlib.sha256(
            f"{config.resampling_seed}/{context.context_id}/{primary}".encode()).hexdigest()[:8], 16)
        for repetition in range(config.cka_sampling_repeats):
            rng = np.random.default_rng(seed_base + repetition)
            indices = _balanced_indices(labels, common_n, rng)
            rho = centered_linear_cka(fp[indices, -1, :], q[indices, -1, :],
                                      chunk_size=config.chunk_size)
            if rho is not None:
                values.append(float(rho))
                cka_by_dataset[context.dataset].append(float(rho))
            detail_rows.append({"stratum": stratum, "condition": primary,
                                "status": "sampling_repetition", "dataset": context.dataset,
                                "repetition": repetition, "seed": seed_base + repetition,
                                "common_n": common_n,
                                "n_classes_sampled": len(np.unique(labels[indices])),
                                "sample_indices_sha256": hashlib.sha256(
                                    indices.astype(np.int64, copy=False).tobytes()).hexdigest(),
                                "sample_order_sha256": hashlib.sha256(
                                    "\n".join(map(str, context.metadata["sample_ids"])).encode()).hexdigest(),
                                "cka": float(rho) if rho is not None else None})
    med = float(np.median(values)) if values else None
    p10, p90 = map(float, np.quantile(values, [0.1, 0.9])) if values else (None, None)
    rows = [{"stratum": stratum, "condition": primary, "n_datasets": len(contexts),
             "common_n": common_n, "sampling_repeats": len(detail_rows),
             "repeats_per_dataset": config.cka_sampling_repeats,
             "n_valid_cka_repeats": len(values),
             "resampling_seed": config.resampling_seed,
             "status": "matched_n_class_balanced_sampling_sensitivity",
             "median_cka": med, "p10_cka": p10, "p90_cka": p90,
             "datasets": ";".join(c.dataset for c in contexts)}]
    for dataset, samples in sorted(cka_by_dataset.items()):
        rows.append({"stratum": stratum, "condition": primary, "n_datasets": 1,
                     "common_n": common_n, "sampling_repeats": len(samples),
                     "n_valid_cka_repeats": len(samples),
                     "status": "dataset_sampling_distribution", "dataset": dataset,
                     "median_cka": float(np.median(samples)),
                     "p10_cka": float(np.quantile(samples, .1)),
                     "p90_cka": float(np.quantile(samples, .9))})
    return [*rows[:1], *detail_rows, *rows[1:]]


def analyze(config: RunConfig, contexts: list[Context], metrics_artifact: Path,
            output: Path) -> dict[str, Any]:
    """Write fixed-condition, group, and repeated-condition comparisons."""
    if not config.primary_condition:
        raise ValueError("Set primary_condition in the config after inspecting inventory")
    primary = config.primary_condition
    metrics_by_context: dict[str, dict[str, list[dict[str, str]]]] = {}
    summary_by_context: dict[str, dict[str, dict[str, str]]] = {}
    for context in contexts:
        folder = metrics_artifact / context.context_id
        layer_rows = read_csv(folder / "layer_metrics.csv")
        condition_rows = read_csv(folder / "condition_summary.csv")
        metrics_by_context[context.context_id] = {"layers": layer_rows}
        summary_by_context[context.context_id] = {r["condition"]: r for r in condition_rows}
    strata: dict[str, list[Context]] = defaultdict(list)
    verified_strata: dict[str, bool] = {}
    for context in contexts:
        key, verified = _stratum(context)
        strata[key].append(context)
        verified_strata[key] = verified
    correlation_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    sampling_rows: list[dict[str, Any]] = []
    core_map, broad_map, mapping_hash = _group_mapping(config)
    if mapping_hash:
        mapped_names = set(core_map) | set(broad_map)
        unknown_names = mapped_names - {context.dataset for context in contexts}
        if unknown_names:
            raise ValueError(f"Group mapping contains datasets outside this cohort: "
                             f"{sorted(unknown_names)}")
    for stratum, cohort in sorted(strata.items()):
        cohort = sorted(cohort, key=lambda c: c.dataset)
        conditions = [primary, *[c for c in config.sensitivity_conditions if c != primary]]
        for condition in conditions:
            if any(condition not in c.conditions for c in cohort):
                correlation_rows.append({"stratum": stratum, "condition": condition,
                                         "status": "incomplete_condition_coverage",
                                         "n_datasets": 0})
                continue
            final_position = len(cohort[0].position_names) - 1
            fixed: list[dict[str, Any]] = []
            for context in cohort:
                fixed.append(fixed_condition_values(
                    context, condition, config.baseline_condition,
                    metrics_by_context[context.context_id]["layers"],
                    summary_by_context[context.context_id]))
            analyses = [
                ("fp_difficulty_vs_gap_deterioration", "difficulty", "gap_deterioration"),
                ("fp_difficulty_vs_inter_distance_change", "difficulty", "inter_distance_change"),
                ("fp_difficulty_vs_intra_distance_change", "difficulty", "intra_distance_change"),
                ("cka_vs_accuracy_ratio", "cka", "accuracy_ratio"),
                ("cka_vs_absolute_accuracy_drop", "cka", "accuracy_drop"),
                ("fp_gap_vs_accuracy_ratio", "fp_gap", "accuracy_ratio"),
                ("fp_gap_vs_absolute_accuracy_drop", "fp_gap", "accuracy_drop"),
                ("gap_deterioration_vs_accuracy_drop", "gap_deterioration", "accuracy_drop"),
            ]
            for name, x_key, y_key in analyses:
                values = [row for row in fixed if np.isfinite(row[x_key]) and np.isfinite(row[y_key])]
                x = np.asarray([row[x_key] for row in values], dtype=float)
                y = np.asarray([row[y_key] for row in values], dtype=float)
                rho = spearman_description(x, y)
                seed = int(hashlib.sha256(f"{config.resampling_seed}/{stratum}/{condition}/{name}".encode()).hexdigest()[:8], 16)
                lo, hi, n_boot, p = (_corr_ci(
                    x, y, config, seed,
                    include_permutation=verified_strata[stratum])
                    if rho is not None else (None, None, 0, None))
                correlation_rows.append({
                    "stratum": stratum, "stratum_metadata_verified": verified_strata[stratum],
                    "condition": condition, "feature_position": cohort[0].position_names[final_position],
                    "analysis": name, "x": x_key, "y": y_key, "n_datasets": len(values),
                    "datasets": ";".join(row["dataset"] for row in values),
                    "excluded_datasets": ";".join(row["dataset"] for row in fixed
                                                  if row not in values),
                    "undefined_reason": ("" if rho is not None else
                                         "insufficient_complete_pairs" if len(values) < 3 else
                                         "constant_metric"),
                    "spearman_rho": rho, "bootstrap_ci95_low": lo,
                    "bootstrap_ci95_high": hi, "valid_bootstrap_repeats": n_boot,
                    **_leave_one_out(x, y),
                    "dataset_permutation_p_value": p, "fdr_q_value": None,
                    "practical_equivalence_bound": config.correlation_equivalence_bound,
                    "bootstrap_ci_within_practical_bound": (
                        None if config.correlation_equivalence_bound is None or
                        lo is None or hi is None else
                        bool(lo >= -config.correlation_equivalence_bound and
                             hi <= config.correlation_equivalence_bound)),
                    "status": "descriptive_only_unknown_context" if not verified_strata[stratum]
                    else ("ok" if rho is not None else "undefined"),
                })
            if condition == primary:
                if core_map:
                    for value in ("cka",):
                        vals = [(row["dataset"], row[value]) for row in fixed]
                        seed = int(hashlib.sha256(f"{config.resampling_seed}/{stratum}/group/{value}".encode()).hexdigest()[:8], 16)
                        group_rows.extend(_group_difference(vals, core_map, config, seed,
                                                            f"{stratum}/core/{value}",
                                                            verified_strata[stratum]))
                        group_rows.extend(_group_difference(vals, broad_map, config, seed + 1,
                                                            f"{stratum}/expanded/{value}",
                                                            verified_strata[stratum]))
                sample_rows = _cka_sampling(cohort, config, primary, stratum)
                sampling_rows.extend(sample_rows if isinstance(sample_rows, list) else [sample_rows])

        # Within each dataset, conditions repeat on one model/checkpoint; summarize separately.
        for context in cohort:
            layers = metrics_by_context[context.context_id]["layers"]
            summaries = summary_by_context[context.context_id]
            geometry_by_condition: dict[str, str] = {}
            for condition in context.conditions:
                row = next((x for x in layers if x["condition"] == condition and
                            int(x["position_id"]) == len(context.position_names) - 1), None)
                if row:
                    geometry_by_condition[condition] = row.get("gap_deterioration", "")
            families: dict[str, list[str]] = defaultdict(list)
            for condition, record in context.conditions.items():
                if condition != config.baseline_condition:
                    families[_condition_family(record["manifest"]["condition"])].append(condition)
            for family, conds in sorted(families.items()):
                pairs = [(c, nullable_number(geometry_by_condition, c),
                          nullable_number(summaries.get(c, {}), "relative_loss")) for c in conds]
                valid = [(c, a, b) for c, a, b in pairs if np.isfinite(a) and np.isfinite(b)]
                a = np.asarray([a for _, a, _ in valid])
                b = np.asarray([b for _, _, b in valid])
                rho = spearman_description(a, b)
                if family == "unknown":
                    rho = None
                correlation_rows.append({
                    "stratum": stratum, "stratum_metadata_verified": verified_strata[stratum],
                    "dataset": context.dataset, "condition": f"family:{family}",
                    "condition_family": family, "conditions": ";".join(c for c, _, _ in valid),
                    "excluded_conditions": ";".join(c for c, a, b in pairs
                                                     if not (np.isfinite(a) and np.isfinite(b))),
                    "analysis": "within_dataset_gap_deterioration_vs_relative_loss",
                    "x": "gap_deterioration", "y": "relative_loss", "n_datasets": 1,
                    "n_conditions": len(valid), "n_conditions_total": len(conds),
                    "spearman_rho": rho,
                    "undefined_reason": ("unknown_condition_family" if family == "unknown" else
                                         "" if rho is not None else
                                         "insufficient_complete_pairs" if len(valid) < 3 else
                                         "constant_metric"),
                    "status": "repeated_conditions_descriptive" if rho is not None else "undefined"})

    _bh(correlation_rows)
    if not group_rows:
        group_rows.append({"status": "group_mapping_not_configured" if not core_map else "no_comparable_groups"})
    if not sampling_rows:
        sampling_rows.append({"status": "not_run_no_comparable_cohort"})
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "comparisons.csv", correlation_rows)
    _write_csv(output / "group_comparisons.csv", group_rows)
    _write_csv(output / "cka_sampling_sensitivity.csv", sampling_rows)
    (output / "comparison_cohorts.json").write_text(json.dumps({
        "strata": {key: {"datasets": [c.dataset for c in group],
                          "context_metadata_verified": verified_strata[key]}
                   for key, group in strata.items()},
        "primary_condition": primary,
        "sensitivity_conditions": list(config.sensitivity_conditions),
        "group_mapping_sha256": mapping_hash,
        "notes": ["Each context is one dataset-checkpoint pair.",
                  "Repeated conditions and positions are not independent datasets.",
                  "Unknown model/head/seed metadata restricts formal interpretation."],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"status": "complete", "strata": len(strata),
            "primary_condition": primary, "group_mapping_sha256": mapping_hash}
