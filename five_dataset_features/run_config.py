"""Configuration and stable context identity for the 20004 analysis."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


CONFIG_SCHEMA = 1


def canonical_hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class RunConfig:
    config_path: Path
    schema_version: int
    run_id: str
    input_root: Path
    output_root: Path
    expected_dataset_count: int
    source_format: str
    baseline_condition: str
    primary_condition: str | None
    sensitivity_conditions: tuple[str, ...]
    conditions: tuple[str, ...] | None
    group_mapping_file: Path | None
    chunk_size: int
    resampling_seed: int
    bootstrap_repeats: int
    permutation_repeats: int
    cka_sampling_repeats: int
    cka_sampling_n_max: int
    correlation_equivalence_bound: float | None
    division_epsilon: float
    feature_shape: tuple[int, int] | None
    position_names: tuple[str, ...] | None

    @property
    def identity(self) -> str:
        return canonical_hash({
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "input_root": str(self.input_root),
            "output_root": str(self.output_root),
            "source_format": self.source_format,
            "expected_dataset_count": self.expected_dataset_count,
            "baseline_condition": self.baseline_condition,
            "primary_condition": self.primary_condition,
            "sensitivity_conditions": self.sensitivity_conditions,
            "conditions": self.conditions,
            "group_mapping_file": (str(self.group_mapping_file)
                                   if self.group_mapping_file else None),
            "chunk_size": self.chunk_size,
            "resampling_seed": self.resampling_seed,
            "bootstrap_repeats": self.bootstrap_repeats,
            "permutation_repeats": self.permutation_repeats,
            "cka_sampling_repeats": self.cka_sampling_repeats,
            "cka_sampling_n_max": self.cka_sampling_n_max,
            "correlation_equivalence_bound": self.correlation_equivalence_bound,
            "division_epsilon": self.division_epsilon,
            "feature_shape": self.feature_shape,
            "position_names": self.position_names,
        })

    @property
    def cohort_identity(self) -> str:
        """Identity of the source cohort and feature layout, excluding analysis choices."""
        return canonical_hash({
            "schema_version": self.schema_version,
            "input_root": str(self.input_root),
            "source_format": self.source_format,
            "expected_dataset_count": self.expected_dataset_count,
            "baseline_condition": self.baseline_condition,
            "conditions": self.conditions,
            "feature_shape": self.feature_shape,
            "position_names": self.position_names,
        })

    def stage_identity(self, stage: str) -> str:
        """Hash only settings that can change the requested stage's output."""
        cohort = self.cohort_identity
        if stage in {"inventory", "validate"}:
            settings: object = {"cohort": cohort}
        elif stage == "metrics":
            settings = {"cohort": cohort, "chunk_size": self.chunk_size,
                        "division_epsilon": self.division_epsilon}
        elif stage == "pca":
            settings = {"cohort": cohort, "chunk_size": self.chunk_size}
        elif stage == "comparisons":
            settings = {
                "cohort": cohort, "primary_condition": self.primary_condition,
                "sensitivity_conditions": self.sensitivity_conditions,
                "group_mapping_file": (str(self.group_mapping_file)
                                        if self.group_mapping_file else None),
                "resampling_seed": self.resampling_seed,
                "bootstrap_repeats": self.bootstrap_repeats,
                "permutation_repeats": self.permutation_repeats,
                "cka_sampling_repeats": self.cka_sampling_repeats,
                "cka_sampling_n_max": self.cka_sampling_n_max,
                "correlation_equivalence_bound": self.correlation_equivalence_bound,
                "chunk_size": self.chunk_size,
            }
        elif stage == "plot":
            settings = {"cohort": cohort, "run_id": self.run_id,
                        "baseline_condition": self.baseline_condition,
                        "primary_condition": self.primary_condition,
                        "sensitivity_conditions": self.sensitivity_conditions}
        else:
            settings = {"full_config": self.identity}
        return canonical_hash(settings)


def load_config(path: str | Path) -> RunConfig:
    """Load paths relative to the config and reject ambiguous settings."""
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError(f"Unsupported config schema: {config_path}")
    base = config_path.parent.parent.parent

    def resolve(value: str | None, default: Path | None = None) -> Path | None:
        if value is None:
            return default
        result = Path(value)
        return (base / result).resolve() if not result.is_absolute() else result.resolve()

    input_root = resolve(raw.get("input_root"), base / "data" / "20004")
    output_root = resolve(raw.get("output_root"), base / "five_dataset_features" /
                          "results" / "20004")
    assert input_root is not None and output_root is not None
    if input_root == output_root or input_root in output_root.parents or output_root in input_root.parents:
        raise ValueError("Input and output roots must be separate trees")
    source_format = raw.get("source_format", "tgda_manifest_v1")
    if source_format != "tgda_manifest_v1":
        raise ValueError(f"Unsupported source_format {source_format!r}; "
                         "inspect the input and add a verified adapter first")
    conditions = raw.get("conditions")
    if conditions is not None:
        if (not isinstance(conditions, list) or
                any(not isinstance(x, str) or not x for x in conditions)):
            raise ValueError("conditions must be null or a list of non-empty names")
        if len(set(conditions)) != len(conditions):
            raise ValueError("conditions must not contain duplicate names")
    baseline = raw.get("baseline_condition", "fp")
    if not isinstance(baseline, str) or not baseline:
        raise ValueError("baseline_condition must be a non-empty string")
    if conditions is not None and baseline not in conditions:
        raise ValueError("conditions must include baseline_condition")
    sensitivities = raw.get("sensitivity_conditions", [])
    if (not isinstance(sensitivities, list) or
            any(not isinstance(x, str) or not x for x in sensitivities)):
        raise ValueError("sensitivity_conditions must be a list of non-empty names")
    if len(set(sensitivities)) != len(sensitivities):
        raise ValueError("sensitivity_conditions must not contain duplicate names")
    primary = raw.get("primary_condition")
    if primary is not None and (not isinstance(primary, str) or not primary):
        raise ValueError("primary_condition must be a non-empty string or null")
    if primary == baseline:
        raise ValueError("primary_condition must differ from baseline_condition")
    if primary and conditions is not None and primary not in conditions:
        raise ValueError("conditions must include primary_condition")
    if baseline in sensitivities:
        raise ValueError("sensitivity_conditions must not include baseline_condition")
    if conditions is not None and not set(sensitivities).issubset(conditions):
        raise ValueError("conditions must include every sensitivity_condition")
    shape = raw.get("feature_shape")
    if shape is not None and (not isinstance(shape, list) or len(shape) != 2 or
                              any(type(x) is not int or x <= 0 for x in shape)):
        raise ValueError("feature_shape must be null or [positions, width]")
    positions = raw.get("position_names")
    if positions is not None:
        if (not isinstance(positions, list) or not positions or
                any(not isinstance(x, str) or not x for x in positions)):
            raise ValueError("position_names must be null or a non-empty list of names")
        if len(set(positions)) != len(positions):
            raise ValueError("position_names must not contain duplicate names")
    if shape and positions and len(positions) != shape[0]:
        raise ValueError("position_names length must match feature_shape[0]")
    group_mapping = resolve(raw.get("group_mapping_file"))
    numeric = {
        "expected_dataset_count": (int, 16), "chunk_size": (int, 1024),
        "resampling_seed": (int, 100), "bootstrap_repeats": (int, 2000),
        "permutation_repeats": (int, 9999),
        "cka_sampling_repeats": (int, 30), "cka_sampling_n_max": (int, 512),
    }
    values = {}
    for name, (kind, default) in numeric.items():
        value = raw.get(name, default)
        if type(value) is not kind or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        values[name] = value
    epsilon = raw.get("division_epsilon", 1e-8)
    if not isinstance(epsilon, (int, float)) or not 0 < epsilon < 1:
        raise ValueError("division_epsilon must be between 0 and 1")
    bound = raw.get("correlation_equivalence_bound")
    if bound is not None and (not isinstance(bound, (int, float)) or not 0 < bound < 1):
        raise ValueError("correlation_equivalence_bound must be null or between 0 and 1")
    run_id = raw.get("run_id", "20004")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    return RunConfig(
        config_path=config_path, schema_version=CONFIG_SCHEMA,
        run_id=run_id, input_root=input_root,
        output_root=output_root, expected_dataset_count=values["expected_dataset_count"],
        source_format=source_format, baseline_condition=baseline,
        primary_condition=primary, sensitivity_conditions=tuple(sensitivities),
        conditions=tuple(conditions) if conditions is not None else None,
        group_mapping_file=group_mapping, chunk_size=values["chunk_size"],
        resampling_seed=values["resampling_seed"],
        bootstrap_repeats=values["bootstrap_repeats"],
        permutation_repeats=values["permutation_repeats"],
        cka_sampling_repeats=values["cka_sampling_repeats"],
        cka_sampling_n_max=values["cka_sampling_n_max"],
        correlation_equivalence_bound=float(bound) if bound is not None else None,
        division_epsilon=float(epsilon),
        feature_shape=tuple(shape) if shape is not None else None,
        position_names=tuple(positions) if positions is not None else None,
    )
