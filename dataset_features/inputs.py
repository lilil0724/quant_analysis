"""Read, inventory, and validate manifest-backed 20004 inference features."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .run_config import RunConfig, canonical_hash


@dataclass(frozen=True)
class Context:
    dataset: str
    context_id: str
    metadata: dict[str, Any]
    conditions: dict[str, dict[str, Any]]
    feature_shape: tuple[int, int, int]
    position_names: tuple[str, ...]
    source_hashes: dict[str, str]


@dataclass(frozen=True)
class Inventory:
    contexts: tuple[Context, ...]
    conditions: tuple[str, ...]
    raw_hashes: dict[str, str]
    errors: tuple[str, ...]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _manifest_identity(metadata: dict[str, Any], condition: dict[str, Any]) -> str:
    identity_data = {
        "schema": metadata["schema"], "checkpoint": metadata["checkpoint_sha256"],
        "data": metadata["data_sha256"], "code": metadata["code_sha256"],
        "hqq": metadata["hqq_commit"], "dtype": metadata["feature_dtype"],
        "execution": metadata["execution"], "condition": condition,
    }
    return hashlib.sha256(json.dumps(
        identity_data, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")).hexdigest()


def _contained_path(root: Path, path: Path, label: str,
                    parent: Path | None = None) -> Path:
    """Resolve a source path and reject symlinks or traversal outside its declared tree."""
    resolved = path.resolve()
    root_resolved = root.resolve()
    if not resolved.is_relative_to(root_resolved):
        raise ValueError(f"{label} escapes input_root: {path}")
    if parent is not None and not resolved.is_relative_to(parent.resolve()):
        raise ValueError(f"{label} escapes its declared directory: {path}")
    return resolved


def discover(config: RunConfig) -> Inventory:
    """Read dataset metadata and array headers to build a source inventory."""
    root = config.input_root
    if not root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {root}")
    contexts: list[Context] = []
    raw_hashes: dict[str, str] = {}
    errors: list[str] = []
    seen_datasets: set[str] = set()
    observed_conditions: set[str] = set()
    for metadata_path in sorted(root.glob("*/dataset.json")):
        folder = metadata_path.parent
        dataset = folder.name
        if dataset in seen_datasets:
            errors.append(f"Duplicate dataset directory: {dataset}")
            continue
        seen_datasets.add(dataset)
        try:
            metadata_file = _contained_path(root, metadata_path,
                                            "dataset metadata")
            dataset_folder = _contained_path(root, folder, "dataset directory")
            if dataset_folder.parent != root.resolve():
                raise ValueError(f"dataset directory is not a direct child of input_root: {folder}")
            metadata = _json(metadata_file)
            if metadata.get("schema") != 1:
                raise ValueError("tgda_manifest_v1 currently accepts only exporter schema 1")
            if metadata.get("dataset") != dataset:
                raise ValueError(f"dataset.json dataset={metadata.get('dataset')!r} "
                                 "does not match its directory")
            meta_rel = metadata_path.relative_to(root).as_posix()
            raw_hashes[meta_rel] = sha256(metadata_file)
            n = metadata["n_samples"]
            sample_ids = metadata["sample_ids"]
            targets = metadata["targets"]
            if (type(n) is not int or n <= 1 or
                    not isinstance(sample_ids, list) or not isinstance(targets, list) or
                    len(sample_ids) != n or len(targets) != n or
                    any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids) or
                    len(set(sample_ids)) != n or
                    any(type(target) is not int or target < 0 for target in targets)):
                raise ValueError("invalid sample count/ID/target types, duplicate IDs, or lengths")
            condition_records: dict[str, dict[str, Any]] = {}
            dataset_hashes = {meta_rel: raw_hashes[meta_rel]}
            for manifest_path in sorted(folder.glob("*/manifest.json")):
                condition_dir = manifest_path.parent
                if (condition_dir.name == "exemplars" or condition_dir.name.startswith(".") or
                        ".invalid." in condition_dir.name):
                    continue
                condition_folder = _contained_path(root, condition_dir,
                                                   "condition directory", dataset_folder)
                if condition_folder.parent != dataset_folder:
                    raise ValueError(f"condition directory is not a direct child of its dataset: "
                                     f"{condition_dir}")
                manifest_file = _contained_path(root, manifest_path,
                                                "condition manifest", condition_folder)
                manifest_rel = manifest_path.relative_to(root).as_posix()
                raw_hashes[manifest_rel] = sha256(manifest_file)
                dataset_hashes[manifest_rel] = raw_hashes[manifest_rel]
                manifest = _json(manifest_file)
                condition_spec = manifest.get("condition")
                if not isinstance(condition_spec, dict):
                    raise ValueError("manifest condition must be an object")
                name = str(condition_spec.get("name", condition_dir.name))
                if name != condition_dir.name or name in condition_records:
                    raise ValueError(f"condition directory/name mismatch or duplicate: {name}")
                observed_conditions.add(name)
                condition_records[name] = {"path": str(condition_folder),
                                           "manifest": manifest,
                                           "relative_manifest": manifest_rel}
            if config.baseline_condition not in condition_records:
                raise ValueError(f"missing baseline condition {config.baseline_condition!r}")
            if config.conditions is not None:
                absent = sorted(set(config.conditions) - condition_records.keys())
                if absent:
                    raise ValueError(f"missing configured conditions: {absent}")
                condition_records = {n: condition_records[n] for n in config.conditions}
            feature_path = Path(condition_records[config.baseline_condition]["path"]) / "features.npy"
            feature_file = _contained_path(root, feature_path, "features.npy",
                                           feature_path.parent)
            features = np.load(feature_file, mmap_mode="r", allow_pickle=False)
            if features.ndim != 3 or features.shape[0] != n or features.shape[2] < 2:
                raise ValueError(f"expected feature array [N,L,D], found {features.shape}")
            if not np.issubdtype(features.dtype, np.floating):
                raise ValueError(f"feature dtype must be floating point, found {features.dtype}")
            declared_shape = metadata.get("feature_shape")
            if declared_shape is not None:
                if (not isinstance(declared_shape, (list, tuple)) or
                        len(declared_shape) != 2 or
                        any(type(x) is not int or x <= 0 for x in declared_shape)):
                    raise ValueError("metadata feature_shape must be [positions, width]")
                declared_shape = tuple(declared_shape)
            if config.feature_shape and declared_shape and config.feature_shape != declared_shape:
                raise ValueError("config feature_shape conflicts with exporter metadata")
            expected_shape = config.feature_shape or declared_shape
            if expected_shape is None:
                raise ValueError("feature_shape is not declared by trusted metadata or config")
            if tuple(features.shape[1:]) != tuple(expected_shape):
                raise ValueError(f"feature positions/width {features.shape[1:]} do not match "
                                 f"declared feature_shape {expected_shape}")
            if str(features.dtype) != str(metadata.get("feature_dtype")):
                raise ValueError("feature array dtype differs from dataset.json feature_dtype")
            declared_positions = metadata.get("position_names")
            if (config.position_names and declared_positions is not None and
                    list(config.position_names) != declared_positions):
                raise ValueError("config position_names conflicts with exporter metadata")
            positions = config.position_names or declared_positions
            if positions is None:
                raise ValueError("position_names must be declared in metadata or config; "
                                 "schema/hash presence does not establish exporter semantics")
            if (not isinstance(positions, (list, tuple)) or
                    any(not isinstance(x, str) or not x for x in positions) or
                    len(set(positions)) != len(positions)):
                raise ValueError("position_names must be a unique ordered list of names")
            if len(positions) != features.shape[1]:
                raise ValueError(f"position_names count {len(positions)} != {features.shape[1]}")
            identity = {
                "dataset": dataset,
                "model": metadata.get("model", metadata.get("model_name", "unknown")),
                "head": metadata.get("head", metadata.get("head_name", "unknown")),
                "checkpoint_sha256": metadata.get("checkpoint_sha256"),
                "training_seed": metadata.get("training_seed", metadata.get("seed")),
                "data_sha256": metadata.get("data_sha256"),
                "preprocess": metadata.get("preprocess"),
                "execution": metadata.get("execution"),
                "feature_shape": list(features.shape[1:]),
                "conditions": sorted(condition_records),
            }
            context_id = f"{dataset}-{canonical_hash(identity)[:12]}"
            contexts.append(Context(dataset, context_id, metadata, condition_records,
                                    tuple(map(int, features.shape)), tuple(positions),
                                    dataset_hashes))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"{dataset}: {exc}")
    if len(contexts) != config.expected_dataset_count:
        errors.append(f"expected {config.expected_dataset_count} valid dataset inventories, "
                      f"found {len(contexts)}")
    conditions = config.conditions or tuple(sorted(observed_conditions))
    if len(conditions) != len(set(conditions)):
        errors.append("duplicate conditions in resolved inventory")
    if config.primary_condition and config.primary_condition not in conditions:
        errors.append(f"primary condition {config.primary_condition!r} is not present")
    for condition in config.sensitivity_conditions:
        if condition not in conditions:
            errors.append(f"sensitivity condition {condition!r} is not present")
    return Inventory(tuple(contexts), tuple(conditions), raw_hashes, tuple(errors))


def validate(config: RunConfig, inventory: Inventory) -> list[dict[str, Any]]:
    """Check all source manifests, checksums, predictions, and feature values."""
    errors = list(inventory.errors)
    coverage: list[dict[str, Any]] = []
    expected_conditions = set(inventory.conditions)
    position_maps = {context.position_names for context in inventory.contexts}
    if len(position_maps) > 1:
        errors.append("contexts do not share one ordered feature-position map; "
                      "separate incompatible model architectures before cross-dataset analysis")
    for context in inventory.contexts:
        metadata = context.metadata
        n, positions, width = context.feature_shape
        sample_ids = list(map(str, metadata["sample_ids"]))
        target_by_id = dict(zip(sample_ids, map(int, metadata["targets"])))
        expected_dtype = str(metadata["feature_dtype"])
        present = set(context.conditions)
        if present != expected_conditions:
            errors.append(f"{context.dataset}: condition coverage {sorted(present)} != "
                          f"{sorted(expected_conditions)}")
        for condition, record in context.conditions.items():
            directory = Path(record["path"])
            manifest = record["manifest"]
            label = f"{context.dataset}/{condition}"
            try:
                directory = _contained_path(config.input_root, directory,
                                            f"{label} condition directory")
                if manifest.get("status") != "complete":
                    raise ValueError("manifest status is not complete")
                if manifest.get("schema") != metadata.get("schema"):
                    raise ValueError("manifest schema differs from dataset.json")
                if manifest.get("dataset") != context.dataset or manifest.get("n_samples") != n:
                    raise ValueError("manifest dataset/sample count does not match dataset.json")
                spec = manifest["condition"]
                if spec.get("name") != condition or manifest.get("identity") != _manifest_identity(metadata, spec):
                    raise ValueError("condition identity does not match dataset metadata")
                declared_files = manifest.get("files")
                required = {"features.npy", "predictions.csv"}
                if not isinstance(declared_files, dict) or not required.issubset(declared_files):
                    raise ValueError("manifest must checksum features.npy and predictions.csv")
                checked_files: dict[str, Path] = {}
                for name, expected in declared_files.items():
                    if not isinstance(name, str) or Path(name).name != name:
                        raise ValueError(f"manifest file key must be a filename: {name!r}")
                    path = _contained_path(config.input_root, directory / name,
                                           f"{label}/{name}", directory)
                    if not path.is_file() or sha256(path) != expected:
                        raise ValueError(f"missing or checksum mismatch: {name}")
                    checked_files[name] = path
                    inventory.raw_hashes[path.relative_to(config.input_root).as_posix()] = expected
                feature = np.load(checked_files["features.npy"], mmap_mode="r",
                                  allow_pickle=False)
                if tuple(feature.shape) != context.feature_shape or str(feature.dtype) != expected_dtype:
                    raise ValueError(f"feature shape/dtype mismatch: {feature.shape}/{feature.dtype}")
                for start in range(0, n, config.chunk_size):
                    chunk = np.asarray(feature[start:start + config.chunk_size], dtype=np.float32)
                    if not np.isfinite(chunk).all():
                        raise ValueError(f"nonfinite features in rows {start}:"
                                         f"{min(start + config.chunk_size, n)}")
                rows = read_csv(checked_files["predictions.csv"])
                if len(rows) != n:
                    raise ValueError(f"prediction row count {len(rows)} != {n}")
                by_id: dict[str, dict[str, str]] = {}
                for row in rows:
                    sid = row.get("sample_id", "")
                    if not sid or sid in by_id:
                        raise ValueError("predictions.csv has missing or duplicate sample_id")
                    by_id[sid] = row
                if set(by_id) != set(sample_ids):
                    raise ValueError("prediction sample IDs differ from dataset.json")
                for sid, row in by_id.items():
                    target, pred = int(row["target"]), int(row["prediction"])
                    correct = int(row["correct"])
                    if target != target_by_id[sid] or correct != int(pred == target):
                        raise ValueError(f"prediction target/correct mismatch for sample {sid}")
                feature_indices = _feature_row_indices(context, condition, rows)
                if "precision_probe.npy" in declared_files:
                    probe = np.load(checked_files["precision_probe.npy"], mmap_mode="r",
                                    allow_pickle=False)
                    if probe.shape != context.feature_shape[1:] and not (
                            probe.ndim == 3 and probe.shape[1:] == context.feature_shape[1:]):
                        raise ValueError(f"precision probe has incompatible shape {probe.shape}")
                    probe_rows = (1 if probe.ndim == 2 else len(probe))
                    for start in range(0, probe_rows, config.chunk_size):
                        if not np.isfinite(probe[start:start + config.chunk_size]).all():
                            raise ValueError("precision probe contains non-finite features")
                coverage.append({"dataset": context.dataset, "context_id": context.context_id,
                                 "condition": condition, "n_samples": n,
                                 "n_positions": positions, "feature_width": width,
                                 "dtype": expected_dtype, "n_predictions": len(rows),
                                 "prediction_rows_reordered": [r["sample_id"] for r in rows] != sample_ids,
                                 "feature_rows_reordered": not np.array_equal(feature_indices, np.arange(n)),
                                 "precision_probe": "precision_probe.npy" in declared_files,
                                 "status": "complete"})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append(f"{label}: {exc}")
                coverage.append({"dataset": context.dataset, "context_id": context.context_id,
                                 "condition": condition, "status": "invalid", "error": str(exc)})
    if errors:
        raise ValueError("Input validation failed:\n- " + "\n- ".join(errors))
    return coverage


def predictions_for(context: Context, condition: str) -> list[dict[str, Any]]:
    """Return prediction rows in dataset.json sample order."""
    path = Path(context.conditions[condition]["path"]) / "predictions.csv"
    rows = read_csv(path)
    by_id = {row["sample_id"]: row for row in rows}
    return [by_id[str(sample_id)] for sample_id in context.metadata["sample_ids"]]


def _feature_row_indices(context: Context, condition: str,
                         predictions: list[dict[str, Any]] | None = None) -> np.ndarray:
    """Map canonical IDs to physical feature rows, whose default order follows CSV."""
    manifest = context.conditions[condition]["manifest"]
    physical_ids = manifest.get("feature_sample_ids")
    if physical_ids is None:
        rows = predictions if predictions is not None else read_csv(
            Path(context.conditions[condition]["path"]) / "predictions.csv")
        physical_ids = [row["sample_id"] for row in rows]
    canonical_ids = context.metadata["sample_ids"]
    if (not isinstance(physical_ids, list) or
            any(not isinstance(sid, str) for sid in physical_ids) or
            len(physical_ids) != len(canonical_ids) or
            len(set(physical_ids)) != len(physical_ids) or
            set(physical_ids) != set(canonical_ids)):
        raise ValueError("feature sample IDs must uniquely match dataset.json")
    by_id = {sid: index for index, sid in enumerate(physical_ids)}
    return np.asarray([by_id[sid] for sid in canonical_ids], dtype=np.int64)


class RowAlignedArray:
    """Lazy row selection over an mmap; layer/slice selection never copies all features."""

    def __init__(self, array: np.ndarray, row_indices: np.ndarray):
        self._array = array
        self._row_indices = row_indices
        self.shape = (len(row_indices), *array.shape[1:])
        self.ndim = array.ndim
        self.dtype = array.dtype

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key: Any) -> Any:
        keys = key if isinstance(key, tuple) else (key,)
        first, *rest = keys
        # Basic tail indexing is a view of the underlying mmap. Materialize only
        # the requested rows, or retain a lazy view for a subsequent chunk slice.
        array = self._array[(slice(None), *rest)]
        indices = self._row_indices[first]
        if isinstance(first, slice):
            return RowAlignedArray(array, indices)
        return array[indices]

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        selected = np.asarray(self._array[self._row_indices], dtype=dtype)
        return selected.copy() if copy else selected


def open_features(context: Context, condition: str) -> np.ndarray | RowAlignedArray:
    """Open validated features lazily in dataset.json sample order."""
    path = Path(context.conditions[condition]["path"]) / "features.npy"
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    indices = _feature_row_indices(context, condition)
    if np.array_equal(indices, np.arange(len(array))):
        return array
    return RowAlignedArray(array, indices)
