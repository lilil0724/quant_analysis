"""Config-driven, resumable full-feature analysis for inference run 20004.

All Python invocations use the repository's ``opencode_env`` as documented in
AGENTS.md. This workflow consumes completed local inference exports only.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .comparisons import analyze as analyze_comparisons
from .inputs import Context, Inventory, discover, read_csv, validate
from .layer_analysis import analyze_context as analyze_metrics, write_json, write_table
from .layer_pca import analyze_context as analyze_pca
from .plot_run import plot
from .report_run import generate as generate_report
from .run_config import RunConfig, canonical_hash, load_config
from .run_snapshot import (SnapshotStore, atomic_json, file_hash, runtime_identity,
                           stage_fingerprint, tree_hash)


STAGES = ("inventory", "validate", "metrics", "pca", "comparisons",
          "plot", "report", "verify", "all")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _manifest_raw_identity(inventory: Inventory) -> dict[str, str]:
    return dict(sorted(inventory.raw_hashes.items()))


def _context_identity(context: Context, raw_hashes: dict[str, str]) -> dict[str, str]:
    prefix = context.dataset + "/"
    return {path: digest for path, digest in sorted(raw_hashes.items())
            if path.startswith(prefix)}


def _contexts(inventory: Inventory, dataset: str | None = None) -> list[Context]:
    values = [context for context in inventory.contexts
              if dataset is None or context.dataset == dataset]
    if dataset and not values:
        raise ValueError(f"Unknown dataset {dataset!r}; run inventory and check its names")
    return sorted(values, key=lambda context: (context.dataset, context.context_id))


def _inventory_artifact(config: RunConfig, inventory: Inventory, store: SnapshotStore,
                        rebuild: bool = False) -> Path:
    identity = _manifest_raw_identity(inventory)
    fingerprint = stage_fingerprint("inventory", config, input_identity=identity)
    scope = "run"
    if not rebuild and store.is_current("inventory", scope, fingerprint):
        artifact = store.current_artifact("inventory", scope)
        assert artifact is not None
        return artifact
    temp = store.build_dir("inventory", scope, fingerprint)
    temp.mkdir(parents=True, exist_ok=False)
    dataset_rows = []
    condition_rows = []
    for context in inventory.contexts:
        dataset_rows.append({"dataset": context.dataset, "context_id": context.context_id,
                             "n_samples": context.feature_shape[0],
                             "n_positions": context.feature_shape[1],
                             "feature_width": context.feature_shape[2],
                             "dtype": context.metadata.get("feature_dtype"),
                             "conditions": ";".join(context.conditions),
                             "position_names": ";".join(context.position_names)})
        for condition, record in context.conditions.items():
            condition_rows.append({"dataset": context.dataset, "context_id": context.context_id,
                                   "condition": condition,
                                   "manifest_sha256": inventory.raw_hashes.get(
                                       record["relative_manifest"], "")})
    _write_csv(temp / "datasets.csv", dataset_rows)
    _write_csv(temp / "conditions.csv", condition_rows)
    write_json(temp / "inventory.json", {
                                          "cohort_identity": config.cohort_identity,
                                          "expected_dataset_count": config.expected_dataset_count,
                                          "conditions": list(inventory.conditions),
                                          "contexts": [c.context_id for c in inventory.contexts],
                                          "raw_manifest_hashes": identity,
                                          "discovery_errors": list(inventory.errors)})
    return store.publish("inventory", scope, fingerprint, temp,
                         {"n_contexts": len(inventory.contexts)})


def _validation_artifact(config: RunConfig, inventory: Inventory, store: SnapshotStore,
                         rebuild: bool = False) -> tuple[Path, dict[str, Any]]:
    inventory_artifact = _inventory_artifact(config, inventory, store, rebuild=False)
    inventory_receipt = store.receipt_path("inventory", "run")
    fingerprint = stage_fingerprint(
        "validate", config, input_identity=_manifest_raw_identity(inventory),
        upstream={"inventory_receipt": file_hash(inventory_receipt)})
    temp = store.build_dir("validate", "run", fingerprint)
    temp.mkdir(parents=True, exist_ok=False)
    try:
        coverage = validate(config, inventory)
        context_raw_hashes = {c.context_id: _context_identity(c, inventory.raw_hashes)
                              for c in inventory.contexts}
        value = {"status": "complete", "cohort_identity": config.cohort_identity,
                 "inventory_artifact": str(inventory_artifact.relative_to(config.output_root)),
                 "n_contexts": len(inventory.contexts), "n_raw_files": len(inventory.raw_hashes),
                 "raw_hashes": dict(sorted(inventory.raw_hashes.items())),
                 "context_raw_hashes": context_raw_hashes,
                 "coverage": coverage,
                 "precision_probe_available": any(row["precision_probe"] for row in coverage)}
        write_json(temp / "validation.json", value)
        _write_csv(temp / "coverage.csv", coverage)
        artifact = store.publish("validate", "run", fingerprint, temp,
                                 {"n_contexts": len(inventory.contexts),
                                  "n_raw_files": len(inventory.raw_hashes)})
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    lock = {"cohort_identity": config.cohort_identity,
            "context_ids": sorted(c.context_id for c in inventory.contexts),
            "conditions": list(inventory.conditions),
            "positions": list(inventory.contexts[0].position_names)
            if inventory.contexts else [],
            "status": "validated", "validation_receipt_sha256":
            file_hash(store.receipt_path("validate", "run"))}
    atomic_json(store.state / "cohort_lock.json", lock)
    pending = store.state / "cohort_lock.pending.json"
    if pending.is_file():
        pending.unlink()
    return artifact, value


def _load_validation(config: RunConfig, store: SnapshotStore) -> tuple[Path, dict[str, Any]]:
    artifact = store.current_artifact("validate", "run")
    if artifact is None:
        raise RuntimeError("No current validation receipt. Run --stage validate first.")
    value = json.loads((artifact / "validation.json").read_text(encoding="utf-8"))
    if value.get("cohort_identity") != config.cohort_identity:
        raise RuntimeError("Source cohort or feature layout differs from validation; "
                           "run --stage validate again.")
    inventory_artifact = store.current_artifact("inventory", "run")
    if inventory_artifact is None or value.get("inventory_artifact") != str(
            inventory_artifact.relative_to(config.output_root)):
        raise RuntimeError("Validation receipt belongs to an older inventory; run --stage validate.")
    inventory_info = json.loads((inventory_artifact / "inventory.json").read_text(encoding="utf-8"))
    expected = stage_fingerprint(
        "validate", config, input_identity=inventory_info["raw_manifest_hashes"],
        upstream={"inventory_receipt": file_hash(store.receipt_path("inventory", "run"))})
    if not store.is_current("validate", "run", expected):
        raise RuntimeError("Validation stage code or inputs changed; run --stage validate again.")
    lock_path = store.state / "cohort_lock.json"
    if not lock_path.is_file():
        raise RuntimeError("Validated cohort lock is missing; rerun --stage validate.")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (lock.get("status") != "validated" or
            lock.get("validation_receipt_sha256") != file_hash(store.receipt_path("validate", "run"))):
        raise RuntimeError("Cohort lock does not match validation receipt; rerun --stage validate.")
    return artifact, value


def _verify_validated_sources(config: RunConfig, validation: dict[str, Any],
                             dataset: str | None = None) -> None:
    """Fail closed when raw bytes have changed since the successful validation stage."""
    errors = []
    root = config.input_root.resolve()
    for relative, expected in sorted(validation["raw_hashes"].items()):
        if dataset and Path(relative).parts[0] != dataset:
            continue
        path = (config.input_root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            errors.append(f"missing or escaped input path: {relative}")
        elif file_hash(path) != expected:
            errors.append(f"raw input changed since validation: {relative}")
    if errors:
        raise ValueError("Validated source check failed; rerun --stage validate:\n- " +
                         "\n- ".join(errors))


def _context_stage(config: RunConfig, contexts: list[Context], validation: dict[str, Any],
                   stage: str, store: SnapshotStore, rebuild: bool) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    expected = {c.context_id for c in contexts}
    if expected - validation["context_raw_hashes"].keys():
        raise RuntimeError("Validation receipt does not cover the requested contexts")
    for context in contexts:
        input_identity = validation["context_raw_hashes"][context.context_id]
        fingerprint = stage_fingerprint(stage, config, context_id=context.context_id,
                                        input_identity=input_identity,
                                        upstream={"validation": canonical_hash(input_identity)})
        if not rebuild and store.is_current(stage, context.context_id, fingerprint):
            artifact = store.current_artifact(stage, context.context_id)
            assert artifact is not None
            outputs[context.context_id] = artifact
            print(f"REUSE {stage} {context.dataset}", flush=True)
            continue
        temp = store.build_dir(stage, context.context_id, fingerprint)
        temp.mkdir(parents=True, exist_ok=False)
        try:
            if stage == "metrics":
                result = analyze_metrics(context, config, temp)
            elif stage == "pca":
                result = analyze_pca(context, config, temp)
            else:
                raise ValueError(f"Unknown context stage {stage}")
            write_json(temp / "receipt.json", {**result, "fingerprint": fingerprint,
                                               "input_identity": input_identity})
            artifact = store.publish(stage, context.context_id, fingerprint, temp, result)
        except Exception:
            shutil.rmtree(temp, ignore_errors=True)
            raise
        outputs[context.context_id] = artifact
    if set(outputs) != expected:
        raise RuntimeError(f"{stage} completed partial coverage")
    return outputs


def _write_aggregate(stage: str, config: RunConfig, contexts: list[Context],
                     context_artifacts: dict[str, Path], store: SnapshotStore,
                     filenames: tuple[str, ...], rebuild: bool = False,
                     extra_identity: Any = None) -> Path:
    receipt_ids = {c.context_id: file_hash(store.receipt_path(stage, c.context_id))
                   for c in contexts}
    fingerprint = stage_fingerprint(stage, config, input_identity=receipt_ids,
                                    upstream=receipt_ids, extra_identity=extra_identity)
    scope = "run"
    if not rebuild and store.is_current(stage, scope, fingerprint):
        artifact = store.current_artifact(stage, scope)
        assert artifact is not None
        return artifact
    temp = store.build_dir(stage, scope, fingerprint)
    temp.mkdir(parents=True, exist_ok=False)
    try:
        index = {}
        for context in contexts:
            source = context_artifacts[context.context_id]
            target = temp / context.context_id
            target.mkdir()
            index[context.context_id] = {"dataset": context.dataset,
                                         "receipt_sha256": receipt_ids[context.context_id]}
            for filename in filenames:
                shutil.copyfile(source / filename, target / filename)
        write_json(temp / "context_index.json", index)
        return store.publish(stage, scope, fingerprint, temp,
                             {"n_contexts": len(contexts), "extra_identity": extra_identity})
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def _run_context_stage(config: RunConfig, inventory: Inventory, validation: dict[str, Any],
                       stage: str, store: SnapshotStore, dataset: str | None,
                       rebuild: bool) -> tuple[dict[str, Path], Path | None]:
    all_contexts = _contexts(inventory)
    selected = _contexts(inventory, dataset)
    outputs = _context_stage(config, selected, validation, stage, store, rebuild)
    if len(selected) != len(all_contexts):
        print(f"{stage}: completed {len(selected)} selected context(s); aggregate remains incomplete")
        return outputs, None
    filenames = ("condition_summary.csv", "layer_metrics.csv", "layer_drop_summary.csv",
                 "metric_status.csv",
                 "precision_probe.csv", "context.json") if stage == "metrics" else ("pca_summary.csv",)
    aggregate = _write_aggregate(stage, config, all_contexts, outputs, store,
                                 filenames, rebuild)
    return outputs, aggregate


def _global_stage(config: RunConfig, stage: str, store: SnapshotStore,
                  upstream: dict[str, Path], builder: Callable[[Path], dict[str, Any]],
                  extra_identity: Any = None, rebuild: bool = False) -> Path:
    upstream_hashes = {name: file_hash(store.receipt_path(name, "run"))
                       for name in upstream}
    fingerprint = stage_fingerprint(stage, config, upstream=upstream_hashes,
                                    extra_identity=extra_identity)
    scope = "run"
    if not rebuild and store.is_current(stage, scope, fingerprint):
        artifact = store.current_artifact(stage, scope)
        assert artifact is not None
        print(f"REUSE {stage}", flush=True)
        return artifact
    temp = store.build_dir(stage, scope, fingerprint)
    temp.mkdir(parents=True, exist_ok=False)
    try:
        result = builder(temp)
        write_json(temp / "receipt.json", {**result, "fingerprint": fingerprint,
                                          "upstream_receipts": upstream_hashes})
        artifact = store.publish(stage, scope, fingerprint, temp, result)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    print(f"COMPLETE {stage}: {artifact}", flush=True)
    return artifact


def _global_is_current(config: RunConfig, stage: str, store: SnapshotStore,
                       upstream: dict[str, Path], extra_identity: Any = None) -> bool:
    upstream_hashes = {name: file_hash(store.receipt_path(name, "run"))
                       for name in upstream}
    fingerprint = stage_fingerprint(stage, config, upstream=upstream_hashes,
                                    extra_identity=extra_identity)
    return store.is_current(stage, "run", fingerprint)


def _aggregate_is_current(config: RunConfig, stage: str, contexts: list[Context],
                          store: SnapshotStore) -> bool:
    try:
        receipt_ids = {c.context_id: file_hash(store.receipt_path(stage, c.context_id))
                       for c in contexts}
    except OSError:
        return False
    fingerprint = stage_fingerprint(stage, config, input_identity=receipt_ids,
                                    upstream=receipt_ids)
    return store.is_current(stage, "run", fingerprint)


def _stage_fp(config: RunConfig, inventory: Inventory, validation: dict[str, Any],
              stage: str, context: Context) -> str:
    identity = validation["context_raw_hashes"][context.context_id]
    return stage_fingerprint(stage, config, context_id=context.context_id,
                              input_identity=identity,
                              upstream={"validation": canonical_hash(identity)})


def verify(config: RunConfig, inventory: Inventory, store: SnapshotStore) -> dict[str, Any]:
    """Rehash raw inputs and reject incomplete/stale output trees."""
    _, validation_data = _load_validation(config, store)
    contexts = _contexts(inventory)
    if len(contexts) != config.expected_dataset_count or validation_data["n_contexts"] != len(contexts):
        raise ValueError("Verification rejected: dataset cohort size changed")
    raw_expected: dict[str, str] = validation_data["raw_hashes"]
    raw_errors = []
    for relative, expected in sorted(raw_expected.items()):
        path = (config.input_root / relative).resolve()
        if not path.is_relative_to(config.input_root.resolve()) or not path.is_file():
            raw_errors.append(f"missing or escaped input path: {relative}")
        elif file_hash(path) != expected:
            raw_errors.append(f"raw input changed: {relative}")
    if raw_errors:
        raise ValueError("Final raw-input verification failed:\n- " + "\n- ".join(raw_errors))
    inventory_artifact = store.current_artifact("inventory", "run")
    if inventory_artifact is None:
        raise ValueError("Inventory stage is missing or stale")
    inventory_info = json.loads((inventory_artifact / "inventory.json").read_text(encoding="utf-8"))
    inventory_fp = stage_fingerprint("inventory", config,
                                     input_identity=inventory_info["raw_manifest_hashes"])
    if not store.is_current("inventory", "run", inventory_fp):
        raise ValueError("Inventory stage is missing or stale")
    validate_fp = stage_fingerprint(
        "validate", config, input_identity=inventory_info["raw_manifest_hashes"],
        upstream={"inventory_receipt": file_hash(store.receipt_path("inventory", "run"))})
    if not store.is_current("validate", "run", validate_fp):
        raise ValueError("Validation stage is missing or stale")
    for stage, files in (("metrics", ("condition_summary.csv", "layer_metrics.csv",
                                       "layer_drop_summary.csv", "metric_status.csv",
                                       "precision_probe.csv",
                                       "context.json")),
                         ("pca", ("pca_summary.csv",))):
        for context in contexts:
            fingerprint = _stage_fp(config, inventory, validation_data, stage, context)
            if not store.is_current(stage, context.context_id, fingerprint):
                raise ValueError(f"{stage} output is missing or stale: {context.dataset}")
            artifact = store.current_artifact(stage, context.context_id)
            assert artifact is not None
            missing = [name for name in files if not (artifact / name).is_file()]
            if missing:
                raise ValueError(f"{stage} output missing for {context.dataset}: {missing}")
            if stage == "metrics":
                conditions = set(context.conditions)
                condition_rows = read_csv(artifact / "condition_summary.csv")
                layer_rows = read_csv(artifact / "layer_metrics.csv")
                status_rows = read_csv(artifact / "metric_status.csv")
                expected_layers = {(condition, position)
                                   for condition in conditions
                                   for position in range(len(context.position_names))}
                actual_layers = {(row["condition"], int(row["position_id"]))
                                 for row in layer_rows}
                actual_condition_rows = {row["condition"] for row in condition_rows}
                if (actual_condition_rows != conditions or
                        len(condition_rows) != len(conditions)):
                    raise ValueError(f"metrics condition coverage is invalid: {context.dataset}")
                if actual_layers != expected_layers or len(layer_rows) != len(expected_layers):
                    raise ValueError(f"metrics layer keys are invalid: {context.dataset}")
                expected_statuses = len(expected_layers) * 4
                if len(status_rows) != expected_statuses:
                    raise ValueError(f"metric status coverage is invalid: {context.dataset}")
            else:
                rows = read_csv(artifact / "pca_summary.csv")
                expected = {(condition, position, outcome)
                            for condition in context.conditions
                            for position in range(len(context.position_names))
                            for outcome in ("all", "C_to_C", "C_to_W", "W_to_C", "W_to_W")}
                actual = {(row["condition"], int(row["position_id"]), row["outcome"])
                          for row in rows}
                if actual != expected or len(rows) != len(expected):
                    raise ValueError(f"PCA row coverage is invalid: {context.dataset}")
                expected_order_hash = hashlib.sha256(
                    "\n".join(map(str, context.metadata["sample_ids"])).encode("utf-8")).hexdigest()
                for position in range(len(context.position_names)):
                    position_rows = [row for row in rows
                                     if int(row["position_id"]) == position]
                    basis_files = {row["basis_file"] for row in position_rows}
                    if (len(basis_files) != 1 or
                            any(row.get("sample_order_sha256") != expected_order_hash
                                for row in position_rows)):
                        raise ValueError(f"PCA basis/sample identity is inconsistent: "
                                         f"{context.dataset}/{position}")
                    basis_path = (artifact / next(iter(basis_files))).resolve()
                    if not basis_path.is_relative_to(artifact.resolve()) or not basis_path.is_file():
                        raise ValueError(f"PCA basis reference is missing or escaped: {basis_path}")
                    with np.load(basis_path, allow_pickle=False) as basis:
                        if (basis["mean"].shape != (context.feature_shape[2],) or
                                basis["components"].shape != (context.feature_shape[2], 2) or
                                basis["explained"].shape != (2,)):
                            raise ValueError(f"PCA basis shape is invalid: {basis_path}")
                for row in rows:
                    coordinate_path = (artifact / row["coordinates_file"]).resolve()
                    if not coordinate_path.is_relative_to(artifact.resolve()) or not coordinate_path.is_file():
                        raise ValueError(f"PCA coordinate reference is missing or escaped: "
                                         f"{coordinate_path}")
                    coordinates = np.load(coordinate_path, mmap_mode="r", allow_pickle=False)
                    if (coordinates.shape != (context.feature_shape[0], 2) or
                            coordinates.dtype != np.dtype("float32")):
                        raise ValueError(f"PCA coordinate shape is invalid: {coordinate_path}")
    if not _aggregate_is_current(config, "metrics", contexts, store):
        raise ValueError("Metrics aggregate is missing or stale; run --stage metrics")
    if not _aggregate_is_current(config, "pca", contexts, store):
        raise ValueError("PCA aggregate is missing or stale; run --stage pca")
    global_stages = ("metrics", "pca", "comparisons", "plot", "report")
    metrics_artifact = store.current_artifact("metrics", "run")
    pca_artifact = store.current_artifact("pca", "run")
    assert metrics_artifact is not None and pca_artifact is not None
    mapping_hash = (file_hash(config.group_mapping_file)
                    if config.group_mapping_file and config.group_mapping_file.is_file() else None)
    if config.group_mapping_file and mapping_hash is None:
        raise ValueError(f"Group mapping missing: {config.group_mapping_file}")
    compare_extra = {"group_mapping_sha256": mapping_hash,
                     "cohort": [c.context_id for c in contexts]}
    if not _global_is_current(config, "comparisons", store,
                              {"metrics": metrics_artifact}, compare_extra):
        raise ValueError("Comparisons are missing or stale; run --stage comparisons")
    comparisons_artifact = store.current_artifact("comparisons", "run")
    assert comparisons_artifact is not None
    if not _global_is_current(config, "plot", store,
                              {"metrics": metrics_artifact,
                               "comparisons": comparisons_artifact,
                               "pca": pca_artifact}):
        raise ValueError("Plots are missing or stale; run --stage plot")
    plot_artifact = store.current_artifact("plot", "run")
    validation_artifact = store.current_artifact("validate", "run")
    assert plot_artifact is not None and validation_artifact is not None
    if not _global_is_current(config, "report", store,
                              {"metrics": metrics_artifact, "comparisons": comparisons_artifact,
                               "pca": pca_artifact, "plot": plot_artifact,
                               "validate": validation_artifact}):
        raise ValueError("Report is missing or stale; run --stage report")
    report_artifact = store.current_artifact("report", "run")
    assert report_artifact is not None
    report = report_artifact / "REPORT.md"
    verification = {"status": "complete", "run_id": config.run_id,
                    "config_identity": config.identity, "n_contexts": len(contexts),
                    "n_raw_files_rehashed": len(raw_expected),
                    "raw_inputs_match_validation": True,
                    "stages": ["inventory", "validate", *global_stages],
                    "report_sha256": file_hash(report),
                    "verified_raw_hashes_sha256": canonical_hash(raw_expected)}
    state_path = config.output_root / "state" / "verification.json"
    atomic_json(state_path, verification)
    return verification


def _inventory_for_stage(config: RunConfig, store: SnapshotStore,
                         rebuild: bool) -> Inventory:
    inventory = discover(config)
    _inventory_artifact(config, inventory, store, rebuild)
    resolved_config = dataclasses.asdict(config)
    for key, value in resolved_config.items():
        if isinstance(value, Path):
            resolved_config[key] = str(value)
        elif isinstance(value, tuple):
            resolved_config[key] = list(value)
    resolved_config["runtime"] = runtime_identity()
    atomic_json(store.state / "run_config.json", resolved_config)
    lock_path = store.state / "cohort_lock.json"
    if lock_path.is_file():
        locked = json.loads(lock_path.read_text(encoding="utf-8"))
        proposed = {"cohort_identity": config.cohort_identity,
                    "context_ids": sorted(c.context_id for c in inventory.contexts),
                    "conditions": list(inventory.conditions),
                    "positions": list(inventory.contexts[0].position_names)
                    if inventory.contexts else []}
        old = {key: locked.get(key) for key in proposed}
        if old != proposed:
            if not rebuild:
                raise ValueError("Run cohort differs from cohort_lock.json. Inspect the "
                                 "inventory and use --rebuild to create a new validated cohort version.")
            archive = lock_path.with_name(
                f"cohort_lock.{canonical_hash(locked)[:12]}.json")
            if not archive.exists():
                shutil.copyfile(lock_path, archive)
            atomic_json(lock_path.with_name("cohort_lock.pending.json"),
                        {**proposed, "status": "pending_validation"})
    return inventory


def run(config: RunConfig, stage: str, dataset: str | None = None,
        rebuild: bool = False) -> None:
    if dataset and stage not in {"metrics", "pca"}:
        raise ValueError("--dataset is supported only by metrics and pca stages")
    if stage == "all" and dataset:
        raise ValueError("--stage all requires the complete cohort; omit --dataset")
    store = SnapshotStore(config)
    if stage == "inventory":
        inventory = _inventory_for_stage(config, store, rebuild)
        print(f"Inventory: {len(inventory.contexts)} contexts, "
              f"{len(inventory.conditions)} conditions; errors={len(inventory.errors)}")
        if inventory.errors:
            raise ValueError("Inventory is incomplete:\n- " + "\n- ".join(inventory.errors))
        return
    inventory = _inventory_for_stage(config, store, rebuild=rebuild)
    if inventory.errors:
        raise ValueError("Inventory is incomplete:\n- " + "\n- ".join(inventory.errors))
    if stage in {"comparisons", "plot", "report", "all"} and not config.primary_condition:
        raise ValueError("Set primary_condition in the config after inspecting inventory")
    if stage == "validate":
        artifact, result = _validation_artifact(config, inventory, store, rebuild)
        print(f"VALIDATED {result['n_contexts']} contexts, {result['n_raw_files']} raw files: {artifact}")
        return
    # Re-run full data validation for `all`; individual stages require the latest receipt.
    if stage == "all":
        _, validation_data = _validation_artifact(config, inventory, store, rebuild)
    else:
        _, validation_data = _load_validation(config, store)
        if stage != "verify":
            _verify_validated_sources(config, validation_data, dataset=dataset)
    if stage in {"metrics", "pca", "all"}:
        metrics_artifacts: dict[str, Path] = {}
        pca_artifacts: dict[str, Path] = {}
        metrics_aggregate = None
        pca_aggregate = None
        if stage in {"metrics", "all"}:
            metrics_artifacts, metrics_aggregate = _run_context_stage(
                config, inventory, validation_data, "metrics", store,
                dataset if stage == "metrics" else None, rebuild)
            if metrics_aggregate is None:
                return
        if stage in {"pca", "all"}:
            pca_artifacts, pca_aggregate = _run_context_stage(
                config, inventory, validation_data, "pca", store,
                dataset if stage == "pca" else None, rebuild)
            if pca_aggregate is None:
                return
        if stage in {"metrics", "pca"}:
            print(f"{stage} complete for {dataset or 'all contexts'}")
            return
    if stage in {"comparisons", "all"}:
        metrics_artifact = store.current_artifact("metrics", "run")
        if metrics_artifact is None:
            raise RuntimeError("Complete metrics coverage required. Run --stage metrics.")
        if not config.primary_condition:
            raise ValueError("Set primary_condition in the config after inspecting inventory")
        if not _aggregate_is_current(config, "metrics", _contexts(inventory), store):
            raise RuntimeError("Metrics aggregate is stale or incomplete; run --stage metrics")
        mapping_hash = file_hash(config.group_mapping_file) if config.group_mapping_file and config.group_mapping_file.is_file() else None
        if config.group_mapping_file and mapping_hash is None:
            raise FileNotFoundError(config.group_mapping_file)
        comparisons_artifact = _global_stage(
            config, "comparisons", store, {"metrics": metrics_artifact},
            lambda out: analyze_comparisons(config, _contexts(inventory), metrics_artifact, out),
            extra_identity={"group_mapping_sha256": mapping_hash,
                            "cohort": [c.context_id for c in _contexts(inventory)]},
            rebuild=rebuild)
    if stage in {"plot", "all"}:
        metrics_artifact = store.current_artifact("metrics", "run")
        comparisons_artifact = store.current_artifact("comparisons", "run")
        pca_artifact = store.current_artifact("pca", "run")
        if any(value is None for value in (metrics_artifact, comparisons_artifact, pca_artifact)):
            raise RuntimeError("Complete metrics, comparisons, and PCA stages are required before plot")
        if not _aggregate_is_current(config, "metrics", _contexts(inventory), store) or not _aggregate_is_current(
                config, "pca", _contexts(inventory), store):
            raise RuntimeError("Metrics or PCA aggregate is stale; rerun its complete stage")
        mapping_hash = (file_hash(config.group_mapping_file)
                        if config.group_mapping_file and config.group_mapping_file.is_file() else None)
        if config.group_mapping_file and mapping_hash is None:
            raise FileNotFoundError(config.group_mapping_file)
        compare_extra = {"group_mapping_sha256": mapping_hash,
                         "cohort": [c.context_id for c in _contexts(inventory)]}
        if not _global_is_current(config, "comparisons", store,
                                  {"metrics": metrics_artifact}, compare_extra):
            raise RuntimeError("Comparisons are stale; run --stage comparisons first")
        context_pca = {c.context_id: store.current_artifact("pca", c.context_id)
                       for c in _contexts(inventory)}
        if any(value is None for value in context_pca.values()):
            raise RuntimeError("At least one context PCA artifact is missing")
        plot_artifact = _global_stage(
            config, "plot", store,
            {"metrics": metrics_artifact, "comparisons": comparisons_artifact,
             "pca": pca_artifact},
            lambda out: plot(config, _contexts(inventory), metrics_artifact,
                             comparisons_artifact, pca_artifact, context_pca, out),
            rebuild=rebuild)
    if stage in {"report", "all"}:
        metrics_artifact = store.current_artifact("metrics", "run")
        comparisons_artifact = store.current_artifact("comparisons", "run")
        pca_artifact = store.current_artifact("pca", "run")
        plot_artifact = store.current_artifact("plot", "run")
        if any(value is None for value in (metrics_artifact, comparisons_artifact,
                                           pca_artifact, plot_artifact)):
            raise RuntimeError("Complete metrics, comparisons, PCA, and plot stages are required before report")
        if (not _aggregate_is_current(config, "metrics", _contexts(inventory), store) or
                not _aggregate_is_current(config, "pca", _contexts(inventory), store)):
            raise RuntimeError("Metrics or PCA aggregate is stale; rerun its complete stage")
        mapping_hash = (file_hash(config.group_mapping_file)
                        if config.group_mapping_file and config.group_mapping_file.is_file() else None)
        compare_extra = {"group_mapping_sha256": mapping_hash,
                         "cohort": [c.context_id for c in _contexts(inventory)]}
        if not _global_is_current(config, "comparisons", store,
                                  {"metrics": metrics_artifact}, compare_extra):
            raise RuntimeError("Comparisons are stale; run --stage comparisons first")
        if not _global_is_current(config, "plot", store,
                                  {"metrics": metrics_artifact,
                                   "comparisons": comparisons_artifact, "pca": pca_artifact}):
            raise RuntimeError("Plots are stale; run --stage plot first")
        validation_artifact, validation_info = _load_validation(config, store)
        _global_stage(config, "report", store,
                      {"metrics": metrics_artifact, "comparisons": comparisons_artifact,
                       "pca": pca_artifact, "plot": plot_artifact,
                       "validate": validation_artifact},
                      lambda out: generate_report(config, _contexts(inventory), metrics_artifact,
                                                  comparisons_artifact, pca_artifact,
                                                  plot_artifact, validation_info, out),
                      rebuild=rebuild)
    if stage in {"verify", "all"}:
        print(json.dumps(verify(config, inventory, store), ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the run JSON config")
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--dataset", help="Restrict metrics or PCA to one dataset")
    parser.add_argument("--rebuild", action="store_true",
                        help="Recompute the selected stage; source validation still applies")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        run(config, args.stage, args.dataset, args.rebuild)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
