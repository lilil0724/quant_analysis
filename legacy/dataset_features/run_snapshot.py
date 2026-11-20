"""Content-addressed receipts for resumable 20004 analysis stages."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import uuid
from pathlib import Path
from typing import Any

from .run_config import RunConfig, canonical_hash


CODE_DEPENDENCIES = {
    "inventory": ("run.py", "run_config.py", "inputs.py", "run_snapshot.py"),
    "validate": ("run.py", "run_config.py", "inputs.py", "run_snapshot.py"),
    "metrics": ("run.py", "run_config.py", "inputs.py", "metrics.py",
                 "layer_analysis.py", "run_snapshot.py"),
    "pca": ("run.py", "run_config.py", "inputs.py", "metrics.py",
            "layer_pca.py", "run_snapshot.py"),
    "comparisons": ("run.py", "run_config.py", "inputs.py", "metrics.py",
                    "comparison_values.py", "comparisons.py", "run_snapshot.py"),
    "plot": ("run.py", "run_config.py", "inputs.py", "comparison_values.py",
             "plot_run.py", "run_snapshot.py"),
    "report": ("run.py", "run_config.py", "report_run.py", "run_snapshot.py"),
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(folder: Path) -> dict[str, str]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing stage artifact directory: {folder}")
    files = sorted(path for path in folder.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"Stage artifact directory is empty: {folder}")
    return {path.relative_to(folder).as_posix(): file_hash(path) for path in files}


def code_hashes(stage: str) -> dict[str, str]:
    if stage not in CODE_DEPENDENCIES:
        raise ValueError(f"No code dependency map for stage {stage}")
    base = Path(__file__).parent
    return {name: file_hash(base / name) for name in CODE_DEPENDENCIES[stage]}


def runtime_identity() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in ("numpy", "scipy", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def stage_fingerprint(stage: str, config: RunConfig, *, context_id: str | None = None,
                      input_identity: Any = None,
                      upstream: dict[str, str] | None = None,
                      extra_identity: Any = None) -> str:
    return canonical_hash({
        "stage": stage, "config": config.stage_identity(stage),
        "context_id": context_id, "input_identity": input_identity,
        "upstream": upstream or {}, "extra_identity": extra_identity,
        "code": code_hashes(stage), "runtime": runtime_identity(),
    })


class SnapshotStore:
    """Read stage receipts and atomically publish content-addressed outputs."""

    def __init__(self, config: RunConfig):
        self.config = config
        self.root = config.output_root
        self.state = self.root / "state"
        self.artifacts = self.root / "artifacts"

    def receipt_path(self, stage: str, scope: str) -> Path:
        safe = hashlib.sha256(scope.encode("utf-8")).hexdigest()[:20]
        return self.state / stage / f"{safe}.json"

    def current_artifact(self, stage: str, scope: str) -> Path | None:
        path = self.receipt_path(stage, scope)
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
            artifact = self.root / receipt["artifact"]
            if not artifact.resolve().is_relative_to(self.artifacts.resolve()):
                return None
            if tree_hash(artifact) != receipt.get("artifact_hashes"):
                return None
            return artifact
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def is_current(self, stage: str, scope: str, fingerprint: str) -> bool:
        path = self.receipt_path(stage, scope)
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
            if receipt.get("fingerprint") != fingerprint:
                return False
            return self.current_artifact(stage, scope) is not None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False

    def record(self, stage: str, scope: str, fingerprint: str,
               artifact: Path, extra: Any = None) -> None:
        artifact = artifact.resolve()
        if not artifact.is_relative_to(self.artifacts.resolve()):
            raise ValueError(f"Refusing to snapshot path outside artifacts: {artifact}")
        receipt = {
            "schema_version": 1, "stage": stage, "scope": scope,
            "fingerprint": fingerprint,
            "artifact": artifact.relative_to(self.root.resolve()).as_posix(),
            "artifact_hashes": tree_hash(artifact), "extra": extra,
        }
        path = self.receipt_path(stage, scope)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
        temp.write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True,
                                   indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temp, path)

    def build_dir(self, stage: str, scope: str, fingerprint: str) -> Path:
        safe_scope = hashlib.sha256(scope.encode("utf-8")).hexdigest()[:20]
        final = self.artifacts / stage / safe_scope / fingerprint[:20]
        final.parent.mkdir(parents=True, exist_ok=True)
        return final.parent / f".{fingerprint[:20]}.{uuid.uuid4().hex}.tmp"

    def publish(self, stage: str, scope: str, fingerprint: str,
                temporary: Path, extra: Any = None) -> Path:
        safe_scope = hashlib.sha256(scope.encode("utf-8")).hexdigest()[:20]
        final = self.artifacts / stage / safe_scope / fingerprint[:20]
        staged_hashes = tree_hash(temporary)
        if not staged_hashes:
            raise ValueError(f"Refusing to publish empty output: {temporary}")
        if final.exists():
            if tree_hash(final) == staged_hashes:
                shutil.rmtree(temporary)
                self.record(stage, scope, fingerprint, final, extra)
                return final
            digest = canonical_hash(staged_hashes)[:12]
            final = final.with_name(final.name + "-" + digest)
            if final.exists() and tree_hash(final) != staged_hashes:
                raise FileExistsError(f"Conflicting immutable artifact version: {final}")
            if final.exists():
                shutil.rmtree(temporary)
                self.record(stage, scope, fingerprint, final, extra)
                return final
        os.replace(temporary, final)
        self.record(stage, scope, fingerprint, final, extra)
        return final

    def json_artifact(self, stage: str, scope: str) -> dict[str, Any]:
        folder = self.current_artifact(stage, scope)
        if folder is None:
            raise RuntimeError(f"Missing or stale {stage} artifact for {scope}")
        index = folder / "receipt.json"
        if not index.is_file():
            raise RuntimeError(f"Missing {stage} receipt payload: {index}")
        return json.loads(index.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                               indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)
