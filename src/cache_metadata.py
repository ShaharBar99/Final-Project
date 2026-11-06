"""Small, deterministic metadata helpers for validating cached artifacts."""
import hashlib
import json
import platform
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
DEPENDENCIES = ("numpy", "pandas", "scikit-learn", "scipy", "networkx", "igraph", "joblib")


class CacheValidationError(RuntimeError):
    """A cache exists but cannot safely be reused."""


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def stable_fingerprint(value):
    encoded = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def table_fingerprint(table):
    """Fingerprint values, row order/index, columns, and dtypes of a Series/DataFrame."""
    frame = table.to_frame() if isinstance(table, pd.Series) else table
    digest = hashlib.sha256()
    description = {
        "columns": [str(c) for c in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "index_names": [str(name) for name in frame.index.names],
        "shape": list(frame.shape),
    }
    digest.update(json.dumps(description, sort_keys=True).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=True, categorize=True).to_numpy().tobytes())
    return digest.hexdigest()


def graph_fingerprint(graph):
    """Fingerprint graph structure and node order (the latter controls result-array alignment)."""
    nodes = [repr(node) for node in graph.nodes]
    edges = sorted((repr(a), repr(b)) for a, b in graph.edges)
    return stable_fingerprint({
        "directed": graph.is_directed(),
        "multigraph": graph.is_multigraph(),
        "node_order": nodes,
        "edges": edges,
    })


def dependency_versions():
    versions = {"python": platform.python_version()}
    for package in DEPENDENCIES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def scaler_spec(scaler):
    if scaler is None:
        return {"available": False}
    fitted = {}
    for name in ("feature_names_in_", "data_min_", "data_max_", "data_range_", "scale_", "min_", "n_features_in_"):
        if hasattr(scaler, name):
            fitted[name] = _jsonable(getattr(scaler, name))
    return {
        "available": True,
        "class": f"{type(scaler).__module__}.{type(scaler).__name__}",
        "parameters": _jsonable(scaler.get_params(deep=False)),
        "fitted_state": fitted,
    }


def scaler_fingerprint(scaler):
    return stable_fingerprint(scaler_spec(scaler))


def metadata_path(artifact_path):
    return Path(f"{artifact_path}.meta.json")


def write_metadata(artifact_path, spec, related_artifacts=None):
    """Write metadata last, so an interrupted artifact write is never marked valid."""
    artifact_path = Path(artifact_path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "spec": _jsonable(spec),
        "spec_fingerprint": stable_fingerprint(spec),
        "artifact_sha256": file_sha256(artifact_path),
        "related_artifacts": _jsonable(related_artifacts or {}),
    }
    path = metadata_path(artifact_path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    temporary.replace(path)
    return payload


def validate_metadata(artifact_path, expected_spec=None):
    artifact_path = Path(artifact_path)
    sidecar = metadata_path(artifact_path)
    if not artifact_path.exists():
        raise CacheValidationError(f"cached artifact is missing: {artifact_path}")
    if not sidecar.exists():
        raise CacheValidationError(f"cache metadata is missing: {sidecar}")
    try:
        with open(sidecar, encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise CacheValidationError(f"cache metadata is unreadable: {sidecar}") from error
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise CacheValidationError(f"unsupported cache metadata schema: {sidecar}")
    if payload.get("artifact_sha256") != file_sha256(artifact_path):
        raise CacheValidationError(f"cached artifact checksum does not match metadata: {artifact_path}")
    if payload.get("spec_fingerprint") != stable_fingerprint(payload.get("spec")):
        raise CacheValidationError(f"cache metadata fingerprint is corrupt: {sidecar}")
    if expected_spec is not None and payload.get("spec_fingerprint") != stable_fingerprint(expected_spec):
        raise CacheValidationError(f"cached artifact configuration does not match the current run: {artifact_path}")
    for filename, expected_hash in payload.get("related_artifacts", {}).items():
        related = artifact_path.parent / filename
        if not related.exists() or file_sha256(related) != expected_hash:
            raise CacheValidationError(f"related cached artifact is missing or corrupt: {related}")
    return payload
