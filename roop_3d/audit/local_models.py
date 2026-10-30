"""Hash and register local weight artifacts without loading legacy Python code.

VERIFIED_LOCAL means bytes were read locally and hashed. It is deliberately
independent of licensing, upstream provenance, and successful inference.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from enum import Enum
import json
import io
import os
from pathlib import Path
from typing import Any, Mapping

from isolation import AuditError, is_link, safe_path, sha256_file, write_json, write_text


class EvidenceState(str, Enum):
    VERIFIED_LOCAL = "VERIFIED_LOCAL"
    VERIFIED_REMOTE = "VERIFIED_REMOTE"
    CANDIDATE = "CANDIDATE"
    UNKNOWN = "UNKNOWN"
    BLOCKED = "BLOCKED"
    UNSUPPORTED = "UNSUPPORTED"


WEIGHT_EXTENSIONS = {".onnx", ".pth", ".pt", ".ckpt", ".safetensors", ".bin"}
PRUNE = {".git", "env", "venv", ".venv", "node_modules", "__pycache__",
         "roop_3d", "roop_3d_ui", "output", "outputs", "temp", "logs", "site-packages"}


def family_hint(name: str) -> str:
    name = name.casefold()
    groups = (
        ("lipsync", ("wav2lip", "musetalk")),
        ("animation", ("liveportrait",)),
        ("swapper", ("inswapper", "hyperswap", "hififace", "simswap")),
        ("detector", ("scrfd", "retinaface", "det_10g", "yoloface", "yunet")),
        ("recognizer", ("arcface", "w600k", "glintr", "adaface", "sface")),
        ("parser", ("bisenet", "xseg", "resnet18", "face_occluder")),
        ("restorer", ("gfpgan", "codeformer", "gpen", "restoreformer", "dmdnet")),
    )
    return next((label for label, hints in groups if any(h in name for h in hints)), "other")


def graph_metadata(path: Path) -> dict[str, Any]:
    if path.suffix.casefold() != ".onnx":
        return {"state": "UNKNOWN", "reason": "Not deserialized; pickle execution is forbidden during discovery"}
    try:
        import onnx
        # External tensor paths are reported as graph metadata, never followed.
        model = onnx.load_model(str(path), load_external_data=False)
        initializers = {tensor.name for tensor in model.graph.initializer}

        def tensor_info(value: Any) -> dict[str, Any]:
            tensor = value.type.tensor_type
            return {"name": value.name, "dtype": onnx.TensorProto.DataType.Name(tensor.elem_type),
                    "shape": [d.dim_value if d.HasField("dim_value") else
                              d.dim_param if d.HasField("dim_param") else None
                              for d in tensor.shape.dim]}

        return {"state": "VERIFIED_LOCAL", "evidence": "ONNX graph metadata parsed locally; no inference",
                "onnx_version": onnx.__version__,
                "inputs": [tensor_info(v) for v in model.graph.input if v.name not in initializers],
                "outputs": [tensor_info(v) for v in model.graph.output],
                "opsets": [{"domain": op.domain, "version": op.version} for op in model.opset_import],
                "external_data": [{"tensor": t.name, "entries": {e.key: e.value for e in t.external_data}}
                                  for t in model.graph.initializer if t.external_data]}
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        return {"state": "UNKNOWN", "reason": f"{type(error).__name__}: {error}"}
    except Exception as error:
        # Corrupt protobufs are evidence of failed inspection, never valid tensor specs.
        return {"state": "UNKNOWN", "reason": f"Graph inspection failed: {type(error).__name__}: {error}"}


def discover(scopes: Mapping[str, Path], expected_hashes: Mapping[str, str] | None = None) -> dict[str, Any]:
    models: list[dict[str, Any]] = []
    seen: set[str] = set()
    scope_evidence = []
    expected_hashes = expected_hashes or {}
    for scope, base in scopes.items():
        if not base.exists():
            scope_evidence.append({"scope": scope, "state": "UNKNOWN", "reason": "Directory absent"})
            continue
        safe_path(base, "probe")
        scope_evidence.append({"scope": scope, "state": "VERIFIED_LOCAL", "evidence": "Directory scanned read-only"})

        def onerror(error: OSError) -> None:
            raise AuditError("MODEL_SCAN_FAILED", str(error))

        for parent, directories, filenames in os.walk(base, followlinks=False, onerror=onerror):
            folder = Path(parent)
            directories[:] = sorted(d for d in directories
                                    if d.casefold() not in PRUNE and not is_link(folder / d))
            for filename in sorted(filenames):
                path = folder / filename
                if path.suffix.casefold() not in WEIGHT_EXTENSIONS or is_link(path):
                    continue
                if path.suffix.casefold() == ".bin" and not (
                    filename.casefold() == "model.bin" or filename.casefold().endswith("_model.bin")
                ):
                    # TensorRT timing caches and executables are not weight files.
                    continue
                identity = os.path.normcase(str(path.absolute()))
                if identity in seen:
                    continue
                seen.add(identity)
                size, digest = sha256_file(path)
                if identity in expected_hashes and expected_hashes[identity] != digest:
                    raise AuditError("MODEL_DRIFT", str(path))
                models.append({"name": filename, "locator": {"scope": scope, "path": path.relative_to(base).as_posix()},
                               "state": "VERIFIED_LOCAL", "size_bytes": size, "sha256": digest,
                               "evidence": "Local file bytes read and SHA-256 computed; not a download checksum",
                               "family_hint": family_hint(filename), "family_evidence": "Filename heuristic only",
                               "provenance_state": "UNKNOWN", "license_state": "UNKNOWN",
                               "commercial_permission": "UNKNOWN", "inference_state": "UNKNOWN",
                               "graph": graph_metadata(path)})
    return {"schema_version": 1, "timestamp": datetime.now(timezone.utc).isoformat(),
            "verification_contract": "VERIFIED_LOCAL verifies local bytes, not inference or licensing",
            "scope_evidence": scope_evidence, "models": models,
            "family_counts": dict(sorted(Counter(m["family_hint"] for m in models).items()))}


def verify_registry(registry: dict[str, Any], scopes: Mapping[str, Path]) -> dict[str, Any]:
    models = registry.get("models")
    if not isinstance(models, list) or not models:
        raise AuditError("INVALID_REGISTRY", "A nonempty models array is required")
    checked = 0
    violations = []
    seen: set[tuple[str, str]] = set()
    for model in models:
        locator = model["locator"]
        key = (locator["scope"], locator["path"])
        if key in seen:
            raise AuditError("INVALID_REGISTRY", f"Duplicate model: {key}")
        seen.add(key)
        try:
            if model["state"] != EvidenceState.VERIFIED_LOCAL.value:
                raise AuditError("INVALID_REGISTRY", f"Unverified model: {key}")
            path = safe_path(scopes[key[0]], key[1])
            size, digest = sha256_file(path)
            if size != model["size_bytes"] or digest != model["sha256"]:
                raise AuditError("MODEL_DRIFT", str(key))
            checked += 1
        except (OSError, AuditError, KeyError) as error:
            violations.append({"locator": locator, "detail": str(error)})
    return {"ok": not violations, "checked": checked, "expected": len(models), "violations": violations}


def default_scopes(repo: Path) -> dict[str, Path]:
    home = Path.home()
    return {"repository": repo, "user_cache": home / ".cache", "insightface_cache": home / ".insightface",
            "torch_cache": home / ".torch"}


def license_register(registry: dict[str, Any]) -> str:
    """Keep artifact integrity separate from permissions we have not verified."""
    output = io.StringIO(newline="")
    fields = ["artifact_name", "family_hint", "verification_state", "locator", "sha256",
              "graph_metadata_state", "inference_state", "license_state", "commercial_permission",
              "download_url_state", "evidence"]
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for model in registry["models"]:
        locator = model["locator"]
        writer.writerow({"artifact_name": model["name"], "family_hint": model["family_hint"],
                         "verification_state": model["state"], "locator": f"{locator['scope']}:{locator['path']}",
                         "sha256": model["sha256"], "graph_metadata_state": model["graph"]["state"],
                         "inference_state": model["inference_state"], "license_state": model["license_state"],
                         "commercial_permission": model["commercial_permission"], "download_url_state": "UNKNOWN",
                         "evidence": model["evidence"]})
    for name in ("FLAME", "Wav2Lip", "InstantID", "PuLID", "SDXL-Turbo", "FLUX.1-Schnell", "remote_provider"):
        writer.writerow({"artifact_name": name, "verification_state": "CANDIDATE", "graph_metadata_state": "UNKNOWN",
                         "inference_state": "UNKNOWN", "license_state": "UNKNOWN", "commercial_permission": "UNKNOWN",
                         "download_url_state": "UNKNOWN",
                         "evidence": "Optional capability; upstream artifacts, permission and execution not verified by this audit"})
    return output.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Read and verify the generated registry")
    arguments = parser.parse_args()
    repo = Path(__file__).absolute().parents[2]
    registry_path = "roop_3d/audit/verified_local_models.json"
    try:
        scopes = default_scopes(repo)
        if arguments.verify:
            registry = json.loads(safe_path(repo, registry_path).read_text("utf-8"))
            result = verify_registry(registry, scopes)
        else:
            old = json.loads(safe_path(repo, "roop_3d/audit/local_models_discovered.json").read_text("utf-8"))
            expected = {os.path.normcase(str(Path(m["path"]).absolute())): m["sha256"] for m in old["models"]}
            registry = discover(scopes, expected)
            result = {"ok": True, "models": len(registry["models"]), "family_counts": registry["family_counts"]}
            # Refuse to silently lose artifacts from the original evidence set.
            identities = {os.path.normcase(str((scopes[m["locator"]["scope"]] / m["locator"]["path"]).absolute()))
                          for m in registry["models"]}
            missing = sorted(set(expected) - identities)
            if missing:
                raise AuditError("MODEL_DISCOVERY_INCOMPLETE", f"Prior models not rediscovered: {missing}")
            write_json(repo, registry_path, registry)
            write_text(repo, "roop_3d/audit/model_and_license_register.csv", license_register(registry))
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1
    except (OSError, ValueError, KeyError, AuditError) as error:
        envelope = error.envelope() if isinstance(error, AuditError) else {
            "ok": False, "error": {"code": "DISCOVERY_FAILED", "detail": str(error)}}
        print(json.dumps(envelope, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
