"""Compatibility-gated updater for the Pinokio project.

The updater deliberately has a narrow apply surface.  A candidate must carry
an immutable, repository-provided ``update_manifest.json`` whose commit,
runtime constraints, hardware profiles, provider policy, checkpoint contract,
and sensitive-file hashes all agree with the current installation.  Only a
source-only fast-forward is currently admissible.  Dependency, model, and
critical-runtime changes are reported for review and are never installed by
this command.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


CLASSIFICATIONS = ("SAFE", "REQUIRES REVIEW", "UNVERIFIED", "INCOMPATIBLE")
MANIFEST_PATH = "update_manifest.json"
CHECKPOINT_SCHEMA = 1
PROCESSING_CONTRACT = "segmented-video-v1"
MANDATORY_HARDWARE_PROFILES = {
    "rtx4070_12gb",
    "rtx3060_laptop_6gb",
}
# These are the compute capabilities recorded by the repository's hardware
# evidence for the two mandatory targets (RTX 4070 SM 8.9, RTX 3060 SM 8.6).
MANDATORY_GPU_ARCHITECTURES = {"8.9", "8.6"}
SENSITIVE_FILES = (
    "app/requirements.txt",
    "app/update_manager.py",
    "app/update_health.py",
    "torch.js",
    "fix_tensorrt.js",
    "update.js",
    "react-ui/package.json",
    "react-ui/package-lock.json",
    "react-ui-v2/package.json",
    "react-ui-v2/package-lock.json",
)
ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_ROOT = ROOT / ".update-snapshots"
STAGING_ROOT = ROOT / ".update-staging"
TRANSACTION_PATH = ROOT / ".update-transaction.json"
HEALTH_TIMEOUT_SECONDS = 900


class UpdateError(RuntimeError):
    """Raised for an updater operation that cannot produce a safe report."""


def _run(command: list[str], cwd: Path = ROOT, check: bool = True,
         timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(command, cwd=str(cwd), text=True,
                                capture_output=True, check=False, timeout=timeout)
    except OSError as exc:
        if check:
            raise UpdateError(f"{command[0]} could not be started: {exc}") from exc
        return subprocess.CompletedProcess(command, 127, "", str(exc))
    except subprocess.TimeoutExpired as exc:
        if check:
            raise UpdateError(f"{command[0]} timed out after {timeout} seconds") from exc
        return subprocess.CompletedProcess(command, 124, exc.stdout or "", exc.stderr or "")
    if check and result.returncode:
        detail = (result.stderr or result.stdout or "command failed").strip()
        raise UpdateError(f"{command[0]} failed: {detail}")
    return result


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Write transaction metadata so a torn write cannot look successful."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent),
                                         prefix=f".{path.name}.", suffix=".tmp",
                                         delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise UpdateError(f"atomic metadata write failed for {path}: {exc}") from exc
    finally:
        if temporary is not None and temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=str(destination.parent),
                                         prefix=f".{destination.name}.", suffix=".tmp",
                                         delete=False) as handle:
            temporary = Path(handle.name)
            with source.open("rb") as source_handle:
                shutil.copyfileobj(source_handle, handle)
            handle.flush()
            os.fsync(handle.fileno())
        shutil.copystat(source, temporary)
        os.replace(temporary, destination)
    except OSError as exc:
        raise UpdateError(f"atomic snapshot copy failed for {source}: {exc}") from exc
    finally:
        if temporary is not None and temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def _git(*args: str, check: bool = True) -> str:
    return _run(["git", *args], check=check).stdout.strip()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None
    match = re.search(r"\d+(?:\.\d+)+", str(value))
    if not match:
        return None
    return tuple(int(part) for part in match.group(0).split("."))


def _version_matches(actual: str | None, requirement: str | None) -> bool | None:
    """Evaluate only simple, explicit version constraints.

    Complex specifier syntax is intentionally unknown here.  Returning None
    prevents an unverified constraint from being treated as compatible.
    """
    if not actual or not requirement:
        return None
    requirement = str(requirement).strip()
    if requirement.startswith("=="):
        return actual.lower() == requirement[2:].strip().lower()
    if requirement.startswith(">="):
        left, right = _version(actual), _version(requirement[2:])
        return left is not None and right is not None and left >= right
    if requirement.startswith("<="):
        left, right = _version(actual), _version(requirement[2:])
        return left is not None and right is not None and left <= right
    if requirement.startswith(">"):
        left, right = _version(actual), _version(requirement[1:])
        return left is not None and right is not None and left > right
    if requirement.startswith("<"):
        left, right = _version(actual), _version(requirement[1:])
        return left is not None and right is not None and left < right
    if re.fullmatch(r"\d+(?:\.\d+)+", requirement):
        left, right = _version(actual), _version(requirement)
        return left is not None and right is not None and left == right
    return None


def _distribution_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _provider_from_config() -> str | None:
    configured = os.environ.get("ROOP_EXECUTION_PROVIDER")
    if configured:
        return configured.strip().lower()
    config = ROOT / "app" / "config.yaml"
    if not config.is_file():
        return None
    for line in config.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"^\s*provider\s*:\s*([^#\s]+)", line, re.IGNORECASE)
        if match:
            return match.group(1).strip().lower()
    return None


def _available_providers() -> list[str] | None:
    try:
        import onnxruntime as ort
        return [str(item) for item in ort.get_available_providers()]
    except Exception:
        return None


def _hardware_evidence() -> dict[str, Any]:
    command = ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
               "--format=csv,noheader,nounits"]
    result = _run(command, check=False)
    if result.returncode:
        return {"profile": None, "gpu": None, "vram_mb": None,
                "driver": None, "compute_capability": None}
    row = next((line.strip() for line in result.stdout.splitlines() if line.strip()), "")
    fields = [item.strip() for item in row.split(",")]
    if len(fields) < 4:
        return {"profile": None, "gpu": None, "vram_mb": None,
                "driver": None, "compute_capability": None}
    gpu, raw_vram, driver, compute = fields[:4]
    try:
        vram_mb = int(float(raw_vram))
    except ValueError:
        vram_mb = None
    lowered = gpu.lower()
    profile = None
    if "4070" in lowered and (vram_mb or 0) >= 11000:
        profile = "rtx4070_12gb"
    elif "3060" in lowered and (vram_mb or 0) < 7000:
        profile = "rtx3060_laptop_6gb"
    return {"profile": profile, "gpu": gpu, "vram_mb": vram_mb,
            "driver": driver, "compute_capability": compute}


def _current_file_hashes() -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for relative in SENSITIVE_FILES:
        path = ROOT / relative
        values[relative] = _sha256_file(path) if path.is_file() else None
    return values


def _active_work() -> list[str]:
    """Find persisted work that makes source activation unsafe right now."""
    # Queue and project state vocabularies are defined in routes_queue.py and
    # project_checkpoint.py.  A paused/recoverable item still binds a future
    # continuation to the current source/runtime generation, so it is not idle
    # for update admission purposes.
    queue_protected = {"QUEUED", "PREPARING", "PROCESSING", "PAUSE_REQUESTED",
                       "PAUSED", "RECOVERABLE", "INTERRUPTED"}
    project_protected = {"PROCESSING", "PAUSED", "INTERRUPTED", "RECOVERABLE"}
    found: list[str] = []
    queue = ROOT / "app" / "queue.json"
    if queue.is_file():
        try:
            value = json.loads(queue.read_text(encoding="utf-8"))
            jobs = value.get("jobs", []) if isinstance(value, dict) else value
            if isinstance(jobs, list):
                for job in jobs:
                    if isinstance(job, dict) and str(job.get("state", "")) in queue_protected:
                        found.append(f"queue job {job.get('id', 'unknown')} is {job['state']}")
        except (OSError, ValueError, json.JSONDecodeError):
            found.append("queue state exists but could not be validated")
    projects = ROOT / "app" / "projects"
    if projects.is_dir():
        for path in projects.glob("*.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(value, dict) and value.get("state") in project_protected:
                    found.append(f"project {value.get('id', path.stem)} is {value['state']}")
            except (OSError, ValueError, json.JSONDecodeError):
                found.append(f"project checkpoint {path.name} could not be validated")
    return found


def _current_identity() -> dict[str, Any]:
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    sha = _git("rev-parse", "HEAD")
    describe = _git("describe", "--tags", "--always")
    remote = _git("remote", "get-url", "origin", check=False)
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    hardware = _hardware_evidence()
    providers = _available_providers()
    runtime = {
        "torch": _distribution_version("torch"),
        "onnxruntime": _distribution_version("onnxruntime-gpu", "onnxruntime"),
        "tensorrt": _distribution_version("tensorrt", "tensorrt-cu12"),
    }
    try:
        import torch
        runtime["cuda"] = str(getattr(torch.version, "cuda", "") or "") or None
    except Exception:
        runtime["cuda"] = None
    ffmpeg = _run(["ffmpeg", "-version"], check=False)
    runtime["ffmpeg"] = (ffmpeg.stdout.splitlines()[0] if ffmpeg.returncode == 0
                          and ffmpeg.stdout else None)
    return {
        "branch": branch,
        "sha": sha,
        "version": f"{branch}@{describe}",
        "remote": remote,
        "dirty": dirty,
        "platform": sys.platform,
        "python": platform.python_version(),
        "provider": _provider_from_config(),
        "available_providers": providers,
        "runtime": runtime,
        "hardware": hardware,
        "tracked_file_hashes": _current_file_hashes(),
        "active_work": _active_work(),
    }


def _candidate_file_hashes(reference: str) -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for relative in SENSITIVE_FILES:
        try:
            result = subprocess.run(["git", "cat-file", "blob", f"{reference}:{relative}"],
                                    cwd=str(ROOT), capture_output=True, check=False)
        except OSError:
            result = None
        values[relative] = (_sha256_bytes(result.stdout)
                            if result is not None and result.returncode == 0 else None)
    return values


def _load_candidate_manifest(reference: str) -> dict[str, Any] | None:
    result = _run(["git", "show", f"{reference}:{MANIFEST_PATH}"], check=False)
    if result.returncode:
        return None
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _add_list(value: Any, label: str, unknown: list[str]) -> list[Any] | None:
    if not isinstance(value, list) or any(not isinstance(item, (str, int, float)) for item in value):
        unknown.append(f"manifest field '{label}' is missing or malformed")
        return None
    return value


def evaluate_manifest(manifest: dict[str, Any] | None, candidate_sha: str,
                      current: dict[str, Any], candidate_files: dict[str, str | None] | None = None) -> dict[str, Any]:
    """Classify one immutable candidate using only explicit evidence."""
    reasons: list[str] = []
    review: list[str] = []
    unknown: list[str] = []
    incompatible: list[str] = []
    if not isinstance(manifest, dict):
        return {"classification": "UNVERIFIED", "reasons": [
            f"candidate does not contain a valid {MANIFEST_PATH}"
        ]}

    if manifest.get("schema_version") != 1:
        unknown.append("update manifest schema is missing or unsupported")
    if manifest.get("source_commit") != candidate_sha:
        unknown.append("manifest source_commit does not equal the fetched candidate commit")
    if manifest.get("activation") != "fast_forward_only":
        unknown.append("activation policy is not the verified fast-forward-only mode")

    compatibility = manifest.get("compatibility")
    if not isinstance(compatibility, dict):
        unknown.append("manifest compatibility block is missing")
        compatibility = {}
    platforms = _add_list(compatibility.get("platforms"), "compatibility.platforms", unknown)
    current_platform = str(current.get("platform") or sys.platform)
    if platforms is not None and current_platform not in [str(item) for item in platforms]:
        incompatible.append(f"platform '{current_platform}' is not supported by the candidate")
    python_rules = compatibility.get("python")
    if not isinstance(python_rules, dict):
        unknown.append("Python compatibility range is missing")
    else:
        result = _version_matches(current.get("python"), f">={python_rules.get('min')}" if python_rules.get("min") else None)
        if result is None:
            unknown.append("current Python or candidate minimum Python is unknown")
        elif not result:
            incompatible.append(f"Python {current.get('python')} is below candidate minimum {python_rules.get('min')}")
        if python_rules.get("max"):
            result = _version_matches(current.get("python"), f"<={python_rules['max']}")
            if result is None:
                unknown.append("candidate maximum Python constraint is not evaluable")
            elif not result:
                incompatible.append(f"Python {current.get('python')} exceeds candidate maximum {python_rules['max']}")

    providers = _add_list(compatibility.get("providers"), "compatibility.providers", unknown)
    provider = current.get("provider")
    if providers is not None:
        normalized = {str(item).lower().replace("executionprovider", "") for item in providers}
        if not provider:
            unknown.append("effective execution provider is not known from the current installation")
        elif str(provider).lower().replace("executionprovider", "") not in normalized:
            incompatible.append(f"provider '{provider}' is not supported by the candidate")
    available_providers = current.get("available_providers")
    if not isinstance(available_providers, list):
        unknown.append("ONNX Runtime execution providers could not be enumerated")
    elif not provider:
        unknown.append("effective execution provider is not known from the current installation")
    elif str(provider).lower().replace("executionprovider", "") not in {
            str(item).lower().replace("executionprovider", "") for item in available_providers}:
        incompatible.append(f"configured provider '{provider}' is not available in the current ONNX Runtime")

    hardware_profiles = _add_list(compatibility.get("hardware_profiles"),
                                  "compatibility.hardware_profiles", unknown)
    if hardware_profiles is not None:
        advertised = {str(item) for item in hardware_profiles}
        missing_profiles = MANDATORY_HARDWARE_PROFILES - advertised
        if missing_profiles:
            incompatible.append("candidate does not declare both mandatory GPU profiles: "
                                + ", ".join(sorted(missing_profiles)))
        current_profile = (current.get("hardware") or {}).get("profile")
        if not current_profile:
            unknown.append("current NVIDIA GPU profile could not be verified")
        elif current_profile not in advertised:
            incompatible.append(f"current GPU profile '{current_profile}' is not supported")

    gpu_architectures = _add_list(compatibility.get("gpu_architectures"),
                                  "compatibility.gpu_architectures", unknown)
    if gpu_architectures is not None:
        advertised_architectures = {str(item) for item in gpu_architectures}
        missing_architectures = MANDATORY_GPU_ARCHITECTURES - advertised_architectures
        if missing_architectures:
            incompatible.append("candidate does not declare both mandatory GPU architectures: "
                                + ", ".join(sorted(missing_architectures)))
        current_architecture = str((current.get("hardware") or {}).get("compute_capability") or "")
        if not current_architecture:
            unknown.append("current GPU compute architecture could not be verified")
        elif current_architecture not in advertised_architectures:
            incompatible.append(f"current GPU compute architecture '{current_architecture}' is not supported")

    contract = compatibility.get("application_contract")
    if not isinstance(contract, dict):
        unknown.append("application checkpoint contract is missing")
    else:
        if contract.get("project_schema") != CHECKPOINT_SCHEMA:
            incompatible.append("candidate project checkpoint schema is incompatible")
        if contract.get("processing_contract") != PROCESSING_CONTRACT:
            incompatible.append("candidate processing checkpoint contract is incompatible")

    requirements_policy = compatibility.get("application_requirements")
    if not isinstance(requirements_policy, dict):
        unknown.append("application requirements compatibility policy is missing")
    elif requirements_policy.get("policy") == "unchanged":
        pass
    elif requirements_policy.get("policy") == "review":
        review.append("candidate application requirements require explicit review")
    else:
        unknown.append("application requirements compatibility policy is unknown")

    model_policy = compatibility.get("models")
    if not isinstance(model_policy, dict):
        unknown.append("model compatibility policy is missing")
    elif model_policy.get("policy") == "unchanged":
        pass
    elif model_policy.get("policy") == "review":
        review.append("candidate model compatibility requires explicit review")
    else:
        unknown.append("model compatibility policy is unknown")

    for field, label in (("critical_runtime_changes", "critical runtime changes"),
                         ("dependency_changes", "dependency changes"),
                         ("model_changes", "model changes")):
        values = manifest.get(field)
        if values is None:
            unknown.append(f"manifest field '{field}' is missing")
        elif not isinstance(values, list):
            unknown.append(f"manifest field '{field}' is malformed")
        elif values:
            review.append(f"candidate declares {label}: " + ", ".join(str(item) for item in values))

    runtime_rules = compatibility.get("runtime")
    if not isinstance(runtime_rules, dict):
        unknown.append("runtime compatibility block is missing or malformed")
        runtime_rules = {}
    for name in ("torch", "onnxruntime", "tensorrt", "cuda"):
        if name not in runtime_rules:
            unknown.append(f"runtime compatibility for {name} is not declared")
    for name, requirement in runtime_rules.items():
        actual = (current.get("runtime") or {}).get(str(name))
        result = _version_matches(actual, str(requirement))
        if result is None:
            unknown.append(f"runtime compatibility for {name} cannot be verified (current={actual!r})")
        elif not result:
            incompatible.append(f"current {name} version {actual} does not satisfy {requirement}")

    declared_hashes = manifest.get("tracked_file_hashes")
    if not isinstance(declared_hashes, dict):
        unknown.append("tracked_file_hashes is missing")
    else:
        for relative in SENSITIVE_FILES:
            declared = declared_hashes.get(relative)
            candidate_hash = (candidate_files or {}).get(relative)
            current_hash = (current.get("tracked_file_hashes") or {}).get(relative)
            if not declared or not candidate_hash or declared != candidate_hash:
                unknown.append(f"candidate hash for {relative} is missing or does not match the fetched tree")
            elif current_hash and candidate_hash != current_hash:
                review.append(f"sensitive file changes: {relative}")

    if current.get("dirty"):
        review.append("tracked working-tree changes exist; update activation requires a clean checkout")
    if current.get("active_work"):
        review.append("active or unvalidated processing state exists: "
                      + "; ".join(str(item) for item in current["active_work"]))

    reasons.extend(incompatible)
    reasons.extend(unknown)
    reasons.extend(review)
    if incompatible:
        classification = "INCOMPATIBLE"
    elif unknown:
        classification = "UNVERIFIED"
    elif review:
        classification = "REQUIRES REVIEW"
    else:
        classification = "SAFE"
        reasons.append("candidate satisfies the explicit compatibility manifest and changes no sensitive dependency/model/runtime files")
    return {"classification": classification, "reasons": reasons,
            "manifest": manifest, "candidate_sha": candidate_sha}


def _candidate_report(current: dict[str, Any]) -> dict[str, Any]:
    branch = current.get("branch")
    remote_url = current.get("remote")
    if not branch or branch == "HEAD":
        return {"classification": "UNVERIFIED", "available": False,
                "current": current, "reasons": ["the current checkout is detached; candidate branch is unknown"]}
    if not remote_url:
        return {"classification": "UNVERIFIED", "available": False,
                "current": current, "reasons": ["origin remote is not configured"]}
    remote_name = "origin"
    remote_sha_result = _run(["git", "ls-remote", remote_name, f"refs/heads/{branch}"], check=False)
    if remote_sha_result.returncode:
        return {"classification": "UNVERIFIED", "available": False,
                "current": current, "reasons": ["remote candidate could not be reached or verified"]}
    remote_sha = next((line.split()[0] for line in remote_sha_result.stdout.splitlines()
                       if line.split()), None)
    if not remote_sha or not re.fullmatch(r"[0-9a-f]{40}", remote_sha):
        return {"classification": "UNVERIFIED", "available": False,
                "current": current, "reasons": ["remote did not return an immutable commit SHA"]}
    if remote_sha == current.get("sha"):
        return {"classification": "SAFE", "available": False,
                "current": current, "candidate_sha": remote_sha,
                "reasons": ["no newer commit is available on the configured branch"]}

    fetch_ref = f"+refs/heads/{branch}:refs/remotes/{remote_name}/{branch}"
    fetched = _run(["git", "fetch", "--no-tags", remote_name, fetch_ref], check=False)
    if fetched.returncode:
        return {"classification": "UNVERIFIED", "available": True,
                "current": current, "candidate_sha": remote_sha,
                "reasons": ["candidate commit was found but could not be fetched for manifest verification"]}
    reference = f"{remote_name}/{branch}"
    manifest = _load_candidate_manifest(reference)
    result = evaluate_manifest(manifest, remote_sha, current,
                               _candidate_file_hashes(reference))
    ancestry = _run(["git", "merge-base", "--is-ancestor", current["sha"], reference], check=False)
    if result.get("classification") == "SAFE":
        if ancestry.returncode == 1:
            result["classification"] = "REQUIRES REVIEW"
            result["reasons"].append("candidate is not a fast-forward descendant of the current commit")
        elif ancestry.returncode != 0:
            result["classification"] = "UNVERIFIED"
            result["reasons"].append("candidate ancestry could not be verified")
    result.update({"available": True, "current": current,
                   "candidate_ref": reference})
    return result


def check() -> dict[str, Any]:
    try:
        return _candidate_report(_current_identity())
    except UpdateError as exc:
        return {"classification": "UNVERIFIED", "available": False,
                "reasons": [f"compatibility evidence collection failed: {exc}"]}


def _text_report(report: dict[str, Any]) -> str:
    lines = ["UPDATE COMPATIBILITY CHECK",
             f"Classification: {report.get('classification', 'UNVERIFIED')}"]
    current = report.get("current") or {}
    if current:
        lines.append(f"Current application: {current.get('version', 'UNKNOWN')}")
        lines.append(f"Python: {current.get('python', 'UNKNOWN')}")
        lines.append(f"Provider: {current.get('provider') or 'UNKNOWN'}")
        hardware = current.get("hardware") or {}
        lines.append(f"GPU profile: {hardware.get('profile') or 'UNKNOWN'}")
    if report.get("candidate_sha"):
        lines.append(f"Candidate commit: {report['candidate_sha']}")
    lines.append("Reasons:")
    lines.extend(f"- {reason}" for reason in report.get("reasons", []))
    if report.get("classification") != "SAFE" or report.get("available"):
        lines.append("No update was installed unless the candidate was explicitly manifest-gated SAFE.")
    return "\n".join(lines)


def _transaction(phase: str, **values: Any) -> dict[str, Any]:
    state = {
        "schema_version": 1,
        "phase": phase,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    state.update(values)
    _atomic_json(TRANSACTION_PATH, state)
    return state


def _create_snapshot(current: dict[str, Any]) -> Path:
    """Capture the reversible part of the current installation.

    Git owns tracked source, so the backup ref is the authoritative source
    snapshot.  The ignored user configuration is copied separately. Models,
    environments, outputs, and project data are intentionally not copied: an
    active persisted job blocks admission and those artifacts need their own
    future content-addressed backup contract.
    """
    sha = str(current.get("sha") or "")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise UpdateError("cannot snapshot an invalid current commit SHA")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot = SNAPSHOT_ROOT / f"{stamp}-{sha[:12]}"
    snapshot.mkdir(parents=True, exist_ok=False)
    backup_ref = f"refs/roop-update-backups/{stamp}-{sha[:12]}"
    _run(["git", "update-ref", backup_ref, sha])
    files: dict[str, Any] = {}
    config = ROOT / "app" / "config.yaml"
    if config.is_file():
        _atomic_copy(config, snapshot / "config.yaml")
        files["app/config.yaml"] = {
            "sha256": _sha256_file(config),
            "snapshot": "config.yaml",
        }
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "SNAPSHOT_READY",
        "prior_commit": sha,
        "backup_ref": backup_ref,
        "current_identity": current,
        "files": files,
        "artifacts_not_copied": [
            "app/env",
            "app/models",
            "app/output",
            "app/projects",
            "app/queue.json",
        ],
        "artifact_policy": "active work is admission-blocking; model, environment, output, and project content are not rollback-copied",
    }
    _atomic_json(snapshot / "snapshot.json", metadata)
    return snapshot


def _record_snapshot(snapshot: Path, **values: Any) -> None:
    metadata_path = snapshot / "snapshot.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise UpdateError(f"snapshot metadata could not be reopened: {exc}") from exc
    metadata.update(values)
    _atomic_json(metadata_path, metadata)


def _run_health(source_root: Path, data_root: Path, skip_launch: bool = False) -> dict[str, Any]:
    script = source_root / "app" / "update_health.py"
    if not script.is_file():
        return {"healthy": False, "checks": [{
            "name": "health-worker",
            "ok": False,
            "detail": f"health worker is missing: {script}",
        }]}
    command = [sys.executable, str(script), "--source-root", str(source_root),
               "--data-root", str(data_root), "--json"]
    if skip_launch:
        command.append("--skip-launch")
    result = _run(command, cwd=source_root, check=False, timeout=HEALTH_TIMEOUT_SECONDS)
    try:
        report = json.loads(result.stdout)
        if not isinstance(report, dict):
            raise ValueError("health worker returned a non-object JSON value")
    except (ValueError, json.JSONDecodeError) as exc:
        report = {"healthy": False, "checks": [{
            "name": "health-worker",
            "ok": False,
            "detail": f"health worker returned invalid JSON: {exc}",
            "stdout": result.stdout[-4000:],
        }]}
    report["returncode"] = result.returncode
    if result.stderr:
        report["stderr"] = result.stderr[-12000:]
    report["stdout"] = result.stdout[-12000:]
    if result.returncode != 0:
        report["healthy"] = False
    return report


def _stage_candidate(reference: str, candidate_sha: str, data_root: Path) -> dict[str, Any]:
    """Validate a candidate checkout before changing the active worktree."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    staging = STAGING_ROOT / f"{stamp}-{candidate_sha[:12]}"
    staging.parent.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        raise UpdateError(f"staging directory already exists: {staging}")
    _run(["git", "worktree", "add", "--detach", str(staging), reference], timeout=180)
    try:
        diff_check = _run(["git", "diff", "--check", str(_git("rev-parse", "HEAD")), reference],
                          check=False, timeout=180)
        if diff_check.returncode:
            raise UpdateError(f"candidate diff has whitespace errors: {diff_check.stderr or diff_check.stdout}")
        health_worker = staging / "app" / "update_health.py"
        if not health_worker.is_file():
            raise UpdateError("candidate does not contain the required runtime health worker")
        compile_check = _run([sys.executable, "-m", "compileall", "-q", str(staging / "app")],
                             cwd=staging, check=False, timeout=180)
        if compile_check.returncode:
            raise UpdateError(f"candidate Python compilation failed: {compile_check.stderr or compile_check.stdout}")
        health = _run_health(staging, data_root, skip_launch=True)
        if not health.get("healthy"):
            raise UpdateError("staged candidate failed pre-activation runtime checks")
        return {"staging": str(staging), "health": health}
    finally:
        removed = _run(["git", "worktree", "remove", "--force", str(staging)], check=False, timeout=180)
        if removed.returncode and staging.exists():
            # The directory is an updater-owned temporary checkout.  Never
            # remove a path outside STAGING_ROOT if Git left it behind.
            try:
                staging.resolve().relative_to(STAGING_ROOT.resolve())
                shutil.rmtree(staging)
            except (OSError, ValueError):
                pass


def _rollback(snapshot: Path, prior_sha: str) -> dict[str, Any]:
    """Restore tracked source and the copied config, then re-health-check it."""
    tracked_changes = _git("status", "--porcelain", "--untracked-files=no", check=False)
    if tracked_changes:
        return {"ok": False,
                "detail": "tracked files changed during failure; refusing destructive rollback",
                "tracked_changes": tracked_changes}
    _run(["git", "reset", "--hard", prior_sha], timeout=180)
    head = _git("rev-parse", "HEAD")
    if head != prior_sha:
        return {"ok": False, "detail": f"rollback stopped at unexpected commit {head}"}
    metadata = json.loads((snapshot / "snapshot.json").read_text(encoding="utf-8"))
    config_record = (metadata.get("files") or {}).get("app/config.yaml")
    config = ROOT / "app" / "config.yaml"
    config_warning = None
    if config_record and (snapshot / str(config_record.get("snapshot"))).is_file():
        if config.is_file() and _sha256_file(config) != config_record.get("sha256"):
            config_warning = "configuration changed during the transaction and was not overwritten"
        else:
            _atomic_copy(snapshot / str(config_record["snapshot"]), config)
    health = _run_health(ROOT, ROOT, skip_launch=False)
    ok = bool(health.get("healthy")) and not config_warning
    return {"ok": ok, "detail": config_warning or "prior source and configuration restored",
            "health": health, "head": head}


def apply() -> int:
    report = check()
    print(_text_report(report))
    if not report.get("available"):
        return 0 if report.get("classification") == "SAFE" else 2
    if report.get("classification") != "SAFE":
        return 2
    reference = report.get("candidate_ref")
    if not reference:
        print("No update installed: candidate reference is unavailable.")
        return 2
    current = report.get("current") or {}
    prior_sha = str(current.get("sha") or "")
    snapshot: Path | None = None
    activated = False
    phase = "PREFLIGHT"
    try:
        _transaction(phase, current_sha=prior_sha, candidate_sha=report.get("candidate_sha"))
        pre_health = _run_health(ROOT, ROOT, skip_launch=False)
        if not pre_health.get("healthy"):
            failure_dir = SNAPSHOT_ROOT / f"failed-preflight-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            failure_dir.mkdir(parents=True, exist_ok=True)
            _atomic_json(failure_dir / "health.json", pre_health)
            _transaction("FAILED", failure=str(failure_dir / "health.json"),
                         reason="prior installation did not pass health validation")
            print("Update not installed: the prior installation is not healthy; diagnostics were captured at "
                  f"{failure_dir / 'health.json'}.")
            return 3
        phase = "SNAPSHOTTING"
        _transaction(phase, current_sha=prior_sha, candidate_sha=report.get("candidate_sha"))
        snapshot = _create_snapshot(current)
        _transaction("STAGING", current_sha=prior_sha,
                     candidate_sha=report.get("candidate_sha"), snapshot=str(snapshot))
        staged = _stage_candidate(reference, str(report["candidate_sha"]), ROOT)
        _record_snapshot(snapshot, staged_health=staged["health"])
        _transaction("ACTIVATING", current_sha=prior_sha,
                     candidate_sha=report.get("candidate_sha"), snapshot=str(snapshot))
        if _git("rev-parse", "HEAD") != prior_sha or _git("status", "--porcelain", "--untracked-files=no"):
            raise UpdateError("active checkout changed while the candidate was being staged")
        _run(["git", "merge", "--ff-only", reference])
        activated = True
        _transaction("POST_UPDATE_HEALTH", current_sha=prior_sha,
                     candidate_sha=report.get("candidate_sha"), snapshot=str(snapshot))
        post_health = _run_health(ROOT, ROOT, skip_launch=False)
        _record_snapshot(snapshot, post_health=post_health,
                         status="HEALTHY" if post_health.get("healthy") else "POST_UPDATE_FAILED")
        if not post_health.get("healthy"):
            raise UpdateError("post-update runtime health validation failed")
        _transaction("HEALTHY", current_sha=prior_sha,
                     candidate_sha=report.get("candidate_sha"), snapshot=str(snapshot),
                     active_sha=_git("rev-parse", "HEAD"))
    except UpdateError as exc:
        detail = str(exc)
        rollback = None
        if snapshot is not None:
            try:
                current_head = _git("rev-parse", "HEAD", check=False)
                if activated or current_head != prior_sha:
                    _transaction("ROLLING_BACK", current_sha=prior_sha,
                                 candidate_sha=report.get("candidate_sha"), snapshot=str(snapshot),
                                 failure=detail)
                    rollback = _rollback(snapshot, prior_sha)
                    _record_snapshot(snapshot, rollback=rollback,
                                    status="ROLLED_BACK" if rollback.get("ok") else "ROLLBACK_FAILED")
                    _transaction("ROLLED_BACK" if rollback.get("ok") else "ROLLBACK_FAILED",
                                 current_sha=prior_sha, candidate_sha=report.get("candidate_sha"),
                                 snapshot=str(snapshot), failure=detail)
            except Exception as rollback_exc:
                rollback = {"ok": False, "detail": f"rollback failed: {rollback_exc}"}
                try:
                    _record_snapshot(snapshot, rollback=rollback, status="ROLLBACK_FAILED")
                    _transaction("ROLLBACK_FAILED", current_sha=prior_sha,
                                 candidate_sha=report.get("candidate_sha"), snapshot=str(snapshot),
                                 failure=detail)
                except Exception:
                    pass
        print(f"Update failed: {detail}. Diagnostics and transaction state were captured.")
        if rollback is not None:
            print(f"Rollback: {'succeeded' if rollback.get('ok') else 'FAILED'} — {rollback.get('detail')}")
        return 3
    except Exception as exc:
        print(f"Update failed with an unexpected error: {exc}")
        return 3
    print("Update applied and runtime health validated: source-only fast-forward. "
          "Dependencies and models were not changed.")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "apply"))
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = check() if args.command == "check" else None
    if args.command == "check":
        print(json.dumps(report, indent=2, sort_keys=True) if args.as_json else _text_report(report))
        return 0 if report.get("classification") == "SAFE" else 2
    return apply()


if __name__ == "__main__":
    raise SystemExit(main())
