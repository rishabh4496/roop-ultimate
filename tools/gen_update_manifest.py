"""Render update_manifest.json from the committed dependency/runtime contract.

    python tools/gen_update_manifest.py                 # (re)write update_manifest.json from the index
    python tools/gen_update_manifest.py --check         # exit 1 if HEAD's committed manifest is missing or stale
    python tools/gen_update_manifest.py --check --ref origin/main

The manifest is what app/update_manager.py evaluates before it fast-forwards an
installation. It is a pure function of the committed contract and is never
hand-written:

  compatibility.runtime       the torch / onnxruntime / tensorrt / CUDA pins in
                              app/provision_runtime.py's NVIDIA branch
  compatibility.providers     the ONNX Runtime wheels provision_runtime.py installs
  tracked_file_hashes         sha256 of every update_manager.SENSITIVE_FILES blob
  hardware_profiles, gpu_architectures, application_contract
                              the constants update_manager.py enforces
  platforms, python           the support matrix below (PLATFORMS / PYTHON)

Identity: the checker verifies each tracked_file_hashes entry against the
fetched candidate tree, so the manifest binds to the commit's dependency
contract. It does NOT record the commit SHA: a committed file cannot contain
the hash of the commit that contains it (schema 1 asked for exactly that, and
no commit ever satisfied it). A commit that changes none of the sensitive
files keeps its manifest valid through merges, rebases and squashes; one that
does change them must regenerate, which .githooks/pre-commit does and CI's
--check enforces.

Write mode reads blobs from the git INDEX -- what `git commit` is about to
record -- so the manifest matches the commit, not a half-staged working tree.
Check mode reads the manifest and the blobs from the given ref through git,
exactly as the updater reads the fetched candidate.
app/tests/test_update_manifest_head.py runs this in --check mode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app")
for _path in (ROOT, APP):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from app import update_manager  # noqa: E402

MANIFEST_PATH = update_manager.MANIFEST_PATH
PROVISION_PATH = "app/provision_runtime.py"
# Support matrix. provision_runtime.py has an install branch for each of these
# platforms; ci.yml runs the suite on 3.10 and numpy 1.26.4 (requirements.txt)
# has no wheels past 3.12.
PLATFORMS = ["win32", "linux", "darwin"]
PYTHON = {"min": "3.10", "max": "3.12"}
# ONNX Runtime wheel -> execution providers it ships, in the lowercase form
# update_manager normalizes to (the "ExecutionProvider" suffix stripped).
ORT_WHEEL_PROVIDERS = {
    "onnxruntime-gpu": ["tensorrt", "cuda"],
    "onnxruntime-directml": ["dml"],
    "onnxruntime-rocm": ["rocm"],
    "onnxruntime_rocm": ["rocm"],
    "onnxruntime-silicon": ["coreml"],
    "onnxruntime": [],
}


class ManifestError(RuntimeError):
    pass


def _git(*args: str, binary: bool = False) -> bytes | str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=False)
    if result.returncode:
        raise ManifestError(f"git {' '.join(args)} failed: "
                            f"{result.stderr.decode('utf-8', 'replace').strip()}")
    return result.stdout if binary else result.stdout.decode("utf-8")


def _blob(source: str, relative: str) -> bytes | None:
    """The bytes of `relative` in the index (source="index") or at a git ref."""
    spec = f":{relative}" if source == "index" else f"{source}:{relative}"
    result = subprocess.run(["git", "cat-file", "blob", spec], cwd=ROOT,
                            capture_output=True, check=False)
    return result.stdout if result.returncode == 0 else None


def _only(values: list[str], label: str) -> str:
    distinct = sorted(set(values))
    if len(distinct) != 1:
        raise ManifestError(f"{PROVISION_PATH}: expected exactly one {label}, found {distinct or 'none'}")
    return distinct[0]


def runtime_contract(provision_source: str) -> dict[str, str]:
    """The NVIDIA runtime pins, as the exact constraints the checker evaluates."""
    match = re.search(r"def _install_nvidia_runtime\(.*?(?=\ndef |\Z)", provision_source, re.S)
    if not match:
        raise ManifestError(f"{PROVISION_PATH}: _install_nvidia_runtime() not found")
    block = match.group(0)
    torch = _only(re.findall(r'"torch==([^"\s]+)"', block), "torch pin")
    cuda_tag = _only(re.findall(r"download\.pytorch\.org/whl/cu(\d+)", block), "CUDA wheel index")
    # cu128 -> 12.8: torch.version.cuda is "<major>.<minor>".
    cuda = f"{cuda_tag[:2]}.{cuda_tag[2:]}"
    onnxruntime = _only(re.findall(r'"onnxruntime-gpu==([^"\s]+)"', block), "onnxruntime-gpu pin")
    tensorrt = _only(re.findall(r'\{package\}==([\d.]+)"', block), "tensorrt pin")
    return {
        # torch reports "<version>+cu<tag>" from the CUDA wheel index.
        "torch": f"=={torch}+cu{cuda_tag}",
        "onnxruntime": f"=={onnxruntime}",
        "tensorrt": f"=={tensorrt}",
        "cuda": f"=={cuda}",
    }


def providers(provision_source: str) -> list[str]:
    """Every provider an ORT wheel provision_runtime.py installs can expose."""
    wheels = set(re.findall(r'_install_onnxruntime\(\s*"(onnxruntime[\w-]*)', provision_source))
    wheels |= set(re.findall(r"(onnxruntime_rocm)-", provision_source))
    found: list[str] = []
    for wheel in sorted(wheels):
        if wheel not in ORT_WHEEL_PROVIDERS:
            raise ManifestError(f"{PROVISION_PATH}: unknown ONNX Runtime wheel {wheel!r}; "
                                f"add it to ORT_WHEEL_PROVIDERS")
        found.extend(ORT_WHEEL_PROVIDERS[wheel])
    ordered = [name for name in ("tensorrt", "cuda", "dml", "rocm", "coreml") if name in found]
    return ordered + ["cpu"]


def build_manifest(source: str = "index") -> dict:
    provision = _blob(source, PROVISION_PATH)
    if provision is None:
        raise ManifestError(f"{PROVISION_PATH} is not in {source}")
    provision_source = provision.decode("utf-8")
    hashes: dict[str, str] = {}
    for relative in update_manager.SENSITIVE_FILES:
        blob = _blob(source, relative)
        if blob is None:
            raise ManifestError(f"sensitive file {relative} is not in {source}")
        hashes[relative] = hashlib.sha256(blob).hexdigest()
    return {
        "schema_version": update_manager.MANIFEST_SCHEMA_VERSION,
        "generated_by": "tools/gen_update_manifest.py",
        "activation": "fast_forward_only",
        "compatibility": {
            "platforms": list(PLATFORMS),
            "python": dict(PYTHON),
            "providers": providers(provision_source),
            "hardware_profiles": sorted(update_manager.MANDATORY_HARDWARE_PROFILES),
            "gpu_architectures": sorted(update_manager.MANDATORY_GPU_ARCHITECTURES),
            "application_contract": {
                "project_schema": update_manager.CHECKPOINT_SCHEMA,
                "processing_contract": update_manager.PROCESSING_CONTRACT,
            },
            # Review of dependency/runtime changes is driven by the checker's
            # per-installation tracked_file_hashes comparison, not by a
            # declaration here; a generated manifest has nothing to declare.
            "application_requirements": {"policy": "unchanged"},
            "models": {"policy": "unchanged"},
            "runtime": runtime_contract(provision_source),
        },
        "critical_runtime_changes": [],
        "dependency_changes": [],
        "model_changes": [],
        "tracked_file_hashes": hashes,
    }


def render(manifest: dict) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def check(ref: str = "HEAD") -> list[str]:
    """Problems with the manifest committed at `ref`; empty means current."""
    expected = build_manifest(ref)
    committed = _blob(ref, MANIFEST_PATH)
    if committed is None:
        return [f"{MANIFEST_PATH} is not committed at {ref}; run: python tools/gen_update_manifest.py"]
    try:
        actual = json.loads(committed.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        return [f"{MANIFEST_PATH} at {ref} is not valid JSON: {exc}"]
    if actual == expected:
        return []
    problems = [f"{MANIFEST_PATH} at {ref} is stale; run: python tools/gen_update_manifest.py"]
    for key in sorted(set(expected) | set(actual)):
        if expected.get(key) != actual.get(key):
            problems.append(f"  {key}: committed {json.dumps(actual.get(key), sort_keys=True)}"
                            f" != expected {json.dumps(expected.get(key), sort_keys=True)}")
    return problems


def _unstaged_sensitive_changes() -> list[str]:
    changed = _git("diff", "--name-only", "--", *update_manager.SENSITIVE_FILES).split()
    return [path for path in changed if path in update_manager.SENSITIVE_FILES]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="verify the manifest committed at --ref instead of writing")
    parser.add_argument("--ref", default="HEAD", help="git ref to check (default HEAD)")
    args = parser.parse_args(argv)
    try:
        if args.check:
            problems = check(args.ref)
            if problems:
                print("\n".join(problems), file=sys.stderr)
                return 1
            print(f"{MANIFEST_PATH} at {args.ref} matches its dependency contract")
            return 0
        manifest = build_manifest("index")
        path = os.path.join(ROOT, MANIFEST_PATH)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(render(manifest))
        print(f"wrote {MANIFEST_PATH} from the index")
        unstaged = _unstaged_sensitive_changes()
        if unstaged:
            print("note: unstaged changes in " + ", ".join(unstaged)
                  + " are not in the manifest; stage them and re-run", file=sys.stderr)
        return 0
    except ManifestError as exc:
        print(f"gen_update_manifest: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
