"""Explicit ONNX Runtime import, namespace, and environment consistency checks.

The detector runs before provider enumeration. It rejects PEP 420 namespace
packages, local shadow files, duplicate package roots, stale bytecode-only
artifacts, imports outside the active interpreter prefix, and distribution
metadata that does not own the imported module.
"""
from __future__ import annotations

import glob
import importlib
import importlib.metadata
import os
import sys
from dataclasses import dataclass, field
from types import ModuleType
from typing import Iterable


ORT_DISTRIBUTIONS = (
    "onnxruntime",
    "onnxruntime-gpu",
    "onnxruntime-directml",
    "onnxruntime-rocm",
    "onnxruntime-silicon",
)


@dataclass
class ORTInspection:
    module_path: str | None = None
    spec_origin: str | None = None
    distribution_name: str | None = None
    distribution_version: str | None = None
    distribution_location: str | None = None
    module_version: str | None = None
    namespace_paths: list[str] = field(default_factory=list)
    shadow_paths: list[str] = field(default_factory=list)
    package_roots: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors


class ORTInspectionError(RuntimeError):
    """The imported ORT module is not a real package in the active venv."""


def _normalise(path: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def _within(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([_normalise(path), _normalise(root)]) == _normalise(root)
    except ValueError:
        return False


def _roots(extra_roots: Iterable[str] = ()) -> list[str]:
    values: list[str] = []
    for value in [*sys.path, *extra_roots]:
        root = value or os.getcwd()
        if not os.path.isdir(root):
            continue
        resolved = _normalise(root)
        if resolved not in values:
            values.append(resolved)
    return values


def scan_shadow_paths(extra_roots: Iterable[str] = ()) -> tuple[list[str], list[str], list[str]]:
    """Return namespace directories, shadow artifacts, and package roots.

    This scan does not import ORT. It checks every import root, not just the
    current directory, so a stale checkout or an earlier sys.path entry cannot
    hide behind a valid site-packages installation.
    """
    namespace_paths: list[str] = []
    shadow_paths: list[str] = []
    package_roots: list[str] = []
    for root in _roots(extra_roots):
        module_file = os.path.join(root, "onnxruntime.py")
        if os.path.isfile(module_file):
            shadow_paths.append(module_file)
        for bytecode in glob.glob(os.path.join(root, "onnxruntime.pyc")):
            shadow_paths.append(bytecode)
        for bytecode in glob.glob(os.path.join(root, "__pycache__", "onnxruntime*.pyc")):
            shadow_paths.append(bytecode)

        package = os.path.join(root, "onnxruntime")
        if not os.path.isdir(package):
            continue
        package_roots.append(package)
        init_file = os.path.join(package, "__init__.py")
        if not os.path.isfile(init_file):
            namespace_paths.append(package)
    return namespace_paths, shadow_paths, package_roots


def _distribution() -> tuple[str | None, importlib.metadata.Distribution | None, list[str]]:
    found: list[tuple[str, importlib.metadata.Distribution]] = []
    for name in ORT_DISTRIBUTIONS:
        try:
            found.append((name, importlib.metadata.distribution(name)))
        except importlib.metadata.PackageNotFoundError:
            continue
    if len(found) == 1:
        return found[0][0], found[0][1], []
    return None, None, [
        "expected exactly one ORT distribution, found "
        + (", ".join(name for name, _dist in found) if found else "none")
    ]


def inspect_onnxruntime(
    *,
    module: ModuleType | None = None,
    extra_roots: Iterable[str] = (),
) -> ORTInspection:
    report = ORTInspection()
    namespace_paths, shadow_paths, package_roots = scan_shadow_paths(extra_roots)
    report.namespace_paths = namespace_paths
    report.shadow_paths = shadow_paths
    report.package_roots = package_roots

    if namespace_paths:
        report.errors.append(
            "onnxruntime package directory is missing __init__.py: "
            + ", ".join(namespace_paths)
        )
    if shadow_paths:
        report.errors.append(
            "shadowed ONNX Runtime artifacts found on import paths: "
            + ", ".join(shadow_paths)
        )

    if module is None:
        try:
            module = importlib.import_module("onnxruntime")
        except Exception as exc:
            report.errors.append(f"onnxruntime import failed: {exc}")
            return report

    report.module_path = getattr(module, "__file__", None)
    report.module_version = getattr(module, "__version__", None)
    spec = getattr(module, "__spec__", None)
    report.spec_origin = getattr(spec, "origin", None) if spec is not None else None

    if report.module_path is None:
        report.errors.append("onnxruntime.__file__ is None")
    if spec is None:
        report.errors.append("onnxruntime.__spec__ is None")
    if report.spec_origin is None:
        report.errors.append("onnxruntime.__spec__.origin is None")

    module_search_paths = [str(path) for path in (getattr(module, "__path__", None) or [])]
    if module_search_paths:
        missing_inits = [
            path for path in module_search_paths
            if not os.path.isfile(os.path.join(path, "__init__.py"))
        ]
        if missing_inits:
            report.namespace_paths.extend(path for path in missing_inits
                                         if path not in report.namespace_paths)
            report.errors.append(
                "onnxruntime.__path__ contains implicit namespace locations: "
                + ", ".join(missing_inits)
            )

    if report.module_path and not _within(report.module_path, sys.prefix):
        report.errors.append(
            f"imported onnxruntime is outside active sys.prefix: {report.module_path}"
        )

    name, distribution, distribution_errors = _distribution()
    report.errors.extend(distribution_errors)
    if distribution is not None and name is not None:
        report.distribution_name = name
        report.distribution_version = distribution.version
        location = str(distribution.locate_file(""))
        report.distribution_location = location
        if not _within(location, sys.prefix):
            report.errors.append(
                f"onnxruntime distribution is outside active sys.prefix: {location}"
            )
        if report.module_path and not _within(report.module_path, location):
            report.errors.append(
                "package metadata exists but imported onnxruntime resolves elsewhere: "
                + report.module_path
            )
        expected_module = str(distribution.locate_file("onnxruntime/__init__.py"))
        expected_root = os.path.dirname(expected_module)
        foreign_roots = [
            root for root in report.package_roots
            if _normalise(root) != _normalise(expected_root)
        ]
        if foreign_roots:
            report.shadow_paths.extend(
                root for root in foreign_roots if root not in report.shadow_paths
            )
            report.errors.append(
                "duplicate or shadowed onnxruntime package roots found: "
                + ", ".join(foreign_roots)
            )
        if report.module_path and _normalise(report.module_path) != _normalise(expected_module):
            report.errors.append(
                "onnxruntime distribution metadata does not point to imported module: "
                + f"metadata={expected_module}, imported={report.module_path}"
            )
        if report.module_version and report.module_version != distribution.version:
            report.errors.append(
                f"onnxruntime version mismatch: module={report.module_version}, "
                f"distribution={distribution.version}"
            )

    if report.errors:
        raise ORTInspectionError("; ".join(dict.fromkeys(report.errors)))
    return report
