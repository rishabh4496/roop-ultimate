#!/usr/bin/env python3
"""Pre-flight for ``start_legacy.js``: say the Gradio UI is deprecated, and fail loudly if it cannot start.

The Gradio interface under ``app/ui/`` is frozen (``app/README.md``); ``start_react.js`` is the supported launcher
and the only one that gets new features. What actually breaks the legacy UI as the backend moves on is not an HTTP
endpoint -- Gradio runs in-process and calls no route -- it is the ``Settings`` object: ``ui/*`` reads
``roop.globals.CFG.<name>`` while it builds its widgets, and deleting one of those attributes stopped the launch once
(docs/CHANGELOG.md, 2026-10-21, item 7: ``keep_frames``, ``skip_audio``, ``video_swapping_method``,
``wait_after_extraction``). Without this check the failure is a Gradio traceback after several minutes of model
loading; with it, it is one ``[LEGACY-UI] FAIL`` line before anything loads.

Run from ``app/`` by Pinokio (``python legacy_ui_check.py``). Exit 0 = the legacy UI can start; 1 = it cannot. The
deprecation notice is printed either way. ``start_legacy.js`` breaks on the ``[LEGACY-UI] FAIL`` marker.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import sys
import tempfile

APP = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(APP, "ui")

DEPRECATION = (
    "[LEGACY-UI] DEPRECATED: this Gradio interface is frozen and receives no new features. "
    "Everything built since the workstation overhaul (multi-track timeline, identity morph lab, relighting, "
    "roto zones, crash recovery console, TensorRT receipts) exists only in the React UI. "
    "Use 'Start' (start_react.js) instead."
)


def settings_attributes_read_by_ui(ui_dir: str = UI_DIR) -> dict[str, set[str]]:
    """``{attribute: {files}}`` for every ``CFG.<attribute>`` LOAD under ``ui/``.

    Stores (``CFG.provider_active = ...``) are not reads and are skipped: the UI is allowed to create those.
    """
    reads: dict[str, set[str]] = {}
    for dirpath, _dirs, files in os.walk(ui_dir):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            with open(path, encoding="utf-8") as fh:
                tree = ast.parse(fh.read(), filename=path)
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)):
                    continue
                owner = node.value
                is_cfg = (isinstance(owner, ast.Name) and owner.id == "CFG") or \
                         (isinstance(owner, ast.Attribute) and owner.attr == "CFG")
                if is_cfg:
                    reads.setdefault(node.attr, set()).add(os.path.relpath(path, ui_dir))
    return reads


def find_issues(app_dir: str = APP) -> list[str]:
    """Every reason the legacy UI cannot start, as one line each; empty when it can."""
    issues: list[str] = []
    ui_dir = os.path.join(app_dir, "ui")
    if not os.path.isfile(os.path.join(ui_dir, "main.py")):
        issues.append("app/ui/main.py is missing, so there is no Gradio UI to launch")
        return issues
    if importlib.util.find_spec("gradio") is None:
        issues.append("the 'gradio' package is not installed in app/env (re-run Install, or use the React UI)")
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    try:
        import settings
        with tempfile.TemporaryDirectory() as tmp:
            current = set(dir(settings.Settings(os.path.join(tmp, "config.yaml"))))
    except Exception as exc:                                 # a broken settings.py is itself the answer
        issues.append("Settings could not be constructed: %s: %s" % (type(exc).__name__, exc))
        return issues
    for attr, files in sorted(settings_attributes_read_by_ui(ui_dir).items()):
        if attr not in current:
            issues.append("ui/%s reads CFG.%s, which Settings no longer defines" % (sorted(files)[0], attr))
    return issues


def main() -> int:
    print(DEPRECATION, flush=True)
    issues = find_issues()
    for line in issues:
        print("[LEGACY-UI] FAIL: %s" % line, flush=True)
    if issues:
        print("[LEGACY-UI] The legacy UI will not start. The React UI is unaffected.", flush=True)
        return 1
    print("[LEGACY-UI] OK: every setting the Gradio UI reads still exists.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
