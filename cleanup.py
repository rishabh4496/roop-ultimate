"""Disk cleanup for the Roop Ultimate launcher (driven by clean.js).

Only ever touches things the app can rebuild by itself. Everything that would
cost you work or a re-download is off-limits and is never referenced here:

    app/models/       the model library (~23GB) - re-downloading is the only way back
    app/env/          the virtualenv - use "Save Disk Space" (link.js) instead
    app/output/       YOUR RENDERED VIDEOS - never touched by this script
    app/facesets/     your saved faceset library
    react-ui/node_modules, source, .git

Run with no arguments for a report; pass one or more target names to delete.
"""

import argparse
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(ROOT, "app")


def _size(path):
    """Bytes used by a file or directory tree. Unreadable entries count as 0."""
    if os.path.isfile(path):
        try:
            return os.path.getsize(path)
        except OSError:
            return 0
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path, onerror=lambda e: None):
        for fn in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, fn))
            except OSError:
                pass
    return total


def _fmt(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0


def _trt_precision():
    """The precision the app is configured for. Its caches are the LIVE ones.

    Read straight out of config.yaml rather than importing settings.py, so the
    cleanup never needs the venv or drags in torch. Falls back to the same
    default settings.py uses, which errs towards keeping more.
    """
    cfg = os.path.join(APP, "config.yaml")
    try:
        with open(cfg, "r", encoding="utf-8") as fh:
            for line in fh:
                key, _, value = line.partition(":")
                if key.strip() == "trt_precision":
                    v = value.strip().strip("'\"")
                    if v:
                        return v
    except OSError:
        pass
    return "mixed"


def _pycache_dirs():
    out = []
    for base in (APP, os.path.join(ROOT, "react-ui", "src")):
        for dirpath, dirnames, _f in os.walk(base, onerror=lambda e: None):
            # Don't descend into the venv's site-packages: those .pyc files are
            # regenerated per import and clearing them only slows the next start.
            dirnames[:] = [d for d in dirnames if d not in ("env", "node_modules", ".git")]
            if "__pycache__" in dirnames:
                out.append(os.path.join(dirpath, "__pycache__"))
    return out


def _trt_dirs(stale_only):
    """TensorRT engine caches. `stale_only` keeps the ones the current
    precision setting actually uses (see Enhance_GPEN / FaceSwapInsightFace,
    which append _gpen_fp32 / _swap_fp32 to the precision folder)."""
    root = os.path.join(APP, "models", "trt_cache")
    if not os.path.isdir(root):
        return []
    live = set()
    if stale_only:
        p = _trt_precision()
        live = {p, p + "_swap_fp32", p + "_gpen_fp32"}
    return [os.path.join(root, d) for d in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, d)) and d not in live]


def _old_logs(keep=5):
    """Every log directory keeps its newest `keep` runs; older ones go.

    'latest' and 'events' are never candidates — they are what the app and the
    troubleshooting docs point at.
    """
    out = []
    base = os.path.join(ROOT, "logs")
    for dirpath, _dirnames, filenames in os.walk(base, onerror=lambda e: None):
        runs = []
        for fn in filenames:
            if fn in ("latest", "events"):
                continue
            full = os.path.join(dirpath, fn)
            try:
                runs.append((os.path.getmtime(full), full))
            except OSError:
                pass
        runs.sort(reverse=True)
        out.extend(full for _mt, full in runs[keep:])
    return out


TARGETS = {
    "uploads": (
        "Uploaded media scratch",
        "Copies of the images and videos you dragged in, plus unpacked facesets. "
        "The app re-creates them on the next upload. Your rendered output is NOT here.",
        lambda: [p for p in (os.path.join(APP, "temp", "api_uploads"),
                             os.path.join(APP, "temp", "faceset"))
                 if os.path.exists(p)],
    ),
    "pycache": (
        "Python bytecode caches",
        "__pycache__ folders under app/. Python rebuilds them automatically; the "
        "first run afterwards is a second or two slower.",
        _pycache_dirs,
    ),
    "trt-stale": (
        "Stale TensorRT engine caches",
        "Engine caches built for a precision mode you are no longer using. The "
        "caches for the CURRENT setting are kept, so runs stay fast.",
        lambda: _trt_dirs(stale_only=True),
    ),
    "trt-all": (
        "ALL TensorRT engine caches",
        "Frees the most space, but every model has to recompile its engine on the "
        "next run - minutes of extra startup, once per model. Nothing breaks.",
        lambda: _trt_dirs(stale_only=False),
    ),
    "logs": (
        "Old run logs",
        "Keeps the newest 5 log files per folder plus 'latest'. Older ones are only "
        "useful for historical debugging.",
        _old_logs,
    ),
    "build": (
        "Front-end build output",
        "react-ui/dist and lint caches. The UI is served by the Vite dev server, so "
        "this is rebuilt on demand and unused at runtime.",
        lambda: [p for p in (os.path.join(ROOT, "react-ui", "dist"),
                             os.path.join(ROOT, ".ruff_cache"),
                             os.path.join(APP, ".ruff_cache"))
                 if os.path.exists(p)],
    ),
}

# trt-all is a superset of trt-stale; running both would double-count the report
# and make the second one a no-op.
EXCLUSIVE = ("trt-all", "trt-stale")


def _remove(path):
    """Returns (freed_bytes, error_or_None). Locked files are reported, not raised:
    the app holds its uploads open while running, and one locked file must not
    abort the rest of the cleanup."""
    size = _size(path)
    try:
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        return size, None
    except OSError as e:
        remaining = _size(path) if os.path.exists(path) else 0
        return max(0, size - remaining), str(e)


def report():
    print("\n=== Reclaimable disk space ===\n")
    total = 0
    for key, (title, desc, finder) in TARGETS.items():
        paths = finder()
        size = sum(_size(p) for p in paths)
        if key != "trt-all":
            total += size
        n = len(paths)
        print(f"  [{key}] {title}")
        print(f"      {_fmt(size):>10}   ({n} item{'s' if n != 1 else ''})")
        print(f"      {desc}")
        print()
    print(f"  Safe total (excluding 'ALL TensorRT caches'): {_fmt(total)}")
    print("\n  Never touched: models, virtualenv, YOUR OUTPUT VIDEOS, saved facesets.\n")


def clean(keys):
    if "trt-all" in keys and "trt-stale" in keys:
        keys = [k for k in keys if k != "trt-stale"]
    freed = 0
    problems = []
    for key in keys:
        entry = TARGETS.get(key)
        if entry is None:
            print(f"  ? unknown target '{key}' - skipped")
            continue
        title, _desc, finder = entry
        paths = finder()
        if not paths:
            print(f"  - {title}: nothing to remove")
            continue
        sub = 0
        for p in paths:
            got, err = _remove(p)
            sub += got
            if err:
                problems.append((p, err))
        freed += sub
        print(f"  + {title}: freed {_fmt(sub)} ({len(paths)} item(s))")
    if problems:
        print(f"\n  {len(problems)} item(s) could not be removed — almost always because "
              f"the app is still running and holding them open. Stop the app and run "
              f"Clean again to finish those.")
        for p, err in problems[:8]:
            print(f"    - {os.path.relpath(p, ROOT)}: {err}")
    print(f"\n=== Freed {_fmt(freed)} ===\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Reclaim regenerable disk space.")
    ap.add_argument("targets", nargs="*", help=f"one or more of: {', '.join(TARGETS)}")
    ap.add_argument("--report", action="store_true", help="only show what could be freed")
    args = ap.parse_args()
    if args.report or not args.targets:
        report()
    else:
        clean(args.targets)
    sys.exit(0)
