"""Disk cleanup for the Roop Ultimate launcher (driven by scripts/clean.js).

Only ever touches things the app can rebuild by itself. Everything that would
cost you work or a re-download is off-limits and is never referenced here:

    app/models/       the model library (~23GB) - re-downloading is the only way back
    app/env/          the virtualenv - use "Save Disk Space" (link.js) instead
    app/output/       YOUR RENDERED VIDEOS - never touched by this script
    app/facesets/     your saved faceset library
    app/projects/     project checkpoints, and any upload a project or queue job still reads
    react-ui/node_modules, source, .git

Run with no arguments for a report; pass one or more target names to delete.
"""

import argparse
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root; this file is in scripts/
APP = os.path.join(ROOT, "app")
# <PINOKIO_HOME>/api/<launcher>: the home is two levels up, but only when this
# really is an api/ launcher. Never assume a drive letter or a different layout.
PINOKIO_HOME = (os.path.dirname(os.path.dirname(ROOT))
                if os.path.basename(os.path.dirname(ROOT)) == "api" else None)
SCRATCH_DAYS = 3  # Pinokio scratch younger than this may belong to a run in progress


def _size(path):
    """Bytes used by a file or directory tree. Unreadable entries count as 0."""
    if os.path.isfile(path):
        return _file_bytes(path)
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path, onerror=lambda e: None):
        for fn in filenames:
            total += _file_bytes(os.path.join(dirpath, fn))
    return total


def _file_bytes(path):
    """Bytes freed by deleting one file. A hard-linked file frees nothing until
    its last link goes: the uv cache shares files with the virtualenv, so counting
    them reported 10 GB for a cleanup that returned 0.7 GB."""
    try:
        st = os.stat(path)
    except OSError:
        return 0
    return 0 if st.st_nlink > 1 else st.st_size


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


_TRT_PRECISIONS = ("mixed", "fp32", "fp16", "int8", "fp8")


def _trt_dirs(stale_only):
    """TensorRT engine caches. `stale_only` keeps the ones the current
    precision setting actually uses (see Enhance_GPEN / FaceSwapInsightFace,
    which append _gpen_fp32 / _swap_fp32 to the precision folder)."""
    root = os.path.join(APP, "models", "trt_cache")
    if not os.path.isdir(root):
        return []
    def stale(name):
        # Cache folders are named "<precision>" or "<precision>_<GPU>_<driver>_..._<model>_fp32".
        # The old exact-name test kept only the first form and would have deleted every
        # current engine (15.8 GB, ~25 min to rebuild) as "stale". Now only what is
        # provably stale qualifies: test leftovers and another precision's folders.
        # Anything else (recognition_*_fp16_*, surgical*) is not provable, so it stays.
        p = _trt_precision()
        others = [q for q in _TRT_PRECISIONS if q != p]
        return name.startswith("_unittest_") or any(
            name == q or name.startswith(q + "_") for q in others)

    return [os.path.join(root, d) for d in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, d)) and (not stale_only or stale(d))]


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


def _upload_candidates():
    """Uploads and unpacked facesets the APP calls safe to delete.

    Reuses storage_manager's rules rather than a copy of them: a file a project
    checkpoint or queue job still points at is protected, and while ANY project is
    active or resumable nothing here is deletable. Importing `api`/`routes_queue`
    would boot the whole backend (CUDA, queue restore), so they are blocked;
    `scan()` then reads queue.json and projects/*.json straight from disk.
    """
    if APP not in sys.path:
        sys.path.insert(0, APP)
    absent = object()
    blocked = {n: sys.modules.get(n, absent) for n in ("api", "routes_queue")}
    for name in blocked:
        sys.modules[name] = None  # None makes the import raise, which scan() tolerates
    try:
        from storage_manager import StorageManager, SAFE_TO_DELETE
        items = StorageManager(project_root=ROOT).scan()["items"]
    except Exception as e:  # unreadable state is protected, never deleted
        print(f"  ! uploads: protection check failed ({e}); skipping uploads")
        return []
    finally:
        for name, prior in blocked.items():
            if prior is absent:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior
    uploads = {os.path.normcase(os.path.join(APP, "temp", "api_uploads")),
               os.path.normcase(os.path.join(APP, "temp", "faceset"))}
    return [i["path"] for i in items
            if i["classification"] == SAFE_TO_DELETE
            and os.path.normcase(os.path.dirname(i["path"])) in uploads]


def _newest_mtime(path):
    try:
        newest = os.path.getmtime(path)
    except OSError:
        return 0.0
    for dirpath, _d, filenames in os.walk(path, onerror=lambda e: None):
        for fn in filenames:
            try:
                newest = max(newest, os.path.getmtime(os.path.join(dirpath, fn)))
            except OSError:
                pass
    return newest


def _pinokio_cache(name):
    """<PINOKIO_HOME>/cache/<name>. Pinokio exports these as environment variables
    (TEMP, TMPDIR, UV_CACHE_DIR ...); fall back to the documented location. Only a
    folder strictly inside <PINOKIO_HOME>/cache is ever returned."""
    if not PINOKIO_HOME:
        return None
    root = os.path.realpath(os.path.join(PINOKIO_HOME, "cache"))
    path = os.path.realpath(os.environ.get(name) or os.path.join(root, name))
    try:
        inside = os.path.commonpath([path, root]) == root and path != root
    except ValueError:  # different drives
        return None
    return path if inside and os.path.isdir(path) else None


def _old_scratch():
    """Finished scratch under Pinokio's TEMP/TMPDIR/TMP: tool scratch, benchmark
    leftovers, test fixtures. Entries nothing has touched for SCRATCH_DAYS days
    (judged by the newest file inside, not the folder's own date); the folders
    themselves stay. TEMP/claude/<project>/ holds one scratch folder per session,
    so that level is listed instead of the project folder."""
    cutoff = time.time() - SCRATCH_DAYS * 86400
    out, seen = [], set()
    for var in ("TEMP", "TMPDIR", "TMP"):
        base = _pinokio_cache(var)
        if not base or base in seen:
            continue
        seen.add(base)
        try:
            entries = [os.path.join(base, n) for n in os.listdir(base)]
        except OSError:
            continue
        for e in entries:
            if os.path.islink(e):
                continue
            if os.path.basename(e) == "claude" and os.path.isdir(e):
                for proj in (os.path.join(e, n) for n in os.listdir(e)):
                    if os.path.isdir(proj) and not os.path.islink(proj):
                        out.extend(c for c in (os.path.join(proj, n) for n in os.listdir(proj))
                                   if _newest_mtime(c) < cutoff)
            elif _newest_mtime(e) < cutoff:
                out.append(e)
    return out


def _package_caches():
    """uv / npm / pip download caches."""
    out = []
    for var, sub in (("UV_CACHE_DIR", ""), ("PIP_CACHE_DIR", ""),
                     ("npm_config_cache", "_cacache")):
        base = _pinokio_cache(var)
        if base and os.path.isdir(os.path.join(base, sub)):
            base = os.path.join(base, sub)
            out.extend(os.path.join(base, n) for n in os.listdir(base))
    return out


TARGETS = {
    "uploads": (
        "Uploaded media scratch",
        "Copies of the images and videos you dragged in, plus unpacked facesets. "
        "Only files no project or queue job still reads, and nothing at all while a "
        "project is active or resumable. Your rendered output is NOT here.",
        _upload_candidates,
    ),
    "pinokio-temp": (
        "Old Pinokio scratch",
        f"Temp folders under <PINOKIO_HOME>/cache (TEMP, TMPDIR, TMP) that nothing has "
        f"touched for {SCRATCH_DAYS}+ days: tool scratch, benchmark leftovers, test fixtures.",
        _old_scratch,
    ),
    "pkg-caches": (
        "Package download caches",
        "uv, npm and pip caches in <PINOKIO_HOME>/cache. Shared by all Pinokio apps and "
        "refilled by the next install, which downloads again. Space shared with an "
        "installed environment is not counted.",
        _package_caches,
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
        "react-ui/dist and lint caches. Rebuilt automatically on the next launch.",
        # node_modules is deliberately NOT here: it is required to launch the
        # client.  react-ui-v2/dist was listed here until React UI 2.0 was
        # removed; there is one client now.
        lambda: [p for p in (os.path.join(ROOT, "react-ui", "dist"),
                             os.path.join(ROOT, ".ruff_cache"),
                             os.path.join(APP, ".ruff_cache"))
                 if os.path.exists(p)],
    ),
}

# trt-all is a superset of trt-stale; running both would double-count the report
# and make the second one a no-op.
EXCLUSIVE = ("trt-all", "trt-stale")


def _clear_readonly_and_retry(func, path, _exc_info):
    """Git marks pack/idx files read-only, and Windows refuses to delete those
    (WinError 5). Clear the flag and retry once; a real lock still raises."""
    os.chmod(path, 0o700)
    func(path)


def _remove(path):
    """Returns (freed_bytes, error_or_None). Locked files are reported, not raised:
    the app holds its uploads open while running, and one locked file must not
    abort the rest of the cleanup."""
    size = _size(path)
    try:
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path, onerror=_clear_readonly_and_retry)
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
