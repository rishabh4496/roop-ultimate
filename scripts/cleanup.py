"""Disk cleanup for the Roop Ultimate launcher (driven by scripts/clean.js).

Off-limits and never referenced here:

    app/models/       the model library (~23GB) - only the TensorRT engine cache under
                      app/models/trt_cache is touched, and only engines that cannot be reused
    app/env/          the virtualenv - use "Save Disk Space" (link.js) instead
    app/facesets/     your saved faceset library
    app/config.yaml   your settings
    react-ui/node_modules, source, .git

Targets in the second group below delete YOUR work products (finished renders, project
pre-scan caches and resume journals, uploaded targets). They are age-gated: nothing
younger than ROOP_CLEAN_KEEP_DAYS (default 3, 0 = everything) is removed, because that is
what a job in progress is still reading or writing. The sidebar does not offer Clean while
the app is running, and a file Windows has open is reported, never force-deleted.

Run with no arguments for a report; pass one or more target names to delete.
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
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
    # The whole repo, not just app/: face_engine/, roop_3d/, tools/ and tests/ leave bytecode too.
    for base in (ROOT,):
        for dirpath, dirnames, _f in os.walk(base, onerror=lambda e: None):
            # Don't descend into the venv's site-packages: those .pyc files are
            # regenerated per import and clearing them only slows the next start.
            # models/, output/ and projects/ hold only data (and are huge to walk).
            dirnames[:] = [d for d in dirnames
                           if d not in ("env", "node_modules", ".git", "models", "output", "projects")]
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


def _keep_seconds():
    """Age below which a work product is left alone (ROOP_CLEAN_KEEP_DAYS, default
    SCRATCH_DAYS, 0 = no protection): it may be what a job in progress is using."""
    try:
        days = float(os.environ.get("ROOP_CLEAN_KEEP_DAYS", SCRATCH_DAYS))
    except ValueError:
        days = float(SCRATCH_DAYS)
    return max(0.0, days) * 86400.0


def _aged(paths):
    """The paths whose newest file is older than the keep window."""
    cutoff = time.time() - _keep_seconds()
    return [p for p in paths if _newest_mtime(p) < cutoff]


def _outermost(paths):
    """Drop every path that sits inside another listed path, so nothing is counted
    or deleted twice (the second removal would only report a spurious error)."""
    keep = []
    for p in sorted(set(paths), key=lambda x: (len(x), x)):
        n = os.path.normcase(p)
        if not any(n == k or n.startswith(k + os.sep) for k in keep):
            keep.append(n)
    by_norm = {os.path.normcase(p): p for p in paths}
    return [by_norm[k] for k in keep]


def _git_tracked(path):
    """True when git tracks anything at `path`. A checkout that commits a folder this
    script lists as scratch must not lose it; no git, or no repo, means not tracked."""
    try:
        r = subprocess.run(["git", "-C", ROOT, "ls-files", "--", path],
                           capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(r.stdout.strip())


def _dev_leftovers():
    """Scratch that development and test runs leave in the repo: pytest and Playwright
    output, ad-hoc probe logs and dirs at the repo root, throwaway engine-cache test
    folders, JS tool caches in app/temp. Nothing the app reads."""
    tmp = os.path.join(APP, "temp")
    cands = [os.path.join(ROOT, n) for n in (
        "scratch", "temp_frames_analysis", "temp_fs_inspect", "original_gfpgan_fp32",
        "test_gpen256pro_out.png", ".pytest_cache")]
    for pat in ("scratch_*.log", "test-trt-cache_*"):
        cands.extend(glob.glob(os.path.join(ROOT, pat)))
    cands += [os.path.join(ROOT, "roop_3d", ".runtime", "pytest"),
              os.path.join(ROOT, "react-ui", "e2e", "test-results"),
              os.path.join(ROOT, "react-ui", "e2e", "playwright-report"),
              os.path.join(APP, "scratch"), os.path.join(APP, ".pytest_cache"),
              os.path.join(tmp, "node-compile-cache"),
              os.path.join(tmp, "playwright-transform-cache")]
    cands.extend(glob.glob(os.path.join(tmp, "tsx-*")))
    return [p for p in cands
            if os.path.lexists(p) and not os.path.islink(p) and not _git_tracked(p)]


_DRV_TAG = re.compile(r"_drv([0-9][0-9.]*)_trt")


def _gpu_driver():
    """Installed NVIDIA driver version as it appears in engine names ('621.14'), or
    None when it cannot be read - in which case no engine is called stale by driver."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = (r.stdout or "").strip().splitlines()
    return lines[0].strip() if r.returncode == 0 and lines else None


def _driver_stale(root, current):
    """Engines whose name embeds a DIFFERENT driver version. TensorRT engine names carry
    '_drv<version>_trt...', and the app only looks up the tag of the installed driver, so
    these can never be hit again after a driver update. Names without a tag are kept."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
        keep = []
        for d in dirnames:
            m = _DRV_TAG.search(d)
            if m and m.group(1) != current:
                out.append(os.path.join(dirpath, d))
            else:
                keep.append(d)
        dirnames[:] = keep
        for fn in filenames:
            m = _DRV_TAG.search(fn)
            if m and m.group(1) != current:
                out.append(os.path.join(dirpath, fn))
    return out


def _trt_stale():
    root = os.path.join(APP, "models", "trt_cache")
    paths = _trt_dirs(stale_only=True)
    current = _gpu_driver()
    if current and os.path.isdir(root):
        paths += _driver_stale(root, current)
    return _outermost(paths)


def _project_caches():
    """Per-project Stage 2 pre-scan caches (*.stage2.npz) and resume journals
    (*.roop.journal). A project re-runs its pre-scan and loses mid-render resume."""
    d = os.path.join(APP, "projects")
    try:
        names = os.listdir(d)
    except OSError:
        return []
    return _aged([os.path.join(d, n) for n in names
                  if n.endswith(".stage2.npz") or n.endswith(".roop.journal")])


_VIDEO_EXT = (".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v")


def _old_renders():
    """Finished renders in app/output (and each render's .provenance.json)."""
    d = os.path.join(APP, "output")
    try:
        names = os.listdir(d)
    except OSError:
        return []
    out = []
    for n in names:
        low = n.lower()
        full = os.path.join(d, n)
        if os.path.isfile(full) and any(low.endswith(e) or low.endswith(e + ".provenance.json")
                                        for e in _VIDEO_EXT):
            out.append(full)
    return _aged(out)


def _old_uploads():
    """Every file in app/temp/api_uploads (target videos, uploaded facesets) older than
    the keep window, whether or not an old project still points at it."""
    base = os.path.join(APP, "temp", "api_uploads")
    out = []
    for dirpath, _d, filenames in os.walk(base, onerror=lambda e: None):
        out.extend(os.path.join(dirpath, fn) for fn in filenames)
    return _aged(out)


def _quality_evidence():
    """Renders the swap-quality harness wrote (output/swap_quality_stage*) and the
    regression benchmark's recorded baselines and golden renders (.roop/regression).
    The next benchmark run records a fresh baseline."""
    cands = glob.glob(os.path.join(ROOT, "output", "swap_quality_stage*"))
    cands.append(os.path.join(ROOT, ".roop", "regression"))
    return _aged([p for p in cands if os.path.lexists(p) and not os.path.islink(p)])


TARGETS = {
    "dev-leftovers": (
        "Dev and test leftovers",
        "pytest / Playwright output, scratch logs and probe folders at the repo root, "
        "throwaway engine-cache test folders, JS tool caches in app/temp. Anything git "
        "tracks is kept. The app reads none of it.",
        _dev_leftovers,
    ),
    "project-caches": (
        "Project pre-scan caches and journals",
        "app/projects/*.stage2.npz and *.roop.journal older than the keep window. A project "
        "scans again the next time you open it and cannot resume a half-finished render.",
        _project_caches,
    ),
    "old-uploads": (
        "Old uploaded targets",
        "Files in app/temp/api_uploads older than the keep window, whether or not an old "
        "project still points at them (the project will ask for its media again).",
        _old_uploads,
    ),
    "old-renders": (
        "Old finished renders",
        "Videos (and their .provenance.json) in app/output older than the keep window. "
        "YOUR OUTPUT: not recoverable.",
        _old_renders,
    ),
    "quality-evidence": (
        "Quality-harness output and benchmark baselines",
        "output/swap_quality_stage* and .roop/regression older than the keep window. "
        "The next regression benchmark records a new baseline.",
        _quality_evidence,
    ),
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
        "Engines that can never be used again: built for a precision mode you no longer "
        "use, or for an older NVIDIA driver (the driver version is in the engine's name). "
        "Engines for the CURRENT setting and driver are kept, so runs stay fast.",
        _trt_stale,
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
        "react-ui/dist, react-ui-v2/dist and lint caches. Rebuilt automatically on the next launch.",
        # node_modules is deliberately NOT here: it is required to build the
        # clients. React UI 2.0 (react-ui-v2/, 1.0's screens with a new visual
        # layer, served at /v2/) is rebuilt by start_react.js like 1.0.
        lambda: [p for p in (os.path.join(ROOT, "react-ui", "dist"),
                             os.path.join(ROOT, "react-ui-v2", "dist"),
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
    print(f"\n  Work products younger than {_keep_seconds() / 86400:g} day(s) are kept "
          f"(set ROOP_CLEAN_KEEP_DAYS to change; 0 = everything).")
    print("  Never touched: model weights, virtualenv, saved facesets, config.yaml.\n")


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
