"""Model integrity manager: SHA256-verified, resumable model downloads.

WHAT IT GUARDS AGAINST.  `utilities.conditional_download` decides a model is
present by ``os.path.exists``.  A file truncated by a full disk, corrupted by a
bad sector, or replaced by an HTML error page therefore counts as installed
forever, and surfaces as a cryptic ONNX load error in whatever stage first opens
it.  An interrupted download was also thrown away and restarted from byte zero,
which on a 554 MB swapper over a flaky link can mean never finishing.

WHAT IT DOES.  `app/model_manifest.json` lists each checked model with its size
and SHA256 (each hash cross-checked against the host's own published digest).
At startup `verify_and_repair` hashes every listed file -- once; the digest is
cached against (size, mtime_ns) in `models/.integrity.json`, so later boots cost
a stat per file -- and downloads what is missing or wrong:

  * downloads write to `<file>.part` and RESUME with an HTTP Range request;
  * a finished download is size- and hash-checked before it is renamed into
    place, so a bad transfer never becomes the model;
  * a local file that fails its hash is moved aside, not deleted.  If the
    replacement cannot be fetched (offline, host down) the original is put
    back, so the check can never leave a machine worse than it found it.

`get_status()` is the progress snapshot the React splash screen polls through
`GET /api/models/integrity` while this runs.

TLS.  The first attempt uses normal certificate verification.  Only on a
certificate failure does it retry unverified -- acceptable here, and only here,
because the payload is then held to a pinned SHA256 before it is used.
"""
from __future__ import annotations

import hashlib
import json
import os
import ssl
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from roop.degrade import swallowed as _swallowed

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST = os.path.join(APP_DIR, "model_manifest.json")
DEFAULT_MODELS_DIR = os.path.join(APP_DIR, "models")
CACHE_NAME = ".integrity.json"
_HASH_CHUNK = 4 * 1024 * 1024
_DOWNLOAD_CHUNK = 1024 * 1024
_USER_AGENT = "roop-ultimate-model-integrity/1"

ProgressFn = Callable[[int, int], None]


class DownloadError(RuntimeError):
    """A model could not be fetched, or what arrived failed verification."""


@dataclass(frozen=True)
class ModelEntry:
    file: str
    sha256: str
    size: int
    urls: tuple
    required: bool = True
    family: str = ""
    description: str = ""


def load_manifest(path: Optional[str] = None) -> List[ModelEntry]:
    with open(path or DEFAULT_MANIFEST, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    entries = []
    for item in data.get("models", []):
        entries.append(ModelEntry(
            file=str(item["file"]),
            sha256=str(item["sha256"]).lower(),
            size=int(item["size"]),
            urls=tuple(item.get("urls") or ()),
            required=bool(item.get("required", True)),
            family=str(item.get("family", "")),
            description=str(item.get("description", "")),
        ))
    return entries


def lookup(filename: str, manifest: Optional[List[ModelEntry]] = None) -> Optional[ModelEntry]:
    """The manifest entry for a file name, or None if the manifest does not list it."""
    name = os.path.basename(filename)
    try:
        entries = manifest if manifest is not None else load_manifest()
    except (OSError, ValueError, KeyError) as exc:
        _swallowed("roop/model_integrity.py:lookup", exc, "treating file as unlisted")
        return None
    for entry in entries:
        if entry.file == name:
            return entry
    return None


# ── hashing ──────────────────────────────────────────────────────────────────

def sha256_file(path: str, progress: Optional[ProgressFn] = None) -> str:
    digest = hashlib.sha256()
    total = os.path.getsize(path)
    done = 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(_HASH_CHUNK)
            if not block:
                break
            digest.update(block)
            done += len(block)
            if progress is not None:
                progress(done, total)
    return digest.hexdigest()


class _DigestCache:
    """sha256 per file, valid only while (size, mtime_ns) are unchanged."""

    def __init__(self, models_dir: str):
        self.path = os.path.join(models_dir, CACHE_NAME)
        self._data: Dict[str, dict] = {}
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                self._data = loaded
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as exc:
            _swallowed("roop/model_integrity.py:cache-load", exc, "rehashing every model")

    def get(self, name: str, stat: os.stat_result) -> Optional[str]:
        row = self._data.get(name)
        if (isinstance(row, dict) and row.get("size") == stat.st_size
                and row.get("mtime_ns") == stat.st_mtime_ns):
            return row.get("sha256")
        return None

    def put(self, name: str, stat: os.stat_result, digest: str) -> None:
        self._data[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                            "sha256": digest}

    def drop(self, name: str) -> None:
        self._data.pop(name, None)

    def save(self) -> None:
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(self._data, handle, indent=1, sort_keys=True)
            os.replace(tmp, self.path)
        except OSError as exc:
            _swallowed("roop/model_integrity.py:cache-save", exc, "digests recomputed next boot")


def file_digest(path: str, cache: Optional[_DigestCache] = None,
                progress: Optional[ProgressFn] = None) -> str:
    stat = os.stat(path)
    name = os.path.basename(path)
    if cache is not None:
        cached = cache.get(name, stat)
        if cached:
            return cached
    digest = sha256_file(path, progress)
    if cache is not None:
        cache.put(name, stat, digest)
    return digest


# ── resumable download ───────────────────────────────────────────────────────

def _open(url: str, offset: int, timeout: float, opener, insecure: bool):
    headers = {"User-Agent": _USER_AGENT}
    if offset > 0:
        headers["Range"] = "bytes=%d-" % offset
    request = urllib.request.Request(url, headers=headers)
    if insecure:
        return opener(request, timeout=timeout, context=ssl._create_unverified_context())
    return opener(request, timeout=timeout)


def _status_of(response) -> int:
    status = getattr(response, "status", None)
    if status is None:
        status = response.getcode()
    return int(status or 200)


def download_resumable(url: str, dest: str, *, expected_size: Optional[int] = None,
                       expected_sha256: Optional[str] = None,
                       progress: Optional[ProgressFn] = None,
                       timeout: float = 30.0, retries: int = 4,
                       backoff: float = 1.5,
                       opener=None, sleep=time.sleep) -> str:
    """Fetch *url* into *dest* through `<dest>.part`, resuming across failures.

    Resumes within this call (each retry continues from the bytes already on
    disk) and across calls (a `.part` left by a killed process is continued,
    not discarded).  Raises DownloadError if the transfer cannot be completed
    or the result does not match *expected_size* / *expected_sha256*; on a
    verification failure the `.part` is deleted, because resuming onto bytes
    that are already wrong can never produce the right file.
    """
    opener = opener or urllib.request.urlopen
    part = dest + ".part"
    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
    insecure = False
    last_error: Optional[BaseException] = None

    for attempt in range(retries + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if expected_size and have > expected_size:
            os.remove(part)
            have = 0
        if expected_size and have == expected_size:
            last_error = None
            break
        try:
            with _open(url, have, timeout, opener, insecure) as response:
                status = _status_of(response)
                if have and status != 206:
                    # The server ignored the Range header: it is sending the
                    # whole file, so the partial bytes must be discarded.
                    have = 0
                length = response.headers.get("Content-Length")
                total = expected_size or ((have + int(length)) if length else 0)
                done = have
                with open(part, "ab" if have else "wb") as out:
                    while True:
                        block = response.read(_DOWNLOAD_CHUNK)
                        if not block:
                            break
                        out.write(block)
                        done += len(block)
                        if progress is not None:
                            progress(done, total)
                if total and done < total:
                    raise DownloadError("connection closed at %d of %d bytes" % (done, total))
            last_error = None
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and have and (not expected_size or have == expected_size):
                # Range starts at the end: the .part already holds everything.
                last_error = None
                break
            last_error = exc
            if exc.code in (401, 403, 404, 410):
                break  # not transient -- the next mirror is the only hope
        except urllib.error.URLError as exc:
            last_error = exc
            if isinstance(exc.reason, ssl.SSLCertVerificationError) and not insecure:
                insecure = True  # payload is SHA256-checked below; see module doc
                continue
        except (OSError, DownloadError, ValueError) as exc:
            last_error = exc
        if attempt < retries:
            sleep(backoff * (2 ** attempt))

    if last_error is not None:
        raise DownloadError("%s: %s" % (url, last_error))
    if not os.path.exists(part):
        raise DownloadError("%s: nothing was downloaded" % url)
    size = os.path.getsize(part)
    if expected_size and size != expected_size:
        os.remove(part)
        raise DownloadError("%s: size %d, expected %d" % (url, size, expected_size))
    if expected_sha256:
        digest = sha256_file(part)
        if digest.lower() != expected_sha256.lower():
            os.remove(part)
            raise DownloadError("%s: sha256 %s, expected %s" % (url, digest, expected_sha256))
    os.replace(part, dest)
    return dest


def fetch_entry(entry: ModelEntry, models_dir: str,
                progress: Optional[ProgressFn] = None, **kwargs) -> str:
    """Download one manifest entry, trying its URLs in order."""
    dest = os.path.join(models_dir, entry.file)
    errors = []
    for url in entry.urls:
        try:
            return download_resumable(url, dest, expected_size=entry.size,
                                      expected_sha256=entry.sha256,
                                      progress=progress, **kwargs)
        except DownloadError as exc:
            errors.append(str(exc))
    raise DownloadError("; ".join(errors) or "%s has no download URL" % entry.file)


# ── progress state (read by GET /api/models/integrity) ───────────────────────

class IntegrityStatus:
    """Thread-safe snapshot of the running check, for the splash screen."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with getattr(self, "_lock", threading.Lock()):
            self.phase = "idle"       # idle|checking|downloading|ready|degraded
            self.message = ""
            self.files: Dict[str, dict] = {}
            self.started_at = None
            self.finished_at = None

    def begin(self, entries: List[ModelEntry]):
        with self._lock:
            self.phase = "checking"
            self.message = "Verifying models"
            self.started_at = time.time()
            self.finished_at = None
            self.files = {e.file: {"file": e.file, "family": e.family,
                                   "required": e.required, "status": "pending",
                                   "bytes_done": 0, "bytes_total": e.size,
                                   "error": ""} for e in entries}

    def update(self, name: str, **fields):
        with self._lock:
            row = self.files.setdefault(name, {"file": name})
            row.update(fields)
            if fields.get("status") == "downloading":
                self.phase = "downloading"
                self.message = "Downloading %s" % name

    def finish(self, phase: str, message: str):
        with self._lock:
            self.phase = phase
            self.message = message
            self.finished_at = time.time()

    def snapshot(self) -> dict:
        with self._lock:
            files = [dict(row) for row in self.files.values()]
            phase, message = self.phase, self.message
            started, finished = self.started_at, self.finished_at
        downloading = [f for f in files if f.get("status") == "downloading"]
        done = sum(int(f.get("bytes_done") or 0) for f in downloading)
        total = sum(int(f.get("bytes_total") or 0) for f in downloading)
        return {
            "phase": phase,
            "message": message,
            "busy": phase in ("checking", "downloading"),
            "files": files,
            "download_bytes_done": done,
            "download_bytes_total": total,
            "download_percent": (100.0 * done / total) if total else None,
            "started_at": started,
            "finished_at": finished,
        }


STATUS = IntegrityStatus()


def get_status() -> dict:
    return STATUS.snapshot()


# ── the startup check ────────────────────────────────────────────────────────

def _throttled(name: str, key: str, status: IntegrityStatus, interval: float = 0.25):
    last = {"t": 0.0}

    def report(done: int, total: int):
        now = time.monotonic()
        if now - last["t"] < interval and done < total:
            return
        last["t"] = now
        status.update(name, **{key: done, "bytes_total": total})
    return report


def verify_and_repair(models_dir: Optional[str] = None,
                      manifest: Optional[List[ModelEntry]] = None, *,
                      download: bool = True,
                      status: Optional[IntegrityStatus] = None,
                      log: Callable[[str], None] = print,
                      **download_kwargs) -> dict:
    """Hash every manifest model; fetch what is missing or corrupt.

    Never raises for a model problem: startup policy for a missing model is
    "warn and continue" (utilities._handle_missing_model), and a model the
    user's configuration does not use must not keep the app from opening.
    The returned report -- and the splash -- say exactly what is wrong.
    """
    models_dir = models_dir or DEFAULT_MODELS_DIR
    status = status or STATUS
    entries = manifest if manifest is not None else load_manifest()
    os.makedirs(models_dir, exist_ok=True)
    cache = _DigestCache(models_dir)
    status.begin(entries)
    results: Dict[str, str] = {}

    for entry in entries:
        path = os.path.join(models_dir, entry.file)
        if os.path.isfile(path):
            status.update(entry.file, status="verifying", bytes_done=0)
            try:
                digest = file_digest(path, cache, _throttled(entry.file, "bytes_done", status))
            except OSError as exc:
                digest = "unreadable: %s" % exc
            if digest == entry.sha256:
                status.update(entry.file, status="ok", bytes_done=entry.size)
                results[entry.file] = "ok"
                continue
            log("[ModelIntegrity] %s FAILED its SHA256 check (got %s, expected %s)"
                % (entry.file, digest[:16], entry.sha256[:16]))
            cache.drop(entry.file)
            if not download:
                status.update(entry.file, status="corrupt", error="sha256 mismatch")
                results[entry.file] = "corrupt"
                continue
            quarantine = path + ".corrupt"
            os.replace(path, quarantine)
            try:
                status.update(entry.file, status="downloading", bytes_done=0)
                fetch_entry(entry, models_dir,
                            _throttled(entry.file, "bytes_done", status), **download_kwargs)
                os.remove(quarantine)
                cache.put(entry.file, os.stat(path), entry.sha256)
                status.update(entry.file, status="ok", bytes_done=entry.size)
                results[entry.file] = "repaired"
                log("[ModelIntegrity] %s re-downloaded and verified" % entry.file)
            except (DownloadError, OSError) as exc:
                # Put the original back: a suspect model beats no model, and
                # this check must never leave the machine worse than before.
                if os.path.exists(quarantine) and not os.path.exists(path):
                    os.replace(quarantine, path)
                status.update(entry.file, status="corrupt", error=str(exc))
                results[entry.file] = "corrupt"
                log("[ModelIntegrity] could not replace %s (%s); kept the existing file"
                    % (entry.file, exc))
            continue

        if not download:
            status.update(entry.file, status="missing")
            results[entry.file] = "missing"
            continue
        try:
            status.update(entry.file, status="downloading", bytes_done=0)
            log("[ModelIntegrity] downloading %s (%.1f MB)" % (entry.file, entry.size / 2 ** 20))
            fetch_entry(entry, models_dir,
                        _throttled(entry.file, "bytes_done", status), **download_kwargs)
            cache.put(entry.file, os.stat(path), entry.sha256)
            status.update(entry.file, status="ok", bytes_done=entry.size)
            results[entry.file] = "downloaded"
        except (DownloadError, OSError) as exc:
            status.update(entry.file, status="missing", error=str(exc))
            results[entry.file] = "missing"
            log("[ModelIntegrity] %s is missing and could not be downloaded: %s"
                % (entry.file, exc))

    cache.save()
    bad = sorted(name for name, result in results.items()
                 if result in ("missing", "corrupt"))
    if bad:
        status.finish("degraded", "Unverified models: " + ", ".join(bad))
    else:
        status.finish("ready", "All %d models verified" % len(results))
    return {"results": results, "ok": not bad, "problems": bad}
