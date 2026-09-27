"""Verified, resumable model downloads.

A file is only ever moved to its final name after its size and SHA256 match
the declaration; partial transfers live beside it as ``<name>.part`` and are
resumed with an HTTP ``Range`` request. Hashing a multi-hundred-MB model on
every start is slow, so a verified hash is remembered in a
``<name>.sha256.json`` sidecar keyed by ``(size, mtime_ns)`` — touching or
replacing the file invalidates it.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Iterable
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 1024
DEFAULT_TIMEOUT = (10.0, 60.0)  # connect, read
DEFAULT_RETRIES = 4  # attempts per URL; a .part file resumes between them
BACKOFF_SECONDS = 1.5


class DownloadError(RuntimeError):
    """Every URL failed."""


class IntegrityError(DownloadError):
    """A file's size or SHA256 does not match its declaration."""


def sha256_file(path: Path, chunk_size: int = CHUNK_SIZE) -> str:
    """Hex SHA256 of a file, streamed."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".sha256.json")


def _stamp(path: Path) -> dict:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def verify_file(path: Path, sha256: str | None, size: int | None = None,
                use_sidecar: bool = True) -> bool:
    """True when ``path`` exists and matches ``size`` and ``sha256``.

    A None ``sha256`` checks existence and size only. The sidecar is trusted
    only for the exact size/mtime it was written for.
    """
    if not path.is_file():
        return False
    if size is not None and path.stat().st_size != size:
        return False
    if sha256 is None:
        return True
    expected = sha256.lower()
    sidecar = _sidecar(path)
    if use_sidecar and sidecar.is_file():
        try:
            record = json.loads(sidecar.read_text(encoding="utf-8"))
            if record.get("sha256") == expected and record.get("stamp") == _stamp(path):
                return True
        except (OSError, ValueError):
            pass
    actual = sha256_file(path)
    if actual != expected:
        return False
    if use_sidecar:
        try:
            sidecar.write_text(json.dumps({"sha256": actual, "stamp": _stamp(path)}),
                               encoding="utf-8")
        except OSError:
            pass
    return True


def _fetch(url: str, part: Path, expected_size: int | None,
           session: requests.Session, show_progress: bool) -> None:
    offset = part.stat().st_size if part.exists() else 0
    if expected_size is not None and offset > expected_size:
        part.unlink()
        offset = 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with session.get(url, stream=True, timeout=DEFAULT_TIMEOUT, headers=headers,
                     allow_redirects=True) as response:
        if response.status_code == 416 and expected_size is not None and offset == expected_size:
            return  # already complete
        response.raise_for_status()
        if offset and response.status_code != 206:
            offset = 0  # server ignored Range: start over
        length = response.headers.get("Content-Length")
        total = offset + int(length) if length is not None else expected_size
        mode = "ab" if offset else "wb"
        with open(part, mode) as handle, tqdm(
                total=total, initial=offset, unit="B", unit_scale=True, unit_divisor=1024,
                desc=part.name[: -len(".part")], disable=not show_progress) as bar:
            for block in response.iter_content(chunk_size=CHUNK_SIZE):
                if block:
                    handle.write(block)
                    bar.update(len(block))


def download_file(urls: Iterable[str], destination: Path, *, sha256: str | None,
                  size: int | None = None, show_progress: bool = True,
                  session: requests.Session | None = None,
                  retries: int = DEFAULT_RETRIES) -> Path:
    """Download ``destination`` from the first URL that yields a verified file.

    Returns immediately when ``destination`` already verifies. URLs are
    alternatives serving the *same bytes*; a URL whose file fails
    verification is discarded and the next one tried. Network errors are
    retried ``retries`` times per URL with exponential backoff, resuming the
    partial file. TLS verification is never disabled.

    Raises:
        DownloadError: no URL was given, or every URL failed.
        IntegrityError: the last failure was a hash/size mismatch.
    """
    destination = Path(destination)
    if verify_file(destination, sha256, size):
        return destination
    url_list = list(urls)
    if not url_list:
        raise DownloadError(f"no download URL declared for {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    own_session = session is None
    http = session or requests.Session()
    failures: list[str] = []
    last_integrity = False
    try:
        for url in url_list:
            error_text = ""
            for attempt in range(max(1, retries)):
                try:
                    _fetch(url, part, size, http, show_progress)
                    error_text = ""
                    break
                except requests.RequestException as exc:
                    error_text = str(exc)
                    logger.warning("download attempt %d/%d from %s failed: %s",
                                   attempt + 1, retries, url, exc)
                    if attempt + 1 < retries:
                        time.sleep(BACKOFF_SECONDS * 2 ** attempt)
            if error_text:
                failures.append(f"{url}: {error_text}")
                continue
            if verify_file(part, sha256, size, use_sidecar=False):
                os.replace(part, destination)
                verify_file(destination, sha256, size)  # writes the sidecar
                return destination
            actual = sha256_file(part) if part.exists() else "missing"
            failures.append(f"{url}: integrity mismatch (sha256 {actual}, "
                            f"size {part.stat().st_size if part.exists() else 0})")
            last_integrity = True
            part.unlink(missing_ok=True)
    finally:
        if own_session:
            http.close()
    error = IntegrityError if last_integrity else DownloadError
    raise error(f"could not fetch {destination.name}: " + " | ".join(failures))
