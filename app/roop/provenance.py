"""Bridge to face_engine.pipeline.provenance for roop app pipelines."""

from __future__ import annotations

import json
import os
import secrets
import tempfile
from pathlib import Path

from face_engine.pipeline.provenance import (
    DEFAULT_DISCLOSURE,
    DEFAULT_VERSION,
    PROVENANCE_SCHEMA,
    compute_file_sha256,
    create_provenance_manifest,
    embed_frequency_watermark,
    extract_and_verify_file,
    extract_frequency_watermark,
    get_stage14_hardware_receipt_hash,
    sign_and_embed_file,
    verify_frequency_watermark,
    verify_provenance_manifest,
)


def is_provenance_enabled() -> bool:
    """Return True if cryptographic provenance signing is enabled in settings."""
    try:
        import roop.globals as g
        cfg = getattr(g, "CFG", None)
        if cfg is not None:
            return bool(getattr(cfg, "enable_provenance_signer", True))
    except Exception:
        pass
    return True


def is_watermark_enabled() -> bool:
    """Return True if frequency-domain watermark embedding is enabled in settings."""
    try:
        import roop.globals as g
        cfg = getattr(g, "CFG", None)
        if cfg is not None:
            return bool(getattr(cfg, "embed_provenance_watermark", False))
    except Exception:
        pass
    return False


def get_provenance_disclosure() -> str:
    """Return user-configured synthetic media disclosure string."""
    try:
        import roop.globals as g
        cfg = getattr(g, "CFG", None)
        if cfg is not None:
            return str(getattr(cfg, "provenance_disclosure", DEFAULT_DISCLOSURE) or DEFAULT_DISCLOSURE)
    except Exception:
        pass
    return DEFAULT_DISCLOSURE


def _signing_key(*, create: bool = False) -> bytes:
    """Keep the delivery key with the local user, outside the repository and outputs."""
    path = Path.home() / '.roop-ultimate' / 'provenance.key'
    if create:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(secrets.token_bytes(32))
    with path.open('rb') as stream:
        key = stream.read(33)
    if len(key) != 32:
        raise ValueError('Invalid local provenance key')
    return key


def sign_render_output(file_path: str) -> dict:
    """Sign a completed user-facing output and write a readable hash manifest."""
    if not is_provenance_enabled():
        return {"enabled": False, "signed": False, "reason": "signer disabled"}
    if not os.path.isfile(file_path):
        return {"enabled": True, "signed": False, "reason": "output is missing"}
    disclosure = get_provenance_disclosure()
    key = _signing_key(create=True)
    embedded, embedded_manifest = sign_and_embed_file(file_path, disclosure=disclosure, secret_key=key)
    # The container bytes change when the embedded record is written. The
    # detached manifest signs the finished bytes, including container metadata.
    manifest = create_provenance_manifest(
        compute_file_sha256(file_path), disclosure=disclosure,
        stage14_receipt_hash=(embedded_manifest or {}).get("stage14_receipt_hash"), secret_key=key)
    sidecar = f"{file_path}.provenance.json"
    fd, tmp = tempfile.mkstemp(prefix=".provenance-", dir=os.path.dirname(os.path.abspath(sidecar)))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
        os.replace(tmp, sidecar)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return {"enabled": True, "signed": True, "embedded": embedded, "manifest": manifest, "sidecar": sidecar}


def verify_render_output(file_path: str) -> dict:
    """Check the delivered bytes against the sidecar, or report a legacy signature."""
    try:
        with open(f"{file_path}.provenance.json", encoding="utf-8") as stream:
            manifest = json.loads(stream.read(64 * 1024))
    except FileNotFoundError:
        try:
            valid, manifest, reason = extract_and_verify_file(file_path, secret_key=_signing_key())
            if not valid and manifest:
                valid, reason = verify_provenance_manifest(manifest)
        except FileNotFoundError:
            valid, manifest, reason = extract_and_verify_file(file_path)
        except (OSError, ValueError) as exc:
            valid, manifest, reason = False, None, str(exc)
        return {"verified": valid, "file_verified": False, "manifest": manifest, "reason": reason}
    except (OSError, TypeError, ValueError) as exc:
        return {"verified": False, "file_verified": False, "reason": f"Invalid hash manifest: {exc}"}
    try:
        valid, reason = verify_provenance_manifest(manifest, compute_file_sha256(file_path), secret_key=_signing_key())
    except FileNotFoundError:
        valid, reason = False, "The local signing key is unavailable"
    except (OSError, TypeError, ValueError) as exc:
        valid, reason = False, str(exc)
    return {"verified": valid, "file_verified": valid, "manifest": manifest, "reason": reason}


__all__ = [
    "DEFAULT_DISCLOSURE",
    "DEFAULT_VERSION",
    "PROVENANCE_SCHEMA",
    "compute_file_sha256",
    "create_provenance_manifest",
    "embed_frequency_watermark",
    "extract_and_verify_file",
    "extract_frequency_watermark",
    "get_provenance_disclosure",
    "get_stage14_hardware_receipt_hash",
    "is_provenance_enabled",
    "is_watermark_enabled",
    "sign_and_embed_file",
    "sign_render_output",
    "verify_frequency_watermark",
    "verify_provenance_manifest",
    "verify_render_output",
]
