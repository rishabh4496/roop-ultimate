"""Bridge to face_engine.pipeline.provenance for roop app pipelines."""

from __future__ import annotations

import os
from typing import Any, Mapping

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
    "verify_frequency_watermark",
    "verify_provenance_manifest",
]
