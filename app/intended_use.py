"""The intended-use terms from NOTICE.md, and whether this install accepted them.

NOTICE.md's "Intended use" section is the text the first-run screen shows;
its version is a hash of that text, so editing the terms asks every install
to accept them again. Acceptance is stored in config.yaml as
`intended_use_acknowledged: <version>` (settings.py). The React UI shows the
screen until the stored version matches; /api/swap refuses to render until
it does, so a client that bypasses the screen is refused too.
"""
from __future__ import annotations

import hashlib
import os
import re

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTICE_PATH = os.path.join(_ROOT, "NOTICE.md")
_HEADING = "## Intended use"
_FALLBACK = (
    "Use this software only on material you have the right to use, and only with the informed "
    "consent of the people whose likenesses are involved. Do not use it to create sexual content "
    "of anyone without their consent, to impersonate real people, to produce misinformation, or "
    "for anything unlawful in your jurisdiction. Where required, label output as synthetic.")


def terms_text() -> str:
    """The Intended-use section of NOTICE.md (fallback: the same words inline)."""
    try:
        with open(NOTICE_PATH, encoding="utf-8") as fh:
            notice = fh.read()
    except OSError as exc:
        print(f"[intended_use] NOTICE.md unreadable ({exc}); using the built-in text", flush=True)
        return _FALLBACK
    m = re.search(re.escape(_HEADING) + r"\s*\n(.*?)(?:\n## |\Z)", notice, re.S)
    body = (m.group(1) if m else "").strip()
    return body or _FALLBACK


def terms_version() -> str:
    return hashlib.sha256(terms_text().encode("utf-8")).hexdigest()[:16]


def acknowledged(cfg) -> bool:
    return bool(cfg) and str(getattr(cfg, "intended_use_acknowledged", "") or "") == terms_version()


def acknowledge(cfg) -> str:
    version = terms_version()
    cfg.intended_use_acknowledged = version
    cfg.save()
    return version
