"""Upload and read validated LUT assets used by the React grading preview."""

from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi import APIRouter, Body, File, HTTPException, UploadFile

import roop.globals as roop_globals
import safe_paths
from roop.lut_utils import parse_cube_text

router = APIRouter()
MAX_LUT_BYTES = 12 * 1024 * 1024


def _lut_dir() -> Path:
    folder = (Path.home() / '.roop-ultimate' / 'luts').resolve()
    folder.mkdir(parents=True, exist_ok=True)
    return folder


@router.get("/api/lut/shelf")
def lut_shelf():
    entries = sorted(_lut_dir().glob("*.cube"), key=lambda path: path.name.casefold())
    return {"files": [
        {"path": str(path), "name": path.stem.rsplit("_", 1)[0] + ".cube"}
        for path in entries if path.is_file() and not path.is_symlink()
    ]}


@router.post("/api/lut/upload")
async def upload_lut(file: UploadFile = File(...)):
    """Persist a bounded, validated .cube file and return its backend path."""
    name = safe_paths.sanitize_filename(file.filename, default="grade.cube")
    if Path(name).suffix.lower() != ".cube":
        raise HTTPException(status_code=400, detail="Only .cube LUT files are supported")
    raw = await file.read(MAX_LUT_BYTES + 1)
    if len(raw) > MAX_LUT_BYTES:
        raise HTTPException(status_code=413, detail="LUT exceeds the 12 MiB upload limit")
    try:
        text = raw.decode("utf-8-sig")
        size, domain_min, domain_max, values = parse_cube_text(text)
    except (UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid .cube LUT: {exc}") from exc
    digest = hashlib.sha256(raw).hexdigest()[:12]
    safe_stem = Path(name).stem[:64] or "grade"
    path = _lut_dir() / f"{safe_stem}_{digest}.cube"
    path.write_bytes(raw)
    return {
        "path": str(path),
        "name": path.name,
        "size": size,
        "entries": len(values),
        "domain_min": domain_min,
        "domain_max": domain_max,
    }


@router.post("/api/lut/read")
def read_lut(payload: dict = Body(...)):
    """Read a shelf asset or the grade already configured on this workstation."""
    path = resolve_lut_asset(payload.get("path"))
    try:
        with path.open("rb") as stream:
            data = stream.read(MAX_LUT_BYTES + 1)
        if len(data) > MAX_LUT_BYTES:
            raise HTTPException(status_code=413, detail="LUT exceeds the 12 MiB limit")
        text = data.decode("utf-8-sig")
        size, _domain_min, _domain_max, _values = parse_cube_text(text)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Stored LUT is invalid: {exc}") from exc
    return {"name": path.name, "size": size, "cube_text": text}


def resolve_lut_asset(value) -> Path:
    path = Path(str(value or "")).expanduser().resolve()
    configured = getattr(getattr(roop_globals, "CFG", None), "lut_path", "")
    configured_path = Path(configured).expanduser().resolve() if configured else None
    if not path.is_relative_to(_lut_dir()) and path != configured_path:
        raise HTTPException(status_code=400, detail="LUT path is outside the grading shelf")
    if not path.is_file() or path.suffix.lower() != ".cube":
        raise HTTPException(status_code=404, detail="LUT file is not available")
    return path


__all__ = ["router"]
