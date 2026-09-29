"""Model catalog and metadata endpoints for React UI."""
from __future__ import annotations

from typing import Any
from fastapi import APIRouter

from face_engine.models.zoo import build_default_registry

router = APIRouter(prefix="/api/models")


@router.get("/catalog")
def get_model_catalog() -> dict[str, Any]:
    """Exposes all registered models, native resolutions, and parameter schemas for the UI."""
    registry = build_default_registry()
    catalog_data = registry.catalog()
    return {
        "status": "ok",
        "models": catalog_data,
        "swappers": [m for m in catalog_data.values() if m["task"] == "swap"],
        "maskers": [m for m in catalog_data.values() if m["task"] in ("occlusion", "parsing")],
        "restorers": [m for m in catalog_data.values() if m["task"] == "restoration"],
    }
