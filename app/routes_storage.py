"""Storage review and explicit cleanup endpoints for the active React client."""

from __future__ import annotations

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

from storage_manager import StorageError, StorageManager


router = APIRouter()
MANAGER = StorageManager()


@router.get("/api/storage")
def storage_review():
    """Return a fresh, reference-aware storage review; never deletes on GET."""
    return MANAGER.scan()


@router.post("/api/storage/delete")
def storage_delete(payload: dict = Body(...)):
    """Delete one explicitly confirmed, freshly revalidated safe item."""
    try:
        item_id = str(payload.get("item_id") or "")
        if not item_id:
            raise StorageError("item_id is required")
        if payload.get("item_ids"):
            raise StorageError("delete one reviewed item at a time")
        return MANAGER.delete_item(item_id, confirm=bool(payload.get("confirm")))
    except StorageError as exc:
        return JSONResponse(status_code=exc.status_code, content={
            "message": str(exc),
            "storage_error": True,
        })


__all__ = ["MANAGER", "router", "storage_delete", "storage_review"]
