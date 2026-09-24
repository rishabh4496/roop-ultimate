"""Model integrity progress for the React splash screen.

`roop.model_integrity.verify_and_repair` runs inside core.pre_check while the
API thread is already serving. This route exposes its snapshot, and is the one
route that must answer before configuration exists -- the splash polls it
precisely during that window -- so it does not consult CFG.
"""

from fastapi import APIRouter

from roop import model_integrity

router = APIRouter(prefix="/api/models")


@router.get("/integrity")
def integrity_status():
    return model_integrity.get_status()
