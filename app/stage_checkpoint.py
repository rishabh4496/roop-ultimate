"""The project-bound side of the Stage 1 / Stage 2 cache.

`roop/stage_cache.py` knows how to write and read the results; `project_checkpoint.py` knows where a
project's record lives and under which key a result is valid. This binds the two for ONE project and
presents the four calls the pipeline makes (`procmgr_batch.run_batch_inmem`):

    save_stage1 / restore_stage1      right after the sparse scan / instead of it
    save_stage2 / restore_stage2      right after the pre-pass      / instead of it

`api._run_swap` installs one as `roop.globals._stage_cache` for the length of a project's run and
clears it in its `finally`; no project, no cache, and the pipeline behaves exactly as before.

EVERY METHOD IS SAFE TO CALL AND NEVER RAISES: a cache that cannot be written costs only the next
resume's speed, and a cache that cannot be read costs only the time the stage would have taken anyway.
What a method could not do is PRINTED, because a cache that is silently ignored cannot be told apart
from one that was never there.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import project_checkpoint as _pc
from roop import stage_cache as _sc
from roop.degrade import swallowed as _swallowed


# ROOP_* variables the application WRITES while it runs, or sets for one harness/UI session. They describe
# the run, not the input, and they persist in os.environ after the run that set them: the runtime optimizer
# exports its chosen profile (`ROOP_RUNTIME_PROFILE_*`), the small-card policy toggles `ROOP_NVDEC` and
# restores it, auto-tune exports `ROOP_BATCH_SWAP_MAX`. Hashing them would make a pause-and-resume inside one
# app session see a different key than the run it resumes (the first run's key was taken before the optimizer
# had written anything) and refuse a cache that is perfectly good.
VOLATILE_ENV_PREFIXES = ("ROOP_RUNTIME_", "ROOP_NVDEC", "ROOP_REACT_CLIENT", "ROOP_HEADLESS",
                         "ROOP_VALIDATION", "ROOP_BATCH_SWAP_MAX", "ROOP_UPSCALE_TILE")


def live_identity(app_version: str = "") -> Dict[str, Any]:
    """What this process contributes to the cache key: the code (version) and the ROOP_* knobs that are
    inputs (not the ones the application itself writes while running - see VOLATILE_ENV_PREFIXES)."""
    env = {k: v for k, v in sorted(os.environ.items())
           if k.startswith("ROOP_") and not k.startswith(VOLATILE_ENV_PREFIXES)}
    return {"app_version": str(app_version or ""),
            "env": hashlib.sha256(repr(env).encode("utf-8", "replace")).hexdigest()}


class ProjectStageCache:
    def __init__(self, project_id: str, app_version: str = "",
                 log: Callable[..., None] = lambda *a, **k: print(*a, flush=True, **k)):
        self.project_id = str(project_id)
        self._live = live_identity(app_version)
        self._log = log
        self._usable: Optional[dict] = None
        self._stage1_ranges_abs: Optional[List[List[int]]] = None   # what Stage 2 must have been built against
        self.report: Dict[str, Any] = {}

    # -- lookup ---------------------------------------------------------------------------------
    def usable(self) -> dict:
        """Which stages of this project may be skipped. The sidecar is hashed ONCE per run."""
        if self._usable is None:
            try:
                self._usable = _pc.usable_stage_cache(self.project_id, self._live)
            except Exception as exc:                      # a broken cache must not break the run
                _swallowed("stage_checkpoint.py:usable", exc, "no Stage 1/2 cache is used for this run")
                self._usable = {"stage1": None, "stage2": None, "reasons": [f"cache lookup failed: {exc}"]}
            for reason in self._usable["reasons"]:
                self._log(f"[StageCache] not used: {reason}")
        return self._usable

    # -- Stage 1 --------------------------------------------------------------------------------
    def save_stage1(self, *, frame_start: int, frame_count: int, ranges, stats) -> bool:
        mode = str((stats or {}).get("mode") or "")
        if mode not in _sc.CACHEABLE_STAGE1_MODES:
            self._log(f"[StageCache] Stage 1 not recorded (scan mode '{mode}' is not a settled result).")
            return False
        abs_ranges = _sc.ranges_to_abs(ranges, frame_start) or []
        self._stage1_ranges_abs = abs_ranges
        try:
            saved = _pc.record_stage(self.project_id, "stage1", {
                "frame_start": int(frame_start), "frame_count": int(frame_count),
                "active_ranges": abs_ranges, "mode": mode,
                "digest": _sc.ranges_digest(abs_ranges),
            }, live=self._live)
        except Exception as exc:
            _swallowed("stage_checkpoint.py:save_stage1", exc, "this run's Stage 1 is not cached")
            self._log(f"[StageCache] Stage 1 could not be recorded: {exc}")
            return False
        if saved is None:
            self._log("[StageCache] Stage 1 not recorded (the project's input digests are not final).")
            return False
        self._log(f"[StageCache] Stage 1 recorded: {len(abs_ranges)} active range(s), mode {mode}.")
        return True

    def restore_stage1(self, *, frame_start: int, frame_count: int
                       ) -> Optional[Tuple[List[Tuple[int, int]], np.ndarray, Dict[str, Any]]]:
        entry = self.usable().get("stage1")
        if not entry:
            return None
        start, count = int(entry["frame_start"]), int(entry["frame_count"])
        if not _sc.covers_window(start, start + count, frame_start, frame_count):
            self._log(f"[StageCache] Stage 1 not used: it covers frames {start}-{start + count - 1}, "
                      f"this run needs {frame_start}-{frame_start + frame_count - 1}.")
            return None
        abs_ranges = [list(map(int, r)) for r in entry["active_ranges"]]
        if _sc.ranges_digest(abs_ranges) != entry.get("digest"):
            self._log("[StageCache] Stage 1 not used: its ranges do not match their recorded digest.")
            return None
        ranges, mask = _sc.rebase_ranges(abs_ranges, frame_start, frame_count)
        active = int(mask.sum())
        stats = {"active_frames": active, "total_frames": int(frame_count),
                 "skipped_frames": int(frame_count) - active,
                 "percentage": round(active / frame_count * 100.0, 1) if frame_count else 0.0,
                 "ranges": ranges, "mode": entry.get("mode"), "restored": True}
        self._stage1_ranges_abs = [[a + frame_start, b + frame_start] for a, b in ranges]
        self.report["stage1"] = {"ranges": len(ranges), "active_frames": active}
        return ranges, mask, stats

    # -- Stage 2 --------------------------------------------------------------------------------
    @staticmethod
    def _current_ranges_abs(pm: Any, frame_start: int):
        """The ranges THIS run's Stage 2 sees (restored or freshly scanned), in absolute frames; None when
        Stage 1 did not run."""
        return _sc.ranges_to_abs(getattr(pm, "_active_ranges", None), frame_start)

    def save_stage2(self, pm: Any, *, frame_start: int, frame_count: int) -> bool:
        path = _pc.stage_sidecar_path(self.project_id, "stage2")
        try:
            entry = _sc.write_stage2(path, pm, frame_start=frame_start, frame_count=frame_count,
                                     ranges_abs=self._current_ranges_abs(pm, frame_start),
                                     plan=getattr(pm, "stage2_frame_plan", None) or {})
        except _sc.Unserializable as exc:
            _swallowed("stage_checkpoint.py:save_stage2", exc, "a value the cache cannot restore exactly; not cached")
            self._log(f"[StageCache] Stage 2 not recorded: {exc}")
            return False
        except Exception as exc:
            _swallowed("stage_checkpoint.py:save_stage2_write", exc, "this run's Stage 2 is not cached")
            self._log(f"[StageCache] Stage 2 not recorded ({type(exc).__name__}: {exc})")
            try:
                os.remove(path)
            except OSError:
                pass
            return False
        try:
            saved = _pc.record_stage(self.project_id, "stage2", entry, live=self._live)
        except Exception as exc:
            _swallowed("stage_checkpoint.py:record_stage2", exc, "this run's Stage 2 is not cached")
            self._log(f"[StageCache] Stage 2 could not be recorded: {exc}")
            saved = None
        if saved is None:
            self._log("[StageCache] Stage 2 not recorded (the project's input digests are not final).")
            return False
        mb = entry["sidecar"]["size"] / 1048576.0
        self._log(f"[StageCache] Stage 2 recorded: {entry['faces']} face(s) on {entry['frames']} frame(s), "
                  f"{entry['plan_frames']} plan frame(s), {mb:.1f} MB.")
        return True

    def restore_stage2(self, pm: Any, *, frame_start: int, frame_count: int) -> bool:
        entry = self.usable().get("stage2")
        if not entry:
            return False
        from roop.stage2_plan import build_stage2_plan
        path = _pc.stage_sidecar_path(self.project_id, "stage2")
        try:
            report = _sc.restore_stage2(path, pm, new_start=frame_start, new_count=frame_count,
                                        current_ranges_abs=self._current_ranges_abs(pm, frame_start),
                                        build_plan=build_stage2_plan)
        except _sc.Unserializable as exc:
            _swallowed("stage_checkpoint.py:restore_stage2", exc, "the cache is refused; Stage 2 runs")
            self._log(f"[StageCache] Stage 2 not used: {exc}")
            return False
        except Exception as exc:
            _swallowed("stage_checkpoint.py:restore_stage2_read", exc, "the cache is refused; Stage 2 runs")
            self._log(f"[StageCache] Stage 2 not used ({type(exc).__name__}: {exc})")
            return False
        self.report["stage2"] = report
        self._log(f"[StageCache] Stage 2 restored from the checkpoint: {report['faces']} face(s) on "
                  f"{report['frames']} frame(s), {report['plan_frames']} plan frame(s) - the pre-pass is skipped.")
        return True
