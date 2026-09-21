"""Does the acceptance measurement actually catch the OLD bug?

The pre-fix UI, with two captured people and ONE source face, emitted [0, 1]
for every highlighted person: the highlighted person was ignored, so both
selections produced the same payload and the same swapped face. This replays
that exact payload against the live backend and shows the bug signature the
acceptance script is designed to fail on.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import acceptance_selected_face_swap as A  # noqa: E402
import fixtures  # noqa: E402
import numpy as np  # noqa: E402


def main():
    try:
        A.get_json("/api/state", timeout=10)
    except Exception as exc:
        print(f"SKIP: backend unreachable ({exc})")
        return 0

    A.post("/api/source/clear", {})
    A.post("/api/target/clear", {})
    src = fixtures.clip("faceset-v1-backup-2026-09-03/person_a.png")
    with open(src, "rb") as fh:
        A.post_file("/api/source/add", "files", os.path.basename(src),
                    fh.read(), "image/png")
    A.post("/api/target/add_path", {"paths": [fixtures.clip("double/d1.mp4")]})
    A.post("/api/target/auto_capture",
           {"index": 0, "people": 2, "replace": True, "time_budget": 60.0},
           timeout=900)

    frame = None
    for cand in range(1, 400, 6):
        r = A.post("/api/preview", {"frame": cand, "index": 0, "target_index": 0,
                                    "source_index": 0, "fake_preview": False,
                                    "detection": "Selected face"}, timeout=60)
        if len({i for i in (r.get("person_ids") or []) if i is not None}) >= 2:
            frame = cand
            break
    if frame is None:
        print("SKIP: no two-person frame")
        return 0

    def pv(mapping):
        r = A.post("/api/preview", {"frame": frame, "index": 0, "target_index": 0,
                                    "source_index": 0, "fake_preview": True,
                                    "detection": "Selected face",
                                    "face_mapping": mapping})
        return A.decode(r["image"])

    base = pv([-1, -1])

    # OLD payload: identical no matter which person was highlighted.
    old_p0 = A.diff_mask(base, pv([0, 1]))
    old_p1 = A.diff_mask(base, pv([0, 1]))
    # NEW payloads: distinct per highlighted person.
    new_p0 = A.diff_mask(base, pv([0, -1]))
    new_p1 = A.diff_mask(base, pv([-1, 0]))

    THRESHOLD = 0.25  # the bar used by acceptance_selected_face_swap.py
    old_iou = A.iou(old_p0, old_p1)
    new_iou = A.iou(new_p0, new_p1)
    print(f"  OLD payload: the two selections overlap IoU={old_iou:.3f} "
          f"(same face moved -> the reported bug)")
    print(f"  NEW payload: the two selections overlap IoU={new_iou:.3f} "
          f"(different faces moved -> fixed)")
    # The old payload does not reach IoU 1.0 even though it swaps the same
    # face twice: the pipeline is not bit-deterministic across calls, so the
    # blended edge differs slightly run to run. What matters is the SEPARATION
    # between the two cases against the acceptance bar.
    ok = old_iou > THRESHOLD and new_iou < THRESHOLD
    print(f"\n  acceptance bar: IoU < {THRESHOLD}")
    print(f"  old payload would FAIL it: {old_iou > THRESHOLD}")
    print(f"  new payload PASSES it:     {new_iou < THRESHOLD}")
    print(f"  separation: {old_iou / max(new_iou, 1e-6):.0f}x")
    print(f"\nDISCRIMINATES -> {ok}")

    A.post("/api/source/clear", {})
    A.post("/api/target/clear", {})
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
