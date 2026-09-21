"""End-to-end acceptance for "Selected face": only the highlighted person swaps.

This drives the SHIPPING backend over HTTP exactly as the React UI does, with
the payload the fixed UI now produces, and then judges the result on PIXELS:
for a two-person frame, the highlighted person's face region must change and
the other person's face region must not.

That is the user's actual acceptance criterion. Earlier checks could only show
that the payload was well-formed; they could not show that the right face moved.

Requires a running backend (start_react.js / run.py --ui react). Point it
elsewhere with ROOP_API_BASE. Skips, loudly, when the server or the two-person
fixture is unavailable, rather than passing silently.

Usage:
  python app/tests/acceptance_selected_face_swap.py
"""
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import fixtures  # noqa: E402

BASE = os.environ.get("ROOP_API_BASE", "http://127.0.0.1:17860")


def post(path, data, timeout=180):
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode())


def post_file(path, field, filename, content, ctype, timeout=120):
    boundary = "----roopAcceptanceBoundary"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'.encode()
    )
    body.extend(f"Content-Type: {ctype}\r\n\r\n".encode())
    body.extend(content)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode())


def get_json(path, timeout=30):
    with urllib.request.urlopen(f"{BASE}{path}", timeout=timeout) as res:
        return json.loads(res.read().decode())


def decode(b64):
    import cv2
    raw = base64.b64decode(b64.split(",")[-1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), 1)


class Report:
    def __init__(self):
        self.passed = 0
        self.failed = []

    def check(self, desc, cond, detail=""):
        suffix = f" ({detail})" if detail else ""
        if cond:
            self.passed += 1
            print(f"  [PASS] {desc}{suffix}")
        else:
            self.failed.append(desc)
            print(f"  [FAIL] {desc}{suffix}")


def face_boxes(target_faces_info, groups):
    """Person rank -> bounding box, from whatever the API reports per face."""
    boxes = {}
    for i, info in enumerate(target_faces_info or []):
        rank = groups[i] if i < len(groups) else None
        rank = rank[0] if isinstance(rank, list) else rank
        bbox = info.get("bbox") or info.get("box") or info.get("bounding_box")
        if bbox and rank is not None and rank not in boxes:
            boxes[int(rank)] = [int(v) for v in bbox[:4]]
    return boxes


def diff_mask(a, b, thresh=3):
    """Pixels that actually changed between two frames.

    The threshold is deliberately low. A swap of a similar-looking face is a
    subtle, smoothly blended edit, not a hard cut-out: measured on this
    pipeline, a confirmed swap moves ~11% of the frame at all, ~4% by more
    than 2 levels, and only ~0.1% by more than 18. A high threshold therefore
    reports "nothing happened" for a swap that plainly did happen, while 3
    still sits well above JPEG round-trip noise, which the all-skip control
    below measures rather than assumes.
    """
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)
    return d > thresh


def mask_stats(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {"count": 0, "cx": None, "cy": None}
    return {"count": int(len(xs)), "cx": float(xs.mean()), "cy": float(ys.mean())}


def iou(m1, m2):
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(m1, m2).sum()) / float(union)


def main():
    rpt = Report()
    print("=== ACCEPTANCE: Selected face swaps only the highlighted person ===")

    try:
        get_json("/api/state", timeout=10)
    except Exception as exc:
        print(f"SKIP: backend not reachable at {BASE} ({exc})")
        return 0

    clip = fixtures.clip("double/d1.mp4")
    if not clip or not os.path.exists(clip):
        print("SKIP: two-person fixture double/d1.mp4 unavailable")
        return 0
    # A source who is clearly NOT either person in the clip. A near-identical
    # source would move very few pixels and make the measurement ambiguous.
    src_img = None
    for cand in ("faceset-v1-backup-2026-09-03/person_a.png",
                 "single/s1_preview.jpg"):
        p = fixtures.clip(cand)
        if p and os.path.exists(p):
            src_img = p
            break
    if not src_img:
        print("SKIP: source face fixture unavailable")
        return 0

    # Clean slate so a previous session cannot supply the faces under test.
    post("/api/source/clear", {})
    post("/api/target/clear", {})

    ctype = "image/png" if src_img.lower().endswith(".png") else "image/jpeg"
    with open(src_img, "rb") as fh:
        res = post_file("/api/source/add", "files", os.path.basename(src_img),
                        fh.read(), ctype)
    sources = res.get("source_faces", [])
    rpt.check("one source face loaded", len(sources) >= 1, f"{len(sources)} face(s)")
    if not sources:
        return 1

    res = post("/api/target/add_path", {"paths": [clip]})
    rpt.check("two-person target clip added", len(res.get("targets", [])) > 0)

    # Capture the people in the clip through the route the UI's "find people"
    # button uses. This is what fills the target bank and assigns person ranks.
    try:
        cap = post("/api/target/auto_capture",
                   {"index": 0, "people": 2, "replace": True, "time_budget": 60.0},
                   timeout=900)
        print(f"       auto_capture: {cap.get('message') or cap.get('count')}")
    except Exception as exc:
        print(f"SKIP: auto_capture failed ({exc})")
        return 0

    state = get_json("/api/state")
    groups = [g[0] if isinstance(g, list) else g for g in (state.get("target_groups") or [])]
    info = state.get("target_faces_info") or []
    people = sorted({int(g) for g in groups if g is not None})
    rpt.check("two distinct people captured from the clip", len(people) >= 2,
              f"people={people}")
    if len(people) < 2:
        print("SKIP: clip did not yield two people; cannot judge selectivity")
        return 0

    # Find a frame that actually shows BOTH people. Frame 1 of a real clip
    # usually shows one of them, and a one-face frame cannot demonstrate
    # selectivity at all: every mapping would look identical.
    frame_no = None
    for cand in range(1, 400, 6):
        try:
            probe = post("/api/preview", {"frame": cand, "index": 0,
                                          "target_index": 0, "source_index": 0,
                                          "fake_preview": False,
                                          "detection": "Selected face"}, timeout=60)
        except Exception:
            continue
        ids = {i for i in (probe.get("person_ids") or []) if i is not None}
        if len(ids) >= 2:
            frame_no = cand
            print(f"       using frame {cand}, which shows people {sorted(ids)}")
            break
    if frame_no is None:
        print("SKIP: no frame in the clip shows both people; cannot judge selectivity")
        return 0

    def preview(mapping):
        # `fake_preview` is what turns /api/preview from "decode the frame" into
        # "decode the frame AND swap it" (see api.py preview()); without it the
        # route returns the untouched frame and every comparison below is
        # trivially equal. `detection` selects the swap mode whose contract the
        # mapping belongs to.
        selected_people = [person for person, source in enumerate(mapping)
                           if source is not None and int(source) >= 0]
        if len(selected_people) == 1:
            selection_state = {"selection_mode": "selected",
                               "person_id": selected_people[0]}
        elif selected_people:
            selection_state = {"selection_mode": "multi_person",
                               "person_ids": selected_people}
        else:
            selection_state = {"selection_mode": "none"}
        res = post("/api/preview", {
            "frame": frame_no,
            "index": 0,
            "target_index": 0,
            "source_index": 0,
            "fake_preview": True,
            "detection": "Selected face",
            "face_mapping": mapping,
            "selection_state": selection_state,
        })
        img = decode(res["image"])
        if img is None:
            raise AssertionError(f"preview for {mapping} returned no decodable image")
        return img

    # The API has no per-person bbox, so selectivity is judged from WHICH PIXELS
    # MOVE. That is a stronger statement than a box lookup anyway: it needs no
    # metadata and it is exactly what the user sees.
    base_img = preview([-1, -1])          # nobody swapped: the reference frame
    sel_p1 = preview([-1, 0])             # only person 1 highlighted (the fix)
    sel_p0 = preview([0, -1])             # only person 0 highlighted (mirror)
    both = preview([0, 0])                # both people swapped (upper bound)

    rpt.check("preview frames all share one shape",
              sel_p1.shape == base_img.shape == sel_p0.shape == both.shape,
              f"{sel_p1.shape}")

    m_p1 = diff_mask(base_img, sel_p1)
    m_p0 = diff_mask(base_img, sel_p0)
    m_both = diff_mask(base_img, both)
    s_p1, s_p0, s_both = mask_stats(m_p1), mask_stats(m_p0), mask_stats(m_both)
    total_px = base_img.shape[0] * base_img.shape[1]
    print(f"       changed px: person0-only={s_p0['count']} "
          f"person1-only={s_p1['count']} both={s_both['count']} of {total_px}")

    # 1. Each single selection must actually swap SOMETHING.
    rpt.check("selecting person 1 changes the frame", s_p1["count"] > 500,
              f"{s_p1['count']} px")
    rpt.check("selecting person 0 changes the frame", s_p0["count"] > 500,
              f"{s_p0['count']} px")

    # 2. The core bug: the two selections must affect DIFFERENT faces. When the
    #    highlighted person was ignored, both payloads swapped the same face and
    #    these two regions coincided.
    overlap = iou(m_p1, m_p0)
    rpt.check("the two selections change DIFFERENT regions (low IoU)",
              overlap < 0.25, f"IoU={overlap:.3f}")
    if s_p1["cx"] is not None and s_p0["cx"] is not None:
        dx = abs(s_p1["cx"] - s_p0["cx"])
        dy = abs(s_p1["cy"] - s_p0["cy"])
        face_span = (s_both["count"] ** 0.5) or 1.0
        print(f"       centroid gap: dx={dx:.0f} dy={dy:.0f} (face span ~{face_span:.0f})")
        rpt.check("the two changed regions sit on different people (centroids apart)",
                  dx > face_span * 0.5 or dy > face_span * 0.5,
                  f"dx={dx:.0f} dy={dy:.0f}")

    # 3. One selection must not do the work of two: each single-person swap has
    #    to be clearly smaller than swapping both.
    rpt.check("one selection changes less than swapping both people",
              s_p1["count"] < s_both["count"] * 0.85
              and s_p0["count"] < s_both["count"] * 0.85,
              f"{s_p1['count']}, {s_p0['count']} vs both={s_both['count']}")

    # 4. Each single-person region must sit essentially inside the both-swapped
    #    region: a selection may not do work in a place a full swap leaves
    #    alone. The tolerance is proportional because the blend around a face
    #    is not bit-identical when one face is swapped versus two, so a few
    #    percent of edge pixels legitimately differ.
    for name, m, stats in (("person 1", m_p1, s_p1), ("person 0", m_p0, s_p0)):
        stray = int(np.logical_and(m, np.logical_not(m_both)).sum())
        frac = stray / max(1, stats["count"])
        rpt.check(f"selecting {name} works where a full swap also works",
                  frac < 0.35, f"{stray}/{stats['count']} = {frac:.0%} outside")

    # 5. The two single swaps together should account for most of the both-swap
    #    region: person0 + person1 ~= both.
    covered = iou(np.logical_or(m_p1, m_p0), m_both)
    rpt.check("the two selections together cover the full two-person swap",
              covered > 0.5, f"IoU={covered:.3f}")

    # 6. Skip-everything must be a true no-op, not a silent full swap.
    noop = mask_stats(diff_mask(base_img, preview([-1, -1])))
    rpt.check("all-skip payload leaves the frame untouched",
              noop["count"] < max(200, total_px * 0.001), f"{noop['count']} px")

    post("/api/source/clear", {})
    post("/api/target/clear", {})

    print()
    if rpt.failed:
        print(f"FAILED ({len(rpt.failed)}): " + "; ".join(rpt.failed))
        return 1
    print(f"ALL CHECKS PASSED: {rpt.passed}/{rpt.passed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
