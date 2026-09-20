"""Stage 15 real-file acceptance: D:\\Monica Bellucci .mp4 x the user's harjot.fsz.

External harness against the SHIPPING backend over HTTP.  It sends the same
canonical `processing_selection` the React UI sends (Stage 14), reads the
same echoes back, and judges identity on the backend's own records plus the
pixels of the preview frames.  It never invents a faceset or a video: both
inputs are discovered and their identity (size/sha256) is written into the
report, and the run aborts if either is missing.

    python tools/acceptance_stage15.py [--base http://127.0.0.1:17860]
                                       [--target "D:/Monica Bellucci .mp4"]
                                       [--faceset app/facesets/harjot.fsz]
                                       [--render-frames 60]

Report: output/acceptance_stage15/report.json (+ preview PNGs).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "acceptance_stage15"


def sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class Api:
    def __init__(self, base):
        self.base = base

    def get(self, path, **kw):
        kw.setdefault("timeout", 120)
        r = requests.get(self.base + path, **kw)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"text": r.text[:2000]}

    def post(self, path, body=None, **kw):
        kw.setdefault("timeout", 600)
        r = requests.post(self.base + path, json=body if body is not None else {}, **kw)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"text": r.text[:2000]}

    def upload(self, path, file_path, field="files"):
        with open(file_path, "rb") as fh:
            r = requests.post(self.base + path, files=[(field, (os.path.basename(file_path), fh,
                                                               "application/octet-stream"))], timeout=600)
        return r.status_code, r.json()


def decode(data_url):
    raw = base64.b64decode(data_url.split(",", 1)[-1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def box_change(a, b, box, thresh=6):
    """Fraction of pixels inside `box` whose absolute difference exceeds thresh."""
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(a.shape[1], x1), min(a.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    d = np.abs(a[y0:y1, x0:x1].astype(np.int16) - b[y0:y1, x0:x1].astype(np.int16)).max(axis=2)
    return float((d > thresh).mean())


class Report:
    def __init__(self):
        self.results = []
        self.evidence = {}

    def add(self, name, status, evidence):
        self.results.append({"test": name, "status": status, "evidence": evidence})
        flag = {"PASS": "PASS", "FAIL": "FAIL", "NOT EXECUTED": "SKIP"}[status]
        print(f"  {flag:4}  {name} -- {evidence if isinstance(evidence, str) else json.dumps(evidence, default=str)[:300]}",
              flush=True)

    def check(self, name, cond, evidence):
        self.add(name, "PASS" if cond else "FAIL", evidence)
        return bool(cond)


def selection_for(state, person_id, source_id, mode="Selected face", version=None, mapping=None):
    """Exactly what react-ui/.../processingSelection.js builds."""
    version = version if version is not None else int(time.time() * 1000)
    mapping = mapping if mapping is not None else ({person_id: source_id} if person_id and source_id else {})
    sel_mode = "selected" if mode == "Selected face" else ("multi_person" if mode == "Selected people" else "none")
    refs = state.get("target_reference_face_ids") or []
    people = state.get("target_person_ids") or []
    ref_id = None
    if person_id and person_id in people:
        ref_id = refs[people.index(person_id)]
    return {
        "schema": 1,
        "request_id": f"acc-{version}-{os.urandom(3).hex()}",
        "selection_version": version,
        "mapping_version": "acc",
        "target_media_id": state.get("target_media_id"),
        "target_person_id": person_id if sel_mode == "selected" else None,
        "target_person_ids": [person_id] if (sel_mode == "selected" and person_id) else list(mapping.keys()) if sel_mode == "multi_person" else [],
        "target_reference_face_id": ref_id,
        "source_identity_id": source_id,
        "detection_mode": mode,
        "target_person_source_mapping": mapping,
        "selection_state": {
            "selection_mode": sel_mode,
            "person_id": person_id if sel_mode == "selected" else None,
            "person_ids": [person_id] if (sel_mode == "selected" and person_id) else list(mapping.keys()) if sel_mode == "multi_person" else [],
            "target_reference_face_id": ref_id,
        },
    }


def preview_body(settings, state, frame, selection, fake=True):
    s = settings
    body = {
        "index": state.get("selected_target_index", 0), "frame": frame, "fake_preview": fake,
        "target_media_id": selection["target_media_id"],
        "enhancer": "None",  # identity is judged on the swap, keep the preview cheap
        "detection": selection["detection_mode"],
        "face_distance": float(s.get("max_face_distance", 0.75)), "blend_ratio": float(s.get("blend_ratio", 0.8)),
        "mask_engine": s.get("mask_engine", "None"), "mask_engine_2": s.get("mask_engine_2", "None"),
        "clip_text": s.get("mask_clip_text", ""),
        "swap_model": s.get("swap_model", "inswapper"),
        "upscale": s.get("subsample_upscale", "256px"),
        "num_swap_steps": 1,
        "processing_selection": selection,
        "selection_version": selection["selection_version"],
        "selection_state": selection["selection_state"],
        "target_person_source_mapping": selection["target_person_source_mapping"],
        "selected_source_id": selection["source_identity_id"],
    }
    return body


def swap_body(settings, state, selection, enhancer="None"):
    body = preview_body(settings, state, 1, selection, fake=True)
    body.pop("index", None); body.pop("frame", None); body.pop("fake_preview", None)
    body.update({
        "enhancer": enhancer, "output_method": "File",
        "video_method": "In-Memory processing",
        "track_identities": bool(settings.get("track_identities", False)),
        "upscale_after_swap": False,
        "interp_after_swap": "off",
    })
    return body


def wait_render(api, log, timeout_s=3600):
    t0 = time.time()
    last_report = 0.0
    while time.time() - t0 < timeout_s:
        code, p = api.get("/api/progress", timeout=30)
        if not p.get("processing"):
            return p
        if time.time() - last_report > 180:
            last_report = time.time()
            print(f"    [render] {p.get('desc')} fps={p.get('fps')} progress={p.get('progress')}", flush=True)
        time.sleep(2)
    return {"timeout": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:17860")
    ap.add_argument("--target", default="D:/Monica Bellucci .mp4")
    ap.add_argument("--faceset", default=str(ROOT / "app" / "facesets" / "harjot.fsz"))
    ap.add_argument("--render-frames", type=int, default=60)
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--providers", default="",
                    help="DEPRECATED: a settings write does not switch loaded sessions; restart the "
                         "backend with --execution-provider and use --tag instead")
    ap.add_argument("--tag", default="", help="report/preview subfolder, e.g. the provider under test")
    ap.add_argument("--face-index", type=int, default=None,
                    help="detected-face index (left-to-right) to capture as the intended target on the first "
                         "face frame; default = largest box. Use 1 for Monica on frame 1 of the real clip.")
    args = ap.parse_args()

    global OUT
    if args.tag:
        OUT = OUT / args.tag
    OUT.mkdir(parents=True, exist_ok=True)
    api = Api(args.base)
    rep = Report()
    target = Path(args.target)
    faceset = Path(args.faceset)

    print("== Inputs")
    rep.evidence["inputs"] = {
        "target": {"path": str(target), "exists": target.exists(),
                   "size": target.stat().st_size if target.exists() else None,
                   "sha256": sha256(target) if target.exists() else None},
        "faceset": {"path": str(faceset), "exists": faceset.exists(),
                    "size": faceset.stat().st_size if faceset.exists() else None,
                    "sha256": sha256(faceset) if faceset.exists() else None},
    }
    print(json.dumps(rep.evidence["inputs"], indent=1))
    if not target.exists() or not faceset.exists():
        rep.add("real-file acceptance", "NOT EXECUTED", "target or faceset missing")
        (OUT / "report.json").write_text(json.dumps({"results": rep.results, "evidence": rep.evidence}, indent=1, default=str))
        return 2

    code, meta = api.get("/api/meta")
    rep.evidence["meta"] = {k: meta.get(k) for k in ("active_provider", "git_version", "providers", "swap_models")}
    print("backend:", rep.evidence["meta"])
    code, settings = api.get("/api/settings")
    rep.evidence["settings"] = {k: settings.get(k) for k in (
        "face_detection_mode", "provider", "swap_model", "mask_engine", "max_face_distance",
        "selected_enhancer", "track_identities", "trt_precision")}
    print("settings:", rep.evidence["settings"])

    # ── 1. clean state ─────────────────────────────────────────────────
    print("== 1. clean state")
    api.post("/api/source/clear"); api.post("/api/target/clear")
    code, st = api.get("/api/state")
    rep.check("clean state", st.get("faceset_count") == 0 and not st.get("targets") and not st.get("target_faces"),
              {"facesets": st.get("faceset_count"), "targets": len(st.get("targets") or []),
               "target_faces": len(st.get("target_faces") or [])})

    # ── 2/3. faceset ───────────────────────────────────────────────────
    print("== 2/3. load Harjot faceset")
    code, res = api.upload("/api/source/add", str(faceset))
    info = (res.get("source_faces_info") or [None])[0]
    rep.check("harjot.fsz loads as ONE source identity", code == 200 and res.get("faceset_count") == 1 and not res.get("errors"),
              {"http": code, "faceset_count": res.get("faceset_count"), "errors": res.get("errors")})
    rep.check("harjot angles stay under one identity (5 refs, front+profiles)",
              bool(info) and info.get("count", 0) >= 2 and len(set(info.get("poses") or [])) >= 2,
              {"count": info and info.get("count"), "poses": info and info.get("poses"), "id": info and info.get("id")})
    source_id = info["id"] if info else None
    code, st = api.get("/api/state")
    rep.check("loading a source populates no target-face state",
              not st.get("target_faces") and not st.get("target_person_ids"),
              {"target_faces": len(st.get("target_faces") or []), "target_person_ids": st.get("target_person_ids")})

    # ── 4. target ──────────────────────────────────────────────────────
    print("== 4. load target video")
    code, res = api.post("/api/target/add_path", {"paths": [str(target)]})
    code, st = api.get("/api/state")
    media_id = st.get("target_media_id")
    tinfo = (st.get("targets") or [{}])[0]
    rep.check("target video loaded with a stable media id", bool(media_id) and len(st.get("targets") or []) == 1,
              {"media_id": media_id, "frames": tinfo.get("frames"), "name": tinfo.get("name")})
    rep.check("loading a target does not touch source identity",
              (st.get("source_faces_info") or [{}])[0].get("id") == source_id and st.get("faceset_count") == 1,
              {"source_id_after": (st.get("source_faces_info") or [{}])[0].get("id")})
    total_frames = int(tinfo.get("frames") or 0)

    # ── 5/6. no explicit target: no swap ───────────────────────────────
    print("== 5/6. preview without any captured target person")
    no_target_sel = selection_for(st, None, source_id)
    # Find a frame with faces first (raw detection only).
    frame = None
    faces = []
    for cand in range(1, min(total_frames, 900), 15):
        code, r = api.post("/api/preview", preview_body(settings, st, cand, no_target_sel, fake=False))
        if len(r.get("faces") or []) >= 1:
            frame, faces = cand, r["faces"]
            break
    rep.check("a frame with detected faces exists", frame is not None, {"frame": frame, "faces": faces})
    if frame is None:
        (OUT / "report.json").write_text(json.dumps({"results": rep.results, "evidence": rep.evidence}, indent=1, default=str))
        return 1
    code, raw = api.post("/api/preview", preview_body(settings, st, frame, no_target_sel, fake=False))
    raw_img = decode(raw["image"]); cv2.imwrite(str(OUT / f"frame{frame}_raw.png"), raw_img)
    code, r = api.post("/api/preview", preview_body(settings, st, frame, no_target_sel, fake=True))
    img = decode(r["image"])
    changes = [round(box_change(raw_img, img, b), 4) for b in faces]
    rep.check("TEST 1: no target selected -> 0 swapped faces, diagnostic returned",
              r.get("selection_diagnostic") in ("target_required", "selection_required") and max(changes) < 0.01,
              {"diagnostic": r.get("selection_diagnostic"), "message": r.get("message"),
               "face_change_fractions": changes, "request_id_echo": r.get("request_id") == no_target_sel["request_id"]})
    code, r = api.post("/api/swap", swap_body(settings, st, no_target_sel))
    rep.check("TEST 1b: /api/swap refuses without a target person", code == 409,
              {"http": code, "body": {k: r.get(k) for k in ("message", "selection_diagnostic")}})

    # ── 7-10. explicit capture ─────────────────────────────────────────
    print("== 7-10. inspect detections, capture ONE face explicitly")
    rep.evidence["detections"] = {"frame": frame, "faces": faces, "person_ids": raw.get("person_ids")}
    # A multi-person frame is preferred for TEST 2; search a little further.
    multi_frame, multi_faces = None, []
    for cand in range(1, min(total_frames, 3000), 25):
        code, r = api.post("/api/preview", preview_body(settings, st, cand, no_target_sel, fake=False))
        if len(r.get("faces") or []) >= 2:
            multi_frame, multi_faces = cand, r["faces"]
            break
    rep.evidence["multi_person_frame"] = {"frame": multi_frame, "faces": multi_faces}
    capture_frame = frame
    face_index = 0
    if args.face_index is not None and 0 <= args.face_index < len(faces):
        face_index = args.face_index
    elif faces and len(faces) > 1:
        face_index = int(np.argmax([(b[2]-b[0])*(b[3]-b[1]) for b in faces]))
    rep.evidence["captured_face"] = {"frame": capture_frame, "face_index": face_index, "bbox": faces[face_index]}
    code, cap = api.post("/api/target/use_face", {"index": 0, "frame": capture_frame, "target_media_id": media_id,
                                                  "face_index": face_index})
    people = cap.get("target_person_ids") or []
    rep.check("capture of ONE face creates exactly one target person",
              code == 200 and len(set(people)) == 1 and len(cap.get("target_faces") or []) == 1,
              {"http": code, "person_ids": people, "selected_target_person_id": cap.get("selected_target_person_id"),
               "target_media_id": cap.get("target_media_id")})
    person_a = people[0] if people else None
    code, st = api.get("/api/state")
    rep.check("target capture did not modify the source faceset identity",
              (st.get("source_faces_info") or [{}])[0].get("id") == source_id
              and (st.get("source_faces_info") or [{}])[0].get("count") == info.get("count"),
              {"source_after": (st.get("source_faces_info") or [{}])[0]})

    # ── 11-13. map to Harjot, preview: only the selected identity routes ─
    print("== 11-13. map person -> Harjot, preview")
    code, ctx = api.post("/api/target/context", {"target_media_id": media_id, "selection_version": int(time.time()*1000),
                                                 "selected_target_person_id": person_a,
                                                 "target_person_source_mapping": {person_a: source_id}})
    rep.check("mapping persisted on the backend", ctx.get("target_person_source_mapping", {}).get(person_a) == source_id
              and ctx.get("selected_target_person_id") == person_a, ctx.get("target_person_source_mapping"))
    code, st = api.get("/api/state")
    sel_a = selection_for(st, person_a, source_id)
    code, pa = api.post("/api/preview", preview_body(settings, st, frame, sel_a, fake=True))
    img_a = decode(pa["image"]); cv2.imwrite(str(OUT / f"frame{frame}_swapped_personA.png"), img_a)
    ch = [round(box_change(raw_img, img_a, b), 4) for b in faces]
    echo = pa.get("processing_selection") or {}
    rep.check("preview echoes the canonical selection (media/person/source)",
              pa.get("request_id") == sel_a["request_id"] and echo.get("target_media_id") == media_id
              and echo.get("target_person_id") == person_a and echo.get("source_identity_id") == source_id,
              {"request_id": pa.get("request_id"), "echo": {k: echo.get(k) for k in ("target_media_id", "target_person_id", "source_identity_id", "target_person_source_mapping")},
               "preview_signature": pa.get("preview_signature"), "diagnostic": pa.get("selection_diagnostic")})
    rep.check("preview swapped the captured face (pixels changed inside its box)", ch[face_index] > 0.05,
              {"face_change_fractions": ch, "captured_face_index": face_index, "swap_audit": pa.get("swap_audit")})
    others = [c for i, c in enumerate(ch) if i != face_index]
    rep.check("preview left every other detected face untouched", all(c < 0.02 for c in others),
              {"other_faces": others})
    preview_route = {"target_media_id": echo.get("target_media_id"), "target_person_id": echo.get("target_person_id"),
                     "source_identity_id": echo.get("source_identity_id"), "mapping": echo.get("target_person_source_mapping")}

    # ── TEST 2: multi-person frame, select person B ────────────────────
    person_b = None
    if multi_frame is not None:
        print("== TEST 2/3. multi-person frame")
        code, r = api.post("/api/preview", preview_body(settings, st, multi_frame, sel_a, fake=False))
        m_faces = r.get("faces") or []
        m_raw = decode(r["image"]); cv2.imwrite(str(OUT / f"frame{multi_frame}_raw.png"), m_raw)
        # Person B = a face that is NOT person A (the preview labels each box with the captured person it matches)
        pids = r.get("person_ids") or []
        b_index = None
        for i, pid in enumerate(pids):
            if pid != 0:
                b_index = i
                break
        if b_index is None and len(m_faces) > 1:
            b_index = 1
        code, capb = api.post("/api/target/use_face", {"index": 0, "frame": multi_frame, "target_media_id": media_id,
                                                       "face_index": b_index})
        people_b = capb.get("target_person_ids") or []
        new_people = [p for p in dict.fromkeys(people_b) if p != person_a]
        person_b = new_people[0] if new_people else None
        rep.check("TEST 2: explicitly selecting person B adds exactly person B",
                  person_b is not None and len(set(people_b)) == 2,
                  {"person_ids": people_b, "person_b": person_b, "b_face_index": b_index, "preview_person_ids": pids})
        if person_b:
            api.post("/api/target/context", {"target_media_id": media_id, "selection_version": int(time.time()*1000),
                                             "selected_target_person_id": person_b,
                                             "target_person_source_mapping": {person_b: source_id}})
            code, st = api.get("/api/state")
            sel_b = selection_for(st, person_b, source_id, mapping={person_b: source_id})
            code, pb = api.post("/api/preview", preview_body(settings, st, multi_frame, sel_b, fake=True))
            img_b = decode(pb["image"]); cv2.imwrite(str(OUT / f"frame{multi_frame}_swapped_personB.png"), img_b)
            chb = [round(box_change(m_raw, img_b, b), 4) for b in m_faces]
            rep.check("TEST 2: only person B's face changes when B is selected",
                      chb[b_index] > 0.05 and all(c < 0.02 for i, c in enumerate(chb) if i != b_index),
                      {"change": chb, "b_index": b_index, "echo_person": (pb.get("processing_selection") or {}).get("target_person_id"),
                       "swap_audit": pb.get("swap_audit")})
            # TEST 3: switch back to A on the same frame -> B must not stay active
            sel_a2 = selection_for(st, person_a, source_id, mapping={person_a: source_id})
            api.post("/api/target/context", {"target_media_id": media_id, "selection_version": int(time.time()*1000),
                                             "selected_target_person_id": person_a,
                                             "target_person_source_mapping": {person_a: source_id}})
            code, pa2 = api.post("/api/preview", preview_body(settings, st, multi_frame, sel_a2, fake=True))
            img_a2 = decode(pa2["image"]); cv2.imwrite(str(OUT / f"frame{multi_frame}_swapped_personA_again.png"), img_a2)
            cha = [round(box_change(m_raw, img_a2, b), 4) for b in m_faces]
            a_present = any(p == 0 for p in pids)
            rep.check("TEST 3: switching the selection B -> A: B is no longer swapped",
                      cha[b_index] < 0.02 and (pa2.get("processing_selection") or {}).get("target_person_id") == person_a,
                      {"change": cha, "b_index": b_index, "person_a_visible_in_frame": a_present})
    else:
        rep.add("TEST 2: multi-person frame", "NOT EXECUTED", "no frame with >=2 detected faces in the first 3000 frames")
        rep.add("TEST 3: switch B -> C", "NOT EXECUTED", "needs the multi-person frame")

    # ── TEST 5: change mapping -> preview uses the new mapping ──────────
    print("== TEST 5. mapping change")
    code, st = api.get("/api/state")
    sel_skip = selection_for(st, person_a, source_id, mapping={})   # explicit: A mapped to nobody
    code, ps = api.post("/api/preview", preview_body(settings, st, frame, sel_skip, fake=True))
    img_s = decode(ps["image"])
    chs = [round(box_change(raw_img, img_s, b), 4) for b in faces]
    rep.check("TEST 5: with person A mapped to no source, A is NOT swapped (no redirect to source 0)",
              max(chs) < 0.02, {"change": chs, "echo_mapping": (ps.get("processing_selection") or {}).get("target_person_source_mapping")})
    sel_a3 = selection_for(st, person_a, source_id, mapping={person_a: source_id})
    code, pr = api.post("/api/preview", preview_body(settings, st, frame, sel_a3, fake=True))
    img_r = decode(pr["image"])
    chr_ = [round(box_change(raw_img, img_r, b), 4) for b in faces]
    rep.check("TEST 5b: restoring the mapping swaps A again", chr_[face_index] > 0.05,
              {"change": chr_, "swap_audit": pr.get("swap_audit")})

    # ── TEST 4: target media A -> B isolation ──────────────────────────
    print("== TEST 4. target media isolation")
    code, res = api.post("/api/target/add_path", {"paths": [str(target)]})   # same file, second entry = media B
    code, st2 = api.get("/api/state")
    ids = [t.get("media_id") for t in st2.get("targets") or []]
    other = [m for m in ids if m != media_id]
    if other:
        code, selb = api.post("/api/target/select", {"target_media_id": other[0]})
        rep.check("TEST 4: media B starts with no people / no mapping from media A",
                  not selb.get("target_person_ids") and not selb.get("target_person_source_mapping")
                  and selb.get("target_media_id") == other[0],
                  {"B": other[0], "people": selb.get("target_person_ids"), "mapping": selb.get("target_person_source_mapping")})
        code, back = api.post("/api/target/select", {"target_media_id": media_id})
        rep.check("TEST 4b: switching back to A restores A's person + mapping",
                  person_a in (back.get("target_person_ids") or []) and back.get("target_person_source_mapping", {}).get(person_a) == source_id,
                  {"people": back.get("target_person_ids"), "mapping": back.get("target_person_source_mapping")})
        api.post("/api/target/remove", {"target_media_id": other[0]})
    else:
        rep.add("TEST 4: media isolation", "NOT EXECUTED", "second target entry was not created")

    # ── TEST 9: session restore ────────────────────────────────────────
    print("== TEST 9. session restore (/api/state rehydrate)")
    code, st = api.get("/api/state")
    rep.check("TEST 9: /api/state restores media/person/mapping/source by id",
              st.get("target_media_id") == media_id and st.get("selected_target_person_id") == person_a
              and st.get("target_person_source_mapping", {}).get(person_a) == source_id
              and st.get("selected_source_id") == source_id,
              {"target_media_id": st.get("target_media_id"), "selected_target_person_id": st.get("selected_target_person_id"),
               "mapping": st.get("target_person_source_mapping"), "selected_source_id": st.get("selected_source_id")})

    # ── 14/15. final render, compare routing with the preview ──────────
    final_route = None
    if not args.skip_render:
        print(f"== 14/15. final render ({args.render_frames} frames)")
        api.post("/api/target/set_frame", {"which": "start", "frame": max(1, frame - 1), "target_media_id": media_id})
        api.post("/api/target/set_frame", {"which": "end", "frame": min(total_frames, frame - 1 + args.render_frames), "target_media_id": media_id})
        code, st = api.get("/api/state")
        sel_final = selection_for(st, person_a, source_id, mapping={person_a: source_id})
        code, outs_before = api.get("/api/output")
        output_dir = outs_before.get("output_path")
        render_started = time.time() - 2
        code, sw = api.post("/api/swap", swap_body(settings, st, sel_final))
        fecho = sw.get("processing_selection") or {}
        final_route = {"target_media_id": fecho.get("target_media_id"), "target_person_id": fecho.get("target_person_id"),
                       "source_identity_id": fecho.get("source_identity_id"), "mapping": fecho.get("target_person_source_mapping")}
        rep.check("final render accepted and froze the same canonical route as the preview",
                  code == 200 and sw.get("status") == "started" and final_route == preview_route,
                  {"http": code, "status": sw.get("status"), "preview_route": preview_route, "final_route": final_route,
                   "request_id": sw.get("request_id")})
        prog = wait_render(api, None)
        code, hist = api.get("/api/history")
        runs = hist.get("history") or hist.get("runs") or hist if isinstance(hist, list) else []
        rep.check("final render completed without error", not prog.get("error") and not prog.get("timeout") and prog.get("progress", 0) >= 0.99,
                  {"desc": prog.get("desc"), "error": prog.get("error"), "progress": prog.get("progress")})
        # Judge the OUTPUT on pixels, not on the return code: the newest video
        # in the output folder, first frame == the render's start frame.
        # /api/output lists only the first 50 names; read the folder directly
        # (same machine) and keep files written by THIS render.
        vids = []
        if output_dir and os.path.isdir(output_dir):
            for name in os.listdir(output_dir):
                full = os.path.join(output_dir, name)
                if name.lower().endswith((".mp4", ".mkv", ".webm", ".mov")) and os.path.getmtime(full) >= render_started:
                    vids.append({"name": name, "absolute_path": full, "mtime": os.path.getmtime(full)})
        newest = max(vids, key=lambda f: f.get("mtime", 0)) if vids else None
        rep.evidence["last_output"] = newest
        if newest and os.path.isfile(newest["absolute_path"]):
            cap = cv2.VideoCapture(newest["absolute_path"])
            ok, out0 = cap.read(); cap.release()
            src = cv2.VideoCapture(str(target)); src.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame - 1))
            ok2, raw0 = src.read(); src.release()
            if ok and ok2 and out0.shape == raw0.shape:
                cv2.imwrite(str(OUT / "render_first_frame.png"), out0)
                cho = [round(box_change(raw0, out0, b, thresh=12), 4) for b in faces]
                rep.check("final render output swaps the selected face and only that face",
                          cho[face_index] > 0.05 and all(c < 0.05 for i, c in enumerate(cho) if i != face_index),
                          {"file": newest["name"], "change_by_face": cho, "captured_face_index": face_index})
            else:
                rep.add("final render output pixel check", "NOT EXECUTED",
                        {"decoded": bool(ok and ok2), "shapes": [getattr(out0, "shape", None), getattr(raw0, "shape", None)]})
        else:
            rep.add("final render output pixel check", "NOT EXECUTED", "no output video found")
    else:
        rep.add("final render", "NOT EXECUTED", "--skip-render")

    # ── TEST 7: providers ──────────────────────────────────────────────
    if args.providers:
        print("== TEST 7. provider parity (routing only)")
        routes = {}
        for prov in [p.strip() for p in args.providers.split(",") if p.strip()]:
            code, r = api.post("/api/settings", {"provider": prov})
            code, st = api.get("/api/state")
            sel = selection_for(st, person_a, source_id, mapping={person_a: source_id})
            code, pp = api.post("/api/preview", preview_body(settings, st, frame, sel, fake=True))
            e = pp.get("processing_selection") or {}
            img_p = decode(pp["image"]) if pp.get("image") else raw_img
            routes[prov] = {"person": e.get("target_person_id"), "source": e.get("source_identity_id"),
                            "change": [round(box_change(raw_img, img_p, b), 4) for b in faces],
                            "active_provider": api.get("/api/meta")[1].get("active_provider"), "http": code}
        api.post("/api/settings", {"provider": settings.get("provider")})
        same = len({(v["person"], v["source"]) for v in routes.values()}) == 1
        rep.check("TEST 7: provider changes runtime, not the person/source route", same, routes)
    else:
        rep.add("TEST 7: CPU vs CUDA vs TensorRT", "NOT EXECUTED", "run with --providers cpu,cuda,tensorrt")

    # ── TEST 8 / PART E: clear, reload, retest ─────────────────────────
    print("== TEST 8 / E. clear target faces, reload target + faceset, re-select")
    api.post("/api/target/clear_faces")
    code, st = api.get("/api/state")
    rep.check("TEST 8: clear target faces removes the old identity",
              not st.get("target_person_ids") and not st.get("selected_target_person_id") and not st.get("target_person_source_mapping"),
              {"people": st.get("target_person_ids"), "selected": st.get("selected_target_person_id")})
    api.post("/api/target/clear"); api.post("/api/source/clear")
    code, res = api.upload("/api/source/add", str(faceset))
    source_id2 = (res.get("source_faces_info") or [{}])[0].get("id")
    code, res = api.post("/api/target/add_path", {"paths": [str(target)]})
    code, st = api.get("/api/state")
    media_id2 = st.get("target_media_id")
    rep.check("E: reloaded target has a NEW media id and no people",
              media_id2 and media_id2 != media_id and not st.get("target_person_ids"),
              {"old": media_id, "new": media_id2, "people": st.get("target_person_ids")})
    sel_none = selection_for(st, None, source_id2)
    code, r = api.post("/api/preview", preview_body(settings, st, frame, sel_none, fake=True))
    img_n = decode(r["image"])
    chn = [round(box_change(raw_img, img_n, b), 4) for b in faces]
    rep.check("E: after reload, no stale person -> no swap", max(chn) < 0.01 and r.get("selection_diagnostic"),
              {"change": chn, "diagnostic": r.get("selection_diagnostic")})
    code, cap2 = api.post("/api/target/use_face", {"index": 0, "frame": capture_frame, "target_media_id": media_id2, "face_index": face_index})
    person_a2 = (cap2.get("target_person_ids") or [None])[0]
    rep.check("E: re-capture creates a FRESH person id (old id gone)", person_a2 and person_a2 != person_a and len(set(cap2.get("target_person_ids") or [])) == 1,
              {"old": person_a, "new": person_a2})
    api.post("/api/target/context", {"target_media_id": media_id2, "selection_version": int(time.time()*1000),
                                     "selected_target_person_id": person_a2, "target_person_source_mapping": {person_a2: source_id2}})
    code, st = api.get("/api/state")
    sel2 = selection_for(st, person_a2, source_id2, mapping={person_a2: source_id2})
    code, p2 = api.post("/api/preview", preview_body(settings, st, frame, sel2, fake=True))
    img2 = decode(p2["image"]); cv2.imwrite(str(OUT / f"frame{frame}_swapped_after_clean_retest.png"), img2)
    ch2 = [round(box_change(raw_img, img2, b), 4) for b in faces]
    rep.check("E: clean-state retest swaps exactly the re-selected face", ch2[face_index] > 0.05 and all(c < 0.02 for i, c in enumerate(ch2) if i != face_index),
              {"change": ch2, "echo_person": (p2.get("processing_selection") or {}).get("target_person_id"),
               "swap_audit": p2.get("swap_audit")})
    # A stale write naming the OLD person must be refused
    code, stale = api.post("/api/target/context", {"target_media_id": media_id2, "selection_version": int(time.time()*1000),
                                                   "selected_target_person_id": person_a})
    rep.check("E: the old person id cannot be written back", code == 422, {"http": code, "body": stale})

    (OUT / "report.json").write_text(json.dumps({"results": rep.results, "evidence": rep.evidence}, indent=1, default=str), encoding="utf-8")
    failed = [r for r in rep.results if r["status"] == "FAIL"]
    skipped = [r for r in rep.results if r["status"] == "NOT EXECUTED"]
    print(f"\n== {len(rep.results) - len(failed) - len(skipped)} PASS, {len(failed)} FAIL, {len(skipped)} NOT EXECUTED -> {OUT / 'report.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
