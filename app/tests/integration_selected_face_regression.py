"""Integration-level regression harness for the "Selected Face" bug.

Runs the SHIPPING pipeline (real detector, real recogniser, real swapper,
real compositor) on a real, deterministic frame with three detectable faces,
and judges the result two ways -- neither of them a screenshot:

  * ROUTING   -- ``ProcessMgr._SWAP_LOG``, the settled (box -> source) decision;
  * IDENTITY  -- the recogniser run on the OUTPUT face: did the selected person
                 actually become the source identity, and did everyone else keep
                 theirs. Identity distance is robust to the codec noise that
                 makes a raw pixel diff useless.

The frame is composed once from four DISTINCT real identities so a correct swap
is unambiguous and reproducible on any machine that has the facesets:

    [ person_b ]    [ person_i ]     [ person_h ]
     person A     person B      bystander
      group 0      group 1      nobody's target

    source faceset = person_a (index 0) -- an identity NOT on the frame, so a
    correct swap of the selected person turns that face INTO person_a and every
    other face keeps its own identity.

Two things about the SCENARIO, both documented rather than worked around:

  * tracking is OFF, so the per-frame ``selected`` routing under test
    (roop.selected_routing) runs, not the identity-lock pre-pass.
  * the foreground occluder is OFF (``ROOP_OCCLUSION_MASK=0``). It is a
    real-content feature that misfires on a portrait pasted on a flat canvas --
    it reads the whole tile as background and masks the paste to nothing. On
    real video it composites correctly; here it would hide a swap that the
    routing correctly performed. This is scenario setup, not a threshold change.

Assertions (the eight the task lists):
  1 person A receives the source        -- A's box -> source 0, and A's OUTPUT
                                           identity is person_a's, not person_b's
  2 person B is unchanged                -- B's box absent; B's OUTPUT identity
                                           still person_i's
  3 no third detected face is swapped    -- exactly one box swapped; bystander's
                                           identity unchanged
  4 changing the selected person         -- select B: B swaps to person_a, A stays
    changes which person is eligible        person_b
  5 removing the selection -> zero swaps -- empty selection: nothing routed, no
                                           identity moves anywhere
  6 preview and render pick the same     -- process_frame is_preview True vs
                                           False route to the same box
  7 CPU/CUDA/TensorRT same routing       -- --providers cpu,cuda,tensorrt: the
                                           (box -> source) map is identical
  8 log the distance for every candidate -- the [SelectedRoute] line carries a
                                           distance for all three detected faces

Usage (from app/):
  env/Scripts/python.exe tests/integration_selected_face_regression.py
  env/Scripts/python.exe tests/integration_selected_face_regression.py --providers cpu,cuda,tensorrt
"""
import argparse
import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import fixtures  # noqa: E402

# Four DISTINCT local facesets (app/facesets/<name>.png + .fsz). The names are
# machine-specific, so they come from the environment; the defaults are
# placeholders that exist on no machine, and the harness says so and exits.
_env = os.environ.get
SOURCE_FACESET = _env("ROOP_TEST_SOURCE", "person_a")        # the identity applied to the selected person
PERSON_A_FACESET = _env("ROOP_TEST_PERSON_A", "person_b")    # person A: the target meant to receive the source
PERSON_B_FACESET = _env("ROOP_TEST_PERSON_B", "person_i")    # person B: an unrelated person
BYSTANDER_FACESET = _env("ROOP_TEST_BYSTANDER", "person_h")  # a third face nobody captured
LABELS = ["A", "B", "bystander"]
DEFAULT_OUT = os.path.join(APP, "output", "selected_face_integration")

# identity-distance bands. Same-identity sits ~0.0-0.2, cross-identity ~0.8-1.0,
# so 0.45 separates "became the source / kept its own" with a wide margin.
SAME = 0.45
DIFF = 0.60


def _portrait(name):
    import cv2
    for cand in (os.path.join(APP, "facesets", f"{name}.png"),
                 fixtures.clip(f"faceset-v1-backup-2026-09-03/{name}.png")):
        if cand and os.path.isfile(cand):
            img = cv2.imdecode(np.fromfile(cand, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                return img
    raise SystemExit(f"portrait not found for {name!r}")


def build_canvas(cell=512, gap=64):
    """Three portraits side by side -> a deterministic frame with three
    detectable faces, left-to-right A, B, bystander."""
    import cv2
    tiles = []
    for n in (PERSON_A_FACESET, PERSON_B_FACESET, BYSTANDER_FACESET):
        img = _portrait(n)
        h, w = img.shape[:2]
        s = cell / max(h, w)
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))))
        tile = np.full((cell, cell, 3), 32, np.uint8)
        y0, x0 = (cell - img.shape[0]) // 2, (cell - img.shape[1]) // 2
        tile[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
        tiles.append(tile)
    canvas = np.full((cell + 2 * gap, cell * 3 + gap * 4, 3), 32, np.uint8)
    for i, tile in enumerate(tiles):
        x = gap + i * (cell + gap)
        canvas[gap:gap + cell, x:x + cell] = tile
    return canvas


def _swap_model():
    import yaml
    with open(os.path.join(APP, "config.yaml"), encoding="utf-8") as fh:
        return str((yaml.safe_load(fh) or {}).get("swap_model", "hyperswap"))


def _nearest(faces, box):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return min(faces, key=lambda f: ((f.bbox[0] + f.bbox[2]) / 2 - cx) ** 2
              + ((f.bbox[1] + f.bbox[3]) / 2 - cy) ** 2)


def _label_of(box, det_boxes):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    i = min(range(len(det_boxes)), key=lambda k:
            (cx - (det_boxes[k][0] + det_boxes[k][2]) / 2) ** 2
            + (cy - (det_boxes[k][1] + det_boxes[k][3]) / 2) ** 2)
    return LABELS[i]


def _setup(provider, tracking):
    import angle_bench as ab
    import two_face_video as tfv  # applies config.yaml perf env at import
    # The occluder misfires on a synthetic portrait canvas (see the header).
    os.environ["ROOP_OCCLUSION_MASK"] = "0"
    os.environ["ROOP_LOG_SELECTED_ROUTE"] = "1"
    g = ab.init_pipeline(provider, _swap_model(), "None", "None", sync_config=True)
    g.execution_threads = 4
    g.face_swap_mode = "selected"
    g.track_identities = bool(tracking)
    g.CFG.track_identities = bool(tracking)
    g.temporal_detection = bool(tracking)
    g.CFG.temporal_detection = bool(tracking)
    g.enable_occlusion_mask = False
    from roop import ProcessMgr as _pm
    _pm._LOG_SELECTED_ROUTE = True
    return g, ab, tfv


def _selection(person):
    from roop.target_selection import normalize_target_selection
    raw = ({"selection_mode": "selected", "person_id": person}
           if person is not None else {"selection_mode": "none"})
    return normalize_target_selection(raw, person_count=2)


# -- the identity-verified render (run_swap: the proven compositing path) -------

def render_identity(provider, selection_person, out_dir, tracking=False):
    """Render one short clip of the canvas and read who became whom.

    Returns dict: routed {label->source}, identity {label->{to_source,to_original}},
    route_log (the last [SelectedRoute] line)."""
    import cv2
    import roop.globals as g
    from roop.face_util import get_all_faces
    from roop.utilities import compute_cosine_distance
    from roop import ProcessMgr as _pm

    g_, ab, tfv = _setup(provider, tracking)
    canvas = build_canvas()
    faces = sorted(get_all_faces(canvas) or [], key=lambda f: float(f.bbox[0]))
    if len(faces) < 3:
        raise SystemExit(f"expected >=3 detectable faces, got {len(faces)}")
    det_boxes = [[float(v) for v in f.bbox] for f in faces[:3]]
    orig_emb = {LABELS[i]: np.asarray(faces[i].embedding) for i in range(3)}
    source = tfv.load_library_faceset(SOURCE_FACESET)
    src_emb = np.asarray(source.faces[0].embedding)

    work = os.path.join(out_dir, f"{provider}_sel{selection_person}")
    os.makedirs(work, exist_ok=True)
    vid = os.path.join(work, "canvas.mp4")
    vw = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"mp4v"), 25,
                         (canvas.shape[1], canvas.shape[0]))
    for _ in range(12):
        vw.write(canvas)
    vw.release()

    options = ab.build_options(g_, _swap_model(), None, False)
    options.selection_state = _selection(selection_person)

    g.INPUT_FACESETS = [source]
    g.TARGET_FACES = [faces[0], faces[1]]
    g.TARGET_FACE_GROUP = [0, 1]

    _pm._SWAP_LOG = {}
    buf = io.StringIO()
    with redirect_stdout(buf):
        out_path, (swap_log, _facelog) = tfv.run_swap(
            vid, [source], [faces[0], faces[1]], [0, 1], options, work)
    routed = {}
    for _f, entries in (swap_log or {}).items():
        for box, src in entries:
            routed[_label_of(box, det_boxes)] = int(src)
    route_lines = [ln for ln in buf.getvalue().splitlines() if "[SelectedRoute]" in ln]

    identity = {}
    if out_path and os.path.exists(out_path):
        cap = cv2.VideoCapture(out_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 6)
        ok, fr = cap.read()
        cap.release()
        if ok:
            out_faces = get_all_faces(fr) or []
            for i in range(3):
                lbl = LABELS[i]
                if not out_faces:
                    continue
                of = _nearest(out_faces, det_boxes[i])
                identity[lbl] = {
                    "to_source": float(compute_cosine_distance(of.embedding, src_emb)),
                    "to_original": float(compute_cosine_distance(of.embedding, orig_emb[lbl])),
                }
    return {"provider": provider, "selection_person": selection_person,
            "routed": routed, "identity": identity,
            "route_log": route_lines[-1] if route_lines else ""}


# -- the routing-only decision (process_frame: preview vs render, per provider) -

def route_decision(provider, selection_person, is_preview, tracking=False):
    """Just the (box -> source) routing for one frame -- cheap, for assertions
    6 (preview vs render) and 7 (provider vs provider)."""
    import roop.globals as g
    from roop.face_util import get_all_faces
    from roop.ProcessMgr import ProcessMgr
    from roop import ProcessMgr as _pm

    g_, ab, tfv = _setup(provider, tracking)
    canvas = build_canvas()
    faces = sorted(get_all_faces(canvas) or [], key=lambda f: float(f.bbox[0]))
    det_boxes = [[float(v) for v in f.bbox] for f in faces[:3]]
    source = tfv.load_library_faceset(SOURCE_FACESET)

    options = ab.build_options(g_, _swap_model(), None, False)
    options.selection_state = _selection(selection_person)
    g.INPUT_FACESETS = [source]
    g.TARGET_FACES = [faces[0], faces[1]]
    g.TARGET_FACE_GROUP = [0, 1]

    _pm._SWAP_LOG = {}
    mgr = ProcessMgr(None)
    mgr.is_preview = bool(is_preview)
    mgr.initialize(g.INPUT_FACESETS, g.TARGET_FACES, options)
    _prev = getattr(g, "processing", False)
    g.processing = True
    try:
        with redirect_stdout(io.StringIO()):
            mgr.process_frame(canvas.copy(), frame_idx=0)
    finally:
        g.processing = _prev
    routed = {}
    for _f, entries in (_pm._SWAP_LOG or {}).items():
        for box, src in entries:
            routed[_label_of(box, det_boxes)] = int(src)
    _pm._SWAP_LOG = None
    for p in list(getattr(mgr, "processors", []) or []):
        try:
            if hasattr(p, "Release"):
                p.Release()
        except Exception:
            pass
    return routed


# -- single-provider assertions (1-6, 8) ---------------------------------------

def check_single_provider(provider, out_dir):
    a = render_identity(provider, 0, out_dir)
    b = render_identity(provider, 1, out_dir)
    none = render_identity(provider, None, out_dir)
    prev_routed = route_decision(provider, 0, is_preview=True)
    rend_routed = route_decision(provider, 0, is_preview=False)

    lines, ok = [], [True]

    def _ok(cond, msg):
        ok[0] = ok[0] and bool(cond)
        lines.append(("  PASS " if cond else "  FAIL ") + msg)

    ai, bi, ni = a["identity"], b["identity"], none["identity"]
    # 1 person A receives the source
    _ok(a["routed"].get("A") == 0, f"[1] A routed to source 0 ({a['routed']})")
    _ok(ai.get("A", {}).get("to_source", 9) < SAME
        and ai.get("A", {}).get("to_original", 0) > DIFF,
        f"[1] A's output identity is the source, not person_b ({ai.get('A')})")
    # 2 person B unchanged
    _ok("B" not in a["routed"], f"[2] B not routed when A selected ({a['routed']})")
    _ok(ai.get("B", {}).get("to_original", 9) < SAME,
        f"[2] B's output identity unchanged ({ai.get('B')})")
    # 3 no third face swapped
    _ok(len(a["routed"]) == 1, f"[3] exactly one face swapped ({a['routed']})")
    _ok(ai.get("bystander", {}).get("to_original", 9) < SAME,
        f"[3] bystander identity unchanged ({ai.get('bystander')})")
    # 4 changing the selected person changes eligibility
    _ok(b["routed"].get("B") == 0 and "A" not in b["routed"],
        f"[4] selecting B routes B not A ({b['routed']})")
    _ok(bi.get("B", {}).get("to_source", 9) < SAME
        and bi.get("A", {}).get("to_original", 9) < SAME,
        f"[4] B became the source, A kept person_b (A={bi.get('A')}, B={bi.get('B')})")
    # 5 removing the selection -> zero swaps
    _ok(none["routed"] == {}, f"[5] no selection -> nothing routed ({none['routed']})")
    _ok(bool(ni) and all(v.get("to_original", 9) < SAME for v in ni.values()),
        f"[5] no selection -> every identity unchanged ({ni})")
    # 6 preview and render choose the same target
    _ok(prev_routed == rend_routed and rend_routed.get("A") == 0,
        f"[6] preview routing == render routing (preview={prev_routed} render={rend_routed})")
    # 8 distance logged for every candidate
    log = a["route_log"]
    _ok("[SelectedRoute]" in log and all(f"face={i}" in log for i in range(3)),
        f"[8] distance logged for all 3 candidates: {log[:150]}")

    return ok[0], lines, {"A": a, "B": b, "none": none,
                          "preview_routed": prev_routed, "render_routed": rend_routed}


# -- assertion 7: same routing across providers (subprocess per provider) -------

def _child_route(provider):
    return {
        "A_render": route_decision(provider, 0, is_preview=False),
        "A_preview": route_decision(provider, 0, is_preview=True),
        "B_render": route_decision(provider, 1, is_preview=False),
        "none_render": route_decision(provider, None, is_preview=False),
    }


def _run_child(provider, out_dir):
    path = os.path.join(out_dir, f"route_{provider}.json")
    log = os.path.join(out_dir, f"route_{provider}.log")
    cmd = [sys.executable, os.path.abspath(__file__),
           "--route-provider", provider, "--emit", path, "--out", out_dir]
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        rc = subprocess.run(cmd, cwd=APP, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not os.path.isfile(path):
        tail = "\n".join(open(log, encoding="utf-8", errors="replace").read().splitlines()[-20:])
        return None, f"exit={rc}\n{tail}"
    return json.load(open(path, encoding="utf-8")), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", default=None,
                    help="comma list -> run assertion 7 across these providers")
    ap.add_argument("--route-provider", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--emit", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.route_provider and args.emit:
        routed = _child_route(args.route_provider)
        with open(args.emit, "w", encoding="utf-8") as fh:
            json.dump(routed, fh)
        print(f"[{args.route_provider}] {routed}")
        return 0

    if args.providers:
        providers = [p.strip() for p in args.providers.split(",") if p.strip()]
        per, fails = {}, {}
        for p in providers:
            print(f"\n=== routing on {p} ===", flush=True)
            r, err = _run_child(p, args.out)
            if err:
                fails[p] = err
                print(f"  FAILED: {err}")
                continue
            per[p] = r
            print(f"  {r}")
        ok = not fails
        if len(per) >= 2:
            base = next(iter(per))
            for p in per:
                if p == base:
                    continue
                same = per[p] == per[base]
                ok = ok and same
                print(f"[7] {base} vs {p}: routing {'IDENTICAL' if same else 'DIFFERS'}")
        print(f"\nVERDICT: {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1

    from roop.backend_manager import canonical_provider_decision
    prov = "cuda" if "cuda" in canonical_provider_decision("cuda").active.lower() else "cpu"
    print(f"=== single provider: {prov} ===")
    ok, lines, _ = check_single_provider(prov, args.out)
    print("\n".join(lines))
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
