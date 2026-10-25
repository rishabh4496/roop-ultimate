"""End-to-end check of the OpenVINO seams on the two models the unit tests only reach through the runner:
SCRFD (Stage 1 detector, ``retinaface._build_one``) and XSeg (Stage 3 mask, ``Mask_XSeg.Initialize``).

    # app venv has no `openvino` on purpose; borrow the package from a scratch venv without touching the venv:
    PYTHONPATH=<dir holding a copy/junction of site-packages/openvino> \
    env/Scripts/python.exe tests/probe_openvino_seams_e2e.py off  out_off.npz
    ROOP_OPENVINO_NATIVE=CPU env/Scripts/python.exe tests/probe_openvino_seams_e2e.py cpu out_cpu.npz
    env/Scripts/python.exe tests/probe_openvino_seams_e2e.py compare out_off.npz out_cpu.npz

Each arm is its own process because the env var is read when a session is built. The `cpu` arm must show a live
``OpenVinoAsyncSession`` for BOTH models with a non-zero call count (a flag that is read but never routes is
consistent with a green result); `compare` fails otherwise. OpenVINO's CPU device stands in for an Intel NPU / iGPU:
it proves the seam routes, the I/O contract and the numbers, not Intel-hardware speed.
"""
import json
import os
import sys

import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
sys.path.insert(0, os.path.join(APP, "tests"))
FRAMES = int(os.environ.get("PROBE_FRAMES", "24"))
CLIP = os.environ.get("PROBE_CLIP", os.path.join(os.path.dirname(APP), "..", "..", "roop-keep", "Love.mp4"))


def _frames():
    import cv2
    cap = cv2.VideoCapture(os.path.abspath(CLIP))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for i in np.linspace(0, max(0, total - 1), FRAMES).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            out.append(f)
    return out


def run_arm(name, path):
    import angle_bench as ab
    from settings import Settings
    cfg = Settings(os.path.join(APP, "config.yaml"))
    ab.init_pipeline(str(cfg.provider), str(cfg.swap_model), "None", "None", sync_config=True)
    from roop import retinaface, openvino_runner
    from roop.processors.Mask_XSeg import Mask_XSeg
    frames = _frames()
    boxes, masks = [], []
    xs = Mask_XSeg()
    xs.Initialize({"devicename": "cuda"})
    det_sess = retinaface.get_detector("10g").session
    for f in frames:
        b, k = retinaface._detect_single_instance(f, det_size=640, det_thresh=0.5, model_type="10g")
        boxes.append(np.asarray(b, dtype=np.float32).reshape(-1, 5)[:, :5] if b is not None else np.zeros((0, 5), np.float32))
        h, w = f.shape[:2]
        side = min(h, w)
        crop = f[(h - side) // 2:(h + side) // 2, (w - side) // 2:(w + side) // 2]
        masks.append(np.asarray(xs.Run(crop, ""), dtype=np.float32))
    rep = openvino_runner.report()
    info = {
        "arm": name,
        "scrfd_session": type(det_sess).__name__,
        "xseg_session": type(xs.model_xseg).__name__,
        "frames": len(frames),
        "faces": int(sum(len(b) for b in boxes)),
        "runner_sessions": rep.get("sessions", []),
        "configured": rep.get("configured"),
    }
    np.savez(path, info=json.dumps(info), n=len(boxes),
             **{f"b{i}": b for i, b in enumerate(boxes)}, masks=np.stack(masks))
    print(json.dumps({k: v for k, v in info.items() if k != "runner_sessions"}))
    for s in info["runner_sessions"]:
        print("  runner:", {k: s.get(k) for k in ("model_key", "device", "precision", "calls", "streams", "queue_requests", "runner")})


def compare(a, b):
    A, B = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
    ia, ib = json.loads(str(A["info"])), json.loads(str(B["info"]))
    bad = []
    if ia["scrfd_session"] == "OpenVinoAsyncSession" or ia["xseg_session"] == "OpenVinoAsyncSession":
        bad.append("baseline arm already ran OpenVINO")
    for key in ("scrfd_session", "xseg_session"):
        if ib[key] != "OpenVinoAsyncSession":
            bad.append(f"{key} in the OpenVINO arm is {ib[key]}, not OpenVinoAsyncSession (seam did not route)")
    calls = [int(s.get("calls") or 0) for s in ib["runner_sessions"]]
    if not calls or min(calls) <= 0:
        bad.append(f"runner call counts {calls}: a session was built but never executed")
    worst_box, miss = 0.0, 0
    for i in range(int(A["n"])):
        ba, bb = A[f"b{i}"], B[f"b{i}"]
        if len(ba) != len(bb):
            miss += 1
            continue
        if len(ba):
            ba, bb = ba[np.argsort(ba[:, 0])], bb[np.argsort(bb[:, 0])]
            worst_box = max(worst_box, float(np.abs(ba[:, :4] - bb[:, :4]).max()))
    d = np.abs(A["masks"] - B["masks"])
    inter = ((A["masks"] > 0.5) & (B["masks"] > 0.5)).sum()
    union = ((A["masks"] > 0.5) | (B["masks"] > 0.5)).sum()
    res = {"frames": int(A["n"]), "faces_off": ia["faces"], "faces_ov": ib["faces"], "frames_with_different_face_count": miss,
           "max_bbox_diff_px": round(worst_box, 4), "xseg_mean_abs": round(float(d.mean()), 6),
           "xseg_max_abs": round(float(d.max()), 4), "xseg_iou": round(float(inter / max(1, union)), 5),
           "runner_calls": calls, "problems": bad}
    print(json.dumps(res, indent=1))
    return 1 if bad or miss else 0


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    run_arm(sys.argv[1], sys.argv[2])
