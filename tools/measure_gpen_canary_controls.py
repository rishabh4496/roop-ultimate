import glob, os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); APP = os.path.join(REPO, "app")
for p in (APP, os.path.join(APP, "tests")): sys.path.insert(0, p)
os.chdir(APP)
import cv2, numpy as np
import trt_precision_fidelity as fid
fid._init_pipeline()
import onnxruntime
import roop.globals as g
from roop import aux_canary as ac, precision_policy as pp
from roop.utilities import get_onnx_session_options

def cos(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

for key, rel in (("gpen_512", "GPEN-BFR-512.onnx"), ("gpen_256", "gpen_bfr_256.onnx")):
    path = os.path.join(APP, "models", rel)
    prov = pp.providers_for(key, g.execution_providers, path)[0]
    live = onnxruntime.InferenceSession(path, get_onnx_session_options(), providers=prov)
    ref = onnxruntime.InferenceSession(path, get_onnx_session_options(), providers=["CPUExecutionProvider"])
    name = live.get_inputs()[0].name; size = live.get_inputs()[0].shape[2]; out = [live.get_outputs()[0].name]
    ims = []
    for kind, seed in (("skin+n0.04", 1), ("skin+n0.08", 2)):
        ims.append(ac._synth(kind, seed, size, size))
    for f in sorted(glob.glob(os.path.join(APP, "facesets", "*.png")))[:6]:
        im = cv2.resize(cv2.imread(f), (size, size), interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float32) / 255.0
        ims.append(im.transpose(2, 0, 1))
    xs = [np.ascontiguousarray((c * 2 - 1)[None], dtype=np.float32) for c in ims]
    lo = [live.run(out, {name: x})[0] for x in xs]
    ro = [ref.run(out, {name: x})[0] for x in xs]
    n = len(ro)
    healthy = [cos(a, b) for a, b in zip(lo, ro)]
    other = [cos(ro[(i + 3) % n], ro[i]) for i in range(n)]
    smear = [cos(cv2.GaussianBlur(r[0].transpose(1, 2, 0), (0, 0), 5).transpose(2, 0, 1)[None], r) for r in ro]
    grey = [cos(np.zeros_like(r), r) for r in ro]
    shift = [cos(np.roll(r, 24, axis=3), r) for r in ro]
    print("[%s] healthy min %.4f | other-face max %.4f min %.4f | smear5 max %.4f | grey max %.4f | shift24 max %.4f" % (
        key, min(healthy), max(other), min(other), max(smear), max(grey), max(shift)), flush=True)
