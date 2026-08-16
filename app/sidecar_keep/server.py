"""KEEP sidecar HTTP server — runs INSIDE sidecar_keep/.venv, never the main env.

Protocol (localhost only, port from argv[1] or KEEP_SIDECAR_PORT):
  GET  /health           → 200 {"ok": true, "model": "KEEP", "device": "..."}
  POST /enhance          → body: PNG/JPEG bytes of an aligned 512x512 BGR face
                            crop; response: PNG bytes of the enhanced crop.
                           500 + text on failure (the client passes through).

Model loading follows KEEP's published inference code: the cloned repo is put
on sys.path, the `KEEP` architecture comes from basicsr's ARCH_REGISTRY (the
repo registers it on import), and weights load from
sidecar_keep/weights/KEEP-b76feb75.pth (the official v1.0.0 release asset).
NOTE: validated plumbing; the model path itself is experimental — see README.
"""

import io
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "KEEP")
WEIGHT = os.path.join(HERE, "weights", "KEEP-b76feb75.pth")

_model = None
_device = "cpu"


def _load_model():
    global _model, _device
    if _model is not None:
        return _model
    import torch
    sys.path.insert(0, REPO)
    # Importing the repo's archs package registers 'KEEP' in ARCH_REGISTRY.
    from basicsr.utils.registry import ARCH_REGISTRY
    try:
        import basicsr.archs  # noqa: F401  (repo overrides/extends this)
    except Exception:
        pass
    try:
        import archs  # noqa: F401  (some KEEP revisions keep archs at repo root)
    except Exception:
        pass
    net_cls = ARCH_REGISTRY.get("KEEP")
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    # Constructor args mirror KEEP's inference script defaults.
    model = net_cls(
        img_size=512, emb_dim=256, dim_embd=512, n_head=8, n_layers=9,
        codebook_size=1024, cft_list=["16", "32", "64"], kalman_attn_head_dim=48,
        num_uncertainty_layers=3, cfa_list=["16", "32"], cfa_nhead=4, cfa_dim=256,
        cond=1,
    ).to(_device)
    ckpt = torch.load(WEIGHT, map_location="cpu")
    state = ckpt.get("params_ema") or ckpt.get("params") or ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    _model = model
    return model


def _enhance(png_bytes: bytes) -> bytes:
    import cv2
    import numpy as np
    import torch

    model = _load_model()
    arr = np.frombuffer(png_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode input image")
    h, w = bgr.shape[:2]
    if (h, w) != (512, 512):
        bgr = cv2.resize(bgr, (512, 512), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    x = (x - 0.5) / 0.5
    # KEEP is temporal: (b, t, c, h, w). Feed t=1 (see README caveat).
    x = x.unsqueeze(1).to(_device)
    with torch.no_grad():
        out = model(x)
    if isinstance(out, (list, tuple)):
        out = out[0]
    out = out.squeeze(0)
    if out.dim() == 4:   # (t, c, h, w)
        out = out[0]
    out = (out.clamp(-1, 1) + 1) / 2
    img = (out.permute(1, 2, 0).cpu().numpy() * 255.0).astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if (h, w) != (512, 512):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("could not encode result")
    return buf.tobytes()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body: bytes, ctype="application/octet-stream"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send(200, ('{"ok": true, "model": "KEEP", "device": "%s"}' % _device).encode(),
                       "application/json")
        else:
            self._send(404, b"not found")

    def do_POST(self):
        if not self.path.startswith("/enhance"):
            self._send(404, b"not found")
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(n)
            self._send(200, _enhance(data), "image/png")
        except Exception:
            tb = traceback.format_exc()
            sys.stderr.write(tb)
            self._send(500, tb.encode("utf-8", "replace"), "text/plain")


def main():
    port = int(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("KEEP_SIDECAR_PORT", "8021"))
    # Warm the model before accepting traffic so /health means "ready".
    try:
        _load_model()
        print(f"[KEEP sidecar] model ready on {_device}", flush=True)
    except Exception:
        traceback.print_exc()
        print("[KEEP sidecar] model failed to load — /enhance will 500", flush=True)
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"[KEEP sidecar] listening on 127.0.0.1:{port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
