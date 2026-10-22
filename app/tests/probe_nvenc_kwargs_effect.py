"""Does each PyNvVideoCodec kwarg CHANGE the bitstream? Acceptance is meaningless (unknown kwargs are ignored).

    app\\env\\Scripts\\python.exe app\\tests\\probe_nvenc_kwargs_effect.py

Measured 2026-10-22 (PyNvVideoCodec 2.2.3, RTX 4070, 90 frames of Weeds.mp4, 4 Mbps VBR HEVC): ``multipass="fullres"``,
``aq=1`` and ``temporalaq=1`` change the bitstream; ``FULL_RESOLUTION``, ``multipass=2``, ``spatial_aq``, ``enableAQ``,
``temporal_aq``, ``enableTemporalAQ`` and a made-up kwarg produce a stream BYTE-IDENTICAL to the preset-only one.
``roop/nvenc_quality.py`` and ``test_nvenc_quality.py`` pin the consequence.
"""
import hashlib
import os
import sys
import time

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import PyNvVideoCodec as nvc    # noqa: E402
import fixtures                 # noqa: E402
from roop import video_io_hw as hw  # noqa: E402

CLIP = fixtures.clip("Weeds.mp4", required=True)
N = 90
cap = cv2.VideoCapture(CLIP)
frames = []
while len(frames) < N:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
H, W = frames[0].shape[:2]
W -= W % 2
H -= H % 2
frames = [f[:H, :W] for f in frames]
print(f"{len(frames)} frames {W}x{H} from Weeds.mp4")
matrix = hw.default_matrix(W, H)
nv12 = []
for f in frames:
    t = torch.from_numpy(f).cuda().flip(-1).permute(2, 0, 1).contiguous()
    nv12.append(hw.rgb_to_nv12(t, matrix))
torch.cuda.synchronize()

base = dict(codec="hevc", gpu_id=0, fps=30, gop=30, preset="P4", rc="vbr", bitrate=4_000_000, maxbitrate=8_000_000)
trials = {
    "baseline P4": {},
    "preset=P6": {"preset": "P6"},
    "P6 multipass=fullres": {"preset": "P6", "multipass": "fullres"},
    "P6 multipass=FULL_RESOLUTION": {"preset": "P6", "multipass": "FULL_RESOLUTION"},
    "P6 multipass=2 (int)": {"preset": "P6", "multipass": 2},
    "P6 multipass=bogus": {"preset": "P6", "multipass": "bogus"},
    "P6 aq=1": {"preset": "P6", "aq": 1},
    "P6 spatial_aq=1": {"preset": "P6", "spatial_aq": 1},
    "P6 enableAQ=1": {"preset": "P6", "enableAQ": 1},
    "P6 totallybogus=1": {"preset": "P6", "totallybogus": 1},
    "P6 temporalaq=1": {"preset": "P6", "temporalaq": 1},
    "P6 temporal_aq=1": {"preset": "P6", "temporal_aq": 1},
    "P6 enableTemporalAQ=1": {"preset": "P6", "enableTemporalAQ": 1},
    "P6 tuning_info=high_quality": {"preset": "P6", "tuning_info": "high_quality"},
}
ref = {}
for name, extra in trials.items():
    try:
        enc = nvc.CreateEncoder(W, H, "NV12", False, **{**base, **extra})
    except Exception as exc:
        print(f"{name:34s} REJECTED {exc}")
        continue
    size, h = 0, hashlib.sha1()
    t0 = time.perf_counter()
    for i, buf in enumerate(nv12):
        p = nvc.NV_ENC_PIC_PARAMS()
        p.inputTimeStamp = i
        for pk in enc.Encode(buf, p):
            b = bytes(pk["data"]); size += len(b); h.update(b)
    for pk in enc.EndEncode():
        b = bytes(pk["data"]); size += len(b); h.update(b)
    dt = time.perf_counter() - t0
    digest = h.hexdigest()[:10]
    same = [k for k, v in ref.items() if v == digest]
    ref.setdefault(name, digest)
    print(f"{name:34s} {size:9d} B  {len(nv12)/dt:6.0f} fps  sha {digest}  "
          + (f"IDENTICAL to {same[0]}" if same else "distinct"))
    del enc
