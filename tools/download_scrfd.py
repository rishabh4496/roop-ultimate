"""Download and SHA256-verify SCRFD-10G (``scrfd_10g_bnkps.onnx``) into ``<repo>/.cache/models``.

    python tools/download_scrfd.py [--engine]

The source is InsightFace's ``buffalo_l/det_10g.onnx`` (Hugging Face mirror
``public-data/insightface``), pinned in ``face_engine/models/zoo.py`` by
size and SHA256; a file that does not verify is downloaded again (resumable,
retried). ``--engine`` also builds the dynamic-shape TensorRT FP16 engine
``GPUSCRFDDetector`` uses (``tools/compile_engines.py --models scrfd_10g_bnkps``).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--engine", action="store_true", help="also compile the TensorRT engine")
    args = ap.parse_args(argv)

    from face_engine.models.zoo import MODEL_ZOO, build_default_registry

    spec = MODEL_ZOO["scrfd_10g_bnkps"]
    registry = build_default_registry()
    path = registry.ensure("scrfd_10g_bnkps", show_progress=True)
    print(f"{path}  ({spec.size} bytes, sha256 {spec.sha256}) verified")
    if args.engine:
        return subprocess.run([sys.executable, str(REPO / "tools" / "compile_engines.py"),
                               "--models", "scrfd_10g_bnkps", "--precision", "fp16"],
                              cwd=REPO, check=False).returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
