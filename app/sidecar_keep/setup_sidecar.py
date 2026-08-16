"""One-shot installer for the KEEP sidecar.

Creates an ISOLATED virtual environment (so KEEP's basicsr-era dependency pins
can never conflict with the main app env), clones the official KEEP repo,
installs its requirements + a CUDA torch build, and downloads the released
checkpoint. Safe to re-run — every step is idempotent.

Run from the `app` folder with the app's own (trusted) interpreter:
    env\\Scripts\\python.exe sidecar_keep\\setup_sidecar.py
"""

import os
import shutil
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(HERE, ".venv")
REPO = os.path.join(HERE, "KEEP")
WEIGHTS = os.path.join(HERE, "weights")
REPO_URL = "https://github.com/jnjaby/KEEP.git"
WEIGHT_URLS = [
    # Official v1.0.0 release assets (verified 2026-07-19).
    "https://github.com/jnjaby/KEEP/releases/download/v1.0.0/KEEP-b76feb75.pth",
]
TORCH_INDEX = "https://download.pytorch.org/whl/cu121"


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def venv_python():
    if os.name == "nt":
        return os.path.join(VENV, "Scripts", "python.exe")
    return os.path.join(VENV, "bin", "python")


def main():
    # 1. Isolated venv (uv is faster and ships with Pinokio; stdlib fallback).
    if not os.path.exists(venv_python()):
        uv = shutil.which("uv")
        if uv:
            run([uv, "venv", VENV, "--python", sys.executable])
        else:
            run([sys.executable, "-m", "venv", VENV])
    py = venv_python()

    def pip(*args):
        uv = shutil.which("uv")
        if uv:
            run([uv, "pip", "install", "--python", py, *args])
        else:
            run([py, "-m", "pip", "install", *args])

    # 2. Clone KEEP (idempotent).
    if not os.path.isdir(os.path.join(REPO, ".git")):
        run(["git", "clone", "--depth", "1", REPO_URL, REPO])

    # 3. Torch (CUDA) first, then the sidecar requirement set, then KEEP's own.
    pip("torch", "torchvision", "--index-url", TORCH_INDEX)
    pip("-r", os.path.join(HERE, "requirements.txt"))
    keep_reqs = os.path.join(REPO, "requirements.txt")
    if os.path.exists(keep_reqs):
        pip("-r", keep_reqs)

    # 4. Checkpoint(s).
    os.makedirs(WEIGHTS, exist_ok=True)
    for url in WEIGHT_URLS:
        dest = os.path.join(WEIGHTS, os.path.basename(url))
        if os.path.exists(dest):
            print(f"weights: {os.path.basename(dest)} already present")
            continue
        print(f"downloading {url} ...", flush=True)
        part = dest + ".part"
        urllib.request.urlretrieve(url, part)
        os.replace(part, dest)

    # 5. Smoke check: can the sidecar env import its stack?
    run([py, "-c", "import torch, basicsr; print('sidecar env OK — torch', torch.__version__, 'cuda', torch.cuda.is_available())"])
    print("\nKEEP sidecar installed. Select the 'KEEP (sidecar)' enhancer in the app.")


if __name__ == "__main__":
    main()
