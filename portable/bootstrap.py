"""Portable runtime bootstrap for Roop Ultimate (stdlib only).

run.bat / run.sh get uv and a uv-managed CPython 3.10 venv under
portable/runtime/, then hand over to this script, which runs inside that venv:

  1. installs the Python dependencies -- the same steps as install.js, in the
     same order, with app/provision_runtime.py choosing the GPU wheels (PyTorch
     CUDA 12.8, ONNX Runtime GPU, TensorRT 10.9) for the hardware it finds;
  2. provides static FFmpeg binaries under portable/runtime/ffmpeg/bin;
  3. makes sure the React build and config.yaml exist;
  4. starts `app/run.py --ui react`, forwarding any arguments it does not own,
     and opens the browser as soon as the API answers -- early enough that the
     model splash screen shows the first-run downloads.

A dependency stamp (portable/runtime/deps.json) makes step 1 a no-op until
requirements.txt, provision_runtime.py or this file changes.

OFFLINE BUNDLES.  `--build-bundle` downloads every wheel provisioning would
install on this machine into portable/wheels/, plus the uv and FFmpeg archives
into portable/vendor/.  Together with portable/runtime/python (the uv-managed
interpreter, which is relocatable -- unlike a venv) the folder then installs
with no network at all; the venv is always rebuilt on the target machine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import webbrowser
import zipfile

BOOTSTRAP_VERSION = 1
PORTABLE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PORTABLE)
APP = os.path.join(ROOT, "app")
UI = os.path.join(ROOT, "react-ui")
RT = os.path.join(PORTABLE, "runtime")
WHEELS = os.path.join(PORTABLE, "wheels")
VENDOR = os.path.join(PORTABLE, "vendor")
FFMPEG_DIR = os.path.join(RT, "ffmpeg", "bin")
STAMP = os.path.join(RT, "deps.json")
IS_WIN = os.name == "nt"
EXE = ".exe" if IS_WIN else ""

# Pinned to the FFmpeg the app is validated against (8.1.x). Each list is tried
# in order; ROOP_FFMPEG_URL overrides. The Windows and macOS builds are
# immutable release tags; BtbN's Linux build tracks the 8.1 branch.
FFMPEG_URLS = {
    "Windows": ["https://github.com/GyanD/codexffmpeg/releases/download/8.1.2/"
                "ffmpeg-8.1.2-essentials_build.zip"],
    "Linux": ["https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
              "ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz"],
    "Darwin": ["https://evermeet.cx/ffmpeg/ffmpeg-8.1.2.zip",
               "https://evermeet.cx/ffmpeg/ffprobe-8.1.2.zip"],
}

# install.js installs these after torch, with --no-deps, so the torch / numpy
# / OpenCV already in the env are never touched.
SAM2_PACKAGES = ["sam2", "hydra-core", "omegaconf", "iopath", "portalocker",
                 "antlr4-python3-runtime==4.9.3"]
NUMPY_PIN = "numpy==1.26.4"


def log(message: str) -> None:
    print("[portable] " + message, flush=True)


def uv_exe() -> str:
    found = os.environ.get("ROOP_PORTABLE_UV")
    if found and os.path.isfile(found):
        return found
    local = os.path.join(RT, "uv", "uv" + EXE)
    if os.path.isfile(local):
        return local
    found = shutil.which("uv")
    if found:
        return found
    raise SystemExit("[portable] uv not found; start through run.bat / run.sh")


def run(command, *, cwd=None, env=None, check=True) -> int:
    log("$ " + " ".join(str(c) for c in command))
    code = subprocess.call(command, cwd=cwd, env=env)
    if check and code != 0:
        raise SystemExit("[portable] command failed (exit %d): %s" % (code, command[0]))
    return code


def child_env(extra: dict | None = None) -> dict:
    env = dict(os.environ)
    paths = [FFMPEG_DIR, os.path.dirname(uv_exe()), os.path.dirname(sys.executable)]
    env["PATH"] = os.pathsep.join(paths + [env.get("PATH", "")])
    env["VIRTUAL_ENV"] = os.path.dirname(os.path.dirname(sys.executable))
    env.update(extra or {})
    return env


# ── hardware ────────────────────────────────────────────────────────────────

def _provision_module():
    sys.path.insert(0, APP)
    try:
        import provision_runtime  # noqa: WPS433 -- app/ is not a package
        return provision_runtime
    finally:
        sys.path.remove(APP)


def hardware_vendor() -> str:
    try:
        return _provision_module().detect_hardware().vendor
    except Exception as exc:  # detection is advisory for the stamp only
        log("hardware detection failed (%s); assuming cpu" % exc)
        return "cpu"


# ── Python dependencies ─────────────────────────────────────────────────────

def _digest(*paths: str) -> str:
    h = hashlib.sha256(str(BOOTSTRAP_VERSION).encode())
    for path in paths:
        with open(path, "rb") as handle:
            h.update(handle.read())
    return h.hexdigest()


def bundle_info() -> dict | None:
    try:
        with open(os.path.join(WHEELS, "bundle.json"), "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def install_dependencies(*, offline: bool, force: bool) -> None:
    vendor = hardware_vendor()
    wheelhouse = WHEELS if os.path.isdir(WHEELS) and os.listdir(WHEELS) else None
    key = {
        "inputs": _digest(os.path.join(APP, "requirements.txt"),
                          os.path.join(APP, "provision_runtime.py"),
                          os.path.abspath(__file__)),
        "vendor": vendor,
        "python": platform.python_version(),
    }
    try:
        with open(STAMP, "r", encoding="utf-8") as handle:
            if json.load(handle) == key and not force:
                log("dependencies up to date (%s runtime)" % vendor)
                return
    except (OSError, ValueError):
        pass
    if offline and not wheelhouse:
        raise SystemExit("[portable] --offline needs portable/wheels (run --build-bundle "
                         "on a connected machine first)")

    log("installing dependencies for a %s machine%s" %
        (vendor, " from the offline wheelhouse" if offline else ""))
    extra = {}
    if wheelhouse:
        extra["ROOP_WHEELHOUSE"] = wheelhouse
    if offline:
        extra["ROOP_OFFLINE"] = "1"
    env = child_env(extra)
    source = (["--no-index"] if offline else []) + (["--find-links", wheelhouse] if wheelhouse else [])
    uv_pip = [uv_exe(), "pip", "install", "--python", sys.executable]

    # The same four steps as install.js, in the same order.
    run(uv_pip + ["-r", os.path.join(APP, "requirements.txt")] + source, env=env)
    run([sys.executable, os.path.join(APP, "provision_runtime.py")], cwd=APP, env=env)
    run(uv_pip + [NUMPY_PIN] + source, env=env)
    run(uv_pip + ["--no-deps"] + SAM2_PACKAGES + source, env=env)
    # The manifest goes under runtime/, not app/.runtime-verification.json:
    # that one belongs to the Pinokio install (install.js / update.js).
    if run([sys.executable, "verify_ort.py", "--manifest-out",
            os.path.join(RT, "runtime-verification.json")],
           cwd=APP, env=env, check=False):
        log("WARNING: verify_ort.py reported a problem with the ONNX Runtime providers "
            "(see above); the app will fall back to what does load")

    os.makedirs(RT, exist_ok=True)
    with open(STAMP, "w", encoding="utf-8") as handle:
        json.dump(key, handle, indent=1)


# ── FFmpeg ──────────────────────────────────────────────────────────────────

def _ffmpeg_urls() -> list:
    override = os.environ.get("ROOP_FFMPEG_URL")
    return [override] if override else FFMPEG_URLS.get(platform.system(), [])


def _vendored(url: str) -> str:
    return os.path.join(VENDOR, url.rstrip("/").rsplit("/", 1)[-1])


def _download(url: str, dest: str) -> None:
    log("downloading " + url)
    tmp = dest + ".part"
    request = urllib.request.Request(url, headers={"User-Agent": "roop-ultimate-portable"})
    with urllib.request.urlopen(request, timeout=60) as response, open(tmp, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done, last = 0, 0.0
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)
            done += len(block)
            if time.monotonic() - last > 2 and total:
                last = time.monotonic()
                log("  %.0f / %.0f MB" % (done / 2 ** 20, total / 2 ** 20))
    os.replace(tmp, dest)


def _extract_binaries(archive: str, names: tuple) -> list:
    """Copy every file called ffmpeg/ffprobe (any depth) into FFMPEG_DIR."""
    os.makedirs(FFMPEG_DIR, exist_ok=True)
    found = []
    if archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                base = member.rsplit("/", 1)[-1]
                if base in names:
                    target = os.path.join(FFMPEG_DIR, base)
                    with zf.open(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    found.append(target)
    else:
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                base = member.name.rsplit("/", 1)[-1]
                if member.isfile() and base in names:
                    target = os.path.join(FFMPEG_DIR, base)
                    with tf.extractfile(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    found.append(target)
    for path in found:
        if not IS_WIN:
            os.chmod(path, 0o755)
    return found


def ensure_ffmpeg(*, offline: bool) -> str:
    binary = os.path.join(FFMPEG_DIR, "ffmpeg" + EXE)
    if os.path.isfile(binary):
        return binary
    names = ("ffmpeg" + EXE, "ffprobe" + EXE)
    for url in _ffmpeg_urls():
        archive = _vendored(url)
        if not os.path.isfile(archive):
            if offline:
                raise SystemExit("[portable] --offline: %s is not in portable/vendor" %
                                 os.path.basename(archive))
            os.makedirs(VENDOR, exist_ok=True)
            _download(url, archive)
        _extract_binaries(archive, names)
    if not os.path.isfile(binary):
        raise SystemExit("[portable] no ffmpeg binary found in the FFmpeg archive(s)")
    out = subprocess.run([binary, "-hide_banner", "-encoders"], capture_output=True, text=True)
    nvenc = sorted({w for w in out.stdout.split() if w.endswith("_nvenc")})
    log("ffmpeg ready at %s (NVENC encoders: %s)" % (binary, ", ".join(nvenc) or "none"))
    return binary


# ── UI build and config ─────────────────────────────────────────────────────

def ensure_ui() -> None:
    if os.path.isfile(os.path.join(UI, "dist", "index.html")):
        return
    npm = shutil.which("npm")
    if not npm:
        raise SystemExit(
            "[portable] react-ui/dist is missing and npm is not available to build it. "
            "Portable bundles ship the built UI: run --build-bundle on a machine with "
            "Node.js, or install Node.js 20+ and run again.")
    run([npm, "ci", "--no-audit", "--no-fund"], cwd=UI)
    run([npm, "run", "build"], cwd=UI)


def ensure_config() -> None:
    config = os.path.join(APP, "config.yaml")
    default = os.path.join(APP, "default_config.yaml")
    if not os.path.exists(config) and os.path.exists(default):
        shutil.copyfile(default, config)
        log("seeded app/config.yaml from default_config.yaml")


# ── bundle ──────────────────────────────────────────────────────────────────

def _uv_archive_name() -> str:
    machine = platform.machine().lower()
    arch = "aarch64" if machine in ("arm64", "aarch64") else "x86_64"
    system = platform.system()
    if system == "Windows":
        return "uv-%s-pc-windows-msvc.zip" % arch
    if system == "Darwin":
        return "uv-%s-apple-darwin.tar.gz" % arch
    return "uv-%s-unknown-linux-gnu.tar.gz" % arch


def build_bundle() -> None:
    """Fill portable/wheels and portable/vendor for an offline install."""
    os.makedirs(WHEELS, exist_ok=True)
    os.makedirs(VENDOR, exist_ok=True)
    uv_version = "0.8.22"
    uv_archive = os.path.join(VENDOR, _uv_archive_name())
    if not os.path.isfile(uv_archive):
        _download("https://github.com/astral-sh/uv/releases/download/%s/%s"
                  % (uv_version, os.path.basename(uv_archive)), uv_archive)
    for url in _ffmpeg_urls():
        if not os.path.isfile(_vendored(url)):
            _download(url, _vendored(url))

    # Ask provisioning which wheels THIS machine needs, without installing.
    record = os.path.join(RT, "provision-record.json")
    if os.path.exists(record):
        os.remove(record)
    run([sys.executable, os.path.join(APP, "provision_runtime.py")], cwd=APP,
        env=child_env({"ROOP_PROVISION_RECORD": record}))
    with open(record, "r", encoding="utf-8") as handle:
        groups = json.load(handle)

    env = child_env()
    run([uv_exe(), "pip", "install", "--python", sys.executable, "pip"], env=env)
    pip_wheel = [sys.executable, "-m", "pip", "wheel", "--wheel-dir", WHEELS]
    for group in groups:
        command = list(pip_wheel)
        if group.get("index_url"):
            command += ["--index-url", group["index_url"],
                        "--extra-index-url", "https://pypi.org/simple"]
        if group.get("extra_index_url"):
            command += ["--extra-index-url", group["extra_index_url"]]
        if group.get("no_deps"):
            command.append("--no-deps")
        command += [p for p in group["packages"] if "://" not in p]
        urls = [p for p in group["packages"] if "://" in p]
        run(command + urls, env=env)
    # requirements.txt LAST, constrained to the provisioning pins and able to
    # see the wheels already fetched: timm depends on torch, and unconstrained
    # it would add PyPI's newest (CPU on Windows) torch to the bundle. With
    # torch==2.7.0 pinned, the local 2.7.0+cu128 wheel outranks PyPI's 2.7.0.
    constraints = os.path.join(RT, "bundle-constraints.txt")
    with open(constraints, "w", encoding="utf-8") as handle:
        for group in groups:
            handle.writelines(p + "\n" for p in group["packages"] if "==" in p and "://" not in p)
    run(pip_wheel + ["--find-links", WHEELS, "-c", constraints,
                     "-r", os.path.join(APP, "requirements.txt"), NUMPY_PIN], env=env)
    run(pip_wheel + ["--no-deps"] + SAM2_PACKAGES, env=env)

    ensure_ui()
    info = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "system": platform.system(), "machine": platform.machine(),
        "python": platform.python_version(), "vendor": hardware_vendor(),
        "wheels": len([f for f in os.listdir(WHEELS) if f.endswith(".whl")]),
    }
    with open(os.path.join(WHEELS, "bundle.json"), "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=1)
    log("bundle ready: %(wheels)d wheels for %(system)s/%(vendor)s" % info)
    log("ship the repository folder with portable/wheels, portable/vendor and "
        "portable/runtime/python (NOT portable/runtime/venv or uv-cache)")


# ── launch ──────────────────────────────────────────────────────────────────

def _free_port(start: int = 8001) -> int:
    for port in range(start, start + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise SystemExit("[portable] no free port in %d-%d" % (start, start + 199))


def _open_when_ready(url: str, child: subprocess.Popen) -> None:
    probe = url + "/api/models/integrity"
    while child.poll() is None:
        try:
            with urllib.request.urlopen(probe, timeout=2) as response:
                if response.status == 200:
                    log("opening " + url)
                    webbrowser.open(url)
                    return
        except OSError:
            pass
        time.sleep(0.5)


def launch(forward: list, *, open_browser: bool) -> int:
    port = int(os.environ.get("ROOP_API_PORT") or _free_port())
    env = child_env({
        "ROOP_API_PORT": str(port),
        "ROOP_GRADIO_PORT": str(port + 1),
        "ROOP_REACT_CLIENT": "1",
        "NO_ALBUMENTATIONS_UPDATE": "1",
        "ROOP_TEMPORAL_STEP": "1",
        "PYTHONUNBUFFERED": "1",
    })
    command = [sys.executable, "run.py", "--ui", "react"] + forward
    log("$ " + " ".join(command))
    child = subprocess.Popen(command, cwd=APP, env=env)
    headless = any(a in forward for a in ("--benchmark", "--diagnose-runtime"))
    if open_browser and not headless:
        threading.Thread(target=_open_when_ready,
                         args=("http://127.0.0.1:%d" % port, child), daemon=True).start()
    try:
        return child.wait()
    except KeyboardInterrupt:
        child.terminate()
        return child.wait()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="run", description="Roop Ultimate portable launcher. Unrecognised "
        "arguments are passed to app/run.py (e.g. --benchmark --benchmark-mode regression).")
    parser.add_argument("--setup-only", action="store_true", help="install/verify, then exit")
    parser.add_argument("--reinstall", action="store_true", help="reinstall dependencies")
    parser.add_argument("--offline", action="store_true",
                        help="never touch the network (default when portable/wheels is a bundle)")
    parser.add_argument("--online", action="store_true", help="ignore a bundle's offline default")
    parser.add_argument("--build-bundle", action="store_true",
                        help="download wheels, uv and FFmpeg for an offline copy of this folder")
    parser.add_argument("--no-browser", action="store_true")
    args, forward = parser.parse_known_args(argv)
    if forward[:1] == ["--"]:
        forward = forward[1:]

    if sys.version_info[:2] != (3, 10):
        log("warning: expected CPython 3.10, running %s" % platform.python_version())
    if args.build_bundle:
        build_bundle()
        return 0
    offline = args.offline or (bundle_info() is not None and not args.online)
    install_dependencies(offline=offline, force=args.reinstall)
    ensure_ffmpeg(offline=offline)
    ensure_ui()
    ensure_config()
    if args.setup_only:
        log("setup complete")
        return 0
    return launch(forward, open_browser=not args.no_browser)


if __name__ == "__main__":
    sys.exit(main())
