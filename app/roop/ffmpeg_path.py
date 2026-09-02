"""Resolve ffmpeg once, without depending on the caller's PATH.

WHY THIS EXISTS.  The render path invoked ffmpeg as the bare name `"ffmpeg"`.
Pinokio's own shell puts ffmpeg on PATH, so under the launcher everything works
and every prior validation passed.  Outside that shell -- `update_health.py`'s
launch probe, the updater's health worker, a benchmark child process, a user
starting the app from an ordinary terminal -- the lookup fails and the encoder
pre-flight aborts the run with:

    Video encoder 'hevc_nvenc' is not working, aborting.
    ffmpeg binary 'ffmpeg' was not found on PATH.
    Processing stopped: video encoder unavailable.

Observed on the physical RTX 3060 host: a 900-frame render "completed" in
seconds with `progress: 1.0` and `desc: 'Done'`, produced NO output file, and
both queued jobs were marked FAILED -- on a machine whose ffmpeg is installed,
works, and exposes NVENC.

This is the same defect class the project already fixed twice: the bare
`shutil.which('ffmpeg')` that made the hardware profile record a machine with
no NVDEC and no NVENC, and the bare `shutil.which('npm')` in the health worker.
Each was fixed with its own private copy of the search, which is how the render
path came to be the one place still using the bare name.  This module is the
single implementation; `HardwareProfiler._resolve_ffmpeg` delegates to it.

Resolution is RUNTIME ONLY.  No absolute path is ever written into a launcher
script or a config file -- the project guide requires `PINOKIO_HOME` to be
resolved, never assumed, and never baked in.
"""

import json
import os
import shutil

_CACHED = None


def _pinokio_home():
    """PINOKIO_HOME, in the order the project guide requires."""
    home = os.environ.get("PINOKIO_HOME")
    if home and os.path.isdir(home):
        return home
    try:
        config = os.path.join(os.path.expanduser("~"), ".pinokio", "config.json")
        with open(config, "r", encoding="utf-8") as handle:
            home = json.load(handle).get("home")
        if home and os.path.isdir(home):
            return home
    except Exception:
        pass
    # <PINOKIO_HOME>/api/<launcher>/app/roop/this_file.py -> up four levels.
    here = os.path.dirname(os.path.abspath(__file__))
    app_dir = os.path.dirname(here)
    return os.path.dirname(os.path.dirname(os.path.dirname(app_dir)))


def ffmpeg_binary(refresh=False):
    """Absolute path to ffmpeg, or the bare name if nothing else resolves.

    Falling back to `"ffmpeg"` keeps the previous behaviour for any environment
    this search does not know about, so the change can only ever add working
    cases -- it cannot take one away.
    """
    global _CACHED
    if _CACHED is not None and not refresh:
        return _CACHED

    found = shutil.which("ffmpeg")
    if not found:
        home = _pinokio_home()
        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        roots = (
            os.path.join(home, "bin", "miniforge", "Library", "bin"),
            os.path.join(home, "bin", "miniconda", "Library", "bin"),
            os.path.join(home, "bin", "miniforge"),
            os.path.join(app_dir, "env", "Library", "bin"),
            os.path.join(app_dir, "env", "Scripts"),
        )
        for root in roots:
            for name in ("ffmpeg.exe", "ffmpeg"):
                candidate = os.path.join(root, name)
                if os.path.isfile(candidate):
                    found = candidate
                    break
            if found:
                break
    _CACHED = found or "ffmpeg"
    return _CACHED


def ffprobe_binary(refresh=False):
    """ffprobe, resolved beside whatever ffmpeg resolved to."""
    found = shutil.which("ffprobe")
    if found:
        return found
    ffmpeg = ffmpeg_binary(refresh=refresh)
    if os.path.isabs(ffmpeg):
        for name in ("ffprobe.exe", "ffprobe"):
            candidate = os.path.join(os.path.dirname(ffmpeg), name)
            if os.path.isfile(candidate):
                return candidate
    return "ffprobe"
