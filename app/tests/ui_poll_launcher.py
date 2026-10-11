"""Run tests/two_face_video.py (same argv) with the REAL API server alive on a thread of the render process.

    ROOP_BENCH_LAUNCHER=tests/ui_poll_launcher.py ROOP_UIPOLL_PORT=<port>   (see tests/ab_ui_poll_contention.py)

WHY. The app serves its UI from a non-reloading uvicorn thread INSIDE the process that renders (api.run_api), so every
UI request - /api/progress, /api/live_frame, a 206 range read - is Python work on the same GIL as the swap workers.
two_face_video.py has no server, so a bench that polls "the app" would poll nothing. This starts `api.app` exactly as
the app does (loopback, one thread, `log_level="error"`, the Windows accept-abort shim) and then runs the render.
The server is up in EVERY arm of the A/B, idle in the control arm, so the arms differ in the clients alone.

`ROOP_UIPOLL_MEDIA` (a file path) is hard-linked/copied into the render's output folder as `uipoll_media.mp4` once
`roop.globals.output_path` is set, so the range-request client has something under /outputs/ to seek in.
"""
import os
import runpy
import shutil
import socket
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _start_server(port):
    import uvicorn
    from roop.win_asyncio_compat import install as _install
    _install()
    import api
    server = uvicorn.Server(uvicorn.Config(api.app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, name="uipoll-uvicorn", daemon=True).start()
    deadline = time.time() + 90
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                print("[UIPoll] server up on 127.0.0.1:%d (thread %s)" % (port, "uipoll-uvicorn"), flush=True)
                return server
        except OSError:
            time.sleep(0.25)
    raise SystemExit("[UIPoll] API server did not come up")


def _stage_media(src):
    """Keep the media in whatever folder /outputs/ currently serves.

    `two_face_video` replaces `output_path` after start-up, so staging once (into the default folder) leaves every later
    range read a 404. Re-check twice a second for the life of the render."""
    import roop.globals as g
    last = None
    while True:
        out = getattr(g, "output_path", None)
        if out and os.path.isdir(out):
            dest = os.path.join(out, "uipoll_media.mp4")
            if not os.path.exists(dest):
                try:
                    os.link(src, dest)
                except OSError:
                    shutil.copyfile(src, dest)
            if dest != last:
                last = dest
                print("[UIPoll] range media staged: %s" % dest, flush=True)
        time.sleep(0.5)


def main():
    port = int(os.environ["ROOP_UIPOLL_PORT"])
    _start_server(port)
    media = os.environ.get("ROOP_UIPOLL_MEDIA")
    if media:
        threading.Thread(target=_stage_media, args=(media,), name="uipoll-media", daemon=True).start()
    sys.argv = [os.path.join(HERE, "two_face_video.py")] + sys.argv[1:]
    runpy.run_path(os.path.join(HERE, "two_face_video.py"), run_name="__main__")


if __name__ == "__main__":
    main()
