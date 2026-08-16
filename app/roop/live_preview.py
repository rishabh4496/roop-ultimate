"""The most recent processed frame, for the UI's processing view.

The pipeline used to hand every finished frame to the UI by keeping a full-frame
COPY of it (`latest_swapped_frame`) — a ~6 MB memcpy per frame, on the hot swap
path, whether or not anyone was looking. That is why it was removed and
`_publish_live` became a no-op: the cost was per frame while the UI could only
ever display a couple of frames a second, so the box showed a still and the
preview looked frozen for the whole run.

This does the same job for a fixed, tiny budget:

  * THROTTLED — at most one frame every ROOP_LIVE_PREVIEW_MS (default 500), so
    the cost does not scale with frame rate. At 20 fps that is 1 frame in 10.
  * DOWNSCALED — to ROOP_LIVE_PREVIEW_WIDTH (default 960) before anything else
    touches it. The first version used 480 on the reasoning that it was "more
    than the box can show", which is only true at devicePixelRatio 1: the box is
    ~480 CSS px wide, so on the HiDPI displays these renders are watched on it
    was being stretched 2x and looked heavily pixelated. 960 covers a 2x box
    with nothing to spare, which is why it is the default rather than more.
  * ENCODED ONCE — to JPEG here rather than per HTTP poll, so N pollers cost
    nothing extra and the API thread never touches a numpy array (no lock held
    across an encode, no copy needed for thread safety).

  * WATCHED-GATED — the full cadence only runs while someone is actually
    fetching the result. Nothing about a hidden Pinokio tab stops the pipeline,
    so this used to encode a frame twice a second for a browser that had not
    asked for one since the run started. See _INTERVAL vs _IDLE_INTERVAL below.

Measured shape of the work (this machine): resize plus imencode is 3.7 ms per
publish from 1080p (median of 30; p90 8.1) and 6.8 ms from 4K, twice a second —
about 0.7% of one thread, and OpenCV releases the GIL for both. A frame that
does NOT publish costs 0.27 us, so the per-frame tax on the swap path is
nothing. The frame it produces is 20-33 KB, i.e. under 70 KB/s over loopback.
Set ROOP_LIVE_PREVIEW=0 to switch it off entirely (publish becomes a bare
return).

For the avoidance of a wrong conclusion: these numbers are why the live preview
is NOT the reason a render goes faster with the Pinokio Terminal on screen. The
cost above is paid whether or not anyone is looking, so it cannot produce a
difference between the two. That difference is on the browser's side of the
loopback — chiefly the backdrop-filter panels layered over this image, which
each re-blur when it changes. See `data-render-lite` in react-ui/src/index.css.
"""

import os
import threading
import time

import cv2
import numpy as np

_ENABLED = os.environ.get('ROOP_LIVE_PREVIEW', '1').strip().lower() not in ('0', 'false', 'off')

try:
    _INTERVAL = max(0.05, float(os.environ.get('ROOP_LIVE_PREVIEW_MS', '500')) / 1000.0)
except ValueError:
    _INTERVAL = 0.5

try:
    _MAX_W = max(160, int(os.environ.get('ROOP_LIVE_PREVIEW_WIDTH', '960')))
except ValueError:
    _MAX_W = 960

# JPEG quality for that frame. 72 is a web-photo default and shows its blocking
# on the flat skin the swap is judged on; 88 costs ~1.5x the bytes of a frame
# that is already tiny and is what stops the live view looking worse than the
# render it is reporting on.
try:
    _QUALITY = min(100, max(40, int(os.environ.get('ROOP_LIVE_PREVIEW_QUALITY', '88'))))
except ValueError:
    _QUALITY = 88

# Cadence when nobody is fetching the result, and how long after a fetch we go
# on believing someone is there.
#
# This is NOT simply "stop publishing when unwatched", because the UI only
# refetches when `seq` changes (it is the image URL's cache key) — so a gate
# that froze `seq` would have no way to notice a viewer coming back, and the
# preview would stay dead for the rest of the run. Publishing slowly instead
# keeps `seq` moving, so a returning tab picks a frame up on its next progress
# poll, at ~1/25th the cost of the watched cadence.
try:
    _IDLE_INTERVAL = max(_INTERVAL, float(os.environ.get('ROOP_LIVE_PREVIEW_IDLE_MS', '3000')) / 1000.0)
except ValueError:
    _IDLE_INTERVAL = 3.0

_WATCH_TTL = 4.0

_lock = threading.Lock()
# jpeg: encoded bytes, seq: bumped on every publish (the UI's cache key),
# t: last publish time (throttle), size: source frame size for the caption,
# fetched: last time anyone read the result (0 = nobody ever has).
_state = {'jpeg': None, 'seq': 0, 't': 0.0, 'size': (0, 0), 'fetched': 0.0}


def reset():
    """Drop the previous run's frame so a new run never shows a stale one."""
    with _lock:
        _state.update({'jpeg': None, 'seq': 0, 't': 0.0, 'size': (0, 0), 'fetched': 0.0})


def note_fetch():
    """Record that something read the live frame — called by /api/live_frame.

    Deliberately a bare timestamp write rather than a counter or a viewer set:
    the only question publish() asks is "has anyone looked recently", and the
    answer must cost nothing to record on a request that runs during a render.
    """
    _state['fetched'] = time.time()


def enabled():
    return _ENABLED


def publish(frame):
    """Offer a finished frame. Cheap and safe to call for every frame — most
    calls return on the throttle check without touching the pixels."""
    if not _ENABLED or frame is None:
        return
    now = time.time()
    # Read without the lock: a torn read here can only mean one extra (or one
    # skipped) publish, and both are harmless. Taking the lock per frame would
    # put every worker thread through a mutex for a preview. (Measured: eight
    # threads released together onto an expired throttle still produce one
    # encode, because the `_state['t']` write lands before any of them reaches
    # the resize — but the design does not depend on that.)
    watched = (now - _state['fetched']) < _WATCH_TTL
    if now - _state['t'] < (_INTERVAL if watched else _IDLE_INTERVAL):
        return
    _state['t'] = now
    try:
        h, w = frame.shape[:2]
        if w > _MAX_W:
            scale = _MAX_W / float(w)
            small = cv2.resize(frame, (_MAX_W, max(1, int(round(h * scale)))),
                               interpolation=cv2.INTER_AREA)
        else:
            small = frame
        ok, buf = cv2.imencode('.jpg', small, [int(cv2.IMWRITE_JPEG_QUALITY), _QUALITY])
        if not ok:
            return
        data = np.asarray(buf).tobytes()
    except Exception:
        return          # a preview must never be able to break a render
    with _lock:
        _state['jpeg'] = data
        _state['seq'] += 1
        _state['size'] = (int(w), int(h))


def seq():
    """Publication counter — 0 until the first frame. The UI uses it as the
    image URL's cache key, so it refetches exactly when there is something new."""
    return _state['seq']


def snapshot():
    """(jpeg_bytes | None, seq, (src_w, src_h))"""
    with _lock:
        return _state['jpeg'], _state['seq'], _state['size']
