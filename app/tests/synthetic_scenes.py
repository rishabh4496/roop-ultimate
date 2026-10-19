"""Deterministic multi-frame scenes for the interaction stress tests (Stage 19).

Drawn faces are not detected, so the "faces" are the app's own real 256x256 faceset
photos (`app/facesets/*.png`) feathered onto a seeded background. Everything is
geometry: every frame's sprite boxes and visible-pixel masks are known exactly, so a
test can say "this region must not have changed" without a detector in the loop.

Scenes (30 frames, 960x540, 30 fps):

  crossing(front='bystander'|'target')
      A non-target face slides horizontally across a fixed target face, offset
      vertically so the sprites' IoU peaks at ~0.75 (not 1.0). `front` is which of
      the two is drawn on top. The detector cannot see a face that is >~55%
      covered, so the two orderings test different things: bystander in front
      (the target disappears behind it - nothing may be swapped onto the
      bystander) and target in front (the target must stay swapped while a
      bystander slides behind it).

  bar_pass(with_bar=True)
      A skin-coloured vertical bar sweeps across the target face (eyes, nose,
      mouth) over the clip. `with_bar=False` renders the same frames clean: the
      detector's output on those is the ground-truth trajectory.

The clips are written losslessly (libx264 crf 0) through ffmpeg so the pipeline's
decode sees exactly these pixels.
"""
import os
import subprocess

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)

W, H, FPS, N_FRAMES = 960, 540, 30, 30
SIZE = 220                      # sprite edge, px
CENTRE = (480, 270)             # the target's centre
TARGET, BYSTANDER = 'anushree', 'anshita'
SOURCE = 'ashna'                # the faceset that is swapped IN (a third person)


def faceset_png(name):
    return os.path.join(APP, 'facesets', f'{name}.png')


def sprite(name, size=SIZE):
    im = cv2.imread(faceset_png(name))
    if im is None:
        raise FileNotFoundError(faceset_png(name))
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_CUBIC)
    a = np.zeros((size, size), np.float32)
    cv2.ellipse(a, (size // 2, size // 2), (int(size * 0.47), int(size * 0.5)), 0, 0, 360, 1.0, -1)
    return im, cv2.GaussianBlur(a, (0, 0), size * 0.03)


def background():
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur(rng.integers(90, 140, (H // 8, W // 8, 3), dtype=np.uint8), (0, 0), 1.0)
    return cv2.resize(base, (W, H), interpolation=cv2.INTER_CUBIC)


def paste(frame, im, alpha, cx, cy):
    """Alpha-composite `im` centred at (cx, cy); returns the full-frame alpha it wrote."""
    size = im.shape[0]
    x0, y0 = int(round(cx - size / 2)), int(round(cy - size / 2))
    full = np.zeros((H, W), np.float32)
    xs, ys = max(0, x0), max(0, y0)
    xe, ye = min(W, x0 + size), min(H, y0 + size)
    if xe <= xs or ye <= ys:
        return full
    a = alpha[ys - y0:ye - y0, xs - x0:xe - x0, None]
    sub = frame[ys:ye, xs:xe].astype(np.float32)
    frame[ys:ye, xs:xe] = (a * im[ys - y0:ye - y0, xs - x0:xe - x0].astype(np.float32)
                           + (1 - a) * sub).astype(np.uint8)
    full[ys:ye, xs:xe] = a[..., 0]
    return full


def ta_full_alpha(alpha, centre):
    """Full-frame alpha of a sprite that is NOT drawn (for control-scene ground truth)."""
    full = np.zeros((H, W), np.float32)
    size = alpha.shape[0]
    x0, y0 = int(round(centre[0] - size / 2)), int(round(centre[1] - size / 2))
    xs, ys = max(0, x0), max(0, y0)
    xe, ye = min(W, x0 + size), min(H, y0 + size)
    full[ys:ye, xs:xe] = alpha[ys - y0:ye - y0, xs - x0:xe - x0]
    return full


def sprite_iou(c1, c2, size=SIZE):
    ox = max(0, size - abs(c1[0] - c2[0]))
    oy = max(0, size - abs(c1[1] - c2[1]))
    inter = ox * oy
    return inter / float(2 * size * size - inter)


class Scene:
    """frames + the ground truth a test needs."""

    def __init__(self):
        self.frames = []
        self.target_centre = []
        self.bystander_centre = []
        self.target_alpha = []        # full-frame alpha of the target sprite, as drawn
        self.bystander_alpha = []
        self.bystander_visible = []   # bool mask: bystander pixels that are actually on screen
        self.target_visible = []
        self.bar = []                 # bool mask of the occluding bar, or None

    def iou(self, k):
        return sprite_iou(self.target_centre[k], self.bystander_centre[k])


def crossing(front='bystander', n=N_FRAMES, peak_iou=0.75, include_target=True):
    assert front in ('bystander', 'target')
    # vertical offset that makes the sprites' IoU at dx = 0 equal `peak_iou`
    overlap = peak_iou * 2 * SIZE * SIZE / (1 + peak_iou)
    dy = int(round(SIZE - overlap / SIZE))
    s = Scene()
    tgt, ta = sprite(TARGET)
    byd, ba = sprite(BYSTANDER)
    step = 30                                  # px per frame; frame n//2 - 1 sits exactly on dx = 0
    for k in range(n):
        bx = CENTRE[0] + step * (k - (n // 2 - 1))
        bc = (int(bx), CENTRE[1] - dy)
        f = background()
        order = [(tgt, ta, CENTRE, 't'), (byd, ba, bc, 'b')]
        if front == 'target':
            order = order[::-1]
        alphas = {}
        for im, a, c, tag in order:
            if tag == 't' and not include_target:
                # CONTROL scene: the target is simply not in the shot, so nothing
                # can be swapped and the pipeline passes every pixel through. The
                # ground-truth masks below still describe the full scene.
                alphas[tag] = np.zeros((H, W), np.float32) + ta_full_alpha(ta, c)
                continue
            alphas[tag] = paste(f, im, a, c[0], c[1])
        # visible = drawn alpha, minus whatever the later (front) sprite covers
        front_tag, back_tag = order[-1][3], order[0][3]
        vis = {front_tag: alphas[front_tag] > 0.5,
               back_tag: (alphas[back_tag] > 0.5) & ~(alphas[front_tag] > 0.5)}
        s.frames.append(f)
        s.target_centre.append(CENTRE)
        s.bystander_centre.append(bc)
        s.target_alpha.append(alphas['t'])
        s.bystander_alpha.append(alphas['b'])
        s.target_visible.append(vis['t'])
        s.bystander_visible.append(vis['b'])
        s.bar.append(None)
    return s


def bar_pass(with_bar=True, n=N_FRAMES, bar_w=70):
    """A skin-coloured bar sweeping DOWN across the target: forehead -> chin."""
    s = Scene()
    tgt, ta = sprite(TARGET)
    ys = np.linspace(CENTRE[1] - SIZE * 0.55, CENTRE[1] + SIZE * 0.55, n)
    skin = np.array([120, 150, 205], np.uint8)            # BGR, flesh-toned
    for k in range(n):
        f = background()
        alpha = paste(f, tgt, ta, CENTRE[0], CENTRE[1])
        bar = np.zeros((H, W), bool)
        y = int(round(ys[k]))
        y0, y1 = max(0, y - bar_w // 2), min(H, y + bar_w // 2)
        x0, x1 = CENTRE[0] - SIZE, CENTRE[0] + SIZE
        bar[y0:y1, x0:x1] = True
        if with_bar:
            rng = np.random.default_rng(k)
            patch = np.clip(skin.astype(np.float32) + rng.normal(0, 5, (y1 - y0, x1 - x0, 3)), 0, 255).astype(np.uint8)
            f[y0:y1, x0:x1] = patch
        s.frames.append(f)
        s.target_centre.append(CENTRE)
        s.bystander_centre.append(None)
        s.target_alpha.append(alpha)
        s.bystander_alpha.append(None)
        s.target_visible.append((alpha > 0.5) & ~bar)
        s.bystander_visible.append(None)
        s.bar.append(bar)
    return s


def write_clip(frames, path, fps=FPS):
    """Lossless H.264 through ffmpeg, so the pipeline decodes exactly these pixels."""
    from roop.ffmpeg_path import ffmpeg_binary
    h, w = frames[0].shape[:2]
    cmd = [ffmpeg_binary(), '-hide_banner', '-y', '-loglevel', 'error', '-f', 'rawvideo',
           '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
           '-c:v', 'libx264', '-crf', '0', '-pix_fmt', 'yuv444p', '-preset', 'veryfast', path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError('ffmpeg failed writing ' + path)
    return path
