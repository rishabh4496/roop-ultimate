"""Measure what the pipeline does when a second face or an object crosses the target (Stage 19).

`synthetic_scenes.py` draws the scenes; this module renders them through the real
`two_face_video.py` harness and grades the output against geometry that is known
exactly. Shared by `tests/test_occlusion_interaction.py` (gates) and
`verify_integration.py` (a printed table).

Three measurements, each from pixels or from the pipeline's own per-face rows:

  crossing   who was swapped (rows.csv: one row per detected face per frame) and how
             many bystander pixels changed. The control for "did anything change" is
             the SAME clip with the target sprite left out and the SAME pinned target
             captured from the real scene: with no target in the shot nothing can be
             swapped, so the pipeline passes every pixel through and any difference
             on the bystander is caused by the swap, not by the codec or the paste.
  alignment  the swap crop's matrix (`estimate_norm` on the detector's 5 keypoints),
             bar scene vs the same frames with no bar.
  flicker    second-order temporal luma derivative on swapped skin, and the LAB
             colour of that skin against the no-bar render of the same frames.

Renders are lossless (libx264 crf 0) and un-trimmed so a pixel difference is the
pipeline's, not an encoder's (see `ROOP_BENCH_CODEC`/`ROOP_BENCH_CRF`/`ROOP_BENCH_NO_TRIM`
in two_face_video.py).
"""
import csv
import glob
import os
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import synthetic_scenes as sc   # noqa: E402

PYTHON = sys.executable
_ERODE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


# ----------------------------------------------------------------------------- rendering
class Render:
    def __init__(self, tag, out_dir, log_path):
        self.tag, self.out_dir, self.log_path = tag, out_dir, log_path
        # the harness writes <out>/<tag>/{rows.csv, work/clip.mp4 (input), work/clip_*.mp4 (output)}
        base = os.path.join(out_dir, tag)
        self.rows_path = os.path.join(base, 'rows.csv')
        outs = [p for p in glob.glob(os.path.join(base, 'work', 'clip_*.mp4'))]
        if not outs:
            raise RuntimeError(f'render {tag} produced no output clip (see {log_path})')
        self.output_path = max(outs, key=os.path.getmtime)

    def frames(self):
        return read_frames(self.output_path)

    def rows(self):
        with open(self.rows_path, newline='') as fh:
            return list(csv.DictReader(fh))


def read_frames(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def render(video, tag, out_dir, *, capture_face, fixture_in=None, fixture_out=None, timeout=900, reuse=False):
    """Render `video` through the production path (realswap + RealityUX, tracking on).

    `reuse=True` returns an existing render of the same tag in `out_dir` instead of redoing it
    (re-grading with different thresholds; never used by the test, which renders fresh)."""
    os.makedirs(out_dir, exist_ok=True)
    if reuse and os.path.exists(os.path.join(out_dir, tag, 'rows.csv')):
        try:
            return Render(tag, out_dir, os.path.join(out_dir, tag + '.log'))
        except RuntimeError:
            pass
    env = dict(os.environ)
    env.update({'ROOP_BENCH_CODEC': 'libx264', 'ROOP_BENCH_CRF': '0', 'ROOP_BENCH_NO_TRIM': '1',
                'NO_ALBUMENTATIONS_UPDATE': '1'})
    env.pop('ROOP_BENCH_FIXTURE_IN', None)
    env.pop('ROOP_BENCH_FIXTURE_OUT', None)
    if fixture_in:
        env['ROOP_BENCH_FIXTURE_IN'] = fixture_in
    if fixture_out:
        env['ROOP_BENCH_FIXTURE_OUT'] = fixture_out
    cmd = [PYTHON, os.path.join(HERE, 'two_face_video.py'),
           '--tag', tag, '--video', video, '--sources', sc.SOURCE,
           '--capture', '0', '--capture-face', str(capture_face),
           '--swap-model', 'realswap', '--mask-engine', 'RealityUX', '--enhancer', 'None',
           '--provider', 'tensorrt', '--threads', '20', '--tracking', '1', '--out', out_dir]
    log_path = os.path.join(out_dir, tag + '.log')
    with open(log_path, 'wb') as log:
        rc = subprocess.run(cmd, cwd=APP, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode
    if rc != 0:
        raise RuntimeError(f'render {tag} exited {rc}; see {log_path}')
    return Render(tag, out_dir, log_path)


# ----------------------------------------------------------------------------- crossing
def classify_rows(rows, scene):
    """Per frame: did the pipeline swap the target's box, and did it swap the bystander's?

    A row is one DETECTED face. It belongs to whichever sprite centre it is nearer to
    (both sprites share the same box-to-sprite offset, so the offset cancels).
    """
    n = len(scene.frames)
    target = [None] * n        # None = target not detected this frame, else swapped bool
    bystander = [None] * n
    for r in rows:
        k = int(r['frame'])
        cx, cy = (int(r['x0']) + int(r['x1'])) / 2.0, (int(r['y0']) + int(r['y1'])) / 2.0
        t, b = scene.target_centre[k], scene.bystander_centre[k]
        # `why` joins the audit's reasons with ' | ' and the swap verdict is not always first
        swapped = any(seg.strip().startswith('swapped') for seg in r['why'].split('|'))
        d_t = (cx - t[0]) ** 2 + (cy - t[1]) ** 2
        d_b = (cx - b[0]) ** 2 + (cy - b[1]) ** 2
        slot = target if d_t <= d_b else bystander
        slot[k] = bool(slot[k]) or swapped if slot[k] is not None else swapped
    return target, bystander


def strict_bystander_mask(scene, k, erode_iter=2):
    """Bystander pixels that are 100% bystander: alpha ~1, not under or beside the target."""
    m = (scene.bystander_alpha[k] >= 0.999) & (scene.target_alpha[k] <= 0.001)
    return cv2.erode(m.astype(np.uint8), _ERODE, iterations=erode_iter).astype(bool)


def bystander_change(scene, swapped_frames, control_frames):
    """Per frame: MSE / changed-pixel count / max |d| of the bystander's strict pixels,
    render vs the target-free control."""
    out = []
    for k in range(min(len(swapped_frames), len(control_frames), len(scene.frames))):
        m = strict_bystander_mask(scene, k)
        if not m.any():
            out.append({'frame': k, 'px': 0, 'mse': 0.0, 'changed': 0, 'max': 0})
            continue
        d = np.abs(swapped_frames[k].astype(np.int16) - control_frames[k].astype(np.int16))[m]
        out.append({'frame': k, 'px': int(m.sum()), 'mse': float((d.astype(np.float64) ** 2).mean()),
                    'changed': int((d.max(axis=1) > 2).sum()), 'max': int(d.max())})
    return out


def summarize_crossing(scene, rows, swapped_frames, control_frames):
    tgt, byd = classify_rows(rows, scene)
    change = bystander_change(scene, swapped_frames, control_frames)
    iou = [scene.iou(k) for k in range(len(scene.frames))]
    return {
        'peak_iou': max(iou),
        'frames': len(scene.frames),
        'target_detected': sum(1 for v in tgt if v is not None),
        'target_swapped': sum(1 for v in tgt if v),
        'target_unswapped': sum(1 for v in tgt if v is False),
        'bystander_detected': sum(1 for v in byd if v is not None),
        'bystander_swapped': sum(1 for v in byd if v),
        'bystander_px': sum(c['px'] for c in change),
        'bystander_mse_max': max(c['mse'] for c in change),
        'bystander_changed_max': max(c['changed'] for c in change),
        'bystander_dmax': max(c['max'] for c in change),
        'bystander_frames_nonzero': sum(1 for c in change if c['mse'] > 0.0),
    }


# ----------------------------------------------------------------------------- bar scene
def _luma(f):
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _lab(f):
    lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return lab


def _d2(frames, mask, k):
    a, b, c = _luma(frames[k - 1]), _luma(frames[k]), _luma(frames[k + 1])
    return float(np.abs(a - 2 * b + c)[mask].mean()) if mask.any() else float('nan')


def flicker(scene, plate, out_bar, out_clean):
    """Second-order luma derivative on swapped skin (|L[t-1] - 2 L[t] + L[t+1]|), and the
    LAB distance of that skin to the no-bar render of the same frame. Skin = the target's
    visible pixels, eroded, with the bar and a margin around it removed."""
    n = min(len(out_bar), len(out_clean), len(scene.frames))
    d2_in, d2_bar, d2_clean, de = [], [], [], []
    for k in range(1, n - 1):
        skin = (scene.target_visible[k - 1] & scene.target_visible[k] & scene.target_visible[k + 1])
        skin = cv2.erode(skin.astype(np.uint8), _ERODE, iterations=3).astype(bool)
        d2_in.append(_d2(plate, skin, k))
        d2_bar.append(_d2(out_bar, skin, k))
        d2_clean.append(_d2(out_clean, skin, k))
    for k in range(n):
        skin = cv2.erode(scene.target_visible[k].astype(np.uint8), _ERODE, iterations=3).astype(bool)
        skin &= ~cv2.dilate(scene.bar[k].astype(np.uint8), _ERODE, iterations=4).astype(bool)
        if skin.sum() < 500:
            continue
        de.append(float(np.linalg.norm(_lab(out_bar[k])[skin].mean(axis=0) - _lab(out_clean[k])[skin].mean(axis=0))))
    return {
        'd2_input': float(np.nanmean(d2_in)),
        'd2_bar': float(np.nanmean(d2_bar)),
        'd2_clean': float(np.nanmean(d2_clean)),
        'd2_excess_max': float(np.nanmax(np.array(d2_bar) - np.array(d2_clean))),
        'de_mean': float(np.mean(de)), 'de_max': float(np.max(de)), 'de_frames': len(de),
    }


_PIPELINE_READY = False


def _init_pipeline():
    global _PIPELINE_READY
    if _PIPELINE_READY:
        return
    import yaml
    import angle_bench as ab
    import two_face_video as tfv
    tfv._apply_startup_runtime_environment()
    cfg = yaml.safe_load(open(os.path.join(APP, 'config.yaml'))) or {}
    ab.init_pipeline(cfg.get('provider') or 'cuda', cfg.get('swap_model') or 'inswapper', 'None', 'None',
                     sync_config=True)
    import roop.globals as G
    G.processing = True
    _PIPELINE_READY = True


def alignment_deviation(scene, bar_frames, clean_frames):
    """Translation (px, of the crop centre in frame space) and rotation (deg) between the
    swap crop's matrix on the bar frames and on the same frames without the bar.

    The matrix is the one the swap uses: `estimate_norm(kps, 256, 'arcface')` on the
    detector's five keypoints. A frame where either run does not find exactly one face is
    reported as a miss, not as zero."""
    _init_pipeline()
    from roop.face_util import estimate_norm, get_all_faces
    from roop.visibility_alignment import decompose

    def kps_of(frames):
        out = {}
        for k, f in enumerate(frames):
            fs = [x for x in (get_all_faces(f) or []) if getattr(x, 'kps', None) is not None]
            if len(fs) == 1:
                out[k] = np.asarray(fs[0].kps, np.float32)
        return out

    def centre(M):
        return (np.linalg.inv(np.vstack([M, [0, 0, 1]])) @ np.array([128, 128, 1.0]))[:2]

    kb, kc = kps_of(bar_frames), kps_of(clean_frames)
    ks = sorted(set(kb) & set(kc))
    tr, ro, over = [], [], []
    h = 40
    cx, cy = sc.CENTRE
    for k in ks:
        Mb, Mc = estimate_norm(kb[k], 256, 'arcface'), estimate_norm(kc[k], 256, 'arcface')
        tr.append(float(np.linalg.norm(centre(Mb) - centre(Mc))))
        ro.append(float(abs(decompose(Mb)[1] - decompose(Mc)[1])))
        over.append(bool(scene.bar[k][cy - h:cy + h, cx - h:cx + h].any()))
    tr, ro, over = np.array(tr), np.array(ro), np.array(over)
    return {
        'frames': len(scene.frames), 'matched': len(ks), 'missed': len(scene.frames) - len(ks),
        'translation_mean': float(tr.mean()), 'translation_max': float(tr.max()),
        'rotation_mean': float(ro.mean()), 'rotation_max': float(ro.max()),
        'bar_frames': int(over.sum()),
        'bar_translation_max': float(tr[over].max()) if over.any() else 0.0,
        'bar_rotation_max': float(ro[over].max()) if over.any() else 0.0,
        'per_frame': [(ks[i], float(tr[i]), float(ro[i])) for i in range(len(ks))],
    }


# ----------------------------------------------------------------------------- all scenes
def run_all(workdir, reuse=False):
    """Render every scene and return the measurements the test and the verifier print."""
    os.makedirs(workdir, exist_ok=True)
    result = {}
    for front in ('bystander', 'target'):
        scene = sc.crossing(front)
        ctrl = sc.crossing(front, include_target=False)
        scene_clip = sc.write_clip(scene.frames, os.path.join(workdir, f'cross_{front}.mp4'))
        ctrl_clip = sc.write_clip(ctrl.frames, os.path.join(workdir, f'ctrl_{front}.mp4'))
        fixture = os.path.join(workdir, f'fix_{front}.pkl')
        # the target is the second face from the left in frame 0 (the bystander starts far left)
        r = render(scene_clip, f'cross_{front}', workdir, capture_face=1, fixture_out=fixture, reuse=reuse)
        c = render(ctrl_clip, f'ctrl_{front}', workdir, capture_face=1, fixture_in=fixture, reuse=reuse)
        result[f'cross_{front}'] = summarize_crossing(scene, r.rows(), r.frames(), c.frames())

    bar = sc.bar_pass(True)
    clean = sc.bar_pass(False)
    bar_clip = sc.write_clip(bar.frames, os.path.join(workdir, 'bar.mp4'))
    clean_clip = sc.write_clip(clean.frames, os.path.join(workdir, 'bar_clean.mp4'))
    rb = render(bar_clip, 'bar', workdir, capture_face=0, reuse=reuse)
    rc = render(clean_clip, 'bar_clean', workdir, capture_face=0, reuse=reuse)
    result['flicker'] = flicker(bar, bar.frames, rb.frames(), rc.frames())
    result['alignment'] = alignment_deviation(bar, bar.frames, clean.frames)
    return result
