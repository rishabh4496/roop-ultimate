"""Multi-threaded VIDEO render through GPEN Ultimate and Restore Ultra.

WHY THIS EXISTS SEPARATELY FROM THE IMAGE HARNESS. The single-frame checks run
one face through one processor on the calling thread. That is the one
configuration in which a TensorRT context-sharing bug CANNOT appear. A real
video render is the opposite: N worker threads call the same processor
concurrently, and a shared execution context entered twice at once corrupts the
CUDA context (error 999) or silently returns garbage for one of the callers.

Both profiles inherit `self_excluding = True` from their bases, which tells
ProcessMgr's enhance stage to SKIP its global lock on the promise that every
session call is routed through `enhance_common.exclusive` -- a pool lease when
the processor owns a pool, its own mutex when it does not. If a subclass had
inherited its parent's `_session_lock` (they are class attributes on both
bases), that promise would be broken in exactly this configuration and nowhere
else. So this harness drives the real `batch_process_with_options` video path
with several workers, under TensorRT, and checks that every frame comes back.

Run: env/Scripts/python.exe tools/verify_ultimate_profiles_video.py
     ROOP_VERIFY_PROVIDER=cuda to use the CUDA EP instead.
     ROOP_VERIFY_THREADS=N to change the worker count (default 4).
"""

import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)

# run.py does this before anything else, and the render path prints status
# lines containing checkmarks. Redirected stdout on Windows defaults to cp1252,
# where those raise UnicodeEncodeError -- which surfaces as a render that
# processed every frame and then produced no file. Match the app's startup.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8')

FACESETS = os.path.join(APP, 'facesets')
ENHANCERS = ['GPEN Ultimate', 'Restore Ultra']
FRAMES = 24


def build_clip(path, face_png, frames=FRAMES, fps=12.0):
    """A short clip that MOVES the face, so tracking and per-frame detection
    both do real work rather than repeating one cached result."""
    img = cv2.imread(face_png)
    assert img is not None, face_png
    img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_CUBIC)
    h, w = 360, 480
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    assert writer.isOpened(), 'could not open the mp4v writer'
    for i in range(frames):
        canvas = np.full((h, w, 3), 40, np.uint8)
        # A slow horizontal drift plus a slight vertical bob.
        x = int(20 + (w - 360) * (i / max(1, frames - 1)))
        y = int(15 + 10 * np.sin(i * 0.5))
        canvas[y:y + 320, x:x + 320] = img
        writer.write(canvas)
    writer.release()
    assert os.path.exists(path) and os.path.getsize(path) > 0
    return path


def count_frames(path):
    cap = cv2.VideoCapture(path)
    n = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    cap.release()
    return n


def main():
    import roop.globals
    from roop.core import decode_execution_providers

    requested = os.environ.get('ROOP_VERIFY_PROVIDER', 'tensorrt')
    threads = int(os.environ.get('ROOP_VERIFY_THREADS', '4'))
    roop.globals.execution_providers = decode_execution_providers([requested])
    roop.globals.execution_threads = threads

    from settings import Settings
    roop.globals.CFG = Settings(os.path.join(APP, 'config.yaml'))

    # `core.run()` copies these out of CFG on startup, and this harness enters
    # below it. Without them `video_encoder` is None and FFMPEG_VideoWriter
    # aborts the render with "Video encoder 'None' is not working" -- which
    # looks like an enhancer failure but is purely a missing global.
    roop.globals.video_encoder = roop.globals.CFG.output_video_codec
    roop.globals.video_quality = roop.globals.CFG.video_quality

    import roop.core as core
    from roop.face_util import extract_face_images
    from roop.FaceSet import FaceSet
    from roop.ProcessEntry import ProcessEntry
    from roop.ProcessOptions import ProcessOptions

    print(f'provider = {requested}   workers = {threads}   frames = {FRAMES}\n')

    tmp = tempfile.mkdtemp(prefix='ultimate_video_')
    try:
        clip = build_clip(os.path.join(tmp, 'clip.mp4'),
                          os.path.join(FACESETS, 'simran.png'))
        expected = count_frames(clip)
        print(f'built {clip} with {expected} frames\n')

        src = extract_face_images(os.path.join(FACESETS, 'lori.png'), (False, 0))
        assert src, 'no face in the source image'
        fs = FaceSet()
        face = src[0][0]
        face.mask_offsets = (0, 0, 0, 0, 1, 20)
        fs.faces.append(face)
        fs.ref_images.append(src[0][1])
        fs.AverageEmbeddings()
        roop.globals.INPUT_FACESETS = [fs]
        roop.globals.TARGET_FACES = []

        roop.globals.output_path = tmp
        roop.globals.target_path = clip
        roop.globals.processing = True

        for name in ENHANCERS:
            roop.globals.selected_enhancer = name
            processors = core.get_processing_plugins('mask_realityux')
            assert 'gpen_ultimate' in processors or 'restore_ultra' in processors, \
                f'{name}: the enhancer did not reach the processor chain'

            options = ProcessOptions(
                processors, roop.globals.distance_threshold,
                roop.globals.blend_ratio, 'all', 0, '', None, 1, 128,
                False, False)

            entry = ProcessEntry(clip, 0, 0, 0)
            before = set(os.listdir(tmp))
            core.batch_process_with_options([entry], options, None)

            # `entry.finalname` is the PRE-MUX temp name ("<stem>__temp.mp4").
            # core.batch_process then muxes audio into a template-expanded
            # destination and deletes the temp, so finalname does not exist by
            # the time this returns. Take whatever new video appeared instead.
            produced = [f for f in set(os.listdir(tmp)) - before
                        if f.lower().endswith(('.mp4', '.mkv', '.webm', '.gif'))]
            assert produced, (
                f'{name}: no output video appeared in {tmp} '
                f'(finalname was {entry.finalname!r})')
            out = os.path.join(tmp, produced[0])
            got = count_frames(out)
            size = os.path.getsize(out)
            assert size > 1024, f'{name}: output is {size} bytes'
            # Every input frame must survive. A worker that died on a corrupted
            # TensorRT context shows up here as a short video, not as a crash.
            assert got == expected, \
                f'{name}: {got} frames out of {expected} -- frames were lost'

            cap = cv2.VideoCapture(out)
            ok, first = cap.read()
            cap.release()
            assert ok and first is not None, f'{name}: output unreadable'
            assert np.isfinite(first.astype(np.float32)).all()
            assert first.std() > 5.0, f'{name}: output frame is flat/blank'

            print(f'  {name:<16} ok  {got}/{expected} frames  '
                  f'{size / 1024:7.1f} KB  first-frame std={first.std():.2f}')
            os.remove(out)

        print('\nVIDEO PIPELINE CHECKS PASSED')
    finally:
        roop.globals.processing = False
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
