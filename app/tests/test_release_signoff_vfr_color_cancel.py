"""Release sign-off: variable-frame-rate sources, colour description, cooperative cancel, installer manifests.

What the brief asked for and where it is ground truth in this repo (a claim needs a named site and a passing test):

1. VFR timestamp passthrough. There is no 3-stream ring, no ``Stream_*`` and no render-path ``NvZeroCopyWriter`` (RECODE_STATUS
   2026-10-26), so "pass each frame's pts through the ring into the writer" has no site. The render is one raw frame per
   SOURCE frame down an ffmpeg pipe, so frame k of the output IS source frame ``trim_start + k``; what was wrong was the
   timestamps: the encoder stamped a uniform grid, the audio kept the source clock. Fixed at the two ends that exist:
   ``roop/vfr_timing.py`` (Stage-1 detection, ``r_frame_rate != avg_frame_rate`` or a frame > 1 ms off the average-rate
   grid) + ``roop/mp4_timing.py`` (re-stamp the intermediate's samples at the source's timestamps before the audio mux), and the
   readers' accurate seek (``vfr_timing.seek_time``): a trimmed VFR render started one frame late.
   Ground truth for "sub-millisecond A/V sync" is NOT a container field: a click is placed in the audio at the exact
   presentation time of a white frame, the file is rendered, and the click is located in the DECODED audio and the flash in
   the DECODED video (``-copyts``), both on the player's clock.
2. Colour. ``roop/source_color.py`` + the two writers on the live path (``ffmpeg_writer``, ``video_stream``): the source's
   matrix, range and tags, measured on the file's own planes. HDR / wide gamut stay with ``hdr_pipeline``.
3. Cancel. ``POST /api/stop`` and ``/api/queue/cancel`` both end in ``roop.globals.processing = False`` (api.stop_swap); the
   scenarios below do exactly that to a real render in its own process and check the exit code, the partial file and the GPU.
4. Installer. The new modules import only the standard library (no new package); the GPU packages the code gates at run
   time are declared with platform markers.
"""
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
ROOT = os.path.dirname(APP)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fixtures  # noqa: E402,F401  (module level: test_fixture_paths requires it)
from fractions import Fraction  # noqa: E402

from roop import mp4_timing, source_color, vfr_timing  # noqa: E402
from roop.ffmpeg_path import ffmpeg_binary, ffprobe_binary  # noqa: E402

FF, FP = ffmpeg_binary(), ffprobe_binary()
SR = 44100
BITS, BLOCK = 12, 16
VFR_PTS = "setpts='(N/30+0.02*floor(N/37)+0.013*sin(N/9))/TB'"        # the harness's phone-like jitter (20 ms steps, 13 ms wobble)
SYNC_TOLERANCE_MS = 1.0
FLASH_ID = (1 << BITS) - 1


# ── media helpers (a measurement, not product code) ─────────────────────────────────────────────────────────────────────
def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, **kw)


def pts_of(path, sel='v:0'):
    out = _run([FP, '-v', 'error', '-select_streams', sel, '-show_entries', 'frame=pts_time', '-of', 'json', path], text=True).stdout
    return sorted(float(f['pts_time']) for f in json.loads(out)['frames'])


def stamp_id(frame, i):
    """Frame number `i` as BITS black/white 16x16 blocks, top-left: survives lossy coding."""
    for b in range(BITS):
        frame[0:BLOCK, b * BLOCK:(b + 1) * BLOCK] = 255 if (i >> b) & 1 else 0
    return frame


def ids_of(path):
    """The stamped number of every decoded frame, in presentation order."""
    raw = _run([FF, '-v', 'error', '-i', path, '-an', '-fps_mode', 'passthrough', '-vf',
                f'crop={BITS * BLOCK}:{BLOCK}:0:0,scale={BITS}:1:flags=area', '-pix_fmt', 'gray', '-f', 'rawvideo', '-']).stdout
    arr = np.frombuffer(raw, np.uint8).reshape(-1, BITS)
    return [int(sum((1 << b) for b in range(BITS) if row[b] > 127)) for row in arr]


def make_clip(path, frames, vfr, flash_every=30, audio=True):
    """Stamped frames -> H.264 (4:2:0, B-frames, GOP 45); VFR with the harness jitter; every `flash_every`-th frame (offset
    half a period) is white, and the audio carries a click at that frame's EXACT presentation time."""
    h, w = frames[0].shape[:2]
    vid = path + '.video.mp4'
    vf = VFR_PTS if vfr else "setpts=N/30/TB"
    cmd = [FF, '-hide_banner', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', '30', '-i', '-',
           '-vf', vf, '-fps_mode', 'vfr', '-c:v', 'libx264', '-crf', '14', '-pix_fmt', 'yuv420p', '-preset', 'veryfast', '-g', '45', vid]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    flash_idx = []
    for i, fr in enumerate(frames):
        fr = stamp_id(fr.copy(), i)
        if flash_every and i % flash_every == flash_every // 2:
            fr = np.full_like(fr, 255)
            flash_idx.append(i)
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
    p.stdin.close()
    assert p.wait() == 0
    pts = pts_of(vid)
    assert len(pts) == len(frames)
    if not audio:
        os.replace(vid, path)
        return flash_idx
    pcm = np.zeros(int((pts[-1] + 2.0) * SR), np.int16)
    for i in flash_idx:
        k = int(round(pts[i] * SR))
        pcm[k:k + 8] = 30000
    wav = path + '.audio.wav'
    with wave.open(wav, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())
    r = _run([FF, '-v', 'error', '-y', '-i', vid, '-i', wav, '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
              '-shortest', path])
    assert r.returncode == 0, r.stderr
    os.remove(vid)
    os.remove(wav)
    return flash_idx


def flash_times(path):
    pts = pts_of(path)
    raw = _run([FF, '-v', 'error', '-i', path, '-an', '-fps_mode', 'passthrough', '-vf', 'scale=32:18', '-pix_fmt', 'gray', '-f', 'rawvideo', '-']).stdout
    mean = np.frombuffer(raw, np.uint8).reshape(-1, 18 * 32).mean(axis=1)
    return [pts[i] for i in range(len(mean)) if mean[i] > 235 and (i == 0 or mean[i - 1] <= 235)]


def click_times(path):
    """Each click on the player's clock: first decoded audio frame's pts (-copyts keeps the container clock) + sample / rate."""
    p = _run([FF, '-v', 'info', '-copyts', '-i', path, '-map', '0:a:0', '-af', 'ashowinfo', '-f', 's16le', '-ac', '1', '-ar', str(SR), '-'])
    pcm = np.frombuffer(p.stdout, np.int16)
    m = re.search(r'pts_time:(-?[0-9.]+)', p.stderr.decode('utf-8', 'replace'))
    t0 = float(m.group(1)) if m else 0.0
    onsets, last = [], -10 ** 9
    for k in np.flatnonzero(np.abs(pcm.astype(np.int32)) > 8000):
        if k - last > SR // 10:
            onsets.append(int(k))
        last = int(k)
    return [t0 + k / SR for k in onsets]


def av_offsets_ms(path):
    v, a = flash_times(path), click_times(path)
    n = min(len(v), len(a))
    return [(a[i] - v[i]) * 1000.0 for i in range(n)], len(v), len(a)


def spacing_error_ms(out, src):
    """Worst difference between the frame-to-frame timing of the output and the source, in ms. Relative to the first
    frame: the final remux moves BOTH streams by the container's B-frame delay (-avoid_negative_ts make_zero), so an
    absolute origin is not a property of the file; the A/V relation (checked separately) is."""
    return max(abs((o - out[0]) - (s - src[0])) for o, s in zip(out, src)) * 1000.0


def decodes_cleanly(path):
    # framemd5 decodes every frame and keeps the frames' own timestamps: the null muxer's constant-rate timebase would
    # "collide" two frames of a VFR file that are less than one average frame apart and report a muxer error that is its own
    r = _run([FF, '-v', 'error', '-i', path, '-f', 'framemd5', '-'], text=True)
    return r.returncode == 0 and not r.stderr.strip(), r.stderr.strip()[:300]


def plain_frames(n, w=640, h=360):
    """Moving, face-free content (a gradient that rolls): every frame differs."""
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([(xx * 255 // w), (yy * 255 // h), ((xx + yy) * 255 // (w + h))], axis=-1).astype(np.uint8)
    return [np.roll(base, i * 3, axis=1) for i in range(n)]


# ═══ 1. VFR: detection, seek, timestamps, audio ═════════════════════════════════════════════════════════════════════════
class VfrDetection(unittest.TestCase):
    def test_r_frame_rate_not_equal_avg_frame_rate_is_vfr(self):
        pts = [Fraction(k, 30) for k in range(30)]
        vfr, why, _ = vfr_timing.classify(Fraction(30), Fraction(2700, 91), pts)
        self.assertTrue(vfr)
        self.assertIn('!=', why)

    def test_a_constant_rate_clip_is_not(self):
        pts = [Fraction(k * 1001, 30000) for k in range(60)]
        vfr, _, dev = vfr_timing.classify(Fraction(30000, 1001), Fraction(30000, 1001), pts)
        self.assertFalse(vfr)
        self.assertLess(dev, 1e-9)

    def test_rounding_to_a_millisecond_timebase_is_not_vfr(self):
        pts = [Fraction(round(k * 1000 / 30), 1000) for k in range(90)]       # mkv-style 33/34 ms steps
        self.assertFalse(vfr_timing.classify(Fraction(30), Fraction(30), pts)[0])

    def test_agreeing_rates_but_frames_off_the_grid_is_vfr(self):
        pts = [Fraction(k, 30) for k in range(60)]
        pts[30] += Fraction(5, 1000)                                            # one frame 5 ms late
        self.assertTrue(vfr_timing.classify(Fraction(30), Fraction(30), pts)[0])

    def test_probing_a_real_vfr_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'v.mp4')
            make_clip(path, plain_frames(120), vfr=True, audio=False)
            t = vfr_timing.probe_source_timing(path)
            self.assertIsNotNone(t)
            self.assertTrue(t.is_vfr, t.reason)
            self.assertEqual(t.frames, 120)
            cfr = os.path.join(d, 'c.mp4')
            make_clip(cfr, plain_frames(120), vfr=False, audio=False)
            self.assertFalse(vfr_timing.probe_source_timing(cfr).is_vfr)


class ReaderSeekOnVfr(unittest.TestCase):
    """The main-pass readers aimed `(frame - 0.5) / average_fps`; on a VFR clip frame N is not at N / fps."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix='signoff_seek_')
        cls.path = os.path.join(cls.tmp, 'v.mp4')
        make_clip(cls.path, plain_frames(240), vfr=True, audio=False)
        t = vfr_timing.probe_source_timing(cls.path)
        cls.fps = float(t.avg_frame_rate)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _first_id(self, start, reader):
        import cv2
        if reader == 'ffmpeg':
            from roop import nvdec_reader
            cap = nvdec_reader.FFmpegVideoReader(self.path, 640, 360, self.fps, hwaccel=None, pix_fmt='bgr24', prefetch_depth=0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            ok, fr = cap.read()
            cap.release()
        else:
            from roop.video_stream import NVHardwareVideoReader
            cap = NVHardwareVideoReader(self.path, 640, 360, fps=self.fps, start_frame=start)
            ok, fr = cap.read()
            cap.close() if hasattr(cap, 'close') else cap.release()
        self.assertTrue(ok)
        raw = np.ascontiguousarray(fr[0:BLOCK, 0:BITS * BLOCK])
        gray = raw.mean(axis=2).reshape(BLOCK, BITS, BLOCK).mean(axis=(0, 2))
        return int(sum(1 << b for b in range(BITS) if gray[b] > 127))

    def test_the_ffmpeg_pipe_reader_starts_on_the_requested_frame(self):
        for start in (1, 10, 66, 67, 68, 100, 150, 200, 239):
            with self.subTest(start=start):
                self.assertEqual(self._first_id(start, 'ffmpeg'), start)

    def test_the_hardware_stream_reader_starts_on_the_requested_frame(self):
        for start in (67, 100, 200):
            with self.subTest(start=start):
                try:
                    got = self._first_id(start, 'hw')
                except (OSError, RuntimeError) as exc:
                    self.skipTest(f'UNVERIFIED - NVDEC reader unavailable here: {exc!r}')
                self.assertEqual(got, start)

    def test_a_constant_rate_clip_keeps_the_established_seek(self):
        with tempfile.TemporaryDirectory() as d:
            cfr = os.path.join(d, 'c.mp4')
            make_clip(cfr, plain_frames(120), vfr=False, audio=False)
            self.assertIsNone(vfr_timing.seek_time(cfr, 67))


class Mp4Retime(unittest.TestCase):
    """The container rewrite, against ffmpeg as the oracle: pixels untouched, timestamps as asked."""

    def _clip(self, d, bframes, faststart):
        path = os.path.join(d, 'in.mp4')
        frames = plain_frames(60, 320, 180)
        cmd = [FF, '-hide_banner', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '320x180', '-r', '30', '-i', '-',
               '-c:v', 'libx264', '-bf', str(bframes), '-g', '30', '-pix_fmt', 'yuv420p'] + (['-movflags', '+faststart'] if faststart else []) + [path]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for fr in frames:
            p.stdin.write(fr.tobytes())
        p.stdin.close()
        self.assertEqual(p.wait(), 0)
        return path

    def _framemd5(self, path):
        out = _run([FF, '-v', 'error', '-i', path, '-f', 'framemd5', '-'], text=True).stdout
        return [l.split(',')[-1].strip() for l in out.splitlines() if l and not l.startswith('#')]

    def _targets(self, n, first=0.0):
        return [Fraction(int(round((k / 30 + 0.02 * (k // 7) + 0.013 * math.sin(k / 3)) * 90000)), 90000) + Fraction(first) for k in range(n)]

    def _check(self, bframes, faststart, first=0.0):
        with tempfile.TemporaryDirectory() as d:
            src = self._clip(d, bframes, faststart)
            dst = os.path.join(d, 'out.mp4')
            n = mp4_timing.read_track(src)[2].samples
            targets = self._targets(n, first)
            rep = mp4_timing.retime(src, dst, targets)
            self.assertEqual(rep['bframes'], bframes > 0)
            self.assertEqual(self._framemd5(src), self._framemd5(dst), 'a coded byte or the frame order changed')
            ok, why = decodes_cleanly(dst)
            self.assertTrue(ok, why)
            got = pts_of(dst)
            worst = max(abs(g - float(t)) for g, t in zip(got, targets)) * 1000
            # a late first frame is an EMPTY EDIT, written in the movie timescale (mvhd: 1000 Hz): half a millisecond of resolution
            self.assertLess(worst, 0.5 if first else 0.002, f'ffprobe sees {worst:.5f} ms from the requested timestamps')
            mine = mp4_timing.presentation_seconds(dst)
            self.assertLess(max(abs(float(m - t)) for m, t in zip(mine, targets)) * 1000, 0.5 if first else 1e-3)

    def test_with_b_frames_and_the_faststart_layout(self):
        self._check(bframes=2, faststart=True)

    def test_with_b_frames_and_moov_at_the_end(self):
        self._check(bframes=2, faststart=False)

    def test_without_b_frames(self):
        self._check(bframes=0, faststart=True)

    def test_a_first_frame_that_starts_late_becomes_an_empty_edit(self):
        self._check(bframes=2, faststart=True, first=0.0667)

    def test_wrong_target_count_or_order_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            src = self._clip(d, 2, True)
            n = mp4_timing.read_track(src)[2].samples
            with self.assertRaises(mp4_timing.Mp4Error):
                mp4_timing.retime(src, os.path.join(d, 'o.mp4'), [Fraction(k, 30) for k in range(n - 1)])
            with self.assertRaises(mp4_timing.Mp4Error):
                mp4_timing.retime(src, os.path.join(d, 'o.mp4'), [Fraction(n - k, 30) for k in range(n)])
            self.assertFalse(os.path.exists(os.path.join(d, 'o.mp4')))


def _pipeline_like(src, start, end, workdir, w, h, fps):
    """Reader -> FFMPEG_VideoWriter -> finalize_render: the live path's I/O and finalisation, minus the models."""
    import cv2
    from roop import finalize_output, nvdec_reader
    from roop.ffmpeg_writer import FFMPEG_VideoWriter
    cap = nvdec_reader.FFmpegVideoReader(src, w, h, fps, hwaccel=None, pix_fmt='bgr24', prefetch_depth=0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    temp = os.path.join(workdir, 'inter__temp.mp4')
    wr = FFMPEG_VideoWriter(temp, (w, h), fps, codec='libx264', crf=14, audiofile=None)
    for _ in range(end - start):
        ok, fr = cap.read()
        assert ok
        wr.write_frame(fr)
    wr.close()
    cap.release()
    final = os.path.join(workdir, 'final.mp4')
    res = finalize_output.finalize_render(temp, final, audio_source=src, trim_start=start if start else None, trim_end=end if start else None,
                                          expected_frames=end - start, video_source=src)
    return final, res


class SyncWithoutTheModels(unittest.TestCase):
    """reader -> writer -> finalize on a clip with flashes and clicks. Fast: no models."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix='signoff_sync_')
        cls.w, cls.h, cls.n = 640, 360, 240
        cls.vfr = os.path.join(cls.tmp, 'vfr.mp4')
        cls.cfr = os.path.join(cls.tmp, 'cfr.mp4')
        make_clip(cls.vfr, plain_frames(cls.n, cls.w, cls.h), vfr=True)
        make_clip(cls.cfr, plain_frames(cls.n, cls.w, cls.h), vfr=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _render(self, src, start, end, tag):
        d = os.path.join(self.tmp, tag)
        os.makedirs(d, exist_ok=True)
        t = vfr_timing.probe_source_timing(src)
        final, res = _pipeline_like(src, start, end, d, self.w, self.h, float(t.avg_frame_rate))
        self.assertTrue(res['ok'], res['problems'])
        return final, res

    def _assert_in_sync(self, path, expect_flashes):
        offs, nv, na = av_offsets_ms(path)
        self.assertEqual((nv, na), (expect_flashes, expect_flashes), 'a flash or a click was lost')
        self.assertLess(max(abs(o) for o in offs), SYNC_TOLERANCE_MS, f'click - flash per event (ms): {[round(o, 3) for o in offs]}')

    def test_vfr_untrimmed_every_frame_is_at_its_source_timestamp_and_audio_is_in_sync(self):
        final, res = self._render(self.vfr, 0, self.n, 'vfr_full')
        self.assertTrue(res['vfr']['retimed'], res['vfr'])
        src, out = pts_of(self.vfr), pts_of(final)
        self.assertEqual(len(out), self.n)
        self.assertLess(spacing_error_ms(out, src), 0.05, 'a frame is off its source timestamp')
        self.assertEqual([i for i in ids_of(final) if i != FLASH_ID], [i for i in range(self.n) if i % 30 != 15])
        self._assert_in_sync(final, 8)

    def test_vfr_trimmed_starts_on_the_requested_frame_and_is_in_sync(self):
        start, end = 67, 203
        final, res = self._render(self.vfr, start, end, 'vfr_trim')
        ids = ids_of(final)
        self.assertEqual(ids[0], start, 'the trimmed render did not start on the requested frame')
        self.assertEqual([i for i in ids if i != FLASH_ID], [i for i in range(start, end) if i % 30 != 15])
        src = pts_of(self.vfr)
        out = pts_of(final)
        self.assertLess(spacing_error_ms(out, src[start:end]), 0.05)
        self._assert_in_sync(final, 5)

    def test_cfr_trimmed_audio_is_cut_to_the_sample_not_to_the_packet(self):
        """Before: the packet-aligned cut put the audio early by the gap to the next AAC packet (19.0 ms measured)."""
        final, res = self._render(self.cfr, 67, 203, 'cfr_trim')
        self.assertEqual(ids_of(final)[0], 67)
        self.assertFalse(res['vfr']['retimed'])
        self._assert_in_sync(final, 5)

    def test_cfr_untrimmed_is_in_sync(self):
        final, _ = self._render(self.cfr, 0, self.n, 'cfr_full')
        self._assert_in_sync(final, 8)

    def test_a_stopped_vfr_render_keeps_its_timestamps(self):
        """A partial render (what a cancel leaves): 61 of the first 100 frames, retimed from the SAME source timestamps."""
        final, res = self._render(self.vfr, 0, 61, 'vfr_partial')
        src, out = pts_of(self.vfr)[:61], pts_of(final)
        self.assertEqual(len(out), 61)
        self.assertLess(spacing_error_ms(out, src), 0.05)

    def test_the_finalize_gate_accepts_the_edit_listed_audio(self):
        """ffprobe's audio duration is the media length and ignores the edit list that cuts it; the gate reads the edits."""
        final, res = self._render(self.cfr, 67, 203, 'gate')
        probe = __import__('roop.finalize_output', fromlist=['x']).probe_container(final)
        video_len = probe['video']['duration_s']
        self.assertLess(abs(probe['audio']['duration_s'] - video_len), 1.0 / 30)
        self.assertLess(abs(probe['audio']['start_s'] - probe['video']['start_s']), 1.0 / 30)


# ═══ 2. Colour: the plan, and the real render path ══════════════════════════════════════════════════════════════════════
class ColorPlan(unittest.TestCase):
    def _plan(self, space='', rng='', prim='', trc='', height=1080):
        return source_color.plan_for({'color': {'color_space': space, 'color_range': rng, 'color_primaries': prim, 'color_transfer': trc}}, height)

    def test_bt709_limited_range_keeps_the_established_path(self):
        self.assertIsNone(self._plan('bt709', 'tv', 'bt709', 'bt709'))
        self.assertIsNone(self._plan('bt709', 'tv'))

    def test_an_untagged_limited_range_source_keeps_it_too(self):
        self.assertIsNone(self._plan('', ''))
        self.assertIsNone(self._plan('unknown', 'unknown', 'unknown', 'unknown'))

    def test_hdr_and_wide_gamut_are_left_to_the_hdr_pipeline(self):
        self.assertIsNone(self._plan('bt2020nc', 'tv', 'bt2020', 'smpte2084'))
        self.assertIsNone(self._plan('bt2020nc', 'tv', 'bt2020', 'arib-std-b67'))

    def test_bt601_is_reproduced_with_the_sources_own_tags(self):
        p = self._plan('smpte170m', 'tv', 'smpte170m', 'smpte170m', 480)
        self.assertEqual((p.matrix, p.scale_matrix, p.range, p.primaries, p.transfer), ('smpte170m', 'bt601', 'tv', 'smpte170m', 'smpte170m'))
        self.assertEqual(self._plan('bt470bg', 'tv', 'bt470bg', 'unknown', 576).primaries, 'bt470bg')

    def test_full_range_is_reproduced_even_when_the_matrix_is_709(self):
        p = self._plan('bt709', 'pc', 'bt709', 'bt709')
        self.assertEqual((p.range, p.scale_matrix), ('pc', 'bt709'))

    def test_untagged_full_range_gets_the_matrix_players_assume(self):
        self.assertEqual(self._plan('', 'pc', height=1080).scale_matrix, 'bt709')
        self.assertEqual(self._plan('', 'pc', height=480).scale_matrix, 'bt601')

    def test_the_filter_sets_the_frame_properties_the_encoder_writes(self):
        """FFmpeg 8.1 drops -color_primaries/-color_trc output options; only frame properties reach the encoder."""
        f = self._plan('smpte170m', 'pc', 'smpte170m', 'smpte170m').video_filter()
        self.assertIn('out_color_matrix=bt601', f)
        self.assertIn('out_range=pc', f)
        self.assertIn('setparams=colorspace=smpte170m:color_primaries=smpte170m:color_trc=smpte170m:range=pc', f)

    def test_the_live_writers_take_the_plan_and_the_default_argv_is_unchanged(self):
        import inspect
        from roop.ffmpeg_writer import FFMPEG_VideoWriter
        from roop.video_stream import NVHardwareVideoWriter
        self.assertIn('color', inspect.signature(FFMPEG_VideoWriter.__init__).parameters)
        self.assertIn('color', inspect.signature(NVHardwareVideoWriter.__init__).parameters)
        plan = self._plan('smpte170m', 'tv', 'smpte170m', 'smpte170m')
        with tempfile.TemporaryDirectory() as d:
            for color, expect in ((None, 'colorspace=bt709:iall=bt601-6-625:fast=1'), (plan, 'setparams=colorspace=smpte170m')):
                wr = NVHardwareVideoWriter(os.path.join(d, 'o.mp4'), 64, 64, 30.0, codec='libx264', color=color)
                cmd = wr._command('libx264')
                wr.proc.stdin.close()
                wr.proc.wait()
                self.assertIn(expect, cmd[cmd.index('-vf') + 1])
                if color is None:
                    i = cmd.index('-colorspace')
                    self.assertEqual(cmd[i:i + 8:2], ['-colorspace', '-color_primaries', '-color_trc', '-color_range'])
                    self.assertEqual(cmd[i + 1:i + 8:2], ['bt709', 'bt709', 'bt709', 'tv'])


# ═══ 3. Cooperative cancel ══════════════════════════════════════════════════════════════════════════════════════════════
ACCESS_VIOLATIONS = {3221225477, -1073741819}          # 0xC0000005 as an unsigned / signed exit code


def _scenario(workdir, stage, stop_after, clip, frames=600, timeout=1500):
    cmd = [sys.executable, os.path.join(HERE, '_cancel_scenario.py'), workdir, stage, str(stop_after), clip, str(frames)]
    env = dict(os.environ, ROOP_STAB_OVERLAP='0')
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=os.path.dirname(APP))
    line = next((l for l in p.stdout.splitlines() if l.startswith('RESULT ')), None)
    return p.returncode, (json.loads(line[7:]) if line else None), (p.stdout + p.stderr)[-1500:]


@pytest.mark.gpu
class CooperativeCancel(unittest.TestCase):
    """POST /api/stop and /api/queue/cancel both end in `roop.globals.processing = False` (api.stop_swap). Each scenario does
    exactly that to a real render running in its own process, so a Windows access violation on teardown is a visible exit code."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch
            if not torch.cuda.is_available():
                raise unittest.SkipTest('UNVERIFIED - no CUDA device here')
        except ImportError:
            raise unittest.SkipTest('UNVERIFIED - torch not installed')
        cls.tmp = tempfile.mkdtemp(prefix='signoff_cancel_')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _ok_exit(self, rc, tail):
        self.assertNotIn(rc, ACCESS_VIOLATIONS, f'access violation on teardown\n{tail}')
        self.assertEqual(rc, 0, tail)

    def test_stop_during_stage_3_leaves_a_playable_file_with_every_retired_frame_and_a_clean_exit(self):
        rc, res, tail = _scenario(os.path.join(self.tmp, 'plain'), '3', 200, 'plain', frames=900)
        self._ok_exit(rc, tail)
        self.assertIsNotNone(res, tail)
        self.assertTrue(res['asked'], 'the stop was never requested (Stage 3 never retired a frame)')
        self.assertFalse(res['alive_after_join'], 'the render thread did not stop')
        self.assertIsNone(res['error'])
        fin = res['finalize']
        self.assertTrue(fin['ok'], fin['problems'])
        self.assertTrue(fin['stopped'])
        out = res['output']
        self.assertTrue(os.path.isfile(out))
        ok, why = decodes_cleanly(out)
        self.assertTrue(ok, 'the partial file is not playable: ' + why)         # the encoder's trailer was written
        info = fin['verify']
        self.assertGreaterEqual(info['frames'], 200)
        self.assertLess(info['frames'], 900, 'the render ran to the end: nothing was cancelled')
        self.assertEqual(info['frames'], res['retired'], 'a frame the writer accepted is missing from the file, or one it did not is in it')
        self.assertTrue(info['audio_present'])
        self.assertEqual(fin['orphans'], [])
        self.assertLess(res['seconds_to_stop'], 60.0)
        self.assertLessEqual(res['cuda_allocated_mb'], 16.0, 'GPU memory still held after the stop')
        self.assertNotIn('release_error', res)
        leftovers = [n for n in os.listdir(os.path.dirname(out)) if n.endswith(('.part', '.mka', '.seg.mp4', '__temp.mp4'))]
        self.assertEqual(leftovers, [])

    def test_stop_during_the_pre_pass_returns_cleanly(self):
        rc, res, tail = _scenario(os.path.join(self.tmp, 'stage2'), '2', 30, 'faces')
        self._ok_exit(rc, tail)
        self.assertTrue(res['asked'], 'the pre-pass never reached 30%')
        self.assertFalse(res['alive_after_join'])
        self.assertIsNone(res['error'])
        self.assertLess(res['seconds_to_stop'], 60.0)
        self.assertLessEqual(res['cuda_allocated_mb'], 16.0)

    def test_stop_with_swap_batches_in_flight_exits_without_an_access_violation(self):
        """Stage 3 running its GPU workers (TensorRT batches in flight, whether or not a frame was written yet)."""
        rc, res, tail = _scenario(os.path.join(self.tmp, 'inflight'), 'inflight', 15, 'faces')
        self._ok_exit(rc, tail)
        self.assertTrue(res['asked'], 'Stage 3 never ran for 15 s')
        self.assertFalse(res['alive_after_join'], 'the render thread did not stop')
        self.assertIsNone(res['error'])
        self.assertLessEqual(res['cuda_allocated_mb'], 16.0)
        if res.get('output'):
            ok, why = decodes_cleanly(res['output'])
            self.assertTrue(ok, why)


# ═══ 4. The real render path (after the cancel scenarios: they need the GPU to themselves) ═══════════════════════════════
COLOR_ARMS = {
    # name: (swscale matrix, container matrix tag, range, primaries/transfer tag)
    'bt709_tv': ('bt709', 'bt709', 'tv', 'bt709'),
    'bt601_tv': ('bt601', 'smpte170m', 'tv', 'smpte170m'),
    'bt709_pc': ('bt709', 'bt709', 'pc', 'bt709'),
    'bt601_pc': ('bt601', 'smpte170m', 'pc', 'smpte170m'),
}
COLOR_KEYS = ('pix_fmt', 'color_space', 'color_primaries', 'color_transfer', 'color_range')


def _stream_tags(path):
    out = _run([FP, '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=' + ','.join(COLOR_KEYS), '-of', 'default=nw=1', path],
               text=True).stdout
    return dict(l.split('=', 1) for l in out.split())


def _encode_color_source(path, arm, frames, w, h):
    mat, tag, rng, prim = COLOR_ARMS[arm]
    pix = 'yuvj420p' if rng == 'pc' else 'yuv420p'
    vf = (f'scale=out_color_matrix={mat}:out_range={rng},setparams=colorspace={tag}:color_primaries={prim}:color_trc={prim}:range={rng},'
          f'format={pix}')
    p = subprocess.Popen([FF, '-hide_banner', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', '30',
                          '-i', '-', '-vf', vf, '-c:v', 'libx264', '-crf', '12', '-preset', 'veryfast', '-g', '30', path], stdin=subprocess.PIPE)
    for fr in frames:
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
    p.stdin.close()
    assert p.wait() == 0


def _planes(path, w, h, frame=3):
    """The file's own Y, U, V values (no conversion) of one frame."""
    pix = _stream_tags(path)['pix_fmt']
    raw = _run([FF, '-v', 'error', '-i', path, '-vf', f'select=eq(n\\,{frame})', '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', pix, '-']).stdout
    b = np.frombuffer(raw, np.uint8)
    q = (w // 2) * (h // 2)
    return (b[:w * h].reshape(h, w).astype(np.int16), b[w * h:w * h + q].reshape(h // 2, w // 2).astype(np.int16),
            b[w * h + q:w * h + 2 * q].reshape(h // 2, w // 2).astype(np.int16))


def _chart(w, h):
    from measure_color_roundtrip import chart
    return chart(w, h)


@pytest.mark.gpu
class RealRenderPath(unittest.TestCase):
    """The same checks through `compare_enhancers_video.render`: the real reader selection, ProcessMgr, writer and finalisation."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch
            if not torch.cuda.is_available():
                raise unittest.SkipTest('UNVERIFIED - no CUDA device here')
            import test_pipeline_3stage_sync_and_fps as E
        except unittest.SkipTest:
            raise
        except Exception as exc:
            raise unittest.SkipTest(f'UNVERIFIED - the GPU stack is not importable: {exc!r}')
        cls.tmp = tempfile.mkdtemp(prefix='signoff_real_')
        cls.rig = E.Rig(cls.tmp)
        try:
            cls.rig.build()
        except BaseException:
            cls.rig.close()
            raise
        base = cls.rig.shots()[0]
        cls.vfr = os.path.join(cls.tmp, 'sync_vfr.mp4')
        make_clip(cls.vfr, [np.roll(base, i * 3, axis=1) for i in range(240)], vfr=True)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, 'rig', None) is not None:
            cls.rig.close()
        if os.environ.get('ROOP_E2E_KEEP') != '1':
            shutil.rmtree(cls.tmp, ignore_errors=True)

    def _sync(self, path, flashes):
        offs, nv, na = av_offsets_ms(path)
        self.assertEqual((nv, na), (flashes, flashes), 'a flash or a click was lost')
        self.assertLess(max(abs(o) for o in offs), SYNC_TOLERANCE_MS, f'click - flash per event (ms): {[round(o, 3) for o in offs]}')

    def test_vfr_source_is_detected_in_stage_1_and_keeps_every_frames_timestamp(self):
        import roop.globals as g
        run = self.rig.render('vfr_full', self.vfr, rng=(0, 0), total=240)
        self.assertIsNone(run['error'])
        fin = run['finalize'][-1]
        self.assertTrue(fin['ok'], fin['problems'])
        self.assertTrue(fin['vfr']['vfr'])
        self.assertTrue(fin['vfr']['retimed'], fin['vfr'])
        self.assertTrue((getattr(g, 'source_timing', None) or {}).get('vfr'), 'Stage 1 did not record the VFR source')
        out = run['output']
        self.assertEqual(len(pts_of(out)), 240)
        self.assertLess(spacing_error_ms(pts_of(out), pts_of(self.vfr)), 0.05)
        self.assertEqual([i for i in ids_of(out) if i != FLASH_ID], [i for i in range(240) if i % 30 != 15])
        self._sync(out, 8)

    def test_a_trimmed_vfr_render_starts_on_the_requested_frame_and_stays_in_sync(self):
        run = self.rig.render('vfr_trim', self.vfr, rng=(67, 203), total=240)
        self.assertIsNone(run['error'])
        self.assertTrue(run['finalize'][-1]['ok'], run['finalize'][-1]['problems'])
        out = run['output']
        ids = ids_of(out)
        self.assertEqual(ids[0], 67, 'a trimmed VFR render must start on the requested frame, not the next')
        self.assertEqual([i for i in ids if i != FLASH_ID], [i for i in range(67, 203) if i % 30 != 15])
        self._sync(out, 5)

    def test_the_sources_colour_description_survives_the_render(self):
        w, h, n = 1280, 720, 30
        frames = [np.roll(_chart(w, h), i * 2, axis=1) for i in range(n)]
        for arm in COLOR_ARMS:
            with self.subTest(arm=arm):
                src = os.path.join(self.tmp, f'src_{arm}.mp4')
                _encode_color_source(src, arm, frames, w, h)
                run = self.rig.render(f'color_{arm}', src, rng=(0, 0), total=n)
                self.assertIsNone(run['error'])
                out = run['output']
                self.assertEqual(_stream_tags(out), _stream_tags(src), 'matrix / primaries / transfer / range / pix_fmt tags changed')
                ys, us, vs = _planes(src, w, h)
                yo, uo, vo = _planes(out, w, h)
                mean = lambda a, b: float(np.abs(a - b).mean())               # noqa: E731
                # the file's own planes: before the fix bt601 / full range read Y 8.5-11.2, U 3.8-6.3, V 2.6-5.6 (bt709 limited: 0.94 / 0.22 / 0.16)
                self.assertLess(mean(ys, yo), 1.5, 'luma moved')
                self.assertLess(mean(us, uo), 0.6, 'Cb moved')
                self.assertLess(mean(vs, vo), 0.4, 'Cr moved')
                if COLOR_ARMS[arm][2] == 'pc':
                    self.assertGreaterEqual(int(yo.max()), 250, 'full range was squeezed into limited range')
                    self.assertLessEqual(int(yo.min()), 5)
                else:
                    self.assertGreaterEqual(int(yo.min()), 14)
                    self.assertLessEqual(int(yo.max()), 236)


# ═══ 5. Installer ═══════════════════════════════════════════════════════════════════════════════════════════════════════
class InstallerSignoff(unittest.TestCase):
    REQ = os.path.join(APP, 'requirements.txt')

    def _requirements(self):
        from packaging.requirements import Requirement
        out = {}
        for line in open(self.REQ, encoding='utf-8'):
            line = line.split('#')[0].strip()
            if line:
                r = Requirement(line)
                out[r.name.lower().replace('_', '-')] = r
        return out

    def test_the_gpu_extras_are_declared_with_platform_markers_that_exclude_mac_and_arm(self):
        reqs = self._requirements()
        envs = {
            'win-x64': {'sys_platform': 'win32', 'platform_machine': 'AMD64'},
            'linux-x64': {'sys_platform': 'linux', 'platform_machine': 'x86_64'},
            'mac-arm': {'sys_platform': 'darwin', 'platform_machine': 'arm64'},
            'win-arm': {'sys_platform': 'win32', 'platform_machine': 'ARM64'},
        }
        expect = {'pynvvideocodec': (True, True, False, False), 'cuda-python': (True, True, False, False),
                  'triton-windows': (True, False, False, False)}
        for name, want in expect.items():
            with self.subTest(package=name):
                self.assertIn(name, reqs, f'{name} is used by the render path but is not in requirements.txt')
                self.assertIsNotNone(reqs[name].marker, f'{name} has no platform marker: it would break the install on Mac / ARM')
                got = tuple(reqs[name].marker.evaluate(env) for env in envs.values())
                self.assertEqual(got, want, dict(zip(envs, got)))

    def test_the_optional_gpu_packages_are_never_imported_at_module_top_level(self):
        """Missing / unsupported hardware must degrade, not fail at import: each is imported inside a function or a try."""
        optional = {'PyNvVideoCodec', 'triton', 'cuda', 'tensorrt', 'cvcuda', 'kornia', 'av', 'modelopt', 'openvino', 'nncf'}
        import ast
        offenders = []
        for sub in ('roop', 'core'):
            for root, dirs, files in os.walk(os.path.join(APP, sub)):
                dirs[:] = [d for d in dirs if d != '__pycache__']
                for f in files:
                    if f.endswith('.py'):
                        path = os.path.join(root, f)
                        tree = ast.parse(open(path, encoding='utf-8', errors='replace').read())
                        for node in tree.body:
                            names = ([a.name.split('.')[0] for a in node.names] if isinstance(node, ast.Import)
                                     else [node.module.split('.')[0]] if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module else [])
                            offenders += [f'{os.path.relpath(path, APP)}: {n}' for n in names if n in optional]
        self.assertEqual(offenders, [])

    # third-party modules imported at module top level without a line of their own in requirements.txt, and why that is fine
    PROVISIONED_ELSEWHERE = {
        'torch': 'torch.js (Pinokio torch step) picks the wheel for the GPU',
        'torchvision': 'torch.js, with torch',
        'onnxruntime': 'provision_runtime.py / ort_package_detector.py choose onnxruntime-gpu vs -directml vs cpu',
        'starlette': 'a hard dependency of fastapi',
        'scipy': 'a hard dependency of insightface and librosa',
        'sklearn': 'a hard dependency of insightface and librosa (scikit-learn)',
        'yaml': 'a hard dependency of gradio (PyYAML)',
        'cv2': 'opencv-python-headless',
        'skimage': 'scikit-image',
    }

    def test_every_unguarded_third_party_import_is_declared(self):
        import ast
        stdlib = set(sys.stdlib_module_names)
        local = {os.path.splitext(d)[0] for d in os.listdir(APP)}
        reqs = set(self._requirements())
        alias = {'cv2': 'opencv-python-headless', 'skimage': 'scikit-image', 'yaml': 'pyyaml', 'sklearn': 'scikit-learn', 'pynvml': 'nvidia-ml-py'}
        undeclared = {}
        for sub in ('roop', 'core', ''):
            base = os.path.join(APP, sub) if sub else APP
            for root, dirs, files in os.walk(base):
                dirs[:] = [d for d in dirs if d not in ('env', '__pycache__', 'tests', 'models', 'node_modules', 'facesets')]
                if not sub and root != base:
                    continue
                for f in files:
                    if not f.endswith('.py'):
                        continue
                    tree = ast.parse(open(os.path.join(root, f), encoding='utf-8', errors='replace').read())
                    for node in tree.body:
                        names = ([a.name.split('.')[0] for a in node.names] if isinstance(node, ast.Import)
                                 else [node.module.split('.')[0]] if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module else [])
                        for n in names:
                            if n in stdlib or n in local or n in ('roop', 'core', 'tests', 'face_engine', '__future__'):
                                continue
                            if alias.get(n, n).lower().replace('_', '-') in reqs or n in self.PROVISIONED_ELSEWHERE:
                                continue
                            undeclared.setdefault(n, set()).add(os.path.relpath(os.path.join(root, f), APP))
        self.assertEqual({k: sorted(v) for k, v in undeclared.items()}, {},
                         'imported at module top level but in neither requirements.txt nor PROVISIONED_ELSEWHERE')

    def test_the_new_modules_need_no_package(self):
        import ast
        stdlib = set(sys.stdlib_module_names)
        for name in ('vfr_timing.py', 'mp4_timing.py', 'source_color.py'):
            with self.subTest(module=name):
                tree = ast.parse(open(os.path.join(APP, 'roop', name), encoding='utf-8').read())
                third = set()
                for node in ast.walk(tree):
                    names = ([a.name.split('.')[0] for a in node.names] if isinstance(node, ast.Import)
                             else [node.module.split('.')[0]] if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module else [])
                    third |= {n for n in names if n not in stdlib and n not in ('roop', '__future__')}
                self.assertEqual(third, set(), 'a new third-party dependency would need a requirements.txt line and an install.js step')

    def test_install_js_installs_the_requirements_and_declares_the_ai_bundle(self):
        js = open(os.path.join(ROOT, 'install.js'), encoding='utf-8').read()
        self.assertIn('uv pip install -r requirements.txt', js)
        self.assertRegex(js, r'bundle:\s*"ai"')
        self.assertIn('torch.js', js)
