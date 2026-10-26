"""End-to-end proof that the 3-stage pipeline is frame-synchronized, reports honest numbers and leaves no artifacts.

Real renders through the production path (`core.batch_process_with_options`, the user's live config.yaml models,
`sparse_auto_range` ON), reusing `verify_three_stage_pipeline.py`'s pipeline bring-up. Needs a CUDA device, the
compiled engines, ffmpeg and the repo's `angle.mp4`; without them the class SKIPS with the reason (UNVERIFIED, not
green). One render at a time: each is cached on the class and shared by the tests that read it.

    Test 1  a 180-frame 1080p clip with hard cuts at 60 and 120, trimmed 10..165 (155 frames), K=4, B<=8:
            Stage 1 finds exactly the cuts, Stage 2's trajectory cache spans exactly 155 frames, Stage 3's muxed
            file decodes to exactly 155 frames, the audio matches the video, and nothing is left behind.
    Test 2  a 2-slice Batch Matrix render: both segments sit under `.segments/<group>/`, neither overwrites the other,
            the join has the clip's frame count and its audio duration.
    Test 3  what the 14-section runtime report says is ACTIVE. Where the brief names an acceleration the render path
            does not have, the test states the requirement and is `xfail(strict=True)` with the reason: it cannot
            pass by accident, and it FAILS the day the feature is wired, so the pin gets updated rather than rotting.
    Test 4  a warm-up render, then a timed 200-frame render bracketed by `torch.cuda.Event`s and ONE full drain
            (no per-batch synchronize), printed as a table.

Names in the brief that do not exist, and what is used instead:
  * `.segments/{job_id}/` is `.segments/{group}/` (`roop/segment_output.py`); the job id is in the file name.
  * "TRT Graph ms" is the TensorRT swap-model call (ORT's CUDA graph is OFF by default); "Fused Post-Process ms" is
    the cv2 paste-back (`blend`); "Warp ms" is the cv2 alignment. The table says so under itself.
  * the runtime canaries record cosine / SSIM / IoU / pixel error, not PSNR; the 48 dB figure is the offline surgeon's
    gate constant. `PRECISION.verification_gates.nan_inf_free` is a hard-coded literal, so NaN-freeness is checked on
    the canaries' own metrics instead.
"""
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
for _p in (APP, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fixtures                                              # noqa: E402  (module level: test_fixture_paths requires it)

FPS, W, H = 30, 1920, 1080
N_FRAMES = 180
CUTS = (60, 120)                       # the first frame of the 2nd and 3rd shot
TRIM = (10, 165)                       # exclusive end -> 155 frames, not divisible by B=8 or by K=4
TRIM_FRAMES = TRIM[1] - TRIM[0]
BENCH_WARMUP, BENCH_FRAMES = 24, 200   # 3 batches of 8, then the timed 200
GROUP = 'e2e-3stage-sync'
# ROOP_STAB_OVERLAP=0: with two stabilization chunks in flight (the default) the first 155-frame render of a process drove
# this 12 GB card to 100% VRAM and the driver spilled into shared memory: swap batches took 100-550 s and the render
# 77-97 min (runs 3-6, and three cold probes that were stopped at 9 min). One chunk in flight: the same render, cold, in
# 89 s with no slow batch. The default is the owner's call (RECODE_STATUS); the test measures the pipeline, not the edge.
ENV = {'ROOP_PREPASS_DETECT_INTERVAL': '4', 'ROOP_BATCH_SWAP': '1', 'ROOP_BATCH_SWAP_MAX': '8', 'ROOP_STAB_OVERLAP': '0'}
ORPHAN_MARKERS = ('__temp', '.seg', '__audio_cut', '.resume.json')


def _ffprobe_json(path, *args):
    from roop.ffmpeg_path import ffprobe_binary
    out = subprocess.run([ffprobe_binary(), '-v', 'error', *args, '-of', 'json', path],
                         capture_output=True, text=True)
    return out.returncode, out.stderr.strip(), (json.loads(out.stdout or '{}'))


def probe_video(path):
    """Decoded frame count (not packets: a dropped tail can hide in either), duration, plus audio duration."""
    rc, err, v = _ffprobe_json(path, '-select_streams', 'v:0', '-count_frames', '-count_packets', '-show_entries',
                               'stream=nb_read_frames,nb_read_packets,duration,width,height,codec_name')
    s = (v.get('streams') or [{}])[0]
    _, _, a = _ffprobe_json(path, '-select_streams', 'a:0', '-show_entries', 'stream=duration')
    sa = (a.get('streams') or [{}])[0]
    return {'rc': rc, 'stderr': err, 'frames': int(s.get('nb_read_frames') or 0),
            'packets': int(s.get('nb_read_packets') or 0), 'video_s': float(s.get('duration') or 0.0),
            'audio_s': float(sa['duration']) if sa.get('duration') else None,
            'width': s.get('width'), 'height': s.get('height')}


def _paste(frame, im, alpha, cx, cy):
    size = im.shape[0]
    x0, y0 = int(round(cx - size / 2)), int(round(cy - size / 2))
    xs, ys, xe, ye = max(0, x0), max(0, y0), min(W, x0 + size), min(H, y0 + size)
    if xe <= xs or ye <= ys:
        return
    a = alpha[ys - y0:ye - y0, xs - x0:xe - x0, None]
    sub = frame[ys:ye, xs:xe].astype(np.float32)
    frame[ys:ye, xs:xe] = (a * im[ys - y0:ye - y0, xs - x0:xe - x0].astype(np.float32) + (1 - a) * sub).astype(np.uint8)


class Rig:
    """The pipeline, the clips and the renders. Built once; a render is run on demand and remembered."""

    def __init__(self, workdir):
        import cv2
        import fixtures
        import verify_three_stage_pipeline as V
        self.cv2, self.V, self.fixtures = cv2, V, fixtures
        self.workdir = workdir
        self.out_root = os.path.join(workdir, 'out')
        self.runs = {}
        self.captured = []
        self._restore = []

    # ── clips ──────────────────────────────────────────────────────────────
    def shots(self):
        """Three luma-distinct, face-free backgrounds: a hard cut between any two is unmistakable."""
        base = self.cv2.resize(self.V.landscape(), (W, H), interpolation=self.cv2.INTER_CUBIC)
        a = base
        b = 255 - self.cv2.flip(base, 1)                                        # mirrored and luma-inverted
        c = (base.astype(np.float32) * 0.25).astype(np.uint8)                    # very dark
        # luma deltas on the detector's 64x36 pooling: A->B 59.5, B->C 86.0 (threshold 28), so no cut is marginal
        return a, b, c

    def frames(self, n, cuts=()):
        shots = self.shots()
        sprites = self._sprites
        out = []
        for i in range(n):
            f = shots[sum(1 for c in cuts if i >= c)].copy()
            src = self.V.PROFILE_SRC[1] + abs((i // 2) % 36 - 18)               # frontal hold: 360..378
            src = min(src, self.V.SRC_LAST)
            im, al = sprites[src]
            _paste(f, im, al, W / 2 + 140 * math.sin(i / 17.0), H / 2 + 30 * math.sin(i / 11.0))
            out.append(f)
        return out

    def write_clip(self, frames, path, audio=True):
        """Ordinary 4:2:0 H.264 (what NVDEC decodes), not the harness's lossless 4:4:4."""
        cmd = [self.V._ffmpeg(), '-hide_banner', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
               '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-']
        if audio:
            cmd += ['-f', 'lavfi', '-i', f'sine=frequency=440:duration={len(frames) / FPS:.3f}:sample_rate=44100']
        cmd += ['-c:v', 'libx264', '-crf', '14', '-pix_fmt', 'yuv420p', '-preset', 'veryfast', '-g', '45']
        if audio:
            cmd += ['-c:a', 'aac', '-shortest']
        p = subprocess.Popen(cmd + [path], stdin=subprocess.PIPE)
        for fr in frames:
            p.stdin.write(np.ascontiguousarray(fr).tobytes())
        p.stdin.close()
        if p.wait() != 0:
            raise RuntimeError('ffmpeg failed writing ' + path)
        return path

    # ── bring-up ───────────────────────────────────────────────────────────
    def build(self):
        V, cv2 = self.V, self.cv2
        angle = self.fixtures.clip('angle.mp4')
        if not os.path.isfile(angle):
            raise unittest.SkipTest(f'UNVERIFIED - real profile footage not found ({angle}); not substituting another clip')
        sprites = V.load_target_sprites(angle)
        # the harness sprites are 220 px on a 960x540 canvas; this clip is 1080p, so the head is drawn 2x
        self._sprites = {k: (cv2.resize(im, (440, 440), interpolation=cv2.INTER_CUBIC),
                             cv2.resize(al, (440, 440), interpolation=cv2.INTER_CUBIC))
                         for k, (im, al) in sprites.items()}
        self.clip = self.write_clip(self.frames(N_FRAMES, CUTS), os.path.join(self.workdir, 'e2e_cuts_180.mp4'))
        self.bench_clip = self.write_clip(self.frames(BENCH_WARMUP + BENCH_FRAMES),
                                          os.path.join(self.workdir, 'e2e_bench_224.mp4'))
        self.P = V.Probe()
        self.ctx = V.prepare_pipeline(types.SimpleNamespace(enhancer=None), self.workdir, None, sprites)
        self._hook_precompute()

    def _hook_precompute(self):
        """Capture Stage 1/2 facts at the moment the pre-pass hands over, restored in `close`."""
        from roop.ProcessMgr import ProcessMgr
        orig = ProcessMgr._precompute_temporal
        rig = self

        def wrapped(mgr, *a, **k):
            r = orig(mgr, *a, **k)
            traj = getattr(mgr, '_trajectory', None)
            frames = sorted({int(x) for x in traj.rows['frame']}) if traj is not None and len(traj) else []
            rig.captured.append({
                'stage1': dict(getattr(mgr, '_active_ranges_stats', None) or {}),
                'plan': sorted(int(x) for x in (getattr(mgr, 'stage2_frame_plan', None) or {}).keys()),
                'traj_meta': dict(traj.meta) if traj is not None else None,
                'traj_frames': frames,
                'shots': sorted(int(c) for c in (getattr(mgr, '_shot_boundaries', None) or ())),
            })
            return r

        ProcessMgr._precompute_temporal = wrapped
        self._restore.append(lambda: setattr(ProcessMgr, '_precompute_temporal', orig))

    def close(self):
        for fn in reversed(self._restore):
            fn()

    # ── a render ───────────────────────────────────────────────────────────
    def render(self, key, clip, rng=(0, 0), env=None, job=None, spec=None, total=N_FRAMES, event_timed=False):
        if key in self.runs:
            return self.runs[key]
        import roop.globals as g
        from roop import stage_telemetry as st
        import torch
        st.reset()
        g.last_finalize = []
        self.captured.clear()
        wall = ev_ms = None
        with mock.patch.dict(os.environ, dict(ENV, **(env or {}))), mock.patch.object(self.V, 'N_FRAMES', total):
            if event_timed:
                torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                t0 = time.perf_counter()
            run = self.V.render(self.ctx, self.P, clip, key, self.out_root, {'sparse_auto_range': True}, rng=rng,
                                job=job, spec=spec)
            if event_timed:
                e1.record()
                torch.cuda.synchronize()                        # the one full drain; nothing synchronizes per batch
                wall = time.perf_counter() - t0
                ev_ms = e0.elapsed_time(e1)
        fins = list(getattr(g, 'last_finalize', None) or [])
        for f in fins:
            st.set_finalize(f)
        run['report'] = st.end_run(ok=run['error'] is None)
        run['finalize'] = fins
        run['capture'] = self.captured[-1] if self.captured else None
        run['rng'] = rng
        run['wall_s'], run['event_ms'] = wall, ev_ms
        run['log_text'] = open(run['log'], encoding='utf-8', errors='replace').read()
        run['leftovers'] = sorted(n for n in os.listdir(self.out_root) if any(m in n for m in ORPHAN_MARKERS))
        self.runs[key] = run
        return run


@pytest.mark.gpu
class ThreeStageSyncAndFps(unittest.TestCase):
    rig = None

    @classmethod
    def setUpClass(cls):
        try:
            import torch
            if not torch.cuda.is_available():
                raise unittest.SkipTest('UNVERIFIED - no CUDA device here')
            from roop.ffmpeg_path import ffmpeg_binary                                # noqa: F401
        except unittest.SkipTest:
            raise
        except Exception as exc:
            raise unittest.SkipTest(f'UNVERIFIED - the GPU stack is not importable: {exc!r}')
        cls.workdir = tempfile.mkdtemp(prefix='e2e3stage_')
        cls.rig = Rig(cls.workdir)
        try:
            cls.rig.build()
        except BaseException:
            cls.rig.close()
            raise

    @classmethod
    def tearDownClass(cls):
        if cls.rig is not None:
            cls.rig.close()
        if os.environ.get('ROOP_E2E_KEEP') == '1':
            print(f'\n[e2e] kept {cls.workdir}')
        else:
            shutil.rmtree(cls.workdir, ignore_errors=True)

    # ── shared renders ─────────────────────────────────────────────────────
    def main_run(self):
        return self.rig.render('trim_10_165', self.rig.clip, rng=TRIM)

    # ═══ Test 1 ═════════════════════════════════════════════════════════════
    def test_1_the_source_clip_is_what_the_test_says_it_is(self):
        info = probe_video(self.rig.clip)
        self.assertEqual((info['width'], info['height']), (W, H))
        self.assertEqual(info['frames'], N_FRAMES)
        self.assertIsNotNone(info['audio_s'])

    def test_1_stage1_stage2_stage3_are_frame_synchronized_over_a_non_divisible_trim(self):
        run = self.main_run()
        rep = run['report']
        S1, S2, S3 = ('STAGE_1_SPARSE_PRESCAN', 'STAGE_2_ACTIVE_PREPASS', 'STAGE_3_MAIN_SWAP')
        self.assertIsNone(run['error'])
        cap = run['capture']
        self.assertIsNotNone(cap, 'the pre-pass never handed over (Stage 1/2 did not run)')

        # Stage 1: exactly the cuts that fall inside the trim, as absolute source frames
        cuts = [int(c) for c in (cap['stage1'].get('scene_cuts_abs') or cap['stage1'].get('scene_cuts') or ())]
        self.assertEqual(sorted(cuts), list(CUTS), f'Stage 1 scene cuts {cuts}; stats keys {sorted(cap["stage1"])}')

        # Stage 2: K=4 really strided the detector, and the cache spans the whole trim
        s2 = rep['stages'][S2]
        self.assertEqual(s2['detect_interval'], 4)
        self.assertLess(s2['detector_calls'], s2['scan_frames'], 'K=4 but the detector ran on every frame')
        self.assertGreater(s2['coast_frames'], 0)
        meta = cap['traj_meta']
        self.assertIsNotNone(meta, 'no trajectory cache was built')
        self.assertEqual(meta['total_frames'], TRIM_FRAMES)
        self.assertEqual(len(cap['traj_frames']), TRIM_FRAMES, 'the cache has frames without a face (or beyond the trim)')
        self.assertEqual(cap['traj_frames'][0] - meta['frame_start'], 0)
        self.assertEqual(cap['traj_frames'][-1] - meta['frame_start'], TRIM_FRAMES - 1)

        # Stage 3: the delivered file, decoded, has exactly the trim and nothing dropped off the tail
        out = run['output']
        self.assertTrue(out and os.path.isfile(out), 'no output file')
        info = probe_video(out)
        self.assertEqual((info['rc'], info['stderr']), (0, ''), 'ffprobe reports container problems')
        self.assertEqual(info['frames'], TRIM_FRAMES, 'decoded frame count')
        self.assertEqual(info['packets'], TRIM_FRAMES, 'packet count')
        self.assertEqual((info['width'], info['height']), (W, H))
        self.assertIsNotNone(info['audio_s'], 'the muxed file lost its audio')
        self.assertLess(abs(info['audio_s'] - info['video_s']), 1.0 / FPS,
                        f"audio {info['audio_s']}s vs video {info['video_s']}s")
        s3 = rep['stages'][S3]
        self.assertEqual(s3['frames_retired'], TRIM_FRAMES, 'frames the writer accepted')
        # the face is in every frame, so Stage 1 must leave no frame of the trim outside an active range (before the
        # edge samples: 126 of 155 active, 29 frames passed through unswapped)
        self.assertEqual((cap['stage1']['active_frames'], cap['stage1']['skipped_frames']), (TRIM_FRAMES, 0))
        self.assertEqual([tuple(r) for r in cap['stage1']['ranges']], [(0, TRIM_FRAMES - 1)])

    def test_1_the_render_leaves_nothing_behind_and_logs_no_failure(self):
        run = self.main_run()
        fin = (run['finalize'] or [{}])[-1]
        self.assertTrue(fin.get('ok'), fin.get('problems'))
        self.assertTrue(fin.get('temp_deleted'))
        self.assertEqual(fin.get('orphans') or [], [])
        self.assertEqual(run['leftovers'], [])
        self.assertEqual({k: v for k, v in run['failure_signatures'].items() if v}, {})

    def test_1_the_requested_batch_ceiling_is_never_exceeded(self):
        """B=8 is a CEILING (clamped by threads and the VRAM governor); the log line states the one that applied."""
        import re
        text = self.main_run()['log_text']
        seen = [int(m) for m in re.findall(r'cross-frame batching ON \(max_batch=(\d+)', text)]
        if not seen:
            self.skipTest('UNVERIFIED - this render path never built the cross-frame swap batcher '
                          '(no "[BatchSwap] cross-frame batching ON" line), so B was not exercised')
        self.assertTrue(all(1 <= b <= 8 for b in seen), seen)

    # ═══ Test 2 ═════════════════════════════════════════════════════════════
    def test_2_batch_matrix_two_segment_join_is_intact(self):
        import routes_queue as q
        from roop import segment_output as so
        import roop.globals as g
        rig = self.rig
        half = N_FRAMES // 2
        jobs, specs = [], []
        for i in range(2):
            job = {'id': f'job{i + 1}', 'frame_start': i * half + 1, 'frame_end': (i + 1) * half,   # UI: 1-based, inclusive
                   'target_name': os.path.basename(rig.clip), 'segment': {'group': GROUP, 'index': i + 1, 'count': 2}}
            jobs.append(job)
            specs.append(q._segment_spec_for(job))
        out_dir = os.path.join(rig.workdir, 'segments')
        os.makedirs(out_dir, exist_ok=True)
        rig_out = rig.out_root
        rig.out_root = out_dir
        expected = [so.segment_destination(out_dir, s, '.mp4') for s in specs]
        try:
            import api
            clear = api._clear_output_folder
        except Exception:
            clear = None
        old_clear = getattr(getattr(g, 'CFG', None), 'clear_output', None)
        before_later, first_stat = [], None
        try:
            for i, (job, spec) in enumerate(zip(jobs, specs)):
                g.output_path = out_dir
                if getattr(g, 'CFG', None) is not None:
                    g.CFG.clear_output = True                                  # the default: each slice used to wipe the last
                g._segment_spec = spec
                if clear is not None:
                    clear()
                before_later.append([os.path.isfile(p) for p in expected[:i]])
                try:
                    run = rig.render(f'segment{i + 1}', rig.clip, job=job, spec=spec)
                finally:
                    g._segment_spec = None
                self.assertIsNone(run['error'])
                if i == 0:
                    first_stat = (os.path.getsize(expected[0]), os.path.getmtime(expected[0]))
        finally:
            rig.out_root = rig_out
            if getattr(g, 'CFG', None) is not None and old_clear is not None:
                g.CFG.clear_output = old_clear

        self.assertEqual(len(set(expected)), 2, 'both slices got the same destination')
        for p in expected:
            self.assertEqual(os.path.basename(os.path.dirname(p)), GROUP)
            self.assertEqual(os.path.basename(os.path.dirname(os.path.dirname(p))), '.segments')
            self.assertTrue(os.path.isfile(p) and os.path.getsize(p) > 0, p)
        self.assertEqual(before_later[1], [True], 'slice 2 started after slice 1 had already been removed')
        self.assertEqual((os.path.getsize(expected[0]), os.path.getmtime(expected[0])), first_stat,
                         'slice 2 rewrote slice 1')
        seg_frames = [probe_video(p)['frames'] for p in expected]
        self.assertEqual(seg_frames, [half, half])

        ranges = [(s.start_frame, s.end_frame) for s in specs]
        joined = so.join_segments(expected, so.joined_path(out_dir, rig.clip, '.mp4'),
                                  so.concat_list_path(out_dir, rig.clip), source_media=rig.clip,
                                  source_audio=so.covers_whole_source(ranges, N_FRAMES))
        info = probe_video(joined)
        self.assertEqual((info['rc'], info['stderr']), (0, ''))
        self.assertEqual(info['frames'], N_FRAMES, 'joined frame count')
        self.assertIsNotNone(info['audio_s'], 'the join lost the audio')
        self.assertLess(abs(info['audio_s'] - N_FRAMES / FPS), 1.0 / FPS, f"joined audio {info['audio_s']}s")
        self.assertLess(abs(info['audio_s'] - info['video_s']), 1.0 / FPS)
        self.assertTrue(all(os.path.isfile(p) for p in expected), 'the join consumed its parts')

    # ═══ Test 3 ═════════════════════════════════════════════════════════════
    def runtime(self):
        self.main_run()
        from roop import runtime_state
        return runtime_state.snapshot()

    def test_3_the_14_section_report_is_complete(self):
        sections = self.runtime()['sections']
        self.assertEqual(len(sections), 14, sorted(sections))
        for name in ('HARDWARE', 'PROVIDER', 'PRECISION', 'PERFORMANCE'):
            self.assertEqual(sections[name]['status'], 'AVAILABLE', name)

    def test_3_decode_and_encode_are_hardware_and_every_step_down_is_explained(self):
        vio = self.main_run()['report']['video_io']
        self.assertIsNotNone(vio)
        self.assertTrue(any(t in vio['decode_backend'] for t in ('NVDEC', 'QSV')), vio['decode_backend'])
        self.assertTrue(any(t in vio['encode_backend'] for t in ('NVENC', 'QSV')), vio['encode_backend'])
        stepped_down = [k for k, v in vio['decoders'].items() if str(v).startswith('CPU')]
        if stepped_down:
            self.assertTrue(vio['fallbacks'], f'{stepped_down} fell to the CPU with no recorded reason (silent fallback)')

    def test_3_stage1_keyframe_scan_decodes_on_nvdec_dlpack(self):
        vio = self.main_run()['report']['video_io']
        self.assertEqual(vio['decoders'].get('stage1-keyframes'), 'NVDEC-DLPack', vio)

    def test_3_the_cpu_warp_and_blend_are_labelled_as_cpu(self):
        ops = self.main_run()['report']['stages']['STAGE_3_MAIN_SWAP']['ops']
        self.assertEqual(ops['warp']['backend'], 'cv2 (CPU)')
        self.assertEqual(ops['blend']['backend'], 'cv2 (CPU)')

    @pytest.mark.xfail(strict=True, reason='NO GPU warp is wired into the render path: ops.warp.backend is cv2 (CPU). '
                       'CV-CUDA has no wheel for Windows/py3.10; CudaAffineBatch / Kornia are 3-4x SLOWER than cv2 with '
                       'host frames (RECODE_STATUS 2026-10-25, closed). Wiring it flips this to XPASS and fails the suite.')
    def test_3_gpu_warp_cvcuda_or_kornia_is_active(self):
        backend = str(self.main_run()['report']['stages']['STAGE_3_MAIN_SWAP']['ops']['warp']['backend']).lower()
        self.assertTrue(any(t in backend for t in ('cv-cuda', 'cvcuda', 'kornia', 'cuda')), backend)

    @pytest.mark.xfail(strict=True, reason='the Triton fused post-process kernel (app/core/fused_postprocess.py) is built '
                       'and tested but has no render-path caller; ops.blend.backend is cv2 (CPU).')
    def test_3_triton_fused_epilogue_is_active(self):
        backend = str(self.main_run()['report']['stages']['STAGE_3_MAIN_SWAP']['ops']['blend']['backend']).lower()
        self.assertIn('triton', backend)

    @pytest.mark.xfail(strict=True, reason="ORT's trt_cuda_graph_enable is OFF by default (ROOP_TRT_CUDA_GRAPH=0): the "
                       'swap model runs as plain TensorRT, label "TRT", not "TRT Graph".')
    def test_3_trt_cuda_graph_is_active(self):
        backend = str(self.main_run()['report']['stages']['STAGE_3_MAIN_SWAP']['ops']['model']['backend']).lower()
        self.assertIn('graph', backend)

    def test_3_mixed_precision_trt_and_its_canaries_hold(self):
        precision = self.runtime()['sections']['PRECISION']['values']
        gates = precision['verification_gates']
        self.assertGreaterEqual(gates['psnr_min_db'], 48.0)                  # the surgeon's gate constant, not a measurement
        self.assertGreaterEqual(gates['cosine_min'], 0.9995)
        self.assertIs(gates['nan_inf_free'], True, 'not established: some model has no passing latest canary verdict')
        self.assertIn(str(precision['effective']).lower().split()[0], ('mixed', 'fp16', 'fp32', 'fp8', 'int8'),
                      precision['effective'])
        verdicts = precision['canary_verdicts']
        self.assertTrue(verdicts, 'no canary verdict is recorded at all')
        failed = {k: v['reason'] for k, v in verdicts.items() if v['passed'] is not True}
        self.assertEqual(failed, {}, 'a canary failed or was skipped')
        # NaN/Inf-freedom from the canaries' OWN metrics (`nan_inf_free` in the report is a literal True)
        from roop.aux_canary import _load_store
        bad = [(k[:8], m, val) for k, v in _load_store().get('verdicts', {}).items()
               for m, val in (v.get('metrics') or {}).items() if not (isinstance(val, (int, float)) and math.isfinite(val))]
        self.assertEqual(bad, [], 'non-finite canary metrics')

    # ═══ Test 4 ═════════════════════════════════════════════════════════════
    def test_4_warm_up_then_a_timed_200_frame_run_with_a_single_drain(self):
        rig = self.rig
        warm = rig.render('bench_warmup', rig.bench_clip, rng=(0, BENCH_WARMUP), total=BENCH_WARMUP + BENCH_FRAMES)
        self.assertIsNone(warm['error'])
        run = rig.render('bench_timed', rig.bench_clip, rng=(BENCH_WARMUP, BENCH_WARMUP + BENCH_FRAMES),
                         total=BENCH_WARMUP + BENCH_FRAMES, event_timed=True)
        self.assertIsNone(run['error'])
        rep = run['report']
        s1, s2, s3 = (rep['stages'][k] for k in ('STAGE_1_SPARSE_PRESCAN', 'STAGE_2_ACTIVE_PREPASS', 'STAGE_3_MAIN_SWAP'))
        self.assertEqual(s3['frames_retired'], BENCH_FRAMES)
        self.assertEqual(probe_video(run['output'])['frames'], BENCH_FRAMES)
        # the event bracket and the wall clock bracket the same drained interval
        self.assertAlmostEqual(run['event_ms'] / 1000.0, run['wall_s'], delta=max(0.25, 0.05 * run['wall_s']))

        ops = s3['ops']
        ms = lambda k: (ops.get(k) or {}).get('ms')                                  # noqa: E731
        row = [s1['keyframes_per_sec'], s2['fps'], s3['effective_output_fps'], ms('decode'), ms('warp'), ms('model'),
               ms('blend'), ms('encode'), (rep['vram']['peak_gb'] or 0) * 1024.0]
        for name, value in zip(('Stage 1 FPS', 'Stage 2 FPS', 'Stage 3 Effective FPS', 'Decode ms', 'Warp ms',
                                'TRT Graph ms', 'Fused Post-Process ms', 'NVENC ms', 'Peak VRAM MB'), row):
            self.assertIsNotNone(value, f'{name} was not measured')
            self.assertGreater(value, 0, name)

        heads = ('Stage 1 FPS', 'Stage 2 FPS', 'Stage 3 Effective FPS', 'Decode ms', 'Warp ms', 'TRT Graph ms',
                 'Fused Post-Process ms', 'NVENC ms', 'Peak VRAM MB')
        cells = [f'{v:,.2f}' for v in row]
        widths = [max(len(h), len(c)) for h, c in zip(heads, cells)]
        line = ' | '.join(h.rjust(w) for h, w in zip(heads, widths))
        print('\n' + line + '\n' + '-+-'.join('-' * w for w in widths) + '\n' + ' | '.join(c.rjust(w) for c, w in zip(cells, widths)))
        print(f"  {BENCH_FRAMES} frames retired in {run['wall_s']:.2f} s wall / {run['event_ms'] / 1000.0:.2f} s cuda-event "
              f"(warm-up: {BENCH_WARMUP} frames = 3 batches of 8, rendered first; one drain at the end)")
        print(f"  Decode: {ops['decode']['backend']} | Warp: {ops['warp']['backend']} (CPU alignment) | "
              f"TRT Graph: {ops['model']['backend']} per call, CUDA graph OFF | Fused Post-Process: "
              f"{ops['blend']['backend']} paste-back (Triton kernel not wired) | NVENC: {ops['encode']['backend']}")
        print('  per-op ms are host-wall of blocking calls; Stage 3 FPS = frames the writer accepted / Stage 3 wall.')


if __name__ == '__main__':
    unittest.main()
