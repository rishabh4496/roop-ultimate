"""Batch video-run orchestration for ProcessMgr.

This boundary owns per-clip lifecycle, scheduling selection, writer cleanup,
and the in-memory batch path. Model-specific operations remain on ProcessMgr
and its focused mixins; this module coordinates them without owning global
application state.
"""

from __future__ import annotations


class BatchProcessingMixin:
    """Own the per-clip batch lifecycle separately from face operations."""

    def run_batch_inmem(self, output_method, source_video, target_video, frame_start, frame_end, fps, threads: int = 1, skip_audio=False):
        self._writer_error = None
        # Dependencies are bound at call time because ProcessMgr owns the
        # shared runtime instrumentation and scheduler singletons. This keeps
        # the extracted batch boundary explicit without creating a second copy
        # of those process-wide authorities.
        import gc
        import os
        import sys
        import cv2
        import roop
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from queue import Queue
        from threading import Thread
        from roop.ProcessMgr import (
            _MAX_STAB_WARMUP, _audit_report, _audit_reset,
            _configure_opencv_worker_threads, _prof, _prof_report, _prof_reset,
            bar_write, set_detailed_profiler, set_runtime_monitor,
            ChunkedProgress, FFMPEG_VideoWriter, PROGRESS_BAR_FORMAT,
            RuntimeMonitor, RuntimeOptimizer, SafeAdaptiveController,
            StageProfiler, StreamingStabilizationHistory,
            StreamWriter, UnifiedRuntimeScheduler,
        )
        from roop.video_stream import (
            NVHardwareVideoWriter,
            hardware_stream_enabled,
            open_video_capture,
        )
        from roop import face_util
        from roop import runtime_banner as _runtime_banner
        import roop.util_ffmpeg as util_ffmpeg
        from roop.degrade import swallowed as _swallowed
        # Stabilization scheduling (temporal smoothing needs frames in order; the
        # multithreaded reader strides them out-of-order):
        #  - kps-only stabilization → 2-pass: precompute smoothed kps sequentially
        #    (pass 1, below once the source is set up) then swap in parallel
        #    (pass 2), keeping multi-thread speed.
        #  - enhancer stabilization smooths the enhanced OUTPUT, which only exists
        #    during the swap, so it can't be precomputed → fall back to the
        #    original single-thread sequential path.
        #  - parallel stabilization (opt-in ROOP_STAB_PARALLEL): process contiguous
        #    frame blocks per thread, each with its own stabilizer + warm-up, so
        #    BOTH kps and enhancer stabilization run multi-threaded.
        self._precomputed_mode = False
        self._precomputed_kps = None
        self._stab_active = False
        self._parallel_stab = False
        self._stab_chunk_queue_capacity = None
        # fps is otherwise a local parameter never retained on self — lip-sync
        # needs it alongside self._tls.frame_idx to map a frame to a timestamp
        # in the driving audio (see roop.lipsync_audio.frame_time).
        self._lipsync_fps = fps
        self._lipsync_frame_start = frame_start
        self._lipsync_audio = None
        self._memory_stage_log = []
        self._stage_profiler = None
        self._stage_profile_report = None
        self._log_memory_stage('phase3:run-start')
        if getattr(roop.globals, 'lipsync_enabled', False):
            try:
                source_mode = getattr(roop.globals, 'lipsync_audio_source', 'original')
                audio_source = (roop.globals.lipsync_audio_path if source_mode == 'upload'
                                else source_video)
                if audio_source and os.path.isfile(audio_source):
                    # Whisper feature extraction reads the whole clip at once and is
                    # the expensive part here, so cache by source path — a batch that
                    # dubs several clips against the SAME uploaded track (source_mode
                    # == 'upload') must not redo it once per clip.
                    if getattr(self, '_lipsync_audio_cache_key', None) != audio_source:
                        from roop.utilities import get_temp_directory_path
                        wav_dir = get_temp_directory_path(audio_source)
                        os.makedirs(wav_dir, exist_ok=True)
                        wav_path = os.path.join(wav_dir, 'lipsync_audio.wav')
                        if util_ffmpeg.extract_audio_wav(audio_source, wav_path):
                            self._lipsync_audio_cache_val = self._lipsync_restorer().build_audio_cache(wav_path, fps)
                        else:
                            bar_write(f"[ProcessMgr] lip-sync: audio extraction failed for {audio_source!r}")
                            self._lipsync_audio_cache_val = None
                        self._lipsync_audio_cache_key = audio_source
                    self._lipsync_audio = self._lipsync_audio_cache_val
                else:
                    bar_write(f"[ProcessMgr] lip-sync: no driving audio at {audio_source!r}, skipping this clip")
            except Exception as e:
                bar_write(f"[ProcessMgr] lip-sync audio setup failed: {e}")
                self._lipsync_audio = None
        _detail_profile = str(os.environ.get('ROOP_PROFILE_DETAIL', '0')).strip().lower() in (
            '1', 'true', 'yes', 'on')
        if _detail_profile:
            try:
                _detail_sync = str(os.environ.get('ROOP_PROFILE_DETAIL_SYNC', '0')).strip().lower() in (
                    '1', 'true', 'yes', 'on')
                _device_id = int(getattr(roop.globals, 'cuda_device_id', 0) or 0)
                self._stage_profiler = StageProfiler(gpu_sync=_detail_sync,
                                                     device_id=_device_id)
                set_detailed_profiler(self._stage_profiler)
                print('[Phase14] detailed stage profiler enabled; '
                      f"mode={'synchronized' if _detail_sync else 'event-only'}; "
                      'external telemetry is required for full-card VRAM.', flush=True)
            except Exception as _exc:
                self._stage_profiler = None
                print(f'[Phase14] detailed stage profiler unavailable: {_exc}', flush=True)
        else:
            set_detailed_profiler(None)
        # Frame indices restart per clip, so a latch carried over from the last
        # one would match a new face by position and hand it a stale verdict.
        self._nonfrontal_router.reset()
        # The temporal identity output history is causal: a frame must consume
        # the previous frame's canonical crop before it can publish its own.
        # Do not send this opt-in path through the out-of-order stabilizer.
        if getattr(self, '_temporal_identity', None) is not None:
            self._temporal_identity.reset()
        if getattr(self, '_temporal_occlusion', None) is not None:
            self._temporal_occlusion.reset()
        if getattr(self, '_temporal_expression', None) is not None:
            self._temporal_expression.reset()
        if getattr(self, '_target_appearance', None) is not None:
            self._target_appearance.reset()
        if getattr(self, '_temporal_compositing', None) is not None:
            self._temporal_compositing.reset()
        if getattr(self, '_temporal_quality', None) is not None:
            self._temporal_quality.reset()
        # Swap-audit counters. Reset HERE, not in initialize(): core.py calls
        # initialize() once and then hands batch_process a whole LIST of files,
        # while the audit is reported at the end of each one — so resetting there
        # made every clip after the first report its own counts plus every
        # previous clip's, which is precisely the confusion the report exists to
        # remove. This block already exists to clear per-clip state.
        _audit_reset()
        # ...and the stage timings, for exactly the same reason and in the same
        # place. These two blocks print one after the other and invite being read
        # together, so one of them accumulating across clips while the other does
        # not is worse than either being wrong alone.
        _prof_reset()
        # Temporal detection (anti-flicker): its pre-pass gap-fills detection
        # misses AND applies the kps/lm106 smoothing itself, so the per-frame
        # kps stabilizer and the kps-only 2-pass become redundant — disable
        # them here so nothing double-smooths. (Enhancer flicker smoothing is
        # output-based and unaffected.)
        _temporal_identity_enabled = bool(
            getattr(getattr(self, '_temporal_identity', None), 'enabled', False))
        _temporal_occlusion_enabled = bool(
            getattr(getattr(self, '_temporal_occlusion', None), 'enabled', False))
        _temporal_expression_enabled = bool(
            getattr(getattr(self, '_temporal_expression', None), 'enabled', False))
        _target_appearance_enabled = bool(
            getattr(getattr(self, '_target_appearance', None), 'enabled', False))
        _temporal_state_enabled = (_temporal_identity_enabled
                                   or _temporal_occlusion_enabled
                                   or _temporal_expression_enabled)
        _temporal_compositing_enabled = bool(
            getattr(getattr(self, '_temporal_compositing', None), 'enabled', False))
        _temporal_quality_enabled = bool(
            getattr(getattr(self, '_temporal_quality', None), 'enabled', False))
        self._temporal_mode = bool(
            getattr(roop.globals, 'temporal_detection', False)
            or _temporal_state_enabled or _temporal_compositing_enabled
            or _temporal_quality_enabled)
        self._temporal_faces = None
        self._temporal_covered = 0
        # A ProcessMgr instance can render more than one clip.  Never let a
        # completed clip's compact stabilization context seed the next one.
        self._stab_history = None
        if self._temporal_mode:
            self.kps_stabilizer = None
            self._kps_stab_factory = None
            # ...and with it the landmark smoother that hangs off `_apply_stab`,
            # which is only reachable through the kps stabilizer just dropped.
            # The tracking pre-pass owns the coupled kps+lm106 smoothing on this
            # path (procmgr_tracking._build_temporal_faces) and reports its own
            # per-track summary, so leaving this instance enabled would be a
            # second, unreachable copy of the same feature -- and would make
            # `_report_smoother_summaries` warn "enabled but never invoked" on
            # every default render, which is a false alarm that would train a
            # reader to ignore the one message that catches a real regression.
            self._landmark_smoother.enabled = False
        _want_kps_stab = self.kps_stabilizer is not None
        _want_enh_stab = self.enh_stabilizer is not None
        _want_mask_stab = self.mask_stabilizer is not None
        # Parallel stabilization is now the DEFAULT. Smoothing is sequential per
        # face, not per clip, so a contiguous block primed with enough warm-up
        # produces the same output as running the whole clip in order — and the
        # warm-up needed for that is derived from the filter (see
        # _stab_warmup_frames), not guessed. Turning it off costs 2-3x on the
        # swap pass, measured, because the fallback is ONE thread.
        # ROOP_STAB_PARALLEL=0 restores the sequential path.
        # A temporal filter is a recurrence.  The normal path preserves one
        # live state over the entire clip in the bounded decode -> CUDA ->
        # encode stream, rather than priming a fresh block and discarding it.
        # Keep the old block path as an explicit rollback only.
        # DEFAULT '0' -- the streaming stream is OPT-IN, measured 2026-09-02.
        #
        # It is a real continuous FIFO with zero warm-up recompute, and that is
        # exactly what makes it slow: `use_unified_scheduler` pins the run to
        # ONE inference owner (`_inference_workers = 1` below), so decode fills
        # a 3-deep queue that a single consumer drains. Counterbalanced ABBA,
        # 141 output frames, production stabilizer settings, RTX 4070:
        #
        #     streaming (cuda_owner=1)   155.5 / 154.8 s   GPU peak 68-73%
        #     parallel blocks (8 wkrs)   126.4 / 139.0 s   GPU peak 92-97%
        #                                -> blocks +16.9%
        #
        # The worst block arm still beat the best streaming arm by 11.4%, and
        # the block path does ~28% MORE inference to get there (361 face
        # instances against 283, the excess being warm-up frames it
        # re-processes). Output frame counts are identical, 141 in / 141 out.
        #
        # Note for anyone reading a thread profile before changing this: the
        # FASTER arm has FEWER OS threads (198 mean against 256). Thread count
        # and the GIL are not what gates this pipeline -- inference
        # concurrency is. See also the Gate E sweep, 0.7% across threads 4..20.
        _streaming_stabilization = (
            (_want_kps_stab or _want_enh_stab or _want_mask_stab)
            and os.environ.get('ROOP_STAB_STREAMING', '0') != '0')
        _parallel_ok = os.environ.get('ROOP_STAB_PARALLEL', '1') != '0'
        self._stab_warmup = self._stab_warmup_frames()
        if _parallel_ok and self._stab_warmup >= _MAX_STAB_WARMUP:
            # A filter this slow never forgets its seed within a bounded warm-up,
            # so no block boundary can be made seam-free. Correctness wins.
            print(f"[Stabilize] smoothing too slow to parallelise safely "
                  f"(needs >{_MAX_STAB_WARMUP} warm-up frames) — staying sequential.")
            _parallel_ok = False
        # The opt-in identity and occlusion engines keep a per-track output
        # history, so they need frames IN ORDER. That used to be satisfied by
        # pinning the whole run to one worker, which cost 2.7x measured
        # (12.90 -> 4.79 fps on the locked fixture, 2026-08-31) -- and the
        # measurement that matters is that a plain `threads=1` control with no
        # flag set was 4.79 too, so the entire cost was the pinning and none of
        # it was the features.
        #
        # Ordered does not mean serial. The parallel-stabilization path already
        # hands each worker a CONTIGUOUS block, runs it in frame order, gives it
        # its own filter instances, and primes it with warm-up frames it then
        # discards -- which is the same problem and the same solution. So these
        # engines now ride that path (`clone_for_block` per block, warm-up from
        # their own recurrences) instead of collapsing it.
        #
        # Expression is not here on purpose: its state is written by the
        # sequential tracking pre-pass and `plan()` only reads, so it has always
        # been safe at full width.
        _want_temporal_identity = _temporal_identity_enabled
        _want_temporal_occlusion = _temporal_occlusion_enabled
        _want_temporal_ordered = (_want_temporal_identity or _want_temporal_occlusion
                                  or _target_appearance_enabled
                                  or _temporal_compositing_enabled
                                  or _temporal_quality_enabled)
        # 3x, i.e. warm-up overhead capped at 33%. Below that the priming costs
        # more than the extra workers return, measured both ways on this fixture.
        self._stab_min_block_multiple = 3 if _want_temporal_ordered else 1
        use_parallel_stab = ((not _streaming_stabilization)
                              and (_want_kps_stab or _want_enh_stab or _want_mask_stab
                               or _want_temporal_ordered)
                              and threads > 1 and _parallel_ok)
        _two_pass_ok = os.environ.get('ROOP_STAB_2PASS', '1') != '0'
        # 2-pass smooths sequentially in pass 1 and then swaps ROUND-ROBIN in
        # pass 2. That is fine for a kps filter, whose work is finished before
        # pass 2 begins, and useless here: the output history advances during the
        # swap. So it is never an option for the temporal engines.
        use_2pass = ((not _streaming_stabilization) and (not use_parallel_stab)
                     and _want_kps_stab and not _want_enh_stab
                     and not _want_mask_stab and not _want_temporal_ordered
                     and threads > 1 and _two_pass_ok)
        if _want_temporal_ordered and not use_parallel_stab and threads != 1:
            # Only reached when the parallel path is unavailable outright
            # (ROOP_STAB_PARALLEL=0, or a filter too slow to prime inside
            # _MAX_STAB_WARMUP). Round-robin workers would advance one track's
            # history out of order, so correctness still wins -- but say what it
            # costs, because the symptom the user sees is `execution_threads=1`
            # and a render at a third of its fps.
            _label = ('TemporalIdentity' if _want_temporal_identity
                      else 'TemporalOcclusion' if _want_temporal_occlusion
                      else 'TemporalCompositing' if _temporal_compositing_enabled
                      else 'TemporalQuality')
            print(f'[{_label}] ordered output history and no parallel-block '
                  f'path available: this run gets ONE worker instead of '
                  f'{threads}.', flush=True)
            threads = 1
        if ((_want_kps_stab or _want_enh_stab or _want_mask_stab
             or _want_temporal_ordered)
                and not _streaming_stabilization and not use_2pass and not use_parallel_stab):
            if threads != 1:
                print("[Stabilize] Forcing single thread for temporal smoothing.")
            threads = 1
            self._stab_active = True
            self._stab_t = 0
            if self.kps_stabilizer is not None:
                self.kps_stabilizer.reset()
            if self.enh_stabilizer is not None:
                self.enh_stabilizer.reset()
            if self.mask_stabilizer is not None:
                self.mask_stabilizer.reset()

        if _streaming_stabilization:
            # The one CUDA owner advances these exactly once per frame.  This
            # is the rolling FIFO: no clone, no block boundary, no rework.
            self._stab_active = True
            self._stab_t = 0
            self._stab_history = StreamingStabilizationHistory()
            for _stab in (self.kps_stabilizer, self.enh_stabilizer,
                          self.mask_stabilizer):
                if _stab is not None:
                    _stab.reset()

        # Animated WebP: OpenCV cannot decode it — use PIL-based reader instead
        is_awebp = source_video.lower().endswith('.webp')
        cap = None
        awebp_frames = None

        if is_awebp:
            from roop.capturer import _load_animated_webp
            import roop.capturer as _capturer_mod
            _load_animated_webp(source_video)
            awebp_frames = _capturer_mod._awebp_frames or []
            if awebp_frames:
                height, width = awebp_frames[0].shape[:2]
            else:
                width, height = 0, 0
            frame_count = len(awebp_frames[frame_start:frame_end]) if frame_end > frame_start else len(awebp_frames[frame_start:])
        else:
            cap = cv2.VideoCapture(source_video)
            # `endframe` is exclusive throughout core.py and both readers. Keep
            # progress, resume, and temporal-prepass accounting on that same
            # contract: frames [start, end) contains exactly end - start frames.
            frame_count = frame_end - frame_start
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # NVDEC: swap the cv2 reader for a GPU-decode ffmpeg pipe when the
            # file probes OK (no-op otherwise; ROOP_NVDEC=0 disables). Must use
            # the SOURCE dims, before any processed_resolution override.
            try:
                from roop.runtime_optimizer import (shared_hardware_profile,
                                                     small_card_decode_policy)
                self._small_card_decode_env_previous = os.environ.get('ROOP_NVDEC')
                self._small_card_decode_env_present = 'ROOP_NVDEC' in os.environ
                _decode_policy = small_card_decode_policy(
                    shared_hardware_profile(
                        getattr(roop.globals, 'cuda_device_id', 0) or 0))
                self._small_card_decode_policy = _decode_policy
                if _decode_policy.get('changed'):
                    os.environ['ROOP_NVDEC'] = '0'
                    print('[RuntimeOptimizer] sub-7GB decode safety: NVDEC -> CPU; '
                          f"{_decode_policy['reason']}. Set ROOP_SMALL_CARD_NVDEC=keep "
                          'or ROOP_NVDEC=1 for an explicit A/B.', flush=True)
            except Exception as _exc:
                print(f'[RuntimeOptimizer] small-card decode policy unavailable: {_exc}',
                      flush=True)
            cap = open_video_capture(
                source_video,
                width,
                height,
                fps,
                fallback_capture=cap,
                tag='swap decode',
            )

        processed_resolution = None
        for p in self.processors:
            if hasattr(p, 'getProcessedResolution'):
                processed_resolution = p.getProcessedResolution(width, height)
                print(f"Processed resolution: {processed_resolution}")
        if processed_resolution is not None:
            width = processed_resolution[0]
            height = processed_resolution[1]

        # Workload-aware runtime profile.  This is deliberately after probing
        # the real video dimensions and before queue/chunk geometry is chosen.
        # It derives bounded hints and records provenance.  On the sub-7GB
        # laptop tier it may clamp the caller's worker count to one because
        # the measured multi-worker path violates the 2.5GB RSS ceiling.
        self.runtime_profile = None
        self._runtime_stab_chunk = None
        self._runtime_stab_workers = None
        self._runtime_stab_small = False
        self._runtime_queue_depth = None
        self._runtime_swap_batch_size = 1
        self._runtime_swap_tile_batch_size = None
        self._runtime_face_concurrency = 1
        self._runtime_in_flight_frames = 1
        self._runtime_ram_buffer_mb = None
        self._runtime_monitor = None
        self._runtime_adaptive = None
        with self.lock:
            self._runtime_worker_busy.clear()
        set_runtime_monitor(None)
        self._small_card_enhancer_policy = None
        self._small_card_decode_policy = None
        self._small_card_decode_env_previous = None
        self._small_card_decode_env_present = False
        self._replay_analysis_released = False
        try:
            _runtime_optimizer = RuntimeOptimizer(settings=getattr(roop.globals, 'CFG', None))
            _target_datas = getattr(self, 'target_face_datas', []) or []
            _target_groups = getattr(self, 'target_face_groups', None) or getattr(roop.globals, 'TARGET_FACE_GROUP', None)
            if _target_groups and (not _target_datas or len(_target_groups) == len(_target_datas)):
                try:
                    _unique_identities = len(set(_target_groups))
                except Exception:
                    _unique_identities = len(_target_groups)
            else:
                _unique_identities = len(_target_datas)
            self.runtime_profile = _runtime_optimizer.profile_video(
                source_video,
                frame_count=frame_count,
                resolution=(width, height),
                output_resolution=(width, height),
                faces_per_frame=max(1, _unique_identities),
                face_count=_unique_identities,
                save=True)
            # Publish the one profiled device to model-session policy. Every
            # provider decision in this workload now uses the same detected
            # architecture, VRAM, software stack, and precision capabilities.
            roop.globals.runtime_hardware_profile = self.runtime_profile.hardware
            RuntimeOptimizer.apply_environment(
                self.runtime_profile, getattr(roop.globals, 'CFG', None))
            cfg_codec = str(getattr(getattr(roop.globals, 'CFG', None),
                                   'output_video_codec', 'auto') or 'auto').strip().lower()
            if cfg_codec in ('', 'auto', 'default', 'none'):
                # Automatic codec selection is applied only at runtime; the
                # saved/user-facing setting is never rewritten. An explicit
                # codec remains authoritative.
                roop.globals.video_encoder = self.runtime_profile.tuning.encoder
            self._runtime_stab_chunk = self.runtime_profile.tuning.stabilization_chunk_size
            self._runtime_stab_workers = self.runtime_profile.tuning.stabilization_workers
            self._runtime_stab_small = self.runtime_profile.hardware.vram_total_gb < 7.0
            self._runtime_queue_depth = self.runtime_profile.tuning.queue_depth
            self._runtime_swap_batch_size = self.runtime_profile.tuning.batch_size
            self._runtime_swap_tile_batch_size = self.runtime_profile.tuning.tile_batch_size
            self._runtime_face_concurrency = self.runtime_profile.tuning.face_concurrency
            self._runtime_in_flight_frames = self.runtime_profile.tuning.in_flight_frames
            self._runtime_ram_buffer_mb = self.runtime_profile.tuning.ram_buffer_mb
            _scheduler_enabled = str(os.environ.get(
                'ROOP_UNIFIED_SCHEDULER', '1')).strip().lower() not in (
                    '0', 'false', 'no', 'off')
            if _scheduler_enabled:
                self._runtime_scheduler = UnifiedRuntimeScheduler(
                    self.runtime_profile.hardware,
                    self.runtime_profile.workload,
                    self.runtime_profile.tuning,
                    settings=getattr(roop.globals, 'CFG', None),
                    monitor=None,
                    adaptive=None)
            monitor_enabled = str(os.environ.get('ROOP_RUNTIME_MONITOR', '0')).strip().lower() in (
                '1', 'true', 'yes', 'on')
            if monitor_enabled:
                self._runtime_monitor = RuntimeMonitor(
                    hardware=self.runtime_profile.hardware,
                    tuning=self.runtime_profile.tuning,
                    settings=getattr(roop.globals, 'CFG', None))
                self._runtime_monitor.start()
                set_runtime_monitor(self._runtime_monitor)
                if self._runtime_scheduler is not None:
                    # Keep the scheduler's optional rolling bottleneck
                    # classifier connected to the same monitor that owns the
                    # application's stage telemetry.
                    self._runtime_scheduler.monitor = self._runtime_monitor
                self._runtime_adaptive = SafeAdaptiveController(
                    self.runtime_profile.hardware, self.runtime_profile.tuning,
                    settings=getattr(roop.globals, 'CFG', None),
                    enabled=str(os.environ.get('ROOP_RUNTIME_ADAPTIVE', '0')).strip().lower() in (
                        '1', 'true', 'yes', 'on'))
            cfg = getattr(roop.globals, 'CFG', None)
            auto_threads = bool(getattr(cfg, 'auto_thread_selection', False)) and bool(
                getattr(cfg, '_threads_auto', False))
            if (auto_threads and not self._runtime_stab_small and threads > 1
                    and self.runtime_profile.tuning.worker_count != threads):
                print(f"[RuntimeOptimizer] workload worker policy: execution threads "
                      f"{threads} -> {self.runtime_profile.tuning.worker_count}; "
                      "bounded by CPU topology, workload complexity, and GPU tier.",
                      flush=True)
                threads = self.runtime_profile.tuning.worker_count
                roop.globals.execution_threads = threads
            print("[RuntimeOptimizer] workload profile: "
                  f"{self.runtime_profile.workload.input_width}x"
                  f"{self.runtime_profile.workload.input_height}, "
                  f"complexity={self.runtime_profile.workload.estimated_complexity:.2f}, "
                  f"workers(recommended)={self.runtime_profile.tuning.worker_count}, "
                  f"queue={self._runtime_queue_depth}, "
                  f"swap_batch={self._runtime_swap_batch_size}, "
                  f"swap_tile_batch={self._runtime_swap_tile_batch_size}, "
                  f"face_concurrency={self._runtime_face_concurrency}, "
                  f"in_flight={self._runtime_in_flight_frames}, "
                  f"stab_chunk={self._runtime_stab_chunk}, "
                  f"profile={self.runtime_profile.cache_key}", flush=True)
            autotune = getattr(self.runtime_profile, 'autotune', {}) or {}
            if autotune:
                selected = autotune.get('selected', {})
                print("[RuntimeAutotune] selected="
                      f"{selected}; candidates={len(autotune.get('candidates_tested', []))}; "
                      f"baseline_fps={autotune.get('baseline_fps', 0):.2f}; "
                      f"best_fps={autotune.get('best_fps', 0):.2f}; "
                      f"improvement={autotune.get('improvement_pct', 0):.2f}%",
                      flush=True)
                best_trial = max(autotune.get('candidates_tested', []) or [],
                                 key=lambda item: item.get('measurement', {}).get('score', 0))
                best_metrics = best_trial.get('measurement', {})
                print("[RuntimeAutotune] resources="
                      f"VRAM={best_metrics.get('peak_vram_gb', 0):.2f}GB; "
                      f"RAM={best_metrics.get('peak_ram_gb', 0):.2f}GB; "
                      f"CPU={best_metrics.get('cpu_utilization_pct', 'n/a')}; "
                      f"GPU={best_metrics.get('gpu_utilization_pct', 'n/a')}", flush=True)
            else:
                print("[RuntimeAutotune] heuristic profile selected; "
                      "run the bounded manual retune to measure candidates.", flush=True)

            # A small-VRAM device still uses one GPU context, but that does not
            # make host-side compositing single-threaded. Keep the configured /
            # runtime-selected worker count here; stabilization's RAM-derived
            # geometry below decides how many frame blocks can run concurrently.
            # This prevents a GPU memory policy from unnecessarily collapsing
            # the entire post-inference pipeline to one worker.
            self._log_memory_stage('phase3:runtime-profiled')
        except Exception as exc:
            print(f"[RuntimeOptimizer] workload profile unavailable: {exc}", flush=True)

        # Parallel stabilization buffers whole chunks of DECODED frames, so its
        # memory budget is counted in frames of this size. Only now, with the
        # dimensions known, can we tell how wide a stabilized run can actually
        # go — so the parallel/sequential choice made above gets one last look.
        self._stab_frame_bytes = int(width) * int(height) * 3
        if use_parallel_stab:
            # The stabilization path owns its chunk queues. Never reuse the
            # frame scheduler's depth for whole decoded chunks.
            self._stab_chunk_queue_capacity = 1
            _wu, _blk, _width, _bpc = self._stab_parallel_geometry(threads)
            if _width < 2:
                # 1-wide is single-threaded anyway, and paying the chunk
                # buffering and block bookkeeping on top of that measured SLOWER
                # than the plain sequential path (2.65 vs 2.92 fps at strength
                # 1.0, 1080p). Take the path that is actually faster.
                #
                # RE-ASK THE 2-PASS QUESTION. It was answered above as
                # `(not use_parallel_stab) and ...` while parallel was still on,
                # so it came out False for a reason that has just stopped being
                # true. Leaving it stale is what turned a memory-tight machine
                # into a ONE-WORKER render: the kps-only path has a way to keep
                # every thread (smooth sequentially in pass 1, swap
                # multi-threaded in pass 2) and it was being skipped silently.
                use_parallel_stab = False
                # `not _want_temporal_ordered` for the same reason it is excluded
                # above: pass 2 swaps round-robin, and the output history the
                # identity/occlusion engines keep advances during the swap.
                use_2pass = (_want_kps_stab and not _want_enh_stab
                             and not _want_mask_stab and not _want_temporal_ordered
                             and threads > 1 and _two_pass_ok)
                if _want_temporal_ordered:
                    threads = 1
                print(f"[Stabilize] warm-up {_wu}f needs {_blk}f blocks and only "
                      f"one fits the memory budget — not chunking "
                      f"(1-wide chunking is slower than the sequential path). "
                      f"Raise ROOP_STAB_CHUNK_MB to widen it.")
                if not use_2pass:
                    # Say what it COSTS, at the moment it is decided. This used
                    # to report only the stabilizer's chunk geometry, while the
                    # thing the user actually sees -- `execution_threads=1` in
                    # the progress bar, and a render at a fraction of its fps --
                    # went unexplained thousands of log lines later.
                    # `_warn_single_worker_on_gpu` cannot cover this: it runs in
                    # batch_process, BEFORE this drop happens.
                    print(f"[Stabilize] this run gets ONE worker thread instead "
                          f"of {threads}: temporal smoothing has to see frames "
                          f"in order, and only the kps-only case can be split "
                          f"into two passes (enhancer={_want_enh_stab}, "
                          f"mask={_want_mask_stab}). Free RAM, or turn "
                          f"stabilize_enhancer/stabilize_mask off to keep the "
                          f"threads.")
                    threads = 1
                    self._stab_active = True
                    self._stab_t = 0
                    if self.kps_stabilizer is not None:
                        self.kps_stabilizer.reset()
                    if self.enh_stabilizer is not None:
                        self.enh_stabilizer.reset()
                    if self.mask_stabilizer is not None:
                        self.mask_stabilizer.reset()
            else:
                # The initial capture was opened for the generic path. The
                # parallel path opens its own sequential decoder below, so do
                # not keep two FFmpeg/NVDEC readers alive for the same clip.
                if cap is not None:
                    try:
                        cap.release()
                    except Exception as _degrade_error:
                        _swallowed("roop/ProcessMgr.py:2447", _degrade_error, "fallback continued")
                        pass
                    cap = None

        self.output_to_file = output_method != "Virtual Camera"
        self.output_to_cam = output_method == "Virtual Camera" or output_method == "Both"

        # Writer creation happens HERE (before the pre-passes) because resume
        # detection may shift frame_start forward — the temporal/SAM2/track
        # pre-passes and the 2-pass stabilizer must then only scan the frames
        # that still need encoding.
        if self.output_to_file:
            use_resume = (not is_awebp) and os.environ.get('ROOP_RESUME', '1') == '1'
            if use_resume:
                from roop.segment_writer import SegmentedVideoWriter
                self.videowriter = SegmentedVideoWriter(
                    target_video, (width, height), fps,
                    codec=roop.globals.video_encoder, crf=roop.globals.video_quality,
                    source_video=source_video, frame_start=frame_start, frame_end=frame_end,
                    signature=str(getattr(roop.globals, '_run_signature', '') or ''),
                    checkpoint_callback=getattr(
                        roop.globals, '_checkpoint_segment_callback', None))
                skip = self.videowriter.resume_frames
                if skip >= frame_count > 0:
                    # Everything was already encoded by the interrupted run —
                    # just finalize (concat) and return; the caller's audio
                    # restore / renaming flow proceeds as if freshly rendered.
                    print(f'[Resume] all {frame_count} frames were already encoded '
                          f'by a previous run — finalizing without re-rendering.')
                    self.videowriter.close()
                    self.videowriter = None
                    if cap is not None:
                        cap.release()
                    return
                if skip > 0:
                    print(f'[Resume] found {skip} already-encoded frames from an '
                          f'interrupted run — resuming at frame {frame_start + skip}. '
                          f'(Delete {os.path.basename(target_video)}.resume.json to force a fresh render.)')
                    frame_start += skip
                    frame_count -= skip
            else:
                codec = roop.globals.video_encoder
                if (hardware_stream_enabled() and
                        codec in {'h264_nvenc', 'hevc_nvenc', 'av1_nvenc'}):
                    self.videowriter = NVHardwareVideoWriter(
                        target_video,
                        width,
                        height,
                        fps,
                        audio_source=None,
                        codec=codec,
                        crf=roop.globals.video_quality,
                    )
                else:
                    self.videowriter = FFMPEG_VideoWriter(
                        target_video,
                        (width, height),
                        fps,
                        codec=codec,
                        crf=roop.globals.video_quality,
                        audiofile=None,
                    )
        if self.output_to_cam:
            self.streamwriter = StreamWriter((width, height), int(fps))

        # 2-pass stabilization, pass 1: precompute smoothed kps sequentially so
        # pass 2 (the swap) can run multi-threaded. Done before auto-tuning so the
        # tuner calibrates the real pass-2 workload.
        if use_2pass:
            self._precomputed_kps = self._precompute_stabilized_kps(
                source_video, awebp_frames, frame_start, frame_end, frame_count)
            self._precomputed_mode = True
            print(f"[Stabilize] 2-pass: precomputed smoothed kps for "
                  f"{len(self._precomputed_kps)} frames; pass 2 runs multi-threaded.")
            self._log_memory_stage('phase3:stabilization-prepass-complete')

        # The caller's explicit thread setting remains authoritative.  When the
        # setting is automatic, the runtime profile has already selected a
        # bounded worker count above. GPU context limits do not force the host
        # writer/compositor to collapse to one worker.
        self.total_frames = frame_count
        self.num_threads = threads
        self.processing_threads = self.num_threads
        _configure_opencv_worker_threads(self.num_threads)
        self.frames_queue = []
        self.processed_queue = []
        # A little buffering per thread smooths variable per-frame times so the
        # reader/writer don't stall worker threads (matters now that CUDA runs
        # workers concurrently instead of serialised behind one GPU lock).
        qdepth = self._runtime_queue_depth if self._runtime_queue_depth is not None else (1 if threads <= 1 else 3)
        qdepth = max(1, min(4, int(qdepth)))
        if self._runtime_in_flight_frames is not None:
            qdepth = min(qdepth, max(1, int(self._runtime_in_flight_frames)))
        # An explicit output-buffer choice is useful for a slow CPU encoder or
        # a RAM-tight laptop. Keep the automatic runtime bound unless the user
        # deliberately supplies this override.
        try:
            output_qdepth = int(os.environ.get('ROOP_OUTPUT_QUEUE_DEPTH', '') or '0')
            if output_qdepth > 0:
                qdepth = max(1, min(4, output_qdepth))
        except ValueError:
            pass
        for _ in range(threads):
            self.frames_queue.append(Queue(qdepth))
            self.processed_queue.append(Queue(qdepth))

        # SAM2 temporal-mask pre-pass: track the faces across the trimmed clip and
        # cache a full-frame mask per frame, so the (still parallel) swap below can
        # look them up. Opt-in — only runs when the SAM2 engine is selected.
        sam2_p = next((p for p in self.processors
                       if getattr(p, 'processorname', None) == 'mask_sam2'), None)
        if sam2_p is not None and not is_awebp:
            try:
                self._precompute_sam2(sam2_p, source_video, frame_start, frame_end, frame_count)
            except Exception as e:
                print(f'[SAM2] pre-pass failed ({e}); falling back to unmasked swap')
                sam2_p.precomputed = {}

        # Identity-lock pre-pass: track each person across the clip and assign a
        # single source per track, so the per-frame embedding match can't flip
        # identities mid-video. Opt-in, only for "selected" mode on real video.
        self._track_mode = False
        self._track_assignments = {}
        self._track_source_map = {}

        # Temporal detection pre-pass (anti-flicker): detect + track every frame
        # once, gap-fill short detection misses and smooth kps/lm106/bbox per
        # track; the (still parallel) swap pass then consumes the cached faces
        # per frame instead of re-detecting, so the swap can't blink out on a
        # missed detection. The same scan yields the identity-lock assignments,
        # so no separate tracking pass is needed when both are enabled.
        if self._temporal_mode:
            try:
                self._log_memory_stage('phase3:temporal-prepass-start')
                self._precompute_temporal(source_video, awebp_frames, frame_start, frame_end, frame_count)
                # `and self._temporal_mode`: the pre-pass turns itself off when it
                # found nothing usable, and then its identity assignments are empty
                # too — so fall through to the standalone track pass below rather
                # than locking identities off an empty scan.
                # Both selected modes: the swap branch that consumes the lock
                # already accepts ("selected", "selected_multi"), but this gate
                # admitted only "selected", so "Lock face identities" was inert
                # in exactly the "Selected people" case it exists for (two
                # people crossing) -- the toggle read as on, the pre-pass
                # printed its per-track assignment, and the swap ran per-frame.
                self._track_mode = (self._temporal_mode
                                    and roop.globals.track_identities
                                    and self.options.swap_mode in ("selected", "selected_multi")
                                    and len(self.target_face_datas) > 0)
                self._log_memory_stage('phase3:temporal-prepass-complete')
                self._release_replayed_analysis(frame_count)
            except Exception as e:
                print(f'[Temporal] detection pre-pass failed ({e}); using per-frame detection')
                self._temporal_mode = False
                self._temporal_faces = None
                self._temporal_covered = 0
                self._log_memory_stage('phase3:temporal-prepass-fallback')

        if (not self._temporal_mode and roop.globals.track_identities and not is_awebp
                and self.options.swap_mode in ("selected", "selected_multi")
                and len(self.target_face_datas) > 0):
            try:
                self._precompute_tracks(source_video, frame_start, frame_end, frame_count)
                self._track_mode = True
                self._log_memory_stage('phase3:tracking-prepass-complete')
            except Exception as e:
                print(f'[Track] identity pre-pass failed ({e}); using per-frame matching')
                self._track_mode = False
                self._log_memory_stage('phase3:tracking-prepass-fallback')

        self._log_memory_stage('phase4:before-main-processing')
        # Avoid generational collections in the frame hot loop.  Frame queues
        # are bounded and frame references are released at encode/checkpoint
        # flushes; the finalizer below performs the only explicit sweep.
        self._hot_loop_gc_was_enabled = gc.isenabled()
        if self._hot_loop_gc_was_enabled:
            gc.disable()

        progress_bar_format = PROGRESS_BAR_FORMAT
        try:
            scheduler_enabled = str(os.environ.get(
                'ROOP_UNIFIED_SCHEDULER', '1')).strip().lower() not in (
                    '0', 'false', 'no', 'off')
            use_unified_scheduler = bool(
                scheduler_enabled and self._runtime_scheduler is not None and
                self._runtime_scheduler.frame_pipeline_allowed(
                    stateful_stabilization=_streaming_stabilization) and
                not use_parallel_stab)
            if scheduler_enabled and self._runtime_scheduler is not None:
                print('[RuntimeScheduler] unified coordinator ON: '
                      f"mode={'frame' if use_unified_scheduler else 'ordered-chunk'}, "
                      f"workers={threads}, queue={self._runtime_scheduler.queue_capacity}, "
                      f"in_flight={self._runtime_scheduler.effective_inflight}", flush=True)
            # Cross-frame swap batching must also serve the ordered-chunk
            # stabilisation path. Construction used to live only inside the
            # sequential fallback, so perf_batch_swap was exported and then
            # silently ignored for the production stabilised render.
            # Keep the unified scheduler unchanged: it owns its own inference
            # coordination and may use a single inference worker.
            if not use_unified_scheduler:
                self._swap_batcher = self._make_swap_batcher(threads)
            # The batch path is only settled here, so this is the first point
            # the banner can name it; it also re-checks the selection snapshot
            # taken at initialize (selection_invariant=...).
            print(_runtime_banner.runtime_selection_line(self, 'video'), flush=True)
            if use_unified_scheduler:
                _inference_workers = 1
                self._active_inference_workers = _inference_workers
                print('[RuntimeScheduler] unified frame pipeline ON: '
                      f"cuda_owner={_inference_workers}, "
                      f"queue={self._runtime_scheduler.queue_capacity}, "
                      f"in_flight={self._runtime_scheduler.effective_inflight}, "
                      f"ram_budget={self._runtime_scheduler.budget.ram_budget_bytes // 2**20}MB"
                      + ('; continuous stabilizer FIFO (zero warm-up recompute)'
                         if _streaming_stabilization else ''),
                      flush=True)
                with ChunkedProgress(total=self.total_frames, desc='Processing',
                                     unit='frames', dynamic_ncols=True,
                                     bar_format=progress_bar_format) as progress:
                    self._run_unified_scheduler(
                        cap, awebp_frames, frame_start, frame_end, frame_count,
                        progress_cb=lambda: self.update_progress(progress),
                        threads=_inference_workers)
            elif use_parallel_stab:
                stab_threads = max(1, min(
                    threads, int(self._runtime_stab_workers or threads)))
                _active = [n for n, w in (("kps", _want_kps_stab), ("enhancer", _want_enh_stab),
                                          ("mask", _want_mask_stab)) if w]
                print(f"[Stabilize] parallel stabilization ON (threads={stab_threads}, warm-up overlap) — "
                      f"{' + '.join(_active)} smoothing run{'s' if len(_active) == 1 else ''} multi-threaded.")
                with ChunkedProgress(total=self.total_frames, desc='Processing', unit='frames', dynamic_ncols=True, bar_format=progress_bar_format) as progress:
                    self._run_stab_parallel(source_video, awebp_frames, frame_start, frame_end,
                                            frame_count, stab_threads, lambda: self.update_progress(progress))
            else:
                if is_awebp:
                    readthread = Thread(target=self.read_frames_webp_thread, args=(awebp_frames, frame_start, frame_end, threads))
                else:
                    readthread = Thread(target=self.read_frames_thread, args=(cap, frame_start, frame_end, threads))
                # daemon: the joins below are `join(timeout=...)`, so a thread that
                # somehow still overruns is ABANDONED, not waited for. Non-daemon,
                # such a thread then blocks interpreter shutdown forever — the
                # symptom that exposed the unbounded sentinel put. Belt and braces
                # with _post_sentinels: that one stops it wedging, this one stops a
                # wedge from being fatal.
                readthread.daemon = True
                readthread.start()

                writethread = Thread(target=self.write_frames_thread)
                writethread.daemon = True
                writethread.start()

                # Round-robin dispatch: worker i gets frames i, i+N, i+2N, so no
                # worker sees adjacent frames. Say so before the workers start —
                # `_dispatch_tracker` reads this to give each thread its own
                # tracker and to refuse coasting across the stride.
                self._dispatch_ordered = (threads <= 1)
                try:
                    with ChunkedProgress(total=self.total_frames, desc='Processing', unit='frames', dynamic_ncols=True, bar_format=progress_bar_format) as progress:
                        with ThreadPoolExecutor(thread_name_prefix='swap_proc', max_workers=self.num_threads) as executor:
                            futures = []
                            for threadindex in range(threads):
                                future = executor.submit(self.process_videoframes, threadindex, lambda: self.update_progress(progress))
                                futures.append(future)
                            for future in as_completed(futures):
                                future.result()
                finally:
                    if self._swap_batcher is not None:
                        self._swap_batcher.stop()
                        self._swap_batcher.report()
                        self._swap_batcher = None
                    # Join with timeouts so an exception path never leaves background
                    # threads running (or holding the videowriter open). Timeouts are a
                    # safety net: normally both threads exit quickly because workers have
                    # already set roop.globals.processing=False and sent their sentinels.
                    readthread.join(timeout=5)
                    writethread.join(timeout=10)
                    if self._writer_error is not None:
                        raise IOError(
                            "Roop Ultimate error: the frame output thread failed: "
                            f"{self._writer_error}") from self._writer_error
        finally:
            if self._swap_batcher is not None:
                self._swap_batcher.stop()
                self._swap_batcher.report()
                self._swap_batcher = None
            # Always release the capture and close writers regardless of which path ran
            # and whether it raised.  The write thread MUST be joined (above) before we
            # close videowriter, otherwise the pipe stdin close races with an in-flight
            # write_frame() call and corrupts the temp file.
            if cap is not None:
                cap.release()
            if self.output_to_file and self.videowriter is not None:
                # Include encoder trailer/segment concat in the measured
                # encode cost; write_frame alone misses the final lifecycle.
                writer = self.videowriter
                failed = self._writer_error is not None or sys.exc_info()[1] is not None
                try:
                    with _prof('encode_finalize'):
                        if failed and hasattr(writer, 'abort'):
                            writer.abort()
                        else:
                            writer.close()
                except Exception as exc:
                    if failed:
                        bar_write(f'[ProcessMgr] output cleanup failed after render error: {exc}')
                    else:
                        raise
                self.videowriter = None
            if self.output_to_cam and self.streamwriter is not None:
                self.streamwriter.Close()
                self.streamwriter = None
            self.frames_queue.clear()
            self.processed_queue.clear()
            self._precomputed_mode = False
            self._precomputed_kps = None
            self._temporal_mode = False
            self._temporal_faces = None
            self._temporal_covered = 0
            self._track_assignments = None
            self._track_source_map = None
            self._lipsync_audio = None
            # release_face_analyser_aux() intentionally leaves a detector-only
            # object for the main pass. Do not let that reduced object escape
            # into post-run callers: clear the pool so the next analysis call
            # rebuilds the normal recognition/landmark contract lazily.
            if getattr(self, '_replay_analysis_released', False):
                face_util.release_face_analyser()
                self._replay_analysis_released = False
            if (getattr(self, '_small_card_decode_policy', None) or {}).get('changed'):
                if self._small_card_decode_env_present:
                    os.environ['ROOP_NVDEC'] = self._small_card_decode_env_previous
                else:
                    os.environ.pop('ROOP_NVDEC', None)
                self._small_card_decode_policy = None
            if getattr(self, '_hot_loop_gc_was_enabled', False):
                gc.enable()
            gc.collect()
            self._log_memory_stage('phase3:run-cleanup-complete')
            self._psutil_proc = None
            self._active_inference_workers = None
            if self._runtime_monitor is not None:
                self._runtime_summary = self._runtime_monitor.finish(
                    queue_depths=self._runtime_queue_snapshot(),
                    worker_utilization_pct=self._runtime_worker_utilization())
                if self._runtime_monitor.diagnostics:
                    print('[RuntimeMonitor] summary=' + str({
                        key: self._runtime_summary.get(key) for key in (
                            'end_to_end_fps', 'stage_fps', 'stage_latency_ms',
                            'cpu_utilization_pct', 'p_core_utilization_pct',
                            'e_core_utilization_pct', 'gpu_utilization_pct',
                            'vram_pressure_pct', 'ram_utilization_pct',
                            'queue_depths', 'worker_utilization_pct', 'bottleneck')
                    }), flush=True)
                # Do this in the finally block too: an exception in a worker or
                # writer must not leave a global timing hook attached to the
                # next video run.
                set_runtime_monitor(None)
            if self._runtime_scheduler is not None:
                self._runtime_scheduler_summary = self._runtime_scheduler.snapshot()
                if self._runtime_monitor is not None and self._runtime_monitor.diagnostics:
                    print('[RuntimeScheduler] summary=' + str({
                        key: self._runtime_scheduler_summary.get(key) for key in (
                            'decoded', 'processed', 'encoded', 'queue_capacity',
                            'effective_inflight', 'max_queue_depths', 'bottleneck',
                            'actions', 'errors')
                    }), flush=True)
                self._runtime_scheduler = None
            if self._stage_profiler is not None:
                self._stage_profile_report = self._stage_profiler.report()
                # Normal completion reaches _prof_report below; this fallback
                # makes exception cleanup safe without leaving a global hook.
                self._stage_profiler.print_report()
                set_detailed_profiler(None)
        _prof_report()
        _audit_report()
        self._report_smoother_summaries()
