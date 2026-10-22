"""Unified hardware video I/O: zero-copy NVDEC/NVENC through PyNvVideoCodec, an optional Intel QSV
keyframe decoder, and the existing ffmpeg / OpenCV pipeline as the fallback for every one of them.

Placement: this is ``roop/video_io_hw.py``, not ``app/core/video_io_hw.py``. ``roop/core.py`` is a module and a
package named ``core`` anywhere on its import path has already shadowed it once (5609800); every I/O module lives
beside ``nvdec_reader.py`` and ``video_stream.py``.

What each path does (all measured on the RTX 4070 with PyNvVideoCodec 2.2.3, see docs/CHANGELOG.md 2026-10-21)

``NvZeroCopyReader``      ``ThreadedDecoder(use_device_memory=True, RGBP)``; ``torch.from_dlpack`` hands back a CUDA
                          ``[3, H, W]`` uint8 RGB tensor with no host round trip. **The tensor is a view into the
                          decoder's output ring, which holds two surfaces however large ``buffer_size`` is**: a tensor
                          is overwritten by the decode after next (measured: 39 of 40 held tensors changed).
                          ``HwFrame`` therefore raises ``StaleFrameError`` instead of returning recycled pixels once
                          the reader has moved on; ``frame.detach()`` makes a device-side copy that outlives it. A raw tensor taken out of
                          ``frame.tensor`` bypasses that guard - after ``close()`` it points at freed device memory
                          (an illegal memory access that poisons the CUDA context), so do not hold one.
``NvSparseKeyframeReader`` demuxes packets and feeds ONLY true random-access packets to an NVDEC decoder. "True" is
                          decided from the NAL unit (H.264 type 5, HEVC 19/20), NOT from the container's key flag:
                          Love.mp4 flags 7 of 36 sync samples that are not IDR (NAL type 1), and decoding those
                          alone returns the wrong picture (3 of 34 frames differed from a full decode by up to 232/255).
                          IDR-only output was bit-identical to a full decode on all three clips tried.
``NvZeroCopyWriter``      ``CreateEncoder`` (H.264 / HEVC / AV1 on NVENC). A CUDA ``[3, H, W]`` RGB tensor is turned
                          into NV12 on the GPU (BT.709 at >= 720p, BT.601 below, which is what the decoder inverts
                          with: the wrong one cost 3.7 dB) and encoded from VRAM; the muxer is
                          ``FFmpegMuxer`` (video only), then ``finalize()`` stream-copies the source audio through the
                          same ``finalize_render`` the rest of the app uses, so the output is verified before the
                          intermediate is deleted.
``IntelQsvReader``        ``ffmpeg -c:v <codec>_qsv`` pipe for keyframes (Stage 1 sparse pre-scan co-processing).
                          **UNVERIFIED on hardware**: this machine has no Intel iGPU in use and the bundled ffmpeg lists
                          no ``*_qsv`` decoder. The command, the probe and the fallback are tested; a QSV decode is not.

Every open_* function walks a ladder and a rung that cannot start (package missing, no CUDA, codec unsupported,
first frame does not decode) falls to the next, ending at the pipeline the app already uses. ``ROOP_HW_IO=0`` forces
the last rung; ``ROOP_QSV=0`` removes QSV.

Host copies are never hidden: ``HwFrame.bgr()`` is the one place a frame leaves VRAM, and it is counted.
"""
from __future__ import annotations

import collections
import ctypes
import gc
import json
import os
import queue
import subprocess
import threading
import time
from fractions import Fraction
from typing import Any, Deque, Dict, Iterator, List, Optional, Tuple

import numpy as np

from roop.degrade import swallowed as _swallowed

BACKEND_NVDEC_DLPACK = "NVDEC-DLPack"
BACKEND_NVDEC_PIPE = "NVDEC-Pipe"       # the existing ``ffmpeg -hwaccel cuda`` raw pipe (nvdec_reader.py)
BACKEND_QSV = "QSV-oneVPL"
BACKEND_CPU = "CPU-Fallback"
BACKEND_NVENC_DIRECT = "NVENC-Direct"
BACKEND_NVENC_PIPE = "NVENC-Pipe"       # the existing ffmpeg h264_nvenc pipe (video_stream.NVHardwareVideoWriter)

#: A read that waited longer than this for the decoder counts as starved (the decode thread had nothing ready).
STARVE_S = 0.002
_SATURATION_WINDOW = 512
#: Search window for the NAL scan of one packet. IDR slices follow SEI/SPS/PPS/VPS, all of which are small.
_NAL_SCAN_BYTES = 8192


class HwUnavailable(RuntimeError):
    """A hardware rung cannot be used; the message says why. Callers fall to the next rung."""


class StaleFrameError(RuntimeError):
    """A zero-copy tensor was used after its reader moved on (the decoder had recycled the surface)."""


def encoder_label(codec: str) -> str:
    """The telemetry label for an ffmpeg-pipe encoder by its codec name (the pipe writers are the existing ones)."""
    c = (codec or "").lower()
    if "nvenc" in c:
        return BACKEND_NVENC_PIPE
    if "qsv" in c:
        return BACKEND_QSV
    return BACKEND_CPU


def hw_io_wanted() -> bool:
    return os.environ.get("ROOP_HW_IO", "").strip() != "0"


def qsv_wanted() -> bool:
    return hw_io_wanted() and os.environ.get("ROOP_QSV", "").strip() != "0"


# --------------------------------------------------------------------------------------------------------------
# Telemetry
# --------------------------------------------------------------------------------------------------------------

class VideoIoTelemetry:
    """Which decode/encode backend is ACTIVE and what it moved over PCIe. Written by the readers and writers,
    read by ``runtime_state`` (the HARDWARE section), the /ws/telemetry frame and the per-job report.

    ``pcie_saved_bytes`` is a labelled ESTIMATE against a host-staged pipeline: every frame handed out as a device
    tensor would have been uploaded once, and every frame encoded straight from VRAM would have been downloaded
    once. ``pcie_moved_bytes`` is what this module actually copied (``HwFrame.bgr()`` and host-frame uploads).
    ``decode_queue_saturation`` is the share of reads served without waiting on the decoder (1.0 = the decode
    thread stays ahead of the consumer, 0.0 = the consumer waits for every frame); None before any NVDEC read."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._decoders: Dict[str, str] = {}
            self._encoders: Dict[str, str] = {}
            self._frames_decoded = 0
            self._frames_encoded = 0
            self._saved = 0
            self._moved = 0
            self._reads: Deque[bool] = collections.deque(maxlen=_SATURATION_WINDOW)
            self._fallbacks: List[str] = []

    def register_decoder(self, tag: str, backend: str) -> None:
        with self._lock:
            self._decoders[str(tag)] = backend

    def register_encoder(self, tag: str, backend: str) -> None:
        with self._lock:
            self._encoders[str(tag)] = backend

    def note_fallback(self, what: str) -> None:
        with self._lock:
            if len(self._fallbacks) < 20:
                self._fallbacks.append(str(what)[:300])

    def note_decoded(self, nbytes: int, device: bool) -> None:
        with self._lock:
            self._frames_decoded += 1
            if device:
                self._saved += int(nbytes)

    def note_encoded(self, nbytes: int, device: bool) -> None:
        with self._lock:
            self._frames_encoded += 1
            if device:
                self._saved += int(nbytes)

    def note_moved(self, nbytes: int) -> None:
        with self._lock:
            self._moved += int(nbytes)

    def note_read(self, wait_s: float) -> None:
        with self._lock:
            self._reads.append(wait_s > STARVE_S)

    @staticmethod
    def _join(values: Dict[str, str]) -> str:
        seen: List[str] = []
        for v in values.values():
            if v not in seen:
                seen.append(v)
        return " + ".join(seen)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            reads = len(self._reads)
            sat = None if not reads else round(1.0 - sum(self._reads) / reads, 3)
            return {
                "decode_backend": self._join(self._decoders) or None,
                "encode_backend": self._join(self._encoders) or None,
                "decoders": dict(self._decoders),
                "encoders": dict(self._encoders),
                "frames_decoded": self._frames_decoded,
                "frames_encoded": self._frames_encoded,
                "pcie_saved_bytes": self._saved,
                "pcie_moved_bytes": self._moved,
                "pcie_saved_is_estimate": True,
                "decode_queue_saturation": sat,
                "decode_queue_reads": reads,
                "fallbacks": list(self._fallbacks),
            }

    def flat(self) -> Dict[str, Any]:
        """Flat, coarse fields for the 4 Hz telemetry frame: whole MB and whole percent, so an idle or steady
        run is silent for the change-only sampler."""
        s = self.snapshot()
        return {
            "io_decode": s["decode_backend"] or "",
            "io_encode": s["encode_backend"] or "",
            "io_pcie_saved_mb": int(s["pcie_saved_bytes"] // (1024 * 1024)),
            "io_queue_sat": None if s["decode_queue_saturation"] is None
            else int(round(s["decode_queue_saturation"] * 100)),
        }


telemetry = VideoIoTelemetry()


def snapshot() -> Dict[str, Any]:
    return telemetry.snapshot()


# --------------------------------------------------------------------------------------------------------------
# Optional-import gates
# --------------------------------------------------------------------------------------------------------------

_nvc_cache: Dict[str, Any] = {}


def _load_nvc():
    """``(PyNvVideoCodec, torch)`` or HwUnavailable with the reason. Cached; the failure is cached too."""
    if "err" in _nvc_cache:
        raise HwUnavailable(_nvc_cache["err"])
    if "mods" in _nvc_cache:
        return _nvc_cache["mods"]
    try:
        import torch
        if not torch.cuda.is_available():
            raise HwUnavailable("torch has no CUDA device")
        import PyNvVideoCodec as nvc
        _nvc_cache["mods"] = (nvc, torch)
        return _nvc_cache["mods"]
    except HwUnavailable as exc:
        _nvc_cache["err"] = str(exc)
    except Exception as exc:        # ImportError, a driver too old for the wheel, ...
        _nvc_cache["err"] = f"PyNvVideoCodec/torch unavailable: {type(exc).__name__}: {exc}"
        _swallowed("roop/video_io_hw.py:load-nvc", exc, "zero-copy NVDEC/NVENC unavailable; using the ffmpeg pipeline")
    raise HwUnavailable(_nvc_cache["err"])


def capabilities(probe: bool = False) -> Dict[str, Any]:
    """What the machine can do. With ``probe=False`` (the telemetry poll) nothing is imported: NVDEC reports what an
    earlier open found, or "not probed". QSV reports only earlier ``qsv_probe`` results."""
    out: Dict[str, Any] = {"hw_io_enabled": hw_io_wanted()}
    if probe or "mods" in _nvc_cache or "err" in _nvc_cache:
        try:
            nvc, torch = _load_nvc()
            out["nvdec_dlpack"] = {"available": True, "pynvvideocodec": getattr(nvc, "__version__", "?"),
                                   "device": torch.cuda.get_device_name(0)}
        except HwUnavailable as exc:
            out["nvdec_dlpack"] = {"available": False, "reason": str(exc)}
    else:
        out["nvdec_dlpack"] = {"available": None, "reason": "not probed yet"}
    with _qsv_lock:
        out["qsv"] = {"probed": dict(_qsv_cache), "enabled": qsv_wanted(), "verified_on_hardware": False}
    return out


# --------------------------------------------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------------------------------------------

class HwFrame:
    """One decoded frame. ``tensor`` is CUDA ``[3, H, W]`` uint8 RGB; ``bgr()`` is host ``[H, W, 3]`` BGR (cv2's
    convention). A zero-copy frame is valid until its reader's next read/close (``StaleFrameError`` after that);
    ``detach()`` copies it on the device so it can be kept."""

    __slots__ = ("index", "pts", "_tensor", "_host", "_owner", "_gen", "_nbytes", "decode_stats", "is_keyframe")

    def __init__(self, index: int, pts: Optional[int], *, tensor: Any = None, host: Optional[np.ndarray] = None,
                 owner: Any = None, gen: int = 0, nbytes: int = 0,
                 decode_stats: Optional[Dict[str, Any]] = None, is_keyframe: bool = True) -> None:
        self.index = index
        self.pts = pts
        self._tensor = tensor
        self._host = host
        self._owner = owner
        self._gen = gen
        self._nbytes = nbytes
        self.decode_stats = decode_stats
        self.is_keyframe = bool(is_keyframe)

    def _check(self) -> None:
        if self._owner is not None and self._owner._gen != self._gen:
            raise StaleFrameError(
                f"frame {self.index} was read before the reader advanced; the decoder has recycled its surface. "
                f"Use frame.detach() to keep a frame across reads.")

    @property
    def tensor(self):
        """The CUDA tensor. A host-decoded frame (QSV / CPU) is uploaded once, counted, and cached."""
        self._check()
        if self._tensor is None:
            _, torch = _load_nvc()
            host = self._host
            self._tensor = torch.from_numpy(np.ascontiguousarray(host[..., ::-1])).to("cuda").permute(2, 0, 1) \
                .contiguous()
            telemetry.note_moved(int(host.nbytes))
        return self._tensor

    def bgr(self) -> np.ndarray:
        self._check()
        if self._host is None:
            t = self._tensor
            self._host = t.flip(0).permute(1, 2, 0).contiguous().cpu().numpy()
            telemetry.note_moved(int(self._host.nbytes))
        return self._host

    def detach(self) -> "HwFrame":
        """A copy that outlives the reader's next read (device-to-device; no PCIe)."""
        self._check()
        if self._tensor is not None:
            return HwFrame(self.index, self.pts, tensor=self._tensor.clone(), host=self._host,
                           nbytes=self._nbytes, decode_stats=self.decode_stats, is_keyframe=self.is_keyframe)
        return HwFrame(self.index, self.pts, host=self._host, nbytes=self._nbytes,
                       decode_stats=self.decode_stats, is_keyframe=self.is_keyframe)


# --------------------------------------------------------------------------------------------------------------
# NAL classification (a pure function so it is tested without a GPU)
# --------------------------------------------------------------------------------------------------------------

def is_random_access_packet(data: bytes, codec: str) -> Optional[bool]:
    """Is this Annex-B packet a picture that decodes with no earlier picture? None = this codec is not classified
    here and the caller should trust the container's key flag.

    H.264: a coded slice of an IDR picture (NAL type 5). HEVC: IDR_W_RADL / IDR_N_LP (19 / 20). CRA (21) is left
    out on purpose: it may carry leading pictures that reference frames before it. An I-slice NAL type 1 that the
    container merely flags as a sync sample is NOT accepted - that is the Love.mp4 defect."""
    c = (codec or "").lower()
    if c not in ("h264", "avc", "hevc", "h265"):
        return None
    h264 = c in ("h264", "avc")
    n = min(len(data), _NAL_SCAN_BYTES)
    i = 0
    while i + 3 < n:
        if data[i] == 0 and data[i + 1] == 0 and data[i + 2] == 1:
            header = data[i + 3]
            nal = (header & 0x1F) if h264 else ((header >> 1) & 0x3F)
            if h264 and nal == 5:
                return True
            if not h264 and nal in (19, 20):
                return True
            i += 3
        else:
            i += 1
    return False


def _codec_name(nvc, codec_id) -> str:
    text = str(codec_id).lower()
    for name in ("h264", "hevc", "av1", "vp9", "vp8", "mpeg"):
        if name in text:
            return name
    return text


# --------------------------------------------------------------------------------------------------------------
# NVIDIA readers
# --------------------------------------------------------------------------------------------------------------

class NvZeroCopyReader:
    """Sequential NVDEC reader returning CUDA tensors (single consumer thread).

    ``read()`` -> HwFrame | None at end of stream. ``read_batch(n)`` -> list (n <= buffer_size); the whole batch
    is valid until the NEXT read of either kind. ``start_frame`` seeks at open (ThreadedDecoder's own seek)."""

    backend = BACKEND_NVDEC_DLPACK

    def __init__(self, path: str, *, gpu_id: int = 0, buffer_size: int = 16, start_frame: int = 0,
                 tag: str = "decode") -> None:
        nvc, torch = _load_nvc()
        self._torch = torch
        self.path = path
        self.buffer_size = max(1, int(buffer_size))
        self._gen = 0
        self._pos = int(start_frame)
        self._closed = False
        try:
            self._dec = nvc.CreateThreadedDecoder(
                encSource=str(path), bufferSize=self.buffer_size, gpuid=int(gpu_id), useDeviceMemory=True,
                decoderCacheSize=1, outputColorType=nvc.OutputColorType.RGBP, startFrame=int(start_frame))
            meta = self._dec.get_stream_metadata()
        except Exception as exc:
            raise HwUnavailable(f"NVDEC could not open {os.path.basename(path)}: {type(exc).__name__}: {exc}") from exc
        self.width = int(getattr(meta, "width", 0))
        self.height = int(getattr(meta, "height", 0))
        self.fps = float(getattr(meta, "average_fps", 0) or 0)
        self.num_frames = int(getattr(meta, "num_frames", 0) or 0)
        self.codec = str(getattr(meta, "codec_name", ""))
        self._frame_bytes = self.width * self.height * 3
        self._peek: Optional[List[Any]] = None
        self._tag = tag
        # Prove the first frame decodes NOW so open_reader() can still fall back; a failure after this is a real error.
        try:
            self._peek = self._dec.get_batch_frames(1)
        except Exception as exc:
            self._shutdown()
            raise HwUnavailable(f"NVDEC could not decode the first frame of {os.path.basename(path)}: "
                                f"{type(exc).__name__}: {exc}") from exc
        if not self._peek:
            self._shutdown()
            raise HwUnavailable(f"NVDEC returned no frames for {os.path.basename(path)}")
        telemetry.register_decoder(tag, self.backend)

    def _wrap(self, f: Any) -> HwFrame:
        t = self._torch.from_dlpack(f)
        frame = HwFrame(self._pos, int(getattr(f, "timestamp", 0)), tensor=t, owner=self, gen=self._gen,
                        nbytes=self._frame_bytes)
        self._pos += 1
        telemetry.note_decoded(self._frame_bytes, device=True)
        return frame

    def read(self) -> Optional[HwFrame]:
        if self._closed:
            return None
        self._gen += 1                      # everything handed out before this call is now stale
        t0 = time.perf_counter()
        if self._peek is not None:
            raw, self._peek = self._peek, None
            telemetry.note_read(0.0)
        else:
            raw = self._dec.get_batch_frames(1)
            telemetry.note_read(time.perf_counter() - t0)
        return self._wrap(raw[0]) if raw else None

    def read_batch(self, n: int) -> List[HwFrame]:
        if self._closed:
            return []
        n = max(1, min(int(n), self.buffer_size))
        self._gen += 1
        t0 = time.perf_counter()
        raw: List[Any] = []
        if self._peek is not None:
            raw, self._peek = list(self._peek), None
        if len(raw) < n:
            raw += list(self._dec.get_batch_frames(n - len(raw)) or [])
        telemetry.note_read(time.perf_counter() - t0)
        return [self._wrap(f) for f in raw]

    def __iter__(self) -> Iterator[HwFrame]:
        while True:
            f = self.read()
            if f is None:
                return
            yield f

    def _shutdown(self) -> None:
        dec, self._dec = getattr(self, "_dec", None), None
        if dec is not None:
            try:
                dec.end()
            except Exception as exc:
                _swallowed("roop/video_io_hw.py:reader-end", exc, "decoder end() failed; releasing anyway")

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._gen += 1
            self._shutdown()

    def __enter__(self) -> "NvZeroCopyReader":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class NvSparseKeyframeReader:
    """Decode only the random-access pictures of a clip, in VRAM. ``keyframes()`` yields independent
    (detached) HwFrames whose ``index`` is the exact display index (the rank of the packet's pts among all packets),
    not an fps estimate. ``keyframe_table()`` lists them without decoding."""

    backend = BACKEND_NVDEC_DLPACK

    def __init__(self, path: str, *, gpu_id: int = 0, tag: str = "stage1-keyframes",
                 adaptive_midpoints: bool = True, max_gop_gap: int = 60,
                 enable_decode_stats: bool = True) -> None:
        nvc, torch = _load_nvc()
        self._nvc, self._torch = nvc, torch
        self.path = path
        self.gpu_id = int(gpu_id)
        self.adaptive_midpoints = bool(adaptive_midpoints)
        self.max_gop_gap = max(1, int(max_gop_gap))
        self.enable_decode_stats = bool(enable_decode_stats)
        self._pts_by_rank: Dict[int, int] = {}
        self._rank_by_pts: Dict[int, int] = {}
        self._total_packet_count: int = 0
        try:
            dmx = nvc.CreateDemuxer(str(path))
            self.width, self.height = int(dmx.Width()), int(dmx.Height())
            self.fps = float(dmx.FrameRate())
            self._codec_id = dmx.GetNvCodecId()
        except Exception as exc:
            raise HwUnavailable(f"NVDEC demux could not open {os.path.basename(path)}: "
                                f"{type(exc).__name__}: {exc}") from exc
        self.codec = _codec_name(nvc, self._codec_id)
        self._table: Optional[List[Tuple[int, int]]] = None
        telemetry.register_decoder(tag, self.backend)

    def _classify(self, packet) -> bool:
        verdict = None
        try:
            if self.codec in ("h264", "hevc"):
                n = min(int(packet.bsl), _NAL_SCAN_BYTES)
                verdict = is_random_access_packet(ctypes.string_at(packet.bsl_data, n), self.codec)
        except Exception as exc:
            _swallowed("roop/video_io_hw.py:nal-scan", exc, "NAL scan failed; trusting the container key flag")
        return bool(packet.key) if verdict is None else verdict

    def _scan(self) -> Tuple[List[Tuple[int, bool]], List[int]]:
        dmx = self._nvc.CreateDemuxer(str(self.path))
        pts_all: List[int] = []
        keys: List[Tuple[int, bool]] = []     # (pts, accepted)
        while True:
            p = dmx.Demux()
            if p.bsl == 0:
                break
            pts_all.append(int(p.pts))
            if p.key:
                keys.append((int(p.pts), self._classify(p)))
        sorted_pts = sorted(pts_all)
        self._total_packet_count = len(sorted_pts)
        self._rank_by_pts = {pts: i for i, pts in enumerate(sorted_pts)}
        self._pts_by_rank = {i: pts for i, pts in enumerate(sorted_pts)}
        return keys, pts_all

    def keyframe_table(self) -> List[Tuple[int, int]]:
        """``[(display_index, pts)]`` of the accepted keyframes."""
        if self._table is None:
            keys, pts_all = self._scan()
            rank = self._rank_by_pts
            self._table = [(rank[pts], pts) for pts, ok in keys if ok]
        return list(self._table)

    def sample_table(self, max_gop_gap: Optional[int] = None) -> List[Tuple[int, int, bool]]:
        """``[(display_index, pts, is_keyframe)]`` of keyframes plus adaptive midpoint frames
        whenever the gap between keyframes exceeds max_gop_gap (default 60)."""
        table = self.keyframe_table()
        if not table:
            return []
        gap_limit = self.max_gop_gap if max_gop_gap is None else max(1, int(max_gop_gap))
        samples: List[Tuple[int, int, bool]] = []
        for i, (idx, pts) in enumerate(table):
            samples.append((idx, pts, True))
            if i + 1 < len(table):
                next_idx, next_pts = table[i + 1]
                gap = next_idx - idx
                if gap > gap_limit:
                    # Adaptive midpoint frame only when GOP length exceeds 60 frames
                    mid_idx = idx + gap // 2
                    mid_pts = self._pts_by_rank.get(mid_idx, int(round((pts + next_pts) / 2)))
                    samples.append((mid_idx, mid_pts, False))
            elif self._total_packet_count > 0:
                gap = self._total_packet_count - idx
                if gap > gap_limit:
                    mid_idx = idx + gap // 2
                    mid_pts = self._pts_by_rank.get(mid_idx, pts + 1)
                    samples.append((mid_idx, mid_pts, False))
        seen = set()
        deduped = []
        for item in sorted(samples, key=lambda x: x[0]):
            if item[0] not in seen:
                seen.add(item[0])
                deduped.append(item)
        return deduped

    def sample_frames(self, max_frames: Optional[int] = None, enable_stats: Optional[bool] = None) -> Iterator[HwFrame]:
        """Decode keyframes plus adaptive midpoints directly into VRAM, attaching decode statistics."""
        return self.keyframes(max_frames=max_frames, sample_midpoints=True, enable_stats=enable_stats)

    def keyframes(self, max_frames: Optional[int] = None, sample_midpoints: bool = False,
                  enable_stats: Optional[bool] = None) -> Iterator[HwFrame]:
        nvc, torch = self._nvc, self._torch
        use_stats = bool(self.enable_decode_stats if enable_stats is None else enable_stats)
        table = self.sample_table() if sample_midpoints else [(idx, pts, True) for idx, pts in self.keyframe_table()]
        if not table:
            return

        has_midpoints = any(not is_key for _, _, is_key in table)
        frame_bytes = self.width * self.height * 3

        # When midpoints or decode stats are requested, use SimpleDecoder to access frames in VRAM
        if has_midpoints or use_stats:
            try:
                dec = nvc.CreateSimpleDecoder(self.path, self.gpu_id, 0, 0, True, 0, 0, False, 1,
                                             nvc.OutputColorType.RGBP, False, use_stats)
                n = 0
                for idx, pts, is_key in table:
                    try:
                        raw = dec[idx]
                        t = torch.from_dlpack(raw).clone()
                        stats = raw.ParseDecodeStats() if use_stats else None
                        telemetry.note_decoded(frame_bytes, device=True)
                        yield HwFrame(idx, pts, tensor=t, nbytes=frame_bytes, decode_stats=stats, is_keyframe=is_key)
                        n += 1
                        if max_frames is not None and n >= max_frames:
                            return
                    except Exception as exc:
                        _swallowed("roop/video_io_hw.py:sample_decode", exc, f"skipping frame {idx}")
                return
            except Exception as exc:
                _swallowed("roop/video_io_hw.py:simple_decoder_fallback", exc, "falling back to demuxer decoder")

        index_of = {pts: idx for idx, pts, _ in table}
        dec = nvc.CreateDecoder(gpuid=self.gpu_id, codec=self._codec_id, usedevicememory=True,
                                outputColorType=nvc.OutputColorType.RGBP,
                                latency=nvc.DisplayDecodeLatencyType.LOW)
        flush = nvc.PacketData()
        flush.bsl = 0
        dmx = nvc.CreateDemuxer(str(self.path))
        n = 0
        while True:
            p = dmx.Demux()
            if p.bsl == 0:
                break
            if not p.key or int(p.pts) not in index_of:
                continue
            # one packet in, then an empty packet to drain it: without the drain the last keyframes stay inside
            # the decoder (4 of 6 on d1.mp4); with it, 29/29, 6/6 and 220/220 matched a full decode bit for bit.
            for chunk in (dec.Decode(p), dec.Decode(flush)):
                for f in chunk:
                    ts = int(f.timestamp)
                    if ts not in index_of:
                        continue
                    telemetry.note_decoded(frame_bytes, device=True)
                    yield HwFrame(index_of[ts], ts, tensor=torch.from_dlpack(f).clone(), nbytes=frame_bytes, is_keyframe=True)
                    n += 1
                    if max_frames is not None and n >= max_frames:
                        return

    def close(self) -> None:
        pass

    def __enter__(self) -> "NvSparseKeyframeReader":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# --------------------------------------------------------------------------------------------------------------
# ffmpeg pipe reader: the QSV rung and the software rung share one implementation
# --------------------------------------------------------------------------------------------------------------

def _ffmpeg_binary() -> str:
    from roop.ffmpeg_writer import FFMPEG_BINARY
    return str(FFMPEG_BINARY)


def _ffprobe_binary() -> str:
    directory = os.path.dirname(_ffmpeg_binary())
    return os.path.join(directory, "ffprobe") if directory else "ffprobe"


def _popen_flags() -> Dict[str, Any]:
    return {"creationflags": 0x08000000} if os.name == "nt" else {}


def probe_geometry(path: str) -> Tuple[int, int, float, str]:
    """(width, height, fps, codec_name) from ffprobe."""
    out = subprocess.check_output(
        [_ffprobe_binary(), "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,codec_name", "-of", "json", path],
        timeout=30, **_popen_flags())
    s = json.loads(out)["streams"][0]
    num, den = str(s.get("r_frame_rate", "0/1")).split("/")
    return int(s["width"]), int(s["height"]), float(num) / max(1.0, float(den)), str(s.get("codec_name", ""))


def keyframe_display_indices(path: str) -> List[int]:
    """Display index of every picture the software decoder emits under ``-skip_frame nokey`` (the same libavcodec
    rule the keyframe pipe uses), as the rank of its pts among all packets' pts.

    NOT the container's sync-sample flag: on an open-GOP clip ffprobe flags 5 packets while the decoder emits 3
    key pictures, so pairing frames to flags by order mislabels them (found by test_video_io_hw)."""
    def run(args: List[str]) -> Any:
        return json.loads(subprocess.check_output([_ffprobe_binary(), "-v", "error", *args, "-of", "json", path],
                                                  timeout=120, **_popen_flags()))
    packets = run(["-select_streams", "v:0", "-show_entries", "packet=pts"]).get("packets", [])
    pts = sorted(int(p["pts"]) for p in packets if p.get("pts") not in (None, "N/A"))
    rank = {v: i for i, v in enumerate(pts)}
    frames = json.loads(subprocess.check_output(
        [_ffprobe_binary(), "-v", "error", "-skip_frame", "nokey", "-select_streams", "v:0", "-show_entries",
         "frame=pts", "-of", "json", path], timeout=300, **_popen_flags())).get("frames", [])
    return [rank[int(f["pts"])] for f in frames if f.get("pts") not in (None, "N/A") and int(f["pts"]) in rank]


_QSV_DECODERS = {"h264": "h264_qsv", "hevc": "hevc_qsv", "av1": "av1_qsv", "vp9": "vp9_qsv",
                 "mpeg2video": "mpeg2_qsv", "vc1": "vc1_qsv"}
_qsv_cache: Dict[Tuple[str, str], bool] = {}
_qsv_lock = threading.Lock()


def qsv_decoder_name(codec: str) -> Optional[str]:
    return _QSV_DECODERS.get((codec or "").lower())


def qsv_probe(path: str, codec: str) -> bool:
    """Can ffmpeg decode one frame of this file with the matching ``*_qsv`` decoder? Cached per (codec, path).
    False when the build has no such decoder, there is no Intel adapter, or the decode fails."""
    name = qsv_decoder_name(codec)
    if not name or not qsv_wanted():
        return False
    key = (name, path)
    with _qsv_lock:
        if key in _qsv_cache:
            return _qsv_cache[key]
    ok = False
    try:
        listing = subprocess.run([_ffmpeg_binary(), "-hide_banner", "-decoders"], capture_output=True, timeout=30,
                                 **_popen_flags())
        if name.encode() in (listing.stdout or b""):
            proc = subprocess.run(
                [_ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-nostdin", "-c:v", name, "-i", path,
                 "-frames:v", "1", "-f", "null", "-"], capture_output=True, timeout=30, **_popen_flags())
            ok = proc.returncode == 0
    except Exception as exc:
        _swallowed("roop/video_io_hw.py:qsv-probe", exc, "QSV probe failed; QSV is treated as unavailable")
    with _qsv_lock:
        _qsv_cache[key] = ok
    return ok


class FfmpegPipeReader:
    """Host-frame reader over an ffmpeg raw BGR pipe. ``decoder='h264_qsv'`` is the Intel path; ``None`` is plain
    software decode (the CPU-Fallback keyframe rung). ``keyframes_only`` decodes only key pictures
    (``-skip_frame nokey``) and the ``select=key`` filter guarantees nothing else is emitted even by a decoder that
    ignores the skip hint. This set is libavcodec's key pictures (IDR plus recovery-point I pictures), so it can be
    DENSER than the NVDEC reader's IDR-only set; every frame it yields is a correct picture and carries its exact
    display index (or -1 when the decoder emitted more than ffprobe predicted). Frames arrive in the DECODER's output
    order, which under ``-skip_frame nokey`` on an open-GOP stream is not display order (a 6 s test clip came out
    60, 30, 0): trust ``frame.index``, sort by it if display order matters."""

    def __init__(self, path: str, *, decoder: Optional[str] = None, keyframes_only: bool = False,
                 backend: str = BACKEND_CPU, tag: str = "decode") -> None:
        self.path = path
        self.backend = backend
        self.keyframes_only = keyframes_only
        try:
            self.width, self.height, self.fps, self.codec = probe_geometry(path)
        except Exception as exc:
            raise HwUnavailable(f"cannot probe {os.path.basename(path)}: {exc}") from exc
        self._indices: Optional[List[int]] = None
        if keyframes_only:
            try:
                self._indices = keyframe_display_indices(path)
            except Exception as exc:
                _swallowed("roop/video_io_hw.py:key-indices", exc, "keyframe indices unavailable; reporting -1")
                self._indices = []
        self._frame_bytes = self.width * self.height * 3
        self._count = 0
        self._gen = 0
        self.command = self.build_command(path, decoder, keyframes_only)
        try:
            self._proc = subprocess.Popen(self.command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                          bufsize=self._frame_bytes * 2, **_popen_flags())
        except OSError as exc:
            raise HwUnavailable(f"cannot start ffmpeg: {exc}") from exc
        telemetry.register_decoder(tag, backend)

    @staticmethod
    def build_command(path: str, decoder: Optional[str], keyframes_only: bool) -> List[str]:
        cmd = [_ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-nostdin", "-noautorotate"]
        if keyframes_only:
            cmd += ["-skip_frame", "nokey"]
        if decoder:
            cmd += ["-c:v", decoder]
        cmd += ["-i", path]
        if keyframes_only:
            cmd += ["-vf", "select=key"]
        cmd += ["-an", "-sn", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        return cmd

    def read(self) -> Optional[HwFrame]:
        raw = self._proc.stdout.read(self._frame_bytes) if self._proc and self._proc.stdout else b""
        if len(raw) < self._frame_bytes:
            return None
        if self._indices is not None:
            # An index that cannot be matched is -1 ("unknown"), never a made-up number.
            index = self._indices[self._count] if self._count < len(self._indices) else -1
        else:
            index = self._count
        self._count += 1
        telemetry.note_decoded(self._frame_bytes, device=False)
        host = np.frombuffer(raw, np.uint8).reshape(self.height, self.width, 3)
        return HwFrame(index, None, host=host, nbytes=self._frame_bytes)

    def __iter__(self) -> Iterator[HwFrame]:
        while True:
            f = self.read()
            if f is None:
                return
            yield f

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.kill()
            except OSError:
                pass
            try:
                proc.wait(timeout=10)
            except Exception as exc:
                _swallowed("roop/video_io_hw.py:pipe-wait", exc, "ffmpeg did not exit within 10 s")
            for stream in (proc.stdout,):
                try:
                    stream and stream.close()
                except OSError:
                    pass

    def __enter__(self) -> "FfmpegPipeReader":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class IntelQsvReader(FfmpegPipeReader):
    """Intel QuickSync / oneVPL keyframe decoder over ffmpeg. Raises HwUnavailable unless ``qsv_probe`` passes.
    UNVERIFIED on hardware (see the module docstring)."""

    def __init__(self, path: str, *, keyframes_only: bool = True, tag: str = "stage1-keyframes") -> None:
        try:
            codec = probe_geometry(path)[3]
        except Exception as exc:
            raise HwUnavailable(f"cannot probe {os.path.basename(path)}: {exc}") from exc
        if not qsv_probe(path, codec):
            raise HwUnavailable(f"no usable Intel QSV decoder for {codec} on this machine")
        super().__init__(path, decoder=qsv_decoder_name(codec), keyframes_only=keyframes_only,
                         backend=BACKEND_QSV, tag=tag)


class LegacyCaptureReader:
    """The existing pipeline behind the same interface: ``cv2.VideoCapture`` through ``nvdec_reader.wrap_capture``
    (an ``ffmpeg -hwaccel cuda`` pipe when it probes, software ffmpeg or plain cv2 otherwise). ``wrap_capture``
    registers which of those it chose with the telemetry, so the label here is the truth, not a guess."""

    def __init__(self, path: str, *, start_frame: int = 0, tag: str = "decode") -> None:
        import cv2
        from roop.nvdec_reader import wrap_capture
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise HwUnavailable(f"cv2 cannot open {path}")
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        self.num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._cv2 = cv2
        self._cap = wrap_capture(cap, path, self.width, self.height, self.fps, tag=tag)
        self._pos = int(start_frame)
        self._gen = 0
        self._frame_bytes = self.width * self.height * 3
        if start_frame:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

    @property
    def backend(self) -> str:
        return telemetry.snapshot()["decoders"].get("decode", BACKEND_CPU)

    def read(self) -> Optional[HwFrame]:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        telemetry.note_decoded(self._frame_bytes, device=False)
        f = HwFrame(self._pos, None, host=frame, nbytes=self._frame_bytes)
        self._pos += 1
        return f

    def __iter__(self) -> Iterator[HwFrame]:
        while True:
            f = self.read()
            if f is None:
                return
            yield f

    def close(self) -> None:
        cap, self._cap = self._cap, None
        if cap is not None:
            cap.release()

    def __enter__(self) -> "LegacyCaptureReader":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# --------------------------------------------------------------------------------------------------------------
# Ladders
# --------------------------------------------------------------------------------------------------------------

def open_reader(path: str, *, start_frame: int = 0, buffer_size: int = 16, gpu_id: int = 0, tag: str = "decode"):
    """Main-pass reader: NVDEC zero-copy, else the app's existing capture pipeline. Never raises for a missing
    accelerator; raises only if even the fallback cannot open the file."""
    if hw_io_wanted():
        try:
            return NvZeroCopyReader(path, gpu_id=gpu_id, buffer_size=buffer_size, start_frame=start_frame, tag=tag)
        except HwUnavailable as exc:
            telemetry.note_fallback(f"{tag}: NVDEC-DLPack -> existing pipeline ({exc})")
    return LegacyCaptureReader(path, start_frame=start_frame, tag=tag)


def open_keyframe_reader(path: str, *, gpu_id: int = 0, tag: str = "stage1-keyframes"):
    """Stage 1 sparse pre-scan source. Ladder: Intel QSV (frees the NVIDIA decoder and the dGPU while TensorRT
    engines build) -> NVDEC in VRAM -> software ffmpeg keyframes. All three expose ``.keyframes_iter()``."""
    if hw_io_wanted():
        if qsv_wanted():
            try:
                return _KeyframeAdapter(IntelQsvReader(path, keyframes_only=True, tag=tag))
            except HwUnavailable as exc:
                telemetry.note_fallback(f"{tag}: QSV-oneVPL skipped ({exc})")
        try:
            return _KeyframeAdapter(NvSparseKeyframeReader(path, gpu_id=gpu_id, tag=tag))
        except HwUnavailable as exc:
            telemetry.note_fallback(f"{tag}: NVDEC-DLPack keyframes -> software ffmpeg ({exc})")
    return _KeyframeAdapter(FfmpegPipeReader(path, keyframes_only=True, backend=BACKEND_CPU, tag=tag))


class _KeyframeAdapter:
    """One iteration protocol over the three keyframe readers."""

    def __init__(self, reader: Any) -> None:
        self.reader = reader
        self.backend = reader.backend

    def keyframes_iter(self) -> Iterator[HwFrame]:
        r = self.reader
        if hasattr(r, "sample_frames"):
            return r.sample_frames()
        return r.keyframes() if hasattr(r, "keyframes") else iter(r)

    def close(self) -> None:
        self.reader.close()

    def __enter__(self) -> "_KeyframeAdapter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class BackgroundKeyframeScan:
    """Decode keyframes on a worker thread into a bounded queue, so Stage 1 can start consuming while the main
    GPU is still building TensorRT engines. Frames are device tensors (NVDEC) or host frames (QSV / software).
    Not yet wired into ``ProcessMgr`` - see CHANGELOG 2026-10-21."""

    _END = object()

    def __init__(self, path: str, *, depth: int = 8, gpu_id: int = 0) -> None:
        self.source = open_keyframe_reader(path, gpu_id=gpu_id)
        self.backend = self.source.backend
        self.frames: "queue.Queue[Any]" = queue.Queue(maxsize=max(1, depth))
        self.error: Optional[BaseException] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="keyframe-prescan", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            for frame in self.source.keyframes_iter():
                while not self._stop.is_set():
                    try:
                        self.frames.put(frame, timeout=0.2)
                        break
                    except queue.Full:
                        continue
                if self._stop.is_set():
                    return
        except BaseException as exc:                    # surfaced to the consumer, never swallowed here
            self.error = exc
            _swallowed("roop/video_io_hw.py:keyframe-scan", exc, "keyframe scan stopped; the consumer re-raises it")
        finally:
            while True:
                try:
                    self.frames.put(self._END, timeout=0.2)
                    return
                except queue.Full:
                    if self._stop.is_set():
                        return

    def __iter__(self) -> Iterator[HwFrame]:
        while True:
            item = self.frames.get()
            if item is self._END:
                if self.error is not None:
                    raise self.error
                return
            yield item

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)
        self.source.close()


# --------------------------------------------------------------------------------------------------------------
# Writer
# --------------------------------------------------------------------------------------------------------------

_MATRIX_BT601 = (0.299, 0.114)
_MATRIX_BT709 = (0.2126, 0.0722)
_BITS_PER_PIXEL = {"h264": 0.12, "hevc": 0.08, "av1": 0.07}


def default_matrix(width: int, height: int) -> Tuple[float, float]:
    """The (Kr, Kb) a decoder assumes for untagged video: BT.709 from 720p / 1280 wide, BT.601 below."""
    return _MATRIX_BT709 if (height >= 720 or width >= 1280) else _MATRIX_BT601


def rgb_to_nv12(rgb, matrix: Tuple[float, float]):
    """CUDA uint8 ``[3, H, W]`` RGB -> CUDA uint8 ``[H*3/2, W]`` NV12, limited range, 2x2 box-averaged chroma."""
    import torch
    kr, kb = matrix
    kg = 1.0 - kr - kb
    x = rgb.float()
    r, g, b = x[0], x[1], x[2]
    y = kr * r + kg * g + kb * b
    cb = (b - y) / (2.0 * (1.0 - kb))
    cr = (r - y) / (2.0 * (1.0 - kr))
    y_out = (16.0 + (219.0 / 255.0) * y).round_().clamp_(16, 235)
    chroma = torch.nn.functional.avg_pool2d(torch.stack([cb, cr])[None], 2)[0]
    uv = (128.0 + (224.0 / 255.0) * chroma).round_().clamp_(16, 240)
    uv = uv.permute(1, 2, 0).reshape(uv.shape[1], uv.shape[2] * 2)
    return torch.cat([y_out, uv], 0).to(torch.uint8).contiguous()


class NvZeroCopyWriter:
    """NVENC writer fed from VRAM. ``write(tensor)`` takes a contiguous CUDA uint8 tensor (``layout`` 'CHW_RGB',
    'HWC_BGR' or 'HWC_RGB') or an ``HwFrame``; a host array is refused, because accepting one would hide the copy
    this class exists to avoid. Even dimensions are required (NV12). ``close()`` finishes the video-only file;
    ``finalize()`` adds the source audio and verifies it."""

    backend = BACKEND_NVENC_DIRECT
    _TIMESCALE = 90000

    def __init__(self, path: str, width: int, height: int, fps: float, *, codec: str = "h264",
                 bitrate: Optional[int] = None, gop: Optional[int] = None, preset: str = "P4", gpu_id: int = 0,
                 matrix: Optional[Tuple[float, float]] = None, tag: str = "encode",
                 quality: Optional[Any] = None) -> None:
        nvc, torch = _load_nvc()
        # Opt-in multipass / spatial AQ / temporal AQ (roop.nvenc_quality; default OFF, see its docstring for the
        # measurements and for why only these kwarg spellings count). `quality` overrides the environment.
        from roop import nvenc_quality
        self.quality = quality if quality is not None else nvenc_quality.options()
        bundle_preset = nvenc_quality.preset_override(self.quality)
        if bundle_preset:
            preset = bundle_preset.upper()
        preset = str(preset).upper()                       # CreateEncoder spells presets P1..P7
        self._nvc, self._torch = nvc, torch
        codec = codec.lower()
        if codec not in _BITS_PER_PIXEL:
            raise HwUnavailable(f"NVENC codec {codec!r} is not one of h264/hevc/av1")
        if width % 2 or height % 2 or width < 64 or height < 64:
            raise HwUnavailable(f"NV12 needs even dimensions of at least 64, got {width}x{height}")
        self.path, self.width, self.height, self.codec = path, int(width), int(height), codec
        self.matrix = matrix or default_matrix(width, height)
        ratio = Fraction(fps).limit_denominator(1001)
        self._fps_num, self._fps_den = ratio.numerator, ratio.denominator
        # Muxer timebase: 1/(fps_num*k) with one frame = fps_den*k ticks is exact for 25, 30, 60 and 24000/1001
        # alike. k lifts the timescale to >= 10000 (an mp4 timescale of 25 is legal but needlessly coarse). The
        # muxer takes only the DENOMINATOR of the timebase: passing 1001/24000 produced a 24000/1 fps file.
        k = max(1, -(-10000 // self._fps_num))
        self._tb_den, self._tick = self._fps_num * k, self._fps_den * k
        self._frame_bytes = self.width * self.height * 3
        self.frames_written = 0
        self.packets = 0
        self._device = torch.device("cuda", int(gpu_id))
        bitrate = int(bitrate or max(1_000_000, self.width * self.height * float(fps) * _BITS_PER_PIXEL[codec]))
        try:
            self._enc = nvc.CreateEncoder(self.width, self.height, "NV12", False, codec=codec, gpu_id=int(gpu_id),
                                          fps=int(round(float(fps))), gop=int(gop or max(1, round(float(fps)))),
                                          preset=preset, rc="vbr", bitrate=bitrate, maxbitrate=bitrate * 2,
                                          **nvenc_quality.encoder_kwargs(self.quality))
            self._mux = nvc.FFmpegMuxer(
                file_path=path, media_format=nvc.GetMediaFormat(path), codec=codec, width=self.width,
                height=self.height, fps_num=self._fps_num, fps_den=self._fps_den, timebase_num=1,
                timebase_den=self._tb_den, extradata=self._enc.GetSequenceParams())
            self._mux.SetUniformPtsIncrement(self._tick)
        except Exception as exc:
            raise HwUnavailable(f"NVENC {codec} could not open: {type(exc).__name__}: {exc}") from exc
        self._closed = False
        telemetry.register_encoder(tag, self.backend)

    def _as_tensor(self, frame: Any, layout: str):
        t = frame.tensor if isinstance(frame, HwFrame) else frame
        torch = self._torch
        if not isinstance(t, torch.Tensor) or not t.is_cuda or t.dtype != torch.uint8:
            raise TypeError("NvZeroCopyWriter.write needs a CUDA uint8 tensor; copying a host frame up is the "
                            "round trip this writer avoids (use the fallback writer for host frames)")
        if t.device != self._device:
            raise TypeError(f"frame is on {t.device}, the encoder is on {self._device}")
        if layout == "HWC_BGR":
            t = t.flip(-1).permute(2, 0, 1)
        elif layout == "HWC_RGB":
            t = t.permute(2, 0, 1)
        elif layout != "CHW_RGB":
            raise ValueError(f"unknown layout {layout!r}")
        if tuple(t.shape) != (3, self.height, self.width):
            raise ValueError(f"frame is {tuple(t.shape)}, the encoder was opened for (3, {self.height}, {self.width})")
        return t

    def write(self, frame: Any, layout: str = "CHW_RGB") -> None:
        if self._closed:
            raise RuntimeError("writer is closed")
        nv12 = rgb_to_nv12(self._as_tensor(frame, layout), self.matrix)
        # The NV12 conversion is queued on torch's stream and NVENC reads the buffer on its own. Unordered, ~30% of
        # 30-frame clips encoded GARBAGE (4 dB round trip, identical on every decode of the file; 0 of 60 with the
        # wait). Passing torch's stream to CreateEncoder(cudastream=...) did NOT fix it (still 8 of 60), so wait.
        self._torch.cuda.current_stream(self._device).synchronize()
        params = self._nvc.NV_ENC_PIC_PARAMS()
        params.inputTimeStamp = self.frames_written
        self._drain(self._enc.Encode(nv12, params))
        self.frames_written += 1
        telemetry.note_encoded(self._frame_bytes, device=True)

    def _drain(self, packets: Any) -> None:
        for pk in packets:
            self._mux.MuxVideoPacket(bytes(pk["data"]), pk["picture_type"], pk["timestamp"])
            self.packets += 1

    def close(self) -> Dict[str, Any]:
        if not self._closed:
            self._closed = True
            try:
                self._drain(self._enc.EndEncode())
                self._mux.Finalize()
            finally:
                # The muxer writes the moov atom when released: opening the file before this gave
                # "Invalid data found when processing input" on all three codecs.
                self._mux = None
                self._enc = None
                gc.collect()
        return {"path": self.path, "frames": self.frames_written, "packets": self.packets,
                "backend": self.backend, "codec": self.codec, "nvenc_quality": self.quality.label()}

    def finalize(self, destination: str, *, audio_source: Optional[str], trim_start: Optional[int] = None,
                 trim_end: Optional[int] = None, expected_frames: Optional[int] = None) -> Dict[str, Any]:
        """Close, stream-copy the source audio, verify, and delete the video-only file only if verification passed
        (the same ``finalize_render`` the NVENC-pipe path uses)."""
        info = self.close()
        from roop.finalize_output import finalize_render
        result = finalize_render(self.path, destination, audio_source=audio_source, trim_start=trim_start,
                                 trim_end=trim_end, expected_frames=expected_frames or self.frames_written)
        result["writer"] = info
        return result

    def __enter__(self) -> "NvZeroCopyWriter":
        return self

    def __exit__(self, exc_type: Any, *rest: Any) -> None:
        self.close()


class CpuFallbackWriter:
    """The existing ``FFMPEG_VideoWriter`` behind the same ``write``/``close`` interface. Device tensors are
    downloaded (counted in ``pcie_moved_bytes``); host BGR arrays go straight through."""

    backend = BACKEND_CPU
    _CODECS = {"h264": "libx264", "hevc": "libx265", "av1": "libx265"}

    def __init__(self, path: str, width: int, height: int, fps: float, *, codec: str = "h264",
                 tag: str = "encode", **_ignored: Any) -> None:
        from roop.ffmpeg_writer import FFMPEG_VideoWriter
        self.path, self.width, self.height = path, int(width), int(height)
        self.frames_written = 0
        self._w = FFMPEG_VideoWriter(path, (self.width, self.height), float(fps),
                                     codec=self._CODECS.get(codec.lower(), "libx264"))
        telemetry.register_encoder(tag, self.backend)

    def write(self, frame: Any, layout: str = "CHW_RGB") -> None:
        if isinstance(frame, HwFrame):
            host = frame.bgr()
        elif isinstance(frame, np.ndarray):
            host = frame
        else:
            t = frame
            if layout == "CHW_RGB":
                t = t.flip(0).permute(1, 2, 0)
            elif layout == "HWC_RGB":
                t = t.flip(-1)
            host = t.contiguous().cpu().numpy()
            telemetry.note_moved(int(host.nbytes))
        self._w.write_frame(host)
        self.frames_written += 1
        telemetry.note_encoded(int(host.nbytes), device=False)

    def close(self) -> Dict[str, Any]:
        self._w.close()
        return {"path": self.path, "frames": self.frames_written, "backend": self.backend}

    def finalize(self, destination: str, *, audio_source: Optional[str], trim_start: Optional[int] = None,
                 trim_end: Optional[int] = None, expected_frames: Optional[int] = None) -> Dict[str, Any]:
        info = self.close()
        from roop.finalize_output import finalize_render
        result = finalize_render(self.path, destination, audio_source=audio_source, trim_start=trim_start,
                                 trim_end=trim_end, expected_frames=expected_frames or self.frames_written)
        result["writer"] = info
        return result

    def __enter__(self) -> "CpuFallbackWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def open_writer(path: str, width: int, height: int, fps: float, *, codec: str = "h264",
                bitrate: Optional[int] = None, gpu_id: int = 0, tag: str = "encode"):
    """NVENC straight from VRAM, else the existing ffmpeg pipe writer."""
    if hw_io_wanted():
        try:
            return NvZeroCopyWriter(path, width, height, fps, codec=codec, bitrate=bitrate, gpu_id=gpu_id, tag=tag)
        except HwUnavailable as exc:
            telemetry.note_fallback(f"{tag}: NVENC-Direct -> existing ffmpeg writer ({exc})")
    return CpuFallbackWriter(path, width, height, fps, codec=codec, tag=tag)
