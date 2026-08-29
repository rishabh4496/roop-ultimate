"""Crash/abort-resumable video encoding via finalized segment files.

Problem: the in-memory pipeline encodes straight into one long ffmpeg pipe. If
the process dies mid-render (power loss, driver reset, accidental window kill),
the temp file has no trailer (moov atom) and every already-encoded frame is
lost — a 10-hour run restarts from zero.

Fix: SegmentedVideoWriter is a drop-in replacement for FFMPEG_VideoWriter that
rotates the encode into numbered segment files, closing (finalizing) each one
every ROOP_RESUME_CHUNK frames (default 1000) and recording it in a manifest
JSON next to the output. Because each completed segment is a valid playable
file, a crash can only ever lose the in-progress segment. On the next run with
the same source/trim/settings, run_batch_inmem reads the manifest, skips the
already-encoded frames, and continues; when the run completes, the segments are
losslessly concatenated (ffmpeg concat demuxer, stream copy) into the expected
single temp video and cleaned up, so everything downstream (audio restore, the
upscale second pass, template naming) is unchanged.

A deliberate Stop is finalized exactly like a completed run: the segments are
concatenated into the temp video, the parts and manifest are deleted, and
batch_process still mixes the audio back and applies the output template — so
stopping yields one properly named partial video, not a pile of parts. Set
ROOP_RESUME_KEEP=1 to keep the parts after a Stop so it can be resumed later
(a crash never reaches the cleanup, so crash-resume works either way).

Disable with ROOP_RESUME=0. Segment files start with '.' so the output-folder
scans (e.g. the upscale pass's _outputs_since) ignore them.
"""

import json
import os
import subprocess
import threading

import roop.globals
from roop.ffmpeg_writer import FFMPEG_VideoWriter, FFMPEG_BINARY
# These lines are emitted from INSIDE the encode thread, where a raised
# exception is not a bad log line but a dead writer — and a dead writer leaves
# every producer blocked on a bounded queue. bar_write already exists for
# exactly this: it degrades unprintable characters instead of raising. Bare
# print() here bypassed it, and a decorative check-mark on a non-UTF-8 console
# was enough to stop a 2400-frame render dead at 58%.
from roop.procmgr_runtime import bar_write

MANIFEST_VERSION = 1


# ── Live parts registry ──────────────────────────────────────────────────────
# The parts are the only thing on disk while a long run is in flight, so the UI
# shows them: /api/progress serves this snapshot and the console groups its log
# lines by the part that was open when each line arrived. Purely observational —
# nothing here feeds encoding or resume, which stay driven by the manifest.
#
# Written only by the encoder thread (one writer per run) and read by request
# threads, so the lock guards the list swap, not the per-frame counter.
_parts_lock = threading.Lock()
_parts = []          # finalized, oldest first
_current = None      # the part being written, or None between/after segments


def reset_parts():
    """Called at the start of a run — the previous run's parts are gone."""
    global _parts, _current
    with _parts_lock:
        _parts, _current = [], None


def parts_snapshot():
    """[{index, file, frames, first, last, bytes, done}] — finalized parts plus
    the one in progress. Frame numbers are 1-based and absolute for the run
    (a resumed run counts from the frames it inherited, not from 1)."""
    with _parts_lock:
        out = list(_parts)
        cur = dict(_current) if _current else None
    if cur:
        cur["frames"] = cur.pop("_written", 0)
        cur["last"] = cur["first"] + max(0, cur["frames"] - 1)
        out.append(cur)
    return out


def current_part_index():
    """Index of the part being written (1-based), or the count of finished ones
    when nothing is open. 0 before any frame is encoded — used to tag log lines."""
    cur = _current
    if cur:
        return cur["index"]
    return len(_parts)


def manifest_path(target_video: str) -> str:
    return target_video + ".resume.json"


def _segments_that_exist(m: dict, seg_dir: str):
    """Contiguous prefix of manifest segments whose files are actually on disk.
    (A gap means someone deleted a file — everything after it is unusable
    because concat order would break.)"""
    segs, done = [], 0
    for s in m.get("segments", []):
        fn = s.get("file", "")
        n = int(s.get("frames", 0) or 0)
        if n <= 0 or not fn:
            break
        # Size, not just existence. A part can be listed with a frame count and
        # still be EMPTY: an encoder that failed to open swallows everything sent
        # to it ("Nothing was written into output file") while the writer counts
        # the frames it handed over. Inheriting that as valid puts a 0-byte file
        # into the final concat.
        try:
            if os.path.getsize(os.path.join(seg_dir, fn)) <= 0:
                break
        except OSError:
            break
        segs.append({"file": fn, "frames": n})
        done += n
    return segs, done


class SegmentedVideoWriter:
    """FFMPEG_VideoWriter-compatible writer (write_frame/close) with rotation,
    a resume manifest, and final lossless concatenation."""

    def __init__(self, target_video, size, fps, codec="libx264", crf=14,
                 source_video="", frame_start=0, frame_end=0, signature="",
                 preset=None, bitrate=None, threads=None,
                 ffmpeg_params=None, colorspace=None):
        self.target_video = target_video
        self.size = size
        self.fps = float(fps)
        self.codec = codec
        self.crf = crf
        self._dir = os.path.dirname(target_video) or "."
        base, ext = os.path.splitext(os.path.basename(target_video))
        self._seg_prefix = f".{base}.seg"
        self._seg_ext = ext or ".mp4"
        raw_chunk = os.environ.get("ROOP_RESUME_CHUNK")
        if raw_chunk is not None and str(raw_chunk).strip():
            try:
                self.chunk = max(50, int(raw_chunk))
            except ValueError:
                self.chunk = self._default_chunk()
        else:
            self.chunk = self._default_chunk()
        self._writer_options = {
            "preset": preset,
            "bitrate": bitrate,
            "threads": threads,
            "ffmpeg_params": ffmpeg_params,
            "colorspace": colorspace,
        }

        # Everything a resume must match on — resuming into a run with different
        # settings/trim/dims would silently mix outputs.
        self._identity = {
            "version": MANIFEST_VERSION,
            "source": os.path.abspath(source_video) if source_video else "",
            "frame_start": int(frame_start),
            "frame_end": int(frame_end),
            "width": int(size[0]),
            "height": int(size[1]),
            "fps": round(self.fps, 3),
            "codec": codec,
            "crf": crf,
            "signature": signature or "",
        }

        self.segments, self.resume_frames = self._load_resume()
        self._seg_index = len(self.segments)
        self._writer = None
        self._cur_seg_file = None
        self._cur_frames = 0
        # Absolute frame number (1-based) the next segment starts at. Registering
        # the inherited parts walks this forward to resume_frames + 1, so the
        # parts a user sees are numbered continuously with the video rather than
        # restarting at 1 on a resumed run.
        self._next_first = 1
        reset_parts()
        for i, s in enumerate(self.segments, 1):     # parts inherited by a resume
            self._register(i, s["file"], int(s.get("frames", 0) or 0),
                           done=True, inherited=True)
        self._write_manifest()

    # ── UI registry ──────────────────────────────────────────────────────────
    def _register(self, index, filename, frames, first=None, done=False,
                  inherited=False):
        first = self._next_first if first is None else first
        entry = {"index": index, "file": filename, "frames": frames,
                 "first": first, "last": first + max(0, frames - 1),
                 "bytes": self._size_of(filename), "done": done,
                 "inherited": inherited}
        with _parts_lock:
            _parts.append(entry)
        self._next_first = entry["last"] + 1

    def _size_of(self, filename):
        try:
            return os.path.getsize(os.path.join(self._dir, filename))
        except OSError:
            return 0

    # ── resume detection ─────────────────────────────────────────────────────
    def _load_resume(self):
        try:
            with open(manifest_path(self.target_video), "r", encoding="utf-8") as fh:
                m = json.load(fh)
            for key, want in self._identity.items():
                have = m.get(key)
                if key == "fps":
                    if abs(float(have or 0) - want) > 0.01:
                        return [], 0
                elif have != want:
                    return [], 0
            return _segments_that_exist(m, self._dir)
        except Exception:
            return [], 0

    def _write_manifest(self):
        try:
            m = dict(self._identity)
            m["segments"] = self.segments
            tmp = manifest_path(self.target_video) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(m, fh, indent=1)
            os.replace(tmp, manifest_path(self.target_video))
        except Exception as e:
            bar_write(f"[Resume] could not write manifest: {e}")

    # ── writing ──────────────────────────────────────────────────────────────
    def _default_chunk(self):
        """Choose a rotation interval from the detected source frame rate.

        Explicit ROOP_RESUME_CHUNK remains authoritative. Automatic mode uses a
        duration rather than a fixed frame count so 24/30/60 FPS clips get
        comparable crash-loss windows while avoiding needless encoder lifecycle
        work. The duration is configurable for finer-grained recovery.
        """
        try:
            seconds = float(os.environ.get("ROOP_RESUME_SEGMENT_SECONDS", "120"))
        except ValueError:
            seconds = 120.0
        return max(50, int(round(max(1.0, seconds) * max(1.0, self.fps))))

    def _open_next_segment(self):
        global _current
        self._cur_seg_file = f"{self._seg_prefix}{self._seg_index:04d}{self._seg_ext}"
        path = os.path.join(self._dir, self._cur_seg_file)
        self._writer = FFMPEG_VideoWriter(path, self.size, self.fps,
                                          codec=self.codec, crf=self.crf,
                                          audiofile=None,
                                          **self._writer_options)
        self._cur_frames = 0
        _current = {"index": len(self.segments) + 1, "file": self._cur_seg_file,
                    "first": self._next_first, "_written": 0, "bytes": 0,
                    "done": False, "inherited": False}

    def write_frame(self, img_array):
        if self._writer is None:
            self._open_next_segment()
        self._writer.write_frame(img_array)
        self._cur_frames += 1
        if _current is not None:
            _current["_written"] = self._cur_frames
        if self._cur_frames >= self.chunk:
            self._finalize_segment()

    def _finalize_segment(self):
        """Close the current segment (writes its trailer → playable) and commit
        it to the manifest. From this moment its frames survive a crash."""
        global _current
        if self._writer is None:
            return
        self._writer.close()
        self._writer = None
        _current = None
        # Committing is a claim that these frames survive a crash, so it has to be
        # true: check the encoder actually produced bytes. A dead encoder still
        # increments _cur_frames, and committing that is what puts an empty part
        # in the manifest for a later resume to trust.
        _seg_path = (os.path.join(self._dir, self._cur_seg_file)
                     if self._cur_seg_file else "")
        try:
            _seg_bytes = os.path.getsize(_seg_path) if _seg_path else 0
        except OSError:
            _seg_bytes = 0
        if self._cur_frames > 0 and _seg_bytes <= 0:
            bar_write(f"[Resume] part {len(self.segments) + 1} is EMPTY after "
                      f"{self._cur_frames} frames — the encoder produced no data, "
                      f"so it is discarded rather than committed. Resume still "
                      f"has parts 1-{len(self.segments)}.")
        if self._cur_frames > 0 and _seg_bytes > 0:
            self.segments.append({"file": self._cur_seg_file, "frames": self._cur_frames})
            self._seg_index += 1
            self._register(len(self.segments), self._cur_seg_file, self._cur_frames,
                           done=True)
            last = _parts[-1]
            # One line per part, so the console says what is safe on disk. This is
            # the only crash-survival signal a long run gives while it is running.
            bar_write(f"[Resume] ✓ part {last['index']} written · frames "
                  f"{last['first']}-{last['last']} · {last['bytes'] / 1048576:.0f} MB")
            self._write_manifest()
        else:
            try:
                os.remove(os.path.join(self._dir, self._cur_seg_file))
            except OSError:
                pass
        self._cur_seg_file = None
        self._cur_frames = 0

    # ── finish ───────────────────────────────────────────────────────────────
    def close(self):
        try:
            self._finalize_segment()
        except Exception as e:
            bar_write(f"[Resume] finalizing last segment failed: {e}")
        if not self.segments:
            return
        # Completed run (nothing signalled a stop) → concat, then clean up.
        # Aborted run → concat too, so the partial video is playable, and clean up
        # as well: a deliberate Stop is a request for "give me what you rendered as
        # one file", and leaving the numbered parts behind made the output folder
        # look like the merge never happened. Set ROOP_RESUME_KEEP=1 to keep the
        # parts + manifest after a Stop so the run can be resumed later instead.
        # A hard crash never reaches this code at all, so crash-resume is unaffected.
        completed = bool(roop.globals.processing)
        keep_after_stop = os.environ.get("ROOP_RESUME_KEEP", "0") == "1"
        preserve_parts = not completed and keep_after_stop
        # A single segment is already a complete MP4. Promote it directly to
        # avoid a second ffmpeg process and stream-copy pass on short clips;
        # retain concat for multi-part files and ROOP_RESUME_KEEP=1.
        ok = (self._promote_single()
              if len(self.segments) == 1 and not preserve_parts
              else self._concat())
        if not completed and ok:
            n = len(self.segments)
            if keep_after_stop:
                bar_write(f"[Resume] stopped — merged {n} segment(s) into "
                      f"{os.path.basename(self.target_video)}; parts kept for resuming "
                      f"(ROOP_RESUME_KEEP=1).")
            else:
                bar_write(f"[Resume] stopped — merged {n} segment(s) into "
                      f"{os.path.basename(self.target_video)} and removed the parts "
                      f"(set ROOP_RESUME_KEEP=1 to keep them for resuming).")
        if ok and (completed or not keep_after_stop):
            self.cleanup()

    def _promote_single(self) -> bool:
        try:
            source = os.path.join(self._dir, self.segments[0]["file"])
            os.replace(source, self.target_video)
            return True
        except Exception as e:
            bar_write(f"[Resume] single-segment finalize failed: {e}")
            return False

    def _concat(self) -> bool:
        list_path = os.path.join(self._dir, f"{self._seg_prefix}list.txt")
        try:
            with open(list_path, "w", encoding="utf-8") as fh:
                for s in self.segments:
                    p = os.path.abspath(os.path.join(self._dir, s["file"])).replace("\\", "/")
                    fh.write("file '" + p.replace("'", "'\\''") + "'\n")
            cmd = [FFMPEG_BINARY, "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "concat", "-safe", "0", "-i", list_path,
                   "-c", "copy", self.target_video]
            kwargs = {}
            if os.name == "nt":
                kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
            proc = subprocess.run(cmd, capture_output=True, **kwargs)
            if proc.returncode != 0:
                err = (proc.stderr or b"").decode("utf-8", "replace")[:400]
                bar_write(f"[Resume] segment concat failed (ffmpeg exit {proc.returncode}): {err}")
                return False
            return True
        except Exception as e:
            bar_write(f"[Resume] segment concat failed: {e}")
            return False
        finally:
            try:
                os.remove(list_path)
            except OSError:
                pass

    def cleanup(self):
        """Remove all segments + the manifest (after a fully completed run)."""
        for s in self.segments:
            try:
                os.remove(os.path.join(self._dir, s["file"]))
            except OSError:
                pass
        try:
            os.remove(manifest_path(self.target_video))
        except OSError:
            pass
        self.segments = []
