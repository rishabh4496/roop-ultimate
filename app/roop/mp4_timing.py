"""Read and rewrite the sample timing of the video track of an MP4, without touching a single coded byte.

The render engine writes a constant-rate intermediate (one raw frame per source frame down a pipe, `-r <fps>`). A
variable-frame-rate source (a phone recording) therefore came out with every frame on a uniform grid while the audio,
stream-copied from the source, kept its own clock. Measured through the real render path on a 180-frame clip with
20 ms steps and 13 ms wobble: -42.6 .. +24.0 ms per frame (7 ms mean), and the audio 23 ms shorter than the video.

A frame's presentation time lives in `stts` (decode-time deltas per sample), `ctts` (composition offsets, present with
B-frames) and the edit list. ``retime`` rewrites them so that frame k (presentation order) is presented at a requested
time, to 1/90000 s (0.011 ms), and fixes everything that moves with them: box sizes, `stco`/`co64` when `moov` precedes
`mdat` (the faststart layout), the `mdhd` / `tkhd` / `mvhd` durations and the edit list. With B-frames the decode
timestamps are re-derived (dts of a sample = the earliest presentation time among the samples from it on), which keeps
them non-decreasing and never after their presentation time.

Pure Python: PyAV is not installed and ffmpeg has no per-frame timestamp-table input. It never edits in place.
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

TIMESCALE = 90000                       # the video track's timescale after a retime (0.011 ms per tick)
_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts"}


class Mp4Error(ValueError):
    pass


@dataclass
class Box:
    type: bytes
    start: int                  # offset of the size field
    header: int                 # 8, or 16 for a 64-bit size
    size: int
    children: List["Box"] = field(default_factory=list)

    @property
    def body(self) -> int:
        return self.start + self.header

    @property
    def end(self) -> int:
        return self.start + self.size


def parse_boxes(data: bytes, start: int = 0, end: Optional[int] = None) -> List[Box]:
    end = len(data) if end is None else end
    out: List[Box] = []
    pos = start
    while pos + 8 <= end:
        size, typ = struct.unpack_from(">I4s", data, pos)
        header = 8
        if size == 1:
            size = struct.unpack_from(">Q", data, pos + 8)[0]
            header = 16
        elif size == 0:
            size = end - pos
        if size < header or pos + size > end:
            raise Mp4Error(f"bad box {typ!r} at {pos}: size {size}")
        box = Box(typ, pos, header, size)
        if typ in _CONTAINERS:
            box.children = parse_boxes(data, box.body, box.end)
        out.append(box)
        pos += size
    return out


def _walk(boxes: Sequence[Box]) -> Iterator[Box]:
    for b in boxes:
        yield b
        yield from _walk(b.children)


def _kids(box: Box, typ: bytes) -> List[Box]:
    return [c for c in box.children if c.type == typ]


def _version(data: bytes, box: Box) -> int:
    return data[box.body]


@dataclass
class VideoTrack:
    trak: Box
    mdhd: Box
    stbl: Box
    timescale: int
    movie_timescale: int
    stts: List[Tuple[int, int]]            # (count, delta)
    ctts: Optional[List[Tuple[int, int]]]  # (count, offset)
    elst: List[Tuple[int, int]]            # (segment_duration in movie ticks, media_time) ; media_time -1 = empty edit
    samples: int

    def decode_times(self) -> List[int]:
        out, t = [], 0
        for count, delta in self.stts:
            for _ in range(count):
                out.append(t)
                t += delta
        return out

    def composition_offsets(self) -> List[int]:
        if not self.ctts:
            return [0] * self.samples
        out: List[int] = []
        for count, off in self.ctts:
            out += [off] * count
        return out[:self.samples]


def _video_track(data: bytes, moov: Box) -> VideoTrack:
    mvhd = _kids(moov, b"mvhd")[0]
    movie_ts = struct.unpack_from(">I", data, mvhd.body + (20 if _version(data, mvhd) == 1 else 12))[0]
    for trak in _kids(moov, b"trak"):
        mdia = _kids(trak, b"mdia")[0]
        hdlr = _kids(mdia, b"hdlr")[0]
        if data[hdlr.body + 8:hdlr.body + 12] != b"vide":
            continue
        mdhd = _kids(mdia, b"mdhd")[0]
        timescale = struct.unpack_from(">I", data, mdhd.body + (20 if _version(data, mdhd) == 1 else 12))[0]
        stbl = _kids(_kids(mdia, b"minf")[0], b"stbl")[0]
        stts_b = _kids(stbl, b"stts")[0]
        n = struct.unpack_from(">I", data, stts_b.body + 4)[0]
        stts = [struct.unpack_from(">II", data, stts_b.body + 8 + 8 * i) for i in range(n)]
        ctts = None
        ctts_b = _kids(stbl, b"ctts")
        if ctts_b:
            fmt = ">Ii" if _version(data, ctts_b[0]) == 1 else ">II"
            n = struct.unpack_from(">I", data, ctts_b[0].body + 4)[0]
            ctts = [tuple(struct.unpack_from(fmt, data, ctts_b[0].body + 8 + 8 * i)) for i in range(n)]
        elst: List[Tuple[int, int]] = []
        edts = _kids(trak, b"edts")
        if edts and _kids(edts[0], b"elst"):
            e = _kids(edts[0], b"elst")[0]
            ev = _version(data, e)
            n = struct.unpack_from(">I", data, e.body + 4)[0]
            for i in range(n):
                if ev == 1:
                    sd, mt = struct.unpack_from(">Qq", data, e.body + 8 + 20 * i)
                else:
                    sd, mt = struct.unpack_from(">Ii", data, e.body + 8 + 12 * i)
                elst.append((sd, mt))
        return VideoTrack(trak, mdhd, stbl, timescale, movie_ts, [tuple(x) for x in stts], ctts, elst,
                          sum(c for c, _ in stts))
    raise Mp4Error("no video track")


def read_track(path: str) -> Tuple[bytes, List[Box], VideoTrack]:
    with open(path, "rb") as fh:
        data = fh.read()
    boxes = parse_boxes(data)
    moov = [b for b in boxes if b.type == b"moov"]
    if not moov:
        raise Mp4Error("no moov box (fragmented or not an MP4)")
    return data, boxes, _video_track(data, moov[0])


def presentation_seconds(path: str) -> List[Fraction]:
    """Presentation time of every video sample, in presentation order, as exact fractions of a second.

    The edit list is applied the way a player does: an empty edit delays the track by its duration, and the media edit
    starts at its media_time (the B-frame delay ffmpeg writes as a leading edit)."""
    _, _, tr = read_track(path)
    ts = tr.timescale
    dts = tr.decode_times()
    pts = sorted(d + o for d, o in zip(dts, tr.composition_offsets()))
    delay = Fraction(0)
    media_start = 0
    for dur, mt in tr.elst:
        if mt < 0:
            delay += Fraction(dur, tr.movie_timescale)
        else:
            media_start = mt
            break
    return [delay + Fraction(p - media_start, ts) for p in pts]


# ── rewrite ──────────────────────────────────────────────────────────────────
def _rle(values: Sequence[int]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for v in values:
        if out and out[-1][1] == v:
            out[-1] = (out[-1][0] + 1, v)
        else:
            out.append((1, v))
    return out


def _box(typ: bytes, body: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(body), typ) + body


def _shift_chunk_offsets(moov: bytearray, shift: int) -> None:
    for b in _walk(parse_boxes(bytes(moov))):
        if b.type == b"stco":
            n = struct.unpack_from(">I", moov, b.body + 4)[0]
            for i in range(n):
                pos = b.body + 8 + 4 * i
                struct.pack_into(">I", moov, pos, struct.unpack_from(">I", moov, pos)[0] + shift)
        elif b.type == b"co64":
            n = struct.unpack_from(">I", moov, b.body + 4)[0]
            for i in range(n):
                pos = b.body + 8 + 8 * i
                struct.pack_into(">Q", moov, pos, struct.unpack_from(">Q", moov, pos)[0] + shift)


def retime(src: str, dst: str, targets: Sequence[Fraction]) -> Dict[str, object]:
    """Write ``dst`` = ``src`` with video sample k (presentation order) presented at ``targets[k]`` seconds.

    ``targets`` has one entry per sample, strictly increasing, the first >= 0 (a later first frame becomes an empty edit, so
    the track starts late exactly as the source did). Raises Mp4Error on anything it cannot rewrite exactly."""
    data, boxes, tr = read_track(src)
    n = tr.samples
    if len(targets) != n:
        raise Mp4Error(f"{len(targets)} target times for {n} samples")
    if targets[0] < 0 or any(b <= a for a, b in zip(targets, targets[1:])):
        raise Mp4Error("target times must be non-negative and strictly increasing")
    ts = TIMESCALE
    pts_ticks = [int(round(Fraction(t) * ts)) for t in targets]          # absolute, presentation order
    first = pts_ticks[0]
    media_pts = [p - first for p in pts_ticks]                           # the media timeline starts at 0

    dts_old = tr.decode_times()
    old_pts = [d + o for d, o in zip(dts_old, tr.composition_offsets())]
    order = sorted(range(n), key=lambda i: (old_pts[i], i))              # presentation rank -> decode index
    new_pts = [0] * n
    for rank, idx in enumerate(order):
        new_pts[idx] = media_pts[rank]
    reorder = tr.ctts is not None and any(o != 0 for o in tr.composition_offsets())
    # Decode timestamps must be STRICTLY increasing and never after the presentation time. Without reordering dts == pts.
    # With B-frames, sample k is decoded `R` places ahead of where it is shown (R = the stream's reorder depth), so its dts
    # is the presentation time of the frame R places earlier in the SHOWN order: natural frame-sized steps, dts <= pts by
    # construction. The first R dts fall before the first frame; the whole track is shifted right by that much and the edit
    # list's media_time hides the shift.
    shift_ticks = 0
    if reorder:
        rho = [0] * n
        for rank, idx in enumerate(order):
            rho[idx] = rank
        depth = max(0, max(k - rho[k] for k in range(n)))
        step0 = media_pts[1] - media_pts[0] if n > 1 else 3000
        new_dts = [media_pts[k - depth] if k >= depth else media_pts[0] - (depth - k) * step0 for k in range(n)]
        shift_ticks = -new_dts[0]
        new_dts = [d + shift_ticks for d in new_dts]
        new_pts = [p + shift_ticks for p in new_pts]
    else:
        new_dts = list(new_pts)
    deltas = [new_dts[k + 1] - new_dts[k] for k in range(n - 1)]
    if any(d <= 0 for d in deltas):
        raise Mp4Error("decode timestamps would not be strictly increasing")
    last = tr.stts[-1][1] * ts // tr.timescale if tr.stts else 1
    last_step = max(1, media_pts[-1] - media_pts[-2]) if n > 1 else max(1, last)
    deltas.append(max(1, last_step))
    # the media ends after the last frame to be SHOWN and after the last frame to be DECODED
    total_media = max(new_dts[-1] + deltas[-1], max(new_pts) + last_step)

    # new leaf boxes
    stts_entries = _rle(deltas)
    stts_body = struct.pack(">BBBBI", 0, 0, 0, 0, len(stts_entries)) + b"".join(struct.pack(">II", c, d) for c, d in stts_entries)
    patches: Dict[int, bytes] = {_kids(tr.stbl, b"stts")[0].start: _box(b"stts", stts_body)}
    ctts_b = _kids(tr.stbl, b"ctts")
    if ctts_b:
        offs = [p - d for p, d in zip(new_pts, new_dts)]
        ents = _rle(offs)
        patches[ctts_b[0].start] = _box(b"ctts", struct.pack(">BBBBI", 0, 0, 0, 0, len(ents))
                                        + b"".join(struct.pack(">II", c, o) for c, o in ents))

    moov = [b for b in boxes if b.type == b"moov"][0]
    mts = tr.movie_timescale
    shown = total_media - shift_ticks                                   # presented length, after the hidden head
    track_movie_dur = int(round(Fraction(first + shown, ts) * mts))
    # mdhd: timescale + duration
    mdhd = bytearray(data[tr.mdhd.start:tr.mdhd.end])
    mv = _version(data, tr.mdhd)
    struct.pack_into(">I", mdhd, 8 + (20 if mv == 1 else 12), ts)
    if mv == 1:
        struct.pack_into(">Q", mdhd, 8 + 24, total_media)
    else:
        struct.pack_into(">I", mdhd, 8 + 16, total_media)
    patches[tr.mdhd.start] = bytes(mdhd)
    # tkhd duration (movie timescale)
    tk = _kids(tr.trak, b"tkhd")[0]
    tkd = bytearray(data[tk.start:tk.end])
    if _version(data, tk) == 1:
        struct.pack_into(">Q", tkd, 8 + 28, track_movie_dur)
    else:
        struct.pack_into(">I", tkd, 8 + 20, track_movie_dur)
    patches[tk.start] = bytes(tkd)
    # edit list: an empty edit for a late first frame, then the whole media from media time 0
    entries: List[Tuple[int, int]] = []
    if first > 0:
        entries.append((int(round(Fraction(first, ts) * mts)), -1))
    entries.append((int(round(Fraction(shown, ts) * mts)), shift_ticks))
    elst_body = struct.pack(">BBBBI", 0, 0, 0, 0, len(entries)) + b"".join(
        struct.pack(">IiHH", d, mt, 1, 0) for d, mt in entries)
    elst_box = _box(b"elst", elst_body)
    edts_old = _kids(tr.trak, b"edts")
    if edts_old:
        patches[_kids(edts_old[0], b"elst")[0].start] = elst_box
    else:
        # no edit list yet: append an edts to the trak (a trak's children may come in any order after tkhd)
        patches[-tr.trak.start - 1] = _box(b"edts", elst_box)

    shift = _write_patched(data, boxes, patches, tr.trak, dst)
    return {"samples": n, "timescale": ts, "bframes": reorder, "moov_shift": shift, "first_pts_s": float(Fraction(first, ts))}


def _write_patched(data: bytes, boxes: List[Box], patches: Dict[int, bytes], trak: Optional[Box], dst: str) -> int:
    """Write ``data`` with the leaf boxes in ``patches`` (keyed by start offset) replaced, every ancestor resized,
    chunk offsets shifted when `moov` precedes `mdat`, and the movie duration refreshed. A patch keyed
    ``-trak.start - 1`` is appended as a new child of ``trak``. Returns the `moov` growth in bytes."""
    moov = [b for b in boxes if b.type == b"moov"][0]

    def rebuild(box: Box) -> bytes:
        if box.start in patches:
            return patches[box.start]
        if not box.children:
            return data[box.start:box.end]
        parts = [data[box.body:box.children[0].start]]
        parts += [rebuild(c) for c in box.children]
        if trak is not None and box is trak and (-trak.start - 1) in patches:
            parts.append(patches[-trak.start - 1])
        inner = b"".join(parts)
        if box.header == 16:
            return struct.pack(">I4sQ", 1, box.type, 16 + len(inner)) + inner
        return struct.pack(">I4s", 8 + len(inner), box.type) + inner

    new_moov = bytearray(rebuild(moov))
    mdat = [b for b in boxes if b.type == b"mdat"]
    shift = len(new_moov) - moov.size if (mdat and moov.start < mdat[0].start) else 0
    if shift:
        _shift_chunk_offsets(new_moov, shift)
    out = bytearray()
    for b in boxes:
        out += bytes(new_moov) if b is moov else data[b.start:b.end]
    _fix_movie_duration(out)
    tmp = dst + ".part"
    with open(tmp, "wb") as fh:
        fh.write(out)
    os.replace(tmp, dst)
    return shift


def edit_span(path: str, handler: bytes = b"soun"):
    """(start_s, duration_s) the first track of ``handler`` type is PRESENTED for, from its edit list: the empty edits
    delay it, the media edits are what plays. None when the file has no edit list for it (ffprobe's numbers then stand).
    ffprobe's per-stream ``duration`` is the media length and ignores an edit that skips the head or ends early.
    """
    data = open(path, "rb").read()
    moov = [b for b in parse_boxes(data) if b.type == b"moov"]
    if not moov:
        return None
    mvhd = _kids(moov[0], b"mvhd")[0]
    mts = struct.unpack_from(">I", data, mvhd.body + (20 if _version(data, mvhd) == 1 else 12))[0]
    for trak in _kids(moov[0], b"trak"):
        hdlr = _kids(_kids(trak, b"mdia")[0], b"hdlr")[0]
        if data[hdlr.body + 8:hdlr.body + 12] != handler:
            continue
        edts = _kids(trak, b"edts")
        if not edts or not _kids(edts[0], b"elst"):
            return None
        el = _kids(edts[0], b"elst")[0]
        ev = _version(data, el)
        n = struct.unpack_from(">I", data, el.body + 4)[0]
        delay = media = 0
        for i in range(n):
            if ev == 1:
                dur, mt = struct.unpack_from(">Qq", data, el.body + 8 + 20 * i)
            else:
                dur, mt = struct.unpack_from(">Ii", data, el.body + 8 + 12 * i)
            if mt < 0:
                delay += dur
            else:
                media += dur
        return delay / mts, media / mts
    return None


def trim_audio_head(path: str, skip_s: float, length_s: float, delay_s: float = 0.0) -> Dict[str, object]:
    """Make the first audio track START ``skip_s`` seconds into its media, last ``length_s`` seconds, and begin
    ``delay_s`` seconds into the timeline (the video's own start), by edit list.
    Sample-exact (1/sample-rate), no audio data touched. A stream-copy cut can only begin on a packet boundary
    (AAC: 1024 samples = 23 ms); cutting a packet-aligned stretch that begins a little BEFORE the wanted start and
    skipping the remainder here puts the wanted instant at exactly t = 0. Rewrites ``path`` in place (atomically).
    """
    data = open(path, "rb").read()
    boxes = parse_boxes(data)
    moov = [b for b in boxes if b.type == b"moov"][0]
    mvhd = _kids(moov, b"mvhd")[0]
    mts = struct.unpack_from(">I", data, mvhd.body + (20 if _version(data, mvhd) == 1 else 12))[0]
    for trak in _kids(moov, b"trak"):
        mdia = _kids(trak, b"mdia")[0]
        hdlr = _kids(mdia, b"hdlr")[0]
        if data[hdlr.body + 8:hdlr.body + 12] != b"soun":
            continue
        mdhd = _kids(mdia, b"mdhd")[0]
        ts = struct.unpack_from(">I", data, mdhd.body + (20 if _version(data, mdhd) == 1 else 12))[0]
        media_time = int(round(skip_s * ts))
        seg = int(round(length_s * mts))
        entries = []
        if delay_s > 0:
            entries.append((int(round(delay_s * mts)), -1, 1, 0))
        entries.append((seg, media_time, 1, 0))
        elst_box = _box(b"elst", struct.pack(">BBBBI", 0, 0, 0, 0, len(entries))
                        + b"".join(struct.pack(">IiHH", *x) for x in entries))
        patches: Dict[int, bytes] = {}
        edts = _kids(trak, b"edts")
        if edts:
            patches[_kids(edts[0], b"elst")[0].start] = elst_box
        else:
            patches[-trak.start - 1] = _box(b"edts", elst_box)
        tk = _kids(trak, b"tkhd")[0]
        tkd = bytearray(data[tk.start:tk.end])
        if _version(data, tk) == 1:
            struct.pack_into(">Q", tkd, 8 + 28, seg + (int(round(delay_s * mts)) if delay_s > 0 else 0))
        else:
            struct.pack_into(">I", tkd, 8 + 20, seg + (int(round(delay_s * mts)) if delay_s > 0 else 0))
        patches[tk.start] = bytes(tkd)
        shift = _write_patched(data, boxes, patches, trak, path)
        return {"skip_s": skip_s, "length_s": length_s, "delay_s": delay_s, "media_time": media_time, "moov_shift": shift}
    raise Mp4Error("no audio track")


def _fix_movie_duration(file_bytes: bytearray) -> None:
    """mvhd.duration = the longest track (movie timescale)."""
    moov = [b for b in parse_boxes(bytes(file_bytes)) if b.type == b"moov"][0]
    mv = _kids(moov, b"mvhd")[0]
    v = file_bytes[mv.body]
    longest = 0
    for trak in _kids(moov, b"trak"):
        tk = _kids(trak, b"tkhd")[0]
        pos = tk.body + (28 if file_bytes[tk.body] == 1 else 20)
        longest = max(longest, struct.unpack_from(">Q" if file_bytes[tk.body] == 1 else ">I", file_bytes, pos)[0])
    struct.pack_into(">Q" if v == 1 else ">I", file_bytes, mv.body + (24 if v == 1 else 16), longest)
