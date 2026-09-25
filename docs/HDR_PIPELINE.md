# HDR / high-bit-depth colour path

`app/roop/hdr_color.py` holds the colour maths and `app/roop/hdr_pipeline.py` the I/O.
Settings live in the **Output** section: `hdr_pipeline`, `hdr_source_transfer`,
`hdr_source_primaries` and `hdr_output_codec` (env `ROOP_HDR`, `ROOP_HDR_TRANSFER`,
`ROOP_HDR_PRIMARIES`, `ROOP_HDR_CODEC`). The pipeline reads them on every render and
preview probe, so a saved change takes effect on the next render without a restart.

## What it does

```
source Y'CbCr 10/12/16-bit ──ffmpeg──> 4:4:4 16-bit planes (the MASTER, never clipped)
   │  source matrix + range -> R'G'B' (super-whites / sub-blacks kept)
   │  camera/HDR curve decode  (PQ, HLG, S-Log3, Canon Log / Log 2 / Log 3, BT.1886)
   │  source primaries -> ACEScg (AP1 linear, FP16)          <- interchange space
   │  ACEScg -> Rec.709 linear -> invertible highlight roll-off -> BT.1886 encode
   ▼
8-bit BGR "working view" ──> detect / swap / enhance / merge   (existing code, unchanged)
   ▼
writer: decode the source AGAIN in lockstep, recompute the working view (w_in),
        compare with the pipeline's frame (w_out):
          pixel unchanged  -> the master's original Y'CbCr code value, bit for bit
          pixel changed    -> w_out + the master's own sub-8-bit residual
                              -> exact inverse chain -> ACEScg (FP16) -> source primaries
                              -> source curve -> source matrix
   ▼
ffmpeg: yuv444p16le -> p010le (NVENC) / yuv4xxp10|12le (libx265), source tags via setparams
```

**Why the models do not see linear ACEScg.** Every detector, recogniser, swapper and
restorer in this app was trained on 8-bit display-referred sRGB/Rec.709 images. Linear
light looks nearly black to them, and a PQ or Log signal read as gamma video looks flat
and grey. So "running the swapper in a standardized colour space" here means one
**fixed, invertible view transform** out of ACEScg. A PQ, HLG, S-Log3 or Canon Log
source all reach the network as the same kind of picture: 18 % grey at code 125,
diffuse white at 232. The HDR headroom above white is rolled off smoothly into the
last 23 codes and restored exactly on the way out.

**Why the writer decodes the source again** instead of carrying 16-bit masters through
the pipeline:

- Every render path already hands the writer frames strictly in source order, so a
  sequential decode lines up with them by construction.
- It costs no pipeline RAM. A 4K 16-bit master is 50 MB.
- None of the ~100 modules between reader and writer had to change.

A frame whose working view disagrees with the pipeline's over more than 60 % of the
picture is counted as *misaligned* and rebuilt from the 8-bit view alone. The same
happens for every frame when a frame upscaler changes the output resolution. The
writer prints its counts at the end of every part:

```
[HDR] .clip__temp.seg0000.mp4: 246 frames via hevc_nvenc (p010le); composited into the
16-bit master 246, rebuilt from 8-bit 0 (misaligned 0); 12.21% of pixels re-encoded,
the rest carry the source's code values.
```

## Policy

| `hdr_pipeline` | Behaviour |
|---|---|
| `auto` (default) | on for PQ/HLG, for a Log override, and for any source deeper than 8 bits; an 8-bit SDR source never enters the path |
| `on` | every video source |
| `off` | the legacy 8-bit path everywhere |

It applies only to the in-memory render. Keep Frames and per-frame masks go through
8-bit frame files and say so in the status line. The output container must be
mp4/mov/m4v/mkv. For WebM or GIF the render stays on the 8-bit path and says why.

**Tags.** The output keeps the source's primaries, transfer and matrix tags. A Log
stream the camera tagged `bt709` stays tagged `bt709`: there is nothing truer to
write, because S-Log3 and Canon Log have no H.273 code. For the same reason Log
footage needs `hdr_source_transfer`. Its primaries default to the camera gamut
(S-Gamut3.Cine for S-Log3, Cinema Gamut for Canon Log) when the tag says `bt709` or
nothing.

**Codec.** `auto` keeps `hevc_nvenc`, `av1_nvenc` or `libx265`. `h264_nvenc` becomes
`hevc_nvenc`, because NVENC H.264 has no 10-bit profile, and anything else becomes
`libx265`.

- NVENC output is `p010le` (4:2:0 10-bit, Main10).
- libx265 keeps the source chroma (4:2:0/4:2:2/4:4:4) and uses 12 bit for a 12-bit
  source.

## Measured facts (FFmpeg 8.1.2, RTX 4070)

- **Output tags need `setparams`.** On this FFmpeg, `-color_primaries` and `-color_trc`
  given as output options are silently dropped: the file comes out `unknown`. This
  holds for libx265, hevc_nvenc and av1_nvenc alike, and only `-colorspace` survives.
  The encoder only sees frame properties, so the writer sets them with
  `setparams=...` and keeps the options as a backup.
  `test_encoder_tags_travel_as_frame_properties` guards the command.
- **HDR10 static metadata (mastering display, MaxCLL/MaxFALL)** is probed from the
  source and written by **libx265** (`-x265-params hdr10=1:master-display=...:max-cll=...`,
  verified with ffprobe). NVENC reads it only from frame side data, and frames arriving
  over a raw pipe carry none; this FFmpeg has no CLI path to attach it. An NVENC HDR10
  output therefore carries the PQ/BT.2020 VUI tags without SEI, and the render says so.
  Set `hdr_output_codec=libx265` when the SEI matters.
- **Curves and matrices** were cross-checked against colour-science 0.4.6 and
  OpenColorIO 2.5.2: every curve to machine precision, and every gamut→ACEScg matrix
  (CAT02) to 1e-10. Canon Log / Log 2 / Log 3 use Canon's v1.2 constants, which are
  defined on the code value (CV/1023), the same domain as Sony S-Log3.
- **Precision.** The fast LUT path (`cv2.transform` + `cv2.remap`) agrees with the
  exact float64 maths within 1 LSB of the working view. Re-viewing a composite returns
  exactly the pipeline's edit. With the output encoded losslessly, luma outside the
  edit is **bit-identical** to the source through decode → view → composite → encode
  (`RoundTrip.test_pq_clip_round_trip`).
- **Cost (1080p, CPU).** The working view costs ~33 ms/frame on the decode thread. The
  composite costs ~30 ms per face-sized region (75k px, 4 threads) on its own writer
  thread. Both run off the render loop.
