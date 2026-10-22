# NVENC quality options (multipass / spatial AQ / temporal AQ) — measured, shipped OPT-IN (2026-10-22)

Code: `app/roop/nvenc_quality.py` (the only place the option spellings live), wired into `ffmpeg_writer.FFMPEG_VideoWriter`
(`_build_cmd`), `video_stream.NVHardwareVideoWriter` (`_command`, `_encoder_preset`) and `video_io_hw.NvZeroCopyWriter`
(`CreateEncoder`). Tests: `app/tests/test_nvenc_quality.py` (command lines of both production writers; bitstream effect on
a real NVENC). Probes: `app/tests/probe_nvenc_kwargs_effect.py`, `app/tests/probe_nvenc_rd.py`.

Switches (all OFF by default): `ROOP_NVENC_QUALITY_FIRST=1` (p6 + multipass fullres + spatial AQ + temporal AQ),
`ROOP_NVENC_MULTIPASS=off|qres|fullres`, `ROOP_NVENC_SPATIAL_AQ`, `ROOP_NVENC_TEMPORAL_AQ`; `ROOP_NVENC_PRESET` still wins.

## 1. The spellings: `PyNvVideoCodec.CreateEncoder` silently ignores what it does not know

90 frames of Weeds.mp4, 1280x720, HEVC, 4 Mbps VBR, P6. "IDENTICAL" = byte-for-byte the same stream as P6 alone.

| kwarg | bitstream |
|---|---|
| `multipass="fullres"` | **different** (effective) |
| `aq=1` | **different** (spatial AQ) |
| `temporalaq=1` | **different** (temporal AQ) |
| `multipass="FULL_RESOLUTION"`, `multipass=2`, `multipass="bogus"` | IDENTICAL (ignored) |
| `spatial_aq=1`, `enableAQ=1` | IDENTICAL (ignored) |
| `temporal_aq=1`, `enableTemporalAQ=1` | IDENTICAL (ignored) |
| `totallybogus=1`, `tuning_info="high_quality"` | IDENTICAL (ignored) |

`enableAQ` / `enableTemporalAQ` are the spellings a `NV_ENC_RC_PARAMS` reader would reach for. They are accepted without
an error and configure nothing. `test_nvenc_quality.py::TestHardwareBitstream` pins both halves, so a future
PyNvVideoCodec that starts honouring one of them fails the test and says so.

## 2. Does any of it help? Rate–distortion through the production ffmpeg path

Love.mp4 and Weeds.mp4, first 150 frames, `hevc_nvenc -tune hq -rc vbr -cq {19,23,27,31}`, raw BGR pipe as the writers do it.
Reference = the source frames; face region = OpenCV Haar box on them (measurement only). "Bytes at equal quality" interpolates
each arm's curve over the quality range all arms cover. Raw: `rd_*_hevc_cq19-31.json`.

| arm | Love: bytes @ equal face PSNR | Love: @ equal face LPIPS | Weeds: @ PSNR | Weeds: @ LPIPS | encode fps (720p) |
|---|---|---|---|---|---|
| p5 (shipped, `ffmpeg_writer`) | 0 | 0 | 0 | 0 | ~440 |
| p4 (shipped, `video_stream`) | +1.2% | +2.2% | -0.8% | +0.2% | ~490 |
| p6 | -2.2% | -2.9% | +1.7% | +2.3% | ~300 |
| p6 + multipass fullres | -0.7% | -3.6% | -1.8% | -2.7% | ~257 |
| p6 + temporal AQ | -1.8% | -3.8% | -2.6% | +2.7% | ~294 |
| p6 + fullres + temporal AQ | -2.0% | -2.5% | -1.8% | +2.1% | ~253 |
| p6 + spatial AQ | **+11.9%** | **+4.5%** | **+15.8%** | **+14.8%** | ~290 |
| p6 + fullres + spatial + temporal (the requested bundle) | **+12.3%** | **+4.4%** | **+10.0%** | **+11.8%** | ~258 |

Reading it:
* Everything except spatial AQ sits inside ±3%, and the sign flips between the two clips (p6 alone: -2.2% on Love, +1.7% on
  Weeds). That is scatter, not an effect.
* Spatial AQ is a consistent loss on both fidelity measures: 4.5–16% more bytes for the same face quality, and a worse face
  LPIPS at the same CQ.
* The full bundle encodes ~40% slower than p5 (still ~18x faster than the render's 14 fps, and NVENC is separate silicon,
  so this is not what limits a render).

## 3. What this does NOT show

* **Visible blocking in flat skin is unmeasured.** That is what spatial AQ exists to reduce, and PSNR / SSIM / a 128 px
  LPIPS crop cannot see it. "No fidelity win" is not "looks no better". A person looking at a smooth cheek in a dark scene
  would be the test.
* Only HEVC, only 720p, only two clips, only quality values 19–31. The live config encodes at `video_quality: 14`, which this
  ladder does not reach (a supplementary run below).
* The swapped output (not the source) is what production encodes; the reference here is the source because it is the only
  clean one. Whether the swap's own detail changes any of this is unmeasured.
* H.264 / AV1 NVENC and the RTX 3060's encoder are unmeasured.

Decision: the options exist and are correct, and nothing is changed for anyone who does not set the env var.
