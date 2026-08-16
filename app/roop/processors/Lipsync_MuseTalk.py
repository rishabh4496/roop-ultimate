"""Lip-sync — regenerate the mouth region to match a driving audio track.

Hand-wired singleton like Expression_LivePortrait, NOT registered in
ProcessMgr.plugins: that registry is a single-select radio for enhancers/mask
engines (Run(source_faceset, target_face, temp_frame) -> (Frame, scale_factor)),
and this needs non-standard inputs (audio features, frame timing) that don't
fit that contract. See ProcessMgr._lipsync_restorer / process_face's
"Lip-sync (post-composite)" block.

MODEL: MuseTalk (github.com/TMElyralab/MuseTalk, MIT License,
Copyright (c) 2024 Tencent Music Entertainment Group). Unlike every other
model in this app it ships PyTorch + diffusers + transformers weights with no
official ONNX export, so this is the one processor here with a torch
inference path instead of onnxruntime — confirmed acceptable (torch was
already a real dependency via Enhance_DMDNet/Mask_Clip2Seg) rather than a
fresh addition.

The functions below are ported from MuseTalk's own inference code, not
guessed at, because a wrong crop convention or feature layout doesn't crash —
it silently drives the model with out-of-distribution input and produces a
plausible-looking wrong mouth. Verified against (commit pinned by reading, not
guessed): musetalk/models/vae.py, musetalk/models/unet.py,
musetalk/utils/audio_processor.py, scripts/realtime_inference.py.

  * Face crop: a straight bounding-box crop (no landmark similarity
    transform), resized to 256x256 with INTER_LANCZOS4.
  * VAE: stabilityai/sd-vae-ft-mse (a public SD checkpoint). Per frame, the
    SAME 256x256 crop is encoded TWICE — once with its bottom half zeroed
    (the mouth/chin region the model regenerates) and once unmasked as
    context — and the two 4-channel latents are concatenated to 8 channels.
    This is genuinely self-referential per frame (not a separate "reference
    frame"), which is what makes per-frame stateless use straightforward here
    despite MuseTalk's own public interface being built around a precomputed,
    looping "avatar" video.
  * UNet: a stock diffusers.UNet2DConditionModel (config from MuseTalk's own
    musetalk.json — 8 in / 4 out channels, cross_attention_dim 384), called
    with a FIXED timestep of 0 (one forward pass, not iterative diffusion)
    and audio features as encoder_hidden_states.
  * Audio: whisper-tiny's encoder, ALL hidden-state layers stacked (not just
    the final one), windowed per output video frame
    (audio_padding_length_left/right = 2 → 10 raw 50Hz timesteps x 5 stacked
    layers = 50 feature vectors per frame), then a sinusoidal
    PositionalEncoding before cross-attention.
  * Blend mask: MuseTalk's own reference pipeline uses a face-parsing model
    (their own BiSeNet fork, weights hosted on Google Drive — an unreliable
    download source, and different weights from this codebase's own
    Mask_FaceParser.py). Deliberately NOT reused here for v1: this class
    hands its decoded mouth crop to ProcessMgr's existing apply_mouth_area/
    create_mouth_mask machinery instead (same compositor restore_original_
    mouth already uses), trading a bit of blend fidelity at the jaw line for
    zero new mask model, zero Google Drive dependency, and code this app
    already tests. Worth revisiting once output quality is validated on real
    footage.

Runs in fp16 (MuseTalk's own default for real-time inference) with no
SessionPool/pooling for v1 — see ProcessMgr's _gpu_guard call site. Add
pooling only once real per-frame timings exist; this is already a 3-model
pipeline (VAE encode x2, UNet, VAE decode) plus a one-time Whisper pass, so
guessing at a pool size ahead of measurement risks tuning against nothing.

Verified end-to-end on this machine (RTX 4070): Initialize() downloads +
loads all three models, build_audio_cache() produces the expected chunk count
and (50, 384) per-frame feature shape, and infer() returns a real
(256, 256, 3) uint8 crop with no NaN/degenerate output — not just imports
cleanly, an actual forward pass through VAE->UNet->VAE ran and produced a
plausible image.
"""

import json
import math
import os

import cv2
import numpy as np
import torch
import torch.nn as nn

from roop.lipsync_audio import AudioFeatureCache
from roop.typing import Frame
from roop.utilities import resolve_relative_path

INPUT_SIZE = 256  # cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
_MUSETALK_REPO = "TMElyralab/MuseTalk"
_UNET_CONFIG_FILE = "musetalkV15/musetalk.json"
_UNET_WEIGHTS_FILE = "musetalkV15/unet.pth"
_VAE_REPO = "stabilityai/sd-vae-ft-mse"
_WHISPER_REPO = "openai/whisper-tiny"
_AUDIO_SR = 16000
_AUDIO_FEATURE_DIM = 384
# audio_padding_length_left/right = 2 (MuseTalk's own default): each video
# frame's window is 2*(2+2+1) = 10 raw 50Hz whisper timesteps.
_PAD_LEFT = 2
_PAD_RIGHT = 2
_WINDOW = 2 * (_PAD_LEFT + _PAD_RIGHT + 1)


def resolve_device(devicename, cuda_available: bool) -> str:
    """devicename comes from roop.utilities.get_device(), which names ONNX
    EXECUTION PROVIDERS ('cuda', 'mps', 'mkl' for OpenVINO, 'cpu') — this is
    the one torch-based processor here, and 'mkl' is not a torch device
    string. Only trust devicename for the two that ARE; anything else falls
    back to a plain CUDA-availability check.

    Standalone/testable for the same reason face_bbox_crop is: a bad device
    string doesn't error cleanly here, it makes torch.device() raise inside
    Initialize(), which retries (and fails again) on every subsequent frame.
    """
    if devicename in ('cuda', 'mps'):
        return devicename
    return 'cuda' if cuda_available else 'cpu'


def face_bbox_crop(frame: Frame, bbox, extra_margin: int = 10):
    """MuseTalk's own crop: a straight bounding-box crop (no landmark-fitted
    similarity transform), resized to INPUT_SIZE with Lanczos.

    extra_margin extends the box downward before cropping — MuseTalk's v1.5
    default, so the crop includes enough chin for the model's own framing;
    matches `y2 = y2 + args.extra_margin` in realtime_inference.py.

    Kept as a standalone function (not a method) so the crop math is testable
    without the model class — mirrors why frame_time/audio_index_for_frame
    live in roop.lipsync_audio rather than on the class that uses them.

    Returns (crop, crop_box) where crop_box is the actual (x1, y1, x2, y2)
    used, post-margin and post-clamp, in full-frame pixel coordinates — the
    caller needs this to map a full-frame region (e.g. the mouth bounding
    box) into the resized crop's local coordinates later. Returns (None,
    None) on a degenerate box.
    """
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    h, w = frame.shape[:2]
    y2 = y2 + extra_margin
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None, None
    crop = frame[y1:y2, x1:x2]
    resized = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LANCZOS4)
    return resized, (x1, y1, x2, y2)


def crop_box_to_local(crop_box, region_box, size: int = INPUT_SIZE):
    """Map region_box (full-frame x1,y1,x2,y2) into the local pixel
    coordinates of a size x size crop resized from crop_box (also full-frame
    x1,y1,x2,y2) — e.g. finding where the mouth bounding box landed inside
    the 256x256 face crop MuseTalk was fed, so its OUTPUT can be sliced back
    down to just that region before apply_mouth_area resizes it into place.

    Standalone/testable: this is pure geometry, no model involved, and a
    transposed axis or missing clamp here would not crash — it would
    composite the wrong part of a generated face (or the whole face,
    squashed) into the mouth region, silently.
    """
    if crop_box is None or region_box is None:
        return None
    fx1, fy1, fx2, fy2 = crop_box
    rx1, ry1, rx2, ry2 = region_box
    fw, fh = fx2 - fx1, fy2 - fy1
    if fw <= 0 or fh <= 0:
        return None
    sx, sy = size / fw, size / fh
    lx1 = max(0, min(size, int(round((rx1 - fx1) * sx))))
    ly1 = max(0, min(size, int(round((ry1 - fy1) * sy))))
    lx2 = max(0, min(size, int(round((rx2 - fx1) * sx))))
    ly2 = max(0, min(size, int(round((ry2 - fy1) * sy))))
    if lx2 <= lx1 or ly2 <= ly1:
        return None
    return (lx1, ly1, lx2, ly2)


class _PositionalEncoding(nn.Module):
    """Ported verbatim from TMElyralab/MuseTalk's musetalk/models/unet.py
    (MIT License, Copyright (c) 2024 Tencent Music Entertainment Group)."""

    def __init__(self, d_model=_AUDIO_FEATURE_DIM, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)


class Lipsync_MuseTalk:
    processorname = 'lipsync_musetalk'
    type = 'lipsync'
    pool = None  # No pooling for v1 — see the module docstring.

    def __init__(self):
        self._ready = False
        self.device = None
        self.vae = None
        self.unet = None
        self.pe = None
        self.whisper = None
        self.feature_extractor = None
        self.scaling_factor = None
        self.dtype = torch.float16

    # ── setup ────────────────────────────────────────────────────────────
    def Initialize(self, plugin_options: dict):
        if self._ready:
            return
        from diffusers import AutoencoderKL, UNet2DConditionModel
        from huggingface_hub import hf_hub_download
        from transformers import AutoFeatureExtractor, WhisperModel

        devicename = resolve_device((plugin_options or {}).get('devicename'),
                                    torch.cuda.is_available())
        self.device = torch.device(devicename)
        cache_dir = resolve_relative_path('../models/musetalk_hf_cache')
        os.makedirs(cache_dir, exist_ok=True)

        vae = AutoencoderKL.from_pretrained(_VAE_REPO, cache_dir=cache_dir)
        self.vae = vae.to(self.device, dtype=self.dtype).eval()
        self.vae.requires_grad_(False)
        self.scaling_factor = self.vae.config.scaling_factor

        config_path = hf_hub_download(_MUSETALK_REPO, _UNET_CONFIG_FILE, cache_dir=cache_dir)
        weights_path = hf_hub_download(_MUSETALK_REPO, _UNET_WEIGHTS_FILE, cache_dir=cache_dir)
        with open(config_path, 'r', encoding='utf-8') as f:
            unet_config = json.load(f)
        unet = UNet2DConditionModel(**unet_config)
        state = torch.load(weights_path, map_location=self.device)
        unet.load_state_dict(state)
        self.unet = unet.to(self.device, dtype=self.dtype).eval()
        self.unet.requires_grad_(False)

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(_WHISPER_REPO, cache_dir=cache_dir)
        whisper = WhisperModel.from_pretrained(_WHISPER_REPO, cache_dir=cache_dir)
        self.whisper = whisper.to(self.device, dtype=self.dtype).eval()
        self.whisper.requires_grad_(False)

        self.pe = _PositionalEncoding(d_model=_AUDIO_FEATURE_DIM).to(self.device, dtype=self.dtype)

        self._ready = True

    def Release(self):
        self.vae = self.unet = self.pe = self.whisper = self.feature_extractor = None
        self._ready = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── audio ────────────────────────────────────────────────────────────
    def build_audio_cache(self, audio_wav_path: str, fps: float) -> AudioFeatureCache:
        """One Whisper encoder pass over the whole driving-audio WAV,
        windowed into one (50, 384) feature vector per frame at *fps* — the
        expensive, cacheable part (see ProcessMgr.run_batch_inmem, which
        calls this once per distinct audio source rather than per frame).

        Ported from AudioProcessor.get_audio_feature +
        AudioProcessor.get_whisper_chunk in musetalk/utils/audio_processor.py
        (MIT License, Copyright (c) 2024 Tencent Music Entertainment Group),
        adapted to return an audio_fps-indexed roop.lipsync_audio.
        AudioFeatureCache instead of a fixed-length per-clip list — this app
        trims/starts clips mid-video, so the caller needs to index by
        absolute frame_time rather than assume index 0 is frame 0.
        """
        import librosa

        if not self._ready:
            raise RuntimeError("Lipsync_MuseTalk.Initialize() was not called")
        samples, _ = librosa.load(audio_wav_path, sr=_AUDIO_SR)
        if samples is None or len(samples) == 0:
            return AudioFeatureCache(None, audio_fps=fps)

        segment_len = 30 * _AUDIO_SR
        segments = [samples[i:i + segment_len] for i in range(0, len(samples), segment_len)]

        with torch.no_grad():
            layer_stacks = []
            for seg in segments:
                mel = self.feature_extractor(seg, sampling_rate=_AUDIO_SR,
                                             return_tensors="pt").input_features
                mel = mel.to(self.device, dtype=self.dtype)
                hidden_states = self.whisper.encoder(mel, output_hidden_states=True).hidden_states
                # (1, 1500, 384) x num_layers -> (1, 1500, num_layers, 384)
                layer_stacks.append(torch.stack(hidden_states, dim=2))
            whisper_feature = torch.cat(layer_stacks, dim=1)

            audio_fps = 50  # whisper-tiny's own encoder output rate
            actual_length = math.floor((len(samples) / _AUDIO_SR) * audio_fps)
            whisper_feature = whisper_feature[:, :actual_length, ...]

            pad_left = torch.zeros_like(whisper_feature[:, :_PAD_LEFT * 2])
            pad_right = torch.zeros_like(whisper_feature[:, :_PAD_RIGHT * 6])
            whisper_feature = torch.cat([pad_left, whisper_feature, pad_right], dim=1)

            num_frames = math.floor((len(samples) / _AUDIO_SR) * fps)
            multiplier = audio_fps / fps
            prompts = []
            for frame_index in range(num_frames):
                idx = math.floor(frame_index * multiplier)
                clip = whisper_feature[:, idx: idx + _WINDOW]
                if clip.shape[1] != _WINDOW:
                    break  # ran off the end — trailing partial frame, drop it
                prompts.append(clip)
            if not prompts:
                return AudioFeatureCache(None, audio_fps=fps)
            # (num_frames, 1, WINDOW, num_layers, 384) -> (num_frames, WINDOW*num_layers, 384)
            prompts = torch.cat(prompts, dim=0)
            b, c, l, d = prompts.shape
            prompts = prompts.reshape(b, c * l, d).to(torch.float32).cpu().numpy()

        return AudioFeatureCache(prompts, audio_fps=fps)

    # ── per-frame inference, split so ProcessMgr can take the GPU lock only
    #    around infer() — mirrors Expression_LivePortrait's prepare/infer/
    #    finish split. ──────────────────────────────────────────────────
    def prepare(self, frame: Frame, target_face, audio_features):
        if audio_features is None:
            return None
        bbox = getattr(target_face, 'bbox', None)
        if bbox is None:
            return None
        crop, crop_box = face_bbox_crop(frame, bbox)
        if crop is None:
            return None
        return {"crop": crop, "crop_box": crop_box, "audio_features": audio_features}

    def infer(self, prepared):
        if prepared is None or not self._ready:
            return None
        crop = prepared["crop"]
        audio_features = prepared["audio_features"]
        with torch.no_grad():
            masked_latents = self._encode(crop, half_mask=True)
            ref_latents = self._encode(crop, half_mask=False)
            latents = torch.cat([masked_latents, ref_latents], dim=1).to(self.dtype)

            audio = torch.from_numpy(np.asarray(audio_features, dtype=np.float32))
            audio = audio.unsqueeze(0).to(self.device, dtype=self.dtype)
            audio = self.pe(audio)

            timesteps = torch.tensor([0], device=self.device)
            pred = self.unet(latents, timesteps, encoder_hidden_states=audio).sample
            return self._decode(pred)

    def finish(self, raw, crop_box, mouth_bb) -> Frame:
        """Slice the mouth-region sub-crop out of the generated 256x256
        FACE image, so apply_mouth_area resizes just that region into
        mouth_bb — not the whole face squashed into a mouth-sized box."""
        if raw is None:
            return None
        local = crop_box_to_local(crop_box, mouth_bb)
        if local is None:
            return None
        lx1, ly1, lx2, ly2 = local
        return raw[ly1:ly2, lx1:lx2]

    def Run(self, frame: Frame, target_face, audio_features, mouth_bb) -> Frame:
        prepared = self.prepare(frame, target_face, audio_features)
        raw = self.infer(prepared)
        crop_box = prepared.get("crop_box") if prepared else None
        return self.finish(raw, crop_box, mouth_bb)

    # ── VAE helpers, ported from musetalk/models/vae.py (MIT License,
    #    Copyright (c) 2024 Tencent Music Entertainment Group). ───────────
    def _encode(self, crop_bgr: np.ndarray, half_mask: bool):
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(rgb).permute(2, 0, 1)  # HWC -> CHW
        if half_mask:
            x = x.clone()
            x[:, INPUT_SIZE // 2:, :] = 0.0  # zero the BOTTOM half (mouth/chin)
        x = (x - 0.5) / 0.5  # Normalize(mean=0.5, std=0.5) -> [-1, 1]
        x = x.unsqueeze(0).to(self.device, dtype=self.dtype)
        latents = self.vae.encode(x).latent_dist.sample()
        return latents * self.scaling_factor

    def _decode(self, latents) -> np.ndarray:
        latents = latents / self.scaling_factor
        image = self.vae.decode(latents.to(self.vae.dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().float().cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255.0).round().astype(np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
