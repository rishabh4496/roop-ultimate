"""GPEN 256 Pro — Upgraded ultra-fast, sharper, high-texture, photo-realistic face restorer.

Speed profile — READ THIS BEFORE ASKING WHY THE GPU IS IDLE.

This processor is HOST-DOMINATED BY DESIGN, and that is the whole shape of its
cost. It pairs a very small network with a large look filter, so most of a
face's time is spent on the CPU and the card has little to do. Measured on an
RTX 4070, one thread, a 256 crop in (tests are in
tests/test_enhancer_gpen256_pro.py, the split in enhance_common.exclusive):

    pre  (LUT gather)              0.49 ms
    GPU  gpen_bfr_256.onnx         3.90 ms      <- the only GPU work
    post cast / collapse guard     0.45 ms
    post photoreal chrominance     0.19 ms
    post texture + sharpen @512   19.03 ms      <- the filter
    -----------------------------------------
    Run()                         24.53 ms      GPU is 16% of it

Nothing here is going to drive a card to 90% utilisation, and chasing that
number is a mistake — see session_pool._advisory_pool_size, where the
configuration reporting 94.5% utilisation was 14x SLOWER than the one reporting
43%. What matters is that the host work does not BLOCK anything: the filter
runs outside every lock (`self_excluding`, enhance_common.exclusive) so N
workers run it concurrently, and it is written as few wide passes rather than
many narrow ones.

- Native 256x256 neural resolution, gpen_bfr_256.onnx.
- Multi-context SessionPool when the card has the VRAM for it; the processor's
  own lock over one session when it does not. Either way the GPU call is
  exclusive and NOTHING ELSE IS.
- Single-pass 256-entry LUT gather in, saturating C++ cast out, one io_binding
  held per pool slot.

Measured and rejected, so it is not re-attempted: capping OpenCV's internal
thread pool (cv2.setNumThreads) to stop it oversubscribing against the worker
threads. At 32 logical cores, 1/4/32 OpenCV threads gave 104.8 / 106.8 / 102.4
faces per second over 8 workers — inside the noise. The oversubscription is
real and it is not what costs anything.

Visual Quality Enhancements:
1. True Photoreal Chrominance (Zero Color Drift):
   - Eliminates GPEN's pink/magenta color cast (2.9+ chroma drift) by transferring restored luminance onto the source chrominance.
2. Structure-Aware Multi-Band Texture Synthesis & Dermal Micro-Porosity:
   - Extracts genuine high-frequency skin pores and micro-texture from the input crop.
   - Applies an edge-stop gate (Sobel gradient magnitude) so sharp contours (eyelids, iris rims, nostrils, lips) do NOT receive ghost creases or halo artifacts.
   - Gated by mid-tone exposure curve so specular catchlights and deep shadows remain clean and noise-free.
   - For heavily blurred inputs, synthesizes subtle organic dermal micro-grain for authentic photographic realism instead of plastic AI smoothing.
3. Feature-Targeted Micro-Sharpening:
   - Selective high-frequency boost on facial features (eyes, lashes, brows, lips, teeth, nose contours) to provide crisp, ultra-sharp detail without making skin look artificial or crunchy.
4. Scale-Aware Output:
   - Supports 256 native output (scale 1) and 512 multi-resolution output (scale 2) for direct paste-upscale compatibility.
"""

import os
import threading

import cv2
import numpy as np
import onnxruntime

import roop.globals
from roop.typing import Face, FaceSet, Frame
from roop.processors.enhance_common import (sized, looks_collapsed, exclusive,
                                            _global_std)
from roop.utilities import resolve_relative_path, conditional_download
from roop import session_pool
from roop.precision_policy import providers_for

try:
    import torch
    import torch.nn.functional as _F
    _TORCH_CUDA = torch.cuda.is_available()
except (ImportError, AttributeError):
    _TORCH_CUDA = False


class Enhance_GPEN256Pro:
    processorname = 'gpen_256_pro'
    # Every session call goes through `exclusive()` below, so no context of
    # this processor's is ever entered twice at once -- the only thing
    # ProcessMgr's enhance-stage lock provides. Declaring that lets the stage
    # skip the lock and stop serialising this class's 34 ms of HOST work
    # against every other worker thread. See enhance_common.exclusive.
    self_excluding = True
    # core.py also maps the legacy label 'GPEN 256 Ultra' here. It is not
    # offered in api.py's enhancer list any more, and is kept only so a
    # config saved under the old name still resolves.
    type = 'enhance'
    model_template = 'ffhq_512'

    _MODEL_FILE = 'gpen_bfr_256.onnx'
    _MODEL_URL = 'https://huggingface.co/facefusion/models-3.0.0/resolve/main/gpen_bfr_256.onnx'
    _SIZE = 256

    # Mid-tone weight for the texture restore exposure gate: full weight where
    # skin lives, smoothly drops to zero in specular highlights or crushed shadows.
    _EXPOSURE_LUT = np.clip(
        np.sin(np.pi * (np.arange(256, dtype=np.float32) / 255.0)), 0.0, 1.0)

    # The same curve with the texture strength folded in, so the injection
    # weight is one multiply instead of two. 0.85 is the texture amount and it
    # is a constant, so it never needed to be a separate pass over 786k floats.
    _EXPOSURE_LUT_TEX = _EXPOSURE_LUT * 0.85

    # Warn ONCE per class, not per face: a look filter must never take a render
    # down, but a filter that fails silently is worse than one that fails loudly.
    # See the consequence documented on _enhance_textures_and_sharpness.
    _warned_texture = False
    _warned_colour = False

    # Synthetic micro-grain, generated ONCE per size and reused.
    #
    # THE FIXED PATTERN IS DELIBERATE — do not "fix" it into a per-call draw.
    # This runs in aligned-crop space, so the field tracks the face; drawing new
    # noise every call would make it re-randomise 25 times a second on a face
    # that is otherwise stable, which is flicker, and flicker is the one artefact
    # this pipeline spends the most effort removing. Deterministic here means
    # temporally stable.
    #
    # The cost of the old spelling was real though: it re-seeded and re-drew a
    # 512x512 gaussian on EVERY face to produce a bit-identical array.
    _GRAIN = {}
    _GRAIN_LOCK = threading.Lock()

    @classmethod
    def _grain(cls, size):
        g = cls._GRAIN.get(size)
        if g is None:
            with cls._GRAIN_LOCK:
                g = cls._GRAIN.get(size)
                if g is None:
                    g = np.random.default_rng(42).normal(
                        0.0, 2.0, (size, size, 1)).astype(np.float32)
                    # Shared between worker threads; it must remain immutable.
                    # The CUDA boundary takes a private copy before wrapping it
                    # as a tensor, avoiding PyTorch's non-writeable-array warning.
                    g.flags.writeable = False
                    cls._GRAIN[size] = g
        return g

    def __init__(self):
        self.plugin_options = None
        self.devicename = None
        self.session = None
        self.io_binding = None
        self.size = self._SIZE
        self.pool = None          # SessionPool of (session, io_binding) slots
        self.in_name = None
        self.out_name = None
        self._lut = None
        self._faces = 0
        self._lock = threading.Lock()
        self._cpu_only = False
        # Guards the SINGLE shared (session, io_binding) used when there is no
        # pool -- a binding's state is not thread-safe. Distinct from _lock,
        # which only counts faces.
        self._session_lock = threading.Lock()

    # ── lifecycle ────────────────────────────────────────────────────────────
    def Initialize(self, plugin_options: dict):
        if self.plugin_options is not None:
            if self.plugin_options.get("devicename") != plugin_options.get("devicename"):
                self.Release()
        self.plugin_options = plugin_options
        self.devicename = plugin_options["devicename"].replace('mps', 'cpu')

        model_dir = resolve_relative_path('../models')
        conditional_download(model_dir, [self._MODEL_URL])
        model_path = os.path.join(model_dir, self._MODEL_FILE)
        from roop.utilities import (get_onnx_session_options,
                                    get_small_card_safe_providers)
        opts = get_onnx_session_options()
        providers = get_small_card_safe_providers(roop.globals.execution_providers)
        providers, _precision = providers_for('gpen_256_pro', providers, model_path)
        self._cpu_only = providers == ['CPUExecutionProvider']

        def _build(_i=0):
            sess = onnxruntime.InferenceSession(
                model_path, opts, providers=providers)
            iob = sess.io_binding()
            iob.bind_output(sess.get_outputs()[0].name, self.devicename)
            return (sess, iob)

        self.session, self.io_binding = _build()
        self.in_name = self.session.get_inputs()[0].name
        self.out_name = self.session.get_outputs()[0].name
        # uint8 -> float32 normalised to [-1, 1], in one gather.
        self._lut = ((np.arange(256, dtype=np.float32) / 127.5) - 1.0)

        # Multi-context SessionPool for lock-free execution across worker threads
        if session_pool.pooling_enabled():
            n = session_pool.pool_size(
                model_key='enhancer:gpen256pro', input_shape=(1, 3, 256, 256))
            cap = plugin_options.get('pool_size')
            if cap:
                n = max(1, min(int(n), int(cap)))
            try:
                forced = int(os.environ.get('ROOP_GPEN256PRO_POOL', '') or 0)
            except ValueError:
                forced = 0
            if forced:
                n = max(1, forced)
            extras = []
            try:
                extras = [_build(i + 1) for i in range(n - 1)]
                primary = (self.session, self.io_binding)
                self.pool = session_pool.SessionPool(
                    lambda i, _e=([primary] + extras): _e[i], n,
                    model_key='enhancer:gpen256pro', input_shape=(1, 3, 256, 256))
            except Exception as e:
                extras.clear()
                self.pool = None
                print(f"[GPEN 256 Pro] multi-context pool unavailable ({e}); "
                      f"falling back to single session")

    def Release(self):
        line = self.cost_summary()
        if line:
            print(line, flush=True)
        if self.pool is not None:
            self.pool.release()
            self.pool = None
        self.io_binding = None
        self.session = None
        self._lut = None

    # ── colour & texture processing ──────────────────────────────────────────
    @classmethod
    def _keep_source_colour(cls, restored, source):
        """Transfers GPEN's restored luminance onto the source's chrominance."""
        return cls._keep_source_colour_cpu(restored, source)

    @classmethod
    def _keep_source_colour_cpu(cls, restored, source):
        try:
            g_r = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)
            g_s = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
            d = cv2.subtract(g_r, g_s, dtype=cv2.CV_16S)
            return cv2.add(source, cv2.merge((d, d, d)), dtype=cv2.CV_8U)
        except cv2.error as e:
            if not cls._warned_colour:
                cls._warned_colour = True
                print(f"[GPEN 256 Pro] colour fix skipped: {e}", flush=True)
            return restored

    _KERNEL_CACHE = {}

    @classmethod
    def _gaussian_kernel_2d_gpu(cls, sigma, device):
        key = (float(sigma), str(device))
        cached = cls._KERNEL_CACHE.get(key)
        if cached is not None:
            return cached
        radius = int(round(3 * sigma))
        x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
        k1d = torch.exp(-0.5 * (x / sigma) ** 2)
        k1d = k1d / k1d.sum()
        k2d = k1d.view(-1, 1) @ k1d.view(1, -1)
        res = (k2d.view(1, 1, k2d.shape[0], k2d.shape[1]), radius)
        cls._KERNEL_CACHE[key] = res
        return res

    @classmethod
    def _enhance_textures_and_sharpness_gpu(cls, restored, source, input_size):
        device = torch.device('cuda')
        try:
            want = int(os.environ.get('ROOP_GPEN256PRO_SIZE', '') or 0)
        except ValueError:
            want = 0
        target_size = want if want in (256, 512, 1024) else (512 if input_size <= 256 else input_size)

        r_t = torch.from_numpy(restored).to(device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        s_t = torch.from_numpy(source).to(device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

        if r_t.shape[2:] != (target_size, target_size):
            rest_f = _F.interpolate(r_t, size=(target_size, target_size), mode='bicubic', align_corners=False)
        else:
            rest_f = r_t

        if s_t.shape[2:] != (target_size, target_size):
            src_f = _F.interpolate(s_t, size=(target_size, target_size), mode='bicubic', align_corners=False)
        else:
            src_f = s_t

        gray = (0.114 * rest_f[:, 0:1] + 0.587 * rest_f[:, 1:2] + 0.299 * rest_f[:, 2:3])

        if not hasattr(cls, '_sobel_x_gpu'):
            cls._sobel_x_gpu = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
            cls._sobel_y_gpu = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
            cls._tex_lut_gpu = torch.tensor(cls._EXPOSURE_LUT_TEX, dtype=torch.float32, device=device)

        gx = _F.conv2d(gray, cls._sobel_x_gpu, padding=1)
        gy = _F.conv2d(gray, cls._sobel_y_gpu, padding=1)
        m2 = gx * gx + gy * gy
        skin_gate = 196.0 / (m2 + 196.0)

        gray_idx = gray.long().clamp(0, 255)
        exp_gate = cls._tex_lut_gpu[gray_idx]
        w_tex = skin_gate * exp_gate
        k_tex, r_ktex = cls._gaussian_kernel_2d_gpu(max(1.0, target_size / 256.0), device)
        padded_src = _F.pad(src_f, (r_ktex, r_ktex, r_ktex, r_ktex), mode='reflect')
        src_blur = _F.conv2d(padded_src, k_tex.repeat(3, 1, 1, 1), groups=3)
        hf_texture = src_f - src_blur
        core = torch.exp(hf_texture * hf_texture * (-1.0 / 256.0))
        injected_texture = hf_texture * core * w_tex

        hf_std = torch.std(hf_texture)
        if hf_std < 3.5:
            k = (1.0 - min(1.0, float(hf_std) / 3.5)) / 0.85
            if not hasattr(cls, '_grain_gpu'):
                cls._grain_gpu = {}
            g_gpu = cls._grain_gpu.get(target_size)
            if g_gpu is None:
                g_gpu = torch.from_numpy(cls._grain(target_size).copy()).to(device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                cls._grain_gpu[target_size] = g_gpu
            injected_texture = injected_texture + g_gpu * (w_tex * k)

        k_sharp, r_ksharp = cls._gaussian_kernel_2d_gpu(0.8 * (target_size / 256.0), device)
        padded_rest = _F.pad(rest_f, (r_ksharp, r_ksharp, r_ksharp, r_ksharp), mode='reflect')
        rest_blur = _F.conv2d(padded_rest, k_sharp.repeat(3, 1, 1, 1), groups=3)
        hf_restored = rest_f - rest_blur
        sharpness_amount = 0.42 - skin_gate * 0.30
        sharpened_features = hf_restored * sharpness_amount

        out = torch.clamp(rest_f + injected_texture + sharpened_features, 0, 255).to(torch.uint8)
        return out[0].permute(1, 2, 0).cpu().numpy()

    @classmethod
    def _enhance_textures_and_sharpness(cls, restored, source, input_size):
        """Applies structure-gated micro-texture injection and edge-targeted sharpening."""
        if _TORCH_CUDA:
            try:
                return cls._enhance_textures_and_sharpness_gpu(restored, source, input_size)
            except Exception as e:
                if not cls._warned_texture:
                    cls._warned_texture = True
                    print(f"[GPEN 256 Pro] GPU texture fix fallback to CPU: {e}", flush=True)
        return cls._enhance_textures_and_sharpness_cpu(restored, source, input_size)

    @classmethod
    def _enhance_textures_and_sharpness_cpu(cls, restored, source, input_size):
        try:
            try:
                want = int(os.environ.get('ROOP_GPEN256PRO_SIZE', '') or 0)
            except ValueError:
                want = 0
            target_size = want if want in (256, 512, 1024) else (512 if input_size <= 256 else input_size)

            # Both are forced to the SAME square shape. The old spelling
            # resized `source` only when its WIDTH differed, so a non-square crop
            # (width already == target, height not) passed straight through and
            # every later `rest_f + injected_texture` broadcast-failed into the
            # bare except below — i.e. silently, as a resolution halving.
            if restored.shape[:2] != (target_size, target_size):
                rest_scaled = cv2.resize(restored, (target_size, target_size),
                                         interpolation=cv2.INTER_LANCZOS4)
            else:
                rest_scaled = restored
            if source.shape[:2] != (target_size, target_size):
                src_scaled = cv2.resize(source, (target_size, target_size),
                                        interpolation=cv2.INTER_CUBIC)
            else:
                src_scaled = source

            rest_f = rest_scaled.astype(np.float32)
            src_f = src_scaled.astype(np.float32)

            # THE SHAPE OF WHAT FOLLOWS IS THE COST, NOT THE OPERATORS.
            # Everything here is a pass over a 512x512x3 float32 buffer (3 MB),
            # so the only thing that moves the clock is HOW MANY passes there
            # are and how wide each one is. Measured on an RTX 4070 (the split
            # is in enhance_common.exclusive): this filter was 32 ms per face
            # against the network's 4.3 ms, i.e. the enhancer was 89% host and
            # the card idled through it. The spelling below is the same maths
            # in fewer, wider passes -- verified against the previous one at
            # max |diff| = 1/255 over the whole frame, which is entirely the
            # final cast rounding instead of truncating (see the return).
            #
            # Three rules came out of measuring it, and they are what to keep:
            #   * a (H,W,1) gate broadcast over (H,W,3) is a STRIDED numpy loop
            #     at 2.47 ms; expanding the gate with cvtColor(GRAY2BGR) and
            #     using cv2.multiply is 0.93 ms for the identical result.
            #   * collapse 1-channel work into 1-channel arrays and fold every
            #     constant into a LUT or a coefficient before it reaches 3
            #     channels.
            #   * cv2 beats numpy on the big buffers (its loops are SIMD and
            #     threaded) and loses on small ones (per-call thread dispatch),
            #     so the 512x512 single-channel gates stay in whichever was
            #     measured faster rather than being made uniform.

            # Gray stays uint8: Sobel reads it directly at CV_32F and the
            # exposure LUT indexes it. The old spelling made a float copy, then
            # clipped and cast it back to uint8 to index -- an exact round trip
            # through two extra full-size passes.
            gray_u8 = cv2.cvtColor(rest_scaled, cv2.COLOR_BGR2GRAY)

            # 1. Structural edge-stop gate.
            #    1/(1 + (hypot(gx,gy)/14)^2) == 196/(196 + gx^2 + gy^2).
            #    The square undoes the square root, so hypot never had to be
            #    computed -- same value, no sqrt over 262k pixels.
            gx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
            m2 = cv2.add(cv2.multiply(gx, gx), cv2.multiply(gy, gy))
            # ~1.0 on smooth skin, -> 0.0 at sharp boundaries (eyelids, pupils, lips)
            skin_gate = cv2.divide(196.0, cv2.add(m2, 196.0))

            # 2. Exposure gate, folded into the texture weight: one 1-channel
            #    array carrying 0.85 * skin_gate * exposure_gate.
            w_tex = cv2.multiply(skin_gate, cls._EXPOSURE_LUT_TEX[gray_u8])

            # 3. High-frequency dermal micro-texture from the pre-restoration crop.
            sigma_texture = max(1.0, target_size / 256.0)
            src_blur = cv2.GaussianBlur(src_f, (0, 0), sigma_texture)
            hf_texture = cv2.subtract(src_f, src_blur)
            # exp(-(hf/16)^2), as one scaled square and one exp.
            core = cv2.exp(cv2.multiply(hf_texture, hf_texture, scale=-1.0 / 256.0))

            hf_std = _global_std(hf_texture)
            injected_texture = cv2.multiply(
                cv2.multiply(hf_texture, core),
                cv2.cvtColor(w_tex, cv2.COLOR_GRAY2BGR))

            # 4. Subtle tactile micro-pore sensor grain if the input was blurry.
            #    w_tex already carries the 0.85, which this term does not want,
            #    so it is divided back out of the scalar rather than out of a
            #    full-size array.
            if hf_std < 3.5:
                k = (1.0 - min(1.0, hf_std / 3.5)) / 0.85
                # numpy, not cv2.add: the grain term is single-channel and is
                # added to ALL THREE channels by broadcast, which is what the
                # previous `+=` did and what makes the grain monochrome rather
                # than coloured. cv2.add does not broadcast (H,W,1) over
                # (H,W,3) -- it raises, and the except below would have turned
                # that into a silent drop to 256 output on exactly the blurry
                # inputs this branch exists for.
                injected_texture += (cls._grain(target_size)
                                     * (w_tex[:, :, np.newaxis] * k))

            # 5. Targeted feature micro-sharpening (eyes, lashes, lips, contours).
            #    0.42*(1 - skin) + 0.12*skin == 0.42 - 0.30*skin, so feature_gate
            #    never has to exist as its own array.
            sigma_sharp = 0.8 * (target_size / 256.0)
            rest_blur = cv2.GaussianBlur(rest_f, (0, 0), sigma_sharp)
            hf_restored = cv2.subtract(rest_f, rest_blur)
            sharpness_amount = cv2.subtract(0.42, cv2.multiply(skin_gate, 0.30))
            sharpened_features = cv2.multiply(
                hf_restored,
                cv2.cvtColor(sharpness_amount, cv2.COLOR_GRAY2BGR))

            # 6. Combine. cv2.add with dtype=CV_8U does the clip and the cast in
            #    one saturating pass instead of np.clip + astype over two. It
            #    ROUNDS where astype truncated, which is the whole of the 1/255
            #    difference from the previous spelling, and is the more correct
            #    of the two.
            return cv2.add(rest_f, cv2.add(injected_texture, sharpened_features),
                           dtype=cv2.CV_8U)
        except Exception as e:
            # THIS RETURN IS NOT FREE, which is why it has to be audible. This
            # method is also what upsamples 256 -> 512, so returning `restored`
            # hands back a 256 image and `sized()` then reports scale 1 instead
            # of 2 — HALF the resolution reaches the frame, and the processor
            # silently becomes plain GPEN-256, the exact outcome the class
            # exists to avoid. Never take a render down over a look filter, but
            # never hide this either. Same lesson as Enhance_UltraMax's
            # `_warned_texture`.
            if not cls._warned_texture:
                cls._warned_texture = True
                print(f"[GPEN 256 Pro] texture/sharpen step skipped: "
                      f"{type(e).__name__}: {e} — output falls back to 256 "
                      f"(scale 1), i.e. plain GPEN-256 quality", flush=True)
            return restored

    # ── run ──────────────────────────────────────────────────────────────────
    def Run(self, source_faceset: FaceSet, target_face: Face, temp_frame: Frame) -> Frame:
        if temp_frame is None or getattr(temp_frame, 'size', 0) == 0:
            return temp_frame, 1

        input_size = temp_frame.shape[1]
        S = self.size
        if temp_frame.shape[0] != S or temp_frame.shape[1] != S:
            src_256 = cv2.resize(temp_frame, (S, S), interpolation=cv2.INTER_CUBIC)
        else:
            src_256 = temp_frame

        # One gather: uint8 BGR HWC -> float32 RGB CHW in [-1, 1]
        x = self._lut[src_256.transpose(2, 0, 1)[::-1]][None]

        def _infer(sess, iob):
            # The CPU-only low-VRAM fallback does not need an I/O binding.
            # Reusing one binding across many CPU calls retains output buffers
            # on this Windows/ORT build; plain Run releases each result.
            if self._cpu_only:
                return sess.run([self.out_name], {self.in_name: x})
            iob.bind_cpu_input(self.in_name, x)
            sess.run_with_iobinding(iob)
            return iob.copy_outputs_to_cpu()

        with exclusive(self.pool, self._session_lock,
                       (self.session, self.io_binding)) as (sess, iob):
            ort_outs = _infer(sess, iob)

        hwc = np.ascontiguousarray(ort_outs[0][0][::-1].transpose(1, 2, 0),
                                   dtype=np.float32)
        del ort_outs

        # Non-finite output guard (e.g. NaN/Inf from overflow)
        if not np.isfinite(hwc.sum()):
            print("[GPEN 256 Pro] non-finite output — using unenhanced frame")
            return sized(temp_frame, input_size)

        np.maximum(hwc, -1.0, out=hwc)
        restored_256 = cv2.convertScaleAbs(hwc, alpha=127.5, beta=127.5)

        # `is_usable` is NOT called here: restored_256 is uint8 and np.isfinite
        # is always True on an integer dtype, so it could never fire. Non-finite
        # is already caught by the sum() above, on the float. `looks_collapsed`
        # is the check that can still see something after the cast.
        if looks_collapsed(restored_256, src_256):
            print("[GPEN 256 Pro] output collapsed (flat) — using unenhanced frame")
            return sized(temp_frame, input_size)

        # 1. Color correction: GPEN luminance with source chrominance
        color_fixed_256 = self._keep_source_colour(restored_256, src_256)

        # 2. Multi-band texture injection, edge-gated sharpening & photo-realism
        enhanced = (self._enhance_textures_and_sharpness_cpu(color_fixed_256, temp_frame, input_size)
                    if self._cpu_only else
                    self._enhance_textures_and_sharpness(color_fixed_256, temp_frame, input_size))

        with self._lock:
            self._faces += 1

        return sized(enhanced, input_size)

    # ── reporting ────────────────────────────────────────────────────────────
    def cost_summary(self):
        with self._lock:
            f = self._faces
        if not f:
            return None
        return (f"[GPEN 256 Pro] {f} faces at {self.size} "
                f"(GPEN-256 neural base, structure-gated micro-textures, photoreal chrominance"
                f"{', pooled' if self.pool is not None else ''})")
