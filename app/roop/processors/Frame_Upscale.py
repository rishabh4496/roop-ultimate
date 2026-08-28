import cv2
import numpy as np
import onnxruntime
import os
import roop.globals
import threading

from roop.utilities import resolve_relative_path
from roop.typing import Frame
from roop.precision_policy import providers_for


def _upscale_providers():
    """Providers for the frame upscaler — TensorRT excluded.

    The ESRGAN-family x4 nets (ultra_sharp, lsdir, esrgan x4, …) overflow /
    mis-convert under TensorRT 'mixed' (FP16), producing all-BLACK output, and
    each model also triggers a slow one-time TRT engine build. They are verified
    correct on CUDA/CPU (FP32), which also starts instantly, so run them there.
    Opt back into TensorRT with ROOP_UPSCALE_TRT=1 (not recommended)."""
    provs = list(roop.globals.execution_providers or [])
    if os.environ.get('ROOP_UPSCALE_TRT', '0') == '1':
        return provs
    def _is_trt(p):
        name = p[0] if isinstance(p, (tuple, list)) else p
        return 'tensorrt' in str(name).lower()
    filtered = [p for p in provs if not _is_trt(p)]
    if not filtered:
        filtered = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    return filtered

class Frame_Upscale():
    plugin_options:dict = None
    model_upscale = None
    devicename = None
    prev_type = None
    _tile_batch_unsupported = False

    processorname = 'upscale'
    type = 'frame_enhancer'

    THREAD_LOCK_UPSCALE = threading.Lock()


    def Initialize(self, plugin_options:dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        self._tile_batch_unsupported = False
        if self.prev_type is not None and self.prev_type != self.plugin_options["subtype"]:
            self.Release()
        self.prev_type = self.plugin_options["subtype"]
        if self.model_upscale is None:
            # replace Mac mps with cpu for the moment
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')
            if self.prev_type == "esrganx4":
                model_path = resolve_relative_path('../models/Frame/real_esrgan_x4.onnx')
                self.scale = 4
            elif self.prev_type == "esrganx2":
                model_path = resolve_relative_path('../models/Frame/real_esrgan_x2.onnx')
                self.scale = 2
            elif self.prev_type == "esrgan_anime_x4":
                model_path = resolve_relative_path('../models/Frame/RealESRGAN_x4plus_anime_6B.onnx')
                self.scale = 4
            elif self.prev_type == "ultrasharp_x4":
                model_path = resolve_relative_path('../models/Frame/ultra_sharp_2_x4.onnx')
                self.scale = 4
            elif self.prev_type == "lsdirx4":
                model_path = resolve_relative_path('../models/Frame/lsdir_x4.onnx')
                self.scale = 4
            elif self.prev_type == "clear_reality_x4":
                model_path = resolve_relative_path('../models/Frame/clear_reality_x4.onnx')
                self.scale = 4
            elif self.prev_type == "span_x4":
                model_path = resolve_relative_path('../models/Frame/span_kendata_x4.onnx')
                self.scale = 4
            elif self.prev_type == "compact_x4":
                # Real-ESRGAN "general" v3 — SRVGGNetCompact. The fastest of the
                # ×4 nets here (fully-convolutional, ~4-5× the RRDB ESRGAN models)
                # while matching Real-ESRGAN quality on real footage. Dynamic-shape
                # ONNX, verified ×4, output in [0,1]. Runs on the CUDA/CPU FP32 path
                # like the other ESRGAN-family nets (see _upscale_providers).
                model_path = resolve_relative_path('../models/Frame/realesr-general-x4v3.onnx')
                self.scale = 4
            elif self.prev_type == "nomos8k_x4":
                model_path = resolve_relative_path('../models/Frame/nomos8k_sc_x4.onnx')
                self.scale = 4
            else:
                # An unrecognised subtype used to leave model_path unbound, so
                # the next line raised NameError — which reads like a broken
                # install rather than "that model name is not one of ours". It
                # reaches here from a saved preset or run-history entry naming a
                # model this build no longer ships. Say so, and fall back to the
                # same default the preview route uses.
                print(f"[Upscale] unknown upscaler '{self.prev_type}' — falling back to esrganx2.")
                self.prev_type = "esrganx2"
                model_path = resolve_relative_path('../models/Frame/real_esrgan_x2.onnx')
                self.scale = 2

            from roop.utilities import get_onnx_session_options
            session_providers, _precision = providers_for(
                f'frame_upscaler:{self.prev_type}', _upscale_providers(), model_path)
            self.model_upscale = onnxruntime.InferenceSession(
                model_path, get_onnx_session_options(), providers=session_providers)
            self.model_inputs = self.model_upscale.get_inputs()
            model_outputs = self.model_upscale.get_outputs()
            self.io_binding = self.model_upscale.io_binding()
            self.io_binding.bind_output(model_outputs[0].name, self.devicename)

    def getProcessedResolution(self, width, height):
        return (width * self.scale, height * self.scale)

# borrowed from facefusion -> https://github.com/facefusion/facefusion
    def prepare_tile_frame(self, tile_frame : Frame) -> Frame:
        tile_frame = np.expand_dims(tile_frame[:, :, ::-1], axis = 0)
        tile_frame = tile_frame.transpose(0, 3, 1, 2)
        tile_frame = tile_frame.astype(np.float32) / 255
        return tile_frame


    def normalize_tile_frame(self, tile_frame : Frame) -> Frame:
        tile_frame = tile_frame.transpose(0, 2, 3, 1).squeeze(0) * 255
        tile_frame = tile_frame.clip(0, 255).astype(np.uint8)[:, :, ::-1]
        return tile_frame

    def create_tile_frames(self, input_frame : Frame, size):
        input_frame = np.pad(input_frame, ((size[1], size[1]), (size[1], size[1]), (0, 0)))
        tile_width = size[0] - 2 * size[2]
        pad_size_bottom = size[2] + tile_width - input_frame.shape[0] % tile_width
        pad_size_right = size[2] + tile_width - input_frame.shape[1] % tile_width
        pad_vision_frame = np.pad(input_frame, ((size[2], pad_size_bottom), (size[2], pad_size_right), (0, 0)))
        pad_height, pad_width = pad_vision_frame.shape[:2]
        row_range = range(size[2], pad_height - size[2], tile_width)
        col_range = range(size[2], pad_width - size[2], tile_width)
        tile_frames = []

        for row_frame in row_range:
            top = row_frame - size[2]
            bottom = row_frame + size[2] + tile_width
            for column_vision_frame in col_range:
                left = column_vision_frame - size[2]
                right = column_vision_frame + size[2] + tile_width
                tile_frames.append(pad_vision_frame[top:bottom, left:right, :])
        return tile_frames, pad_width, pad_height


    def merge_tile_frames(self, tile_frames, temp_width : int, temp_height : int, pad_width : int, pad_height : int, size) -> Frame:
        merge_frame = np.zeros((pad_height, pad_width, 3)).astype(np.uint8)
        tile_width = tile_frames[0].shape[1] - 2 * size[2]
        tiles_per_row = min(pad_width // tile_width, len(tile_frames))

        for index, tile_frame in enumerate(tile_frames):
            tile_frame = tile_frame[size[2]:-size[2], size[2]:-size[2]]
            row_index = index // tiles_per_row
            col_index = index % tiles_per_row
            top = row_index * tile_frame.shape[0]
            bottom = top + tile_frame.shape[0]
            left = col_index * tile_frame.shape[1]
            right = left + tile_frame.shape[1]
            merge_frame[top:bottom, left:right, :] = tile_frame
        merge_frame = merge_frame[size[1] : size[1] + temp_height, size[1]: size[1] + temp_width, :]
        return merge_frame

    @staticmethod
    def _detected_vram_gb() -> float:
        """Best-effort total VRAM for the automatic tile-batch tier."""
        forced = os.environ.get('ROOP_VRAM_GB')
        if forced:
            try:
                return float(forced)
            except ValueError:
                pass
        try:
            import torch
            if torch.cuda.is_available():
                return (torch.cuda.get_device_properties(
                    torch.cuda.current_device()).total_memory / (1024 ** 3))
        except Exception:
            pass
        return 0.0

    def _tile_batch_size(self) -> int:
        """Resolve a bounded tile batch without confusing it with swap batches."""
        raw = os.environ.get('ROOP_UPSCALE_TILE_BATCH')
        if raw is None or raw == '':
            raw = os.environ.get('ROOP_RUNTIME_UPSCALE_TILE_BATCH')
        if raw is None or raw == '':
            # Unknown and sub-7GB hardware stays conservative.  Larger cards
            # also stay at the measured batch-one baseline until this specific
            # model/workload has a persisted benchmark result; VRAM alone does
            # not imply that a wider batch is faster.
            return 1
        try:
            return max(1, min(4, int(raw)))
        except (TypeError, ValueError):
            return 1

    def _run_tile_single(self, tile_frame, input_name, thread_safe):
        if thread_safe:
            return self.model_upscale.run(None, {input_name: tile_frame})[0]
        with self.THREAD_LOCK_UPSCALE:
            self.io_binding.bind_cpu_input(input_name, tile_frame)
            self.model_upscale.run_with_iobinding(self.io_binding)
            return self.io_binding.copy_outputs_to_cpu()[0]

    def _run_tile_batch(self, prepared, input_name, thread_safe):
        """Run one bounded batch and return outputs in input order."""
        if len(prepared) == 1:
            return [self._run_tile_single(prepared[0], input_name, thread_safe)]
        batch = np.ascontiguousarray(np.concatenate(prepared, axis=0))
        if thread_safe:
            output = self.model_upscale.run(None, {input_name: batch})[0]
        else:
            with self.THREAD_LOCK_UPSCALE:
                self.io_binding.bind_cpu_input(input_name, batch)
                self.model_upscale.run_with_iobinding(self.io_binding)
                output = self.io_binding.copy_outputs_to_cpu()[0]
        if not hasattr(output, 'shape') or output.shape[0] != len(prepared):
            raise RuntimeError(
                f'upscaler returned batch {getattr(output, "shape", None)} '
                f'for {len(prepared)} tile inputs')
        return [output[i:i + 1] for i in range(len(prepared))]


    def _run_impl(self, temp_frame: Frame, thread_safe: bool) -> Frame:
        # Tile canvas size. The model runs once per tile, so a small tile (the
        # old fixed 128) means 100+ tiny inferences per 1080p×4 frame — mostly
        # launch/copy overhead, and the fixed 2px overlap is re-computed on every
        # tile edge. A larger tile does the SAME per-pixel work in far fewer,
        # bigger inferences (better GPU efficiency) with proportionally less
        # overlap waste, and — because each output pixel still comes from one
        # tile's interior — the result is effectively unchanged. Tune with
        # ROOP_UPSCALE_TILE (px); lower it if VRAM is tight on heavy ×4 models.
        try:
            tile_px = max(64, int(os.environ.get('ROOP_UPSCALE_TILE', '256')))
        except ValueError:
            tile_px = 256
        size = (tile_px, 8, 2)
        temp_height, temp_width = temp_frame.shape[:2]
        upscale_tile_frames, pad_width, pad_height = self.create_tile_frames(temp_frame, size)
        input_name = self.model_inputs[0].name

        batch_size = (1 if self._tile_batch_unsupported
                      else self._tile_batch_size())
        start = 0
        while start < len(upscale_tile_frames):
            if self._tile_batch_unsupported:
                batch_size = 1
            chunk = upscale_tile_frames[start:start + batch_size]
            prepared = [self.prepare_tile_frame(tile) for tile in chunk]
            try:
                results = self._run_tile_batch(prepared, input_name, thread_safe)
            except Exception as batch_error:
                if len(prepared) <= 1:
                    raise
                self._tile_batch_unsupported = True
                print(f"[Upscale] '{self.prev_type}' rejected tile batch "
                      f"{len(prepared)} ({batch_error!r}); falling back to "
                      "batch 1 for the rest of this run.", flush=True)
                results = [self._run_tile_single(tile, input_name, thread_safe)
                           for tile in prepared]
            for index, result in enumerate(results, start=start):
                upscale_tile_frames[index] = self.normalize_tile_frame(result)
            start += len(chunk)
        final_frame = self.merge_tile_frames(upscale_tile_frames, temp_width * self.scale
                                                    , temp_height * self.scale
                                                    , pad_width * self.scale, pad_height * self.scale
                                                    , (size[0] * self.scale, size[1] * self.scale, size[2] * self.scale))
        return final_frame.astype(np.uint8)

    def Run(self, temp_frame: Frame) -> Frame:
        """Single-thread path with bounded tile batching and io_binding."""
        return self._run_impl(temp_frame, thread_safe=False)

    def RunThreadSafe(self, temp_frame: Frame) -> Frame:
        """Concurrency-safe path — multiple threads may call this on ONE shared
        session at once (plain ORT run, no io_binding, no lock). Tile batches
        remain ordered within each frame."""
        return self._run_impl(temp_frame, thread_safe=True)



    def Release(self):
        del self.model_upscale
        self.model_upscale = None
        del self.io_binding
        self.io_binding = None
        self._tile_batch_unsupported = False

