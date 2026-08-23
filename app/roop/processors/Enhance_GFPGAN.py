from typing import Any, List, Callable
import cv2 
import numpy as np
import onnxruntime
import roop.globals

from roop.typing import Face, Frame, FaceSet
from roop.utilities import resolve_relative_path
from roop.processors.enhance_common import (is_usable, sized,
                                            fp32_trt_providers,
                                            looks_collapsed)


# THREAD_LOCK = threading.Lock()


class Enhance_GFPGAN():
    plugin_options:dict = None

    model_gfpgan = None
    name = None
    devicename = None

    processorname = 'gfpgan'
    type = 'enhance'
    _warned_collapse = False
    # FFHQ-trained — see Enhance_CodeFormer.model_template for the measured
    # mismatch against each swapper's crop, and ProcessMgr for the re-warp.
    model_template = 'ffhq_512'


    def Initialize(self, plugin_options:dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.model_gfpgan is None:
            model_path = resolve_relative_path('../models/GFPGANv1.4.onnx')
            # FORCED FP32 UNDER TENSORRT, and this one was silent.
            #
            # GFPGAN v1.4 does not survive TensorRT's FP16 kernels, but it does
            # not overflow to NaN the way GPEN 1024/2048 does — it COLLAPSES.
            # Measured 2026-08-24 on an RTX 4070, same input, same pre/post:
            #
            #   TRT fp16   raw range [-0.47, -0.14]  pixel std 16.0  detail 0.08
            #   TRT fp32   raw range [-1.00,  1.00]  pixel std 65.2  detail 4.35
            #   CUDA       raw range [-1.00,  1.00]  pixel std 65.2  detail 4.35
            #
            # fp32 matches the CUDA reference to 0.03/255; fp16 differs from it
            # by 59/255. Every fp16 value is finite, so `is_usable` never fired
            # and the enhancer shipped returning a uniform grey face that still
            # looked like an image. It cost 65 s to build the FP32 engine once
            # (cached thereafter) and runs at 93 ms against CUDA's 568 ms.
            providers = fp32_trt_providers(roop.globals.execution_providers,
                                           'gfpgan')
            self.model_gfpgan = onnxruntime.InferenceSession(model_path, None, providers=providers)
            # replace Mac mps with cpu for the moment
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')

        self.name = self.model_gfpgan.get_inputs()[0].name
        self.output_name = self.model_gfpgan.get_outputs()[0].name

    def Run(self, source_faceset: FaceSet, target_face: Face, temp_frame: Frame) -> Frame:
        # preprocess
        input_size = temp_frame.shape[1]
        temp_frame = cv2.resize(temp_frame, (512, 512), interpolation=cv2.INTER_CUBIC)
        fallback_bgr = temp_frame   # resized input, kept for the non-finite guard

        temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
        temp_frame = temp_frame.astype('float32') / 255.0
        temp_frame = (temp_frame - 0.5) / 0.5
        temp_frame = np.expand_dims(temp_frame, axis=0).transpose(0, 3, 1, 2)

        io_binding = self.model_gfpgan.io_binding()
        io_binding.bind_cpu_input(self.name, temp_frame)
        io_binding.bind_output(self.output_name, self.devicename)
        self.model_gfpgan.run_with_iobinding(io_binding)
        ort_outs = io_binding.copy_outputs_to_cpu()
        result = ort_outs[0][0]

        # np.clip does not remove NaN and uint8(NaN) is 0, so a single
        # overflowed value paints black and a saturated graph paints a black
        # FACE — silently. See enhance_common.is_usable.
        if not is_usable(result):
            print("[GFPGAN] non-finite output — using unenhanced frame "
                  "(FP16 overflow? try an fp32 provider)")
            return sized(fallback_bgr.astype(np.uint8), input_size)

        # post-process
        result = np.clip(result, -1, 1)
        result = (result + 1) / 2
        result = result.transpose(1, 2, 0) * 255.0
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        result = result.astype(np.uint8)

        # The guard `is_usable` cannot provide: a finite but COLLAPSED output.
        # The FP32 provider above is the real fix; this is the net that would
        # have caught the failure in the first place instead of letting a flat
        # grey face render for months.
        if looks_collapsed(result, fallback_bgr):
            if not Enhance_GFPGAN._warned_collapse:
                Enhance_GFPGAN._warned_collapse = True
                print("[GFPGAN] output has collapsed to a near-uniform image "
                      "(finite, but no dynamic range) — using the unenhanced "
                      "frame. This is the TensorRT FP16 failure; unset "
                      "ROOP_GFPGAN_FP16 to get the forced-FP32 engine back.")
            return sized(fallback_bgr.astype(np.uint8), input_size)

        return sized(result, input_size)


    def Release(self):
        del self.model_gfpgan
        self.model_gfpgan = None











