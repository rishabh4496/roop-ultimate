import asyncio
import os
import shutil
import time
import warnings
import gradio as gr
import roop.globals
import roop.metadata
import roop.utilities as util
import ui.globals as uii
import ui.globals

from ui.tabs.faceswap_tab import faceswap_tab, MASKING_HEAD_JS
from ui.tabs.facemgr_tab import facemgr_tab
from ui.tabs.extras_tab import extras_tab
from ui.tabs.settings_tab import settings_tab

roop.globals.keep_fps = None
roop.globals.keep_frames = None
roop.globals.skip_audio = None
roop.globals.use_batch = None

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def prepare_environment():
    roop.globals.output_path = os.path.abspath(os.path.join(os.getcwd(), "output"))
    os.makedirs(roop.globals.output_path, exist_ok=True)
    if not roop.globals.CFG.use_os_temp_folder:
        os.environ["TEMP"] = os.environ["TMP"] = os.path.abspath(os.path.join(os.getcwd(), "temp"))
    os.makedirs(os.environ["TEMP"], exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = os.environ["TEMP"]
    os.environ['GRADIO_ANALYTICS_ENABLED'] = '0'

def run():
    from roop.core import decode_execution_providers, set_display_ui

    prepare_environment()

    set_display_ui(show_msg)
    if roop.globals.CFG.provider in ("cuda", "tensorrt") and util.has_cuda_device() == False:
        import onnxruntime as _ort
        _available = _ort.get_available_providers()
        _asked = roop.globals.CFG.provider
        if 'DmlExecutionProvider' in _available:
            roop.globals.CFG.provider = "dml"
        elif 'ROCMExecutionProvider' in _available:
            roop.globals.CFG.provider = "rocm"
        else:
            roop.globals.CFG.provider = "cpu"
        # SAY SO. This is a 10x downgrade and it used to happen in silence.
        # Worse, it silently takes the thread count with it: batch_process
        # forces execution_threads to 1 for dml/rocm (suggest_execution_threads),
        # so the user's Max Threads is discarded too. The reported shape of that
        # -- "I set 7 threads, it runs one thread at 1-2 fps" -- is impossible to
        # diagnose from the app without this line.
        print("=" * 75)
        print(f"  [PROVIDER FALLBACK] '{_asked}' was requested but torch reports "
              f"NO USABLE CUDA DEVICE.")
        print(f"                      Falling back to "
              f"'{roop.globals.CFG.provider}'. Expect roughly 1-2 fps.")
        if roop.globals.CFG.provider in ("dml", "rocm"):
            print("                      This provider is also FORCED TO ONE "
                  "WORKER THREAD, whatever Max Threads says.")
        print("                      Almost always a broken torch/CUDA install, "
              "not a setting:")
        print("                      run  env/Scripts/python.exe tests/diag_device.py")
        print("=" * 75)

    # If TensorRT is selected, verify its runtime DLLs are actually loadable.
    # onnxruntime lists TensorrtExecutionProvider as "available" even when the
    # TensorRT runtime libraries (nvinfer.dll etc.) are missing from the system.
    # Attempting to use it then produces error 126 and falls back silently to CPU,
    # losing all GPU acceleration.  Detect this early and fall back to CUDA instead.
    if roop.globals.CFG.provider == "tensorrt":
        _trt_ok = False
        try:
            import tensorrt  # noqa: F401
            _trt_ok = True
        except ImportError:
            pass
        if not _trt_ok:
            print("TensorRT runtime libraries not found – falling back to CUDA provider.")
            roop.globals.CFG.provider = "cuda"

    # Configure execution providers. If TensorRT is selected, also register CUDA as a fallback
    # to prevent silent fallback to CPU if TensorRT initialization fails at run time.
    providers = [roop.globals.CFG.provider]
    if roop.globals.CFG.provider == "tensorrt":
        providers.append("cuda")
    roop.globals.execution_providers = decode_execution_providers(providers)
    gputype = util.get_device()
    if gputype == 'cuda':
        util.print_cuda_info()
        
    print(f'Using provider {roop.globals.execution_providers} - Device:{gputype}')
    
    run_server = True
    uii.ui_restart_server = False
    mycss = """
    /* ════════════════════════════════════════════════════════════════════════
       CARBON DARK  -  legacy Gradio theme (frozen; the React UI is the default)
       Surface scale (deep → elevated):
         #0d0d0d  page / body
         #151515  app container
         #1c1c1c  block / card surfaces
         #242424  nested panels, accordions
         #2c2c2c  input fields, dropdowns
         #383838  borders
         #484848  hover borders / muted separators
       Text:
         #eeeeee  primary   |  #999999  labels/hints  |  #555555  placeholders
       Accent (emerald):
         #50a070  focus ring / active indicator
         #3d8059  primary button fill
         #2e6645  primary button hover
       Danger:
         #7a2020  stop/cancel fill  |  #9a2a2a  hover
    ════════════════════════════════════════════════════════════════════════ */

    :root, .dark {
        color-scheme: dark !important;

        --color-accent:                            #E94560;
        --color-accent-soft:                       rgba(233,69,96,0.15);
        --border-color-accent:                     #E94560;
        --border-color-primary:                    rgba(255, 255, 255, 0.2);

        --link-text-color:                         #E94560;
        --link-text-color-hover:                   #FF6B8B;
        --link-text-color-active:                  #FF6B8B;
        --link-text-color-visited:                 #D63447;

        --body-background-fill:                    transparent;
        --background-fill-primary:                 rgba(255, 255, 255, 0.05);
        --background-fill-secondary:               rgba(255, 255, 255, 0.02);

        --block-background-fill:                   rgba(255, 255, 255, 0.1);
        --block-border-color:                      rgba(255, 255, 255, 0.2);
        --block-border-width:                      1px;
        --block-label-background-fill:             transparent;
        --block-label-text-color:                  #dddddd;
        --block-title-text-color:                  #ffffff;
        --block-info-text-color:                   #bbbbbb;
        --block-radius:                            12px;

        --panel-background-fill:                   rgba(255, 255, 255, 0.05);
        --panel-border-color:                      rgba(255, 255, 255, 0.2);

        --input-background-fill:                   rgba(255, 255, 255, 0.1);
        --input-background-fill-focus:             rgba(255, 255, 255, 0.2);
        --input-border-color:                      rgba(255, 255, 255, 0.3);
        --input-border-color-focus:                #E94560;
        --input-border-color-hover:                rgba(255, 255, 255, 0.4);
        --input-shadow:                            none;
        --input-shadow-focus:                      0 0 0 3px rgba(233,69,96,0.3);
        --input-placeholder-color:                 #dddddd;
        --input-text-color:                        #ffffff;
        --input-radius:                            8px;

        --button-primary-background-fill:          #E94560;
        --button-primary-background-fill-hover:    #D63447;
        --button-primary-text-color:               #ffffff;
        --button-primary-border-color:             #E94560;
        --button-primary-border-color-hover:       #D63447;
        --button-primary-shadow:                   0 4px 15px rgba(233,69,96,0.4);
        --button-primary-shadow-hover:             0 6px 20px rgba(233,69,96,0.6);

        --button-secondary-background-fill:        rgba(255, 255, 255, 0.1);
        --button-secondary-background-fill-hover:  rgba(255, 255, 255, 0.2);
        --button-secondary-text-color:             #ffffff;
        --button-secondary-border-color:           rgba(255, 255, 255, 0.3);
        --button-secondary-border-color-hover:     rgba(255, 255, 255, 0.5);

        --button-cancel-background-fill:           rgba(255, 255, 255, 0.1);
        --button-cancel-background-fill-hover:     #D63447;
        --button-cancel-text-color:                #ffffff;
        --button-cancel-border-color:              rgba(255, 255, 255, 0.3);

        --checkbox-background-color:               rgba(255, 255, 255, 0.1);
        --checkbox-background-color-focus:         rgba(255, 255, 255, 0.2);
        --checkbox-background-color-selected:      #E94560;
        --checkbox-background-color-hover:         rgba(255, 255, 255, 0.2);
        --checkbox-border-color:                   rgba(255, 255, 255, 0.3);
        --checkbox-border-color-focus:             #E94560;
        --checkbox-border-color-selected:          #E94560;
        --checkbox-border-color-hover:             rgba(255, 255, 255, 0.5);
        --checkbox-label-background-fill:          transparent;
        --checkbox-label-background-fill-hover:    rgba(255, 255, 255, 0.05);
        --checkbox-label-background-fill-selected: rgba(233,69,96,0.1);
        --checkbox-label-text-color:               #ffffff;

        --slider-color:                            #E94560;

        --table-odd-background-fill:               rgba(255, 255, 255, 0.02);
        --table-even-background-fill:              rgba(255, 255, 255, 0.05);
        --table-row-focus:                         rgba(233,69,96,0.1);

        --shadow-drop:                             0 8px 32px 0 rgba(31, 38, 135, 0.37);
        --shadow-drop-lg:                          0 8px 32px 0 rgba(31, 38, 135, 0.37);
        --shadow-inset:                            inset 0 1px 3px rgba(255, 255, 255, 0.1);

        --neutral-50:  rgba(255, 255, 255, 0.9);
        --neutral-100: rgba(255, 255, 255, 0.8);
        --neutral-200: rgba(255, 255, 255, 0.7);
        --neutral-300: rgba(255, 255, 255, 0.6);
        --neutral-400: rgba(255, 255, 255, 0.5);
        --neutral-500: rgba(255, 255, 255, 0.4);
        --neutral-600: rgba(255, 255, 255, 0.3);
        --neutral-700: rgba(255, 255, 255, 0.2);
        --neutral-800: rgba(255, 255, 255, 0.1);
        --neutral-900: rgba(255, 255, 255, 0.05);
        --neutral-950: transparent;
    }

    /* ── Page & container ── */
    html, body {
        background: linear-gradient(135deg, #1A1A2E, #16213E, #0F3460, #E94560) !important;
        background-attachment: fixed !important;
        color: #eeeeee !important; 
    }
    .gradio-container, .gradio-container.dark {
        background: transparent !important;
        color: #eeeeee !important;
        max-width: 100% !important;
    }

    /* ── Blocks / cards ── */
    .block, .panel, fieldset, .form {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        transition: border-color 0.15s ease !important;
    }
    .block:hover { border-color: rgba(255, 255, 255, 0.4) !important; }
    .gap, .contain, .tabs { background: transparent !important; border: none !important; }

    /* ── Labels & text ── */
    .block-label, .block > .label-wrap > span,
    .block > label > span, label > span {
        color: #999999 !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
    }
    .block p, .block h1, .block h2, .block h3 { color: #eeeeee !important; }
    .block span { color: #eeeeee !important; }
    .block div  { color: #eeeeee !important; }
    .block .info, .block .description { color: #999999 !important; font-size: 0.82rem !important; }

    /* ── Inputs ── */
    input:not([type=range]):not([type=checkbox]):not([type=radio]), textarea, select {
        background: #2c2c2c !important;
        border: 1px solid #383838 !important;
        border-radius: 6px !important;
        color: #eeeeee !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease, background 0.12s ease !important;
    }
    input:not([type=range]):not([type=checkbox]):not([type=radio]):hover,
    textarea:hover, select:hover { border-color: #484848 !important; }
    input:not([type=range]):not([type=checkbox]):not([type=radio]):focus,
    textarea:focus {
        border-color: #50a070 !important;
        box-shadow: 0 0 0 3px rgba(80,160,112,0.20) !important;
        background: #323232 !important;
        outline: none !important;
    }
    ::placeholder { color: #555555 !important; opacity: 1; }

    /* ── Dropdowns ── */
    .wrap, ul.options {
        background: #2c2c2c !important;
        border: 1px solid #383838 !important;
        border-radius: 6px !important;
        color: #eeeeee !important;
    }
    ul.options { border-radius: 0 0 6px 6px !important; }
    ul.options li {
        color: #eeeeee !important;
        background: #2c2c2c !important;
        padding: 6px 10px !important;
        transition: background 0.1s ease !important;
    }
    ul.options li:hover    { background: #3d8059 !important; color: #f0f0f0 !important; }
    ul.options li.selected { background: #2e6645 !important; color: #f0f0f0 !important; }

    /* ── Buttons ── */
    button {
        border-radius: 6px !important;
        font-weight: 500 !important;
        transition: background 0.12s ease, border-color 0.12s ease,
                    box-shadow 0.12s ease, transform 0.1s ease !important;
        cursor: pointer !important;
    }
    button:hover  { transform: translateY(-1px) !important; }
    button:active { transform: translateY(0px)  !important; }

    button.primary, .btn-primary {
        background: #3d8059 !important;
        border-color: #3d8059 !important;
        color: #f0f0f0 !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.4) !important;
    }
    button.primary:hover, .btn-primary:hover {
        background: #2e6645 !important;
        border-color: #2e6645 !important;
        box-shadow: 0 4px 14px rgba(61,128,89,0.38) !important;
    }
    button.secondary, .btn-secondary {
        background: #242424 !important;
        border-color: #383838 !important;
        color: #bbbbbb !important;
    }
    button.secondary:hover, .btn-secondary:hover {
        background: #2c2c2c !important;
        border-color: #484848 !important;
        color: #eeeeee !important;
    }
    button.stop, button.cancel, .btn-cancel {
        background: #7a2020 !important;
        border-color: #7a2020 !important;
        color: #f0f0f0 !important;
    }
    button.stop:hover, button.cancel:hover, .btn-cancel:hover {
        background: #9a2a2a !important;
        border-color: #9a2a2a !important;
    }

    /* ── Sliders ── */
    input[type=range] { accent-color: #50a070; }
    input[type=range]::-webkit-slider-thumb {
        background: #50a070 !important;
        transition: transform 0.1s ease !important;
    }
    input[type=range]::-webkit-slider-thumb:hover { transform: scale(1.25) !important; }
    input[type=range]::-moz-range-thumb            { background: #50a070 !important; }
    input[type=range]::-webkit-slider-runnable-track { background: #2c2c2c !important; }

    /* ── Checkboxes – native rendering for reliable checked state ── */
    input[type=checkbox] {
        appearance: auto !important;
        -webkit-appearance: checkbox !important;
        accent-color: #50a070 !important;
        width: 16px !important;
        height: 16px !important;
        cursor: pointer !important;
        background: unset !important;
        border: unset !important;
        box-shadow: none !important;
        transition: transform 0.1s ease !important;
    }
    input[type=checkbox]:hover { transform: scale(1.1) !important; }
    input[type=radio]  { accent-color: #50a070; }

    /* ── Upload / drop zones ── */
    .upload-container, .file-preview, .drop-container {
        background: #1c1c1c !important;
        border: 2px dashed #383838 !important;
        border-radius: 8px !important;
        color: #999999 !important;
        transition: border-color 0.15s ease, background 0.15s ease !important;
    }
    .drop-container:hover {
        border-color: #50a070 !important;
        background: rgba(80,160,112,0.04) !important;
    }

    /* ── Gallery ── */
    .gallery, .gallery-container, .grid-container { background: transparent !important; }
    .gallery-item, .thumbnail-item {
        border: 1px solid #383838 !important;
        border-radius: 6px !important;
        overflow: hidden !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease, transform 0.15s ease !important;
    }
    .gallery-item:hover, .thumbnail-item:hover {
        border-color: #50a070 !important;
        box-shadow: 0 4px 16px rgba(80,160,112,0.25) !important;
        transform: translateY(-2px) !important;
    }

    /* ── Header bar ── */
    .compact {
        background: #111111 !important;
        border-bottom: 1px solid #272727 !important;
        padding: 6px 12px !important;
    }

    /* ── Tab bar ── */
    .tab-nav {
        background: transparent !important;
        border-bottom: 1px solid rgba(255,255,255,0.2) !important;
        padding: 0 4px !important;
    }
    .tab-nav button {
        color: #888888 !important;
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        padding: 10px 18px !important;
        font-weight: 500 !important;
        transform: none !important;
        transition: color 0.15s ease, border-color 0.15s ease, background 0.15s ease !important;
    }
    .tab-nav button:hover {
        color: #cccccc !important;
        background: rgba(255,255,255,0.04) !important;
        border-bottom-color: #484848 !important;
        transform: none !important;
    }
    .tab-nav button.selected {
        color: #f0f0f0 !important;
        border-bottom: 2px solid #50a070 !important;
        font-weight: 700 !important;
        background: rgba(80,160,112,0.07) !important;
        transform: none !important;
    }

    /* ── Accordion headers ── */
    .label-wrap {
        background: #242424 !important;
        border: 1px solid #383838 !important;
        border-radius: 6px !important;
        cursor: pointer !important;
        transition: background 0.12s ease, border-color 0.12s ease !important;
    }
    .label-wrap:hover { background: #2c2c2c !important; border-color: #484848 !important; }
    .label-wrap span  { color: #eeeeee !important; font-weight: 500 !important; }

    /* ── Scrollbars ── */
    ::-webkit-scrollbar       { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #181818; border-radius: 3px; }
    ::-webkit-scrollbar-thumb { background: #383838; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #50a070; }

    /* ── Progress / generating ── */
    .progress-bar {
        background: linear-gradient(90deg, #3d8059, #50a070) !important;
        border-radius: 3px !important;
    }
    .generating { border-color: #50a070 !important; }

    /* ── Toasts ── */
    /* The wrapper is always in the DOM; keep it invisible when empty */
    .toast-wrap {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    /* Style individual toast items instead */
    .toast-wrap .toast {
        background: #1c1c1c !important;
        border: 1px solid #383838 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.55) !important;
    }
    .toast-title { color: #eeeeee !important; font-weight: 600 !important; }
    .toast-text  { color: #999999 !important; }

    /* ── Markdown / prose ── */
    .prose, .prose p, .prose li { color: #eeeeee !important; }
    .prose a       { color: #50a070 !important; }
    .prose a:hover { color: #6dba8a !important; }
    .prose code {
        background: #2c2c2c !important;
        border: 1px solid #383838 !important;
        border-radius: 4px !important;
        color: #6dba8a !important;
        padding: 1px 5px !important;
    }

    /* ── Preserved layout rules ── */
    span { color: var(--block-info-text-color) }
    /* Remove the visible block border from the version-info HTML element */
    #versions { border: none !important; background: transparent !important; }
    #fixedheight {
        max-height: 238.4px;
        overflow-y: auto !important;
    }
    .image-container.svelte-1l6wqyv { height: 100% }
    /* Original frame component — kept in DOM for JS access but never shown on the page */
    #roop_original_frame { display: none !important; }
    """

    while run_server:
        # Clean Gradio temp dir on each (re)start to prevent asyncio event loop
        # mismatch errors caused by stale asyncio.locks.Event objects from a
        # previous server instance referencing the old event loop.
        try:
            gradio_temp = os.environ.get("GRADIO_TEMP_DIR", "")
            if gradio_temp and os.path.exists(gradio_temp):
                shutil.rmtree(gradio_temp, ignore_errors=True)
                os.makedirs(gradio_temp, exist_ok=True)
        except Exception:
            pass
        server_name = roop.globals.CFG.server_name
        if server_name is None or len(server_name) < 1:
            server_name = None
        try:
            server_port = int(os.environ.get("ROOP_GRADIO_PORT", roop.globals.CFG.server_port))
        except ValueError:
            server_port = roop.globals.CFG.server_port
        if server_port <= 0:
            server_port = None
        ssl_verify = True
        with gr.Blocks(title=f'{roop.metadata.name} {roop.metadata.version}', theme=gr.themes.Base(), css=mycss, delete_cache=(60, 86400), head=MASKING_HEAD_JS) as ui:
            with gr.Row(variant='compact'):
                    gr.HTML(util.create_version_html(), elem_id="versions")
                    bt_save_session = gr.Button("💾 Save Settings", size='sm', variant='primary', scale=0)
                    bt_load_session = gr.Button("📂 Load Settings", size='sm', scale=0)
            bt_destfiles = faceswap_tab()
            facemgr_tab()
            extras_tab(bt_destfiles)
            settings_tab()
            # Wire Save/Load after all tabs so ui.globals component refs are populated
            _comps = _session_components()
            bt_save_session.click(fn=save_session, inputs=_comps, outputs=[])
            bt_load_session.click(fn=load_session, inputs=[], outputs=_comps)
        launch_browser = roop.globals.CFG.launch_browser

        uii.ui_restart_server = False

        # Suppress the benign Windows ProactorEventLoop noise that fires when a
        # browser tab drops an SSE/WebSocket connection mid-stream.
        # ConnectionResetError [WinError 10054] is harmless — Gradio handles it
        # internally — but asyncio prints a traceback to stderr by default.
        def _suppress_connection_reset(loop, context):
            exc = context.get('exception')
            if isinstance(exc, ConnectionResetError):
                return
            loop.default_exception_handler(context)
        try:
            asyncio.get_event_loop().set_exception_handler(_suppress_connection_reset)
        except RuntimeError:
            pass  # no running loop yet; Gradio will create one

        try:
            ui.queue().launch(inbrowser=launch_browser, server_name=server_name, server_port=server_port, share=roop.globals.CFG.server_share, ssl_verify=ssl_verify, prevent_thread_lock=True, show_error=True)
        except Exception as e:
            print(f'Exception {e} when launching Gradio Server!')
            uii.ui_restart_server = True
            run_server = False
        try:
            while uii.ui_restart_server == False:
                time.sleep(1.0)

        except (KeyboardInterrupt, OSError):
            print("Keyboard interruption in main thread... closing server.")
            run_server = False
        ui.close()


def show_msg(msg: str):
    gr.Info(msg)


_SESSION_CFG_KEYS = [
    'face_detection_mode', 'num_swap_steps', 'selected_enhancer', 'max_face_distance',
    'subsample_upscale', 'blend_ratio', 'video_swapping_method', 'no_face_action',
    'vr_mode', 'autorotate_faces', 'skip_audio', 'keep_frames', 'wait_after_extraction',
    'output_method', 'mask_engine', 'mask_clip_text', 'show_mask_offsets',
    'restore_original_mouth', 'mask_top', 'mask_bottom', 'mask_left', 'mask_right',
    'face_mask_blend', 'mouth_mask_blend',
    'mouth_top_scale', 'mouth_bottom_scale', 'mouth_left_scale', 'mouth_right_scale',
    'use_3d_recon',
    'use_source_bank', 'use_frontalization', 'frontalization_threshold', 'swap_model',
]


def _session_components():
    return [
        ui.globals.ui_selected_face_detection,
        ui.globals.ui_num_swap_steps,
        ui.globals.ui_selected_enhancer,
        ui.globals.ui_max_face_distance,
        ui.globals.ui_upscale,
        ui.globals.ui_blend_ratio,
        ui.globals.ui_video_swapping_method,
        ui.globals.ui_no_face_action,
        ui.globals.ui_vr_mode,
        ui.globals.ui_autorotate,
        ui.globals.ui_skip_audio,
        ui.globals.ui_keep_frames,
        ui.globals.ui_wait_after_extraction,
        ui.globals.ui_output_method,
        ui.globals.ui_selected_mask_engine,
        ui.globals.ui_clip_text,
        ui.globals.ui_chk_showmaskoffsets,
        ui.globals.ui_chk_restoreoriginalmouth,
        ui.globals.ui_mask_top,
        ui.globals.ui_mask_bottom,
        ui.globals.ui_mask_left,
        ui.globals.ui_mask_right,
        ui.globals.ui_face_mask_blend,
        ui.globals.ui_mouth_mask_blend,
        ui.globals.ui_mouth_top_scale,
        ui.globals.ui_mouth_bottom_scale,
        ui.globals.ui_mouth_left_scale,
        ui.globals.ui_mouth_right_scale,
        ui.globals.ui_chk_use_3d_recon,
        ui.globals.ui_chk_use_source_bank,
        ui.globals.ui_chk_use_frontalization,
        ui.globals.ui_sld_frontalization_threshold,
        ui.globals.ui_dd_swap_model,
    ]


def save_session(*values):
    cfg = roop.globals.CFG
    for key, val in zip(_SESSION_CFG_KEYS, values):
        setattr(cfg, key, val)
    cfg.save()
    gr.Info('Settings saved!')


def load_session():
    roop.globals.CFG.load()
    cfg = roop.globals.CFG
    return tuple(getattr(cfg, key) for key in _SESSION_CFG_KEYS)

