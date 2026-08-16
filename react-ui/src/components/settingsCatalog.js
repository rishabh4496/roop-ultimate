// ── What the Settings panel exposes, as data ──────────────────────────────
// The command palette needs to offer settings by name, and App.jsx cannot read
// them out of Settings.jsx: that panel is a lazy chunk, and its controls are
// JSX rather than a list. So the catalogue lives here, importable by both.
//
// Duplication between this and the panel would be exactly the kind of silent
// drift the light-theme class list used to cause — a setting renamed in one
// place and not the other just quietly stops being findable. So
// test_ui_settings_catalog.py holds the two in step: every key here must be
// bound by Settings.jsx, and every key Settings.jsx binds must appear here.
//
// `section` matches the panel's section title, so a palette hit can say where
// it is going. `label` should match the control's label — that is what the
// panel's own search box matches on, and jumping works by handing it that
// string.
export const SETTINGS_CATALOG = [
  // Server
  { key: 'server_share', label: 'Public server (share)', section: 'Server' },
  { key: 'clear_output', label: 'Clear output folder before each run', section: 'Server' },
  { key: 'server_name', label: 'Server name', section: 'Server' },
  { key: 'server_port', label: 'Server port', section: 'Server' },
  { key: 'output_template', label: 'Filename output template', section: 'Server' },
  { key: 'faceset_library_path', label: 'Faceset library folder', section: 'Server' },

  // Performance
  { key: 'provider', label: 'Provider', section: 'Performance' },
  { key: 'trt_precision', label: 'Precision mode (TensorRT)', section: 'Performance' },
  { key: 'force_cpu', label: 'Force CPU for face analyser', section: 'Performance' },
  { key: 'auto_thread_selection', label: 'Auto thread selection', section: 'Performance' },
  { key: 'face_detector_threshold', label: 'Face detection threshold', section: 'Performance' },
  { key: 'face_detector_nms', label: 'Overlap NMS threshold', section: 'Performance' },
  { key: 'max_threads', label: 'Max threads', section: 'Performance' },
  { key: 'memory_limit', label: 'Max memory (GB)', section: 'Performance' },

  // Advanced performance
  { key: 'perf_trt_pool', label: 'Swapper TRT pool', section: 'Advanced performance' },
  { key: 'perf_detmask_pool', label: 'Detect/Mask pool', section: 'Advanced performance' },
  { key: 'perf_detector_pool', label: 'Detector pool', section: 'Advanced performance' },
  { key: 'perf_expr_pool', label: 'Expression pool', section: 'Advanced performance' },
  { key: 'perf_encoder_preset', label: 'Encoder preset', section: 'Advanced performance' },
  { key: 'perf_nvdec', label: 'GPU video decode (NVDEC)', section: 'Advanced performance' },
  { key: 'perf_batch_swap', label: 'Batched swap', section: 'Advanced performance' },
  { key: 'perf_profile', label: 'Stage profiling (terminal)', section: 'Advanced performance' },

  // Output
  { key: 'output_image_format', label: 'Image format', section: 'Output' },
  { key: 'output_video_format', label: 'Video format', section: 'Output' },
  { key: 'output_video_codec', label: 'Video codec', section: 'Output' },
  { key: 'video_quality', label: 'Video quality', section: 'Output' },
  { key: 'use_os_temp_folder', label: 'Use OS temp folder', section: 'Output' },
  { key: 'output_show_video', label: 'Show video in browser (re-encodes)', section: 'Output' },
];

// The event the palette fires to send the Settings panel to one control. The
// panel is a lazy chunk that may not be mounted yet when the command runs, so
// this is a window event with a retry on the panel's side rather than a direct
// call — the same bus pattern the Face Swap actions already use.
export const FOCUS_SETTING_EVENT = 'roop:focus-setting';

export const focusSetting = (key) => {
  window.dispatchEvent(new CustomEvent(FOCUS_SETTING_EVENT, { detail: { key } }));
};
