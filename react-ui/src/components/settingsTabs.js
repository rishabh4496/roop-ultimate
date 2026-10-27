// Which Settings tab a setting lives on.
//
// The panel used to be one masonry of sixteen sections. It is now five tabs:
//
//   providers    Execution Providers & TensorRT Engines
//   cpu          CPU P/E-Core & Thread Pools
//   paths        Paths & Storage
//   diagnostics  Runtime Diagnostics & Benchmark Runner
//   general      everything else: theme, the 3-stage pipeline, identity,
//                mouth/occlusion, lighting, detection defaults
//
// (`general` is a fifth tab the brief did not name. Those sections are real
// settings with nowhere else to go, and dropping them would have removed controls.)
//
// The command palette opens a setting by key (App.jsx -> focusSetting), and the
// control it names may be on a tab that is not showing, so the tab for a key has
// to be computable from the catalog's section name plus the few keys that sit in a
// section but belong elsewhere. Kept as data, here, so the test can check it.

export const TAB_LABELS = {
  providers: 'Execution Providers & TensorRT Engines',
  cpu: 'CPU P/E-Core & Thread Pools',
  paths: 'Paths & Storage',
  diagnostics: 'Runtime Diagnostics & Benchmark Runner',
  general: 'Appearance & Pipeline Defaults',
};

// Keys whose section says one thing and whose tab says another.
const KEY_TAB = {
  // "Performance" holds the providers, but also the thread controls ...
  auto_thread_selection: 'cpu',
  max_threads: 'cpu',
  memory_limit: 'cpu',
  // ... and three detector defaults that have nothing to do with performance.
  face_detector_threshold: 'general',
  face_detector_nms: 'general',
  detector_scale_pyramid: 'general',
  // "Advanced performance" is mostly pools and threads, with the GPU-side knobs mixed in.
  trt_builder_optimization_level: 'providers',
  trt_auxiliary_streams: 'providers',
  trt_cuda_graph: 'providers',
  perf_ort_arena_strategy: 'providers',
  perf_cudnn_conv_algo: 'providers',
  perf_gpu_mem_limit: 'providers',
  perf_gpu_affine: 'providers',
  perf_nvdec: 'providers',
  perf_nvenc_preset: 'providers',
  perf_encoder_preset: 'providers',
};

const SECTION_TAB = (section) => {
  const s = String(section || '').toLowerCase();
  if (s.startsWith('advanced performance')) return 'cpu';
  if (s.startsWith('performance')) return 'providers';
  if (s.startsWith('server') || s.startsWith('output')) return 'paths';
  return 'general';
};

/** The tab holding setting `key`, given its catalog section name. */
export const tabForSetting = (key, section) => KEY_TAB[key] || SECTION_TAB(section);
