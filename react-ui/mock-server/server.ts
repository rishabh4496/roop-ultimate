import express from 'express';
import http from 'http';
import path from 'path';
import { fileURLToPath } from 'url';
import { WebSocketServer, WebSocket } from 'ws';
import { createServer as createViteServer } from 'vite';
import multer from 'multer';

// This is a MOCK of app/api.py: simulated swaps, invented telemetry, SVG
// placeholders instead of frames. It exists so the React UI can be developed
// without a GPU or the Python backend. Nothing in production runs it.
// Paths are relative to this file, not to the working directory.
const UI_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const upload = multer({ limits: { fileSize: 500 * 1024 * 1024 } });

// SVG helper to generate crisp preview thumbnails
function createSampleFaceSvg(label: string, subtitle: string, bg1: string, bg2: string, seed = 1): string {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
    <defs>
      <linearGradient id="bg_${seed}" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="${bg1}" />
        <stop offset="100%" stop-color="${bg2}" />
      </linearGradient>
      <filter id="shadow_${seed}">
        <feDropShadow dx="0" dy="8" stdDeviation="16" flood-color="#000" flood-opacity="0.4" />
      </filter>
    </defs>
    <rect width="512" height="512" fill="url(#bg_${seed})" />
    <!-- Grid overlay -->
    <path d="M0 64h512M0 128h512M0 192h512M0 256h512M0 320h512M0 384h512M0 448h512M64 0v512M128 0v512M192 0v512M256 0v512M320 0v512M384 0v512M448 0v512" stroke="rgba(255,255,255,0.05)" stroke-width="1"/>
    
    <!-- Stylized Face silhouette -->
    <g filter="url(#shadow_${seed})">
      <!-- Head / Hair -->
      <path d="M176 180 C160 100, 352 100, 336 180 C368 200, 368 300, 336 340 C304 400, 208 400, 176 340 C144 300, 144 200, 176 180 Z" fill="#2d3748" />
      <!-- Face shape -->
      <ellipse cx="256" cy="245" rx="76" ry="96" fill="#fed7aa" />
      <!-- Eyes -->
      <ellipse cx="226" cy="230" rx="9" ry="6" fill="#1e293b" />
      <ellipse cx="286" cy="230" rx="9" ry="6" fill="#1e293b" />
      <!-- Pupils / reflection -->
      <circle cx="228" cy="229" r="2.5" fill="#fff" />
      <circle cx="288" cy="229" r="2.5" fill="#fff" />
      <!-- Nose bridge & tip -->
      <path d="M256 226 L254 252 L262 254" stroke="#d97706" stroke-width="2.5" fill="none" stroke-linecap="round" />
      <!-- Mouth -->
      <path d="M236 280 Q256 294 276 280" stroke="#b91c1c" stroke-width="3" fill="none" stroke-linecap="round" />
      <!-- Jaw contour line -->
      <path d="M192 240 Q256 350 320 240" stroke="rgba(217, 119, 6, 0.4)" stroke-width="2" fill="none" stroke-dasharray="4 4"/>
      
      <!-- Keypoint landmarks dots -->
      <circle cx="226" cy="230" r="3" fill="#38bdf8" />
      <circle cx="286" cy="230" r="3" fill="#38bdf8" />
      <circle cx="256" cy="254" r="3" fill="#38bdf8" />
      <circle cx="236" cy="280" r="3" fill="#38bdf8" />
      <circle cx="276" cy="280" r="3" fill="#38bdf8" />
    </g>

    <!-- UI Badge / Text -->
    <rect x="24" y="420" width="464" height="68" rx="12" fill="rgba(15, 23, 42, 0.85)" stroke="rgba(255,255,255,0.15)" stroke-width="1" />
    <text x="44" y="450" fill="#f8fafc" font-family="system-ui, -apple-system, sans-serif" font-size="17" font-weight="700">${label}</text>
    <text x="44" y="472" fill="#94a3b8" font-family="system-ui, -apple-system, sans-serif" font-size="13">${subtitle}</text>
    <rect x="420" y="438" width="52" height="24" rx="6" fill="#0284c7" />
    <text x="446" y="455" fill="#fff" font-family="system-ui, -apple-system, sans-serif" font-size="11" font-weight="700" text-anchor="middle">512px</text>
  </svg>`;
  return `data:image/svg+xml;base64,${Buffer.from(svg).toString('base64')}`;
}

function createSampleTargetSvg(label: string, frame: number, total: number, isVideo: boolean, seed = 2): string {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 360" width="640" height="360">
    <defs>
      <linearGradient id="tbg_${seed}" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#0f172a" />
        <stop offset="100%" stop-color="#1e1b4b" />
      </linearGradient>
    </defs>
    <rect width="640" height="360" fill="url(#tbg_${seed})" />
    <!-- Subtle camera scene bokeh -->
    <circle cx="140" cy="110" r="90" fill="#3b82f6" opacity="0.12" />
    <circle cx="510" cy="240" r="120" fill="#6366f1" opacity="0.1" />
    <circle cx="340" cy="180" r="60" fill="#06b6d4" opacity="0.15" />
    
    <!-- Target Actor Figure -->
    <g transform="translate(60, 10)">
      <ellipse cx="260" cy="140" rx="55" ry="70" fill="#fed7aa" />
      <!-- Eyes and features -->
      <ellipse cx="240" cy="130" rx="7" ry="5" fill="#0f172a" />
      <ellipse cx="280" cy="130" rx="7" ry="5" fill="#0f172a" />
      <path d="M260 128 L258 148 L264 150" stroke="#ea580c" stroke-width="2" fill="none" />
      <path d="M246 170 Q260 180 274 170" stroke="#991b1b" stroke-width="2.5" fill="none" />
      <!-- Body / Shoulders -->
      <path d="M170 290 C180 210, 340 210, 350 290 Z" fill="#334155" />
      <!-- Detection bounding box overlay -->
      <rect x="190" y="60" width="140" height="160" rx="8" fill="none" stroke="#22c55e" stroke-width="2" stroke-dasharray="6 4" />
      <rect x="190" y="42" width="80" height="18" rx="4" fill="#22c55e" />
      <text x="230" y="55" fill="#0f172a" font-family="system-ui" font-size="11" font-weight="800" text-anchor="middle">Face #0 · 99%</text>
    </g>

    <!-- Cinema Bars / Info Footer -->
    <rect x="0" y="305" width="640" height="55" fill="rgba(10, 15, 30, 0.9)" />
    <text x="24" y="332" fill="#f8fafc" font-family="system-ui" font-size="14" font-weight="600">${label}</text>
    <text x="24" y="348" fill="#94a3b8" font-family="system-ui" font-size="11">${isVideo ? `Video clip (30 FPS) · Frame ${frame} of ${total}` : `Still high-res photograph`}</text>
    
    <rect x="530" y="320" width="86" height="24" rx="4" fill="#1e293b" stroke="#334155" stroke-width="1" />
    <text x="573" y="336" fill="#38bdf8" font-family="monospace" font-size="11" font-weight="700" text-anchor="middle">${isVideo ? `FR ${frame}/${total}` : 'STILL'}</text>
  </svg>`;
  return `data:image/svg+xml;base64,${Buffer.from(svg).toString('base64')}`;
}

// Initial source faces and target videos/images
const sourceThumbs = [
  createSampleFaceSvg('Source Alpha (Frontal)', 'Pitch: 0.0° · Yaw: -1.2° · Roll: 0.5°', '#1e293b', '#0f172a', 1),
  createSampleFaceSvg('Source Beta (3/4 Angle)', 'Pitch: -2.4° · Yaw: 24.1° · Roll: 1.1°', '#1e1b4b', '#090d16', 2),
  createSampleFaceSvg('Source Gamma (Studio)', 'Pitch: 1.1° · Yaw: -8.4° · Roll: -0.2°', '#14532d', '#052e16', 3),
];

const sourceInfo = [
  { index: 0, path: 'source_alpha.png', width: 512, height: 512, count: 1, poses: ['Frontal'], angles: { pitch: 0.0, yaw: -1.2, roll: 0.5 }, quality: 0.98, landmarks: 106 },
  { index: 1, path: 'source_beta.png', width: 512, height: 512, count: 1, poses: ['Right 3/4'], angles: { pitch: -2.4, yaw: 24.1, roll: 1.1 }, quality: 0.95, landmarks: 106 },
  { index: 2, path: 'source_gamma.png', width: 512, height: 512, count: 1, poses: ['Frontal'], angles: { pitch: 1.1, yaw: -8.4, roll: -0.2 }, quality: 0.97, landmarks: 106 },
];

const targetEntries = [
  {
    name: 'interview_scene_1080p.mp4',
    filename: 'interview_scene_1080p.mp4',
    preview_available: true,
    startframe: 0,
    endframe: 180,
    start_frame: 0,
    end_frame: 180,
    frames: 180,
    fps: 30,
    is_video: true,
  },
  {
    name: 'commercial_portrait.png',
    filename: 'commercial_portrait.png',
    preview_available: true,
    startframe: 0,
    endframe: 1,
    start_frame: 0,
    end_frame: 1,
    frames: 1,
    fps: 0,
    is_video: false,
  },
];

const targetThumbs = [
  createSampleTargetSvg('interview_scene_1080p.mp4', 1, 180, true, 101),
  createSampleTargetSvg('commercial_portrait.png', 1, 1, false, 102),
];

const targetFacesList: string[] = [
  createSampleFaceSvg('Target Face (Lead)', 'Person 1 · Detected at frame 1', '#1e293b', '#0f172a', 51),
];
let targetGroupsList: number[] = [0];
let targetNamesList: string[] = ['Actor Lead (Person 1)'];
let targetFacesInfoList: any[] = [
  { index: 0, person: 'Person 1', landmarks: 106, gender: 'female', age: 26, bbox: [190, 60, 140, 160], pose: 'Front' },
];
let storedProfiles: any[] = [];

function getTargetFacesPayload(extra: Record<string, any> = {}) {
  return {
    target_faces: targetFacesList,
    target_groups: targetGroupsList,
    target_names: targetNamesList,
    target_faces_info: targetFacesInfoList,
    count: targetFacesList.length,
    ...extra,
  };
}

// App global state
let activeHardwareProfile = 'desktop'; // 'desktop' (RTX 4070) or 'laptop' (RTX 3060)

let appSettings: Record<string, any> = {
  provider: 'cuda',
  trt_precision: 'mixed',
  force_cpu: false,
  auto_thread_selection: true,
  face_detector_threshold: 0.50,
  face_detector_nms: 0.3,
  detector_scale_pyramid: 'auto',
  max_threads: 10,
  memory_limit: 0,
  perf_trt_pool: 2,
  perf_detmask_pool: 2,
  perf_detector_pool: 2,
  perf_expr_pool: 1,
  perf_encoder_preset: 'auto',
  perf_nvdec: 'auto',
  perf_batch_swap: 'auto',
  perf_profile: 'Desktop Workstation (RTX 4070)',
  cpu_ort_intra_threads: 'auto',
  cpu_ort_inter_threads: 'auto',
  cpu_ffmpeg_threads: 'auto',
  perf_ort_arena_strategy: 'auto',
  perf_cudnn_conv_algo: 'auto',
  perf_gpu_mem_limit: 'auto',
  recognizer: 'default',
  face_demarcate: 'auto',
  track_stitch: 'auto',
  verify_swap: 'auto',
  upright_remeasure: 'auto',
  process_priority: 'auto',
  output_image_format: 'png',
  output_video_format: 'mp4',
  output_video_codec: 'libx264',
  video_quality: 14,
  use_os_temp_folder: false,
  output_show_video: true,
  server_share: false,
  clear_output: false,
  server_name: '',
  server_port: 3000,
  output_template: '{file}_{time}',
  faceset_library_path: '',
  selected_theme: 'Default',
  custom_themes: [],
  theme_follow_system: false,
  theme_dark: 'Default',
  theme_light: 'Glass Light',
  selected_enhancer: 'UltraMax',
  swap_model: 'realswap',
  mask_engine: 'RealityUX',
  mask_engine_2: 'None',
  mask_clip_text: 'cup,hands,hair,banana',
  face_detection_mode: 'Selected face',
  blend_ratio: 0.85,
  face_mask_blend: 25.0,
  mouth_mask_blend: 10.0,
  stabilize_face: true,
  stabilize_enhancer: true,
  stabilize_enhancer_strength: 0.6,
  merger_sharpen: 0.55,
  color_transfer_mode: 'lct',
  codeformer_fidelity: 0.75,
  subsample_size: 256,
  upscale: '256px',
};

// Target and source selection
let selectedSourceIndex = 0;
let selectedTargetIndex = 0;
let currentFrame = 1;

// Progress / swap job state
interface ProgressState {
  processing: boolean;
  paused: boolean;
  pause_requested: boolean;
  stop_requested: boolean;
  progress: number;
  desc: string;
  error: string;
  current_frame: number;
  total_frames: number;
  fps: number;
  eta_seconds: number;
  output_filename: string;
  runtime: any;
}

function getHardwareTelemetry(isProcessing = false) {
  const isDesktop = activeHardwareProfile === 'desktop';
  return {
    sections: {
      HARDWARE: {
        values: {
          gpu: { value: isDesktop ? 'NVIDIA GeForce RTX 4070 (12GB)' : 'NVIDIA GeForce RTX 3060 Laptop (6GB)' },
          vram: { total_gb: isDesktop ? 12.0 : 6.0, free_gb: isDesktop ? 9.07 : 3.82 },
          ram: { total_gb: isDesktop ? 32.0 : 16.0, free_gb: isDesktop ? 22.4 : 10.1 },
          tdp: { value: isDesktop ? '142W / 200W' : '68W / 85W' },
          temperature: { value: '54°C' },
          pcie: { value: 'PCIe 4.0 x16 (Full Bandwidth)' },
        },
      },
      POOLING: {
        values: {
          workers: { configured: isDesktop ? 10 : 4, active: isProcessing ? (isDesktop ? 10 : 4) : 0 },
          trt_pool: { configured: isDesktop ? 2 : 0, active: isProcessing ? (isDesktop ? 2 : 0) : 0 },
          detmask_pool: { configured: isDesktop ? 2 : 0, active: isProcessing ? (isDesktop ? 2 : 0) : 0 },
          detector_pool: { configured: isDesktop ? 2 : 0, active: isProcessing ? (isDesktop ? 2 : 0) : 0 },
          hard_cap_mb: { value: isDesktop ? '4096 MB' : '1536 MB' },
        },
      },
      STABILIZATION: {
        values: {
          blend_ratio: { value: 0.85 },
          face_mask_blend: { value: 25 },
          merger_sharpen: { value: 0.55 },
          stabilize_enhancer_strength: { value: 0.6 },
          one_euro_filter: { value: 'Active (Adaptive Frequency)' },
        },
      },
    },
  };
}

let progressState: ProgressState = {
  processing: false,
  paused: false,
  pause_requested: false,
  stop_requested: false,
  progress: 0.0,
  desc: 'Ready',
  error: '',
  current_frame: 0,
  total_frames: 0,
  fps: 0,
  eta_seconds: 0,
  output_filename: '',
  runtime: getHardwareTelemetry(false),
};

// History & Output galleries
const historyRuns: any[] = [
  {
    id: 'run_sample_01',
    timestamp: Date.now() - 3600000,
    target: 'interview_scene_1080p.mp4',
    source: 'source_alpha.png',
    frames: 180,
    duration: '24.2s',
    avg_fps: 28.5,
    enhancer: 'UltraMax',
    model: 'realswap',
    output: '/api/file/output_interview_swapped.mp4',
  },
];

const outputFiles: any[] = [
  {
    name: 'interview_swapped_final.mp4',
    size: '18.4 MB',
    date: new Date(Date.now() - 3600000).toISOString(),
    url: '/api/file/interview_swapped_final.mp4',
    thumb: targetThumbs[0],
    frames: 180,
    resolution: '1920x1080',
  },
  {
    name: 'portrait_swapped_ultramax.png',
    size: '3.2 MB',
    date: new Date(Date.now() - 7200000).toISOString(),
    url: '/api/file/portrait_swapped_ultramax.png',
    thumb: targetThumbs[1],
    frames: 1,
    resolution: '1024x1024',
  },
];

// Active swap job timer
let swapInterval: NodeJS.Timeout | null = null;

function simulateSwapProcess() {
  if (swapInterval) clearInterval(swapInterval);
  const total = targetEntries[selectedTargetIndex]?.frames || 120;
  let frame = 0;
  progressState = {
    ...progressState,
    processing: true,
    paused: false,
    pause_requested: false,
    stop_requested: false,
    progress: 0.0,
    current_frame: 0,
    total_frames: total,
    fps: activeHardwareProfile === 'desktop' ? 31.4 : 18.2,
    eta_seconds: Math.round(total / (activeHardwareProfile === 'desktop' ? 31.4 : 18.2)),
    desc: `Swapping with ${appSettings.swap_model} + ${appSettings.selected_enhancer}...`,
    error: '',
    output_filename: '',
    runtime: getHardwareTelemetry(),
  };

  swapInterval = setInterval(() => {
    if (!progressState.processing) {
      if (swapInterval) clearInterval(swapInterval);
      return;
    }
    if (progressState.paused) return;

    frame += 4;
    if (frame >= total) {
      frame = total;
      progressState.processing = false;
      progressState.progress = 1.0;
      progressState.current_frame = total;
      progressState.desc = 'Swap finished successfully';
      const outName = `output_${Date.now().toString().slice(-4)}_${targetEntries[selectedTargetIndex]?.name || 'media.mp4'}`;
      progressState.output_filename = outName;
      outputFiles.unshift({
        name: outName,
        size: '14.2 MB',
        date: new Date().toISOString(),
        url: `/api/file/${outName}`,
        thumb: targetThumbs[selectedTargetIndex],
        frames: total,
        resolution: '1920x1080',
      });
      historyRuns.unshift({
        id: `run_${Date.now()}`,
        timestamp: Date.now(),
        target: targetEntries[selectedTargetIndex]?.name || 'media.mp4',
        source: sourceInfo[selectedSourceIndex]?.path || 'source.png',
        frames: total,
        duration: '12.6s',
        avg_fps: progressState.fps,
        enhancer: appSettings.selected_enhancer,
        model: appSettings.swap_model,
        output: `/api/file/${outName}`,
      });
      if (swapInterval) clearInterval(swapInterval);
    } else {
      progressState.current_frame = frame;
      progressState.progress = Number((frame / total).toFixed(3));
      const remainingFrames = total - frame;
      progressState.eta_seconds = Math.max(1, Math.round(remainingFrames / progressState.fps));
      progressState.desc = `Processing frame ${frame}/${total} (${progressState.fps} fps) · ${appSettings.selected_enhancer}`;
    }
  }, 100);
}

async function startServer() {
  const app = express();
  const PORT = Number(process.env.PORT) || 3000;

  // Every response says so, so a UI pointed at the wrong port cannot mistake
  // simulated numbers for a real render.
  app.use((req, res, next) => {
    res.setHeader('X-Mock-Server', 'true');
    next();
  });
  app.use(express.json({ limit: '50mb' }));
  app.use(express.urlencoded({ extended: true, limit: '50mb' }));

  // Create HTTP server & WebSocket for telemetry
  const server = http.createServer(app);
  const wss = new WebSocketServer({ server, path: '/ws/telemetry' });

  wss.on('connection', (ws: WebSocket) => {
    const timer = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        const isDesktop = activeHardwareProfile === 'desktop';
        ws.send(JSON.stringify({
          fps: progressState.processing ? progressState.fps : 0,
          current_frame: progressState.current_frame,
          total_frames: progressState.total_frames,
          progress: progressState.progress,
          vram_used_gb: isDesktop ? (progressState.processing ? 9.07 : 3.2) : (progressState.processing ? 4.8 : 2.1),
          vram_total_gb: isDesktop ? 12.0 : 6.0,
          gpu_util: progressState.processing ? 98 : 12,
          temp_c: 56,
          active_workers: progressState.processing ? (isDesktop ? 10 : 4) : 0,
          status: progressState.desc,
        }));
      }
    }, 500);

    ws.on('close', () => clearInterval(timer));
  });

  // ── Meta & Settings Endpoints ──────────────────────────────────────────
  app.get('/api/meta', (req, res) => {
    res.json({
      mock: true,
      git_version: 'v2.5.0-ultimate',
      installed_commit: { sha: 'a'.repeat(40), short: 'a'.repeat(12), date: '2026-09-22T00:00:00+00:00' },
      providers: ['cuda', 'tensorrt', 'cpu'],
      trt_precisions: ['fp32', 'fp16', 'mixed'],
      enhancers: [
        'None',
        'Adaptive',
        'Codeformer',
        'Codeformer (fp16)',
        'DMDNet',
        'GFPGAN',
        'GPEN 256',
        'GPEN 256 Pro',
        'GPEN Realistic',
        'GPEN',
        'GPEN 1024',
        'GPEN 2048',
        'GPEN Ultimate',
        'Restoreformer++',
        'Restore Ultra',
        'UltraMax',
        'KEEP (sidecar)',
      ],
      swap_models: [
        'realswap',
        'inswapper',
        'reswapper',
        'hyperswap',
        'hyperswap_1b',
        'hyperswap_1c',
        'ghost_1',
        'ghost_2',
        'ghost_3',
        'simswap',
        'simswap_512',
        'hififace',
        'blendswap',
        'uniface',
        'instyleswapper_a',
        'instyleswapper_b',
        'instyleswapper_c',
        'cscs',
      ],
      face_detection_modes: [
        'First found',
        'All input faces',
        'All female',
        'All male',
        'All faces',
        'Selected face',
      ],
      mask_engines: [
        'None',
        'Clip2Seg',
        'DFL XSeg',
        'Face Parser (BiSeNet)',
        'RealityUX',
        'Face Occluder',
        'Face Occluder v3 (XSeg-3)',
        'Segment Anything (MobileSAM)',
        'Segment Anything (FastSAM)',
        'Segment Anything 2 (tracked)',
      ],
      sam2_model_sizes: ['tiny', 'small', 'base_plus', 'large'],
      color_transfer_modes: ['none', 'rct', 'lct', 'mkl', 'idt'],
      detector_engines: ['scrfd', 'yoloface', 'retinaface', 'retinaface_r50', 'yunet'],
      encoder_presets: ['auto', 'ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'],
      pool_sizes: ['auto', '1', '2', '3', '4', '5', '6', '7', '8', '10', '12', '16'],
      tristate: ['auto', 'on', 'off'],
      recognizers: ['default', 'adaface'],
      priorities: ['auto', 'high', 'above_normal', 'normal'],
      no_face_actions: ['Use untouched original frame', 'Blank frame', 'Skip frame'],
      upscale: ['128px', '256px', '512px'],
      video_methods: ['Extract Frames to media', 'In-Memory processing'],
      output_methods: ['File', 'Virtual Camera', 'Both'],
      image_formats: ['jpg', 'png', 'webp'],
      video_formats: ['avi', 'mkv', 'mp4', 'webm'],
      video_codecs: ['libx264', 'libx265', 'libvpx-vp9', 'h264_nvenc', 'hevc_nvenc'],
    });
  });

  app.get('/api/settings', (req, res) => {
    res.json(appSettings);
  });

  app.get('/api/settings/defaults', (req, res) => {
    res.json(appSettings);
  });

  app.post('/api/settings', (req, res) => {
    appSettings = { ...appSettings, ...req.body };
    res.json(appSettings);
  });

  app.get('/api/progress', (req, res) => {
    progressState.runtime = getHardwareTelemetry();
    res.json(progressState);
  });

  // ── State Endpoints (Sources & Targets) ──────────────────────────────
  app.get('/api/state', (req, res) => {
    res.json({
      source_faces: sourceThumbs,
      source_faces_info: sourceInfo,
      target_faces: targetFacesList,
      target_groups: targetGroupsList,
      target_faces_info: targetFacesInfoList,
      target_names: targetNamesList,
      targets: targetEntries,
      selected_target_index: selectedTargetIndex,
      selected_source_index: selectedSourceIndex,
      faceset_count: 3,
    });
  });

  // ── Profiles Endpoints ──────────────────────────────────────────────
  app.get('/api/profiles', (req, res) => {
    res.json({ profiles: storedProfiles });
  });

  app.post('/api/profiles', (req, res) => {
    if (req.body.profile === 'laptop' || req.body.profile === 'desktop') {
      activeHardwareProfile = req.body.profile;
      if (activeHardwareProfile === 'desktop') {
        appSettings.max_threads = 10;
        appSettings.perf_trt_pool = 2;
        appSettings.perf_detmask_pool = 2;
        appSettings.perf_detector_pool = 2;
      } else {
        appSettings.max_threads = 4;
        appSettings.perf_trt_pool = 0;
        appSettings.perf_detmask_pool = 0;
        appSettings.perf_detector_pool = 0;
      }
      return res.json({ ok: true, active_profile: activeHardwareProfile });
    }
    const list = req.body.profiles;
    if (Array.isArray(list)) {
      storedProfiles = list;
    }
    res.json({ status: 'success', count: storedProfiles.length, profiles: storedProfiles });
  });

  // ── Source Face operations ──────────────────────────────────────────
  app.post('/api/source/select', (req, res) => {
    const idx = Number(req.body.index ?? 0);
    selectedSourceIndex = Math.max(0, Math.min(idx, sourceThumbs.length - 1));
    res.json({ ok: true, selected_source_index: selectedSourceIndex });
  });

  app.post('/api/source/add', upload.array('files'), (req, res) => {
    const files = req.files as Express.Multer.File[] || [];
    for (const f of files) {
      const idx = sourceThumbs.length;
      sourceThumbs.push(createSampleFaceSvg(f.originalname, 'Custom Upload · 512px', '#1e293b', '#334155', idx + 10));
      sourceInfo.push({
        index: idx,
        path: f.originalname,
        width: 512,
        height: 512,
        count: 1,
        poses: ['Frontal'],
        angles: { pitch: 0, yaw: 0, roll: 0 },
        quality: 0.96,
        landmarks: 106,
      });
    }
    res.json({
      source_faces: sourceThumbs,
      source_faces_info: sourceInfo,
      selected_source_index: selectedSourceIndex,
    });
  });

  app.post('/api/source/add-folder', upload.none(), (req, res) => {
    res.json({ source_faces: sourceThumbs, source_faces_info: sourceInfo });
  });

  app.post('/api/source/remove', (req, res) => {
    const idx = Number(req.body.index ?? 0);
    if (sourceThumbs.length > 1 && idx >= 0 && idx < sourceThumbs.length) {
      sourceThumbs.splice(idx, 1);
      sourceInfo.splice(idx, 1);
      sourceInfo.forEach((info, i) => { info.index = i; });
      // Removing an entry BEFORE the selection shifts the selection down by
      // one; removing the selected entry keeps the slot (now the next face),
      // clamped to the end; removing one after it changes nothing.
      if (idx < selectedSourceIndex) {
        selectedSourceIndex -= 1;
      } else if (idx === selectedSourceIndex) {
        selectedSourceIndex = Math.min(selectedSourceIndex, sourceThumbs.length - 1);
      }
    }
    res.json({ source_faces: sourceThumbs, source_faces_info: sourceInfo, selected_source_index: selectedSourceIndex });
  });

  app.post('/api/source/clear', (req, res) => {
    sourceThumbs.length = 0;
    sourceInfo.length = 0;
    selectedSourceIndex = 0;
    res.json({ source_faces: [], source_faces_info: [], selected_source_index: 0 });
  });

  // ── Target Operations ───────────────────────────────────────────────
  app.post('/api/target/select', (req, res) => {
    const idx = Number(req.body.index ?? 0);
    selectedTargetIndex = Math.max(0, Math.min(idx, targetEntries.length - 1));
    res.json({ targets: targetEntries, selected_target_index: selectedTargetIndex, fps: targetEntries[selectedTargetIndex]?.fps || 30 });
  });

  app.post('/api/target/add', upload.array('files'), (req, res) => {
    const files = req.files as Express.Multer.File[] || [];
    for (const f of files) {
      const isVid = /\.(mp4|avi|mkv|mov|webm)$/i.test(f.originalname);
      targetEntries.push({
        name: f.originalname,
        filename: f.originalname,
        preview_available: true,
        startframe: 0,
        endframe: isVid ? 240 : 1,
        start_frame: 0,
        end_frame: isVid ? 240 : 1,
        frames: isVid ? 240 : 1,
        fps: isVid ? 30 : 0,
        is_video: isVid,
      });
      targetThumbs.push(createSampleTargetSvg(f.originalname, 1, isVid ? 240 : 1, isVid, targetEntries.length + 100));
    }
    res.json({ targets: targetEntries, selected_target_index: selectedTargetIndex, fps: targetEntries[selectedTargetIndex]?.fps || 30 });
  });

  app.post('/api/target/add_path', (req, res) => {
    res.json({ targets: targetEntries, selected_target_index: selectedTargetIndex });
  });

  app.post('/api/target/remove', (req, res) => {
    const idx = Number(req.body.index ?? 0);
    if (targetEntries.length > 1 && idx >= 0 && idx < targetEntries.length) {
      targetEntries.splice(idx, 1);
      targetThumbs.splice(idx, 1);
      selectedTargetIndex = Math.max(0, selectedTargetIndex - 1);
    }
    res.json({ targets: targetEntries, selected_target_index: selectedTargetIndex, fps: targetEntries[selectedTargetIndex]?.fps || 30 });
  });

  app.post('/api/target/clear', (req, res) => {
    targetEntries.length = 0;
    targetThumbs.length = 0;
    selectedTargetIndex = 0;
    res.json({ targets: [], selected_target_index: 0, fps: 0 });
  });

  app.post('/api/target/set_frame', (req, res) => {
    currentFrame = Number(req.body.frame ?? currentFrame);
    res.json({ ok: true, frame: currentFrame });
  });

  app.post('/api/target/use_face', (req, res) => {
    const idx = targetFacesList.length;
    targetFacesList.push(createSampleFaceSvg(`Target Face ${idx + 1}`, `Frame ${currentFrame}`, '#1e293b', '#0f172a', idx + 80));
    targetGroupsList.push(0);
    targetFacesInfoList.push({
      index: idx,
      person: targetNamesList[0] || 'Person 1',
      landmarks: 106,
      gender: 'female',
      age: 26,
      bbox: [190, 60, 140, 160],
      pose: 'Front'
    });
    res.json(getTargetFacesPayload({ count: 1 }));
  });

  app.post('/api/target/remove_face', (req, res) => {
    const idx = Number(req.body.index ?? -1);
    if (idx >= 0 && idx < targetFacesList.length) {
      targetFacesList.splice(idx, 1);
      targetGroupsList.splice(idx, 1);
      targetFacesInfoList.splice(idx, 1);
    }
    res.json(getTargetFacesPayload());
  });

  app.post('/api/target/clear_faces', (req, res) => {
    targetFacesList.length = 0;
    targetGroupsList.length = 0;
    targetFacesInfoList.length = 0;
    targetNamesList.length = 0;
    res.json(getTargetFacesPayload({ count: 0 }));
  });

  app.post('/api/target/add_angle', (req, res) => {
    const person = Number(req.body.person ?? 0);
    const idx = targetFacesList.length;
    targetFacesList.push(createSampleFaceSvg(`Angle ${idx + 1}`, 'Profile Angle', '#334155', '#1e293b', idx + 60));
    targetGroupsList.push(person);
    targetFacesInfoList.push({
      index: idx,
      person: targetNamesList[person] || `Person ${person + 1}`,
      landmarks: 106,
      gender: 'female',
      age: 26,
      bbox: [180, 50, 150, 170],
      pose: 'Profile'
    });
    res.json(getTargetFacesPayload());
  });

  app.post('/api/target/auto_angles', (req, res) => {
    const person = Number(req.body.person ?? 0);
    ['Left Profile', 'Right Profile'].forEach((pose) => {
      const idx = targetFacesList.length;
      targetFacesList.push(createSampleFaceSvg(`${pose}`, 'Auto-harvested', '#1e293b', '#0f172a', idx + 70));
      targetGroupsList.push(person);
      targetFacesInfoList.push({
        index: idx,
        person: targetNamesList[person] || `Person ${person + 1}`,
        landmarks: 106,
        gender: 'female',
        age: 26,
        bbox: [185, 55, 145, 165],
        pose
      });
    });
    res.json(getTargetFacesPayload({ scanned: 180, seconds: 1.2, bins: 5, new_angles: 2 }));
  });

  app.post('/api/target/auto_capture', (req, res) => {
    res.json(getTargetFacesPayload({ count: targetFacesList.length }));
  });

  app.post('/api/target/autocluster', (req, res) => {
    res.json(getTargetFacesPayload({ clustered: targetFacesList.length }));
  });

  app.post('/api/target/name', (req, res) => {
    const person = Number(req.body.person ?? 0);
    const name = String(req.body.name ?? '').trim();
    if (person >= 0) {
      while (targetNamesList.length <= person) {
        targetNamesList.push('');
      }
      targetNamesList[person] = name;
    }
    res.json(getTargetFacesPayload());
  });

  app.post('/api/target/group', (req, res) => {
    const groups = req.body.groups;
    if (Array.isArray(groups)) {
      targetGroupsList = groups.slice(0, targetFacesList.length).map((g: any) =>
        typeof g === 'number' ? g : (Array.isArray(g) ? (g[0] ?? 0) : parseInt(g, 10) || 0)
      );
    }
    res.json(getTargetFacesPayload());
  });

  // ── Preview Grid endpoint (binary JPEG stream) ─────────────────────
  app.get('/api/target/preview_grid', (req, res) => {
    const framesParam = req.query.frames ? String(req.query.frames).split(',').map(Number).filter(n => !isNaN(n)) : [1];
    const totalFrames = Math.max(1, Math.min(framesParam.length, 64));

    // Valid minimal 1x1 JPEG:
    const tinyJpg = Buffer.from('/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA=', 'base64');

    const chunks: Buffer[] = [];
    for (let i = 0; i < totalFrames; i++) {
      const lenBuf = Buffer.alloc(4);
      lenBuf.writeUInt32BE(tinyJpg.length, 0);
      chunks.push(lenBuf);
      chunks.push(tinyJpg);
    }
    res.setHeader('Content-Type', 'application/octet-stream');
    res.send(Buffer.concat(chunks));
  });

  // ── Preview Generation ──────────────────────────────────────────────
  app.get('/api/target/preview', (req, res) => {
    const idx = Number(req.query.index ?? selectedTargetIndex);
    const frame = Number(req.query.frame ?? currentFrame);
    const target = targetEntries[idx] || targetEntries[0];
    const thumb = createSampleTargetSvg(target?.name || 'media', frame, target?.frames || 100, target?.is_video || false, idx + 200 + frame);
    res.setHeader('Content-Type', 'image/svg+xml');
    res.send(Buffer.from(thumb.split(',')[1], 'base64'));
  });

  app.post('/api/preview', (req, res) => {
    const idx = Number(req.body.index ?? selectedTargetIndex);
    const frame = Number(req.body.frame ?? currentFrame);
    const target = targetEntries[idx] || targetEntries[0];
    const image = createSampleTargetSvg(
      `SWAPPED: ${sourceInfo[selectedSourceIndex]?.path || 'source'} -> ${target?.name || 'target'}`,
      frame,
      target?.frames || 100,
      target?.is_video || false,
      idx + 500 + frame
    );

    res.json({
      image,
      faces: [[190, 60, 140, 160]],
      person_ids: ['Person 1 (Target Match)'],
      kps: [[[240, 130], [280, 130], [260, 150], [246, 170], [274, 170]]],
      pose: [{ pitch: -1.4, yaw: 8.2, roll: 0.4 }],
      enhancer: appSettings.selected_enhancer,
      swap_model: appSettings.swap_model,
    });
  });

  app.post('/api/preview_upscale', (req, res) => {
    const target = targetEntries[selectedTargetIndex] || targetEntries[0];
    const image = createSampleTargetSvg(
      `UPSCALED (${appSettings.selected_enhancer}): ${target?.name}`,
      currentFrame,
      target?.frames || 100,
      target?.is_video || false,
      999
    );
    res.json({ image });
  });

  // ── Swap Execution Endpoints ─────────────────────────────────────────
  app.post('/api/swap', (req, res) => {
    if (progressState.processing) {
      return res.status(409).json({ message: 'Already processing' });
    }
    simulateSwapProcess();
    res.json({ status: 'started', message: 'Swap process initiated successfully' });
  });

  app.post('/api/stop', (req, res) => {
    progressState.processing = false;
    progressState.stop_requested = true;
    progressState.desc = 'Swap stopped by user';
    if (swapInterval) clearInterval(swapInterval);
    res.json({ ok: true, message: 'Process stopped' });
  });

  app.post('/api/pause', (req, res) => {
    progressState.paused = true;
    progressState.pause_requested = true;
    progressState.desc = 'Swap paused';
    res.json({ ok: true, paused: true });
  });

  app.post('/api/resume', (req, res) => {
    progressState.paused = false;
    progressState.pause_requested = false;
    progressState.desc = 'Swap resumed';
    res.json({ ok: true, paused: false });
  });

  app.post('/api/reveal', (req, res) => {
    res.json({ ok: true, path: req.body.path || '/output' });
  });

  // ── Gallery & History ───────────────────────────────────────────────
  app.get('/api/output', (req, res) => {
    res.json({ outputs: outputFiles });
  });

  app.post('/api/output/delete', (req, res) => {
    const name = req.body.filename;
    const idx = outputFiles.findIndex((o) => o.name === name);
    if (idx >= 0) outputFiles.splice(idx, 1);
    res.json({ ok: true, outputs: outputFiles });
  });

  app.get('/api/history', (req, res) => {
    res.json({ history: historyRuns });
  });

  app.post('/api/history/delete', (req, res) => {
    const id = req.body.id;
    const idx = historyRuns.findIndex((h) => h.id === id);
    if (idx >= 0) historyRuns.splice(idx, 1);
    res.json({ ok: true, history: historyRuns });
  });

  // ── System & Hardware (RTX 4070 Desktop / RTX 3060 Laptop Profiles) ──
  app.get('/api/system/hardware', (req, res) => {
    const isDesktop = activeHardwareProfile === 'desktop';
    res.json({
      gpu: isDesktop ? 'NVIDIA GeForce RTX 4070' : 'NVIDIA GeForce RTX 3060 Laptop GPU',
      gpu_count: 1,
      vram_total_gb: isDesktop ? 12.0 : 6.0,
      vram_free_gb: isDesktop ? 9.07 : 3.82,
      cuda_version: '12.4',
      tensorrt_version: '10.2.0',
      cudnn_version: '9.1.0',
      cpu_cores: isDesktop ? 24 : 8,
      cpu_threads: isDesktop ? 32 : 16,
      ram_total_gb: isDesktop ? 32.0 : 16.0,
      ram_free_gb: isDesktop ? 22.4 : 10.1,
      active_profile: activeHardwareProfile,
      profiles: {
        desktop: {
          name: 'Main Device (Desktop Workstation)',
          gpu: 'RTX 4070 Desktop (12GB VRAM, 200W TDP)',
          workers: 10,
          trt_pool: 2,
          hard_cap_mb: 4096,
        },
        laptop: {
          name: 'Secondary Device (Laptop Workstation)',
          gpu: 'RTX 3060 Laptop GPU (6GB VRAM, Mobile TDP)',
          workers: 4,
          trt_pool: 0,
          hard_cap_mb: 1536,
        },
      },
    });
  });

  app.get('/api/system/profile', (req, res) => {
    res.json({ active_profile: activeHardwareProfile, telemetry: getHardwareTelemetry() });
  });

  app.post('/api/system/profile', (req, res) => {
    if (req.body.profile === 'laptop' || req.body.profile === 'desktop') {
      activeHardwareProfile = req.body.profile;
      if (activeHardwareProfile === 'desktop') {
        appSettings.max_threads = 10;
        appSettings.perf_trt_pool = 2;
        appSettings.perf_detmask_pool = 2;
        appSettings.perf_detector_pool = 2;
      } else {
        appSettings.max_threads = 4;
        appSettings.perf_trt_pool = 0;
        appSettings.perf_detmask_pool = 0;
        appSettings.perf_detector_pool = 0;
      }
    }
    res.json({ ok: true, active_profile: activeHardwareProfile });
  });

  app.get('/api/system/telemetry', (req, res) => {
    res.json(getHardwareTelemetry());
  });

  // ── Faceset & Face Manager ──────────────────────────────────────────
  app.get('/api/faceset/library', (req, res) => {
    res.json({
      libraries: [
        { name: 'Cinematic Actors', count: 18, modified: '2026-03-15' },
        { name: 'Studio Cast 2026', count: 24, modified: '2026-04-01' },
      ],
    });
  });

  app.post('/api/faceset/library/save', (req, res) => res.json({ ok: true }));
  app.post('/api/faceset/library/load', (req, res) => res.json({ ok: true }));
  app.post('/api/faceset/library/delete', (req, res) => res.json({ ok: true }));
  app.post('/api/faceset/library/rename', (req, res) => res.json({ ok: true }));
  app.post('/api/faceset/library/rebuild_thumbs', (req, res) => res.json({ ok: true }));
  app.post('/api/faceset/library/open', (req, res) => res.json({ ok: true }));

  app.post('/api/facemgr/add', upload.array('files'), (req, res) => res.json({ ok: true, added: 1 }));
  app.post('/api/facemgr/build', (req, res) => res.json({ ok: true, built: true }));
  app.post('/api/facemgr/clear', (req, res) => res.json({ ok: true }));
  app.post('/api/facemgr/cut', (req, res) => res.json({ ok: true }));
  app.post('/api/facemgr/prune', (req, res) => res.json({ ok: true }));
  app.post('/api/facemgr/remove', (req, res) => res.json({ ok: true }));

  // ── Extras & Quality ────────────────────────────────────────────────
  app.get('/api/extras/frame_ops', (req, res) => {
    res.json({
      ops: [
        { id: 'denoise', name: 'Temporal Denoise', description: 'Removes grain and sensor noise' },
        { id: 'deflicker', name: 'Face Deflicker', description: 'Stabilizes inter-frame luminance fluctuations' },
        { id: 'sharpen', name: 'Unsharp Mask', description: 'Edge micro-contrast boost' },
        { id: 'color_match', name: 'Reinhard Color Match', description: 'Matches tone to reference' },
      ],
    });
  });

  app.post('/api/quality/analyze', (req, res) => {
    res.json({
      score: 94.2,
      recommendation: 'Optimal configuration. Lighting angle matches source identity.',
      warnings: [],
    });
  });

  app.post('/api/advisor', (req, res) => {
    res.json({
      advice: 'RTX 4070 desktop detected with 12GB VRAM. TensorRT FP16 execution recommended with UltraMax enhancer.',
      suggested_settings: {
        provider: 'cuda',
        trt_precision: 'mixed',
        max_threads: 10,
        selected_enhancer: 'UltraMax',
      },
    });
  });

  app.post('/api/runtime_estimate', (req, res) => {
    res.json({ estimated_seconds: 14.5, fps: 28.5 });
  });

  // ── Presets & Queue ─────────────────────────────────────────────────
  app.get('/api/export/presets', (req, res) => {
    res.json({
      presets: [
        { id: 'web_1080p', name: 'Web Streaming (1080p MP4, CRF 18)', codec: 'libx264', crf: 18 },
        { id: 'cinema_4k', name: 'Cinematic Master (4K ProRes / HEVC)', codec: 'hevc_nvenc', crf: 12 },
        { id: 'lossless_png', name: 'Lossless Image Sequence (PNG)', codec: 'png', crf: 0 },
      ],
    });
  });

  app.post('/api/export/apply', (req, res) => res.json({ ok: true }));

  app.get('/api/queue', (req, res) => res.json({ queue: [], running: progressState.processing }));
  app.post('/api/queue/join', (req, res) => res.json({ ok: true }));

  app.get('/api/jobs/active', (req, res) => res.json({ jobs: [] }));
  app.get('/api/projects', (req, res) => res.json({ projects: [] }));
  app.post('/api/projects/:action', (req, res) => res.json({ ok: true }));
  app.get('/api/faceset/library', (req, res) => res.json({ items: [] }));
  app.post('/api/faceset/library/:action', (req, res) => res.json({ ok: true }));
  app.get('/api/update/check', (req, res) => res.json({
    classification: 'SAFE', available: false, reasons: ['no newer commit is available on the configured branch'],
    candidate_sha: 'a'.repeat(40), candidate_date: '2026-09-22T00:00:00+00:00', candidate_ref: null,
    candidate_manifest: { present: true, valid: true, problems: [] },
    current: { sha: 'a'.repeat(40), date: '2026-09-22T00:00:00+00:00', branch: 'main', version: 'main@mock' },
    apply_channel: 'pinokio', apply_gated: false, checked_at: Date.now() / 1000, cached: false,
  }));
  app.get('/api/storage', (req, res) => res.json({ total_gb: 512, used_gb: 128, temp_gb: 4.2 }));
  app.post('/api/storage/delete', (req, res) => res.json({ ok: true }));
  app.get('/api/runtime/state', (req, res) => res.json({ state: 'ready', initialized: true }));

  // ── Hardware Benchmark & Optimization Suite ─────────────────────────
  interface BenchmarkProgressState {
    running: boolean;
    cancelled: boolean;
    done: boolean;
    error: string | null;
    phase: string;
    status: string;
    frame: number;
    total_frames: number;
    progress_pct: number;
    current_fps: number;
    average_fps: number;
    elapsed_sec: number;
    eta_sec: number;
    vram_used_mb: number;
    vram_pct: number;
    gpu_pct: number;
    faces: string;
    mode: string;
  }

  let benchmarkState: BenchmarkProgressState = {
    running: false,
    cancelled: false,
    done: false,
    error: null,
    phase: 'idle',
    status: 'Ready to evaluate',
    frame: 0,
    total_frames: 90,
    progress_pct: 0,
    current_fps: 0,
    average_fps: 0,
    elapsed_sec: 0,
    eta_sec: 0,
    vram_used_mb: 3200,
    vram_pct: 26,
    gpu_pct: 0,
    faces: '1',
    mode: 'quick',
  };

  let benchmarkInterval: any = null;
  let lastBenchmarkReport: any = {
    ready: true,
    run_id: 'run_baseline_rtx4070_prev',
    timestamp: new Date(Date.now() - 3600000).toISOString(),
    score: 94,
    badge: 'Optimal Balanced',
    badge_tone: 'good',
    badge_detail: 'High throughput achieved with safe 3.1 GB unfragmented VRAM buffer.',
    bottleneck: 'GPU Compute Bound (Expected)',
    bottleneck_evidence: [
      'Engine saturation reached 96% with zero PCIe bandwidth throttling.',
      'Frame pacing stability within 4.2% jitter margin.',
      'Dual TRT execution pool fully occupied without context eviction.',
    ],
    average_fps: 24.8,
    p1_low_fps: 20.4,
    avg_latency_ms: 40.3,
    p99_latency_ms: 54.1,
    peak_vram_mb: 8940,
    applied: true,
    device: {
      gpu_name: 'NVIDIA GeForce RTX 4070 Desktop (12GB)',
      vram_total_mb: 12288,
      cpu_logical_cores: 32,
    },
    active_models: {
      swapper: 'inswapper_128',
      enhancer: 'None',
      mask_engine: 'box_tight',
    },
    thermal: {
      throttling_detected: false,
      retention_pct: 99.2,
    },
    presets: {
      balanced: {
        threads: 10,
        temp_format: 'png',
        recommended_settings: {
          max_threads: 10,
          perf_trt_pool: 2,
          perf_detmask_pool: 2,
          perf_detector_pool: 2,
          temp_frame_format: 'png',
        },
      },
      max_throughput: {
        threads: 14,
        temp_format: 'jpg',
        recommended_settings: {
          max_threads: 14,
          perf_trt_pool: 3,
          perf_detmask_pool: 2,
          perf_detector_pool: 2,
          temp_frame_format: 'jpg',
        },
      },
      stable_low_power: {
        threads: 6,
        temp_format: 'png',
        recommended_settings: {
          max_threads: 6,
          perf_trt_pool: 1,
          perf_detmask_pool: 1,
          perf_detector_pool: 1,
          temp_frame_format: 'png',
        },
      },
    },
    recommended_settings: {
      max_threads: 10,
      perf_trt_pool: 2,
      perf_detmask_pool: 2,
      perf_detector_pool: 2,
      temp_frame_format: 'png',
    },
    comparison: [
      { setting: 'Execution Threads', key: 'max_threads', current: 10, recommended: 10, changed: false, note: 'Matches 24 physical cores optimal dispatch', requires_restart: false },
      { setting: 'TensorRT Swap Pool', key: 'perf_trt_pool', current: 2, recommended: 2, changed: false, note: 'Parallel dual-instance inference without paging', requires_restart: true },
      { setting: 'Detection Mask Pool', key: 'perf_detmask_pool', current: 2, recommended: 2, changed: false, note: 'Prevents CPU-GPU mask synchronization stalls', requires_restart: true },
      { setting: 'Detector Model Pool', key: 'perf_detector_pool', current: 2, recommended: 2, changed: false, note: 'Smooth multi-face detection pipeline', requires_restart: true },
      { setting: 'Temp Frame Format', key: 'temp_frame_format', current: 'png', recommended: 'png', changed: false, note: 'Preserves lossless 10-bit source fidelity', requires_restart: false },
    ],
  };

  let savedBenchmarkProfiles: any[] = [
    {
      run_id: 'run_baseline_rtx4070_prev',
      timestamp: new Date(Date.now() - 3600000).toISOString(),
      score: 94,
      avg_fps: 24.8,
      p1_low_fps: 20.4,
      workload: '1 Face (Solo) · Quick',
      active_models: { swapper: 'inswapper_128', enhancer: 'None' },
      applied: true,
      recommended_settings: {
        max_threads: 10,
        perf_trt_pool: 2,
        perf_detmask_pool: 2,
        perf_detector_pool: 2,
        temp_frame_format: 'png',
      },
    },
    {
      run_id: 'run_laptop_rtx3060_mobile',
      timestamp: new Date(Date.now() - 86400000).toISOString(),
      score: 79,
      avg_fps: 15.2,
      p1_low_fps: 12.1,
      workload: '2 Faces (Duo) · Full',
      active_models: { swapper: 'inswapper_128', enhancer: 'GPEN-BFR-512' },
      applied: false,
      recommended_settings: {
        max_threads: 4,
        perf_trt_pool: 0,
        perf_detmask_pool: 0,
        perf_detector_pool: 0,
        temp_frame_format: 'png',
      },
    },
  ];

  app.get('/api/benchmark/prompt', (req, res) => {
    const isDesktop = activeHardwareProfile === 'desktop';
    res.json({
      can_run: !progressState.processing,
      gpu_name: isDesktop ? 'NVIDIA GeForce RTX 4070 Desktop (12GB)' : 'NVIDIA GeForce RTX 3060 Laptop GPU (6GB)',
      active_models: {
        swapper: appSettings.swap_model || 'inswapper_128',
        enhancer: appSettings.selected_enhancer || 'None',
        mask_engine: appSettings.face_demarcate || 'box_tight',
      },
      default_faces: '1',
      default_mode: 'quick',
      warnings: progressState.processing
        ? ['A render process is actively utilizing the GPU. Benchmarking now would measure a busy card and degrade rendering throughput.']
        : [],
    });
  });

  app.post('/api/benchmark/start', (req, res) => {
    if (progressState.processing) {
      return res.status(409).json({ message: 'A render is in progress — benchmarking now would measure a busy GPU.' });
    }
    if (benchmarkState.running) {
      return res.status(409).json({ message: 'A benchmark is already running.' });
    }

    const faces = String(req.body?.faces || '1');
    const mode = String(req.body?.mode || 'quick');
    const totalFrames = mode === 'full' ? 300 : 90;
    const isDesktop = activeHardwareProfile === 'desktop';
    const baseFps = isDesktop ? (faces === '1' ? 24.5 : faces === '2' ? 18.2 : 13.5) : (faces === '1' ? 15.2 : faces === '2' ? 10.4 : 7.1);

    benchmarkState = {
      running: true,
      cancelled: false,
      done: false,
      error: null,
      phase: 'prepare',
      status: 'Allocating benchmark tensors and warming up CUDA runtime...',
      frame: 0,
      total_frames: totalFrames,
      progress_pct: 0,
      current_fps: 0,
      average_fps: 0,
      elapsed_sec: 0,
      eta_sec: Math.ceil(totalFrames / baseFps),
      vram_used_mb: isDesktop ? 4500 : 2600,
      vram_pct: isDesktop ? 36 : 43,
      gpu_pct: 45,
      faces,
      mode,
    };

    let stepCount = 0;
    if (benchmarkInterval) clearInterval(benchmarkInterval);

    benchmarkInterval = setInterval(() => {
      if (!benchmarkState.running) {
        clearInterval(benchmarkInterval);
        return;
      }

      stepCount++;
      benchmarkState.elapsed_sec = stepCount * 0.5;

      if (stepCount < 3) {
        benchmarkState.phase = 'prepare';
        benchmarkState.status = 'Pre-allocating pinned memory buffers & profiling tensor graph...';
        benchmarkState.gpu_pct = 50 + stepCount * 10;
        return;
      }

      benchmarkState.phase = 'measuring';
      const frameDelta = Math.max(1, Math.round(baseFps * 0.5 + (Math.random() * 2 - 1)));
      benchmarkState.frame = Math.min(benchmarkState.total_frames, benchmarkState.frame + frameDelta);
      benchmarkState.progress_pct = (benchmarkState.frame / benchmarkState.total_frames) * 100;
      benchmarkState.current_fps = +(baseFps + (Math.random() * 1.8 - 0.9)).toFixed(1);
      benchmarkState.average_fps = +(baseFps * 0.98 + (Math.random() * 0.4)).toFixed(1);
      benchmarkState.gpu_pct = Math.min(99, Math.round(92 + Math.random() * 6));
      benchmarkState.vram_used_mb = isDesktop ? Math.round(8600 + Math.random() * 300) : Math.round(4100 + Math.random() * 200);
      benchmarkState.vram_pct = isDesktop ? Math.round((benchmarkState.vram_used_mb / 12288) * 100) : Math.round((benchmarkState.vram_used_mb / 6144) * 100);

      const remainingFrames = benchmarkState.total_frames - benchmarkState.frame;
      benchmarkState.eta_sec = Math.max(0, Math.ceil(remainingFrames / baseFps));
      benchmarkState.status = `Processing benchmark frame ${benchmarkState.frame}/${benchmarkState.total_frames} (${benchmarkState.current_fps} FPS)`;

      if (benchmarkState.frame >= benchmarkState.total_frames) {
        clearInterval(benchmarkInterval);
        benchmarkState.running = false;
        benchmarkState.done = true;
        benchmarkState.phase = 'complete';
        benchmarkState.status = 'Benchmark evaluation complete. Compiling dashboard report...';
        benchmarkState.progress_pct = 100;

        // Build report
        const finalAvgFps = +(baseFps * 1.01).toFixed(2);
        const finalP1Low = +(baseFps * 0.82).toFixed(2);
        const finalScore = isDesktop ? 94 : 81;
        const newRunId = `run_${Date.now()}`;

        lastBenchmarkReport = {
          ready: true,
          run_id: newRunId,
          timestamp: new Date().toISOString(),
          score: finalScore,
          badge: isDesktop ? 'Optimal Balanced' : 'Stable Mobile Tuned',
          badge_tone: isDesktop ? 'good' : 'neutral',
          badge_detail: isDesktop
            ? 'Optimal 100% on-device VRAM residency with 0 PCIe thrash.'
            : 'Balanced profile constrained to safe 1.5GB parallel stabilization buffer.',
          bottleneck: isDesktop ? 'GPU Bound (Optimal)' : 'VRAM Headroom Guard Active',
          bottleneck_evidence: isDesktop
            ? [
                'GPU core utilization sustained at 96% throughout all measurement passes.',
                '1% low frame pacing remains above 80% of mean execution throughput.',
                'TRT Engine memory footprint comfortably beneath 12.0 GB hard budget.',
              ]
            : [
                'GPU memory constrained to 6.0GB VRAM tier; TRT pool set to 0 for single-context safety.',
                'Adaptive block sizing engaged to prevent system paging memory spikes.',
              ],
          average_fps: finalAvgFps,
          p1_low_fps: finalP1Low,
          avg_latency_ms: +(1000 / finalAvgFps).toFixed(1),
          p99_latency_ms: +(1000 / finalP1Low * 1.08).toFixed(1),
          peak_vram_mb: isDesktop ? 8920 : 4380,
          applied: false,
          device: {
            gpu_name: isDesktop ? 'NVIDIA GeForce RTX 4070 Desktop (12GB)' : 'NVIDIA GeForce RTX 3060 Laptop GPU (6GB)',
            vram_total_mb: isDesktop ? 12288 : 6144,
            cpu_logical_cores: isDesktop ? 32 : 16,
          },
          active_models: {
            swapper: appSettings.swap_model || 'inswapper_128',
            enhancer: appSettings.selected_enhancer || 'None',
            mask_engine: appSettings.face_demarcate || 'box_tight',
          },
          thermal: {
            throttling_detected: false,
            retention_pct: isDesktop ? 99.4 : 96.8,
          },
          presets: {
            balanced: {
              threads: isDesktop ? 10 : 4,
              temp_format: 'png',
              recommended_settings: {
                max_threads: isDesktop ? 10 : 4,
                perf_trt_pool: isDesktop ? 2 : 0,
                perf_detmask_pool: isDesktop ? 2 : 0,
                perf_detector_pool: isDesktop ? 2 : 0,
                temp_frame_format: 'png',
              },
            },
            max_throughput: {
              threads: isDesktop ? 14 : 6,
              temp_format: 'jpg',
              recommended_settings: {
                max_threads: isDesktop ? 14 : 6,
                perf_trt_pool: isDesktop ? 3 : 1,
                perf_detmask_pool: isDesktop ? 2 : 1,
                perf_detector_pool: isDesktop ? 2 : 1,
                temp_frame_format: 'jpg',
              },
            },
            stable_low_power: {
              threads: isDesktop ? 6 : 2,
              temp_format: 'png',
              recommended_settings: {
                max_threads: isDesktop ? 6 : 2,
                perf_trt_pool: isDesktop ? 1 : 0,
                perf_detmask_pool: isDesktop ? 1 : 0,
                perf_detector_pool: isDesktop ? 1 : 0,
                temp_frame_format: 'png',
              },
            },
          },
          recommended_settings: {
            max_threads: isDesktop ? 10 : 4,
            perf_trt_pool: isDesktop ? 2 : 0,
            perf_detmask_pool: isDesktop ? 2 : 0,
            perf_detector_pool: isDesktop ? 2 : 0,
            temp_frame_format: 'png',
          },
          comparison: [
            {
              setting: 'Execution Threads',
              key: 'max_threads',
              current: appSettings.max_threads,
              recommended: isDesktop ? 10 : 4,
              changed: appSettings.max_threads !== (isDesktop ? 10 : 4),
              note: isDesktop ? 'Balances CPU extraction and GPU submission queues' : 'Caps host CPU dispatch to avoid laptop thermal throttling',
              requires_restart: false,
            },
            {
              setting: 'TensorRT Swap Pool',
              key: 'perf_trt_pool',
              current: appSettings.perf_trt_pool,
              recommended: isDesktop ? 2 : 0,
              changed: appSettings.perf_trt_pool !== (isDesktop ? 2 : 0),
              note: isDesktop ? 'Dual inference engines in VRAM' : 'Zero pool single-context for 6GB VRAM limit',
              requires_restart: true,
            },
            {
              setting: 'Detection Mask Pool',
              key: 'perf_detmask_pool',
              current: appSettings.perf_detmask_pool,
              recommended: isDesktop ? 2 : 0,
              changed: appSettings.perf_detmask_pool !== (isDesktop ? 2 : 0),
              note: 'Eliminates mask generation bottleneck',
              requires_restart: true,
            },
            {
              setting: 'Detector Pool',
              key: 'perf_detector_pool',
              current: appSettings.perf_detector_pool,
              recommended: isDesktop ? 2 : 0,
              changed: appSettings.perf_detector_pool !== (isDesktop ? 2 : 0),
              note: 'Concurrent face detector passes',
              requires_restart: true,
            },
            {
              setting: 'Temporary Frame Format',
              key: 'temp_frame_format',
              current: appSettings.temp_frame_format || 'png',
              recommended: 'png',
              changed: false,
              note: 'Lossless disk frame extraction',
              requires_restart: false,
            },
          ],
        };

        savedBenchmarkProfiles.unshift({
          run_id: newRunId,
          timestamp: lastBenchmarkReport.timestamp,
          score: finalScore,
          avg_fps: finalAvgFps,
          p1_low_fps: finalP1Low,
          workload: `${faces} Face${faces === '1' ? '' : 's'} · ${mode === 'full' ? 'Full' : 'Quick'}`,
          active_models: lastBenchmarkReport.active_models,
          applied: false,
          recommended_settings: lastBenchmarkReport.recommended_settings,
        });
      }
    }, 500);

    res.json({
      status: 'started',
      message: 'Hardware benchmark execution initiated',
      frames: totalFrames,
    });
  });

  app.get('/api/benchmark/progress', (req, res) => {
    res.json(benchmarkState);
  });

  app.post('/api/benchmark/cancel', (req, res) => {
    if (benchmarkInterval) clearInterval(benchmarkInterval);
    benchmarkState.running = false;
    benchmarkState.cancelled = true;
    benchmarkState.phase = 'cancelled';
    benchmarkState.status = 'Benchmark cancelled by user';
    res.json({ ok: true, status: 'cancelled' });
  });

  app.get('/api/benchmark/result', (req, res) => {
    if (!lastBenchmarkReport) {
      return res.json({ ready: false, running: benchmarkState.running, error: benchmarkState.error });
    }
    res.json({ ...lastBenchmarkReport, ready: true });
  });

  app.post('/api/benchmark/apply', (req, res) => {
    const recommended = req.body?.recommended_settings || lastBenchmarkReport?.recommended_settings;
    if (recommended && typeof recommended === 'object') {
      Object.assign(appSettings, recommended);
      if (lastBenchmarkReport) lastBenchmarkReport.applied = true;
      const targetRunId = req.body?.run_id || lastBenchmarkReport?.run_id;
      if (targetRunId) {
        const found = savedBenchmarkProfiles.find((p) => p.run_id === targetRunId);
        if (found) found.applied = true;
      }
      return res.json({ status: 'applied', message: 'Recommended settings applied successfully' });
    }
    res.status(400).json({ status: 'error', message: 'No recommended settings provided' });
  });

  app.post('/api/benchmark/decline', (req, res) => {
    if (lastBenchmarkReport) lastBenchmarkReport.applied = false;
    res.json({ status: 'declined', message: 'Benchmark recommendations declined' });
  });

  app.post('/api/benchmark/revert', (req, res) => {
    const isDesktop = activeHardwareProfile === 'desktop';
    appSettings.max_threads = isDesktop ? 10 : 4;
    appSettings.perf_trt_pool = isDesktop ? 2 : 0;
    appSettings.perf_detmask_pool = isDesktop ? 2 : 0;
    appSettings.perf_detector_pool = isDesktop ? 2 : 0;
    appSettings.temp_frame_format = 'png';
    if (lastBenchmarkReport) lastBenchmarkReport.applied = false;
    res.json({ status: 'reverted', message: 'Restored factory stock defaults for benchmark-owned settings' });
  });

  app.get('/api/benchmark/defaults', (req, res) => {
    res.json({
      owned_settings: ['max_threads', 'perf_trt_pool', 'perf_detmask_pool', 'perf_detector_pool', 'temp_frame_format'],
      defaults: {
        max_threads: 10,
        perf_trt_pool: 2,
        perf_detmask_pool: 2,
        perf_detector_pool: 2,
        temp_frame_format: 'png',
      },
    });
  });

  app.get('/api/benchmark/profiles', (req, res) => {
    res.json({ profiles: savedBenchmarkProfiles });
  });

  app.post('/api/benchmark/profiles/apply', (req, res) => {
    const runId = req.body?.run_id;
    let rec = req.body?.recommended_settings;
    if (!rec && runId) {
      const p = savedBenchmarkProfiles.find((x) => x.run_id === runId);
      rec = p?.recommended_settings;
    }
    if (rec && typeof rec === 'object') {
      Object.assign(appSettings, rec);
      savedBenchmarkProfiles.forEach((p) => {
        p.applied = p.run_id === runId;
      });
      return res.json({ status: 'applied', message: `Profile ${runId} applied successfully` });
    }
    res.status(404).json({ status: 'error', message: 'Profile not found' });
  });

  app.get('/api/settings/benchmark_threads', (req, res) => {
    res.json({
      threads: [2, 4, 6, 8, 10, 12, 16],
      fps: [8.2, 14.1, 18.9, 22.4, 25.1, 24.8, 23.5],
      optimal_threads: activeHardwareProfile === 'desktop' ? 10 : 4,
    });
  });

  // ── Livecam ─────────────────────────────────────────────────────────
  let livecamActive = false;
  app.get('/api/livecam/status', (req, res) => res.json({ active: livecamActive, fps: 30 }));
  app.post('/api/livecam/start', (req, res) => {
    livecamActive = true;
    res.json({ ok: true, active: true });
  });
  app.post('/api/livecam/stop', (req, res) => {
    livecamActive = false;
    res.json({ ok: true, active: false });
  });
  app.get('/api/livecam/frame', (req, res) => {
    const thumb = createSampleTargetSvg('Virtual LiveCam Stream', 1, 1, false, 888);
    res.setHeader('Content-Type', 'image/svg+xml');
    res.send(Buffer.from(thumb.split(',')[1], 'base64'));
  });

  // Mock static file responder for preview/downloads
  app.get('/api/file/:name', (req, res) => {
    const name = req.params.name;
    const thumb = createSampleTargetSvg(`File: ${name}`, 1, 100, true, 303);
    res.setHeader('Content-Type', 'image/svg+xml');
    res.send(Buffer.from(thumb.split(',')[1], 'base64'));
  });

  // Fallback for any unhandled /api/* endpoints so they never fall through to Vite SPA
  app.all('/api/*', (req, res) => {
    res.status(404).json({ error: 'Endpoint not found', path: req.path });
  });

  // ── Vite middleware (Dev) OR Static files (Prod) ────────────────────
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      configFile: path.join(UI_ROOT, 'vite.config.js'),
      root: UI_ROOT,
      server: { middlewareMode: true, proxy: {} },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(UI_ROOT, 'dist');
    app.use(express.static(distPath));
    app.get('*all', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  server.listen(PORT, '0.0.0.0', () => {
    console.log([
      '',
      '==========================================================',
      '  ROOP ULTIMATE  --  MOCK API SERVER  (not the real backend)',
      '  Swaps are simulated, telemetry is invented, previews are SVG.',
      '  The real backend is app/api.py (python run.py).',
      `  Listening on http://0.0.0.0:${PORT}  (PORT env var)`,
      '  Every response carries X-Mock-Server: true; /api/meta has mock: true.',
      '==========================================================',
      '',
    ].join('\n'));
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
