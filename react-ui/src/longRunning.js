// The endpoints that are ALLOWED to run past api.js's default 15 s deadline.
//
// api.js gives every request 15 s unless the call passes `timeout: 0`. That default
// exists because a backend that accepts the socket and then stalls (mid-GPU-stall, or
// killed between accept and response) otherwise leaves the promise pending forever, and
// a UI that waits forever never says anything is wrong.
//
// A request is LONG-RUNNING when how long it takes scales with its INPUT or can trigger
// a cold start, rather than with server latency: it decodes or scans media, runs model
// inference (the first call can build a TensorRT engine: minutes), encodes, deletes
// gigabytes, or carries an upload whose size the browser does not control. A deadline on
// those turns a slow success into a reported failure, so they say `timeout: 0` at the
// call site, in the open, where a reviewer sees it.
//
// THIS LIST IS THE SOURCE OF TRUTH. scripts/api-callsites.mjs (npm run lint:api-timeouts,
// and test_ui_api_timeouts.py in the Python suite) enumerates every getJSON / postJSON /
// postFile(s) call site and fails when
//   * an endpoint listed here is called without an explicit `timeout: 0`;
//   * a call passes `timeout: 0` for an endpoint that is NOT listed here (an unexplained
//     opt-out: add it here, with the reason, or drop the opt-out);
//   * a call has a computed path and says nothing about its timeout (use timeoutFor());
//   * an entry here matches no call site (a stale entry).
//
// Keys are `METHOD /path`; `{}` matches one path segment. Add the reason: the next person
// deciding whether their new endpoint belongs here reads it.
export const LONG_RUNNING = {
  // ── starting a render
  'POST /api/swap': 'validates and starts a render; waits on model / engine initialisation',

  // ── per-request inference (the first call can build a TensorRT engine: minutes)
  'POST /api/preview': 'swaps one frame; cold detector / swapper / enhancer load',
  'POST /api/preview_upscale': 'upscales one frame on the GPU',
  'POST /api/advisor': 'analyses the target clip (detection over sampled frames)',
  'POST /api/quality/analyze': 'detects and embeds faces across an output',
  'POST /api/extras/apply': 'media edit / re-encode of an uploaded file',
  'POST /api/extras/enhance': 'AI upscale / colorize of an uploaded file',

  // ── scans of the target clip
  'POST /api/target/add_path': 'opens and analyses a video named by path',
  'POST /api/target/add_angle': 'detects and embeds one captured frame',
  'POST /api/target/use_face': 'detects and embeds a captured face',
  'POST /api/target/auto_capture': 'scans the whole clip for people',
  'POST /api/target/auto_angles': 'scans the whole clip for each person\'s angles',
  'POST /api/target/autocluster': 'embeds and clusters every captured face',
  'POST /api/target/face_bank': 'embeds and clusters faces across the clip',

  // ── uploads (size is the user's, not ours; then the server analyses what arrived)
  'POST /api/source/add': 'upload + face detection',
  'POST /api/source/add-folder': 'upload of a folder + one clustered identity',
  'POST /api/target/add': 'upload + media analysis',
  'POST /api/lipsync/audio/add': 'upload of an audio track',
  'POST /api/facemgr/add': 'upload + face extraction (video: every frame)',
  'POST /api/facemgr/faceset': 'upload of a faceset archive',
  'POST /api/faceset/library/import': 'upload / copy of a faceset archive',

  // ── face manager and faceset operations that run detection or write archives
  'POST /api/facemgr/cut': 'detects faces on one decoded frame',
  'POST /api/faceset/library/save': 'detects, embeds and writes a faceset archive',
  'POST /api/faceset/library/load': 'reads an archive and re-detects its faces',
  'POST /api/faceset/library/rebuild_thumbs': 'regenerates thumbnails for the whole library',

  // ── angle capture
  'POST /api/angle-scan/override': 'detects and embeds a replacement frame',
  'POST /api/angle-scan/apply': 'adds the portfolio\'s frames to the person\'s angle bank',
  'POST /api/angle-scan/source-portfolio': 'fuses a source\'s faces into an angle portfolio',

  // ── encodes, devices and bulk deletes
  'POST /api/queue/join': 'ffmpeg concatenation of finished outputs',
  'POST /api/export/apply': 'ffmpeg re-encode to an export preset',
  'POST /api/livecam/start': 'opens the camera and loads models',
  'POST /api/trt_cache/clear': 'deletes gigabytes of compiled engines',

  // ── project verbs (the path is `/api/projects/{id}/{verb}`; `validate` is quick)
  'POST /api/projects/{}/load': 'reloads the project\'s media, sources and settings',
  'POST /api/projects/{}/resume': 'reloads a project and restarts its run',
};

const escape = (s) => s.replace(/[.*+?^$()|[\]\\]/g, '\\$&');
const PATTERNS = Object.keys(LONG_RUNNING).map((key) => {
  const [method, path] = key.split(' ');
  return { key, method, re: new RegExp(`^${escape(path).replace(/\{\}/g, '[^/]+')}$`) };
});

/** The registry key a request matches, or null. `path` may carry a query string. */
export function longRunningKey(method, path) {
  const bare = String(path).split('?')[0].replace(/\/+$/, '');
  const hit = PATTERNS.find((p) => p.method === method && p.re.test(bare));
  return hit ? hit.key : null;
}
