/**
 * Stage 21: Stereoscopic 3D & WebGL2 Equirectangular VR Viewport Render Checks.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  STEREO_INSPECTION_MODES,
  VR_PROJECTION_MODES,
  STEREO_LAYOUTS,
  DEFAULT_VR_PARAMS,
} from '../src/components/faceswap/StereoVRViewport.jsx';
import { GlRenderer } from '../src/components/player/glRenderer.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, '..');

let total = 0;
let passed = 0;

function ok(name, condition) {
  total += 1;
  if (!condition) {
    console.error(`  FAIL  ${name}`);
  } else {
    passed += 1;
    console.log(`  PASS  ${name}`);
  }
}

console.log('── Stage 21: Stereoscopic 3D & VR Viewport Checks ──────────');

// 1. Constants & Defaults
ok('DEFAULT_VR_PARAMS has vrMode="off"', DEFAULT_VR_PARAMS.vrMode === 'off');
ok('DEFAULT_VR_PARAMS has default fov=90', DEFAULT_VR_PARAMS.fov === 90);
ok('DEFAULT_VR_PARAMS has stereoMode="off"', DEFAULT_VR_PARAMS.stereoMode === 'off');
ok('DEFAULT_VR_PARAMS has stereoLayout="sbs"', DEFAULT_VR_PARAMS.stereoLayout === 'sbs');

// 2. Inspection Modes
const inspIds = STEREO_INSPECTION_MODES.map((m) => m.id);
ok('STEREO_INSPECTION_MODES includes left eye', inspIds.includes('left'));
ok('STEREO_INSPECTION_MODES includes right eye', inspIds.includes('right'));
ok('STEREO_INSPECTION_MODES includes side_by_side', inspIds.includes('side_by_side'));
ok('STEREO_INSPECTION_MODES includes anaglyph', inspIds.includes('anaglyph'));
ok('STEREO_INSPECTION_MODES includes disparity_diff', inspIds.includes('disparity_diff'));

// 3. VR Projection Modes
const vrIds = VR_PROJECTION_MODES.map((m) => m.id);
ok('VR_PROJECTION_MODES includes off (2D)', vrIds.includes('off'));
ok('VR_PROJECTION_MODES includes vr180', vrIds.includes('vr180'));
ok('VR_PROJECTION_MODES includes vr360', vrIds.includes('vr360'));

// 4. GlRenderer VR Integration
const mockCanvas = {
  getContext: () => null,
  width: 640,
  height: 480,
  addEventListener: () => {},
  removeEventListener: () => {},
};
const renderer = new GlRenderer(mockCanvas, { force2d: true });
ok('GlRenderer initializes vrParams', typeof renderer.vrParams === 'object' && renderer.vrParams.vrMode === 'off');
ok('GlRenderer has setVrParams method', typeof renderer.setVrParams === 'function');
renderer.setVrParams({ vrMode: 'vr180', yaw: 0.5, pitch: -0.2, fov: 75 });
ok('setVrParams updates vrMode', renderer.vrParams.vrMode === 'vr180');
ok('setVrParams updates yaw and pitch', renderer.vrParams.yaw === 0.5 && renderer.vrParams.pitch === -0.2);
ok('setVrParams updates fov', renderer.vrParams.fov === 75);

// 5. Codebase Wiring Checks
const ipCode = fs.readFileSync(path.join(ROOT, 'src/components/faceswap/InteractivePreview.jsx'), 'utf-8');
ok('InteractivePreview imports StereoVRViewport', ipCode.includes('import StereoVRViewport'));
ok('InteractivePreview has 3D / VR button', ipCode.includes('3D / VR'));
ok('InteractivePreview renders <StereoVRViewport', ipCode.includes('<StereoVRViewport'));

const extrasCode = fs.readFileSync(path.join(ROOT, 'src/components/Extras.jsx'), 'utf-8');
ok('Extras imports StereoVRViewport', extrasCode.includes('import StereoVRViewport'));
ok('Extras has 3D / VR button', extrasCode.includes('3D / VR'));
ok('Extras renders <StereoVRViewport', extrasCode.includes('<StereoVRViewport'));

const apiContractCode = fs.readFileSync(path.join(ROOT, 'src/types/api-contract.ts'), 'utf-8');
ok('api-contract.ts defines StereoMode', apiContractCode.includes('export type StereoMode'));
ok('api-contract.ts defines StereoInspectionMode', apiContractCode.includes('export type StereoInspectionMode'));
ok('api-contract.ts defines VrProjectionMode', apiContractCode.includes('export type VrProjectionMode'));

const mockServerCode = fs.readFileSync(path.join(ROOT, 'mock-server/server.ts'), 'utf-8');
ok('mock-server/server.ts has stereo_mode setting', mockServerCode.includes("stereo_mode: 'MONO_2D'"));

console.log(`\n${passed} of ${total} checks passed`);
globalThis.__RENDER_CHECK_FAILURES__ = total - passed;
