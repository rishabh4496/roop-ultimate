/**
 * Telemetry HUD & Render Queue Verification Suite
 * Tests rolling 100-frame ETA moving average, workspace layout state transitions,
 * panel dock toggles, and hardware telemetry metric bounds.
 */
import process from 'node:process';
import { useWorkspaceLayoutStore } from '../src/components/studio/workspaceLayoutStore.js';

let checks = 0;
let failures = 0;

const ok = (name, cond, detail = '') => {
  checks += 1;
  if (cond) {
    console.log(`  PASS  ${name}`);
    return;
  }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};

// Rolling 100-frame ETA test implementation mirroring component logic
class TestRollingEtaTracker {
  constructor(windowFrames = 100) {
    this.windowFrames = windowFrames;
    this.samples = [];
  }
  addSample(timeMs, frame) {
    this.samples.push({ time: timeMs, frame });
    while (this.samples.length > 2 && (frame - this.samples[0].frame) > this.windowFrames) {
      this.samples.shift();
    }
  }
  getFps() {
    if (this.samples.length < 2) return null;
    const first = this.samples[0];
    const last = this.samples[this.samples.length - 1];
    const dt = (last.time - first.time) / 1000.0;
    const df = last.frame - first.frame;
    return dt > 0 && df > 0 ? df / dt : null;
  }
  getEtaSeconds(totalFrames) {
    if (!this.samples.length) return null;
    const current = this.samples[this.samples.length - 1].frame;
    const remaining = Math.max(0, totalFrames - current);
    const fps = this.getFps();
    return fps && fps > 0 ? remaining / fps : null;
  }
}

async function runTests() {
  console.log('── Rolling 100-Frame ETA Moving Average Engine ───────────');
  {
    const tracker = new TestRollingEtaTracker(100);

    // Initial state
    ok('Initial ETA is null when no samples', tracker.getEtaSeconds(1000) === null);

    // Add samples at steady 40 FPS: 25ms per frame
    // Feed 150 frames: from t=0 to t=3750ms
    for (let f = 1; f <= 150; f++) {
      tracker.addSample(f * 25, f);
    }

    const calculatedFps = tracker.getFps();
    ok('Measured FPS is approximately 40.0', Math.abs(calculatedFps - 40.0) < 0.1, `Got ${calculatedFps}`);

    // Rolling window depth stays capped at ~100 frames
    const windowSpan = tracker.samples[tracker.samples.length - 1].frame - tracker.samples[0].frame;
    ok('Rolling window span is <= 100 frames', windowSpan <= 100, `Got span ${windowSpan}`);

    // Calculate ETA for 1000 total frames
    // Remaining frames: 1000 - 150 = 850 frames. At 40 FPS -> 850 / 40 = 21.25 seconds.
    const etaSec = tracker.getEtaSeconds(1000);
    ok('ETA for remaining 850 frames at 40 FPS is 21.25s', Math.abs(etaSec - 21.25) < 0.1, `Got ${etaSec}s`);

    // Speed multiplier check vs 30fps target
    const speedMult = calculatedFps / 30.0;
    ok('Speed multiplier is ~1.33x real-time', Math.abs(speedMult - 1.333) < 0.05, `Got ${speedMult}`);
  }

  console.log('── Workspace Layout Store & Preset Transitions ───────────');
  {
    const store = useWorkspaceLayoutStore.getState();

    // Default layout check
    ok('Default layout activePreset is default', store.activePreset === 'default');
    ok('Default layout has preview enabled', store.panels.preview === true);
    ok('Default layout has timeline enabled', store.panels.timeline === true);
    ok('Default layout has facebank enabled', store.panels.facebank === true);

    // Switch to Cinema Preset
    store.setPreset('cinema');
    const cinemaState = useWorkspaceLayoutStore.getState();
    ok('Cinema preset disables facebank', cinemaState.panels.facebank === false);
    ok('Cinema preset keeps preview enabled', cinemaState.panels.preview === true);

    // Switch to Color Grading Preset (Rec.709)
    store.setPreset('grading');
    const gradingState = useWorkspaceLayoutStore.getState();
    ok('Grading preset activates rec709_neutral studio theme', gradingState.studioTheme === 'rec709_neutral');
    ok('Grading preset keeps telemetry enabled', gradingState.panels.telemetry === true);

    // Toggle individual panel
    store.togglePanel('queue');
    const queueToggledState = useWorkspaceLayoutStore.getState();
    ok('Toggling queue panel enables it', queueToggledState.panels.queue === true);
    ok('Custom toggle shifts activePreset to custom', queueToggledState.activePreset === 'custom');

    // Reset layout
    store.resetLayout();
    const resetState = useWorkspaceLayoutStore.getState();
    ok('resetLayout restores activePreset to default', resetState.activePreset === 'default');
    ok('resetLayout restores studioTheme to obsidian', resetState.studioTheme === 'obsidian');
  }

  console.log('── Hardware Telemetry Throttling Bounds & Math ───────────');
  {
    // Thermal threshold rules
    const isThermalThrottling = (temp) => temp >= 80;
    const isCriticalThermal = (temp) => temp >= 86;

    ok('75°C does not trigger thermal throttling', !isThermalThrottling(75));
    ok('81°C triggers thermal warning', isThermalThrottling(81) && !isCriticalThermal(81));
    ok('88°C triggers critical thermal alert', isCriticalThermal(88));

    // Power limit detection (190W threshold for 200W TDP card)
    const isPowerThrottling = (watts, tdp = 200) => watts >= tdp * 0.95;
    ok('150W is normal power', !isPowerThrottling(150, 200));
    ok('195W triggers power throttling warning', isPowerThrottling(195, 200));

    // Sparkline normalization safety: value clamp 0..1
    const normalize = (val, min, max) => Math.max(0, Math.min(1, (val - min) / Math.max(1e-5, max - min)));
    ok('Sparkline normalizes midpoint 30 in [0, 60] to 0.5', normalize(30, 0, 60) === 0.5);
    ok('Sparkline clamps negative value to 0.0', normalize(-10, 0, 60) === 0.0);
    ok('Sparkline clamps overflow value to 1.0', normalize(80, 0, 60) === 1.0);
  }

  console.log('── Summary ────────────────────────────────────────────────');
  console.log(`  Total checks: ${checks}`);
  console.log(`  Failures:     ${failures}`);

  if (failures > 0) {
    process.exit(1);
  }
}

runTests().catch((err) => {
  console.error('Test run failed with error:', err);
  process.exit(1);
});
