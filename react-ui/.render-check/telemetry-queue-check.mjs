/**
 * Telemetry HUD & Render Queue Verification Suite
 * Tests rolling 100-frame ETA moving average, workspace layout state transitions,
 * panel dock toggles, and hardware telemetry metric bounds.
 */
import process from 'node:process';
import { useWorkspaceLayoutStore } from '../src/components/studio/workspaceLayoutStore.js';
import { RollingEtaTracker, summarizeQueueOutcome } from '../src/components/queue/queueMetrics.js';
import {
  frameTimeMs, vramUsage, thermalLevel, isPowerLimited, sparkNorm,
} from '../src/components/telemetry/hudMetrics.js';

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

async function runTests() {
  console.log('── Rolling 100-Frame ETA Moving Average Engine ───────────');
  {
    const tracker = new RollingEtaTracker(100);

    // Initial state
    ok('Initial ETA is null when no samples', tracker.getEtaSeconds(1000) === null);

    // Add samples at steady 40 FPS: 25ms per frame
    // Feed 150 frames: from t=0 to t=3750ms
    for (let f = 1; f <= 150; f++) {
      tracker.addSample(f, f * 25);
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

    // Next job: the frame counter restarts. The window must restart too.
    tracker.addSample(5, 10000);
    ok('Counter going backwards resets the window (no negative rate)',
      tracker.samples.length === 1 && tracker.getFps() === null);
    tracker.addSample(25, 11000);
    ok('Rate after reset uses only the new job', Math.abs(tracker.getFps() - 20) < 1e-9);
    ok('ETA is 0 at the last frame', tracker.getEtaSeconds(25) === 0);
    ok('ETA is null without a total', tracker.getEtaSeconds(0) === null);
  }

  console.log('── Queue completion outcome (real summarizeQueueOutcome) ─');
  {
    const good = summarizeQueueOutcome(['COMPLETED', 'COMPLETED']);
    ok('All COMPLETED is a success', good.allSucceeded && good.title === 'Render Queue Complete');
    const bad = summarizeQueueOutcome(['FAILED', 'FAILED', 'FAILED']);
    ok('All FAILED is NOT a success', !bad.allSucceeded && bad.title === 'Render Queue Failed');
    ok('All FAILED body reports the failures', bad.body === '0 of 3 completed, 3 failed.', bad.body);
    const mixed = summarizeQueueOutcome(['COMPLETED', 'CANCELLED', 'INTERRUPTED', 'FAILED']);
    ok('Mixed queue is NOT a success', !mixed.allSucceeded);
    ok('Mixed body counts every outcome',
      mixed.body === '1 of 4 completed, 1 failed, 1 cancelled, 1 interrupted.', mixed.body);
    ok('Empty queue is not a success', !summarizeQueueOutcome([]).allSucceeded);
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

  console.log('── HUD metrics (real hudMetrics used by HardwareTelemetryHud) ─');
  {
    ok('75°C is no alert', thermalLevel(75) === null);
    ok('81°C is a warning', thermalLevel(81) === 'warn');
    ok('88°C is critical', thermalLevel(88) === 'critical');
    ok('Missing temperature is no alert', thermalLevel(undefined) === null);

    // Power: relative to the board's own limit, never a fixed wattage.
    ok('4070: 150 of 200W is not limited', !isPowerLimited(150, 200));
    ok('4070: 195 of 200W is limited', isPowerLimited(195, 200));
    ok('3060 Laptop: 92 of 95W is limited (a fixed 190W could never fire)', isPowerLimited(92, 95));
    ok('3060 Laptop: 60 of 95W is not limited', !isPowerLimited(60, 95));
    ok('No reported limit -> no alert, even at 250W', !isPowerLimited(250, undefined));

    // VRAM: unknown total is unknown, not a share of a 12 GB card.
    ok('VRAM with no total is null', vramUsage({ vram_used: 3 }) === null);
    ok('VRAM with total 0 (CPU only) is null', vramUsage({ vram_used: 0, vram_total: 0 }) === null);
    const v = vramUsage({ vram_used: 3, vram_total: 6 });
    ok('6 GB card: 3 GB used is 50%', v && v.pct === 50 && v.total === 6);
    ok('VRAM pct clamps at 100', vramUsage({ vram_used: 9, vram_total: 6 }).pct === 100);

    // Frame time: the backend's number or nothing; no invented default.
    ok('frame_ms passes through', frameTimeMs({ frame_ms: 96.2 }) === 96.2);
    ok('No frame_ms -> null (not 25ms)', frameTimeMs({ fps: 10 }) === null);
    ok('frame_ms null -> null', frameTimeMs({ frame_ms: null }) === null);

    ok('Sparkline normalizes midpoint 30 in [0, 60] to 0.5', sparkNorm(30, 0, 60) === 0.5);
    ok('Sparkline clamps negative value to 0.0', sparkNorm(-10, 0, 60) === 0.0);
    ok('Sparkline clamps overflow value to 1.0', sparkNorm(80, 0, 60) === 1.0);
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
