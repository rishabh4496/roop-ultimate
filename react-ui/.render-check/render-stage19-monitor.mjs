/**
 * Stage 19: BroadcastChannel Synchronization Bridge & Pop-Out Reference Monitor Tests
 *
 * Exercises:
 *   - createReferenceChannel with asynchronous message delivery
 *   - Zero-copy message broadcasting (FRAME, TRANSPORT, COMPARE, GRADING, PING/PONG, TEARDOWN)
 *   - Deterministic channel teardown on unmount
 *   - Static markup render of ReferenceMonitor
 *
 * Run with:
 *   node .render-check/run-module.mjs render-stage19-monitor.mjs
 */
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import {
  createReferenceChannel,
  closeReferenceChannel,
  broadcastFrame,
  broadcastTransport,
  broadcastCompare,
  broadcastGrading,
  MESSAGE_TYPES,
  CHANNEL_NAME,
} from '../src/components/monitor/referenceChannel.js';
import ReferenceMonitor from '../src/components/monitor/ReferenceMonitor.jsx';

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) {
    console.log(`  PASS  ${name}`);
    return;
  }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};

const delay = (ms = 25) => new Promise((resolve) => setTimeout(resolve, ms));

console.log('── Stage 19: BroadcastChannel synchronization bridge ────────────────');
{
  const receivedA = [];
  const receivedB = [];

  const chanA = createReferenceChannel((data) => receivedA.push(data));
  const chanB = createReferenceChannel((data) => receivedB.push(data));

  ok('channel name is roop_ultimate_reference_out', chanA.name === CHANNEL_NAME);

  // 1. Broadcast Transport
  broadcastTransport(chanA, {
    isPlaying: true,
    currentFrame: 42,
    totalFrames: 120,
    currentTime: 1.4,
    duration: 4.0,
    fps: 30,
  });
  await delay(20);

  ok('chanB receives transport packet', receivedB.length === 1);
  ok('transport packet type is TRANSPORT', receivedB[0]?.type === MESSAGE_TYPES.TRANSPORT);
  ok('transport payload preserves frame count and playing state',
    receivedB[0]?.payload?.currentFrame === 42 && receivedB[0]?.payload?.isPlaying === true);

  // 2. Broadcast Compare Coordinates
  broadcastCompare(chanA, {
    mode: 'slider',
    wipe: 65,
    compareDir: 'vertical',
    wipeAngle: 45,
    diffGain: 3.5,
  });
  await delay(20);

  ok('chanB receives compare packet', receivedB.length === 2);
  ok('compare payload matches wipe coordinates',
    receivedB[1]?.type === MESSAGE_TYPES.COMPARE && receivedB[1]?.payload?.wipe === 65 && receivedB[1]?.payload?.wipeAngle === 45);

  // 3. Broadcast Grading & Zebra
  broadcastGrading(chanA, {
    lut_path: 'profiles/cinema.cube',
    lut_opacity: 0.85,
    zebraActive: true,
    color_temperature: 15,
  });
  await delay(20);

  ok('chanB receives grading packet', receivedB.length === 3);
  ok('grading payload carries lut and zebra flag',
    receivedB[2]?.type === MESSAGE_TYPES.GRADING &&
    receivedB[2]?.payload?.zebraActive === true &&
    receivedB[2]?.payload?.lut_path === 'profiles/cinema.cube');

  // 4. Broadcast Frame
  const mockFrame = { width: 1920, height: 1080, id: 'frame_bitmap_mock' };
  broadcastFrame(chanA, mockFrame, {
    width: 1920,
    height: 1080,
    frameIndex: 42,
    timestamp: 1000,
  });
  await delay(20);

  ok('chanB receives frame packet', receivedB.length === 4);
  ok('frame payload preserves dimensions and index',
    receivedB[3]?.type === MESSAGE_TYPES.FRAME &&
    receivedB[3]?.payload?.width === 1920 &&
    receivedB[3]?.payload?.frameIndex === 42);

  // 5. Deterministic Teardown
  closeReferenceChannel(chanA);
  await delay(20);
  ok('chanB receives teardown on channel close',
    receivedB.some((msg) => msg.type === MESSAGE_TYPES.TEARDOWN));

  closeReferenceChannel(chanB);
}

console.log('── Stage 19: ReferenceMonitor component static render ───────────────');
{
  const html = renderToStaticMarkup(React.createElement(ReferenceMonitor));
  ok('renders clean-feed monitor canvas', html.includes('<canvas'));
  ok('renders reference monitor HUD bar', html.includes('CONNECTING...') || html.includes('LIVE CLEAN FEED'));
  ok('has chrome-free popup close and fullscreen controls', html.includes('Close') && html.includes('Fullscreen'));
}

console.log('── Stage 19: Summary ────────────────────────────────────────────────');
globalThis.__RENDER_CHECK_FAILURES__ = failures;
console.log(failures === 0 ? 'stage19-monitor: all checks passed' : `stage19-monitor: ${failures} check(s) FAILED`);
