/**
 * Unit verification for Pro Keyboard Shortcuts & Shuttle Ergonomics.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

globalThis.window = globalThis;
globalThis.location = { origin: 'http://127.0.0.1:1', protocol: 'http:', host: '127.0.0.1:1' };

const {
  IDLE,
  SHUTTLE_SPEEDS,
  shuttleKey,
  shuttleRate,
  shuttleLabel,
  reverseTick,
  stepSize,
} = await import('../src/components/faceswap/timeline/shuttle.js');

const {
  isInputFocused,
  isModalOpen,
  isInteractiveButtonFocused,
  getMarkersStorageKey,
} = await import('../src/components/faceswap/useWorkstationShortcuts.ts');

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) {
    console.log(`  PASS  ${name}`);
    return;
  }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};

console.log('── J / K / L Shuttle Ergonomics & Speed Ladder ───────────────────');

{
  // 1. Initial idle state
  ok('initial state is idle dir 0 level 0', IDLE.dir === 0 && IDLE.level === 0);
  ok('shuttleRate for idle is 0', shuttleRate(IDLE) === 0);
  ok('shuttleLabel for idle is empty', shuttleLabel(IDLE) === '');

  // 2. Forward shuttle (L): 1x -> 2x -> 4x -> 4x (capped)
  let s = shuttleKey(IDLE, 'l');
  ok('first L press moves forward at 1x', s.dir === 1 && s.level === 0 && shuttleRate(s) === 1);
  ok('forward 1x label displays ▶ 1×', shuttleLabel(s).includes('▶') && shuttleLabel(s).includes('1×'));

  s = shuttleKey(s, 'l');
  ok('second L press advances to 2x', s.dir === 1 && s.level === 1 && shuttleRate(s) === 2);
  ok('forward 2x label displays ▶ 2×', shuttleLabel(s).includes('▶') && shuttleLabel(s).includes('2×'));

  s = shuttleKey(s, 'l');
  ok('third L press advances to 4x', s.dir === 1 && s.level === 2 && shuttleRate(s) === 4);
  ok('forward 4x label displays ▶ 4×', shuttleLabel(s).includes('▶') && shuttleLabel(s).includes('4×'));

  s = shuttleKey(s, 'l');
  ok('fourth L press stays capped at 4x', s.dir === 1 && s.level === 2 && shuttleRate(s) === 4);

  // 3. Pause (K): returns to IDLE immediately
  s = shuttleKey(s, 'k');
  ok('K press halts shuttle and returns to IDLE', s.dir === 0 && s.level === 0);

  // 4. Reverse shuttle (J): -1x -> -2x -> -4x
  s = shuttleKey(IDLE, 'j');
  ok('first J press moves reverse at -1x', s.dir === -1 && s.level === 0 && shuttleRate(s) === 1);
  ok('reverse 1x label displays ◀ 1×', shuttleLabel(s).includes('◀') && shuttleLabel(s).includes('1×'));

  s = shuttleKey(s, 'j');
  ok('second J press advances to -2x', s.dir === -1 && s.level === 1 && shuttleRate(s) === 2);

  s = shuttleKey(s, 'j');
  ok('third J press advances to -4x', s.dir === -1 && s.level === 2 && shuttleRate(s) === 4);

  // 5. Direction change: L while reversing immediately switches to forward 1x
  s = shuttleKey(s, 'l');
  ok('L while reversing immediately switches to forward 1x', s.dir === 1 && s.level === 0 && shuttleRate(s) === 1);

  // J while going forward immediately switches to reverse -1x
  s = shuttleKey(s, 'j');
  ok('J while going forward immediately switches to reverse -1x', s.dir === -1 && s.level === 0 && shuttleRate(s) === 1);
}

console.log('── Sub-frame Reverse rAF Clock Math ────────────────────────────');

{
  // 30 fps, 100ms dt at 1x = 3 frames
  const tick1 = reverseTick({ frame: 50, carry: 0 }, 100, 1, 30, 1);
  ok('100ms at 30fps moves exactly 3 frames backward', tick1.frame === 47 && !tick1.atStart);

  // Fractional carry accumulation: 16.666ms at 25 fps = 0.4166 frames
  const tick2 = reverseTick({ frame: 50, carry: 0 }, 16.666, 1, 25, 1);
  ok('fractional dt preserves carry without rounding jitter', tick2.frame === 50 && tick2.carry > 0.4);

  const tick3 = reverseTick({ frame: tick2.frame, carry: tick2.carry }, 30, 1, 25, 1);
  ok('accumulated carry steps frame when crossing integer boundary', tick3.frame === 49);

  // Reaching clip floor (frame 1)
  const floorTick = reverseTick({ frame: 2, carry: 0 }, 1000, 4, 30, 1);
  ok('reverse tick clamps to floor 1 and signals atStart', floorTick.frame === 1 && floorTick.atStart === true);
}

console.log('── Stepping Ergonomics ─────────────────────────────────────────');

{
  ok('unmodified stepSize is 1 frame', stepSize(false) === 1);
  ok('Shift stepSize is 10 frames', stepSize(true) === 10);
}

console.log('── Input Guard & Modal Isolation Helpers ───────────────────────');

{
  // Mock document environment
  globalThis.document = {
    activeElement: null,
    querySelector: () => null,
  };

  ok('isInputFocused false when activeElement is null', isInputFocused() === false);
  ok('isModalOpen false when no role=dialog', isModalOpen() === false);

  globalThis.document.activeElement = { tagName: 'INPUT' };
  ok('isInputFocused true for INPUT', isInputFocused() === true);

  globalThis.document.activeElement = { tagName: 'TEXTAREA' };
  ok('isInputFocused true for TEXTAREA', isInputFocused() === true);

  globalThis.document.activeElement = { tagName: 'SELECT' };
  ok('isInputFocused true for SELECT', isInputFocused() === true);

  globalThis.document.activeElement = { tagName: 'DIV', isContentEditable: true };
  ok('isInputFocused true for contentEditable DIV', isInputFocused() === true);

  globalThis.document.activeElement = {
    tagName: 'BUTTON',
    closest: (sel) => sel.includes('button') ? {} : null,
  };
  ok('isInputFocused false for regular button', isInputFocused() === false);
  ok('isInteractiveButtonFocused true for button', isInteractiveButtonFocused(globalThis.document.activeElement) === true);

  globalThis.document.querySelector = (sel) => sel === '[role="dialog"]' ? {} : null;
  ok('isModalOpen true when dialog element is present', isModalOpen() === true);
}

console.log('── Chapter Marker Storage Key ──────────────────────────────────');

{
  ok('markers key with targetKey', getMarkersStorageKey('clip_abc') === 'roop_markers_clip_abc');
  ok('markers key without targetKey uses default', getMarkersStorageKey('') === 'roop_markers_default');
}

console.log('── 100% Offline Compliance & Zero Remote URLs ──────────────────');

{
  const targetFiles = [
    'src/components/faceswap/useWorkstationShortcuts.ts',
    'src/components/faceswap/NleTimeline.tsx',
    'src/components/faceswap/PreviewViewport.tsx',
    'src/components/faceswap/runBarSync.ts',
  ];

  for (const rel of targetFiles) {
    const fullPath = path.resolve(__dirname, '..', rel);
    if (!fs.existsSync(fullPath)) continue;
    const content = fs.readFileSync(fullPath, 'utf8');

    // Remote URL regex (excluding comments mentioning docs / specs)
    const urlMatches = content.match(/https?:\/\/(?!localhost|127\.0\.0\.1)[^\s'")`]+/gi) || [];
    const remoteUrls = urlMatches.filter((u) => !u.includes('w3.org') && !u.includes('schema.org'));
    ok(`offline compliance for ${rel}: 0 remote URLs`, remoteUrls.length === 0, `Found: ${remoteUrls.join(', ')}`);
  }
}

globalThis.__RENDER_CHECK_FAILURES__ = failures;
if (failures > 0) {
  console.error(`\n${failures} check(s) failed`);
  process.exit(1);
} else {
  console.log('\nAll Pro Shortcuts & Workstation Ergonomics checks PASSED');
}
