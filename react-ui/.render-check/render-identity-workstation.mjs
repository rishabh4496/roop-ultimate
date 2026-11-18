/**
 * Identity Workstation's pure logic (Face Manager, Stage 6): pose coverage, selection, marquee, cluster edits,
 * quality filtering, library filtering.
 *
 *   node .render-check/run-module.mjs render-identity-workstation.mjs
 */
import {
  applyClick, clampThumb, computeCoverage, facePose, filterLibrary, fmtBytes, libraryTags, marqueeSelect, mergeGroups,
  outliersBelow, poseFromLabel, qualityFromWidth, ranksOf, rectFromPoints, resolveHero, splitGroup, zoneOf,
} from '../src/components/facemgr/identityModel.js';

let failures = 0;
const ok = (name, cond, detail = '') => {
  if (cond) { console.log(`  PASS  ${name}`); return; }
  failures += 1;
  console.log(`  FAIL  ${name}${detail ? `\n          ${detail}` : ''}`);
};
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const pts = (list) => list.map(([yaw, pitch], i) => ({ key: i, yaw, pitch }));

console.log('pose');
ok('label Front -> 0/0 and flagged approximate', eq(poseFromLabel('Front'), { yaw: 0, pitch: 0, approx: true }));
ok('label "Left Profile + Up Tilt" -> left and up', (() => { const p = poseFromLabel('Left Profile + Up Tilt'); return p.yaw < 0 && p.pitch > 0; })());
ok('label Right Profile -> right', poseFromLabel('Right Profile').yaw > 0);
ok('numeric yaw/pitch win over a label and are NOT approximate', eq(facePose({ yaw: 12, pitch: -3, pose: 'Left Profile' }), { yaw: 12, pitch: -3, approx: false }));
ok('a record with neither falls back to Front', eq(facePose({}), { yaw: 0, pitch: 0, approx: true }));

console.log('coverage');
ok('straight-on lands in F/M', zoneOf(0, 0) === 'F/M');
ok('yaw edges are clamped, not lost', zoneOf(120, 0) === 'R90/M' && zoneOf(-120, 0) === 'L90/M');
ok('a zone boundary belongs to exactly one zone', zoneOf(22, 0) === 'R45/M' && zoneOf(-22, 0) === 'F/M' && zoneOf(-22.01, 0) === 'L45/M');
ok('+90 yaw (the inclusive last edge) is a right profile', zoneOf(90, 0) === 'R90/M');
ok('+45 pitch (inclusive last edge) is Upward', zoneOf(0, 45) === 'F/U');
{
  const none = computeCoverage([]);
  ok('an empty set is missing every needed angle', none.missing.length === none.required && none.covered === 0);
  const front = computeCoverage(pts([[0, 0]]));
  ok('a frontal-only set names the profiles and pitch it lacks',
    front.missing.some((m) => m.label === 'Missing Left 90° profile') && front.missing.some((m) => m.label === 'Missing Upward pitch')
    && !front.missing.some((m) => m.id === 'F/M'), JSON.stringify(front.missing));
  ok('the 45° wording matches the brief', computeCoverage(pts([[0, 0]])).missing.some((m) => m.label === 'Missing Left 45° profile'));
  const full = computeCoverage(pts([[0, 0], [-40, 0], [40, 0], [-75, 0], [75, 0], [0, 25], [0, -25]]));
  ok('one photo in each needed zone leaves nothing missing', full.missing.length === 0 && full.covered === full.required);
  ok('counts add up to the number of faces', Object.values(computeCoverage(pts([[0, 0], [1, 1], [50, 30]])).zones).reduce((a, z) => a + z.count, 0) === 3);
  ok('approx propagates', computeCoverage([{ key: 0, yaw: 0, pitch: 0, approx: true }]).approx === true && front.approx === false);
}

console.log('selection');
const order = [0, 1, 2, 3, 4, 5];
{
  let s = applyClick({ sel: [], anchor: null }, 2, {}, order);
  ok('plain click selects only that face and sets the anchor', eq(s, { sel: [2], anchor: 2 }));
  s = applyClick(s, 4, { shift: true }, order);
  ok('Shift+Click selects the inclusive range from the anchor', eq(s.sel, [2, 3, 4]) && s.anchor === 2);
  s = applyClick(s, 0, { shift: true }, order);
  ok('Shift range works backwards and REPLACES the old range', eq(s.sel, [0, 1, 2]));
  s = applyClick({ sel: [1], anchor: 1 }, 3, { ctrl: true }, order);
  ok('Ctrl+Click adds', eq(s.sel, [1, 3]) && s.anchor === 3);
  s = applyClick(s, 1, { ctrl: true }, order);
  ok('Ctrl+Click on a selected face removes it', eq(s.sel, [3]));
  s = applyClick({ sel: [5], anchor: 5 }, 1, { shift: true, ctrl: true }, order);
  ok('Ctrl+Shift extends the existing selection with the range', eq([...s.sel].sort(), [1, 2, 3, 4, 5]));
  s = applyClick({ sel: [], anchor: null }, 3, { shift: true }, order);
  ok('Shift with no anchor behaves as a plain click', eq(s, { sel: [3], anchor: 3 }));
}

console.log('marquee');
{
  const tiles = [0, 1, 2, 3].map((i) => ({ index: i, rect: { left: i * 100, top: 0, right: i * 100 + 90, bottom: 90 } }));
  const r = rectFromPoints({ x: 250, y: 80 }, { x: 95, y: 10 });
  ok('rectFromPoints normalises any drag direction', eq(r, { left: 95, top: 10, right: 250, bottom: 80 }));
  ok('a marquee picks the tiles it touches', eq(marqueeSelect(r, tiles), [1, 2]));
  ok('a marquee that misses everything selects nothing', eq(marqueeSelect({ left: 1000, top: 1000, right: 1100, bottom: 1100 }, tiles), []));
  ok('additive marquee keeps the prior selection without duplicates', eq(marqueeSelect(r, tiles, [0, 1], true).sort(), [0, 1, 2]));
  ok('edges that only touch do not count', eq(marqueeSelect({ left: 90, top: 0, right: 100, bottom: 90 }, tiles), []));
}

console.log('cluster edits');
{
  const groups = [0, 0, 1, 1, 2];
  ok('ranksOf lists the identities a selection touches, ascending', eq(ranksOf(groups, [4, 0, 2]), [0, 1, 2]));
  ok('merge folds every touched rank into the target', eq(mergeGroups(groups, [0, 2], 0), [0, 0, 1, 1, 0]));
  ok('merge does not mutate its input', eq(groups, [0, 0, 1, 1, 2]));
  const sp = splitGroup(groups, [1]);
  ok('split moves just the chosen face into a brand-new rank', eq(sp.groups, [0, 3, 1, 1, 2]) && sp.rank === 3);
  ok('split of every face of a person still yields a new rank (rank-preserved elsewhere)', eq(splitGroup([0, 0], [0, 1]).groups, [1, 1]));
  ok('split on an empty set is safe', eq(splitGroup([], []).groups, []));
}

console.log('quality');
{
  ok('outliers are strictly below the threshold', eq(outliersBelow([0.9, 0.3, 0.5, 0.2], 0.5), [1, 3]));
  ok('a face with no score is never an outlier', eq(outliersBelow([null, 0.1], 0.5), [1]));
  ok('threshold 0 drops nothing', eq(outliersBelow([0, 0.1], 0), []));
  ok('width -> quality: 24px is 0, 160px+ is 1', qualityFromWidth(24) === 0 && qualityFromWidth(160) === 1 && qualityFromWidth(500) === 1);
  ok('width unknown -> null, not 0', qualityFromWidth(undefined) === null && qualityFromWidth(0) === null);
  ok('thumbnail size is clamped to 64..240', clampThumb(10) === 64 && clampThumb(999) === 240 && clampThumb('112') === 112 && clampThumb('x') === 112);
}

console.log('library');
{
  const lib = [
    { name: 'Lead Portrait', faces: 1, size: 84 * 1024 },
    { name: 'Cinematic Actors', faces: 18, size: 18 * 1024 * 1024 },
    { name: 'Studio Cast', faces: 7, size: 2 * 1024 * 1024 },
  ];
  ok('search is case-insensitive substring', eq(filterLibrary(lib, 'ACTORS').map((e) => e.name), ['Cinematic Actors']));
  ok('tag filter narrows', eq(filterLibrary(lib, '', 'single').map((e) => e.name), ['Lead Portrait']));
  ok('query and tag combine', filterLibrary(lib, 'cast', 'rich').length === 0 && filterLibrary(lib, 'cast', 'multi-angle').length === 1);
  ok('tags are derived from the entry', eq(libraryTags(lib[1]), ['rich', 'large']) && eq(libraryTags(lib[0]), ['single']));
  ok('byte formatting', fmtBytes(512) === '512 B' && fmtBytes(2048) === '2.0 KB' && fmtBytes(5 * 1024 * 1024) === '5.0 MB');
}

console.log('hero');
{
  ok('a stored hero that still exists wins', resolveHero({ p1: 3 }, 'p1', [2, 3, 4]) === 3);
  ok('a stale hero falls back to the first face', resolveHero({ p1: 9 }, 'p1', [2, 3, 4]) === 2);
  ok('no hero stored -> first face', resolveHero({}, 'p1', [5, 6]) === 5);
  ok('an empty person has no hero', resolveHero({}, 'p1', []) === null);
}

console.log(failures === 0 ? '\nall identity workstation logic checks passed' : `\n${failures} identity workstation logic check(s) FAILED`);
globalThis.__RENDER_CHECK_FAILURES__ = failures;
