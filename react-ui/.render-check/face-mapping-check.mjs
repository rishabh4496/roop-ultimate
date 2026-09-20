// Behavioural checks for the target-person -> source-face mapping.
// These import and RUN the real module the UI ships, rather than asserting on
// source text, so a refactor that keeps the strings but breaks the decision
// still fails here. Run with: node .render-check/face-mapping-check.mjs
import assert from 'node:assert/strict';
import process from 'node:process';
import { buildFaceMappingArray, mapPerson, selectedPersonOf, SKIP }
  from '../src/components/faceswap/faceMapping.js';

let pass = 0;
const fails = [];
const check = (name, fn) => {
  try { fn(); pass++; console.log(`  PASS  ${name}`); }
  catch (e) { fails.push(name); console.log(`  FAIL  ${name}\n        ${e.message}`); }
};
const group = (t) => console.log(`\n── ${t} ${'─'.repeat(Math.max(0, 50 - t.length))}`);

const SEL = 'Selected face';
const build = (o) => buildFaceMappingArray({ faceMapping: {}, ...o });

group('Selected face: only the highlighted person swaps');
check('two people, highlight person 1 -> only person 1 gets the source', () => {
  // The reported bug: highlighting the second person swapped the first.
  assert.deepEqual(
    build({ targetGroups: [0, 1], selTargetFace: 1, selectedSource: 0, sourceCount: 1, faceSelection: SEL }),
    [SKIP, 0],
  );
});
check('two people, highlight person 0 -> only person 0 gets the source', () => {
  assert.deepEqual(
    build({ targetGroups: [0, 1], selTargetFace: 0, selectedSource: 0, sourceCount: 1, faceSelection: SEL }),
    [0, SKIP],
  );
});
check('the OLD behaviour ([0,1] for one source) is gone', () => {
  const out = build({ targetGroups: [0, 1], selTargetFace: 1, selectedSource: 0, sourceCount: 1, faceSelection: SEL });
  assert.notDeepEqual(out, [0, 1]);
  assert.ok(out.every((v) => v === SKIP || v < 1), 'no index past the source gallery');
});
check('a second source face is honoured when it is the one selected', () => {
  assert.deepEqual(
    build({ targetGroups: [0, 1], selTargetFace: 1, selectedSource: 1, sourceCount: 2, faceSelection: SEL }),
    [SKIP, 1],
  );
});
check('many target faces of the same person still yield one entry', () => {
  assert.deepEqual(
    build({ targetGroups: [0, 0, 1, 1, 1], selTargetFace: 4, selectedSource: 0, sourceCount: 1, faceSelection: SEL }),
    [SKIP, 0],
  );
});
check('three people: exactly one non-skip', () => {
  const out = build({ targetGroups: [0, 1, 2], selTargetFace: 2, selectedSource: 0, sourceCount: 3, faceSelection: SEL });
  assert.equal(out.filter((v) => v !== SKIP).length, 1);
  assert.equal(out[2], 0);
});

group('Selected face: edge cases that used to send junk');
check('no source faces loaded -> everyone skipped, nothing out of range', () => {
  assert.deepEqual(
    build({ targetGroups: [0, 1], selTargetFace: 0, selectedSource: 0, sourceCount: 0, faceSelection: SEL }),
    [SKIP, SKIP],
  );
});
check('selected source past the gallery is clamped, never out of range', () => {
  const out = build({ targetGroups: [0], selTargetFace: 0, selectedSource: 7, sourceCount: 2, faceSelection: SEL });
  assert.deepEqual(out, [1]);
});
check('negative selected source is a skip, not a Python tail index', () => {
  assert.deepEqual(
    build({ targetGroups: [0], selTargetFace: 0, selectedSource: -1, sourceCount: 2, faceSelection: SEL }),
    [SKIP],
  );
});
check('non-contiguous person ranks map by rank order, not array position', () => {
  assert.deepEqual(
    build({ targetGroups: [0, 3], selTargetFace: 1, selectedSource: 0, sourceCount: 1, faceSelection: SEL }),
    [SKIP, 0],
  );
});
check('array-valued group entry resolves to its person', () => {
  assert.equal(selectedPersonOf([[0], [1]], 1), 1);
});
check('unresolvable highlight skips everyone rather than guessing', () => {
  assert.deepEqual(
    build({ targetGroups: [0, 1], selTargetFace: 9, selectedSource: 0, sourceCount: 1, faceSelection: SEL }),
    [SKIP, SKIP],
  );
});

group('Explicit dropdown choices still win');
check('explicit mapping overrides the selected-face default', () => {
  assert.deepEqual(
    build({ targetGroups: [0, 1], faceMapping: { 0: 1 }, selTargetFace: 1, selectedSource: 0, sourceCount: 2, faceSelection: SEL }),
    [1, 0],
  );
});
check('explicit Skip is preserved for the highlighted person', () => {
  assert.deepEqual(
    build({ targetGroups: [0], faceMapping: { 0: -1 }, selTargetFace: 0, selectedSource: 0, sourceCount: 2, faceSelection: SEL }),
    [SKIP],
  );
});
check('explicit out-of-range index degrades to Skip', () => {
  assert.deepEqual(
    build({ targetGroups: [0], faceMapping: { 0: 5 }, selTargetFace: 0, selectedSource: 0, sourceCount: 2, faceSelection: SEL }),
    [SKIP],
  );
});
check('explicit string index from storage is honoured', () => {
  assert.deepEqual(
    build({ targetGroups: [0], faceMapping: { 0: '1' }, selTargetFace: 0, selectedSource: 0, sourceCount: 2, faceSelection: SEL }),
    [1],
  );
});
check('explicit garbage degrades to Skip, not NaN', () => {
  const out = build({ targetGroups: [0], faceMapping: { 0: 'abc' }, selTargetFace: 0, selectedSource: 0, sourceCount: 2, faceSelection: SEL });
  assert.deepEqual(out, [SKIP]);
  assert.ok(Number.isFinite(out[0]));
});

group('Other detection modes keep the legacy person-rank default');
check('All faces: person rank is the source index', () => {
  assert.deepEqual(
    build({ targetGroups: [0, 1], selTargetFace: 0, selectedSource: 0, sourceCount: 2, faceSelection: 'All faces' }),
    [0, 1],
  );
});
check('All faces: rank past the gallery becomes Skip, not out of range', () => {
  assert.deepEqual(
    build({ targetGroups: [0, 1, 2], selTargetFace: 0, selectedSource: 0, sourceCount: 2, faceSelection: 'All faces' }),
    [0, 1, SKIP],
  );
});

group('The payload never violates the backend contract');
check('every entry is -1 or a valid gallery index, across a mode/size sweep', () => {
  const modes = [SEL, 'All faces', 'First found', 'All female', 'All male', undefined];
  for (const faceSelection of modes) {
    for (let sourceCount = 0; sourceCount <= 3; sourceCount++) {
      for (const targetGroups of [[], [0], [0, 1], [0, 1, 2], [0, 0, 1], [2, 0]]) {
        for (let selTargetFace = 0; selTargetFace <= targetGroups.length; selTargetFace++) {
          for (const selectedSource of [-1, 0, 1, 5]) {
            const out = build({ targetGroups, selTargetFace, selectedSource, sourceCount, faceSelection });
            for (const v of out) {
              assert.ok(Number.isInteger(v), `non-integer ${v}`);
              assert.ok(v === SKIP || (v >= 0 && v < sourceCount),
                `bad entry ${v} for sourceCount=${sourceCount} mode=${faceSelection}`);
            }
          }
        }
      }
    }
  }
});
check('Selected face never returns more than one non-skip entry', () => {
  for (let sourceCount = 1; sourceCount <= 3; sourceCount++) {
    for (const targetGroups of [[0], [0, 1], [0, 1, 2], [0, 0, 1, 2]]) {
      for (let selTargetFace = 0; selTargetFace < targetGroups.length; selTargetFace++) {
        const out = build({ targetGroups, selTargetFace, selectedSource: 0, sourceCount, faceSelection: SEL });
        assert.ok(out.filter((v) => v !== SKIP).length <= 1);
      }
    }
  }
});

group('The dropdown shows what the payload will do');
check('row value equals payload entry for every person, in both modes', () => {
  for (const faceSelection of [SEL, 'All faces']) {
    for (const faceMapping of [{}, { 0: 1 }, { 1: -1 }, { 0: 9 }]) {
      const targetGroups = [0, 1, 2];
      const selTargetFace = 1;
      const sourceCount = 2;
      const selectedSource = 1;
      const payload = build({ targetGroups, faceMapping, selTargetFace, selectedSource, sourceCount, faceSelection });
      [0, 1, 2].forEach((person, i) => {
        const shown = mapPerson({
          person, faceMapping, sourceCount, faceSelection,
          selectedPerson: 1, selectedSource,
        });
        assert.equal(shown, payload[i], `person ${person} row ${shown} != payload ${payload[i]}`);
      });
    }
  }
});

console.log(`\n${fails.length ? `FAILED: ${fails.join(', ')}` : `ALL GREEN: ${pass}/${pass} checks passed`}`);
process.exit(fails.length ? 1 : 0);
