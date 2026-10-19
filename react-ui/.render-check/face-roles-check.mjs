// Behavioural checks for the target-pin / bystander-exclusion decision the Media
// Canvas badges and the Person Groups dropdown share. They import and RUN the real
// module (src/components/faceswap/faceRoles.js).
//
//   node .render-check/face-roles-check.mjs
//
// The first group is the defect that motivated the module: "Ignore / Skip" used to
// DELETE the person's key, which only skips in "Selected face"; in every other mode
// an unmapped person falls back to the legacy rank default and is swapped anyway.
import assert from 'node:assert/strict';
import process from 'node:process';
import { buildFaceMappingArray, SKIP } from '../src/components/faceswap/faceMapping.js';
import {
  ROLE, excludePerson, faceRole, includePerson, isExcluded,
} from '../src/components/faceswap/faceRoles.js';

let pass = 0;
const fails = [];
const check = (name, fn) => {
  try { fn(); pass++; console.log(`  PASS  ${name}`); }
  catch (e) { fails.push(name); console.log(`  FAIL  ${name}\n        ${e.message}`); }
};
const group = (t) => console.log(`\n── ${t} ${'─'.repeat(Math.max(0, 50 - t.length))}`);

// Two captured people, two sources, person A is the highlighted one.
const base = {
  targetGroups: [0, 1], targetPersonIds: ['pA', 'pB'], sourceCount: 2,
  sourceIdentityIds: ['srcA', 'srcB'], selectedSource: 0, selectedTargetPersonId: 'pA', selTargetFace: 0,
};
const payload = (mode, faceMapping) => buildFaceMappingArray({ ...base, faceSelection: mode, faceMapping });

group('The motivating defect: a deleted key is not a skip');
for (const mode of ['Selected people', 'All faces']) {
  check(`${mode}: DELETING person B's key still swaps B (the old Skip)`, () => {
    assert.deepEqual(payload(mode, { pA: 'srcA' }), [0, 1], 'control: the old behaviour must be reproducible');
  });
  check(`${mode}: an explicit never-swap entry skips B`, () => {
    assert.deepEqual(payload(mode, excludePerson({ pA: 'srcA' }, 'pB')), [0, SKIP]);
  });
}
check('Selected face: B is not swapped either way (it was never the target)', () => {
  assert.deepEqual(payload('Selected face', { pA: 'srcA' }), [0, SKIP]);
  assert.deepEqual(payload('Selected face', excludePerson({ pA: 'srcA' }, 'pB')), [0, SKIP]);
});

group('excludePerson / includePerson / isExcluded');
check('exclude returns a new object and leaves the input untouched', () => {
  const before = { pA: 'srcA' };
  const after = excludePerson(before, 'pB');
  assert.deepEqual(before, { pA: 'srcA' });
  assert.ok(Object.prototype.hasOwnProperty.call(after, 'pB'));
  assert.equal(after.pB, null);
});
check('include drops the key, returning the person to the mode default', () => {
  const next = includePerson(excludePerson({ pA: 'srcA' }, 'pB'), 'pB');
  assert.equal(Object.prototype.hasOwnProperty.call(next, 'pB'), false);
  assert.deepEqual(payload('Selected people', next), [0, 1]);
});
check('exclude with no person id is a no-op', () => {
  assert.deepEqual(excludePerson({ pA: 'srcA' }, null), { pA: 'srcA' });
  assert.deepEqual(excludePerson({ pA: 'srcA' }, ''), { pA: 'srcA' });
});
check('isExcluded recognises every spelling of skip, and not a real source', () => {
  for (const v of [null, '', -1, '-1', 'skip', 'Ignore', 'none']) {
    assert.equal(isExcluded({ pB: v }, 'pB', 2), true, String(v));
  }
  assert.equal(isExcluded({ pB: 'srcB' }, 'pB', 2), false);
  assert.equal(isExcluded({ pB: 1 }, 'pB', 2), false);
  assert.equal(isExcluded({ pB: 5 }, 'pB', 2), true, 'an index past the gallery is a skip');
  assert.equal(isExcluded({}, 'pB', 2), false, 'no key is not an exclusion');
});

group('faceRole: what the badge shows');
const role = (o) => faceRole({ ...base, ...o });
check('Selected face: the pinned person is the TARGET and cannot be excluded', () => {
  const r = role({ personId: 'pA', faceSelection: 'Selected face', faceMapping: {} });
  assert.equal(r.role, ROLE.TARGET);
  assert.equal(r.canExclude, false);
  assert.match(r.reason, /pinned target/i);
});
check('Selected face: every other captured person is a bystander', () => {
  const r = role({ personId: 'pB', faceSelection: 'Selected face', faceMapping: {} });
  assert.equal(r.role, ROLE.BYSTANDER);
  assert.equal(r.excluded, false);
  assert.equal(r.canExclude, true);
});
check('Selected people: a mapped person is SWAPPED and can be excluded', () => {
  const r = role({ personId: 'pB', faceSelection: 'Selected people', faceMapping: { pA: 'srcA', pB: 'srcB' } });
  assert.equal(r.role, ROLE.SWAPPED);
  assert.equal(r.canExclude, true);
});
check('Selected people: after excluding, the badge reads BYSTANDER / excluded', () => {
  const r = role({ personId: 'pB', faceSelection: 'Selected people', faceMapping: excludePerson({ pA: 'srcA' }, 'pB') });
  assert.equal(r.role, ROLE.BYSTANDER);
  assert.equal(r.excluded, true);
});
check('the role agrees with the payload for every person in every mode', () => {
  for (const mode of ['Selected face', 'Selected people', 'All faces']) {
    for (const fm of [{}, { pA: 'srcA' }, { pA: 'srcA', pB: 'srcB' }, excludePerson({ pA: 'srcA' }, 'pB'), excludePerson({}, 'pA')]) {
      const arr = payload(mode, fm);
      ['pA', 'pB'].forEach((id, i) => {
        const r = role({ personId: id, faceSelection: mode, faceMapping: fm });
        const swapped = arr[i] !== SKIP;
        assert.equal(r.role !== ROLE.BYSTANDER, swapped,
          `${mode} ${JSON.stringify(fm)} ${id}: badge says ${r.role}, payload says ${arr[i]}`);
      });
    }
  }
});
check('a face nobody captured: bystander in the people modes', () => {
  for (const mode of ['Selected face', 'Selected people']) {
    const r = role({ personId: null, faceSelection: mode });
    assert.equal(r.role, ROLE.BYSTANDER);
    assert.equal(r.canExclude, false);
  }
});
check('a face nobody captured in "All faces" is swapped and cannot be excluded without capturing', () => {
  const r = role({ personId: null, faceSelection: 'All faces' });
  assert.equal(r.role, ROLE.SWAPPED);
  assert.equal(r.canExclude, false);
  assert.match(r.reason, /capture/i);
});

console.log(`\n${fails.length ? `FAILED: ${fails.join(', ')}` : `ALL GREEN: ${pass}/${pass} checks passed`}`);
process.exit(fails.length ? 1 : 0);
