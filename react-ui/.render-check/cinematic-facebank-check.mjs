/**
 * FaceBankRouter Verification Suite
 * Tests IndexedDB/memory crop caching, 1-to-many and many-to-1 mapping logic,
 * parameter override semantics, and actor discovery contracts.
 */
import process from 'node:process';
import {
  cacheCrop,
  getCachedCrop,
  deleteCachedCrop,
  clearCropCache,
  getCacheStats,
} from '../src/components/facebank/faceBankDb.js';
import {
  assignSource,
  removeSource,
  normalizeOverrides,
  UNASSIGNED,
} from '../src/components/facebank/mappingOps.js';

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
  console.log('── Face Bank Cache Lifecycle & IndexedDB Fallback ────────');
  {
    await clearCropCache();

    // Stats empty
    const stats0 = await getCacheStats();
    ok('Initial cache count is 0', stats0.count === 0);

    // Cache a sample crop (base64 data URL)
    const testDataUrl = 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP...';
    const putSuccess = await cacheCrop('target_001', testDataUrl, {
      actorName: 'Actor 1',
      clusterSize: 48,
      detScore: 0.94,
    });
    ok('cacheCrop returns true on successful insertion', putSuccess === true);

    // Retrieve crop
    const retrieved = await getCachedCrop('target_001');
    ok('getCachedCrop retrieves stored record', retrieved !== null && retrieved.key === 'target_001');
    ok('getCachedCrop preserves payload data', retrieved.data === testDataUrl);
    ok('getCachedCrop preserves metadata', retrieved.metadata?.actorName === 'Actor 1');

    // Stats updated
    const stats1 = await getCacheStats();
    ok('Cache count incremented to 1', stats1.count === 1);

    // Non-existent key
    const missing = await getCachedCrop('unknown_key');
    ok('getCachedCrop returns null for missing key', missing === null);

    // Delete single crop
    const delSuccess = await deleteCachedCrop('target_001');
    ok('deleteCachedCrop returns true', delSuccess === true);
    const postDelete = await getCachedCrop('target_001');
    ok('Deleted crop is no longer present', postDelete === null);

    // Clear all
    await cacheCrop('k1', 'val1');
    await cacheCrop('k2', 'val2');
    await clearCropCache();
    const stats2 = await getCacheStats();
    ok('clearCropCache empties cache completely', stats2.count === 0);
  }

  console.log('── Mapping mutations (real mappingOps used by FaceBankRouter) ──');
  {
    const m0 = Object.freeze({});
    const m1 = assignSource(m0, 'tp_1', 0);
    const m2 = assignSource(m1, 'tp_3', 0);
    ok('1-to-many: one source on two targets', m2.tp_1 === 0 && m2.tp_3 === 0);
    ok('assignSource never mutates its input', Object.keys(m0).length === 0 && m1.tp_3 === undefined);

    const m3 = assignSource(m2, 'tp_2', 0, true);
    ok('append onto an EMPTY target assigns a scalar', m3.tp_2 === 0);
    const m4 = assignSource(m3, 'tp_2', 2, true);
    ok('Many-to-1: append makes [0, 2]', Array.isArray(m4.tp_2) && m4.tp_2.join() === '0,2');
    const m5 = assignSource(m4, 'tp_2', 2, true);
    ok('append does not duplicate a source', m5.tp_2.join() === '0,2');
    ok('append onto UNASSIGNED replaces it', assignSource({ tp_9: UNASSIGNED }, 'tp_9', 4, true).tp_9 === 4);
    ok('non-append replaces a set with a scalar', assignSource(m5, 'tp_2', 7).tp_2 === 7);
    ok('numeric cluster ids key as strings', assignSource({}, 3, 1)['3'] === 1);

    const r1 = removeSource(m5, 'tp_2', 2);
    ok('removing from [0, 2] collapses to scalar 0', r1.tp_2 === 0);
    const r2 = removeSource(m5, 'tp_2', '2');
    ok('removal matches ids across string/number', r2.tp_2 === 0);
    ok('removing the last source unassigns', removeSource(r1, 'tp_2', 0).tp_2 === UNASSIGNED);
    ok('removing from [a, b, c] keeps an array', removeSource({ t: [1, 2, 3] }, 't', 2).t.join() === '1,3');
    ok('removing on an unknown cluster returns the same mapping', removeSource(m5, 'nope', 0) === m5);
    ok('removal leaves other targets alone', r1.tp_1 === 0 && r1.tp_3 === 0);
  }

  console.log('── Override defaults (real normalizeOverrides) ────────────');
  {
    const d = normalizeOverrides();
    ok('defaults: threshold 0.60, offset 0, action swap',
      d.cosineThreshold === 0.60 && d.maskOffset === 0 && d.action === 'swap');
    const z = normalizeOverrides({ cosineThreshold: 0, maskOffset: 0 });
    ok('an explicit 0 is kept, not replaced by the default', z.cosineThreshold === 0 && z.maskOffset === 0);
    ok('an explicit action is kept', normalizeOverrides({ action: 'keep' }).action === 'keep');
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
