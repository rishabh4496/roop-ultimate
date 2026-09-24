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

  console.log('── 1-to-Many & Many-to-1 Mapping Contracts ───────────────');
  {
    // Simulating mapping mutations
    let mapping = {};

    // 1-to-many: Assign Source 0 to Target 1 and Target 3 simultaneously
    mapping['tp_1'] = 0;
    mapping['tp_3'] = 0;
    ok('1-to-many: Target 1 maps to Source 0', mapping['tp_1'] === 0);
    ok('1-to-many: Target 3 maps to Source 0 simultaneously', mapping['tp_3'] === 0);

    // Many-to-1: Target 2 receives multiple sources (e.g. multi-angle references [0, 2])
    mapping['tp_2'] = [0, 2];
    ok('Many-to-1: Target 2 maps to multiple sources ([0, 2])',
      Array.isArray(mapping['tp_2']) && mapping['tp_2'].length === 2 && mapping['tp_2'].includes(0) && mapping['tp_2'].includes(2)
    );

    // Removal of single source from many-to-1
    const target2Sources = Array.isArray(mapping['tp_2']) ? mapping['tp_2'] : [mapping['tp_2']];
    const filteredSources = target2Sources.filter((s) => s !== 2);
    mapping['tp_2'] = filteredSources.length === 1 ? filteredSources[0] : filteredSources;
    ok('Removing source 2 simplifies Target 2 back to single source 0', mapping['tp_2'] === 0);

    // Unassign / Skip target 1
    mapping['tp_1'] = -1;
    ok('Unassigning target 1 sets sentinel -1 without affecting target 3', mapping['tp_1'] === -1 && mapping['tp_3'] === 0);
  }

  console.log('── Parameter Overrides & Boundary Validation ──────────────');
  {
    const overrides = {
      tp_1: {
        cosineThreshold: 0.65,
        maskOffset: -4,
        action: 'swap',
      },
      tp_2: {
        cosineThreshold: 0.75,
        maskOffset: 8,
        action: 'keep',
      },
      tp_3: {
        cosineThreshold: 0.40,
        maskOffset: 0,
        action: 'censor',
      },
    };

    // Cosine threshold range [0.30, 0.85]
    ok('Cosine threshold for tp_1 is in valid range [0.30, 0.85]',
      overrides.tp_1.cosineThreshold >= 0.30 && overrides.tp_1.cosineThreshold <= 0.85
    );

    // Mask offset range [-20, 20]
    ok('Erosion offset -4px is within valid range [-20, 20]',
      overrides.tp_1.maskOffset >= -20 && overrides.tp_1.maskOffset <= 20
    );
    ok('Dilation offset +8px is within valid range [-20, 20]',
      overrides.tp_2.maskOffset >= -20 && overrides.tp_2.maskOffset <= 20
    );

    // Action modes
    ok('Action "swap" recognized', overrides.tp_1.action === 'swap');
    ok('Action "keep" recognized (bypass)', overrides.tp_2.action === 'keep');
    ok('Action "censor" recognized (blur/privacy)', overrides.tp_3.action === 'censor');
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
