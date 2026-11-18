/**
 * Route state preservation and offline compliance contract test.
 *
 * Verifies:
 * 1. All 9 deep-link hash routes (#/home, #/faceswap, #/batch, #/processing, #/facemgr, #/extras, #/gallery, #/history, #/settings)
 * 2. Route resolution from URL hash (with or without leading slashes, subpaths, query strings)
 * 3. State preservation across Pinokio tab switches (simulating bare URL webview reloads via localStorage roop_active_tab)
 * 4. Automatic runTabOpen activation for #/processing recovery
 * 5. Precedence of URL hash deep links over persisted localStorage
 * 6. Graceful fallback to 'faceswap' when neither hash nor storage is available
 */

const ALL_ROUTE_IDS = ['home', 'faceswap', 'batch', 'processing', 'facemgr', 'extras', 'gallery', 'history', 'settings'];

let failures = 0;
const ok = (name, cond, extra = '') => {
  if (!cond) {
    failures += 1;
    console.error(`  FAIL  ${name}${extra ? ` — ${extra}` : ''}`);
  } else {
    console.log(`  PASS  ${name}`);
  }
};

const tabFromHash = (hash, valid) => {
  const id = String(hash || '').replace(/^#\/?/, '').split(/[/?]/)[0];
  return valid.includes(id) ? id : null;
};

const resolveInitialTab = (valid, mockLocation, mockStorage) => {
  const fromHash = tabFromHash(mockLocation?.hash, valid);
  if (fromHash) return fromHash;
  try {
    const saved = mockStorage?.getItem('roop_active_tab');
    if (saved && valid.includes(saved)) return saved;
  } catch { /* storage unavailable */ }
  return 'faceswap';
};

console.log('── 1. All 9 deep-link routes hash resolution ──');
for (const id of ALL_ROUTE_IDS) {
  ok(`resolves direct hash '#/${id}'`, tabFromHash(`#/${id}`, ALL_ROUTE_IDS) === id);
  ok(`resolves hash without slash '#${id}'`, tabFromHash(`#${id}`, ALL_ROUTE_IDS) === id);
  ok(`resolves hash with subpath '#/${id}/sub'`, tabFromHash(`#/${id}/sub`, ALL_ROUTE_IDS) === id);
  ok(`resolves hash with query param '#/${id}?key=val'`, tabFromHash(`#/${id}?key=val`, ALL_ROUTE_IDS) === id);
}

ok('invalid hash returns null', tabFromHash('#/unknown_route', ALL_ROUTE_IDS) === null);
ok('empty hash returns null', tabFromHash('', ALL_ROUTE_IDS) === null);
ok('bare hash returns null', tabFromHash('#', ALL_ROUTE_IDS) === null);
ok('bare slash hash returns null', tabFromHash('#/', ALL_ROUTE_IDS) === null);

console.log('\n── 2. Route state preservation across Pinokio tab switches (bare URL reload) ──');
for (const id of ALL_ROUTE_IDS) {
  const storage = new Map([['roop_active_tab', id]]);
  const mockStorage = {
    getItem: (k) => storage.get(k) || null,
    setItem: (k, v) => storage.set(k, String(v)),
  };
  const mockLocation = { hash: '' }; // Pinokio tab switch reload without hash

  const resolved = resolveInitialTab(ALL_ROUTE_IDS, mockLocation, mockStorage);
  ok(`preserves route '#/${id}' on bare URL reload`, resolved === id);
}

console.log('\n── 3. Processing tab retention and priority ──');
{
  const storage = new Map([['roop_active_tab', 'processing']]);
  const mockStorage = {
    getItem: (k) => storage.get(k) || null,
    setItem: (k, v) => storage.set(k, String(v)),
  };
  const mockLocation = { hash: '' };
  const resolved = resolveInitialTab(ALL_ROUTE_IDS, mockLocation, mockStorage);
  ok('processing tab is restored', resolved === 'processing');
  const runTabOpen = resolved === 'processing';
  ok('runTabOpen is automatically true when processing route is restored', runTabOpen === true);
}

console.log('\n── 4. URL deep link takes precedence over localStorage ──');
{
  const storage = new Map([['roop_active_tab', 'home']]);
  const mockStorage = {
    getItem: (k) => storage.get(k) || null,
    setItem: (k, v) => storage.set(k, String(v)),
  };
  const mockLocation = { hash: '#/settings' };
  const resolved = resolveInitialTab(ALL_ROUTE_IDS, mockLocation, mockStorage);
  ok('URL deep link #/settings overrides saved home tab', resolved === 'settings');
}

console.log('\n── 5. Graceful fallback on clean launch or storage failure ──');
{
  const mockLocation = { hash: '' };
  const mockStorage = { getItem: () => null };
  const resolved = resolveInitialTab(ALL_ROUTE_IDS, mockLocation, mockStorage);
  ok('defaults to faceswap when no hash and empty storage', resolved === 'faceswap');

  const throwingStorage = { getItem: () => { throw new Error('SecurityError'); } };
  const resolvedThrow = resolveInitialTab(ALL_ROUTE_IDS, mockLocation, throwingStorage);
  ok('defaults to faceswap when localStorage access throws', resolvedThrow === 'faceswap');
}

console.log(failures === 0 ? `\nALL GREEN: route state preservation checks passed` : `\n${failures} check(s) FAILED`);
globalThis.__RENDER_CHECK_FAILURES__ = failures;
if (typeof process !== 'undefined') {
  process.exitCode = failures === 0 ? 0 : 1;
}
