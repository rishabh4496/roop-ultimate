import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { TABS, openTab, loadAllowlist, saveAllowlist, UPDATE_BASELINE } from './helpers.js';

// axe on every tab. A tab's known violations are recorded in allowlist.json as
// `rule id -> node count`; the test fails when a rule appears that is not there
// or when a recorded rule matches MORE nodes than the baseline. Fixing things
// never fails the run -- it annotates, and `npm run test:e2e:baseline` ratchets
// the file down.
const TAGS = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'best-practice'];

for (const tab of TABS) {
  test(`axe: ${tab.id}`, async ({ page }, testInfo) => {
    await openTab(page, tab.id);

    const { violations } = await new AxeBuilder({ page }).withTags(TAGS).analyze();
    const found = Object.fromEntries(violations.map((v) => [v.id, v.nodes.length]));

    if (UPDATE_BASELINE) {
      saveAllowlist((a) => { a.axe[tab.id] = found; });
      return;
    }

    const known = loadAllowlist().axe[tab.id] ?? {};
    const regressions = violations
      .filter((v) => v.nodes.length > (known[v.id] ?? 0))
      .map((v) => ({
        rule: v.id,
        impact: v.impact,
        baseline: known[v.id] ?? 0,
        now: v.nodes.length,
        help: v.help,
        targets: v.nodes.slice(0, 5).map((n) => n.target.join(' ')),
      }));

    const improved = Object.keys(known).filter((id) => (found[id] ?? 0) < known[id]);
    if (improved.length) {
      testInfo.annotations.push({
        type: 'tighten allowlist',
        description: `${tab.id}: ${improved.join(', ')} improved -- run npm run test:e2e:baseline`,
      });
    }

    expect(regressions, `new or worse axe violations on ${tab.id}`).toEqual([]);
  });
}
