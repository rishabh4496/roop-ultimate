import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { VIEWS, openTab, loadAllowlist, saveAllowlist, UPDATE_BASELINE } from './helpers.js';

// axe on every tab, and on the Batch Matrix strategies (see helpers.STATES). A
// view's known violations are recorded in allowlist.json as `rule id -> node
// count`; the test fails when a rule appears that is not there or when a recorded
// rule matches MORE nodes than the baseline. Fixing things never fails the run --
// it annotates, and `npm run test:e2e:baseline` ratchets the file down.
//
// The allowlist is EMPTY for axe today: every view has zero violations, so any
// violation at all -- not just serious/critical -- is a regression.
//
// What axe cannot decide (text over this app's gradient and glass: ~1,000 nodes
// land in `incomplete`) is measured instead by contrast.spec.js.
const TAGS = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'best-practice'];

for (const view of VIEWS) {
  test(`axe: ${view.id}`, async ({ page }, testInfo) => {
    await view.open(page);

    const { violations } = await new AxeBuilder({ page }).withTags(TAGS).analyze();
    const found = Object.fromEntries(violations.map((v) => [v.id, v.nodes.length]));

    if (UPDATE_BASELINE) {
      saveAllowlist((a) => { a.axe[view.id] = found; });
      return;
    }

    const known = loadAllowlist().axe[view.id] ?? {};
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
        description: `${view.id}: ${improved.join(', ')} improved -- run npm run test:e2e:baseline`,
      });
    }

    expect(regressions, `new or worse axe violations on ${view.id}`).toEqual([]);
  });
}

// WCAG 2.5.3 (label in name): the visible text of a control must be inside its
// accessible name, or speech-input users cannot say what they see. The zoom
// readout is a button whose visible text is the percentage.
test('the zoom button\'s accessible name contains its visible "100%"', async ({ page }) => {
  await openTab(page, 'faceswap');
  // The app zoom lives in the header; the preview stage has its own controls.
  const zoom = page.locator('header').getByRole('button', { name: /100%/ });
  await expect(zoom).toHaveCount(1);
  await expect(zoom).toHaveText('100%');
  await expect(zoom).toHaveAccessibleName(/reset zoom/i);
});
