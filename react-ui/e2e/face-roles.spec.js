import { test, expect } from '@playwright/test';
import { openTab, makePng } from './helpers.js';

// The Media Canvas's per-face controls: every detected box says what the render will do
// with it (Target / Swapped / Bystander), and a captured person can be locked out with one
// click. The decision itself is pinned by .render-check/face-roles-check.mjs; this is the
// DOM contract, in a real browser:
//   * pinning a box makes it the Target, and the Target cannot be excluded;
//   * excluding another person writes an EXPLICIT never-swap entry (null), not a deleted
//     key (a deleted key falls back to the rank default in every mode but "Selected face");
//   * "Include again" removes the entry;
//   * a face nobody captured has a role but no Exclude button (nothing to key it on).

const FRAME = { w: 640, h: 360 };
const P0 = 'legacy-target-person-0';
const P1 = 'legacy-target-person-1';

async function openTwoPeople(page) {
  const png = makePng(FRAME.w, FRAME.h);
  const contextWrites = [];

  await page.route('**/api/target/preview*', (route) => route.fulfill({ status: 200, contentType: 'image/png', body: png }));

  // Two captured people instead of the mock's one.
  await page.route('**/api/state', async (route) => {
    const res = await route.fetch();
    const json = await res.json();
    const face = json.target_faces?.[0];
    json.target_faces = [face, face];
    json.target_groups = [0, 1];
    json.target_names = ['Lead', 'Guest'];
    json.target_faces_info = [
      { index: 0, person: 'Person 1', landmarks: 106, gender: 'female', age: 26, bbox: [40, 50, 120, 160], pose: 'Front' },
      { index: 1, person: 'Person 2', landmarks: 106, gender: 'male', age: 41, bbox: [360, 60, 120, 160], pose: 'Front' },
    ];
    await route.fulfill({ response: res, json });
  });

  // Three boxes: person 0, person 1, and a face nobody captured.
  await page.route('**/api/preview', (route) => {
    if (route.request().method() !== 'POST') return route.fallback();
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        image: `data:image/png;base64,${png.toString('base64')}`,
        faces: [[40, 50, 160, 210], [260, 60, 380, 220], [470, 70, 590, 230]],
        // The backend numbers faces that match no captured person AFTER the known ones.
        person_ids: [0, 1, 2],
        kps: [], pose: [],
      }),
    });
  });

  // The mock has no target-context route; record what the UI writes.
  await page.route('**/api/target/context*', (route) => {
    if (route.request().method() === 'POST') {
      try { contextWrites.push(JSON.parse(route.request().postData() || '{}')); } catch { /* not JSON */ }
      return route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
    }
    return route.fallback();
  });

  await openTab(page, 'faceswap');
  await expect(page.locator('[data-face-index="0"]')).toBeVisible({ timeout: 20_000 });
  return contextWrites;
}

const box = (page, i) => page.locator(`[data-face-index="${i}"]`);
const mappingOf = (write) => write.target_person_source_mapping ?? write.face_mapping;
const lastMapping = (writes) => {
  for (let i = writes.length - 1; i >= 0; i--) {
    const m = mappingOf(writes[i]);
    if (m && typeof m === 'object' && !Array.isArray(m)) return m;
  }
  return null;
};

test('every detected box shows what the render will do with it', async ({ page }) => {
  await openTwoPeople(page);
  for (const i of [0, 1, 2]) {
    await expect(box(page, i)).toHaveAttribute('data-face-role', /^(target|swapped|bystander)$/);
    await expect(page.locator(`[data-face-role-label="${i}"]`)).toBeVisible();
  }
  // The face nobody captured is never swapped in the people modes, and has no Exclude button.
  await expect(box(page, 2)).toHaveAttribute('data-face-role', 'bystander');
  await expect(page.locator('[data-face-exclude="2"]')).toHaveCount(0);
});

test('pin a box: it becomes the Target, and the Target cannot be excluded', async ({ page }) => {
  await openTwoPeople(page);
  await box(page, 0).click();
  await expect(box(page, 0)).toHaveAttribute('data-face-role', 'target');
  await expect(page.locator('[data-face-role-label="0"]')).toHaveText('Target');
  await expect(page.locator('[data-face-exclude="0"]')).toHaveCount(0);
  // The other captured person is now a bystander (Selected face swaps only the pinned one).
  await expect(box(page, 1)).toHaveAttribute('data-face-role', 'bystander');
});

test('exclude a bystander: an explicit never-swap entry is written, and Include again removes it', async ({ page }) => {
  const writes = await openTwoPeople(page);
  await box(page, 0).click();                                    // pin person 0
  await expect(box(page, 0)).toHaveAttribute('data-face-role', 'target');

  const exclude = page.locator('[data-face-exclude="1"]');
  await expect(exclude).toHaveText('Exclude (never swap)');
  const before = writes.length;
  await exclude.click();

  await expect(page.locator('[data-face-role-label="1"]')).toHaveText('Excluded · never swapped');
  await expect(exclude).toHaveText('Include again');
  await expect(exclude).toHaveAttribute('aria-pressed', 'true');

  await expect.poll(() => writes.length, { timeout: 5_000 }).toBeGreaterThan(before);
  const excluded = lastMapping(writes);
  expect(excluded, 'the write must carry a mapping object').not.toBeNull();
  expect(Object.prototype.hasOwnProperty.call(excluded, P1), 'an EXPLICIT entry, not a deleted key').toBe(true);
  expect(excluded[P1], 'never-swap is null').toBeNull();
  expect(excluded[P0] ?? 'absent').not.toBeNull();               // the pinned person is untouched

  const mid = writes.length;
  await exclude.click();                                          // include again
  await expect(page.locator('[data-face-role-label="1"]')).toHaveText('Bystander · not swapped');
  await expect(exclude).toHaveText('Exclude (never swap)');
  await expect.poll(() => writes.length, { timeout: 5_000 }).toBeGreaterThan(mid);
  const included = lastMapping(writes);
  expect(Object.prototype.hasOwnProperty.call(included, P1), 'the entry is gone').toBe(false);
});

test('clicking Exclude does not also pin the box (the badge stops the click)', async ({ page }) => {
  await openTwoPeople(page);
  await box(page, 0).click();
  await page.locator('[data-face-exclude="1"]').click();
  await expect(box(page, 0)).toHaveAttribute('data-face-role', 'target');
  await expect(box(page, 1)).not.toHaveAttribute('data-face-role', 'target');
});
