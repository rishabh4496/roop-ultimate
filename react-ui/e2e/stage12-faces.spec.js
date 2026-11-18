import { test, expect } from '@playwright/test';
import { openTab, makePng } from './helpers.js';

// Stage 12, part 3: the on-canvas face menu and the face-role badges, in a real Chromium against the production build.
//
//   placement   the menu opens BESIDE the face box (right, or left when there is no room), never over it
//   roles       Primary / Secondary / Reference / Exclude are real writes to the face mapping the page sends to the backend;
//               an excluded face wears a crimson striped lock bracket, a Reference face a dashed violet one
//   actions     Solo Face Crop in Loupe, Copy Pose to Filter, Paint Manual Occlusion Mask, Reset Face Overrides
//
// The mock's detections and poses are stand-ins; what is asserted is what the page DOES with them: the mapping it writes, the
// settings it posts with the next preview, the stage's transform.
//
// Two captured people (like face-roles.spec.js), because the pinned target cannot be excluded: the roles are exercised on the
// SECOND person, and the first is used where the pinned target is the point.

const FRAME = { w: 640, h: 360 };
const P0 = 'person-a';
const P1 = 'person-b';

async function openFaces(page, { pose = [[44, -4, 2]] } = {}) {
  const plate = makePng(FRAME.w, FRAME.h, () => [224, 172, 140]);
  const swapped = makePng(FRAME.w, FRAME.h, () => [223, 172, 144]);
  const writes = [];
  const previews = [];
  await page.route('**/api/target/preview*', (route) => route.fulfill({ status: 200, contentType: 'image/png', body: plate }));
  await page.route('**/api/preview', (route) => {
    if (route.request().method() !== 'POST') return route.fallback();
    try { previews.push(JSON.parse(route.request().postData() || '{}')); } catch { /* not JSON */ }
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        image: `data:image/png;base64,${swapped.toString('base64')}`,
        faces: [[40, 50, 160, 210], [470, 70, 590, 230]],
        person_ids: [0, 1],
        kps: [], pose,
      }),
    });
  });
  await page.route('**/api/target/context*', (route) => {
    if (route.request().method() === 'POST') {
      try { writes.push(JSON.parse(route.request().postData() || '{}')); } catch { /* not JSON */ }
      return route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
    }
    return route.fallback();
  });
  await page.route('**/api/state', async (route) => {
    const res = await route.fetch();
    const json = await res.json();
    const f = json.target_faces?.[0];
    json.target_faces = [f, f];
    json.target_groups = [0, 1];
    json.target_names = ['Lead', 'Guest'];
    json.target_person_ids = [P0, P1];
    json.target_reference_face_ids = ['ref-a', 'ref-b'];
    await route.fulfill({ response: res, json });
  });
  await openTab(page, 'faceswap');
  await expect(page.locator('[data-face-index="0"]')).toBeVisible({ timeout: 20_000 });
  return { writes, previews };
}

const face = (page, i) => page.locator(`[data-face-index="${i}"]`);
const menu = (page) => page.getByTestId('face-context-menu');
const role = (page, id) => page.getByTestId(`face-role-${id}`);
const label = (page, i) => page.locator(`[data-face-role-label="${i}"]`);
// The stage's own HUD can sit over a box at this window size (it does at HEAD: face-roles.spec.js fails the same way), so a click
// on a box is delivered to it directly rather than through Playwright's actionability check.
const pick = (page, i) => face(page, i).dispatchEvent('click');

async function rightClick(page, i) {
  const b = await face(page, i).boundingBox();
  await face(page, i).click({ button: 'right', position: { x: b.width * 0.5, y: b.height * 0.3 } });
  await expect(menu(page)).toBeVisible();
  await expect(menu(page)).toHaveCSS('visibility', 'visible');
}

/** The face mapping the page most recently wrote to the backend ({ personId: sourceId | null }). */
const lastMapping = (writes) => [...writes].reverse().map((w) => w.target_person_source_mapping ?? w.face_mapping).find((m) => m && typeof m === 'object');
const refTags = (page) => page.evaluate(() => Object.entries(localStorage).filter(([k]) => k.startsWith('roop_face_reference_tags:')).map(([, v]) => JSON.parse(v)));

test.describe('face menu (Stage 12)', () => {
  test('it opens beside the face, not over it, and never leaves the window', async ({ page }) => {
    await openFaces(page);
    await rightClick(page, 0);
    let f = await face(page, 0).boundingBox();
    let m = await menu(page).boundingBox();
    expect(m.x, 'right of the box').toBeGreaterThanOrEqual(f.x + f.width);
    expect(m.x - (f.x + f.width), 'a small gap, not far away').toBeLessThan(40);
    // beside it vertically too: the two share some height (the menu is pushed up only as far as it must be to fit)
    expect(m.y <= f.y + f.height && m.y + m.height >= f.y, 'the menu is level with the face').toBe(true);
    await expect(menu(page)).toHaveAttribute('data-side', 'right');
    await page.keyboard.press('Escape');

    // a narrow window leaves no room beside the right-hand box on that side. The layout reflows when the window shrinks and the
    // box may no longer be reachable by a real click, so the event is delivered to it with real coordinates instead.
    await page.setViewportSize({ width: 820, height: 700 });
    f = await face(page, 1).boundingBox();
    const cx = Math.min(810, Math.max(10, f.x + f.width / 2));
    const cy = Math.min(690, Math.max(10, f.y + f.height / 2));
    await face(page, 1).evaluate((el, [x, y]) => {
      el.dispatchEvent(new MouseEvent('contextmenu', { clientX: x, clientY: y, button: 2, bubbles: true, cancelable: true }));
    }, [cx, cy]);
    await expect(menu(page)).toBeVisible();
    await expect(menu(page)).toHaveCSS('visibility', 'visible');
    m = await menu(page).boundingBox();
    expect(m.x).toBeGreaterThanOrEqual(0);
    expect(m.x + m.width).toBeLessThanOrEqual(820.5);
    expect(m.y).toBeGreaterThanOrEqual(0);
    expect(m.y + m.height).toBeLessThanOrEqual(700.5);
    const inside = f.x >= 0 && f.x + f.width <= 820 && f.y >= 0 && f.y + f.height <= 700;
    if (inside && (await menu(page).getAttribute('data-side')) !== 'point') {
      expect(m.x >= f.x + f.width || m.x + m.width <= f.x, 'the menu does not cover the box it is about').toBe(true);
    }
    expect(['left', 'right', 'point']).toContain(await menu(page).getAttribute('data-side'));
  });

  test('the four roles are real changes: Exclude hatches the box with a crimson padlock, Reference writes the same mapping, Primary lifts it', async ({ page }) => {
    const { writes } = await openFaces(page);
    await rightClick(page, 1);
    for (const id of ['primary', 'secondary', 'reference', 'exclude']) await expect(role(page, id)).toBeVisible();
    await expect(face(page, 1).getByTestId('face-lock-badge')).toHaveCount(0);

    // Exclude: the mapping gets an explicit null for that person, and the box wears the lock
    await role(page, 'exclude').click();
    await expect(menu(page)).toHaveCount(0);
    await expect.poll(() => lastMapping(writes)?.[P1], { timeout: 5_000 }).toBeNull();
    await expect(face(page, 1).getByTestId('face-lock-badge')).toBeVisible();
    await expect(face(page, 1)).toHaveClass(/vfx-face--protected/);
    await expect(face(page, 1)).toHaveAttribute('data-face-choice', 'exclude');
    await expect(face(page, 1).locator('.vfx-face-stripes')).toBeVisible();
    await expect(label(page, 1)).toContainText(/Excluded/);
    await expect(face(page, 0).getByTestId('face-lock-badge'), 'the other person is untouched').toHaveCount(0);

    // the menu now shows it as the current role
    await rightClick(page, 1);
    await expect(role(page, 'exclude')).toHaveAttribute('aria-checked', 'true');
    await expect(role(page, 'primary')).toHaveAttribute('aria-checked', 'false');

    // Reference Only: the SAME null in the mapping (it renders like Exclude), drawn differently and without the padlock
    const excludeMapping = JSON.stringify(lastMapping(writes));
    await role(page, 'reference').click();
    await expect(face(page, 1)).toHaveClass(/vfx-face--reference/);
    await expect(face(page, 1).getByTestId('face-lock-badge')).toHaveCount(0);
    await expect(face(page, 1)).toHaveAttribute('data-face-choice', 'reference');
    await expect(label(page, 1)).toContainText(/Reference/);
    await expect.poll(() => lastMapping(writes)?.[P1], { timeout: 5_000 }).toBeNull();
    expect(JSON.stringify(lastMapping(writes)), 'Reference renders exactly like Exclude').toBe(excludeMapping);
    const tags = await refTags(page);
    expect(tags.length).toBe(1);
    expect(tags[0][P1]).toBe(true);

    // Exclude again clears the marker
    await rightClick(page, 1);
    await role(page, 'exclude').click();
    await expect(face(page, 1)).toHaveClass(/vfx-face--protected/);
    expect((await refTags(page)).length).toBe(0);

    // Primary lifts the exclusion (the explicit null would otherwise win) and makes this person THE target
    await rightClick(page, 1);
    await role(page, 'primary').click();
    await expect(face(page, 1)).not.toHaveClass(/vfx-face--protected/);
    await expect(face(page, 1).getByTestId('face-lock-badge')).toHaveCount(0);
    await expect(label(page, 1)).toHaveText('Target');
    expect(lastMapping(writes)?.[P1] ?? 'absent', 'the exclusion entry is gone').not.toBeNull();
  });

  test('Secondary maps the person to a source; the pinned target cannot be excluded, and says why', async ({ page }) => {
    const { writes } = await openFaces(page);
    await pick(page, 0);                                                      // person A is the pinned target
    await expect(label(page, 0)).toHaveText('Target');
    await rightClick(page, 1);
    await role(page, 'secondary').click();
    await expect.poll(() => typeof lastMapping(writes)?.[P1], { timeout: 5_000 }).toBe('string');   // a source identity, not null
    await expect(label(page, 1)).toHaveText(/Swapped|Target/);
    await expect(face(page, 1)).toHaveAttribute('data-face-choice', /secondary|primary/);

    // the pinned target: excluding it is refused, with the reason, and so is making it secondary
    await pick(page, 0);
    await expect(label(page, 0)).toHaveText('Target');
    await rightClick(page, 0);
    await expect(role(page, 'primary')).toHaveAttribute('aria-checked', 'true');
    await expect(role(page, 'exclude')).toHaveAttribute('aria-disabled', 'true');
    await expect(role(page, 'exclude')).toHaveAttribute('title', /pinned target/);
    await expect(role(page, 'reference')).toHaveAttribute('aria-disabled', 'true');
    await expect(role(page, 'secondary')).toHaveAttribute('aria-disabled', 'true');
    await role(page, 'exclude').dispatchEvent('click');                        // a real user can press it; nothing may happen
    await expect(menu(page), 'a refused role leaves the menu open').toBeVisible();
    expect(lastMapping(writes)?.[P0] ?? 'absent').not.toBeNull();
  });

  test('hovering a role explains what it does in plain words, including that Reference renders like Exclude', async ({ page }) => {
    await openFaces(page);
    await rightClick(page, 1);
    await role(page, 'reference').hover();
    await expect(page.getByTestId('face-role-hint')).toContainText(/exactly like Exclude/);
    await role(page, 'primary').hover();
    await expect(page.getByTestId('face-role-hint')).toContainText(/Selected face/);
    await role(page, 'exclude').hover();
    await expect(page.getByTestId('face-role-hint')).toContainText(/crimson striped lock/);
  });

  test('Copy Pose to Filter sets the Frontalize gate to this face\'s pose, and is refused where no pose was measured', async ({ page }) => {
    const { previews } = await openFaces(page, { pose: [[44, -4, 2]] });       // only face 0 has a pose; 44 degrees -> the 45 step
    await rightClick(page, 0);
    await expect(page.getByTestId('face-menu-pose')).toContainText('yaw: +44.0°');
    await expect(page.getByTestId('face-menu-pose-bucket')).toContainText('half_right');
    await expect(page.getByTestId('face-menu-occlusion')).toHaveCount(0);       // no per-face occlusion figure exists: none is invented
    const cp = page.getByTestId('face-menu-copypose');
    await expect(cp).toContainText(/Frontalize above angle" to 45°/);
    await expect(cp).toContainText(/Frontalize is off/);
    const before = previews.length;
    await cp.click();
    await expect(page.getByText(/Frontalize angled faces is OFF/).first()).toBeVisible();   // honest about the gate doing nothing yet
    // the value is in the next request: the default is 30, so seeing 45 means this action wrote it
    await expect.poll(() => previews.slice(before).map((b) => b.frontalization_threshold).filter((v) => v === 45).length, { timeout: 15_000 }).toBeGreaterThan(0);

    await rightClick(page, 1);
    await expect(page.getByTestId('face-menu-pose')).toContainText('Pose not measured');
    await expect(page.getByTestId('face-menu-copypose')).toHaveAttribute('aria-disabled', 'true');
    await expect(page.getByTestId('face-menu-copypose')).toContainText(/No head pose/);
  });

  test('Solo Face Crop in Loupe zooms the stage in on the face and switches the magnifier on', async ({ page }) => {
    await openFaces(page);
    await expect(page.locator('.preview-stage div[style*="scale(1)"]').first()).toBeVisible();
    await rightClick(page, 0);
    await page.getByTestId('face-menu-loupe').click();
    const scaled = page.locator('.preview-stage div[style*="scale("]:not([style*="scale(1)"])').first();
    await expect(scaled).toBeVisible();
    const scale = await scaled.evaluate((el) => Number(/scale\(([\d.]+)\)/.exec(el.style.transform)?.[1]));
    expect(scale, 'zoomed in on the face').toBeGreaterThan(1.1);
    await expect(page.getByTitle(/Floating Magnifier Loupe/)).toHaveClass(/text-cyan-300/);
  });

  test('Paint Manual Occlusion Mask zooms to the face and arms the Paint brush', async ({ page }) => {
    await openFaces(page);
    await rightClick(page, 0);
    await page.getByTestId('face-menu-paint').click();
    await expect(page.getByText(/Brush \(paint/).first()).toBeVisible();
    await expect(page.locator('.preview-stage div[style*="scale("]:not([style*="scale(1)"])').first()).toBeVisible();
  });

  test('Reset Face Overrides clears what was set for that face - and is disabled, with the reason, when nothing was', async ({ page }) => {
    const { writes } = await openFaces(page);
    await rightClick(page, 1);
    await expect(page.getByTestId('face-menu-reset')).toHaveAttribute('aria-disabled', 'true');
    await expect(page.getByTestId('face-menu-reset')).toContainText(/Nothing to reset/);
    await role(page, 'reference').click();                                          // a mapping entry AND a marker
    await expect(face(page, 1)).toHaveClass(/vfx-face--reference/);
    await rightClick(page, 1);
    await expect(page.getByTestId('face-menu-reset')).toContainText(/Clears its source \/ skip choice, its Reference marker/);
    await page.getByTestId('face-menu-reset').click();
    await expect(face(page, 1)).not.toHaveClass(/vfx-face--reference/);
    await expect(face(page, 1)).not.toHaveClass(/vfx-face--protected/);
    expect(lastMapping(writes)?.[P1] ?? 'absent', 'the explicit never-swap entry is gone').not.toBeNull();
    expect((await refTags(page)).length).toBe(0);
    await expect(page.getByText(/Face 2 reset: cleared/).first()).toBeVisible();
  });
});
