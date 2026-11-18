// The four comparison modes that `C` cycles through, and the viewport sub-mode each one drives.
//
// Pure, so the render-check can pin the order without mounting the Face Swap screen. FaceSwap owns the
// compare / side-by-side state; InteractivePreview owns the sub-mode (its compareMode). The step's `sub` is sent
// to the preview as a `roop:viewport` event, and the palette uses the same vocabulary.
//
//   wipe        the split-wipe slider over the before / after picture
//   crossfade   the blend between the two pictures (InteractivePreview's 'blend')
//   diff        the absolute difference between the two pictures
//   side        the two pictures side by side (the split view, on the slider sub-mode)

export const COMPARE_STEPS = Object.freeze([
  Object.freeze({ id: 'wipe', label: 'Split wipe', sub: 'slider', split: false }),
  Object.freeze({ id: 'crossfade', label: 'Crossfade', sub: 'blend', split: false }),
  Object.freeze({ id: 'diff', label: 'Diff', sub: 'diff', split: false }),
  Object.freeze({ id: 'side', label: 'Side-by-side', sub: 'slider', split: true }),
]);

/**
 * The step `C` moves to. `current` is the index of the step last applied, or -1 when none is known.
 * Starting from "off" (or an unknown step) lands on the first step, so the first press always shows something.
 */
export function nextCompareIndex(current, compareOn) {
  if (!compareOn || !Number.isInteger(current) || current < 0) return 0;
  return (current + 1) % COMPARE_STEPS.length;
}

/** The label for the step at `index`, for the transient read-out. */
export const compareLabel = (index) => COMPARE_STEPS[index]?.label ?? '';
