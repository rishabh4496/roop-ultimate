// The last still the Face Swap tab had on screen.
//
// The processing view lives in its own tab now, and tabs are mutually
// exclusive — so while the run is being watched, the Face Swap component that
// owns `previewSrc` is unmounted. LiveProcessingPeek still wants that still as
// its fallback for the window before the first live frame is published, which
// is precisely the start of a run.
//
// Module scope rather than localStorage on purpose: `previewSrc` is a
// multi-megabyte data URL, and this only has to survive a tab switch inside one
// document, not a reload.
export const lastPreview = { previewSrc: '', rawUrl: '', frame: 1, maxFrames: 1 };

export function setLastPreview(patch) {
  Object.assign(lastPreview, patch);
}
