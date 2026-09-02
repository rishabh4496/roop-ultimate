// The last still the Face Swap tab had on screen.
//
// The processing view lives in its own tab now, and tabs are mutually
// exclusive — so while the run is being watched, the Face Swap component that
// owns `previewSrc` is unmounted. LiveProcessingPeek still wants that still as
// its fallback for the window before the first live frame is published, which
// is precisely the start of a run.
//
// Module scope rather than localStorage on purpose: this only has to survive a
// tab switch inside one document, not a reload.
//
// `previewSrc` MUST be self-contained bytes (a data URL), not the blob URL the
// Face Swap panel displays. The whole point of this store is to be read AFTER
// that panel unmounts — and unmounting is exactly when the panel revokes the
// blob URLs it owns. Parking the reference here instead of the bytes would give
// the processing tab a url that is guaranteed dead by the time it reads it, and
// the symptom would be a broken-image icon at the start of every run: precisely
// the window this exists to cover.
export const lastPreview = { previewSrc: '', rawUrl: '', frame: 1, maxFrames: 1 };

export function setLastPreview(patch) {
  Object.assign(lastPreview, patch);
}
