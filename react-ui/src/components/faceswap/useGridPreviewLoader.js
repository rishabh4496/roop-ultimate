import { useEffect } from 'react';
import { postJSON } from '../../api';
import { dataUrlToOwnedBlobUrl, revokeUrl } from './objectUrls';

// ── One comparison grid's preview loader ──────────────────────────────────
// Renders one preview per selected variant, holding every OTHER setting fixed,
// so the grid isolates the effect of the one thing being compared.
//
// There were three of these — enhancers, mask engines, swapper models — and
// they were the same seventy lines three times over. Their own comments said
// so ("Identical shape to loadMaskPreviews/loadEnhancerPreviews"). The only
// real differences were which settings key gets varied, which list of values is
// legal, and which set of state setters to write into.
//
// The AI-upscale grid is deliberately NOT built on this. It is a genuinely
// different shape: it swaps the frame ONCE and then upscales that single result
// with each model (one swap + N upscales, against a different endpoint, with no
// per-cell preview cache), rather than re-running the whole swap per cell.
// Forcing it through here would mean a parameter that switches the body wholesale.
//
// ── Cancellation ──────────────────────────────────────────────────────────
// Cells render one at a time, in sequence, and each is a full swap — so a grid
// of four can be in flight for many seconds. `activeCheck` is consulted before
// and after every await: when the effect is torn down (grid closed, frame
// scrubbed, a setting changed) the in-flight cell finishes but its result is
// dropped rather than written into the state of a grid that has moved on.

const TIMER_TICK_MS = 100;

export default function useGridPreviewLoader({
  enabled,            // the grid's `comparing` flag
  selection,          // the chosen variants, in display order
  allowed,            // values the backend actually supports (undefined = allow all)
  paramKey,           // the settings key this grid varies
  setPreviews, setTimes, setTimers, intervalsRef,   // from useCompareGrid
  settings,           // current params; each cell overrides `paramKey` on a copy
  fakePreview,
  selTarget, frame, targetCount,
  buildPreviewPayload, previewSignature, previewCacheRef,
  cacheSuffix,        // the source/target/selection part of the cache key
  reloadKey,          // previewKey — any preview-relevant setting change
  owner,              // object-URL owner token; shared with the main preview cache
}) {
  /* eslint-disable react-hooks/exhaustive-deps -- intentional: `load` is
     rebuilt every render and closes over current values on purpose; the effect
     is keyed on the things that should actually re-render the grid. Depending
     on the closure itself would re-run every render and never settle. */
  useEffect(() => {
    if (!enabled || targetCount === 0) return undefined;
    let active = true;
    const activeCheck = () => active;

    const load = async () => {
      const available = allowed ? selection.filter((v) => allowed.includes(v)) : selection;

      // Drop cells for variants no longer selected, keep the ones still shown —
      // deselecting one must not blank the others back to a spinner.
      const keepOnly = (prev) => {
        const reset = {};
        for (const v of available) if (prev[v]) reset[v] = prev[v];
        return reset;
      };
      setPreviews(keepOnly);
      setTimes(keepOnly);
      setTimers(keepOnly);

      for (const value of available) {
        if (!activeCheck()) return;
        const localParams = { ...settings, [paramKey]: value };
        const cacheKey = `${selTarget}_${frame}_${previewSignature(localParams, fakePreview)}_${cacheSuffix}`;

        if (previewCacheRef.current[cacheKey]) {
          if (!activeCheck()) return;
          setPreviews((prev) => ({ ...prev, [value]: previewCacheRef.current[cacheKey].image }));
          setTimes((prev) => ({ ...prev, [value]: 'Cached' }));
          continue;
        }

        const stopTimer = () => {
          if (intervalsRef.current[value]) {
            clearInterval(intervalsRef.current[value]);
            delete intervalsRef.current[value];
          }
        };

        try {
          const start = Date.now();
          setTimers((prev) => ({ ...prev, [value]: '0.0s' }));
          intervalsRef.current[value] = setInterval(() => {
            setTimers((prev) => ({ ...prev, [value]: `${((Date.now() - start) / 1000).toFixed(1)}s` }));
          }, TIMER_TICK_MS);

          const res = await postJSON('/api/preview', buildPreviewPayload(localParams, {
            index: selTarget, frame, fake: fakePreview,
          }));
          const duration = ((Date.now() - start) / 1000).toFixed(2);
          stopTimer();
          if (!activeCheck()) return;
          if (res.image) {
            // Same reason as the main preview: a grid of six cells held six
            // full-resolution base64 strings in React state AND six more in the
            // cache. As blob URLs the bytes exist once each and the state holds
            // tokens. Owned by the panel's token, so the panel's teardown frees
            // the grid's frames along with everything else.
            const blobSrc = dataUrlToOwnedBlobUrl(res.image, owner);
            setPreviews((prev) => {
              // Replacing a cell's url without revoking the outgoing one leaks
              // a frame per re-render of the grid.
              const old = prev[value];
              if (old && old !== blobSrc && previewCacheRef.current[cacheKey]?.image !== old) {
                revokeUrl(old);
              }
              return { ...prev, [value]: blobSrc };
            });
            setTimes((prev) => ({ ...prev, [value]: `${duration}s` }));
            setTimers((prev) => ({ ...prev, [value]: null }));
            previewCacheRef.current[cacheKey] = { faces: res.faces || [], image: blobSrc };
          }
        } catch {
          stopTimer();
          setTimers((prev) => ({ ...prev, [value]: null }));
          // Fail silently, per cell. One variant failing is normal and must not
          // take the grid down: a swapper model can fail to download on first
          // use, and SAM2-tracked masking needs a video pre-pass so it may skip
          // a single frame.
        }
      }
    };

    load();
    return () => {
      active = false;
      // Timers outlive the request they were measuring if the grid is torn down
      // mid-cell, so clear the whole set rather than the current one.
      if (intervalsRef.current) {
        Object.values(intervalsRef.current).forEach(clearInterval);
        intervalsRef.current = {};
      }
    };
  }, [enabled, selection, frame, selTarget, targetCount, cacheSuffix, reloadKey]);
  /* eslint-enable react-hooks/exhaustive-deps */
}
