import { useEffect, useRef, useState } from 'react';

// ── One comparison grid's state ───────────────────────────────────────────
// The Face Swap tab can put up to four variants of the same frame side by side
// for four different things: enhancers, mask engines, swapper models and AI
// upscalers. Each of those carried its own copy of the identical eight
// declarations — a `comparing` flag, the chosen model list with its
// localStorage round-trip and validation, a previews map, a times map, a
// per-cell render timer map, and a ref of live intervals.
//
// Four copies of one idea is four places to fix a bug in and four chances for
// them to drift, which had already started: the enhancer grid's timer state is
// still named `liveRenderingTimers` rather than following the other three.
//
// The differences between them turned out to be exactly two — the storage key
// and what counts as a valid selection — so both are parameters and everything
// else is shared.
//
// Returns generic names (`comparing`, `selected`, `previews`…). Call sites
// rename them on destructure to whatever they already called them, which is
// what keeps this a pure move rather than a rewrite of the thousands of lines
// downstream that use those names.

const isString = (x) => typeof x === 'string';

/**
 * @param {string}   storageKey  localStorage key holding the chosen list.
 * @param {string[]} defaults    used when nothing valid is stored.
 * @param {(x:any)=>boolean} [isValid]  per-item check; defaults to "a string".
 */
export default function useCompareGrid({ storageKey, defaults, isValid = isString }) {
  const [comparing, setComparing] = useState(false);

  const [selected, setSelected] = useState(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(storageKey) || 'null');
      // 1..4 cells: the grid lays out at most four, and an empty list would
      // render a comparison with nothing in it.
      if (Array.isArray(saved) && saved.length >= 1 && saved.length <= 4 && saved.every(isValid)) {
        return saved;
      }
    } catch { /* fall through to default */ }
    return defaults;
  });

  useEffect(() => {
    localStorage.setItem(storageKey, JSON.stringify(selected));
  }, [storageKey, selected]);

  // Rendered cell images, their measured cost, and the ticking "rendering for
  // N s" counters — plus the interval handles those counters run on, which are
  // a ref because they are cleared imperatively when a cell resolves.
  const [previews, setPreviews] = useState({});
  const [times, setTimes] = useState({});
  const [timers, setTimers] = useState({});
  const intervalsRef = useRef({});

  return {
    comparing, setComparing,
    selected, setSelected,
    previews, setPreviews,
    times, setTimes,
    timers, setTimers,
    intervalsRef,
  };
}
