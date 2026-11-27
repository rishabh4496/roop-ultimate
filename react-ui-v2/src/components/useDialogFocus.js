import { useEffect, useRef } from 'react';

// Keyboard behaviour every modal overlay owes the user, in one place:
//   * Escape closes it (when `onClose` is given),
//   * focus moves into it when it opens (the first [autofocus] / focusable element),
//   * Tab and Shift+Tab cycle inside it instead of walking the page behind the backdrop,
//   * focus goes back to whatever opened it when it closes.
// The command palette, Preset Studio, Quality Profiles and the Batch health check had none of
// this: Tab walked the page behind the backdrop and, for three of them, Escape did nothing.
//
// Usage: const ref = useDialogFocus(open, onClose);  <div ref={ref} role="dialog" aria-modal="true" ...>
const FOCUSABLE = [
  'a[href]', 'button:not([disabled])', 'input:not([disabled]):not([type="hidden"])',
  'select:not([disabled])', 'textarea:not([disabled])', '[tabindex]:not([tabindex="-1"])',
].join(',');

const focusables = (root) => [...root.querySelectorAll(FOCUSABLE)]
  .filter((el) => el.getAttribute('tabindex') !== '-1' && el.getClientRects().length > 0);

export default function useDialogFocus(open = true, onClose) {
  const ref = useRef(null);
  const closeRef = useRef(onClose);
  closeRef.current = onClose;

  useEffect(() => {
    if (!open) return undefined;
    const opener = document.activeElement;
    const root = ref.current;
    // After paint, so a child that focuses itself (the palette's input) wins.
    const t = setTimeout(() => {
      if (!root || root.contains(document.activeElement)) return;
      const first = root.querySelector('[autofocus]') || focusables(root)[0];
      (first || root).focus?.({ preventScroll: true });
    }, 30);

    const onKey = (e) => {
      if (!root) return;
      // A dialog stacked on top (a confirm() from inside this one) owns the keyboard.
      const top = document.activeElement?.closest?.('[role="dialog"], [role="alertdialog"]');
      if (top && top !== root && !root.contains(top)) return;
      if (e.key === 'Escape' && closeRef.current) {
        e.preventDefault();
        e.stopPropagation();
        closeRef.current();
        return;
      }
      if (e.key !== 'Tab') return;
      // Every Tab is moved by hand. Leaving the middle of the cycle to the browser let focus out:
      // Chrome makes a scrollable box a tab stop of its own, which is not in this list, and the
      // next Tab from it went to <body>.
      e.preventDefault();
      const items = focusables(root);
      if (!items.length) { root.focus?.(); return; }
      const idx = items.indexOf(document.activeElement);
      const next = idx === -1
        ? (e.shiftKey ? items.length - 1 : 0)
        : (idx + (e.shiftKey ? -1 : 1) + items.length) % items.length;
      items[next].focus();
    };
    // Capture phase, so the app's global shortcuts (which also listen for Escape) do not act first.
    window.addEventListener('keydown', onKey, true);
    return () => {
      clearTimeout(t);
      window.removeEventListener('keydown', onKey, true);
      if (opener && typeof opener.focus === 'function' && document.contains(opener)) {
        opener.focus({ preventScroll: true });
      }
    };
  }, [open]);

  return ref;
}
