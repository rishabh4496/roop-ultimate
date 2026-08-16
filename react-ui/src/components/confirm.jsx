import React, { useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence, spring } from '../motion';
import { Icon } from '../icons';

// Promise-based, themed replacements for window.confirm / window.prompt so
// destructive actions don't pop a jarring native OS dialog in the premium UI.
// Any component can `await confirmDialog({...})` without prop-drilling — a single
// <ConfirmHost/> mounted at the app root renders the modal. If the host isn't
// mounted yet (shouldn't happen), we fall back to the native dialog so callers
// never hang.

let _emit = null;

export function confirmDialog(opts = {}) {
  return new Promise((resolve) => {
    if (!_emit) { resolve(window.confirm(opts.message || '')); return; }
    _emit({ ...opts, kind: 'confirm', resolve });
  });
}

export function promptDialog(opts = {}) {
  return new Promise((resolve) => {
    if (!_emit) { resolve(window.prompt(opts.message || '', opts.defaultValue || '')); return; }
    _emit({ ...opts, kind: 'prompt', resolve });
  });
}

export function ConfirmHost() {
  const [dialog, setDialog] = useState(null);
  const [text, setText] = useState('');
  const inputRef = useRef(null);

  useEffect(() => {
    _emit = (d) => { setText(d.defaultValue || ''); setDialog(d); };
    return () => { _emit = null; };
  }, []);

  const close = useCallback((value) => {
    if (dialog) dialog.resolve(value);
    setDialog(null);
  }, [dialog]);
  const onConfirm = useCallback(
    () => close(dialog?.kind === 'prompt' ? text : true),
    [close, dialog?.kind, text],
  );
  const onCancel = useCallback(
    () => close(dialog?.kind === 'prompt' ? null : false),
    [close, dialog?.kind],
  );

  // Focus the input (prompt) for immediate typing; Esc cancels, Enter confirms.
  useEffect(() => {
    if (dialog?.kind !== 'prompt') return undefined;
    const timer = setTimeout(() => inputRef.current?.select(), 30);
    return () => clearTimeout(timer);
  }, [dialog]);
  useEffect(() => {
    if (!dialog) return;
    const onKey = (e) => {
      if (e.key === 'Escape') { e.preventDefault(); onCancel(); }
      else if (e.key === 'Enter' && dialog.kind !== 'prompt') { e.preventDefault(); onConfirm(); }
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [dialog, onCancel, onConfirm]);

  const danger = dialog?.danger;

  return createPortal(
    <AnimatePresence>
      {dialog && (
        <motion.div
          className="fixed inset-0 z-[90] flex items-center justify-center p-4"
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
          onMouseDown={(e) => { if (e.target === e.currentTarget) onCancel(); }}
          role="dialog" aria-modal="true"
          aria-label={dialog.title || 'Confirm'}
        >
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />
          <motion.div
            initial={{ scale: 0.92, y: 16, opacity: 0 }}
            animate={{ scale: 1, y: 0, opacity: 1 }}
            exit={{ scale: 0.95, y: 8, opacity: 0 }}
            transition={spring.snappy}
            className="relative w-full max-w-sm rounded-2xl glass-panel border border-white/10 p-6 shadow-2xl"
          >
            <h2 className="text-lead font-bold text-white/95 flex items-center gap-2">
              {danger
                ? <Icon.warning size={17} className="text-red-400" />
                : <Icon.question size={17} className="text-white/50" />}
              {dialog.title || 'Are you sure?'}
            </h2>
            {dialog.message && (
              <p className="mt-2 text-compact leading-relaxed text-white/60 selectable">{dialog.message}</p>
            )}
            {dialog.kind === 'prompt' && (
              <input
                ref={inputRef}
                value={text}
                onChange={(e) => setText(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); onConfirm(); } }}
                placeholder={dialog.placeholder || ''}
                className="mt-4 w-full px-3 py-2.5 rounded-xl glass-input text-white text-compact focus:outline-none placeholder:text-white/25 selectable"
              />
            )}
            <div className="mt-6 flex justify-end gap-2.5">
              <button
                type="button" onClick={onCancel}
                className="px-4 py-2 rounded-xl text-compact font-semibold bg-white/[0.05] hover:bg-white/[0.09] text-white/70 hover:text-white border border-white/10 transition-colors"
              >{dialog.cancelLabel || 'Cancel'}</button>
              <button
                type="button" onClick={onConfirm} autoFocus={dialog.kind !== 'prompt'}
                className={`px-4 py-2 rounded-xl text-compact font-semibold border transition-colors ${
                  danger
                    ? 'bg-red-500/15 hover:bg-red-500/25 text-red-300 border-red-500/30'
                    : 'bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white border-white/10'}`}
              >{dialog.confirmLabel || 'Confirm'}</button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>,
    document.body
  );
}
