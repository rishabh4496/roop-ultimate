import React, { useRef } from 'react';

/**
 * Pro-Workstation Source Identity Gallery for React UI 2.0.
 * Displays source face thumbnails, embedding confidence, quick-pinning,
 * re-ordering, and multi-face selection.
 */
export function SourceGallery({
  sourceFaces = [],
  selectedSourceIndex = 0,
  onSelectSource,
  onRemoveSource,
  onMoveSource,
  onClearSources,
  onAddFiles,
  onOpenLibrary,
  pinnedIdentities = [],
  onTogglePin,
  busy = false,
  className = '',
}) {
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0 && onAddFiles) {
      onAddFiles(files);
    }
    e.target.value = '';
  };

  return (
    <div className={`flex flex-col gap-2.5 p-3 rounded-xl bg-[#0b0d13] border border-white/10 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <span className="text-[11px] font-mono font-bold uppercase tracking-wider text-[var(--accent-primary,#38bdf8)]">
            Source Identities
          </span>
          <span className="px-1.5 py-0.5 rounded-full bg-white/10 text-[10px] font-mono text-white/70">
            {sourceFaces.length}
          </span>
        </div>

        <div className="flex items-center gap-1">
          {onOpenLibrary && (
            <button
              type="button"
              onClick={onOpenLibrary}
              title="Open Faceset Library (.fsz)"
              className="px-2 py-0.5 rounded bg-white/5 hover:bg-white/10 border border-white/10 text-[10px] font-medium text-white/80 transition-colors"
            >
              Library .fsz
            </button>
          )}

          {sourceFaces.length > 0 && onClearSources && (
            <button
              type="button"
              onClick={onClearSources}
              title="Clear all source faces"
              disabled={busy}
              className="px-1.5 py-0.5 rounded hover:bg-red-500/20 text-white/40 hover:text-red-400 text-[10px] font-medium transition-colors"
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Quick-Pin Identity Strip */}
      {pinnedIdentities.length > 0 && (
        <div className="flex items-center gap-1 overflow-x-auto pb-1 border-b border-white/5">
          <span className="text-[9px] font-mono uppercase text-white/40 shrink-0">Pinned:</span>
          {pinnedIdentities.map((pin, i) => (
            <button
              key={`pin-${pin.id || i}`}
              type="button"
              onClick={() => onSelectSource?.(pin.sourceIndex)}
              className="px-1.5 py-0.5 rounded bg-white/5 hover:bg-white/10 border border-amber-400/30 text-[10px] font-mono text-amber-300 flex items-center gap-1 shrink-0"
            >
              <span>★</span>
              <span>{pin.name || `Face #${pin.sourceIndex + 1}`}</span>
            </button>
          ))}
        </div>
      )}

      {/* Source Thumbnails Strip / Grid */}
      {sourceFaces.length > 0 ? (
        <div className="grid grid-cols-3 sm:grid-cols-4 gap-2 max-h-[220px] overflow-y-auto pr-1">
          {sourceFaces.map((face, index) => {
            const isSelected = selectedSourceIndex === index;
            const isPinned = pinnedIdentities.some((p) => p.sourceIndex === index);
            const thumbSrc = typeof face === 'string' ? face : face.thumb || face.image || '';

            return (
              <div
                key={`src-face-${index}`}
                onClick={() => onSelectSource?.(index)}
                className={`group relative flex flex-col items-center rounded-lg p-1 transition-all cursor-pointer select-none border ${
                  isSelected
                    ? 'bg-[var(--accent-primary,#38bdf8)]/15 border-[var(--accent-primary,#38bdf8)] shadow-[0_0_12px_rgba(56,189,248,0.25)]'
                    : 'bg-white/[0.03] border-white/10 hover:border-white/20 hover:bg-white/[0.06]'
                }`}
              >
                {/* Thumbnail */}
                <div className="relative w-full aspect-square rounded-md overflow-hidden bg-black/60">
                  {thumbSrc ? (
                    <img
                      src={thumbSrc}
                      alt={`Source #${index + 1}`}
                      decoding="async"
                      className="w-full h-full object-cover pointer-events-none"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-white/20 text-xs font-mono">
                      #{index + 1}
                    </div>
                  )}

                  {/* Top-Left Face Index Badge */}
                  <div className="absolute top-1 left-1 px-1 py-0.2 rounded bg-black/80 backdrop-blur-sm text-[9px] font-mono font-bold text-white/90">
                    #{index + 1}
                  </div>

                  {/* Top-Right Quick Pin Button */}
                  {onTogglePin && (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        onTogglePin(index);
                      }}
                      title={isPinned ? 'Unpin face' : 'Pin face identity'}
                      className={`absolute top-1 right-1 w-4 h-4 rounded flex items-center justify-center text-[10px] transition-opacity ${
                        isPinned
                          ? 'bg-amber-500 text-black font-bold opacity-100'
                          : 'bg-black/70 text-white/60 opacity-0 group-hover:opacity-100 hover:text-amber-300'
                      }`}
                    >
                      ★
                    </button>
                  )}

                  {/* Hover Actions Bar (Move & Delete) */}
                  <div className="absolute inset-x-0 bottom-0 p-0.5 bg-black/85 backdrop-blur-sm flex items-center justify-between opacity-0 group-hover:opacity-100 transition-opacity">
                    {onMoveSource && index > 0 ? (
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          onMoveSource(index, index - 1);
                        }}
                        title="Move left"
                        className="px-1 text-[9px] text-white/70 hover:text-white"
                      >
                        ←
                      </button>
                    ) : (
                      <span className="w-2" />
                    )}

                    {onRemoveSource && (
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          onRemoveSource(index);
                        }}
                        title="Remove face"
                        className="px-1 text-[10px] text-red-400 hover:text-red-300"
                      >
                        ✕
                      </button>
                    )}

                    {onMoveSource && index < sourceFaces.length - 1 ? (
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          onMoveSource(index, index + 1);
                        }}
                        title="Move right"
                        className="px-1 text-[9px] text-white/70 hover:text-white"
                      >
                        →
                      </button>
                    ) : (
                      <span className="w-2" />
                    )}
                  </div>
                </div>

                {/* Selection Ring Status */}
                <div className="w-full mt-1 flex items-center justify-between text-[10px]">
                  <span className={`font-mono ${isSelected ? 'text-[var(--accent-primary,#38bdf8)] font-bold' : 'text-white/50'}`}>
                    {isSelected ? 'Active' : 'Standby'}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        /* Empty State */
        <div
          onClick={() => fileInputRef.current?.click()}
          className="flex flex-col items-center justify-center gap-1.5 p-4 rounded-lg border border-dashed border-white/15 hover:border-[var(--accent-primary,#38bdf8)]/50 hover:bg-white/[0.02] transition-colors cursor-pointer text-center select-none"
        >
          <span className="text-xl text-white/30">+</span>
          <div className="space-y-0.5">
            <span className="text-[11px] font-semibold text-white/70">Add Source Face</span>
            <p className="text-[10px] text-white/40">Upload or drop photo / .fsz archive</p>
          </div>
        </div>
      )}

      {/* Hidden File Input */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept="image/*,.fsz"
        onChange={handleFileChange}
        className="hidden"
      />

      {/* Add More Faces Button */}
      {sourceFaces.length > 0 && (
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={busy}
          className="w-full py-1 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-[11px] font-semibold text-white/80 transition-colors flex items-center justify-center gap-1"
        >
          <span>+</span>
          <span>Add More Source Faces</span>
        </button>
      )}
    </div>
  );
}
