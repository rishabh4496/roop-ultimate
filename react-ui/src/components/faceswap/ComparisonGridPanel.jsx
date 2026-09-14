import CompareGrid from './CompareGrid';

/**
 * Shared comparison-grid chrome for enhancer, mask, swapper, and upscaler
 * comparisons. The parent owns selection state and preview loading; this
 * component owns only the repeated selector and result-grid presentation.
 */
export default function ComparisonGridPanel({
  title,
  subtitle = null,
  availableItems = [],
  selectedItems = [],
  setSelectedItems,
  previews,
  times,
  timers,
  notify,
  itemNoun = 'items',
}) {
  const activeItems = selectedItems.filter((item) => availableItems.includes(item));
  const gridColsClass = activeItems.length === 1 ? 'grid-cols-1' : 'grid-cols-2';

  const toggleItem = (item) => {
    const isSelected = selectedItems.includes(item);
    if (isSelected) {
      if (selectedItems.length > 1) {
        setSelectedItems((prev) => prev.filter((value) => value !== item));
      }
      return;
    }
    if (selectedItems.length >= 4) {
      notify(`You can select a maximum of 4 ${itemNoun} for grid comparison.`, 'warning');
      return;
    }
    setSelectedItems((prev) => [...prev, item]);
  };

  return (
    <div className="space-y-4">
      <div className="p-3.5 rounded-xl bg-black/45 border border-white/5 space-y-2 select-none">
        <div className="flex items-center justify-between gap-2">
          <span className="text-micro font-semibold uppercase tracking-[0.14em] text-white/45 block">
            {title}
          </span>
          {subtitle && <span className="text-micro text-white/45">{subtitle}</span>}
        </div>
        <div className="flex flex-wrap gap-2">
          {availableItems.map((item) => {
            const isSelected = selectedItems.includes(item);
            return (
              <button
                key={item}
                type="button"
                onClick={() => toggleItem(item)}
                className={`px-3 py-1.5 rounded-lg text-mini font-semibold border transition-all duration-200 ${isSelected ? 'bg-[var(--accent)]/15 border-[var(--accent)]/40 text-white' : 'bg-white/[0.02] border-white/10 text-white/50 hover:border-white/20 hover:text-white/85'}`}
              >
                {item}
              </button>
            );
          })}
        </div>
      </div>

      <CompareGrid
        items={activeItems}
        gridColsClass={gridColsClass}
        previews={previews}
        times={times}
        timers={timers}
      />
    </div>
  );
}
