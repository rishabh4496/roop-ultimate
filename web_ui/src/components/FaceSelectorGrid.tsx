import type { DetectedFace, SourceFace } from "../lib/types";

export interface FaceSelectorGridProps {
  people: DetectedFace[];
  sources: SourceFace[];
  /** person id -> source id */
  assignments: Record<string, string>;
  onAssign: (personId: string, sourceId: string | null) => void;
  /** Currently highlighted person (e.g. to jump the player to their frame). */
  selected?: string | null;
  onSelect?: (person: DetectedFace) => void;
  disabled?: boolean;
}

/**
 * Detected people in the target, each with a source picker. With no
 * assignment at all the server swaps EVERY face with the first source; as soon
 * as one person is assigned, only assigned people are swapped (multi-target
 * swaps: give each person their own source).
 */
export function FaceSelectorGrid(props: FaceSelectorGridProps) {
  const { people, sources, assignments } = props;
  const assignedCount = Object.keys(assignments).length;
  const sourceName = (id: string) => sources.find((s) => s.id === id)?.name ?? "?";

  if (people.length === 0) {
    return <p className="text-sm text-zinc-400">No faces detected yet.</p>;
  }

  return (
    <section aria-label="Detected faces" className="flex flex-col gap-3">
      <p className="text-xs text-zinc-400" data-testid="swap-mode">
        {assignedCount === 0
          ? `No assignments: every face will get ${sources[0]?.name ?? "the first source"}.`
          : `${assignedCount} of ${people.length} people assigned; the others stay untouched.`}
      </p>
      <ul className="grid grid-cols-2 gap-3 xl:grid-cols-3">
        {people.map((person) => {
          const assigned = assignments[person.id];
          const isSelected = props.selected === person.id;
          return (
            <li
              key={person.id}
              data-person={person.id}
              className={`flex flex-col gap-2 rounded-lg border p-2 ${
                assigned ? "border-sky-500 bg-sky-950/40" : "border-zinc-800 bg-zinc-900"
              } ${isSelected ? "ring-2 ring-amber-400" : ""}`}
            >
              <button
                type="button"
                onClick={() => props.onSelect?.(person)}
                className="overflow-hidden rounded"
                aria-label={`Show person seen ${person.count} times at frame ${person.frame}`}
              >
                <img
                  src={person.thumbnail_url}
                  alt={`Detected person, score ${person.score.toFixed(2)}`}
                  className="aspect-square w-full object-cover"
                />
              </button>
              <div className="flex min-w-0 flex-col text-xs text-zinc-400">
                <span>{person.count}× seen</span>
                <span className="truncate text-sky-300" title={assigned ? sourceName(assigned) : undefined}>
                  {assigned ? `→ ${sourceName(assigned)}` : " "}
                </span>
              </div>
              <select
                aria-label={`Source for person ${person.id}`}
                value={assigned ?? ""}
                disabled={props.disabled || sources.length === 0}
                onChange={(e) => props.onAssign(person.id, e.target.value || null)}
                className="rounded border border-zinc-700 bg-zinc-950 px-1 py-1 text-xs"
              >
                <option value="">— keep original —</option>
                {sources.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.name}
                  </option>
                ))}
              </select>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
