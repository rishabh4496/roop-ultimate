import { useState } from "react";
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
 * Source faces and the people detected in the target, with interactive
 * matching: click a source (it stays "armed"), then click every person who
 * should become that source; click an assigned person again with the same
 * source armed to release them. Each person also has a source picker (the
 * keyboard / screen-reader path).
 *
 * With no assignment at all the server swaps EVERY face with the first source;
 * as soon as one person is assigned, only assigned people are swapped
 * (multi-face targeting: give each person their own source).
 */
export function FaceSelectorGrid(props: FaceSelectorGridProps) {
  const { people, sources, assignments } = props;
  const [armed, setArmed] = useState<string | null>(null);
  const assignedCount = Object.keys(assignments).length;
  const sourceName = (id: string) => sources.find((s) => s.id === id)?.name ?? "?";

  const clickPerson = (person: DetectedFace) => {
    props.onSelect?.(person);
    if (armed === null || props.disabled) return;
    props.onAssign(person.id, assignments[person.id] === armed ? null : armed);
  };

  return (
    <section aria-label="Face matching" className="flex flex-col gap-3">
      {sources.length > 0 && (
        <div className="flex flex-col gap-1">
          <span className="text-[10px] uppercase tracking-wide text-zinc-500">
            Sources {armed ? "· click people to assign" : "· pick one to assign by clicking"}
          </span>
          <ul className="flex flex-wrap gap-2" aria-label="Source faces">
            {sources.map((s) => (
              <li key={s.id}>
                <button
                  type="button"
                  aria-pressed={armed === s.id}
                  aria-label={`Assign with source ${s.name}`}
                  disabled={props.disabled}
                  onClick={() => setArmed((a) => (a === s.id ? null : s.id))}
                  className={`flex flex-col items-center rounded p-1 text-[10px] text-zinc-400 ${
                    armed === s.id ? "bg-sky-900 ring-2 ring-sky-400" : "hover:bg-zinc-800"
                  } disabled:opacity-50`}
                >
                  <img src={s.thumbnail_url} alt={s.name} className="h-14 w-14 rounded object-cover" />
                  <span className="max-w-14 truncate">{s.name}</span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      {people.length === 0 ? (
        <p className="text-sm text-zinc-400">No faces detected yet.</p>
      ) : (
        <>
          <p className="text-xs text-zinc-400" data-testid="swap-mode">
            {assignedCount === 0
              ? `No assignments: every face will get ${sources[0]?.name ?? "the first source"}.`
              : `${assignedCount} of ${people.length} people assigned; the others stay untouched.`}
          </p>
          <ul className="grid grid-cols-2 gap-3 xl:grid-cols-3" aria-label="Detected faces">
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
                    onClick={() => clickPerson(person)}
                    className={`overflow-hidden rounded ${armed ? "cursor-copy" : ""}`}
                    aria-label={
                      armed
                        ? `Assign ${sourceName(armed)} to person seen ${person.count} times`
                        : `Show person seen ${person.count} times at frame ${person.frame}`
                    }
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
                      {assigned ? `→ ${sourceName(assigned)}` : " "}
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
        </>
      )}
    </section>
  );
}
