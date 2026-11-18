// The command palette's search, as pure functions (no React, no DOM) so the render-check can pin the ranking.
//
// A command matches when its title, subtitle or section contains the query's characters in order (a subsequence).
// A TITLE match outranks a match that only lands in the subtitle or section, because "Compare" should find
// "Compare: Split wipe" before a subtitle that happens to mention compare.

/** Subsequence match with a relevance score: 0 = no match. A contiguous run scores by how early it starts. */
export function scoreText(query, text) {
  if (!query) return 1;
  const q = String(query).toLowerCase();
  const t = String(text || '').toLowerCase();
  if (!t) return 0;
  if (t.includes(q)) return 100 - t.indexOf(q); // a contiguous match ranks highest, earlier is better
  let qi = 0;
  for (let ti = 0; ti < t.length && qi < q.length; ti += 1) {
    if (t[ti] === q[qi]) qi += 1;
  }
  return qi === q.length ? 10 : 0; // a scattered subsequence match is still a match, just a weak one
}

/** The command's overall score for `query`: a title match gets a large lift over a match elsewhere. */
export function scoreCommand(query, command) {
  if (!query) return 1;
  const onTitle = scoreText(query, command.title);
  if (onTitle > 0) return 1000 + onTitle;
  return scoreText(query, `${command.subtitle || ''} ${command.section || ''}`);
}

/** The commands that match `query`, best first. An empty query keeps the commands in the order they were given. */
export function rankCommands(commands, query) {
  const list = Array.isArray(commands) ? commands : [];
  if (!query || !String(query).trim()) return list.slice();
  return list
    .map((c, i) => ({ c, s: scoreCommand(String(query).trim(), c), i }))
    .filter((x) => x.s > 0)
    .sort((a, b) => (b.s - a.s) || (a.i - b.i))   // stable: equal scores keep the palette's own order
    .map((x) => x.c);
}

/** Group ranked commands by section, in the order each section first appears. */
export function groupBySection(ranked) {
  const groups = [];
  const seen = new Map();
  ranked.forEach((c, i) => {
    const key = c.section || 'Actions';
    if (!seen.has(key)) { seen.set(key, groups.length); groups.push({ section: key, items: [] }); }
    groups[seen.get(key)].items.push({ c, i });
  });
  return groups;
}
