// Arithmetic for numeric fields: type `512/2`, `24*1.5` or `+10%` where a number goes.
//
// There is no eval() and no Function() here. The text is tokenised by hand and
// parsed by recursive descent over a fixed grammar, so the only things an input
// can do are the things the grammar names - it cannot reach a global, call
// anything or loop. Length and nesting are capped so a pasted wall of
// parentheses fails with a message instead of exhausting the stack.
//
//   expr    = term { ('+' | '-') term }
//   term    = unary { ('*' | '/') unary }
//   unary   = ('+' | '-') unary | power
//   power   = postfix [ '^' unary ]            right-associative: 2^3^2 = 2^9
//   postfix = primary { '%' }
//   primary = number | 'pi' | name | call | '(' expr ')'
//   call    = ('min' | 'max' | 'abs' | 'round' | 'floor' | 'ceil' | 'clamp') '(' expr { ',' expr } ')'
//
// RELATIVE FORMS (Stage 15). Typed over a field that already holds a value, an input that STARTS with an operator edits
// that value instead of replacing it:
//
//   *2  /1.5        multiply / divide the current value ( *2 on 40 -> 80 )
//   +16             add to the current value ( +16 on 40 -> 56 ); the right side is any expression ( +2*8 )
//   -10%  +10%      scale the current value (below)
//
// A bare `-5` is still the number minus five (see the last paragraph), so to subtract a number write `v-16`: `v` (also
// `value` / `current`) is the field's current value. `min(v*2, 512)` and `max(v/2, 16)` read it too.
//
// PERCENT has three meanings, and which one applies is decided by where it sits:
//
//   +10%  /  -10%     the WHOLE input: scale the current value ( +10% of 40 -> 44 )
//   50%               the WHOLE input, unsigned: a share of the RANGE ( 0..1 -> 0.5 )
//   a + b%  a - b%    b percent OF a, as a spreadsheet reads it ( 200 + 10% -> 220 )
//   a * b%            b/100 ( 200 * 50% -> 100 )
//
// A bare `-5` is the number minus five, not "five below the current value": a
// field that was just selected and typed over means the value typed.

export const MAX_EXPR_LENGTH = 120;
export const MAX_DEPTH = 40;

/**
 * Stage 26: Registered expression math parameters available for formula evaluation.
 */
export const REGISTERED_MATH_PARAMETERS = Object.freeze([
  'stabilization_strength',
  'yaw_erosion_bias',
  'expression_restorer_weight',
  'pixel_boost',
  'temporal_restore_alpha',
  'restore_identity_weight',
  'grain_match_strength',
  'motion_blur_match',
]);

export const DEFAULT_MATH_PARAMETER_VALUES = Object.freeze({
  stabilization_strength: 0.5,
  yaw_erosion_bias: 0.0,
  expression_restorer_weight: 0.6,
  pixel_boost: 512,
  temporal_restore_alpha: 0.5,
  restore_identity_weight: 0.6,
  grain_match_strength: 0.35,
  motion_blur_match: 0.0,
});

const NUMBER = '(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][+-]?\\d+)?';
const SIGNED_PERCENT = new RegExp(`^([+-])\\s*(${NUMBER})\\s*%$`);
const PLAIN_PERCENT = new RegExp(`^(${NUMBER})\\s*%$`);

const fail = (error) => ({ ok: false, error });

// The functions the grammar knows, with their allowed argument counts. Anything else followed by `(` is an error, so
// `alert(1)` and `constructor(...)` can never name anything.
export const FUNCTIONS = Object.freeze({
  min: { min: 1, max: 8, fn: (a) => Math.min(...a) },
  max: { min: 1, max: 8, fn: (a) => Math.max(...a) },
  abs: { min: 1, max: 1, fn: (a) => Math.abs(a[0]) },
  round: { min: 1, max: 1, fn: (a) => Math.round(a[0]) },
  floor: { min: 1, max: 1, fn: (a) => Math.floor(a[0]) },
  ceil: { min: 1, max: 1, fn: (a) => Math.ceil(a[0]) },
  clamp: { min: 3, max: 3, fn: (a) => Math.min(a[2], Math.max(a[1], a[0])) },
});
const CALLS_A_FUNCTION = /[a-z_][a-z0-9_]*\s*\(/i;
// The names that mean "the value the field holds now".
const CURRENT_NAMES = Object.freeze(['v', 'value', 'current']);

/** Typographic operators and decimal commas, folded to ASCII before anything reads the text. */
export function normalizeExpression(text) {
  const folded = String(text ?? '')
    .replace(/[−–—]/g, '-')
    .replace(/[×⋅·]/g, '*')
    .replace(/[÷∕]/g, '/');
  // `1,5` is a decimal comma - unless the text calls a function, where the comma separates arguments: `min(1,5)`.
  return (CALLS_A_FUNCTION.test(folded) ? folded : folded.replace(/(\d),(\d)/g, '$1.$2')).trim();
}

function tokenize(src, ctx = {}) {
  const tokens = [];
  let i = 0;
  while (i < src.length) {
    const c = src[i];
    if (c === ' ' || c === '\t') { i += 1; continue; }
    if ('+-*/^()%,'.includes(c)) { tokens.push({ t: c }); i += 1; continue; }
    const num = new RegExp(`^${NUMBER}`).exec(src.slice(i));
    if (num) { tokens.push({ t: 'num', v: Number(num[0]) }); i += num[0].length; continue; }
    if (src.slice(i, i + 2).toLowerCase() === 'pi' && !/[a-z0-9_]/i.test(src[i + 2] || '')) {
      tokens.push({ t: 'num', v: Math.PI });
      i += 2;
      continue;
    }
    const identMatch = new RegExp(`^[a-zA-Z_][a-zA-Z0-9_]*`).exec(src.slice(i));
    if (identMatch) {
      const ident = identMatch[0];
      const lower = ident.toLowerCase();
      if (/^\s*\(/.test(src.slice(i + ident.length))) {
        if (!Object.hasOwn(FUNCTIONS, lower)) return { error: `Unknown function "${ident}"` };
        tokens.push({ t: 'fn', name: lower });
        i += ident.length;
        continue;
      }
      if (CURRENT_NAMES.includes(lower) && Number.isFinite(ctx.current)) {
        tokens.push({ t: 'num', v: ctx.current });
        i += ident.length;
        continue;
      }
      const paramVal = ctx.variables?.[ident] ?? ctx.params?.[ident] ?? ctx[ident]
        ?? (REGISTERED_MATH_PARAMETERS.includes(ident) ? DEFAULT_MATH_PARAMETER_VALUES[ident] : undefined);
      if (paramVal !== undefined && Number.isFinite(Number(paramVal))) {
        tokens.push({ t: 'num', v: Number(paramVal) });
        i += ident.length;
        continue;
      }
      return { error: `Unexpected "${ident}"` };
    }
    return { error: `Unexpected "${c}"` };
  }
  return { tokens };
}

// Each production returns { v, pct }: `pct` marks a value that is a bare percentage so the additive level can
// reinterpret it relative to its left operand. It is dropped the moment anything else touches the value.
function parse(tokens) {
  let pos = 0;
  let depth = 0;
  const peek = () => (pos < tokens.length ? tokens[pos].t : null);
  const take = () => tokens[pos++];

  const enter = () => {
    depth += 1;
    if (depth > MAX_DEPTH) throw new Error('Too deeply nested');
  };
  const leave = () => { depth -= 1; };

  function primary() {
    const tk = peek();
    if (tk === 'num') return { v: take().v, pct: false };
    if (tk === '(') {
      take();
      enter();
      const inner = expr();
      leave();
      if (peek() !== ')') throw new Error('Missing closing parenthesis');
      take();
      return { v: inner.v, pct: false };
    }
    if (tk === 'fn') {
      const { name } = take();
      const spec = FUNCTIONS[name];
      if (peek() !== '(') throw new Error(`${name} needs parentheses`);
      take();
      enter();
      const args = [];
      if (peek() === ')') throw new Error(`${name}() needs a value`);
      for (;;) {
        args.push(expr().v);
        if (peek() === ',') { take(); continue; }
        break;
      }
      leave();
      if (peek() !== ')') throw new Error('Missing closing parenthesis');
      take();
      if (args.length < spec.min || args.length > spec.max) {
        throw new Error(spec.min === spec.max ? `${name} takes ${spec.min} value${spec.min === 1 ? '' : 's'}` : `${name} takes ${spec.min} to ${spec.max} values`);
      }
      return { v: spec.fn(args), pct: false };
    }
    if (tk === null) throw new Error('Expression ends early');
    throw new Error(`Unexpected "${tk}"`);
  }

  function postfix() {
    let r = primary();
    while (peek() === '%') { take(); r = { v: r.v / 100, pct: true }; }
    return r;
  }

  function power() {
    const base = postfix();
    if (peek() !== '^') return base;
    take();
    enter();
    const exp = unary();
    leave();
    return { v: base.v ** exp.v, pct: false };
  }

  function unary() {
    const tk = peek();
    if (tk === '+' || tk === '-') {
      take();
      enter();
      const r = unary();
      leave();
      return { v: tk === '-' ? -r.v : r.v, pct: r.pct };
    }
    return power();
  }

  function term() {
    let left = unary();
    while (peek() === '*' || peek() === '/') {
      const op = take().t;
      const right = unary();
      if (op === '/' && right.v === 0) throw new Error('Division by zero');
      left = { v: op === '*' ? left.v * right.v : left.v / right.v, pct: false };
    }
    return left;
  }

  function expr() {
    let left = term();
    while (peek() === '+' || peek() === '-') {
      const op = take().t;
      const right = term();
      // `a + b%` is b percent of a. Only a term that IS a bare percentage reinterprets; `a + 2*10%` does not.
      const delta = right.pct ? left.v * right.v : right.v;
      left = { v: op === '+' ? left.v + delta : left.v - delta, pct: false };
    }
    return left;
  }

  const result = expr();
  if (pos < tokens.length) throw new Error(`Unexpected "${tokens[pos].t === 'num' ? tokens[pos].v : tokens[pos].t === 'fn' ? tokens[pos].name : tokens[pos].t}"`);
  return result.v;
}

/**
 * Evaluate what was typed into a numeric field.
 * `current` anchors a signed percentage; `min`/`max` anchor an unsigned one.
 * Returns { ok: true, value } or { ok: false, error }. The value is NOT clamped or rounded here -
 * see settleTypedValue for that.
 */
export function evaluateExpression(text, ctx = {}) {
  const { current = 0, min = -Infinity, max = Infinity } = ctx;
  const src = normalizeExpression(text);
  if (!src) return fail('Type a number or an expression');
  if (src.length > MAX_EXPR_LENGTH) return fail('Expression is too long');

  const signed = SIGNED_PERCENT.exec(src);
  if (signed) {
    const n = Number(signed[2]) / 100;
    const base = Number.isFinite(current) ? current : 0;
    return { ok: true, value: base * (1 + (signed[1] === '-' ? -n : n)) };
  }
  const plain = PLAIN_PERCENT.exec(src);
  if (plain) {
    const n = Number(plain[1]) / 100;
    const value = Number.isFinite(min) && Number.isFinite(max) ? min + n * (max - min) : n;
    return { ok: true, value };
  }

  // `*2`, `/1.5`, `+16`: the input starts with an operator, so it edits the value the field holds.
  const lead = /^([*/+])\s*(\S[\s\S]*)$/.exec(src);
  const rel = lead && !/^[+*/]/.test(lead[2]) ? lead : null;
  const body = rel ? rel[2] : src;

  const lexed = tokenize(body, ctx);
  if (lexed.error) return fail(lexed.error);
  let value;
  try {
    value = parse(lexed.tokens);
  } catch (err) {
    return fail(err.message);
  }
  if (rel) {
    const base = Number.isFinite(ctx.current) ? ctx.current : 0;
    if (rel[1] === '/' && value === 0) return fail('Division by zero');
    value = rel[1] === '*' ? base * value : rel[1] === '/' ? base / value : base + value;
  }
  if (!Number.isFinite(value)) return fail('Result is out of range');
  return { ok: true, value };
}

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/**
 * Evaluate, then make the result fit the control: an integer-stepped control takes a whole number
 * (`512/3` is 171, not 170.3333), everything else keeps four decimals, and the value lands inside [min, max].
 * `clamped` tells the caller the typed number was out of range so it can say so.
 */
export function settleTypedValue(text, opts = {}) {
  const { current = 0, min = -Infinity, max = Infinity, step = 0 } = opts;
  const r = evaluateExpression(text, opts);
  if (!r.ok) return r;
  const rounded = Number.isInteger(step) && step >= 1 ? Math.round(r.value) : Number(r.value.toFixed(4));
  const value = clamp(rounded, min, max);
  return { ok: true, value, clamped: value !== rounded };
}

// A leading `+` is NOT plain: `+16` is "add 16 to the current value", so it gets its pill. A leading `-` is the number.
const PLAIN_NUMBER = new RegExp(`^-?\\s*${NUMBER}$`);

/**
 * What the live preview pill under a field being edited says (Stage 15). `null` when the text is empty or just a number -
 * a pill that repeats what you typed is noise. Otherwise
 *   { formula, ok: true,  result, text, clamped }   `text` is the result as the field will take it ("256", "4096 (max)")
 *   { formula, ok: false, error }
 * The result is what Enter would commit: the same settleTypedValue the field calls, so the preview cannot disagree with it.
 */
export function previewExpression(text, opts = {}) {
  const formula = normalizeExpression(text);
  if (!formula || PLAIN_NUMBER.test(formula)) return null;
  const r = settleTypedValue(formula, opts);
  if (!r.ok) return { formula, ok: false, error: r.error };
  const { min = -Infinity, max = Infinity } = opts;
  const bound = r.clamped ? (r.value === max ? ' (max)' : r.value === min ? ' (min)' : '') : '';
  const digits = Number.isInteger(r.value) ? String(r.value) : String(Number(r.value.toFixed(4)));
  return { formula, ok: true, result: r.value, text: `${digits}${bound}`, clamped: r.clamped };
}
