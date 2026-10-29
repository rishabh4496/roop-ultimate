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
//   primary = number | 'pi' | '(' expr ')'
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

const NUMBER = '(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][+-]?\\d+)?';
const SIGNED_PERCENT = new RegExp(`^([+-])\\s*(${NUMBER})\\s*%$`);
const PLAIN_PERCENT = new RegExp(`^(${NUMBER})\\s*%$`);

const fail = (error) => ({ ok: false, error });

/** Typographic operators and decimal commas, folded to ASCII before anything reads the text. */
export function normalizeExpression(text) {
  return String(text ?? '')
    .replace(/[−–—]/g, '-')
    .replace(/[×⋅·]/g, '*')
    .replace(/[÷∕]/g, '/')
    .replace(/(\d),(\d)/g, '$1.$2')
    .trim();
}

function tokenize(src) {
  const tokens = [];
  let i = 0;
  while (i < src.length) {
    const c = src[i];
    if (c === ' ' || c === '\t') { i += 1; continue; }
    if ('+-*/^()%'.includes(c)) { tokens.push({ t: c }); i += 1; continue; }
    const num = new RegExp(`^${NUMBER}`).exec(src.slice(i));
    if (num) { tokens.push({ t: 'num', v: Number(num[0]) }); i += num[0].length; continue; }
    if (src.slice(i, i + 2).toLowerCase() === 'pi' && !/[a-z0-9_]/i.test(src[i + 2] || '')) {
      tokens.push({ t: 'num', v: Math.PI });
      i += 2;
      continue;
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
  if (pos < tokens.length) throw new Error(`Unexpected "${tokens[pos].t === 'num' ? tokens[pos].v : tokens[pos].t}"`);
  return result.v;
}

/**
 * Evaluate what was typed into a numeric field.
 * `current` anchors a signed percentage; `min`/`max` anchor an unsigned one.
 * Returns { ok: true, value } or { ok: false, error }. The value is NOT clamped or rounded here -
 * see settleTypedValue for that.
 */
export function evaluateExpression(text, { current = 0, min = -Infinity, max = Infinity } = {}) {
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

  const lexed = tokenize(src);
  if (lexed.error) return fail(lexed.error);
  let value;
  try {
    value = parse(lexed.tokens);
  } catch (err) {
    return fail(err.message);
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
export function settleTypedValue(text, { current = 0, min = -Infinity, max = Infinity, step = 0 } = {}) {
  const r = evaluateExpression(text, { current, min, max });
  if (!r.ok) return r;
  const rounded = Number.isInteger(step) && step >= 1 ? Math.round(r.value) : Number(r.value.toFixed(4));
  const value = clamp(rounded, min, max);
  return { ok: true, value, clamped: value !== rounded };
}
