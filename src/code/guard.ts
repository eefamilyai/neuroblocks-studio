/**
 * Infinite-loop protection for code mode.
 *
 * User code runs inside the training worker AND on the main thread (the 3D
 * showcase). A `while (true) {}` would freeze the page, so before running a
 * program we parse it and give every loop its own counter:
 *
 *     while (x) body      →   { let __l0 = 0; while (x) { if (++__l0 > LIMIT) __loopLimit(); body } }
 *
 * Only text is inserted (never newlines), so line numbers in error messages
 * still match what you typed.
 */
import { parser } from '@lezer/javascript';
import type { SyntaxNode } from '@lezer/common';

export const LOOP_LIMIT = 1_000_000;

export interface GuardResult {
  code: string;
  loops: number;
  /** first parse error found by the parser (position in the original source), if any */
  syntaxError: { from: number; to: number } | null;
}

const LOOPS = new Set(['ForStatement', 'WhileStatement', 'DoStatement']);
const isStatement = (n: SyntaxNode) => n.name === 'Block' || n.name.endsWith('Statement') || n.name.endsWith('Declaration');

function loopBody(node: SyntaxNode): SyntaxNode | null {
  if (node.name === 'DoStatement') {
    for (let c = node.firstChild; c; c = c.nextSibling) if (isStatement(c)) return c;
    return null;
  }
  const last = node.lastChild;
  return last && isStatement(last) ? last : null;
}

interface Insert {
  pos: number;
  text: string;
  close: boolean;
  seq: number;
}

export function guardLoops(src: string): GuardResult {
  const tree = parser.parse(src);
  const ins: Insert[] = [];
  let seq = 0;
  let loops = 0;
  let syntaxError: GuardResult['syntaxError'] = null;
  const push = (pos: number, text: string, close: boolean) => ins.push({ pos, text, close, seq: seq++ });
  tree.iterate({
    enter: (ref) => {
      if (ref.type.isError) {
        if (!syntaxError) syntaxError = { from: ref.from, to: Math.max(ref.to, ref.from + 1) };
        return;
      }
      if (!LOOPS.has(ref.name)) return;
      const node = ref.node;
      const body = loopBody(node);
      if (!body) return;
      const id = `__l${loops++}`;
      let outer: SyntaxNode = node;
      while (outer.parent && outer.parent.name === 'LabeledStatement') outer = outer.parent;
      const check = `if (++${id} > ${LOOP_LIMIT}) __loopLimit();`;
      push(outer.from, `{ let ${id} = 0; `, false);
      push(outer.to, ' }', true);
      if (body.name === 'Block') push(body.from + 1, ` ${check}`, false);
      else {
        push(body.from, `{ ${check} `, false);
        push(body.to, ' }', true);
      }
    },
  });
  // same position: closing braces first (innermost first), then openings (outermost first)
  ins.sort((a, b) => a.pos - b.pos || (a.close === b.close ? (a.close ? b.seq - a.seq : a.seq - b.seq) : a.close ? -1 : 1));
  let out = '';
  let last = 0;
  for (const i of ins) {
    out += src.slice(last, i.pos) + i.text;
    last = i.pos;
  }
  out += src.slice(last);
  return { code: out, loops, syntaxError };
}

/** 1-based line and column of a character offset. */
export function lineCol(src: string, pos: number): { line: number; col: number } {
  let line = 1;
  let lastNl = -1;
  for (let i = 0; i < pos && i < src.length; i++) {
    if (src.charCodeAt(i) === 10) {
      line++;
      lastNl = i;
    }
  }
  return { line, col: pos - lastNl };
}

/** Character offset of the start of a 1-based line. */
export function lineStart(src: string, line: number): number {
  let pos = 0;
  for (let l = 1; l < line; l++) {
    const nl = src.indexOf('\n', pos);
    if (nl < 0) return src.length;
    pos = nl + 1;
  }
  return pos;
}
