/**
 * The code-mode editor (CodeMirror 6). This module is loaded on demand, so
 * people who only use blocks never download it.
 */
import { EditorView, keymap, lineNumbers, highlightActiveLine, highlightActiveLineGutter, drawSelection, dropCursor, highlightSpecialChars, rectangularSelection } from '@codemirror/view';
import { EditorState } from '@codemirror/state';
import { defaultKeymap, history, historyKeymap, indentWithTab } from '@codemirror/commands';
import { javascript } from '@codemirror/lang-javascript';
import { autocompletion, closeBrackets, closeBracketsKeymap, completionKeymap, snippetCompletion } from '@codemirror/autocomplete';
import type { Completion, CompletionContext, CompletionResult } from '@codemirror/autocomplete';
import { lintGutter, setDiagnostics } from '@codemirror/lint';
import type { Diagnostic } from '@codemirror/lint';
import { HighlightStyle, bracketMatching, foldGutter, foldKeymap, indentOnInput, syntaxHighlighting, syntaxTree } from '@codemirror/language';
import { highlightSelectionMatches, searchKeymap } from '@codemirror/search';
import { tags as t } from '@lezer/highlight';
import { ACTIVATION_WORDS, DOCS, TAG_WORDS, ZONE_WORDS } from '../code/docs.ts';
import type { CodeDiagnostic } from '../code/runtime.ts';

export interface CodeEditor {
  getValue(): string;
  setValue(s: string): void;
  setDiagnostics(d: CodeDiagnostic[]): void;
  focus(): void;
  goto(pos: number): void;
  insert(text: string): void;
  destroy(): void;
}

const theme = EditorView.theme(
  {
    '&': { color: '#dfe4f0', backgroundColor: 'var(--bg-1)' },
    '.cm-content': { caretColor: '#9fb3ff', padding: '10px 0' },
    '.cm-cursor, .cm-dropCursor': { borderLeftColor: '#9fb3ff', borderLeftWidth: '2px' },
    '&.cm-focused .cm-selectionBackground, .cm-selectionBackground, .cm-content ::selection': { backgroundColor: 'rgba(110,139,255,0.28) !important' },
    '.cm-selectionMatch': { backgroundColor: 'rgba(110,139,255,0.14)' },
    '.cm-matchingBracket': { backgroundColor: 'rgba(255,255,255,0.08)', outline: '1px solid rgba(255,255,255,0.18)' },
    '.cm-foldPlaceholder': { backgroundColor: 'var(--bg-3)', border: 'none', color: 'var(--text-3)' },
    '.cm-lineNumbers .cm-gutterElement': { padding: '0 10px 0 14px', minWidth: '36px' },
  },
  { dark: true },
);

const highlight = HighlightStyle.define([
  { tag: t.keyword, color: '#c9a7ff' },
  { tag: [t.definitionKeyword, t.modifier, t.controlKeyword, t.operatorKeyword], color: '#c9a7ff' },
  { tag: [t.string, t.special(t.string)], color: '#9fe3a8' },
  { tag: [t.number, t.bool, t.null], color: '#ffc58a' },
  { tag: t.comment, color: '#5f6a82', fontStyle: 'italic' },
  { tag: [t.function(t.variableName), t.function(t.propertyName)], color: '#82c4ff' },
  { tag: t.propertyName, color: '#a8c7fa' },
  { tag: t.variableName, color: '#e7eaf1' },
  { tag: [t.definition(t.variableName)], color: '#ffd98a' },
  { tag: [t.operator, t.punctuation, t.separator], color: '#8a93a8' },
  { tag: t.bracket, color: '#a9b0c2' },
  { tag: t.invalid, color: '#ff8ea0' },
]);

// ---------------------------------------------------------------------------
// Autocomplete: builder & api functions, plus tags/zones inside quotes
// ---------------------------------------------------------------------------
function docCompletion(d: (typeof DOCS)[number], prefixLen: number): Completion {
  const label = d.name.slice(prefixLen);
  const base: Completion = {
    label,
    detail: d.sig.length > 46 ? d.sig.slice(0, 44) + '…)' : d.sig,
    info: d.doc,
    type: d.group.startsWith('api') ? 'method' : 'function',
    boost: d.group.startsWith('api') ? 1 : 0,
  };
  if (!d.snippet) return { ...base, apply: label + (d.sig.startsWith('(') ? '()' : '') };
  return snippetCompletion(d.snippet.slice(prefixLen), base);
}

function complete(ctx: CompletionContext): CompletionResult | null {
  const node = syntaxTree(ctx.state).resolveInner(ctx.pos, -1);
  // inside a string: offer tags / zones / activations
  if (node.name === 'String') {
    const before = ctx.state.sliceDoc(Math.max(0, node.from - 30), node.from);
    const words = /zone\s*:\s*$|clearZone\($/.test(before) ? ZONE_WORDS : /layers?\([^)]*,\s*$/.test(before) ? ACTIVATION_WORDS : TAG_WORDS;
    const quote = ctx.state.sliceDoc(node.from, node.from + 1);
    return { from: node.from + 1, to: node.to - (ctx.state.sliceDoc(node.to - 1, node.to) === quote && node.to - node.from > 1 ? 1 : 0), options: words.map((w) => ({ label: w, type: 'constant' })), validFor: /^\w*$/ };
  }
  if (node.name === 'LineComment' || node.name === 'BlockComment') return null;
  // "obj." → member completion for a builder / api
  const m = ctx.matchBefore(/[A-Za-z_]\w*\.\w*$/);
  if (m) {
    const dot = m.text.indexOf('.');
    const obj = m.text.slice(0, dot + 1);
    const items = DOCS.filter((d) => d.name.startsWith(obj));
    if (!items.length) return null;
    return { from: m.from + dot + 1, options: items.map((d) => docCompletion(d, obj.length)), validFor: /^\w*$/ };
  }
  const w = ctx.matchBefore(/[A-Za-z_]\w*$/);
  if (!w && !ctx.explicit) return null;
  const roots = ['robot', 'sense', 'actions', 'act', 'brain', 'train', 'world', 'on', 'api', 'Math', 'console'];
  const opts: Completion[] = roots.map((r) => ({ label: r, type: r === 'robot' ? 'function' : 'namespace', info: DOCS.find((d) => d.name === r || d.name.startsWith(r + '.'))?.group }));
  opts.push(
    snippetCompletion("on.step(() => {\n\t${}\n});", { label: 'on.step', detail: 'every step', type: 'keyword', boost: 2 }),
    snippetCompletion("on.touch('${flag}', (obj) => {\n\t${}\n});", { label: 'on.touch', detail: 'when I touch…', type: 'keyword', boost: 2 }),
    snippetCompletion("api.reward(${1}, '${label}');", { label: 'api.reward', detail: '(x, label)', type: 'keyword', boost: 2 }),
    snippetCompletion('if (${condition}) {\n\t${}\n}', { label: 'if', type: 'keyword' }),
    snippetCompletion('for (let ${i} = 0; ${i} < ${n}; ${i}++) {\n\t${}\n}', { label: 'for', type: 'keyword' }),
    snippetCompletion('const ${name} = ${value};', { label: 'const', type: 'keyword' }),
    snippetCompletion('let ${name} = ${value};', { label: 'let', type: 'keyword' }),
  );
  return { from: w ? w.from : ctx.pos, options: opts, validFor: /^\w*$/ };
}

export function createCodeEditor(host: HTMLElement, opts: { doc: string; onChange: (doc: string) => void; onRun: () => void; onSave: () => void; readOnly?: boolean }): CodeEditor {
  let silent = false;
  const view = new EditorView({
    parent: host,
    state: EditorState.create({
      doc: opts.doc,
      extensions: [
        lineNumbers(),
        foldGutter({ openText: '▾', closedText: '▸' }),
        lintGutter(),
        highlightActiveLineGutter(),
        highlightSpecialChars(),
        history(),
        drawSelection(),
        dropCursor(),
        EditorState.allowMultipleSelections.of(true),
        indentOnInput(),
        bracketMatching(),
        closeBrackets(),
        autocompletion({ override: [complete], icons: false, activateOnTyping: true, maxRenderedOptions: 60 }),
        rectangularSelection(),
        highlightActiveLine(),
        highlightSelectionMatches(),
        javascript(),
        syntaxHighlighting(highlight),
        theme,
        EditorView.lineWrapping,
        EditorState.tabSize.of(2),
        keymap.of([
          { key: 'Mod-Enter', run: () => (opts.onRun(), true), preventDefault: true },
          { key: 'Mod-s', run: () => (opts.onSave(), true), preventDefault: true },
          ...closeBracketsKeymap,
          ...defaultKeymap,
          ...searchKeymap,
          ...historyKeymap,
          ...foldKeymap,
          ...completionKeymap,
          indentWithTab,
        ]),
        EditorView.updateListener.of((u) => {
          if (u.docChanged && !silent) opts.onChange(u.state.doc.toString());
        }),
        EditorState.readOnly.of(!!opts.readOnly),
      ],
    }),
  });
  return {
    getValue: () => view.state.doc.toString(),
    setValue(s: string) {
      silent = true;
      view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: s } });
      silent = false;
    },
    setDiagnostics(d: CodeDiagnostic[]) {
      const len = view.state.doc.length;
      const diags: Diagnostic[] = d.map((x) => ({ from: Math.min(x.from, len), to: Math.min(Math.max(x.to, x.from), len), severity: x.severity, message: x.message }));
      view.dispatch(setDiagnostics(view.state, diags));
    },
    focus: () => view.focus(),
    goto(pos: number) {
      const p = Math.max(0, Math.min(pos, view.state.doc.length));
      view.dispatch({ selection: { anchor: p }, effects: EditorView.scrollIntoView(p, { y: 'center' }) });
      view.focus();
    },
    insert(text: string) {
      const sel = view.state.selection.main;
      view.dispatch({ changes: { from: sel.from, to: sel.to, insert: text }, selection: { anchor: sel.from + text.length } });
      view.focus();
    },
    destroy: () => view.destroy(),
  };
}
