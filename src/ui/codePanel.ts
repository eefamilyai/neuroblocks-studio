/** </> Code tab: everything the blocks turned into, as real code. */
import type { BrainSnapshot } from '../ml/algos/common.ts';
import type { ObsLayout } from '../sim/env.ts';
import type { AgentSpec } from '../sim/types.ts';
import { pytorchModelCode } from '../export/pytorch.ts';
import { btn, clear, escapeHtml, h, section, toast } from './dom.ts';

function highlight(code: string, lang: 'js' | 'py' | 'json'): string {
  const kw =
    lang === 'py'
      ? /\b(import|from|class|def|return|self|super|with|as|for|in|if|else|elif|not|and|or|None|True|False|try|except|lambda)\b/g
      : /\b(function|var|let|const|return|if|else|for|while|new|true|false|null|this|of|in)\b/g;
  let s = escapeHtml(code);
  // order matters: comments & strings first (protect them with placeholders)
  const keep: string[] = [];
  const stash = (m: string, cls: string) => {
    keep.push(`<span class="${cls}">${m}</span>`);
    // encode the index with private-use characters so later regexes (numbers!) can't touch it
    const id = String(keep.length - 1).replace(/\d/g, (d) => String.fromCharCode(0xe000 + Number(d)));
    return `\u0000${id}\u0000`;
  };
  s = s.replace(lang === 'py' ? /(#[^\n]*|"""[\s\S]*?""")/g : /(\/\/[^\n]*)/g, (m) => stash(m, 'com'));
  s = s.replace(/(&#39;[^&\n]*?&#39;|&quot;[^\n]*?&quot;)/g, (m) => stash(m, 'str'));
  s = s.replace(kw, (m) => stash(m, 'kw'));
  s = s.replace(/\b(api)\.(\w+)/g, (_m, a, f) => `${a}.${stash(f, 'fn')}`);
  s = s.replace(/\b(-?\d+\.?\d*(e-?\d+)?)\b/g, (m) => stash(m, 'num'));
  return s.replace(/\u0000([\ue000-\ue009]+)\u0000/g, (_m, i: string) => keep[Number(i.replace(/[\ue000-\ue009]/g, (c) => String(c.charCodeAt(0) - 0xe000)))]);
}

export class CodePanel {
  readonly el: HTMLElement;
  private body: HTMLElement;

  constructor() {
    this.body = h('div');
    this.el = h('div', null, this.body);
  }

  private block(title: string, help: string, code: string, lang: 'js' | 'py' | 'json'): HTMLElement {
    const pre = h('pre', { class: 'code', html: highlight(code, lang) });
    const copy = btn({
      icon: 'copy',
      label: 'Copy',
      cls: 'xs ghost',
      onclick: () => {
        navigator.clipboard?.writeText(code).then(
          () => toast('Copied to the clipboard', 2000, 'copy'),
          () => toast('Copy failed: select the text manually'),
        );
      },
    });
    return section(title, null, [h('div', { class: 'note', style: { margin: '0 0 8px' } }, help), pre], [copy]);
  }

  update(spec: AgentSpec | null, layout: ObsLayout | null, snap: BrainSnapshot | null, program: string | null = null): void {
    clear(this.body);
    if (!spec) {
      this.body.append(h('div', { class: 'note' }, 'Fix the errors in your program to see the generated code.'));
      return;
    }
    if (spec.source !== 'code' && program)
      this.body.append(this.block('Your blocks as a code program', 'Exactly the same robot, written for code mode. Switch the editor to Code to edit it as text: every setting is spelled out.', program, 'js'));
    if (spec.source !== 'code')
      this.body.append(this.block('What runs in every world', 'The event blocks compile to this JavaScript. It runs inside every training world, 20 times per second. "api" is the bridge to the simulation (src/sim/env.ts → ScriptAPI).', spec.script, 'js'));
    if (snap) this.body.append(this.block('The brain as PyTorch', 'The same network written for PyTorch, the most popular ML library. Export → "PyTorch script" gives you this file WITH the trained weights.', pytorchModelCode(snap), 'py'));
    if (layout) {
      const rows = layout.labels.map((l, i) => `x[${String(i).padStart(2)}]  ${l}`).join('\n');
      this.body.append(this.block('Input vector layout', 'The exact order of the numbers fed into the network (the "observation"). Exported brains expect inputs in this order.', rows, 'js'));
    }
    const { script: _script, ...rest } = spec;
    this.body.append(this.block('Compiled spec (JSON)', 'Everything the program configures, as data. This is what is sent to the training worker.', JSON.stringify(rest, (_k, v) => (typeof v === 'number' && !Number.isFinite(v) ? null : v), 2), 'json'));
  }
}
