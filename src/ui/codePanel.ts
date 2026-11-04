/** </> Code tab: everything the blocks turned into, as real code. */
import type { BrainSnapshot } from '../ml/algos/common.ts';
import type { ObsLayout } from '../sim/env.ts';
import type { AgentSpec } from '../sim/types.ts';
import { pytorchModelCode } from '../export/pytorch.ts';
import { clear, escapeHtml, h, toast } from './dom.ts';

function highlight(code: string, lang: 'js' | 'py' | 'json'): string {
  const kw =
    lang === 'py'
      ? /\b(import|from|class|def|return|self|super|with|as|for|in|if|else|elif|not|and|or|None|True|False|try|except|lambda)\b/g
      : /\b(function|var|let|const|return|if|else|for|while|new|true|false|null|this)\b/g;
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
    return h(
      'div',
      { class: 'section' },
      h(
        'h3',
        null,
        title,
        h('span', { class: 'grow' }),
        h(
          'button',
          {
            class: 'btn tiny',
            onclick: () => {
              navigator.clipboard?.writeText(code).then(
                () => toast('Copied to clipboard 📋'),
                () => toast('Copy failed: select the text manually'),
              );
            },
          },
          '📋 copy',
        ),
      ),
      h('div', { class: 'small muted', style: { marginBottom: '6px' } }, help),
      pre,
    );
  }

  update(spec: AgentSpec | null, layout: ObsLayout | null, snap: BrainSnapshot | null): void {
    clear(this.body);
    if (!spec) {
      this.body.append(h('div', { class: 'muted' }, 'Fix the errors in your blocks to see the generated code.'));
      return;
    }
    this.body.append(
      this.block(
        '⚡ Your event blocks → JavaScript',
        'This is the code your reward & event blocks compile to. It runs inside every training world, 20 times per second. "api" is the bridge to the simulation (see src/sim/env.ts → ScriptAPI).',
        spec.script,
        'js',
      ),
    );
    if (snap) this.body.append(this.block('🐍 The brain as PyTorch', 'The same network written for PyTorch, the most popular ML library. Export → "PyTorch script" gives you this file WITH the trained weights.', pytorchModelCode(snap), 'py'));
    if (layout) {
      const rows = layout.labels.map((l, i) => `x[${String(i).padStart(2)}]  ${l}`).join('\n');
      this.body.append(this.block('🔢 Input vector layout', 'The exact order of the numbers fed into the network (the "observation"). Exported brains expect inputs in this order.', rows, 'js'));
    }
    const { script: _script, ...rest } = spec;
    this.body.append(this.block('📦 Compiled spec (JSON)', 'Everything the blocks configure, as data. This is what is sent to the training worker.', JSON.stringify(rest, null, 2), 'json'));
  }
}
