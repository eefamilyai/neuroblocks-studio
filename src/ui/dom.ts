/** Tiny DOM helpers (no framework). */
export type Child = Node | string | number | null | undefined | false;

export function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, unknown> | null = null,
  ...children: (Child | Child[])[]
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v === undefined || v === null || v === false) continue;
      if (k === 'class') el.className = String(v);
      else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
      else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2).toLowerCase(), v as EventListener);
      else if (k === 'html') el.innerHTML = String(v);
      else el.setAttribute(k, v === true ? '' : String(v));
    }
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export function clear(el: Element): void {
  while (el.firstChild) el.removeChild(el.firstChild);
}

export function fmt(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return '–';
  if (digits === 0) return Math.round(v).toLocaleString();
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  if (a >= 100) return v.toFixed(1);
  return v.toFixed(digits);
}

export function fmtSteps(n: number): string {
  if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 1 : 2) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(n >= 1e5 ? 0 : 1) + 'k';
  return String(Math.round(n));
}

export function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);
}

export function toast(msg: string, ms = 2600): void {
  const t = h('div', { class: 'toast' }, msg);
  document.body.append(t);
  setTimeout(() => t.remove(), ms);
}

export function download(name: string, data: string | Uint8Array, type = 'application/octet-stream'): void {
  const blob = new Blob([typeof data === 'string' ? data : new Uint8Array(data)], { type });
  const url = URL.createObjectURL(blob);
  const a = h('a', { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

export function pickFile(accept: string): Promise<File | null> {
  return new Promise((resolve) => {
    const inp = h('input', { type: 'file', accept, style: { display: 'none' } });
    inp.addEventListener('change', () => {
      resolve(inp.files && inp.files[0] ? inp.files[0] : null);
      inp.remove();
    });
    document.body.append(inp);
    inp.click();
  });
}

export function modal(title: string, body: Node, opts: { narrow?: boolean; foot?: Node[]; onClose?: () => void } = {}): { close: () => void; el: HTMLElement } {
  const back = h('div', { class: 'modal-back' });
  const close = () => {
    back.remove();
    document.removeEventListener('keydown', onKey, true);
    opts.onClose?.();
  };
  const onKey = (e: KeyboardEvent) => {
    if (e.key === 'Escape') close();
  };
  const m = h(
    'div',
    { class: 'modal' + (opts.narrow ? ' narrow' : '') },
    h('div', { class: 'modal-head' }, h('h2', null, title), h('div', { class: 'spacer' }), h('button', { class: 'btn small ghost', onclick: close, title: 'Close (Esc)' }, '✕')),
    h('div', { class: 'modal-body' }, body),
    opts.foot ? h('div', { class: 'modal-foot' }, ...opts.foot) : null,
  );
  back.append(m);
  back.addEventListener('mousedown', (e) => {
    if (e.target === back) close();
  });
  document.addEventListener('keydown', onKey, true);
  document.body.append(back);
  return { close, el: m };
}

export function menu(anchor: HTMLElement, items: ({ label: string; hint?: string; action: () => void } | 'sep')[]): void {
  const r = anchor.getBoundingClientRect();
  const m = h('div', { class: 'menu' });
  const close = () => {
    m.remove();
    document.removeEventListener('mousedown', outside, true);
  };
  const outside = (e: MouseEvent) => {
    if (!m.contains(e.target as Node)) close();
  };
  for (const it of items) {
    if (it === 'sep') m.append(h('hr'));
    else
      m.append(
        h(
          'button',
          {
            onclick: () => {
              close();
              it.action();
            },
          },
          it.label,
          it.hint ? h('small', null, it.hint) : null,
        ),
      );
  }
  document.body.append(m);
  const mw = m.offsetWidth;
  m.style.left = Math.max(8, Math.min(window.innerWidth - mw - 8, r.right - mw)) + 'px';
  m.style.top = r.bottom + 6 + 'px';
  setTimeout(() => document.addEventListener('mousedown', outside, true), 0);
}
