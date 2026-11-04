/** Tiny DOM helpers (no framework). */
import { icon } from './icons.ts';
import type { IconName } from './icons.ts';

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

/** Set text only when it changed (avoids needless layout work in hot loops). */
export function setText(el: { textContent: string | null }, s: string): void {
  if (el.textContent !== s) el.textContent = s;
}

export interface BtnOpts {
  icon?: IconName;
  label?: string;
  title?: string;
  cls?: string;
  onclick?: (e: MouseEvent) => void;
  kbd?: string;
}

/** A button with an optional icon and label. */
export function btn(o: BtnOpts): HTMLButtonElement {
  const b = h('button', { class: `btn ${o.cls ?? ''}${o.label ? '' : ' icon'}`.trim(), title: o.title ?? o.label, type: 'button' }) as HTMLButtonElement;
  if (o.icon) b.append(icon(o.icon));
  if (o.label) b.append(h('span', { class: 'tb-label-txt' }, o.label));
  if (o.kbd) b.append(h('span', { class: 'kbd' }, o.kbd));
  if (o.onclick) b.addEventListener('click', o.onclick);
  return b;
}

/** A segmented control. Returns the element and a setter for the selected value. */
export function seg<T extends string>(items: { value: T; label?: string; icon?: IconName; title?: string }[], onPick: (v: T) => void, cls = ''): { el: HTMLElement; set: (v: T) => void } {
  const el = h('div', { class: `seg ${cls}`.trim(), role: 'tablist' });
  const buttons = items.map((it) => {
    const b = h('button', { type: 'button', title: it.title ?? it.label, 'data-v': it.value, onclick: () => onPick(it.value) });
    if (it.icon) b.append(icon(it.icon));
    if (it.label) b.append(h('span', null, it.label));
    el.append(b);
    return b;
  });
  return {
    el,
    set: (v) => buttons.forEach((b) => b.classList.toggle('on', b.dataset.v === v)),
  };
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

export function fmtTime(sec: number): string {
  if (!Number.isFinite(sec)) return '–';
  const s = Math.round(sec);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${String(s % 60).padStart(2, '0')}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, '0')}m`;
}

export function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);
}

let toastHost: HTMLElement | null = null;
export function toast(msg: string, ms = 2600, ic?: IconName): void {
  if (!toastHost || !toastHost.isConnected) {
    toastHost = h('div', { class: 'toasts' });
    document.body.append(toastHost);
  }
  const t = h('div', { class: 'toast' }, ic ? icon(ic) : null, msg);
  toastHost.append(t);
  while (toastHost.childElementCount > 3) toastHost.firstElementChild!.remove();
  setTimeout(() => t.remove(), ms);
}

export function download(name: string, data: string | Uint8Array | Blob, type = 'application/octet-stream'): void {
  const blob = data instanceof Blob ? data : new Blob([typeof data === 'string' ? data : new Uint8Array(data)], { type });
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

export function modal(
  title: string,
  body: Node,
  opts: { size?: 'narrow' | 'wide'; narrow?: boolean; foot?: Node[]; onClose?: () => void } = {},
): { close: () => void; el: HTMLElement } {
  const back = h('div', { class: 'modal-back' });
  const close = () => {
    back.remove();
    document.removeEventListener('keydown', onKey, true);
    opts.onClose?.();
  };
  const onKey = (e: KeyboardEvent) => {
    if (e.key === 'Escape') {
      e.stopPropagation();
      close();
    }
  };
  const size = opts.size ?? (opts.narrow ? 'narrow' : '');
  const m = h(
    'div',
    { class: `modal ${size}`.trim(), role: 'dialog', 'aria-label': title },
    h('div', { class: 'modal-head' }, h('h2', null, title), h('div', { class: 'spacer' }), btn({ icon: 'x', cls: 'ghost sm', title: 'Close (Esc)', onclick: close })),
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

export type MenuItem =
  | { label: string; hint?: string; icon?: IconName; checked?: boolean; disabled?: boolean; action: () => void }
  | { heading: string }
  | 'sep';

export function menu(anchor: HTMLElement, items: MenuItem[], align: 'left' | 'right' = 'right'): void {
  document.querySelectorAll('.menu').forEach((m) => m.remove());
  const r = anchor.getBoundingClientRect();
  const m = h('div', { class: 'menu', role: 'menu' });
  const close = () => {
    m.remove();
    document.removeEventListener('mousedown', outside, true);
    document.removeEventListener('keydown', onKey, true);
  };
  const outside = (e: MouseEvent) => {
    if (!m.contains(e.target as Node) && !anchor.contains(e.target as Node)) close();
  };
  const onKey = (e: KeyboardEvent) => {
    const bs = [...m.querySelectorAll('button')] as HTMLButtonElement[];
    const i = bs.indexOf(document.activeElement as HTMLButtonElement);
    if (e.key === 'Escape') {
      e.stopPropagation();
      close();
      anchor.focus();
    } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      const n = bs.length;
      bs[(i + (e.key === 'ArrowDown' ? 1 : -1) + n) % n]?.focus();
    }
  };
  const hasChecks = items.some((it) => typeof it === 'object' && 'checked' in it && it.checked !== undefined);
  for (const it of items) {
    if (it === 'sep') m.append(h('hr'));
    else if ('heading' in it) m.append(h('div', { class: 'menu-label' }, it.heading));
    else {
      const b = h(
        'button',
        {
          type: 'button',
          role: 'menuitem',
          disabled: it.disabled,
          onclick: () => {
            close();
            it.action();
          },
        },
        hasChecks ? h('span', { class: 'check' }, it.checked ? icon('check') : null) : null,
        !hasChecks && it.icon ? icon(it.icon) : null,
        h('span', null, it.label),
        it.hint ? h('small', null, it.hint) : null,
      );
      m.append(b);
    }
  }
  document.body.append(m);
  const mw = m.offsetWidth;
  const mh = m.offsetHeight;
  const left = align === 'right' ? r.right - mw : r.left;
  m.style.left = Math.max(8, Math.min(window.innerWidth - mw - 8, left)) + 'px';
  m.style.top = (r.bottom + 6 + mh > window.innerHeight - 8 ? Math.max(8, r.top - mh - 6) : r.bottom + 6) + 'px';
  setTimeout(() => {
    document.addEventListener('mousedown', outside, true);
    document.addEventListener('keydown', onKey, true);
  }, 0);
}

/** Section with a small uppercase heading and optional actions on the right. */
export function section(title: string, ic: IconName | null, body: (Node | null | false)[], actions: Node[] = []): HTMLElement {
  return h('div', { class: 'section' }, h('div', { class: 'section-head' }, h('h3', null, ic ? icon(ic) : null, title), h('div', { class: 'spacer' }), ...actions), ...body.filter((b): b is Node => !!b));
}

/** Make a horizontal or vertical splitter resize `target` (keeps the size in localStorage). */
export function makeSplitter(sp: HTMLElement, target: HTMLElement, opts: { dir: 1 | -1; axis: 'x' | 'y'; min: number; max: () => number; key?: string; onResize?: () => void }): void {
  const prop = opts.axis === 'x' ? 'width' : 'height';
  if (opts.key) {
    try {
      const v = Number(localStorage.getItem(opts.key));
      if (v > 0) target.style.flex = `0 0 ${Math.max(opts.min, Math.min(opts.max(), v))}px`;
    } catch {
      /* storage blocked */
    }
  }
  sp.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    sp.classList.add('drag');
    sp.setPointerCapture(e.pointerId);
    const start = opts.axis === 'x' ? e.clientX : e.clientY;
    const startSize = target.getBoundingClientRect()[prop];
    let size = startSize;
    const move = (ev: PointerEvent) => {
      const d = (opts.axis === 'x' ? ev.clientX : ev.clientY) - start;
      size = Math.max(opts.min, Math.min(opts.max(), startSize + d * opts.dir));
      target.style.flex = `0 0 ${size}px`;
      opts.onResize?.();
    };
    const up = () => {
      sp.classList.remove('drag');
      sp.removeEventListener('pointermove', move);
      sp.removeEventListener('pointerup', up);
      if (opts.key)
        try {
          localStorage.setItem(opts.key, String(Math.round(size)));
        } catch {
          /* storage blocked */
        }
    };
    sp.addEventListener('pointermove', move);
    sp.addEventListener('pointerup', up);
  });
}

/** Small persistent settings in localStorage (never throws). */
export const prefs = {
  get<T>(key: string, d: T): T {
    try {
      const v = localStorage.getItem('nb.' + key);
      return v === null ? d : (JSON.parse(v) as T);
    } catch {
      return d;
    }
  },
  set(key: string, v: unknown): void {
    try {
      localStorage.setItem('nb.' + key, JSON.stringify(v));
    } catch {
      /* storage blocked */
    }
  },
};
