/**
 * The application frame: top bar, the three panes (editor | world |
 * inspector), the console dock and the mobile tab bar. Pure layout: the
 * App fills it with content and wires up the buttons.
 */
import { btn, clear, h, makeSplitter, prefs, seg, setText } from '../ui/dom.ts';
import { icon } from '../ui/icons.ts';
import type { IconName } from '../ui/icons.ts';

export type Tab = 'train' | 'brain' | 'senses' | 'math' | 'code' | 'experiments';
export type EditorMode = 'blocks' | 'code';
export type LogLevel = 'info' | 'warn' | 'err' | 'good' | 'learn';

export interface ShellActions {
  train(): void;
  step(): void;
  stop(): void;
  resetBrain(): void;
  projectMenu(anchor: HTMLElement): void;
  missions(): void;
  lab(): void;
  learn(): void;
  exportMenu(anchor: HTMLElement): void;
  moreMenu(anchor: HTMLElement): void;
  setEditorMode(m: EditorMode): void;
  setTab(t: Tab): void;
  toggleProblems(): void;
  layoutChanged(): void;
}

const TABS: [Tab, string, IconName, string][] = [
  ['train', 'Training', 'chart', 'Is it learning? Curves, reward breakdown, checkpoints, evaluation'],
  ['brain', 'Brain', 'brain', 'The neural network: neurons, weights, activations'],
  ['senses', 'Senses', 'senses', 'The exact numbers going in and coming out of the brain'],
  ['math', 'Math', 'sigma', 'The learning algorithm, step by step, with live numbers'],
  ['code', 'Code', 'file-code', 'Generated code: JavaScript, PyTorch, the input layout'],
  ['experiments', 'Experiments', 'tubes', 'Compare settings: train several brains side by side'],
];

export class Shell {
  readonly root: HTMLElement;
  // top bar
  readonly projectName: HTMLElement;
  readonly trainBtn: HTMLButtonElement;
  readonly stepBtn: HTMLButtonElement;
  readonly stopBtn: HTMLButtonElement;
  readonly resetBtn: HTMLButtonElement;
  readonly exportBtn: HTMLButtonElement;
  private dot: HTMLElement;
  private phase: HTMLElement;
  private stats: Record<'steps' | 'episodes' | 'reward' | 'success' | 'sps', HTMLElement>;
  // editor
  readonly workspace: HTMLElement;
  readonly editorPane: HTMLElement;
  readonly editorBody: HTMLElement;
  readonly blocklyHost: HTMLElement;
  readonly codeHost: HTMLElement;
  readonly editorTools: HTMLElement;
  readonly editorTitle: HTMLElement;
  readonly statusPill: HTMLButtonElement;
  readonly problems: HTMLElement;
  private modeSeg: { el: HTMLElement; set: (v: EditorMode) => void };
  // world
  readonly worldPane: HTMLElement;
  readonly viewportHost: HTMLElement;
  readonly console: HTMLElement;
  private dock: HTMLElement;
  // inspector
  readonly inspector: HTMLElement;
  private tabBtns = new Map<Tab, HTMLElement>();
  private tabBodies = new Map<Tab, HTMLElement>();
  tab: Tab = 'train';

  constructor(root: HTMLElement, a: ShellActions) {
    this.root = root;
    clear(root);
    // ------------------------------------------------------------ top bar
    this.projectName = h('span', { class: 'name' }, 'Untitled');
    const projectBtn = h('button', { class: 'btn ghost project-btn', title: 'Project: new, open, save, rename', onclick: () => a.projectMenu(projectBtn) }, icon('puzzle'), this.projectName, icon('down', 14));
    this.trainBtn = btn({ icon: 'play', label: 'Train', cls: 'primary train', title: 'Start / pause training (Ctrl+Enter)', onclick: () => a.train() });
    this.stepBtn = btn({ icon: 'step', cls: 'ghost', title: 'Train exactly ONE update, then pause: see what a single learning step changes', onclick: () => a.step() });
    this.stopBtn = btn({ icon: 'stop', cls: 'ghost', title: 'Stop training (keeps the brain)', onclick: () => a.stop() });
    this.resetBtn = btn({ icon: 'reset', cls: 'ghost', title: 'New brain: throw the brain away and start from random weights', onclick: () => a.resetBrain() });
    this.dot = h('span', { class: 'dot' });
    this.phase = h('span', null, 'ready');
    const stat = (k: string, label: string, opt = false) => {
      const b = h('b', null, '–');
      return [b, h('div', { class: `stat${opt ? ' opt' : ''}` }, h('small', null, label), b)] as const;
    };
    const [sSteps, eSteps] = stat('steps', 'steps');
    const [sEps, eEps] = stat('episodes', 'episodes', true);
    const [sRew, eRew] = stat('reward', 'reward');
    const [sSucc, eSucc] = stat('success', 'success');
    const [sSps, eSps] = stat('sps', 'steps/s', true);
    this.stats = { steps: sSteps, episodes: sEps, reward: sRew, success: sSucc, sps: sSps };
    this.exportBtn = btn({ icon: 'download', label: 'Export', cls: 'sm', title: 'Export the brain (JSON, ONNX, PyTorch) or training logs', onclick: () => a.exportMenu(this.exportBtn) });
    const moreBtn = btn({ icon: 'more', cls: 'ghost sm', title: 'Settings, help, keyboard shortcuts', onclick: () => a.moreMenu(moreBtn) });
    const topbar = h(
      'header',
      { class: 'topbar' },
      h('div', { class: 'brand', title: 'NeuroBlocks Studio' }, h('span', { class: 'mark' }, icon('brain')), h('span', { class: 'word' }, 'NeuroBlocks')),
      h('div', { class: 'tb-sep opt' }),
      projectBtn,
      h('div', { class: 'tb-sep' }),
      h('div', { class: 'run-group' }, this.trainBtn, this.stepBtn, this.stopBtn, this.resetBtn),
      h('div', { class: 'status' }, h('div', { class: 'phase', title: 'What the learning algorithm is doing right now' }, this.dot, this.phase), eSteps, eEps, eRew, eSucc, eSps),
      h('div', { class: 'spacer' }),
      btn({ icon: 'target', label: 'Missions', cls: 'ghost sm tb-btn', title: 'Ready-made missions and the zones of the world', onclick: () => a.missions() }),
      btn({ icon: 'flask', label: 'Lab', cls: 'ghost sm tb-btn', title: 'Neuron Lab: watch a network learn to classify points (supervised learning)', onclick: () => a.lab() }),
      btn({ icon: 'learn', label: 'Learn', cls: 'ghost sm tb-btn', title: 'The ideas behind it all, and a suggested path', onclick: () => a.learn() }),
      this.exportBtn,
      moreBtn,
    );

    // ------------------------------------------------------------ editor pane
    this.modeSeg = seg<EditorMode>(
      [
        { value: 'blocks', label: 'Blocks', icon: 'blocks', title: 'Snap blocks together (Scratch-style)' },
        { value: 'code', label: 'Code', icon: 'code', title: 'Write the program in JavaScript: everything the blocks do, and more' },
      ],
      (m) => a.setEditorMode(m),
    );
    this.editorTitle = h('span', { class: 'pane-title' });
    this.editorTools = h('div', { class: 'row', style: { gap: '2px' } });
    this.statusPill = h('button', { class: 'status-pill', title: 'Problems in your program', onclick: () => a.toggleProblems() }) as HTMLButtonElement;
    this.blocklyHost = h('div', { id: 'blockly' });
    this.codeHost = h('div', { class: 'code-host hidden' });
    this.editorBody = h('div', { class: 'editor-body' }, this.blocklyHost, this.codeHost);
    this.problems = h('div', { class: 'problems hidden' });
    this.editorPane = h('section', { class: 'pane', id: 'pane-editor' }, h('div', { class: 'pane-head' }, this.modeSeg.el, this.editorTitle, h('div', { class: 'spacer' }), this.statusPill, this.editorTools), this.editorBody, this.problems);

    // ------------------------------------------------------------ world pane
    this.viewportHost = h('div', { class: 'viewport' });
    this.console = h('div', { class: 'console' });
    const filter = seg(
      [
        { value: 'all', label: 'All' },
        { value: 'learn', label: 'Tips' },
        { value: 'warn', label: 'Warnings' },
        { value: 'err', label: 'Errors' },
      ],
      (v) => {
        filter.set(v);
        this.console.className = `console${v === 'all' ? '' : ' f-' + v}`;
      },
      'sm',
    );
    filter.set('all');
    const collapse = btn({ icon: 'down', cls: 'ghost xs icon', title: 'Hide / show the console' });
    this.dock = h(
      'div',
      { class: 'dock' },
      h('div', { class: 'dock-head' }, icon('terminal', 14), h('span', { class: 'dock-title' }, 'Console'), h('div', { class: 'spacer' }), filter.el, btn({ icon: 'trash', cls: 'ghost xs icon', title: 'Clear the console', onclick: () => clear(this.console) }), collapse),
      this.console,
    );
    const toggleDock = (force?: boolean) => {
      const c = this.dock.classList.toggle('collapsed', force);
      collapse.replaceChildren(icon(c ? 'up' : 'down'));
      prefs.set('dockCollapsed', c);
      a.layoutChanged();
    };
    collapse.addEventListener('click', () => toggleDock());
    if (prefs.get('dockCollapsed', false)) toggleDock(true);
    const hsplit = h('div', { class: 'hsplit' });
    this.worldPane = h('section', { class: 'pane', id: 'pane-world' }, this.viewportHost, hsplit, this.dock);

    // ------------------------------------------------------------ inspector
    const tabs = h('nav', { class: 'tabs', role: 'tablist' });
    this.inspector = h('section', { class: 'pane', id: 'pane-inspector' }, h('div', { class: 'pane-head', style: { paddingLeft: '6px' } }, tabs));
    for (const [id, label, ic, title] of TABS) {
      const b = h('button', { class: 'tab', role: 'tab', title, onclick: () => a.setTab(id) }, icon(ic), h('span', { class: 'lbl' }, label));
      tabs.append(b);
      this.tabBtns.set(id, b);
      const body = h('div', { class: 'tab-body hidden', role: 'tabpanel' });
      this.tabBodies.set(id, body);
      this.inspector.append(body);
    }

    const sp1 = h('div', { class: 'splitter' });
    const sp2 = h('div', { class: 'splitter' });
    this.workspace = h('main', { class: 'workspace', 'data-mobile': 'world' }, this.editorPane, sp1, this.worldPane, sp2, this.inspector);
    const mobile = h('div', { class: 'mobile-tabs' });
    for (const [k, l, ic] of [
      ['editor', 'Program', 'puzzle'],
      ['world', 'World', 'map'],
      ['inspector', 'Inspect', 'chart'],
    ] as const)
      mobile.append(btn({ icon: ic, label: l, cls: 'sm', onclick: () => this.mobileShow(k) }));
    root.append(topbar, mobile, this.workspace);
    makeSplitter(sp1, this.editorPane, { dir: 1, axis: 'x', min: 260, max: () => window.innerWidth * 0.6, key: 'nb.w.editor', onResize: () => a.layoutChanged() });
    makeSplitter(sp2, this.inspector, { dir: -1, axis: 'x', min: 280, max: () => window.innerWidth * 0.55, key: 'nb.w.inspector', onResize: () => a.layoutChanged() });
    makeSplitter(hsplit, this.dock, { dir: -1, axis: 'y', min: 60, max: () => this.worldPane.clientHeight * 0.7, key: 'nb.h.dock', onResize: () => a.layoutChanged() });
  }

  mobileShow(k: 'editor' | 'world' | 'inspector'): void {
    this.workspace.setAttribute('data-mobile', k);
  }

  /** Put content into an inspector tab. */
  setTabContent(t: Tab, el: HTMLElement): void {
    this.tabBodies.get(t)!.replaceChildren(el);
  }

  showTab(t: Tab): void {
    this.tab = t;
    for (const [id, b] of this.tabBtns) b.classList.toggle('on', id === t);
    for (const [id, b] of this.tabBodies) b.classList.toggle('hidden', id !== t);
  }

  setEditorMode(m: EditorMode): void {
    this.modeSeg.set(m);
    this.blocklyHost.classList.toggle('hidden', m !== 'blocks');
    this.codeHost.classList.toggle('hidden', m !== 'code');
  }

  /** Focus mode: hide the editor and inspector to give the world the whole window. */
  setFocus(on: boolean): void {
    this.workspace.classList.toggle('focus', on);
  }

  setRunState(s: 'idle' | 'running' | 'paused', phase: string): void {
    this.dot.className = 'dot ' + (s === 'running' ? 'running' : s === 'paused' ? 'paused' : '');
    this.trainBtn.replaceChildren(icon(s === 'running' ? 'pause' : 'play'), h('span', null, s === 'running' ? 'Pause' : s === 'paused' ? 'Resume' : 'Train'));
    this.trainBtn.className = 'btn train ' + (s === 'running' ? 'warn' : 'primary');
    this.stepBtn.classList.toggle('disabled', s === 'running');
    this.stopBtn.classList.toggle('disabled', s === 'idle');
    this.setPhase(phase);
  }

  setPhase(p: string): void {
    setText(this.phase, p);
    this.phase.parentElement!.title = p;
  }

  setStats(s: { steps: string; episodes: string; reward: string; success: string; sps: string; good?: boolean }): void {
    setText(this.stats.steps, s.steps);
    setText(this.stats.episodes, s.episodes);
    setText(this.stats.reward, s.reward);
    setText(this.stats.success, s.success);
    setText(this.stats.sps, s.sps);
    this.stats.success.classList.toggle('good', !!s.good);
  }

  /** The compile status pill + problem list. */
  setProblems(errors: { msg: string; line?: number; pos?: number }[], warnings: { msg: string; line?: number; pos?: number }[], onPick: ((pos: number) => void) | null, expand: boolean | null): void {
    const pill = this.statusPill;
    pill.className = 'status-pill' + (errors.length ? ' err' : warnings.length ? ' warn' : '');
    pill.replaceChildren(icon(errors.length ? 'error' : warnings.length ? 'warn' : 'check'), errors.length ? `${errors.length} error${errors.length > 1 ? 's' : ''}` : warnings.length ? `${warnings.length} warning${warnings.length > 1 ? 's' : ''}` : 'ready');
    pill.title = errors.length || warnings.length ? 'Click to show / hide the problems' : 'No problems: press Train';
    clear(this.problems);
    const row = (m: { msg: string; line?: number; pos?: number }, kind: 'err' | 'warn') => {
      const clickable = onPick && m.pos !== undefined;
      this.problems.append(h('div', { class: `problem ${kind}${clickable ? ' click' : ''}`, onclick: clickable ? () => onPick!(m.pos!) : null }, icon(kind === 'err' ? 'error' : 'warn'), h('span', null, m.msg), m.line ? h('span', { class: 'ln' }, `line ${m.line}`) : null));
    };
    errors.forEach((e) => row(e, 'err'));
    warnings.forEach((w) => row(w, 'warn'));
    if (expand !== null) this.problems.classList.toggle('hidden', !expand || (!errors.length && !warnings.length));
  }

  toggleProblems(): void {
    if (!this.problems.childElementCount) return;
    this.problems.classList.toggle('hidden');
  }

  log(msg: string, level: LogLevel = 'info'): void {
    const c = this.console;
    const t = new Date();
    const ts = `${String(t.getHours()).padStart(2, '0')}:${String(t.getMinutes()).padStart(2, '0')}:${String(t.getSeconds()).padStart(2, '0')}`;
    const stick = c.scrollHeight - c.scrollTop - c.clientHeight < 30;
    c.append(h('div', { class: `ln ${level}` }, h('span', { class: 't' }, ts), h('span', null, msg)));
    while (c.childElementCount > 400) c.firstElementChild!.remove();
    if (stick) c.scrollTop = c.scrollHeight;
  }
}
