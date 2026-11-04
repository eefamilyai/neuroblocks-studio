/**
 * The Language studio: build, train and look inside mini language models.
 *
 *   builder (data, tokenizer, model, training)  |  playground · tokens · data  |  training · attention · embeddings · model
 *
 * Training runs in a worker (LLMClient). The worker sends the weights every
 * ~1.5 s; this side keeps a copy of the model for generating text and for
 * the "under the hood" views.
 */
import type { Shell } from '../../app/Shell.ts';
import { choiceDialog, confirmDialog, promptText } from '../../app/dialogs.ts';
import { btn, download, fmt, fmtSteps, fmtTime, h, makeSplitter, menu, modal, pickFile, prefs, setText, toast } from '../../ui/dom.ts';
import { icon } from '../../ui/icons.ts';
import type { IconName } from '../../ui/icons.ts';
import { LLMClient, defaultThreads } from '../client.ts';
import { getDataset } from '../datasets.ts';
import type { Corpus } from '../datasets.ts';
import { pythonScript } from '../exportPy.ts';
import { GPT, validateConfig } from '../gpt.ts';
import type { FromLLM, LLMProject, LLMStats } from '../protocol.ts';
import { loadModel, saveModel, thin, validateLLMFile } from '../serialize.ts';
import type { LLMFile } from '../serialize.ts';
import { Tokenizer } from '../tokenizer.ts';
import { BuilderPanel } from './BuilderPanel.ts';
import type { ChangeKind } from './BuilderPanel.ts';
import { defaultProject, PRESETS } from './presets.ts';
import type { Preset } from './presets.ts';
import { Playground } from './Playground.ts';
import { KeyboardView } from './KeyboardView.ts';
import { DataView, TokensView } from './TokensView.ts';
import { TrainingView } from './TrainingView.ts';
import { AttentionView } from './AttentionView.ts';
import { EmbeddingView } from './EmbeddingView.ts';
import { ModelView } from './ModelView.ts';

export type LLMLogLevel = 'info' | 'warn' | 'err' | 'good' | 'learn';

/** The trained model (a copy on this thread). */
export interface Trained {
  model: GPT;
  tok: Tokenizer;
  step: number;
  tokens: number;
  /** the settings it was built with (tokenizer + architecture + seed) */
  archKey: string;
  /** the text it was (last) trained on */
  dataKey: string;
  stats: LLMStats | null;
}

const AUTOSAVE_KEY = 'nb.llm.autosave';
const MAX_TEXT = 1_500_000;

function hashStr(s: string): string {
  let x = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    x ^= s.charCodeAt(i);
    x = Math.imul(x, 0x01000193);
  }
  return (x >>> 0).toString(36) + ':' + s.length;
}

export function dataKey(p: LLMProject): string {
  return p.data.dataset === 'custom' ? `custom:${hashStr(p.data.text ?? '')}${p.data.align ? ':lines' : ''}` : `${p.data.dataset}:${p.data.seed}`;
}

export function archKey(p: LLMProject): string {
  const t = p.tokenizer;
  const m = p.model;
  return [t.kind, t.kind === 'char' ? 0 : t.vocabSize, m.ctx, m.d, m.layers, m.heads, m.mlp, m.act, m.pos, m.tie, p.train.seed].join('|');
}

const clone = <T>(o: T): T => JSON.parse(JSON.stringify(o)) as T;

type CenterTab = 'play' | 'keyboard' | 'tokens' | 'data';
type RightTab = 'train' | 'attention' | 'embeddings' | 'model';

function tabBar<T extends string>(defs: [T, string, IconName, string][], pick: (t: T) => void): { el: HTMLElement; bodies: Map<T, HTMLElement>; set: (t: T) => void } {
  const el = h('nav', { class: 'tabs', role: 'tablist' });
  const btns = new Map<T, HTMLElement>();
  const bodies = new Map<T, HTMLElement>();
  for (const [id, label, ic, title] of defs) {
    const b = h('button', { class: 'tab', role: 'tab', title, onclick: () => pick(id) }, icon(ic), h('span', { class: 'lbl' }, label));
    el.append(b);
    btns.set(id, b);
    bodies.set(id, h('div', { class: 'tab-body hidden', role: 'tabpanel' }));
  }
  return {
    el,
    bodies,
    set: (t) => {
      for (const [id, b] of btns) b.classList.toggle('on', id === t);
      for (const [id, b] of bodies) b.classList.toggle('hidden', id !== t);
    },
  };
}

export class LLMStudio {
  project: LLMProject;
  trained: Trained | null = null;
  /** bumps whenever the trained weights change (views recompute lazily) */
  version = 0;
  state: 'idle' | 'running' | 'paused' = 'idle';
  private shell: Shell;
  private client: LLMClient | null = null;
  /** the worker holds a trainer for `trained` (pause/resume/step work without restarting) */
  private hostReady = false;
  private run = 0;
  private pending: { project: LLMProject; archKey: string; dataKey: string; mode: 'new' | 'continue' | 'finetune' } | null = null;
  private corpusCache: { key: string; corpus: Corpus } | null = null;
  private draft: { key: string; tok: Tokenizer } | null = null;
  private draftTimer = 0;
  private autosaveTimer = 0;
  private lastAutosave = 0;
  private visible = false;
  private raf = 0;
  private lastRender = 0;
  private center: CenterTab = 'play';
  private right: RightTab = 'train';
  private tps = 0;
  private finishedAt = -1;
  /** CPU threads for training (0 = automatic) */
  private threads = prefs.get<number>('llmThreads', 0);
  // top bar
  private nameEl: HTMLElement;
  private trainBtn: HTMLButtonElement;
  private stepBtn: HTMLButtonElement;
  private resetBtn: HTMLButtonElement;
  private dot: HTMLElement;
  private phase: HTMLElement;
  private stat: Record<'step' | 'loss' | 'val' | 'tps', HTMLElement>;
  // views
  readonly builder: BuilderPanel;
  readonly playground: Playground;
  readonly keyboard: KeyboardView;
  readonly tokensView: TokensView;
  readonly dataView: DataView;
  readonly trainingView: TrainingView;
  readonly attentionView: AttentionView;
  readonly embeddingView: EmbeddingView;
  readonly modelView: ModelView;
  private centerTabs: ReturnType<typeof tabBar<CenterTab>>;
  private rightTabs: ReturnType<typeof tabBar<RightTab>>;
  private ws: HTMLElement;

  constructor(shell: Shell) {
    this.shell = shell;
    const saved = this.readAutosave();
    this.project = saved?.project ?? defaultProject();

    // ------------------------------------------------------------ top bar
    this.nameEl = h('span', { class: 'name' }, this.project.name);
    const projectBtn = h('button', { class: 'btn ghost project-btn', title: 'Language model: presets, your own text, open, save, rename', onclick: () => this.projectMenu(projectBtn) }, icon('chat'), this.nameEl, icon('down', 14));
    this.trainBtn = btn({ icon: 'play', label: 'Train', cls: 'primary train', title: 'Start / pause training (Ctrl+Enter)', onclick: () => void this.toggleTrain() });
    this.stepBtn = btn({ icon: 'step', cls: 'ghost', title: 'Train exactly ONE step (one batch of text), then pause', onclick: () => void this.stepOnce() });
    this.resetBtn = btn({ icon: 'reset', cls: 'ghost', title: 'New model: throw the trained model away and start again from random weights', onclick: () => void this.newModel() });
    this.dot = h('span', { class: 'dot' });
    this.phase = h('span', null, 'ready');
    const stat = (label: string, title: string, opt = false) => {
      const b = h('b', null, '–');
      return [b, h('div', { class: `stat${opt ? ' opt' : ''}`, title }, h('small', null, label), b)] as const;
    };
    const [sStep, eStep] = stat('steps', 'Training steps done (one step = one batch of text)');
    const [sLoss, eLoss] = stat('loss', 'Training loss (cross-entropy): how surprised the model is by the real next token. Lower is better.');
    const [sVal, eVal] = stat('val loss', 'Loss on text the model never trains on: the honest score');
    const [sTps, eTps] = stat('tokens/s', 'Training speed', true);
    this.stat = { step: sStep, loss: sLoss, val: sVal, tps: sTps };
    shell.llmLeft.append(
      projectBtn,
      h('div', { class: 'tb-sep' }),
      h('div', { class: 'run-group' }, this.trainBtn, this.stepBtn, this.resetBtn),
      h('div', { class: 'status' }, h('div', { class: 'phase', title: 'What is happening right now' }, this.dot, this.phase), eStep, eLoss, eVal, eTps),
    );
    shell.llmRight.append(btn({ icon: 'sparkles', label: 'Presets', cls: 'ghost sm tb-btn', title: 'Ready-made language models to train', onclick: () => this.openPresets() }));

    // ------------------------------------------------------------ views
    const cores = navigator.hardwareConcurrency || 2;
    this.builder = new BuilderPanel(this.project, (_p, kind) => this.settingsChanged(kind), () => this.editText(), {
      threads: () => this.threads,
      setThreads: (n) => {
        this.threads = n;
        prefs.set('llmThreads', n);
        if (this.client) {
          const got = this.client.setThreads(n || defaultThreads());
          this.log(`Training on ${got} thread${got > 1 ? 's' : ''}.`, 'info');
        }
      },
      auto: defaultThreads(),
      max: Math.max(1, Math.min(8, cores)),
    });
    this.playground = new Playground(this);
    this.keyboard = new KeyboardView(this);
    this.tokensView = new TokensView(this);
    this.dataView = new DataView(this);
    this.trainingView = new TrainingView(this);
    this.attentionView = new AttentionView(this);
    this.embeddingView = new EmbeddingView(this);
    this.modelView = new ModelView(this);

    // ------------------------------------------------------------ layout
    const buildPane = h(
      'section',
      { class: 'pane', id: 'llm-build' },
      h('div', { class: 'pane-head' }, icon('hammer', 15), h('span', { class: 'pane-title' }, 'Model builder'), h('div', { class: 'spacer' }), btn({ icon: 'sparkles', label: 'Presets', cls: 'ghost xs', title: 'Start from a ready-made model', onclick: () => this.openPresets() })),
      h('div', { class: 'tab-body' }, this.builder.el),
    );
    this.centerTabs = tabBar<CenterTab>(
      [
        ['play', 'Playground', 'sparkles', 'Generate text with your model and see how sure it is of every token'],
        ['keyboard', 'Keyboard', 'keyboard', 'Word prediction and autocorrect with your model, like a phone keyboard'],
        ['tokens', 'Tokens', 'type', 'How the text is cut into tokens; the vocabulary'],
        ['data', 'Data', 'scroll', 'The training text'],
      ],
      (t) => this.setCenter(t),
    );
    this.centerTabs.bodies.get('play')!.append(this.playground.el);
    this.centerTabs.bodies.get('keyboard')!.append(this.keyboard.el);
    this.centerTabs.bodies.get('tokens')!.append(this.tokensView.el);
    this.centerTabs.bodies.get('data')!.append(this.dataView.el);
    const centerPane = h('section', { class: 'pane', id: 'llm-center' }, h('div', { class: 'pane-head', style: { paddingLeft: '6px' } }, this.centerTabs.el), ...this.centerTabs.bodies.values());
    this.rightTabs = tabBar<RightTab>(
      [
        ['train', 'Training', 'chart', 'Is it learning? Loss curves, live samples'],
        ['attention', 'Attention', 'eye', 'Which earlier tokens each token looks at, per layer and head'],
        ['embeddings', 'Embeddings', 'orbit', 'The learned token vectors, squashed to 2D'],
        ['model', 'Model', 'layers', 'Every weight matrix: sizes and statistics'],
      ],
      (t) => this.setRight(t),
    );
    this.rightTabs.bodies.get('train')!.append(this.trainingView.el);
    this.rightTabs.bodies.get('attention')!.append(this.attentionView.el);
    this.rightTabs.bodies.get('embeddings')!.append(this.embeddingView.el);
    this.rightTabs.bodies.get('model')!.append(this.modelView.el);
    const rightPane = h('section', { class: 'pane', id: 'llm-inspect' }, h('div', { class: 'pane-head', style: { paddingLeft: '6px' } }, this.rightTabs.el), ...this.rightTabs.bodies.values());
    const sp1 = h('div', { class: 'splitter' });
    const sp2 = h('div', { class: 'splitter' });
    this.ws = h('main', { class: 'workspace llm-ws', 'data-mobile': 'center' }, buildPane, sp1, centerPane, sp2, rightPane);
    const mobile = h('div', { class: 'mobile-tabs llm-mobile' });
    for (const [k, l, ic] of [
      ['build', 'Build', 'hammer'],
      ['center', 'Play', 'sparkles'],
      ['inspect', 'Inspect', 'chart'],
    ] as const)
      mobile.append(btn({ icon: ic, label: l, cls: 'sm', onclick: () => this.ws.setAttribute('data-mobile', k) }));
    shell.llmHost.replaceChildren(mobile, this.ws);
    makeSplitter(sp1, buildPane, { dir: 1, axis: 'x', min: 280, max: () => window.innerWidth * 0.45, key: 'nb.w.llmBuild' });
    makeSplitter(sp2, rightPane, { dir: -1, axis: 'x', min: 300, max: () => window.innerWidth * 0.55, key: 'nb.w.llmInspect' });
    this.setCenter('play');
    this.setRight('train');

    // ------------------------------------------------------------ restore
    if (saved?.trained) {
      try {
        const { model, tok } = loadModel(saved.trained, this.project.model);
        this.trained = { model, tok, step: saved.trained.step, tokens: saved.trained.tokens, archKey: archKey(this.project), dataKey: dataKey(this.project), stats: null };
        this.version++;
      } catch (e) {
        console.warn('could not restore the language model', e);
      }
    }
    if (saved) this.trainingView.restore(saved.history, saved.samples);
    this.trainingView.setProbe(this.probeInfo());
    this.setState('idle', this.trained ? `model restored (${fmtSteps(this.trained.step)} steps)` : 'ready');
    this.refreshTopStats();
    this.refreshData();
    this.scheduleDraft(0);
    this.updateBanner();
    window.addEventListener('beforeunload', () => this.autosave());
    if (!saved) this.log('Welcome to the language studio! Press Train to train a tiny GPT on short stories, or pick a preset. Everything (the transformer, backpropagation, the tokenizer) is written from scratch and runs in your browser.', 'learn');
  }

  // =========================================================================
  // Visibility & the render loop
  // =========================================================================

  show(): void {
    this.visible = true;
    if (!this.raf) this.raf = requestAnimationFrame(this.loop);
    this.nameEl.textContent = this.project.name;
  }

  hide(): void {
    this.visible = false;
    this.autosave();
  }

  private loop = (): void => {
    this.raf = 0;
    if (!this.visible) return;
    const now = performance.now();
    if (now - this.lastRender > 60) {
      this.lastRender = now;
      this.render();
    }
    this.raf = requestAnimationFrame(this.loop);
  };

  private render(): void {
    this.playground.render(this.center === 'play');
    this.keyboard.render(this.center === 'keyboard');
    this.tokensView.render(this.center === 'tokens');
    this.dataView.render(this.center === 'data');
    this.trainingView.render(this.right === 'train');
    this.attentionView.render(this.right === 'attention');
    this.embeddingView.render(this.right === 'embeddings');
    this.modelView.render(this.right === 'model');
  }

  setCenter(t: CenterTab): void {
    this.center = t;
    this.centerTabs.set(t);
    this.render();
  }

  setRight(t: RightTab): void {
    this.right = t;
    this.rightTabs.set(t);
    this.render();
  }

  // =========================================================================
  // What the views read
  // =========================================================================

  /** The training text for the current settings (cached). */
  get corpus(): Corpus {
    const key = dataKey(this.project);
    if (this.corpusCache?.key !== key) {
      const p = this.project;
      let corpus: Corpus;
      if (p.data.dataset === 'custom') corpus = { text: p.data.text ?? '', align: p.data.align ? 'start' : undefined };
      else {
        const ds = getDataset(p.data.dataset) ?? getDataset('stories')!;
        corpus = ds.make(p.data.seed);
      }
      this.corpusCache = { key, corpus };
    }
    return this.corpusCache.corpus;
  }

  /** Does the trained model match the current tokenizer/architecture settings? */
  get modelMatches(): boolean {
    return !!this.trained && this.trained.archKey === archKey(this.project);
  }

  /** The tokenizer to show: the trained model's, or one made for the current settings (null while it is being made). */
  get tok(): Tokenizer | null {
    if (this.trained && this.modelMatches) return this.trained.tok;
    return this.draft?.key === this.draftKey() ? this.draft.tok : null;
  }

  /** The dataset's task with right answers, if any. */
  private probeInfo(): { name: string; help: string } | null {
    const pr = this.project.data.dataset === 'custom' ? undefined : getDataset(this.project.data.dataset)?.probe;
    return pr ? { name: pr.name, help: pr.help } : null;
  }

  get valLoss(): number {
    return this.trainingView.lastVal;
  }

  log(msg: string, level: LLMLogLevel = 'info'): void {
    this.trainingView.log(msg, level);
  }

  /** Show the attention for a piece of text (from the playground). */
  inspectAttention(text: string, row = -1): void {
    this.attentionView.setText(text, row);
    this.setRight('attention');
    if (window.innerWidth <= 860) this.ws.setAttribute('data-mobile', 'inspect');
  }

  // =========================================================================
  // Settings changes
  // =========================================================================

  /** Something in the project settings changed (from the builder or another view). */
  settingsChanged(kind: ChangeKind): void {
    if ((kind === 'train' || kind === 'sample') && this.hostReady && this.client) this.client.send({ type: 'config', train: { ...this.project.train }, sample: { ...this.project.sample } });
    if (kind === 'data') this.refreshData();
    if (kind === 'data' || kind === 'tokenizer') this.scheduleDraft();
    if (kind === 'model') this.builder.setTokenizerInfo(this.tokInfo(), this.tok?.size ?? 0);
    this.updateBanner();
    this.tokensView.invalidate();
    this.scheduleAutosave();
  }

  private draftKey(): string {
    const t = this.project.tokenizer;
    return `${t.kind}|${t.kind === 'char' ? 0 : t.vocabSize}|${dataKey(this.project)}`;
  }

  /** Make a tokenizer for the current settings (to preview tokens and count weights before training). */
  private scheduleDraft(ms = 150): void {
    clearTimeout(this.draftTimer);
    this.draftTimer = window.setTimeout(() => {
      const key = this.draftKey();
      if (this.draft?.key !== key && !this.modelMatches) {
        const text = this.corpus.text;
        if (text.length) {
          try {
            this.draft = { key, tok: Tokenizer.train(this.project.tokenizer.kind, text.length > 400_000 ? text.slice(0, 400_000) : text, this.project.tokenizer.vocabSize) };
          } catch (e) {
            this.log(`Could not make the tokenizer: ${(e as Error).message}`, 'err');
          }
        }
      }
      this.builder.setTokenizerInfo(this.tokInfo(), this.tok?.size ?? 0);
      this.tokensView.invalidate();
    }, ms);
  }

  private tokInfo(): string {
    const tok = this.tok;
    if (!tok) return this.corpus.text.length ? 'Making the tokenizer…' : 'No text yet.';
    const kind = tok.kind === 'char' ? 'Every character is a token' : tok.kind === 'word' ? 'Whole words (rare ones are spelled out)' : 'Byte-pair pieces';
    const sample = this.corpus.text.slice(0, 20000);
    const n = tok.encode(sample).length;
    return `${kind}: ${tok.size} different tokens · ${fmt(sample.length / Math.max(1, n), 2)} characters per token on this text${this.modelMatches ? ' (the trained model’s tokenizer)' : ''}.`;
  }

  private refreshData(): void {
    const c = this.corpus;
    const chars = new Set(c.text).size;
    const val = c.valStart !== undefined ? c.text.length - c.valStart : Math.round(c.text.length * 0.1);
    this.builder.setDataInfo(c.text.length ? `${c.text.length.toLocaleString()} characters · ${chars} different characters · last ${val.toLocaleString()} characters kept apart for validation.` : 'No text yet: press "Edit text" and paste some.');
    this.dataView.invalidate();
  }

  private updateBanner(): void {
    const tr = this.trained;
    let msg: string | null = null;
    if (tr && tr.archKey !== archKey(this.project)) msg = 'You changed the model or tokenizer: the next Train starts a NEW model. The current one keeps working until then.';
    else if (tr && tr.dataKey !== dataKey(this.project)) msg = 'New training text: the next Train can fine-tune the current model on it, or start a new one.';
    this.builder.setBanner(msg);
  }

  // =========================================================================
  // Training
  // =========================================================================

  private ensureClient(): LLMClient {
    if (!this.client) {
      this.client = new LLMClient();
      this.client.onMessage = (m) => this.onMsg(m);
      const n = this.client.setThreads(this.threads || defaultThreads());
      if (this.client.mode === 'worker') this.log(`Training runs in background workers on ${n} CPU thread${n > 1 ? 's' : ''} (change it under Training → CPU threads).`, 'info');
      else this.log('Web Workers are not available: training runs on the page itself (slower).', 'warn');
    }
    return this.client;
  }

  private validate(): string | null {
    const p = this.project;
    const text = this.corpus.text;
    if (text.length < 200) return 'The training text is too short: paste at least a few paragraphs (more is better).';
    return validateConfig({ ...p.model, vocab: Math.max(2, this.tok?.size ?? 64) });
  }

  async toggleTrain(): Promise<void> {
    if (this.state === 'running') {
      this.client?.send({ type: 'pause' });
      this.setState('paused', 'paused');
      return;
    }
    await this.begin(false);
  }

  async stepOnce(): Promise<void> {
    if (this.state === 'running') return;
    await this.begin(true);
  }

  /** Start, resume or continue training (asks what to do if the settings changed). */
  private async begin(stepOnly: boolean): Promise<void> {
    const p = this.project;
    const err = this.validate();
    if (err) {
      toast(err, 4000, 'warn');
      this.log(err, 'err');
      return;
    }
    const ak = archKey(p);
    const dk = dataKey(p);
    let mode: 'new' | 'continue' | 'finetune' = 'new';
    const tr = this.trained;
    if (tr) {
      if (tr.archKey !== ak) {
        const ok = await confirmDialog('Start a new model?', `You changed the tokenizer or the architecture, so the current model (${tr.step.toLocaleString()} steps) cannot continue with these settings. Training starts a NEW model from random weights. (Export the current one first if you want to keep it.)`, 'Start a new model');
        if (!ok) return;
        mode = 'new';
      } else if (tr.dataKey !== dk) {
        const c = await choiceDialog('New training text', 'The training text changed. Fine-tune the current model on the new text (it keeps what it already learned and its tokenizer; characters it has never seen are skipped), or start a new model from scratch?', [
          { value: 'new', label: 'New model' },
          { value: 'finetune', label: 'Fine-tune this model', primary: true },
        ]);
        if (!c) return;
        mode = c;
      } else mode = 'continue';
    }
    if (mode !== 'new' && tr) {
      // reached the step limit: train for the same number of steps again
      if (p.train.steps > 0 && tr.step >= p.train.steps) {
        const more = 1000;
        p.train.steps = tr.step + more;
        this.builder.refresh();
        this.log(`Training ${more.toLocaleString()} more steps (until step ${p.train.steps.toLocaleString()}).`, 'info');
      }
    }
    this.finishedAt = -1;
    const client = this.ensureClient();
    if (mode === 'continue' && this.hostReady) {
      client.send({ type: 'config', train: { ...p.train }, sample: { ...p.sample } });
      client.send(stepOnly ? { type: 'step' } : { type: 'resume' });
      this.setState(stepOnly ? 'paused' : 'running', stepOnly ? 'one step…' : 'training');
      return;
    }
    // (re)start the worker's trainer
    if (this.hostReady) client.send({ type: 'stop' });
    this.hostReady = false;
    const corpus = this.corpus;
    const init = mode !== 'new' && tr ? { tokenizer: tr.tok.toJSON(), weights: tr.model.getWeights(), step: tr.step, tokens: tr.tokens } : null;
    if (mode === 'new') {
      this.trained = null;
      this.version++;
      this.trainingView.reset();
      this.playground.reset();
    }
    if (mode === 'finetune') this.log('Fine-tuning: the model continues learning, now on the new text.', 'learn');
    this.run++;
    this.pending = { project: clone(p), archKey: ak, dataKey: dk, mode };
    this.setState('running', p.tokenizer.kind === 'char' || init ? 'starting…' : 'making the tokenizer…');
    client.send({ type: 'start', project: clone(p), text: corpus.text, valStart: corpus.valStart, align: corpus.align, init, stepOnly, run: this.run }, init ? init.weights.map((w) => w.buffer) : []);
    this.updateBanner();
  }

  private stopHost(): void {
    if (this.client && this.hostReady) this.client.send({ type: 'stop' });
    this.run++;
    this.hostReady = false;
    this.pending = null;
  }

  async newModel(): Promise<void> {
    if (this.trained && this.trained.step > 0) {
      const ok = await confirmDialog('New model?', `Throw away the trained model (${this.trained.step.toLocaleString()} steps) and start again from random weights the next time you press Train?`, 'New model', true);
      if (!ok) return;
    }
    this.stopHost();
    this.trained = null;
    this.version++;
    this.finishedAt = -1;
    this.trainingView.reset();
    this.playground.reset();
    this.setState('idle', 'ready');
    this.refreshTopStats();
    this.updateBanner();
    this.scheduleDraft(0);
    this.scheduleAutosave();
  }

  private onMsg(m: FromLLM): void {
    if (m.run !== undefined && m.run !== this.run) return; // from an older run
    switch (m.type) {
      case 'ready': {
        const pend = this.pending;
        this.pending = null;
        this.hostReady = true;
        if (!pend) return;
        const tok = new Tokenizer(m.tokenizer);
        if (pend.mode === 'new' || !this.trained) {
          this.trained = { model: new GPT({ ...pend.project.model, vocab: tok.size }, pend.project.train.seed), tok, step: m.step, tokens: 0, archKey: pend.archKey, dataKey: pend.dataKey, stats: m.stats };
          this.log(`New model: ${m.stats.params.toLocaleString()} weights · ${m.stats.vocab} different tokens · ${m.stats.trainTokens.toLocaleString()} training tokens, ${m.stats.valTokens.toLocaleString()} for validation.`, 'info');
          this.trainingView.setBaseline(Math.log(m.stats.vocab));
        } else {
          this.trained.dataKey = pend.dataKey;
          this.trained.stats = m.stats;
        }
        this.trainingView.setProbe(m.probe);
        this.version++;
        this.finishedAt = -1;
        this.builder.setTokenizerInfo(this.tokInfo(), tok.size);
        this.tokensView.invalidate();
        this.updateBanner();
        if (this.state === 'running') this.setPhase('training');
        break;
      }
      case 'progress': {
        this.trainingView.addProgress(m);
        this.tps = m.tokPerSec;
        this.refreshTopStats();
        if (this.state === 'running') this.setPhase(this.etaText(m.step));
        break;
      }
      case 'sample':
        this.trainingView.addSample(m.step, m.text);
        break;
      case 'weights': {
        const tr = this.trained;
        if (!tr) return;
        try {
          tr.model.setWeights(m.weights);
        } catch {
          return;
        }
        tr.step = m.step;
        tr.tokens = m.tokens;
        this.version++;
        // keep the autosave fresh while training (at most every 20 s)
        if (performance.now() - this.lastAutosave > 20000) this.autosave();
        break;
      }
      case 'log':
        this.log(m.msg, m.level);
        break;
      case 'error':
        this.log(m.msg, 'err');
        toast(m.msg, 5000, 'error');
        this.stopHost();
        this.setState('idle', 'stopped');
        this.updateBanner();
        break;
      case 'paused':
        this.setState('paused', `paused after ${m.reason}`);
        break;
      case 'finished':
        this.finishedAt = this.trained?.step ?? 0;
        this.setState('paused', 'finished');
        this.log(`Training finished: ${m.reason}. Press Train to keep going, or try it in the Playground.`, 'good');
        toast('Training finished. Try your model in the Playground!', 3500, 'check');
        this.autosave();
        break;
    }
  }

  private etaText(step: number): string {
    const p = this.project.train;
    if (!(p.steps > 0) || !(this.tps > 0)) return 'training';
    const left = ((p.steps - step) * p.batch * this.project.model.ctx) / this.tps;
    return left > 1 ? `training · ${fmtTime(left)} left` : 'training';
  }

  private setState(s: 'idle' | 'running' | 'paused', phase: string): void {
    this.state = s;
    this.dot.className = 'dot ' + (s === 'running' ? 'running' : s === 'paused' ? 'paused' : '');
    const label = s === 'running' ? 'Pause' : this.finishedAt >= 0 && this.trained ? 'Train more' : s === 'paused' || this.trained ? 'Resume' : 'Train';
    this.trainBtn.replaceChildren(icon(s === 'running' ? 'pause' : 'play'), h('span', { class: 'tb-label-txt' }, label));
    this.trainBtn.title = label === 'Train more' ? `Training finished: train ${(1000).toLocaleString()} more steps (Ctrl+Enter)` : 'Start / pause training (Ctrl+Enter)';
    this.trainBtn.className = 'btn train ' + (s === 'running' ? 'warn' : 'primary');
    this.stepBtn.classList.toggle('disabled', s === 'running');
    this.setPhase(phase);
    if (s !== 'running') this.tps = 0;
    this.refreshTopStats();
  }

  private setPhase(p: string): void {
    setText(this.phase, p);
    this.phase.parentElement!.title = p;
  }

  private refreshTopStats(): void {
    const tv = this.trainingView;
    setText(this.stat.step, this.trained ? fmtSteps(this.trained.step) : '–');
    setText(this.stat.loss, fmt(tv.lastLoss, 3));
    setText(this.stat.val, fmt(tv.lastVal, 3));
    setText(this.stat.tps, this.tps > 0 ? fmtSteps(this.tps) : '–');
  }

  // =========================================================================
  // Projects: presets, your own text, files, autosave
  // =========================================================================

  private projectMenu(anchor: HTMLElement): void {
    menu(
      anchor,
      [
        { heading: 'Start from a preset' },
        ...PRESETS.map((p) => ({ label: `${p.emoji}  ${p.title}`, hint: p.time, action: () => void this.loadPreset(p) })),
        'sep',
        { label: 'Train on your own text…', icon: 'pencil', action: () => this.editText() },
        { label: 'Open…', icon: 'open', hint: '.json', action: () => void this.openFile() },
        { label: 'Save to a file', icon: 'save', hint: 'Ctrl+S', action: () => this.saveFile() },
        { label: 'Rename…', icon: 'pencil', action: () => void this.rename() },
        'sep',
        { label: 'New model (forget the weights)', icon: 'reset', action: () => void this.newModel() },
      ],
      'left',
    );
  }

  openPresets(): void {
    const grid = h('div', { class: 'mission-grid' });
    const m = modal('Language model presets', h('div', null, h('p', { class: 'dim', style: { marginTop: 0 } }, 'Each preset sets the training text, the tokenizer, the model size and the training settings. They all train in a few minutes in your browser; change anything afterwards in the Model builder.'), grid), { size: 'wide' });
    for (const p of PRESETS)
      grid.append(
        h(
          'button',
          {
            class: 'mcard',
            type: 'button',
            onclick: () => {
              m.close();
              void this.loadPreset(p);
            },
          },
          h('div', { class: 'row' }, h('span', { style: { fontSize: '22px' } }, p.emoji), h('b', null, p.title), h('div', { class: 'spacer' }), h('span', { class: 'badge' }, p.time)),
          h('div', { class: 'dim small' }, p.blurb),
        ),
      );
  }

  private async loadPreset(pr: Preset): Promise<void> {
    if (this.trained && this.trained.step > 0) {
      const ok = await confirmDialog(`Load "${pr.title}"?`, 'This replaces the current model and its settings. (Export or save the current one first if you want to keep it.)', 'Load preset');
      if (!ok) return;
    }
    this.setProject(pr.make(), null);
    this.log(`${pr.emoji} ${pr.title}: ${pr.blurb} Press Train.`, 'learn');
  }

  /** Replace the whole project (and the model, if the file had one). */
  private setProject(p: LLMProject, file: LLMFile | null): void {
    this.stopHost();
    this.project = p;
    this.trained = null;
    this.version++;
    this.finishedAt = -1;
    this.trainingView.reset();
    this.playground.reset();
    if (file?.trained) {
      const { model, tok } = loadModel(file.trained, p.model);
      this.trained = { model, tok, step: file.trained.step, tokens: file.trained.tokens, archKey: archKey(p), dataKey: dataKey(p), stats: null };
      this.trainingView.setBaseline(Math.log(tok.size));
      this.version++;
    }
    if (file) this.trainingView.restore(file.history, file.samples);
    this.trainingView.setProbe(this.probeInfo());
    this.nameEl.textContent = p.name;
    this.builder.setProject(p);
    this.playground.setPrompt(p.sample.prompt);
    this.refreshData();
    this.scheduleDraft(0);
    this.updateBanner();
    this.setState('idle', this.trained ? `model loaded (${fmtSteps(this.trained.step)} steps)` : 'ready');
    this.tokensView.invalidate();
    this.scheduleAutosave(200);
  }

  editText(): void {
    const ta = h('textarea', { class: 'input llm-textarea', spellcheck: 'false', placeholder: 'Paste any text here: stories, poems, song lyrics, recipes, code, a chat log… The more text, the better (at least a few thousand characters).' }) as HTMLTextAreaElement;
    ta.value = this.project.data.dataset === 'custom' ? (this.project.data.text ?? '') : '';
    const count = h('span', { class: 'muted small' });
    const upd = () => setText(count, `${ta.value.length.toLocaleString()} characters${ta.value.length > MAX_TEXT ? ` (only the first ${MAX_TEXT.toLocaleString()} are used)` : ''}`);
    ta.addEventListener('input', upd);
    upd();
    const load = btn({
      icon: 'upload',
      label: 'Load a .txt file',
      cls: 'ghost sm',
      onclick: async () => {
        const f = await pickFile('.txt,.md,.csv,.py,.js,.ts,.json,.html,text/*');
        if (!f) return;
        ta.value = await f.text();
        upd();
      },
    });
    const sample = btn({
      icon: 'scroll',
      label: 'Start from the current text',
      cls: 'ghost sm',
      onclick: () => {
        ta.value = this.corpus.text;
        upd();
      },
    });
    const m = modal('Train on your own text', h('div', { class: 'col', style: { gap: '8px' } }, h('p', { class: 'dim', style: { margin: 0 } }, 'Your text stays in your browser. The model learns to continue text that looks like it. The last 10% is kept apart to measure how well it does on text it never trained on.'), ta, h('div', { class: 'row' }, load, sample, h('div', { class: 'spacer' }), count)), {
      size: 'wide',
      foot: [
        btn({ label: 'Cancel', cls: 'ghost', onclick: () => m.close() }),
        btn({
          label: 'Use this text',
          cls: 'primary',
          onclick: () => {
            const text = ta.value.slice(0, MAX_TEXT);
            if (text.trim().length < 200) {
              toast('Please paste at least a few paragraphs.', 3000, 'warn');
              return;
            }
            m.close();
            this.project.data = { dataset: 'custom', seed: this.project.data.seed, text, align: !!this.project.data.align };
            this.builder.refresh();
            this.settingsChanged('data');
            this.setCenter('data');
            this.log(`Your text: ${text.length.toLocaleString()} characters. Press Train.`, 'info');
          },
        }),
      ],
    });
    setTimeout(() => ta.focus(), 30);
  }

  private async rename(): Promise<void> {
    const n = await promptText('Rename', 'Name of this language model', this.project.name);
    if (!n) return;
    this.project.name = n.slice(0, 80);
    this.nameEl.textContent = this.project.name;
    this.scheduleAutosave();
  }

  private fileData(withModel: boolean, maxPoints = 3000): LLMFile {
    const f: LLMFile = { ...clone(this.project), savedAt: new Date().toISOString() };
    const tr = this.trained;
    if (withModel && tr && tr.archKey === archKey(this.project)) f.trained = saveModel(tr.model, tr.tok, tr.step, tr.tokens);
    f.history = this.trainingView.history(maxPoints);
    f.samples = this.trainingView.samples.slice(-20);
    return f;
  }

  saveFile(): void {
    const f = this.fileData(true);
    download(`${slugify(this.project.name)}.llm.json`, JSON.stringify(f), 'application/json');
    toast(f.trained ? 'Saved the model, its tokenizer and settings' : 'Saved the settings (no trained model yet)', 2600, 'save');
  }

  private async openFile(): Promise<void> {
    const f = await pickFile('.json,application/json');
    if (!f) return;
    try {
      const j = validateLLMFile(JSON.parse(await f.text()), defaultProject());
      if (this.trained && this.trained.step > 0 && !(await confirmDialog('Open this file?', 'It replaces the current model and settings.', 'Open'))) return;
      const { trained: _m, history: _h, samples: _s, savedAt: _t, ...proj } = j;
      this.setProject(proj, j);
      toast(`Opened "${j.name}"${j.trained ? ` (${j.trained.step.toLocaleString()} steps trained)` : ''}`, 2600, 'open');
    } catch (e) {
      toast(`Could not open the file: ${(e as Error).message}`, 4000, 'error');
    }
  }

  private scheduleAutosave(ms = 1500): void {
    clearTimeout(this.autosaveTimer);
    this.autosaveTimer = window.setTimeout(() => this.autosave(), ms);
  }

  autosave(): void {
    clearTimeout(this.autosaveTimer);
    this.lastAutosave = performance.now();
    const tries: (() => LLMFile)[] = [
      () => this.fileData(true, 1500),
      () => this.fileData(false, 1500),
      () => {
        const f = this.fileData(false, 500);
        if (f.data.dataset === 'custom') f.data.text = (f.data.text ?? '').slice(0, 100_000);
        return f;
      },
    ];
    for (const t of tries) {
      try {
        localStorage.setItem(AUTOSAVE_KEY, JSON.stringify(t()));
        return;
      } catch {
        /* too big for local storage: try smaller */
      }
    }
  }

  private readAutosave(): (LLMFile & { project: LLMProject }) | null {
    try {
      const s = localStorage.getItem(AUTOSAVE_KEY);
      if (!s) return null;
      const f = validateLLMFile(JSON.parse(s), defaultProject());
      const { trained: _m, history: _h, samples: _s, savedAt: _t, ...proj } = f;
      return { ...f, project: proj };
    } catch {
      return null;
    }
  }

  // =========================================================================
  // Exports
  // =========================================================================

  exportMenu(anchor: HTMLElement): void {
    const tr = this.trained;
    const name = slugify(this.project.name);
    menu(anchor, [
      { heading: 'Export the language model' },
      {
        label: 'Python script (NumPy + PyTorch)',
        icon: 'file-code',
        hint: '.py',
        disabled: !tr,
        action: () => {
          download(`${name.replace(/-/g, '_')}.py`, pythonScript(tr!.model, tr!.tok, this.project.name, { steps: tr!.step, valLoss: this.valLoss }), 'text/x-python');
          toast('Run it with: python file.py "your prompt"', 3500, 'file-code');
        },
      },
      { label: 'Model, tokenizer and settings', icon: 'json', hint: '.json', action: () => this.saveFile() },
      'sep',
      { label: 'Training log', icon: 'chart', hint: '.csv', disabled: !this.trainingView.hasData, action: () => download(`${name}-training.csv`, this.trainingView.csv(), 'text/csv') },
      { label: 'Live samples', icon: 'chat', hint: '.txt', disabled: !this.trainingView.samples.length, action: () => download(`${name}-samples.txt`, this.trainingView.samples.map((s) => `--- step ${s.step} ---\n${s.text}\n`).join('\n'), 'text/plain') },
      { label: 'Playground text', icon: 'sparkles', hint: '.txt', disabled: !this.playground.text(), action: () => download(`${name}-generated.txt`, this.playground.text(), 'text/plain') },
      { label: 'Training text', icon: 'scroll', hint: '.txt', action: () => download(`${name}-data.txt`, this.corpus.text, 'text/plain') },
    ]);
  }
}

export function slugify(s: string): string {
  return (
    s
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '')
      .slice(0, 40) || 'model'
  );
}
