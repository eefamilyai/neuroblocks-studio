/**
 * NeuroBlocks Studio: the application. Connects the program editor (blocks
 * or code), the 3D world, the training worker and the "under the hood"
 * panels.
 */
import type { CompileResult } from '../blocks/compiler.ts';
import { MISSIONS, getMission } from '../blocks/missions.ts';
import type { Mission } from '../blocks/missions.ts';
import { blocksToCode, CODE_TEMPLATE } from '../code/fromBlocks.ts';
import { CUSTOM_ALGO_TEMPLATE } from '../code/templates.ts';
import { DOCS, DOC_GROUPS } from '../code/docs.ts';
import type { CodeEditor } from './codeEditor.ts';
import { WorldRuntime, editsKey, mergeEdits } from '../sim/runtime.ts';
import type { WorldEdits } from '../sim/runtime.ts';
import { obsLayout } from '../sim/env.ts';
import type { ObsLayout } from '../sim/env.ts';
import { actionSpaceOf } from '../sim/spec.ts';
import type { AgentSpec } from '../sim/types.ts';
import { getWorld } from '../sim/worldCache.ts';
import { Brain } from '../ml/algos/common.ts';
import type { BrainSnapshot } from '../ml/algos/common.ts';
import { MLP } from '../ml/nn.ts';
import { RNG } from '../ml/rng.ts';
import { jointActionCount, policyOutputSize } from '../ml/policy.ts';
import { Showcase } from '../showcase.ts';
import type { DriveMode } from '../showcase.ts';
import { TrainerClient } from '../worker/client.ts';
import type { FromWorker } from '../worker/protocol.ts';
import { TrainingPanel } from '../ui/trainingPanel.ts';
import type { CheckpointInfo } from '../ui/trainingPanel.ts';
import { SensesPanel } from '../ui/sensesPanel.ts';
import { BrainPanel } from '../ui/brainPanel.ts';
import { MathPanel } from '../ui/mathPanel.ts';
import { CodePanel } from '../ui/codePanel.ts';
import { btn, clear, download, fmt, fmtSteps, h, menu, modal, pickFile, prefs, toast } from '../ui/dom.ts';
import { exportOnnx } from '../export/onnx.ts';
import { pytorchScript, slug } from '../export/pytorch.ts';
import { loadLocal, markMissionDone, missionsDone, saveLocal, validateBrain, validateProject, welcomeSeen } from '../project/storage.ts';
import type { ProjectFile } from '../project/storage.ts';
import { Shell } from './Shell.ts';
import type { AppMode, EditorMode, LogLevel, Tab } from './Shell.ts';
import { BlocksEditor } from './BlocksEditor.ts';
import { WorldView } from './WorldView.ts';
import { ExperimentsPanel } from './Experiments.ts';
import { confirmCode, confirmDialog, explainExports, openLearn, openMissions, openSettings, openShortcuts, openWelcome, promptText } from './dialogs.ts';
import type { Settings } from './dialogs.ts';

type CodeModule = typeof import('../code/runtime.ts');
type LLMStudioT = import('../llm/ui/LLMStudio.ts').LLMStudio;
type EditorModule = typeof import('./codeEditor.ts');

interface Checkpoint extends CheckpointInfo {
  snap: BrainSnapshot;
}

const DEFAULT_SETTINGS: Settings = { quality: 'auto', maxFps: 0, throttle: 0, autosave: true, showFps: false };

export class App {
  readonly world = getWorld();
  edits: WorldEdits = { objects: [], walls: [] };
  rt: WorldRuntime;
  private rtKey = '';
  compiled: CompileResult | null = null;
  private draftKey = '';
  layout: ObsLayout | null = null;
  private trainingKey = '';
  /** the newest brain from training (or loaded) */
  snap: BrainSnapshot | null = null;
  brain: Brain | null = null;
  /** a checkpoint / experiment brain being watched instead of the newest one */
  private watched: { snap: BrainSnapshot; label: string } | null = null;
  state: 'idle' | 'running' | 'paused' = 'idle';
  private trainMode: 'rl' | 'bc' = 'rl';
  client = new TrainerClient();
  private evalClient: TrainerClient | null = null;
  showcase = new Showcase();
  missionId: string | null = null;
  projectName = 'My robot';
  private runCounter = 0;
  private stats = { steps: 0, episodes: 0, sps: 0, phase: 'ready' };
  private compileTimer = 0;
  private autosaveTimer = 0;
  private lastSuccessMilestone = 0;
  private firstSuccessShown = false;
  private checkpoints: Checkpoint[] = [];
  private settings: Settings;
  private focus = false;
  // editor
  editorMode: EditorMode = 'blocks';
  code = '';
  /** code from a file someone else wrote is not run until you say so */
  private codeTrusted = true;
  /** the code was edited by hand since it was generated from blocks */
  private codeEdited = false;
  private codeMod: CodeModule | null = null;
  private editorMod: EditorModule | null = null;
  private codeEditor: CodeEditor | null = null;
  private codeLoading: Promise<void> | null = null;
  private problemsOpen: boolean | null = null;
  // UI
  readonly shell: Shell;
  readonly blocks: BlocksEditor;
  readonly worldView: WorldView;
  private trainingPanel: TrainingPanel;
  private sensesPanel = new SensesPanel();
  private brainPanel = new BrainPanel();
  private mathPanel = new MathPanel();
  private codePanel = new CodePanel();
  private experiments: ExperimentsPanel;
  private lastPanels = 0;
  private codeDirty = true;
  /** the language-model studio (loaded on demand) */
  private llm: LLMStudioT | null = null;
  private llmLoading: Promise<LLMStudioT> | null = null;

  constructor(root: HTMLElement) {
    this.settings = { ...DEFAULT_SETTINGS, ...prefs.get<Partial<Settings>>('settings', {}) };
    this.rt = new WorldRuntime(this.world, this.edits);
    this.shell = new Shell(root, {
      train: () => this.toggleTrain(),
      step: () => this.stepOnce(),
      stop: () => this.stopTraining(),
      resetBrain: () => this.resetBrain(),
      projectMenu: (a) => this.projectMenu(a),
      missions: () => this.openMissions(),
      lab: () => this.openLab(),
      learn: () => openLearn(() => this.openMissions(), () => this.openLab(), this.shell.appMode),
      exportMenu: (a) => (this.shell.appMode === 'language' ? this.llm?.exportMenu(a) : this.exportMenu(a)),
      moreMenu: (a) => this.moreMenu(a),
      setEditorMode: (m) => void this.setEditorMode(m),
      setTab: (t) => this.setTab(t),
      toggleProblems: () => {
        this.shell.toggleProblems();
        this.problemsOpen = !this.shell.problems.classList.contains('hidden');
      },
      layoutChanged: () => {
        this.blocks?.resize();
        this.worldView?.view.resize();
      },
      setAppMode: (m) => void this.setAppMode(m),
    });
    this.trainingPanel = new TrainingPanel({
      onWatchCheckpoint: (i) => this.watchCheckpoint(i),
      onResumeCheckpoint: (i) => void this.resumeCheckpoint(i),
      onEvaluate: (n, det) => this.evaluate(n, det),
    });
    this.experiments = new ExperimentsPanel({
      spec: () => (this.compiled && !this.compiled.errors.length ? this.compiled.spec : null),
      edits: () => this.edits,
      watch: (snap, label) => this.watch(snap, label),
      log: (m, l) => this.log(m, l),
    });
    this.shell.setTabContent('train', this.trainingPanel.el);
    this.shell.setTabContent('brain', this.brainPanel.el);
    this.shell.setTabContent('senses', this.sensesPanel.el);
    this.shell.setTabContent('math', this.mathPanel.el);
    this.shell.setTabContent('code', this.codePanel.el);
    this.shell.setTabContent('experiments', this.experiments.el);

    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const app = this;
    this.blocks = new BlocksEditor(this.shell.blocklyHost);
    this.blocks.onChange = () => {
      if (this.editorMode === 'blocks') this.scheduleCompile(200);
    };
    this.worldView = new WorldView(
      this.shell.viewportHost,
      {
        showcase: this.showcase,
        world: this.world,
        get edits() {
          return app.edits;
        },
        isTraining: () => this.state === 'running',
        trainingSteps: () => this.stats.steps,
        setZone: (id) => this.setZone(id),
        setSpawn: (x, z, zone) => this.setSpawn(x, z, zone),
        editsChanged: (msg) => this.onEditsChanged(msg),
        setDriveMode: (m) => this.setDriveMode(m),
        toggleRecord: () => this.toggleRecord(),
        startCloning: () => this.startCloning(),
        ghostsChanged: (on) => this.client.send({ type: 'ghosts', on }),
        log: (m, l) => this.log(m, l),
        frame: () => this.frame(),
      },
      this.settings.quality,
    );
    this.worldView.onFocusToggle = () => this.toggleFocus();
    this.worldView.view.maxFps = this.settings.maxFps;
    this.worldView.toggles.fps = this.settings.showFps;
    this.worldView.setRuntime(this.rt);
    this.rtKey = editsKey(this.edits);
    this.setEditorTools();

    this.client.onMessage = (m) => this.onWorker(m);
    this.client.onFallback = () => this.log('Background workers are not available here, so training runs on the main thread (a bit slower, same results).', 'warn');
    this.showcase.onStep = (info) => {
      this.worldView.onStep(info);
      this.sensesPanel.addEvents(info.events, info.reward);
    };
    this.showcase.onBeforeStep = () => this.worldView.objLayer.snapshotPrev();
    this.showcase.onError = (m) => this.log(m, 'err');
    this.showcase.onLog = (m) => this.log(`📝 ${m}`, 'info');
    this.initKeyboard();
    this.setTab('train');
    this.setState('idle');
    this.setDriveMode('ai');

    const saved = loadLocal();
    let restored = false;
    if (saved) {
      try {
        this.loadProject(saved, { quiet: true, trusted: true });
        restored = true;
      } catch (e) {
        console.warn('Could not restore the autosave', e);
        this.log(`Your last session could not be restored (${(e as Error).message}). Starting fresh.`, 'warn');
      }
    }
    if (!restored) this.loadMission(MISSIONS[0], true);
    if (prefs.get<AppMode>('appMode', 'robots') === 'language') void this.setAppMode('language');
    if (!welcomeSeen()) setTimeout(() => this.welcome(), 400);
    this.log(`Welcome to NeuroBlocks Studio! Training runs in ${this.client.mode === 'worker' ? 'a background worker thread' : 'the main thread (no Web Worker support)'}.`, 'learn');
    window.addEventListener('beforeunload', () => this.autosave());
    setInterval(() => this.autosave(), 15000);
    // fetch the on-demand parts in the background once the page is idle: code mode opens
    // instantly later, and the installed app (service worker cache) also works offline
    const idle = (f: () => void) => ('requestIdleCallback' in window ? window.requestIdleCallback(f, { timeout: 4000 }) : setTimeout(f, 1000));
    setTimeout(
      () =>
        idle(() => {
          void this.loadCodeModules().catch(() => {});
          void import('../ui/lab.ts').catch(() => {});
          void import('../llm/ui/LLMStudio.ts').catch(() => {});
        }),
      6000,
    );
  }

  // =========================================================================
  // Layout & tabs
  // =========================================================================

  private setTab(t: Tab): void {
    this.shell.showTab(t);
    this.showcase.wantTrace = t === 'brain' || t === 'senses';
    this.showcase.wantSaliency = t === 'senses';
    this.showcase.computeTrace();
    if (t === 'code' && this.codeDirty) this.refreshCodePanel();
    if (t === 'math') this.mathPanel.render(true, true);
  }

  /** Switch between the robot studio and the language-model studio (loaded on first use). */
  private async setAppMode(m: AppMode): Promise<void> {
    prefs.set('appMode', m);
    if (m === 'robots') {
      this.shell.setAppMode('robots');
      this.llm?.hide();
      setTimeout(() => {
        this.worldView.view.resize();
        this.blocks.resize();
      }, 0);
      return;
    }
    this.shell.setAppMode('language');
    if (!this.llm) {
      if (!this.llmLoading) {
        this.shell.llmHost.replaceChildren(h('div', { class: 'loading', style: { position: 'relative', flex: '1' } }, 'Loading the language studio…'));
        this.llmLoading = import('../llm/ui/LLMStudio.ts').then(({ LLMStudio }) => new LLMStudio(this.shell));
      }
      try {
        this.llm = await this.llmLoading;
      } catch (e) {
        this.llmLoading = null;
        this.log(`Could not load the language studio: ${(e as Error).message}`, 'err');
        toast('Could not load the language studio (offline?)', 3500, 'error');
        this.shell.setAppMode('robots');
        prefs.set('appMode', 'robots');
        return;
      }
    }
    if (this.shell.appMode === 'language') this.llm.show();
  }

  private toggleFocus(): void {
    this.focus = !this.focus;
    this.shell.setFocus(this.focus);
    this.worldView.setFocusIcon(this.focus);
    setTimeout(() => {
      this.worldView.view.resize();
      this.blocks.resize();
    }, 0);
  }

  private setEditorTools(): void {
    const t = this.shell.editorTools;
    clear(t);
    if (this.editorMode === 'blocks') {
      t.append(
        btn({ icon: 'sparkles', cls: 'ghost sm icon', title: 'Arrange the blocks neatly', onclick: () => this.blocks.tidy() }),
        btn({ icon: 'undo', cls: 'ghost sm icon', title: 'Undo (Ctrl+Z)', onclick: () => this.blocks.undo(false) }),
        btn({ icon: 'redo', cls: 'ghost sm icon', title: 'Redo (Ctrl+Shift+Z)', onclick: () => this.blocks.undo(true) }),
      );
    } else {
      t.append(
        btn({ icon: 'book', cls: 'ghost sm icon', title: 'Reference: every function you can use', onclick: () => this.openDocs() }),
        btn({ icon: 'blocks', cls: 'ghost sm icon', title: 'Replace the code with the current blocks', onclick: () => void this.codeFromBlocks(true) }),
      );
    }
  }

  // =========================================================================
  // Program: compile blocks or code
  // =========================================================================

  private scheduleCompile(ms = 200): void {
    clearTimeout(this.compileTimer);
    this.compileTimer = window.setTimeout(() => this.compile(), ms);
  }

  private compile(): void {
    let r: CompileResult;
    try {
      if (this.editorMode === 'code') {
        if (!this.codeMod) return;
        if (!this.codeTrusted) {
          this.shell.setProblems([{ msg: 'This code has not been run yet. Press Train (or the button in the banner) to trust and run it.' }], [], null, true);
          return;
        }
        r = this.codeMod.compileCode(this.code, this.edits.objects);
      } else {
        r = this.blocks.compile(this.edits.objects);
        this.blocks.updateLabels(r);
      }
    } catch (e) {
      this.shell.setProblems([{ msg: `Compiler crashed: ${(e as Error).message}` }], [], null, true);
      return;
    }
    this.compiled = r;
    this.showProblems(r);
    if (!r.errors.length) this.applyDraftSpec(r.spec);
    this.scheduleAutosave();
  }

  private showProblems(r: CompileResult): void {
    const d = r.diagnostics;
    const toItem = (msg: string, severity: 'error' | 'warning') => {
      if (!d) return { msg };
      const hit = d.find((x) => x.severity === severity && msg.endsWith(x.message));
      return hit ? { msg: hit.message, line: hit.line, pos: hit.from } : { msg };
    };
    const errors = r.errors.map((e) => toItem(e, 'error'));
    const warnings = r.warnings.slice(0, 12).map((w) => toItem(w, 'warning'));
    const expand = errors.length ? true : this.problemsOpen;
    this.shell.setProblems(errors, warnings, (pos) => this.codeEditor?.goto(pos), expand);
    if (this.codeEditor && d) this.codeEditor.setDiagnostics(d);
  }

  private specKey(s: AgentSpec): string {
    const { warnings: _w, name: _n, ...rest } = s;
    return JSON.stringify(rest);
  }

  private applyDraftSpec(spec: AgentSpec): void {
    const key = this.specKey(spec);
    const changed = key !== this.draftKey;
    const prevZone = this.showcase.spec?.zone ?? null;
    this.draftKey = key;
    if (this.state !== 'idle') {
      // training continues with its own spec; tell the user how to apply edits
      this.worldView.setBanner(key !== this.trainingKey ? 'Program changed while training: stop and train again (or New brain) to use it' : null);
      return;
    }
    this.worldView.setBanner(null);
    if (!changed) return;
    // the program may change the world itself (code mode world.add / clearZone)
    const merged = mergeEdits(this.edits, spec);
    const ek = editsKey(merged);
    if (ek !== this.rtKey) {
      this.rtKey = ek;
      this.rt = new WorldRuntime(this.world, merged);
      this.worldView.setRuntime(this.rt);
    }
    this.layout = obsLayout(spec);
    this.showcase.setSpec(spec, this.rt);
    this.worldView.bindSpec(spec, prevZone !== spec.zone);
    this.sensesPanel.setLayout(this.layout);
    // keep the current brain if it still fits, otherwise grow a fresh random one
    if (!this.snap || !this.snapFits(this.snap, spec)) this.setSnapshot(this.freshSnapshot(spec), false);
    else this.refreshShowcaseBrain();
    this.trainingPanel.setAlgo(spec.train.algo);
    this.mathPanel.algo = spec.train.algo;
    this.codeDirty = true;
    if (this.shell.tab === 'code') this.refreshCodePanel();
  }

  // ---- editor mode -----------------------------------------------------------

  private async loadCodeModules(): Promise<void> {
    if (this.codeMod && this.editorMod) return;
    if (!this.codeLoading)
      this.codeLoading = (async () => {
        const [cm, em] = await Promise.all([import('../code/runtime.ts'), import('./codeEditor.ts')]);
        this.codeMod = cm;
        this.editorMod = em;
      })().catch((e) => {
        this.codeLoading = null; // allow a retry (e.g. when back online)
        throw e;
      });
    await this.codeLoading;
  }

  private async ensureCodeEditor(): Promise<CodeEditor> {
    if (this.codeEditor) return this.codeEditor;
    const host = this.shell.codeHost;
    const spinner = h('div', { class: 'loading' }, 'Loading the code editor…');
    host.append(spinner);
    try {
      await this.loadCodeModules();
    } catch (e) {
      spinner.textContent = 'Could not load the code editor (are you offline?). Try again in a moment.';
      throw e;
    }
    spinner.remove();
    this.codeEditor = this.editorMod!.createCodeEditor(host, {
      doc: this.code,
      onChange: (doc) => {
        this.code = doc;
        this.codeEdited = true;
        this.scheduleCompile(400);
      },
      onRun: () => this.toggleTrain(),
      onSave: () => this.saveProjectFile(),
    });
    return this.codeEditor;
  }

  private async setEditorMode(m: EditorMode, opts: { quiet?: boolean } = {}): Promise<void> {
    if (m === this.editorMode && (m === 'blocks' || this.codeEditor)) return;
    if (m === 'code') {
      const fresh = !this.code.trim() || !this.codeEdited;
      this.editorMode = 'code';
      this.shell.setEditorMode('code');
      this.setEditorTools();
      const ed = await this.ensureCodeEditor();
      if (fresh && this.compiled?.events) {
        // convert the current blocks
        this.code = blocksToCode(this.compiled);
        this.codeEdited = false;
        ed.setValue(this.code);
        if (!opts.quiet) this.log('Your blocks are now code: the same robot, every setting spelled out. Ctrl+Space shows what you can type; 📖 lists every function.', 'learn');
      } else if (!this.code.trim()) {
        this.code = CODE_TEMPLATE;
        ed.setValue(this.code);
      }
      this.draftKey = '';
      this.compile();
      setTimeout(() => ed.focus(), 30);
    } else {
      this.editorMode = 'blocks';
      this.shell.setEditorMode('blocks');
      this.setEditorTools();
      setTimeout(() => this.blocks.resize(), 0);
      if (this.codeEdited && !opts.quiet) this.log('Back to blocks. Code changes do not turn back into blocks, but your code is kept: switch to Code to see it again.', 'info');
      this.draftKey = '';
      this.compile();
    }
    this.scheduleAutosave();
  }

  private async codeFromBlocks(ask: boolean): Promise<void> {
    if (ask && this.codeEdited && !(await confirmDialog('Replace your code?', 'This rewrites the code from the current blocks. Changes you typed will be lost.', 'Replace', true))) return;
    const r = this.blocks.compile(this.edits.objects);
    this.code = blocksToCode(r);
    this.codeEdited = false;
    this.codeTrusted = true;
    this.codeEditor?.setValue(this.code);
    this.compile();
  }

  private openDocs(): void {
    const list = h('div', { class: 'docs' });
    const search = h('input', { class: 'input docs-search', placeholder: 'Search functions…' }) as HTMLInputElement;
    let m: { close: () => void } | null = null;
    const render = () => {
      clear(list);
      const q = search.value.trim().toLowerCase();
      for (const g of DOC_GROUPS) {
        const items = DOCS.filter((d) => d.group === g && (!q || `${d.name} ${d.doc}`.toLowerCase().includes(q)));
        if (!items.length) continue;
        list.append(h('h4', null, g));
        for (const d of items)
          list.append(
            h(
              'div',
              { class: 'doc-item' },
              h('div', { class: 'row' }, h('code', null, d.name, h('span', { class: 'muted' }, d.sig)), h('div', { class: 'spacer' }), d.snippet ? btn({ icon: 'plus', cls: 'xs ghost', label: 'Insert', onclick: () => (m?.close(), this.codeEditor?.insert(d.snippet!.replace(/\$\{([^}]*)\}/g, '$1').replace(/\t/g, '  '))) }) : null),
              h('p', null, d.doc),
            ),
          );
      }
    };
    search.addEventListener('input', render);
    render();
    m = modal('Code reference', h('div', null, search, list), { size: 'narrow' });
    setTimeout(() => search.focus(), 30);
  }

  // =========================================================================
  // Brains
  // =========================================================================

  private outSizeFor(spec: AgentSpec): number {
    const space = actionSpaceOf(spec);
    return spec.train.algo === 'dqn' ? jointActionCount(space) : policyOutputSize(space);
  }

  /** Does this brain fit this spec (same inputs, outputs, layers, algorithm family)? */
  private snapFits(s: BrainSnapshot, spec: AgentSpec): boolean {
    const L = obsLayout(spec);
    if (s.net.inSize !== L.size) return false;
    // your own algorithm builds its own network: only the inputs and outputs have to match
    if (spec.train.algo === 'custom') return s.kind === 'policy' && s.space.kind === 'discrete' && s.net.layers[s.net.layers.length - 1].outSize === this.outSizeFor(spec);
    if ((s.kind === 'q') !== (spec.train.algo === 'dqn')) return false;
    if (s.net.layers[s.net.layers.length - 1].outSize !== this.outSizeFor(spec)) return false;
    if ((s.space.kind === 'continuous') !== (spec.actionMode === 'continuous')) return false;
    const hidden = s.net.layers.slice(0, -1);
    if (hidden.length !== spec.brain.layers.length) return false;
    return hidden.every((l, i) => l.outSize === spec.brain.layers[i].units && l.activation === spec.brain.layers[i].activation);
  }

  private freshSnapshot(spec: AgentSpec): BrainSnapshot {
    const L = obsLayout(spec);
    const space = actionSpaceOf(spec);
    const isQ = spec.train.algo === 'dqn';
    const rng = new RNG(spec.train.seed * 7 + 11);
    const layers = spec.brain.layers.length ? spec.brain.layers : [{ units: 32, activation: 'tanh' as const }];
    const net = new MLP(L.size, layers, this.outSizeFor(spec), 'linear', rng, isQ ? 0.1 : 0.01);
    const critic = spec.train.algo === 'ppo' || (spec.train.algo === 'reinforce' && spec.train.baseline) ? new MLP(L.size, layers, 1, 'linear', rng) : null;
    return {
      format: 'neuroblocks-brain',
      version: 1,
      algo: spec.train.algo,
      kind: isQ ? 'q' : 'policy',
      net: net.toJSON(),
      critic: critic ? critic.toJSON() : null,
      logStd: space.kind === 'continuous' ? new Array(space.branches.length).fill(-0.5) : null,
      obsNorm: null,
      space,
      obsLabels: L.labels,
      steps: 0,
      updates: 0,
    };
  }

  private setSnapshot(s: BrainSnapshot, fromTraining: boolean): void {
    this.snap = s;
    this.brain = new Brain(s);
    if (!fromTraining) this.codeDirty = true;
    if (!this.watched) this.refreshShowcaseBrain();
  }

  /** Show either the newest brain or the one being watched. */
  private refreshShowcaseBrain(): void {
    const b = this.watched ? new Brain(this.watched.snap) : this.brain;
    this.showcase.setBrain(b);
    this.showcase.label = this.watched?.label ?? '';
    this.brainPanel.setBrain(b);
    this.mathPanel.algo = this.trainMode === 'bc' && this.state !== 'idle' ? 'bc' : (b?.snap.algo ?? 'ppo');
  }

  /** Watch an older / experimental brain in the 3D view (training is not affected). */
  private watch(snap: BrainSnapshot | null, label = ''): void {
    this.watched = snap ? { snap, label } : null;
    if (snap && !this.showcase.brainFits()) {
      const b = new Brain(snap);
      if (b.net.inSize !== this.showcase.env?.obs.length) {
        toast('That brain does not fit the current program (different senses)');
        this.watched = null;
      }
    }
    this.refreshShowcaseBrain();
    if (this.watched) this.log(`Watching ${label} in the 3D view. Training (if any) continues with the newest brain.`, 'info');
  }

  private watchCheckpoint(i: number | null): void {
    if (i === null) return this.watch(null);
    const c = this.checkpoints[i];
    if (c) this.watch(c.snap, `checkpoint ${fmtSteps(c.steps)}`);
  }

  private async resumeCheckpoint(i: number): Promise<void> {
    const c = this.checkpoints[i];
    if (!c) return;
    if (!(await confirmDialog('Train from this checkpoint?', `The brain goes back to how it was after ${fmtSteps(c.steps)} steps. Checkpoints after it are dropped.`, 'Go back'))) return;
    if (this.state !== 'idle') this.stopTraining(true);
    this.watched = null;
    this.checkpoints = this.checkpoints.slice(0, i + 1);
    this.setSnapshot(c.snap, false);
    this.pushCheckpoints();
    this.trainingPanel.watchLatest();
    this.log(`Back to the brain from ${fmtSteps(c.steps)} steps. Press Train to continue from there.`, 'good');
  }

  private addCheckpoint(s: BrainSnapshot): void {
    const last = this.checkpoints[this.checkpoints.length - 1];
    const gap = Math.max(4000, s.steps * 0.08);
    if (last && s.steps - last.steps < gap && s.steps > last.steps) {
      // keep the latest snapshot in the last slot
      return;
    }
    const r = this.trainingPanel.recent(100);
    this.checkpoints.push({ steps: s.steps, updates: s.updates, meanReward: r.mean, success: r.success, snap: s });
    if (this.checkpoints.length > 40) {
      // thin out: keep every other one (but always the first, best and last)
      const best = this.bestCheckpoint();
      this.checkpoints = this.checkpoints.filter((c, k, a) => k === 0 || k === a.length - 1 || k % 2 === 0 || c === best);
    }
    this.pushCheckpoints();
  }

  private bestCheckpoint(): Checkpoint | null {
    let best: Checkpoint | null = null;
    for (const c of this.checkpoints) if (Number.isFinite(c.meanReward) && (!best || c.meanReward > best.meanReward)) best = c;
    return best;
  }

  private pushCheckpoints(): void {
    const best = this.bestCheckpoint();
    this.trainingPanel.setCheckpoints(this.checkpoints.map((c) => ({ steps: c.steps, updates: c.updates, meanReward: c.meanReward, success: c.success, best: c === best })));
  }

  // =========================================================================
  // Training
  // =========================================================================

  private async ensureTrustedCode(): Promise<boolean> {
    if (this.editorMode !== 'code' || this.codeTrusted) return true;
    const ok = await confirmCode(this.projectName, this.code);
    if (ok) {
      this.codeTrusted = true;
      this.compile();
    }
    return ok;
  }

  private async toggleTrain(): Promise<void> {
    if (this.state === 'running') {
      this.client.send({ type: 'pause' });
      this.setState('paused');
      this.log('Training paused. The robot keeps its brain.', 'info');
      return;
    }
    if (!(await this.ensureTrustedCode())) return;
    if (this.state === 'paused') {
      if (this.trainingKey !== this.draftKey) {
        this.stopTraining(true);
        this.startTraining('rl');
        return;
      }
      this.client.send({ type: 'resume' });
      this.setState('running');
      this.log('Training resumed.', 'info');
      return;
    }
    this.startTraining('rl');
  }

  private async stepOnce(): Promise<void> {
    if (this.state === 'running') return;
    if (!(await this.ensureTrustedCode())) return;
    if (this.state === 'paused' && this.trainingKey === this.draftKey) {
      this.client.send({ type: 'step' });
      this.setState('running');
      return;
    }
    if (this.state === 'paused') this.stopTraining(true);
    this.startTraining('rl', true);
  }

  private startTraining(mode: 'rl' | 'bc', stepOnly = false): void {
    const r = this.compiled;
    if (!r || r.errors.length) {
      toast('Fix the problems in your program first', 2600, 'error');
      this.shell.problems.classList.remove('hidden');
      return;
    }
    const spec = r.spec;
    this.applyDraftSpec(spec);
    let init: BrainSnapshot | null = null;
    let demos = null;
    if (mode === 'bc') {
      demos = this.showcase.demos();
      if (!demos) {
        toast('Record some driving first (You → Record)', 2600, 'gamepad');
        return;
      }
      this.newRun(spec, 'bc');
      this.log(`Behaviour cloning on ${demos.count} recorded steps: the network learns to copy your driving.`, 'good');
    } else if (this.snap && this.snap.steps > 0 && this.snapFits(this.snap, spec)) {
      init = this.snap;
      this.log(`Continuing to train the current brain (${fmtSteps(this.snap.steps)} steps of experience) with ${spec.train.algo.toUpperCase()}${stepOnly ? ', one update' : ''}.`, 'info');
    } else {
      this.newRun(spec);
      this.log(`Training a fresh brain with ${spec.train.algo.toUpperCase()} in ${spec.train.numEnvs} parallel worlds (${this.world.zones.find((z) => z.id === spec.zone)?.name})${stepOnly ? ': one update, then pause' : ''}.`, 'good');
    }
    this.trainingKey = this.specKey(spec);
    this.trainMode = mode;
    this.trainingPanel.setAlgo(mode === 'bc' ? 'bc' : spec.train.algo);
    this.mathPanel.algo = mode === 'bc' ? 'bc' : spec.train.algo;
    if (this.watched) {
      this.watched = null;
      this.trainingPanel.watchLatest();
    }
    this.client.send({ type: 'ghosts', on: this.worldView.toggles.ghosts });
    this.client.send({ type: 'throttle', sps: this.settings.throttle });
    this.client.send({ type: 'start', spec, edits: this.edits, init, mode, demos, stepOnly });
    this.setState('running');
    if (this.showcase.mode === 'paused') this.setDriveMode('ai');
  }

  private newRun(spec: AgentSpec, algo?: string): void {
    this.runCounter++;
    this.trainingPanel.newRun(`run ${this.runCounter}`);
    this.worldView.heat.clear();
    this.stats.steps = 0;
    this.stats.episodes = 0;
    this.lastSuccessMilestone = 0;
    this.firstSuccessShown = false;
    this.checkpoints = [];
    this.pushCheckpoints();
    this.trainingPanel.setAlgo(algo ?? spec.train.algo);
    this.updateTopStats();
  }

  private stopTraining(silent = false): void {
    if (this.state === 'idle') return;
    this.client.send({ type: 'stop' });
    this.setState('idle');
    this.worldView.setGhosts(null);
    if (!silent) this.log('Training stopped. The brain is kept: press Train to continue, New brain to start over.', 'info');
    if (this.compiled && !this.compiled.errors.length) {
      this.draftKey = '';
      this.applyDraftSpec(this.compiled.spec);
    }
  }

  private async resetBrain(): Promise<void> {
    const wasRunning = this.state === 'running';
    if (this.state !== 'idle') this.stopTraining(true);
    const spec = this.compiled?.spec;
    if (!spec || this.compiled?.errors.length) return;
    this.watched = null;
    this.setSnapshot(this.freshSnapshot(spec), false);
    this.newRun(spec);
    this.showcase.reset();
    this.worldView.clearTrail();
    this.log('Fresh random brain. Everything it learned is gone (the previous curve stays dashed for comparison).', 'warn');
    if (wasRunning) this.startTraining('rl');
  }

  private setState(s: 'idle' | 'running' | 'paused'): void {
    this.state = s;
    const phase = s === 'idle' ? (this.snap && this.snap.steps ? `stopped · ${fmtSteps(this.snap.steps)} steps learned` : 'ready') : s === 'paused' ? 'paused' : this.stats.phase;
    this.shell.setRunState(s, phase);
  }

  private updateTopStats(): void {
    const r = this.trainingPanel.recent(100);
    this.shell.setStats({
      steps: fmtSteps(this.stats.steps),
      episodes: this.stats.episodes.toLocaleString(),
      reward: r.n ? fmt(r.mean, 2) : '–',
      success: r.n ? `${(r.success * 100).toFixed(0)}%` : '–',
      sps: this.state === 'running' ? fmtSteps(this.stats.sps) : '–',
      good: r.success >= 0.8,
    });
  }

  private onWorker(m: FromWorker): void {
    switch (m.type) {
      case 'started':
        this.log(`Worker started ${m.algo.toUpperCase()} with ${m.obsSize} inputs.`, 'info');
        break;
      case 'progress': {
        this.stats.steps = m.steps;
        this.stats.episodes = m.episodeCount;
        this.stats.sps = m.sps;
        this.stats.phase = m.phase;
        this.trainingPanel.add(m.episodes, m.metrics);
        this.trainingPanel.setFitness(m.fitness);
        this.trainingPanel.lr = m.lr;
        this.mathPanel.info = m.explain;
        this.mathPanel.phase = m.phase;
        for (const mt of m.metrics) this.logMetric(mt);
        this.checkMilestones();
        this.updateTopStats();
        if (this.state === 'running') this.shell.setPhase(m.phase);
        break;
      }
      case 'brain':
        this.setSnapshot(m.snap, true);
        if (this.trainMode === 'rl' && m.snap.steps > 0) this.addCheckpoint(m.snap);
        if (this.shell.tab === 'code' && m.snap.updates <= 1) this.refreshCodePanel();
        else this.codeDirty = true;
        break;
      case 'ghosts':
        if (this.state === 'running') this.worldView.setGhosts(m.data);
        break;
      case 'log':
        this.log(m.msg, m.level);
        break;
      case 'error':
        this.log(`⛔ ${m.msg}`, 'err');
        toast(m.msg, 5000, 'error');
        this.setState('idle');
        break;
      case 'paused':
        this.setState('paused');
        this.log(`Paused after ${m.reason}. Look at the Math and Brain tabs to see what changed, then step again.`, 'learn');
        break;
      case 'finished':
        this.log(`Training finished: ${m.reason}.`, 'good');
        toast(`Training finished: ${m.reason}`, 3500, 'check');
        this.setState('idle');
        this.worldView.setGhosts(null);
        break;
    }
  }

  private lastMetricLog = 0;
  private logMetric(m: Record<string, number>): void {
    const now = performance.now();
    if (now - this.lastMetricLog < 5000) return;
    this.lastMetricLog = now;
    const r = this.trainingPanel.recent(100);
    const sr = Number.isFinite(r.success) ? ` · success ${(r.success * 100).toFixed(0)}%` : '';
    const extra = m.entropy !== undefined ? ` · entropy ${fmt(m.entropy, 2)}` : m.epsilon !== undefined ? ` · ε ${fmt(m.epsilon, 2)}` : m.bestFitness !== undefined ? ` · best fitness ${fmt(m.bestFitness, 2)}` : m.accuracy !== undefined ? ` · accuracy ${(m.accuracy * 100).toFixed(0)}%` : '';
    this.log(`update ${m.update} @ ${fmtSteps(m.step)} steps · mean reward ${fmt(r.mean, 2)}${sr}${extra}`, 'info');
  }

  private checkMilestones(): void {
    const eps = this.trainingPanel.episodes;
    if (!this.firstSuccessShown && eps.some((e) => e.success)) {
      this.firstSuccessShown = true;
      this.log('First success! The robot reached its goal at least once. Now it has to learn to do it reliably.', 'good');
    }
    const r = this.trainingPanel.recent(50);
    if (r.n >= 50) {
      for (const ms of [0.25, 0.5, 0.8, 0.95]) {
        if (r.success >= ms && this.lastSuccessMilestone < ms) {
          this.lastSuccessMilestone = ms;
          this.log(`Success rate passed ${ms * 100}% (last 50 episodes).`, 'good');
          if (ms >= 0.8 && this.missionId) {
            const was = missionsDone().has(this.missionId);
            markMissionDone(this.missionId);
            if (!was) toast(`Mission complete: ${getMission(this.missionId)?.title}!`, 4000, 'trophy');
          }
        }
      }
    }
  }

  private evaluate(episodes: number, deterministic: boolean): void {
    const spec = this.showcase.spec;
    const snap = this.watched?.snap ?? this.snap;
    if (!spec || !snap || !this.compiled || this.compiled.errors.length) return void toast('Nothing to evaluate yet');
    if (!this.showcase.brainFits()) return void toast('This brain does not fit the current program');
    this.evalClient?.terminate();
    const c = new TrainerClient();
    this.evalClient = c;
    this.trainingPanel.evalProgress(0, episodes);
    c.onMessage = (m) => {
      if (m.type === 'evalProgress') this.trainingPanel.evalProgress(m.done, m.total);
      else if (m.type === 'evaluated') {
        this.trainingPanel.showEval(m.result);
        this.log(`Evaluation of the brain @ ${fmtSteps(m.result.brainSteps)} steps: success ${(m.result.successRate * 100).toFixed(0)}%, reward ${fmt(m.result.meanReward, 2)} ± ${fmt(m.result.stdReward, 2)} over ${m.result.episodes} episodes.`, 'good');
        c.terminate();
        if (this.evalClient === c) this.evalClient = null;
      } else if (m.type === 'error') {
        this.trainingPanel.evalFailed();
        this.log(`⛔ ${m.msg}`, 'err');
        c.terminate();
      }
    };
    c.send({ type: 'evaluate', spec, edits: this.edits, snap, episodes, deterministic });
  }

  private startCloning(): void {
    if (this.state !== 'idle') this.stopTraining(true);
    this.showcase.recording = false;
    this.setDriveMode('ai');
    this.startTraining('bc');
  }

  // =========================================================================
  // World: zones, spawn, build mode, driving
  // =========================================================================

  private setZone(id: string): void {
    const z = this.world.zones.find((q) => q.id === id);
    if (!z) return;
    if (this.editorMode === 'blocks') {
      if (!this.blocks.setRobotField('ZONE', id)) return void toast('Add a robot block first (Robot category)');
    } else if (!this.editCode((src) => this.replaceRobotOption(src, 'zone', `'${id}'`))) return;
    this.log(`Robot moved to ${z.emoji} ${z.name}: ${z.blurb}`, 'info');
    if (this.worldView.view.mode === 'follow' || this.worldView.view.mode === 'orbit') this.worldView.setCam('zone');
    else this.worldView.flyToZone(id, true);
  }

  private setSpawn(x: number, z: number, zone: string): void {
    if (this.editorMode === 'blocks') {
      if (!this.blocks.robot()) return;
      this.blocks.setRobotField('SPAWN', 'point');
      this.blocks.setRobotField('X', x);
      this.blocks.setRobotField('Z', z);
      if (zone !== 'world') this.blocks.setRobotField('ZONE', zone);
    } else {
      const ok = this.editCode((src) => {
        let s = this.replaceRobotOption(src, 'spawn', `{ x: ${x}, z: ${z} }`);
        if (s !== null && zone !== 'world') s = this.replaceRobotOption(s, 'zone', `'${zone}'`);
        return s;
      });
      if (!ok) return;
    }
    this.log(`Start point set to x=${x}, z=${z} (${zone}).`, 'info');
  }

  /** Change one option inside robot({...}) in the code (adds it if missing). */
  private replaceRobotOption(src: string, key: string, value: string): string | null {
    const m = /robot\(\s*\{/.exec(src);
    if (!m) return null;
    const start = m.index + m[0].length;
    let depth = 1;
    let end = start;
    for (; end < src.length && depth > 0; end++) {
      const c = src[end];
      if (c === '{' || c === '[' || c === '(') depth++;
      else if (c === '}' || c === ']' || c === ')') depth--;
    }
    const body = src.slice(start, end - 1);
    const re = new RegExp(`(\\b${key}\\s*:\\s*)(\\{[^}]*\\}|'[^']*'|"[^"]*"|[^,}\\n]+)`);
    const nb = re.test(body) ? body.replace(re, `$1${value}`) : ` ${key}: ${value},${body.startsWith('\n') ? '' : ' '}${body.trimStart()}`;
    return src.slice(0, start) + nb + src.slice(end - 1);
  }

  private editCode(f: (src: string) => string | null): boolean {
    const next = f(this.code);
    if (next === null) {
      toast('Add a robot({ ... }) call to your code first');
      return false;
    }
    this.code = next;
    this.codeEdited = true;
    this.codeEditor?.setValue(next);
    this.scheduleCompile(50);
    return true;
  }

  private onEditsChanged(msg: string): void {
    this.rtKey = '';
    this.draftKey = '';
    if (this.state === 'idle') this.compile();
    else {
      // the 3D view shows the new world right away; training keeps the old one until restarted
      const merged = mergeEdits(this.edits, this.compiled?.spec ?? null);
      this.rtKey = editsKey(merged);
      this.rt = new WorldRuntime(this.world, merged);
      this.worldView.setRuntime(this.rt);
      this.worldView.setBanner('World changed: stop and train again to use your builds');
    }
    this.log(`${msg} (${this.edits.objects.length} objects, ${this.edits.walls.length} blocks placed)`, 'info');
    this.scheduleAutosave();
  }

  private setDriveMode(m: DriveMode): void {
    this.showcase.mode = m;
    if (m !== 'drive') this.showcase.recording = false;
    this.updateDriveUi();
    if (m === 'drive') {
      this.worldView.view.renderer.domElement.focus();
      this.log('You are driving: W/↑ forward, S/↓ back, A/D or ←/→ turn, Q/E slide, Space jump, F kick. Click the 3D view first if keys do nothing.', 'learn');
    }
  }

  private updateDriveUi(): void {
    this.worldView.setDriveUi(this.showcase.mode, this.showcase.recording, this.showcase.demoCount);
  }

  private toggleRecord(): void {
    this.showcase.recording = !this.showcase.recording;
    if (this.showcase.recording) this.log('Recording: every step you drive is saved as (what the robot saw → what you pressed). Drive well!', 'learn');
    else this.log(`Recorded ${this.showcase.demoCount} steps. Press "Learn from me" to train the brain to copy you.`, 'info');
    this.updateDriveUi();
  }

  private initKeyboard(): void {
    const isTyping = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      return !!t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable || !!t.closest?.('.blocklyWidgetDiv, .blocklyDropDownDiv, .cm-editor'));
    };
    window.addEventListener('keydown', (e) => {
      if (e.defaultPrevented) return;
      const language = this.shell.appMode === 'language';
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        if (language) void this.llm?.toggleTrain();
        else void this.toggleTrain();
        return;
      }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
        e.preventDefault();
        if (language) this.llm?.saveFile();
        else this.saveProjectFile();
        return;
      }
      if (isTyping(e) || language) return;
      if (this.worldView.building && this.worldView.handleKey(e)) {
        e.preventDefault();
        return;
      }
      if (this.showcase.mode === 'drive' && !(e.target as HTMLElement)?.closest?.('#blockly')) {
        if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space'].includes(e.code)) e.preventDefault();
        this.showcase.keys.add(e.code);
      }
    });
    window.addEventListener('keyup', (e) => this.showcase.keys.delete(e.code));
    window.addEventListener('blur', () => this.showcase.keys.clear());
  }

  // =========================================================================
  // Per-frame panel updates (called by the 3D view's animation loop)
  // =========================================================================

  private frame(): void {
    const now = performance.now();
    if (now - this.lastPanels < 66) return;
    this.lastPanels = now;
    const tab = this.shell.tab;
    const sc = this.showcase;
    this.trainingPanel.render(tab === 'train');
    this.sensesPanel.render(tab === 'senses', sc);
    this.mathPanel.render(tab === 'math');
    this.experiments.render(tab === 'experiments');
    if (tab === 'brain') {
      this.brainPanel.viz.setTrace(sc.lastTrace, sc.action, sc.brain?.critic ? sc.brain.lastValue : null);
      this.brainPanel.render(true);
    }
  }

  private refreshCodePanel(): void {
    this.codeDirty = false;
    const ok = this.compiled && !this.compiled.errors.length;
    const program = ok && this.compiled!.spec.source !== 'code' && this.compiled!.events ? blocksToCode(this.compiled!) : null;
    this.codePanel.update(ok ? this.compiled!.spec : null, this.layout, this.watched?.snap ?? this.snap, program);
  }

  // =========================================================================
  // Menus
  // =========================================================================

  private projectMenu(anchor: HTMLElement): void {
    menu(
      anchor,
      [
        { heading: 'Project' },
        { label: 'New blocks project', icon: 'blocks', action: () => void this.newProject('blocks') },
        { label: 'New code project', icon: 'code', action: () => void this.newProject('code') },
        { label: 'New code project: your own algorithm', icon: 'flask', action: () => void this.newProject('algo') },
        { label: 'Missions…', icon: 'target', action: () => this.openMissions() },
        'sep',
        { label: 'Open…', icon: 'open', hint: '.json', action: () => void this.openProjectFile() },
        { label: 'Save to a file', icon: 'save', hint: 'Ctrl+S', action: () => this.saveProjectFile() },
        { label: 'Rename…', icon: 'pencil', action: () => void this.rename() },
        'sep',
        { label: 'Import a brain (weights)…', icon: 'upload', hint: '.json', action: () => void this.importBrain() },
        { label: 'Clear comparison curves', icon: 'chart', action: () => this.trainingPanel.clearComparisons() },
        { label: 'Reset everything…', icon: 'trash', action: () => void this.resetAll() },
      ],
      'left',
    );
  }

  private moreMenu(anchor: HTMLElement): void {
    menu(anchor, [
      { label: 'Missions', icon: 'target', action: () => this.openMissions() },
      { label: 'Neuron Lab', icon: 'flask', action: () => void this.openLab() },
      { label: 'Learn', icon: 'learn', action: () => openLearn(() => this.openMissions(), () => void this.openLab(), this.shell.appMode) },
      'sep',
      { label: 'Settings', icon: 'settings', action: () => this.openSettings() },
      { label: 'Keyboard shortcuts', icon: 'keyboard', action: () => openShortcuts() },
      { label: 'Welcome tour', icon: 'rocket', action: () => this.welcome() },
      { label: 'Source code on GitHub', icon: 'code', action: () => window.open('https://github.com/eefamilyai/neuroblocks-studio', '_blank', 'noopener') },
    ]);
  }

  private exportMenu(anchor: HTMLElement): void {
    const name = this.compiled?.spec.name ?? 'robot';
    const snap = this.watched?.snap ?? this.snap;
    const ok = () => {
      if (!snap) {
        toast('No brain yet');
        return false;
      }
      if (!snap.steps) toast('This brain is untrained (random weights). Exporting anyway.', 3000, 'warn');
      return true;
    };
    const which = this.watched ? ` (${this.watched.label})` : '';
    menu(anchor, [
      { heading: `Brain${which}` },
      { label: 'Weights as JSON', icon: 'json', hint: 're-importable', action: () => ok() && download(`${slug(name)}_brain.json`, JSON.stringify(this.brainExport(snap!), null, 1), 'application/json') },
      { label: 'ONNX model', icon: 'box', hint: 'onnxruntime, Unity', action: () => ok() && download(`${slug(name)}.onnx`, exportOnnx(snap!, name), 'application/octet-stream') },
      { label: 'PyTorch / NumPy script', icon: 'file-code', hint: '.py', action: () => ok() && download(`${slug(name)}_brain.py`, pytorchScript(snap!, name), 'text/x-python') },
      { heading: 'Training logs' },
      { label: 'Episodes (CSV)', icon: 'file-down', hint: 'reward, length…', action: () => this.exportCsv('episodes') },
      { label: 'Metrics (CSV)', icon: 'file-down', hint: 'losses, entropy…', action: () => this.exportCsv('metrics') },
      'sep',
      { label: 'Program as code (.js)', icon: 'code', action: () => this.exportProgram() },
      { label: 'What are these formats?', icon: 'help', action: () => explainExports() },
    ]);
  }

  private exportProgram(): void {
    const r = this.compiled;
    const code = this.editorMode === 'code' ? this.code : r?.events ? blocksToCode(r) : null;
    if (!code) return void toast('Fix the program first');
    download(`${slug(this.projectName)}.neuroblocks.js`, code, 'text/javascript');
  }

  private brainExport(s: BrainSnapshot): object {
    const fs = this.compiled?.spec.frameStack ?? 1;
    return {
      ...s,
      exportedAt: new Date().toISOString(),
      robot: this.compiled?.spec.name,
      zone: this.compiled?.spec.zone,
      frameStack: fs,
      howToUse:
        'net.layers[i].W is a row-major [outSize x inSize] matrix (PyTorch nn.Linear layout), b the bias. Forward: x = (obs - obsNorm.mean)/sqrt(obsNorm.var) clipped to ±obsNorm.clip (if obsNorm present); then for each layer x = activation(W·x + b). Outputs: see space (discrete = one softmax per branch; continuous = means; kind "q" = Q-values for every joint action).' +
        (fs > 1 ? ` The input is ${fs} observations in a row (newest first).` : ''),
    };
  }

  private exportCsv(which: 'episodes' | 'metrics'): void {
    const name = slug(this.compiled?.spec.name ?? 'robot');
    if (which === 'episodes') {
      const eps = this.trainingPanel.episodes;
      if (!eps.length) return void toast('No finished episodes yet');
      const labels = [...new Set(eps.flatMap((e) => Object.keys(e.parts ?? {})))];
      const esc = (s: string) => `"${s.replace(/"/g, '""')}"`;
      const rows = [['step', 'reward', 'length', 'success', ...labels.map((l) => esc(`reward: ${l}`))].join(','), ...eps.map((e) => [e.step, e.reward, e.length, e.success ? 1 : 0, ...labels.map((l) => e.parts?.[l] ?? 0)].join(','))];
      download(`${name}_episodes.csv`, rows.join('\n'), 'text/csv');
      return;
    }
    const ms = this.trainingPanel.metrics;
    if (!ms.length) return void toast('No training updates yet');
    const keys = [...new Set(ms.flatMap((m) => Object.keys(m)))];
    const rows = [keys.join(','), ...ms.map((m) => keys.map((k) => (m[k] === undefined ? '' : String(m[k]))).join(','))];
    download(`${name}_metrics.csv`, rows.join('\n'), 'text/csv');
  }

  private openSettings(): void {
    openSettings(
      this.settings,
      (s) => {
        const qChanged = s.quality !== this.settings.quality;
        this.settings = { ...s };
        prefs.set('settings', this.settings);
        if (qChanged) this.worldView.setQuality(s.quality);
        this.worldView.view.maxFps = s.maxFps;
        this.worldView.toggles.fps = s.showFps;
        (this.shell.viewportHost.querySelector('.fps') as HTMLElement).style.display = s.showFps ? '' : 'none';
        this.client.send({ type: 'throttle', sps: s.throttle });
      },
      { workers: this.client.mode === 'worker' ? 'a background worker' : 'the main thread', dpr: this.worldView.view.pixelRatio, cores: navigator.hardwareConcurrency || 1 },
    );
  }

  private openMissions(): void {
    openMissions(this.world, {
      load: (m) => this.loadMission(m),
      flyTo: (z) => {
        this.worldView.setCam('zone');
        this.worldView.flyToZone(z, true);
      },
      trainHere: (z) => this.setZone(z),
    });
  }

  private async openLab(): Promise<void> {
    const { openNeuronLab } = await import('../ui/lab.ts');
    openNeuronLab();
  }

  private welcome(): void {
    openWelcome({
      startFirst: () => {
        if (this.missionId !== 'reach-flag') this.loadMission(getMission('reach-flag')!);
        setTimeout(() => this.startTraining('rl'), 300);
      },
      missions: () => this.openMissions(),
      code: () => void this.newProject('code'),
      language: () => void this.setAppMode('language'),
    });
  }

  // =========================================================================
  // Projects
  // =========================================================================

  private projectData(): ProjectFile {
    return {
      format: 'neuroblocks-project',
      version: 1,
      name: this.projectName,
      savedAt: new Date().toISOString(),
      missionId: this.missionId,
      workspace: this.blocks.save(),
      edits: this.edits,
      brain: this.snap,
      episodes: this.trainingPanel.episodes.slice(-3000),
      editor: this.editorMode,
      code: this.code || undefined,
    };
  }

  private saveProjectFile(): void {
    const p = this.projectData();
    download(`${slug(p.name)}.neuroblocks.json`, JSON.stringify(p), 'application/json');
    toast('Project saved (program + world builds + brain)', 2600, 'save');
  }

  private async openProjectFile(): Promise<void> {
    const f = await pickFile('.json,application/json');
    if (!f) return;
    try {
      const p = JSON.parse(await f.text());
      if (!validateProject(p)) throw new Error('not a NeuroBlocks project file');
      const backup = this.projectData();
      try {
        this.loadProject(p, { quiet: false, trusted: !(p.editor === 'code' && p.code) });
      } catch (err) {
        this.loadProject(backup, { quiet: true, trusted: true });
        throw err;
      }
      toast(`Opened "${p.name}"`, 2600, 'open');
    } catch (e) {
      toast(`Could not open: ${(e as Error).message}`, 4000, 'error');
    }
  }

  private async importBrain(): Promise<void> {
    const f = await pickFile('.json,application/json');
    if (!f) return;
    try {
      const b = JSON.parse(await f.text());
      if (!validateBrain(b)) throw new Error('not a NeuroBlocks brain file');
      if (this.state !== 'idle') this.stopTraining(true);
      this.watched = null;
      this.setSnapshot(b, false);
      const fits = this.compiled && this.snapFits(b, this.compiled.spec);
      this.log(fits ? `Brain imported (${fmtSteps(b.steps)} steps of training). Press Train to keep training it.` : 'Brain imported, but it does not match your current senses/actions/brain, so the robot cannot use it. Load the matching program first.', fits ? 'good' : 'warn');
    } catch (e) {
      toast(`Could not import: ${(e as Error).message}`, 4000, 'error');
    }
  }

  private async resetAll(): Promise<void> {
    if (!(await confirmDialog('Reset everything?', 'This clears your autosaved project, settings and mission progress, then reloads the page.', 'Reset', true))) return;
    this.settings.autosave = false;
    try {
      localStorage.clear();
    } catch {
      /* ignore */
    }
    location.reload();
  }

  private async rename(): Promise<void> {
    const n = await promptText('Rename project', 'Project name', this.projectName);
    if (!n) return;
    this.projectName = n.slice(0, 60);
    this.shell.projectName.textContent = this.projectName;
    this.scheduleAutosave();
  }

  private async newProject(kind: 'blocks' | 'code' | 'algo'): Promise<void> {
    if (kind === 'blocks') {
      this.loadMission(getMission('blank')!);
      if (this.editorMode !== 'blocks') await this.setEditorMode('blocks', { quiet: true });
      return;
    }
    if (this.state !== 'idle') this.stopTraining(true);
    this.loadMission(getMission('reach-flag')!, true);
    this.missionId = null;
    this.code = kind === 'algo' ? CUSTOM_ALGO_TEMPLATE : CODE_TEMPLATE;
    this.codeEdited = true;
    this.codeTrusted = true;
    this.setProjectName(kind === 'algo' ? 'My own algorithm' : 'My code robot');
    this.editorMode = 'blocks';
    await this.setEditorMode('code', { quiet: true });
    this.codeEditor?.setValue(this.code);
    this.compile();
    this.log(kind === 'algo' ? 'Your own learning algorithm: REINFORCE written from scratch in train.custom(...). Press Train, watch it learn, then change the maths!' : 'New code project: a robot that learns to reach the flag, written in JavaScript. Press Train, then change it!', 'learn');
  }

  private setProjectName(n: string): void {
    this.projectName = n;
    this.shell.projectName.textContent = n;
    this.shell.editorTitle.textContent = '';
  }

  private loadProject(p: ProjectFile, o: { quiet: boolean; trusted: boolean }): void {
    if (this.state !== 'idle') this.stopTraining(true);
    this.edits = p.edits ?? { objects: [], walls: [] };
    this.rtKey = '';
    this.missionId = p.missionId;
    this.blocks.load(p.workspace);
    this.setProjectName(p.name || 'My robot');
    this.code = p.code ?? '';
    this.codeEdited = !!p.code;
    this.codeTrusted = o.trusted;
    this.snap = null;
    this.watched = null;
    this.draftKey = '';
    const mode: EditorMode = p.editor === 'code' && p.code ? 'code' : 'blocks';
    const finish = () => {
      if (p.brain && validateBrain(p.brain) && this.compiled && this.snapFits(p.brain, this.compiled.spec)) this.setSnapshot(p.brain, false);
      this.trainingPanel.newRun('');
      if (p.episodes?.length) this.trainingPanel.add(p.episodes, []);
      this.updateTopStats();
      if (!o.quiet) this.log(`Loaded project "${p.name}".`, 'info');
      else if (this.snap?.steps) this.log(`Restored your last session (brain trained for ${fmtSteps(this.snap.steps)} steps).`, 'info');
      if (!o.trusted) {
        this.worldView.setBanner('This project contains code that has not been run. Read it, then press Train to run it.');
        this.log('This project contains JavaScript code from a file. It will not run until you press Train and confirm.', 'warn');
      }
    };
    if (mode === 'code') {
      this.editorMode = 'blocks';
      void this.setEditorMode('code', { quiet: true }).then(() => {
        this.codeEditor?.setValue(this.code);
        this.compile();
        finish();
      });
    } else {
      if (this.editorMode === 'code') {
        this.editorMode = 'blocks';
        this.shell.setEditorMode('blocks');
        this.setEditorTools();
      }
      this.compile();
      finish();
    }
    setTimeout(() => this.blocks.ws.scroll(0, 0), 50);
  }

  loadMission(m: Mission, quiet = false): void {
    if (this.state !== 'idle') this.stopTraining(true);
    this.missionId = m.id;
    this.blocks.load(m.workspace());
    this.blocks.tidy();
    this.setProjectName(`${m.emoji} ${m.title}`);
    this.snap = null;
    this.watched = null;
    this.draftKey = '';
    this.codeTrusted = true;
    if (this.editorMode === 'code') {
      // missions are block programs: convert them for code mode
      const r = this.blocks.compile(this.edits.objects);
      this.code = blocksToCode(r);
      this.codeEdited = false;
      this.codeEditor?.setValue(this.code);
    } else {
      this.codeEdited = false;
      this.code = '';
    }
    this.compile();
    if (this.compiled && !this.compiled.errors.length) this.newRun(this.compiled.spec);
    this.trainingPanel.clearComparisons();
    this.showcase.clearDemos();
    this.updateDriveUi();
    if (!quiet) {
      this.log(`Mission: ${m.emoji} ${m.title}. ${m.summary}`, 'learn');
      this.log(`Try next: ${m.tryNext}`, 'learn');
    }
    if (this.worldView.view.mode !== 'follow') this.worldView.flyToZone(m.zone, true);
  }

  // =========================================================================

  log(msg: string, level: LogLevel = 'info'): void {
    this.shell.log(msg, level);
  }

  private scheduleAutosave(): void {
    clearTimeout(this.autosaveTimer);
    this.autosaveTimer = window.setTimeout(() => this.autosave(), 1500);
  }

  private autosave(): void {
    if (!this.blocks || !this.settings.autosave) return;
    saveLocal(this.projectData());
  }
}

