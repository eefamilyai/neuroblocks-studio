/**
 * NeuroBlocks Studio: the application shell. Connects the block editor,
 * the 3D world, the training worker and the "under the hood" panels.
 */
import Blockly from './blocks/blockly.ts';
import type * as BT from 'blockly/core';
import 'blockly/blocks';
import * as En from 'blockly/msg/en';
import * as THREE from 'three';
import { defineBlocks } from './blocks/defs.ts';
import { compileWorkspace } from './blocks/compiler.ts';
import type { CompileResult } from './blocks/compiler.ts';
import { toolbox } from './blocks/toolbox.ts';
import { neuroTheme } from './blocks/theme.ts';
import { MISSIONS, getMission } from './blocks/missions.ts';
import type { Mission } from './blocks/missions.ts';
import { buildWorld } from './sim/worldgen.ts';
import { WorldRuntime, zoneAt } from './sim/runtime.ts';
import type { CustomWall, WorldEdits } from './sim/runtime.ts';
import { obsLayout } from './sim/env.ts';
import type { ObsLayout } from './sim/env.ts';
import { actionSpaceOf } from './sim/spec.ts';
import type { AgentSpec, ObjType } from './sim/types.ts';
import { Brain } from './ml/algos/common.ts';
import type { BrainSnapshot } from './ml/algos/common.ts';
import { MLP } from './ml/nn.ts';
import { RNG } from './ml/rng.ts';
import { jointActionCount, policyOutputSize } from './ml/policy.ts';
import { Showcase } from './showcase.ts';
import type { DriveMode } from './showcase.ts';
import { TrainerClient } from './worker/client.ts';
import type { FromWorker } from './worker/protocol.ts';
import { Viewport } from './render/viewport.ts';
import type { CamMode } from './render/viewport.ts';
import { buildWorldMeshes } from './render/worldMesh.ts';
import type { WorldMeshes } from './render/worldMesh.ts';
import { GhostLayer, ObjectLayer, RobotMesh, SensorLayer, Trail } from './render/actors.ts';
import { GridOverlay } from './render/overlays.ts';
import { TrainingPanel } from './ui/trainingPanel.ts';
import { SensesPanel } from './ui/sensesPanel.ts';
import { BrainPanel } from './ui/brainPanel.ts';
import { MathPanel } from './ui/mathPanel.ts';
import { CodePanel } from './ui/codePanel.ts';
import { LearnPanel } from './ui/learnPanel.ts';
import { openNeuronLab } from './ui/lab.ts';
import { clear, download, fmt, fmtSteps, h, menu, modal, pickFile, toast } from './ui/dom.ts';
import { exportOnnx } from './export/onnx.ts';
import { pytorchScript, slug } from './export/pytorch.ts';
import { loadLocal, markMissionDone, missionsDone, saveLocal, setWelcomeSeen, validateBrain, validateProject, welcomeSeen } from './project/storage.ts';
import type { ProjectFile } from './project/storage.ts';

type Tab = 'train' | 'brain' | 'senses' | 'math' | 'code' | 'learn';
type BuildTool = ObjType | CustomWall['tag'] | 'erase' | 'spawn';

const BUILD_TOOLS: { tool: BuildTool; emoji: string; label: string }[] = [
  { tool: 'spawn', emoji: '📍', label: 'start' },
  { tool: 'coin', emoji: '🪙', label: 'coin' },
  { tool: 'flag', emoji: '🚩', label: 'flag' },
  { tool: 'ball', emoji: '⚽', label: 'ball' },
  { tool: 'crate', emoji: '📦', label: 'crate' },
  { tool: 'pad', emoji: '🎯', label: 'pad' },
  { tool: 'trophy', emoji: '🏆', label: 'trophy' },
  { tool: 'runner', emoji: '🐰', label: 'runner' },
  { tool: 'chaser', emoji: '👾', label: 'chaser' },
  { tool: 'wall', emoji: '🧱', label: 'wall' },
  { tool: 'ramp', emoji: '📐', label: 'ramp' },
  { tool: 'lava', emoji: '🔥', label: 'lava' },
  { tool: 'ice', emoji: '🧊', label: 'ice' },
  { tool: 'bouncy', emoji: '🟣', label: 'bounce' },
  { tool: 'erase', emoji: '🧽', label: 'erase' },
];

export class App {
  world = buildWorld();
  edits: WorldEdits = { objects: [], walls: [] };
  rt: WorldRuntime;
  ws!: BT.WorkspaceSvg;
  compiled: CompileResult | null = null;
  private draftKey = '';
  layout: ObsLayout | null = null;
  trainingSpec: AgentSpec | null = null;
  private trainingKey = '';
  snap: BrainSnapshot | null = null;
  brain: Brain | null = null;
  state: 'idle' | 'running' | 'paused' = 'idle';
  private trainMode: 'rl' | 'bc' = 'rl';
  client = new TrainerClient();
  showcase = new Showcase();
  missionId: string | null = null;
  private runCounter = 0;
  private stats = { steps: 0, episodes: 0, sps: 0, phase: 'idle', meanReward: NaN, success: NaN };
  private compileTimer = 0;
  private autosaveTimer = 0;
  private lastSuccessMilestone = 0;
  private firstSuccessShown = false;

  // UI
  private el: Record<string, HTMLElement> = {};
  private tab: Tab = 'train';
  private trainingPanel = new TrainingPanel();
  private sensesPanel = new SensesPanel();
  private brainPanel = new BrainPanel();
  private mathPanel = new MathPanel();
  private codePanel = new CodePanel();
  private learnPanel: LearnPanel;
  private tabBodies = new Map<Tab, HTMLElement>();
  private lastHud = 0;
  private lastPanels = 0;
  private lastLabels = 0;
  private zoneLabels: { el: HTMLElement; x: number; z: number; id: string }[] = [];
  private popups: { el: HTMLElement; x: number; y: number; z: number; born: number }[] = [];
  private codeDirty = true;

  // 3D
  view!: Viewport;
  private worldMeshes: WorldMeshes | null = null;
  private robot = new RobotMesh('#4fc3f7');
  private objLayer = new ObjectLayer();
  private ghosts = new GhostLayer();
  private sensors = new SensorLayer();
  private trail = new Trail();
  private heat = new GridOverlay(64);
  private valueMap = new GridOverlay(48);
  private toggles = { rays: true, ghosts: true, trail: true, heat: false, value: false, labels: true };
  private buildTool: BuildTool | null = null;
  private lastValueMap = 0;
  private heatDirty = false;

  constructor(root: HTMLElement) {
    this.rt = new WorldRuntime(this.world, this.edits);
    this.learnPanel = new LearnPanel(() => this.openMissions());
    this.buildDom(root);
    this.initBlockly();
    this.initViewport();
    this.client.onMessage = (m) => this.onWorker(m);
    this.client.onFallback = () => this.log('ℹ️ Background workers are not available here, so training runs on the main thread (a bit slower, same results).', 'warn');
    this.showcase.onStep = (info) => this.onShowcaseStep(info);
    this.showcase.onBeforeStep = () => this.objLayer.snapshotPrev();
    this.showcase.onError = (m) => this.log(m, 'err');
    this.showcase.onLog = (m) => this.log(`📝 ${m}`, 'info');
    this.initKeyboard();
    const saved = loadLocal();
    if (saved) this.loadProject(saved, true);
    else this.loadMission(MISSIONS[0], true);
    if (!welcomeSeen()) setTimeout(() => this.openWelcome(), 400);
    this.log(`Welcome to NeuroBlocks Studio! Training runs in ${this.client.mode === 'worker' ? 'a background worker thread' : 'the main thread (no Web Worker support)'}.`, 'learn');
    window.addEventListener('beforeunload', () => this.autosave());
  }

  // =========================================================================
  // DOM
  // =========================================================================

  private buildDom(root: HTMLElement): void {
    clear(root);
    const E = this.el;
    E.dot = h('span', { class: 'status-dot' });
    E.steps = h('b', null, '0');
    E.episodes = h('b', null, '0');
    E.reward = h('b', null, '–');
    E.sps = h('b', null, '0');
    E.phase = h('b', null, 'ready');
    E.trainBtn = h('button', { class: 'btn primary', onclick: () => this.toggleTrain(), title: 'Start / pause training (Ctrl+Enter)' }, '▶ Train');
    E.stopBtn = h('button', { class: 'btn', onclick: () => this.stopTraining(), title: 'Stop training (keeps the brain)' }, '⏹');
    E.resetBtn = h('button', { class: 'btn', onclick: () => this.resetBrain(), title: 'Throw the brain away and start from random weights' }, '⟲ New brain');
    E.fileBtn = h('button', { class: 'btn', onclick: () => this.openFileMenu() }, '📁 File ▾');
    E.exportBtn = h('button', { class: 'btn accent', onclick: () => this.openExportMenu() }, '💾 Export brain ▾');
    const topbar = h(
      'header',
      { id: 'topbar' },
      h('div', { class: 'brand' }, h('span', { class: 'logo' }, '🧠'), h('span', { class: 'txt' }, 'NeuroBlocks ', h('span', null, 'Studio'))),
      h('button', { class: 'btn', onclick: () => this.openMissions() }, '🎯 Missions'),
      h('button', { class: 'btn', onclick: () => openNeuronLab(), title: 'Neuron Lab: watch a network learn to classify 2D points (supervised learning basics)' }, '🧪 Lab'),
      h('div', { class: 'vsep' }),
      E.trainBtn,
      E.stopBtn,
      E.resetBtn,
      h('div', { class: 'vsep' }),
      h(
        'div',
        { class: 'pills' },
        h('div', { class: 'pill keep phase' }, h('small', null, E.dot, 'status'), E.phase),
        h('div', { class: 'pill keep' }, h('small', null, 'steps'), E.steps),
        h('div', { class: 'pill' }, h('small', null, 'episodes'), E.episodes),
        h('div', { class: 'pill' }, h('small', null, 'avg reward'), E.reward),
        h('div', { class: 'pill' }, h('small', null, 'steps/s'), E.sps),
      ),
      h('div', { class: 'spacer' }),
      E.exportBtn,
      E.fileBtn,
      h('button', { class: 'btn', onclick: () => this.openWelcome(), title: 'Help' }, '❓'),
    );

    // blocks pane
    E.blockly = h('div', { id: 'blockly' });
    E.compileStatus = h('div', { id: 'compile-status' });
    E.missionTitle = h('span', { class: 'pane-title' }, '🧩 Blocks');
    const blocksPane = h(
      'section',
      { class: 'pane', id: 'pane-blocks' },
      h(
        'div',
        { class: 'pane-head' },
        E.missionTitle,
        h('div', { class: 'spacer' }),
        h('button', { class: 'btn tiny', onclick: () => this.tidyBlocks(), title: 'Arrange blocks neatly' }, '✨ Tidy'),
        h('button', { class: 'btn tiny', onclick: () => this.ws.undo(false), title: 'Undo (Ctrl+Z)' }, '↶'),
        h('button', { class: 'btn tiny', onclick: () => this.ws.undo(true), title: 'Redo' }, '↷'),
      ),
      h('div', { id: 'blockly-host' }, E.blockly, E.compileStatus),
    );

    // world pane
    E.viewport = h('div', { id: 'viewport' });
    E.zoneSel = h('select', { class: 'sel', title: 'Train in which zone?', onchange: () => this.setZone((E.zoneSel as HTMLSelectElement).value) });
    for (const z of this.world.zones) E.zoneSel.append(h('option', { value: z.id }, `${z.emoji} ${z.name}`));
    const camSeg = h('div', { class: 'seg', id: 'cam-seg' });
    for (const [m, l, t] of [
      ['follow', '🎥', 'Chase camera'],
      ['orbit', '🔄', 'Orbit around the robot'],
      ['zone', '🗺️', 'Zone overview'],
      ['top', '⬇️', 'Top-down'],
      ['free', '🕊️', 'Free camera'],
    ] as const)
      camSeg.append(h('button', { 'data-v': m, title: t, onclick: () => this.setCam(m) }, l));
    const driveSeg = h('div', { class: 'seg', id: 'drive-seg' });
    for (const [m, l, t] of [
      ['ai', '🧠 AI', 'Watch the current brain drive'],
      ['drive', '🎮 You', 'You drive: WASD / arrows, Space = jump, F = kick'],
      ['paused', '⏸', 'Freeze the showcase world'],
    ] as const)
      driveSeg.append(h('button', { 'data-v': m, title: t, onclick: () => this.setDriveMode(m) }, l));
    E.speedSel = h('select', { class: 'sel', title: 'Showcase speed', onchange: () => (this.showcase.speed = Number((E.speedSel as HTMLSelectElement).value)) });
    for (const s of [0.25, 0.5, 1, 2, 4, 8]) E.speedSel.append(h('option', { value: String(s), selected: s === 1 }, `${s}×`));
    E.viewBtn = h('button', { class: 'btn small', onclick: () => this.openViewMenu(), title: 'What to show in the 3D view: rays, ghosts, heatmap, value map…' }, '👁️ ▾');
    E.buildBtn = h('button', { class: 'btn small', onclick: () => this.toggleBuild(), title: 'Build mode: place your own objects and walls in the world' }, '🧱');
    E.recBtn = h('button', { class: 'btn small', style: { display: 'none' }, onclick: () => this.toggleRecord(), title: 'Record your driving as training examples' }, '⏺ Record');
    E.cloneBtn = h('button', { class: 'btn small warn', style: { display: 'none' }, onclick: () => this.startCloning(), title: 'Train the brain to copy your recorded driving (supervised learning)' }, '🎓 Learn from me');
    E.hud = h('div', { class: 'hud hud-tl' });
    E.hudTR = h('div', { class: 'hud hud-tr' });
    E.hudBL = h('div', { class: 'hud hud-bl' });
    E.palette = h('div', { id: 'build-palette', style: { display: 'none' } });
    E.hint = h('div', { class: 'cursor-hint', style: { display: 'none' } });
    E.banner = h('div', { class: 'banner', style: { display: 'none' } });
    E.overlayLayer = h('div', { style: { position: 'absolute', inset: '0', pointerEvents: 'none', overflow: 'hidden' } });
    E.viewport.append(E.overlayLayer, E.hud, E.hudTR, E.hudBL, E.palette, E.hint, E.banner);
    E.console = h('div', { id: 'console' });
    const worldPane = h(
      'section',
      { class: 'pane', id: 'pane-world' },
      h('div', { class: 'pane-head world-tools' }, E.zoneSel, camSeg, driveSeg, E.recBtn, E.cloneBtn, h('div', { class: 'spacer' }), E.speedSel, E.viewBtn, E.buildBtn),
      E.viewport,
      E.console,
    );

    // hood pane
    const tabs = h('div', { class: 'tabs' });
    const hoodBody = h('div', { style: { flex: '1', display: 'flex', flexDirection: 'column', minHeight: '0' } });
    const tabDefs: [Tab, string, HTMLElement][] = [
      ['train', '📈 Train', this.trainingPanel.el],
      ['brain', '🧠 Brain', this.brainPanel.el],
      ['senses', '👁️ Senses', this.sensesPanel.el],
      ['math', '∑ Math', this.mathPanel.el],
      ['code', '</> Code', this.codePanel.el],
      ['learn', '📚 Learn', this.learnPanel.el],
    ];
    for (const [id, label, body] of tabDefs) {
      tabs.append(h('button', { class: 'tab', 'data-tab': id, onclick: () => this.setTab(id) }, label));
      const wrap = h('div', { class: 'tab-body hidden' }, body);
      this.tabBodies.set(id, wrap);
      hoodBody.append(wrap);
    }
    const hoodPane = h('section', { class: 'pane', id: 'pane-hood' }, h('div', { class: 'pane-head', title: 'Under the hood: everything the robot and the learning algorithm are doing' }, h('span', { class: 'pane-title' }, '🔧'), tabs), hoodBody);
    E.tabs = tabs;

    const split1 = h('div', { class: 'splitter' });
    const split2 = h('div', { class: 'splitter' });
    E.main = h('main', { id: 'main', 'data-mobile': 'world' }, blocksPane, split1, worldPane, split2, hoodPane);
    const mobile = h('div', { id: 'mobile-tabs' });
    for (const [k, l] of [
      ['blocks', '🧩 Blocks'],
      ['world', '🌍 World'],
      ['hood', '🔧 Hood'],
    ])
      mobile.append(h('button', { class: 'btn small', onclick: () => E.main.setAttribute('data-mobile', k) }, l));
    root.append(topbar, mobile, E.main);
    this.makeSplitter(split1, blocksPane, 1);
    this.makeSplitter(split2, hoodPane, -1);
    this.setTab('train');
    this.renderPalette();
  }

  private makeSplitter(sp: HTMLElement, pane: HTMLElement, dir: 1 | -1): void {
    sp.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      sp.classList.add('drag');
      sp.setPointerCapture(e.pointerId);
      const startX = e.clientX;
      const startW = pane.getBoundingClientRect().width;
      const move = (ev: PointerEvent) => {
        const w = Math.max(220, Math.min(window.innerWidth * 0.7, startW + (ev.clientX - startX) * dir));
        pane.style.flex = `0 0 ${w}px`;
        Blockly.svgResize(this.ws);
      };
      const up = () => {
        sp.classList.remove('drag');
        sp.removeEventListener('pointermove', move);
        sp.removeEventListener('pointerup', up);
      };
      sp.addEventListener('pointermove', move);
      sp.addEventListener('pointerup', up);
    });
  }

  private setTab(t: Tab): void {
    this.tab = t;
    this.el.tabs.querySelectorAll('.tab').forEach((b) => b.classList.toggle('on', (b as HTMLElement).dataset.tab === t));
    for (const [id, body] of this.tabBodies) body.classList.toggle('hidden', id !== t);
    this.showcase.wantTrace = t === 'brain' || t === 'senses';
    this.showcase.wantSaliency = t === 'senses';
    this.showcase.computeTrace();
    if (t === 'code' && this.codeDirty) this.refreshCode();
    if (t === 'math') this.mathPanel.render(true, true);
  }

  // =========================================================================
  // Blocks
  // =========================================================================

  private initBlockly(): void {
    // 'blockly/core' ships without UI strings: load English explicitly
    Blockly.setLocale(En as unknown as { [key: string]: string });
    defineBlocks();
    this.ws = Blockly.inject(this.el.blockly, {
      toolbox: toolbox() as BT.utils.toolbox.ToolboxDefinition,
      theme: neuroTheme(),
      renderer: 'zelos',
      media: './media/',
      // the single-file build has no media folder: hide the icon-based controls there
      trashcan: import.meta.env.MODE !== 'single',
      sounds: false,
      zoom: { controls: import.meta.env.MODE !== 'single', wheel: true, startScale: 0.62, maxScale: 1.6, minScale: 0.25, scaleSpeed: 1.12 },
      grid: { spacing: 26, length: 2, colour: '#262c48', snap: false },
      move: { scrollbars: true, drag: true, wheel: false },
    });
    this.ws.addChangeListener((e: BT.Events.Abstract) => {
      if (e.isUiEvent) return;
      this.scheduleCompile();
    });
    new ResizeObserver(() => Blockly.svgResize(this.ws)).observe(this.el.blockly);
  }

  private scheduleCompile(): void {
    clearTimeout(this.compileTimer);
    this.compileTimer = window.setTimeout(() => this.compile(), 220);
  }

  private compile(): void {
    let r: CompileResult;
    try {
      r = compileWorkspace(this.ws, this.edits.objects);
    } catch (e) {
      this.showCompileStatus([`Compiler crashed: ${(e as Error).message}`], []);
      return;
    }
    this.compiled = r;
    this.updateBlockLabels(r);
    this.showCompileStatus(r.errors, r.warnings);
    if (!r.errors.length) this.applyDraftSpec(r.spec);
    this.scheduleAutosave();
  }

  private specKey(s: AgentSpec): string {
    const { warnings: _w, name: _n, ...rest } = s;
    return JSON.stringify(rest);
  }

  private applyDraftSpec(spec: AgentSpec): void {
    const key = this.specKey(spec);
    const changed = key !== this.draftKey;
    const prevZone = this.compiled && this.draftKey ? JSON.parse(this.draftKey).zone : null;
    this.draftKey = key;
    if (this.state !== 'idle') {
      // training continues with its own spec; tell the user how to apply edits
      this.showBanner(key !== this.trainingKey ? '✏️ Blocks changed while training: press ⟲ New brain or ⏹ + ▶ to apply' : null);
      return;
    }
    this.showBanner(null);
    if (!changed) return;
    this.layout = obsLayout(spec);
    this.showcase.setSpec(spec, this.rt);
    this.robot.setColor(spec.body.color);
    this.objLayer.bind(this.showcase.env!);
    this.trail.clear();
    this.sensesPanel.setLayout(this.layout);
    this.placeOverlays(spec.zone);
    // keep the current brain if it still fits, otherwise grow a fresh random one
    if (!this.snap || !this.snapFits(this.snap, spec)) this.setSnapshot(this.freshSnapshot(spec), false);
    else this.showcase.setBrain(this.brain);
    (this.el.zoneSel as HTMLSelectElement).value = spec.zone;
    if (prevZone !== spec.zone) this.flyToZone(spec.zone);
    this.trainingPanel.setAlgo(spec.train.algo);
    this.mathPanel.algo = spec.train.algo;
    this.codeDirty = true;
    if (this.tab === 'code') this.refreshCode();
    this.updateZoneLabelHighlight(spec.zone);
  }

  private updateBlockLabels(r: CompileResult): void {
    const sensesHat = this.ws.getTopBlocks(false).find((b) => b.type === 'nb_senses');
    const brainHat = this.ws.getTopBlocks(false).find((b) => b.type === 'nb_brain');
    Blockly.Events.disable();
    try {
      if (sensesHat) sensesHat.setFieldValue(`= ${r.obsSize} numbers into the brain`, 'INFO');
      if (brainHat) {
        const hidden = r.spec.brain.layers.map((l) => l.units).join(' → ');
        brainHat.setFieldValue(`${r.obsSize} in → ${hidden} → ${r.outSize} out · ${r.params.toLocaleString()} weights`, 'INFO');
      }
    } finally {
      Blockly.Events.enable();
    }
  }

  private showCompileStatus(errors: string[], warnings: string[]): void {
    const host = this.el.compileStatus;
    clear(host);
    for (const e of errors) host.append(h('div', { class: 'cmsg err' }, '⛔', h('span', null, e)));
    for (const w of warnings.slice(0, 3)) {
      const m = h('div', { class: 'cmsg warn' }, '⚠️', h('span', null, w), h('span', { class: 'x', onclick: () => m.remove() }, '✕'));
      host.append(m);
    }
  }

  private tidyBlocks(): void {
    const tops = this.ws.getTopBlocks(true);
    const left = ['nb_robot', 'nb_senses', 'nb_actions', 'nb_brain'];
    let y1 = 20;
    let y2 = 20;
    let colW = 0;
    const leftBlocks = tops.filter((b) => left.includes(b.type)).sort((a, b) => left.indexOf(a.type) - left.indexOf(b.type));
    for (const b of leftBlocks) colW = Math.max(colW, b.getHeightWidth().width);
    for (const b of leftBlocks) {
      b.moveTo(new Blockly.utils.Coordinate(20, y1));
      y1 += b.getHeightWidth().height + 28;
    }
    const x2 = 20 + colW + 50;
    for (const b of tops) {
      if (left.includes(b.type)) continue;
      b.moveTo(new Blockly.utils.Coordinate(x2, y2));
      y2 += b.getHeightWidth().height + 28;
    }
    // zoom so both columns fit side by side
    let rightW = 0;
    for (const b of tops) if (!left.includes(b.type)) rightW = Math.max(rightW, b.getHeightWidth().width);
    const total = x2 + rightW + 20;
    const view = this.ws.getMetrics().viewWidth / this.ws.scale;
    const scale = Math.max(0.42, Math.min(0.72, (this.ws.scale * view) / total));
    this.ws.setScale(scale);
    setTimeout(() => this.ws.scroll(0, 0), 0);
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
    const wantQ = spec.train.algo === 'dqn';
    if ((s.kind === 'q') !== wantQ) return false;
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
    this.showcase.setBrain(this.brain);
    this.brainPanel.setBrain(this.brain);
    if (!fromTraining) this.codeDirty = true;
    this.mathPanel.algo = s.algo;
  }

  // =========================================================================
  // Training
  // =========================================================================

  private toggleTrain(): void {
    if (this.state === 'running') {
      this.client.send({ type: 'pause' });
      this.setState('paused');
      this.log('⏸ Training paused. The robot keeps its brain.', 'info');
      return;
    }
    if (this.state === 'paused') {
      if (this.trainingKey !== this.draftKey) {
        this.stopTraining(true);
        this.startTraining('rl');
        return;
      }
      this.client.send({ type: 'resume' });
      this.setState('running');
      this.log('▶ Training resumed.', 'info');
      return;
    }
    this.startTraining('rl');
  }

  private startTraining(mode: 'rl' | 'bc'): void {
    const r = this.compiled;
    if (!r || r.errors.length) {
      toast('⛔ Fix the errors in your blocks first');
      return;
    }
    const spec = r.spec;
    this.applyDraftSpec(spec);
    let init: BrainSnapshot | null = null;
    if (mode === 'rl' && this.snap && this.snap.steps > 0 && this.snapFits(this.snap, spec)) {
      init = this.snap;
      this.log(`▶ Continuing to train the current brain (${fmtSteps(this.snap.steps)} steps of experience) with ${spec.train.algo.toUpperCase()}.`, 'info');
    } else if (mode === 'bc') {
      init = null;
    } else {
      this.newRun(spec);
      this.log(`▶ Training a fresh brain with ${spec.train.algo.toUpperCase()} in ${spec.train.numEnvs} parallel worlds (${this.world.zones.find((z) => z.id === spec.zone)?.name}).`, 'good');
    }
    let demos = null;
    if (mode === 'bc') {
      demos = this.showcase.demos();
      if (!demos) {
        toast('🎮 Record some driving first (You drive → ⏺ Record)');
        return;
      }
      this.newRun(spec, 'bc');
      this.log(`🎓 Behaviour cloning on ${demos.count} recorded steps: the network learns to copy your driving.`, 'good');
    }
    this.trainingSpec = spec;
    this.trainingKey = this.specKey(spec);
    this.trainMode = mode;
    this.trainingPanel.setAlgo(mode === 'bc' ? 'bc' : spec.train.algo);
    this.mathPanel.algo = mode === 'bc' ? 'bc' : spec.train.algo;
    this.client.send({ type: 'ghosts', on: this.toggles.ghosts });
    this.client.send({ type: 'start', spec, edits: this.edits, init, mode, demos });
    this.setState('running');
    if (this.showcase.mode === 'paused') this.setDriveMode('ai');
  }

  private newRun(spec: AgentSpec, algo?: string): void {
    this.runCounter++;
    this.trainingPanel.newRun(`run ${this.runCounter}`);
    this.heat.clear();
    this.stats.steps = 0;
    this.stats.episodes = 0;
    this.stats.meanReward = NaN;
    this.stats.success = NaN;
    this.el.reward.textContent = '–';
    this.el.steps.textContent = '0';
    this.el.episodes.textContent = '0';
    this.lastSuccessMilestone = 0;
    this.firstSuccessShown = false;
    this.trainingPanel.setAlgo(algo ?? spec.train.algo);
  }

  private stopTraining(silent = false): void {
    if (this.state === 'idle') return;
    this.client.send({ type: 'stop' });
    this.setState('idle');
    this.ghosts.set(null);
    if (!silent) this.log('⏹ Training stopped. The brain is kept: press ▶ to continue, ⟲ New brain to start over.', 'info');
    if (this.compiled && !this.compiled.errors.length) {
      this.draftKey = '';
      this.applyDraftSpec(this.compiled.spec);
    }
  }

  private resetBrain(): void {
    const wasRunning = this.state === 'running';
    if (this.state !== 'idle') this.stopTraining(true);
    const spec = this.compiled?.spec;
    if (!spec || this.compiled?.errors.length) return;
    this.setSnapshot(this.freshSnapshot(spec), false);
    this.newRun(spec);
    this.showcase.reset();
    this.trail.clear();
    this.log('⟲ Fresh random brain. Everything it learned is gone (previous curve kept dashed for comparison).', 'warn');
    if (wasRunning) this.startTraining('rl');
  }

  private setState(s: 'idle' | 'running' | 'paused'): void {
    this.state = s;
    this.el.dot.className = 'status-dot ' + (s === 'running' ? 'running' : s === 'paused' ? 'paused' : '');
    this.el.trainBtn.textContent = s === 'running' ? '⏸ Pause' : s === 'paused' ? '▶ Resume' : '▶ Train';
    this.el.trainBtn.className = 'btn ' + (s === 'running' ? 'warn' : 'primary');
    if (s === 'idle') this.el.phase.textContent = this.snap && this.snap.steps ? `stopped · ${fmtSteps(this.snap.steps)} steps learned` : 'ready';
  }

  private onWorker(m: FromWorker): void {
    switch (m.type) {
      case 'started':
        this.log(`🧠 Worker started ${m.algo.toUpperCase()} with ${m.obsSize} inputs.`, 'info');
        break;
      case 'progress': {
        this.stats.steps = m.steps;
        this.stats.episodes = m.episodeCount;
        this.stats.sps = m.sps;
        this.stats.phase = m.phase;
        this.trainingPanel.add(m.episodes, m.metrics);
        this.trainingPanel.setFitness(m.fitness);
        this.mathPanel.info = m.explain;
        this.mathPanel.phase = m.phase;
        const eps = this.trainingPanel.episodes.slice(-100);
        if (eps.length) {
          this.stats.meanReward = eps.reduce((a, e) => a + e.reward, 0) / eps.length;
          this.stats.success = eps.filter((e) => e.success).length / eps.length;
        }
        for (const mt of m.metrics) this.logMetric(mt);
        this.checkMilestones();
        this.el.steps.textContent = fmtSteps(m.steps);
        this.el.episodes.textContent = m.episodeCount.toLocaleString();
        this.el.reward.textContent = Number.isFinite(this.stats.meanReward) ? fmt(this.stats.meanReward, 2) : '–';
        this.el.sps.textContent = Math.round(m.sps).toLocaleString();
        this.el.phase.textContent = m.phase;
        this.el.phase.title = m.phase;
        break;
      }
      case 'brain':
        this.setSnapshot(m.snap, true);
        if (this.tab === 'code' && m.snap.updates <= 1) this.refreshCode();
        else this.codeDirty = true;
        break;
      case 'ghosts':
        if (this.toggles.ghosts && this.state === 'running') this.ghosts.set(m.data);
        for (let i = 0; i < m.data.length; i += 4) this.heat.addVisit(m.data[i], m.data[i + 2]);
        this.heatDirty = true;
        break;
      case 'log':
        this.log(m.msg, m.level);
        break;
      case 'error':
        this.log(`⛔ ${m.msg}`, 'err');
        toast(`⛔ ${m.msg}`, 5000);
        this.setState('idle');
        break;
      case 'finished':
        this.log(`🏁 Training finished: ${m.reason}.`, 'good');
        toast(`🏁 Training finished: ${m.reason}`);
        this.setState('idle');
        this.ghosts.set(null);
        break;
    }
  }

  private lastMetricLog = 0;
  private logMetric(m: Record<string, number>): void {
    const now = performance.now();
    if (now - this.lastMetricLog < 4000) return;
    this.lastMetricLog = now;
    const sr = Number.isFinite(this.stats.success) ? ` · success ${(this.stats.success * 100).toFixed(0)}%` : '';
    const extra =
      m.entropy !== undefined ? ` · entropy ${fmt(m.entropy, 2)}` : m.epsilon !== undefined ? ` · ε ${fmt(m.epsilon, 2)}` : m.bestFitness !== undefined ? ` · best fitness ${fmt(m.bestFitness, 2)}` : m.accuracy !== undefined ? ` · accuracy ${(m.accuracy * 100).toFixed(0)}%` : '';
    this.log(`📈 update ${m.update} @ ${fmtSteps(m.step)} steps · mean reward ${fmt(this.stats.meanReward, 2)}${sr}${extra}`, 'info');
  }

  private checkMilestones(): void {
    const eps = this.trainingPanel.episodes;
    if (!this.firstSuccessShown && eps.some((e) => e.success)) {
      this.firstSuccessShown = true;
      this.log('🎉 First success! The robot reached its goal at least once. Now it has to learn to do it reliably.', 'good');
    }
    const last = eps.slice(-50);
    if (last.length >= 50) {
      const sr = last.filter((e) => e.success).length / 50;
      for (const ms of [0.25, 0.5, 0.8, 0.95]) {
        if (sr >= ms && this.lastSuccessMilestone < ms) {
          this.lastSuccessMilestone = ms;
          this.log(`🏅 Success rate passed ${ms * 100}% (last 50 episodes).`, 'good');
          if (ms >= 0.8 && this.missionId) {
            const was = missionsDone().has(this.missionId);
            markMissionDone(this.missionId);
            if (!was) toast(`🏆 Mission complete: ${getMission(this.missionId)?.title}!`, 4000);
          }
        }
      }
    }
  }

  private startCloning(): void {
    if (this.state !== 'idle') this.stopTraining(true);
    this.showcase.recording = false;
    this.updateDriveUi();
    this.setDriveMode('ai');
    this.startTraining('bc');
  }

  // =========================================================================
  // 3D
  // =========================================================================

  private initViewport(): void {
    this.view = new Viewport(this.el.viewport);
    const sc = this.view.scene;
    this.rebuildWorldMeshes();
    sc.add(this.robot, this.objLayer.group, this.ghosts.mesh, this.sensors.group, this.trail.line, this.heat.mesh, this.valueMap.mesh);
    this.view.onFrame = (dt, t) => this.frame(dt, t);
    // zone labels
    for (const z of this.world.zones) {
      if (z.id === 'world') continue;
      const el = h('div', { class: 'zone-label', onclick: () => this.setZone(z.id) }, `${z.emoji} ${z.name}`);
      el.title = `${z.blurb}\n\nClick to train here.`;
      this.el.overlayLayer.append(el);
      this.zoneLabels.push({ el, x: (z.rect.minX + z.rect.maxX) / 2, z: (z.rect.minZ + z.rect.maxZ) / 2, id: z.id });
    }
    // picking (spawn point / build mode)
    const canvas = this.view.renderer.domElement;
    let down: [number, number] | null = null;
    canvas.addEventListener('pointerdown', (e) => (down = [e.clientX, e.clientY]));
    canvas.addEventListener('pointerup', (e) => {
      if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return;
      down = null;
      if (this.buildTool) this.applyBuildTool(e.clientX, e.clientY);
    });
    this.setCam('follow');
    this.setDriveMode('ai');
  }

  private rebuildWorldMeshes(): void {
    if (this.worldMeshes) {
      this.view.scene.remove(this.worldMeshes.group);
      this.worldMeshes.dispose();
    }
    this.worldMeshes = buildWorldMeshes(this.world, this.rt.statics);
    this.view.scene.add(this.worldMeshes.group);
    this.view.pickables = this.worldMeshes.group.children.filter((c) => (c as THREE.Mesh).isMesh && !(c as THREE.Mesh).geometry.type.startsWith('Sphere'));
    if (this.worldMeshes.gates) this.worldMeshes.gates.visible = this.compiled?.spec.zone !== 'world';
  }

  private frame(dt: number, t: number): void {
    const sc = this.showcase;
    sc.update(dt);
    const env = sc.env;
    if (env) {
      const a = env.a;
      const p = sc.prev;
      const al = sc.alpha;
      const jump = Math.abs(a.x - p.x) + Math.abs(a.z - p.z) > 4;
      const x = jump ? a.x : p.x + (a.x - p.x) * al;
      const y = jump ? a.y : p.y + (a.y - p.y) * al;
      const z = jump ? a.z : p.z + (a.z - p.z) * al;
      let dy = a.yaw - p.yaw;
      while (dy > Math.PI) dy -= Math.PI * 2;
      while (dy < -Math.PI) dy += Math.PI * 2;
      const yaw = p.yaw + dy * al;
      this.robot.position.set(x, y, z);
      this.robot.animate(dt, t, yaw, a.vx, a.vz, env.turn);
      this.view.focus.set(x, y, z);
      this.view.focusYaw = yaw;
      this.objLayer.update(al, t, dt);
      this.sensors.group.visible = this.toggles.rays;
      if (this.toggles.rays) this.sensors.update(env, x, y + 0.15, z);
      if (this.worldMeshes?.gates) this.worldMeshes.gates.visible = env.spec.zone !== 'world';
    }
    this.ghosts.mesh.visible = this.toggles.ghosts && this.state === 'running';
    this.trail.line.visible = this.toggles.trail;
    this.worldMeshes?.update(t);
    const now = performance.now();
    if (this.heatDirty && this.toggles.heat && now - this.lastValueMap > 600) {
      this.heat.paintHeat();
      this.heatDirty = false;
      this.lastValueMap = now;
    }
    if (now - this.lastHud > 100) {
      this.lastHud = now;
      this.renderHud();
    }
    if (now - this.lastLabels > 50) {
      this.lastLabels = now;
      this.positionOverlays();
    }
    if (now - this.lastPanels > 66) {
      this.lastPanels = now;
      this.trainingPanel.render(this.tab === 'train');
      this.sensesPanel.render(this.tab === 'senses', sc);
      this.mathPanel.render(this.tab === 'math');
      if (this.tab === 'brain') {
        this.brainPanel.viz.setTrace(sc.lastTrace, sc.action, sc.brain?.critic ? sc.brain.lastValue : null);
      }
    }
    this.brainPanel.render(this.tab === 'brain');
  }

  private onShowcaseStep(info: { reward: number; done: boolean; success: boolean; events: string[] }): void {
    const env = this.showcase.env!;
    this.robot.react(info.reward);
    if (this.toggles.trail) this.trail.push(env.a.x, env.a.y - 0.35, env.a.z);
    this.heat.addVisit(env.a.x, env.a.z, 0.2);
    if (Math.abs(info.reward) >= 0.25) this.popup(info.reward > 0 ? `+${fmt(info.reward, 2)}` : fmt(info.reward, 2), info.reward > 0 ? 'pos' : 'neg');
    for (const e of info.events) if (/lap|kick/.test(e)) this.popup(e, 'info');
    if (info.done) {
      this.popup(info.success ? '✅ success!' : env.truncated ? '⏱️ time up' : '❌', info.success ? 'pos' : 'info');
      this.trail.clear();
    }
    this.sensesPanel.addEvents(info.events, info.reward);
  }

  private popup(text: string, cls: 'pos' | 'neg' | 'info'): void {
    const env = this.showcase.env;
    if (!env) return;
    const el = h('div', { class: `popup ${cls}` }, text);
    this.el.overlayLayer.append(el);
    this.popups.push({ el, x: env.a.x, y: env.a.y + 1.3, z: env.a.z, born: performance.now() });
    setTimeout(() => el.remove(), 1100);
    if (this.popups.length > 30) this.popups.shift();
  }

  private positionOverlays(): void {
    const now = performance.now();
    this.popups = this.popups.filter((p) => now - p.born < 1100);
    for (const p of this.popups) {
      const s = this.view.project(p.x, p.y, p.z);
      if (s) {
        p.el.style.left = s[0] + 'px';
        p.el.style.top = s[1] + 'px';
      }
    }
    const cam = this.view.camera.position;
    for (const l of this.zoneLabels) {
      const d = Math.hypot(cam.x - l.x, cam.z - l.z);
      const show = this.toggles.labels && d > 30 && d < 420;
      const s = show ? this.view.project(l.x, 16, l.z) : null;
      if (!s) {
        l.el.style.display = 'none';
        continue;
      }
      l.el.style.display = '';
      l.el.style.left = s[0] + 'px';
      l.el.style.top = s[1] + 'px';
      l.el.style.opacity = String(Math.max(0.35, Math.min(1, (420 - d) / 200)));
    }
  }

  private updateZoneLabelHighlight(zone: string): void {
    for (const l of this.zoneLabels) l.el.classList.toggle('current', l.id === zone);
  }

  private renderHud(): void {
    const env = this.showcase.env;
    const spec = this.showcase.spec;
    if (!env || !spec) return;
    const zone = this.world.zones.find((z) => z.id === spec.zone);
    clear(this.el.hud);
    this.el.hud.append(
      h(
        'div',
        { class: 'hud-card' },
        h('div', { class: 'hud-zone' }, `${zone?.emoji ?? ''} ${zone?.name ?? ''}`),
        h('div', { class: 'hud-big', style: { color: env.epReturn >= 0 ? '#6dffaf' : '#ff7a8f' } }, fmt(env.epReturn, 2)),
        h('div', { class: 'hud-stats' }, 'episode', h('b', null, String(env.episode)), 'step', h('b', null, `${env.steps} / ${env.maxSteps}`), 'last reward', h('b', null, fmt(env.stepReward, 3)), this.showcase.brain?.critic ? 'critic V(s)' : '', this.showcase.brain?.critic ? h('b', null, fmt(this.showcase.brain.lastValue, 2)) : ''),
      ),
    );
    clear(this.el.hudTR);
    const b = this.showcase.brain;
    if (this.showcase.mode === 'drive') this.el.hudTR.append(h('span', { class: 'badge you' }, '🎮 You are driving'));
    else if (this.showcase.mode === 'paused') this.el.hudTR.append(h('span', { class: 'badge' }, '⏸ Paused'));
    else this.el.hudTR.append(h('span', { class: 'badge ai' }, b && this.showcase.brainFits() ? `🧠 ${b.snap.steps ? `brain @ ${fmtSteps(b.snap.steps)} steps` : 'untrained brain (random)'}` : '🎲 random actions'));
    if (this.showcase.recording) this.el.hudTR.append(h('span', { class: 'badge rec' }, `⏺ REC ${this.showcase.demoCount} steps`));
    else if (this.showcase.demoCount) this.el.hudTR.append(h('span', { class: 'badge' }, `📼 ${this.showcase.demoCount} recorded steps`));
    if (this.state === 'running') this.el.hudTR.append(h('span', { class: 'badge' }, `🏋️ training · ${fmtSteps(this.stats.steps)} steps`));
    // action bars
    clear(this.el.hudBL);
    const bars = h('div', { class: 'hud-card act-bars' });
    const probs = b && this.showcase.brainFits() && b.snap.kind === 'policy' ? b.lastProbs : null;
    spec.axes.forEach((ax, k) => {
      const row = h('div', { class: 'act-row' }, h('span', { class: 'lbl' }, ax.axis));
      const opts = h('div', { class: 'act-opts' });
      if (spec.actionMode === 'discrete') {
        ax.labels.forEach((lab, j) => {
          const p = probs?.[k]?.[j];
          const chosen = this.showcase.action[k] === j;
          opts.append(h('div', { class: 'act-opt' + (chosen ? ' chosen' : '') }, h('i', { style: { width: `${(p ?? (chosen ? 1 : 0)) * 100}%` } }), h('span', null, lab)));
        });
      } else {
        const v = this.showcase.action[k];
        opts.append(h('div', { class: 'act-opt' }, h('i', { style: { width: `${((Math.max(-1, Math.min(1, v)) + 1) / 2) * 100}%` } }), h('span', null, fmt(v, 2))));
      }
      row.append(opts);
      bars.append(row);
    });
    this.el.hudBL.append(bars);
  }

  private setCam(m: CamMode): void {
    this.view.setMode(m);
    if (m === 'zone') this.flyToZone(this.showcase.spec?.zone ?? 'meadow');
    this.el.viewport.parentElement!.querySelectorAll('#cam-seg button').forEach((b) => b.classList.toggle('on', (b as HTMLElement).dataset.v === m));
  }

  private flyToZone(id: string): void {
    const z = this.world.zones.find((q) => q.id === id);
    if (!z) return;
    const c = new THREE.Vector3((z.rect.minX + z.rect.maxX) / 2, z.floorY > -5 ? z.floorY : 0, (z.rect.minZ + z.rect.maxZ) / 2);
    const size = Math.max(z.rect.maxX - z.rect.minX, z.rect.maxZ - z.rect.minZ);
    if (this.view.mode === 'follow' || this.view.mode === 'orbit') return;
    this.view.flyTo(c, id === 'world' ? 230 : size * 1.25, 0.75, Math.PI * 0.15);
  }

  private setZone(id: string): void {
    const robot = this.ws.getTopBlocks(false).find((b) => b.type === 'nb_robot');
    if (!robot) {
      toast('Add a 🤖 robot block first (Robot category)');
      return;
    }
    robot.setFieldValue(id, 'ZONE');
    const z = this.world.zones.find((q) => q.id === id);
    if (z) {
      this.log(`🗺️ Robot moved to ${z.emoji} ${z.name}: ${z.blurb}`, 'info');
      if (this.view.mode === 'follow' || this.view.mode === 'orbit') this.setCam('zone');
      else this.flyToZone(id);
    }
  }

  private placeOverlays(zoneId: string): void {
    const z = this.world.zones.find((q) => q.id === zoneId);
    if (!z) return;
    const y = (z.floorY > -5 ? z.floorY : 0) + 0.06;
    this.heat.place(z.rect, y);
    this.valueMap.place(z.rect, y + 0.02);
    this.heat.clear();
    this.valueMap.mesh.visible = false;
    this.toggles.value = false;
  }

  private computeValueMap(): void {
    const sc = this.showcase;
    const env = sc.env;
    const b = sc.brain;
    if (!env || !b || !sc.brainFits() || (!b.critic && b.snap.kind !== 'q')) {
      toast('The value map needs a brain with a critic (PPO, REINFORCE with baseline) or a Q-network (DQN)');
      this.toggles.value = false;
      return;
    }
    const a = env.a;
    const saved = { ...a };
    const savedG = env.grounded;
    const vals = new Float32Array(this.valueMap.res * this.valueMap.res);
    const yaws = [0, Math.PI / 2, Math.PI, -Math.PI / 2];
    for (let k = 0; k < vals.length; k++) {
      const [x, z] = this.valueMap.cellCenter(k);
      const gy = env.groundY(x, z, 16);
      if (gy === null) {
        vals[k] = NaN;
        continue;
      }
      let sum = 0;
      for (const yw of yaws) {
        a.x = x;
        a.y = gy + 0.52;
        a.z = z;
        a.vx = a.vy = a.vz = 0;
        a.yaw = yw;
        env.grounded = true;
        env.observe();
        const nx = b.normalize(env.obs);
        if (b.critic) sum += b.critic.forward(nx, 1)[0];
        else {
          const q = b.net.forward(nx, 1);
          let m = -Infinity;
          for (let i = 0; i < b.net.outSize; i++) m = Math.max(m, q[i]);
          sum += m;
        }
      }
      vals[k] = sum / yaws.length;
    }
    Object.assign(a, saved);
    env.grounded = savedG;
    env.observe();
    const r = this.valueMap.paintValues(vals);
    this.valueMap.mesh.visible = true;
    this.log(`🗺️ Value map: what the critic thinks each spot is worth (yellow = high ${fmt(r.max, 2)}, purple = low ${fmt(r.min, 2)}), averaged over 4 headings, with objects where they are now.`, 'learn');
  }

  // =========================================================================
  // Drive / record / build
  // =========================================================================

  private setDriveMode(m: DriveMode): void {
    this.showcase.mode = m;
    if (m !== 'drive') this.showcase.recording = false;
    this.el.viewport.parentElement!.querySelectorAll('#drive-seg button').forEach((b) => b.classList.toggle('on', (b as HTMLElement).dataset.v === m));
    this.updateDriveUi();
    if (m === 'drive') {
      this.view.renderer.domElement.focus();
      this.log('🎮 You are driving: W/↑ forward, S/↓ back, A/D or ←/→ turn, Q/E slide, Space jump, F kick. Click the 3D view first if keys do nothing.', 'learn');
    }
  }

  private updateDriveUi(): void {
    const drive = this.showcase.mode === 'drive';
    this.el.recBtn.style.display = drive ? '' : 'none';
    this.el.recBtn.textContent = this.showcase.recording ? '⏹ Stop rec' : '⏺ Record';
    this.el.recBtn.className = 'btn small' + (this.showcase.recording ? ' danger' : '');
    this.el.cloneBtn.style.display = this.showcase.demoCount > 0 ? '' : 'none';
    this.el.cloneBtn.textContent = `🎓 Learn from me (${this.showcase.demoCount})`;
  }

  private toggleRecord(): void {
    this.showcase.recording = !this.showcase.recording;
    if (this.showcase.recording) this.log('⏺ Recording: every step you drive is saved as (what the robot saw → what you pressed). Drive well!', 'learn');
    else this.log(`⏹ Recorded ${this.showcase.demoCount} steps. Press 🎓 Learn from me to train the brain to copy you.`, 'info');
    this.updateDriveUi();
  }

  private initKeyboard(): void {
    const isTyping = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      return t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable || !!t.closest?.('.blocklyWidgetDiv, .blocklyDropDownDiv'));
    };
    window.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        this.toggleTrain();
        return;
      }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
        e.preventDefault();
        this.saveProjectFile();
        return;
      }
      if (e.key === 'Escape' && this.buildTool) this.toggleBuild(false);
      if (this.showcase.mode === 'drive' && !isTyping(e) && !(e.target as HTMLElement)?.closest?.('#blockly')) {
        if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space'].includes(e.code)) e.preventDefault();
        this.showcase.keys.add(e.code);
      }
    });
    window.addEventListener('keyup', (e) => this.showcase.keys.delete(e.code));
    window.addEventListener('blur', () => this.showcase.keys.clear());
  }

  private toggleBuild(on?: boolean): void {
    const next = on ?? !this.buildTool;
    this.buildTool = next ? 'coin' : null;
    this.el.palette.style.display = next ? '' : 'none';
    this.el.buildBtn.classList.toggle('on', next);
    this.renderPalette();
    if (next) {
      if (this.view.mode === 'follow') this.setCam('zone');
      this.log('🧱 Build mode: pick an item below, then click on the ground to place it. 🧽 erases things you placed. Changes apply to training the next time you press ▶.', 'learn');
    }
    this.updateHint();
  }

  private renderPalette(): void {
    const p = this.el.palette;
    if (!p) return;
    clear(p);
    for (const t of BUILD_TOOLS) {
      p.append(
        h(
          'button',
          {
            class: this.buildTool === t.tool ? 'on' : '',
            title: t.tool === 'spawn' ? 'Set the robot start point' : `Place a ${t.label}`,
            onclick: () => {
              this.buildTool = t.tool;
              this.renderPalette();
              this.updateHint();
            },
          },
          t.emoji,
          h('small', null, t.label),
        ),
      );
    }
    p.append(
      h(
        'button',
        {
          title: 'Remove everything you placed',
          onclick: () => {
            this.edits = { objects: [], walls: [] };
            this.onEditsChanged('🧹 Removed all your builds.');
          },
        },
        '🧹',
        h('small', null, 'clear'),
      ),
    );
  }

  private updateHint(): void {
    const t = this.buildTool;
    this.el.hint.style.display = t ? '' : 'none';
    if (t) this.el.hint.textContent = t === 'erase' ? '🧽 Click something you placed to remove it' : t === 'spawn' ? '📍 Click where the robot should start' : `Click on the ground to place: ${BUILD_TOOLS.find((b) => b.tool === t)?.emoji} ${t}`;
  }

  private applyBuildTool(cx: number, cy: number): void {
    const p = this.view.pick(cx, cy);
    const tool = this.buildTool;
    if (!p || !tool) return;
    const x = Math.round(p.x * 2) / 2;
    const z = Math.round(p.z * 2) / 2;
    if (tool === 'spawn') {
      const robot = this.ws.getTopBlocks(false).find((b) => b.type === 'nb_robot');
      if (!robot) return;
      const zone = zoneAt(this.world, x, z);
      robot.setFieldValue('point', 'SPAWN');
      robot.setFieldValue(x, 'X');
      robot.setFieldValue(z, 'Z');
      if (zone !== 'world') robot.setFieldValue(zone, 'ZONE');
      this.log(`📍 Start point set to x=${x}, z=${z} (${zone}). The 🤖 robot block was updated.`, 'info');
      return;
    }
    if (tool === 'erase') {
      let best = -1;
      let bd = 2.5;
      this.edits.objects.forEach((o, i) => {
        const d = Math.hypot(o.x - p.x, o.z - p.z);
        if (d < bd) {
          bd = d;
          best = i;
        }
      });
      if (best >= 0) {
        this.edits.objects.splice(best, 1);
        this.onEditsChanged('🧽 Removed an object.');
        return;
      }
      let bw = -1;
      bd = 3;
      this.edits.walls.forEach((w, i) => {
        const d = Math.hypot(w.x - p.x, w.z - p.z);
        if (d < bd + Math.max(w.w, w.d) / 2) {
          bd = d;
          bw = i;
        }
      });
      if (bw >= 0) {
        this.edits.walls.splice(bw, 1);
        this.onEditsChanged('🧽 Removed a block.');
      } else toast('Only things you placed can be erased');
      return;
    }
    if (tool === 'wall' || tool === 'ramp' || tool === 'lava' || tool === 'ice' || tool === 'bouncy' || tool === 'spike') {
      const yaw = Math.round(this.view.controls.getAzimuthalAngle() / (Math.PI / 2)) * (Math.PI / 2);
      const wall: CustomWall = { x, z, w: tool === 'wall' ? 4 : 3, d: tool === 'wall' ? 0.5 : tool === 'ramp' ? 5 : 3, h: tool === 'wall' ? 1.6 : tool === 'ramp' ? 1.5 : 0.12, yaw, tag: tool };
      this.edits.walls.push(wall);
      this.onEditsChanged(`Placed ${tool}.`);
      return;
    }
    const type = tool as ObjType;
    const y = p.y + (type === 'ball' ? 0.6 : type === 'crate' ? 0.65 : type === 'runner' || type === 'chaser' ? 0.55 : type === 'pad' ? 0.1 : 0);
    this.edits.objects.push({ type, x, y, z, zone: zoneAt(this.world, x, z), hx: type === 'pad' ? 1.8 : undefined, hy: type === 'pad' ? 0.6 : undefined, hz: type === 'pad' ? 1.8 : undefined });
    this.onEditsChanged(`Placed ${type}.`);
  }

  private onEditsChanged(msg: string): void {
    this.rt = new WorldRuntime(this.world, this.edits);
    this.rebuildWorldMeshes();
    this.draftKey = '';
    this.compile();
    if (this.state !== 'idle') this.showBanner('🧱 World changed: stop & restart training to use your builds');
    this.log(`${msg} (${this.edits.objects.length} objects, ${this.edits.walls.length} blocks placed)`, 'info');
    this.scheduleAutosave();
  }

  // =========================================================================
  // Menus, dialogs, files
  // =========================================================================

  private openViewMenu(): void {
    const t = this.toggles;
    const mk = (label: string, key: keyof typeof t, after?: () => void) => ({
      label: `${t[key] ? '☑' : '☐'} ${label}`,
      action: () => {
        t[key] = !t[key];
        after?.();
      },
    });
    menu(this.el.viewBtn, [
      mk('Ray-vision lines', 'rays'),
      mk('Ghosts (all training worlds)', 'ghosts', () => this.client.send({ type: 'ghosts', on: t.ghosts })),
      mk('Trail', 'trail'),
      mk('Zone labels', 'labels'),
      'sep',
      mk('Exploration heatmap', 'heat', () => {
        this.heat.mesh.visible = t.heat;
        if (t.heat) {
          this.heat.paintHeat();
          this.log('🔥 Heatmap: where the training robots spend their time (brighter = more visits).', 'learn');
        }
      }),
      mk('Value map (critic)', 'value', () => {
        if (t.value) this.computeValueMap();
        else this.valueMap.mesh.visible = false;
      }),
      'sep',
      {
        label: this.showcase.deterministic ? '☑ Best action only (no randomness)' : '☐ Best action only (no randomness)',
        action: () => {
          this.showcase.deterministic = !this.showcase.deterministic;
          this.log(this.showcase.deterministic ? 'The showcase robot now always picks its most likely action.' : 'The showcase robot samples actions like during training (stochastic policy).', 'info');
        },
      },
      { label: '🔄 Restart the showcase episode', action: () => this.showcase.reset() },
      { label: '📸 Save a screenshot', action: () => download('neuroblocks.png', dataUrlToBytes(this.view.screenshot()), 'image/png') },
      { label: '🎬 Record a 10 s video', action: () => this.recordVideo(10) },
    ]);
  }

  private openFileMenu(): void {
    menu(this.el.fileBtn, [
      { label: '✏️ New blank project', action: () => this.loadMission(getMission('blank')!) },
      { label: '📂 Open project…', hint: '.json', action: () => this.openProjectFile() },
      { label: '💾 Save project', hint: 'Ctrl+S', action: () => this.saveProjectFile() },
      'sep',
      { label: '📥 Import a brain (weights)…', hint: '.json', action: () => this.importBrain() },
      'sep',
      { label: '🧹 Clear comparison curves', action: () => this.trainingPanel.clearComparisons() },
      { label: '🗑️ Reset everything (clear autosave)', action: () => this.resetAll() },
    ]);
  }

  private openExportMenu(): void {
    const name = this.compiled?.spec.name ?? 'robot';
    const ok = () => {
      if (!this.snap) {
        toast('No brain yet');
        return false;
      }
      if (!this.snap.steps) toast('⚠️ This brain is untrained (random weights). Exporting anyway.');
      return true;
    };
    menu(this.el.exportBtn, [
      {
        label: '🧾 Weights as JSON',
        hint: 'readable, re-importable',
        action: () => ok() && download(`${slug(name)}_brain.json`, JSON.stringify(this.brainExport(), null, 1), 'application/json'),
      },
      {
        label: '🔷 ONNX model',
        hint: 'onnxruntime, Unity, Netron',
        action: () => ok() && download(`${slug(name)}.onnx`, exportOnnx(this.snap!, name), 'application/octet-stream'),
      },
      {
        label: '🐍 PyTorch / NumPy script',
        hint: '.py with weights',
        action: () => ok() && download(`${slug(name)}_brain.py`, pytorchScript(this.snap!, name), 'text/x-python'),
      },
      'sep',
      { label: '📊 Training log: episodes (CSV)', hint: 'step, reward, length…', action: () => this.exportCsv('episodes') },
      { label: '📊 Training log: metrics (CSV)', hint: 'losses, entropy…', action: () => this.exportCsv('metrics') },
      'sep',
      { label: 'ℹ️ What are these formats?', action: () => this.explainExports() },
    ]);
  }

  private brainExport(): object {
    const s = this.snap!;
    return {
      ...s,
      exportedAt: new Date().toISOString(),
      robot: this.compiled?.spec.name,
      zone: this.compiled?.spec.zone,
      howToUse:
        'net.layers[i].W is a row-major [outSize x inSize] matrix (PyTorch nn.Linear layout), b the bias. Forward: x = (obs - obsNorm.mean)/sqrt(obsNorm.var) clipped to ±obsNorm.clip (if obsNorm present); then for each layer x = activation(W·x + b). Outputs: see space (discrete = one softmax per branch; continuous = means; kind "q" = Q-values for every joint action).',
    };
  }

  private exportCsv(which: 'episodes' | 'metrics'): void {
    const name = slug(this.compiled?.spec.name ?? 'robot');
    if (which === 'episodes') {
      const eps = this.trainingPanel.episodes;
      if (!eps.length) {
        toast('No finished episodes yet');
        return;
      }
      const rows = ['step,reward,length,success', ...eps.map((e) => `${e.step},${e.reward},${e.length},${e.success ? 1 : 0}`)];
      download(`${name}_episodes.csv`, rows.join('\n'), 'text/csv');
      return;
    }
    const ms = this.trainingPanel.metrics;
    if (!ms.length) {
      toast('No training updates yet');
      return;
    }
    const keys = [...new Set(ms.flatMap((m) => Object.keys(m)))];
    const rows = [keys.join(','), ...ms.map((m) => keys.map((k) => (m[k] === undefined ? '' : String(m[k]))).join(','))];
    download(`${name}_metrics.csv`, rows.join('\n'), 'text/csv');
  }

  private recordVideo(seconds = 10): void {
    const canvas = this.view.renderer.domElement as HTMLCanvasElement & { captureStream?: (fps: number) => MediaStream };
    if (!canvas.captureStream || typeof MediaRecorder === 'undefined') {
      toast('Video recording is not supported in this browser');
      return;
    }
    const stream = canvas.captureStream(30);
    const type = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4'].find((t) => MediaRecorder.isTypeSupported(t)) ?? '';
    const rec = new MediaRecorder(stream, type ? { mimeType: type, videoBitsPerSecond: 6_000_000 } : undefined);
    const chunks: Blob[] = [];
    rec.ondataavailable = (e) => {
      if (e.data.size) chunks.push(e.data);
    };
    rec.onstop = () => {
      const blob = new Blob(chunks, { type: rec.mimeType || 'video/webm' });
      const url = URL.createObjectURL(blob);
      const a = h('a', { href: url, download: `neuroblocks-${Date.now()}.${(rec.mimeType || '').includes('mp4') ? 'mp4' : 'webm'}` });
      document.body.append(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 5000);
      toast('🎬 Video saved');
    };
    rec.start(250);
    toast(`🎬 Recording the 3D view for ${seconds} s…`, 2000);
    setTimeout(() => rec.state !== 'inactive' && rec.stop(), seconds * 1000);
  }

  private explainExports(): void {
    modal(
      '💾 Exporting your robot\'s brain',
      h(
        'div',
        { class: 'explain' },
        h('p', null, h('b', null, '🧾 JSON: '), 'all weights as plain numbers, plus the input/output layout and normalisation statistics. Human-readable, and you can import it back with File → Import brain.'),
        h('p', null, h('b', null, '🔷 ONNX: '), 'the industry-standard neural network file. Open it in ', h('a', { href: 'https://netron.app', target: '_blank', rel: 'noopener' }, 'Netron'), ' to see the graph, run it with onnxruntime in Python/C#/JavaScript, or load it in Unity (Sentis). Input "obs" = the senses, outputs = logits / Q-values / action means (+ "value" for the critic).'),
        h('p', null, h('b', null, '🐍 PyTorch script: '), 'a single .py file with the weights embedded. Run it with plain NumPy, or call torch_model() to get a real PyTorch nn.Module you can fine-tune.'),
        h('p', { class: 'muted' }, 'Every export is verified to give the same outputs as the browser (to about 0.0000001).'),
      ),
      { narrow: true },
    );
  }

  private projectData(): ProjectFile {
    return {
      format: 'neuroblocks-project',
      version: 1,
      name: this.compiled?.spec.name ?? 'robot',
      savedAt: new Date().toISOString(),
      missionId: this.missionId,
      workspace: Blockly.serialization.workspaces.save(this.ws),
      edits: this.edits,
      brain: this.snap,
      episodes: this.trainingPanel.episodes.slice(-3000),
    };
  }

  private saveProjectFile(): void {
    const p = this.projectData();
    download(`${slug(p.name)}.neuroblocks.json`, JSON.stringify(p), 'application/json');
    toast('💾 Project saved (blocks + world builds + brain)');
  }

  private async openProjectFile(): Promise<void> {
    const f = await pickFile('.json,application/json');
    if (!f) return;
    try {
      const p = JSON.parse(await f.text());
      if (!validateProject(p)) throw new Error('not a NeuroBlocks project file');
      this.loadProject(p, false);
      toast(`📂 Opened "${p.name}"`);
    } catch (e) {
      toast(`⛔ Could not open: ${(e as Error).message}`, 4000);
    }
  }

  private async importBrain(): Promise<void> {
    const f = await pickFile('.json,application/json');
    if (!f) return;
    try {
      const b = JSON.parse(await f.text());
      if (!validateBrain(b)) throw new Error('not a NeuroBlocks brain file');
      if (this.state !== 'idle') this.stopTraining(true);
      this.setSnapshot(b, false);
      const fits = this.compiled && this.snapFits(b, this.compiled.spec);
      this.log(fits ? `📥 Brain imported (${fmtSteps(b.steps)} steps of training). Press ▶ to keep training it.` : '📥 Brain imported, but it does not match your current senses/actions/brain blocks, so the robot cannot use it. Load the matching blocks first.', fits ? 'good' : 'warn');
    } catch (e) {
      toast(`⛔ Could not import: ${(e as Error).message}`, 4000);
    }
  }

  private resetAll(): void {
    if (!confirm('Reset everything? This clears your autosaved project and mission progress.')) return;
    try {
      localStorage.clear();
    } catch {
      /* ignore */
    }
    location.reload();
  }

  private loadProject(p: ProjectFile, quiet: boolean): void {
    if (this.state !== 'idle') this.stopTraining(true);
    this.edits = p.edits ?? { objects: [], walls: [] };
    this.rt = new WorldRuntime(this.world, this.edits);
    this.rebuildWorldMeshes();
    this.missionId = p.missionId;
    Blockly.Events.disable();
    try {
      this.ws.clear();
      Blockly.serialization.workspaces.load(p.workspace as never, this.ws);
    } finally {
      Blockly.Events.enable();
    }
    this.el.missionTitle.textContent = p.missionId ? `🧩 ${getMission(p.missionId)?.emoji ?? ''} ${getMission(p.missionId)?.title ?? p.name}` : `🧩 ${p.name}`;
    this.snap = null;
    this.draftKey = '';
    this.compile();
    if (p.brain && validateBrain(p.brain) && this.compiled && this.snapFits(p.brain, this.compiled.spec)) this.setSnapshot(p.brain, false);
    this.trainingPanel.newRun('');
    if (p.episodes?.length) this.trainingPanel.add(p.episodes, []);
    setTimeout(() => this.ws.scroll(0, 0), 50);
    if (!quiet) this.log(`📂 Loaded project "${p.name}".`, 'info');
    else {
      const restored = this.snap as BrainSnapshot | null;
      if (restored?.steps) this.log(`♻️ Restored your last session (brain trained for ${fmtSteps(restored.steps)} steps).`, 'info');
    }
  }

  loadMission(m: Mission, quiet = false): void {
    if (this.state !== 'idle') this.stopTraining(true);
    this.missionId = m.id;
    Blockly.Events.disable();
    try {
      this.ws.clear();
      Blockly.serialization.workspaces.load(m.workspace() as never, this.ws);
    } finally {
      Blockly.Events.enable();
    }
    this.tidyBlocks();
    this.el.missionTitle.textContent = `🧩 ${m.emoji} ${m.title}`;
    this.snap = null;
    this.draftKey = '';
    this.compile();
    if (this.compiled && !this.compiled.errors.length) this.newRun(this.compiled.spec);
    this.trainingPanel.clearComparisons();
    this.showcase.clearDemos();
    this.updateDriveUi();
    if (!quiet) {
      this.log(`🎯 Mission: ${m.emoji} ${m.title}. ${m.summary}`, 'learn');
      this.log(`💡 Try next: ${m.tryNext}`, 'learn');
    }
    if (this.view.mode === 'follow') this.flyToZone(m.zone);
  }

  private openMissions(): void {
    const done = missionsDone();
    const grid = h('div', { class: 'mission-grid' });
    let close: () => void = () => {};
    for (const m of MISSIONS) {
      const z = this.world.zones.find((q) => q.id === m.zone);
      grid.append(
        h(
          'div',
          {
            class: 'mcard',
            onclick: () => {
              close();
              this.loadMission(m);
            },
          },
          done.has(m.id) ? h('span', { class: 'done', title: 'Completed (80% success)' }, '🏆') : null,
          h('div', { class: 'memoji' }, m.emoji),
          h('h4', null, m.title),
          h('div', { class: 'meta' }, h('span', { class: 'stars', title: `difficulty ${m.level}/5` }, '★'.repeat(m.level), h('span', { style: { opacity: '0.25' } }, '★'.repeat(5 - m.level))), h('span', { class: 'tagchip' }, m.algo), h('span', { class: 'tagchip' }, `⏱ ${m.time}`), h('span', { class: 'tagchip' }, `${z?.emoji} ${z?.name}`)),
          h('p', null, m.summary),
          h('ul', null, ...m.learn.map((l) => h('li', null, l))),
        ),
      );
    }
    const zones = h('div', { class: 'zone-grid' });
    for (const z of this.world.zones) {
      zones.append(
        h(
          'div',
          { class: 'zcard' },
          h('div', { class: 'zt' }, `${z.emoji} ${z.name}`),
          h('p', null, z.blurb),
          h(
            'div',
            { class: 'row' },
            h(
              'button',
              {
                class: 'btn tiny',
                onclick: () => {
                  close();
                  this.setCam('zone');
                  this.flyToZone(z.id);
                },
              },
              '🛸 Fly there',
            ),
            h(
              'button',
              {
                class: 'btn tiny accent',
                onclick: () => {
                  close();
                  this.setZone(z.id);
                },
              },
              '🤖 Train here',
            ),
          ),
        ),
      );
    }
    ({ close } = modal(
      '🎯 Missions & the world',
      h(
        'div',
        null,
        h('p', { class: 'muted', style: { marginTop: '0' } }, 'Each mission is a ready-made block program: load one, press ▶ Train, and read the blocks to see how it works. Then change them!'),
        grid,
        h('h3', { style: { margin: '18px 0 8px' } }, '🏝️ The world: 12 zones + free roam'),
        h('p', { class: 'muted', style: { marginTop: '0' } }, 'Pick where your robot trains. Combine any zone with your own senses, actions and rewards, or add objects in 🧱 Build mode.'),
        zones,
      ),
    ));
  }

  private openWelcome(): void {
    setWelcomeSeen();
    let close: () => void = () => {};
    ({ close } = modal(
      '👋 Welcome to NeuroBlocks Studio',
      h(
        'div',
        { class: 'welcome' },
        h('p', { class: 'muted', style: { marginTop: 0 } }, 'Like Scratch, but you program a robot\'s LEARNING, not its moves. You decide what it senses, what it can do and what it gets rewarded for. A neural network figures out the rest, by trial and error, right in front of you.'),
        h(
          'div',
          { class: 'steps' },
          h('div', { class: 'step' }, h('b', null, '🧩 1. Blocks = the recipe'), h('p', null, 'Left side. 🤖 where it lives, 👁️ what it senses (inputs), 🕹️ what it can do (outputs), 🧠 brain size, 🏋️ learning algorithm, and ⚡ events with 🍬 rewards.')),
          h('div', { class: 'step' }, h('b', null, '🌍 2. The world'), h('p', null, 'Middle. 12 training zones on an island: mazes, lava, soccer, parkour… Click a zone label to train there, or 🧱 Build your own course.')),
          h('div', { class: 'step' }, h('b', null, '▶ 3. Train & watch'), h('p', null, '16 copies of your robot train in parallel (👻 ghosts). The robot you see always uses the newest brain. Grab the controls with 🎮 You drive.')),
          h('div', { class: 'step' }, h('b', null, '🔧 4. Under the hood'), h('p', null, 'Right side. Live learning curves, every neuron and weight, the exact sensor numbers, the maths, and the generated code. Export the trained brain as JSON / ONNX / PyTorch.')),
        ),
        h('p', { class: 'small muted' }, 'Everything (neural network, backpropagation, PPO, DQN, evolution, physics) is written from scratch in TypeScript and runs in your browser. No servers, no ML libraries, nothing hidden.'),
        h('p', { class: 'small muted' }, '⌨️ Ctrl+Enter train/pause · Ctrl+S save project · WASD/arrows drive (🎮 mode) · Space jump · F kick · Esc leave build mode · mouse: drag to orbit, right-drag to pan, wheel to zoom.'),
      ),
      {
        foot: [
          h(
            'button',
            {
              class: 'btn',
              onclick: () => {
                close();
                this.openMissions();
              },
            },
            '🎯 Browse missions',
          ),
          h(
            'button',
            {
              class: 'btn primary',
              onclick: () => {
                close();
                if (this.missionId !== 'reach-flag') this.loadMission(getMission('reach-flag')!);
                setTimeout(() => this.startTraining('rl'), 300);
              },
            },
            '🚩 Start the first mission',
          ),
        ],
      },
    ));
  }

  private refreshCode(): void {
    this.codeDirty = false;
    this.codePanel.update(this.compiled && !this.compiled.errors.length ? this.compiled.spec : null, this.layout, this.snap);
  }

  private showBanner(msg: string | null): void {
    this.el.banner.style.display = msg ? '' : 'none';
    if (msg) this.el.banner.textContent = msg;
  }

  // =========================================================================

  log(msg: string, level: 'info' | 'warn' | 'err' | 'good' | 'learn' = 'info'): void {
    const c = this.el.console;
    const t = new Date();
    const ts = `${String(t.getHours()).padStart(2, '0')}:${String(t.getMinutes()).padStart(2, '0')}:${String(t.getSeconds()).padStart(2, '0')}`;
    c.append(h('div', { class: `ln ${level}` }, h('span', { class: 't' }, ts), msg));
    while (c.childElementCount > 300) c.firstElementChild!.remove();
    c.scrollTop = c.scrollHeight;
  }

  private scheduleAutosave(): void {
    clearTimeout(this.autosaveTimer);
    this.autosaveTimer = window.setTimeout(() => this.autosave(), 1500);
  }

  private autosave(): void {
    if (!this.ws) return;
    saveLocal(this.projectData());
  }

  startAutosaveLoop(): void {
    setInterval(() => this.autosave(), 15000);
  }
}

function dataUrlToBytes(url: string): Uint8Array {
  const b64 = url.split(',')[1] ?? '';
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
