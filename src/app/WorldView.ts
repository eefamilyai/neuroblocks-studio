/**
 * Everything about the 3D world view: the viewport, robot & objects, ghosts,
 * ray lines, overlays (heatmap, value map), the HUD, zone labels, reward
 * popups, camera modes and build mode (place / select / move / rotate).
 */
import * as THREE from 'three';
import { Viewport } from '../render/viewport.ts';
import type { CamMode, Quality } from '../render/viewport.ts';
import { buildWorldMeshes } from '../render/worldMesh.ts';
import type { WorldMeshes } from '../render/worldMesh.ts';
import { GhostLayer, ObjectLayer, RobotMesh, SensorLayer, Trail } from '../render/actors.ts';
import { GridOverlay } from '../render/overlays.ts';
import { zoneAt } from '../sim/runtime.ts';
import type { CustomWall, WorldEdits, WorldRuntime } from '../sim/runtime.ts';
import type { AgentSpec, ObjType, WorldDef } from '../sim/types.ts';
import type { Showcase, StepInfo } from '../showcase.ts';
import type { DriveMode } from '../showcase.ts';
import { btn, clear, download, fmt, fmtSteps, h, menu, seg, setText, toast } from '../ui/dom.ts';
import type { MenuItem } from '../ui/dom.ts';
import { icon } from '../ui/icons.ts';
import type { IconName } from '../ui/icons.ts';

type BuildTool = ObjType | CustomWall['tag'] | 'erase' | 'spawn' | 'select';

const BUILD_TOOLS: { tool: BuildTool; emoji: string; label: string }[] = [
  { tool: 'select', emoji: '👆', label: 'select' },
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

export interface WorldViewHost {
  showcase: Showcase;
  world: WorldDef;
  edits: WorldEdits;
  isTraining(): boolean;
  trainingSteps(): number;
  setZone(id: string): void;
  setSpawn(x: number, z: number, zone: string): void;
  editsChanged(msg: string): void;
  setDriveMode(m: DriveMode): void;
  toggleRecord(): void;
  startCloning(): void;
  ghostsChanged(on: boolean): void;
  log(msg: string, level?: 'info' | 'warn' | 'err' | 'good' | 'learn'): void;
  /** called every rendered frame (for panels) */
  frame(dt: number): void;
}

type Sel = { kind: 'obj'; i: number } | { kind: 'wall'; i: number } | null;

export class WorldView {
  readonly view: Viewport;
  private host: WorldViewHost;
  private el: HTMLElement;
  private meshes: WorldMeshes | null = null;
  private robot = new RobotMesh('#4fc3f7');
  readonly objLayer = new ObjectLayer();
  private ghosts = new GhostLayer();
  private sensors = new SensorLayer();
  private trail = new Trail();
  readonly heat = new GridOverlay(64);
  private valueMap = new GridOverlay(48);
  toggles = { rays: true, ghosts: true, trail: true, heat: false, value: false, labels: true, fps: false };
  private heatDirty = false;
  private lastHeatPaint = 0;
  // overlays
  private overlay: HTMLElement;
  private zoneLabels: { el: HTMLElement; x: number; z: number; id: string; shown: boolean }[] = [];
  private popups: { el: HTMLElement; x: number; y: number; z: number; born: number }[] = [];
  private lastLabels = 0;
  private lastHud = 0;
  private hudCard: { zone: HTMLElement; big: HTMLElement; ep: HTMLElement; step: HTMLElement; last: HTMLElement; vLbl: HTMLElement; v: HTMLElement };
  private badges: HTMLElement;
  private actHud: HTMLElement;
  private actKey = '';
  private actCells: { fill: HTMLElement; cell: HTMLElement; text: HTMLElement | null }[][] = [];
  private banner: HTMLElement;
  private bannerText: HTMLElement;
  private hint: HTMLElement;
  private fpsEl: HTMLElement;
  // toolbar
  private camSeg: { el: HTMLElement; set: (v: CamMode) => void };
  private driveSeg: { el: HTMLElement; set: (v: DriveMode) => void };
  private zoneSel: HTMLSelectElement;
  private recBtn: HTMLButtonElement;
  private cloneBtn: HTMLButtonElement;
  private buildBtn: HTMLButtonElement;
  private focusBtn: HTMLButtonElement;
  private speedSel: HTMLSelectElement;
  onFocusToggle: (() => void) | null = null;
  // build mode
  private buildTool: BuildTool | null = null;
  private palette: HTMLElement;
  private inspect: HTMLElement;
  private sel: Sel = null;
  private selBox: THREE.LineSegments;
  private drag: { pointerId: number; ox: number; oz: number } | null = null;
  private spec: AgentSpec | null = null;

  constructor(el: HTMLElement, host: WorldViewHost, quality: Quality) {
    this.el = el;
    this.host = host;
    this.view = new Viewport(el, quality);
    const sc = this.view.scene;
    sc.add(this.robot, this.objLayer.group, this.ghosts.mesh, this.sensors.group, this.trail.line, this.heat.mesh, this.valueMap.mesh);
    this.selBox = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)), new THREE.LineBasicMaterial({ color: 0x8fa6ff, depthTest: false, transparent: true }));
    this.selBox.renderOrder = 10;
    this.selBox.visible = false;
    sc.add(this.selBox);
    this.view.onFrame = (dt, t) => this.frame(dt, t);

    // ---- overlays & HUD
    this.overlay = h('div', { style: { position: 'absolute', inset: '0', pointerEvents: 'none', overflow: 'hidden', zIndex: '3' } });
    const kv = (label: string) => {
      const b = h('b', null, '–');
      return [h('span', null, label), b] as const;
    };
    const [l1, ep] = kv('episode');
    const [l2, step] = kv('step');
    const [l3, last] = kv('last reward');
    const [vLbl, v] = kv('critic V(s)');
    this.hudCard = { zone: h('div', { class: 'zone' }), big: h('div', { class: 'big' }, '0.00'), ep, step, last, vLbl, v };
    const hudBL = h('div', { class: 'hud hud-bl' }, h('div', { class: 'hud-card' }, this.hudCard.zone, this.hudCard.big, h('div', { class: 'hud-kv' }, l1, ep, l2, step, l3, last, vLbl, v)));
    this.actHud = h('div', { class: 'hud-card act-card' });
    const hudBR = h('div', { class: 'hud hud-br' }, this.actHud);
    this.badges = h('div', { class: 'hud hud-tc' });
    this.bannerText = h('span');
    this.banner = h('div', { class: 'banner', style: { display: 'none' } }, icon('info'), this.bannerText);
    this.hint = h('div', { class: 'cursor-hint', style: { display: 'none' } });
    this.palette = h('div', { class: 'build-bar', style: { display: 'none' } });
    this.inspect = h('div', { class: 'inspect-card', style: { display: 'none' } });
    this.fpsEl = h('div', { class: 'fps', style: { display: 'none' } });

    // ---- toolbars
    this.zoneSel = h('select', { class: 'sel', title: 'Which zone the robot trains in', onchange: () => host.setZone(this.zoneSel.value) }) as HTMLSelectElement;
    for (const z of host.world.zones) this.zoneSel.append(h('option', { value: z.id }, `${z.emoji} ${z.name}`));
    this.camSeg = seg<CamMode>(
      [
        { value: 'follow', icon: 'camcorder', title: 'Chase camera' },
        { value: 'orbit', icon: 'orbit', title: 'Orbit around the robot' },
        { value: 'zone', icon: 'map', title: 'Zone overview' },
        { value: 'top', icon: 'scan', title: 'Top-down' },
        { value: 'free', icon: 'move', title: 'Free camera (drag to orbit, right-drag to pan, wheel to zoom)' },
      ],
      (m) => this.setCam(m),
      'icons',
    );
    this.driveSeg = seg<DriveMode>(
      [
        { value: 'ai', icon: 'bot', label: 'AI', title: 'Watch the current brain drive' },
        { value: 'drive', icon: 'gamepad', label: 'You', title: 'You drive: WASD / arrows, Space = jump, F = kick' },
        { value: 'paused', icon: 'pause', title: 'Freeze the 3D view' },
      ],
      (m) => host.setDriveMode(m),
    );
    this.speedSel = h('select', { class: 'sel', title: 'Speed of the 3D view (not of training)', onchange: () => (host.showcase.speed = Number(this.speedSel.value)) }) as HTMLSelectElement;
    for (const s of [0.25, 0.5, 1, 2, 4, 8]) this.speedSel.append(h('option', { value: String(s), selected: s === 1 }, `${s}×`));
    this.recBtn = btn({ icon: 'record', label: 'Record', cls: 'sm', title: 'Record your driving as training examples', onclick: () => host.toggleRecord() });
    this.cloneBtn = btn({ icon: 'learn', label: 'Learn from me', cls: 'sm warn', title: 'Train the brain to copy your recorded driving (supervised learning)', onclick: () => host.startCloning() });
    this.recBtn.style.display = this.cloneBtn.style.display = 'none';
    const layersBtn = btn({ icon: 'layers', cls: 'ghost sm', title: 'What to show: rays, ghosts, heatmap, value map…' });
    layersBtn.addEventListener('click', () => this.openViewMenu(layersBtn));
    this.buildBtn = btn({ icon: 'hammer', cls: 'ghost sm', title: 'Build mode: place, move and delete your own objects and walls', onclick: () => this.toggleBuild() });
    this.focusBtn = btn({ icon: 'maximize', cls: 'ghost sm', title: 'Focus: give the world the whole window', onclick: () => this.onFocusToggle?.() });
    const tl = h('div', { class: 'vp-bar vp-tl' }, this.zoneSel, h('div', { class: 'tb-sep' }), this.camSeg.el);
    const tr = h('div', { class: 'vp-bar vp-tr' }, this.recBtn, this.cloneBtn, this.driveSeg.el, this.speedSel, layersBtn, this.buildBtn, this.focusBtn);
    el.append(this.overlay, tl, tr, this.badges, hudBL, hudBR, this.banner, this.hint, this.palette, this.inspect, this.fpsEl);

    // ---- zone labels
    for (const z of host.world.zones) {
      if (z.id === 'world') continue;
      const lbl = h('div', { class: 'zone-label', title: `${z.blurb}\n\nClick to train here.`, onclick: () => host.setZone(z.id) }, `${z.emoji} ${z.name}`);
      this.overlay.append(lbl);
      this.zoneLabels.push({ el: lbl, x: (z.rect.minX + z.rect.maxX) / 2, z: (z.rect.minZ + z.rect.maxZ) / 2, id: z.id, shown: true });
    }

    // ---- picking (build mode)
    const canvas = this.view.renderer.domElement;
    let down: [number, number] | null = null;
    canvas.addEventListener('pointerdown', (e) => {
      down = [e.clientX, e.clientY];
      if (this.buildTool === 'select' && this.sel && e.button === 0) this.maybeStartDrag(e);
    });
    canvas.addEventListener('pointermove', (e) => this.dragMove(e));
    canvas.addEventListener('pointerup', (e) => {
      if (this.drag) {
        this.endDrag(e);
        down = null;
        return;
      }
      if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return;
      down = null;
      if (this.buildTool) this.applyBuildTool(e.clientX, e.clientY);
    });
    this.setCam('follow');
    this.renderPalette();
  }

  // ------------------------------------------------------------------ world
  setRuntime(rt: WorldRuntime): void {
    if (this.meshes) {
      this.view.scene.remove(this.meshes.group);
      this.meshes.dispose();
    }
    this.meshes = buildWorldMeshes(rt.def, rt.statics);
    this.view.scene.add(this.meshes.group);
    this.view.pickables = this.meshes.group.children.filter((c) => (c as THREE.Mesh).isMesh && !(c as THREE.Mesh).geometry.type.startsWith('Sphere'));
    if (this.meshes.gates) this.meshes.gates.visible = this.spec?.zone !== 'world';
    this.refreshSelection();
  }

  /** The showcase got a new spec: re-bind objects, colour, overlays. */
  bindSpec(spec: AgentSpec, zoneChanged: boolean): void {
    this.spec = spec;
    this.robot.setColor(spec.body.color);
    if (this.host.showcase.env) this.objLayer.bind(this.host.showcase.env);
    this.trail.clear();
    this.zoneSel.value = spec.zone;
    for (const l of this.zoneLabels) l.el.classList.toggle('current', l.id === spec.zone);
    const z = this.host.world.zones.find((q) => q.id === spec.zone);
    if (z) {
      const y = (z.floorY > -5 ? z.floorY : 0) + 0.06;
      this.heat.place(z.rect, y);
      this.valueMap.place(z.rect, y + 0.02);
    }
    if (zoneChanged) {
      this.heat.clear();
      this.valueMap.mesh.visible = false;
      this.toggles.value = false;
      this.flyToZone(spec.zone);
    }
    if (this.meshes?.gates) this.meshes.gates.visible = spec.zone !== 'world';
  }

  setGhosts(data: Float32Array | null): void {
    this.ghosts.set(this.toggles.ghosts ? data : null);
    if (data) {
      for (let i = 0; i < data.length; i += 4) this.heat.addVisit(data[i], data[i + 2]);
      this.heatDirty = true;
    }
  }

  clearTrail(): void {
    this.trail.clear();
  }

  setBanner(msg: string | null): void {
    this.banner.style.display = msg ? '' : 'none';
    if (msg) setText(this.bannerText, msg);
  }

  setDriveUi(mode: DriveMode, recording: boolean, demos: number): void {
    this.driveSeg.set(mode);
    this.recBtn.style.display = mode === 'drive' ? '' : 'none';
    this.recBtn.replaceChildren(icon(recording ? 'stop' : 'record'), h('span', null, recording ? 'Stop' : 'Record'));
    this.recBtn.className = 'btn sm' + (recording ? ' danger' : '');
    this.cloneBtn.style.display = demos > 0 ? '' : 'none';
    this.cloneBtn.replaceChildren(icon('learn'), h('span', null, `Learn from me (${demos})`));
  }

  setFocusIcon(on: boolean): void {
    this.focusBtn.replaceChildren(icon(on ? 'minimize' : 'maximize'));
    this.focusBtn.title = on ? 'Show the editor and inspector again' : 'Focus: give the world the whole window';
  }

  // ------------------------------------------------------------------ per frame
  private frame(dt: number, t: number): void {
    const sc = this.host.showcase;
    sc.update(dt);
    const env = sc.env;
    const training = this.host.isTraining();
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
    }
    this.ghosts.mesh.visible = this.toggles.ghosts && training;
    this.trail.line.visible = this.toggles.trail;
    this.meshes?.update(t);
    this.view.idle = sc.mode === 'paused' && !training;
    const now = performance.now();
    if (this.heatDirty && this.toggles.heat && now - this.lastHeatPaint > 600) {
      this.heat.paintHeat();
      this.heatDirty = false;
      this.lastHeatPaint = now;
    }
    if (now - this.lastHud > 100) {
      this.lastHud = now;
      this.renderHud();
    }
    if (now - this.lastLabels > 33) {
      this.lastLabels = now;
      this.positionOverlays();
    }
    this.host.frame(dt);
  }

  onStep(info: StepInfo): void {
    const env = this.host.showcase.env!;
    this.robot.react(info.reward);
    if (this.toggles.trail) this.trail.push(env.a.x, env.a.y - 0.35, env.a.z);
    this.heat.addVisit(env.a.x, env.a.z, 0.2);
    if (Math.abs(info.reward) >= 0.25) this.popup(info.reward > 0 ? `+${fmt(info.reward, 2)}` : fmt(info.reward, 2), info.reward > 0 ? 'pos' : 'neg');
    for (const e of info.events) if (/lap|kick/.test(e)) this.popup(e, 'info');
    if (info.done) {
      this.popup(info.success ? 'success!' : env.truncated ? 'time up' : 'failed', info.success ? 'pos' : 'info');
      this.trail.clear();
    }
  }

  private popup(text: string, cls: 'pos' | 'neg' | 'info'): void {
    const env = this.host.showcase.env;
    if (!env) return;
    const el = h('div', { class: `popup ${cls}` }, text);
    this.overlay.append(el);
    this.popups.push({ el, x: env.a.x, y: env.a.y + 1.3, z: env.a.z, born: performance.now() });
    setTimeout(() => el.remove(), 1100);
    if (this.popups.length > 30) this.popups.shift()!.el.remove();
    this.placeOne(this.popups[this.popups.length - 1]);
  }

  private placeOne(p: { el: HTMLElement; x: number; y: number; z: number }): void {
    const s = this.view.project(p.x, p.y, p.z);
    if (s) p.el.style.transform = `translate(${s[0].toFixed(1)}px, ${s[1].toFixed(1)}px)`;
  }

  private positionOverlays(): void {
    const now = performance.now();
    if (this.popups.length) {
      this.popups = this.popups.filter((p) => now - p.born < 1100);
      for (const p of this.popups) this.placeOne(p);
    }
    // zone labels only move when the camera does
    if (!this.view.cameraMoved) return;
    this.view.cameraMoved = false;
    const cam = this.view.camera.position;
    for (const l of this.zoneLabels) {
      const d = Math.hypot(cam.x - l.x, cam.z - l.z);
      const s = this.toggles.labels && d > 30 && d < 420 ? this.view.project(l.x, 16, l.z) : null;
      if (!s) {
        if (l.shown) {
          l.el.style.display = 'none';
          l.shown = false;
        }
        continue;
      }
      if (!l.shown) {
        l.el.style.display = '';
        l.shown = true;
      }
      l.el.style.transform = `translate(${(s[0] - l.el.offsetWidth / 2).toFixed(1)}px, ${(s[1] - 12).toFixed(1)}px)`;
      l.el.style.opacity = String(Math.max(0.35, Math.min(1, (420 - d) / 200)).toFixed(2));
    }
  }

  private renderHud(): void {
    const sc = this.host.showcase;
    const env = sc.env;
    const spec = sc.spec;
    if (this.toggles.fps) setText(this.fpsEl, `${this.view.fps.toFixed(0)} fps · ${this.view.pixelRatio.toFixed(2)}×`);
    if (!env || !spec) return;
    const zone = this.host.world.zones.find((z) => z.id === spec.zone);
    const c = this.hudCard;
    setText(c.zone, `${zone?.emoji ?? ''} ${zone?.name ?? ''}`);
    setText(c.big, fmt(env.epReturn, 2));
    c.big.className = 'big ' + (env.epReturn > 0.005 ? 'pos' : env.epReturn < -0.005 ? 'neg' : '');
    setText(c.ep, String(env.episode));
    setText(c.step, `${env.steps} / ${env.maxSteps}`);
    setText(c.last, fmt(env.stepReward, 3));
    const b = sc.brain;
    const fits = !!b && sc.brainFits();
    const hasV = fits && !!b!.critic;
    c.vLbl.style.display = c.v.style.display = hasV ? '' : 'none';
    if (hasV) setText(c.v, fmt(b!.lastValue, 2));
    // badges
    const badges: [IconName, string, string][] = [];
    if (sc.mode === 'drive') badges.push(['gamepad', 'You are driving', 'you']);
    else if (sc.mode === 'paused') badges.push(['pause', 'Paused', '']);
    else badges.push(['brain', fits ? (b!.snap.steps ? `Brain @ ${fmtSteps(b!.snap.steps)} steps${sc.label ? ' · ' + sc.label : ''}` : 'Untrained brain (random)') : 'Random actions', '']);
    if (sc.recording) badges.push(['record', `REC ${sc.demoCount} steps`, 'rec']);
    else if (sc.demoCount) badges.push(['camcorder', `${sc.demoCount} recorded steps`, '']);
    if (this.host.isTraining()) badges.push(['activity', `Training · ${fmtSteps(this.host.trainingSteps())}`, '']);
    const key = badges.map((x) => x.join()).join('|');
    if (this.badges.dataset.key !== key) {
      this.badges.dataset.key = key;
      clear(this.badges);
      for (const [ic, text, cls] of badges) this.badges.append(h('span', { class: `hud-badge ${cls}` }, cls === 'rec' ? h('span', { class: 'dot' }) : icon(ic), text));
    }
    this.renderActionHud(fits ? b : null);
  }

  private renderActionHud(b: Showcase['brain'] | null): void {
    const sc = this.host.showcase;
    const spec = sc.spec!;
    const key = `${spec.actionMode}|${spec.axes.map((a) => a.labels.join(',')).join(';')}`;
    if (key !== this.actKey) {
      this.actKey = key;
      clear(this.actHud);
      this.actCells = [];
      for (const ax of spec.axes) {
        const opts = h('div', { class: 'act-opts' });
        const cells: { fill: HTMLElement; cell: HTMLElement; text: HTMLElement | null }[] = [];
        const labels = spec.actionMode === 'discrete' ? ax.labels : [''];
        for (const lab of labels) {
          const fill = h('i');
          const text = h('span', null, lab);
          const cell = h('div', { class: 'act-opt' }, fill, text);
          opts.append(cell);
          cells.push({ fill, cell, text });
        }
        this.actHud.append(h('div', { class: 'act-row' }, h('span', { class: 'lbl' }, ax.axis), opts));
        this.actCells.push(cells);
      }
    }
    const probs = b && b.snap.kind === 'policy' ? b.lastProbs : null;
    spec.axes.forEach((_ax, k) => {
      const cells = this.actCells[k];
      if (!cells) return;
      if (spec.actionMode === 'discrete') {
        cells.forEach((c, j) => {
          const chosen = sc.action[k] === j;
          const p = probs?.[k]?.[j] ?? (chosen ? 1 : 0);
          c.fill.style.width = `${(p * 100).toFixed(1)}%`;
          c.cell.classList.toggle('chosen', chosen);
        });
      } else {
        const v = sc.action[k];
        cells[0].fill.style.width = `${(((Math.max(-1, Math.min(1, v)) + 1) / 2) * 100).toFixed(1)}%`;
        if (cells[0].text) setText(cells[0].text, fmt(v, 2));
      }
    });
  }

  // ------------------------------------------------------------------ camera
  setCam(m: CamMode): void {
    this.view.setMode(m);
    if (m === 'zone') this.flyToZone(this.spec?.zone ?? 'meadow', true);
    this.camSeg.set(m);
  }

  flyToZone(id: string, force = false): void {
    const z = this.host.world.zones.find((q) => q.id === id);
    if (!z) return;
    if (!force && (this.view.mode === 'follow' || this.view.mode === 'orbit')) return;
    const c = new THREE.Vector3((z.rect.minX + z.rect.maxX) / 2, z.floorY > -5 ? z.floorY : 0, (z.rect.minZ + z.rect.maxZ) / 2);
    const size = Math.max(z.rect.maxX - z.rect.minX, z.rect.maxZ - z.rect.minZ);
    this.view.flyTo(c, id === 'world' ? 230 : size * 1.25, 0.75, Math.PI * 0.15);
  }

  // ------------------------------------------------------------------ view menu
  private openViewMenu(anchor: HTMLElement): void {
    const t = this.toggles;
    const sc = this.host.showcase;
    const mk = (label: string, key: keyof typeof t, after?: () => void): MenuItem => ({
      label,
      checked: t[key],
      action: () => {
        t[key] = !t[key];
        after?.();
      },
    });
    menu(anchor, [
      { heading: 'Show' },
      mk('Ray-vision lines', 'rays'),
      mk('Ghosts (all training worlds)', 'ghosts', () => this.host.ghostsChanged(t.ghosts)),
      mk('Trail', 'trail'),
      mk('Zone labels', 'labels', () => (this.view.cameraMoved = true)),
      mk('Frame rate', 'fps', () => (this.fpsEl.style.display = t.fps ? '' : 'none')),
      { heading: 'Maps on the ground' },
      mk('Exploration heatmap', 'heat', () => {
        this.heat.mesh.visible = t.heat;
        if (t.heat) {
          this.heat.paintHeat();
          this.host.log('Heatmap: where the training robots spend their time (brighter = more visits).', 'learn');
        }
      }),
      mk('Value map (critic)', 'value', () => {
        if (t.value) this.computeValueMap();
        else this.valueMap.mesh.visible = false;
      }),
      { heading: 'Robot' },
      {
        label: 'Best action only (no randomness)',
        checked: sc.deterministic,
        action: () => {
          sc.deterministic = !sc.deterministic;
          this.host.log(sc.deterministic ? 'The 3D robot now always picks its most likely action.' : 'The 3D robot samples actions like during training (stochastic policy).', 'info');
        },
      },
      { label: 'Restart the episode', checked: false, action: () => sc.reset() },
      'sep',
      { label: 'Save a screenshot', checked: false, action: () => this.saveScreenshot() },
      { label: 'Record a 10 s video', checked: false, action: () => this.recordVideo(10) },
    ]);
  }

  saveScreenshot(): void {
    const url = this.view.screenshot();
    const b64 = url.split(',')[1] ?? '';
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    download('neuroblocks.png', out, 'image/png');
  }

  recordVideo(seconds = 10): void {
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
      download(`neuroblocks-${Date.now()}.${(rec.mimeType || '').includes('mp4') ? 'mp4' : 'webm'}`, new Blob(chunks, { type: rec.mimeType || 'video/webm' }));
      toast('Video saved', 2600, 'camcorder');
    };
    const wasCap = this.view.maxFps;
    this.view.maxFps = 0;
    rec.start(250);
    toast(`Recording the 3D view for ${seconds} s…`, 2000, 'record');
    setTimeout(() => {
      if (rec.state !== 'inactive') rec.stop();
      this.view.maxFps = wasCap;
    }, seconds * 1000);
  }

  /** Paint V(s) over the zone: what the critic thinks each spot is worth. */
  computeValueMap(): void {
    const sc = this.host.showcase;
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
    const savedObs = Float32Array.from(env.obs);
    const vm = this.valueMap;
    const vals = new Float32Array(vm.res * vm.res);
    const yaws = [0, Math.PI / 2, Math.PI, -Math.PI / 2];
    for (let k = 0; k < vals.length; k++) {
      const [x, z] = vm.cellCenter(k);
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
        env.observe('peek');
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
    env.obs.set(savedObs);
    const r = vm.paintValues(vals);
    vm.mesh.visible = true;
    this.host.log(`Value map: what the critic thinks each spot is worth (yellow = high ${fmt(r.max, 2)}, purple = low ${fmt(r.min, 2)}), averaged over 4 headings.`, 'learn');
  }

  // ------------------------------------------------------------------ build mode
  get building(): boolean {
    return !!this.buildTool;
  }

  toggleBuild(on?: boolean): void {
    const next = on ?? !this.buildTool;
    this.buildTool = next ? 'coin' : null;
    this.palette.style.display = next ? '' : 'none';
    this.el.classList.toggle('building', next);
    this.buildBtn.classList.toggle('on', next);
    this.select(null);
    this.renderPalette();
    if (next) {
      if (this.view.mode === 'follow') this.setCam('zone');
      this.host.log('Build mode: pick an item, then click the ground to place it. 👆 Select lets you move (drag), rotate (R), resize or delete (Del) what you placed. Changes apply to training the next time you press Train.', 'learn');
    }
    this.updateHint();
  }

  private renderPalette(): void {
    const p = this.palette;
    clear(p);
    BUILD_TOOLS.forEach((t, i) => {
      if (i === 2 || i === 10 || i === 15) p.append(h('div', { class: 'sep' }));
      p.append(
        h(
          'button',
          {
            class: this.buildTool === t.tool ? 'on' : '',
            title: t.tool === 'spawn' ? 'Set the robot start point' : t.tool === 'select' ? 'Select what you placed: drag to move, R to rotate, Del to delete' : t.tool === 'erase' ? 'Erase things you placed' : `Place a ${t.label}`,
            onclick: () => {
              this.buildTool = t.tool;
              if (t.tool !== 'select') this.select(null);
              this.renderPalette();
              this.updateHint();
            },
          },
          t.emoji,
          h('small', null, t.label),
        ),
      );
    });
    p.append(
      h(
        'button',
        {
          title: "Remove the zone's built-in objects (a blank canvas) / bring them back",
          onclick: () => {
            const zone = this.spec?.zone;
            if (!zone || zone === 'world') return toast('Pick a single zone first');
            const e = this.host.edits;
            const hidden = new Set(e.hidden ?? []);
            if (hidden.has(zone)) hidden.delete(zone);
            else hidden.add(zone);
            e.hidden = [...hidden];
            this.host.editsChanged(hidden.has(zone) ? 'Removed the zone\'s built-in objects.' : 'Brought the zone\'s built-in objects back.');
          },
        },
        '🫥',
        h('small', null, 'blank'),
      ),
      h(
        'button',
        {
          title: 'Remove everything you placed',
          onclick: () => {
            const e = this.host.edits;
            e.objects.length = 0;
            e.walls.length = 0;
            this.select(null);
            this.host.editsChanged('Removed all your builds.');
          },
        },
        '🧹',
        h('small', null, 'clear'),
      ),
    );
  }

  private updateHint(): void {
    const t = this.buildTool;
    this.hint.style.display = t ? '' : 'none';
    if (!t) return;
    setText(this.hint, t === 'erase' ? 'Click something you placed to remove it' : t === 'spawn' ? 'Click where the robot should start' : t === 'select' ? (this.sel ? 'Drag to move · R rotate · Del delete · Esc deselect' : 'Click something you placed') : `Click the ground to place: ${BUILD_TOOLS.find((b) => b.tool === t)?.emoji} ${t}`);
  }

  private nearestEdit(px: number, pz: number): Sel {
    const e = this.host.edits;
    let best: Sel = null;
    let bd = 2.5;
    e.objects.forEach((o, i) => {
      const d = Math.hypot(o.x - px, o.z - pz);
      if (d < bd) {
        bd = d;
        best = { kind: 'obj', i };
      }
    });
    if (best) return best;
    let bw = Infinity;
    e.walls.forEach((w, i) => {
      // distance in the wall's own frame
      const dx = px - w.x;
      const dz = pz - w.z;
      const c = Math.cos(w.yaw);
      const s = Math.sin(w.yaw);
      const lx = Math.abs(dx * c - dz * s) - w.w / 2;
      const lz = Math.abs(dx * s + dz * c) - w.d / 2;
      const d = Math.hypot(Math.max(0, lx), Math.max(0, lz));
      if (d < 1.2 && d < bw) {
        bw = d;
        best = { kind: 'wall', i };
      }
    });
    return best;
  }

  private applyBuildTool(cx: number, cy: number): void {
    const p = this.view.pick(cx, cy);
    const tool = this.buildTool;
    if (!p || !tool) return;
    const x = Math.round(p.x * 2) / 2;
    const z = Math.round(p.z * 2) / 2;
    const e = this.host.edits;
    if (tool === 'select') {
      this.select(this.nearestEdit(p.x, p.z));
      return;
    }
    if (tool === 'spawn') {
      this.host.setSpawn(x, z, zoneAt(this.host.world, x, z));
      return;
    }
    if (tool === 'erase') {
      const s = this.nearestEdit(p.x, p.z);
      if (!s) return toast('Only things you placed can be erased');
      if (s.kind === 'obj') e.objects.splice(s.i, 1);
      else e.walls.splice(s.i, 1);
      this.host.editsChanged(s.kind === 'obj' ? 'Removed an object.' : 'Removed a block.');
      return;
    }
    if (tool === 'wall' || tool === 'ramp' || tool === 'lava' || tool === 'ice' || tool === 'bouncy' || tool === 'spike') {
      const yaw = Math.round(this.view.controls.getAzimuthalAngle() / (Math.PI / 2)) * (Math.PI / 2);
      e.walls.push({ x, z, w: tool === 'wall' ? 4 : 3, d: tool === 'wall' ? 0.5 : tool === 'ramp' ? 5 : 3, h: tool === 'wall' ? 1.6 : tool === 'ramp' ? 1.5 : 0.12, yaw, tag: tool });
      this.host.editsChanged(`Placed ${tool}.`);
      return;
    }
    const type = tool as ObjType;
    const y = p.y + (type === 'ball' ? 0.6 : type === 'crate' ? 0.65 : type === 'runner' || type === 'chaser' ? 0.55 : type === 'pad' ? 0.1 : 0);
    e.objects.push({ type, x, y, z, zone: zoneAt(this.host.world, x, z), hx: type === 'pad' ? 1.8 : undefined, hy: type === 'pad' ? 0.6 : undefined, hz: type === 'pad' ? 1.8 : undefined });
    this.host.editsChanged(`Placed ${type}.`);
  }

  // ---- selection, inspector card, drag & rotate
  private select(s: Sel): void {
    this.sel = s;
    this.refreshSelection();
    this.updateHint();
  }

  /** Keep the highlight box & card in sync with the edits (they may have changed). */
  private refreshSelection(): void {
    const s = this.sel;
    const e = this.host.edits;
    if (s && ((s.kind === 'obj' && !e.objects[s.i]) || (s.kind === 'wall' && !e.walls[s.i]))) this.sel = null;
    const sel = this.sel;
    this.selBox.visible = !!sel;
    this.inspect.style.display = sel ? '' : 'none';
    if (!sel) return;
    if (sel.kind === 'obj') {
      const o = e.objects[sel.i];
      this.selBox.position.set(o.x, o.y + 0.6, o.z);
      this.selBox.rotation.set(0, 0, 0);
      this.selBox.scale.set(1.6, 1.6, 1.6);
    } else {
      const w = e.walls[sel.i];
      const flat = w.tag !== 'wall' && w.tag !== 'ramp';
      const hh = flat ? 0.2 : w.h;
      this.selBox.position.set(w.x, hh / 2, w.z);
      this.selBox.rotation.set(0, w.yaw, 0);
      this.selBox.scale.set(w.w + 0.1, hh + 0.1, w.d + 0.1);
    }
    this.renderInspect();
  }

  private renderInspect(): void {
    const sel = this.sel;
    if (!sel) return;
    const e = this.host.edits;
    clear(this.inspect);
    const num = (label: string, v: number, set: (n: number) => void, step = 0.5) => {
      const inp = h('input', { class: 'input', type: 'number', step: String(step), value: String(Math.round(v * 100) / 100) }) as HTMLInputElement;
      inp.addEventListener('change', () => {
        const n = Number(inp.value);
        if (!Number.isFinite(n)) return;
        set(n);
        this.host.editsChanged('Edited a placed item.');
      });
      return h('label', { class: 'field' }, h('span', { class: 'lbl' }, label), inp);
    };
    const remove = btn({
      icon: 'trash',
      label: 'Delete',
      cls: 'xs',
      onclick: () => this.deleteSelection(),
    });
    if (sel.kind === 'obj') {
      const o = e.objects[sel.i];
      this.inspect.append(
        h('div', { class: 'row' }, h('b', null, o.type), h('div', { class: 'spacer' }), remove),
        h('div', { class: 'grid' }, num('x', o.x, (n) => (o.x = n)), num('z', o.z, (n) => (o.z = n))),
      );
    } else {
      const w = e.walls[sel.i];
      this.inspect.append(
        h('div', { class: 'row' }, h('b', null, w.tag), h('div', { class: 'spacer' }), btn({ icon: 'rotate-cw', cls: 'xs icon', title: 'Rotate 90° (R)', onclick: () => this.rotateSelection(Math.PI / 2) }), btn({ icon: 'rotate-cw', label: '15°', cls: 'xs', title: 'Rotate 15° (Shift+R)', onclick: () => this.rotateSelection(Math.PI / 12) }), remove),
        h('div', { class: 'grid' }, num('x', w.x, (n) => (w.x = n)), num('z', w.z, (n) => (w.z = n)), num('width', w.w, (n) => (w.w = Math.max(0.2, Math.min(60, n)))), num('depth', w.d, (n) => (w.d = Math.max(0.2, Math.min(60, n))))),
      );
      if (w.tag === 'wall' || w.tag === 'ramp') this.inspect.append(h('div', { class: 'grid' }, num('height', w.h, (n) => (w.h = Math.max(0.1, Math.min(20, n))), 0.1), num('angle°', Math.round(((w.yaw * 180) / Math.PI) * 10) / 10, (n) => (w.yaw = (n * Math.PI) / 180), 15)));
    }
  }

  deleteSelection(): void {
    const s = this.sel;
    if (!s) return;
    const e = this.host.edits;
    if (s.kind === 'obj') e.objects.splice(s.i, 1);
    else e.walls.splice(s.i, 1);
    this.select(null);
    this.host.editsChanged('Deleted a placed item.');
  }

  rotateSelection(by: number): void {
    const s = this.sel;
    if (!s || s.kind !== 'wall') return;
    const w = this.host.edits.walls[s.i];
    // keep angles on whole degrees
    const deg = Math.round((((w.yaw + by) * 180) / Math.PI) % 360);
    w.yaw = (deg * Math.PI) / 180;
    this.host.editsChanged('Rotated a placed item.');
  }

  /** Keyboard in build mode: Del, R, Esc. Returns true if handled. */
  handleKey(e: KeyboardEvent): boolean {
    if (!this.buildTool) return false;
    if (e.key === 'Escape') {
      if (this.sel) this.select(null);
      else this.toggleBuild(false);
      return true;
    }
    if (!this.sel) return false;
    if (e.key === 'Delete' || e.key === 'Backspace') {
      this.deleteSelection();
      return true;
    }
    if (e.key.toLowerCase() === 'r') {
      this.rotateSelection(e.shiftKey ? Math.PI / 12 : Math.PI / 2);
      return true;
    }
    return false;
  }

  private maybeStartDrag(e: PointerEvent): void {
    const p = this.view.pick(e.clientX, e.clientY);
    if (!p || !this.sel) return;
    const near = this.nearestEdit(p.x, p.z);
    if (!near || near.kind !== this.sel.kind || near.i !== this.sel.i) return;
    const it = this.sel.kind === 'obj' ? this.host.edits.objects[this.sel.i] : this.host.edits.walls[this.sel.i];
    this.drag = { pointerId: e.pointerId, ox: it.x - p.x, oz: it.z - p.z };
    this.view.controls.enabled = false;
    this.view.renderer.domElement.setPointerCapture(e.pointerId);
  }

  private dragMove(e: PointerEvent): void {
    if (!this.drag || !this.sel) return;
    const p = this.view.pick(e.clientX, e.clientY);
    if (!p) return;
    const it = this.sel.kind === 'obj' ? this.host.edits.objects[this.sel.i] : this.host.edits.walls[this.sel.i];
    it.x = Math.round((p.x + this.drag.ox) * 2) / 2;
    it.z = Math.round((p.z + this.drag.oz) * 2) / 2;
    this.selBox.position.x = it.x;
    this.selBox.position.z = it.z;
  }

  private endDrag(e: PointerEvent): void {
    if (!this.drag) return;
    this.view.renderer.domElement.releasePointerCapture?.(e.pointerId);
    this.drag = null;
    this.view.controls.enabled = true;
    if (this.sel?.kind === 'obj') {
      const o = this.host.edits.objects[this.sel.i];
      o.zone = zoneAt(this.host.world, o.x, o.z);
    }
    this.host.editsChanged('Moved a placed item.');
  }

  setQuality(q: Quality): void {
    this.view.setQuality(q);
  }
}
