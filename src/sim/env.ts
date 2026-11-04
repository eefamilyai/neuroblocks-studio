/**
 * Env: one copy of the world with one robot in it. This is the classic
 * reinforcement-learning loop:
 *
 *     obs = env.reset()
 *     loop:  action = brain(obs)
 *            obs, reward, done = env.step(action)
 *
 * Physics runs at 60 Hz; the brain decides every 3 physics ticks (20 Hz).
 */
import { RNG } from '../ml/rng.ts';
import { pointInYawBox, rayVsCollider, rayVsSphere, rayVsYawBox, sphereVsCollider, wrapAngle } from './geom.ts';
import type { Collider, Contact } from './geom.ts';
import type { AgentSpec, ObjType, Rect, SensorSpec, Tag, ZoneDef } from './types.ts';
import { OBJ_TYPES, STATIC_TAGS } from './types.ts';
import type { WorldRuntime } from './runtime.ts';

export const PHYS = {
  dt: 1 / 60,
  substeps: 3,
  gravity: -22,
  agentR: 0.5,
  agentMass: 1,
  accel: 20,
  strafeAccel: 14,
  groundDrag: 3.2,
  airDrag: 0.25,
  turnRate: 3.2,
  jumpVel: 8.2,
  ballR: 0.55,
  ballMass: 0.45,
  crateH: 0.6,
  crateMass: 2.5,
  npcR: 0.5,
  runnerSpeed: 4.6,
  chaserSpeed: 3.1,
  fallY: -3,
};

const DEG = Math.PI / 180;

/** All tags in a fixed order (index used for compact visualisation data). */
export const ALL_TAGS: Tag[] = [...STATIC_TAGS, ...OBJ_TYPES];
export const TAG_INDEX: Record<string, number> = Object.fromEntries(ALL_TAGS.map((t, i) => [t, i]));

export interface Body {
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
}

export interface ObjState extends Body {
  id: number;
  type: ObjType;
  zone: string;
  yaw: number;
  hx: number;
  hy: number;
  hz: number;
  r: number;
  homeX: number;
  homeY: number;
  homeZ: number;
  homeYaw: number;
  active: boolean;
  index: number;
  custom: boolean;
  grounded: boolean;
  stun: number;
  wander: number;
  flash: number;
  area: Rect;
}

const SOLID: Record<ObjType, boolean> = {
  coin: false,
  flag: false,
  trophy: false,
  ball: true,
  crate: true,
  goal: false,
  pad: false,
  checkpoint: false,
  runner: true,
  chaser: true,
};
const TRANSPARENT: Record<ObjType, boolean> = {
  coin: false,
  flag: false,
  trophy: false,
  ball: false,
  crate: false,
  goal: true,
  pad: true,
  checkpoint: true,
  runner: false,
  chaser: false,
};
const WALKABLE = new Set(['ground', 'ramp', 'rock', 'ice', 'bouncy']);

// ---------------------------------------------------------------------------
// Observation layout: which numbers go into the brain, and what they mean.
// ---------------------------------------------------------------------------

export interface ObsGroup {
  name: string;
  kind: SensorSpec['kind'];
  start: number;
  len: number;
  spec: SensorSpec;
}

export interface ObsLayout {
  size: number;
  labels: string[];
  groups: ObsGroup[];
}

const AXIS_NAME: Record<string, string> = { move: 'move', turn: 'turn', strafe: 'strafe', jump: 'jump', kick: 'kick' };

export function targetLabel(tag: string): string {
  if (tag === 'next_checkpoint') return 'next checkpoint';
  if (tag === 'spawn') return 'start spot';
  return tag;
}

export function obsLayout(spec: AgentSpec): ObsLayout {
  const labels: string[] = [];
  const groups: ObsGroup[] = [];
  const add = (name: string, s: SensorSpec, ls: string[]) => {
    groups.push({ name, kind: s.kind, start: labels.length, len: ls.length, spec: s });
    labels.push(...ls);
  };
  for (const s of spec.sensors) {
    switch (s.kind) {
      case 'rays': {
        const ls: string[] = [];
        for (let i = 0; i < s.count; i++) {
          ls.push(`eye${i + 1} distance`);
          for (const t of s.detect) ls.push(`eye${i + 1} sees ${t}`);
        }
        add(`👁️ Ray eyes ×${s.count}`, s, ls);
        break;
      }
      case 'target': {
        const n = targetLabel(s.tag);
        add(`🧭 ${n} direction`, s, [`${n} side (sin)`, `${n} ahead (cos)`, `${n} distance`]);
        break;
      }
      case 'pair':
        add(`🔀 ${s.a}→${s.b}`, s, [`${s.a}→${s.b} side`, `${s.a}→${s.b} ahead`, `${s.a}→${s.b} distance`]);
        break;
      case 'velocity':
        add('💨 My velocity', s, ['speed forward', 'speed sideways', 'speed up/down']);
        break;
      case 'heading':
        add('🧭 Compass', s, ['compass sin', 'compass cos']);
        break;
      case 'position':
        add('📍 Position', s, ['position x', 'position z']);
        break;
      case 'grounded':
        add('🦶 On ground?', s, ['on ground']);
        break;
      case 'height':
        add('📏 Height', s, ['height']);
        break;
      case 'ground_probe': {
        const ls: string[] = [];
        for (let i = 0; i < s.count; i++) ls.push(`probe${i + 1} hole`, `probe${i + 1} lava`);
        add(`🦯 Ground feelers ×${s.count}`, s, ls);
        break;
      }
      case 'time':
        add('⏱️ Time left', s, ['time left']);
        break;
      case 'last_action':
        add('↩️ Last action', s, spec.axes.map((a) => `last ${AXIS_NAME[a.axis]}`));
        break;
      case 'custom':
        add(`✨ ${s.name}`, s, [s.name]);
        break;
    }
  }
  return { size: labels.length, labels, groups };
}

// ---------------------------------------------------------------------------
// Script handlers (compiled from the event blocks)
// ---------------------------------------------------------------------------

export interface ScriptHandlers {
  start: (() => void)[];
  step: (() => void)[];
  touch: [string, (obj: ObjState | null) => void][];
  pair: [string, string, (a: ObjState, b: ObjState) => void][];
  fall: (() => void)[];
  end: (() => void)[];
  senses: (() => number)[];
}

export type ScriptFactory = (api: ScriptAPI, Math: Math) => ScriptHandlers;

export function compileScript(src: string): ScriptFactory {
  return new Function(`"use strict";\nreturn (${src});`)() as ScriptFactory;
}

const EMPTY_FACTORY: ScriptFactory = () => ({ start: [], step: [], touch: [], pair: [], fall: [], end: [], senses: [] });

// ---------------------------------------------------------------------------

export class Env {
  readonly rt: WorldRuntime;
  readonly spec: AgentSpec;
  readonly rng: RNG;
  readonly zone: ZoneDef;
  readonly freeRoam: boolean;
  readonly layout: ObsLayout;
  readonly obs: Float32Array;
  readonly maxSteps: number;

  /** the robot */
  a: Body & { yaw: number } = { x: 0, y: 0.5, z: 0, vx: 0, vy: 0, vz: 0, yaw: 0 };
  grounded = false;
  groundTag = '';
  private gnx = 0;
  private gny = 1;
  private gnz = 0;
  private jumpCd = 0;
  private kickCd = 0;
  kicked = 0;
  /** current control values */
  move = 0;
  turn = 0;
  strafe = 0;
  jump = 0;
  kick = 0;
  readonly lastActions: Float32Array;

  objs: ObjState[] = [];
  steps = 0;
  episode = 0;
  epReturn = 0;
  stepReward = 0;
  done = false;
  truncated = false;
  success = false;
  fell = false;

  // event bookkeeping ("enter" semantics: fire when contact starts)
  private touchStatic = new Set<string>();
  private prevStatic = new Set<string>();
  private touchObj = new Set<number>();
  private prevObj = new Set<number>();
  private pairNow = new Set<string>();
  private pairPrev = new Set<string>();

  /** per-ray visual data: [dx, dy, dz, t, tagIndex] */
  rayVis: Float32Array;
  rayOrigin: [number, number, number] = [0, 0, 0];
  probeVis: Float32Array;
  /** human-readable events of the last step (for the console / inspector) */
  events: string[] = [];

  private tracked = new Set<string>();
  private prevD = new Map<string, number>();
  private visited: Uint8Array;
  private visitedList: number[] = [];
  private vNx: number;
  private vMinX: number;
  private vMinZ: number;
  newCell = false;
  cellsVisited = 0;
  nextCp = 0;
  laps = 0;
  counters = new Map<string, number>();

  handlers: ScriptHandlers;
  readonly api: ScriptAPI;
  onError: ((msg: string) => void) | null = null;
  onLog: ((msg: string) => void) | null = null;
  private dead = new Set<unknown>();
  private cands: Collider[] = new Array(256);
  private contact: Contact = { nx: 0, ny: 0, nz: 0, depth: 0 };
  private numCheckpoints = 0;
  private skipGate: ((c: Collider) => boolean) | null;
  private detectSets = new Map<SensorSpec, Set<string>>();

  constructor(rt: WorldRuntime, spec: AgentSpec, factory: ScriptFactory | null, seed: number) {
    this.rt = rt;
    this.spec = spec;
    this.rng = new RNG(seed);
    this.zone = rt.zone(spec.zone);
    this.freeRoam = spec.zone === 'world';
    this.skipGate = this.freeRoam ? (c: Collider) => c.gate : null;
    this.layout = obsLayout(spec);
    this.obs = new Float32Array(this.layout.size);
    this.maxSteps = spec.maxSteps;
    this.lastActions = new Float32Array(spec.axes.length);
    const rays = spec.sensors.find((s) => s.kind === 'rays');
    this.rayVis = new Float32Array((rays && rays.kind === 'rays' ? rays.count : 0) * 5);
    const probes = spec.sensors.find((s) => s.kind === 'ground_probe');
    this.probeVis = new Float32Array((probes && probes.kind === 'ground_probe' ? probes.count : 0) * 4);
    const b = rt.def.bounds;
    this.vMinX = b.minX;
    this.vMinZ = b.minZ;
    this.vNx = Math.ceil((b.maxX - b.minX) / 2) + 1;
    this.visited = new Uint8Array(this.vNx * (Math.ceil((b.maxZ - b.minZ) / 2) + 1));

    let id = 0;
    for (const d of rt.objects) {
      if (!this.freeRoam && d.zone !== spec.zone) continue;
      const zn = rt.zone(d.zone);
      const box = d.type === 'crate';
      const r = d.type === 'ball' ? PHYS.ballR : d.type === 'runner' || d.type === 'chaser' ? PHYS.npcR : d.type === 'trophy' ? 0.8 : d.type === 'flag' ? 0.5 : 0.6;
      this.objs.push({
        id: id++,
        type: d.type,
        zone: d.zone,
        x: d.x,
        y: d.y,
        z: d.z,
        vx: 0,
        vy: 0,
        vz: 0,
        yaw: d.yaw ?? 0,
        hx: d.hx ?? (box ? PHYS.crateH : r),
        hy: d.hy ?? (box ? PHYS.crateH : r),
        hz: d.hz ?? (box ? PHYS.crateH : r),
        r,
        homeX: d.x,
        homeY: d.y,
        homeZ: d.z,
        homeYaw: d.yaw ?? 0,
        active: true,
        index: d.index ?? 0,
        custom: !!d.custom,
        grounded: false,
        stun: 0,
        wander: 0,
        flash: 0,
        area: zn.area,
      });
      if (d.type === 'checkpoint') this.numCheckpoints++;
    }
    this.api = new ScriptAPI(this);
    this.handlers = EMPTY_FACTORY(this.api, Math);
    try {
      this.handlers = (factory ?? EMPTY_FACTORY)(this.api, Math);
    } catch (e) {
      this.reportError(`Script failed to start: ${(e as Error).message}`);
    }
  }

  // -------------------------------------------------------------------------
  // Episode lifecycle
  // -------------------------------------------------------------------------

  reset(): Float32Array {
    this.episode++;
    this.steps = 0;
    this.epReturn = 0;
    this.stepReward = 0;
    this.done = false;
    this.truncated = false;
    this.success = false;
    this.fell = false;
    this.nextCp = 0;
    this.laps = 0;
    this.counters.clear();
    this.prevStatic.clear();
    this.prevObj.clear();
    this.pairPrev.clear();
    this.prevD.clear();
    this.lastActions.fill(0);
    this.move = this.turn = this.strafe = this.jump = this.kick = 0;
    for (const i of this.visitedList) this.visited[i] = 0;
    this.visitedList.length = 0;
    this.cellsVisited = 0;
    for (const o of this.objs) this.resetObj(o);
    this.spawnAgent();
    this.markVisited();
    this.newCell = false;
    for (const h of this.handlers.start) this.call(h);
    for (const key of this.tracked) this.prevD.set(key, this.measure(key));
    this.stepReward = 0;
    this.observe();
    return this.obs;
  }

  private spawnAgent(): void {
    const sp = this.spec.spawn;
    const zone = this.zone;
    let px = zone.spawn.x;
    let pz = zone.spawn.z;
    let yaw = zone.spawn.yaw;
    if (sp.mode === 'point') {
      px = sp.x;
      pz = sp.z;
    }
    let placed: [number, number, number] | null = null;
    if (sp.mode === 'random') {
      placed = this.randomPoint(zone.spawnArea ?? zone.area, PHYS.agentR, false);
    } else if (sp.jitter > 0) {
      const r = sp.jitter;
      placed = this.randomPoint({ minX: px - r, maxX: px + r, minZ: pz - r, maxZ: pz + r }, PHYS.agentR, false);
    }
    if (!placed) {
      const gy = this.groundY(px, pz, 20);
      placed = [px, (gy ?? 0) + PHYS.agentR + 0.02, pz];
    }
    if (sp.yawMode === 'random') yaw = this.rng.uniform(-Math.PI, Math.PI);
    else if (sp.yawMode === 'fixed') yaw = sp.yaw;
    const a = this.a;
    a.x = placed[0];
    a.y = placed[1];
    a.z = placed[2];
    a.vx = a.vy = a.vz = 0;
    a.yaw = yaw;
    this.grounded = true;
    this.groundTag = 'ground';
  }

  private resetObj(o: ObjState): void {
    o.x = o.homeX;
    o.y = o.homeY;
    o.z = o.homeZ;
    o.yaw = o.homeYaw;
    o.vx = o.vy = o.vz = 0;
    o.active = true;
    o.stun = 0;
    o.flash = 0;
    o.wander = this.rng.uniform(-Math.PI, Math.PI);
  }

  /** Ground height under (x, z), or null if there is nothing walkable / it's lava. */
  groundY(x: number, z: number, fromY: number): number | null {
    const hit = this.rt.grid.raycast(x, fromY, z, 0, -1, 0, fromY + 50, this.skipGate);
    if (!hit.c) return null;
    if (!WALKABLE.has(hit.c.tag)) return null;
    return fromY - hit.t;
  }

  /** A random collision-free spot on walkable ground inside `area`. */
  randomPoint(area: Rect, r: number, avoidAgent: boolean): [number, number, number] | null {
    const c = this.contact;
    for (let attempt = 0; attempt < 40; attempt++) {
      const x = this.rng.uniform(area.minX, area.maxX);
      const z = this.rng.uniform(area.minZ, area.maxZ);
      const gy = this.groundY(x, z, 16);
      if (gy === null) continue;
      const y = gy + r + 0.02;
      if (avoidAgent && Math.hypot(x - this.a.x, z - this.a.z) < 3) continue;
      let blocked = false;
      const n = this.rt.grid.gather(x - r - 0.3, z - r - 0.3, x + r + 0.3, z + r + 0.3, this.cands);
      for (let i = 0; i < n; i++) {
        const col = this.cands[i];
        if (this.skipGate && col.gate) continue;
        if (sphereVsCollider(col, x, y + 0.05, z, r + 0.25, c) && c.ny < 0.9) {
          blocked = true;
          break;
        }
      }
      if (blocked) continue;
      for (const o of this.objs) {
        if (!o.active || !SOLID[o.type]) continue;
        if (Math.hypot(o.x - x, o.z - z) < r + Math.max(o.r, o.hx) + 0.5) {
          blocked = true;
          break;
        }
      }
      if (!blocked) return [x, y, z];
    }
    return null;
  }

  /** Move an object to a random valid spot in its zone. */
  respawnObj(o: ObjState): void {
    o.active = true;
    o.vx = o.vy = o.vz = 0;
    const r = o.type === 'crate' ? 0.85 : o.type === 'flag' ? 0.8 : Math.max(o.r, 0.5);
    const p = this.randomPoint(o.area, r, true);
    if (!p) {
      this.resetObj(o);
      return;
    }
    const lift = o.type === 'ball' || o.type === 'runner' || o.type === 'chaser' ? o.r : o.type === 'crate' ? PHYS.crateH : 0;
    o.x = p[0];
    o.z = p[2];
    o.y = p[1] - r - 0.02 + lift;
    if (o.type === 'goal' || o.type === 'pad' || o.type === 'checkpoint') o.y = p[1] - r + o.hy;
  }

  // -------------------------------------------------------------------------
  // Stepping
  // -------------------------------------------------------------------------

  private decodeAction(act: ArrayLike<number>, off: number): void {
    const axes = this.spec.axes;
    this.move = this.turn = this.strafe = this.jump = this.kick = 0;
    for (let i = 0; i < axes.length; i++) {
      const ax = axes[i];
      let v: number;
      if (this.spec.actionMode === 'discrete') {
        const idx = Math.max(0, Math.min(ax.options.length - 1, act[off + i] | 0));
        v = ax.options[idx];
      } else {
        v = act[off + i];
        v = v > 1 ? 1 : v < -1 ? -1 : v;
        if (ax.axis === 'jump' || ax.axis === 'kick') v = v > 0 ? 1 : 0;
        else if (ax.axis === 'move' && Math.min(...ax.options) >= 0) v = (v + 1) / 2; // forward-only
      }
      this.lastActions[i] = v;
      switch (ax.axis) {
        case 'move':
          this.move = v;
          break;
        case 'turn':
          this.turn = v;
          break;
        case 'strafe':
          this.strafe = v;
          break;
        case 'jump':
          this.jump = v;
          break;
        case 'kick':
          this.kick = v;
          break;
      }
    }
  }

  /** Apply an action (values per action axis) and advance the world by one decision step. */
  step(act: ArrayLike<number>, off = 0): void {
    this.decodeAction(act, off);
    this.stepReward = 0;
    this.events.length = 0;
    this.touchStatic.clear();
    this.touchObj.clear();
    this.pairNow.clear();
    this.kicked = Math.max(0, this.kicked - 1);
    const dt = PHYS.dt;
    if (this.kick > 0.5 && this.kickCd <= 0) this.doKick();
    for (let s = 0; s < PHYS.substeps; s++) {
      this.agentPhysics(dt);
      this.objectPhysics(dt);
    }
    const T = dt * PHYS.substeps;
    this.jumpCd -= T;
    this.kickCd -= T;
    if (this.a.y < PHYS.fallY) this.fell = true;
    if (!this.freeRoam) {
      const r = this.zone.rect;
      const m = 3;
      if (this.a.x < r.minX - m || this.a.x > r.maxX + m || this.a.z < r.minZ - m || this.a.z > r.maxZ + m) this.fell = true;
    }

    // triggers & contacts with objects
    const a = this.a;
    for (const o of this.objs) {
      if (o.flash > 0) o.flash -= T;
      if (!o.active) continue;
      if (this.agentTouches(o)) this.touchObj.add(o.id);
    }
    const pairs = this.handlers.pair;
    for (let p = 0; p < pairs.length; p++) {
      const [ta, tb] = pairs[p];
      for (const oa of this.objs) {
        if (!oa.active || oa.type !== ta) continue;
        for (const ob of this.objs) {
          if (!ob.active || ob.type !== tb || ob === oa) continue;
          if (this.objOverlap(oa, ob)) this.pairNow.add(`${p}:${oa.id}:${ob.id}`);
        }
      }
    }
    this.steps++;
    this.markVisited();

    // ---- fire events (Scratch-style hats) ----
    for (const tag of this.touchStatic) {
      if (this.prevStatic.has(tag)) continue;
      for (const [t, fn] of this.handlers.touch) if (t === tag) this.call(fn, null);
    }
    for (const id of this.touchObj) {
      if (this.prevObj.has(id)) continue;
      const o = this.objs[id];
      if (o.type === 'chaser') this.bumpChaser(o);
      for (const [t, fn] of this.handlers.touch) if (t === o.type && o.active) this.call(fn, o);
    }
    for (const key of this.pairNow) {
      if (this.pairPrev.has(key)) continue;
      const [p, ia, ib] = key.split(':').map(Number);
      const oa = this.objs[ia];
      const ob = this.objs[ib];
      if (oa.active && ob.active) this.call(pairs[p][2], oa, ob);
    }
    if (this.fell) {
      for (const h of this.handlers.fall) this.call(h);
      if (!this.done) {
        this.done = true;
        this.success = false;
      }
      this.events.push('fell off the world');
    }
    for (const h of this.handlers.step) this.call(h);
    if (!this.done && this.steps >= this.maxSteps) {
      this.done = true;
      this.truncated = true;
    }
    if (this.done) for (const h of this.handlers.end) this.call(h);

    // swap contact sets
    let t1 = this.prevStatic;
    this.prevStatic = this.touchStatic;
    this.touchStatic = t1;
    const t2 = this.prevObj;
    this.prevObj = this.touchObj;
    this.touchObj = t2;
    t1 = this.pairPrev;
    this.pairPrev = this.pairNow;
    this.pairNow = t1;

    for (const key of this.tracked) this.prevD.set(key, this.measure(key));
    if (!Number.isFinite(this.stepReward)) this.stepReward = 0;
    this.epReturn += this.stepReward;
    this.observe();
    void a;
  }

  private call(fn: (...args: never[]) => unknown, ...args: unknown[]): void {
    if (this.dead.has(fn)) return;
    try {
      (fn as (...a: unknown[]) => unknown)(...args);
    } catch (e) {
      this.dead.add(fn);
      this.reportError(`Script error (block disabled): ${(e as Error).message}`);
    }
  }

  private reportError(msg: string): void {
    if (this.onError) this.onError(msg);
  }

  private markVisited(): void {
    const ix = Math.floor((this.a.x - this.vMinX) / 2);
    const iz = Math.floor((this.a.z - this.vMinZ) / 2);
    const k = iz * this.vNx + ix;
    this.newCell = false;
    if (k >= 0 && k < this.visited.length && !this.visited[k]) {
      this.visited[k] = 1;
      this.visitedList.push(k);
      this.cellsVisited++;
      this.newCell = this.steps > 0;
    }
  }

  private doKick(): void {
    const a = this.a;
    const fx = Math.sin(a.yaw);
    const fz = Math.cos(a.yaw);
    let any = false;
    for (const o of this.objs) {
      if (!o.active || (o.type !== 'ball' && o.type !== 'crate')) continue;
      const dx = o.x - a.x;
      const dz = o.z - a.z;
      const d = Math.hypot(dx, dz);
      if (d > 1.9 || d < 1e-6) continue;
      if ((dx * fx + dz * fz) / d < 0.3) continue;
      const power = o.type === 'ball' ? 10 : 3;
      o.vx += fx * power;
      o.vz += fz * power;
      o.vy += o.type === 'ball' ? 2.5 : 0;
      any = true;
    }
    this.kickCd = 0.5;
    if (any) {
      this.kicked = 6;
      this.events.push('kick!');
    }
  }

  private bumpChaser(o: ObjState): void {
    const dx = o.x - this.a.x;
    const dz = o.z - this.a.z;
    const d = Math.hypot(dx, dz) || 1;
    o.vx = (dx / d) * 7;
    o.vz = (dz / d) * 7;
    o.stun = 1.5;
  }

  // -------------------------------------------------------------------------
  // Physics
  // -------------------------------------------------------------------------

  private agentPhysics(dt: number): void {
    const a = this.a;
    const body = this.spec.body;
    const P = PHYS;
    a.yaw = wrapAngle(a.yaw - this.turn * P.turnRate * body.turn * dt);
    const sy = Math.sin(a.yaw);
    const cy = Math.cos(a.yaw);
    const ice = this.grounded && this.groundTag === 'ice';
    const ctrl = this.grounded ? (ice ? 0.3 : 1) : 0.35;
    const acc = P.accel * body.speed * ctrl;
    const sacc = P.strafeAccel * body.speed * ctrl;
    // forward = (sin yaw, cos yaw); right = (−cos yaw, sin yaw)
    a.vx += (sy * this.move * acc - cy * this.strafe * sacc) * dt;
    a.vz += (cy * this.move * acc + sy * this.strafe * sacc) * dt;
    const drag = this.grounded ? (ice ? 0.2 : P.groundDrag) : P.airDrag;
    const k = Math.max(0, 1 - drag * dt);
    a.vx *= k;
    a.vz *= k;
    const g = P.gravity;
    if (this.grounded) {
      // gravity split into "into the ground" + a little "down the slope" (friction holds the rest)
      const gn = g * this.gny;
      const fr = ice ? 1 : 0.15;
      a.vx += (gn * this.gnx + fr * (-gn * this.gnx)) * dt;
      a.vy += (gn * this.gny + fr * (g - gn * this.gny)) * dt;
      a.vz += (gn * this.gnz + fr * (-gn * this.gnz)) * dt;
    } else {
      a.vy += g * dt;
    }
    if (this.jump > 0.5 && this.grounded && this.jumpCd <= 0) {
      a.vy = P.jumpVel * body.jump;
      this.grounded = false;
      this.jumpCd = 0.35;
    }
    a.x += a.vx * dt;
    a.y += a.vy * dt;
    a.z += a.vz * dt;

    // collide with the static world
    this.grounded = false;
    let bestNy = 0.55;
    const r = P.agentR;
    const c = this.contact;
    for (let iter = 0; iter < 2; iter++) {
      const n = this.rt.grid.gather(a.x - r, a.z - r, a.x + r, a.z + r, this.cands);
      for (let i = 0; i < n; i++) {
        const col = this.cands[i];
        if (this.skipGate && col.gate) continue;
        if (!sphereVsCollider(col, a.x, a.y, a.z, r, c)) continue;
        a.x += c.nx * c.depth;
        a.y += c.ny * c.depth;
        a.z += c.nz * c.depth;
        const vn = a.vx * c.nx + a.vy * c.ny + a.vz * c.nz;
        if (vn < 0) {
          a.vx -= vn * c.nx;
          a.vy -= vn * c.ny;
          a.vz -= vn * c.nz;
        }
        if (c.ny > bestNy) {
          bestNy = c.ny;
          this.grounded = true;
          this.groundTag = col.tag;
          this.gnx = c.nx;
          this.gny = c.ny;
          this.gnz = c.nz;
        }
        if (col.tag === 'bouncy' && c.ny > 0.55) {
          a.vy = 12;
          this.grounded = false;
        }
        this.touchStatic.add(col.tag);
      }
    }
    // solid objects
    for (const o of this.objs) {
      if (!o.active || !SOLID[o.type]) continue;
      if (o.type === 'crate') this.sphereVsCrate(a, P.agentR, 1 / P.agentMass, o, true);
      else this.sphereVsSphere(a, P.agentR, 1 / P.agentMass, o, o.r, o.type === 'ball' ? 1 / P.ballMass : 1, 0.35, true);
    }
  }

  private sphereVsSphere(a: Body, ra: number, wa: number, b: ObjState, rb: number, wb: number, e: number, isAgent: boolean): void {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const dz = b.z - a.z;
    const d2 = dx * dx + dy * dy + dz * dz;
    const rs = ra + rb;
    if (d2 >= rs * rs || d2 < 1e-12) return;
    const d = Math.sqrt(d2);
    const nx = dx / d;
    const ny = dy / d;
    const nz = dz / d;
    this.resolve(a, wa, b, wb, nx, ny, nz, rs - d, e, isAgent);
  }

  /** Push bodies apart along n (from a to b) and exchange momentum. */
  private resolve(a: Body, wa: number, b: Body, wb: number, nx: number, ny: number, nz: number, depth: number, e: number, isAgent: boolean): void {
    const wsum = wa + wb;
    a.x -= nx * depth * (wa / wsum);
    a.y -= ny * depth * (wa / wsum);
    a.z -= nz * depth * (wa / wsum);
    b.x += nx * depth * (wb / wsum);
    b.y += ny * depth * (wb / wsum);
    b.z += nz * depth * (wb / wsum);
    const vrel = (b.vx - a.vx) * nx + (b.vy - a.vy) * ny + (b.vz - a.vz) * nz;
    if (vrel < 0) {
      const j = (-(1 + e) * vrel) / wsum;
      a.vx -= j * wa * nx;
      a.vy -= j * wa * ny;
      a.vz -= j * wa * nz;
      b.vx += j * wb * nx;
      b.vy += j * wb * ny;
      b.vz += j * wb * nz;
    }
    if (isAgent && -ny > 0.55) {
      // standing on top of an object
      this.grounded = true;
      this.groundTag = 'ground';
      this.gnx = -nx;
      this.gny = -ny;
      this.gnz = -nz;
    }
  }

  private sphereVsCrate(a: Body, r: number, wa: number, o: ObjState, isAgent: boolean): void {
    const h = o.hx;
    const qx = Math.max(o.x - h, Math.min(a.x, o.x + h));
    const qy = Math.max(o.y - h, Math.min(a.y, o.y + h));
    const qz = Math.max(o.z - h, Math.min(a.z, o.z + h));
    let dx = a.x - qx;
    let dy = a.y - qy;
    let dz = a.z - qz;
    const d2 = dx * dx + dy * dy + dz * dz;
    let depth: number;
    if (d2 > 1e-12) {
      if (d2 >= r * r) return;
      const d = Math.sqrt(d2);
      dx /= d;
      dy /= d;
      dz /= d;
      depth = r - d;
    } else {
      const px = h - Math.abs(a.x - o.x);
      const pz = h - Math.abs(a.z - o.z);
      dy = 0;
      if (px < pz) {
        dx = a.x > o.x ? 1 : -1;
        dz = 0;
        depth = px + r;
      } else {
        dz = a.z > o.z ? 1 : -1;
        dx = 0;
        depth = pz + r;
      }
    }
    // n points from crate to sphere; resolve() wants a→b, so pass -n
    this.resolve(a, wa, o, 1 / PHYS.crateMass, -dx, -dy, -dz, depth, 0.05, isAgent);
  }

  private objectPhysics(dt: number): void {
    const g = PHYS.gravity;
    const c = this.contact;
    const ax = this.a.x;
    const az = this.a.z;
    for (const o of this.objs) {
      if (!o.active) continue;
      const t = o.type;
      if (t !== 'ball' && t !== 'crate' && t !== 'runner' && t !== 'chaser') {
        continue;
      }
      if (t === 'runner' || t === 'chaser') this.npcBrain(o, ax, az, dt);
      o.vy += g * dt;
      const drag = o.grounded ? (t === 'ball' ? 0.5 : t === 'crate' ? 7 : 0) : 0.05;
      const k = Math.max(0, 1 - drag * dt);
      o.vx *= k;
      o.vz *= k;
      o.x += o.vx * dt;
      o.y += o.vy * dt;
      o.z += o.vz * dt;
      o.grounded = false;
      if (t === 'crate') {
        this.crateVsStatic(o);
      } else {
        const r = o.r;
        const n = this.rt.grid.gather(o.x - r, o.z - r, o.x + r, o.z + r, this.cands);
        for (let i = 0; i < n; i++) {
          const col = this.cands[i];
          if (this.skipGate && col.gate) continue;
          if (!sphereVsCollider(col, o.x, o.y, o.z, r, c)) continue;
          o.x += c.nx * c.depth;
          o.y += c.ny * c.depth;
          o.z += c.nz * c.depth;
          const vn = o.vx * c.nx + o.vy * c.ny + o.vz * c.nz;
          if (vn < 0) {
            const bounce = t === 'ball' ? (Math.abs(vn) > 1.5 ? 0.55 : 0) : 0;
            o.vx -= (1 + bounce) * vn * c.nx;
            o.vy -= (1 + bounce) * vn * c.ny;
            o.vz -= (1 + bounce) * vn * c.nz;
          }
          if (c.ny > 0.55) o.grounded = true;
        }
      }
      if (o.y < -20) this.resetObj(o);
    }
    // object vs object
    const objs = this.objs;
    for (let i = 0; i < objs.length; i++) {
      const p = objs[i];
      if (!p.active || !SOLID[p.type]) continue;
      for (let j = i + 1; j < objs.length; j++) {
        const q = objs[j];
        if (!q.active || !SOLID[q.type]) continue;
        if (p.type === 'crate' && q.type === 'crate') this.crateVsCrate(p, q);
        else if (p.type === 'crate') this.sphereVsCrate(q, q.r, this.massInv(q), p, false);
        else if (q.type === 'crate') this.sphereVsCrate(p, p.r, this.massInv(p), q, false);
        else this.sphereVsSphere(p, p.r, this.massInv(p), q, q.r, this.massInv(q), 0.5, false);
      }
    }
  }

  private massInv(o: ObjState): number {
    return o.type === 'ball' ? 1 / PHYS.ballMass : o.type === 'crate' ? 1 / PHYS.crateMass : 1;
  }

  private crateVsStatic(o: ObjState): void {
    const h = o.hx;
    const c = this.contact;
    const n = this.rt.grid.gather(o.x - h, o.z - h, o.x + h, o.z + h, this.cands);
    for (let i = 0; i < n; i++) {
      const col = this.cands[i];
      if (this.skipGate && col.gate) continue;
      if (col.kind === 'box' && col.axisAligned) {
        const ox = Math.min(o.x + h, col.maxX) - Math.max(o.x - h, col.minX);
        if (ox <= 0) continue;
        const oy = Math.min(o.y + h, col.maxY) - Math.max(o.y - h, col.minY);
        if (oy <= 0) continue;
        const oz = Math.min(o.z + h, col.maxZ) - Math.max(o.z - h, col.minZ);
        if (oz <= 0) continue;
        if (oy <= ox && oy <= oz) {
          const up = o.y > col.cy;
          o.y += up ? oy : -oy;
          if (up) {
            o.grounded = true;
            if (o.vy < 0) o.vy = 0;
          } else if (o.vy > 0) o.vy = 0;
        } else if (ox <= oz) {
          const s = o.x > col.cx ? 1 : -1;
          o.x += s * ox;
          if (o.vx * s < 0) o.vx = 0;
        } else {
          const s = o.z > col.cz ? 1 : -1;
          o.z += s * oz;
          if (o.vz * s < 0) o.vz = 0;
        }
      } else if (sphereVsCollider(col, o.x, o.y, o.z, h, c)) {
        o.x += c.nx * c.depth;
        o.y += c.ny * c.depth;
        o.z += c.nz * c.depth;
        const vn = o.vx * c.nx + o.vy * c.ny + o.vz * c.nz;
        if (vn < 0) {
          o.vx -= vn * c.nx;
          o.vy -= vn * c.ny;
          o.vz -= vn * c.nz;
        }
        if (c.ny > 0.55) o.grounded = true;
      }
    }
  }

  private crateVsCrate(p: ObjState, q: ObjState): void {
    const h = p.hx + q.hx;
    const ox = h - Math.abs(p.x - q.x);
    const oy = h - Math.abs(p.y - q.y);
    const oz = h - Math.abs(p.z - q.z);
    if (ox <= 0 || oy <= 0 || oz <= 0) return;
    if (ox < oz) {
      const s = p.x > q.x ? 1 : -1;
      p.x += (s * ox) / 2;
      q.x -= (s * ox) / 2;
      const v = (p.vx + q.vx) / 2;
      p.vx = q.vx = v;
    } else {
      const s = p.z > q.z ? 1 : -1;
      p.z += (s * oz) / 2;
      q.z -= (s * oz) / 2;
      const v = (p.vz + q.vz) / 2;
      p.vz = q.vz = v;
    }
  }

  /** Simple scripted bots: the runner flees, the chaser hunts. */
  private npcBrain(o: ObjState, ax: number, az: number, dt: number): void {
    const dx = o.x - ax;
    const dz = o.z - az;
    const d = Math.hypot(dx, dz) || 1e-6;
    let wx = 0;
    let wz = 0;
    let speed: number;
    o.wander += this.rng.uniform(-0.25, 0.25);
    if (o.type === 'runner') {
      speed = PHYS.runnerSpeed;
      if (d < 10) {
        const f = ((10 - d) / 10) * 3;
        wx += (dx / d) * f;
        wz += (dz / d) * f;
      } else speed *= 0.45;
      wx += Math.sin(o.wander) * 0.7;
      wz += Math.cos(o.wander) * 0.7;
    } else {
      speed = PHYS.chaserSpeed;
      if (o.stun > 0) {
        o.stun -= dt;
        return;
      }
      wx -= (dx / d) * 2;
      wz -= (dz / d) * 2;
      wx += Math.sin(o.wander) * 0.4;
      wz += Math.cos(o.wander) * 0.4;
    }
    // stay away from the zone edges
    const r = this.rt.zone(o.zone).rect;
    const m = 4.5;
    if (o.x - r.minX < m) wx += (m - (o.x - r.minX)) * 0.8;
    if (r.maxX - o.x < m) wx -= (m - (r.maxX - o.x)) * 0.8;
    if (o.z - r.minZ < m) wz += (m - (o.z - r.minZ)) * 0.8;
    if (r.maxZ - o.z < m) wz -= (m - (r.maxZ - o.z)) * 0.8;
    const wl = Math.hypot(wx, wz) || 1;
    const tx = (wx / wl) * speed;
    const tz = (wz / wl) * speed;
    const blend = Math.min(1, 8 * dt);
    o.vx += (tx - o.vx) * blend;
    o.vz += (tz - o.vz) * blend;
    o.yaw = Math.atan2(o.vx, o.vz);
  }

  // -------------------------------------------------------------------------
  // Contacts & queries
  // -------------------------------------------------------------------------

  agentTouches(o: ObjState): boolean {
    const a = this.a;
    const R = PHYS.agentR;
    switch (o.type) {
      case 'coin': {
        const dy = a.y - (o.y + 0.8);
        return Math.hypot(a.x - o.x, a.z - o.z) < o.r + R && Math.abs(dy) < 1.3;
      }
      case 'flag':
        return Math.hypot(a.x - o.x, a.z - o.z) < o.r + R + 0.2 && a.y > o.y - 0.6 && a.y < o.y + 3.2;
      case 'trophy':
        return Math.hypot(a.x - o.x, a.y - (o.y + 0.9), a.z - o.z) < o.r + R + 0.1;
      case 'goal':
      case 'pad':
      case 'checkpoint':
        return pointInYawBox(a.x, a.y, a.z, o.x, o.y, o.z, o.hx, o.hy, o.hz, o.yaw, 0.35);
      case 'crate': {
        const h = o.hx + R + 0.06;
        return Math.abs(a.x - o.x) < h && Math.abs(a.z - o.z) < h && Math.abs(a.y - o.y) < h;
      }
      default:
        return Math.hypot(a.x - o.x, a.y - o.y, a.z - o.z) < o.r + R + 0.06;
    }
  }

  private objOverlap(p: ObjState, q: ObjState): boolean {
    const isBox = (o: ObjState) => o.type === 'goal' || o.type === 'pad' || o.type === 'checkpoint';
    if (isBox(q)) return pointInYawBox(p.x, p.y, p.z, q.x, q.y, q.z, q.hx, q.hy, q.hz, q.yaw, p.type === 'crate' ? 0.35 : 0);
    if (isBox(p)) return pointInYawBox(q.x, q.y, q.z, p.x, p.y, p.z, p.hx, p.hy, p.hz, p.yaw, 0);
    const rp = p.type === 'crate' ? p.hx : p.r;
    const rq = q.type === 'crate' ? q.hx : q.r;
    const py = p.type === 'coin' || p.type === 'flag' ? p.y + 0.8 : p.y;
    const qy = q.type === 'coin' || q.type === 'flag' ? q.y + 0.8 : q.y;
    return Math.hypot(p.x - q.x, py - qy, p.z - q.z) < rp + rq + 0.05;
  }

  /** Nearest thing with a tag: returns [x, y, z, distance, objIndex] or null. */
  nearest(tag: string, fromX = this.a.x, fromZ = this.a.z): [number, number, number, number, number] | null {
    if (tag === 'spawn') {
      const sp = this.spec.spawn.mode === 'point' ? this.spec.spawn : this.zone.spawn;
      return [sp.x, 0, sp.z, Math.hypot(sp.x - fromX, sp.z - fromZ), -1];
    }
    if (tag === 'next_checkpoint') {
      for (const o of this.objs)
        if (o.active && o.type === 'checkpoint' && o.index === this.nextCp) return [o.x, o.y, o.z, Math.hypot(o.x - fromX, o.z - fromZ), o.id];
      return null;
    }
    let best: [number, number, number, number, number] | null = null;
    if ((OBJ_TYPES as string[]).includes(tag)) {
      for (const o of this.objs) {
        if (!o.active || o.type !== tag) continue;
        const d = Math.hypot(o.x - fromX, o.z - fromZ);
        if (!best || d < best[3]) best = [o.x, o.y, o.z, d, o.id];
      }
      return best;
    }
    // static tags: closest point on the nearest collider
    const cols = this.rt.collidersWithTag(this.zone.id, tag);
    const y = this.a.y;
    for (const c of cols) {
      if (this.skipGate && c.gate) continue;
      let qx: number;
      let qz: number;
      if (c.kind === 'cyl') {
        const dx = fromX - c.cx;
        const dz = fromZ - c.cz;
        const d = Math.hypot(dx, dz) || 1e-6;
        const k = Math.min(1, c.r / d);
        qx = c.cx + dx * k;
        qz = c.cz + dz * k;
      } else {
        const R = c.R;
        const dx = fromX - c.cx;
        const dy = y - c.cy;
        const dz = fromZ - c.cz;
        let lx = R[0] * dx + R[3] * dy + R[6] * dz;
        let ly = R[1] * dx + R[4] * dy + R[7] * dz;
        let lz = R[2] * dx + R[5] * dy + R[8] * dz;
        lx = Math.max(-c.hx, Math.min(c.hx, lx));
        ly = Math.max(-c.hy, Math.min(c.hy, ly));
        lz = Math.max(-c.hz, Math.min(c.hz, lz));
        qx = c.cx + R[0] * lx + R[1] * ly + R[2] * lz;
        qz = c.cz + R[6] * lx + R[7] * ly + R[8] * lz;
      }
      const d = Math.hypot(qx - fromX, qz - fromZ);
      if (!best || d < best[3]) best = [qx, y, qz, d, -1];
    }
    return best;
  }

  /** Measure a tracked quantity (for "change in ..." blocks). */
  measure(key: string): number {
    if (key === '#height') return this.a.y;
    if (key.startsWith('pair:')) {
      const [, ta, tb] = key.split(':');
      return this.pairDistance(ta, tb);
    }
    const n = this.nearest(key);
    return n ? n[3] : -1;
  }

  pairDistance(ta: string, tb: string): number {
    const a = this.nearest(ta);
    if (!a) return -1;
    const b = this.nearest(tb, a[0], a[2]);
    return b ? b[3] : -1;
  }

  track(key: string): void {
    if (this.tracked.has(key)) return;
    this.tracked.add(key);
    this.prevD.set(key, this.measure(key));
  }

  delta(key: string): number {
    this.track(key);
    const now = this.measure(key);
    const prev = this.prevD.get(key);
    if (prev === undefined || prev < 0 || now < 0) return 0;
    const d = now - prev;
    return d > 1 ? 1 : d < -1 ? -1 : d;
  }

  /** Relative angle (radians, + = to the right) from the robot to a point. */
  relAngle(x: number, z: number): number {
    const a = this.a;
    const vx = x - a.x;
    const vz = z - a.z;
    const sy = Math.sin(a.yaw);
    const cy = Math.cos(a.yaw);
    const fwd = vx * sy + vz * cy;
    const right = -vx * cy + vz * sy;
    return Math.atan2(right, fwd);
  }

  // -------------------------------------------------------------------------
  // Sensors -> observation vector
  // -------------------------------------------------------------------------

  /** Cast a ray; returns [t, tagIndex] (t = -1 if nothing within range). */
  castRay(ox: number, oy: number, oz: number, dx: number, dy: number, dz: number, maxT: number, detect: Set<string> | null): [number, number] {
    const hit = this.rt.grid.raycast(ox, oy, oz, dx, dy, dz, maxT, this.skipGate);
    let bestT = hit.c ? hit.t : maxT + 1;
    let bestTag = hit.c ? TAG_INDEX[hit.c.tag] : -1;
    for (const o of this.objs) {
      if (!o.active) continue;
      if (TRANSPARENT[o.type] && !(detect && detect.has(o.type))) continue;
      let t = -1;
      switch (o.type) {
        case 'coin':
          t = rayVsSphere(ox, oy, oz, dx, dy, dz, o.x, o.y + 0.8, o.z, 0.55, bestT);
          break;
        case 'trophy':
          t = rayVsSphere(ox, oy, oz, dx, dy, dz, o.x, o.y + 0.9, o.z, 0.8, bestT);
          break;
        case 'flag':
          t = rayVsYawBox(ox, oy, oz, dx, dy, dz, o.x, o.y + 1.6, o.z, 0.35, 1.6, 0.35, 0, bestT);
          break;
        case 'crate':
          t = rayVsYawBox(ox, oy, oz, dx, dy, dz, o.x, o.y, o.z, o.hx, o.hy, o.hz, 0, bestT);
          break;
        case 'goal':
        case 'pad':
        case 'checkpoint':
          t = rayVsYawBox(ox, oy, oz, dx, dy, dz, o.x, o.y, o.z, o.hx, o.hy, o.hz, o.yaw, bestT);
          break;
        default:
          t = rayVsSphere(ox, oy, oz, dx, dy, dz, o.x, o.y, o.z, o.r, bestT);
      }
      if (t >= 0 && t < bestT) {
        bestT = t;
        bestTag = TAG_INDEX[o.type];
      }
    }
    return bestT <= maxT ? [bestT, bestTag] : [-1, -1];
  }

  observe(): void {
    const o = this.obs;
    const a = this.a;
    let k = 0;
    for (const g of this.layout.groups) {
      const s = g.spec;
      switch (s.kind) {
        case 'rays': {
          const n = s.count;
          const tilt = s.tilt * DEG;
          const ct = Math.cos(tilt);
          const st = Math.sin(tilt);
          const ox = a.x;
          const oy = a.y + 0.15;
          const oz = a.z;
          this.rayOrigin[0] = ox;
          this.rayOrigin[1] = oy;
          this.rayOrigin[2] = oz;
          let det = this.detectSets.get(s);
          if (!det) this.detectSets.set(s, (det = new Set<string>(s.detect)));
          for (let i = 0; i < n; i++) {
            const ang = n === 1 ? 0 : (-s.spread / 2 + (s.spread * i) / (n - 1)) * DEG;
            const yr = a.yaw - ang;
            const dx = ct * Math.sin(yr);
            const dy = -st;
            const dz = ct * Math.cos(yr);
            const [t, tagI] = this.castRay(ox, oy, oz, dx, dy, dz, s.range, det);
            o[k++] = t < 0 ? 1 : t / s.range;
            const hitTag = tagI >= 0 ? ALL_TAGS[tagI] : '';
            for (const tg of s.detect) o[k++] = t >= 0 && hitTag === tg ? 1 : 0;
            const v = i * 5;
            if (v + 4 < this.rayVis.length) {
              this.rayVis[v] = dx;
              this.rayVis[v + 1] = dy;
              this.rayVis[v + 2] = dz;
              this.rayVis[v + 3] = t < 0 ? s.range : t;
              this.rayVis[v + 4] = tagI;
            }
          }
          break;
        }
        case 'target': {
          const t = this.nearest(s.tag);
          if (t) {
            const ang = this.relAngle(t[0], t[2]);
            o[k++] = Math.sin(ang);
            o[k++] = Math.cos(ang);
            o[k++] = Math.min(1, t[3] / s.range);
          } else {
            o[k++] = 0;
            o[k++] = 0;
            o[k++] = 1;
          }
          break;
        }
        case 'pair': {
          const pa = this.nearest(s.a);
          const pb = pa ? this.nearest(s.b, pa[0], pa[2]) : null;
          if (pa && pb) {
            const vx = pb[0] - pa[0];
            const vz = pb[2] - pa[2];
            const d = Math.hypot(vx, vz) || 1e-6;
            const sy = Math.sin(a.yaw);
            const cy = Math.cos(a.yaw);
            o[k++] = (-vx * cy + vz * sy) / d;
            o[k++] = (vx * sy + vz * cy) / d;
            o[k++] = Math.min(1, d / s.range);
          } else {
            o[k++] = 0;
            o[k++] = 0;
            o[k++] = 1;
          }
          break;
        }
        case 'velocity': {
          const sy = Math.sin(a.yaw);
          const cy = Math.cos(a.yaw);
          o[k++] = clamp((a.vx * sy + a.vz * cy) / 8, -1.5, 1.5);
          o[k++] = clamp((-a.vx * cy + a.vz * sy) / 8, -1.5, 1.5);
          o[k++] = clamp(a.vy / 10, -1.5, 1.5);
          break;
        }
        case 'heading':
          o[k++] = Math.sin(a.yaw);
          o[k++] = Math.cos(a.yaw);
          break;
        case 'position': {
          const r = this.zone.rect;
          o[k++] = clamp(((a.x - r.minX) / (r.maxX - r.minX)) * 2 - 1, -1.5, 1.5);
          o[k++] = clamp(((a.z - r.minZ) / (r.maxZ - r.minZ)) * 2 - 1, -1.5, 1.5);
          break;
        }
        case 'grounded':
          o[k++] = this.grounded ? 1 : 0;
          break;
        case 'height':
          o[k++] = clamp((a.y - PHYS.agentR) / 6, -1.5, 1.5);
          break;
        case 'ground_probe': {
          const n = s.count;
          for (let i = 0; i < n; i++) {
            const ang = n === 1 ? 0 : (-45 + (90 * i) / (n - 1)) * DEG;
            const yr = a.yaw - ang;
            const px = a.x + Math.sin(yr) * s.dist;
            const pz = a.z + Math.cos(yr) * s.dist;
            const hit = this.rt.grid.raycast(px, a.y + 0.5, pz, 0, -1, 0, 3, this.skipGate);
            const hole = hit.c ? 0 : 1;
            const lava = hit.c && hit.c.tag === 'lava' ? 1 : 0;
            o[k++] = hole;
            o[k++] = lava;
            const v = i * 4;
            if (v + 3 < this.probeVis.length) {
              this.probeVis[v] = px;
              this.probeVis[v + 1] = hit.c ? a.y + 0.5 - hit.t : a.y - 2.5;
              this.probeVis[v + 2] = pz;
              this.probeVis[v + 3] = hole ? 1 : lava ? 2 : 0;
            }
          }
          break;
        }
        case 'time':
          o[k++] = 1 - this.steps / this.maxSteps;
          break;
        case 'last_action':
          for (let i = 0; i < this.lastActions.length; i++) o[k++] = this.lastActions[i];
          break;
        case 'custom': {
          const fn = this.handlers.senses[s.index];
          let v = 0;
          if (fn && !this.dead.has(fn)) {
            try {
              v = Number(fn());
            } catch (e) {
              this.dead.add(fn);
              this.reportError(`Custom sense "${s.name}" failed: ${(e as Error).message}`);
            }
          }
          o[k++] = Number.isFinite(v) ? clamp(v, -10, 10) : 0;
          break;
        }
      }
    }
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

// ---------------------------------------------------------------------------
// The API that block programs call. Every block in the "Rewards & Events"
// palette turns into one of these method calls in the generated JavaScript.
// ---------------------------------------------------------------------------

export class ScriptAPI {
  private e: Env;
  constructor(e: Env) {
    this.e = e;
  }
  reward(x: number): void {
    const v = Number(x);
    if (Number.isFinite(v)) this.e.stepReward += v;
  }
  end(success: boolean): void {
    if (this.e.done) return;
    this.e.done = true;
    this.e.success = !!success;
    this.e.events.push(success ? 'episode ended: success ✅' : 'episode ended ❌');
  }
  dist(tag: string): number {
    const n = this.e.nearest(tag);
    return n ? n[3] : 999;
  }
  delta(tag: string): number {
    return this.e.delta(tag);
  }
  angle(tag: string): number {
    const n = this.e.nearest(tag);
    return n ? (this.e.relAngle(n[0], n[2]) * 180) / Math.PI : 0;
  }
  facing(tag: string, deg: number): boolean {
    const n = this.e.nearest(tag);
    return !!n && Math.abs((this.e.relAngle(n[0], n[2]) * 180) / Math.PI) <= deg;
  }
  pairDist(a: string, b: string): number {
    const d = this.e.pairDistance(a, b);
    return d < 0 ? 999 : d;
  }
  pairDelta(a: string, b: string): number {
    return this.e.delta(`pair:${a}:${b}`);
  }
  speed(): number {
    const a = this.e.a;
    return Math.hypot(a.vx, a.vz);
  }
  fwdSpeed(): number {
    const a = this.e.a;
    return a.vx * Math.sin(a.yaw) + a.vz * Math.cos(a.yaw);
  }
  height(): number {
    return this.e.a.y - PHYS.agentR;
  }
  dHeight(): number {
    return this.e.delta('#height');
  }
  grounded(): boolean {
    return this.e.grounded;
  }
  touching(tag: string): boolean {
    if ((STATIC_TAGS as string[]).includes(tag)) return this.e.groundTag === tag && this.e.grounded ? true : this.touchingStatic(tag);
    for (const o of this.e.objs) if (o.active && o.type === tag && this.e.agentTouches(o)) return true;
    return false;
  }
  private touchingStatic(tag: string): boolean {
    // re-test contacts at the current position (cheap: only nearby colliders)
    const a = this.e.a;
    const c = { nx: 0, ny: 0, nz: 0, depth: 0 };
    const cols = this.e.rt.collidersWithTag(this.e.zone.id, tag);
    for (const col of cols) if (sphereVsCollider(col, a.x, a.y, a.z, PHYS.agentR + 0.05, c)) return true;
    return false;
  }
  steps(): number {
    return this.e.steps;
  }
  maxSteps(): number {
    return this.e.maxSteps;
  }
  timeLeft(): number {
    return 1 - this.e.steps / this.e.maxSteps;
  }
  respawn(o: ObjState | null): void {
    if (o) {
      this.e.respawnObj(o);
      o.flash = 0.4;
    }
  }
  resetObj(o: ObjState | null): void {
    if (!o) return;
    o.x = o.homeX;
    o.y = o.homeY;
    o.z = o.homeZ;
    o.vx = o.vy = o.vz = 0;
    o.active = true;
  }
  hide(o: ObjState | null): void {
    if (o) o.active = false;
  }
  randomize(tag: string): void {
    for (const o of this.e.objs) if (o.type === tag) this.e.respawnObj(o);
  }
  resetAll(tag: string): void {
    for (const o of this.e.objs) if (o.type === tag) this.resetObj(o);
  }
  showAll(tag: string): void {
    for (const o of this.e.objs) if (o.type === tag) o.active = true;
  }
  count(tag: string): number {
    let n = 0;
    for (const o of this.e.objs) if (o.active && o.type === tag) n++;
    return n;
  }
  newArea(): boolean {
    return this.e.newCell;
  }
  areasVisited(): number {
    return this.e.cellsVisited;
  }
  isNext(o: ObjState | null): boolean {
    return !!o && o.type === 'checkpoint' && o.index === this.e.nextCp;
  }
  advance(): void {
    const total = this.e.objs.filter((o) => o.type === 'checkpoint').length;
    if (!total || this.e.nextCp >= total) return;
    this.e.nextCp++;
    if (this.e.nextCp >= total) {
      if (this.e.zone.loopCheckpoints || this.e.freeRoam) {
        // a circuit: start the next lap
        this.e.nextCp = 0;
        this.e.laps++;
        this.e.events.push(`lap ${this.e.laps}! 🏁`);
      } else {
        // a route (maze, summit): all done, there is no "next checkpoint" any more
        this.e.events.push('all checkpoints passed! 🏁');
      }
    }
  }
  checkpoint(): number {
    return this.e.nextCp;
  }
  laps(): number {
    return this.e.laps;
  }
  addCounter(name: string, n: number): void {
    this.e.counters.set(name, (this.e.counters.get(name) ?? 0) + Number(n || 0));
  }
  counter(name: string): number {
    return this.e.counters.get(name) ?? 0;
  }
  episodeReward(): number {
    return this.e.epReturn + this.e.stepReward;
  }
  random(a: number, b: number): number {
    return this.e.rng.uniform(Math.min(a, b), Math.max(a, b));
  }
  lastAction(axis: string): number {
    const i = this.e.spec.axes.findIndex((x) => x.axis === axis);
    return i >= 0 ? this.e.lastActions[i] : 0;
  }
  respawnMe(): void {
    // teleport the robot back to its start
    const a = this.e.a;
    const sp = this.e.zone.spawn;
    a.x = sp.x;
    a.z = sp.z;
    a.y = 1.2;
    a.vx = a.vy = a.vz = 0;
  }
  log(msg: unknown): void {
    if (this.e.onLog) this.e.onLog(String(msg));
  }
  note(msg: string): void {
    this.e.events.push(String(msg));
  }
}
