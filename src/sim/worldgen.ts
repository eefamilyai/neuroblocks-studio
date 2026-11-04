/**
 * The prebuilt world: an island with 12 fenced training zones.
 *
 *            x = -84         -28           28           84
 *  z = -56   🌀 Maze        🌋 Lava Lake  ⛰️ Summit     🏎️ Race Track
 *  z =   0   🪙 Coin Field  🌼 Meadow     ⚽ Soccer     ☁️ Sky Parkour
 *  z =  56   🌲 Forest      📦 Warehouse  🐰 Tag Arena  👾 Dodge Arena
 *
 * Everything is deterministic (seeded), so every training run sees the same world.
 */
import { RNG } from '../ml/rng.ts';
import { rotYawPitch } from './geom.ts';
import type { DecorDef, Material, ObjDef, ObjType, Rect, StaticDef, StaticTag, WorldDef, ZoneDef } from './types.ts';

const PLOT = 22; // half-size of every zone plot
const COLS = [-84, -28, 28, 84];
const ROWS = [-56, 0, 56];
export const FENCE_H = 1.6;

const C = {
  grass: 0x7cc957,
  grass2: 0x8fd16a,
  grassDark: 0x5f9e3f,
  forest: 0x5a8a3c,
  dirt: 0xa08060,
  stone: 0x9a9aa6,
  stoneDark: 0x6f6f7d,
  wood: 0xb07a46,
  woodDark: 0x8a5a32,
  hedge: 0x2f7a3a,
  concrete: 0xa6a6ad,
  brick: 0xb0563c,
  track: 0x4a4a55,
  sand: 0xe0bc78,
  rubber: 0x3d5a99,
  lava: 0xff5a1f,
  white: 0xf2f2f2,
  metal: 0x8fa3b8,
  pitch: 0x4caf50,
  sky: 0xbfd9ff,
  rock: 0x8d8d99,
};

class Builder {
  statics: StaticDef[] = [];
  objects: ObjDef[] = [];
  zones: ZoneDef[] = [];
  decor: DecorDef[] = [];

  box(
    cx: number,
    cy: number,
    cz: number,
    hx: number,
    hy: number,
    hz: number,
    tag: StaticTag,
    mat: Material,
    color: number,
    zone: string,
    opts: { yaw?: number; pitch?: number; gate?: boolean; invisible?: boolean } = {},
  ): void {
    const yaw = opts.yaw ?? 0;
    const pitch = opts.pitch ?? 0;
    this.statics.push({
      kind: 'box',
      c: [cx, cy, cz],
      h: [hx, hy, hz],
      R: rotYawPitch(yaw, pitch),
      yaw,
      pitch,
      tag,
      mat,
      color,
      zone,
      gate: opts.gate,
      invisible: opts.invisible,
    });
  }

  /** Axis-aligned box from min/max corners. */
  aabb(x0: number, y0: number, z0: number, x1: number, y1: number, z1: number, tag: StaticTag, mat: Material, color: number, zone: string, gate = false): void {
    this.box((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2, Math.abs(x1 - x0) / 2, Math.abs(y1 - y0) / 2, Math.abs(z1 - z0) / 2, tag, mat, color, zone, { gate });
  }

  /** A wall segment between two points on the ground. */
  wall(x0: number, z0: number, x1: number, z1: number, height: number, thick: number, tag: StaticTag, mat: Material, color: number, zone: string, base = 0, gate = false): void {
    const dx = x1 - x0;
    const dz = z1 - z0;
    const len = Math.hypot(dx, dz);
    if (len < 1e-6) return;
    // axis-aligned walls stay unrotated (exact & fast collisions for crates)
    if (Math.abs(dz) < 1e-9) {
      this.box((x0 + x1) / 2, base + height / 2, z0, len / 2, height / 2, thick / 2, tag, mat, color, zone, { gate });
      return;
    }
    if (Math.abs(dx) < 1e-9) {
      this.box(x0, base + height / 2, (z0 + z1) / 2, thick / 2, height / 2, len / 2, tag, mat, color, zone, { gate });
      return;
    }
    const yaw = Math.atan2(dx, dz); // local +Z runs along the wall
    this.box((x0 + x1) / 2, base + height / 2, (z0 + z1) / 2, thick / 2, height / 2, len / 2, tag, mat, color, zone, { yaw, gate });
  }

  cyl(x: number, z: number, y0: number, y1: number, r: number, tag: StaticTag, mat: Material, color: number, zone: string): void {
    this.statics.push({ kind: 'cyl', c: [x, z], y0, y1, r, tag, mat, color, zone });
  }

  /** A ramp whose top surface goes from A (height ay) to B (height by). */
  ramp(ax: number, az: number, ay: number, bx: number, bz: number, by: number, width: number, zone: string, color = C.wood, thick = 0.4): void {
    const dx = bx - ax;
    const dz = bz - az;
    const run = Math.hypot(dx, dz);
    const rise = by - ay;
    const len = Math.hypot(run, rise);
    const yaw = Math.atan2(dx, dz);
    const pitch = -Math.atan2(rise, run);
    const R = rotYawPitch(yaw, pitch);
    // centre = midpoint of the top surface minus half the thickness along local up
    const mx = (ax + bx) / 2 - R[1] * (thick / 2);
    const my = (ay + by) / 2 - R[4] * (thick / 2);
    const mz = (az + bz) / 2 - R[7] * (thick / 2);
    this.box(mx, my, mz, width / 2, thick / 2, len / 2, 'ramp', 'wood', color, zone, { yaw, pitch });
  }

  obj(type: ObjType, x: number, y: number, z: number, zone: string, extra: Partial<ObjDef> = {}): void {
    this.objects.push({ type, x, y, z, zone, ...extra });
  }

  /**
   * Four walls around a rect, with an opening closed by a gate collider
   * (gates disappear in free-roam so the zones connect).
   */
  fence(r: Rect, zone: string, gate: { side: 'n' | 's' | 'e' | 'w'; at: number; width: number } | null, opts: { height?: number; thick?: number; mat?: Material; color?: number; base?: number } = {}): void {
    const h = opts.height ?? FENCE_H;
    const t = opts.thick ?? 0.4;
    const mat = opts.mat ?? 'wood';
    const color = opts.color ?? C.wood;
    const base = opts.base ?? 0;
    const sides: [string, number, number, number, number][] = [
      ['n', r.minX, r.minZ, r.maxX, r.minZ],
      ['s', r.minX, r.maxZ, r.maxX, r.maxZ],
      ['w', r.minX, r.minZ, r.minX, r.maxZ],
      ['e', r.maxX, r.minZ, r.maxX, r.maxZ],
    ];
    for (const [side, x0, z0, x1, z1] of sides) {
      if (gate && gate.side === side) {
        const horiz = side === 'n' || side === 's';
        const a = gate.at - gate.width / 2;
        const b = gate.at + gate.width / 2;
        if (horiz) {
          this.wall(x0, z0, a, z1, h, t, 'wall', mat, color, zone, base);
          this.wall(b, z0, x1, z1, h, t, 'wall', mat, color, zone, base);
          this.wall(a, z0, b, z1, h, t * 0.8, 'wall', 'wood', C.woodDark, zone, base, true);
        } else {
          this.wall(x0, z0, x1, a, h, t, 'wall', mat, color, zone, base);
          this.wall(x0, b, x1, z1, h, t, 'wall', mat, color, zone, base);
          this.wall(x0, a, x1, b, h, t * 0.8, 'wall', 'wood', C.woodDark, zone, base, true);
        }
      } else {
        this.wall(x0, z0, x1, z1, h, t, 'wall', mat, color, zone, base);
      }
    }
  }

  floor(r: Rect, mat: Material, color: number, zone: string, top = 0, depth = 4): void {
    this.aabb(r.minX, top - depth, r.minZ, r.maxX, top, r.maxZ, 'ground', mat, color, zone);
  }
}

function plotRect(cx: number, cz: number, inset = 0): Rect {
  return { minX: cx - PLOT + inset, maxX: cx + PLOT - inset, minZ: cz - PLOT + inset, maxZ: cz + PLOT - inset };
}
function localRect(cx: number, cz: number, x0: number, z0: number, x1: number, z1: number): Rect {
  return { minX: cx + Math.min(x0, x1), maxX: cx + Math.max(x0, x1), minZ: cz + Math.min(z0, z1), maxZ: cz + Math.max(z0, z1) };
}

/** Subtract holes from a rectangle -> list of non-overlapping rectangles. */
function subtractRects(base: Rect, holes: Rect[]): Rect[] {
  let out: Rect[] = [base];
  for (const h of holes) {
    const next: Rect[] = [];
    for (const r of out) {
      if (h.maxX <= r.minX || h.minX >= r.maxX || h.maxZ <= r.minZ || h.minZ >= r.maxZ) {
        next.push(r);
        continue;
      }
      if (h.minZ > r.minZ) next.push({ minX: r.minX, maxX: r.maxX, minZ: r.minZ, maxZ: h.minZ });
      if (h.maxZ < r.maxZ) next.push({ minX: r.minX, maxX: r.maxX, minZ: h.maxZ, maxZ: r.maxZ });
      const z0 = Math.max(r.minZ, h.minZ);
      const z1 = Math.min(r.maxZ, h.maxZ);
      if (h.minX > r.minX) next.push({ minX: r.minX, maxX: h.minX, minZ: z0, maxZ: z1 });
      if (h.maxX < r.maxX) next.push({ minX: h.maxX, maxX: r.maxX, minZ: z0, maxZ: z1 });
    }
    out = next;
  }
  return out;
}

function addZone(b: Builder, z: Omit<ZoneDef, 'floorY'> & { floorY?: number }): void {
  b.zones.push({ floorY: 0, ...z });
}

// ---------------------------------------------------------------------------
// Zones
// ---------------------------------------------------------------------------

function meadow(b: Builder, cx: number, cz: number, rng: RNG): void {
  const id = 'meadow';
  b.floor(plotRect(cx, cz), 'grass', C.grass, id);
  b.fence(plotRect(cx, cz), id, { side: 'e', at: cz, width: 4 });
  b.obj('flag', cx, 0, cz - 12, id);
  for (let i = 0; i < 70; i++) {
    const x = cx + rng.uniform(-20, 20);
    const z = cz + rng.uniform(-20, 20);
    b.decor.push({ kind: rng.next() < 0.5 ? 'flower' : 'tuft', x, y: 0, z, s: rng.uniform(0.6, 1.2), color: [0xffd43b, 0xff6fae, 0xffffff, 0xb28dff, 0xff922b][rng.int(5)] });
  }
  addZone(b, {
    id,
    name: 'Meadow',
    emoji: '🌼',
    blurb: 'A wide open field with a flag. The perfect first classroom: no obstacles, just learn to steer.',
    rect: plotRect(cx, cz),
    spawn: { x: cx, z: cz + 14, yaw: Math.PI },
    area: plotRect(cx, cz, 4),
    color: '#7cc957',
    missions: ['reach-flag', 'reach-flag-dqn', 'reach-flag-evo', 'reach-flag-reinforce', 'imitation'],
  });
}

function coinField(b: Builder, cx: number, cz: number, rng: RNG): void {
  const id = 'coins';
  b.floor(plotRect(cx, cz), 'grass', C.grass2, id);
  b.fence(plotRect(cx, cz), id, { side: 'e', at: cz, width: 4 });
  const placed: [number, number][] = [[cx, cz]];
  const free = (x: number, z: number, d: number) => placed.every(([px, pz]) => Math.hypot(px - x, pz - z) > d);
  for (let i = 0; i < 10; i++) {
    for (let k = 0; k < 50; k++) {
      const x = cx + rng.uniform(-17, 17);
      const z = cz + rng.uniform(-17, 17);
      if (!free(x, z, 6)) continue;
      placed.push([x, z]);
      const r = rng.uniform(0.8, 1.5);
      if (i % 2 === 0) b.cyl(x, z, 0, rng.uniform(0.9, 1.8), r, 'rock', 'stone', C.rock, id);
      else b.cyl(x, z, 0, 1.2, r, 'tree', 'hedge', C.hedge, id);
      break;
    }
  }
  for (let i = 0; i < 12; i++) {
    for (let k = 0; k < 60; k++) {
      const x = cx + rng.uniform(-18, 18);
      const z = cz + rng.uniform(-18, 18);
      if (!free(x, z, 3)) continue;
      placed.push([x, z]);
      b.obj('coin', x, 0, z, id);
      break;
    }
  }
  addZone(b, {
    id,
    name: 'Coin Field',
    emoji: '🪙',
    blurb: 'Coins scattered between rocks and bushes. Coins respawn when collected: how many can you grab?',
    rect: plotRect(cx, cz),
    spawn: { x: cx, z: cz, yaw: 0 },
    area: plotRect(cx, cz, 3),
    color: '#ffcc33',
    missions: ['coin-collector', 'coin-evo'],
  });
}

function maze(b: Builder, cx: number, cz: number): void {
  const id = 'maze';
  const N = 8;
  const S = 5;
  const half = (N * S) / 2; // 20
  b.floor(plotRect(cx, cz), 'dirt', 0x8d7a5a, id);
  // carve with a seeded recursive backtracker
  const rng = new RNG(20251);
  const hWall: boolean[][] = Array.from({ length: N + 1 }, () => Array(N).fill(true)); // [row boundary][col]
  const vWall: boolean[][] = Array.from({ length: N }, () => Array(N + 1).fill(true)); // [row][col boundary]
  const seen: boolean[][] = Array.from({ length: N }, () => Array(N).fill(false));
  const stack: [number, number][] = [[0, N - 1]];
  seen[N - 1][0] = true;
  while (stack.length) {
    const [c, r] = stack[stack.length - 1];
    const nb: [number, number, string][] = [];
    if (r > 0 && !seen[r - 1][c]) nb.push([c, r - 1, 'n']);
    if (r < N - 1 && !seen[r + 1][c]) nb.push([c, r + 1, 's']);
    if (c > 0 && !seen[r][c - 1]) nb.push([c - 1, r, 'w']);
    if (c < N - 1 && !seen[r][c + 1]) nb.push([c + 1, r, 'e']);
    if (!nb.length) {
      stack.pop();
      continue;
    }
    const [nc, nr, d] = nb[rng.int(nb.length)];
    if (d === 'n') hWall[r][c] = false;
    if (d === 's') hWall[r + 1][c] = false;
    if (d === 'w') vWall[r][c] = false;
    if (d === 'e') vWall[r][c + 1] = false;
    seen[nr][nc] = true;
    stack.push([nc, nr]);
  }
  // knock out a few extra walls so there are loops (more interesting)
  for (let k = 0; k < 6; k++) {
    const r = 1 + rng.int(N - 2);
    const c = rng.int(N);
    hWall[r][c] = false;
  }
  // entrance on the south of cell (0, N-1) is a gate
  hWall[N][0] = false;
  const x0 = cx - half;
  const z0 = cz - half;
  const H = 2.4;
  const T = 0.6;
  // merge runs of horizontal walls
  for (let r = 0; r <= N; r++) {
    let c = 0;
    while (c < N) {
      if (!hWall[r][c]) {
        c++;
        continue;
      }
      let e = c;
      while (e + 1 < N && hWall[r][e + 1]) e++;
      b.aabb(x0 + c * S - T / 2, 0, z0 + r * S - T / 2, x0 + (e + 1) * S + T / 2, H, z0 + r * S + T / 2, 'wall', 'hedge', C.hedge, id);
      c = e + 1;
    }
  }
  for (let c = 0; c <= N; c++) {
    let r = 0;
    while (r < N) {
      if (!vWall[r][c]) {
        r++;
        continue;
      }
      let e = r;
      while (e + 1 < N && vWall[e + 1][c]) e++;
      b.aabb(x0 + c * S - T / 2, 0, z0 + r * S + T / 2, x0 + c * S + T / 2, H, z0 + (e + 1) * S - T / 2, 'wall', 'hedge', C.hedge, id);
      r = e + 1;
    }
  }
  // gate at the entrance
  b.aabb(x0 + T / 2, 0, z0 + N * S - T / 2, x0 + S - T / 2, H, z0 + N * S + T / 2, 'wall', 'wood', C.woodDark, id, true);
  // solve with BFS for the breadcrumb checkpoints
  const key = (c: number, r: number) => r * N + c;
  const prev = new Map<number, number>();
  const q: [number, number][] = [[0, N - 1]];
  prev.set(key(0, N - 1), -1);
  while (q.length) {
    const [c, r] = q.shift()!;
    if (c === N - 1 && r === 0) break;
    const tryMove = (nc: number, nr: number, open: boolean) => {
      if (!open || nc < 0 || nr < 0 || nc >= N || nr >= N || prev.has(key(nc, nr))) return;
      prev.set(key(nc, nr), key(c, r));
      q.push([nc, nr]);
    };
    tryMove(c, r - 1, !hWall[r][c]);
    tryMove(c, r + 1, r + 1 < N && !hWall[r + 1][c]);
    tryMove(c - 1, r, !vWall[r][c]);
    tryMove(c + 1, r, !vWall[r][c + 1]);
  }
  const path: number[] = [];
  for (let k = key(N - 1, 0); k !== -1; k = prev.get(k) ?? -1) path.unshift(k);
  const centre = (k: number) => [x0 + ((k % N) + 0.5) * S, z0 + (Math.floor(k / N) + 0.5) * S];
  let idx = 0;
  for (let i = 2; i < path.length - 1; i += 2) {
    const [x, z] = centre(path[i]);
    const [px, pz] = centre(path[i - 1]);
    const yaw = Math.atan2(x - px, z - pz) + Math.PI / 2;
    b.obj('checkpoint', x, 1.2, z, id, { index: idx++, hx: 0.4, hy: 1.2, hz: S / 2 - 0.4, yaw });
  }
  const [tx, tz] = centre(path[path.length - 1]);
  b.obj('trophy', tx, 0, tz, id);
  const [sx, sz] = centre(path[0]);
  addZone(b, {
    id,
    name: 'Hedge Maze',
    emoji: '🌀',
    blurb: 'An 8×8 hedge maze with a trophy in the far corner. Purple checkpoints mark the route: use them, or make your robot figure it out alone.',
    rect: plotRect(cx, cz),
    spawn: { x: sx, z: sz, yaw: Math.PI },
    area: { minX: x0 + 1.5, maxX: x0 + N * S - 1.5, minZ: z0 + 1.5, maxZ: z0 + N * S - 1.5 },
    spawnArea: { minX: sx - 1, maxX: sx + 1, minZ: sz - 1, maxZ: sz + 1 },
    color: '#2f7a3a',
    missions: ['maze-runner'],
  });
}

function lavaLake(b: Builder, cx: number, cz: number, rng: RNG): void {
  const id = 'lava';
  const r = plotRect(cx, cz);
  // lava fills the plot (it's a hole in the island)
  b.aabb(r.minX, -4, r.minZ, r.maxX, -0.5, r.maxZ, 'lava', 'lava', C.lava, id);
  b.fence(r, id, { side: 's', at: cx, width: 4 }, { mat: 'stone', color: C.stoneDark, base: -0.5, height: FENCE_H + 0.5 });
  const plat = (x0: number, z0: number, x1: number, z1: number, top = 0) =>
    b.aabb(cx + x0, -4, cz + z0, cx + x1, top, cz + z1, 'ground', 'stone', C.stone, id);
  plat(-6, 14, 6, 22); // start (touches the south fence / gate)
  plat(-1.2, 6, 1.2, 14); // bridge 1
  plat(-4, 0, 4, 6); // island A
  // diagonal bridge to island B
  const bridge = (ax: number, az: number, bx: number, bz: number, w: number) => {
    const yaw = Math.atan2(bx - ax, bz - az);
    const len = Math.hypot(bx - ax, bz - az);
    b.box(cx + (ax + bx) / 2, -2, cz + (az + bz) / 2, w / 2, 2, len / 2, 'ground', 'stone', C.stone, id, { yaw });
  };
  bridge(2, 2, 10, -6, 2.4);
  plat(7, -12, 15, -4); // island B
  bridge(9, -11, 2, -16.5, 2.2);
  plat(-6, -21.6, 6, -15); // goal
  // decorative stepping stones (also walkable)
  for (let i = 0; i < 9; i++) {
    const x = rng.uniform(-19, 19);
    const z = rng.uniform(-19, 19);
    if (Math.abs(x) < 9 && z > -22 && z < 22) continue;
    const s = rng.uniform(0.8, 1.6);
    b.cyl(cx + x, cz + z, -4, -0.25, s, 'rock', 'stone', C.stoneDark, id);
  }
  b.obj('flag', cx, 0, cz - 18, id);
  addZone(b, {
    id,
    name: 'Lava Lake',
    emoji: '🌋',
    blurb: 'Narrow stone bridges over bubbling lava. Reach the flag on the far side without touching the lava.',
    rect: r,
    spawn: { x: cx, z: cz + 18, yaw: Math.PI },
    area: localRect(cx, cz, -4, 0, 4, 6),
    spawnArea: localRect(cx, cz, -4, 16, 4, 20),
    floorY: -0.5,
    color: '#ff5a1f',
    missions: ['lava-crossing'],
  });
}

function summit(b: Builder, cx: number, cz: number, rng: RNG): void {
  const id = 'summit';
  const r = plotRect(cx, cz);
  b.floor(r, 'grass', C.grassDark, id);
  b.fence(r, id, { side: 's', at: cx, width: 4 });
  const tier = (h: number, top: number, color: number) => b.aabb(cx - h, 0, cz - h, cx + h, top, cz + h, 'rock', 'stone', color, id);
  tier(12, 2, 0x8b8f7a);
  tier(6, 4, 0x9c9f88);
  tier(3, 6, 0xb0b39a);
  b.ramp(cx, cz + 20, 0, cx, cz + 12, 2, 4, id);
  b.ramp(cx + 9, cz + 8, 2, cx + 9, cz - 2, 4, 4, id);
  b.aabb(cx + 6, 0, cz - 6, cx + 12, 4, cz - 2, 'rock', 'stone', 0x9c9f88, id); // landing 2
  b.ramp(cx - 4.5, cz + 6, 4, cx - 4.5, cz, 6, 3, id);
  b.aabb(cx - 6, 0, cz - 6, cx - 3, 6, cz, 'rock', 'stone', 0xb0b39a, id); // landing 3
  b.obj('flag', cx, 6, cz, id);
  for (let i = 0; i < 10; i++) {
    const a = rng.uniform(0, Math.PI * 2);
    const d = rng.uniform(15, 19);
    const x = Math.cos(a) * d;
    const z = Math.sin(a) * d;
    if (z > 12 && Math.abs(x) < 5) continue;
    b.cyl(cx + x, cz + z, 0, rng.uniform(0.6, 1.4), rng.uniform(0.6, 1.2), 'rock', 'stone', C.rock, id);
  }
  addZone(b, {
    id,
    name: 'Summit',
    emoji: '⛰️',
    blurb: 'A three-tier mountain with ramps winding up to a flag at the top (6 m high). Teaches climbing and long-term planning.',
    rect: r,
    spawn: { x: cx, z: cz + 18, yaw: Math.PI },
    area: plotRect(cx, cz, 3),
    spawnArea: localRect(cx, cz, -8, 15, 8, 20),
    color: '#9c9f88',
    missions: ['hill-climb'],
  });
}

function raceTrack(b: Builder, cx: number, cz: number): void {
  const id = 'race';
  const r = plotRect(cx, cz);
  b.floor(r, 'track', C.track, id);
  b.fence(r, id, { side: 's', at: cx, width: 4 }, { mat: 'concrete', color: 0xdedede, height: 1.2 });
  b.aabb(cx - 9, 0, cz - 9, cx + 9, 1.0, cz + 9, 'wall', 'grass', C.grass, id);
  const gates: [number, number, number][] = [
    [0, 15.5, 0],
    [15.5, 15.5, Math.PI / 4],
    [15.5, 0, Math.PI / 2],
    [15.5, -15.5, (3 * Math.PI) / 4],
    [0, -15.5, 0],
    [-15.5, -15.5, (-3 * Math.PI) / 4],
    [-15.5, 0, Math.PI / 2],
    [-15.5, 15.5, -Math.PI / 4],
  ];
  gates.forEach(([x, z, yaw], i) => {
    const diag = i % 2 === 1;
    b.obj('checkpoint', cx + x, 1.2, cz + z, id, { index: i, yaw, hx: 0.6, hy: 1.2, hz: diag ? 9.2 : 6.5 });
  });
  for (const [x, z] of [[-20, -20], [20, -20], [20, 20], [-20, 20]] as const) b.decor.push({ kind: 'barrel', x: cx + x, y: 0, z: cz + z, s: 1, color: 0x222222 });
  b.decor.push({ kind: 'banner', x: cx, y: 0, z: cz + 15.5, w: 13, text: 'START', yaw: 0 });
  addZone(b, {
    id,
    name: 'Race Track',
    emoji: '🏎️',
    blurb: 'A square circuit with 8 checkpoint gates. Drive laps as fast as you can without hitting the walls.',
    rect: r,
    spawn: { x: cx - 5, z: cz + 15.5, yaw: Math.PI / 2 },
    area: plotRect(cx, cz, 2),
    spawnArea: localRect(cx, cz, -7, 13, -3, 18),
    color: '#4a4a55',
    missions: ['race-laps'],
  });
}

function soccer(b: Builder, cx: number, cz: number): void {
  const id = 'soccer';
  const r = plotRect(cx, cz);
  b.floor(r, 'grass', C.pitch, id);
  b.fence(r, id, { side: 'w', at: cz, width: 4 });
  const H = 1.0;
  const T = 0.4;
  const W = 16;
  const L = 18;
  b.wall(cx - W, cz - L, cx - W, cz + L, H, T, 'wall', 'wood', C.white, id);
  b.wall(cx + W, cz - L, cx + W, cz + L, H, T, 'wall', 'wood', C.white, id);
  b.wall(cx - W, cz + L, cx + W, cz + L, H, T, 'wall', 'wood', C.white, id);
  b.wall(cx - W, cz - L, cx - 4.3, cz - L, H, T, 'wall', 'wood', C.white, id);
  b.wall(cx + 4.3, cz - L, cx + W, cz - L, H, T, 'wall', 'wood', C.white, id);
  // goal recess
  b.wall(cx - 4.5, cz - L, cx - 4.5, cz - L - 3.5, 2.6, 0.3, 'wall', 'metal', 0xeeeeee, id);
  b.wall(cx + 4.5, cz - L, cx + 4.5, cz - L - 3.5, 2.6, 0.3, 'wall', 'metal', 0xeeeeee, id);
  b.wall(cx - 4.5, cz - L - 3.5, cx + 4.5, cz - L - 3.5, 2.6, 0.3, 'wall', 'metal', 0xeeeeee, id);
  b.obj('goal', cx, 1.25, cz - L - 1.6, id, { hx: 4.2, hy: 1.25, hz: 1.6 });
  b.obj('ball', cx, 0.55, cz - 2, id);
  b.decor.push({ kind: 'net', x: cx, y: 0, z: cz - L - 1.75, w: 9, d: 3.5, h: 2.6 });
  addZone(b, {
    id,
    name: 'Soccer Pitch',
    emoji: '⚽',
    blurb: 'A walled pitch with a ball and a goal at the north end. Dribble the ball into the net!',
    rect: r,
    spawn: { x: cx, z: cz + 10, yaw: Math.PI },
    area: localRect(cx, cz, -10, -9, 10, 7),
    spawnArea: localRect(cx, cz, -12, 6, 12, 15),
    color: '#4caf50',
    missions: ['soccer'],
  });
}

function skyParkour(b: Builder, cx: number, cz: number): void {
  const id = 'sky';
  const r = plotRect(cx, cz);
  const plat = (x0: number, z0: number, x1: number, z1: number, top: number, color: number) =>
    b.aabb(cx + x0, top - 1.2, cz + z0, cx + x1, top, cz + z1, 'ground', 'metal', color, id);
  plat(-5, 14, 5, 21, 0, 0x7fa7d6);
  plat(-3, 8, 3, 12, 0, 0x8fb5e0);
  plat(-3, 2, 3, 6, 0.8, 0x9fc3ea);
  plat(4, -3, 9, 1, 0.8, 0xafcff2);
  plat(-2, -9, 3, -5, 1.6, 0xbfd9f7);
  plat(-6, -21, 6, -12, 1.6, 0xd0e4fa);
  b.obj('flag', cx, 1.6, cz - 17, id);
  addZone(b, {
    id,
    name: 'Sky Parkour',
    emoji: '☁️',
    blurb: 'Floating platforms over the void. Needs the JUMP action. Fall off and the episode is over.',
    rect: r,
    spawn: { x: cx, z: cz + 17.5, yaw: Math.PI },
    area: localRect(cx, cz, -2, 9, 2, 11),
    spawnArea: localRect(cx, cz, -3, 16, 3, 19),
    floorY: -40,
    color: '#bfd9ff',
    needsJump: true,
    missions: ['parkour'],
  });
}

function forest(b: Builder, cx: number, cz: number, rng: RNG): void {
  const id = 'forest';
  const r = plotRect(cx, cz);
  b.floor(r, 'grass', C.forest, id);
  b.fence(r, id, { side: 'n', at: cx, width: 4 });
  const pts: [number, number][] = [];
  for (let k = 0; k < 2000 && pts.length < 46; k++) {
    const x = rng.uniform(-19.5, 19.5);
    const z = rng.uniform(-19.5, 19.5);
    if (Math.hypot(x, z) < 4 || Math.hypot(x - 12, z + 12) < 2.5) continue;
    if (pts.some(([px, pz]) => Math.hypot(px - x, pz - z) < 3.4)) continue;
    pts.push([x, z]);
  }
  for (const [x, z] of pts) {
    const h = rng.uniform(4, 7);
    b.cyl(cx + x, cz + z, 0, h, rng.uniform(0.35, 0.55), 'tree', 'wood', 0x7a5230, id);
  }
  b.obj('flag', cx + 12, 0, cz - 12, id);
  for (let i = 0; i < 40; i++) b.decor.push({ kind: 'tuft', x: cx + rng.uniform(-20, 20), y: 0, z: cz + rng.uniform(-20, 20), s: rng.uniform(0.8, 1.4), color: 0x4c7a30 });
  addZone(b, {
    id,
    name: 'Forest',
    emoji: '🌲',
    blurb: 'A dense pine forest. Find the flag hidden among the trees: needs eyes (ray sensors) to avoid trunks.',
    rect: r,
    spawn: { x: cx, z: cz, yaw: 0 },
    area: plotRect(cx, cz, 3),
    spawnArea: localRect(cx, cz, -2.5, -2.5, 2.5, 2.5),
    color: '#2f7d32',
    missions: ['forest-flag'],
  });
}

function warehouse(b: Builder, cx: number, cz: number): void {
  const id = 'warehouse';
  const r = plotRect(cx, cz);
  b.floor(r, 'concrete', C.concrete, id);
  b.fence(r, id, { side: 'n', at: cx, width: 4 }, { mat: 'brick', color: C.brick, height: 3 });
  // shelving along the sides
  b.aabb(cx - 21, 0, cz - 16, cx - 17.5, 2.4, cz + 16, 'wall', 'wood', C.wood, id);
  b.aabb(cx + 17.5, 0, cz - 16, cx + 21, 2.4, cz + 16, 'wall', 'wood', C.wood, id);
  b.aabb(cx - 3, 0, cz - 1.5, cx + 3, 1.2, cz + 1.5, 'wall', 'metal', C.metal, id);
  const crates: [number, number][] = [[-8, 6], [0, 7], [8, 6]];
  crates.forEach(([x, z], i) => b.obj('crate', cx + x, 0.6, cz + z, id, { index: i }));
  const pads: [number, number][] = [[-8, -14], [0, -14], [8, -14]];
  pads.forEach(([x, z], i) => b.obj('pad', cx + x, 0.1, cz + z, id, { index: i, hx: 1.8, hy: 0.6, hz: 1.8 }));
  addZone(b, {
    id,
    name: 'Warehouse',
    emoji: '📦',
    blurb: 'Crates and target pads. Push a crate onto a pad: a hard puzzle that needs planning.',
    rect: r,
    spawn: { x: cx, z: cz + 15, yaw: Math.PI },
    area: localRect(cx, cz, -12, 4, 12, 10),
    spawnArea: localRect(cx, cz, -12, 12, 12, 18),
    color: '#c8893f',
    missions: ['crate-push'],
  });
}

function tagArena(b: Builder, cx: number, cz: number): void {
  const id = 'tag';
  const r = plotRect(cx, cz);
  b.floor(r, 'sand', C.sand, id);
  b.fence(r, id, { side: 'n', at: cx, width: 4 });
  for (const [x, z] of [[-7, -7], [7, -7], [7, 7], [-7, 7]] as const) b.cyl(cx + x, cz + z, 0, 2.2, 1.2, 'rock', 'stone', 0xc0a070, id);
  b.obj('runner', cx, 0.45, cz - 8, id);
  addZone(b, {
    id,
    name: 'Tag Arena',
    emoji: '🐰',
    blurb: 'A runner bot that flees from you. Catch it! The runner is a little slower than you, but it dodges.',
    rect: r,
    spawn: { x: cx, z: cz + 12, yaw: Math.PI },
    area: plotRect(cx, cz, 3),
    color: '#e0bc78',
    missions: ['tag'],
  });
}

function dodgeArena(b: Builder, cx: number, cz: number, rng: RNG): void {
  const id = 'dodge';
  const r = plotRect(cx, cz);
  b.floor(r, 'rubber', C.rubber, id);
  b.fence(r, id, { side: 'n', at: cx, width: 4 }, { mat: 'metal', color: 0x5a6a8a });
  b.obj('chaser', cx - 12, 0.55, cz - 12, id);
  b.obj('chaser', cx + 12, 0.55, cz - 12, id);
  for (let i = 0; i < 10; i++) b.obj('coin', cx + rng.uniform(-16, 16), 0, cz + rng.uniform(-16, 16), id);
  addZone(b, {
    id,
    name: 'Dodge Arena',
    emoji: '👾',
    blurb: 'Two chaser bots hunt you while you collect coins. Survive and score!',
    rect: r,
    spawn: { x: cx, z: cz + 14, yaw: Math.PI },
    area: plotRect(cx, cz, 3),
    color: '#3d5a99',
    missions: ['dodge'],
  });
}

// ---------------------------------------------------------------------------

let cached: WorldDef | null = null;

export function buildWorld(): WorldDef {
  if (cached) return cached;
  const b = new Builder();
  const rng = new RNG(424242);
  const bounds: Rect = { minX: -114, maxX: 114, minZ: -86, maxZ: 86 };

  // island slab with holes where zones have their own floors
  const holes: Rect[] = [];
  for (const x of COLS) for (const z of ROWS) holes.push(plotRect(x, z));
  for (const rr of subtractRects(bounds, holes)) b.aabb(rr.minX, -4, rr.minZ, rr.maxX, 0, rr.maxZ, 'ground', 'grass', C.grass, 'world');

  maze(b, COLS[0], ROWS[0]);
  lavaLake(b, COLS[1], ROWS[0], rng);
  summit(b, COLS[2], ROWS[0], rng);
  raceTrack(b, COLS[3], ROWS[0]);
  coinField(b, COLS[0], ROWS[1], rng);
  meadow(b, COLS[1], ROWS[1], rng);
  soccer(b, COLS[2], ROWS[1]);
  skyParkour(b, COLS[3], ROWS[1]);
  forest(b, COLS[0], ROWS[2], rng);
  warehouse(b, COLS[1], ROWS[2]);
  tagArena(b, COLS[2], ROWS[2]);
  dodgeArena(b, COLS[3], ROWS[2], rng);

  // free roam = the whole island
  b.zones.push({
    id: 'world',
    name: 'Free Roam (whole island)',
    emoji: '🏝️',
    blurb: 'All gates open. The whole island is the playground: a big, hard, open-ended world.',
    rect: bounds,
    spawn: { x: 0, z: 0, yaw: Math.PI },
    area: { minX: bounds.minX + 4, maxX: bounds.maxX - 4, minZ: bounds.minZ + 4, maxZ: bounds.maxZ - 4 },
    spawnArea: { minX: -3, maxX: 3, minZ: -3, maxZ: 3 },
    floorY: 0,
    color: '#66ccff',
    missions: [],
  });

  // ---- decoration (render only) ----
  for (const z of [-28, 28]) b.decor.push({ kind: 'path', x: 0, y: 0.02, z, w: 224, d: 5, color: 0xcfb58a });
  for (const x of [-56, 0, 56]) b.decor.push({ kind: 'path', x, y: 0.02, z: 0, w: 5, d: 168, color: 0xcfb58a });
  for (const x of [-56, 0, 56])
    for (const z of [-28, 28]) {
      b.decor.push({ kind: 'lamp', x: x + 3.2, y: 0, z: z + 3.2 });
      b.decor.push({ kind: 'lamp', x: x - 3.2, y: 0, z: z - 3.2 });
    }
  for (const zn of b.zones) {
    if (zn.id === 'world') continue;
    const cxz = (zn.rect.minX + zn.rect.maxX) / 2;
    const czz = (zn.rect.minZ + zn.rect.maxZ) / 2;
    b.decor.push({ kind: 'sign', x: cxz, y: 0, z: czz, text: `${zn.emoji} ${zn.name}` });
  }
  // trees & bushes around the island edges
  for (let i = 0; i < 90; i++) {
    const side = rng.int(4);
    let x: number;
    let z: number;
    if (side < 2) {
      x = rng.uniform(bounds.minX + 2, bounds.maxX - 2);
      z = side === 0 ? rng.uniform(bounds.minZ + 1, bounds.minZ + 5) : rng.uniform(bounds.maxZ - 5, bounds.maxZ - 1);
    } else {
      z = rng.uniform(bounds.minZ + 2, bounds.maxZ - 2);
      x = side === 2 ? rng.uniform(bounds.minX + 1, bounds.minX + 5) : rng.uniform(bounds.maxX - 5, bounds.maxX - 1);
    }
    b.decor.push({ kind: 'bush', x, y: 0, z, s: rng.uniform(0.7, 1.6), color: rng.next() < 0.5 ? 0x3d8b3d : 0x2f6f2f });
  }
  for (let i = 0; i < 28; i++) {
    const a = (i / 28) * Math.PI * 2 + rng.uniform(-0.05, 0.05);
    const d = rng.uniform(240, 330);
    b.decor.push({ kind: 'mountain', x: Math.cos(a) * d, y: -6, z: Math.sin(a) * d * 0.85, s: rng.uniform(30, 75), color: rng.next() < 0.5 ? 0x7a8fa6 : 0x8c9fb5 });
  }
  for (let i = 0; i < 22; i++) b.decor.push({ kind: 'cloud', x: rng.uniform(-260, 260), y: rng.uniform(45, 80), z: rng.uniform(-220, 220), s: rng.uniform(8, 18) });

  cached = { statics: b.statics, objects: b.objects, zones: b.zones, decor: b.decor, bounds, waterY: -6 };
  return cached;
}

export function getZone(world: WorldDef, id: string): ZoneDef {
  return world.zones.find((z) => z.id === id) ?? world.zones[0];
}
