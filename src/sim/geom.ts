/**
 * Collision geometry: static colliders, sphere contacts, ray casts and a
 * uniform spatial grid so we only test things that are nearby.
 */
import type { StaticDef, StaticTag, Rect } from './types.ts';

/** Rotation local→world from yaw (about Y) then pitch (about local X). Row-major. */
export function rotYawPitch(yaw: number, pitch: number): number[] {
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  // Ry * Rx
  return [cy, sy * sp, sy * cp, 0, cp, -sp, -sy, cy * sp, cy * cp];
}

export interface Collider {
  idx: number;
  kind: 'box' | 'cyl';
  tag: StaticTag;
  zone: string;
  gate: boolean;
  // box
  cx: number;
  cy: number;
  cz: number;
  hx: number;
  hy: number;
  hz: number;
  R: Float64Array; // local→world (row-major)
  axisAligned: boolean;
  // cylinder
  r: number;
  y0: number;
  y1: number;
  // bounds
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  minZ: number;
  maxZ: number;
}

export function compileStatic(s: StaticDef, idx: number): Collider {
  if (s.kind === 'box') {
    const R = new Float64Array(s.R);
    const [hx, hy, hz] = s.h;
    // AABB of a rotated box: extent_i = Σ_j |R_ij| h_j
    const ex = Math.abs(R[0]) * hx + Math.abs(R[1]) * hy + Math.abs(R[2]) * hz;
    const ey = Math.abs(R[3]) * hx + Math.abs(R[4]) * hy + Math.abs(R[5]) * hz;
    const ez = Math.abs(R[6]) * hx + Math.abs(R[7]) * hy + Math.abs(R[8]) * hz;
    const aa = Math.abs(R[0] - 1) < 1e-9 && Math.abs(R[4] - 1) < 1e-9 && Math.abs(R[8] - 1) < 1e-9;
    return {
      idx,
      kind: 'box',
      tag: s.tag,
      zone: s.zone,
      gate: !!s.gate,
      cx: s.c[0],
      cy: s.c[1],
      cz: s.c[2],
      hx,
      hy,
      hz,
      R,
      axisAligned: aa,
      r: 0,
      y0: 0,
      y1: 0,
      minX: s.c[0] - ex,
      maxX: s.c[0] + ex,
      minY: s.c[1] - ey,
      maxY: s.c[1] + ey,
      minZ: s.c[2] - ez,
      maxZ: s.c[2] + ez,
    };
  }
  return {
    idx,
    kind: 'cyl',
    tag: s.tag,
    zone: s.zone,
    gate: false,
    cx: s.c[0],
    cy: (s.y0 + s.y1) / 2,
    cz: s.c[1],
    hx: s.r,
    hy: (s.y1 - s.y0) / 2,
    hz: s.r,
    R: new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
    axisAligned: true,
    r: s.r,
    y0: s.y0,
    y1: s.y1,
    minX: s.c[0] - s.r,
    maxX: s.c[0] + s.r,
    minY: s.y0,
    maxY: s.y1,
    minZ: s.c[1] - s.r,
    maxZ: s.c[1] + s.r,
  };
}

/** Result of a sphere contact query (reused object to avoid garbage). */
export interface Contact {
  nx: number;
  ny: number;
  nz: number;
  depth: number;
}

/**
 * Sphere (px,py,pz,r) vs collider. Returns true and fills `out` with the
 * push-out normal and penetration depth when they overlap.
 */
export function sphereVsCollider(c: Collider, px: number, py: number, pz: number, r: number, out: Contact): boolean {
  if (px + r < c.minX || px - r > c.maxX || py + r < c.minY || py - r > c.maxY || pz + r < c.minZ || pz - r > c.maxZ)
    return false;
  if (c.kind === 'box') {
    const R = c.R;
    const dx = px - c.cx;
    const dy = py - c.cy;
    const dz = pz - c.cz;
    // world → local = Rᵀ d
    const lx = R[0] * dx + R[3] * dy + R[6] * dz;
    const ly = R[1] * dx + R[4] * dy + R[7] * dz;
    const lz = R[2] * dx + R[5] * dy + R[8] * dz;
    const qx = lx < -c.hx ? -c.hx : lx > c.hx ? c.hx : lx;
    const qy = ly < -c.hy ? -c.hy : ly > c.hy ? c.hy : ly;
    const qz = lz < -c.hz ? -c.hz : lz > c.hz ? c.hz : lz;
    const ex = lx - qx;
    const ey = ly - qy;
    const ez = lz - qz;
    const d2 = ex * ex + ey * ey + ez * ez;
    let nlx: number;
    let nly: number;
    let nlz: number;
    let depth: number;
    if (d2 > 1e-12) {
      if (d2 >= r * r) return false;
      const d = Math.sqrt(d2);
      nlx = ex / d;
      nly = ey / d;
      nlz = ez / d;
      depth = r - d;
    } else {
      // centre inside the box: push out through the nearest face
      const fx = c.hx - Math.abs(lx);
      const fy = c.hy - Math.abs(ly);
      const fz = c.hz - Math.abs(lz);
      nlx = 0;
      nly = 0;
      nlz = 0;
      if (fy <= fx && fy <= fz) {
        nly = ly >= 0 ? 1 : -1;
        depth = fy + r;
      } else if (fx <= fz) {
        nlx = lx >= 0 ? 1 : -1;
        depth = fx + r;
      } else {
        nlz = lz >= 0 ? 1 : -1;
        depth = fz + r;
      }
    }
    // local → world = R n
    out.nx = R[0] * nlx + R[1] * nly + R[2] * nlz;
    out.ny = R[3] * nlx + R[4] * nly + R[5] * nlz;
    out.nz = R[6] * nlx + R[7] * nly + R[8] * nlz;
    out.depth = depth;
    return true;
  }
  // vertical cylinder
  const dx = px - c.cx;
  const dz = pz - c.cz;
  const dh = Math.sqrt(dx * dx + dz * dz);
  let qx = px;
  let qz = pz;
  if (dh > c.r) {
    qx = c.cx + (dx / dh) * c.r;
    qz = c.cz + (dz / dh) * c.r;
  }
  const qy = py < c.y0 ? c.y0 : py > c.y1 ? c.y1 : py;
  const ex = px - qx;
  const ey = py - qy;
  const ez = pz - qz;
  const d2 = ex * ex + ey * ey + ez * ez;
  if (d2 > 1e-12) {
    if (d2 >= r * r) return false;
    const d = Math.sqrt(d2);
    out.nx = ex / d;
    out.ny = ey / d;
    out.nz = ez / d;
    out.depth = r - d;
    return true;
  }
  const radial = c.r - dh;
  const up = c.y1 - py;
  if (up < radial) {
    out.nx = 0;
    out.ny = 1;
    out.nz = 0;
    out.depth = up + r;
  } else {
    const inv = dh > 1e-9 ? 1 / dh : 0;
    out.nx = dh > 1e-9 ? dx * inv : 1;
    out.ny = 0;
    out.nz = dh > 1e-9 ? dz * inv : 0;
    out.depth = radial + r;
  }
  return true;
}

/** Ray vs collider. Returns hit distance t in [0, maxT] or -1. Direction must be unit length. */
export function rayVsCollider(c: Collider, ox: number, oy: number, oz: number, dx: number, dy: number, dz: number, maxT: number): number {
  if (c.kind === 'box') {
    const R = c.R;
    const rx = ox - c.cx;
    const ry = oy - c.cy;
    const rz = oz - c.cz;
    const lox = R[0] * rx + R[3] * ry + R[6] * rz;
    const loy = R[1] * rx + R[4] * ry + R[7] * rz;
    const loz = R[2] * rx + R[5] * ry + R[8] * rz;
    const ldx = R[0] * dx + R[3] * dy + R[6] * dz;
    const ldy = R[1] * dx + R[4] * dy + R[7] * dz;
    const ldz = R[2] * dx + R[5] * dy + R[8] * dz;
    let tmin = 0;
    let tmax = maxT;
    // slab test on each axis
    const o = [lox, loy, loz];
    const d = [ldx, ldy, ldz];
    const h = [c.hx, c.hy, c.hz];
    for (let i = 0; i < 3; i++) {
      if (Math.abs(d[i]) < 1e-12) {
        if (o[i] < -h[i] || o[i] > h[i]) return -1;
      } else {
        const inv = 1 / d[i];
        let t1 = (-h[i] - o[i]) * inv;
        let t2 = (h[i] - o[i]) * inv;
        if (t1 > t2) {
          const t = t1;
          t1 = t2;
          t2 = t;
        }
        if (t1 > tmin) tmin = t1;
        if (t2 < tmax) tmax = t2;
        if (tmin > tmax) return -1;
      }
    }
    return tmin;
  }
  // cylinder: side
  let best = -1;
  const fx = ox - c.cx;
  const fz = oz - c.cz;
  const a = dx * dx + dz * dz;
  if (a > 1e-12) {
    const b = 2 * (fx * dx + fz * dz);
    const cc = fx * fx + fz * fz - c.r * c.r;
    const disc = b * b - 4 * a * cc;
    if (disc >= 0) {
      const sq = Math.sqrt(disc);
      let t = (-b - sq) / (2 * a);
      if (t < 0 && cc < 0) t = 0; // origin inside
      if (t >= 0 && t <= maxT) {
        const y = oy + dy * t;
        if (y >= c.y0 && y <= c.y1) best = t;
      }
    }
  }
  // top cap
  if (Math.abs(dy) > 1e-9) {
    const t = (c.y1 - oy) / dy;
    if (t >= 0 && t <= maxT && (best < 0 || t < best)) {
      const x = ox + dx * t - c.cx;
      const z = oz + dz * t - c.cz;
      if (x * x + z * z <= c.r * c.r) best = t;
    }
  }
  return best;
}

/** Uniform grid over the world, each cell lists the colliders that overlap it. */
export class StaticGrid {
  readonly cell: number;
  readonly minX: number;
  readonly minZ: number;
  readonly nx: number;
  readonly nz: number;
  readonly cells: Int32Array[];
  readonly colliders: Collider[];
  private stamp: Int32Array;
  private stampId = 1;

  constructor(colliders: Collider[], bounds: Rect, cell = 4) {
    this.cell = cell;
    this.colliders = colliders;
    this.minX = bounds.minX - 8;
    this.minZ = bounds.minZ - 8;
    this.nx = Math.ceil((bounds.maxX + 8 - this.minX) / cell);
    this.nz = Math.ceil((bounds.maxZ + 8 - this.minZ) / cell);
    const lists: number[][] = Array.from({ length: this.nx * this.nz }, () => []);
    for (const c of colliders) {
      const x0 = this.cx(c.minX);
      const x1 = this.cx(c.maxX);
      const z0 = this.cz(c.minZ);
      const z1 = this.cz(c.maxZ);
      for (let z = z0; z <= z1; z++) for (let x = x0; x <= x1; x++) lists[z * this.nx + x].push(c.idx);
    }
    this.cells = lists.map((l) => Int32Array.from(l));
    this.stamp = new Int32Array(colliders.length);
  }

  cx(x: number): number {
    const i = Math.floor((x - this.minX) / this.cell);
    return i < 0 ? 0 : i >= this.nx ? this.nx - 1 : i;
  }
  cz(z: number): number {
    const i = Math.floor((z - this.minZ) / this.cell);
    return i < 0 ? 0 : i >= this.nz ? this.nz - 1 : i;
  }

  private nextStamp(): number {
    this.stampId++;
    if (this.stampId > 2e9) {
      this.stamp.fill(0);
      this.stampId = 1;
    }
    return this.stampId;
  }

  /** Fills `out` with colliders near the box (deduplicated). Returns the count. */
  gather(minX: number, minZ: number, maxX: number, maxZ: number, out: Collider[]): number {
    const s = this.nextStamp();
    const x0 = this.cx(minX);
    const x1 = this.cx(maxX);
    const z0 = this.cz(minZ);
    const z1 = this.cz(maxZ);
    let n = 0;
    for (let z = z0; z <= z1; z++)
      for (let x = x0; x <= x1; x++) {
        const list = this.cells[z * this.nx + x];
        for (let k = 0; k < list.length; k++) {
          const i = list[k];
          if (this.stamp[i] === s) continue;
          this.stamp[i] = s;
          out[n++] = this.colliders[i];
        }
      }
    return n;
  }

  /** Calls fn for every collider whose cell range overlaps the box (deduplicated). */
  query(minX: number, minZ: number, maxX: number, maxZ: number, fn: (c: Collider) => void): void {
    const s = this.nextStamp();
    const x0 = this.cx(minX);
    const x1 = this.cx(maxX);
    const z0 = this.cz(minZ);
    const z1 = this.cz(maxZ);
    for (let z = z0; z <= z1; z++)
      for (let x = x0; x <= x1; x++) {
        const list = this.cells[z * this.nx + x];
        for (let k = 0; k < list.length; k++) {
          const i = list[k];
          if (this.stamp[i] === s) continue;
          this.stamp[i] = s;
          fn(this.colliders[i]);
        }
      }
  }

  /**
   * Ray cast with a 2D DDA walk through grid cells (mostly-horizontal rays).
   * `skip` lets the caller ignore some colliders (e.g. open gates).
   */
  raycast(
    ox: number,
    oy: number,
    oz: number,
    dx: number,
    dy: number,
    dz: number,
    maxT: number,
    skip: ((c: Collider) => boolean) | null,
  ): { t: number; c: Collider | null } {
    const s = this.nextStamp();
    let bestT = maxT;
    let best: Collider | null = null;
    let ix = this.cx(ox);
    let iz = this.cz(oz);
    const stepX = dx > 0 ? 1 : -1;
    const stepZ = dz > 0 ? 1 : -1;
    const cell = this.cell;
    const nextBX = this.minX + (ix + (dx > 0 ? 1 : 0)) * cell;
    const nextBZ = this.minZ + (iz + (dz > 0 ? 1 : 0)) * cell;
    let tMaxX = Math.abs(dx) > 1e-12 ? (nextBX - ox) / dx : Infinity;
    let tMaxZ = Math.abs(dz) > 1e-12 ? (nextBZ - oz) / dz : Infinity;
    const tDX = Math.abs(dx) > 1e-12 ? cell / Math.abs(dx) : Infinity;
    const tDZ = Math.abs(dz) > 1e-12 ? cell / Math.abs(dz) : Infinity;
    for (let iter = 0; iter < 256; iter++) {
      const list = this.cells[iz * this.nx + ix];
      for (let k = 0; k < list.length; k++) {
        const i = list[k];
        if (this.stamp[i] === s) continue;
        this.stamp[i] = s;
        const c = this.colliders[i];
        if (skip && skip(c)) continue;
        const t = rayVsCollider(c, ox, oy, oz, dx, dy, dz, bestT);
        if (t >= 0 && t < bestT) {
          bestT = t;
          best = c;
        }
      }
      const tNext = tMaxX < tMaxZ ? tMaxX : tMaxZ;
      if (tNext >= bestT) break;
      if (tMaxX < tMaxZ) {
        ix += stepX;
        if (ix < 0 || ix >= this.nx) break;
        tMaxX += tDX;
      } else {
        iz += stepZ;
        if (iz < 0 || iz >= this.nz) break;
        tMaxZ += tDZ;
      }
    }
    return { t: best ? bestT : -1, c: best };
  }
}

/** Ray vs sphere; returns t or -1. */
export function rayVsSphere(ox: number, oy: number, oz: number, dx: number, dy: number, dz: number, cx: number, cy: number, cz: number, r: number, maxT: number): number {
  const fx = ox - cx;
  const fy = oy - cy;
  const fz = oz - cz;
  const b = fx * dx + fy * dy + fz * dz;
  const c = fx * fx + fy * fy + fz * fz - r * r;
  if (c > 0 && b > 0) return -1;
  const disc = b * b - c;
  if (disc < 0) return -1;
  let t = -b - Math.sqrt(disc);
  if (t < 0) t = 0;
  return t <= maxT ? t : -1;
}

/** Ray vs yaw-rotated box (centre c, half h). */
export function rayVsYawBox(ox: number, oy: number, oz: number, dx: number, dy: number, dz: number, cx: number, cy: number, cz: number, hx: number, hy: number, hz: number, yaw: number, maxT: number): number {
  const cs = Math.cos(yaw);
  const sn = Math.sin(yaw);
  // world→local for yaw rotation: inverse of Ry
  const rx = ox - cx;
  const rz = oz - cz;
  const lox = cs * rx - sn * rz;
  const loz = sn * rx + cs * rz;
  const ldx = cs * dx - sn * dz;
  const ldz = sn * dx + cs * dz;
  const o = [lox, oy - cy, loz];
  const d = [ldx, dy, ldz];
  const h = [hx, hy, hz];
  let tmin = 0;
  let tmax = maxT;
  for (let i = 0; i < 3; i++) {
    if (Math.abs(d[i]) < 1e-12) {
      if (o[i] < -h[i] || o[i] > h[i]) return -1;
    } else {
      const inv = 1 / d[i];
      let t1 = (-h[i] - o[i]) * inv;
      let t2 = (h[i] - o[i]) * inv;
      if (t1 > t2) {
        const t = t1;
        t1 = t2;
        t2 = t;
      }
      if (t1 > tmin) tmin = t1;
      if (t2 < tmax) tmax = t2;
      if (tmin > tmax) return -1;
    }
  }
  return tmin;
}

/** Point inside yaw-rotated box (with margin)? */
export function pointInYawBox(px: number, py: number, pz: number, cx: number, cy: number, cz: number, hx: number, hy: number, hz: number, yaw: number, margin: number): boolean {
  const cs = Math.cos(yaw);
  const sn = Math.sin(yaw);
  const rx = px - cx;
  const rz = pz - cz;
  const lx = cs * rx - sn * rz;
  const lz = sn * rx + cs * rz;
  return Math.abs(lx) <= hx + margin && Math.abs(py - cy) <= hy + margin && Math.abs(lz) <= hz + margin;
}

export function wrapAngle(a: number): number {
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a < -Math.PI) a += 2 * Math.PI;
  return a;
}
