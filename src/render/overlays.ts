/**
 * Ground overlays for a zone:
 *  • Exploration heatmap: where have the training robots been?
 *  • Value map: what does the critic think each spot is worth?
 */
import * as THREE from 'three';
import type { Rect } from '../sim/types.ts';

export class GridOverlay {
  mesh: THREE.Mesh;
  private tex: THREE.DataTexture;
  private data: Uint8Array;
  readonly res: number;
  values: Float32Array;
  rect: Rect = { minX: 0, maxX: 1, minZ: 0, maxZ: 1 };

  constructor(res = 64) {
    this.res = res;
    this.data = new Uint8Array(res * res * 4);
    this.values = new Float32Array(res * res);
    this.tex = new THREE.DataTexture(this.data, res, res, THREE.RGBAFormat);
    this.tex.magFilter = THREE.LinearFilter;
    this.tex.minFilter = THREE.LinearFilter;
    this.tex.needsUpdate = true;
    this.mesh = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2),
      new THREE.MeshBasicMaterial({ map: this.tex, transparent: true, depthWrite: false, opacity: 0.85 }),
    );
    this.mesh.renderOrder = 2;
    this.mesh.visible = false;
  }

  place(rect: Rect, y: number): void {
    this.rect = rect;
    this.mesh.position.set((rect.minX + rect.maxX) / 2, y, (rect.minZ + rect.maxZ) / 2);
    this.mesh.scale.set(rect.maxX - rect.minX, 1, rect.maxZ - rect.minZ);
  }

  clear(): void {
    this.values.fill(0);
    this.data.fill(0);
    this.tex.needsUpdate = true;
  }

  cellOf(x: number, z: number): number {
    const r = this.rect;
    const i = Math.floor(((x - r.minX) / (r.maxX - r.minX)) * this.res);
    const j = Math.floor(((z - r.minZ) / (r.maxZ - r.minZ)) * this.res);
    if (i < 0 || j < 0 || i >= this.res || j >= this.res) return -1;
    return j * this.res + i;
  }

  /** world-space centre of a cell */
  cellCenter(k: number): [number, number] {
    const r = this.rect;
    const i = k % this.res;
    const j = Math.floor(k / this.res);
    return [r.minX + ((i + 0.5) / this.res) * (r.maxX - r.minX), r.minZ + ((j + 0.5) / this.res) * (r.maxZ - r.minZ)];
  }

  addVisit(x: number, z: number, w = 1): void {
    const k = this.cellOf(x, z);
    if (k >= 0) this.values[k] += w;
  }

  /** Heat colours: transparent → yellow → red (log scale). */
  paintHeat(): void {
    let max = 0;
    for (const v of this.values) max = Math.max(max, v);
    const lm = Math.log1p(max) || 1;
    for (let k = 0; k < this.values.length; k++) {
      const v = Math.log1p(this.values[k]) / lm;
      const o = this.flipIndex(k) * 4;
      if (v <= 0) {
        this.data[o + 3] = 0;
        continue;
      }
      this.data[o] = 255;
      this.data[o + 1] = Math.round(255 * (1 - v * 0.85));
      this.data[o + 2] = Math.round(80 * (1 - v));
      this.data[o + 3] = Math.round(60 + 170 * v);
    }
    this.tex.needsUpdate = true;
  }

  /** Diverging colours: blue (low value) → white → red (high value). NaN = transparent. */
  paintValues(vals: Float32Array): { min: number; max: number } {
    let min = Infinity;
    let max = -Infinity;
    for (const v of vals) {
      if (!Number.isFinite(v)) continue;
      min = Math.min(min, v);
      max = Math.max(max, v);
    }
    const span = max - min || 1;
    for (let k = 0; k < vals.length; k++) {
      const o = this.flipIndex(k) * 4;
      const v = vals[k];
      if (!Number.isFinite(v)) {
        this.data[o + 3] = 0;
        continue;
      }
      const t = (v - min) / span;
      // viridis-like ramp
      const c = ramp(t);
      this.data[o] = c[0];
      this.data[o + 1] = c[1];
      this.data[o + 2] = c[2];
      this.data[o + 3] = 200;
    }
    this.tex.needsUpdate = true;
    return { min, max };
  }

  // texture rows go bottom→top while our grid rows go minZ→maxZ (plane rotated -90° about X)
  private flipIndex(k: number): number {
    const i = k % this.res;
    const j = Math.floor(k / this.res);
    return (this.res - 1 - j) * this.res + i;
  }
}

const STOPS: [number, number, number][] = [
  [68, 1, 84],
  [59, 82, 139],
  [33, 145, 140],
  [94, 201, 98],
  [253, 231, 37],
];
export function ramp(t: number): [number, number, number] {
  t = Math.max(0, Math.min(1, t)) * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(t));
  const f = t - i;
  const a = STOPS[i];
  const b = STOPS[i + 1];
  return [Math.round(a[0] + (b[0] - a[0]) * f), Math.round(a[1] + (b[1] - a[1]) * f), Math.round(a[2] + (b[2] - a[2]) * f)];
}
