/**
 * WorldRuntime: the compiled, shared, read-only part of the world
 * (colliders + spatial grid + per-zone lookup tables). Many Envs share one.
 */
import { compileStatic, StaticGrid } from './geom.ts';
import type { Collider } from './geom.ts';
import type { ObjDef, StaticDef, WorldDef, ZoneDef, Rect } from './types.ts';
import { rotYawPitch } from './geom.ts';

export interface CustomWall {
  x: number;
  z: number;
  w: number;
  d: number;
  h: number;
  yaw: number;
  tag: 'wall' | 'lava' | 'ice' | 'bouncy' | 'ramp' | 'spike';
}

export interface WorldEdits {
  objects: ObjDef[];
  walls: CustomWall[];
}

export function customWallToStatic(w: CustomWall, zone: string): StaticDef {
  const colors: Record<CustomWall['tag'], number> = {
    wall: 0xd9cbb3,
    lava: 0xff5a1f,
    ice: 0xa8e6ff,
    bouncy: 0xc264ff,
    ramp: 0xc9a46b,
    spike: 0x777788,
  };
  const flat = w.tag === 'lava' || w.tag === 'ice' || w.tag === 'bouncy' || w.tag === 'spike';
  const h = flat ? 0.12 : w.h;
  const pitch = w.tag === 'ramp' ? -Math.atan2(w.h, w.d) : 0;
  return {
    kind: 'box',
    c: [w.x, w.tag === 'ramp' ? w.h / 2 - 0.1 : h / 2, w.z],
    h: [w.w / 2, w.tag === 'ramp' ? 0.2 : h / 2, w.tag === 'ramp' ? Math.hypot(w.d, w.h) / 2 : w.d / 2],
    R: rotYawPitch(w.yaw, pitch),
    yaw: w.yaw,
    pitch,
    tag: w.tag,
    mat: w.tag === 'lava' ? 'lava' : w.tag === 'ice' ? 'ice' : w.tag === 'wall' ? 'brick' : w.tag === 'ramp' ? 'wood' : 'rubber',
    color: colors[w.tag],
    zone,
  };
}

export function zoneAt(world: WorldDef, x: number, z: number): string {
  for (const zn of world.zones) {
    if (zn.id === 'world') continue;
    const r = zn.rect;
    if (x >= r.minX && x <= r.maxX && z >= r.minZ && z <= r.maxZ) return zn.id;
  }
  return 'world';
}

export class WorldRuntime {
  readonly def: WorldDef;
  readonly statics: StaticDef[];
  readonly objects: ObjDef[];
  readonly colliders: Collider[];
  readonly grid: StaticGrid;
  private byZoneTag = new Map<string, Collider[]>();
  readonly edits: WorldEdits;

  constructor(def: WorldDef, edits: WorldEdits = { objects: [], walls: [] }) {
    this.def = def;
    this.edits = edits;
    const customStatics = edits.walls.map((w) => customWallToStatic(w, zoneAt(def, w.x, w.z)));
    this.statics = [...def.statics, ...customStatics];
    this.objects = [
      ...def.objects,
      ...edits.objects.map((o) => ({ ...o, zone: zoneAt(def, o.x, o.z), custom: true })),
    ];
    this.colliders = this.statics.map((s, i) => compileStatic(s, i));
    this.grid = new StaticGrid(this.colliders, def.bounds, 4);
    for (const c of this.colliders) {
      for (const key of [`${c.zone}|${c.tag}`, `*|${c.tag}`]) {
        let l = this.byZoneTag.get(key);
        if (!l) this.byZoneTag.set(key, (l = []));
        l.push(c);
      }
    }
  }

  collidersWithTag(zone: string, tag: string): Collider[] {
    return this.byZoneTag.get(zone === 'world' ? `*|${tag}` : `${zone}|${tag}`) ?? [];
  }

  zone(id: string): ZoneDef {
    return this.def.zones.find((z) => z.id === id) ?? this.def.zones[0];
  }

  zoneRect(id: string): Rect {
    return this.zone(id).rect;
  }
}
