/** The island is deterministic, so it is generated once and shared. */
import { buildWorld } from './worldgen.ts';
import type { WorldDef } from './types.ts';

let cached: WorldDef | null = null;

export function getWorld(): WorldDef {
  return (cached ??= buildWorld());
}
