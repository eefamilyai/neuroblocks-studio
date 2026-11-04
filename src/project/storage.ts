/** Saving & loading projects (files and browser autosave). */
import type { BrainSnapshot, EpisodeRecord } from '../ml/algos/common.ts';
import type { WorldEdits } from '../sim/runtime.ts';

export interface ProjectFile {
  format: 'neuroblocks-project';
  version: 1;
  name: string;
  savedAt: string;
  missionId: string | null;
  workspace: object;
  edits: WorldEdits;
  brain: BrainSnapshot | null;
  episodes?: EpisodeRecord[];
}

const KEY = 'neuroblocks.autosave.v1';
const DONE_KEY = 'neuroblocks.missions.done';
const SEEN_KEY = 'neuroblocks.welcome.seen';

export function saveLocal(p: ProjectFile): boolean {
  try {
    let s = JSON.stringify(p);
    if (s.length > 4_000_000) s = JSON.stringify({ ...p, episodes: [], brain: null });
    localStorage.setItem(KEY, s);
    return true;
  } catch {
    try {
      localStorage.setItem(KEY, JSON.stringify({ ...p, episodes: [], brain: null }));
    } catch {
      /* storage full or blocked: ignore */
    }
    return false;
  }
}

export function loadLocal(): ProjectFile | null {
  try {
    const s = localStorage.getItem(KEY);
    if (!s) return null;
    const p = JSON.parse(s);
    return validateProject(p) ? p : null;
  } catch {
    return null;
  }
}

export function validateProject(p: unknown): p is ProjectFile {
  const o = p as ProjectFile;
  return !!o && o.format === 'neuroblocks-project' && typeof o.workspace === 'object' && !!o.workspace;
}

export function validateBrain(b: unknown): b is BrainSnapshot {
  const o = b as BrainSnapshot;
  return (
    !!o &&
    o.format === 'neuroblocks-brain' &&
    !!o.net &&
    Array.isArray(o.net.layers) &&
    o.net.layers.every((l) => Array.isArray(l.W) && Array.isArray(l.b) && l.W.length === l.inSize * l.outSize && l.b.length === l.outSize) &&
    !!o.space &&
    Array.isArray(o.space.branches)
  );
}

export function missionsDone(): Set<string> {
  try {
    return new Set(JSON.parse(localStorage.getItem(DONE_KEY) ?? '[]'));
  } catch {
    return new Set();
  }
}

export function markMissionDone(id: string): void {
  try {
    const s = missionsDone();
    s.add(id);
    localStorage.setItem(DONE_KEY, JSON.stringify([...s]));
  } catch {
    /* ignore */
  }
}

export function welcomeSeen(): boolean {
  try {
    return localStorage.getItem(SEEN_KEY) === '1';
  } catch {
    return false;
  }
}

export function setWelcomeSeen(): void {
  try {
    localStorage.setItem(SEEN_KEY, '1');
  } catch {
    /* ignore */
  }
}
