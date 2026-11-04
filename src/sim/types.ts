/**
 * Shared types for the world, the simulation and the agent spec that the
 * block editor compiles to.
 */
import type { LayerSpec } from '../ml/nn.ts';
import type { OptimizerKind } from '../ml/optim.ts';

/** Everything in the world has a tag: it's what sensors "see" and events react to. */
export type StaticTag = 'ground' | 'wall' | 'lava' | 'ramp' | 'tree' | 'rock' | 'ice' | 'bouncy' | 'spike';
export type ObjType = 'coin' | 'flag' | 'trophy' | 'ball' | 'crate' | 'goal' | 'pad' | 'checkpoint' | 'runner' | 'chaser';
export type Tag = StaticTag | ObjType;

export const OBJ_TYPES: ObjType[] = ['coin', 'flag', 'trophy', 'ball', 'crate', 'goal', 'pad', 'checkpoint', 'runner', 'chaser'];
export const STATIC_TAGS: StaticTag[] = ['ground', 'wall', 'lava', 'ramp', 'tree', 'rock', 'ice', 'bouncy', 'spike'];

export const TAG_INFO: Record<Tag, { label: string; emoji: string; color: string }> = {
  ground: { label: 'ground', emoji: '🟩', color: '#6fbf4b' },
  wall: { label: 'wall', emoji: '🧱', color: '#b9a48a' },
  lava: { label: 'lava', emoji: '🔥', color: '#ff5a1f' },
  ramp: { label: 'ramp', emoji: '📐', color: '#c9b28f' },
  tree: { label: 'tree', emoji: '🌲', color: '#2f7d32' },
  rock: { label: 'rock', emoji: '🪨', color: '#8d8d99' },
  ice: { label: 'ice', emoji: '🧊', color: '#a8e6ff' },
  bouncy: { label: 'bouncy pad', emoji: '🟣', color: '#c264ff' },
  spike: { label: 'spikes', emoji: '📍', color: '#d33' },
  coin: { label: 'coin', emoji: '🪙', color: '#ffcc33' },
  flag: { label: 'flag', emoji: '🚩', color: '#ff4466' },
  trophy: { label: 'trophy', emoji: '🏆', color: '#ffd700' },
  ball: { label: 'ball', emoji: '⚽', color: '#ffffff' },
  crate: { label: 'crate', emoji: '📦', color: '#c8893f' },
  goal: { label: 'goal', emoji: '🥅', color: '#66ddff' },
  pad: { label: 'pad', emoji: '🎯', color: '#33dd88' },
  checkpoint: { label: 'checkpoint', emoji: '🏁', color: '#a77bff' },
  runner: { label: 'runner bot', emoji: '🐰', color: '#ff8fd1' },
  chaser: { label: 'chaser bot', emoji: '👾', color: '#ff3b3b' },
};

export type Material = 'grass' | 'stone' | 'wood' | 'metal' | 'lava' | 'brick' | 'sand' | 'ice' | 'rubber' | 'hedge' | 'dirt' | 'concrete' | 'glass' | 'track';

/** Static solid box (may be rotated: ramps, angled walls). */
export interface StaticBox {
  kind: 'box';
  c: [number, number, number];
  h: [number, number, number];
  /** rotation, local → world, row-major 3×3 */
  R: number[];
  yaw: number;
  pitch: number;
  tag: StaticTag;
  mat: Material;
  color: number;
  zone: string;
  /** gates are removed in free-roam so zones connect */
  gate?: boolean;
  /** render-only flag (e.g. hidden ground slab parts) */
  invisible?: boolean;
}

/** Static vertical cylinder (trees, pillars, rocks). */
export interface StaticCyl {
  kind: 'cyl';
  c: [number, number]; // x, z
  y0: number;
  y1: number;
  r: number;
  tag: StaticTag;
  mat: Material;
  color: number;
  zone: string;
}

export type StaticDef = StaticBox | StaticCyl;

export interface ObjDef {
  type: ObjType;
  x: number;
  y: number;
  z: number;
  yaw?: number;
  zone: string;
  /** checkpoint order, pad/crate pairing index */
  index?: number;
  /** half-size for box-shaped triggers (goal, pad, checkpoint) */
  hx?: number;
  hy?: number;
  hz?: number;
  /** placed by the user in build mode */
  custom?: boolean;
}

/** A wall, ramp or floor patch placed in build mode or by a program. */
export interface CustomWall {
  x: number;
  z: number;
  w: number;
  d: number;
  h: number;
  yaw: number;
  tag: 'wall' | 'lava' | 'ice' | 'bouncy' | 'ramp' | 'spike';
}

export interface Rect {
  minX: number;
  maxX: number;
  minZ: number;
  maxZ: number;
}

export interface ZoneDef {
  id: string;
  name: string;
  emoji: string;
  blurb: string;
  rect: Rect;
  /** where agents spawn by default */
  spawn: { x: number; z: number; yaw: number; y?: number };
  /** random spawn/respawn area for objects & agent */
  area: Rect;
  /** optional separate spawn area for the agent */
  spawnArea?: Rect;
  floorY: number;
  color: string;
  needsJump?: boolean;
  /** checkpoints form a loop (race track): after the last one, start over and count a lap */
  loopCheckpoints?: boolean;
  /** suggested mission ids */
  missions: string[];
}

export type DecorKind = 'flower' | 'grass' | 'lamp' | 'sign' | 'bush' | 'mountain' | 'cloud' | 'path' | 'tuft' | 'barrel' | 'cone' | 'banner' | 'net' | 'shelf';

export interface DecorDef {
  kind: DecorKind;
  x: number;
  y: number;
  z: number;
  s?: number; // scale
  yaw?: number;
  color?: number;
  text?: string;
  w?: number;
  d?: number;
  h?: number;
}

export interface WorldDef {
  statics: StaticDef[];
  objects: ObjDef[];
  zones: ZoneDef[];
  decor: DecorDef[];
  bounds: Rect;
  waterY: number;
}

// ---------------------------------------------------------------------------
// The agent spec: what the block program compiles into.
// ---------------------------------------------------------------------------

export type SensorSpec =
  | { kind: 'rays'; count: number; spread: number; range: number; tilt: number; detect: Tag[] }
  | { kind: 'target'; tag: string; range: number }
  | { kind: 'pair'; a: string; b: string; range: number }
  | { kind: 'velocity' }
  | { kind: 'heading' }
  | { kind: 'position' }
  | { kind: 'grounded' }
  | { kind: 'height' }
  | { kind: 'ground_probe'; count: number; dist: number }
  | { kind: 'time' }
  | { kind: 'last_action' }
  | { kind: 'custom'; name: string; index: number }
  /** frame stacking (layout only: the previous observations, `lag` steps ago) */
  | { kind: 'memory'; frames: number; lag: number };

export type AxisKind = 'move' | 'turn' | 'strafe' | 'jump' | 'kick';

export interface ActionAxis {
  axis: AxisKind;
  /** discrete option values, e.g. [-1, 0, 1] */
  options: number[];
  labels: string[];
}

export type AlgoKind = 'ppo' | 'dqn' | 'reinforce' | 'evolution' | 'custom';

export interface TrainSpec {
  algo: AlgoKind;
  numEnvs: number;
  seed: number;
  optimizer: OptimizerKind;
  lr: number;
  gamma: number;
  // PPO
  lambda: number;
  clip: number;
  entropy: number;
  epochs: number;
  minibatch: number;
  horizon: number;
  valueCoef: number;
  maxGradNorm: number;
  normObs: boolean;
  normReward: boolean;
  // DQN
  bufferSize: number;
  batchSize: number;
  epsStart: number;
  epsEnd: number;
  epsDecay: number;
  targetUpdate: number;
  learnStart: number;
  trainEvery: number;
  double: boolean;
  // REINFORCE
  episodesPerUpdate: number;
  baseline: boolean;
  // Evolution
  population: number;
  eliteFrac: number;
  mutation: number;
  mutationDecay: number;
  episodesPerEval: number;
  // schedules: the learning rate (and PPO/REINFORCE entropy bonus) can change during training
  lrSchedule: 'constant' | 'linear' | 'cosine';
  /** learning rate at the end of the schedule, as a fraction of the starting value */
  lrEnd: number;
  /** entropy bonus at the end of the schedule (absolute); NaN = keep it constant */
  entropyEnd: number;
  /** steps over which schedules run (0 = use the stop limit, or 1M steps) */
  scheduleSteps: number;
  // general
  stopSteps: number;
  /** stop when the success rate over the last 100 episodes reaches this (0 = off) */
  targetSuccess: number;
  /** stop when the mean reward over the last 100 episodes reaches this (NaN = off) */
  targetReward: number;
  /** stop after this many minutes of training (0 = off) */
  stopMinutes: number;
}

/** World physics that a program can change (per training run). */
export interface PhysicsSpec {
  /** downward acceleration in m/s² (Earth ≈ 9.8; the default is snappier) */
  gravity: number;
  /** how quickly the robot slows down on the ground */
  friction: number;
  /** how quickly the robot slows down in the air */
  airDrag: number;
  /** multiplier for the runner/chaser bots' speed */
  npcSpeed: number;
  /** multiplier for the kick strength */
  kickPower: number;
  /** mass of balls and crates (kg) */
  ballMass: number;
  crateMass: number;
  /** physics ticks (1/60 s) per brain decision: 3 = the brain decides 20× per second */
  decisionTicks: number;
  /** upward speed given by bouncy pads */
  bounce: number;
}

export const DEFAULT_PHYSICS: PhysicsSpec = {
  gravity: 22,
  friction: 3.2,
  airDrag: 0.25,
  npcSpeed: 1,
  kickPower: 1,
  ballMass: 0.45,
  crateMass: 2.5,
  decisionTicks: 3,
  bounce: 12,
};

/** Bounds used by the compiler and code mode to keep physics sane. */
export const PHYSICS_LIMITS: Record<keyof PhysicsSpec, [number, number]> = {
  gravity: [0, 60],
  friction: [0, 20],
  airDrag: [0, 10],
  npcSpeed: [0, 3],
  kickPower: [0, 4],
  ballMass: [0.05, 20],
  crateMass: [0.2, 50],
  decisionTicks: [1, 12],
  bounce: [0, 30],
};

export interface BodySpec {
  speed: number; // multiplier
  turn: number; // multiplier
  jump: number; // jump strength multiplier
  color: string;
  name: string;
}

export interface SpawnSpec {
  /** zone = the zone's start spot, random = anywhere in the zone's spawn area, point = custom x/z */
  mode: 'zone' | 'random' | 'point';
  x: number;
  z: number;
  /** default = the zone's usual heading, random = any direction, fixed = `yaw` (radians) */
  yawMode: 'default' | 'random' | 'fixed';
  yaw: number;
  /** random offset radius in metres */
  jitter: number;
}

export interface AgentSpec {
  name: string;
  zone: string;
  spawn: SpawnSpec;
  body: BodySpec;
  sensors: SensorSpec[];
  actionMode: 'discrete' | 'continuous';
  axes: ActionAxis[];
  brain: { layers: LayerSpec[] };
  train: TrainSpec;
  maxSteps: number;
  /** generated JavaScript: source of a factory `function(api){ ... return S; }` */
  script: string;
  customSenses: string[];
  /** objects/walls placed in build mode */
  customObjects: ObjDef[];
  warnings: string[];
  /** physics overrides (missing keys use DEFAULT_PHYSICS) */
  physics?: Partial<PhysicsSpec>;
  /** feed the last N observations to the brain (1 = only the current one) */
  frameStack?: number;
  /** where the program came from */
  source?: 'blocks' | 'code';
  /** code mode: lines of the generated factory before the user's first line (for error line numbers) */
  codeLineOffset?: number;
  /** world changes made by the program itself (code mode `world.add(...)`, or "clear the zone") */
  world?: { objects: ObjDef[]; walls: CustomWall[]; clearZones: string[] };
}

export function defaultTrain(algo: AlgoKind = 'ppo'): TrainSpec {
  return {
    algo,
    numEnvs: 16,
    seed: 1,
    optimizer: 'adam',
    lr: algo === 'dqn' ? 5e-4 : algo === 'reinforce' ? 1e-3 : 3e-4,
    gamma: 0.99,
    lambda: 0.95,
    clip: 0.2,
    entropy: 0.01,
    epochs: 4,
    minibatch: 256,
    horizon: 128,
    valueCoef: 0.5,
    maxGradNorm: 0.5,
    normObs: true,
    normReward: true,
    bufferSize: 50000,
    batchSize: 64,
    epsStart: 1,
    epsEnd: 0.05,
    epsDecay: 60000,
    targetUpdate: 4000,
    learnStart: 2000,
    trainEvery: 4,
    double: true,
    episodesPerUpdate: 16,
    baseline: true,
    population: 48,
    eliteFrac: 0.2,
    mutation: 0.1,
    mutationDecay: 0.995,
    episodesPerEval: 1,
    lrSchedule: 'constant',
    lrEnd: 0.1,
    entropyEnd: NaN,
    scheduleSteps: 0,
    stopSteps: 0,
    targetSuccess: 0,
    targetReward: NaN,
    stopMinutes: 0,
  };
}
