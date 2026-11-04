/**
 * The block compiler: reads the block program and produces an AgentSpec
 * (configuration + generated JavaScript for the event scripts).
 *
 *  🤖 robot / 👁️ senses / 🕹️ actions / 🧠 brain / 🏋️ training hats → configuration
 *  ⚡ event hats (with reward blocks inside)                       → JavaScript
 */
import type * as Blockly from 'blockly/core';
import type { ActivationName } from './types.ts';
import { chainCode, gen, installGenerators } from './generators.ts';
import { AXIS_PRESETS, sensorCost } from '../sim/spec.ts';
import type { ActionAxis, AgentSpec, AlgoKind, PhysicsSpec, SensorSpec, Tag, TrainSpec } from '../sim/types.ts';
import { DEFAULT_PHYSICS, PHYSICS_LIMITS, defaultTrain } from '../sim/types.ts';
import { getWorld } from '../sim/worldCache.ts';
import { ACTIVATIONS } from '../ml/nn.ts';
import type { ObjDef } from '../sim/types.ts';

export interface CompileResult {
  spec: AgentSpec;
  errors: string[];
  warnings: string[];
  obsSize: number;
  outSize: number;
  params: number;
  /** code mode: problems with positions in the source */
  diagnostics?: import('../code/runtime.ts').CodeDiagnostic[];
  /** blocks mode: the event scripts, for converting blocks to code */
  events?: BlockEvent[];
  /** blocks mode: top-level variable declarations of the generated code */
  defs?: string;
}

export interface BlockEvent {
  kind: 'start' | 'step' | 'touch' | 'pair' | 'fall' | 'end' | 'sense';
  tag?: string;
  a?: string;
  b?: string;
  /** sense name (kind 'sense') */
  name?: string;
  body: string;
}

const SECTION_TYPES = ['nb_robot', 'nb_senses', 'nb_actions', 'nb_brain', 'nb_world'];
const TRAIN_TYPES = ['nb_train_ppo', 'nb_train_dqn', 'nb_train_reinforce', 'nb_train_evolution'];
const EVENT_TYPES = ['nb_on_start', 'nb_on_step', 'nb_on_touch', 'nb_on_pair', 'nb_on_fall', 'nb_on_end'];

const n = (b: Blockly.Block, f: string, d: number, lo = -Infinity, hi = Infinity): number => {
  const v = Number(b.getFieldValue(f));
  return Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : d;
};
const s = (b: Blockly.Block, f: string, d = ''): string => String(b.getFieldValue(f) ?? d);
const tag = (b: Blockly.Block, f: string): string => s(b, f).replace(/[^a-z_]/g, '');

function chain(first: Blockly.Block | null): Blockly.Block[] {
  const out: Blockly.Block[] = [];
  for (let b = first; b; b = b.getNextBlock()) if (b.isEnabled()) out.push(b);
  return out;
}

function indent(code: string, pad = '  '): string {
  return code
    .split('\n')
    .map((l) => (l.length ? pad + l : l))
    .join('\n');
}

export function compileWorkspace(ws: Blockly.Workspace, customObjects: ObjDef[] = []): CompileResult {
  installGenerators();
  const errors: string[] = [];
  const warnings: string[] = [];
  const tops = ws.getTopBlocks(true).filter((b) => b.isEnabled());
  const first = (type: string) => {
    const all = tops.filter((b) => b.type === type);
    if (all.length > 1) warnings.push(`There are ${all.length} "${type.replace('nb_', '')}" blocks: only the top one is used.`);
    return all[0] ?? null;
  };
  const world = getWorld();

  // ---- robot ----
  const robot = first('nb_robot');
  const spec: AgentSpec = {
    name: 'Robo',
    zone: 'meadow',
    spawn: { mode: 'zone', x: 0, z: 0, yawMode: 'default', yaw: 0, jitter: 1.5 },
    body: { speed: 1, turn: 1, jump: 1, color: '#4fc3f7', name: 'Robo' },
    sensors: [],
    actionMode: 'discrete',
    axes: [],
    brain: { layers: [] },
    train: defaultTrain('ppo'),
    maxSteps: 400,
    script: '',
    customSenses: [],
    customObjects,
    warnings,
    source: 'blocks',
  };
  let clearZone = false;
  if (!robot) warnings.push('No 🤖 robot block: using the Meadow with default settings.');
  else {
    spec.name = s(robot, 'NAME', 'Robo').slice(0, 40) || 'Robo';
    spec.body.name = spec.name;
    const zone = s(robot, 'ZONE', 'meadow');
    spec.zone = world.zones.some((z) => z.id === zone) ? zone : 'meadow';
    const mode = s(robot, 'SPAWN', 'zone');
    spec.spawn.mode = mode === 'random' || mode === 'point' ? mode : 'zone';
    const facing = s(robot, 'FACING', 'default');
    spec.spawn.yawMode = facing === 'random' || facing === 'fixed' ? facing : 'default';
    spec.spawn.x = n(robot, 'X', 0, -500, 500);
    spec.spawn.z = n(robot, 'Z', 0, -500, 500);
    // block angle: 0° = facing north (−z), clockwise positive (like a compass)
    spec.spawn.yaw = Math.PI - (n(robot, 'ANGLE', 0, -360, 360) * Math.PI) / 180;
    spec.spawn.jitter = n(robot, 'JITTER', 1.5, 0, 60);
    spec.maxSteps = Math.round(n(robot, 'STEPS', 400, 20, 5000));
    for (const t of chain(robot.getNextBlock())) {
      if (t.type === 'nb_trait_speed') spec.body.speed = n(t, 'V', 1, 0.2, 3);
      if (t.type === 'nb_trait_turn') spec.body.turn = n(t, 'V', 1, 0.2, 3);
      if (t.type === 'nb_trait_jump') spec.body.jump = n(t, 'V', 1, 0.3, 2);
      if (t.type === 'nb_trait_color') spec.body.color = /^#[0-9a-f]{6}$/i.test(s(t, 'C')) ? s(t, 'C') : '#4fc3f7';
      if (t.type === 'nb_clear_zone') clearZone = true;
    }
    if (spec.spawn.mode === 'point') {
      const r = world.bounds;
      if (spec.spawn.x < r.minX || spec.spawn.x > r.maxX || spec.spawn.z < r.minZ || spec.spawn.z > r.maxZ) warnings.push('The spawn point is outside the island.');
    }
  }

  if (clearZone) {
    if (spec.zone === 'world') warnings.push('🧹 "remove the built-in objects" works on a single zone, not the whole island.');
    else spec.world = { objects: [], walls: [], clearZones: [spec.zone] };
  }

  // ---- world rules (physics) ----
  const rules = first('nb_world');
  if (rules) {
    const phys: Partial<PhysicsSpec> = {};
    for (const b of chain(rules.getNextBlock())) {
      if (b.type !== 'nb_phys') continue;
      const key = s(b, 'KEY') as keyof PhysicsSpec;
      if (!(key in DEFAULT_PHYSICS)) continue;
      const [lo, hi] = PHYSICS_LIMITS[key];
      phys[key] = n(b, 'V', DEFAULT_PHYSICS[key], lo, hi);
      if (key === 'decisionTicks') phys[key] = Math.round(phys[key]!);
    }
    if (Object.keys(phys).length) spec.physics = phys;
  }

  // ---- senses ----
  const senses = first('nb_senses');
  const customSenseBlocks: Blockly.Block[] = [];
  if (senses) {
    for (const b of chain(senses.getNextBlock())) {
      let sp: SensorSpec | null = null;
      switch (b.type) {
        case 'nb_sense_rays': {
          const detect: Tag[] = [];
          for (const d of chain(b.getInputTargetBlock('DETECT'))) {
            const t = tag(d, 'TAG') as Tag;
            if (t && !detect.includes(t)) detect.push(t);
          }
          sp = { kind: 'rays', count: Math.round(n(b, 'COUNT', 7, 1, 31)), spread: n(b, 'SPREAD', 120, 0, 360), range: n(b, 'RANGE', 15, 1, 60), tilt: n(b, 'TILT', 0, -10, 80), detect };
          break;
        }
        case 'nb_sense_target':
          sp = { kind: 'target', tag: tag(b, 'TAG'), range: n(b, 'RANGE', 40, 1, 300) };
          break;
        case 'nb_sense_pair':
          sp = { kind: 'pair', a: tag(b, 'A'), b: tag(b, 'B'), range: 40 };
          break;
        case 'nb_sense_velocity':
          sp = { kind: 'velocity' };
          break;
        case 'nb_sense_heading':
          sp = { kind: 'heading' };
          break;
        case 'nb_sense_position':
          sp = { kind: 'position' };
          break;
        case 'nb_sense_grounded':
          sp = { kind: 'grounded' };
          break;
        case 'nb_sense_height':
          sp = { kind: 'height' };
          break;
        case 'nb_sense_probe':
          sp = { kind: 'ground_probe', count: Math.round(n(b, 'COUNT', 3, 1, 9)), dist: n(b, 'DIST', 1.5, 0.5, 6) };
          break;
        case 'nb_sense_time':
          sp = { kind: 'time' };
          break;
        case 'nb_sense_last':
          sp = { kind: 'last_action' };
          break;
        case 'nb_sense_memory':
          spec.frameStack = Math.round(n(b, 'K', 3, 1, 8));
          break;
        case 'nb_sense_custom': {
          const name = s(b, 'NAME', 'sense').slice(0, 30) || 'sense';
          sp = { kind: 'custom', name, index: customSenseBlocks.length };
          customSenseBlocks.push(b);
          spec.customSenses.push(name);
          break;
        }
      }
      if (sp) spec.sensors.push(sp);
    }
  }
  if (!spec.sensors.length) errors.push('Your robot has no senses. Add blocks under 👁️ "my robot can sense": without inputs the brain is blind.');

  // ---- actions ----
  const actions = first('nb_actions');
  if (actions) {
    spec.actionMode = s(actions, 'MODE') === 'continuous' ? 'continuous' : 'discrete';
    const seen = new Set<string>();
    for (const b of chain(actions.getNextBlock())) {
      let ax: ActionAxis | null = null;
      if (b.type === 'nb_act_move') ax = s(b, 'DIR') === 'one' ? AXIS_PRESETS.move.one! : AXIS_PRESETS.move.both;
      if (b.type === 'nb_act_turn') ax = AXIS_PRESETS.turn.both;
      if (b.type === 'nb_act_strafe') ax = AXIS_PRESETS.strafe.both;
      if (b.type === 'nb_act_jump') ax = AXIS_PRESETS.jump.both;
      if (b.type === 'nb_act_kick') ax = AXIS_PRESETS.kick.both;
      if (!ax) continue;
      if (seen.has(ax.axis)) {
        warnings.push(`The "${ax.axis}" action is listed twice.`);
        continue;
      }
      seen.add(ax.axis);
      spec.axes.push(ax);
    }
  }
  if (!spec.axes.length) errors.push('Your robot cannot do anything. Add blocks under 🕹️ "my robot can".');
  const zoneDef = world.zones.find((z) => z.id === spec.zone);
  if (zoneDef?.needsJump && !spec.axes.some((a) => a.axis === 'jump')) warnings.push(`${zoneDef.emoji} ${zoneDef.name} has gaps: your robot probably needs the 🦘 jump action.`);

  // ---- brain ----
  const brain = first('nb_brain');
  if (brain) {
    for (const b of chain(brain.getNextBlock())) {
      if (b.type !== 'nb_layer') continue;
      const act = s(b, 'ACT', 'relu') as ActivationName;
      spec.brain.layers.push({ units: Math.round(n(b, 'UNITS', 64, 1, 512)), activation: ACTIVATIONS.includes(act) ? act : 'relu' });
    }
  }
  if (!spec.brain.layers.length) {
    warnings.push('No 🧠 brain layers: using one layer of 32 tanh neurons.');
    spec.brain.layers.push({ units: 32, activation: 'tanh' });
  }

  // ---- training ----
  const trainBlocks = tops.filter((b) => TRAIN_TYPES.includes(b.type));
  if (trainBlocks.length > 1) warnings.push('More than one 🏋️ training block: only the top one is used.');
  const tb = trainBlocks[0];
  if (!tb) warnings.push('No 🏋️ training block: using PPO with default settings.');
  else spec.train = readTrain(tb);
  if (spec.train.algo === 'dqn' && spec.actionMode === 'continuous') errors.push('DQN needs discrete actions: set 🕹️ to "pick from choices" or train with PPO.');

  // ---- events → JavaScript ----
  gen.init(ws);
  const parts: string[] = [];
  const events: BlockEvent[] = [];
  let rewardBlocks = 0;
  for (const hat of tops) {
    if (!EVENT_TYPES.includes(hat.type)) continue;
    const body = chainCode(hat);
    for (const d of hat.getDescendants(false)) if ((d.type === 'nb_reward' || d.type === 'nb_penalty') && d.isEnabled()) rewardBlocks++;
    if (!body.trim()) continue;
    const fn = (args: string) => `function (${args}) {\n${indent(body.trimEnd())}\n}`;
    const b = body.trimEnd();
    switch (hat.type) {
      case 'nb_on_start':
        parts.push(`// 🎬 when an episode starts\nS.start.push(${fn('')});`);
        events.push({ kind: 'start', body: b });
        break;
      case 'nb_on_step':
        parts.push(`// 🔁 every step\nS.step.push(${fn('')});`);
        events.push({ kind: 'step', body: b });
        break;
      case 'nb_on_touch':
        parts.push(`// 💥 when I touch ${tag(hat, 'TAG')}\nS.touch.push(['${tag(hat, 'TAG')}', ${fn('obj')}]);`);
        events.push({ kind: 'touch', tag: tag(hat, 'TAG'), body: b });
        break;
      case 'nb_on_pair':
        parts.push(`// 🤝 when ${tag(hat, 'A')} touches ${tag(hat, 'B')}\nS.pair.push(['${tag(hat, 'A')}', '${tag(hat, 'B')}', ${fn('objA, objB')}]);`);
        events.push({ kind: 'pair', a: tag(hat, 'A'), b: tag(hat, 'B'), body: b });
        break;
      case 'nb_on_fall':
        parts.push(`// 🕳️ when I fall off the world\nS.fall.push(${fn('')});`);
        events.push({ kind: 'fall', body: b });
        break;
      case 'nb_on_end':
        parts.push(`// 🏁 when the episode ends\nS.end.push(${fn('')});`);
        events.push({ kind: 'end', body: b });
        break;
    }
  }
  customSenseBlocks.forEach((b, i) => {
    const code = gen.valueToCode(b, 'VALUE', 99) || '0';
    parts.push(`// ✨ custom sense #${i + 1}: ${spec.customSenses[i].replace(/[\r\n*/]/g, ' ')}\nS.senses.push(function () { return Number(${code}); });`);
    events.push({ kind: 'sense', name: spec.customSenses[i], body: code });
  });
  const MARK = '/*@@nb-code@@*/';
  const all = gen.finish(MARK);
  const defs = all.slice(0, all.indexOf(MARK)).trim();
  const finished = `${defs ? defs + '\n\n' : ''}${parts.join('\n\n')}`.replace(/^\s*\n/, '').replace(/\n{3,}/g, '\n\n');
  spec.script = `function (api, Math) {\n  var S = { start: [], step: [], touch: [], pair: [], fall: [], end: [], senses: [] };\n${indent(finished)}\n  return S;\n}`;
  if (!rewardBlocks) warnings.push('There are no 🍬 reward or ⚡ penalty blocks inside event hats. Without rewards the robot has nothing to learn!');
  for (const b of tops) {
    if (b.previousConnection && !SECTION_TYPES.includes(b.type)) {
      warnings.push('Some blocks are not attached to any hat, so they are ignored (they look faded in Scratch too).');
      break;
    }
  }
  try {
    new Function(`return (${spec.script});`);
  } catch (e) {
    errors.push(`Generated code has a syntax error: ${(e as Error).message}`);
  }

  // ---- sizes for the brain label ----
  const obsSize = spec.sensors.reduce((a, sp) => a + sensorCost(sp, spec.axes.length), 0) * (spec.frameStack ?? 1);
  const outSize = spec.train.algo === 'dqn' ? spec.axes.reduce((a, x) => a * x.options.length, 1) : spec.actionMode === 'discrete' ? spec.axes.reduce((a, x) => a + x.options.length, 0) : spec.axes.length;
  let params = 0;
  let prev = obsSize;
  for (const l of spec.brain.layers) {
    params += prev * l.units + l.units;
    prev = l.units;
  }
  params += prev * outSize + outSize;
  return { spec, errors, warnings, obsSize, outSize, params, events, defs };
}

function readTrain(b: Blockly.Block): TrainSpec {
  const algo: AlgoKind = b.type === 'nb_train_dqn' ? 'dqn' : b.type === 'nb_train_reinforce' ? 'reinforce' : b.type === 'nb_train_evolution' ? 'evolution' : 'ppo';
  const t = defaultTrain(algo);
  const opt = s(b, 'OPT', 'adam');
  if (['adam', 'sgd', 'momentum', 'rmsprop'].includes(opt)) t.optimizer = opt as TrainSpec['optimizer'];
  t.numEnvs = Math.round(n(b, 'WORLDS', 16, 1, 64));
  t.stopSteps = Math.round(n(b, 'STOP', 0, 0, 1e9));
  t.seed = Math.round(n(b, 'SEED', 1, 0, 99999));
  if (algo === 'ppo') {
    t.lr = n(b, 'LR', 3e-4, 1e-6, 1);
    t.gamma = n(b, 'GAMMA', 0.99, 0, 1);
    t.lambda = n(b, 'LAMBDA', 0.95, 0, 1);
    t.clip = n(b, 'CLIP', 0.2, 0.01, 1);
    t.entropy = n(b, 'ENT', 0.01, 0, 1);
    t.horizon = Math.round(n(b, 'HORIZON', 128, 8, 4096));
    t.epochs = Math.round(n(b, 'EPOCHS', 4, 1, 50));
    t.minibatch = Math.round(n(b, 'MB', 256, 16, 8192));
    t.normObs = b.getFieldValue('NORMOBS') !== 'FALSE';
    t.normReward = b.getFieldValue('NORMREW') !== 'FALSE';
  } else if (algo === 'dqn') {
    t.lr = n(b, 'LR', 5e-4, 1e-6, 1);
    t.gamma = n(b, 'GAMMA', 0.99, 0, 1);
    t.bufferSize = Math.round(n(b, 'BUFFER', 50000, 1000, 1e6));
    t.batchSize = Math.round(n(b, 'BATCH', 64, 8, 1024));
    t.epsStart = n(b, 'EPS0', 1, 0, 1);
    t.epsEnd = n(b, 'EPS1', 0.05, 0, 1);
    t.epsDecay = Math.round(n(b, 'EPSDECAY', 60000, 1, 1e8));
    t.targetUpdate = Math.round(n(b, 'TARGET', 4000, 1, 1e7));
    t.learnStart = Math.round(n(b, 'LEARNSTART', 2000, 0, 1e7));
    t.trainEvery = Math.round(n(b, 'TRAINEVERY', 4, 1, 1000));
    t.double = b.getFieldValue('DOUBLE') !== 'FALSE';
  } else if (algo === 'reinforce') {
    t.lr = n(b, 'LR', 1e-3, 1e-6, 1);
    t.gamma = n(b, 'GAMMA', 0.99, 0, 1);
    t.episodesPerUpdate = Math.round(n(b, 'EPISODES', 16, 1, 1000));
    t.baseline = b.getFieldValue('BASELINE') !== 'FALSE';
    t.entropy = n(b, 'ENT', 0.01, 0, 1);
    t.normObs = b.getFieldValue('NORMOBS') !== 'FALSE';
  } else {
    t.population = Math.round(n(b, 'POP', 48, 4, 1000));
    t.eliteFrac = n(b, 'ELITE', 0.2, 0.01, 1);
    t.mutation = n(b, 'SIGMA', 0.1, 0.001, 5);
    t.mutationDecay = n(b, 'DECAY', 0.995, 0.5, 1);
    t.episodesPerEval = Math.round(n(b, 'EVALS', 1, 1, 20));
  }
  for (const o of chain(b.getInputTargetBlock('MORE'))) {
    if (o.type === 'nb_train_schedule') {
      const mode = s(o, 'MODE', 'linear');
      t.lrSchedule = mode === 'cosine' || mode === 'constant' ? mode : 'linear';
      t.lrEnd = n(o, 'END', 10, 0, 100) / 100;
    } else if (o.type === 'nb_train_stop') {
      const what = s(o, 'WHAT', 'success');
      if (what === 'success') t.targetSuccess = n(o, 'V', 95, 0, 100) / 100;
      else if (what === 'reward') t.targetReward = n(o, 'V', 10, -1e9, 1e9);
      else if (what === 'minutes') t.stopMinutes = n(o, 'V', 5, 0, 1e5);
    } else if (o.type === 'nb_train_opt') {
      const key = s(o, 'KEY');
      if (key === 'entropyEnd') t.entropyEnd = n(o, 'V', 0.001, 0, 1);
      else if (key === 'valueCoef') t.valueCoef = n(o, 'V', 0.5, 0, 10);
      else if (key === 'maxGradNorm') t.maxGradNorm = n(o, 'V', 0.5, 0.01, 100);
      else if (key === 'scheduleSteps') t.scheduleSteps = Math.round(n(o, 'V', 1e6, 0, 1e10));
    }
  }
  return t;
}
