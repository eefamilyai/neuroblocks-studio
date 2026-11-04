/** Helpers for building AgentSpecs (used by the compiler, tests and templates). */
import type { ActionAxis, AgentSpec, AxisKind, SensorSpec, TrainSpec } from './types.ts';
import { defaultTrain } from './types.ts';
import type { ActionSpace } from '../ml/policy.ts';

export const AXIS_PRESETS: Record<AxisKind, { both: ActionAxis; one?: ActionAxis }> = {
  move: {
    both: { axis: 'move', options: [-1, 0, 1], labels: ['back', 'stop', 'forward'] },
    one: { axis: 'move', options: [0, 1], labels: ['stop', 'forward'] },
  },
  turn: { both: { axis: 'turn', options: [-1, 0, 1], labels: ['left', 'straight', 'right'] } },
  strafe: { both: { axis: 'strafe', options: [-1, 0, 1], labels: ['slide left', 'none', 'slide right'] } },
  jump: { both: { axis: 'jump', options: [0, 1], labels: ['stay', 'jump'] } },
  kick: { both: { axis: 'kick', options: [0, 1], labels: ['no kick', 'kick'] } },
};

export function makeSpec(over: Partial<AgentSpec> & { train?: Partial<TrainSpec> } = {}): AgentSpec {
  const train = { ...defaultTrain(over.train?.algo ?? 'ppo'), ...(over.train ?? {}) };
  const base: AgentSpec = {
    name: 'Robo',
    zone: 'meadow',
    spawn: { mode: 'zone', x: 0, z: 0, yawMode: 'default', yaw: 0, jitter: 2 },
    body: { speed: 1, turn: 1, jump: 1, color: '#4fc3f7', name: 'Robo' },
    sensors: [{ kind: 'target', tag: 'flag', range: 40 }, { kind: 'velocity' }],
    actionMode: 'discrete',
    axes: [AXIS_PRESETS.move.both, AXIS_PRESETS.turn.both],
    brain: { layers: [{ units: 64, activation: 'tanh' }, { units: 64, activation: 'tanh' }] },
    train,
    maxSteps: 400,
    script: 'function(api, Math){ return {start:[],step:[],touch:[],pair:[],fall:[],end:[],senses:[]}; }',
    customSenses: [],
    customObjects: [],
    warnings: [],
  };
  return { ...base, ...over, train } as AgentSpec;
}

export function actionSpaceOf(spec: AgentSpec): ActionSpace {
  if (spec.actionMode === 'continuous') {
    return {
      kind: 'continuous',
      branches: spec.axes.map(() => 1),
      labels: spec.axes.map((a) => a.axis),
      optionLabels: spec.axes.map((a) => [a.axis]),
    };
  }
  return {
    kind: 'discrete',
    branches: spec.axes.map((a) => a.options.length),
    labels: spec.axes.map((a) => a.axis),
    optionLabels: spec.axes.map((a) => a.labels),
  };
}

export function sensorCost(s: SensorSpec, axes: number): number {
  switch (s.kind) {
    case 'rays':
      return s.count * (1 + s.detect.length);
    case 'target':
    case 'pair':
    case 'velocity':
      return 3;
    case 'heading':
    case 'position':
      return 2;
    case 'ground_probe':
      return s.count * 2;
    case 'last_action':
      return axes;
    default:
      return 1;
  }
}
