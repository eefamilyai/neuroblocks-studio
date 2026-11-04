/**
 * Code generators: turn event & reward blocks into JavaScript that calls the
 * ScriptAPI. Field values are sanitised so a crafted project file can never
 * inject code (only [a-z_] tags and finite numbers get through).
 */
import type * as Blockly from 'blockly/core';
import { javascriptGenerator, Order } from 'blockly/javascript';

export const gen = javascriptGenerator;

const safeTag = (v: unknown): string => String(v ?? '').replace(/[^a-z_]/g, '');
const safeNum = (v: unknown, d = 0): string => {
  const n = Number(v);
  return Number.isFinite(n) ? String(n) : String(d);
};

/** What hat is this block under? (to resolve "the touched thing") */
function hatOf(b: Blockly.Block): Blockly.Block | null {
  return b.getRootBlock();
}

function touchedRef(b: Blockly.Block): string {
  const h = hatOf(b);
  if (!h) return 'null';
  if (h.type === 'nb_on_touch') return 'obj';
  if (h.type === 'nb_on_pair') return 'objA';
  return 'null';
}

/** What a measure block is about, in a few words (for reward labels). */
function describe(b: Blockly.Block): string | null {
  const t = (n = 'TAG') => safeTag(b.getFieldValue(n)).replace('next_checkpoint', 'checkpoint');
  switch (b.type) {
    case 'nb_closer':
      return `closer to ${t()}`;
    case 'nb_distance':
      return `distance to ${t()}`;
    case 'nb_angle':
    case 'nb_facing':
      return `facing ${t()}`;
    case 'nb_pair_closer':
    case 'nb_pair_distance':
      return `${t('A')} to ${t('B')}`;
    case 'nb_speed':
      return 'speed';
    case 'nb_fwd_speed':
      return 'forward speed';
    case 'nb_height':
      return 'height';
    case 'nb_height_gain':
      return 'climbing';
    case 'nb_new_area':
    case 'nb_areas':
      return 'exploring';
    case 'nb_time_left':
      return 'time left';
    case 'nb_count':
      return `${t()} count`;
    case 'nb_touching':
      return `touching ${t()}`;
    default:
      return null;
  }
}

/**
 * A short label for a reward block, used in the "where does the reward come
 * from" chart: what the value measures, else which event it is in.
 */
function rewardLabel(b: Blockly.Block): string {
  const v = b.getInputTargetBlock('VALUE');
  if (v) for (const d of [v, ...v.getDescendants(true)]) {
    const s = describe(d);
    if (s) return s;
  }
  for (let p = b.getSurroundParent(); p; p = p.getSurroundParent()) {
    if (p.type !== 'controls_if') continue;
    const cond = p.getInputTargetBlock('IF0');
    if (cond) for (const d of [cond, ...cond.getDescendants(true)]) {
      const s = describe(d);
      if (s) return `when ${s}`;
    }
  }
  const h = hatOf(b);
  const tg = (n = 'TAG') => safeTag(h?.getFieldValue(n));
  switch (h?.type) {
    case 'nb_on_touch':
      return `touch ${tg()}`;
    case 'nb_on_pair':
      return `${tg('A')} touches ${tg('B')}`;
    case 'nb_on_step':
      return 'every step';
    case 'nb_on_fall':
      return 'falling';
    case 'nb_on_end':
      return 'episode end';
    case 'nb_on_start':
      return 'start';
    default:
      return 'reward';
  }
}

let installed = false;
export function installGenerators(): void {
  if (installed) return;
  installed = true;
  gen.addReservedWords('api,S,obj,objA,objB,Math');
  const f = gen.forBlock as Record<string, (b: Blockly.Block, g: typeof gen) => string | [string, number] | null>;

  // section hats produce no code (they are read by the compiler)
  for (const t of [
    'nb_robot',
    'nb_trait_speed',
    'nb_trait_turn',
    'nb_trait_jump',
    'nb_trait_color',
    'nb_clear_zone',
    'nb_world',
    'nb_phys',
    'nb_sense_memory',
    'nb_train_schedule',
    'nb_train_stop',
    'nb_train_opt',
    'nb_senses',
    'nb_sense_rays',
    'nb_detect',
    'nb_sense_target',
    'nb_sense_pair',
    'nb_sense_velocity',
    'nb_sense_heading',
    'nb_sense_position',
    'nb_sense_grounded',
    'nb_sense_height',
    'nb_sense_probe',
    'nb_sense_time',
    'nb_sense_last',
    'nb_sense_custom',
    'nb_actions',
    'nb_act_move',
    'nb_act_turn',
    'nb_act_strafe',
    'nb_act_jump',
    'nb_act_kick',
    'nb_brain',
    'nb_layer',
    'nb_train_ppo',
    'nb_train_dqn',
    'nb_train_reinforce',
    'nb_train_evolution',
    'nb_on_start',
    'nb_on_step',
    'nb_on_touch',
    'nb_on_pair',
    'nb_on_fall',
    'nb_on_end',
  ])
    f[t] = () => '';

  const val = (b: Blockly.Block, name: string, order = Order.NONE, d = '0') => gen.valueToCode(b, name, order) || d;

  f.nb_reward = (b) => `api.reward(${val(b, 'VALUE', Order.NONE, '1')}, '${rewardLabel(b)}');\n`;
  f.nb_penalty = (b) => `api.penalty(${val(b, 'VALUE', Order.NONE, '1')}, '${rewardLabel(b)}');\n`;
  f.nb_end = (b) => `api.end(${b.getFieldValue('RESULT') === 'true' ? 'true' : 'false'});\n`;
  f.nb_respawn = (b) => `api.respawn(${touchedRef(b)});\n`;
  f.nb_hide = (b) => `api.hide(${touchedRef(b)});\n`;
  f.nb_scatter = (b) => `api.randomize('${safeTag(b.getFieldValue('TAG'))}');\n`;
  f.nb_reset_all = (b) => `api.resetAll('${safeTag(b.getFieldValue('TAG'))}');\n`;
  f.nb_show_all = (b) => `api.showAll('${safeTag(b.getFieldValue('TAG'))}');\n`;
  f.nb_advance = () => 'api.advance();\n';
  f.nb_respawn_me = () => 'api.respawnMe();\n';
  f.nb_log = (b) => `api.log(${val(b, 'VALUE', Order.NONE, "''")});\n`;

  const T = (b: Blockly.Block, n = 'TAG') => `'${safeTag(b.getFieldValue(n))}'`;
  f.nb_distance = (b) => [`api.dist(${T(b)})`, Order.FUNCTION_CALL];
  f.nb_closer = (b) => [`api.closer(${T(b)})`, Order.FUNCTION_CALL];
  f.nb_angle = (b) => [`api.angle(${T(b)})`, Order.FUNCTION_CALL];
  f.nb_facing = (b) => [`api.facing(${T(b)}, ${safeNum(b.getFieldValue('DEG'), 20)})`, Order.FUNCTION_CALL];
  f.nb_pair_distance = (b) => [`api.pairDist(${T(b, 'A')}, ${T(b, 'B')})`, Order.FUNCTION_CALL];
  f.nb_pair_closer = (b) => [`api.pairCloser(${T(b, 'A')}, ${T(b, 'B')})`, Order.FUNCTION_CALL];
  f.nb_speed = () => ['api.speed()', Order.FUNCTION_CALL];
  f.nb_fwd_speed = () => ['api.fwdSpeed()', Order.FUNCTION_CALL];
  f.nb_height = () => ['api.height()', Order.FUNCTION_CALL];
  f.nb_height_gain = () => ['api.dHeight()', Order.FUNCTION_CALL];
  f.nb_grounded = () => ['api.grounded()', Order.FUNCTION_CALL];
  f.nb_touching = (b) => [`api.touching(${T(b)})`, Order.FUNCTION_CALL];
  f.nb_steps = () => ['api.steps()', Order.FUNCTION_CALL];
  f.nb_time_left = () => ['api.timeLeft()', Order.FUNCTION_CALL];
  f.nb_count = (b) => [`api.count(${T(b)})`, Order.FUNCTION_CALL];
  f.nb_new_area = () => ['api.newArea()', Order.FUNCTION_CALL];
  f.nb_areas = () => ['api.areasVisited()', Order.FUNCTION_CALL];
  f.nb_is_next = (b) => [`api.isNext(${touchedRef(b)})`, Order.FUNCTION_CALL];
  f.nb_laps = () => ['api.laps()', Order.FUNCTION_CALL];
  f.nb_cp_count = () => ['api.checkpoint()', Order.FUNCTION_CALL];
  f.nb_ep_reward = () => ['api.episodeReward()', Order.FUNCTION_CALL];
  f.nb_random = (b) => [`api.random(${safeNum(b.getFieldValue('A'))}, ${safeNum(b.getFieldValue('B'), 1)})`, Order.FUNCTION_CALL];
  f.nb_last_action = (b) => [`api.lastAction(${T(b, 'AXIS')})`, Order.FUNCTION_CALL];
}

/** Code for the chain of statements hanging under a hat block. */
export function chainCode(hat: Blockly.Block): string {
  const next = hat.getNextBlock();
  if (!next) return '';
  const code = gen.blockToCode(next);
  return typeof code === 'string' ? code : code[0];
}
