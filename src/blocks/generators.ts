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

  f.nb_reward = (b) => `api.reward(${val(b, 'VALUE', Order.NONE, '1')});\n`;
  f.nb_penalty = (b) => `api.reward(-(${val(b, 'VALUE', Order.NONE, '1')}));\n`;
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
  f.nb_closer = (b) => [`(-api.delta(${T(b)}))`, Order.ATOMIC];
  f.nb_angle = (b) => [`api.angle(${T(b)})`, Order.FUNCTION_CALL];
  f.nb_facing = (b) => [`api.facing(${T(b)}, ${safeNum(b.getFieldValue('DEG'), 20)})`, Order.FUNCTION_CALL];
  f.nb_pair_distance = (b) => [`api.pairDist(${T(b, 'A')}, ${T(b, 'B')})`, Order.FUNCTION_CALL];
  f.nb_pair_closer = (b) => [`(-api.pairDelta(${T(b, 'A')}, ${T(b, 'B')}))`, Order.ATOMIC];
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
