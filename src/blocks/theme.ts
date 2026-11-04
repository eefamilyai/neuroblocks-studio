import Blockly from './blockly.ts';
import type * as BT from 'blockly/core';
import { COLORS } from './defs.ts';

function style(primary: string, hat = false) {
  return { colourPrimary: primary, colourSecondary: shade(primary, 0.85), colourTertiary: shade(primary, 0.7), hat: hat ? 'cap' : '' };
}

function shade(hex: string, k: number): string {
  const n = parseInt(hex.slice(1), 16);
  const r = Math.round(((n >> 16) & 255) * k);
  const g = Math.round(((n >> 8) & 255) * k);
  const b = Math.round((n & 255) * k);
  return '#' + ((1 << 24) | (r << 16) | (g << 8) | b).toString(16).slice(1);
}

let theme: BT.Theme | null = null;

export function neuroTheme(): BT.Theme {
  if (theme) return theme;
  theme = Blockly.Theme.defineTheme('neuroblocks', {
    name: 'neuroblocks',
    base: Blockly.Themes.Zelos,
    blockStyles: {
      robot_hat: style(COLORS.robot, true),
      robot_blocks: style(COLORS.robot),
      senses_hat: style(COLORS.senses, true),
      senses_blocks: style(COLORS.senses),
      actions_hat: style(COLORS.actions, true),
      actions_blocks: style(COLORS.actions),
      brain_hat: style(COLORS.brain, true),
      brain_blocks: style(COLORS.brain),
      train_hat: style(COLORS.train, true),
      event_hat: style(COLORS.events, true),
      reward_blocks: style(COLORS.rewards),
      measure_blocks: style(COLORS.measure),
      logic_blocks: style(COLORS.logic),
      math_blocks: style(COLORS.math),
      variable_blocks: style(COLORS.vars),
      loop_blocks: style(COLORS.control),
      text_blocks: style('#59C059'),
    },
    categoryStyles: {
      robot_category: { colour: COLORS.robot },
      senses_category: { colour: COLORS.senses },
      actions_category: { colour: COLORS.actions },
      brain_category: { colour: COLORS.brain },
      train_category: { colour: COLORS.train },
      events_category: { colour: COLORS.events },
      rewards_category: { colour: COLORS.rewards },
      measure_category: { colour: COLORS.measure },
      logic_category: { colour: COLORS.logic },
      math_category: { colour: COLORS.math },
      variable_category: { colour: COLORS.vars },
    },
    componentStyles: {
      workspaceBackgroundColour: '#161a2b',
      toolboxBackgroundColour: '#1d2236',
      toolboxForegroundColour: '#e6e9f5',
      flyoutBackgroundColour: '#23293f',
      flyoutForegroundColour: '#cfd5ea',
      flyoutOpacity: 0.97,
      scrollbarColour: '#4a5274',
      scrollbarOpacity: 0.6,
      insertionMarkerColour: '#ffffff',
      insertionMarkerOpacity: 0.3,
      cursorColour: '#ffd166',
    },
    fontStyle: { family: '"Nunito", "Segoe UI", system-ui, sans-serif', weight: '700', size: 11 },
    startHats: true,
  });
  return theme;
}
