/** The Blockly block editor: setup, layout ("tidy"), labels and robot-block shortcuts. */
import Blockly from '../blocks/blockly.ts';
import type * as BT from 'blockly/core';
import 'blockly/blocks';
import * as En from 'blockly/msg/en';
import { defineBlocks } from '../blocks/defs.ts';
import { compileWorkspace } from '../blocks/compiler.ts';
import type { CompileResult } from '../blocks/compiler.ts';
import { toolbox } from '../blocks/toolbox.ts';
import { neuroTheme } from '../blocks/theme.ts';
import type { ObjDef } from '../sim/types.ts';

const LEFT = ['nb_robot', 'nb_world', 'nb_senses', 'nb_actions', 'nb_brain'];

export class BlocksEditor {
  readonly ws: BT.WorkspaceSvg;
  private host: HTMLElement;
  onChange: (() => void) | null = null;
  private quiet = 0;

  constructor(host: HTMLElement) {
    this.host = host;
    // 'blockly/core' ships without UI strings: load English explicitly
    Blockly.setLocale(En as unknown as { [key: string]: string });
    defineBlocks();
    this.ws = Blockly.inject(host, {
      toolbox: toolbox() as BT.utils.toolbox.ToolboxDefinition,
      theme: neuroTheme(),
      renderer: 'zelos',
      media: './media/',
      // the single-file build has no media folder: hide the icon-based controls there
      trashcan: import.meta.env.MODE !== 'single',
      sounds: false,
      zoom: { controls: import.meta.env.MODE !== 'single', wheel: true, startScale: 0.7, maxScale: 1.6, minScale: 0.3, scaleSpeed: 1.1 },
      grid: { spacing: 24, length: 2, colour: '#1b2130', snap: false },
      move: { scrollbars: true, drag: true, wheel: false },
    });
    this.ws.addChangeListener((e: BT.Events.Abstract) => {
      if (e.isUiEvent || this.quiet) return;
      this.onChange?.();
    });
    new ResizeObserver(() => this.resize()).observe(host);
  }

  resize(): void {
    if (this.host.offsetParent) Blockly.svgResize(this.ws);
  }

  compile(customObjects: ObjDef[]): CompileResult {
    return compileWorkspace(this.ws, customObjects);
  }

  save(): object {
    return Blockly.serialization.workspaces.save(this.ws);
  }

  load(json: object): void {
    this.quiet++;
    Blockly.Events.disable();
    try {
      this.ws.clear();
      Blockly.serialization.workspaces.load(json as never, this.ws);
    } finally {
      Blockly.Events.enable();
      this.quiet--;
    }
  }

  undo(redo = false): void {
    this.ws.undo(redo);
  }

  robot(): BT.Block | null {
    return this.ws.getTopBlocks(false).find((b) => b.type === 'nb_robot') ?? null;
  }

  /** Show the input count and network size on the hat blocks. */
  updateLabels(r: CompileResult): void {
    const tops = this.ws.getTopBlocks(false);
    const senses = tops.find((b) => b.type === 'nb_senses');
    const brain = tops.find((b) => b.type === 'nb_brain');
    this.quiet++;
    Blockly.Events.disable();
    try {
      if (senses) senses.setFieldValue(`= ${r.obsSize} numbers into the brain`, 'INFO');
      if (brain) {
        const hidden = r.spec.brain.layers.map((l) => l.units).join(' → ');
        brain.setFieldValue(`${r.obsSize} in → ${hidden} → ${r.outSize} out · ${r.params.toLocaleString()} weights`, 'INFO');
      }
    } finally {
      Blockly.Events.enable();
      this.quiet--;
    }
  }

  /** Arrange the blocks neatly: settings on the left, training & events on the right (one column when narrow). */
  tidy(): void {
    const ws = this.ws;
    const tops = ws.getTopBlocks(true);
    const left = tops.filter((b) => LEFT.includes(b.type)).sort((a, b) => LEFT.indexOf(a.type) - LEFT.indexOf(b.type));
    const right = tops.filter((b) => !LEFT.includes(b.type));
    const wOf = (bs: BT.BlockSvg[]) => bs.reduce((m, b) => Math.max(m, b.getHeightWidth().width), 0);
    const leftW = wOf(left);
    const rightW = wOf(right);
    const viewPx = Math.max(200, this.host.clientWidth - 110);
    const gap = 40;
    const twoCols = (leftW + gap + rightW + 40) * 0.6 <= viewPx;
    let y = 20;
    const place = (bs: BT.BlockSvg[], x: number, y0: number) => {
      let yy = y0;
      for (const b of bs) {
        b.moveTo(new Blockly.utils.Coordinate(x, yy));
        yy += b.getHeightWidth().height + 26;
      }
      return yy;
    };
    let total: number;
    if (twoCols) {
      place(left, 20, y);
      place(right, 20 + leftW + gap, y);
      total = 20 + leftW + gap + rightW + 20;
    } else {
      y = place(left, 20, y);
      place(right, 20, y + 10);
      total = Math.max(leftW, rightW) + 40;
    }
    const scale = Math.max(0.5, Math.min(0.78, viewPx / total));
    ws.setScale(scale);
    setTimeout(() => ws.scroll(0, 0), 0);
  }

  setRobotField(field: string, value: unknown): boolean {
    const r = this.robot();
    if (!r) return false;
    r.setFieldValue(value, field);
    return true;
  }
}
