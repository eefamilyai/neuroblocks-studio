import Blockly from '../src/blocks/blockly.ts';
import 'blockly/blocks';
import { defineBlocks } from '../src/blocks/defs.ts';
import { compileWorkspace } from '../src/blocks/compiler.ts';
import { MISSIONS } from '../src/blocks/missions.ts';
import { Trainer } from '../src/ml/algos/trainer.ts';
import { buildWorld } from '../src/sim/worldgen.ts';
import { test, assert } from './harness.ts';

export function compileMission(id: string) {
  defineBlocks();
  const m = MISSIONS.find((x) => x.id === id)!;
  const ws = new Blockly.Workspace();
  Blockly.serialization.workspaces.load(m.workspace() as never, ws);
  const r = compileWorkspace(ws);
  ws.dispose();
  return r;
}

for (const m of MISSIONS) {
  test(`mission "${m.id}" compiles and trains`, () => {
    const r = compileMission(m.id);
    assert(r.errors.length === 0, 'errors: ' + r.errors.join(' | '));
    const real = r.warnings.filter((w) => !w.includes('no 🍬') || m.id !== 'blank');
    if (m.id !== 'blank') assert(real.length === 0, 'warnings: ' + real.join(' | '));
    const errs: string[] = [];
    const tr = new Trainer({ spec: r.spec, world: buildWorld(), onError: (e) => errs.push(e) });
    while (tr.totalSteps < 3000) tr.runFor(50);
    assert(errs.length === 0, 'script errors: ' + errs.join(' | '));
    console.log(`      obs=${r.obsSize} out=${r.outSize} params=${r.params} algo=${r.spec.train.algo}`);
  });
}
