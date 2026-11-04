/**
 * Benchmark: train one mission headlessly and report the learning curve.
 *   node --experimental-strip-types tests/bench-mission.ts <mission-id> <steps>
 */
import { compileMission } from './missions.test.ts';
import { Trainer } from '../src/ml/algos/trainer.ts';
import { buildWorld } from '../src/sim/worldgen.ts';

const id = process.argv[2];
const budget = Number(process.argv[3] ?? 300000);
const r = compileMission(id);
const tr = new Trainer({ spec: r.spec, world: buildWorld(), onError: (e) => console.log('ERR', e) });
const t0 = performance.now();
let next = budget / 10;
const lines: string[] = [];
while (tr.totalSteps < budget) {
  tr.runFor(500);
  if (tr.totalSteps >= next) {
    next += budget / 10;
    const eps = tr.algo.stats.episodes.slice(-100);
    const succ = eps.filter((e) => e.success).length / Math.max(1, eps.length);
    const meanR = eps.reduce((a, e) => a + e.reward, 0) / Math.max(1, eps.length);
    const len = eps.reduce((a, e) => a + e.length, 0) / Math.max(1, eps.length);
    lines.push(`${String(tr.totalSteps).padStart(8)} R=${meanR.toFixed(2).padStart(7)} succ=${(succ * 100).toFixed(0).padStart(3)}% len=${len.toFixed(0)} ${(tr.totalSteps / ((performance.now() - t0) / 1000)).toFixed(0)} st/s`);
  }
}
console.log(`=== ${id} (${r.spec.train.algo}, ${((performance.now() - t0) / 1000).toFixed(0)} s)\n` + lines.join('\n'));
