import { buildWorld } from '../src/sim/worldgen.ts';
import { WorldRuntime } from '../src/sim/runtime.ts';
import { Env, compileScript } from '../src/sim/env.ts';
import { makeSpec, AXIS_PRESETS } from '../src/sim/spec.ts';
import { RNG } from '../src/ml/rng.ts';
import { test, assert } from './harness.ts';

const world = buildWorld();
const rt = new WorldRuntime(world);

test('world builds with all zones', () => {
  assert(world.zones.length === 13, `zones: ${world.zones.length}`);
  assert(world.statics.length > 100, `statics: ${world.statics.length}`);
  console.log(`      statics=${world.statics.length} objects=${world.objects.length} decor=${world.decor.length}`);
});

for (const zone of world.zones) {
  test(`random robot survives zone "${zone.id}"`, () => {
    const spec = makeSpec({
      zone: zone.id,
      spawn: { mode: 'zone', x: 0, z: 0, yawMode: 'default', yaw: 0, jitter: 1 },
      sensors: [
        { kind: 'rays', count: 7, spread: 160, range: 15, tilt: 0, detect: ['wall', 'coin', 'lava', 'ball'] },
        { kind: 'target', tag: 'flag', range: 40 },
        { kind: 'velocity' },
        { kind: 'ground_probe', count: 3, dist: 1.5 },
        { kind: 'time' },
      ],
      axes: [AXIS_PRESETS.move.both, AXIS_PRESETS.turn.both, AXIS_PRESETS.jump.both, AXIS_PRESETS.kick.both],
      maxSteps: 300,
    });
    const errors: string[] = [];
    const env = new Env(rt, spec, null, 7);
    env.onError = (m) => errors.push(m);
    const rng = new RNG(3);
    env.reset();
    // spawn must be on solid ground
    assert(env.a.y > -1, `spawned at y=${env.a.y}`);
    const act = new Float32Array(4);
    let steps = 0;
    let episodes = 0;
    const t0 = performance.now();
    while (steps < 3000) {
      act[0] = rng.int(3);
      act[1] = rng.int(3);
      act[2] = rng.next() < 0.05 ? 1 : 0;
      act[3] = rng.next() < 0.1 ? 1 : 0;
      env.step(act);
      steps++;
      for (let i = 0; i < env.obs.length; i++) assert(Number.isFinite(env.obs[i]), `obs[${i}] not finite`);
      assert(Number.isFinite(env.a.x + env.a.y + env.a.z), 'agent position NaN');
      assert(Math.abs(env.a.vx) < 60 && Math.abs(env.a.vz) < 60, `velocity blew up ${env.a.vx},${env.a.vz}`);
      if (zone.id !== 'world' && zone.id !== 'sky' && zone.id !== 'lava') {
        const r = zone.rect;
        assert(env.a.x > r.minX - 1 && env.a.x < r.maxX + 1 && env.a.z > r.minZ - 1 && env.a.z < r.maxZ + 1, `escaped zone at ${env.a.x.toFixed(1)},${env.a.z.toFixed(1)}`);
      }
      if (env.done) {
        episodes++;
        env.reset();
      }
    }
    const ms = performance.now() - t0;
    assert(errors.length === 0, errors.join('; '));
    console.log(`      ${(steps / (ms / 1000)).toFixed(0)} steps/s, ${episodes} episodes, obs=${env.obs.length}`);
  });
}

test('script events: coin pickup gives reward and respawns', () => {
  const src = `function(api, Math){
    var S = {start:[],step:[],touch:[],pair:[],fall:[],end:[],senses:[]};
    S.touch.push(['coin', function(obj){ api.reward(1); api.respawn(obj); }]);
    return S; }`;
  const spec = makeSpec({ zone: 'coins', sensors: [{ kind: 'target', tag: 'coin', range: 40 }], maxSteps: 2000 });
  const env = new Env(rt, spec, compileScript(src), 1);
  env.reset();
  // greedy controller: turn toward the nearest coin and drive
  let total = 0;
  for (let i = 0; i < 1500; i++) {
    const ang = env.obs[0];
    const act = [2, Math.abs(ang) < 0.1 ? 1 : ang > 0 ? 2 : 0];
    env.step(act);
    total += env.stepReward;
  }
  console.log(`      greedy coin bot collected ${total} coins`);
  assert(total >= 5, `only ${total} coins`);
});
