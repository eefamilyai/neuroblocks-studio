/** Starter programs for code mode. */

/** Write your own learning algorithm: REINFORCE from scratch, in ~40 lines. */
export const CUSTOM_ALGO_TEMPLATE = `// 🧪 Your own learning algorithm: REINFORCE (policy gradient), written from scratch.
// The world, senses, actions and rewards are set up as usual; train.custom(...)
// below replaces PPO with YOUR code. Press ▶ Train, then change the algorithm!

robot({ zone: 'meadow', spawn: 'random', facing: 'random', maxSteps: 300 });
sense.target('flag');
sense.velocity();
actions.discrete();
act.move();
act.turn();

on.start(() => api.randomize('flag'));
on.step(() => {
  api.reward(api.closer('flag') * 0.5, 'closer to flag');
  api.penalty(0.002, 'time');
});
on.touch('flag', () => {
  api.reward(1, 'goal');
  api.end(true);
});

train.custom({
  numEnvs: 16,          // parallel worlds
  gamma: 0.99,          // your own settings live here too: this.gamma
  episodesPerUpdate: 16,

  // Runs once: build the brain. this.policy is what the robot uses and what gets exported.
  setup(ctx) {
    this.policy = ctx.mlp([32, 32], 'tanh');   // obs → 32 → 32 → one logit per choice
    this.opt = ctx.adam(this.policy, 0.003);
    this.running = ctx.worlds(() => []);       // the episode in progress, per world
    this.finished = [];                        // whole episodes, waiting to be learned from
  },

  // Every step, for every world: pick an action by sampling from π(a|s).
  act(obs, ctx) {
    return ctx.sample(this.policy.forward(obs));
  },

  // See what happened after the action.
  observe(t, ctx) {
    this.running[t.world].push({ obs: t.obs, action: t.action, reward: t.reward });
    if (t.done) {
      this.finished.push(this.running[t.world]);
      this.running[t.world] = [];
    }
  },

  // Learn from whole episodes.
  update(ctx) {
    if (this.finished.length < this.episodesPerUpdate) return false;
    // 1. the return of every step: G_t = r_t + γ·r_t+1 + γ²·r_t+2 + …
    const steps = [];
    for (const ep of this.finished) {
      let G = 0;
      for (let t = ep.length - 1; t >= 0; t--) {
        G = ep[t].reward + this.gamma * G;
        steps.push({ obs: ep[t].obs, action: ep[t].action, G });
      }
    }
    // 2. advantages: how much better than average was each step? (a simple baseline)
    const n = steps.length;
    const mean = steps.reduce((a, s) => a + s.G, 0) / n;
    const std = Math.sqrt(steps.reduce((a, s) => a + (s.G - mean) ** 2, 0) / n) + 1e-8;
    // 3. loss = −mean(A · log π(a|s)): push up the probability of good actions
    const logits = this.policy.forward(ctx.batch(steps.map((s) => s.obs)), n);
    const dLogits = new Float32Array(logits.length);
    let entropy = 0;
    steps.forEach((s, i) => {
      const A = (s.G - mean) / std;
      ctx.addLogProbGrad(logits, i, s.action, -A / n, dLogits);   // ∂loss/∂logits
      ctx.addEntropyGrad(logits, i, -0.01 / n, dLogits);          // a small bonus for exploring
      entropy += ctx.entropy(logits, i) / n;
    });
    // 4. backpropagation + one Adam step
    this.policy.zeroGrad();
    this.policy.backward(dLogits, n);
    ctx.clipGrad(this.policy, 0.5);
    this.opt.step();
    ctx.log({ meanReturn: mean, entropy, steps: n });   // → charts in the Training tab
    this.finished = [];
    return true;                                         // = one update of the brain
  },
});
`;
