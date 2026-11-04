/** ∑ Math tab: the algorithm explained step by step, with live numbers (built once, values updated in place). */
import { clear, fmt, h, setText } from './dom.ts';

type Info = Record<string, number | string>;

const n = (v: unknown, d = 3) => (typeof v === 'number' ? fmt(v, d) : v === undefined ? '…' : String(v));
const F = (...parts: (string | Node)[]) => h('div', { class: 'formula' }, ...parts);
const P = (...parts: (string | Node)[]) => h('p', null, ...parts);

export class MathPanel {
  readonly el: HTMLElement;
  private body: HTMLElement;
  private last = 0;
  algo = 'ppo';
  info: Info = {};
  phase = '';
  /** extra, e.g. Q-values of the current showcase state */
  extra: Info = {};
  private built = '';
  private lives: { el: HTMLElement; get: () => unknown; d: number }[] = [];
  private stages: { el: HTMLElement; on: (ph: string) => boolean; label?: () => string }[] = [];

  constructor() {
    this.body = h('div', { class: 'explain' });
    this.el = h('div', null, this.body);
  }

  render(visible: boolean, force = false): void {
    if (!visible) return;
    const now = performance.now();
    if (!force && now - this.last < 400) return;
    this.last = now;
    if (this.built !== this.algo) this.build();
    for (const l of this.lives) setText(l.el, n(l.get(), l.d));
    const ph = this.phase.toLowerCase();
    for (const s of this.stages) {
      s.el.classList.toggle('on', s.on(ph));
      if (s.label) setText(s.el, s.label());
    }
  }

  private build(): void {
    this.built = this.algo;
    clear(this.body);
    this.lives = [];
    this.stages = [];
    const live = (key: string, d = 3, f?: (v: number) => number) => {
      const el = h('span', { class: 'live' });
      this.lives.push({ el, d, get: () => { const v = this.info[key]; return f && typeof v === 'number' ? f(v) : v; } });
      return el;
    };
    const extra = (key: string) => {
      const el = h('span', { class: 'live' });
      this.lives.push({ el, d: 3, get: () => this.extra[key] });
      return el;
    };
    const stage = (label: () => string, on: (ph: string) => boolean) => {
      const el = h('div', { class: 'stage' }, label());
      this.stages.push({ el, on, label });
      return el;
    };
    const arrow = (): HTMLElement => h('span', { class: 'arrow' }, '→');
    switch (this.algo) {
      case 'ppo':
        this.body.append(
          h('h3', null, 'PPO: Proximal Policy Optimization'),
          h('div', { class: 'pipeline' }, stage(() => `① Collect ${n(this.info.batch, 0)} steps from ${n(this.info.worlds, 0)} worlds`, (ph) => ph.includes('collect')), arrow(), stage(() => '② Score each step (GAE)', () => false), arrow(), stage(() => `③ Learn: ${n(this.info.epochs, 0)} epochs × minibatches of ${n(this.info.minibatch, 0)}`, (ph) => ph.includes('learn'))),
          P(h('b', null, '① Collect. '), 'The current policy plays in every parallel world for ', live('horizon', 0), ' steps. For each step we store the observation s, the action a, its probability π(a|s), the reward r and the critic\'s guess V(s).'),
          P(h('b', null, '② Advantage. '), 'Was each action better or worse than expected? First the one-step surprise (TD error):'),
          F('δₜ = rₜ + γ·V(sₜ₊₁) − V(sₜ)    with γ = ', live('gamma')),
          P('then GAE blends surprises over many future steps (λ = ', live('lambda'), ' trades bias for variance):'),
          F('Aₜ = δₜ + γλ·δₜ₊₁ + (γλ)²·δₜ₊₂ + …'),
          P('A real sample from the last batch: reward ', live('sample_reward'), ', critic predicted ', live('sample_value'), ', advantage ', live('sample_advantage'), ', so the return target for the critic is ', live('sample_ret'), '.'),
          P(h('b', null, '③ Learn. '), 'Make good actions more likely, but not too much at once. With the probability ratio ρ = π_new(a|s) / π_old(a|s):'),
          F('L_clip = −E[ min( ρ·A ,  clip(ρ, 1−ε, 1+ε)·A ) ]    ε = ', live('clip')),
          F('L = L_clip  +  c·(V(s) − R)²  −  β·H(π)      β = ', live('entropyCoef')),
          P('Last update: policy loss ', live('policyLoss'), ', value loss ', live('valueLoss'), ', entropy ', live('entropy'), ', approx. KL ', live('approxKL', 4), ', clipped ', live('clipFraction', 1, (v) => v * 100), '% of samples, explained variance ', live('explainedVariance'), '.'),
          P('The gradients of L flow backwards through the network (backpropagation) and Adam nudges every weight by a tiny amount (learning rate ', live('lr', 5), '). Updates so far: ', live('updates', 0), '.'),
        );
        break;
      case 'dqn':
        this.body.append(
          h('h3', null, 'DQN: Deep Q-Network'),
          h('div', { class: 'pipeline' }, stage(() => '① Act ε-greedy', (ph) => ph.includes('playing') || ph.includes('filling')), arrow(), stage(() => '② Store in replay memory', (ph) => ph.includes('filling') || ph.includes('playing')), arrow(), stage(() => '③ Learn from random memories', (ph) => ph.includes('learning'))),
          P(h('b', null, 'Q(s, a)'), ' = "total future reward if I do a now and play well afterwards". The network outputs one Q per action combination (', live('jointActions', 0), ' of them) and the robot takes the biggest.'),
          P(h('b', null, '① Explore: '), 'with probability ε = ', live('epsilon'), ' do something random, otherwise the best action.'),
          P(h('b', null, '② Remember: '), 'every (s, a, r, s′) goes into the replay memory: ', live('replaySize', 0), ' / ', live('replayCapacity', 0), ' memories.'),
          P(h('b', null, '③ Learn: '), 'pick ', live('batch', 0), ' random memories and move Q(s,a) toward the Bellman target:'),
          F('y = r + γ · Q_target(s′, argmaxₐ′ Q(s′, a′))     γ = ', live('gamma')),
          F('loss = Huber( Q(s, a) − y )'),
          P('Double DQN: ', live('double'), '. The target network is a frozen copy, refreshed every ', live('targetUpdate', 0), ' steps (', live('targetSyncs', 0), ' times so far), so the targets do not chase their own tail.'),
          P('Last 100 updates: loss ', live('loss', 4), ', mean Q ', live('meanQ'), ', mean |TD error| ', live('tdError'), '.'),
          P('Right now the robot sees Q-values ', extra('qInfo'), '.'),
        );
        break;
      case 'reinforce':
        this.body.append(
          h('h3', null, 'REINFORCE (Monte-Carlo policy gradient)'),
          h('div', { class: 'pipeline' }, stage(() => `① Play ${n(this.info.episodesPerUpdate, 0)} full episodes`, (ph) => ph.includes('playing')), arrow(), stage(() => '② Compute returns Gₜ', () => false), arrow(), stage(() => '③ One gradient step', (ph) => ph.includes('learning'))),
          P('After an episode, every step gets the discounted sum of the rewards that followed it:'),
          F('Gₜ = rₜ + γ·rₜ₊₁ + γ²·rₜ₊₂ + …      γ = ', live('gamma')),
          P('Then push up the log-probability of each action in proportion to how good its return was (minus a baseline b(s) to reduce noise; baseline: ', live('baseline'), '):'),
          F('∇J ≈ Σₜ ∇ log π(aₜ|sₜ) · (Gₜ − b(sₜ))'),
          P('Last update used ', live('batchSteps', 0), ' steps, mean return ', live('meanReturn'), ', entropy ', live('entropy'), '.'),
          P('Because every update needs whole episodes and uses each sample once, REINFORCE is slower and noisier than PPO, which reuses samples safely thanks to clipping.'),
        );
        break;
      case 'evolution':
        this.body.append(
          h('h3', null, 'Neuroevolution (genetic algorithm)'),
          h('div', { class: 'pipeline' }, stage(() => `① Evaluate ${n(this.info.population, 0)} brains`, () => true), arrow(), stage(() => `② Keep the best ${n(this.info.elites, 0)}`, () => false), arrow(), stage(() => '③ Mutate to refill', () => false)),
          P('No gradients, no backprop. Each brain is just a long vector of ', live('params', 0), ' numbers (its weights). Its fitness is the total reward it earns in an episode.'),
          F('child = parent + σ · 𝒩(0, 1)     σ = ', live('sigma', 4)),
          P('Generation ', live('generation', 0), ': best fitness ', live('bestFitness'), ', mean ', live('meanFitness'), ', worst ', live('worstFitness'), '. The elites survive unchanged (elitism), so the best score can never get worse… unless luck in evaluation fooled us.'),
          P('Evolution explores by changing the BRAIN, while PPO/REINFORCE explore by randomising ACTIONS. That is why evolved robots act deterministically.'),
        );
        break;
      case 'custom': {
        const keys = h('div', { class: 'kv' });
        this.body.append(
          h('h3', null, 'Your own algorithm'),
          P('This brain is trained by YOUR code in train.custom({...}): setup() builds the networks, act() picks actions, observe() sees every transition and update() changes the weights.'),
          P('The toolkit does the heavy lifting with the same from-scratch library as PPO: ', h('b', null, 'net.forward'), ' (the forward pass), ', h('b', null, 'net.backward'), ' (backpropagation of ∂loss/∂outputs), and an optimizer step. For a policy gradient, ∂log π(a|s)/∂logits is simply:'),
          F('∂ log softmax(z)ₐ / ∂zᵢ  =  [i = a] − pᵢ        (ctx.addLogProbGrad)'),
          P('Numbers your algorithm logged with ctx.log({...}) at the last update:'),
          keys,
        );
        this.stages.push({
          el: keys,
          on: () => {
            const k = JSON.stringify(Object.keys(this.info));
            if (keys.dataset.k !== k) {
              keys.dataset.k = k;
              clear(keys);
              for (const name of Object.keys(this.info)) keys.append(h('span', null, name), live(name, 4));
            }
            return false;
          },
        });
        break;
      }
      case 'bc':
        this.body.append(
          h('h3', null, 'Behaviour cloning (supervised learning)'),
          P('Your recorded driving is a labelled dataset: input = what the robot saw, label = the key you pressed. We train the network like an image classifier:'),
          F('loss = −(1/N) Σ log π(a_you | s)'),
          P('Epoch ', live('epoch', 0), ': loss ', live('loss'), ', accuracy ', live('accuracy', 1, (v) => v * 100), '% on ', live('demos', 0), ' recorded steps.'),
          P('Watch out for distribution shift: if the copy drifts into a situation you never demonstrated, it has no idea what to do. Fine-tuning with PPO fixes that.'),
        );
        break;
    }
    this.body.append(
      h('h3', { style: { marginTop: '16px' } }, 'Every brain does the same forward pass'),
      F('h₁ = f(W₁·x + b₁),   h₂ = f(W₂·h₁ + b₂),   …,   output = Wₙ·hₙ₋₁ + bₙ'),
      P('x are the senses (see 👁️), W and b are the weights you can export, f is the activation function you chose in the 🧠 brain blocks.'),
    );
  }
}
