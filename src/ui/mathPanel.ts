/** ∑ Math tab: the algorithm explained step by step, with live numbers. */
import { clear, fmt, h } from './dom.ts';

type Info = Record<string, number | string>;

const n = (v: unknown, d = 3) => (typeof v === 'number' ? fmt(v, d) : v === undefined ? '…' : String(v));
const live = (v: unknown, d = 3) => h('span', { class: 'live' }, n(v, d));
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

  constructor() {
    this.body = h('div', { class: 'explain' });
    this.el = h('div', null, this.body);
  }

  render(visible: boolean, force = false): void {
    if (!visible) return;
    const now = performance.now();
    if (!force && now - this.last < 500) return;
    this.last = now;
    clear(this.body);
    const i = this.info;
    const ph = this.phase.toLowerCase();
    const stage = (label: string, on: boolean) => h('div', { class: 'stage' + (on ? ' on' : '') }, label);
    switch (this.algo) {
      case 'ppo':
        this.body.append(
          h('h3', null, 'PPO: Proximal Policy Optimization'),
          h('div', { class: 'pipeline' }, stage(`① Collect ${n(i.batch, 0)} steps from ${n(i.worlds, 0)} worlds`, ph.includes('collect')), h('span', { class: 'arrow' }, '→'), stage('② Score each step (GAE)', false), h('span', { class: 'arrow' }, '→'), stage(`③ Learn: ${n(i.epochs, 0)} epochs × minibatches of ${n(i.minibatch, 0)}`, ph.includes('learn'))),
          P(h('b', null, '① Collect. '), 'The current policy plays in every parallel world for ', live(i.horizon, 0), ' steps. For each step we store the observation s, the action a, its probability π(a|s), the reward r and the critic\'s guess V(s).'),
          P(h('b', null, '② Advantage. '), 'Was each action better or worse than expected? First the one-step surprise (TD error):'),
          F('δₜ = rₜ + γ·V(sₜ₊₁) − V(sₜ)    with γ = ', live(i.gamma)),
          P('then GAE blends surprises over many future steps (λ = ', live(i.lambda), ' trades bias for variance):'),
          F('Aₜ = δₜ + γλ·δₜ₊₁ + (γλ)²·δₜ₊₂ + …'),
          P('A real sample from the last batch: reward ', live(i.sample_reward), ', critic predicted ', live(i.sample_value), ', advantage ', live(i.sample_advantage), ', so the return target for the critic is ', live(i.sample_ret), '.'),
          P(h('b', null, '③ Learn. '), 'Make good actions more likely, but not too much at once. With the probability ratio ρ = π_new(a|s) / π_old(a|s):'),
          F('L_clip = −E[ min( ρ·A ,  clip(ρ, 1−ε, 1+ε)·A ) ]    ε = ', live(i.clip)),
          F('L = L_clip  +  c·(V(s) − R)²  −  β·H(π)      β = ', live(i.entropyCoef)),
          P('Last update: policy loss ', live(i.policyLoss), ', value loss ', live(i.valueLoss), ', entropy ', live(i.entropy), ', approx. KL ', live(i.approxKL, 4), ', clipped ', live(typeof i.clipFraction === 'number' ? i.clipFraction * 100 : undefined, 1), '% of samples, explained variance ', live(i.explainedVariance), '.'),
          P('The gradients of L flow backwards through the network (backpropagation) and Adam nudges every weight by a tiny amount (learning rate ', live(i.lr, 5), '). Updates so far: ', live(i.updates, 0), '.'),
        );
        break;
      case 'dqn':
        this.body.append(
          h('h3', null, 'DQN: Deep Q-Network'),
          h('div', { class: 'pipeline' }, stage('① Act ε-greedy', ph.includes('playing') || ph.includes('filling')), h('span', { class: 'arrow' }, '→'), stage('② Store in replay memory', ph.includes('filling') || ph.includes('playing')), h('span', { class: 'arrow' }, '→'), stage('③ Learn from random memories', ph.includes('learning'))),
          P(h('b', null, 'Q(s, a)'), ' = "total future reward if I do a now and play well afterwards". The network outputs one Q per action combination (', live(i.jointActions, 0), ' of them) and the robot takes the biggest.'),
          P(h('b', null, '① Explore: '), 'with probability ε = ', live(i.epsilon), ' do something random, otherwise the best action.'),
          P(h('b', null, '② Remember: '), 'every (s, a, r, s′) goes into the replay memory: ', live(i.replaySize, 0), ' / ', live(i.replayCapacity, 0), ' memories.'),
          P(h('b', null, '③ Learn: '), 'pick ', live(i.batch, 0), ' random memories and move Q(s,a) toward the Bellman target:'),
          F('y = r + γ · Q_target(s′, argmaxₐ′ Q(s′, a′))     γ = ', live(i.gamma)),
          F('loss = Huber( Q(s, a) − y )'),
          P('Double DQN: ', live(i.double), '. The target network is a frozen copy, refreshed every ', live(i.targetUpdate, 0), ' steps (', live(i.targetSyncs, 0), ' times so far), so the targets do not chase their own tail.'),
          P('Last 100 updates: loss ', live(i.loss, 4), ', mean Q ', live(i.meanQ), ', mean |TD error| ', live(i.tdError), '.'),
          this.extra.qInfo ? P('Right now the robot sees Q-values ', live(this.extra.qInfo), '.') : '',
        );
        break;
      case 'reinforce':
        this.body.append(
          h('h3', null, 'REINFORCE (Monte-Carlo policy gradient)'),
          h('div', { class: 'pipeline' }, stage(`① Play ${n(i.episodesPerUpdate, 0)} full episodes`, ph.includes('playing')), h('span', { class: 'arrow' }, '→'), stage('② Compute returns Gₜ', false), h('span', { class: 'arrow' }, '→'), stage('③ One gradient step', ph.includes('learning'))),
          P('After an episode, every step gets the discounted sum of the rewards that followed it:'),
          F('Gₜ = rₜ + γ·rₜ₊₁ + γ²·rₜ₊₂ + …      γ = ', live(i.gamma)),
          P('Then push up the log-probability of each action in proportion to how good its return was (minus a baseline b(s) to reduce noise; baseline: ', live(i.baseline), '):'),
          F('∇J ≈ Σₜ ∇ log π(aₜ|sₜ) · (Gₜ − b(sₜ))'),
          P('Last update used ', live(i.batchSteps, 0), ' steps, mean return ', live(i.meanReturn), ', entropy ', live(i.entropy), '.'),
          P('Because every update needs whole episodes and uses each sample once, REINFORCE is slower and noisier than PPO, which reuses samples safely thanks to clipping.'),
        );
        break;
      case 'evolution':
        this.body.append(
          h('h3', null, 'Neuroevolution (genetic algorithm)'),
          h('div', { class: 'pipeline' }, stage(`① Evaluate ${n(i.population, 0)} brains`, true), h('span', { class: 'arrow' }, '→'), stage(`② Keep the best ${n(i.elites, 0)}`, false), h('span', { class: 'arrow' }, '→'), stage('③ Mutate to refill', false)),
          P('No gradients, no backprop. Each brain is just a long vector of ', live(i.params, 0), ' numbers (its weights). Its fitness is the total reward it earns in an episode.'),
          F('child = parent + σ · 𝒩(0, 1)     σ = ', live(i.sigma, 4)),
          P('Generation ', live(i.generation, 0), ': best fitness ', live(i.bestFitness), ', mean ', live(i.meanFitness), ', worst ', live(i.worstFitness), '. The elites survive unchanged (elitism), so the best score can never get worse… unless luck in evaluation fooled us.'),
          P('Evolution explores by changing the BRAIN, while PPO/REINFORCE explore by randomising ACTIONS. That is why evolved robots act deterministically.'),
        );
        break;
      case 'bc':
        this.body.append(
          h('h3', null, 'Behaviour cloning (supervised learning)'),
          P('Your recorded driving is a labelled dataset: input = what the robot saw, label = the key you pressed. We train the network like an image classifier:'),
          F('loss = −(1/N) Σ log π(a_you | s)'),
          P('Epoch ', live(i.epoch, 0), ': loss ', live(i.loss), ', accuracy ', live(typeof i.accuracy === 'number' ? i.accuracy * 100 : undefined, 1), '% on ', live(i.demos, 0), ' recorded steps.'),
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
