/** 📚 Learn tab: a compact field guide to the ideas behind the studio. */
import { h } from './dom.ts';

const CONCEPTS: [string, string, string][] = [
  ['🔁 The reinforcement learning loop', 'An AGENT (your robot) observes the world, takes an ACTION, and receives a REWARD. Repeat. Over thousands of tries it learns which actions lead to more total reward. No one tells it the right answer; it discovers it by trial and error.', 'Watch the 👁️ Senses tab while the robot drives: that IS the loop.'],
  ['👁️ Observation (state)', 'The list of numbers the robot perceives each step: distances from ray eyes, the direction to a target, its own speed… The network can only use what is in this vector, so good senses make learning possible.', 'Code tab → "Input vector layout".'],
  ['🕹️ Action', 'What the robot does each step. Discrete = choose one option per control (like pressing keys). Continuous = output a number between −1 and 1 (like a joystick).', 'The 🕹️ blocks define the action space.'],
  ['🍬 Reward', 'A single number after every step. It is the ONLY feedback. The robot maximises the SUM of rewards, so "what you reward is what you get", sometimes in surprising ways (see reward hacking).', 'Event blocks with 🍬 reward / ⚡ penalty.'],
  ['📼 Episode', 'One attempt, from reset to the end (success, failure or time-out). Training = thousands of episodes.', 'The robot block sets the episode length.'],
  ['🔭 Return & discount γ', 'The return is the total future reward. γ (gamma, e.g. 0.99) makes far-future rewards count a bit less: a reward 100 steps away is worth 0.99¹⁰⁰ ≈ 0.37 of one received now. γ close to 1 = patient, far-sighted robot.', 'Training block → discount γ.'],
  ['🧠 Neural network', 'A function made of layers of simple units. Each neuron computes z = w₁x₁ + w₂x₂ + … + b and passes it through an activation function f(z). Stacking layers lets it represent complicated behaviour. Learning = adjusting the weights w and biases b.', '🧠 Brain tab: click a neuron to see its computation.'],
  ['⚡ Activation functions', 'Without them, any stack of layers would collapse into one straight-line (linear) function. ReLU = max(0, z), tanh squashes to (−1, 1), sigmoid to (0, 1). They add the "bends" that make networks powerful.', 'Change the activation in a 🧠 layer block.'],
  ['📉 Loss & gradient descent', 'A loss is a number measuring how wrong the network is. Its gradient (the slope with respect to every weight) says which way to nudge each weight to reduce the loss. Take a small step downhill, repeat: gradient descent.', '📈 Training tab: the loss curves.'],
  ['↩️ Backpropagation', 'The chain rule applied layer by layer, from the output back to the input, to compute every weight\'s gradient efficiently. Our implementation is in src/ml/nn.ts (look for backward()).', 'All of it is hand-written: no ML library in this app.'],
  ['🏃 Learning rate', 'How big each gradient step is. Too high: training explodes or oscillates. Too low: it barely moves. 3e-4 (0.0003) is a classic starting point with Adam.', 'Try 0.003 vs 0.00003 on the flag mission.'],
  ['🧮 Optimizers (SGD, Adam)', 'SGD takes plain steps along the gradient. Momentum adds inertia. Adam keeps per-weight running averages of gradients and their squares, so every weight gets its own adaptive step size. Adam is the default almost everywhere.', 'Training block → optimizer.'],
  ['📦 Batches, minibatches & epochs', 'Instead of updating after every sample, we average gradients over a minibatch (e.g. 256 samples): less noisy, much faster. An epoch = one pass over the collected data. PPO does several epochs over each batch.', 'PPO block → epochs & minibatch.'],
  ['🎭 Policy', 'The robot\'s strategy: a mapping from observations to (probabilities of) actions, π(a|s). In PPO/REINFORCE the network IS the policy.', '🕹️ "What the brain decides" in the Senses tab.'],
  ['💎 Value function', 'V(s): the expected future reward from state s, according to the critic network. Q(s,a): the same but for taking action a first. Values let the robot judge situations before the reward arrives.', '🗺️ Maps → "value map" paints V(s) over the zone.'],
  ['➕ Advantage', 'A(s,a) = how much better an action turned out than the critic expected. Positive → make that action more likely; negative → less likely. It is the heart of actor-critic methods like PPO.', '∑ Math tab shows a real sample.'],
  ['🎲 Exploration vs exploitation', 'To find better strategies the robot must sometimes try things it believes are worse (explore). To score, it must use what it knows (exploit). ε-greedy, entropy bonuses and mutation are all ways to balance this.', 'Raise/lower the entropy bonus and watch the Entropy chart.'],
  ['🌀 Entropy', 'A measure of how random the policy is. High entropy = still exploring. An entropy bonus in the loss keeps the robot curious for longer.', '📈 Entropy chart (PPO).'],
  ['🎯 Reward shaping', 'Sparse rewards (only at the goal) are hard to discover by chance. Shaping adds hints, like "+ how much closer did I get", that point the way. Done carelessly, shaping can change what the robot actually optimises.', 'Remove the "closer" reward in the flag mission and compare.'],
  ['🕳️ Reward hacking', 'When the robot finds a loophole that scores reward without doing what you meant: e.g. circling near a goal to farm "closer" rewards. It is a real problem in AI safety, and it means the reward was mis-specified, not that the robot is "cheating".', 'Soccer mission → "try next".'],
  ['🐌 Sparse reward & curriculum', 'Hard tasks (mazes, crates) rarely give reward early on. Fixes: shaping, checkpoints (breaking the task into steps), curiosity bonuses, or a curriculum that starts easy and gets harder.', 'Maze mission: checkpoints vs. curiosity.'],
  ['📏 Normalisation', 'Networks learn best when inputs have roughly mean 0 and std 1. We keep running statistics of observations (and of returns, to scale rewards). These stats are exported with the weights.', 'Senses tab → "show normalised".'],
  ['👯 Parallel environments', 'Running 16 worlds at once gives more varied, less correlated experience per update and uses the CPU efficiently. Turn on 👻 ghosts to see them all.', 'Training block → parallel worlds.'],
  ['🧑‍🏫 Imitation learning', 'Supervised learning on demonstrations: copy what an expert (you!) did. Fast to learn, but the copy only knows situations it has seen. Real robots are often pre-trained by imitation, then improved with RL.', 'Mission "Teach by Example".'],
  ['🧬 Neuroevolution', 'Optimise the weights with a genetic algorithm instead of gradients: evaluate a population, keep the best, mutate. Simple and parallel; used by OpenAI and Uber AI on control tasks.', 'Mission "Reach the Flag by Evolution".'],
  ['🎓 Overfitting & generalisation', 'A brain that memorised one exact situation fails in others. Randomising spawn points and targets (🎲 scatter) forces general skills.', 'Turn off scattering and see how a trained robot handles a moved flag.'],
  ['💾 What are "weights", really?', 'Just numbers: a grid W and a list b for every layer. Export them (JSON, ONNX, PyTorch) and any program can run your robot\'s brain: Python, Unity, a microcontroller…', '🧠 Brain tab → weight matrices; File → Export brain.'],
];

export class LearnPanel {
  readonly el: HTMLElement;
  constructor(openMissions: () => void) {
    this.el = h(
      'div',
      null,
      h(
        'div',
        { class: 'section' },
        h('h3', null, '🗺️ Suggested path'),
        h(
          'div',
          { class: 'card small', style: { lineHeight: '1.6' } },
          h('div', null, '1️⃣ 🚩 First Steps → watch the reward curve rise.'),
          h('div', null, '2️⃣ Open 🧠 Brain while it drives; click neurons.'),
          h('div', null, '3️⃣ Same task with 🧬 Evolution and 🎯 DQN: compare the dashed curves.'),
          h('div', null, '4️⃣ 🎮 Teach by Example: supervised learning.'),
          h('div', null, '5️⃣ 🪙 Coins, 🌲 Forest, 🌋 Lava: senses & penalties.'),
          h('div', null, '6️⃣ ⚽ Soccer, 🏎️ Race, 🌀 Maze: reward design for real.'),
          h('div', null, '7️⃣ ✏️ Blank Canvas: invent your own task, export the brain.'),
          h('button', { class: 'btn small accent', style: { marginTop: '8px' }, onclick: openMissions }, '🎯 Open missions'),
        ),
      ),
      h(
        'div',
        { class: 'section' },
        h('h3', null, '📖 Concepts'),
        h('div', { class: 'concepts' }, ...CONCEPTS.map(([t, p, see]) => h('details', { class: 'concept' }, h('summary', null, t), h('p', null, p), h('div', { class: 'see' }, '👉 ', see)))),
      ),
    );
  }
}
