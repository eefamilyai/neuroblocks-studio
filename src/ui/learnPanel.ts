/** 📚 Learn: a compact field guide to the ideas behind the studio (shown in a dialog). */
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
  ['🕘 Memory (frame stacking)', 'A plain network only sees the present moment. Feeding it the last few observations too ("remember the last 3 moments") lets it feel motion and remember things that just went out of view. The classic Atari DQN stacked 4 frames.', '👁️ Senses block "remember the last … moments", or sense.memory(3) in code.'],
  ['📉 Learning-rate schedules', 'Big steps early to learn fast, small careful steps later to fine-tune: decaying the learning rate (linearly or with a cosine curve) often gives a better final brain. The entropy bonus can be decayed the same way: explore early, exploit later.', 'Training block → advanced → "learning rate decays…".'],
  ['🧪 Evaluation vs training', 'Training curves include exploration noise and a brain that changes every update. To know how good a brain REALLY is, freeze it and test it on fresh episodes: that is evaluation, the RL version of a test set.', '📈 Training tab → Evaluate.'],
  ['🔬 Experiments & hyperparameters', 'Learning rate, γ, network size… are hyperparameters: choices the algorithm cannot learn by itself. Researchers compare them with sweeps: the same task trained several times with one setting changed (and several random seeds, because RL is noisy).', '🧪 Experiments tab.'],
  ['🧾 Reward breakdown', 'When several rewards add up, it is hard to tell which one the robot is chasing. Splitting the episode reward by source shows it: if a shaping reward keeps growing while the goal reward does not, the robot found a loophole.', '📈 Training tab → "Where does the reward come from?".'],
  ['⌨️ Code mode', 'Everything the blocks do, as JavaScript, plus more: your own senses, rewards computed with any maths, levels built from code, every hyperparameter. Blocks convert to code with one click, so you can start visual and go textual when you are ready.', 'Editor → Code. Ctrl+Space for suggestions, 📖 for the reference.'],
  ['💾 What are "weights", really?', 'Just numbers: a grid W and a list b for every layer. Export them (JSON, ONNX, PyTorch) and any program can run your robot\'s brain: Python, Unity, a microcontroller…', '🧠 Brain tab → weight matrices; File → Export brain.'],
];

/** Language models (the Language studio). */
const LLM_CONCEPTS: [string, string, string][] = [
  ['💬 What is a language model?', 'A network that reads some text and predicts the NEXT token. That is all. To write, it predicts, picks one token, appends it, and repeats. ChatGPT works exactly like this, with a much bigger network trained on much more text.', 'Playground → "+1 token": watch one prediction at a time.'],
  ['✂️ Tokens & tokenizers', 'The model never sees letters, only numbers. A tokenizer cuts text into tokens (characters, words or pieces of words) and gives each one an id. The list of all tokens is the vocabulary.', 'Tokens tab: type anything and see it cut up.'],
  ['🧩 Byte-pair encoding (BPE)', 'GPT\'s tokenizer. Start from single characters and repeatedly glue together the pair that appears most often ("t"+"h" → "th", "th"+"e" → "the"…). Common words become one token, rare words are spelled with pieces. Fewer tokens per sentence = the model sees more text at once.', 'Tokenizer → BPE, then Tokens tab → merges.'],
  ['📍 Embeddings', 'Token id 7 becomes row 7 of a learned table: a vector of numbers. Training moves tokens that are used in similar ways close together: vowels, digits, punctuation, capital letters each form their own neighbourhood.', 'Embeddings tab: click a token to see its neighbours.'],
  ['🧭 Position embeddings', 'Attention by itself cannot tell "dog bites man" from "man bites dog": it has no sense of order. Adding a different vector for every position fixes that. Learned (a trained table) or sinusoidal (fixed waves, from the original Transformer paper).', 'Builder → Positions → "none", and watch it struggle.'],
  ['👀 Self-attention', 'Every token makes a query ("what am I looking for?"), a key ("what do I offer?") and a value ("what I pass on"). It mixes the values of earlier tokens, weighted by softmax(query·key / √d). Several heads look for different things at the same time.', 'Attention tab: each row is one token deciding what to read.'],
  ['🚫 The causal mask', 'When predicting the next token, peeking at it would be cheating. So a token may only attend to itself and EARLIER tokens: the upper half of every attention map is blacked out.', 'Attention tab: everything above the diagonal is black.'],
  ['🧱 The transformer block', 'LayerNorm → attention → add the result back (residual), then LayerNorm → MLP → add back. Stack a few of these. The "residual stream" carries information straight through, which is why transformers can be made very deep.', 'Builder → Architecture: click every box.'],
  ['🎯 Cross-entropy loss', 'loss = −log(probability the model gave to the real next token). A perfect prediction gives 0; random guessing among V tokens gives ln(V). Training = making the real next token more and more likely.', 'Training tab: the dashed "random guess" line.'],
  ['🤔 Perplexity', 'e^loss. A perplexity of 4 means the model is as unsure as if it had to choose between 4 equally likely tokens every time. Lower is better; 1 would be a mind reader.', 'Training tab → Perplexity.'],
  ['🌡️ Temperature, top-k & top-p', 'How to pick from the probabilities. Temperature divides the scores: low = always the safest token (boring, repetitive), high = wild. Top-k only keeps the k most likely tokens; top-p keeps the smallest set that covers p of the probability.', 'Playground sliders: the "next token" bars change live.'],
  ['🧪 Validation & overfitting', 'Part of the text is never used for training. If the training loss keeps falling but the validation loss rises, the model is memorising instead of learning. Small data + big model = memorising.', 'Calculator preset: can it add sums it has never seen?'],
  ['📈 Warm-up, cosine decay, AdamW, clipping', 'The recipe real LLMs are trained with: start with a tiny learning rate (warm-up), then slowly decay it (cosine); AdamW gives every weight its own step size plus a little weight decay; clipping caps sudden huge gradients.', 'Training tab → learning-rate and gradient-norm charts.'],
  ['🧵 Data parallelism', 'Split every batch over several workers, compute the gradients in parallel, add them up: the same update, faster. Big LLMs train like this on thousands of GPUs; here it is your CPU cores.', 'Builder → Training → CPU threads.'],
  ['🔧 Fine-tuning', 'Take a trained model and keep training it on new text: it keeps its general skills and picks up the new style. That is how base models become chat assistants or code helpers.', 'Train a story model, then "Train on your own text" → Fine-tune.'],
  ['⚡ The KV cache', 'Generating token 200 should not redo the work for tokens 1–199. Each layer remembers the keys and values of earlier tokens, so a new token only computes its own. Every chatbot does this.', 'src/llm/decode.ts'],
  ['📐 Scale', 'Your model: about 100,000 weights and a context of 64 tokens. GPT-2 small: 124 million weights, 1,024 tokens. GPT-3: 175 billion. Same blocks, same loss, same training loop: mostly just bigger, and trained on far more text.', 'Model tab: where the weights are.'],
];

export class LearnPanel {
  readonly el: HTMLElement;
  constructor(openMissions: () => void, openLab: () => void, mode: 'robots' | 'language' = 'robots') {
    const path: [string, (() => void) | null][] = [
      ['Neuron Lab (top bar): see a network learn to separate dots, the basics of all ML.', openLab],
      ['Mission "First Steps": press Train and watch the reward curve rise.', openMissions],
      ['Open the Brain tab while it drives; click neurons.', null],
      ['Same task with Evolution and DQN: compare the dashed curves.', null],
      ['"Teach by Example": supervised learning from your own driving.', null],
      ['Coins, Forest, Lava: senses & penalties.', null],
      ['Soccer, Race, Maze: reward design for real.', null],
      ['Experiments tab: sweep the learning rate, see why hyperparameters matter.', null],
      ['Switch the editor to Code: write your own rewards, senses and levels, then export the brain.', null],
    ];
    const llmPath = [
      'Story writer: press Train and watch the live samples go from random letters to words to sentences.',
      'Playground: generate with temperature 0.2, then 1.5. Use "+1 token" and pick tokens yourself.',
      'Tokens tab: switch the tokenizer between Characters and BPE and compare.',
      'Copy machine preset: open the Attention tab and find the head that points at the letter to copy.',
      'Calculator preset: does it learn to ADD sums it never saw, or only memorise?',
      'Train on your own text, then fine-tune a trained model on something new.',
      'Export → Python script, and run your model with NumPy or PyTorch.',
    ];
    const concepts = (list: [string, string, string][]) => h('div', { class: 'concepts' }, ...list.map(([t, p, see]) => h('details', { class: 'concept' }, h('summary', null, t), h('p', null, p), h('div', { class: 'see' }, see))));
    const robots = [
      h('h3', { class: 'h-sub', style: { marginTop: mode === 'robots' ? '0' : undefined } }, 'Robots: suggested path'),
      h('ol', { class: 'path-list' }, ...path.map(([t, go]) => h('li', null, go ? h('a', { href: '#', onclick: (e: Event) => (e.preventDefault(), go()) }, t) : t))),
      h('h3', { class: 'h-sub' }, 'Robots: concepts'),
      concepts(CONCEPTS),
    ];
    const language = [
      h('h3', { class: 'h-sub', style: { marginTop: mode === 'language' ? '0' : undefined } }, 'Language models: suggested path'),
      h('ol', { class: 'path-list' }, ...llmPath.map((t) => h('li', null, t))),
      h('h3', { class: 'h-sub' }, 'Language models: concepts'),
      concepts(LLM_CONCEPTS),
    ];
    this.el = h(
      'div',
      null,
      ...(mode === 'language' ? [...language, ...robots] : [...robots, ...language]),
    );
  }
}
