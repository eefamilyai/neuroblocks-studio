<div align="center">

# 🧠 NeuroBlocks Studio

**Scratch-style blocks (or real code) for machine learning.** Build a robot's senses, actions, brain and rewards, drop it into a 3D world, and watch a neural network learn, with every number under the hood on display. Write your own learning algorithms, run experiments, and export the trained weights as JSON, ONNX or PyTorch.

**…and build your own mini language models.** Switch to **Language** to design a tiny GPT, train it on stories, names, sums, text messages or your own writing, turn it into a phone keyboard that predicts your next word and fixes your typos, and take it apart: tokens, attention, embeddings, every weight.

### ▶ [Open the studio in your browser](https://eefamilyai.github.io/neuroblocks-studio/)

Runs entirely in your browser. No install, no server, no account.<br>
The neural networks, backpropagation, PPO, DQN, REINFORCE, neuroevolution, the transformer, the tokenizers and the physics are all written from scratch in TypeScript: no ML libraries, nothing hidden.

![NeuroBlocks Studio training a robot to reach a flag](docs/screenshots/hero.jpg)

</div>

---

## What you can do

| | |
|---|---|
| 🧩 **Program learning, not moves** | Snap together blocks for what the robot **senses** (network inputs), what it **can do** (outputs), its **brain** (layers & activations), the **learning algorithm** and **rewards**: Scratch-style event hats like *"when I touch 🪙 coin → reward 1"*. |
| ⌨️ **…or write code** | Switch the editor to **Code** and the same program becomes JavaScript, with every setting spelled out. Autocomplete, a searchable reference, inline errors with line numbers and infinite-loop protection. Code can do more than blocks: your own senses with any maths, levels built from code (`world.add`, `world.wall`), every hyperparameter. |
| 🧪 **Your own learning algorithm** | `train.custom({ setup, act, observe, update })` replaces PPO with **your** code, using a small toolkit on the same from-scratch library: `ctx.mlp`, `net.forward`, `net.backward`, `ctx.adam`, sampling, log-probability gradients. A complete REINFORCE in ~40 lines is included as a template, and it learns. |
| 🏝️ **A prebuilt 3D world** | An island with 12 training zones (meadow, coin field, hedge maze, lava lake, mountain, race track, soccer pitch, sky parkour, forest, warehouse, tag arena, dodge arena) plus free roam. |
| 🧱 **Build mode** | Place coins, flags, balls, crates, pads, runner/chaser bots, walls, ramps, lava, ice and bounce pads. **Select** what you placed to drag it, rotate it (R), resize it or delete it. Empty a zone to build a level from scratch. |
| 🌍 **World rules** | Change gravity, friction, air drag, bot speed, kick power, masses and how often the brain decides, per program. |
| 🏋️ **5 learning methods + yours** | **PPO** (discrete or continuous), **DQN** (replay, target network, double DQN), **REINFORCE**, **neuroevolution**, **behaviour cloning** (drive with the keyboard, the network copies you) and **train.custom**. |
| 🎛️ **Training controls** | Train **one update at a time** and look at what changed. Learning-rate & entropy **schedules**, **stop** at a success rate, reward or time, **memory** (frame stacking), a **speed limit** to watch the ghosts learn in real time. |
| 🕰️ **Brain timeline** | Brains are saved as training goes. Scrub back to watch how the robot drove at any point, jump to the **best** one, or train again from there. |
| ✅ **Evaluation** | Test a frozen brain on fresh episodes (no exploration, no learning): success rate, reward ± std, a histogram and the reward breakdown. The RL version of a test set. |
| 🔬 **Experiments** | Hyperparameter sweeps: train several brains side by side in parallel background workers, changing one setting (learning rate, γ, entropy, network width/depth, worlds, algorithm, seed, memory…), and compare the curves. |
| 🔧 **Under the hood** | Live learning curves with run-vs-run comparison, **where the reward comes from** (per reward source), a clickable network diagram (every neuron's `z = Σ w·x + b` with the actual numbers), weight matrices, the exact input vector with saliency, action probabilities, critic value, Q-values, the algorithm's equations with live values, a heatmap and a **value map** painted on the ground. |
| 💾 **Export** | **JSON** (re-importable), **ONNX** (Netron, onnxruntime, Unity Sentis…), a standalone **PyTorch/NumPy `.py`**, training logs as **CSV**, the program as **.js**. Exports are tested to reproduce the browser's outputs to about 1e‑7. |
| 🧪 **Neuron Lab** | Supervised learning playground: a network separates 2D data (circle, XOR, spiral, moons…) or **points you draw yourself**. Watch the decision boundary bend; play with features, layers, learning rate, optimizer, batch size, L2 regularisation and the train/test split. |
| 🎯 **17 missions** | Ready-made, editable programs, from "reach the flag in 15 seconds" to soccer, racing with smooth controls, mazes, parkour and crate pushing. They open as blocks or as code. |
| 💬 **Mini language models** | A second studio for **tiny GPTs**: choose the training text, the tokenizer (characters, words or GPT-style BPE), the architecture and the training recipe, then train in background workers on several CPU cores. Generate text in a playground that colours every token by the model's confidence, and look at attention maps, the embedding space and every weight matrix. [More below](#mini-language-models). |

<table>
<tr>
<td width="50%"><img src="docs/screenshots/code.jpg" alt="Code mode"><br><b>Code mode.</b> The same robot as JavaScript; train straight from the editor.</td>
<td width="50%"><img src="docs/screenshots/custom-algorithm.jpg" alt="Your own algorithm"><br><b>Your own algorithm.</b> REINFORCE written in <code>train.custom</code>; its logged numbers become charts.</td>
</tr>
<tr>
<td><img src="docs/screenshots/experiments.jpg" alt="Experiments tab"><br><b>Experiments.</b> Four learning rates trained in parallel workers.</td>
<td><img src="docs/screenshots/brain.jpg" alt="Brain tab"><br><b>Brain tab.</b> Click any neuron to see its weighted sum and activation.</td>
</tr>
<tr>
<td><img src="docs/screenshots/soccer-senses.jpg" alt="Senses tab in the soccer zone"><br><b>Senses tab.</b> Every input number, the decision probabilities and the critic's value, live.</td>
<td><img src="docs/screenshots/build.jpg" alt="Build mode"><br><b>Build mode.</b> Place, select, rotate and resize your own course.</td>
</tr>
<tr>
<td><img src="docs/screenshots/island.jpg" alt="The island"><br><b>The island.</b> 12 zones; click a label to train there.</td>
<td><img src="docs/screenshots/lab.jpg" alt="Neuron Lab"><br><b>Neuron Lab.</b> A network learning a spiral, with engineered features.</td>
</tr>
</table>

## Quick start

1. Open the studio. The first mission (**🚩 Reach the Flag**) is already loaded.
2. Press **▶ Train**. Within about 15 seconds the reward curve climbs and the success rate hits 100%.
3. Open **Brain** while it drives and click neurons. Open **Senses** to see exactly what the network receives.
4. Change something: delete the *"how much closer I got"* reward, add ray eyes, try **Evolution** instead of PPO. Press **New brain** and compare the curves (the old run stays dashed). Or run it as an **Experiment**.
5. Switch the editor to **Code** to see (and edit) the program as JavaScript.
6. **Export** the brain to take it anywhere.
7. New to neural networks? Start with the **Lab** in the top bar.

Keyboard: `Ctrl+Enter` train/pause · `Ctrl+S` save · `Ctrl+Space` code suggestions · in "You" mode `WASD`/arrows drive, `Space` jump, `F` kick, `Q`/`E` slide · in build mode `R` rotate, `Del` delete, `Esc` deselect.

## Mini language models

Press **Language** in the top-left corner. The same studio, now for text: you build a **tiny GPT**, train it in your browser and look at everything it does.

<table>
<tr>
<td width="50%"><img src="docs/screenshots/llm-train.jpg" alt="Training a story model"><br><b>Training.</b> The loss falls from "random guessing"; live samples go from noise to sentences.</td>
<td width="50%"><img src="docs/screenshots/llm-keyboard.jpg" alt="The keyboard"><br><b>Keyboard.</b> Your model as a phone keyboard: next-word suggestions and autocorrect, with the scores behind every choice.</td>
</tr>
<tr>
<td><img src="docs/screenshots/llm-playground.jpg" alt="The playground"><br><b>Playground.</b> Every token coloured by how sure the model was; hover for the other candidates.</td>
<td><img src="docs/screenshots/llm-attention.jpg" alt="Attention maps"><br><b>Attention.</b> Which earlier tokens each token reads, for every layer and head.</td>
</tr>
<tr>
<td><img src="docs/screenshots/llm-calculator.jpg" alt="Held-out accuracy of the calculator"><br><b>Does it generalise?</b> The calculator gets about 90% of sums it has never seen right.</td>
<td><img src="docs/screenshots/llm-embeddings.jpg" alt="Embedding map"><br><b>Embeddings.</b> The learned token vectors in 2D, with nearest neighbours.</td>
</tr>
</table>

**Start from a preset** (top bar → Presets), press **Train**, and watch:

| | Preset | Trains in* | What you see |
|---|---|---|---|
| 📖 | Story writer | 2–5 min | Random letters → words → sentences, in the live samples |
| ⌨️ | Phone keyboard | 1–3 min | Word prediction and autocorrect on everyday text messages: for messages it never saw, the real next word is among its 3 suggestions about 90% of the time |
| 🏷️ | Name generator | ~30 s | New names that sound real; train it too long and it starts memorising (validation loss goes up) |
| ➕ | Calculator | 2–6 min | Learns to add: about 90% of sums it has **never seen** come out right |
| 🔁 | Copy machine | 2–4 min | Copies and reverses new strings perfectly; one attention head points at exactly the letter to write next |
| 🐍 | Code helper | 2–5 min | A miniature code model: colons, indentation, `return` |
| 🧩 | Story writer with BPE | 2–5 min | The same stories with GPT-style tokens |

<sub>* The shorter time with several CPU threads (the default on a 4-core computer), the longer one on a single thread.</sub>

**Build it.** The model builder on the left has every choice a real LLM has, at toy scale:

- **Data**: tiny stories, text messages, first names, addition problems, copy & reverse puzzles, small Python programs, or **your own text** (paste it or load a `.txt`; it never leaves your browser). The last 10% is kept apart as validation text. For lists (names, messages, problems) every line is one example, and training windows line up with the lines: fixed-format tasks start every problem at position 0 (so each digit always sits in the same place), while names and messages keep the line break in front as a "start" token (so the model also learns how a line begins).
- **Tokenizer**: characters, whole words (rare words spelled out), or **byte-pair encoding** like GPT, with a vocabulary size you choose. The Tokens tab shows how any text is cut up, the whole vocabulary and the BPE merges in the order they were learned.
- **Architecture**: context length, embedding size, layers, attention heads, MLP width, activation (GELU, ReLU, SiLU), positions (learned, sinusoidal or none) and weight tying, with a live diagram you can click for explanations and the exact number of weights.
- **Training**: learning rate, batch, steps, warm-up, cosine decay, AdamW weight decay, gradient clipping, validation interval, seed, and **CPU threads**: every batch is split over several background workers whose gradients are added up (data parallelism, the way big models are trained on many GPUs).

**Watch it learn.** Train and validation loss against the *random guess* line (ln of the vocabulary size), perplexity, learning rate, gradient norm, tokens per second, live samples every few steps and, for tasks with right answers (sums, copying), the **accuracy on problems it never saw**. Pause, train one step at a time, change the learning rate while it trains, or press Train again after it finishes to keep going.

**Use it.** The playground generates text with temperature, top-k and top-p, colours each token by its probability and shows, for the next token, the full ranking of candidates: click one to choose it yourself. Click any token to see its attention.

**Type with it.** The **Keyboard** tab turns any trained model into a phone keyboard. While you type, it suggests the three most likely next words, or ways to finish the word you started (tap one or press Tab). The model predicts tokens, not words, so a beam search follows the most likely token sequences until each one completes a word. When you finish a word it doesn't know, **autocorrect** looks at known words one or two letters away and picks the one that fits best: the language model's score in this context minus a cost per typo (the classic "noisy channel"). Backspace right after a correction undoes it. Everything it weighs is shown under the box. Train it on your own messages or essays to get a keyboard that writes like you.

**Look inside.** *Attention*: heatmaps for every layer and head (or all heads side by side); follow one token to see what it reads and what it predicts. *Embeddings*: PCA of the token (or position) vectors, with nearest neighbours by cosine similarity. *Model*: where the weights are (embedding, attention, MLP, LayerNorm), and every tensor with its shape, mean, spread and histogram.

**Keep it.** Save the model (weights + tokenizer + settings) to a file and open it later; it is also autosaved in the browser. **Fine-tune**: train a model on one text, switch to another and press Train: it keeps what it knows and adapts. **Export** a single Python file:

```bash
python story_writer.py "Once upon a time"     # NumPy only: tokenizer, transformer and sampling, all in plain code
```

```python
from story_writer import generate, torch_model
print(generate("Once upon a time", n=300, temperature=0.8))
model = torch_model()   # the same network as a PyTorch nn.Module with the trained weights, ready to keep training
```

The export is tested: the Python forward pass reproduces the browser's next-token scores to about 1e‑7.

## Code mode in 30 seconds

```js
robot({ zone: 'meadow', spawn: 'random', facing: 'random', maxSteps: 300 });

sense.target('flag');          // 3 inputs: direction + distance to the flag
sense.velocity();              // 3 inputs
sense.custom('speed', () => api.speed() / 6);   // your own input

actions.discrete();
act.move();
act.turn();

brain.layer(32, 'tanh');
brain.layer(32, 'tanh');

train.ppo({ lr: 3e-4, numEnvs: 16, lrSchedule: 'linear', targetSuccess: 0.95 });
world.physics({ gravity: 9.8 });

on.start(() => api.randomize('flag'));
on.step(() => {
  api.reward(api.closer('flag') * 0.5, 'closer to flag');   // the label shows up in the reward breakdown
  api.penalty(0.002, 'time');
});
on.touch('flag', () => {
  api.reward(1, 'goal');
  api.end(true);
});
```

The 📖 button in the editor lists every function (`robot`, `sense.*`, `act.*`, `brain.*`, `train.*`, `world.*`, `on.*`, and the ~50 `api.*` calls available inside events). Code from a project file someone else made is not run until you confirm.

### Your own algorithm

```js
train.custom({
  numEnvs: 16,
  setup(ctx) {
    this.policy = ctx.mlp([32, 32], 'tanh');      // the brain: what the robot uses and what gets exported
    this.opt = ctx.adam(this.policy, 0.003);
    this.batch = [];
  },
  act(obs, ctx) {
    return ctx.sample(this.policy.forward(obs));  // one choice per action
  },
  observe(t, ctx) {
    this.batch.push(t);                           // { obs, action, reward, done, next, world, … }
  },
  update(ctx) {
    // …compute a loss gradient dLogits, then:
    // this.policy.zeroGrad(); this.policy.backward(dLogits, n); this.opt.step();
    ctx.log({ loss: 0 });                         // → a chart in the Training tab
    return false;                                 // true when the brain changed
  },
});
```

**Project → New code project: your own algorithm** opens a complete REINFORCE written this way.

## How it works (the "backend")

```
blocks (Blockly) ─┐
                  ├─compile─► AgentSpec ──► Web Worker ──► VecEnv (16 × Env) ◄──► algorithm (PPO / DQN / … / yours)
code (JavaScript) ┘              │                │                                     │
                                 │                │ brain snapshots, stats, ghosts      │ MLP + backprop + Adam
                                 ▼                ▼                                     ▼
                         3D view (Three.js) runs the newest brain at real-time speed; panels show everything
```

| Folder | What's inside |
|---|---|
| [`src/ml/nn.ts`](src/ml/nn.ts) | Dense layers with hand-written `forward`/`backward` (the chain rule, layer by layer), 8 activation functions, weight init. The matrix loops are register-tiled (2 samples × 4 neurons per pass) for ~2× speed, and tested against the plain textbook loops. |
| [`src/ml/optim.ts`](src/ml/optim.ts) | SGD, momentum, RMSProp, Adam, gradient clipping. |
| [`src/ml/policy.ts`](src/ml/policy.ts) | Softmax/multi-discrete and Gaussian policies: sampling, log-probabilities, entropy and their gradients. |
| [`src/ml/algos/`](src/ml/algos) | `ppo.ts`, `dqn.ts`, `reinforce.ts`, `evolution.ts`, `bc.ts`, `custom.ts` (your own), `evaluate.ts`, the vectorised environment and the trainer (schedules). Each file starts with a plain-English explanation. |
| [`src/sim/`](src/sim) | The world generator, collision geometry (spheres vs rotated boxes and cylinders, ray casts on a spatial grid), robot physics (configurable), sensors, frame stacking, NPC bots, and the `ScriptAPI` that rewards call. |
| [`src/blocks/`](src/blocks) | Block definitions, code generators, the block compiler and the missions. |
| [`src/code/`](src/code) | Code mode: the builder runtime and validation (`runtime.ts`), infinite-loop protection (`guard.ts`), blocks → code (`fromBlocks.ts`), the API reference (`docs.ts`), templates. |
| [`src/app/`](src/app) | The studio: shell & layout, block and code editors, the 3D world view (HUD, build mode), experiments, dialogs. |
| [`src/render/`](src/render) | Three.js world (per-zone chunks), robot, objects, ghosts, rays, heatmap and value map; adaptive render quality. |
| [`src/ui/`](src/ui) | Panels (training, brain, senses, math, code), charts, Neuron Lab, icons. |
| [`src/export/`](src/export) | ONNX writer (a tiny protobuf encoder) and the PyTorch script generator. |
| [`src/llm/`](src/llm) | The mini language models: a decoder-only transformer with hand-written backpropagation through attention, LayerNorm, GELU and the tied output head (`gpt.ts`), KV-cache text generation and sampling (`decode.ts`), character / word / BPE tokenizers (`tokenizer.ts`), the generated datasets, the AdamW trainer with warm-up and cosine decay (`train.ts`), multi-core data-parallel training (`parallel.ts`), held-out accuracy (`probe.ts`) and the Python export. The UI is in [`src/llm/ui/`](src/llm/ui). |

Physics runs at 60 Hz and the brain decides at 20 Hz (every 3 physics ticks) by default, like most robot-learning setups.

### Performance

- **Training** runs in a background worker. The first mission trains at roughly 15,000 environment steps per second on our test machine (about 2.5× faster than the first version), bigger networks proportionally slower.
- **The 3D view** adapts: in *Auto* quality it lowers the render resolution (and, if needed, turns shadows off) when frames get slow, redraws shadows every other frame, and slows to a few frames per second when nothing moves. Settings let you pick High/Medium/Low and cap the frame rate.
- **Language models** train at about 2,500 tokens per second on one CPU thread for the 100,000-weight story model. Splitting each batch over several threads (the default is one less than your number of cores, at most 4) made it about 2.4× faster with 3 threads and 2.8× with 4 on our 4-core test machine. The language studio is its own file, loaded the first time you open it.
- **Loading**: the code editor, the Lab and the training worker are separate files, downloaded only when needed.

## Using an exported brain

**Python + ONNX** (`pip install onnxruntime numpy`):

```python
import numpy as np, onnxruntime as ort

sess = ort.InferenceSession("robo.onnx")
obs = np.zeros((1, sess.get_inputs()[0].shape[1]), dtype=np.float32)  # your sensor values, in the order listed in the Code tab
outputs = sess.run(None, {"obs": obs})
print(outputs[0])   # logits (or q_values / action_mean); outputs[1] = critic value if present
```

Observation normalisation is baked into the ONNX graph, so feed raw sensor values. With memory (frame stacking) the input is the last N observations in a row, newest first.

**PyTorch script**: `python robo_brain.py` runs a NumPy forward pass and, if PyTorch is installed, the same network as an `nn.Module` (`torch_model()`) that you can fine-tune.

**JSON**: `net.layers[i].W` is a row-major `[out × in]` matrix (PyTorch `nn.Linear` layout), `b` the bias, plus `obsNorm`, the action `space` and `obsLabels`. Import it back with *Project → Import a brain*.

## Run it locally

```bash
git clone https://github.com/eefamilyai/neuroblocks-studio
cd neuroblocks-studio
npm install
npm run dev            # http://localhost:5173
```

```bash
npm test               # gradient checks (MLPs and the whole transformer), physics, every mission, code-mode errors, tokenizers, a tiny GPT learning…
npm run test:export    # exports brains and language models and checks them with onnxruntime / NumPy (needs: pip install onnx onnxruntime numpy)
npm run build          # static site in dist/
npm run build:single   # ONE self-contained HTML file in dist-single/ (works offline, double-click to open)
npm run bench -- soccer 1000000   # train a mission headlessly in Node and print its learning curve
```

**Install it like an app:** in Chrome or Edge, use the install icon in the address bar (it's a PWA and works offline after the first visit). An offline single-file copy is also published at [`/neuroblocks-studio-offline.html`](https://eefamilyai.github.io/neuroblocks-studio/neuroblocks-studio-offline.html).

## Missions

| | Mission | Algorithm | Teaches |
|---|---|---|---|
| 🚩 | First Steps: Reach the Flag | PPO | observations → network → actions, reward shaping |
| 🎯 | Reach the Flag with DQN | DQN | Q-learning, ε-greedy, replay memory, target networks |
| 🧬 | Reach the Flag by Evolution | Neuroevolution | genetic algorithms, fitness, mutation |
| 📜 | Reach the Flag with REINFORCE | REINFORCE | policy gradients, baselines |
| 🎮 | Teach by Example | Behaviour cloning | supervised vs reinforcement learning |
| 🪙 | Coin Collector | PPO | ray-cast vision, dense rewards, penalties |
| 🧪 | Coins: Evolution vs PPO | Neuroevolution | comparing algorithms |
| 🌲 | Find the Flag in the Forest | PPO | combining sensors, obstacle avoidance |
| 🐰 | Tag! Catch the Runner | PPO | moving targets, pursuit |
| 🌋 | Lava Crossing | PPO | terminal penalties, ground probes, discount γ |
| ⚽ | Soccer Star | PPO | multi-part rewards, reward hacking |
| 🌀 | Maze Runner | PPO | sparse rewards, curricula, exploration |
| 🏎️ | Race Track | PPO (continuous) | Gaussian policies, exploration noise |
| ⛰️ | Climb the Summit | PPO | progress rewards, local optima |
| ☁️ | Sky Parkour | PPO | timing (jumping), terminal states |
| 👾 | Dodge & Collect | PPO | competing objectives, 360° vision |
| 📦 | Warehouse Crate Push | PPO | hard exploration, indirect control |

## License

MIT. Use it, remix it, teach with it.
