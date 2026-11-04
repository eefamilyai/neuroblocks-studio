/**
 * Reference for code mode: every function a program can call. Used by the
 * editor's autocomplete and by the 📖 reference drawer.
 */
export interface DocEntry {
  group: string;
  /** what you type, e.g. "sense.target" or "api.reward" */
  name: string;
  /** signature shown next to the name */
  sig: string;
  doc: string;
  /** snippet inserted by autocomplete (${} = placeholders) */
  snippet?: string;
}

export const DOC_GROUPS = ['Robot', 'Senses', 'Actions', 'Brain', 'Training', 'World', 'Events', 'api: rewards & episode', 'api: measuring', 'api: objects & world', 'Your own algorithm', 'JavaScript'];

export const DOCS: DocEntry[] = [
  // ---- robot
  { group: 'Robot', name: 'robot', sig: "({ zone, spawn, facing, jitter, maxSteps, name, color, speed, turn, jump })", doc: "Where the robot lives and how episodes start. zone: 'meadow', 'coins', 'maze', 'lava', 'summit', 'race', 'soccer', 'sky', 'forest', 'warehouse', 'tag', 'dodge' or 'world'. spawn: 'zone', 'random' or { x, z }. facing: 'default', 'random' or a compass angle (0 = north). maxSteps: episode length (20 steps = 1 s). speed/turn/jump: body multipliers.", snippet: "robot({ zone: '${meadow}', spawn: '${random}', maxSteps: ${300} });" },
  // ---- senses
  { group: 'Senses', name: 'sense.target', sig: "(tag, { range })", doc: 'Direction (sin, cos) and distance to the nearest thing with that tag: 3 inputs. Tags: flag, coin, trophy, ball, crate, goal, pad, next_checkpoint, runner, chaser, wall, lava, tree, rock, spawn.', snippet: "sense.target('${flag}');" },
  { group: 'Senses', name: 'sense.rays', sig: "({ count, spread, range, tilt, detect })", doc: 'Ray-cast "eyes" fanned out in front of the robot. Each ray gives its distance (0…1) plus one 0/1 input per detected tag. tilt (degrees) points them down to see holes.', snippet: "sense.rays({ count: ${7}, spread: ${120}, range: ${15}, detect: ['${wall}'] });" },
  { group: 'Senses', name: 'sense.pair', sig: '(a, b)', doc: 'Direction and distance from thing a to thing b (e.g. ball → goal): 3 inputs.', snippet: "sense.pair('${ball}', '${goal}');" },
  { group: 'Senses', name: 'sense.velocity', sig: '()', doc: 'My speed forward, sideways and up/down: 3 inputs.', snippet: 'sense.velocity();' },
  { group: 'Senses', name: 'sense.heading', sig: '()', doc: 'Compass: sin and cos of my heading: 2 inputs.', snippet: 'sense.heading();' },
  { group: 'Senses', name: 'sense.position', sig: '()', doc: 'My x and z, scaled to −1…1 across the zone: 2 inputs.', snippet: 'sense.position();' },
  { group: 'Senses', name: 'sense.grounded', sig: '()', doc: '1 when standing on something, 0 in the air: 1 input.', snippet: 'sense.grounded();' },
  { group: 'Senses', name: 'sense.height', sig: '()', doc: 'How high I am: 1 input.', snippet: 'sense.height();' },
  { group: 'Senses', name: 'sense.groundProbes', sig: '({ count, dist })', doc: 'Feelers in front of me that detect holes and lava: 2 inputs each.', snippet: 'sense.groundProbes({ count: ${3}, dist: ${1.5} });' },
  { group: 'Senses', name: 'sense.timeLeft', sig: '()', doc: '1 at the start of an episode, 0 at the end: 1 input.', snippet: 'sense.timeLeft();' },
  { group: 'Senses', name: 'sense.lastAction', sig: '()', doc: 'What I did last step: 1 input per action.', snippet: 'sense.lastAction();' },
  { group: 'Senses', name: 'sense.custom', sig: '(name, () => number)', doc: 'Your own input: any number you can compute with api (clamped to ±10). Runs every step.', snippet: "sense.custom('${my sense}', () => ${api.speed() / 6});" },
  { group: 'Senses', name: 'sense.memory', sig: '(frames)', doc: 'Frame stacking: also feed the observations of the last frames−1 steps (multiplies the inputs).', snippet: 'sense.memory(${3});' },
  // ---- actions
  { group: 'Actions', name: 'actions.discrete', sig: '()', doc: 'The brain picks from choices (like pressing keys). Works with every algorithm.', snippet: 'actions.discrete();' },
  { group: 'Actions', name: 'actions.continuous', sig: '()', doc: 'The brain outputs smooth values from −1 to 1 (like a joystick). PPO, REINFORCE or evolution.', snippet: 'actions.continuous();' },
  { group: 'Actions', name: 'act.move', sig: '({ backwards })', doc: 'Drive forward/back. { backwards: false } = forward only.', snippet: 'act.move();' },
  { group: 'Actions', name: 'act.turn', sig: '()', doc: 'Steer left/right.', snippet: 'act.turn();' },
  { group: 'Actions', name: 'act.strafe', sig: '()', doc: 'Slide sideways without turning.', snippet: 'act.strafe();' },
  { group: 'Actions', name: 'act.jump', sig: '()', doc: 'Jump (when standing on something).', snippet: 'act.jump();' },
  { group: 'Actions', name: 'act.kick', sig: '()', doc: 'Kick a ball or shove a crate in front of me.', snippet: 'act.kick();' },
  // ---- brain
  { group: 'Brain', name: 'brain.layer', sig: "(units, activation)", doc: "Add a hidden layer. activation: 'relu', 'tanh', 'leaky_relu', 'sigmoid', 'elu', 'swish', 'softplus' or 'linear'.", snippet: "brain.layer(${64}, '${tanh}');" },
  { group: 'Brain', name: 'brain.layers', sig: '([units…], activation)', doc: 'Several hidden layers at once, e.g. brain.layers([64, 64], "tanh").', snippet: "brain.layers([${64}, ${64}], '${tanh}');" },
  // ---- training
  { group: 'Training', name: 'train.ppo', sig: '({ lr, gamma, lambda, clip, entropy, horizon, epochs, minibatch, numEnvs, … })', doc: 'Proximal Policy Optimization, the default. Also: valueCoef, maxGradNorm, normObs, normReward, optimizer, seed, lrSchedule (constant/linear/cosine), lrEnd, entropyEnd, scheduleSteps, stopSteps, targetSuccess, targetReward, stopMinutes.', snippet: 'train.ppo({ lr: ${3e-4}, numEnvs: ${16} });' },
  { group: 'Training', name: 'train.dqn', sig: '({ lr, gamma, bufferSize, batchSize, epsStart, epsEnd, epsDecay, targetUpdate, learnStart, trainEvery, double, … })', doc: 'Deep Q-Network: learns the value of every action combination. Discrete actions only.', snippet: 'train.dqn({ lr: ${5e-4} });' },
  { group: 'Training', name: 'train.reinforce', sig: '({ lr, gamma, episodesPerUpdate, baseline, entropy, … })', doc: 'Vanilla policy gradient on whole episodes.', snippet: 'train.reinforce({ lr: ${1e-3} });' },
  { group: 'Training', name: 'train.evolution', sig: '({ population, eliteFrac, mutation, mutationDecay, episodesPerEval, … })', doc: 'Neuroevolution: a genetic algorithm, no gradients.', snippet: 'train.evolution({ population: ${48} });' },
  { group: 'Training', name: 'train.custom', sig: '({ setup, act, observe, update, numEnvs, … })', doc: 'Your OWN learning algorithm. setup(ctx) builds this.policy (the brain), act(obs, ctx) returns a choice per action, observe(t, ctx) sees every transition, update(ctx) learns (return true when the brain changed). See "New code project: your own algorithm" for a full REINFORCE.', snippet: 'train.custom({\n  numEnvs: 16,\n  setup(ctx) {\n    this.policy = ctx.mlp([32, 32], \'tanh\');\n    this.opt = ctx.adam(this.policy, 0.003);\n  },\n  act(obs, ctx) {\n    return ctx.sample(this.policy.forward(obs));\n  },\n  observe(t, ctx) {\n    ${}\n  },\n  update(ctx) {\n    return false;\n  },\n});' },
  // ---- world
  { group: 'World', name: 'world.physics', sig: '({ gravity, friction, airDrag, npcSpeed, kickPower, ballMass, crateMass, bounce, decisionTicks })', doc: 'Change the physics of the training worlds. decisionTicks = physics ticks (1/60 s) between brain decisions (3 = 20 decisions per second).', snippet: 'world.physics({ gravity: ${9.8} });' },
  { group: 'World', name: 'world.add', sig: "(type, { x, z, y })", doc: 'Place an object: coin, flag, trophy, ball, crate, pad, runner, chaser or checkpoint (checkpoints take an index).', snippet: "world.add('${coin}', { x: ${-28}, z: ${4} });" },
  { group: 'World', name: 'world.wall', sig: "({ x, z, w, d, h, yaw, tag })", doc: "Place a wall (or tag: 'ramp', 'lava', 'ice', 'bouncy', 'spike'). yaw in degrees.", snippet: 'world.wall({ x: ${-28}, z: ${6}, w: ${6}, d: ${0.5}, h: ${1.6} });' },
  { group: 'World', name: 'world.clearZone', sig: '(zone?)', doc: "Remove a zone's built-in objects (default: the robot's zone) to build your own level.", snippet: 'world.clearZone();' },
  // ---- events
  { group: 'Events', name: 'on.start', sig: '(() => { … })', doc: 'Runs at the start of every episode.', snippet: 'on.start(() => {\n  ${}\n});' },
  { group: 'Events', name: 'on.step', sig: '(() => { … })', doc: 'Runs after every decision (20× per second). Shaping rewards go here.', snippet: 'on.step(() => {\n  ${}\n});' },
  { group: 'Events', name: 'on.touch', sig: '(tag, (obj) => { … })', doc: 'Runs when I start touching something with that tag. obj = the object (or null for walls, lava…).', snippet: "on.touch('${flag}', (obj) => {\n  ${api.reward(1, 'goal');}\n});" },
  { group: 'Events', name: 'on.pair', sig: '(a, b, (objA, objB) => { … })', doc: 'Runs when thing a starts touching thing b (e.g. ball enters goal).', snippet: "on.pair('${ball}', '${goal}', (ball, goal) => {\n  ${}\n});" },
  { group: 'Events', name: 'on.fall', sig: '(() => { … })', doc: 'Runs when I fall off the world (the episode then ends).', snippet: 'on.fall(() => {\n  ${}\n});' },
  { group: 'Events', name: 'on.end', sig: '(() => { … })', doc: 'Runs once when an episode ends (success, failure or time-out).', snippet: 'on.end(() => {\n  ${}\n});' },
  // ---- api: rewards
  { group: 'api: rewards & episode', name: 'api.reward', sig: '(x, label?)', doc: 'Give reward (negative = punishment). The label groups it in the "where does the reward come from" chart.', snippet: "api.reward(${1}, '${label}');" },
  { group: 'api: rewards & episode', name: 'api.penalty', sig: '(x, label?)', doc: 'Same as api.reward(-x).', snippet: "api.penalty(${0.01}, '${label}');" },
  { group: 'api: rewards & episode', name: 'api.end', sig: '(success)', doc: 'End the episode now. true = success ✅, false = failure.', snippet: 'api.end(${true});' },
  { group: 'api: rewards & episode', name: 'api.steps', sig: '()', doc: 'Steps since the episode started.' },
  { group: 'api: rewards & episode', name: 'api.maxSteps', sig: '()', doc: 'Episode length in steps.' },
  { group: 'api: rewards & episode', name: 'api.timeLeft', sig: '()', doc: '1 at the start of the episode, 0 at the end.' },
  { group: 'api: rewards & episode', name: 'api.time', sig: '()', doc: 'Seconds since the episode started.' },
  { group: 'api: rewards & episode', name: 'api.episode', sig: '()', doc: 'How many episodes this world has started.' },
  { group: 'api: rewards & episode', name: 'api.episodeReward', sig: '()', doc: 'Reward collected so far this episode.' },
  { group: 'api: rewards & episode', name: 'api.log', sig: '(message)', doc: 'Write to the console (from the first training world and the 3D view). console.log works too.' },
  { group: 'api: rewards & episode', name: 'api.note', sig: '(message)', doc: 'Add a line to the Events list of the Senses tab.' },
  // ---- api: measuring
  { group: 'api: measuring', name: 'api.closer', sig: '(tag)', doc: 'Metres I got closer to the nearest tag this step (negative = moved away). The classic shaping reward.', snippet: "api.closer('${flag}')" },
  { group: 'api: measuring', name: 'api.dist', sig: '(tag)', doc: 'Metres to the nearest tag (999 if none).', snippet: "api.dist('${flag}')" },
  { group: 'api: measuring', name: 'api.angle', sig: '(tag)', doc: 'Degrees to the nearest tag: 0 = ahead, negative = left, positive = right.', snippet: "api.angle('${flag}')" },
  { group: 'api: measuring', name: 'api.facing', sig: '(tag, degrees)', doc: 'True if the tag is within ±degrees of straight ahead.', snippet: "api.facing('${flag}', ${20})" },
  { group: 'api: measuring', name: 'api.nearest', sig: '(tag)', doc: 'The nearest tag as { x, y, z, dist, angle }, or null.', snippet: "api.nearest('${coin}')" },
  { group: 'api: measuring', name: 'api.pairDist', sig: '(a, b)', doc: 'Distance between two things (e.g. ball and goal).', snippet: "api.pairDist('${ball}', '${goal}')" },
  { group: 'api: measuring', name: 'api.pairCloser', sig: '(a, b)', doc: 'How much closer a got to b this step.', snippet: "api.pairCloser('${ball}', '${goal}')" },
  { group: 'api: measuring', name: 'api.speed', sig: '()', doc: 'Horizontal speed in m/s.' },
  { group: 'api: measuring', name: 'api.fwdSpeed', sig: '()', doc: 'Speed in the direction I face (negative when reversing).' },
  { group: 'api: measuring', name: 'api.position', sig: '()', doc: 'My position { x, y, z } in metres.' },
  { group: 'api: measuring', name: 'api.velocity', sig: '()', doc: 'My velocity { x, y, z } in m/s.' },
  { group: 'api: measuring', name: 'api.heading', sig: '()', doc: 'Compass heading in degrees (0 = north, 90 = east).' },
  { group: 'api: measuring', name: 'api.height', sig: '()', doc: 'Metres above sea level.' },
  { group: 'api: measuring', name: 'api.dHeight', sig: '()', doc: 'Height gained this step (positive when climbing).' },
  { group: 'api: measuring', name: 'api.grounded', sig: '()', doc: 'True when standing on something.' },
  { group: 'api: measuring', name: 'api.groundTag', sig: '()', doc: "What I'm standing on ('ground', 'ice', 'ramp', …) or '' in the air." },
  { group: 'api: measuring', name: 'api.touching', sig: '(tag)', doc: 'True while touching the tag (not just the first moment).', snippet: "api.touching('${lava}')" },
  { group: 'api: measuring', name: 'api.ray', sig: '(angleDeg, range)', doc: 'Cast a ray: angle 0 = ahead, + = right. Returns { dist, tag } or null.', snippet: 'api.ray(${0}, ${20})' },
  { group: 'api: measuring', name: 'api.count', sig: '(tag)', doc: 'How many of that object are still visible.', snippet: "api.count('${coin}')" },
  { group: 'api: measuring', name: 'api.newArea', sig: '()', doc: 'True when I step onto a 2×2 m tile I have not visited this episode (curiosity!).' },
  { group: 'api: measuring', name: 'api.areasVisited', sig: '()', doc: 'How many tiles I visited this episode.' },
  { group: 'api: measuring', name: 'api.lastAction', sig: "(axis)", doc: "The value I chose for an action last step: 'move', 'turn', 'strafe', 'jump', 'kick'.", snippet: "api.lastAction('${move}')" },
  { group: 'api: measuring', name: 'api.random', sig: '(a, b)', doc: 'A random number between a and b (reproducible with the seed).' },
  { group: 'api: measuring', name: 'api.zone', sig: '()', doc: 'The id of the zone I train in.' },
  // ---- api: objects & world
  { group: 'api: objects & world', name: 'api.respawn', sig: '(obj)', doc: 'Move an object (e.g. the touched coin) to a random spot.', snippet: 'api.respawn(obj);' },
  { group: 'api: objects & world', name: 'api.hide', sig: '(obj)', doc: 'Remove an object until the next episode.', snippet: 'api.hide(obj);' },
  { group: 'api: objects & world', name: 'api.randomize', sig: '(tag)', doc: 'Move every object with that tag to random spots.', snippet: "api.randomize('${flag}');" },
  { group: 'api: objects & world', name: 'api.resetAll', sig: '(tag)', doc: 'Put every object with that tag back where it started.', snippet: "api.resetAll('${ball}');" },
  { group: 'api: objects & world', name: 'api.showAll', sig: '(tag)', doc: 'Bring back hidden objects.', snippet: "api.showAll('${coin}');" },
  { group: 'api: objects & world', name: 'api.objects', sig: '(tag?)', doc: 'Every object (of a type), as live objects with x, y, z, vx, vy, vz, active… You can move them!', snippet: "api.objects('${coin}')" },
  { group: 'api: objects & world', name: 'api.teleport', sig: '(x, z, y?)', doc: 'Teleport me (the episode continues).', snippet: 'api.teleport(${0}, ${0});' },
  { group: 'api: objects & world', name: 'api.push', sig: '(x, y, z)', doc: 'Add to my velocity (m/s): a push, a jet, a gust of wind…', snippet: 'api.push(${0}, ${5}, ${0});' },
  { group: 'api: objects & world', name: 'api.respawnMe', sig: '()', doc: 'Teleport me back to the start.', snippet: 'api.respawnMe();' },
  { group: 'api: objects & world', name: 'api.isNext', sig: '(obj)', doc: 'Is this checkpoint the next one in order?', snippet: 'api.isNext(obj)' },
  { group: 'api: objects & world', name: 'api.advance', sig: '()', doc: 'Count the next checkpoint as passed.', snippet: 'api.advance();' },
  { group: 'api: objects & world', name: 'api.checkpoint', sig: '()', doc: 'How many checkpoints were passed (this lap).' },
  { group: 'api: objects & world', name: 'api.laps', sig: '()', doc: 'Laps completed (race track).' },
  { group: 'api: objects & world', name: 'api.addCounter', sig: '(name, n)', doc: 'Add n to a named counter (reset every episode).', snippet: "api.addCounter('${coins}', 1);" },
  { group: 'api: objects & world', name: 'api.counter', sig: '(name)', doc: 'Read a named counter.', snippet: "api.counter('${coins}')" },
  // ---- your own algorithm
  { group: 'Your own algorithm', name: 'ctx.mlp', sig: "(hidden, activation, outSize?, inSize?)", doc: 'A new neural network. By default obs → hidden layers → one logit per choice (the right shape for this.policy). Pass outSize 1 for a critic/value network.', snippet: "ctx.mlp([${32}, ${32}], '${tanh}')" },
  { group: 'Your own algorithm', name: 'ctx.adam', sig: '(net, lr)', doc: 'An Adam optimizer for a network. opt.step() applies the gradients; opt.lr can be changed any time.', snippet: 'ctx.adam(${this.policy}, ${0.003})' },
  { group: 'Your own algorithm', name: 'ctx.sgd', sig: '(net, lr)', doc: 'Plain gradient descent: w ← w − lr·g.', snippet: 'ctx.sgd(${this.policy}, ${0.01})' },
  { group: 'Your own algorithm', name: 'ctx.optimizer', sig: "(net, lr, kind)", doc: "Any optimizer: 'adam', 'sgd', 'momentum' or 'rmsprop'." },
  { group: 'Your own algorithm', name: 'ctx.sample', sig: '(logits, row?)', doc: 'Pick one choice per action, at random, with softmax probabilities.', snippet: 'ctx.sample(${logits})' },
  { group: 'Your own algorithm', name: 'ctx.best', sig: '(logits, row?)', doc: 'The most likely choice per action (no randomness).', snippet: 'ctx.best(${logits})' },
  { group: 'Your own algorithm', name: 'ctx.probs', sig: '(logits, row?)', doc: 'Softmax probabilities, one array per action.' },
  { group: 'Your own algorithm', name: 'ctx.logProb', sig: '(logits, action, row?)', doc: 'log π(a|s): the log-probability of an action (summed over the actions).' },
  { group: 'Your own algorithm', name: 'ctx.entropy', sig: '(logits, row?)', doc: 'How random the policy is in that state.' },
  { group: 'Your own algorithm', name: 'ctx.addLogProbGrad', sig: '(logits, row, action, scale, out)', doc: 'Adds scale · ∂log π(a|s)/∂logits = scale · (onehot(a) − p) into out (for a batch: row i of n). The heart of every policy-gradient method.', snippet: 'ctx.addLogProbGrad(${logits}, ${i}, ${action}, ${scale}, ${dLogits});' },
  { group: 'Your own algorithm', name: 'ctx.addEntropyGrad', sig: '(logits, row, scale, out)', doc: 'Adds scale · ∂H/∂logits: use a negative scale in a loss to reward exploring.' },
  { group: 'Your own algorithm', name: 'ctx.batch', sig: '([obs, obs, …])', doc: 'Stack observations into one array for a batched forward pass: net.forward(ctx.batch(list), list.length).', snippet: 'ctx.batch(${rows})' },
  { group: 'Your own algorithm', name: 'ctx.clipGrad', sig: '(net, maxNorm)', doc: 'Gradient clipping: shrink the gradient if it is longer than maxNorm. Returns the norm before clipping.', snippet: 'ctx.clipGrad(${this.policy}, ${0.5});' },
  { group: 'Your own algorithm', name: 'ctx.worlds', sig: '(() => value)', doc: 'One value per parallel world, e.g. ctx.worlds(() => []) for an episode buffer each.', snippet: 'ctx.worlds(() => ${[]})' },
  { group: 'Your own algorithm', name: 'ctx.log', sig: '({ name: number, … })', doc: 'Record numbers: each one gets a chart in the Training tab and shows up in the Math tab.', snippet: 'ctx.log({ ${loss}: ${0} });' },
  { group: 'Your own algorithm', name: 'ctx.random', sig: '()', doc: 'A random number in [0, 1) (seeded: runs are reproducible). Also ctx.gauss() and ctx.randint(n).' },
  { group: 'Your own algorithm', name: 'ctx.obsSize', sig: '', doc: 'How many numbers each observation has. Also ctx.outputs (logits), ctx.branches (choices per action), ctx.actions (names), ctx.numEnvs, ctx.steps.' },
  { group: 'Your own algorithm', name: 'net.forward', sig: '(x, batch = 1)', doc: 'Run the network on batch inputs laid out one after another. Returns a new Float32Array of batch × outputs.' },
  { group: 'Your own algorithm', name: 'net.backward', sig: '(dOut, batch)', doc: 'Backpropagation: given ∂loss/∂outputs for the batch of the LAST forward call, add ∂loss/∂weights to the gradients. Call net.zeroGrad() first, then opt.step().' },
  { group: 'Your own algorithm', name: 'net.zeroGrad', sig: '()', doc: 'Reset the accumulated gradients to 0.' },
  { group: 'Your own algorithm', name: 'net.copyFrom', sig: '(other)', doc: 'Copy the weights of another network of the same shape (e.g. a DQN target network). net.clone() makes a copy.' },
  // ---- JS
  { group: 'JavaScript', name: 'let / const', sig: '', doc: 'Variables at the top level are shared by all events of one world and survive between episodes (reset them in on.start if needed).' },
  { group: 'JavaScript', name: 'Math', sig: '', doc: 'Math.abs, Math.min, Math.max, Math.sqrt, Math.exp, Math.log, Math.sin, Math.cos, Math.atan2, Math.hypot, Math.PI… (avoid Math.random: use api.random for reproducible training).' },
  { group: 'JavaScript', name: 'loops', sig: '', doc: 'Loops are protected: one that runs more than a million times stops with an error instead of freezing the page.' },
];

/** Tags and zone ids offered inside quotes. */
export const TAG_WORDS = ['flag', 'coin', 'trophy', 'ball', 'crate', 'goal', 'pad', 'checkpoint', 'next_checkpoint', 'runner', 'chaser', 'wall', 'lava', 'ramp', 'tree', 'rock', 'ice', 'bouncy', 'spike', 'ground', 'spawn'];
export const ZONE_WORDS = ['meadow', 'coins', 'maze', 'lava', 'summit', 'race', 'soccer', 'sky', 'forest', 'warehouse', 'tag', 'dodge', 'world'];
export const ACTIVATION_WORDS = ['relu', 'tanh', 'leaky_relu', 'sigmoid', 'elu', 'swish', 'softplus', 'linear'];
