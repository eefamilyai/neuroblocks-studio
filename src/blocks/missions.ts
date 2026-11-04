/**
 * Ready-made block programs ("missions"). Each one is real blocks you can
 * read and change: nothing is hidden. They double as a guided course.
 */

type J = Record<string, unknown>;

const b = (type: string, fields: J = {}, inputs: J = {}): J => ({ type, fields, inputs });
const stack = (...bs: J[]): J => {
  for (let i = bs.length - 2; i >= 0; i--) bs[i].next = { block: bs[i + 1] };
  return bs[0];
};
const numIn = (v: number) => ({ shadow: { type: 'math_number', fields: { NUM: v } } });
const blkIn = (block: J, d = 1) => ({ block, shadow: { type: 'math_number', fields: { NUM: d } } });
const mul = (a: J, k: number): J => b('math_arithmetic', { OP: 'MULTIPLY' }, { A: blkIn(a), B: numIn(k) });
const reward = (v: number | J) => b('nb_reward', {}, { VALUE: typeof v === 'number' ? numIn(v) : blkIn(v) });
const penalty = (v: number | J) => b('nb_penalty', {}, { VALUE: typeof v === 'number' ? numIn(v) : blkIn(v) });
const end = (success: boolean) => b('nb_end', { RESULT: String(success) });
const ifThen = (cond: J, ...then: J[]) => b('controls_if', {}, { IF0: { block: cond }, DO0: { block: stack(...then) } });
const closer = (tag: string) => b('nb_closer', { TAG: tag });

interface RobotOpts {
  spawn?: 'zone' | 'random' | 'point';
  facing?: 'default' | 'random' | 'fixed';
  steps?: number;
  jitter?: number;
  name?: string;
  traits?: J[];
}
const robot = (zone: string, o: RobotOpts = {}) =>
  stack(
    b('nb_robot', {
      NAME: o.name ?? 'Robo',
      ZONE: zone,
      SPAWN: o.spawn ?? 'zone',
      FACING: o.facing ?? 'default',
      X: 0,
      Z: 0,
      ANGLE: 0,
      JITTER: o.jitter ?? 1.5,
      STEPS: o.steps ?? 400,
    }),
    ...(o.traits ?? []),
  );
const senses = (...items: J[]) => stack(b('nb_senses'), ...items);
const rays = (count: number, spread: number, range: number, tilt: number, detect: string[]) =>
  b('nb_sense_rays', { COUNT: count, SPREAD: spread, RANGE: range, TILT: tilt }, detect.length ? { DETECT: { block: stack(...detect.map((t) => b('nb_detect', { TAG: t }))) } } : {});
const target = (tag: string, range = 40) => b('nb_sense_target', { TAG: tag, RANGE: range });
const actions = (mode: 'discrete' | 'continuous', ...items: J[]) => stack(b('nb_actions', { MODE: mode }), ...items);
const move = (dir: 'both' | 'one' = 'both') => b('nb_act_move', { DIR: dir });
const turn = () => b('nb_act_turn');
const brain = (...layers: [number, string][]) => stack(b('nb_brain'), ...layers.map(([u, a]) => b('nb_layer', { UNITS: u, ACT: a })));
const ppo = (o: J = {}) =>
  b('nb_train_ppo', { WORLDS: 16, LR: 0.0003, GAMMA: 0.99, LAMBDA: 0.95, CLIP: 0.2, ENT: 0.01, HORIZON: 128, EPOCHS: 4, MB: 256, OPT: 'adam', NORMOBS: true, NORMREW: true, STOP: 0, SEED: 1, ...o });
const dqn = (o: J = {}) =>
  b('nb_train_dqn', { WORLDS: 16, LR: 0.0005, GAMMA: 0.99, BUFFER: 50000, BATCH: 64, EPS0: 1, EPS1: 0.05, EPSDECAY: 40000, TARGET: 4000, DOUBLE: true, LEARNSTART: 1000, TRAINEVERY: 4, OPT: 'adam', STOP: 0, SEED: 1, ...o });
const reinforce = (o: J = {}) =>
  b('nb_train_reinforce', { WORLDS: 16, LR: 0.003, GAMMA: 0.99, EPISODES: 16, BASELINE: true, ENT: 0.01, NORMOBS: true, OPT: 'adam', STOP: 0, SEED: 1, ...o });
const evolution = (o: J = {}) => b('nb_train_evolution', { POP: 32, WORLDS: 16, ELITE: 0.2, SIGMA: 0.1, DECAY: 0.995, EVALS: 1, STOP: 0, SEED: 1, ...o });
const onStart = (...s: J[]) => stack(b('nb_on_start'), ...s);
const onStep = (...s: J[]) => stack(b('nb_on_step'), ...s);
const onTouch = (tag: string, ...s: J[]) => stack(b('nb_on_touch', { TAG: tag }), ...s);
const onPair = (a: string, bb: string, ...s: J[]) => stack(b('nb_on_pair', { A: a, B: bb }), ...s);
const onFall = (...s: J[]) => stack(b('nb_on_fall'), ...s);

function workspace(...tops: J[]): J {
  return { blocks: { languageVersion: 0, blocks: tops.map((t, i) => ({ ...t, x: 0, y: i * 10 })) } };
}

export interface Mission {
  id: string;
  title: string;
  emoji: string;
  zone: string;
  level: 1 | 2 | 3 | 4 | 5;
  time: string;
  algo: string;
  summary: string;
  /** what this mission teaches (shown in the mission browser) */
  learn: string[];
  /** a hint about what to watch for / try next */
  tryNext: string;
  workspace: () => J;
}

const reachFlagParts = (train: J, extra: Partial<RobotOpts> = {}) => [
  robot('meadow', { spawn: 'random', facing: 'random', steps: 300, ...extra }),
  senses(target('flag', 40), b('nb_sense_velocity')),
  actions('discrete', move(), turn()),
  brain([32, 'tanh'], [32, 'tanh']),
  train,
  onStart(b('nb_scatter', { TAG: 'flag' })),
  onStep(reward(mul(closer('flag'), 0.5)), penalty(0.002)),
  onTouch('flag', reward(1), end(true)),
];

export const MISSIONS: Mission[] = [
  {
    id: 'reach-flag',
    title: 'First Steps: Reach the Flag',
    emoji: '🚩',
    zone: 'meadow',
    level: 1,
    time: '≈ 30 s',
    algo: 'PPO',
    summary: 'The "hello world" of robot learning. The robot knows where the flag is, but not how to drive. Watch it go from spinning randomly to beelining.',
    learn: ['Observations → network → actions', 'Reward shaping ("how much closer did I get?")', 'Why we randomise the flag each episode'],
    tryNext: 'Delete the 🍬 "closer" reward and keep only the +1 for touching the flag. Learning gets MUCH slower: that is the sparse-reward problem.',
    workspace: () => workspace(...reachFlagParts(ppo())),
  },
  {
    id: 'reach-flag-dqn',
    title: 'Reach the Flag with DQN',
    emoji: '🎯',
    zone: 'meadow',
    level: 2,
    time: '≈ 1 min',
    algo: 'DQN',
    summary: 'Same task, different brain: the network now predicts the VALUE of every action (Q-values) and picks the best one.',
    learn: ['Q-learning & the Bellman equation', 'ε-greedy exploration', 'Replay memory and target networks'],
    tryNext: 'Set "sync target net every" to 200 and watch training become unstable: the target keeps moving!',
    workspace: () => workspace(...reachFlagParts(dqn())),
  },
  {
    id: 'reach-flag-evo',
    title: 'Reach the Flag by Evolution',
    emoji: '🧬',
    zone: 'meadow',
    level: 1,
    time: '≈ 20 s',
    algo: 'Neuroevolution',
    summary: 'No gradients at all. 32 random brains compete; the best ones have mutated children. Survival of the fittest, for neural nets.',
    learn: ['Genetic algorithms', 'Fitness = total reward', 'Exploration through mutation'],
    tryNext: 'Turn on 👻 ghosts to see the whole population at once. Try mutation σ = 0.5 (wild children) vs 0.02 (tiny changes).',
    workspace: () => workspace(...reachFlagParts(evolution())),
  },
  {
    id: 'reach-flag-reinforce',
    title: 'Reach the Flag with REINFORCE',
    emoji: '📜',
    zone: 'meadow',
    level: 2,
    time: '≈ 1 min',
    algo: 'REINFORCE',
    summary: 'The original policy-gradient algorithm from 1992. Simple and elegant, but noisy: compare its learning curve with PPO.',
    learn: ['Policy gradients: ∇log π · return', 'Baselines reduce variance', 'Why PPO was invented'],
    tryNext: 'Untick "baseline (critic)" and see how much noisier the reward curve becomes.',
    workspace: () => workspace(...reachFlagParts(reinforce())),
  },
  {
    id: 'imitation',
    title: 'Teach by Example (Imitation)',
    emoji: '🎮',
    zone: 'meadow',
    level: 1,
    time: '≈ 2 min',
    algo: 'Behaviour cloning',
    summary: 'YOU drive first. Press 🎮 Drive, hit ⏺ Record and steer to the flag a few times with WASD/arrows. Then press 🎓 Learn from me: the network copies your driving with supervised learning.',
    learn: ['Supervised learning vs reinforcement learning', 'Cross-entropy loss & accuracy', 'Distribution shift: the copy only knows situations you showed it'],
    tryNext: 'After cloning, press ▶ Train: PPO starts from your copied brain and polishes it (that is how many real robots are trained).',
    workspace: () => workspace(...reachFlagParts(ppo())),
  },
  {
    id: 'coin-collector',
    title: 'Coin Collector',
    emoji: '🪙',
    zone: 'coins',
    level: 2,
    time: '≈ 2 min',
    algo: 'PPO',
    summary: 'Collect as many coins as possible in 30 seconds while dodging rocks and bushes. Coins respawn when collected.',
    learn: ['Ray-cast vision (like LIDAR)', 'Dense rewards from repeated goals', 'Penalties for bumping into things'],
    tryNext: 'Remove the 🧭 "where is the nearest coin" sense. Can it still find coins with its eyes alone? (Yes, but slower.)',
    workspace: () =>
      workspace(
        robot('coins', { spawn: 'random', facing: 'random', steps: 600 }),
        senses(rays(9, 180, 15, 0, ['coin', 'rock', 'tree', 'wall']), target('coin', 30), b('nb_sense_velocity')),
        actions('discrete', move(), turn()),
        brain([64, 'tanh'], [64, 'tanh']),
        ppo(),
        onTouch('coin', reward(1), b('nb_respawn')),
        onTouch('rock', penalty(0.05)),
        onTouch('tree', penalty(0.05)),
        onTouch('wall', penalty(0.05)),
        onStep(reward(mul(closer('coin'), 0.1)), penalty(0.001)),
      ),
  },
  {
    id: 'coin-evo',
    title: 'Coin Collector: Evolution vs PPO',
    emoji: '🧪',
    zone: 'coins',
    level: 2,
    time: '≈ 1 min',
    algo: 'Neuroevolution',
    summary: 'The same coin task solved by evolution. Evolution is great at fast, reactive behaviours: compare the curves!',
    learn: ['Comparing algorithms on the same task', 'Deterministic vs stochastic policies'],
    tryNext: 'Bump the population to 96. More brains = better search but slower generations.',
    workspace: () =>
      workspace(
        robot('coins', { spawn: 'random', facing: 'random', steps: 600 }),
        senses(rays(7, 150, 12, 0, ['coin', 'rock', 'tree', 'wall']), target('coin', 30), b('nb_sense_velocity')),
        actions('discrete', move(), turn()),
        brain([32, 'tanh']),
        evolution({ POP: 48 }),
        onTouch('coin', reward(1), b('nb_respawn')),
        onStep(penalty(0.001)),
      ),
  },
  {
    id: 'forest-flag',
    title: 'Find the Flag in the Forest',
    emoji: '🌲',
    zone: 'forest',
    level: 2,
    time: '≈ 1 min',
    algo: 'PPO',
    summary: 'Navigate a dense forest to a randomly placed flag. The robot must combine its compass with its eyes to steer around trunks.',
    learn: ['Combining sensors', 'Obstacle avoidance emerges from penalties', 'Generalisation: random flag positions'],
    tryNext: 'Open 🗺️ Maps → value map: see which places the critic thinks are "good".',
    workspace: () =>
      workspace(
        robot('forest', { spawn: 'zone', facing: 'random', steps: 500 }),
        senses(rays(9, 150, 12, 0, ['tree', 'flag', 'wall']), target('flag', 40), b('nb_sense_velocity')),
        actions('discrete', move(), turn()),
        brain([64, 'tanh'], [64, 'tanh']),
        ppo(),
        onStart(b('nb_scatter', { TAG: 'flag' })),
        onStep(reward(mul(closer('flag'), 0.5)), penalty(0.002)),
        onTouch('flag', reward(1), end(true)),
        onTouch('tree', penalty(0.05)),
      ),
  },
  {
    id: 'tag',
    title: 'Tag! Catch the Runner',
    emoji: '🐰',
    zone: 'tag',
    level: 2,
    time: '≈ 3 min',
    algo: 'PPO',
    summary: 'A runner bot flees from you. It is a bit slower than you but dodges: corner it!',
    learn: ['Moving targets', 'Pursuit strategies emerge', 'Episode endings as rewards'],
    tryNext: 'Add 💨 "my velocity" for the runner using a custom sense… or add the 🧭 compass and compare.',
    workspace: () =>
      workspace(
        robot('tag', { spawn: 'random', facing: 'random', steps: 600 }),
        senses(target('runner', 40), rays(7, 180, 15, 0, ['wall', 'rock', 'runner']), b('nb_sense_velocity')),
        actions('discrete', move(), turn()),
        brain([64, 'tanh'], [64, 'tanh']),
        ppo(),
        onStep(reward(mul(closer('runner'), 0.3)), penalty(0.002)),
        onTouch('runner', reward(2), end(true)),
      ),
  },
  {
    id: 'lava-crossing',
    title: 'Lava Crossing',
    emoji: '🌋',
    zone: 'lava',
    level: 3,
    time: '≈ 1 min',
    algo: 'PPO',
    summary: 'Narrow stone bridges over lava. The robot must feel the ground ahead (like a cane) to avoid stepping off the path.',
    learn: ['Safety & terminal penalties', 'Ground probes (edge detection)', 'Why the discount γ matters for long tasks'],
    tryNext: 'Lower γ to 0.9: the robot becomes short-sighted and gets reckless near edges.',
    workspace: () =>
      workspace(
        robot('lava', { spawn: 'zone', facing: 'default', steps: 600, jitter: 1 }),
        senses(target('flag', 50), b('nb_sense_probe', { COUNT: 5, DIST: 1.6 }), b('nb_sense_velocity')),
        actions('discrete', move(), turn()),
        brain([64, 'tanh'], [64, 'tanh']),
        ppo(),
        onStep(reward(mul(closer('flag'), 0.5)), penalty(0.002)),
        onTouch('flag', reward(2), end(true)),
        onTouch('lava', penalty(1), end(false)),
      ),
  },
  {
    id: 'soccer',
    title: 'Soccer Star',
    emoji: '⚽',
    zone: 'soccer',
    level: 3,
    time: '≈ 6 min',
    algo: 'PPO',
    summary: 'Dribble (or kick!) the ball into the goal. Two shaping rewards: get to the ball, and move the ball toward the goal.',
    learn: ['Multi-part reward design', 'Relational senses (ball → goal)', 'Reward hacking: watch for cheap tricks'],
    tryNext: 'Set the "closer to ball" reward to 1.0 and watch the robot hug the ball instead of scoring. That is reward hacking!',
    workspace: () =>
      workspace(
        robot('soccer', { spawn: 'random', facing: 'random', steps: 600 }),
        senses(target('ball', 40), b('nb_sense_pair', { A: 'ball', B: 'goal' }), rays(5, 120, 12, 0, ['wall', 'ball', 'goal']), b('nb_sense_velocity')),
        actions('discrete', move(), turn(), b('nb_act_kick')),
        brain([64, 'tanh'], [64, 'tanh']),
        ppo(),
        onStart(b('nb_scatter', { TAG: 'ball' })),
        onStep(reward(mul(closer('ball'), 0.1)), reward(mul(b('nb_pair_closer', { A: 'ball', B: 'goal' }), 1)), penalty(0.001)),
        onPair('ball', 'goal', reward(5), end(true)),
      ),
  },
  {
    id: 'maze-runner',
    title: 'Maze Runner',
    emoji: '🌀',
    zone: 'maze',
    level: 4,
    time: '≈ 8 min',
    algo: 'PPO',
    summary: 'Reach the trophy in the hedge maze. Purple checkpoints mark the route; following them breaks one huge problem into many small ones.',
    learn: ['Sparse rewards & exploration', 'Curriculum via checkpoints', 'Wall-following behaviour'],
    tryNext: 'Replace the checkpoint rewards with 🍬 reward 0.05 when "entered a new area?" (curiosity!) and see if it can still solve it.',
    workspace: () =>
      workspace(
        robot('maze', { spawn: 'zone', facing: 'default', steps: 1500, jitter: 0.5 }),
        senses(rays(7, 180, 10, 0, ['wall']), target('next_checkpoint', 30), target('trophy', 60), b('nb_sense_velocity')),
        actions('discrete', move(), turn()),
        brain([64, 'tanh'], [64, 'tanh']),
        ppo({ HORIZON: 256 }),
        onStep(reward(mul(closer('next_checkpoint'), 0.3)), penalty(0.002)),
        onTouch('checkpoint', ifThen(b('nb_is_next'), reward(0.5), b('nb_advance'))),
        onTouch('trophy', reward(3), end(true)),
        onTouch('wall', penalty(0.02)),
      ),
  },
  {
    id: 'race-laps',
    title: 'Race Track (smooth control)',
    emoji: '🏎️',
    zone: 'race',
    level: 3,
    time: '≈ 2 min',
    algo: 'PPO · continuous',
    summary: 'Drive laps through 8 checkpoint gates using SMOOTH controls: the network outputs a throttle and steering value, like a joystick.',
    learn: ['Continuous actions (Gaussian policies)', 'Exploration noise σ shrinking over time', 'Racing lines emerge'],
    tryNext: 'Watch "action std" in the 📈 charts: the robot gets more confident (less random) as it learns.',
    workspace: () =>
      workspace(
        robot('race', { spawn: 'zone', facing: 'default', steps: 1000, jitter: 1 }),
        senses(rays(9, 180, 20, 0, ['wall']), target('next_checkpoint', 30), b('nb_sense_velocity')),
        actions('continuous', move('one'), turn()),
        brain([64, 'tanh'], [64, 'tanh']),
        ppo(),
        onStep(reward(mul(closer('next_checkpoint'), 0.2))),
        onTouch('checkpoint', ifThen(b('nb_is_next'), reward(1), b('nb_advance'))),
        onTouch('wall', penalty(0.1)),
      ),
  },
  {
    id: 'hill-climb',
    title: 'Climb the Summit',
    emoji: '⛰️',
    zone: 'summit',
    level: 3,
    time: '≈ 2 min',
    algo: 'PPO',
    summary: 'Find the spiral of ramps up a three-tier mountain to the flag on top. Checkpoints at the foot of each ramp break the climb into steps; height gains are rewarded too.',
    learn: ['Rewarding progress (height)', 'Local optima: the cliff is closer than the ramp', 'Breaking a long task into sub-goals'],
    tryNext: 'Remove the checkpoint rewards and keep only "height gained". Does it still find the ramps? (Much harder: a classic exploration problem.)',
    workspace: () =>
      workspace(
        robot('summit', { spawn: 'zone', facing: 'default', steps: 900, jitter: 2 }),
        senses(target('next_checkpoint', 30), target('flag', 40), rays(7, 150, 10, 12, ['rock', 'ramp', 'wall']), b('nb_sense_height'), b('nb_sense_velocity')),
        actions('discrete', move(), turn()),
        brain([64, 'tanh'], [64, 'tanh']),
        ppo({ HORIZON: 256 }),
        onStep(reward(mul(closer('next_checkpoint'), 0.3)), reward(mul(b('nb_height_gain'), 1)), penalty(0.002)),
        onTouch('checkpoint', ifThen(b('nb_is_next'), reward(0.5), b('nb_advance'))),
        onTouch('flag', reward(5), end(true)),
      ),
  },
  {
    id: 'parkour',
    title: 'Sky Parkour (jumping)',
    emoji: '☁️',
    zone: 'sky',
    level: 4,
    time: '≈ 3 min',
    algo: 'PPO',
    summary: 'Jump across floating platforms to reach the flag. One wrong step and it falls into the void.',
    learn: ['Timing actions (jump at the edge)', 'Terminal states', 'Exploration vs. safety'],
    tryNext: 'Increase the entropy bonus to 0.03 if the robot gets stuck being too careful.',
    workspace: () =>
      workspace(
        robot('sky', { spawn: 'zone', facing: 'default', steps: 600, jitter: 1 }),
        senses(target('flag', 50), b('nb_sense_probe', { COUNT: 3, DIST: 1.5 }), b('nb_sense_probe', { COUNT: 3, DIST: 3 }), b('nb_sense_grounded'), b('nb_sense_height'), b('nb_sense_velocity')),
        actions('discrete', move(), turn(), b('nb_act_jump')),
        brain([64, 'tanh'], [64, 'tanh']),
        ppo({ ENT: 0.02 }),
        onStep(reward(mul(closer('flag'), 0.5)), reward(mul(b('nb_height_gain'), 0.5)), penalty(0.002)),
        onTouch('flag', reward(3), end(true)),
        onFall(penalty(1)),
      ),
  },
  {
    id: 'dodge',
    title: 'Dodge & Collect',
    emoji: '👾',
    zone: 'dodge',
    level: 3,
    time: '≈ 6 min',
    algo: 'PPO',
    summary: 'Two chaser bots hunt the robot while it collects coins. Greed vs. survival: a trade-off the network has to balance.',
    learn: ['Competing objectives', '360° vision', 'Survival bonuses'],
    tryNext: 'Make the chaser penalty smaller (0.5) and watch the robot take bigger risks for coins.',
    workspace: () =>
      workspace(
        robot('dodge', { spawn: 'random', facing: 'random', steps: 800 }),
        senses(rays(12, 330, 12, 0, ['chaser', 'coin', 'wall']), target('coin', 30), target('chaser', 30), b('nb_sense_velocity')),
        actions('discrete', move(), turn()),
        brain([64, 'tanh'], [64, 'tanh']),
        ppo(),
        onTouch('coin', reward(1), b('nb_respawn')),
        onTouch('chaser', penalty(2), end(false)),
        onStep(reward(0.002)),
      ),
  },
  {
    id: 'crate-push',
    title: 'Warehouse: Push a Crate onto a Pad',
    emoji: '📦',
    zone: 'warehouse',
    level: 5,
    time: '≈ 6 min',
    algo: 'PPO',
    summary: 'A real puzzle: get behind a crate and push it onto any green pad. Expect long training: a great test of your reward design skills.',
    learn: ['Hard exploration', 'Indirect control (moving an object)', 'Patience: some tasks need millions of steps'],
    tryNext: 'Start with the crate closer to the pad (🧱 Build mode) to create a curriculum, then move it back.',
    workspace: () =>
      workspace(
        robot('warehouse', { spawn: 'random', facing: 'random', steps: 800 }),
        senses(target('crate', 40), b('nb_sense_pair', { A: 'crate', B: 'pad' }), rays(7, 180, 12, 0, ['wall', 'crate', 'pad']), b('nb_sense_velocity')),
        actions('discrete', move(), turn()),
        brain([64, 'tanh'], [64, 'tanh']),
        ppo({ HORIZON: 256 }),
        onStep(reward(mul(closer('crate'), 0.05)), reward(mul(b('nb_pair_closer', { A: 'crate', B: 'pad' }), 1)), penalty(0.001)),
        onPair('crate', 'pad', reward(5), end(true)),
      ),
  },
  {
    id: 'blank',
    title: 'Blank Canvas',
    emoji: '✏️',
    zone: 'meadow',
    level: 1,
    time: 'you decide',
    algo: 'PPO',
    summary: 'The bare minimum: a robot, a sense, two actions, a brain and empty event hats. Build your own experiment!',
    learn: ['Designing your own task from scratch'],
    tryNext: 'Pick a zone, decide what "good" means, and express it as rewards.',
    workspace: () =>
      workspace(
        robot('meadow', { spawn: 'zone', steps: 400 }),
        senses(b('nb_sense_velocity')),
        actions('discrete', move(), turn()),
        brain([32, 'tanh'], [32, 'tanh']),
        ppo(),
        b('nb_on_start'),
        b('nb_on_step'),
      ),
  },
];

export function getMission(id: string): Mission | undefined {
  return MISSIONS.find((m) => m.id === id);
}
