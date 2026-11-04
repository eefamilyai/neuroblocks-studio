/** Dialogs: welcome, missions, settings, shortcuts, export formats, code trust, rename. */
import { MISSIONS } from '../blocks/missions.ts';
import type { Mission } from '../blocks/missions.ts';
import type { WorldDef } from '../sim/types.ts';
import type { Quality } from '../render/viewport.ts';
import { btn, h, modal, seg } from '../ui/dom.ts';
import { icon } from '../ui/icons.ts';
import type { IconName } from '../ui/icons.ts';
import { missionsDone, setWelcomeSeen } from '../project/storage.ts';
import { LearnPanel } from '../ui/learnPanel.ts';

export function openWelcome(o: { startFirst: () => void; missions: () => void; code: () => void; language: () => void }): void {
  setWelcomeSeen();
  const step = (n: IconName, title: string, text: string) => h('div', { class: 'welcome-step' }, h('span', { class: 'n' }, icon(n)), h('div', null, h('b', null, title), h('p', null, text)));
  const m = modal(
    'Welcome to NeuroBlocks Studio',
    h(
      'div',
      null,
      h('p', { class: 'dim', style: { margin: '0', lineHeight: '1.55' } }, "Like Scratch, but you program a robot's LEARNING, not its moves. You decide what it senses, what it can do and what it gets rewarded for. A neural network figures out the rest by trial and error, right in front of you."),
      h(
        'div',
        { class: 'welcome-grid' },
        step('puzzle', '1 · The program', 'Left. Blocks (or JavaScript code): where the robot lives, what it senses (network inputs), what it can do (outputs), its brain, how it learns and what earns rewards.'),
        step('map', '2 · The world', 'Middle. An island with 12 training zones: mazes, lava, soccer, parkour… Click a zone label to train there, or build your own course.'),
        step('play', '3 · Train & watch', '16 copies of your robot train in parallel (the ghosts). The robot you see always uses the newest brain. Take the controls with "You".'),
        step('chart', '4 · Under the hood', 'Right. Learning curves, every neuron and weight, the exact sensor numbers, the maths, experiments. Export the trained brain as JSON, ONNX or PyTorch.'),
      ),
      h(
        'button',
        { class: 'welcome-llm', type: 'button', onclick: () => (m.close(), o.language()) },
        h('span', { class: 'n' }, icon('chat')),
        h('div', null, h('b', null, 'New: build mini language models'), h('p', null, 'Switch to Language (top left) to design and train your own tiny GPT on stories, names, sums or your own text, and see its tokens, attention and embeddings.')),
      ),
      h('p', { class: 'small muted', style: { margin: '0' } }, 'Everything (neural networks, backpropagation, PPO, DQN, evolution, transformers, physics) is written from scratch in TypeScript and runs in your browser. No servers, no ML libraries, nothing hidden.'),
    ),
    {
      foot: [
        btn({ icon: 'code', label: 'Start in code', cls: 'ghost', onclick: () => (m.close(), o.code()) }),
        btn({ icon: 'target', label: 'Browse missions', onclick: () => (m.close(), o.missions()) }),
        btn({ icon: 'play', label: 'Start the first mission', cls: 'primary', onclick: () => (m.close(), o.startFirst()) }),
      ],
    },
  );
}

export function openMissions(world: WorldDef, o: { load: (m: Mission) => void; flyTo: (zone: string) => void; trainHere: (zone: string) => void }): void {
  const done = missionsDone();
  const grid = h('div', { class: 'mission-grid' });
  const search = h('input', { class: 'input', placeholder: 'Search missions…', style: { width: '220px' } }) as HTMLInputElement;
  let algo = 'all';
  const algos = ['all', ...new Set(MISSIONS.map((m) => m.algo))];
  const filter = seg(
    algos.map((a) => ({ value: a, label: a === 'all' ? 'All' : a })),
    (v) => {
      algo = v;
      filter.set(v);
      render();
    },
    'sm',
  );
  filter.set('all');
  let close: () => void = () => {};
  const render = () => {
    grid.replaceChildren();
    const q = search.value.trim().toLowerCase();
    for (const m of MISSIONS) {
      if (algo !== 'all' && m.algo !== algo) continue;
      if (q && !`${m.title} ${m.summary} ${m.learn.join(' ')} ${m.algo}`.toLowerCase().includes(q)) continue;
      const z = world.zones.find((x) => x.id === m.zone);
      grid.append(
        h(
          'div',
          { class: 'mcard', tabindex: '0', onclick: () => (close(), o.load(m)), onkeydown: (e: KeyboardEvent) => e.key === 'Enter' && (close(), o.load(m)) },
          done.has(m.id) ? h('span', { class: 'done', title: 'Completed (80% success)' }, icon('trophy')) : null,
          h('div', { class: 'top' }, h('span', { class: 'memoji' }, m.emoji), h('h4', null, m.title)),
          h('p', null, m.summary),
          h('div', { class: 'meta' }, h('span', { class: 'chip stars', title: `difficulty ${m.level}/5` }, '★'.repeat(m.level), h('span', { style: { opacity: '0.25' } }, '★'.repeat(5 - m.level))), h('span', { class: 'chip' }, m.algo), h('span', { class: 'chip' }, icon('timer', 12), m.time), h('span', { class: 'chip' }, `${z?.emoji ?? ''} ${z?.name ?? m.zone}`)),
        ),
      );
    }
    if (!grid.childElementCount) grid.append(h('div', { class: 'muted' }, 'No mission matches.'));
  };
  search.addEventListener('input', render);
  render();
  const zones = h('div', { class: 'zone-grid' });
  for (const z of world.zones) {
    zones.append(
      h(
        'div',
        { class: 'zcard' },
        h('div', { class: 'zt' }, `${z.emoji} ${z.name}`),
        h('p', null, z.blurb),
        h('div', { class: 'row' }, btn({ icon: 'locate', label: 'Fly there', cls: 'xs', onclick: () => (close(), o.flyTo(z.id)) }), btn({ icon: 'bot', label: 'Train here', cls: 'xs primary', onclick: () => (close(), o.trainHere(z.id)) })),
      ),
    );
  }
  ({ close } = modal(
    'Missions & the world',
    h(
      'div',
      null,
      h('p', { class: 'dim', style: { marginTop: '0' } }, 'Each mission is a ready-made program: load one, press Train, then read and change it. Missions open in the editor you are using (blocks or code).'),
      h('div', { class: 'filter-row' }, search, filter.el),
      grid,
      h('h3', { class: 'h-sub' }, 'The world: 12 zones + free roam'),
      h('p', { class: 'dim', style: { marginTop: '0' } }, 'Pick where your robot trains. Combine any zone with your own senses, actions and rewards, or build your own course.'),
      zones,
    ),
    { size: 'wide' },
  ));
  setTimeout(() => search.focus(), 50);
}

export function openLearn(openMissions: () => void, openLab: () => void, mode: 'robots' | 'language' = 'robots'): void {
  let close: () => void = () => {};
  const panel = new LearnPanel(
    () => (close(), openMissions()),
    () => (close(), openLab()),
    mode,
  );
  const m = modal('Learn: the ideas behind it all', panel.el, { size: 'narrow' });
  close = m.close;
  m.el.style.width = 'min(720px, 100%)';
}

export interface Settings {
  quality: Quality;
  maxFps: number;
  throttle: number;
  autosave: boolean;
  showFps: boolean;
}

export function openSettings(cur: Settings, apply: (s: Settings) => void, info: { workers: string; dpr: number; cores: number }): void {
  const s = { ...cur };
  const row = (t: string, d: string, ctl: HTMLElement) => h('div', { class: 'set-row' }, h('div', { class: 't' }, t), h('div', { class: 'd' }, d), ctl);
  const q = seg<Quality>(
    [
      { value: 'auto', label: 'Auto' },
      { value: 'high', label: 'High' },
      { value: 'medium', label: 'Medium' },
      { value: 'low', label: 'Low' },
    ],
    (v) => {
      s.quality = v;
      q.set(v);
      apply(s);
    },
    'sm',
  );
  q.set(s.quality);
  const fps = seg<string>(
    [
      { value: '0', label: 'Max' },
      { value: '60', label: '60' },
      { value: '30', label: '30' },
    ],
    (v) => {
      s.maxFps = Number(v);
      fps.set(v);
      apply(s);
    },
    'sm',
  );
  fps.set(String(s.maxFps));
  const thr = seg<string>(
    [
      { value: '0', label: 'Full speed' },
      { value: '2000', label: '2k steps/s' },
      { value: '320', label: 'Real time' },
    ],
    (v) => {
      s.throttle = Number(v);
      thr.set(v);
      apply(s);
    },
    'sm',
  );
  thr.set(String(s.throttle));
  const chk = (val: boolean, set: (b: boolean) => void) => {
    const c = h('input', { type: 'checkbox', checked: val }) as HTMLInputElement;
    c.addEventListener('change', () => {
      set(c.checked);
      apply(s);
    });
    return c;
  };
  modal(
    'Settings',
    h(
      'div',
      { class: 'settings' },
      row('3D quality', `Auto lowers the resolution when frames get slow (now ${info.dpr.toFixed(2)}× pixels). Low turns shadows off: use it on older laptops.`, q.el),
      row('Frame rate cap', 'Capping the 3D view at 30 fps saves battery and leaves more CPU for training.', fps.el),
      row('Training speed', `"Real time" trains slowly enough to watch the 16 ghosts learn (16 worlds × 20 steps/s). Training runs in ${info.workers} (${info.cores} CPU cores).`, thr.el),
      row('Show the frame rate', 'A small fps counter in the 3D view.', chk(s.showFps, (b) => (s.showFps = b))),
      row('Autosave', 'Keep your program, builds and brain in this browser between visits.', chk(s.autosave, (b) => (s.autosave = b))),
    ),
    { size: 'narrow' },
  );
}

export function openShortcuts(): void {
  const k = (...keys: string[]) => h('span', null, ...keys.map((x) => h('span', { class: 'kbd-key' }, x)));
  modal(
    'Keyboard shortcuts',
    h(
      'div',
      { class: 'shortcuts' },
      k('Ctrl', 'Enter'),
      'Train / pause',
      k('Ctrl', 'S'),
      'Save the project to a file',
      k('Ctrl', 'Space'),
      'Code mode: suggestions',
      k('W', 'A', 'S', 'D'),
      'Drive (in "You" mode); arrows work too',
      k('Space'),
      'Jump',
      k('F'),
      'Kick',
      k('Q', 'E'),
      'Slide sideways',
      k('Del'),
      'Build mode: delete the selected item',
      k('R'),
      'Build mode: rotate the selected wall (Shift = 15°)',
      k('Esc'),
      'Close dialogs, deselect, leave build mode',
      k('Mouse'),
      'Drag to orbit, right-drag to pan, wheel to zoom',
    ),
    { size: 'narrow' },
  );
}

export function explainExports(): void {
  modal(
    'Exporting the brain',
    h(
      'div',
      { class: 'explain' },
      h('p', null, h('b', null, 'JSON: '), 'all weights as plain numbers, plus the input/output layout and normalisation statistics. Human-readable, and you can import it back (project menu → Import a brain).'),
      h('p', null, h('b', null, 'ONNX: '), 'the industry-standard neural network file. Open it in ', h('a', { href: 'https://netron.app', target: '_blank', rel: 'noopener' }, 'Netron'), ' to see the graph, run it with onnxruntime in Python/C#/JavaScript, or load it in Unity (Sentis). Input "obs" = the senses, outputs = logits / Q-values / action means (+ "value" for the critic).'),
      h('p', null, h('b', null, 'PyTorch script: '), 'a single .py file with the weights embedded. Run it with plain NumPy, or call torch_model() to get a real PyTorch nn.Module you can fine-tune.'),
      h('p', null, h('b', null, 'Memory (frame stacking): '), 'if your robot remembers the last N moments, the exported network expects N observations in a row (newest first), exactly like the input layout in the Code tab.'),
      h('p', { class: 'muted' }, 'Every export is verified to give the same outputs as the browser (to about 0.0000001).'),
    ),
    { size: 'narrow' },
  );
}

/** Imported code runs in your browser: ask first. Resolves true = run it. */
export function confirmCode(name: string, code: string): Promise<boolean> {
  return new Promise((resolve) => {
    let decided = false;
    const done = (v: boolean) => {
      if (decided) return;
      decided = true;
      m.close();
      resolve(v);
    };
    const preview = h('pre', { class: 'code', style: { maxHeight: '220px' } }, code.slice(0, 4000) + (code.length > 4000 ? '\n…' : ''));
    const m = modal(
      'This project contains code',
      h('div', { class: 'explain' }, h('p', null, `"${name}" is a code-mode project. Its JavaScript runs in your browser when you train or watch the robot, so only run code from people you trust. You can always read it first: open it without running and look at it in the editor.`), preview),
      {
        size: 'narrow',
        foot: [btn({ label: 'Open without running', cls: 'ghost', onclick: () => done(false) }), btn({ icon: 'play', label: 'Run it', cls: 'primary', onclick: () => done(true) })],
        onClose: () => done(false),
      },
    );
  });
}

export function promptText(title: string, label: string, value: string): Promise<string | null> {
  return new Promise((resolve) => {
    let decided = false;
    const inp = h('input', { class: 'input', value, style: { width: '100%' } }) as HTMLInputElement;
    const done = (v: string | null) => {
      if (decided) return;
      decided = true;
      m.close();
      resolve(v);
    };
    inp.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') done(inp.value.trim() || null);
    });
    const m = modal(title, h('label', { class: 'field' }, h('span', { class: 'lbl' }, label), inp), {
      size: 'narrow',
      foot: [btn({ label: 'Cancel', cls: 'ghost', onclick: () => done(null) }), btn({ label: 'OK', cls: 'primary', onclick: () => done(inp.value.trim() || null) })],
      onClose: () => done(null),
    });
    setTimeout(() => (inp.focus(), inp.select()), 30);
  });
}

export function confirmDialog(title: string, text: string, ok = 'OK', danger = false): Promise<boolean> {
  return new Promise((resolve) => {
    let decided = false;
    const done = (v: boolean) => {
      if (decided) return;
      decided = true;
      m.close();
      resolve(v);
    };
    const m = modal(title, h('p', { class: 'dim', style: { margin: 0 } }, text), {
      size: 'narrow',
      foot: [btn({ label: 'Cancel', cls: 'ghost', onclick: () => done(false) }), btn({ label: ok, cls: danger ? 'danger' : 'primary', onclick: () => done(true) })],
      onClose: () => done(false),
    });
  });
}

/** Ask the user to pick one of several options (null = cancelled). */
export function choiceDialog<T extends string>(title: string, text: string, options: { value: T; label: string; primary?: boolean }[]): Promise<T | null> {
  return new Promise((resolve) => {
    let decided = false;
    const done = (v: T | null) => {
      if (decided) return;
      decided = true;
      m.close();
      resolve(v);
    };
    const m = modal(title, h('p', { class: 'dim', style: { margin: 0 } }, text), {
      size: 'narrow',
      foot: [btn({ label: 'Cancel', cls: 'ghost', onclick: () => done(null) }), ...options.map((o) => btn({ label: o.label, cls: o.primary ? 'primary' : '', onclick: () => done(o.value) }))],
      onClose: () => done(null),
    });
  });
}
