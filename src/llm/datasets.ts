/**
 * Training text for the mini language models. All of it is generated here
 * (no downloads): small, varied and structured enough that a tiny
 * transformer visibly learns it within minutes. You can also paste or load
 * your own text.
 */
import { RNG } from '../ml/rng.ts';

export interface Corpus {
  text: string;
  /** where the validation part starts (default: the last 10 %) */
  valStart?: number;
  /** every line is one example: training windows start at the beginning of a line */
  align?: boolean;
}

/** A task with right answers (e.g. sums): test cases taken from the validation text. */
export interface Probe {
  /** what the accuracy measures, e.g. "sums correct" */
  name: string;
  help: string;
  /** prompt → expected completion (up to the end of the line) */
  cases(valText: string): { prompt: string; answer: string }[];
}

export interface DatasetInfo {
  id: string;
  name: string;
  emoji: string;
  blurb: string;
  make(seed: number): Corpus;
  probe?: Probe;
  /** suggested settings for this dataset */
  suggest?: { tokenizer?: 'char' | 'word' | 'bpe'; ctx?: number; vocab?: number };
}

const pick = <T>(rng: RNG, a: readonly T[]): T => a[rng.int(a.length)];

// ---------------------------------------------------------------------------
// Tiny stories: a story grammar with many small variations
// ---------------------------------------------------------------------------
const NAMES = ['Mia', 'Leo', 'Ava', 'Sam', 'Zoe', 'Max', 'Lily', 'Ben', 'Ella', 'Finn', 'Ruby', 'Theo', 'Nora', 'Jack', 'Ivy', 'Omar', 'Lucy', 'Kai', 'Rosa', 'Tom'];
const ANIMALS = ['cat', 'dog', 'rabbit', 'fox', 'bear', 'duck', 'owl', 'mouse', 'frog', 'turtle', 'bird', 'horse', 'pig', 'lamb', 'squirrel'];
const ADJS = ['little', 'big', 'happy', 'shy', 'brave', 'curious', 'sleepy', 'kind', 'silly', 'tiny', 'clever', 'gentle'];
const PLACES = ['forest', 'garden', 'park', 'farm', 'village', 'meadow', 'house by the lake', 'small town', 'castle', 'cozy house'];
const OBJECTS = ['red ball', 'shiny stone', 'blue kite', 'old map', 'golden key', 'small box', 'yellow flower', 'paper boat', 'green hat', 'magic stick', 'big apple', 'soft blanket'];
const ACTIVITIES = ['run in the grass', 'read books', 'sing songs', 'build towers', 'paint pictures', 'climb trees', 'dance in the rain', 'look at the stars', 'bake cookies', 'play hide and seek', 'jump in puddles', 'pick flowers'];
const FEELINGS = ['happy', 'sad', 'surprised', 'excited', 'scared', 'proud', 'curious', 'sleepy'];
const WEATHER = ['sunny', 'rainy', 'windy', 'snowy', 'warm', 'cold', 'cloudy'];
const SAY = ['said', 'asked', 'shouted', 'whispered', 'laughed'];
const EXCLAIM = ['Wow!', 'Look at this!', 'What is it?', 'How pretty!', 'Oh no!', 'Yay!', 'Can we play?'];
const EVENTS = ['It started to rain.', 'The wind blew the {o} away.', 'A loud noise came from the trees.', 'The sun went behind a cloud.', 'A butterfly landed on the {o}.', 'The {o} rolled down the hill.'];
const SOLVE = ['{N} ran after it and caught it.', '{N} asked {F} for help, and together they found it.', '{N} was brave and did not give up.', 'They looked everywhere until they found it under a tree.', '{F} had a good idea and fixed the problem.'];
const ENDINGS = ['From that day on, {N} and {F} were best friends.', 'They learned that sharing makes everyone happy.', '{N} smiled and went home for dinner.', 'It was the best day ever.', 'That night, {N} slept with a big smile.', 'And they all lived happily ever after.'];

function story(rng: RNG): string {
  const N = pick(rng, NAMES);
  let F = pick(rng, NAMES);
  while (F === N) F = pick(rng, NAMES);
  const a = pick(rng, ANIMALS);
  const fa = pick(rng, ANIMALS);
  const o = pick(rng, OBJECTS);
  const fill = (s: string) => s.replace(/\{N\}/g, N).replace(/\{F\}/g, F).replace(/\{o\}/g, o);
  const lines: string[] = [];
  const opening = rng.int(3);
  if (opening === 0) lines.push(`Once upon a time, there was a ${pick(rng, ADJS)} ${a} named ${N}.`);
  else if (opening === 1) lines.push(`${N} was a ${pick(rng, ADJS)} ${a} who lived in a ${pick(rng, PLACES)}.`);
  else lines.push(`One ${pick(rng, WEATHER)} morning, a ${pick(rng, ADJS)} ${a} named ${N} woke up early.`);
  if (opening !== 1) lines.push(`${N} lived in a ${pick(rng, PLACES)}.`);
  lines.push(`${N} loved to ${pick(rng, ACTIVITIES)}.`);
  lines.push(`One ${pick(rng, WEATHER)} day, ${N} found a ${o}.`);
  lines.push(`"${pick(rng, EXCLAIM)}" ${pick(rng, SAY)} ${N}.`);
  lines.push(`${N} wanted to show the ${o} to ${F}, a ${pick(rng, ADJS)} ${fa}.`);
  lines.push(`${F} was ${pick(rng, FEELINGS)} to see it.`);
  if (rng.next() < 0.6) {
    lines.push(fill(pick(rng, EVENTS)));
    lines.push(`${N} felt ${pick(rng, FEELINGS)}.`);
    lines.push(fill(pick(rng, SOLVE)));
  }
  lines.push(`They decided to ${pick(rng, ACTIVITIES)} together.`);
  lines.push(fill(pick(rng, ENDINGS)));
  return lines.join(' ');
}

// ---------------------------------------------------------------------------
// Names (the classic "makemore" task: invent new names)
// ---------------------------------------------------------------------------
const NAME_LIST =
  'emma olivia ava isabella sophia mia charlotte amelia harper evelyn abigail emily ella elizabeth camila luna sofia avery mila aria scarlett penelope layla chloe victoria madison eleanor grace nora riley zoey hannah hazel lily ellie violet lillian zoe stella aurora natalie emilia everly leah aubrey willow addison lucy audrey bella nova brooklyn paisley savannah claire skylar isla genesis naomi elena caroline eliana anna maya valentina ruby kennedy ivy ariana aaliyah cora madelyn alice kinsley hailey gabriella allison gianna serenity samantha sarah autumn quinn eva piper sophie sadie delilah josephine nevaeh adeline arya emery lydia clara vivian madeline peyton julia rylee brielle reagan daisy jade mary rose iris ' +
  'liam noah oliver elijah william james benjamin lucas henry alexander mason michael ethan daniel jacob logan jackson levi sebastian mateo jack owen theodore aiden samuel joseph john david wyatt matthew luke asher carter julian grayson leo jayden gabriel isaac lincoln anthony hudson dylan ezra thomas charles christopher jaxon maverick josiah isaiah andrew elias joshua nathan caleb ryan adrian miles eli nolan christian aaron cameron ezekiel colton luca landon hunter jonathan santiago axel easton cooper jeremiah angel roman connor jameson robert greyson jordan ian carson jaxson leonardo nicholas dominic austin everett brooks xavier kai jose parker adam jace wesley kayden silas bennett declan waylon weston evan emmett micah ryder beau damian brayden gael rowan harrison bryson sawyer amir kingston jason giovanni vincent ayden chase myles diego nathaniel legend jonah river tyler cole braxton george milo zachary ashton luis jasper kaiden adriel gavin bentley calvin zion juan maxwell max ryker carlos emmanuel jayce lorenzo ivan jude august kevin malachi elliott rhett archer karter arthur luka elliot thiago brandon camden justin jesus maddox king theo enzo matteo emiliano dean hayden finn ' +
  'amara keisha malik omar fatima yusuf amina hamza zainab ibrahim leila tariq nadia samir yara karim salma rami dina hassan mariam aiko haruto yuki sakura kenji hana ren mei takumi rina chen wei jun lin ming xiu hui priya arjun ananya rohan kavya vikram meera aditya isha nikhil sofia diego lucia mateo valeria pablo camila javier elena andres paula luca giulia marco chiara matteo francesca lukas lena jonas mila felix hanna emil frida lars ingrid nils astrid sven freya olga ivan katya dmitri anya boris vera ana joao beatriz tiago ines rafael marta';

// ---------------------------------------------------------------------------
// Tiny Python programs
// ---------------------------------------------------------------------------
const VARS = ['x', 'y', 'n', 'a', 'b', 'value', 'count', 'total', 'size', 'score'];
const NOUNS = ['numbers', 'items', 'scores', 'values', 'prices', 'ages', 'points'];
function pyProgram(rng: RNG): string {
  const v = pick(rng, VARS);
  const w = pick(rng, VARS.filter((q) => q !== v));
  const k = 1 + rng.int(9);
  const list = pick(rng, NOUNS);
  switch (rng.int(9)) {
    case 0:
      return `def add(${v}, ${w}):\n    return ${v} + ${w}\n`;
    case 1:
      return `def times_${k}(${v}):\n    return ${v} * ${k}\n`;
    case 2:
      return `def is_even(${v}):\n    return ${v} % 2 == 0\n`;
    case 3:
      return `def biggest(${v}, ${w}):\n    if ${v} > ${w}:\n        return ${v}\n    return ${w}\n`;
    case 4:
      return `def total_of(${list}):\n    total = 0\n    for item in ${list}:\n        total += item\n    return total\n`;
    case 5:
      return `for i in range(${k + 1}):\n    print(i * ${k})\n`;
    case 6:
      return `def count_big(${list}, limit):\n    count = 0\n    for item in ${list}:\n        if item > limit:\n            count += 1\n    return count\n`;
    case 7:
      return `${v} = ${k}\nwhile ${v} > 0:\n    print(${v})\n    ${v} -= 1\n`;
    default:
      return `def greet(name):\n    return "Hello, " + name + "!"\n`;
  }
}

function lettersOf(rng: RNG, n: number): string {
  let s = '';
  for (let i = 0; i < n; i++) s += String.fromCharCode(97 + rng.int(26));
  return s;
}

export const DATASETS: DatasetInfo[] = [
  {
    id: 'stories',
    name: 'Tiny stories',
    emoji: '📖',
    blurb: 'Simple children\'s stories from a story grammar. Watch random letters turn into words, then sentences.',
    make(seed) {
      const rng = new RNG(seed);
      const parts: string[] = [];
      let len = 0;
      while (len < 120000) {
        const s = story(rng);
        parts.push(s);
        len += s.length + 2;
      }
      return { text: parts.join('\n\n') + '\n' };
    },
    suggest: { tokenizer: 'char', ctx: 64 },
  },
  {
    id: 'names',
    name: 'Names',
    emoji: '🏷️',
    blurb: 'First names from many languages, one per line. The model learns how names sound and invents new ones. The validation names are never used for training: watch train and validation loss drift apart as it starts memorising the training names.',
    make(seed) {
      const rng = new RNG(seed);
      const names = NAME_LIST.split(/\s+/).filter(Boolean);
      const unique = [...new Set(names)];
      rng.shuffle(unique);
      const nHeld = Math.round(unique.length * 0.1);
      const held = unique.slice(0, nHeld);
      const train = unique.slice(nHeld);
      const out: string[] = [];
      for (let r = 0; r < 12; r++) {
        const copy = train.slice();
        rng.shuffle(copy);
        out.push(...copy);
      }
      const text = out.join('\n') + '\n';
      const val: string[] = [];
      for (let r = 0; r < 3; r++) {
        const copy = held.slice();
        rng.shuffle(copy);
        val.push(...copy);
      }
      return { text: text + val.join('\n') + '\n', valStart: text.length, align: true };
    },
    suggest: { tokenizer: 'char', ctx: 16 },
  },
  {
    id: 'arithmetic',
    name: 'Addition',
    emoji: '➕',
    blurb: 'Lines like 07+58=065: every number has a fixed number of digits, so each digit has its own place (like column addition on paper). The validation part only has sums that never appear in training: does the model learn to ADD, or only memorise?',
    make(seed) {
      const rng = new RNG(seed);
      const pairs: [number, number][] = [];
      for (let a = 0; a < 100; a++) for (let b = 0; b < 100; b++) pairs.push([a, b]);
      rng.shuffle(pairs);
      const held = pairs.slice(0, 800);
      const train = pairs.slice(800);
      const line = ([a, b]: [number, number]) => `${String(a).padStart(2, '0')}+${String(b).padStart(2, '0')}=${String(a + b).padStart(3, '0')}\n`;
      const lines: string[] = [];
      for (let r = 0; r < 2; r++) for (const p of train) lines.push(line(p));
      rng.shuffle(lines);
      const text = lines.join('');
      return { text: text + held.map(line).join(''), valStart: text.length, align: true };
    },
    probe: {
      name: 'new sums correct',
      help: 'Of 50 sums the model never saw while training, how many does it answer exactly right (always picking the most likely token)?',
      cases: (val) =>
        val
          .split('\n')
          .map((l) => /^(\d\d\+\d\d=)(\d\d\d)$/.exec(l))
          .filter((m): m is RegExpExecArray => !!m)
          .map((m) => ({ prompt: m[1], answer: m[2] })),
    },
    suggest: { tokenizer: 'char', ctx: 30 },
  },
  {
    id: 'copy',
    name: 'Copy & reverse',
    emoji: '🔁',
    blurb: 'Puzzles like hello>hello (copy) and hello<olleh (reverse), with random 5-letter strings. Great for seeing attention heads look back at exactly the right letter.',
    make(seed) {
      const rng = new RNG(seed);
      const lines: string[] = [];
      for (let i = 0; i < 9000; i++) {
        const s = lettersOf(rng, 5);
        lines.push(rng.next() < 0.5 ? `${s}>${s}` : `${s}<${[...s].reverse().join('')}`);
      }
      return { text: lines.join('\n') + '\n', align: true };
    },
    probe: {
      name: 'new puzzles correct',
      help: 'Of 50 letter strings from the validation text (new ones), how many does the model copy or reverse exactly right?',
      cases: (val) =>
        val
          .split('\n')
          .map((l) => /^([a-z]{5}[<>])([a-z]{5})$/.exec(l))
          .filter((m): m is RegExpExecArray => !!m)
          .map((m) => ({ prompt: m[1], answer: m[2] })),
    },
    suggest: { tokenizer: 'char', ctx: 24 },
  },
  {
    id: 'python',
    name: 'Tiny Python',
    emoji: '🐍',
    blurb: 'Small Python functions and loops. A miniature "code model": does it learn indentation and colons?',
    make(seed) {
      const rng = new RNG(seed);
      const parts: string[] = [];
      let len = 0;
      while (len < 90000) {
        const p = pyProgram(rng);
        parts.push(p);
        len += p.length + 1;
      }
      return { text: parts.join('\n') };
    },
    suggest: { tokenizer: 'char', ctx: 64 },
  },
];

export function getDataset(id: string): DatasetInfo | undefined {
  return DATASETS.find((d) => d.id === id);
}
