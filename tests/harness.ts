/** Minimal test harness so tests run with plain `node` (no framework needed). */
type TestFn = () => void | Promise<void>;
const tests: { name: string; fn: TestFn }[] = [];

export function test(name: string, fn: TestFn): void {
  tests.push({ name, fn });
}

export function assert(cond: unknown, msg = 'assertion failed'): asserts cond {
  if (!cond) throw new Error(msg);
}

export async function runAll(filter?: string): Promise<boolean> {
  let pass = 0;
  let fail = 0;
  for (const t of tests) {
    if (filter && !t.name.toLowerCase().includes(filter.toLowerCase())) continue;
    const t0 = performance.now();
    try {
      await t.fn();
      pass++;
      console.log(`  \x1b[32m✓\x1b[0m ${t.name} \x1b[90m(${(performance.now() - t0).toFixed(0)} ms)\x1b[0m`);
    } catch (e) {
      fail++;
      console.log(`  \x1b[31m✗ ${t.name}\x1b[0m\n      ${(e as Error).message}`);
    }
  }
  console.log(`\n${pass} passed, ${fail} failed`);
  return fail === 0;
}
