import { runAll } from './harness.ts';
import './ml.test.ts';
import './sim.test.ts';

const filter = process.argv[2];
const ok = await runAll(filter);
process.exit(ok ? 0 : 1);
