/**
 * Turn exceptions thrown by generated/user code into "line N: message".
 *
 * Scripts are compiled with `new Function`, and every browser numbers the
 * lines of such code a little differently. So we calibrate once: compile a
 * probe that throws on a known line and see what the engine reports.
 */
import type { AgentSpec } from './types.ts';

let calibration: number | null | undefined;

function stackLine(e: unknown): number | null {
  const st = (e as Error)?.stack;
  if (typeof st !== 'string') return null;
  // Chrome/Edge/Node: "<anonymous>:12:5"   Firefox: "> Function:12:5"   Safari: "Function:12:5" or "anonymous:12:5"
  const m = st.match(/(?:<anonymous>|Function|anonymous):(\d+):(\d+)/);
  return m ? Number(m[1]) : null;
}

/** Same wrapper as compileScript() in env.ts. */
function compileProbe(): unknown {
  try {
    const f = new Function(`"use strict";\nreturn (function (api, Math) {\nthrow new Error('probe');\n});`)() as () => void;
    f();
  } catch (e) {
    return e;
  }
  return null;
}

/** How many lines the engine adds in front of a `new Function` script (null = unknown). */
function offset(): number | null {
  if (calibration !== undefined) return calibration;
  const l = stackLine(compileProbe());
  // the probe throws on line 2 of the factory source
  calibration = l === null ? null : l - 2;
  return calibration;
}

/** Line in the user's program (1-based) where `e` was thrown, if we can tell. */
export function userLine(e: unknown, spec: Pick<AgentSpec, 'codeLineOffset'>): number | null {
  if (spec.codeLineOffset === undefined) return null;
  const off = offset();
  const l = stackLine(e);
  if (off === null || l === null) return null;
  const line = l - off - spec.codeLineOffset;
  return line >= 1 ? line : null;
}

export function describeScriptError(e: unknown, spec: Pick<AgentSpec, 'codeLineOffset'>): string {
  const msg = (e as Error)?.message ?? String(e);
  const line = userLine(e, spec);
  return line ? `line ${line}: ${msg}` : msg;
}
