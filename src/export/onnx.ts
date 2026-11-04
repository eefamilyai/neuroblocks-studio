/**
 * ONNX export, written by hand (no libraries): a tiny protobuf encoder plus
 * the handful of ONNX messages we need. The result opens in Netron,
 * onnxruntime (Python/C#/JS), Unity Sentis, etc.
 *
 * Graph:  obs → (x−μ)/σ → clip(±5) → Gemm → act → … → Gemm → policy
 *                                    └─ (critic) Gemm → act → … → value
 */
import type { BrainSnapshot } from '../ml/algos/common.ts';
import type { MLPJSON } from '../ml/nn.ts';

class PB {
  private buf: number[] = [];

  private varint(v: number): void {
    // ONNX fields here are always small non-negative ints
    let n = Math.floor(v);
    while (n > 127) {
      this.buf.push((n & 127) | 128);
      n = Math.floor(n / 128);
    }
    this.buf.push(n);
  }
  private tag(field: number, wire: number): void {
    this.varint(field * 8 + wire);
  }
  int(field: number, v: number): this {
    this.tag(field, 0);
    this.varint(v);
    return this;
  }
  float(field: number, v: number): this {
    this.tag(field, 5);
    const b = new Uint8Array(new Float32Array([v]).buffer);
    this.buf.push(...b);
    return this;
  }
  bytes(field: number, data: Uint8Array): this {
    this.tag(field, 2);
    this.varint(data.length);
    for (let i = 0; i < data.length; i++) this.buf.push(data[i]);
    return this;
  }
  str(field: number, s: string): this {
    return this.bytes(field, new TextEncoder().encode(s));
  }
  msg(field: number, build: (p: PB) => void): this {
    const sub = new PB();
    build(sub);
    return this.bytes(field, sub.out());
  }
  out(): Uint8Array {
    return Uint8Array.from(this.buf);
  }
}

const FLOAT = 1;

function tensor(p: PB, name: string, dims: number[], data: Float32Array): void {
  for (const d of dims) p.int(1, d);
  p.int(2, FLOAT);
  p.str(8, name);
  const le = new Uint8Array(data.length * 4);
  const dv = new DataView(le.buffer);
  data.forEach((v, i) => dv.setFloat32(i * 4, v, true));
  p.bytes(9, le);
}

function valueInfo(p: PB, name: string, width: number): void {
  p.str(1, name);
  p.msg(2, (t) =>
    t.msg(1, (tt) => {
      tt.int(1, FLOAT);
      tt.msg(2, (shape) => {
        shape.msg(1, (d) => d.str(2, 'batch'));
        shape.msg(1, (d) => d.int(1, width));
      });
    }),
  );
}

interface Node {
  op: string;
  inputs: string[];
  outputs: string[];
  name: string;
  attrs?: { name: string; f?: number; i?: number }[];
}

export function exportOnnx(snap: BrainSnapshot, title: string): Uint8Array {
  const nodes: Node[] = [];
  const inits: { name: string; dims: number[]; data: Float32Array }[] = [];
  const S = snap.net.inSize;
  let x = 'obs';
  if (snap.obsNorm) {
    const mean = Float32Array.from(snap.obsNorm.mean);
    const std = Float32Array.from(snap.obsNorm.var, (v) => Math.sqrt(v + 1e-8));
    inits.push({ name: 'obs_mean', dims: [S], data: mean });
    inits.push({ name: 'obs_std', dims: [S], data: std });
    inits.push({ name: 'clip_min', dims: [], data: Float32Array.of(-snap.obsNorm.clip) });
    inits.push({ name: 'clip_max', dims: [], data: Float32Array.of(snap.obsNorm.clip) });
    nodes.push({ op: 'Sub', inputs: ['obs', 'obs_mean'], outputs: ['obs_centered'], name: 'normalize_sub' });
    nodes.push({ op: 'Div', inputs: ['obs_centered', 'obs_std'], outputs: ['obs_scaled'], name: 'normalize_div' });
    nodes.push({ op: 'Clip', inputs: ['obs_scaled', 'clip_min', 'clip_max'], outputs: ['obs_norm'], name: 'normalize_clip' });
    x = 'obs_norm';
  }
  const chain = (net: MLPJSON, prefix: string, outName: string) => {
    let h = x;
    net.layers.forEach((l, i) => {
      const w = `${prefix}.layer${i}.weight`;
      const b = `${prefix}.layer${i}.bias`;
      inits.push({ name: w, dims: [l.outSize, l.inSize], data: Float32Array.from(l.W) });
      inits.push({ name: b, dims: [l.outSize], data: Float32Array.from(l.b) });
      const last = i === net.layers.length - 1;
      const z = last && l.activation === 'linear' ? outName : `${prefix}.z${i}`;
      nodes.push({ op: 'Gemm', inputs: [h, w, b], outputs: [z], name: `${prefix}.gemm${i}`, attrs: [{ name: 'transB', i: 1 }] });
      const a = last ? outName : `${prefix}.a${i}`;
      const act = l.activation;
      if (act === 'linear') {
        h = z;
        return;
      }
      const map: Record<string, string> = { relu: 'Relu', tanh: 'Tanh', sigmoid: 'Sigmoid', softplus: 'Softplus' };
      if (map[act]) nodes.push({ op: map[act], inputs: [z], outputs: [a], name: `${prefix}.${act}${i}` });
      else if (act === 'leaky_relu') nodes.push({ op: 'LeakyRelu', inputs: [z], outputs: [a], name: `${prefix}.lrelu${i}`, attrs: [{ name: 'alpha', f: 0.01 }] });
      else if (act === 'elu') nodes.push({ op: 'Elu', inputs: [z], outputs: [a], name: `${prefix}.elu${i}`, attrs: [{ name: 'alpha', f: 1 }] });
      else if (act === 'swish') {
        nodes.push({ op: 'Sigmoid', inputs: [z], outputs: [`${prefix}.sig${i}`], name: `${prefix}.swish_sig${i}` });
        nodes.push({ op: 'Mul', inputs: [z, `${prefix}.sig${i}`], outputs: [a], name: `${prefix}.swish_mul${i}` });
      }
      h = a;
    });
  };
  const policyName = snap.kind === 'q' ? 'q_values' : snap.space.kind === 'continuous' ? 'action_mean' : 'logits';
  chain(snap.net, 'policy', policyName);
  const outputs: [string, number][] = [[policyName, snap.net.layers[snap.net.layers.length - 1].outSize]];
  if (snap.critic) {
    chain(snap.critic, 'critic', 'value');
    outputs.push(['value', 1]);
  }

  const doc = `NeuroBlocks Studio brain "${title}". Algorithm ${snap.algo}, ${snap.steps} training steps. Input "obs": ${S} sensor values (${snap.obsLabels.join(', ')}). Output "${policyName}": ${describeOutputs(snap)}.`;
  const model = new PB();
  model.int(1, 7); // ir_version
  model.str(2, 'NeuroBlocks Studio');
  model.str(3, '1.0');
  model.str(6, doc);
  model.msg(7, (g) => {
    for (const nd of nodes)
      g.msg(1, (p) => {
        for (const i of nd.inputs) p.str(1, i);
        for (const o of nd.outputs) p.str(2, o);
        p.str(3, nd.name);
        p.str(4, nd.op);
        for (const at of nd.attrs ?? [])
          p.msg(5, (a) => {
            a.str(1, at.name);
            if (at.f !== undefined) {
              a.float(2, at.f);
              a.int(20, 1);
            } else {
              a.int(3, at.i ?? 0);
              a.int(20, 2);
            }
          });
      });
    g.str(2, 'neuroblocks_brain');
    for (const t of inits) g.msg(5, (p) => tensor(p, t.name, t.dims, t.data));
    g.msg(11, (p) => valueInfo(p, 'obs', S));
    for (const [name, w] of outputs) g.msg(12, (p) => valueInfo(p, name, w));
  });
  model.msg(8, (o) => o.int(2, 13)); // opset 13
  return model.out();
}

export function describeOutputs(snap: BrainSnapshot): string {
  const sp = snap.space;
  if (snap.kind === 'q') return `one Q-value per joint action (${sp.branches.reduce((a, b) => a * b, 1)}), pick the argmax`;
  if (sp.kind === 'continuous') return `mean action per control [${sp.labels.join(', ')}], clip to [-1, 1]`;
  return sp.labels.map((l, i) => `${l}: ${sp.branches[i]} logits [${sp.optionLabels[i].join('/')}]`).join('; ') + ' (softmax each group, pick argmax)';
}
