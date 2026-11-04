/** Generates a self-contained PyTorch (and NumPy) script for a brain. */
import type { BrainSnapshot } from '../ml/algos/common.ts';
import type { MLPJSON } from '../ml/nn.ts';
import { describeOutputs } from './onnx.ts';

const ACT_TORCH: Record<string, string> = {
  relu: 'nn.ReLU()',
  tanh: 'nn.Tanh()',
  sigmoid: 'nn.Sigmoid()',
  leaky_relu: 'nn.LeakyReLU(0.01)',
  elu: 'nn.ELU()',
  swish: 'nn.SiLU()',
  softplus: 'nn.Softplus()',
  linear: '',
};

const ACT_NUMPY: Record<string, string> = {
  relu: 'np.maximum(x, 0)',
  tanh: 'np.tanh(x)',
  sigmoid: '1 / (1 + np.exp(-x))',
  leaky_relu: 'np.where(x > 0, x, 0.01 * x)',
  elu: 'np.where(x > 0, x, np.exp(x) - 1)',
  swish: 'x / (1 + np.exp(-x))',
  softplus: 'np.log1p(np.exp(x))',
  linear: 'x',
};

function seq(net: MLPJSON, indent: string): string {
  const lines: string[] = [];
  for (const l of net.layers) {
    lines.push(`${indent}nn.Linear(${l.inSize}, ${l.outSize}),`);
    if (ACT_TORCH[l.activation]) lines.push(`${indent}${ACT_TORCH[l.activation]},`);
  }
  return lines.join('\n');
}

const num = (v: number) => {
  if (v === 0) return '0';
  const s = v.toPrecision(7);
  return s.includes('e') ? s : String(Number(s));
};
const list1 = (a: ArrayLike<number>) => '[' + Array.from(a, num).join(', ') + ']';
function list2(a: ArrayLike<number>, rows: number, cols: number): string {
  const out: string[] = [];
  for (let r = 0; r < rows; r++) out.push('    [' + Array.from({ length: cols }, (_, c) => num(a[r * cols + c])).join(', ') + ']');
  return '[\n' + out.join(',\n') + '\n  ]';
}

/** Architecture only (shown in the Code tab). */
export function pytorchModelCode(snap: BrainSnapshot): string {
  const S = snap.net.inSize;
  return `import torch
import torch.nn as nn

class Policy(nn.Module):
    """${snap.kind === 'q' ? 'Q-network' : 'Policy network'}: ${S} senses -> ${describeOutputs(snap)}"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
${seq(snap.net, '            ')}
        )
${snap.obsNorm ? `        # running statistics of the observations, learned during training\n        self.register_buffer("obs_mean", torch.zeros(${S}))\n        self.register_buffer("obs_std", torch.ones(${S}))\n` : ''}
    def forward(self, obs):${snap.obsNorm ? `\n        obs = ((obs - self.obs_mean) / self.obs_std).clamp(-${snap.obsNorm.clip}, ${snap.obsNorm.clip})` : ''}
        return self.net(obs)
${
  snap.critic
    ? `
class Critic(nn.Module):
    """Value network V(s): predicts the future reward from here."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
${seq(snap.critic, '            ')}
        )

    def forward(self, obs_normalised):
        return self.net(obs_normalised)
`
    : ''
}`;
}

/** Full standalone script with embedded weights. */
export function pytorchScript(snap: BrainSnapshot, title: string): string {
  const S = snap.net.inSize;
  const sp = snap.space;
  const weights: string[] = [];
  snap.net.layers.forEach((l, i) => {
    weights.push(`W${i} = ${list2(l.W, l.outSize, l.inSize)}`);
    weights.push(`b${i} = ${list1(l.b)}`);
  });
  const torchLoad = snap.net.layers.map((_, i) => `        model.net[${linearIndex(snap.net, i)}].weight.copy_(torch.tensor(W${i}))\n        model.net[${linearIndex(snap.net, i)}].bias.copy_(torch.tensor(b${i}))`).join('\n');
  const npForward = snap.net.layers
    .map((l, i) => `    x = x @ np.array(W${i}, dtype=np.float32).T + np.array(b${i}, dtype=np.float32)\n    x = ${ACT_NUMPY[l.activation]}`)
    .join('\n');
  const decode =
    snap.kind === 'q'
      ? `def choose_action(output):
    """Q-values for every joint action: take the best one and split it per control."""
    idx = int(np.argmax(output))
    action = []
    for n in reversed(BRANCHES):
        action.insert(0, idx % n)
        idx //= n
    return [OPTIONS[k][a] for k, a in enumerate(action)]`
      : sp.kind === 'continuous'
        ? `def choose_action(output):
    """Continuous controls: the outputs are the mean actions, clipped to [-1, 1]."""
    return {name: float(np.clip(v, -1, 1)) for name, v in zip(CONTROLS, output)}`
        : `def choose_action(output):
    """One softmax per control ("branch"): pick the most likely option in each."""
    action, i = {}, 0
    for name, n, opts in zip(CONTROLS, BRANCHES, OPTIONS):
        logits = output[i:i + n]
        probs = np.exp(logits - logits.max()); probs /= probs.sum()
        action[name] = opts[int(np.argmax(probs))]
        i += n
    return action`;
  return `"""
NeuroBlocks Studio: exported brain "${title}"
================================================
Algorithm : ${snap.algo.toUpperCase()}  (${snap.kind === 'q' ? 'Q-network' : 'policy network'})
Trained   : ${snap.steps.toLocaleString()} environment steps, ${snap.updates} updates
Inputs    : ${S} numbers (the robot's senses, in this order):
${snap.obsLabels.map((l, i) => `              [${i}] ${l}`).join('\n')}
Outputs   : ${describeOutputs(snap)}

Run it:   python ${slug(title)}_brain.py          (needs numpy; torch optional)
"""
import numpy as np

CONTROLS = ${JSON.stringify(sp.labels)}
BRANCHES = ${JSON.stringify(sp.branches)}
OPTIONS = ${JSON.stringify(sp.optionLabels)}
${snap.obsNorm ? `OBS_MEAN = ${list1(snap.obsNorm.mean)}\nOBS_STD = ${list1(snap.obsNorm.var.map((v) => Math.sqrt(v + 1e-8)))}\nOBS_CLIP = ${snap.obsNorm.clip}\n` : ''}
# ---- weights (PyTorch layout: [out_features, in_features]) ----
${weights.join('\n')}


def forward_numpy(obs):
    """Pure-NumPy forward pass: exactly what the browser computes."""
    x = np.asarray(obs, dtype=np.float32)
${snap.obsNorm ? '    x = np.clip((x - np.array(OBS_MEAN, dtype=np.float32)) / np.array(OBS_STD, dtype=np.float32), -OBS_CLIP, OBS_CLIP)\n' : ''}${npForward}
    return x


${decode}


def torch_model():
    """The same network as a PyTorch nn.Module (for fine-tuning or export)."""
    import torch
    import torch.nn as nn

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
${seq(snap.net, '                ')}
            )
${snap.obsNorm ? '            self.register_buffer("obs_mean", torch.tensor(OBS_MEAN))\n            self.register_buffer("obs_std", torch.tensor(OBS_STD))\n' : ''}
        def forward(self, obs):
${snap.obsNorm ? '            obs = ((obs - self.obs_mean) / self.obs_std).clamp(-OBS_CLIP, OBS_CLIP)\n' : ''}            return self.net(obs)

    model = Policy()
    with torch.no_grad():
${torchLoad}
    return model.eval()


if __name__ == "__main__":
    obs = np.zeros(${S}, dtype=np.float32)  # replace with real sensor values
    out = forward_numpy(obs)
    print("network output:", np.round(out, 4))
    print("chosen action :", choose_action(out))
    try:
        import torch
        model = torch_model()
        print("torch output  :", model(torch.from_numpy(obs)[None]).detach().numpy().round(4))
    except ImportError:
        print("(install torch to also run the PyTorch version)")
`;
}

function linearIndex(net: MLPJSON, layer: number): number {
  let idx = 0;
  for (let i = 0; i < layer; i++) idx += net.layers[i].activation === 'linear' ? 1 : 2;
  return idx;
}

export function slug(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '') || 'robot';
}
