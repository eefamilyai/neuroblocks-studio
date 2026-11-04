/**
 * Export a mini language model as ONE Python file: the weights (base64
 * float32), the tokenizer, a NumPy implementation of the transformer and a
 * text generator. If PyTorch is installed, torch_model() gives the same
 * network as an nn.Module you can keep training.
 */
import type { GPT } from './gpt.ts';
import type { Tokenizer } from './tokenizer.ts';

function b64(a: Float32Array): string {
  const bytes = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
  // Float32Array is little-endian on every browser/Node platform we run on
  let s = '';
  const CH = 0x8000;
  for (let i = 0; i < bytes.length; i += CH) s += String.fromCharCode(...bytes.subarray(i, i + CH));
  return btoa(s);
}

const py = (v: unknown) => JSON.stringify(v);

export function pythonScript(model: GPT, tok: Tokenizer, name: string, info: { steps: number; valLoss: number }): string {
  const c = model.cfg;
  const shapes: Record<string, number[]> = {};
  const weights: string[] = [];
  for (const p of model.params) {
    shapes[p.name] = p.shape;
    weights.push(`    ${py(p.name)}: "${b64(p.w)}",`);
  }
  const tied = c.tie;
  return `# ${name}: a mini language model made in NeuroBlocks Studio
# ${model.numParams.toLocaleString()} weights · trained for ${info.steps.toLocaleString()} steps · validation loss ${Number.isFinite(info.valLoss) ? info.valLoss.toFixed(3) : '?'}
#
#   python this_file.py "Once upon a time"        generate text (needs numpy)
#   torch_model()                                 the same network in PyTorch (optional)
#
# Everything is plain code: the tokenizer, the transformer forward pass and sampling.
import base64, re, sys
import numpy as np

CONFIG = ${py({ vocab: c.vocab, ctx: c.ctx, d: c.d, layers: c.layers, heads: c.heads, mlp: c.mlp, act: c.act, pos: c.pos, tie: c.tie }).replace(/\btrue\b/g, 'True').replace(/\bfalse\b/g, 'False')}
TOKENIZER = ${py(tok.toJSON())}
SHAPES = ${py(shapes)}
WEIGHTS_B64 = {
${weights.join('\n')}
}

W = {k: np.frombuffer(base64.b64decode(v), dtype="<f4").reshape(SHAPES[k]).astype(np.float32) for k, v in WEIGHTS_B64.items()}

# ---------------------------------------------------------------- tokenizer
PRETOKEN = re.compile(r" ?[A-Za-z]+| ?[0-9]| ?[^\\sA-Za-z0-9]+|\\s+(?!\\S)|\\s+")
VOCAB = TOKENIZER["vocab"]
INDEX = {t: i for i, t in enumerate(VOCAB)}
MERGES = [tuple(m) for m in TOKENIZER.get("merges", [])]
RANKS = {m: i for i, m in enumerate(MERGES)}


def _bpe(word):
    syms = [ch for ch in word if ch in INDEX]
    while len(syms) > 1:
        best = min(range(len(syms) - 1), key=lambda i: RANKS.get((syms[i], syms[i + 1]), 1 << 30))
        if (syms[best], syms[best + 1]) not in RANKS:
            break
        a, b = syms[best], syms[best + 1]
        out, i = [], 0
        while i < len(syms):
            if i + 1 < len(syms) and syms[i] == a and syms[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(syms[i])
                i += 1
        syms = out
    return [INDEX[s] for s in syms]


def encode(text):
    """Text → token ids (characters the model never saw are skipped)."""
    kind = TOKENIZER["kind"]
    if kind == "char":
        return [INDEX[ch] for ch in text if ch in INDEX]
    ids = []
    for w in PRETOKEN.findall(text):
        if kind == "word":
            ids += [INDEX[w]] if w in INDEX else [INDEX[ch] for ch in w if ch in INDEX]
        else:
            ids += _bpe(w)
    return ids


def decode(ids):
    return "".join(VOCAB[i] for i in ids)


# ---------------------------------------------------------------- the transformer
def layer_norm(x, g, b):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + 1e-5) * g + b


def activation(x):
    if CONFIG["act"] == "relu":
        return np.maximum(x, 0)
    if CONFIG["act"] == "swish":
        return x / (1 + np.exp(-x))
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))  # GELU (tanh approximation)


def positions(T):
    d = CONFIG["d"]
    if CONFIG["pos"] == "learned":
        return W["wpe"][:T]
    p = np.zeros((T, d), dtype=np.float32)
    if CONFIG["pos"] == "sinusoidal":
        for i in range(0, d, 2):
            f = np.arange(T) / 10000 ** (i / d)
            p[:, i] = np.sin(f)
            if i + 1 < d:
                p[:, i + 1] = np.cos(f)
    return p


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def forward(ids):
    """Token ids (at most CONFIG['ctx']) → next-token scores (logits) at every position: [T, vocab]."""
    T, d, H = len(ids), CONFIG["d"], CONFIG["heads"]
    dh = d // H
    x = W["wte"][ids] + positions(T)
    mask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), 1)  # causal: no peeking ahead
    for l in range(CONFIG["layers"]):
        p = f"h{l}."
        h = layer_norm(x, W[p + "ln1.weight"], W[p + "ln1.bias"])
        qkv = h @ W[p + "attn.qkv.weight"].T + W[p + "attn.qkv.bias"]
        q, k, v = (qkv[:, i * d:(i + 1) * d].reshape(T, H, dh).transpose(1, 0, 2) for i in range(3))
        att = softmax(q @ k.transpose(0, 2, 1) / np.sqrt(dh) + mask)
        y = (att @ v).transpose(1, 0, 2).reshape(T, d)
        x = x + y @ W[p + "attn.proj.weight"].T + W[p + "attn.proj.bias"]
        h = layer_norm(x, W[p + "ln2.weight"], W[p + "ln2.bias"])
        x = x + activation(h @ W[p + "mlp.fc.weight"].T + W[p + "mlp.fc.bias"]) @ W[p + "mlp.proj.weight"].T + W[p + "mlp.proj.bias"]
    x = layer_norm(x, W["lnf.weight"], W["lnf.bias"])
    return x @ W[${tied ? '"wte"' : '"head.weight"'}].T


def generate(prompt="", n=200, temperature=0.8, top_k=0, seed=None):
    rng = np.random.default_rng(seed)
    ids = encode(prompt) or [0]
    for _ in range(n):
        logits = forward(ids[-CONFIG["ctx"]:])[-1].astype(np.float64)
        if temperature <= 0:
            ids.append(int(logits.argmax()))
            continue
        logits /= temperature
        if top_k:
            logits[np.argsort(logits)[:-top_k]] = -np.inf
        ids.append(int(rng.choice(len(logits), p=softmax(logits))))
    return decode(ids)


# ---------------------------------------------------------------- PyTorch (optional)
def torch_model():
    """The same network as a PyTorch nn.Module, with the trained weights loaded."""
    import torch
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            d = CONFIG["d"]
            self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
            self.fc, self.mproj = nn.Linear(d, CONFIG["mlp"] * d), nn.Linear(CONFIG["mlp"] * d, d)
            self.act = {"relu": nn.ReLU(), "swish": nn.SiLU()}.get(CONFIG["act"], nn.GELU(approximate="tanh"))

        def forward(self, x):
            B, T, d = x.shape
            H = CONFIG["heads"]
            q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
            q, k, v = (t.view(B, T, H, d // H).transpose(1, 2) for t in (q, k, v))
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + self.proj(y.transpose(1, 2).reshape(B, T, d))
            return x + self.mproj(self.act(self.fc(self.ln2(x))))

    class MiniGPT(nn.Module):
        def __init__(self):
            super().__init__()
            V, d = CONFIG["vocab"], CONFIG["d"]
            self.wte = nn.Embedding(V, d)
            self.register_buffer("pos", torch.tensor(positions(CONFIG["ctx"])))
            self.h = nn.ModuleList([Block() for _ in range(CONFIG["layers"])])
            self.lnf = nn.LayerNorm(d)
            self.head = nn.Linear(d, V, bias=False)
            if CONFIG["tie"]:
                self.head.weight = self.wte.weight

        def forward(self, idx):
            x = self.wte(idx) + (self.pos[: idx.shape[1]] if CONFIG["pos"] != "learned" else 0)
            if CONFIG["pos"] == "learned":
                x = x + self.wpe[: idx.shape[1]]
            for blk in self.h:
                x = blk(x)
            return self.head(self.lnf(x))

    m = MiniGPT()
    if CONFIG["pos"] == "learned":
        m.wpe = nn.Parameter(torch.tensor(W["wpe"]))
    t = lambda k: torch.tensor(W[k])
    with torch.no_grad():
        m.wte.weight.copy_(t("wte"))
        for l, blk in enumerate(m.h):
            p = f"h{l}."
            for mod, name in [(blk.ln1, "ln1"), (blk.ln2, "ln2")]:
                mod.weight.copy_(t(p + name + ".weight"))
                mod.bias.copy_(t(p + name + ".bias"))
            for mod, name in [(blk.qkv, "attn.qkv"), (blk.proj, "attn.proj"), (blk.fc, "mlp.fc"), (blk.mproj, "mlp.proj")]:
                mod.weight.copy_(t(p + name + ".weight"))
                mod.bias.copy_(t(p + name + ".bias"))
        m.lnf.weight.copy_(t("lnf.weight"))
        m.lnf.bias.copy_(t("lnf.bias"))
        if not CONFIG["tie"]:
            m.head.weight.copy_(t("head.weight"))
    return m


if __name__ == "__main__":
    print(generate(sys.argv[1] if len(sys.argv) > 1 else "", n=300))
`;
}
