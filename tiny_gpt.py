"""tiny_gpt.py - a character-level GPT (decoder-only transformer) in plain PyTorch.
Read top to bottom: data -> tokenizer -> model -> training loop. Then sample with generate.py.
  python tiny_gpt.py --preset cpu   (~5 min on 2 CPU cores)  |  python tiny_gpt.py --preset gpu   (Colab)
  python tiny_gpt.py --data notes.txt --n_layer 2 --max_iters 500   (any text file, any flags)"""
import argparse, math, os, time, urllib.request
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------- data
def load_text(path=None):
    """Read --data, or download Tiny Shakespeare (1.1 MB) once and cache it in ./data/."""
    if path is None:
        path = os.path.join(HERE, "data", "tinyshakespeare.txt")
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print(f"downloading Tiny Shakespeare -> {path}")
            urllib.request.urlretrieve(DATA_URL, path)
    with open(path, encoding="utf-8") as f:
        return f.read()

class CharTokenizer:
    """Vocabulary = every distinct character (65 for Shakespeare): nothing is ever 'unknown', but
    sequences are long. Real LLMs use BPE subwords (50k-200k tokens) so each token carries more text."""
    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)

def get_batch(data, block_size, batch_size, device, generator=None):
    """Random windows of text. y is x shifted left by one, so EVERY position is a training example
    ('given chars 0..t, predict char t+1'): one window = block_size predictions at once."""
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ----------------------------------------------------------------------------- model
@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 128  # max context; there is no position embedding beyond it
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must split evenly across heads"
        self.n_head, self.head_dim = cfg.n_head, cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)  # one matmul makes queries, keys and values
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)     # mixes what the heads found back together
        self.attn_drop, self.resid_drop = nn.Dropout(cfg.dropout), nn.Dropout(cfg.dropout)
        # mask[i, j] is True when position i may look at position j, i.e. j <= i (never the future)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x, cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim): each head attends using its own slice of channels
        q, k, v = (t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v))
        if cache is not None:     # generation with a KV cache (see generate.py)
            if "k" in cache:      # earlier tokens' keys/values never change, so reuse them
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        past = k.size(2) - T      # how many earlier positions came from the cache
        # Score = how well each query matches each key. Dividing by sqrt(head_dim) keeps the scale
        # independent of head size; otherwise softmax saturates and its gradients vanish.
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)              # (B, nh, T, past+T)
        att = att.masked_fill(~self.mask[past:past + T, :past + T], float("-inf"))  # exp(-inf) = 0
        att = self.attn_drop(F.softmax(att, dim=-1))  # each row: weights over visible positions, sum 1
        y = att @ v                                   # weighted average of the values
        # Fused + faster equivalent (when past == 0): F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))

class Block(nn.Module):
    """Attention = tokens exchange information; MLP = each token processes what it gathered.
    Residual '+' gives gradients a direct path to early layers; pre-LN keeps that path clean (stable)."""
    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.n_embd), nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = nn.Sequential(nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(),
                                 nn.Linear(4 * cfg.n_embd, cfg.n_embd), nn.Dropout(cfg.dropout))

    def forward(self, x, cache=None):
        x = x + self.attn(self.ln1(x), cache)
        return x + self.mlp(self.ln2(x))

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # WHAT the token is
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)  # WHERE it is (attention is order-blind)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weight tying: one matrix for token -> vector (input) AND vector -> token scores (output)
        self.head.weight = self.tok_emb.weight
        for m in self.modules():  # small init -> near-uniform predictions -> step-0 loss ~ ln(vocab)
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        for name, p in self.named_parameters():  # GPT-2: layers that ADD into the residual stream start
            if name.endswith(("proj.weight", "mlp.2.weight")):  # smaller, so its variance doesn't grow with depth
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))

    def forward(self, idx, targets=None, cache=None, start_pos=0):
        """idx: (B, T) token ids. cache: list of one dict per layer (generation only).
        start_pos: absolute position of idx[:, 0] = number of tokens already in the cache."""
        B, T = idx.shape
        assert start_pos + T <= self.cfg.block_size, "sequence is longer than block_size"
        pos = torch.arange(start_pos, start_pos + T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for i, block in enumerate(self.blocks):
            x = block(x, None if cache is None else cache[i])
        logits = self.head(self.ln_f(x))  # (B, T, vocab): a score for every possible next char
        loss = None
        if targets is not None:  # cross-entropy = mean over all B*T positions of -log p(true next char)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# ----------------------------------------------------------------------------- training
PRESETS = {  # explicit flags still override a preset
    # 0.82M params, 3000 steps x 2048 tokens: ~4.7 min and val loss ~1.61 on 2 CPU cores
    "cpu": dict(n_layer=4, n_head=4, n_embd=128, block_size=128, batch_size=16, lr=2e-3,
                max_iters=3000, warmup_iters=100, eval_interval=500, eval_iters=20, dropout=0.0),
    # 10.7M params (nanoGPT's shakespeare-char size); dropout because it can overfit 1 MB of text
    "gpu": dict(n_layer=6, n_head=6, n_embd=384, block_size=256, batch_size=64, lr=1e-3,
                max_iters=3000, warmup_iters=100, eval_interval=250, eval_iters=50, dropout=0.2),
}

def get_args(argv=None):
    p = argparse.ArgumentParser(description="Train a character-level GPT.")
    a = p.add_argument
    a("--preset", choices=list(PRESETS), help="cpu: ~5 min on 2 cores; gpu: Colab-sized model")
    a("--data", help="any UTF-8 text file (default: Tiny Shakespeare, downloaded once)")
    a("--out", default=os.path.join(HERE, "checkpoints", "tiny_gpt.pt"), help="checkpoint; '' = don't save")
    a("--n_layer", type=int, default=4); a("--n_head", type=int, default=4); a("--n_embd", type=int, default=128)
    a("--block_size", type=int, default=128, help="context length in characters"); a("--dropout", type=float, default=0.0)
    a("--batch_size", type=int, default=32, help="windows per step (tokens/step = batch*block)")
    a("--lr", type=float, default=1e-3, help="peak learning rate (after warmup)")
    a("--min_lr", type=float, default=None, help="final learning rate of the cosine (default lr/10)")
    a("--warmup_iters", type=int, default=100); a("--max_iters", type=int, default=2000)
    a("--weight_decay", type=float, default=0.1); a("--grad_clip", type=float, default=1.0, help="0 = off")
    a("--eval_interval", type=int, default=250); a("--eval_iters", type=int, default=20)
    a("--seed", type=int, default=1337); a("--device", default="auto", help="auto | cpu | cuda | mps")
    known, _ = p.parse_known_args(argv)
    if known.preset:
        p.set_defaults(**PRESETS[known.preset])
    args = p.parse_args(argv)
    args.min_lr = args.lr / 10 if args.min_lr is None else args.min_lr
    return args

def pick_device(name="auto"):
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def get_lr(it, args):
    """Linear warmup (early Adam steps are noisy; big steps from a random init can blow up), then
    cosine decay to min_lr (big steps to explore early, small steps to settle into a minimum late)."""
    if it < args.warmup_iters:
        return args.lr * (it + 1) / args.warmup_iters
    progress = min(1.0, (it - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters))
    return args.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (args.lr - args.min_lr)

@torch.no_grad()
def estimate_loss(model, splits, args, device):
    """Mean loss over eval_iters batches. A fixed seed gives the SAME batches every time, so differences
    between evals (or between runs of a sweep) come from the model, not from the sample."""
    model.eval()  # turns dropout off
    out = {}
    for name, data in splits.items():
        g = torch.Generator().manual_seed(0)
        out[name] = sum(model(*get_batch(data, args.block_size, args.batch_size, device, g))[1].item()
                        for _ in range(args.eval_iters)) / args.eval_iters
    model.train()
    return out

def train(args, log=print):
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    text = load_text(args.data)
    tok = CharTokenizer(sorted(set(text)))
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))  # contiguous split: val is a passage the model never trains on
    splits = {"train": data[:n], "val": data[n:]}
    cfg = GPTConfig(len(tok.chars), args.block_size, args.n_layer, args.n_head, args.n_embd, args.dropout)
    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())  # tied weights are counted once
    log(f"device {device} | {len(data):,} chars, vocab {cfg.vocab_size} | {n_params:,} parameters | "
        f"uniform-guess loss ln({cfg.vocab_size}) = {math.log(cfg.vocab_size):.3f}")
    log(f"{cfg} | {args.batch_size}x{args.block_size} = {args.batch_size * args.block_size} tokens/step | "
        f"lr {args.lr:g} -> {args.min_lr:g}, warmup {args.warmup_iters}, {args.max_iters} steps")
    # Weight decay (pull weights toward 0) only on matrices; on LayerNorm gains/biases it just fights the norm
    groups = [{"params": [p for p in model.parameters() if p.dim() >= 2], "weight_decay": args.weight_decay},
              {"params": [p for p in model.parameters() if p.dim() < 2], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.99))
    history, best_val, diverged_at, grad_norm = [], float("inf"), None, float("nan")
    train_time, tokens, t_start = 0.0, 0, time.time()
    for it in range(args.max_iters + 1):
        if it % args.eval_interval == 0 or it == args.max_iters:
            losses = estimate_loss(model, splits, args, device)
            tps = tokens / train_time if train_time else 0.0
            history.append(dict(step=it, train=losses["train"], val=losses["val"], lr=get_lr(it, args),
                                grad_norm=float(grad_norm), tok_per_sec=tps))
            log(f"step {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | "
                f"val ppl {math.exp(min(losses['val'], 50)):7.2f} | lr {get_lr(it, args):.1e} | "
                f"grad {f'{grad_norm:5.2f}' if it else '    -'} | {tps:,.0f} tok/s | {time.time() - t_start:4.0f}s")
            if losses["val"] < best_val:
                best_val = losses["val"]
                if args.out:  # keep the best-on-val model; config + vocab make the file self-contained
                    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
                    torch.save({"model": model.state_dict(), "config": asdict(cfg), "vocab": tok.chars,
                                "step": it, "val_loss": best_val}, args.out)
        if it == args.max_iters:
            break
        t0 = time.time()
        for group in opt.param_groups:
            group["lr"] = get_lr(it, args)
        x, y = get_batch(splits["train"], args.block_size, args.batch_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Clip: if the gradient's norm > grad_clip, rescale it, so one bad batch can't fling the weights away
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip or float("inf"))
        opt.step()
        if not math.isfinite(loss.item()):  # .item() also waits for the GPU, so timing stays honest
            log(f"step {it}: loss is {loss.item()} -> diverged, stopping")
            diverged_at = it
            break
        train_time += time.time() - t0
        tokens += x.numel()
    return dict(model=model, tok=tok, cfg=cfg, device=device, params=n_params, history=history,
                best_val=best_val, diverged=diverged_at is not None, diverged_at=diverged_at, train_time=train_time,
                wall_time=time.time() - t_start, tok_per_sec=tokens / max(train_time, 1e-9))

if __name__ == "__main__":
    args = get_args()
    r = train(args)
    print(f"done: best val loss {r['best_val']:.4f} (ppl {math.exp(min(r['best_val'], 50)):.2f}) | "
          f"train {r['train_time']:.0f}s, wall {r['wall_time']:.0f}s | {r['tok_per_sec']:,.0f} tok/s"
          + (f" | best checkpoint -> {args.out}" if args.out else ""))
    from generate import generate  # same folder; reuse its sampler for a quick look
    print("--- sample (temperature 0.8) ---\n" + generate(r["model"], r["tok"], "\n", 300, temperature=0.8, seed=args.seed))
