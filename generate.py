"""
generate.py - sample from a tiny_gpt.py checkpoint. Sampling is written by hand, and there are two
generation loops side by side: naive (recompute everything) and KV-cached.

  python generate.py --prompt "ROMEO:" --temperature 0.8 --top_p 0.9
  python generate.py --temperature 0                 # greedy
  python generate.py --compare --max_new_tokens 500  # same tokens? how much faster is the cache?
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F

from tiny_gpt import GPT, GPTConfig, CharTokenizer, HERE, pick_device


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)  # weights_only by default: tensors + plain types only
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval(), CharTokenizer(ckpt["vocab"]), ckpt  # eval(): dropout off for inference


def sample_next(logits, temperature=1.0, top_k=0, top_p=1.0):
    """logits: (B, vocab) scores for the next token -> (B, 1) chosen ids."""
    if temperature == 0:                         # greedy: always the single most likely token
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature                # T < 1 sharpens the distribution, T > 1 flattens it
    if top_k > 0:                                # keep only the k highest-scoring tokens
        kth_best = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
        logits = logits.masked_fill(logits < kth_best, float("-inf"))
    if top_p < 1.0:                              # nucleus: smallest set of tokens holding >= top_p mass
        sorted_logits, order = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        mass_before = probs.cumsum(dim=-1) - probs          # mass of all strictly better tokens
        sorted_logits[mass_before > top_p] = float("-inf")  # the top token always survives
        logits = torch.full_like(logits, float("-inf")).scatter(1, order, sorted_logits)
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)  # draw from what is left


@torch.no_grad()
def generate_naive(model, idx, max_new_tokens, **sampling):
    """No cache: every step re-runs the WHOLE context, although only the last position's logits are
    used. Step t pushes all t tokens through every layer: ~n^2/2 token-passes for n tokens, not n."""
    for _ in range(max_new_tokens):
        context = idx[:, -model.cfg.block_size:]  # positions only exist up to block_size - 1
        logits, _ = model(context)
        idx = torch.cat([idx, sample_next(logits[:, -1, :], **sampling)], dim=1)
    return idx


@torch.no_grad()
def generate_cached(model, idx, max_new_tokens, **sampling):
    """KV cache: run the prompt once ('prefill') and keep every layer's keys and values; then feed
    ONLY the newest token. Its query attends over the stored K/V and its own K/V are appended.
    Earlier tokens' K/V never change (the causal mask stops them seeing later tokens), so the
    naive loop recomputes them for nothing. Identical tokens to generate_naive while
    prompt + output fits in block_size; see the comment below for what happens after that."""
    block = model.cfg.block_size

    def prefill(context):
        cache = [{} for _ in model.blocks]  # per layer: {'k': (B, heads, T, head_dim), 'v': same}
        logits, _ = model(context, cache=cache)
        return cache, logits[:, -1, :]

    cache, logits = prefill(idx[:, -block:])
    for step in range(max_new_tokens):
        idx = torch.cat([idx, sample_next(logits, **sampling)], dim=1)
        if step == max_new_tokens - 1:
            break                                   # last token chosen; no need to run the model again
        n_cached = cache[0]["k"].size(2)
        if n_cached < block:                        # usual case: 1 token in, 1 position appended
            logits, _ = model(idx[:, -1:], cache=cache, start_pos=n_cached)
            logits = logits[:, -1, :]
        else:
            # Window full. With learned ABSOLUTE positions each cached K/V means "this token at
            # position p"; sliding the window renumbers every token, so the whole cache goes stale.
            # Rebuild it from the newest half-window (one prefill) and carry on: one prefill per
            # block/2 tokens, but the context shrinks to block/2 right after a rebuild, which is why
            # long outputs can differ from the naive loop. (Sliding-window models such as Mistral
            # use a rolling buffer instead.)
            cache, logits = prefill(idx[:, -max(1, block // 2):])
    return idx


def generate(model, tok, prompt, max_new_tokens, temperature=1.0, top_k=0, top_p=1.0, seed=None,
             use_cache=True):
    """Text in, text out (prompt + continuation)."""
    if seed is not None:
        torch.manual_seed(seed)  # torch.multinomial draws from the global RNG
    prompt = prompt or tok.chars[0]  # empty prompt -> start from the first vocab char ('\n' here)
    unknown = sorted(set(prompt) - set(tok.stoi))
    if unknown:
        raise ValueError(f"prompt contains characters the model never saw: {unknown}")
    model.eval()
    idx = torch.tensor([tok.encode(prompt)], device=next(model.parameters()).device)
    loop = generate_cached if use_cache else generate_naive
    out = loop(model, idx, max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
    return tok.decode(out[0].tolist())


def _sync(device):  # GPU kernels run asynchronously; wait for them before reading the clock
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def compare(model, tok, prompt, max_new_tokens=500, repeats=3, check=True):
    """Greedy decoding both ways. (1) check: within one context window the tokens must be identical.
    (2) Best-of-`repeats` wall-clock time for max_new_tokens tokens, naive vs cached."""
    model.eval()
    device = next(model.parameters()).device
    idx = torch.tensor([tok.encode(prompt)], device=device)
    n_fit = model.cfg.block_size - idx.size(1)  # new tokens that fit in one window
    res = {"n_fit": n_fit, "identical_in_window": None, "new_tokens": max_new_tokens}
    if check:
        res["identical_in_window"] = torch.equal(generate_naive(model, idx, n_fit, temperature=0),
                                                 generate_cached(model, idx, n_fit, temperature=0))
    outs = {}
    for name, loop in (("naive", generate_naive), ("cached", generate_cached)):
        loop(model, idx, 8, temperature=0)  # warm-up: first calls pay one-off allocation costs
        times = []
        for _ in range(repeats):
            _sync(device)
            t0 = time.perf_counter()
            outs[name] = loop(model, idx, max_new_tokens, temperature=0)
            _sync(device)
            times.append(time.perf_counter() - t0)
        res[f"{name}_s"] = min(times)
    a, b = outs["naive"][0, idx.size(1):].tolist(), outs["cached"][0, idx.size(1):].tolist()
    res["matching_prefix"] = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), len(a))
    res["speedup"] = res["naive_s"] / res["cached_s"]
    return res


def main():
    p = argparse.ArgumentParser(description="Sample from a tiny_gpt.py checkpoint.")
    p.add_argument("--ckpt", default=os.path.join(HERE, "checkpoints", "tiny_gpt.pt"))
    p.add_argument("--prompt", default="ROMEO:")
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    p.add_argument("--top_k", type=int, default=0, help="keep the k most likely tokens (0 = off)")
    p.add_argument("--top_p", type=float, default=1.0, help="nucleus sampling mass (1.0 = off)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--no_cache", action="store_true", help="use the naive recompute-everything loop")
    p.add_argument("--compare", action="store_true", help="naive vs KV cache: correctness + speed")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    args = p.parse_args()
    device = pick_device(args.device)
    model, tok, ckpt = load_model(args.ckpt, device)
    print(f"loaded {args.ckpt} (step {ckpt['step']}, val loss {ckpt['val_loss']:.4f}) on {device}")
    if args.compare:
        r = compare(model, tok, args.prompt, args.max_new_tokens)
        print(f"greedy, {r['n_fit']} tokens inside one window: identical = {r['identical_in_window']}")
        print(f"{r['new_tokens']} new tokens: naive {r['naive_s']:.2f}s ({r['new_tokens'] / r['naive_s']:.0f} tok/s)"
              f" | cached {r['cached_s']:.2f}s ({r['new_tokens'] / r['cached_s']:.0f} tok/s)"
              f" | speedup {r['speedup']:.1f}x | outputs agree for the first {r['matching_prefix']} tokens")
        return
    for i in range(args.num_samples):
        t0 = time.perf_counter()
        text = generate(model, tok, args.prompt, args.max_new_tokens, args.temperature, args.top_k,
                        args.top_p, seed=args.seed + i, use_cache=not args.no_cache)
        print(f"--- sample {i + 1} ({args.max_new_tokens / (time.perf_counter() - t0):.0f} tok/s) ---\n{text}")


if __name__ == "__main__":
    main()
