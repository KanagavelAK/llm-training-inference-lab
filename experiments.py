"""
experiments.py - runs the experiments whose numbers you quote and writes them to results.md.
  (a) the --preset cpu training run         (b) learning-rate sweep with equal budgets
  (c) temperature / top-p vs diversity      (d) KV cache vs naive generation speed
  python experiments.py                     # all four (~10-15 min on 2 CPU cores)
  python experiments.py d                   # re-run one section; the others in results.md are kept
                                            # (c and d reuse the checkpoint written by a)
  python experiments.py --quick             # tiny budgets: checks that everything runs (~1 min)
"""
import argparse, json, math, os, platform, re, time
from dataclasses import asdict

import torch

import tiny_gpt as tg
from generate import load_model, generate, compare

PROMPT = "ROMEO:\n"


def hardware(device):
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        cpu = platform.processor() or platform.machine()
    if device == "cuda":
        gpu = torch.cuda.get_device_name(0)
    elif device == "mps":
        gpu = "Apple GPU (mps)"
    else:
        gpu = "GPU present but not used" if torch.cuda.is_available() else "no GPU"
    return (f"{os.cpu_count()} CPU cores ({cpu}), {gpu} | device `{device}` | torch {torch.__version__} "
            f"({torch.get_num_threads()} threads) | Python {platform.python_version()} | {platform.system()}")


# ----------------------------------------------------------------------------- (a) training
def run_a(o):
    argv = ["--preset", "cpu", "--out", o.ckpt, "--device", o.device]
    if o.quick:
        argv += ["--max_iters", "60", "--eval_interval", "20", "--eval_iters", "2"]
    args = tg.get_args(argv)
    r = tg.train(args)
    model, tok, ckpt = load_model(o.ckpt, torch.device(r["device"]))
    sample = generate(model, tok, "\n", 300, temperature=0.8, seed=args.seed)
    h, n_tok, n_chars = r["history"], args.max_iters * args.batch_size * args.block_size, len(tg.load_text())
    md = ["## (a) Training run: `python tiny_gpt.py --preset cpu`", "",
          f"- Model: {r['cfg'].n_layer} layers, {r['cfg'].n_head} heads, n_embd {r['cfg'].n_embd}, "
          f"block_size {r['cfg'].block_size}, dropout {r['cfg'].dropout}, tied embeddings: **{r['params']:,} parameters**",
          f"- Data: Tiny Shakespeare, {n_chars:,} characters, vocab {r['cfg'].vocab_size}, 90/10 train/val split",
          f"- Budget: {args.max_iters} steps x {args.batch_size}x{args.block_size} tokens = {n_tok / 1e6:.1f}M tokens "
          f"(~{n_tok / int(0.9 * n_chars):.1f} passes over the training text); AdamW, peak lr {args.lr:g}, "
          f"{args.warmup_iters} warmup steps, cosine to {args.min_lr:g}, weight decay {args.weight_decay}, clip {args.grad_clip}",
          f"- Start: val loss {h[0]['val']:.3f} (uniform guess over {r['cfg'].vocab_size} chars = "
          f"ln {r['cfg'].vocab_size} = {math.log(r['cfg'].vocab_size):.3f})",
          f"- End: train {h[-1]['train']:.4f}, **val {h[-1]['val']:.4f}, perplexity {math.exp(h[-1]['val']):.2f}** "
          f"(saved checkpoint: step {ckpt['step']}, val {ckpt['val_loss']:.4f})",
          f"- Time: **{r['wall_time']:.0f} s wall** ({r['wall_time'] / 60:.1f} min, incl. evals), "
          f"{r['train_time']:.0f} s in training steps; **{r['tok_per_sec']:,.0f} training tokens/s**", "",
          "| step | train loss | val loss | val perplexity | lr |", "|---:|---:|---:|---:|---:|"]
    md += [f"| {e['step']} | {e['train']:.4f} | {e['val']:.4f} | {math.exp(e['val']):.2f} | {e['lr']:.1e} |" for e in h]
    md += ["", f"Perplexity = exp(loss): {math.exp(h[0]['val']):.0f} at step 0 (about a uniform guess among "
           f"{r['cfg'].vocab_size} characters), "
           f"{math.exp(h[-1]['val']):.1f} at the end (as unsure as choosing among ~{math.exp(h[-1]['val']):.0f} characters).",
           "", "Sample: 300 new characters, temperature 0.8, seed 1337, prompt = a newline:", "", "```text",
           sample.strip("\n"), "```"]
    return "\n".join(md)


# ----------------------------------------------------------------------------- (b) learning rates
def run_b(o):
    lrs, iters = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2], (40 if o.quick else 400)
    runs = []
    for lr in lrs:
        argv = ["--preset", "cpu", "--lr", str(lr), "--max_iters", str(iters), "--warmup_iters", str(iters // 10),
                "--eval_interval", str(iters // 4), "--out", "", "--device", o.device]
        r = tg.train(tg.get_args(argv + (["--eval_iters", "2"] if o.quick else [])),
                     log=lambda s, lr=lr: print(f"[lr {lr:g}] {s}"))
        runs.append((lr, r))
    steps = [e["step"] for e in max((r["history"] for _, r in runs), key=len)]
    ok = [(r["history"][-1]["val"], lr) for lr, r in runs if not r["diverged"]]
    best_val, best_lr = min(ok) if ok else (float("nan"), None)
    md = [f"## (b) Learning-rate sweep: same model, seed and batches, {iters} steps each "
          f"({iters // 10} warmup, cosine to lr/10)", "",
          "| peak lr | " + " | ".join(f"val @ {s}" for s in steps) + " | outcome |",
          "|---:|" + "---:|" * len(steps) + "---|"]
    for lr, r in runs:
        vals = {e["step"]: e["val"] for e in r["history"]}
        final, lowest = r["history"][-1]["val"], min(vals.values())
        if r["diverged"]:
            outcome = f"**diverged**: loss became NaN/inf at step {r['diverged_at']}, run stopped"
        elif lr == best_lr:
            outcome = "**best**"
        elif final > lowest + 0.05:
            outcome = f"unstable: val rose from {lowest:.3f} back up to {final:.3f}"
        else:
            outcome = f"+{final - best_val:.3f} vs best"
        md.append(f"| {lr:g} | " + " | ".join(f"{vals[s]:.3f}" if s in vals else "-" for s in steps) + f" | {outcome} |")
    md += ["", f"Same seed means identical initial weights and identical batches, so the learning rate is the only "
           f"difference between rows. Too small: every step is safe but tiny, so a fixed budget ends early on the "
           f"curve. Too large: Adam moves every weight by roughly lr per step, which overshoots the minimum "
           f"(warmup and clipping delay this, they do not prevent it). Best of these: "
           f"{'none (all diverged)' if best_lr is None else f'{best_lr:g}'}. "
           f"The best LR depends on the budget: short runs reward aggressive LRs, longer runs (like the "
           f"{tg.PRESETS['cpu']['max_iters']}-step preset, which uses {tg.PRESETS['cpu']['lr']:g}) can afford "
           f"smaller, steadier steps."]
    return "\n".join(md)


# ----------------------------------------------------------------------------- (c) sampling
def words(s):
    return re.findall(r"[a-z]+(?:'[a-z]+)*", s.lower())


def diversity(samples, vocab):
    ws = [words(s) for s in samples]
    flat = [w for x in ws for w in x]
    bigrams = [b for x in ws for b in zip(x, x[1:])]
    rep3 = [1 - len(set(t)) / len(t) for x in ws if (t := list(zip(x, x[1:], x[2:])))]
    return dict(unique=len(set(samples)), d1=len(set(flat)) / max(1, len(flat)),
                d2=len(set(bigrams)) / max(1, len(bigrams)), rep3=sum(rep3) / max(1, len(rep3)),
                real=sum(w in vocab for w in flat) / max(1, len(flat)))


def run_c(o):
    model, tok, _ = load_model(o.ckpt, torch.device(o.device))
    vocab = set(words(tg.load_text()))
    n, new = (2, 40) if o.quick else (5, 200)
    settings = [("greedy (T=0)", 0.0, 1.0), ("T=0.5", 0.5, 1.0), ("T=1.0", 1.0, 1.0), ("T=1.5", 1.5, 1.0),
                ("T=1.0, top-p 0.9", 1.0, 0.9), ("T=1.5, top-p 0.9", 1.5, 0.9)]
    rows, snippets = [], []
    for name, t, p in settings:
        samples = [generate(model, tok, PROMPT, new, temperature=t, top_p=p, seed=s)[len(PROMPT):] for s in range(n)]
        d = diversity(samples, vocab)
        print(f"[{name}] {d}")
        rows.append(f"| {name} | {d['unique']}/{n} | {d['d1']:.2f} | {d['d2']:.2f} | {d['rep3']:.0%} | {d['real']:.0%} |")
        snippets.append(f"[{name}] " + samples[0][:150].replace("\n", " / "))
    md = [f"## (c) Temperature and top-p: prompt `ROMEO:\\n`, {n} samples x {new} new characters per setting (seeds 0-{n - 1})", "",
          "| setting | distinct samples | distinct-1 (unique words / words) | distinct-2 (unique word pairs / pairs) "
          "| repeated word 3-grams within a sample | real words (seen in Tiny Shakespeare) |",
          "|---|---:|---:|---:|---:|---:|", *rows, "",
          "Higher distinct-n = more variety across samples; repeated 3-grams = loops; real words = coherence. "
          "Temperature divides the logits before softmax (T<1 sharpens, T>1 flattens); top-p keeps only the most "
          "likely tokens covering 90% of the probability and renormalises, cutting the unreliable tail.", "",
          "First 150 characters of sample #1 per setting (` / ` = newline):", "", "```text", *snippets, "```"]
    return "\n".join(md)


# ----------------------------------------------------------------------------- (d) KV cache
def run_d(o):
    device = torch.device(o.device)
    model, tok, _ = load_model(o.ckpt, device)
    lengths, long_lengths = ([10, 20], [32, 64]) if o.quick else ([30, 60, 120, 500], [128, 256, 512, 1000])
    trained = [compare(model, tok, PROMPT, n, repeats=3, check=(i == 0)) for i, n in enumerate(lengths)]
    torch.manual_seed(0)  # same architecture, block_size 1024, random weights: speed doesn't depend on values
    big = tg.GPT(tg.GPTConfig(**{**asdict(model.cfg), "block_size": 1024})).to(device).eval()
    longc = [compare(big, tok, PROMPT, n, repeats=3, check=False) for n in long_lengths]
    for r in trained + longc:
        print(f"[kv] {r}")

    def table(rs):
        out = ["| new tokens | naive (s) | KV cache (s) | naive tok/s | cached tok/s | speedup | same tokens for first |",
               "|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rs:
            n = r["new_tokens"]
            out.append(f"| {n} | {r['naive_s']:.3f} | {r['cached_s']:.3f} | {n / r['naive_s']:.0f} | "
                       f"{n / r['cached_s']:.0f} | **{r['speedup']:.1f}x** | {r['matching_prefix']} of {n} |")
        return out
    t0 = trained[0]
    md = ["## (d) KV cache vs naive generation (greedy, prompt `ROMEO:\\n`, best of 3 timings)", "",
          f"Correctness: greedy decoding of the {t0['n_fit']} new tokens that fit in one {model.cfg.block_size}-token window "
          f"gives **identical tokens: {t0['identical_in_window']}**.", "",
          f"Trained model (block_size {model.cfg.block_size}):", "", *table(trained), "",
          "Same architecture with block_size 1024 and random weights, to see longer contexts:", "",
          *table(longc), "",
          "Why the speedup grows with length: the naive loop re-runs all t tokens through every layer at step t "
          "(O(n^2) total work for n tokens) while the cached loop runs only the new token and reads t stored "
          "keys/values (O(n) through the MLPs), so the gap widens roughly linearly with length.",
          "",
          f"Why it is only ~{trained[-1]['speedup']:.1f}x at {trained[-1]['new_tokens']} tokens for the trained model: "
          f"block_size {model.cfg.block_size} "
          f"caps the naive context at {model.cfg.block_size} tokens, the cache is rebuilt from the last "
          f"{model.cfg.block_size // 2} tokens whenever the window is full (learned absolute positions go stale when the "
          "window slides), and a one-token step of this tiny model is dominated by fixed per-operation overhead "
          "(~100 small PyTorch ops per token). After a rebuild the cached loop sees a shorter context than the naive one, "
          "so outputs past the first window may differ (the 'same tokens' column)."]
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sections", nargs="*", default=["a", "b", "c", "d"], help="any of a b c d")
    ap.add_argument("--out", help="default: results.md (results_quick.md with --quick)")
    ap.add_argument("--quick", action="store_true", help="tiny budgets, separate checkpoint: a smoke test")
    ap.add_argument("--device", default="auto")
    o = ap.parse_args()
    o.device = tg.pick_device(o.device)
    o.out = o.out or os.path.join(tg.HERE, "results_quick.md" if o.quick else "results.md")
    o.ckpt = os.path.join(tg.HERE, "checkpoints", "quick.pt" if o.quick else "tiny_gpt.pt")
    if any(s in o.sections for s in "cd") and "a" not in o.sections and not os.path.exists(o.ckpt):
        raise SystemExit(f"{o.ckpt} not found: run section a first (python experiments.py a)")
    store_path = os.path.splitext(o.out)[0] + ".json"  # every section's text, so re-running one keeps the rest
    store = {}
    if os.path.exists(store_path):
        with open(store_path, encoding="utf-8") as f:
            store = json.load(f)
    for s in o.sections:
        t0 = time.time()
        print(f"\n===== section ({s}) =====")
        md = {"a": run_a, "b": run_b, "c": run_c, "d": run_d}[s](o)
        store[s] = {"md": md, "minutes": (time.time() - t0) / 60, "hardware": hardware(o.device),
                    "load": os.getloadavg()[0] if hasattr(os, "getloadavg") else float("nan")}
        with open(store_path, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=1)
        render(store, o)
        print(f"[wrote {o.out}]")


def render(store, o):
    keys = [k for k in "abcd" if k in store]
    hw = sorted({store[k]["hardware"] for k in keys})
    out = ["# Results: tiny GPT experiments" + (" (QUICK smoke test, not real numbers)" if o.quick else ""), "",
           f"- Hardware: {hw[0] if len(hw) == 1 else ' / '.join(hw) + ' (see each section)'}",
           f"- Written by `experiments.py` (last update {time.strftime('%Y-%m-%d')}). Compute: "
           + ", ".join(f"({k}) {store[k]['minutes']:.1f} min" for k in keys)
           + f"; total {sum(store[k]['minutes'] for k in keys):.1f} min",
           "- 1-min load average at the end of each section: " + ", ".join(f"({k}) {store[k]['load']:.1f}" for k in keys)
           + f". About {torch.get_num_threads()}.0 = this script alone; higher = other jobs were competing for the CPU, "
           "so times and tokens/s are slower and noisier than on an idle machine. Losses do not depend on speed.", ""]
    for k in keys:
        head, _, body = store[k]["md"].partition("\n")
        extra = f" | {store[k]['hardware']}" if len(hw) > 1 else ""
        out += [head, f"_Section ({k}): {store[k]['minutes']:.1f} min, load {store[k]['load']:.1f}{extra}_", body, ""]
    with open(o.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    main()
