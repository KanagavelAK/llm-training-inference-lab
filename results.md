# Results: tiny GPT experiments

- Hardware: 16 CPU cores (Intel64 Family 6 Model 154 Stepping 3, GenuineIntel), no GPU | device `cpu` | torch 2.14.1+cpu (12 threads) | Python 3.11.1 | Windows
- Written by `experiments.py` (last update 2026-10-12). Compute: (a) 11.4 min, (b) 7.4 min, (c) 0.4 min, (d) 2.6 min; total 21.9 min
- 1-min load average at the end of each section: (a) nan, (b) nan, (c) nan, (d) nan. About 12.0 = this script alone; higher = other jobs were competing for the CPU, so times and tokens/s are slower and noisier than on an idle machine. Losses do not depend on speed.

## (a) Training run: `python tiny_gpt.py --preset cpu`
_Section (a): 11.4 min, load nan_

- Model: 4 layers, 4 heads, n_embd 128, block_size 128, dropout 0.0, tied embeddings: **818,048 parameters**
- Data: Tiny Shakespeare, 1,115,394 characters, vocab 65, 90/10 train/val split
- Budget: 3000 steps x 16x128 tokens = 6.1M tokens (~6.1 passes over the training text); AdamW, peak lr 0.002, 100 warmup steps, cosine to 0.0002, weight decay 0.1, clip 1.0
- Start: val loss 4.205 (uniform guess over 65 chars = ln 65 = 4.174)
- End: train 1.3940, **val 1.6007, perplexity 4.96** (saved checkpoint: step 3000, val 1.6007)
- Time: **682 s wall** (11.4 min, incl. evals), 670 s in training steps; **9,163 training tokens/s**

| step | train loss | val loss | val perplexity | lr |
|---:|---:|---:|---:|---:|
| 0 | 4.2094 | 4.2052 | 67.04 | 2.0e-05 |
| 500 | 2.1170 | 2.1656 | 8.72 | 1.9e-03 |
| 1000 | 1.7761 | 1.9266 | 6.87 | 1.6e-03 |
| 1500 | 1.6033 | 1.8018 | 6.06 | 1.1e-03 |
| 2000 | 1.4959 | 1.6996 | 5.47 | 6.8e-04 |
| 2500 | 1.4279 | 1.6367 | 5.14 | 3.3e-04 |
| 3000 | 1.3940 | 1.6007 | 4.96 | 2.0e-04 |

Perplexity = exp(loss): 67 at step 0 (about a uniform guess among 65 characters), 5.0 at the end (as unsure as choosing among ~5 characters).

Sample: 300 new characters, temperature 0.8, seed 1337, prompt = a newline:

```text
Cannot thee enemy honour's nate wail, for and thy weak
Thou to state and have heimself fland;
To it smy more shallow's and Buckio's news:
If I am power your in this father's troath;
And with the look of heavily be he hand my heart,
And ever longer than patied Roman, the chail
for lie. But how is the
```

## (b) Learning-rate sweep: same model, seed and batches, 400 steps each (40 warmup, cosine to lr/10)
_Section (b): 7.4 min, load nan_

| peak lr | val @ 0 | val @ 100 | val @ 200 | val @ 300 | val @ 400 | outcome |
|---:|---:|---:|---:|---:|---:|---|
| 0.0003 | 4.205 | 2.675 | 2.524 | 2.476 | 2.451 | +0.362 vs best |
| 0.001 | 4.205 | 2.536 | 2.424 | 2.363 | 2.321 | +0.232 vs best |
| 0.003 | 4.205 | 2.529 | 2.414 | 2.278 | 2.196 | +0.107 vs best |
| 0.01 | 4.205 | 2.548 | 2.431 | 2.199 | 2.089 | **best** |
| 0.03 | 4.205 | 3.130 | 2.921 | 2.758 | 2.719 | +0.631 vs best |

Same seed means identical initial weights and identical batches, so the learning rate is the only difference between rows. Too small: every step is safe but tiny, so a fixed budget ends early on the curve. Too large: Adam moves every weight by roughly lr per step, which overshoots the minimum (warmup and clipping delay this, they do not prevent it). Best of these: 0.01. The best LR depends on the budget: short runs reward aggressive LRs, longer runs (like the 3000-step preset, which uses 0.002) can afford smaller, steadier steps.

## (c) Temperature and top-p: prompt `ROMEO:\n`, 5 samples x 200 new characters per setting (seeds 0-4)
_Section (c): 0.4 min, load nan_

| setting | distinct samples | distinct-1 (unique words / words) | distinct-2 (unique word pairs / pairs) | repeated word 3-grams within a sample | real words (seen in Tiny Shakespeare) |
|---|---:|---:|---:|---:|---:|
| greedy (T=0) | 1/5 | 0.07 | 0.10 | 41% | 100% |
| T=0.5 | 5/5 | 0.53 | 0.89 | 0% | 97% |
| T=1.0 | 5/5 | 0.72 | 0.99 | 0% | 80% |
| T=1.5 | 5/5 | 0.88 | 1.00 | 0% | 57% |
| T=1.0, top-p 0.9 | 5/5 | 0.68 | 0.98 | 0% | 92% |
| T=1.5, top-p 0.9 | 5/5 | 0.78 | 1.00 | 0% | 69% |

Higher distinct-n = more variety across samples; repeated 3-grams = loops; real words = coherence. Temperature divides the logits before softmax (T<1 sharpens, T>1 flattens); top-p keeps only the most likely tokens covering 90% of the probability and renormalises, cutting the unreliable tail.

First 150 characters of sample #1 per setting (` / ` = newline):

```text
[greedy (T=0)] I have thee thee would not the son of the sea / To be the stand of the state of the state, / That will the state of the state of the state, / That will be t
[T=0.5] Thou make the present of thee have and the strength / And you that here hath the seems of men. /  / PRINCE EDWARD: / Will you have not the rest; my lord, the 
[T=1.0] Thou must know self Warwick exparitate I to thee, / For wedged now here hath thou beheld for serve / We attory or wake a when my fear's, none hands / How th
[T=1.5] thorefhick haw! Then, o'ch nevery youd I tost, greach,--you fot; / Thre hath Doth behelinBeans, impellation / portick, Richard idard, trunk-davantas! / He p
[T=1.0, top-p 0.9] Thou must there not made a means! you will strange / of thy hours; here hath not seemely so. /  / Shepherd: / No your way, a man's is did to lend him the even
[T=1.5, top-p 0.9] Thyrefore keep selter? / Hou nexhall, in I to tell, come with forward. / Our judgmes of lies good injuriation / To yielder ancosenate; were no motisfy! / Hope
```

## (d) KV cache vs naive generation (greedy, prompt `ROMEO:\n`, best of 3 timings)
_Section (d): 2.6 min, load nan_

Correctness: greedy decoding of the 121 new tokens that fit in one 128-token window gives **identical tokens: True**.

Trained model (block_size 128):

| new tokens | naive (s) | KV cache (s) | naive tok/s | cached tok/s | speedup | same tokens for first |
|---:|---:|---:|---:|---:|---:|---:|
| 30 | 0.146 | 0.115 | 206 | 260 | **1.3x** | 30 of 30 |
| 60 | 0.304 | 0.223 | 197 | 269 | **1.4x** | 60 of 60 |
| 120 | 0.724 | 0.462 | 166 | 260 | **1.6x** | 120 of 120 |
| 500 | 3.491 | 1.948 | 143 | 257 | **1.8x** | 256 of 500 |

Same architecture with block_size 1024 and random weights, to see longer contexts:

| new tokens | naive (s) | KV cache (s) | naive tok/s | cached tok/s | speedup | same tokens for first |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.749 | 0.483 | 171 | 265 | **1.6x** | 128 of 128 |
| 256 | 1.955 | 1.000 | 131 | 256 | **2.0x** | 256 of 256 |
| 512 | 6.595 | 2.169 | 78 | 236 | **3.0x** | 512 of 512 |
| 1000 | 25.352 | 4.410 | 39 | 227 | **5.7x** | 1000 of 1000 |

Why the speedup grows with length: the naive loop re-runs all t tokens through every layer at step t (O(n^2) total work for n tokens) while the cached loop runs only the new token and reads t stored keys/values (O(n) through the MLPs), so the gap widens roughly linearly with length.

Why it is only ~1.8x at 500 tokens for the trained model: block_size 128 caps the naive context at 128 tokens, the cache is rebuilt from the last 64 tokens whenever the window is full (learned absolute positions go stale when the window slides), and a one-token step of this tiny model is dominated by fixed per-operation overhead (~100 small PyTorch ops per token). After a rebuild the cached loop sees a shorter context than the naive one, so outputs past the first window may differ (the 'same tokens' column).
