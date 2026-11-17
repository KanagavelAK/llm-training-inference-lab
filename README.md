# LLM Training & Inference Lab

A character-level GPT written from scratch in plain PyTorch, trained on Tiny Shakespeare on a laptop CPU,
then used to measure how sampling settings and a KV cache change what the model writes and how fast it writes it.
A second part sends the same inference parameters to real OpenAI-compatible servers (Groq, Ollama, vLLM).

No GPU was used. Every number below comes from [`results.md`](results.md), written by `experiments.py`
on a 16-core Intel CPU (Windows, PyTorch 2.14 CPU build, October 2026).

## Results

| What | Result |
|---|---|
| Model | 4 layers, 4 heads, 128-wide, 128-character context, tied embeddings: **818,048 parameters** |
| Training | 3,000 AdamW steps (warmup + cosine decay, weight decay, gradient clipping), 6.1M tokens, 11.4 min on CPU |
| Validation loss | **4.21 → 1.60** (perplexity **67 → 5.0**) |
| Learning-rate sweep | 5 rates, same seed and batches: 0.01 was best at 400 steps; 0.03 overshot |
| KV cache | **Identical greedy tokens** to the naive loop; **1.6× faster at 128 tokens, 5.7× at 1,000 tokens** |

Sample after training (temperature 0.8):

```text
And with the look of heavily be he hand my heart,
And ever longer than patied Roman, the chail
```

### Temperature and top-p

5 samples of 200 characters per setting, prompt `ROMEO:`. "Real words" = share of words that appear in the training text.

| Setting | Distinct samples | Repeated word 3-grams | Real words |
|---|---:|---:|---:|
| Greedy (T = 0) | 1 / 5 | 41% | 100% |
| T = 0.5 | 5 / 5 | 0% | 97% |
| T = 1.0 | 5 / 5 | 0% | 80% |
| T = 1.5 | 5 / 5 | 0% | 57% |
| T = 1.0, top-p 0.9 | 5 / 5 | 0% | 92% |
| T = 1.5, top-p 0.9 | 5 / 5 | 0% | 69% |

Greedy decoding loops; high temperature produces non-words; top-p 0.9 cuts the unreliable tail and recovers
most of the coherence at the same temperature.

### Why the KV-cache speedup grows with length

The naive loop re-runs every previous token through every layer at each step (O(n²) work for n tokens).
The cached loop runs only the new token and reads stored keys and values (O(n)). At short lengths this tiny
model is dominated by fixed per-step overhead, so the gain is small; at 1,000 tokens it reaches 5.7×.
With learned absolute positions the cache can't slide, so it is rebuilt when the 128-token window fills.

## Files

| File | What it does |
|---|---|
| [`tiny_gpt.py`](tiny_gpt.py) | Tokenizer, causal self-attention, GPT model, training loop with warmup + cosine LR, checkpoints |
| [`generate.py`](generate.py) | Hand-written temperature / top-k / top-p sampling; naive vs KV-cached generation; `--compare` |
| [`experiments.py`](experiments.py) | Runs training, the LR sweep, the sampling study and the KV-cache benchmark, and writes `results.md` |
| [`params_lab.py`](params_lab.py) | Tests temperature, top_p, max_tokens, stop, seed and streaming (time to first token, tokens/s) against any OpenAI-compatible API using raw `httpx` |
| [`mock_openai.py`](mock_openai.py) | A fake OpenAI-compatible FastAPI server for testing `params_lab.py` offline (checks plumbing only) |
| [`checkpoints/tiny_gpt.pt`](checkpoints/tiny_gpt.pt) | The trained 818K model (3.3 MB), so `generate.py` works without training first |

## Run it

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build; skip on a GPU machine
pip install -r requirements.txt

python generate.py --prompt "ROMEO:" --temperature 0.8 --top_p 0.9   # sample the trained model
python generate.py --compare --max_new_tokens 500                    # KV cache vs naive: same tokens? how much faster?
python tiny_gpt.py --preset cpu                                      # retrain (~11 min on CPU; downloads Tiny Shakespeare once)
python experiments.py                                                # all four experiments -> results.md
```

Device is picked automatically (`cuda`, then `mps`, then `cpu`). `python tiny_gpt.py --preset gpu` trains a
10.7M-parameter version for a GPU runtime such as Colab.

### Real inference servers

```bash
# Ollama on your machine (ollama pull llama3.2:1b)
BASE_URL=http://localhost:11434/v1 API_KEY=ollama MODEL=llama3.2:1b python params_lab.py

# Groq
BASE_URL=https://api.groq.com/openai/v1 API_KEY=<your key> MODEL=openai/gpt-oss-120b python params_lab.py
```

## Concepts in the code

- **Initial loss ≈ ln(65) = 4.17**: a small init makes the model guess uniformly over the 65 characters.
- **Causal mask**: one slice covers both training and cached one-token decoding.
- **AdamW decay only on weight matrices**, warmup + cosine schedule and gradient clipping, each with a short "why" comment.
- **Perplexity = e^loss**: 5.0 means the model is as unsure as choosing among about 5 characters.
- **Temperature vs top-k vs top-p**: temperature reshapes the whole distribution, top-k keeps a fixed count, top-p keeps a fixed probability mass.

Architecture follows the GPT-2 / [nanoGPT](https://github.com/karpathy/nanoGPT) design; the dataset is
[Tiny Shakespeare](https://github.com/karpathy/char-rnn).
