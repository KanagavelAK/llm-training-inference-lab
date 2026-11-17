"""
params_lab.py - see what each inference parameter does, against ANY OpenAI-compatible chat endpoint.
Uses httpx (plain HTTP + JSON, no SDK), so you see exactly what goes over the wire.

  # Groq: its free models (openai/gpt-oss-120b, openai/gpt-oss-20b) are reasoning models whose hidden
  # thinking uses up max_tokens, so the short-reply tests come back empty. Use Groq for speed only,
  # and run the parameter tests on Ollama.
  export BASE_URL=https://api.groq.com/openai/v1 API_KEY=gsk_... MODEL=openai/gpt-oss-120b
  # Ollama:  BASE_URL=http://localhost:11434/v1 API_KEY=ollama MODEL=llama3.2:1b
  # vLLM:    BASE_URL=http://localhost:8000/v1  API_KEY=x      MODEL=<the model vllm is serving>
  # offline: python mock_openai.py &  then  BASE_URL=http://127.0.0.1:8001/v1 API_KEY=x MODEL=mock-1
  python params_lab.py
"""
import json
import os
import time

import httpx

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/")
MODEL = os.environ.get("MODEL", "mock-1")
client = httpx.Client(base_url=BASE_URL, timeout=120,
                      headers={"Authorization": f"Bearer {os.environ.get('API_KEY', 'none')}"})
NAME_PROMPT = "Invent a name for a new coffee shop. Reply with the name only."
ROWS = []


def chat(prompt, **params):
    """One non-streaming call. Returns (text, finish_reason, usage)."""
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], **params}
    r = client.post("/chat/completions", json=body)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    choice = data["choices"][0]
    return (choice["message"].get("content") or "").strip(), choice.get("finish_reason"), data.get("usage") or {}


def stream(prompt, include_usage=True, **params):
    """Streaming call (server-sent events). Measures time to first token (TTFT: queue + prefill)
    and decode speed (tokens/s after the first token)."""
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "stream": True, **params}
    if include_usage:  # ask the server to append exact token counts in a final chunk
        body["stream_options"] = {"include_usage": True}
    t0, first, last, pieces, usage = time.perf_counter(), None, None, [], None
    with client.stream("POST", "/chat/completions", json=body) as r:
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.read().decode()[:200]}")
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue  # SSE: blank lines and comments separate events
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            usage = chunk.get("usage") or (chunk.get("x_groq") or {}).get("usage") or usage  # Groq: x_groq
            for ch in chunk.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    last = time.perf_counter()
                    first = first or last
                    pieces.append(piece)
    if first is None:
        raise RuntimeError("no content tokens streamed (a reasoning model may put them elsewhere)")
    n = usage["completion_tokens"] if usage else len(pieces)  # fall back to counting chunks
    tps = (n - 1) / (last - first) if last > first else float("nan")
    return "".join(pieces), first - t0, tps, n, "usage" if usage else "chunks", time.perf_counter() - t0


def row(test, setting, result):
    ROWS.append((test, setting, result))
    print(f"[{test}] {setting} -> {result}")


def short(texts, width=28):
    return " | ".join(t.replace("\n", " ")[:width] for t in texts)


def run(name, fn):
    try:
        fn()
    except Exception as e:  # one unsupported parameter should not stop the other tests
        row(name, "ERROR", str(e)[:150])


def t_temperature():  # T rescales logits before softmax: 0 = argmax, higher = flatter = more varied
    for t in (0, 0.7, 1.3):
        outs = [chat(NAME_PROMPT, temperature=t, max_tokens=20)[0] for _ in range(3)]
        row("temperature", f"T={t}, 3 calls", f"{len(set(outs))}/3 distinct: {short(outs)}")


def t_top_p():  # nucleus: sample only from the smallest token set holding top_p of the probability
    for p in (0.1, 1.0):
        outs = [chat(NAME_PROMPT, temperature=1.0, top_p=p, max_tokens=20)[0] for _ in range(3)]
        row("top_p", f"top_p={p}, T=1.0, 3 calls", f"{len(set(outs))}/3 distinct: {short(outs)}")


def t_max_tokens():  # a hard cap on generated tokens; finish_reason says WHY generation stopped
    for n in (5, 300):
        text, fr, usage = chat("Explain what a KV cache is in two sentences.", temperature=0, max_tokens=n)
        row("max_tokens", f"max_tokens={n}",
            f"finish_reason={fr}, completion_tokens={usage.get('completion_tokens')}, {len(text)} chars")


def t_stop():  # generation ends BEFORE the first stop string; the string itself is not returned
    text, fr, _ = chat("Count from 1 to 10, separated by commas. Output only the numbers.",
                       temperature=0, max_tokens=50, stop=["5"])
    row("stop", 'stop=["5"]', f"finish_reason={fr}, output={text!r}")


def t_seed():  # same seed + same params should give the same sample, IF the server honours seed
    a, b, c = (chat(NAME_PROMPT, temperature=1.0, max_tokens=20, seed=s)[0] for s in (42, 42, 7))
    row("seed", "T=1.0: seed 42, 42, 7", f"42==42: {a == b}, 42==7: {a == c} ({short([a, c], 20)})")


def t_stream():
    prompt, params = "Write a four-line poem about Chennai rain.", dict(temperature=0.7, max_tokens=120)
    try:
        text, ttft, tps, n, how, total = stream(prompt, **params)
    except RuntimeError:  # some servers reject stream_options: retry and count chunks instead
        text, ttft, tps, n, how, total = stream(prompt, include_usage=False, **params)
    row("stream", "T=0.7, max_tokens=120",
        f"TTFT {ttft * 1000:.0f} ms, decode {tps:.1f} tok/s, {n} tokens ({how}), total {total:.2f}s")


if __name__ == "__main__":
    print(f"endpoint {BASE_URL} | model {MODEL}\n")
    for name, fn in [("temperature", t_temperature), ("top_p", t_top_p), ("max_tokens", t_max_tokens),
                     ("stop", t_stop), ("seed", t_seed), ("stream", t_stream)]:
        run(name, fn)
    w1, w2 = max(len(r[0]) for r in ROWS), max(len(r[1]) for r in ROWS)
    print(f"\n{'test':<{w1}}  {'setting':<{w2}}  result\n{'-' * w1}  {'-' * w2}  {'-' * 40}")
    for test, setting, result in ROWS:
        print(f"{test:<{w1}}  {setting:<{w2}}  {result}")
    print("\nNotes: seed is best-effort (some servers ignore it; batching and GPU kernels can still "
          "change outputs).\nEven T=0 is not guaranteed bit-identical on every hosted API, for the same reason.")
