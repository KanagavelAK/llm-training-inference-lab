"""
mock_openai.py - a fake OpenAI-compatible chat server, for testing params_lab.py offline.

It checks the PLUMBING only: request/response JSON shapes, streaming (SSE), finish_reason, usage,
stop sequences, seed handling. The "model" samples words from a fixed toy distribution using the
temperature/top_p/seed you send, so parameter effects are visible, but it is not a language model.

  python mock_openai.py --port 8001
  BASE_URL=http://127.0.0.1:8001/v1 API_KEY=x MODEL=mock-1 python params_lab.py
"""
import argparse
import asyncio
import json
import math
import random
import time
import uuid

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="mock OpenAI-compatible server")
WORDS = ("Copper Bean Lantern Harbour Morning Velvet Orbit Cedar Ember Maple Quiet River Atlas North "
         "Studio Grind Roast Corner").split()
NATURAL_LEN = 12            # the toy model 'decides' to stop after this many tokens
FIRST_TOKEN_DELAY = 0.15    # pretend prefill time (seconds) for streaming
PER_TOKEN_DELAY = 0.02      # pretend decode time per token -> ~50 tok/s


def pick(logits, temperature, top_p, rng):
    """Same recipe as generate.py: temperature-scaled softmax, nucleus cut, then draw."""
    if temperature == 0:
        return max(range(len(logits)), key=logits.__getitem__)
    m = max(logits)
    weights = [math.exp((x - m) / temperature) for x in logits]
    ranked = sorted(((w / sum(weights), i) for i, w in enumerate(weights)), reverse=True)
    kept, mass = [], 0.0
    for p, i in ranked:
        kept.append((p, i))
        mass += p
        if mass >= top_p:
            break
    r = rng.random() * mass
    for p, i in kept:
        r -= p
        if r <= 0:
            return i
    return kept[-1][1]


def toy_completion(body, temperature, top_p):
    """Returns (tokens, finish_reason). Each token is a word followed by a space."""
    prompt = " ".join(str(m.get("content", "")) for m in body["messages"]).lower()
    max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or 256
    if "1 to 10" in prompt:   # canned answer so the stop-sequence test has something to cut
        tokens = [f"{i}, " for i in range(1, 10)] + ["10"]
    else:
        seed = body.get("seed")
        rng = random.Random(seed) if seed is not None else random.Random()
        V = len(WORDS)
        tokens = []
        for pos in range(NATURAL_LEN):  # fixed scores: a different favourite word at each position
            logits = [-0.6 * ((i - 7 * pos) % V) for i in range(V)]
            tokens.append(WORDS[pick(logits, temperature, top_p, rng)] + " ")
    finish = "stop"
    if len(tokens) > max_tokens:
        tokens, finish = tokens[:max_tokens], "length"
    stops = body.get("stop") or []
    stops = [stops] if isinstance(stops, str) else stops
    text = "".join(tokens)
    cut = min((text.find(s) for s in stops if s in text), default=-1)
    if cut >= 0:  # OpenAI semantics: output ends BEFORE the stop string, which is not included
        kept, n = [], 0
        for t in tokens:
            if n + len(t) > cut:
                if cut > n:
                    kept.append(t[:cut - n])
                break
            kept.append(t)
            n += len(t)
        tokens, finish = kept, "stop"
    return tokens, finish


def echo(body):
    return {k: body.get(k) for k in ("temperature", "top_p", "max_tokens", "stop", "seed", "stream")}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "mock-1", "object": "model", "owned_by": "mock"}]}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    t = 1.0 if body.get("temperature") is None else body["temperature"]  # OpenAI defaults: 1.0, 1.0
    p = 1.0 if body.get("top_p") is None else body["top_p"]
    if not (0 <= t <= 2) or not (0 < p <= 1) or not body.get("messages"):
        return JSONResponse({"error": {"message": "temperature must be in [0,2], top_p in (0,1], "
                                       "messages non-empty", "type": "invalid_request_error"}}, 400)
    tokens, finish = toy_completion(body, t, p)
    usage = {"prompt_tokens": sum(len(str(m.get("content", "")).split()) for m in body["messages"]),
             "completion_tokens": len(tokens)}
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    rid, created, model = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time()), body.get("model", "mock-1")
    if not body.get("stream"):
        return {"id": rid, "object": "chat.completion", "created": created, "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(tokens).strip()},
                             "finish_reason": finish}],
                "usage": usage, "mock_echo": echo(body)}

    def sse(choices, **extra):
        chunk = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
                 "choices": choices, **extra}
        return f"data: {json.dumps(chunk)}\n\n"

    async def events():
        yield sse([{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}])
        await asyncio.sleep(FIRST_TOKEN_DELAY)
        for i, tok in enumerate(tokens):
            if i:
                await asyncio.sleep(PER_TOKEN_DELAY)
            yield sse([{"index": 0, "delta": {"content": tok}, "finish_reason": None}])
        yield sse([{"index": 0, "delta": {}, "finish_reason": finish}], mock_echo=echo(body))
        if (body.get("stream_options") or {}).get("include_usage"):
            yield sse([], usage=usage)
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
