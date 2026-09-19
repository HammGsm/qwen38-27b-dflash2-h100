#!/usr/bin/env python3
"""Single-stream greedy decode benchmark for an OpenAI-compatible vLLM server.

Reports tok/s derived from the *median inter-chunk gap* scaled by tokens-per-step,
which is far more stable than end-to-end time when a draft model is in play.

    python3 bench.py http://127.0.0.1:8000/v1/chat/completions 5

Env: MODEL (default Qwen3.8-27B-VL-MTP-Tools), MAX_TOKENS (default 1024),
     API_KEY (optional; sent as `Authorization: Bearer ...`).
Discards 2 warmup calls first: the first requests after startup are wrong until
CUDA graphs are captured.
"""
import json, os, statistics, sys, time, urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/v1/chat/completions"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 5
MODEL = os.environ.get("MODEL", "Qwen3.8-27B-VL-MTP-Tools")
MAXTOK = int(os.environ.get("MAX_TOKENS", "1024"))
PROMPT = "Write a detailed technical manual about Slurm cluster administration."


HEADERS = {"Content-Type": "application/json"}
if os.environ.get("API_KEY"):
    HEADERS["Authorization"] = "Bearer " + os.environ["API_KEY"]


def post(body):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers=HEADERS)
    return urllib.request.urlopen(req, timeout=900)


for _ in range(2):  # warmup
    with post({"model": MODEL, "messages": [{"role": "user", "content": "say ok"}],
               "max_tokens": 256, "temperature": 0.0}) as f:
        f.read()

results = []
for i in range(N):
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": MAXTOK, "temperature": 0.0,
            "stream": True, "stream_options": {"include_usage": True}}
    t0, chunks, usage, gaps, last = time.time(), 0, None, [], None
    with post(body) as f:
        for line in f:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            try:
                j = json.loads(data)
            except ValueError:
                continue
            if j.get("usage"):
                usage = j["usage"]
            ch = j.get("choices") or []
            delta = (ch[0].get("delta") or {}) if ch else {}
            if delta.get("content") or delta.get("reasoning"):
                now = time.time()
                if last:
                    gaps.append((now - last) * 1000)
                last = now
                chunks += 1
    elapsed = time.time() - t0
    ct = usage["completion_tokens"]
    med = statistics.median(gaps)
    tps = 1000 / med * (ct / chunks)
    results.append(tps)
    print(f"  run{i+1}: gen={ct} steps={chunks} tok/step={ct/chunks:.3f} "
          f"step={med:.2f}ms REAL={tps:.1f} t/s e2e={ct/elapsed:.1f}", flush=True)

print("  MEDIAN %.1f t/s  min %.1f  max %.1f  (n=%d)"
      % (statistics.median(results), min(results), max(results), N), flush=True)
