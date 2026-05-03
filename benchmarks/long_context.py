"""
Long-context memory-vs-context benchmark on M3 Pro 36GB.

For each target prompt length, runs prefill + a fixed decode budget and
records prefill tok/s, decode tok/s, and peak GPU memory. Produces the
memory-scaling curve that KV-cache compression (KVTC, Syntropic) targets.

Pairs with benchmarks/baseline.py (short-context throughput baseline).
"""
import time
import json
from pathlib import Path
from datetime import datetime

import mlx.core as mx
from mlx_lm import load, stream_generate

MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
DECODE_TOKENS = 128
CONTEXT_LENGTHS = [1024, 4096, 8192, 16384]


def build_prompt(tokenizer, target_tokens):
    seed = (
        "Transformer language models maintain a key-value cache during "
        "autoregressive decoding. Each new token requires reading the entire "
        "cache, making decode latency memory-bandwidth-bound. KV cache "
        "compression reduces both memory footprint and bandwidth pressure. "
    )
    seed_tokens = tokenizer.encode(seed)
    repeats = (target_tokens // len(seed_tokens)) + 2
    text = seed * repeats
    tokens = tokenizer.encode(text)[:target_tokens]
    return tokenizer.decode(tokens) + "\n\nContinue:"


def run_one(model, tokenizer, prompt, decode_tokens):
    mx.clear_cache()
    mx.reset_peak_memory()
    final = None
    start = time.time()
    for response in stream_generate(model, tokenizer, prompt, max_tokens=decode_tokens):
        final = response
    wall = time.time() - start
    return {
        "prompt_tokens": final.prompt_tokens,
        "prompt_tps": final.prompt_tps,
        "generation_tokens": final.generation_tokens,
        "generation_tps": final.generation_tps,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "wall_seconds": wall,
    }


def main():
    print(f"Loading {MODEL}...")
    model, tokenizer = load(MODEL)

    print("Warm-up...")
    run_one(model, tokenizer, "Hello, world.", 16)

    results = []
    for ctx in CONTEXT_LENGTHS:
        print(f"\nContext target: {ctx} tokens")
        prompt = build_prompt(tokenizer, ctx)
        try:
            r = run_one(model, tokenizer, prompt, DECODE_TOKENS)
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"context_target": ctx, "error": str(e)})
            continue
        r["context_target"] = ctx
        results.append(r)
        print(f"  prefill: {r['prompt_tokens']:>6} tok @ {r['prompt_tps']:8.1f} tok/s")
        print(f"  decode:  {r['generation_tokens']:>6} tok @ {r['generation_tps']:8.1f} tok/s")
        print(f"  peak memory: {r['peak_memory_gb']:.2f} GB")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "hardware": "M3 Pro 36GB",
        "decode_tokens_per_run": DECODE_TOKENS,
        "results": results,
    }
    out_path = Path("benchmarks/long_context_results.json")
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
