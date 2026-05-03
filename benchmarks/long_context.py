"""
Long-context memory-vs-context benchmark on M3 Pro 36GB.

For each target prompt length, runs prefill + a fixed decode budget and
records prefill tok/s, decode tok/s, and peak GPU memory. Produces the
memory-scaling curve that KV-cache compression (KVTC, Syntropic) targets.

Usage:
  python benchmarks/long_context.py
  python benchmarks/long_context.py --model mlx-community/Qwen2.5-7B-Instruct-4bit
  python benchmarks/long_context.py --contexts 1024,4096,16384,32768
"""
import argparse
import time
import json
from pathlib import Path
from datetime import datetime

import mlx.core as mx
from mlx_lm import load, stream_generate

DEFAULT_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
DEFAULT_CONTEXTS = [1024, 4096, 8192, 16384, 32768]
DECODE_TOKENS = 128


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


def slug(model_name):
    return model_name.split("/")[-1].lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--contexts", default=",".join(str(c) for c in DEFAULT_CONTEXTS))
    ap.add_argument("--decode-tokens", type=int, default=DECODE_TOKENS)
    args = ap.parse_args()

    contexts = [int(c) for c in args.contexts.split(",")]

    print(f"Loading {args.model}...")
    model, tokenizer = load(args.model)

    print("Warm-up...")
    run_one(model, tokenizer, "Hello, world.", 16)

    results = []
    for ctx in contexts:
        print(f"\nContext target: {ctx} tokens")
        prompt = build_prompt(tokenizer, ctx)
        try:
            r = run_one(model, tokenizer, prompt, args.decode_tokens)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            results.append({"context_target": ctx, "error": f"{type(e).__name__}: {e}"})
            continue
        r["context_target"] = ctx
        results.append(r)
        print(f"  prefill: {r['prompt_tokens']:>6} tok @ {r['prompt_tps']:8.1f} tok/s")
        print(f"  decode:  {r['generation_tokens']:>6} tok @ {r['generation_tps']:8.1f} tok/s")
        print(f"  peak memory: {r['peak_memory_gb']:.2f} GB")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "hardware": "M3 Pro 36GB",
        "decode_tokens_per_run": args.decode_tokens,
        "results": results,
    }
    out_path = Path(f"benchmarks/long_context_{slug(args.model)}.json")
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
