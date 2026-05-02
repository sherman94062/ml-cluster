"""
Baseline MLX inference benchmark on M3 Pro 36GB.
Records tokens/sec for the Phase 1 sanity-check model.
Numbers from this script become the comparison baseline
for Phase 2 Syntropic compression work.
"""
import time
import json
from pathlib import Path
from datetime import datetime

from mlx_lm import load, generate

MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
PROMPT = "Explain the concept of KV cache compression in transformer models."
MAX_TOKENS = 256
NUM_RUNS = 3


def run_benchmark():
    print(f"Loading {MODEL}...")
    model, tokenizer = load(MODEL)

    results = []
    for i in range(NUM_RUNS):
        print(f"\nRun {i + 1}/{NUM_RUNS}")
        start = time.time()
        response = generate(
            model,
            tokenizer,
            prompt=PROMPT,
            max_tokens=MAX_TOKENS,
            verbose=False,
        )
        elapsed = time.time() - start
        token_count = len(tokenizer.encode(response))
        tps = token_count / elapsed
        print(f"  Generated {token_count} tokens in {elapsed:.2f}s = {tps:.2f} tok/s")
        results.append({"run": i + 1, "tokens": token_count, "seconds": elapsed, "tokens_per_second": tps})

    avg_tps = sum(r["tokens_per_second"] for r in results) / len(results)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "hardware": "M3 Pro 36GB",
        "max_tokens": MAX_TOKENS,
        "runs": results,
        "average_tokens_per_second": avg_tps,
    }

    out_path = Path("benchmarks/baseline_results.json")
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Average: {avg_tps:.2f} tokens/sec ===")
    print(f"Results saved to {out_path}")
    return summary


if __name__ == "__main__":
    run_benchmark()
