"""
MLX eval harness for Phase 2.

Loads an MLX model, runs a registered task at a target context length on
N samples, scores each sample, writes a JSON result keyed by
(model, task, context_length, compression_config).

Usage:
  python benchmarks/eval/harness.py \\
      --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \\
      --task niah \\
      --context 4096 \\
      --samples 5

Compression hook: --compression none is the only option today. When
Syntropic primitives land, register them here and pipe through to the
generation call.
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, generate

from tasks import TASKS


def slug(s: str) -> str:
    return s.replace("/", "_").replace("-", "_").lower()


def format_prompt(tokenizer, user_prompt: str) -> str:
    """Apply the model's chat template if available, else return raw."""
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    return user_prompt


def run_eval(model_name, task_name, context_length, n_samples, seed, max_new_tokens, compression):
    if compression != "none":
        raise NotImplementedError(f"compression={compression!r} not implemented yet (Phase 2 work)")
    if task_name not in TASKS:
        raise ValueError(f"unknown task {task_name!r}; available: {list(TASKS)}")

    task = TASKS[task_name]

    print(f"Loading {model_name}...")
    t0 = time.time()
    model, tokenizer = load(model_name)
    load_seconds = time.time() - t0

    print(f"Generating {n_samples} samples at context~{context_length}...")
    samples = task.generate_samples(tokenizer, context_length, n_samples, seed)

    results = []
    for i, sample in enumerate(samples, 1):
        prompt = format_prompt(tokenizer, sample.prompt)
        prompt_tokens = len(tokenizer.encode(prompt))

        mx.clear_cache()
        mx.reset_peak_memory()
        t_start = time.time()
        output = generate(model, tokenizer, prompt=prompt, max_tokens=max_new_tokens, verbose=False)
        elapsed = time.time() - t_start

        eval_fields = task.evaluate(sample, output)
        results.append({
            "sample_id": sample.sample_id,
            "prompt_tokens": prompt_tokens,
            "elapsed_seconds": round(elapsed, 3),
            "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 3),
            **sample.metadata,
            **eval_fields,
        })
        mark = "✓" if eval_fields["correct"] else "✗"
        print(f"  [{i}/{n_samples}] {sample.sample_id} {mark} expected={sample.expected} prompt_tok={prompt_tokens}")

    correct = sum(r["correct"] for r in results)
    accuracy = correct / len(results) if results else 0.0

    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "task": task_name,
        "context_target": context_length,
        "n_samples": n_samples,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "compression_config": compression,
        "load_seconds": round(load_seconds, 2),
        "hardware": "M3 Pro 36GB",
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": len(results),
        "results": results,
    }

    out_dir = Path("benchmarks/eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slug(model_name)}__{task_name}__ctx{context_length}__{compression}.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAccuracy: {correct}/{len(results)} = {accuracy:.1%}")
    print(f"Saved to {out_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="niah", choices=list(TASKS))
    ap.add_argument("--context", type=int, required=True)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--compression", default="none")
    args = ap.parse_args()
    run_eval(args.model, args.task, args.context, args.samples, args.seed, args.max_new_tokens, args.compression)


if __name__ == "__main__":
    main()
