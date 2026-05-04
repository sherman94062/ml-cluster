# Eval harness (Phase 2)

Pluggable accuracy benchmarks for MLX models, designed to host the
head-to-head comparison against KVTC's published numbers (see CLAUDE.md
§ Phase 2).

## What's here

- `harness.py` — CLI runner. Loads an MLX model, runs a task at a target
  context length, writes a JSON result keyed by
  `(model, task, context_length, compression_config)`.
- `tasks.py` — task registry. One task implemented today:
  - `niah` — Needle-in-a-Haystack (RULER-style single-needle retrieval).
- `results/` — JSON outputs (gitignored).

## Run

```bash
cd ~/ml-cluster && source mlx-env/bin/activate

python benchmarks/eval/harness.py \
    --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
    --task niah \
    --context 4096 \
    --samples 10
```

Output JSON lands in `benchmarks/eval/results/`.

## Adding a task

Implement a `Task` subclass in `tasks.py`:

```python
class MyTask(Task):
    name = "my_task"
    def generate_samples(self, tokenizer, context_length, n_samples, seed): ...
    def evaluate(self, sample, model_output): ...
```

Register it in `TASKS` at the bottom of `tasks.py`. The harness picks it
up automatically.

## What's missing (deferred)

- **RULER full suite** — currently only the single-needle variant.
  Multi-needle, multi-key, multi-value, common-words, frequent-words,
  etc. need adapters.
- **LongBench** — requires HuggingFace dataset downloads
  (`datasets.load_dataset("THUDM/LongBench", ...)`); deferred until the
  external SSD is mounted (don't pile multi-GB datasets on internal).
- **Compression hook** — `--compression` only accepts `none` today. When
  Syntropic primitives port to MLX, plug them into `harness.run_eval`
  via an injectable cache wrapper.

## Why this design

Single source of truth for results format. Every JSON has the same keys
regardless of task or model, so a future `aggregate.py` can build the
M2 head-to-head table by globbing `results/*.json` without
task-specific parsing.
