# CLAUDE.md — Personal AI Cluster: MLX Foundation + Syntropic Port

**Phase 1: complete (2026-05-02).** Baseline 107.61 tok/s on Qwen 2.5 1.5B-Instruct (4-bit MLX). Repo: https://github.com/sherman94062/ml-cluster. See `~/ml-cluster/RESEARCH_LOG.md` for the full entry and the KVTC strategic-context update.

**Active phase: Phase 2 — Syntropic MLX port + eval harness.** Plan in [§ Phase 2](#phase-2-syntropic-mlx-port--eval-harness) below.

## Project Context

Mike is building a personal AI cluster across two MacBook Pros to support serious ML research, with a primary focus on porting **Syntropic KV cache compression** from CUDA to Apple's MLX framework. The work doubles as IP for Syntropic (a startup Mike co-founded around KV cache compression).

**Strategic context as of 2026-05-02:** NVIDIA Research published **KVTC (KV Cache Transform Coding)** — Nov 2025, v2 March 2026 — achieving 20× compression (40× in specific cases) with <1-pt accuracy delta on standard benchmarks, validated on Llama 3, Mistral NeMo, R1-Qwen 2.5. KVTC is architecturally adjacent to Syntropic's "Core" primitive and ships in TensorRT-LLM (CUDA-only). This reframes Syntropic's positioning toward (a) Predict (the only Syntropic primitive that beats KVTC head-to-head at 26.5×), (b) **non-CUDA hardware** (Apple Silicon, AMD, Gaudi, Trainium, Groq, Cerebras), and (c) Layer 4 multimodal — segments KVTC structurally cannot serve. See Mike's memo "Syntropic positioning — post-KVTC" for the full reframe.

You are running on the **16" MacBook Pro (the primary ML machine)**. A second 14" MacBook Pro will join later as always-on agent infrastructure.

## Hardware (this machine)

- **Apple M3 Pro** (12-core CPU: 6P + 6E, 18-core GPU)
- **36 GB unified memory** (~24-26 GB usable for models after macOS overhead)
- **150 GB/s memory bandwidth** (M3 Pro spec — note: lower than M2 Pro's 200 GB/s; this is the primary inference bottleneck)
- **494 GB internal SSD**, currently ~133 GB used / 361 GB free
- **macOS Tahoe 26.3.1** (will update to 26.4.1 in a later phase)
- 3x Thunderbolt 4 ports

## Phase 1 Objective (complete — see § Phase 1 Execution Plan below for the reproducible setup, useful for the 14" Mac in Phase 3)

Stand up a working MLX development environment, validate the full inference path, establish baseline performance numbers, and scaffold a project structure ready for Phase 2 (the Syntropic compression port).

**Success criteria — all met:**
1. ✅ MLX 0.31.2 installed, GPU verified (`Device(gpu, 0)`)
2. ✅ Qwen 2.5 1.5B-Instruct (4-bit) loads and generates via `mlx-lm`
3. ✅ Baseline recorded: 107.61 tok/s avg (3 runs × 256 tokens)
4. ✅ `~/ml-cluster` scaffolded, git-initialized, pushed to https://github.com/sherman94062/ml-cluster
5. ✅ `RESEARCH_LOG.md` started with baseline + KVTC strategic-context entry

## What NOT to do in Phase 1

- **Do not download large models (13B+) to internal SSD.** An external Thunderbolt SSD (SanDisk PRO-G40 2TB) is in transit. Large models go there, not on the internal drive.
- **Do not install distributed inference frameworks (exo, distributed-llama) yet.** That's Phase 3, after the second Mac is configured and the Thunderbolt Bridge is up.
- **Do not start the Syntropic port yet.** That's Phase 2, after the foundation is solid and we've established baselines.
- **Do not modify system Python.** Use `uv` for environment management.
- **Do not commit large binaries or model files to git.** The .gitignore should exclude these.

## Phase 1 Execution Plan (historical reference — preserved for setting up the 14" Mac in Phase 3)

### Step 1: Install Homebrew (if not already installed)

Check first:

```bash
which brew
```

If not present, install:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

After install, follow the on-screen instructions to add Homebrew to PATH in `~/.zprofile`. On Apple Silicon, this typically means:

```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

Verify:

```bash
brew --version
```

### Step 2: Install core tooling

```bash
# Python toolchain
brew install python@3.12 uv

# Useful CLI utilities for ML work
brew install wget htop tree jq ripgrep git

# Tailscale (needed in Phase 3, install now since it's free and small)
brew install --cask tailscale
```

Verify:

```bash
python3 --version   # should be 3.12.x or newer
uv --version
git --version
```

### Step 3: Create project structure

```bash
mkdir -p ~/ml-cluster
cd ~/ml-cluster

mkdir -p models datasets checkpoints notebooks src benchmarks logs docs
```

Create `.gitignore`:

```
# Python
__pycache__/
*.pyc
*.pyo
.Python
*.egg-info/

# Virtual environments
.venv/
mlx-env/
venv/
env/

# ML artifacts (these will live on external SSD when it arrives)
models/
datasets/
checkpoints/
logs/
*.safetensors
*.bin
*.gguf

# OS
.DS_Store

# Editor
.vscode/
.idea/
```

Initialize git:

```bash
git init
git add .gitignore
git commit -m "Initial project scaffold for ML cluster"
```

### Step 4: Set up MLX environment

```bash
cd ~/ml-cluster
uv venv mlx-env
source mlx-env/bin/activate

uv pip install mlx mlx-lm
```

Verify MLX is functional and using GPU:

```bash
python -c "import mlx.core as mx; a = mx.array([1.0, 2.0, 3.0]); b = a * 2; mx.eval(b); print(b); print(f'Default device: {mx.default_device()}')"
```

Expected output: array showing `[2, 4, 6]` and `Default device: Device(gpu, 0)`.

### Step 5: Validate inference end-to-end

Download and run a small model to confirm the full stack works. Qwen 2.5 1.5B at 4-bit is ~1 GB — small enough for internal SSD as a sanity check:

```bash
mlx_lm.generate \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --prompt "Explain KV cache compression in two sentences." \
  --max-tokens 200
```

This downloads the model on first run (cached to `~/.cache/huggingface/hub/`). Note the tokens/sec output at the end.

### Step 6: Benchmark script

Create `benchmarks/baseline.py`:

```python
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
```

Run it:

```bash
cd ~/ml-cluster
python benchmarks/baseline.py
```

### Step 7: Initialize the research log

Create `RESEARCH_LOG.md` with the Phase 1 baseline entry. Pull the actual numbers from `benchmarks/baseline_results.json`:

```markdown
# Personal AI Cluster — Research Log

## Phase 1: MLX Foundation

### YYYY-MM-DD — Environment established

**Hardware:** 16" MacBook Pro, M3 Pro, 36 GB, macOS Tahoe 26.3.1
**MLX version:** [run `python -c "import mlx; print(mlx.__version__)"`]
**mlx-lm version:** [run `python -c "import mlx_lm; print(mlx_lm.__version__)"`]

**Baseline benchmark — Qwen 2.5 1.5B-Instruct (4-bit):**
- Runs: 3
- Average tokens/sec: [from baseline_results.json]
- Notes: [any observations — thermals, memory pressure, etc.]

**Next:** Drive arrives → upgrade 14" to Tahoe → set up Thunderbolt Bridge → Phase 2 (Syntropic MLX port).
```

### Step 8: Commit the foundation

```bash
cd ~/ml-cluster
git add benchmarks/baseline.py RESEARCH_LOG.md
git commit -m "Phase 1: MLX foundation, baseline benchmark, research log"
```

## Reporting Back

When done, summarize for Mike:

1. Versions installed (Homebrew, Python, uv, MLX, mlx-lm)
2. Output of the GPU verification (Step 4)
3. Tokens/sec from the benchmark (Step 6) — three runs and average
4. Anything that didn't go cleanly or required deviation from this plan
5. Final state of `~/ml-cluster` directory tree (`tree -L 2 ~/ml-cluster`)

## Phase 2: Syntropic MLX Port + Eval Harness

### Objective

Produce a credible head-to-head **accuracy-and-ratio** table comparing Syntropic's compression primitives (ported to MLX) against KVTC's published numbers on a model family KVTC tested. This is the M2 deliverable per Mike's post-KVTC memo. Speed is secondary — accuracy at ratio is what determines whether Syntropic stays in the conversation.

### Locked decisions

- **Primary model:** `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` (broadest external recognition; primary KVTC test target).
- **Fallback model:** R1-Qwen 2.5 7B (4-bit MLX) if the Llama 3.1 8B port hits MLX-specific blockers.
- **Excluded models:** Gemma 4 (KVTC didn't test it; any win wouldn't land in the comparison frame). Mistral NeMo (sparse MLX-format coverage).
- **Eval harness first, port second.** The uncompressed Llama 3.1 8B baseline on MLX with RULER + LongBench must exist before any Syntropic code is touched. No comparison without a calibrated reference.
- **Model weights live on the external Thunderbolt SSD (when it arrives), not internal.** Llama 3.1 8B at 4-bit is ~5 GB; safe on internal as a stopgap, but external once available.

### Success criteria

1. **Eval harness operational.** RULER (long-context retrieval) and LongBench (multi-task long-context) running on uncompressed Llama 3.1 8B-Instruct-4bit via MLX. Reproducible via `python benchmarks/eval_harness.py --model llama-3.1-8b-4bit --suite ruler`.
2. **Memory + accuracy curves** at context lengths {2K, 8K, 32K, 128K} for the uncompressed baseline. These are the reference points the Syntropic port must beat or match.
3. **Syntropic primitive ported.** At least one of {Core, Predict} runs end-to-end on MLX, generating coherent output, with the KV cache routed through the compression layer.
4. **Head-to-head table published in `RESEARCH_LOG.md`.** Columns: model, context length, compression ratio, accuracy delta vs uncompressed, vs KVTC's published number on the same task.
5. **Decision gate met:** if Syntropic's ported primitive matches or beats KVTC at the same ratio on at least one (model, task) pair, proceed to Layer 4 multimodal scoping. If not, document the gap honestly and consult Mike on whether to iterate or pivot.

### Execution plan

**Step 1 — Eval harness scaffold (can start immediately, no SSD dependency).**

```bash
cd ~/ml-cluster && source mlx-env/bin/activate
uv pip install lm-eval datasets jsonlines
mkdir -p benchmarks/{ruler,longbench,results}
```

Stand up `benchmarks/eval_harness.py` that loads an MLX model, runs a configurable subset of RULER and LongBench tasks, and writes JSON results keyed by (model, task, context_length, compression_config). Compression_config = "uncompressed" for the baseline pass.

**Step 2 — Pull Llama 3.1 8B (after external SSD is mounted).**

```bash
# Once external SSD is mounted at /Volumes/MLDrive (or whatever the mountpoint is):
export HF_HOME=/Volumes/MLDrive/hf-cache
mlx_lm.generate --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit --prompt "test" --max-tokens 8
```

Confirms the model loads from the external drive cleanly. If `HF_HOME` isn't honored end-to-end by mlx-lm, fall back to a manual `huggingface-cli download --local-dir`.

**Step 3 — Uncompressed baseline pass.**

Run RULER + LongBench on the uncompressed Llama 3.1 8B at the four target context lengths. Capture: accuracy per task, peak memory, time-to-first-token, decode tok/s. Commit `benchmarks/results/llama31_8b_uncompressed.json` and a markdown summary appended to `RESEARCH_LOG.md`.

**Step 4 — Syntropic port (the actual research work).**

Out of scope for this plan to specify in detail — depends on what the CUDA reference implementation looks like and which primitive (Core vs Predict) gets ported first. Ground rule: **port one primitive at a time, end-to-end, with the eval harness wired in before the next primitive is started.** No big-bang ports.

**Step 5 — Comparison table.**

Re-run RULER + LongBench with the Syntropic-compressed cache enabled. Append results to the markdown summary alongside the baseline numbers and KVTC's reported numbers from the paper. This is the M2 deliverable artifact.

### What NOT to do in Phase 2

- **Do not start with the compression port.** Eval harness first, always. Mike's memo is explicit that "M2 gets dismissed as 'how does that compare to NVIDIA's paper?'" without the comparison frame.
- **Do not measure tok/s alone.** It's necessary but insufficient. Accuracy-at-ratio is the deliverable; speed is a side metric.
- **Do not stand up distributed inference yet** — that's still Phase 3, after the 14" is configured and Thunderbolt Bridge is up. The 8B model fits on this 16" alone.
- **Do not chase Gemma 4, Mistral NeMo, or any other model not on KVTC's tested list.** Single-model focus until M2 ships.
- **Do not commit model weights to git.** `.gitignore` already excludes safetensors/bin/gguf — verify before each commit.

### Open questions (requires Mike input)

1. **Which Syntropic primitive ports first — Core or Predict?** Predict is the only primitive that beats KVTC head-to-head (26.5× vs 20×), so Predict is the strategically obvious choice. But Core is "architecturally adjacent to KVTC" (per Mike's memo), so it may be the simpler port. Need Mike's read.
2. **Is Apple Silicon explicitly in or out of the non-CUDA hardware story?** Mike's memo names AMD MI300X / Intel Gaudi / Trainium / Groq / Cerebras but not Apple Silicon. If Apple Silicon is in, the cluster work becomes a customer-facing demo platform and the bar is higher. If out, this stays research/IP exploration.
3. **What's the Syntropic CUDA reference implementation's API surface?** Need to see it before scoping the MLX port. Path? Repo?

## Key Reference Info

- **MLX docs:** https://ml-explore.github.io/mlx/
- **mlx-lm repo:** https://github.com/ml-explore/mlx-lm
- **MLX community models on HuggingFace:** https://huggingface.co/mlx-community
- **Project repo:** https://github.com/sherman94062/ml-cluster (local: `~/ml-cluster`)
- **KVTC paper (NVIDIA Research):** referenced in Mike's positioning memo; v1 Nov 2025, v2 March 2026 — get arxiv link from Mike if not already in `~/Downloads/`
- **Strategic positioning memo:** `~/Downloads/syntropic positioning post kvtc.docx` (Mike → Josef, 2026-05-02)

## Style and Tone

Mike prefers direct, honest framing. If something doesn't work or a step seems off given current versions, say so — don't fake success. Concise actionable output over verbose explanation. Build real working systems, not theoretical scaffolding.
