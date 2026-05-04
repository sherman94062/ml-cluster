# Personal AI Cluster — Research Log

## Phase 1: MLX Foundation

### 2026-05-02 — Environment established

**Hardware:** 16" MacBook Pro, M3 Pro (12-core CPU 6P+6E, 18-core GPU), 36 GB unified, 150 GB/s mem bw, macOS Tahoe 26.3.1
**Python:** 3.12.11 (managed by uv, isolated in `mlx-env/`)
**MLX version:** 0.31.2 (mlx-metal 0.31.2)
**mlx-lm version:** 0.31.3

**GPU verification:** `mx.default_device()` → `Device(gpu, 0)` ✓

**Baseline benchmark — Qwen 2.5 1.5B-Instruct (4-bit):**
- Runs: 3 × 256-token generations, prompt = "Explain the concept of KV cache compression in transformer models."
- Run 1: 107.04 tok/s
- Run 2: 108.94 tok/s
- Run 3: 106.84 tok/s
- **Average: 107.61 tokens/sec**
- Peak memory (from initial sanity-check generate): 0.98 GB
- Notes: very tight run-to-run variance (~2%), indicating thermals + scheduler are stable. Model weights only ~1 GB on disk; well within unified-memory headroom.

**Next:** Drive arrives → upgrade 14" to Tahoe → set up Thunderbolt Bridge → Phase 2 (Syntropic MLX port).

---

### 2026-05-02 — Strategic context: KVTC reframe

Source: Mike's "Syntropic positioning — post-KVTC" memo to Josef, May 2, 2026 (`~/Downloads/syntropic positioning post kvtc.docx`).

**What changed externally.** NVIDIA Research published KVTC (KV Cache Transform Coding) in November 2025; v2 in March 2026. PCA decorrelation + DP-allocated adaptive quantization + entropy coding, calibration-only (~10 min), weights untouched. **20× compression (40× in specific cases), <1-point accuracy delta on MMLU/RULER/AIME25/LiveCodeBench/LongBench, up to 8× TTFT on long-context cache hits.** Validated on Llama 3, Mistral NeMo, R1-Qwen 2.5. Architecturally adjacent to QuaRot/SpinQuant — closest in spirit to Syntropic's "Core" primitive.

**What it means for this cluster's work:**

1. **Phase 2 model is now constrained, not free.** M2 deliverable per Mike's memo requires "head-to-head accuracy-and-ratio table on at least one model family KVTC tested (Llama 3 or Mistral NeMo) on RULER / LongBench." → **Phase 2 lock: Llama 3.1 8B-Instruct (4-bit MLX)** as primary, R1-Qwen 2.5 7B as fallback. Gemma 4 explicitly out — KVTC didn't test it, so any win wouldn't land in the comparison frame.

2. **Phase 1 baseline remains valid and gains relevance.** Qwen 2.5 family is on KVTC's tested list; the 107.61 tok/s number is on the same model family we'll publish against, not a side-quest.

3. **Phase 2 first deliverable is the eval harness, not the compression port.** RULER + LongBench against an *uncompressed* Llama 3.1 8B baseline on MLX must exist before Syntropic code is touched. Without it, the port has nothing to prove against.

4. **Apple Silicon as a strategic non-CUDA platform.** Mike's memo names AMD MI300X / Intel Gaudi / Trainium / Groq / Cerebras as the non-CUDA buyer segment KVTC structurally cannot reach (it ships in TensorRT-LLM). Apple Silicon is in the same bucket but isn't named. Open question for Mike: is that an oversight or scoping decision? Either way, this cluster's MLX work is no longer pure research — it's potentially the demo platform for the non-CUDA story, which raises the bar (customer-runnable, not just research artifact).

5. **Layer 4 (multimodal/audio) becomes the post-M2 priority** per Mike's recommendation — uncontested by KVTC, multiplicative-ratio story is most credible there. Cluster Phase 3 scoping should anticipate audio encoder + LLM pipelines (Whisper-large-v3 → compressed-cache LLM), longer effective contexts, more memory pressure.

**Decision log:**
- Phase 2 model locked: Llama 3.1 8B-Instruct (4-bit MLX). Fallback: R1-Qwen 2.5 7B.
- Gemma 4 dropped from consideration.
- Phase 2 work order: eval harness (RULER + LongBench on uncompressed baseline) → Syntropic port → head-to-head table.

**Next:** Wait for external SSD before pulling Llama 3.1 8B. Begin Phase 2 step 1 (eval harness scaffolding) — this can start without the SSD since harness code is independent of model weights.

---

### 2026-05-03 — Long-context memory & throughput curve (Qwen 2.5 1.5B-Instruct 4-bit)

First memory-vs-context data point on M3 Pro. Same model as the Phase 1 baseline; same hardware. `benchmarks/long_context.py` sweeps prompt length, runs 128 decode tokens per sample, records prefill tok/s, decode tok/s, and peak GPU memory.

| Context (tok) | Prefill (tok/s) | Decode (tok/s) | Peak GPU mem (GB) |
|---:|---:|---:|---:|
| 1,027 | 1,721.8 | 110.5 | 1.61 |
| 4,077 | 1,685.7 | 99.4 | 1.72 |
| 8,067 | 1,567.2 | 90.2 | 1.82 |
| 16,089 | 1,290.5 | 74.0 | 2.06 |

**Observations:**
- Decode throughput degrades **~33%** from 1K to 16K context (110.5 → 74.0 tok/s) — pure memory-bandwidth pressure as the KV cache grows.
- Peak memory grows **~0.45 GB** from 1K to 16K. Back-of-envelope: 28 layers × 2 (K+V) × 2 KV heads (GQA) × 128 head_dim × 2 bytes (bf16) = **~28 KB per token**. 15K extra tokens × 28 KB ≈ 430 MB — matches observed delta.
- At 16K context, per-decode-step read traffic ≈ 1 GB (model weights) + 459 MB (KV cache) = **~1.46 GB/token**. At 74 tok/s, that's **~108 GB/s ≈ 72% of M3 Pro's 150 GB/s theoretical bandwidth.** Healthy utilization — confirms decode is bandwidth-bound, not compute-bound, exactly the regime KV compression targets.
- Prefill degrades only ~25% over the same range — prefill is more compute-bound than decode, less affected by cache size.

**Implications for Phase 2:**
- This 1.5B-model curve is the *floor* of compression interest — KV cache here only reaches ~460 MB at 16K. The Llama 3.1 8B Phase 2 target will have ~4× the per-token cache footprint (32 layers, larger head_dim) and stretch to 128K context, where uncompressed cache approaches 16-20 GB. That's where compression actually matters.
- The 72% bandwidth utilization figure is the headroom upper bound for any compression scheme: we cannot get more than (1 / 0.72) ≈ 1.4× decode speedup from removing the cache entirely on this hardware. The real wins are (a) **fitting larger contexts in memory** and (b) **batched/multi-request scenarios** where cache dominates total memory.
- M3 Pro's 150 GB/s bandwidth is the structural ceiling. Any ratio comparison against KVTC's H100-published numbers (3 TB/s) needs that context — same compression ratio, very different absolute throughput.

---

### 2026-05-03 — Model-size sweep (Qwen 2.5 1.5B vs 7B, both 4-bit)

Pulled `mlx-community/Qwen2.5-7B-Instruct-4bit` (~4.4 GB to internal cache — temporary; will move to external SSD when it arrives). Re-ran the long-context sweep on both sizes for a direct architecture comparison. Same harness (`benchmarks/long_context.py`), same prompt seed, same 128-token decode budget.

**Side-by-side: decode tok/s and peak GPU memory**

| Context | 1.5B tok/s | 1.5B mem | 7B tok/s | 7B mem |
|---:|---:|---:|---:|---:|
| 1K | 108.2 | 1.61 GB | 27.7 | 5.12 GB |
| 4K | 100.9 | 1.72 GB | 26.7 | 5.24 GB |
| 8K | 90.0 | 1.82 GB | 25.3 | 5.46 GB |
| 16K | 74.0 | 2.06 GB | 22.8 | 5.89 GB |
| 32K | 53.0 | 2.61 GB | 18.8 | 6.74 GB |

**Per-token KV cache cost (architectural prediction vs observed):**

| Model | Layers | KV heads | Head dim | Predicted (bf16) | Observed (1K→32K) |
|---|---:|---:|---:|---:|---:|
| Qwen 2.5 1.5B | 28 | 2 | 128 | 28 KB/tok | ~32 KB/tok |
| Qwen 2.5 7B | 28 | 4 | 128 | 56 KB/tok | ~52 KB/tok |

Observed ≈ predicted within ~10% noise on both. The 7B's per-token KV cost is **2× the 1.5B's**, driven entirely by the doubled KV-head count (GQA grouping ratio 7:1 vs 6:1).

**Bandwidth utilization (per-decode-step traffic ÷ M3 Pro's 150 GB/s ceiling):**

| Model | Context | Per-step read | Decode tok/s | GB/s consumed | % of 150 GB/s |
|---|---:|---:|---:|---:|---:|
| 1.5B | 1K | ~1.03 GB | 108.2 | 111 | 74% |
| 1.5B | 32K | ~2.0 GB | 53.0 | 106 | 71% |
| 7B | 1K | ~4.5 GB | 27.7 | 125 | 83% |
| 7B | 32K | ~6.1 GB | 18.8 | 115 | 77% |

Both models sit at **70–85% of theoretical bandwidth across the entire context range**. The 7B pushes higher because the model weights dominate the per-step read, masking inefficiencies. **Decode is unambiguously bandwidth-bound on this hardware**, regardless of model size or context length.

**Key implications for Phase 2:**

1. **Llama 3.1 8B will look like Qwen 2.5 7B with ~14% more cache.** Llama 3.1 8B has 32 layers (vs Qwen's 28) and 8 KV heads (vs 4) but with smaller head_dim 128 — net per-token KV ≈ 128 KB (vs Qwen 7B's 56 KB). At 32K context that's ~4 GB of cache; at 128K, ~16 GB. The cache becomes the dominant memory cost only past ~16K context — which is exactly where compression starts to matter.
2. **Decode-speedup ceiling on this hardware is small.** At 77–83% bandwidth utilization with the cache present, removing the cache entirely caps speedup at ~1.2–1.3×. The compression value on M3 Pro is **context length and concurrent-request capacity**, not raw decode tok/s.
3. **The KVTC comparison frame stays valid because ratios travel.** A 20× compression ratio means 20× more context fits, or 20× more concurrent prompts share cache budget — those benefits transfer cleanly from H100 to M3 Pro even though absolute tok/s does not.
4. **7B decode at 32K is already painful (18.8 tok/s).** Llama 3.1 8B at 128K uncompressed will likely be in the single-digit tok/s range and consume ~22 GB total. That's the regime where the M3 Pro 36 GB starts hurting and where compression ROI on this hardware becomes obvious.

**Phase 2 prep checklist updates:**
- ✅ Harness reusable across model sizes (verified on 1.5B + 7B)
- ✅ Memory measurement reliable (predicted matches observed)
- ✅ Bandwidth-bound regime confirmed
- ⏳ Need: external SSD mounted before pulling Llama 3.1 8B (4-bit ~5 GB, smaller than Qwen 7B but still belongs on external)
- ⏳ Need: RULER + LongBench eval harness (Phase 2 Step 1, can begin without SSD)

**Next:** wait on SSD for Llama 3.1 8B; meanwhile begin scaffolding `benchmarks/eval_harness.py` for RULER + LongBench.

---

### 2026-05-04 — Eval harness scaffold + NIAH smoke test

Phase 2 Step 1 deliverable: a pluggable accuracy benchmark harness for MLX models. Lives at `benchmarks/eval/` (see `benchmarks/eval/README.md` for usage). One task implemented end-to-end: **Needle-in-a-Haystack (NIAH)**, modeled on RULER's `niah_single_1` — a 5-digit "magic number" inserted at a random depth in a long filler passage; model is asked to retrieve it.

**Architecture:**
- `harness.py` — CLI runner. Loads any MLX model, runs a registered task at a target context length, applies the model's chat template, scores each sample, writes JSON keyed by `(model, task, context_length, compression_config)`.
- `tasks.py` — task registry. Adding a new task = subclass `Task`, register in `TASKS` dict.
- Compression hook present (`--compression none` is the only option today). Plug-point reserved for Syntropic primitives without architectural changes.
- Results gitignored (regenerable from the harness with a fixed seed).

**Smoke test results** (n=5 per cell, seed=42, `max_new_tokens=32`):

| Model | Context (actual) | Accuracy |
|---|---:|---:|
| Qwen 2.5 1.5B-Instruct 4bit | 952 | 5/5 |
| Qwen 2.5 1.5B-Instruct 4bit | 8,120 | 5/5 |
| Qwen 2.5 1.5B-Instruct 4bit | 16,312 | 5/5 |
| Qwen 2.5 1.5B-Instruct 4bit | 32,590 | 5/5 |
| Qwen 2.5 7B-Instruct 4bit | 8,120 | 5/5 |
| Qwen 2.5 7B-Instruct 4bit | 16,312 | 5/5 |

**Why 100% across the board is the right result for a smoke test.** NIAH single-needle retrieval is "easy" for capable instruct models on contexts up to their trained range. The point of this run is not to find weaknesses in the base models — it's to confirm the harness produces clean accuracy numbers end-to-end (chat template applied, sample IDs deterministic by seed, JSON shape consistent across runs). All confirmed.

**What this enables:** when Syntropic compression lands as a `--compression syntropic-core` (or `-predict`) option, re-running the same model/task/context/seed combination will produce a directly comparable accuracy number. Any degradation = the compression cost on this task. Combined with the long-context memory & throughput data from 2026-05-02/03, we'll have **accuracy + ratio + speed in one comparable schema** — exactly the M2 deliverable shape.

**Deferred (explicitly):**
- Full RULER suite (multi-needle, multi-key, multi-value, common-words, frequent-words, etc.). NIAH single-needle is the entry point; expanding the suite is mechanical once a task pattern works end-to-end.
- LongBench. Requires HuggingFace dataset downloads; defer until external SSD is mounted.
- Aggregation script (`aggregate.py`) to glob `results/*.json` into a single comparison table. Add when there are enough cells to merit it.

**Phase 2 status:**
- ✅ Step 1 (eval harness scaffold) — done with one working task
- ✅ Bandwidth-utilization analysis grounding the M2 narrative
- ⏳ Pull Llama 3.1 8B (gated on external SSD)
- ⏳ Expand task suite (NIAH multi-needle, MK, MV — can do without SSD on Qwen models as warm-up)
- ⏳ Scope the Syntropic CUDA reference implementation (Mike-input gated)
