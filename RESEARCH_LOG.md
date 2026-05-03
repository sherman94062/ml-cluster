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
