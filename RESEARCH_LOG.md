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
