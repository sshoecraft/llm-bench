---
name: diffusiongemma-vllm-nightly-tp-patch
description: DiffusionGemma needs vllm nightly (>=0.22.1rc1.dev466) + local TP fix patched into venv's diffusion_gemma.py — any vllm upgrade wipes the patch
metadata:
  type: project
tags: [diffusiongemma, vllm, launch-script, parallel-serving, benchmarking]
---

# DiffusionGemma on vLLM: nightly + local TP patch (2026-06-12)

`google/diffusiongemma-26B-A4B-it` is NOT supported by any stable vLLM release as of 2026-06-12. Support merged to main 2026-06-10 (PR #45163); installed via nightly wheel:

```
pip install --pre --extra-index-url https://wheels.vllm.ai/nightly 'vllm==0.22.1rc1.dev466+gb7f9b6ab2'
```

## Local patch (LOST ON ANY vllm UPGRADE/REINSTALL)

Upstream's self-conditioning matmul is broken under tensor parallelism (only tested single-GPU): it multiplies full-vocab `probs` [.., 262144] by the rank-local `VocabParallelEmbedding` shard [65536, 2816].

Patched `~/venvs/vllm/lib/python3.12/site-packages/vllm/model_executor/models/diffusion_gemma.py` in two places (`_compiled_sample_step` ~line 630 and `compute_self_conditioning` ~line 270): each rank slices its vocab range from probs, matmuls its local shard, then `tensor_model_parallel_all_reduce`. Backup at `diffusion_gemma.py.backup` alongside. Verified working with TP=4 and coherent output. Should be reported/PR'd upstream; check whether upstream fixed it before re-applying after an upgrade.

Verified still intact 2026-07-16: vllm 0.22.1rc1.dev466+gb7f9b6ab2, all_reduce at lines 286 and 653.

## Launch script constraints (~/models/tests/diffusiongemma-26B-A4B-it)

- NO `--speculative-config`: diffusion reuses the spec-decode data path for its 256-token canvas (num_speculative_tokens=255); a draft model collides with it.
- NO custom `cudagraph_capture_sizes` list — CONFIRMED EMPIRICALLY 2026-07-16: `[1,2,4,8]` fails fast in `initialize_kv_cache` → `ValueError: No valid cudagraph sizes after rounding to multiple of 256`. All TP workers die before warmup. Let vLLM derive the sizes. (`max_cudagraph_capture_size` is the sanctioned knob per the error message itself — see Parallel serving below.)
- `--max-num-seqs 4` and `--gpu-memory-utilization 0.80` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: the compiled sampler materializes [num_seqs, 256, vocab] fp32 transients the profiler can't see — higher values OOM the 24 GiB cards during warmup (prior finding; the 2026-07-16 0.97/8-seq run crashed on cudagraphs before reaching warmup, so not re-tested).
- `--max-num-batched-tokens` IS NOT FREE at 0.80: the profiling peak scales with it (logits/activation transients). At 32768 → only 2.53 GiB KV, auto-fit CUT max_model_len to 96496. At 8192 → KV pool 498,480 tokens (36,641 blocks × 16), full 262144 context, kv_cache_max_concurrency=1.90. Keep 8192 (diffusion decode only needs max_num_seqs*256=1024).
- Sampler params come from the checkpoint's generation_config.json; `--diffusion-config`/`--hf-overrides` not needed.
- `--language-model-only`: valid, skips the always-present Gemma4 vision tower, frees VRAM; server then rejects image inputs (fine for text benchmarks). No measurable effect on decode speed.

## Parallel serving measurements (2026-07-16, 4×24 GiB, TP=4, config above)

- 4 concurrent connections verified working (max-num-seqs=4 slots; 5th request queues, not rejected). KV pool 498k tokens ≈ 124k/stream average; one stream can still take the full 262k.
- Warm single-stream: ~230–360 t/s depending on content. Cold first request ~30 t/s — the compiled sampler JITs per batch-size shape on FIRST use, so after any restart the first request at each new concurrency level stalls ~15–25 s.
- Warm 4-way: ~26 t/s per stream, ~99 t/s AGGREGATE — LESS than one stream alone. Cause hypothesis: derived cudagraph sizes cap at max_cudagraph_capture_size=512, so 1–2-seq canvas steps (256/512 tokens) run full-cudagraph but 3–4-seq steps (768/1024) fall to piecewise → kernel-launch overhead dominates the small-active-params MoE.
- UNTESTED candidate fix for concurrent throughput: `--compilation-config '{"max_cudagraph_capture_size":1024}'` (raise the max, still let vLLM derive sizes — NOT a custom size list). Risk: larger graph pools spend the same unprofiled headroom the warmup-OOMs live in; test before trusting.

## Benchmark variance — do NOT compare 5-run means (2026-07-16)

Diffusion t/s is content-dependent (tokens accepted per 256-token canvas step vary with text predictability), so openai_bench/api_bench 5-run means swing ±10%. Measured same server, same minute: means 296.2 / 302.5 / 318.8; individual runs 252–363 t/s. A ~6% mean delta between two invocations (e.g. 321.3 on 6/12 vs 302.5 on 7/16) is NOISE, not regression — the 7/16 config matched 6/12 speed (best single run 362.7 beat 6/12's 357.6). For cross-date/model comparisons at ±5%, run ≥20 iterations.
