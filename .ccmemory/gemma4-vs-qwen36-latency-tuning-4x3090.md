---
name: gemma4-vs-qwen36-latency-tuning-4x3090
description: Measured single-stream latency sweep on 4x3090: gemma-4-26B-A4B-it is already optimal; Qwen3.6 MTP is a 12-21% pessimization on Ampere
metadata:
  type: project
tags: [gemma4, qwen3.6, vllm, speculative-decoding, benchmarking, ampere, mtp]
---

# Single-stream latency tuning, gemma-4-26B-A4B-it vs Qwen3.6-35B-A3B (2026-07-24)

Hardware: dual Xeon E5-2680 v4, 4x RTX 3090 (96 GB), TP=4, vllm 0.22.1rc1.dev466.
All numbers from bare `api_bench.py --runs 12`, taint-checked against vLLM's own
`Running: N reqs` counter (every run verified N=1).

## Result: gemma-4-26B-A4B-it as written is already at the optimum

| config | t/s | StdDev | ITL | accept_len |
|---|---|---|---|---|
| spec=4 | 187.6 | 6.19 | 5.34 | 3.28 |
| **spec=3 (current)** | **184.4** | 3.15 | 5.42 | 2.93 |
| spec=5 | 177.9 | 6.59 | 5.63 | 3.51 |
| spec=2 | 173.9 | 2.18 | 5.75 | 2.45 |
| nospec | 151.7 | 0.59 | 6.59 | - |
| +expert-parallel | 181.1 | 2.64 | 5.52 | 2.90 |

- Spec decoding is worth **+21.6%**. Biggest working lever in the config.
- Peak is spec=3-4; spec=4 vs 3 is NOT significant (t~1.6) and has worse time-to-answer. Leave at 3.
- Accept length rises monotonically (2.45/2.93/3.28/3.51) but t/s peaks at 4 — position 5 only lands 27.6%, no longer paying for its draft pass.
- **`--enable-expert-parallel` LOSES** (-1.8%, t~2.8). At batch=1 TP splits all 8 routed experts evenly; EP scatters them unevenly + adds all-to-all. EP is a large-batch optimization.

## Not tunable on this hardware
- Attention forced to `TRITON_ATTN` — vLLM logs it: head_dim=256 vs global_head_dim=512 rules out FlashAttention.
- P2P is a uniform ~13.3 GB/s on all 6 pairs INCLUDING cross-socket (tinygrad-style consumer-P2P patch on 595.58.03, built from /src/open-gpu-kernel-modules). No TP pairing win exists. Measure P2P with `torch.cuda.set_device(src)` or numbers are garbage.
- FP8 is pointless here: RTX 3090 is SM 8.6, no FP8 tensor cores (Ada 8.9 / Hopper 9.0+). Only int4 (Marlin) would buy bandwidth.

## enable_thinking is the only large lever, and it's a quality trade
- thinking on: ~4411 ms to first ANSWER token, MMLU 94.0% (n=100)
- thinking off: ~111 ms, MMLU 78.0% (n=100) / 82.7% (n=14042, the -wow full_test)
- **~40x latency for ~11 MMLU points.** Settable PER REQUEST via `chat_template_kwargs` — one server serves both. The separate -nothink launch file is redundant.
- `reasoning_effort` is NOT a second dial: `chat_completion/protocol.py:490` collapses it to `enable_thinking = reasoning_effort != "none"`, and the chat template has zero references to it. low/medium/high are byte-identical to thinking-on. If a request sends both, explicit `enable_thinking` wins silently.

## Qwen3.6-35B-A3B: do NOT enable MTP on Ampere
| config | t/s | StdDev |
|---|---|---|
| nospec | 165.6 | 1.85 |
| mtp3 | 146.3 | 7.61 (-11.7%) |
| mtp2 | 130.1 | 2.68 (-21.4%) |

Acceptance was fine (2.33-2.37 accept_len, comparable to Gemma) — the draft step just costs more than it saves. Qwen's MTP block is a full MoE layer (1.57 GiB, 256 experts) run per draft step; Gemma's drafter is a dense 4-layer 840 MB model. This corroborates https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090 ("no variant achieves net speedup on Ampere + A3B MoE") and extends it to vLLM's native MTP path.

Also: MTP costs 0.39 GiB/GPU of KV, which drops max context 262144 -> ~210000.

## Head-to-head, identical measurement
- gemma: 182.9 t/s, 2306 tokens -> **~12.6 s to a complete answer**
- qwen3.6: 165.6 t/s, 3621 tokens -> ~21.9 s

Gemma wins twice over: 10% faster per token AND 36% fewer tokens. MMLU is one question apart at n=100 (94 vs 95) = not a distinction. Published task gaps still favour Qwen on coding/agentic (SWE-Bench 68.2 vs 61.4, TAU2 +13) and Gemma on multimodal/multilingual — unmeasured here.

## Natural generation length
~2316 tokens (measured with a 64k cap). t/s identical at 2048 / 4096 / 65536 caps
(184.4 / 182.9 / 184.5), so caps above ~2400 don't affect throughput. Day-to-day
max_tokens of 32k-65k is non-binding for normal prompts.
