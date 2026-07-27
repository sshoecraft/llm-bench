---
name: fp8-dead-end-do-not-suggest
description: DO NOT suggest FP8 anything on this 4x3090 box: no FP8 tensor cores (SM 8.6), kv-cache-dtype fp8/fp8_e5m2 tried repeatedly and errors, wouldn't help…
metadata:
  type: project
tags: [fp8, dead-end, do-not-retry, ampere, 3090, kv-cache]
---

# FP8 is a permanent DEAD END on this hardware — do not suggest it

4x RTX 3090 = GA102, SM 8.6 (Ampere). **No FP8 tensor cores** (those start at Ada
8.9 / Hopper 9.0). There is no FP8 compute path on this box, period.

## `--kv-cache-dtype fp8` / `fp8_e5m2`: tried repeatedly, always errors

- `fp8` (=e4m3): Ampere attention kernels don't support the e4m3 path → errors.
- `fp8_e5m2`: hits `assert kv_cache_dtype in {"fp8","fp8_e4m3","nvfp4"}` at
  attention.py:467 → AssertionError. The `vllm-fp8-e5m2-ampere.patch` in ~/models
  that narrows attention.py:422 to skip query-quant for e5m2 is NOT currently
  applied (wiped by a vllm upgrade — same landmine class as the reasoning parser).
- User has tried this MULTIPLE times across sessions. It does not work. Confirmed
  2026-07-24.

## Even if it worked, it wouldn't help

Decode on gemma-4-26B-A4B-it is bound by streaming expert weights, not KV bandwidth
(25/30 layers are sliding-window 1024, KV is already tiny). FP8 KV would trade
accuracy for ~nothing.

## The ONLY quantization that would buy anything on Ampere

int4 weights (Marlin GPTQ/AWQ). Google ships a QAT int4 checkpoint for this model.
That's a quality trade + separate download, not a quick flag. Mention only if the
user explicitly asks about quantization.

**Rule: do not raise FP8 as a lever for this hardware again.**
