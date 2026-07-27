---
name: vllm-local-patches-and-config-landmines
description: vLLM local patches wiped by upgrades, the max-num-batched-tokens landmine (Qwen KV cliff + gemma OOM), and the api_bench temp-0.7 default worth ~6% n…
metadata:
  type: project
tags: [vllm, patches, landmines, api_bench, qwen3.6, gemma4, kv-cache, oom, benchmarking, speculative-decoding]
---

# vLLM local patches + config landmines (found 2026-07-24, corrected 2026-07-25)

## LOCAL PATCH — wiped by any vllm upgrade/reinstall

`~/venvs/vllm/lib/python3.12/site-packages/vllm/reasoning/gemma4_reasoning_parser.py`
is locally modified (12,463 bytes vs 9,297 in the adjacent `.bak`). Adds a
"spelled-out-marker fallback" to `is_reasoning_end()`: under INT8 the model emits
control markers as ordinary BPE tokens carrying none of the special ids, so the
token-id scan finds nothing, the tool-call phase never engages, and raw
`<|tool_call>call:...` leaks out as content. The patch re-runs the same precedence
over a decoded 64-token tail (`STRING_SCAN_TOKENS = 64`).

`gemma4_utils.py` is byte-identical to its `.bak` — only the parser is patched.

Same hazard class as the diffusion_gemma TP patch. **Check both after any vllm change.**
Verified present 2026-07-25 (12,463 bytes, `STRING_SCAN_TOKENS` at line 61) on vllm
`0.22.1rc1.dev466+gb7f9b6ab2`, installed 2026-06-12 and unchanged since.

Measured: this patch does NOT cost per-token latency. ITL with thinking on is
*lower* than with it off (5.09 vs 5.26 ms) across three configs, so whatever the
per-token `list(input_ids)` copy costs, it's under the noise floor.

Control token ids (referenced by the parser): `<|tool_call>`=48, `<tool_call|>`=49,
`<|channel>`=100, `<channel|>`=101.

## `--max-num-batched-tokens 32768` bites TWICE — startup AND runtime

Two different failures, same root knob. **8192 is the safe value on this box for
every model.** 32768 is not a throughput win worth either failure mode.

### Failure 1 — Qwen3.6 startup KV cliff

`Qwen3.6-35B-A3B` (plain, not -wow) failed at startup: `_check_enough_kv_cache_memory`
raised because a 262144 context needs 2.53 GiB KV but only 2.30 was available.

**The real driver is `--max-num-batched-tokens`, NOT gpu-memory-utilization.** The
memory profiler's peak scales with it — bigger prefill chunks = bigger logit/activation
transients reserved during profiling, straight out of the KV pool. Measured on this box,
same 262144 / util 0.97:
- batched-tokens 32768 -> KV 2.57 GiB (~265k tokens) — on the cliff edge
- batched-tokens  8192 -> KV 5.05 GiB (522,203 tokens) — comfortable

Same finding as the diffusiongemma memory ("32768 -> 2.53 GiB KV, auto-fit cut
max_model_len; 8192 -> 498,480 tokens, full context"). Same model family, same 2.53 GiB.

**Fixed: `--max-num-batched-tokens 32768 -> 8192`.** Full 262144 context, util stays 0.97.

DEAD END (don't repeat): bumping `--gpu-memory-utilization 0.97 -> 0.975` "works" (2.57
GiB, starts) but only scrapes 0.08 GiB over the line — treats the symptom, no margin.

### Failure 2 — gemma-4-26B-A4B-it whole-engine OOM crash at runtime (2 requests!)

The plain `gemma-4-26B-A4B-it` launch script ran fine for a while, then all 4 workers
died simultaneously with `torch.OutOfMemoryError` mid-serve at only `Running: 2 reqs`.

**Not KV pressure — `GPU KV cache usage: 21.2%` in the log line immediately before.**
The failing allocation, in the inductor-compiled gemma4 graph:

```
buf24 = empty_strided_cuda((s59, 2816), (2816, 1), torch.float32)   # 352.00 MiB
```

352 MiB / 4 bytes / 2816 = **32768 exactly** = `--max-num-batched-tokens`. One fp32 MLP
activation for one full-size prefill chunk. GPU had **357.25 MiB free**. Missed by ~5 MiB.

Why the profiler didn't reserve for it: it profiles that chunk *in isolation*. At crash
time a chunked prefill was co-resident with a spec-decode step (drafter activations +
rejection-sampler scratch) and the MM encoder — none of which exist during profiling.
At `--gpu-memory-utilization 0.97` there is only ~0.35 GiB of total slack on a 24 GB
card, so that discrepancy has nowhere to land.

Cascade to be aware of when reading the log: the OOM is followed by
`KeyError: '<request-id>'` at `scheduler.py:1475 update_from_output` ->
`EngineDeadError` -> 500s on `/v1/messages`. That KeyError is the async batch-queue path
losing the dead worker's output. **It is downstream noise, not a second bug.** Don't chase it.

**Fix: `--max-num-batched-tokens 32768 -> 8192`.** Cuts that buffer 4x to 88 MiB and
shrinks the profile-vs-runtime gap by the same factor. Decode only needs
`max_num_seqs * small` batched tokens, so 8192 is plenty.

Corroboration: `gemma-4-26B-A4B-it-wow` already used 8192 + util 0.94 and survived a
64-concurrent-client sweep. The plain config at 32768 + util 0.97 died at **2 requests**.

Second lever if it ever recurs: drop util 0.97 -> 0.95 (~0.47 GiB/GPU of real headroom;
KV would still be ~4.6 GiB, far above the 2.53 GiB that 262144 context needs). Not
applied — the batched-tokens fix is structural, util is the symptom knob.

## api_bench.py sampling default — worth ~6%, NOT ~1% (CORRECTED 2026-07-25)

### What the patch was

`/src/aitest/api_bench.py` sent `temperature=0.7` and `top_p=1.0` **unconditionally**
on the openai and gemini paths, silently overriding the model's
generation_config.json (Gemma 4 ships temperature=1.0 / top_p=0.95 / top_k=64).
`top_k` was already correctly gated, so the server's 64 survived. The anthropic path
sent temperature unconditionally but gated top_p behind `< 1.0`.

Fixed 2026-07-24 16:59: defaults are now `None`, all paths send only what was explicitly
asked for. At 18:17 `--max-tokens` default was raised 2048 -> 4096.

There was a SECOND patch session the same day that this memory previously omitted:
`-src-wowbot` session, 2026-07-24 12:51-13:08, ~20 edits adding `--thinking` /
`--reasoning` request switches, `check_switch_took_effect()`, `reasoning_chars`,
`content_chars`, `finish_reason`, and `ttfct` (time-to-first-CONTENT-token, printed as
"Time to First ANSWER Token"). **That session did NOT change the throughput math** —
`tps = tokens / generation_time`, `ITL = 1000/tps`, and `first_token_time` are
unchanged, and `tokens` is overwritten by the server's `usage.completion_tokens`
anyway. Verified: ITL == 1000/tps holds run-by-run in both June and July .bench files.

### THE CORRECTION — the old "~1%" figure was wrong

This memory previously said the temp-0.7 effect was "only ~1%. Smaller than feared."
**That is wrong and it cost a full session chasing a phantom regression.** Measured
2026-07-25, n=12 per arm, same patched tool, same server, single stream, taint-checked:

| effective temp | t/s | StdDev | SEM |
|---|---|---|---|
| 1.0 (Gemma stock, nothing sent) | 179.43 | 7.71 | 2.23 |
| 0.7 (forced) | **190.17** | 5.61 | 1.62 |

**Δ = 10.74 t/s = +6.0%, t = 3.89, p ≈ 0.001.** Lower temperature also cuts variance
(StdDev 7.71 -> 5.61) — peakier distribution, steadier draft acceptance.

Cross-check that isolates temperature as the ONLY tool difference: the unpatched
shepherd copy (`/tmp/shepherd/scripts/api_bench.py`, forces 0.7) gave 189.69 ± 9.33
(n=12) against the patched copy's 190.17 ± 5.61 at forced 0.7 — **t = 0.15, identical
to within 0.5 t/s.** So nothing else in the patch (4096 cap, reasoning switches, ttfct)
affects throughput. Temperature is the whole story.

### RULE: pre-patch and post-patch benchmarks are NOT comparable

Anything measured before 2026-07-24 16:59 ran at temp 0.7 and is inflated ~6% relative
to a post-patch run that sends nothing. To compare against a historical .bench, either
pass `--temperature 0.7 --top_p 1.0` explicitly or discount the old number by ~6%.

### The phantom regression this caused (2026-07-25)

A "progressive decay" of 193.73 (Jun 18) -> 184.4 (Jul 24 sweep) -> 174.40 (Jul 25)
looked like a real server regression. It was three different measurement conditions read
as one time series. Eliminated in order, all negative:
- vllm build: unchanged since 2026-06-12
- driver 595.58.03 / kernel 6.8.0-101: unchanged; P2P all 6 pairs OK, patched
  open-gpu-kernel-modules .ko loaded (33,558,032 bytes, May 15)
- GPU power limit: `gpu-power-limit.service` moved 280W -> 300W on 2026-07-23 21:18.
  **Wrong direction** — the FAST Jul 17/19 benches ran at 280W. Confirmed irrelevant:
  at batch 1 decode is memory-bandwidth bound and draw never approaches the cap.
- model weights: two HF snapshots (01e5b3ee Jul 17, 4d7ae498 Jul 22, refs/main -> the
  latter) but they **resolve to the same blobs**; config.json, generation_config.json
  and chat_template.jinja all byte-identical. Metadata-only revision bump.
- thinking toggle / token length / max-model-len: controlled or immaterial

**Resolution: at matched temperature, today == Jul 17.** 190.17 (n=12, temp 0.7) vs
193.98 (Jul 17, n=5, temp 0.7): t = 1.00, p ≈ 0.33. No regression ever existed.

### Statistical power — n=5 is not enough

At StdDev 7-10 t/s, n=5 resolves only ~8-10%. Every n=5 pair in this investigation was
non-significant (p ranged 0.12-0.37) and two of them were read as real effects.
**Use `--runs 12` minimum for any comparison under ~8%.** Also discard obvious warm-up
runs — one n=12 set had runs 1-2 at ~170 t/s then 186-198 thereafter, dragging the mean
~3 t/s below the median; prefer the median or drop the warm-up.

`test_model.py` was always correct (`default=None` + conditional payload), so the
MMLU/.test numbers are unaffected.

## Benchmarking discipline that caught real errors

Always verify no other client shares the server: `grep -oE "Running: [0-9]+ reqs"`
in the vLLM log over the benchmark window. Anything >1 invalidates the run. A
duplicate sweep runner produced a 3.6x "slowdown" that was pure contention.
When the server runs under systemd, `journalctl -u <unit>` has this; when launched from
a shell script it does NOT — check `ps` to know which you're looking at.

`pkill -f "sweep/run.sh"` does NOT match `/bin/bash ./run.sh` (relative path) —
kill by PID or match on `bin/vllm serve`. And `pgrep -f <pattern>` self-matches the
shell wrapper containing the pattern; trust `nvidia-smi` memory instead.

Store benchmark RESULTS in the project dir (or .ccmemory), never the /tmp scratchpad —
a power loss mid-session wiped all raw .bench files; only what was written to .ccmemory
survived.

## Free repetition-loop detector: draft acceptance rate

vLLM logs `SpecDecoding metrics: ... Per-position acceptance rate: ... Avg Draft
acceptance rate: N%` every ~10s while the engine is active. Healthy gemma-4-26B is
~68-85%. **Sustained >=99% with mean acceptance length pinned at the ceiling (4.00 for
3 spec tokens) means the model is in a degenerate repetition loop** — a 4-layer drafter
only predicts the 26B target perfectly when the text is perfectly predictable.

Observed 2026-07-25 06:22: `1.000, 1.000, 1.000`, `Accepted: 291, Drafted: 291`,
acceptance length 4.00 — corroborating aitrader's independently-counted hedge repeats.
Costs nothing, needs no instrumentation, and reads far faster than session-level repeat
counts. NOTE: dropping `--speculative-config` destroys this signal.
