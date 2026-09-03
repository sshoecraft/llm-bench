---
name: qwen38-27b-int8-mtp-measured-plus40pct-ampere
description: MEASURED on 4x3090: Qwen3.8-27B INT8 native MTP is +40.5% (64.04 -> 89.98 t/s). Reverses the "no MTP on Ampere" rule, which was MoE-specific to Qwen3…
metadata:
  type: project
---

## Qwen3.8-27B-Uncensored-INT8: MTP is a large WIN on Ampere (measured 2026-08-30)

Hardware: 4x RTX 3090, TP=4, vllm 0.22.1rc1.dev466. Both runs `./api_bench.py
--runs 12` with no other args, identical configs except `--speculative-config`,
both at `--max-model-len 262144`. Taint-checked: 59/60 log samples `Running: 1 reqs`
(one `0`), so single-stream throughout.

| config | t/s mean | StdDev | ITL (ms) | TTFT (ms) |
|---|---|---|---|---|
| plain INT8 | 64.04 | 0.14 | 15.62 | 197.1 |
| **+ MTP, num_speculative_tokens 3** | **89.98** | 4.47 | **11.14** | 213.7 |

- **+40.5% throughput. ITL -28.7%.** Far bigger than Gemma's +21.6% spec gain.
- Draft acceptance rate 62-66%, acceptance length 2.86-2.97.
- TTFT is ~8% worse (197 -> 214 ms) -- the draft pass on prefill. Irrelevant next to
  the decode win.
- StdDev goes 0.14 -> 4.47. Expected: acceptance varies with content. Still a
  completely unambiguous separation (P5 of MTP = 80.72 > P95 of plain = 64.40).

### This REVERSES the old "do not enable MTP on Ampere" rule
`gemma4-vs-qwen36-latency-tuning-4x3090` recorded mtp3 -11.7% / mtp2 -21.4% and it
is tempting to generalize. Do not. That result was **specific to Qwen3.6-35B-A3B**,
whose MTP block is a full 256-expert MoE layer costing 1.57 GiB per draft step --
acceptance was fine there too (2.33-2.37), the draft was just too expensive.
Qwen3.8-27B is **dense**, so its draft step is a dense block, and the economics
invert completely. Rule of thumb going forward: **MTP cost scales with the drafter's
architecture, not with the vendor.** Dense drafter -> enable it; MoE drafter on
Ampere -> measure before trusting.

### KV is a non-issue
Plain INT8: GPU KV cache 921,732 tokens at 262144 context (3.5x headroom).
With MTP: 833,816 tokens (3.2x). MTP costs ~88k tokens of KV and nothing else. The
earlier hedge of dropping `--max-model-len` to 196608 for MTP was unnecessary and
has been reverted in the script.

### Use `./Qwen3.8-27B-Uncensored-INT8-mtp` as the default launcher.

### Caveats on the measurement
- Every run generated **exactly 4096 tokens** = `api_bench`'s default `--max-tokens`
  cap, so all answers were truncated. t/s is unaffected by the cap (confirmed in the
  earlier gemma sweep across 2048/4096/65536), but "time to a complete answer" is
  NOT measurable from these runs.
- "Time to first ANSWER token" is noisy and should not be read as a regression:
  plain 9904 ms mean / 4974 median vs MTP 11258 / 11480. Reasoning length is
  non-deterministic (mean reasoning chars 2669 vs 3558; reasoning share 16.8% vs
  22.2%), so that metric is dominated by how long the model chose to think, not by
  serving speed. **ITL is the clean latency measure here, and it improved 28.7%.**

### Context vs the gemma service this replaced
gemma-4-26B-A4B-it measured 182.9 t/s. Even at 89.98 t/s, Qwen3.8-27B is ~2x slower
per token -- gemma is MoE with ~4B active params, this is a dense 27B reading all
weights. The trade is uncensoring + Qwen coding/agentic strength for ~2x latency.

### Also fixed this session
`api_bench.py` in the working tree had shebang `#!/usr/bin/env python3.13`; no 3.13
exists on this box (system is 3.12.3) and every other script in the repo uses
`python3`. Running `./api_bench.py` failed with
`/usr/bin/env: 'python3.13': No such file or directory`. Reverted to `python3`;
compiles clean on 3.12.
