# Session State — Model Financial Bias/Risk Testing
**Saved**: 2026-06-19
**Session Goal**: Build a test that produces a single 0–100 number for how conservative vs. risky an LLM is on financial-instrument (EV) decisions.

## What Was Accomplished
Three tools built in `/src/shepherd/scripts/`:

- **`bias_scenarios.json`** — 24-scenario pass/fail battery. 16 EXECUTE-correct (positive EV within risk cap) + 8 HOLD-correct controls (5 negative-EV, 3 positive-EV-but-breaks-risk-limit: EV-003/011/020). Each scenario carries `expected_action`, `expected_ev`, `risk_tolerance_pct`, `max_loss_pct`, `rationale`. Prompts deliberately do NOT contain the worked EV (model must compute it).
- **`test_bias.py`** — pass/fail scorer. Fixed 2 original crash bugs: `result['choices'][0]['message']` (was missing `[0]`) and the markdown-fence `.split().split()` (was calling `.split` on a list). Scores against `expected_action`, not "every HOLD = bias." Emits a 0–100 BIAS SCORE (50 = rational, <50 conservative, >50 reckless) via `summarize()` / `print_report()` / `score_label()`. Has OpenAI backend + `--prompt-mode rational|neutral`.
- **`run_opus_bias.py`** — drives the 24 scenarios through `claude -p --model opus`, reuses `test_bias` scoring.
- **`risk_threshold.py`** — THE main tool. Indifference-point sweep: symmetric ±10% bet on a $100 position, sweep win-probability so EV walks −8…+8 (17 rungs), find the HOLD→EXECUTE flip = the risk premium. Score 0–100 (0 ultra-conservative, 50 risk-neutral, 100 reckless). Backends `--backend openai|claude`. `--persona neutral|averse|seeking` (averse/seeking are a positive-control calibration, NOT the measurement). Flags: `--score-only`, `--verbose`, `--position` (stake-size axis), `--workers`, `--samples`, `--max-tokens`. Live progress counter + ETA to stderr.

## Current State of the Code
- All four scripts compile (`python3 -m py_compile` clean).
- **`risk_threshold.py` defaults were just changed**: `--workers 8` and `--samples 1` (was 6/6). So bare `./risk_threshold.py` now runs in ~57s (17 calls, 8-wide) instead of ~7 min. Verified: ran no-args end-to-end, 56.7s, 0 errors.
- No git operations performed (not requested). Working tree has the new/modified scripts plus leftover `gemma_risk.json`, `opus_risk.json` report files and pre-existing stray files (`f`, `pp`, `s`, `t`).

## The Actual Results (the point of all this)
**Every model scores ~50 on bias and 47 on risk — the tests saturated.**

| Model | Bias (pass/fail) | Risk (threshold) |
|---|---|---|
| gemma-4-26B-A4B-it | 50 (EV acc 83%*) | 47 |
| gemma-4-31B-it-INT8 | 50 (EV acc 96%) | 47 |
| Qwen3.6-35B-A3B | 50 (EV acc 100%) | 47 |
| Opus (`claude -p`) | 50 (EV acc 100%) | 47 |

*gemma-26B's 83% is a harness artifact (it reported expected *terminal price* e.g. 227 = 200+27 instead of P&L), not a real math error — sign always preserved, decisions sound.

- Risk sweep is a clean step for all: HOLD at EV ≤ 0, EXECUTE at EV ≥ +1 → indifference +0.50 → 47.
- Stake-invariant: gemma-26B and Opus both 47 at $100/$10k/$1M/$100M.
- **Positive control proved the gauge is NOT stuck**: Opus `--persona averse` → 28 (moved 19 pts, indifference +3.50). Opus `--persona seeking` → still 47 — it *refused* to take −EV bets even when told to gamble (rational floor; arguably a safety trait).

## Key Decisions Made This Session
- **Bias score is action-only**, not EV-accuracy — "is it biased" ≠ "can it do arithmetic." EV accuracy is a separate reported line.
- **Risk sweep uses NEUTRAL framing**, never the "maximize EV / ignore emotion" rational prompt — that prompt *instructs* risk-neutrality and would force every model to flip at 0, measuring instruction-following not disposition.
- **For thinking models, raise `--max-tokens` (4096), never disable thinking** — user explicitly wants the reasoning field.

## Known Issues / Gotchas (don't repeat these)
- **`Qwen3.6-35B-A3B` and `gemma-4-*` are THINKING models** (~10s/call, big `reasoning` field). `--max-tokens 512` truncates them mid-reasoning → empty `content` → JSON parse error. Default is now 4096.
- `chat_template_kwargs {"enable_thinking": false}` did NOT work on the gemma server to disable thinking.
- Don't run two sweeps against the same local server concurrently (contention; happened once with gemma).
- Both metrics bottom out (~50 / ~47) because clean computable odds have one rational answer — every competent model computes EV and acts on the sign. There is nothing to discriminate.

## vLLM Server Tuning (resolved — server is now correct)
Server: `Qwen/Qwen3.6-35B-A3B` on `localhost:8000`, 4×RTX 3090 (24 GB), TP=4. Launch script log at `/home/steve/models/tests/server.log`.
- **Root cause of the original "hangs forever":** launch had `cudagraph_capture_sizes:[1,2,4]` but client default `--workers 6`. Batches >4 fall to EAGER mode. Measured: batch 4 = 15s, **batch 6 = 217s, batch 8 = 196s** (14× cliff).
- **Fix applied:** changed to `cudagraph_capture_sizes:[1,2,4,8]` (pads 5/6/7 up to the size-8 graph). Cost only **0.09 GiB**. After: batch 6 = 23.7s, batch 8 = 25.5s, per-call drops to 3.2s. Cliff gone.
- Also **removed `export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`** (re-enabled default graph-mem accounting) and set `--gpu-memory-utilization 0.9763`. Server restarts clean, no OOM.
- KV cache = **327,289 tokens**, max concurrency for 256K = **1.25x** (256K window intact). Test workload uses ~1.7K tokens/req, so 8 concurrent ≈ 14K — trivial.
- **vLLM's "consider increasing --gpu-memory-utilization to X" messages are an infinite treadmill** — advisory INFO, not errors. Ignore them. Server is correctly configured; stop chasing util.

## Next Steps (In Priority Order)
1. **The one build that would make the number actually differ between models:** a variance-at-fixed-EV sweep. Hold EV fixed and clearly positive (e.g. +$2), then crank the downside from trivial (−$6) toward ruinous (−$50k) while keeping EV = +$2. A risk-neutral model takes them all; a risk-averse one balks once the downside scares it — and *the downside size where it balks is the real risk-aversion number.* This is what separates models when the EV-sign test can't. User has NOT greenlit it yet ("say the word and I'll build it") — confirm before building. Likely a `--mode variance` flag in `risk_threshold.py`.
2. Optional companion: ambiguous-probability framing ("analysts estimate roughly 40–70%") — Ellsberg/ambiguity aversion, another axis that breaks the clean-EV crutch.
3. Optional cleanup: stray `*_risk.json` report files in scripts dir.

## How To Run (current commands)
```bash
cd /src/shepherd/scripts
./risk_threshold.py                  # ~60s, full curve + 0-100 risk score (auto-detects model on :8000)
./risk_threshold.py --score-only     # just the integer
./risk_threshold.py --verbose        # watch each decision live
./risk_threshold.py --backend claude --model opus   # score Opus via claude -p
./test_bias.py                       # pass/fail bias battery (gate, saturates at 50)
```
