# llm-bench

A working toolkit for benchmarking locally-served LLMs. Everything here talks to an
OpenAI-compatible endpoint (in practice a [vLLM](https://github.com/vllm-project/vllm)
server on `localhost:8000`), plus a shim that puts the `claude` CLI behind the same
interface so hosted models can be scored with the identical harness.

This is a real bench, not a demo. It grew out of tuning a 4×RTX 3090 rig, so the repo
carries both the tools and the recorded results and launch configurations that produced
them.

## What's in here

### Accuracy

| Tool | What it measures |
|---|---|
| `test_model.py` | MMLU and HellaSwag accuracy against an OpenAI-compatible endpoint or the `shepherd` CLI. Subject filtering, seeded sampling, configurable context/backends. |
| `human_eval.py` | The 164 HumanEval programming problems via `/v1/completions` (true completion mode, not chat). Self-activating venv wrapper. |
| `mmlu`, `fin` | One-line `test_model.py` invocations pinning a 100-question quantitative/financial MMLU subject set at seed 42. |

### Throughput and latency

| Tool | What it measures |
|---|---|
| `llm_bench.py` | Unified benchmark across OpenAI-compatible, Anthropic, Gemini and claude-CLI endpoints — pooled decode throughput, TTFT/TTFA, inter-chunk latency, over a fixed diverse prompt set to defeat prefix caching. Supports paired A/B via `--variant`, judged by sign test. Replaces `api_bench.py` (now in `archive/`). |
| `concurrent_sweep.py` | Sweeps concurrency levels (1→128) and reports per-request tok/s, TTFT, ITL and aggregate throughput at each level. This is what exposes CUDA-graph capture cliffs. |
| `tpbench.py` | Lean concurrent throughput probe with synthetic fixed-length input/output. |

### Behavioral / risk

| Tool | What it measures |
|---|---|
| `risk_profile.py` | Prospect-theory risk battery: Tanaka-Camerer-Nguyen lottery series (σ curvature, λ loss aversion, α probability weighting), Holt-Laury price list (CRRA *r*), Grable-Lytton 13-item stated scale, and an Asian-disease / Ellsberg framing probe. Emits a 0-100 conservatism index. `--selftest` validates the estimators against synthetic known-parameter agents with no API calls. See [`risk_profile.md`](risk_profile.md). |

### Infrastructure

| File | Purpose |
|---|---|
| `claude_shim.py` | OpenAI-compatible HTTP server that proxies to the `claude` CLI, so any tool here can score Claude Code without a separate code path. |
| `adapters/` | Backend adapter layer (`base.py`, `claude.py`). |
| `test_eviction.py` | Exercises the `shepherd` client's context-eviction and RAG archival behavior via stdin/stdout, verifying evicted turns land in the archive DB. |
| `test_one_swebench.py`, `swebench_safety_wrapper.sh` | Single SWE-bench task driver and a sandbox wrapper that pins the agent to `/tmp/swebench_*`, scrubs SSH/AWS env, and sets a restrictive umask. |
| `check_alloc.cu` | Minimal CUDA allocation probe. |
| `test_clyde.int.c` | Large single-file C source used as a long-context payload. |

## vLLM launch scripts

The extensionless executables at the repo root are `vllm serve` launch scripts, one per
model configuration:

- `gemma-4-26B-A4B-it`, `-nothink`, `-tp2`, `-wow`
- `gemma-4-31B-it-INT8`
- `Qwen3.6-35B-A3B`, `-wow`, `-wow-bench`

They encode the tuning that actually matters on this hardware — tensor-parallel size,
`cudagraph_capture_sizes`, `max-num-batched-tokens`, speculative-decoding config, KV cache
dtype and GPU memory utilization. `archive/` and `old/` hold superseded launchers and
earlier iterations of the risk tooling.

## Result files

Benchmark output is committed alongside the configuration that produced it, named after
the launch script:

- `*.bench` — throughput/latency runs
- `*.test`, `*.full_test` — accuracy runs
- `*.risk`, `*.finance` — risk-battery and financial-MMLU results
- `*_risk.json`, `grable_lytton.json` — structured risk output and the verbatim
  Grable & Lytton (1999) instrument

## Notes

- [`my_hw.md`](my_hw.md) — the rig these numbers come from (dual Xeon E5-2680 v4, 96 GB
  DDR4 ECC, 4× RTX 3090 / 96 GB VRAM). Numbers here are Ampere numbers; they do not
  transfer to Hopper or Blackwell.
- [`risk_profile.md`](risk_profile.md) — the math, instrument design, and calibration
  history behind the risk battery.
- [`state.md`](state.md) — session notes from the financial bias/risk work, including why
  the earlier EV-threshold approach saturated at the same score for every model.

## Requirements

Python 3.12, `openai` (async client), `scipy` for the risk estimators, and a running
OpenAI-compatible endpoint. Most tools auto-detect the served model name from
`localhost:8000`.

```bash
./test_model.py --benchmark mmlu --count 100
./concurrent_sweep.py --steps 1,8,16,32,64,96,128 --max-tokens 2048
./risk_profile.py --score-only
./risk_profile.py --selftest          # no API calls
```

## License

MIT — see [LICENSE](LICENSE).
