#!/usr/bin/env python3
"""
llm_bench.py -- LLM endpoint benchmark.  Rewrite of api_bench.py.

WHAT THIS MEASURES
------------------
End-to-end, client-observed serving performance: what an application talking to
this endpoint over this network actually experiences.  It is NOT a pure backend
generation benchmark -- the decode rate includes server scheduling, serialization,
SSE framing and network jitter.  That is deliberate and useful; just label the
numbers that way when you quote them.

CHUNKS ARE NOT TOKENS.  A server may pack many tokens into one SSE frame, and
different backends frame differently.  So:
  * token counts ALWAYS come from the server usage block; when usage is missing
    we fall back to chars/4 and flag the run (tokens_estimated), and the summary
    reports how many runs were estimated.  Estimated runs never silently blend
    into measured ones.
  * the per-event latency metric is ICL (inter-CHUNK latency), not ITL, and it
    is computed from real chunk arrival timestamps -- not as 1/throughput, which
    would just be the same number printed twice.
  * tokens-per-chunk is reported so you can see the framing granularity.

PRIMARY THROUGHPUT IS POOLED:  sum(tokens-1) / sum(decode_time)  over all runs,
not the mean of per-run rates.  Mean-of-ratios weights a 40-token run the same
as a 4000-token run.  Decode rate uses N-1 intervals between N tokens.

WORKLOAD CONTROL.  Comparing models by "tokens/sec at --max-tokens 4096" compares
verbosity as much as speed, and decode rate itself decays as the KV cache grows.
Use --min-tokens (vLLM/llama.cpp) or --ignore-eos (vLLM) to pin output length,
and check the reported output-token spread before trusting any A/B.

PAIRED A/B.  Hosted endpoints are non-stationary: 50 runs of config A followed by
50 of config B measures time-of-day as much as config.  --variant runs configs
interleaved round-robin against an identical prompt sequence and reports paired
per-round deltas.  Use it instead of two separate invocations.

Every run is emitted to --json / --csv with input tokens, output tokens, finish
reason, timings and flags, so results can be re-analysed later.
"""

import argparse
import csv
import json
import math
import os
import platform
import queue
import random
import statistics
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

VERSION = "2.0"

# Fixed corpus, fixed order.  Order is only randomised when --prompt-mode shuffled
# is given, and then only under an explicit --seed, so a comparison is repeatable.
BENCHMARK_CORPUS = [
    "Explain how neural networks learn through backpropagation.",
    "What are the key differences between TCP and UDP protocols?",
    "Describe the process of photosynthesis in plants.",
    "How does a compiler transform source code into machine code?",
    "Explain the concept of entropy in thermodynamics.",
    "What are the main principles of object-oriented programming?",
    "How do vaccines train the immune system?",
    "Describe the architecture of a modern CPU.",
    "What causes the seasons on Earth?",
    "Explain how public key cryptography works.",
    "What are the stages of the software development lifecycle?",
    "How does natural selection drive evolution?",
    "Describe the structure and function of DNA.",
    "What is the difference between machine learning and deep learning?",
    "Explain how HTTP requests and responses work.",
    "What are the fundamental forces in physics?",
    "How do databases maintain ACID properties?",
    "Describe the water cycle and its importance.",
    "What is the role of mitochondria in cells?",
    "Explain the concept of recursion in programming.",
    "How does the human brain process visual information?",
    "What are the key features of functional programming?",
    "Describe how earthquakes occur along fault lines.",
    "What is the difference between REST and GraphQL APIs?",
    "Explain how batteries store and release energy.",
]

# A decode window shorter than this with >1 token means the server buffered the
# whole response and flushed it at once (fake streaming).  Such runs are excluded
# from decode-rate stats and counted separately rather than silently rewritten.
BUFFERED_DECODE_FLOOR_S = 0.002

# Below this many samples the extreme percentiles are not estimates, they are the
# extreme samples wearing labels: at n=12, P95 IS the second-largest point and P99
# IS the max.  Below this threshold they are suppressed and an explicit min/max is
# printed instead.  P50/P90 still survive and are printed at any n.
PERCENTILE_MIN_N = 20


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    """Per-call request settings.  None means: do not send the field at all, so
    the server's own default (vLLM reads generation_config.json) applies."""
    max_tokens: int = 512
    min_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    ignore_eos: bool = False
    stream: bool = True
    debug: bool = False


@dataclass
class RawRun:
    """Raw observations for one request.  No derived metrics live here -- they are
    all computed centrally in compute_metrics() so that every backend uses exactly
    the same definition of generation_time.  (The old script defined it three
    different ways across three clients, which made cross-provider numbers
    incomparable by construction.)"""
    variant: str = "default"
    run_index: int = 0
    round_index: int = 0
    prompt_id: int = -1
    stream: bool = True
    start: float = 0.0
    end: float = 0.0
    first_chunk: Optional[float] = None      # first streamed event of any kind
    first_content: Optional[float] = None    # first event of the actual ANSWER
    last_chunk: Optional[float] = None       # last content-bearing event
    chunk_times: List[float] = field(default_factory=list)
    output_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    tokens_estimated: bool = False
    content_chars: int = 0
    reasoning_chars: int = 0
    finish_reason: Optional[str] = None
    spawn_ms: Optional[float] = None         # claude-cli process spawn, excluded
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def sign_test_p(wins: int, n: int) -> Optional[float]:
    """Two-sided sign test over paired rounds.  n is the number of non-tied pairs,
    wins the number favouring the variant.  Returns the probability of a split at
    least this lopsided if the two configs were indistinguishable and each round
    were a coin flip.  This tests DIRECTION ONLY -- it says a winner is consistent,
    not that the win is large.  Read the median delta for magnitude.

    At n=12:  12/12 p=0.000   11/12 p=0.006   10/12 p=0.039   9/12 p=0.146
              8/12 p=0.388 -- i.e. a 10-2 split is callable, an 8-4 split is not."""
    if n <= 0:
        return None
    k = max(wins, n - wins)
    tail = sum(math.comb(n, i) for i in range(k, n + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def _sign_verdict(wins: int, losses: int) -> str:
    n = wins + losses
    p = sign_test_p(wins, n)
    if p is None:
        return ""
    if p < 0.05:
        verdict = "consistent" if wins > losses else "consistently worse"
    else:
        verdict = f"not callable at n={n}"
    return f"  sign test p={p:.3f} ({verdict})"


def pct(values: List[float], p: float) -> Optional[float]:
    """Linear-interpolated percentile.  p is 0..100 and means what it says: for a
    rate metric, P95 is the FAST tail and P5 is the slow tail.  The old script's
    sorted(v)[int(len(v)*p)] returned the max for any p>=0.95 at n<=20 and the min
    for any p<=0.05 at n<20."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return float(s[int(k)])
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


def fmt(v: Optional[float], nd: int = 2) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def compute_metrics(r: RawRun) -> Dict[str, Any]:
    """Derive everything from one RawRun.  Single definition, all backends."""
    m: Dict[str, Any] = {
        "variant": r.variant, "run": r.run_index, "round": r.round_index,
        "prompt_id": r.prompt_id, "stream": r.stream,
        "prompt_tokens": r.prompt_tokens, "output_tokens": r.output_tokens,
        "reasoning_tokens": r.reasoning_tokens, "cached_tokens": r.cached_tokens,
        "tokens_estimated": r.tokens_estimated,
        "content_chars": r.content_chars, "reasoning_chars": r.reasoning_chars,
        "finish_reason": r.finish_reason, "spawn_ms": r.spawn_ms,
        "error": r.error,
    }
    tok = r.output_tokens or 0
    total = max(0.0, r.end - r.start)
    m["total_ms"] = total * 1000
    m["ok"] = r.error is None and tok > 0
    m["empty_answer"] = bool(m["ok"] and r.content_chars == 0)
    m["ttft_ms"] = (r.first_chunk - r.start) * 1000 if r.first_chunk else None
    m["ttfa_ms"] = (r.first_content - r.start) * 1000 if r.first_content else None
    m["chunks"] = len(r.chunk_times)
    m["tokens_per_chunk"] = (tok / len(r.chunk_times)) if (tok and r.chunk_times) else None

    decode = None
    if r.stream and r.first_chunk is not None and r.last_chunk is not None:
        decode = max(0.0, r.last_chunk - r.first_chunk)
    m["decode_ms"] = decode * 1000 if decode is not None else None
    m["buffered"] = bool(decode is not None and tok > 1 and decode < BUFFERED_DECODE_FLOOR_S)

    # N tokens have N-1 intervals.  Dividing the window by N overstates the rate
    # by N/(N-1) -- 5% at 20 tokens, and it is a systematic bias, not noise.
    if decode and tok >= 2 and not m["buffered"]:
        m["decode_tps"] = (tok - 1) / decode
    else:
        m["decode_tps"] = None
    m["e2e_tps"] = (tok / total) if (total > 0 and tok > 0) else None

    gaps = [(b - a) * 1000 for a, b in zip(r.chunk_times, r.chunk_times[1:])]
    m["icl_p50_ms"] = pct(gaps, 50)
    m["icl_p99_ms"] = pct(gaps, 99)
    m["icl_max_ms"] = max(gaps) if gaps else None
    return m


# ---------------------------------------------------------------------------
# Clients.  Each returns a RawRun; none of them compute metrics.
# ---------------------------------------------------------------------------

class APIClient(ABC):
    @abstractmethod
    def chat(self, prompt: str, cfg: RunConfig) -> RawRun:
        ...

    @abstractmethod
    def list_models(self) -> List[str]:
        ...

    def close(self) -> None:
        pass


def estimate_tokens_if_needed(r: RawRun) -> None:
    """chars/4 fallback.  Always flagged so it can be excluded from analysis --
    the old script mixed estimates into the same distribution as measured counts
    with nothing in the output to say which was which."""
    if r.output_tokens is None:
        r.output_tokens = max(0, (r.content_chars + r.reasoning_chars) // 4)
        r.tokens_estimated = True


class OpenAIClient(APIClient):
    """OpenAI-compatible endpoint (vLLM, llama.cpp, SGLang, TGI, OpenAI itself)."""

    def __init__(self, base_url, api_key, model, extra_params=None, insecure=False):
        from openai import OpenAI
        self._http = None
        if insecure:
            import httpx
            self._http = httpx.Client(verify=False)
        self.client = OpenAI(base_url=base_url, api_key=api_key, http_client=self._http)
        self.model = model
        self.extra_params = extra_params or {}

    def close(self):
        if self._http is not None:
            self._http.close()
            self._http = None

    def list_models(self):
        return [m.id for m in self.client.models.list().data]

    @staticmethod
    def _apply_usage(r: RawRun, usage) -> None:
        if not usage:
            return

        def g(obj, key):
            if obj is None:
                return None
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key, None)

        ct = g(usage, "completion_tokens")
        if ct is not None:
            r.output_tokens = ct
        pt = g(usage, "prompt_tokens")
        if pt is not None:
            r.prompt_tokens = pt
        rt = g(g(usage, "completion_tokens_details"), "reasoning_tokens")
        if rt is not None:
            r.reasoning_tokens = rt
        cached = g(g(usage, "prompt_tokens_details"), "cached_tokens")
        if cached is not None:
            r.cached_tokens = cached

    def chat(self, prompt: str, cfg: RunConfig) -> RawRun:
        r = RawRun(stream=cfg.stream)

        extra_body: Dict[str, Any] = {}
        if cfg.top_k is not None:
            extra_body["top_k"] = cfg.top_k
        if cfg.min_tokens is not None:
            extra_body["min_tokens"] = cfg.min_tokens
        if cfg.ignore_eos:
            extra_body["ignore_eos"] = True
        for k, v in self.extra_params.items():
            if v is not None:
                extra_body[k] = v

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": cfg.max_tokens,
            "stream": cfg.stream,
        }
        if cfg.temperature is not None:
            kwargs["temperature"] = cfg.temperature
        if cfg.top_p is not None:
            kwargs["top_p"] = cfg.top_p
        if cfg.stream:
            kwargs["stream_options"] = {"include_usage": True}
        if extra_body:
            kwargs["extra_body"] = extra_body

        r.start = time.perf_counter()
        resp = self.client.chat.completions.create(**kwargs)
        t_returned = time.perf_counter()

        if cfg.stream:
            usage = None
            for chunk in resp:
                now = time.perf_counter()
                u = getattr(chunk, "usage", None)
                if u:
                    usage = u
                if not chunk.choices:
                    continue                      # usage-only trailer frame
                choice = chunk.choices[0]
                if choice.finish_reason:
                    r.finish_reason = choice.finish_reason
                delta = getattr(choice, "delta", None)
                content = reasoning = None
                if delta is not None:
                    extra = getattr(delta, "model_extra", None) or {}
                    content = getattr(delta, "content", None)
                    reasoning = (getattr(delta, "reasoning_content", None)
                                 or extra.get("reasoning_content")
                                 or extra.get("reasoning"))
                if cfg.debug:
                    print(f"  chunk content={content!r} reasoning={reasoning!r}")
                if not content and not reasoning:
                    continue
                r.content_chars += len(content or "")
                r.reasoning_chars += len(reasoning or "")
                if r.first_chunk is None:
                    r.first_chunk = now
                if content and r.first_content is None:
                    r.first_content = now
                r.last_chunk = now
                r.chunk_times.append(now)
            r.end = time.perf_counter()
            self._apply_usage(r, usage)
        else:
            r.end = t_returned
            r.first_chunk = t_returned
            if resp.choices:
                choice = resp.choices[0]
                r.finish_reason = choice.finish_reason
                msg = choice.message
                extra = getattr(msg, "model_extra", None) or {}
                reasoning = (getattr(msg, "reasoning_content", None)
                             or extra.get("reasoning_content")
                             or extra.get("reasoning") or "")
                content = msg.content or ""
                r.content_chars = len(content)
                r.reasoning_chars = len(reasoning)
                if content:
                    r.first_content = t_returned
            # Read usage unconditionally.  The old script nested this inside
            # `if content or reasoning:` while its own comment said the point was
            # to catch billed-but-empty responses -- so it missed exactly the
            # case it was written for.
            self._apply_usage(r, getattr(resp, "usage", None))

        estimate_tokens_if_needed(r)
        return r


class AnthropicClient(APIClient):
    """Anthropic Messages API.  Iterates raw stream events rather than
    text_stream so thinking deltas are visible and usage is read properly."""

    def __init__(self, base_url, api_key, model, extra_params=None, insecure=False):
        import anthropic
        self._http = None
        if insecure:
            import httpx
            self._http = httpx.Client(verify=False)
        kw: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kw["base_url"] = base_url
        if self._http is not None:
            kw["http_client"] = self._http
        self.client = anthropic.Anthropic(**kw)
        self.model = model
        self.extra_params = extra_params or {}

    def close(self):
        if self._http is not None:
            self._http.close()
            self._http = None

    def list_models(self):
        # There IS a models endpoint now; the old hardcoded list auto-selected
        # claude-3-opus-20240229, which is retired.
        try:
            return [m.id for m in self.client.models.list().data]
        except Exception:
            return []

    @staticmethod
    def _apply_usage(r: RawRun, usage) -> None:
        if usage is None:
            return
        ot = getattr(usage, "output_tokens", None)
        if ot is not None:                       # `is not None`, not `> 1`
            r.output_tokens = ot
        it = getattr(usage, "input_tokens", None)
        if it is not None:
            r.prompt_tokens = it
        cached = getattr(usage, "cache_read_input_tokens", None)
        if cached is not None:
            r.cached_tokens = cached

    def chat(self, prompt: str, cfg: RunConfig) -> RawRun:
        r = RawRun(stream=cfg.stream)
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": cfg.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if cfg.temperature is not None:
            kwargs["temperature"] = cfg.temperature
        if cfg.top_p is not None:
            kwargs["top_p"] = cfg.top_p
        if cfg.top_k is not None:
            kwargs["top_k"] = cfg.top_k
        for k, v in self.extra_params.items():
            if v is not None and k not in ("effort", "session_mode", "timeout"):
                kwargs[k] = v

        r.start = time.perf_counter()
        if cfg.stream:
            final = None
            with self.client.messages.stream(**kwargs) as stream:
                for event in stream:
                    now = time.perf_counter()
                    if getattr(event, "type", "") != "content_block_delta":
                        continue
                    delta = getattr(event, "delta", None)
                    text = getattr(delta, "text", None)
                    thinking = getattr(delta, "thinking", None)
                    if cfg.debug:
                        print(f"  chunk text={text!r} thinking={thinking!r}")
                    if not text and not thinking:
                        continue
                    r.content_chars += len(text or "")
                    r.reasoning_chars += len(thinking or "")
                    if r.first_chunk is None:
                        r.first_chunk = now
                    if text and r.first_content is None:
                        r.first_content = now
                    r.last_chunk = now
                    r.chunk_times.append(now)
                try:
                    final = stream.get_final_message()
                except Exception:
                    final = None
            # end_time is assigned unconditionally.  The old script only set it
            # inside `if last_token_time:`, so an empty stream raised
            # UnboundLocalError instead of recording a failed run.
            r.end = time.perf_counter()
            if final is not None:
                r.finish_reason = getattr(final, "stop_reason", None)
                self._apply_usage(r, getattr(final, "usage", None))
        else:
            resp = self.client.messages.create(**kwargs)
            r.end = time.perf_counter()
            r.first_chunk = r.end
            r.finish_reason = getattr(resp, "stop_reason", None)
            for block in (getattr(resp, "content", None) or []):
                btype = getattr(block, "type", "")
                if btype == "thinking":
                    r.reasoning_chars += len(getattr(block, "thinking", "") or "")
                else:
                    text = getattr(block, "text", "") or ""
                    r.content_chars += len(text)
                    if text and r.first_content is None:
                        r.first_content = r.end
            self._apply_usage(r, getattr(resp, "usage", None))

        estimate_tokens_if_needed(r)
        return r


class GeminiClient(APIClient):
    """Google Gemini.  Prefers the current google-genai SDK, falls back to the
    deprecated google-generativeai if that is what is installed."""

    def __init__(self, base_url, api_key, model, extra_params=None, insecure=False):
        self.model = model
        self.extra_params = extra_params or {}
        try:
            from google import genai
            self._sdk = "new"
            self._genai = genai
            self._client = genai.Client(api_key=api_key)
        except ImportError:
            import google.generativeai as genai_old
            genai_old.configure(api_key=api_key)
            self._sdk = "old"
            self._genai = genai_old
            self._client = None

    def list_models(self):
        if self._sdk == "new":
            return [m.name for m in self._client.models.list()]
        return [m.name for m in self._genai.list_models()
                if "generateContent" in getattr(m, "supported_generation_methods", [])]

    @staticmethod
    def _safe_text(chunk) -> Optional[str]:
        # chunk.text RAISES when a candidate has no parts (safety stop, empty
        # candidate).  hasattr does not protect you -- the property itself throws.
        try:
            return chunk.text
        except Exception:
            return None

    @staticmethod
    def _apply_usage(r: RawRun, um) -> None:
        if um is None:
            return
        ot = getattr(um, "candidates_token_count", None)
        if ot is not None:
            r.output_tokens = ot
        it = getattr(um, "prompt_token_count", None)
        if it is not None:
            r.prompt_tokens = it
        rt = getattr(um, "thoughts_token_count", None)
        if rt is not None:
            r.reasoning_tokens = rt
        cached = getattr(um, "cached_content_token_count", None)
        if cached is not None:
            r.cached_tokens = cached

    def _make_stream(self, prompt: str, cfg: RunConfig, stream: bool):
        if self._sdk == "new":
            from google.genai import types
            gc: Dict[str, Any] = {"max_output_tokens": cfg.max_tokens}
            if cfg.temperature is not None:
                gc["temperature"] = cfg.temperature
            if cfg.top_p is not None:
                gc["top_p"] = cfg.top_p
            if cfg.top_k is not None:
                gc["top_k"] = cfg.top_k
            config = types.GenerateContentConfig(**gc)
            if stream:
                return self._client.models.generate_content_stream(
                    model=self.model, contents=prompt, config=config)
            return self._client.models.generate_content(
                model=self.model, contents=prompt, config=config)
        gc = {"max_output_tokens": cfg.max_tokens}
        if cfg.temperature is not None:
            gc["temperature"] = cfg.temperature
        if cfg.top_p is not None:
            gc["top_p"] = cfg.top_p
        if cfg.top_k is not None:
            gc["top_k"] = cfg.top_k
        config = self._genai.GenerationConfig(**gc)
        model = self._genai.GenerativeModel(self.model)
        return model.generate_content(prompt, generation_config=config, stream=stream)

    def chat(self, prompt: str, cfg: RunConfig) -> RawRun:
        r = RawRun(stream=cfg.stream)
        r.start = time.perf_counter()
        resp = self._make_stream(prompt, cfg, cfg.stream)

        if cfg.stream:
            for chunk in resp:
                now = time.perf_counter()
                self._apply_usage(r, getattr(chunk, "usage_metadata", None))
                text = self._safe_text(chunk)
                if cfg.debug:
                    print(f"  chunk text={text!r}")
                if not text:
                    continue
                r.content_chars += len(text)
                if r.first_chunk is None:
                    r.first_chunk = now
                if r.first_content is None:
                    r.first_content = now
                r.last_chunk = now
                r.chunk_times.append(now)
            r.end = time.perf_counter()
            self._apply_usage(r, getattr(resp, "usage_metadata", None))
        else:
            r.end = time.perf_counter()
            r.first_chunk = r.end
            text = self._safe_text(resp) or ""
            r.content_chars = len(text)
            if text:
                r.first_content = r.end
            self._apply_usage(r, getattr(resp, "usage_metadata", None))

        cands = getattr(resp, "candidates", None) or []
        if cands:
            fr = getattr(cands[0], "finish_reason", None)
            r.finish_reason = getattr(fr, "name", None) or (str(fr) if fr else None)
        estimate_tokens_if_needed(r)
        return r


class ClaudeCLIClient(APIClient):
    """Drives the real `claude` CLI via adapters/claude.py -- a --print session
    with full Claude Code behaviour (CLAUDE.md, hooks, MCP servers, tools).

    Session mode matters and is now explicit, because it changes the workload:

      stateless  -- fresh process per run, every run is turn 1 of its own
                    conversation.  Spawn cost is measured separately (spawn_ms)
                    and excluded from the run clock, so TTFT is the model's, not
                    the CLI's.  This is the mode for comparing models.
      persistent -- one process, run N is turn N of one growing conversation.
                    Context grows monotonically, so TTFT drifts upward across the
                    run; that is a real property of long agent sessions, but it
                    is a trend, not noise, and mean/median will hide it.  The
                    summary prints a first-third vs last-third TTFT drift line.

    Note that whatever this measures includes hook execution, MCP connection and
    tool turns.  Do not compare these tokens/sec against an --type openai run.
    """

    def __init__(self, base_url, api_key, model, extra_params=None, insecure=False):
        from adapters.claude import ClaudeAdapter
        self._ClaudeAdapter = ClaudeAdapter
        self.model = model if model and model != "placeholder" else None
        ep = extra_params or {}
        self.effort = ep.get("effort")
        self.session_mode = ep.get("session_mode") or "stateless"
        self.timeout = float(ep.get("timeout") or 600)
        self._adapter = None
        self._events = None

    def list_models(self):
        return []

    def _spawn(self, max_tokens: int) -> float:
        t0 = time.perf_counter()
        events: "queue.Queue" = queue.Queue()
        params: Dict[str, Any] = {"max_output_tokens": max_tokens} if max_tokens else {}
        if self.effort:
            params["effort"] = self.effort
        adapter = self._ClaudeAdapter(model=self.model, params=params)
        adapter.on_delta = lambda frag: events.put(("delta", frag, time.perf_counter()))
        adapter.on_result = lambda cost, turns, usage, is_error: events.put(
            ("done", {"usage": usage, "is_error": is_error}, time.perf_counter()))
        adapter.on_exit = lambda code: events.put(("exit", code, time.perf_counter()))
        adapter.spawn()
        self._adapter = adapter
        self._events = events
        return (time.perf_counter() - t0) * 1000

    def close(self):
        if self._adapter is not None:
            try:
                self._adapter.kill()
            except Exception:
                pass
            self._adapter = None
            self._events = None

    def chat(self, prompt: str, cfg: RunConfig) -> RawRun:
        r = RawRun(stream=cfg.stream)

        if self.session_mode == "stateless":
            self.close()
        if self._adapter is None or not getattr(self._adapter, "alive", False):
            r.spawn_ms = self._spawn(cfg.max_tokens)

        r.start = time.perf_counter()
        self._adapter.send(prompt)

        payload: Dict[str, Any] = {}
        while True:
            try:
                kind, data, ts = self._events.get(timeout=self.timeout)
            except queue.Empty:
                # A timeout is a FAILED run, not a run with partial data quietly
                # averaged into the results.
                r.error = f"timeout after {self.timeout:.0f}s"
                r.end = time.perf_counter()
                self.close()
                return r
            if kind == "delta":
                if cfg.debug:
                    print(f"  chunk {data!r}")
                if data:
                    if r.first_chunk is None:
                        r.first_chunk = ts
                    if r.first_content is None:
                        r.first_content = ts
                    r.last_chunk = ts
                    r.chunk_times.append(ts)
                    r.content_chars += len(str(data))
            elif kind == "done":
                payload = data or {}
                r.end = ts
                break
            elif kind == "exit":
                r.error = f"session exited (code {data})"
                r.end = ts
                self._adapter = None
                self._events = None
                break

        if not r.end:
            r.end = time.perf_counter()

        usage = payload.get("usage") or {}
        if isinstance(usage, dict):
            ot = usage.get("output_tokens")
            it = usage.get("input_tokens")
            cached = usage.get("cache_read_input_tokens")
        else:
            ot = getattr(usage, "output_tokens", None)
            it = getattr(usage, "input_tokens", None)
            cached = getattr(usage, "cache_read_input_tokens", None)
        if ot is not None:
            r.output_tokens = ot
        if it is not None:
            r.prompt_tokens = it
        if cached is not None:
            r.cached_tokens = cached
        if payload.get("is_error") and not r.chunk_times:
            r.error = r.error or "claude reported error with no output"
        if r.output_tokens is None:
            # chunk_count is text fragments, not tokens.  Flag it.
            r.output_tokens = max(0, r.content_chars // 4)
            r.tokens_estimated = True
        if self.session_mode == "stateless":
            self.close()
        return r


def create_client(api_type, base_url, api_key, model, extra_params=None, insecure=False) -> APIClient:
    if api_type == "openai":
        return OpenAIClient(base_url, api_key, model, extra_params, insecure=insecure)
    if api_type == "anthropic":
        return AnthropicClient(base_url, api_key, model, extra_params, insecure=insecure)
    if api_type == "gemini":
        return GeminiClient(base_url, api_key, model, extra_params, insecure=insecure)
    if api_type == "claude-cli":
        return ClaudeCLIClient(base_url, api_key, model, extra_params, insecure=insecure)
    raise ValueError(f"Unknown API type: {api_type}")


# ---------------------------------------------------------------------------
# Variants and the run loop
# ---------------------------------------------------------------------------

VARIANT_KEYS = {"model", "effort", "thinking", "reasoning", "max_tokens",
                "min_tokens", "temperature", "top_p", "top_k", "ignore_eos"}


@dataclass
class Variant:
    name: str
    overrides: Dict[str, Any] = field(default_factory=dict)
    client: Optional[APIClient] = None
    cfg: Optional[RunConfig] = None
    thinking: Optional[str] = None
    reasoning: Optional[str] = None
    model: Optional[str] = None


def parse_variant(spec: str) -> Variant:
    """NAME:key=value,key=value   e.g.  think_off:thinking=off,max_tokens=1024"""
    if ":" in spec:
        name, _, rest = spec.partition(":")
    else:
        name, rest = spec, ""
    overrides: Dict[str, Any] = {}
    for pair in [p for p in rest.split(",") if p.strip()]:
        if "=" not in pair:
            raise ValueError(f"variant '{name}': expected key=value, got '{pair}'")
        k, _, v = pair.partition("=")
        k = k.strip()
        if k not in VARIANT_KEYS:
            raise ValueError(f"variant '{name}': unknown key '{k}' "
                             f"(allowed: {', '.join(sorted(VARIANT_KEYS))})")
        overrides[k] = v.strip()
    return Variant(name=name.strip() or "default", overrides=overrides)


def build_prompt_sequence(args, n_needed: int) -> List[int]:
    """Returns prompt indices into the corpus, or [-1]*n for --prompt-mode repeated.
    The SAME sequence is handed to every variant, which is what makes the paired
    comparison paired."""
    if args.prompt_mode == "repeated":
        return [-1] * n_needed
    order = list(range(len(BENCHMARK_CORPUS)))
    if args.prompt_mode == "shuffled":
        random.Random(args.seed).shuffle(order)
    seq: List[int] = []
    while len(seq) < n_needed:
        seq.extend(order)
    return seq[:n_needed]


def prompt_for(args, pid: int) -> str:
    return args.prompt if pid < 0 else BENCHMARK_CORPUS[pid]


def safe_chat(v: Variant, prompt: str, cfg: RunConfig) -> RawRun:
    """One request.  A transient 429/500/timeout must not discard the other 49
    runs, so failures are recorded and the benchmark continues."""
    try:
        return v.client.chat(prompt, cfg)
    except Exception as e:
        now = time.perf_counter()
        return RawRun(stream=cfg.stream, start=now, end=now,
                      error=f"{type(e).__name__}: {e}")


def run_benchmark(args, variants: List[Variant]) -> List[RawRun]:
    total_prompts = args.warmup + args.runs
    seq = build_prompt_sequence(args, total_prompts)

    if args.warmup > 0:
        print(f"Warming up ({args.warmup} runs x {len(variants)} variant(s))...")
        for v in variants:
            for i in range(args.warmup):
                safe_chat(v, prompt_for(args, seq[i]), v.cfg)

    print(f"Running benchmark ({args.runs} rounds x {len(variants)} variant(s))"
          f"{' [interleaved, paired]' if len(variants) > 1 else ''}...")

    raws: List[RawRun] = []
    counters = {v.name: 0 for v in variants}
    for i in range(args.runs):
        pid = seq[args.warmup + i]
        prompt = prompt_for(args, pid)
        # Round-robin so that A and B see the same endpoint conditions.  Running
        # all of A then all of B measures time-of-day as much as configuration.
        for v in variants:
            r = safe_chat(v, prompt, v.cfg)
            r.variant = v.name
            r.round_index = i
            r.run_index = counters[v.name]
            r.prompt_id = pid
            counters[v.name] += 1
            raws.append(r)
            m = compute_metrics(r)
            if args.quiet:
                mark = "x" if not m["ok"] else ("~" if m["tokens_estimated"] else ".")
                print(mark, end="", flush=True)
            else:
                tag = f"[{v.name}] " if len(variants) > 1 else ""
                if r.error:
                    print(f"  Round {i+1}: {tag}FAILED -- {r.error}")
                else:
                    print(f"  Round {i+1}: {tag}"
                          f"{fmt(m['decode_tps'])} tok/s decode, "
                          f"TTFT {fmt(m['ttft_ms'], 1)}ms, "
                          f"out {m['output_tokens']}"
                          f"{' (est)' if m['tokens_estimated'] else ''}"
                          f"{' [buffered]' if m['buffered'] else ''}")
    if args.quiet:
        print()
    return raws


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _counts(vals: List[Any]) -> str:
    out: Dict[Any, int] = {}
    for v in vals:
        out[v] = out.get(v, 0) + 1
    return " ".join(f"{k}={v}" for k, v in sorted(out.items(), key=lambda kv: str(kv[0])))


def _pline(label: str, vals: List[float], nd: int = 1, rate: bool = False) -> str:
    """Below PERCENTILE_MIN_N samples the tail percentiles are the tail samples
    relabelled, so print an honest min/max instead of four confident-looking
    numbers derived from a dozen points.  P50/P90 are kept at every n."""
    if not vals:
        return f"  {label:<8} n/a"
    n = len(vals)
    if rate:
        if n < PERCENTILE_MIN_N:
            return (f"  {label:<8} P50 {fmt(pct(vals,50),nd)}  P25 {fmt(pct(vals,25),nd)}  "
                    f"min {fmt(min(vals),nd)} (slow tail)  max {fmt(max(vals),nd)}")
        return (f"  {label:<8} P50 {fmt(pct(vals,50),nd)}  P25 {fmt(pct(vals,25),nd)}  "
                f"P5 {fmt(pct(vals,5),nd)} (slow tail)  min {fmt(min(vals),nd)}")
    if n < PERCENTILE_MIN_N:
        return (f"  {label:<8} P50 {fmt(pct(vals,50),nd)}  P90 {fmt(pct(vals,90),nd)}  "
                f"min {fmt(min(vals),nd)}  max {fmt(max(vals),nd)} (slow tail)")
    return (f"  {label:<8} P50 {fmt(pct(vals,50),nd)}  P90 {fmt(pct(vals,90),nd)}  "
            f"P95 {fmt(pct(vals,95),nd)}  P99 {fmt(pct(vals,99),nd)}")


def summarize_variant(name: str, mets: List[Dict[str, Any]], args) -> Dict[str, Any]:
    ok = [m for m in mets if m["ok"]]
    failed = [m for m in mets if not m["ok"]]
    est = [m for m in ok if m["tokens_estimated"]]
    buffered = [m for m in ok if m["buffered"]]

    print()
    print("=" * 70)
    print(f"VARIANT: {name}" if len(args._variant_names) > 1 else "RESULTS")
    print("=" * 70)
    print(f"Runs: {len(mets)} total | {len(ok)} ok | {len(failed)} failed"
          f" | {len(est)} estimated-tokens | {len(buffered)} buffered-stream")
    if failed:
        errs = _counts([(m["error"] or "zero tokens").split(":")[0] for m in failed])
        print(f"  failure kinds: {errs}")
    if not ok:
        print("\nNo successful runs to analyse.")
        return {"variant": name, "ok": 0}

    toks = [m["output_tokens"] for m in ok]
    ptoks = [m["prompt_tokens"] for m in ok if m["prompt_tokens"] is not None]
    print(f"Output tokens: min {min(toks)}  P50 {fmt(pct(toks,50),0)}  max {max(toks)}")
    spread = (max(toks) / min(toks)) if min(toks) else None
    if spread and spread >= 2.0 and not (args.min_tokens or args.ignore_eos):
        print(f"  WARNING: output length varies {spread:.1f}x across runs. Per-run rate")
        print(f"           is being compared across different-sized workloads, and decode")
        print(f"           rate decays as the KV cache grows. Pin length with --min-tokens")
        print(f"           (vLLM/llama.cpp) or --ignore-eos (vLLM) before A/B-ing models.")
    if ptoks:
        cached = [m["cached_tokens"] for m in ok if m["cached_tokens"] is not None]
        extra = f"  cached P50 {fmt(pct(cached,50),0)}" if cached else ""
        print(f"Prompt tokens: P50 {fmt(pct(ptoks,50),0)}{extra}")
    frs = [m["finish_reason"] for m in ok if m["finish_reason"]]
    if frs:
        print(f"Finish reasons: {_counts(frs)}")

    # --- pooled throughput: the headline number ---------------------------
    dec = [m for m in ok if m["decode_tps"] is not None]
    pooled_decode = None
    if dec:
        num = sum(m["output_tokens"] - 1 for m in dec)
        den = sum(m["decode_ms"] for m in dec) / 1000.0
        pooled_decode = num / den if den > 0 else None
    tot_tok = sum(toks)
    tot_s = sum(m["total_ms"] for m in ok) / 1000.0
    pooled_e2e = tot_tok / tot_s if tot_s > 0 else None

    print()
    print("Throughput (pooled -- sum tokens / sum time, not mean of per-run rates)")
    if pooled_decode is not None:
        print(f"  Decode, excl. TTFT: {fmt(pooled_decode)} tok/s"
              f"   [{sum(m['output_tokens']-1 for m in dec)} tok over "
              f"{sum(m['decode_ms'] for m in dec)/1000.0:.1f}s, {len(dec)} runs]")
    print(f"  End-to-end:         {fmt(pooled_e2e)} tok/s"
          f"   [{tot_tok} tok over {tot_s:.1f}s, {len(ok)} runs]")

    print()
    print("Per-run decode rate (tok/s)")
    print(_pline("rate", [m["decode_tps"] for m in dec], 2, rate=True))
    if len(ok) < PERCENTILE_MIN_N:
        print(f"  (n={len(ok)}: P95/P99 suppressed -- below ~{PERCENTILE_MIN_N} runs they are")
        print("   the extreme samples relabelled, not estimates. min/max shown instead.)")

    print()
    print("Latency (ms)")
    print(_pline("TTFT", [m["ttft_ms"] for m in ok if m["ttft_ms"] is not None]))
    ttfa = [m["ttfa_ms"] for m in ok if m["ttfa_ms"] is not None]
    if any(m["reasoning_chars"] or (m["reasoning_tokens"] or 0) for m in ok) and ttfa:
        print(_pline("TTFA", ttfa))
        print("           ^ time to first ANSWER token; gap vs TTFT is the reasoning pass")
    print(_pline("E2E", [m["total_ms"] for m in ok]))
    icl = [m["icl_p50_ms"] for m in ok if m["icl_p50_ms"] is not None]
    if icl:
        tpc = [m["tokens_per_chunk"] for m in ok if m["tokens_per_chunk"]]
        worst = [m["icl_max_ms"] for m in ok if m["icl_max_ms"] is not None]
        print(f"  ICL      P50-of-run-P50 {fmt(pct(icl,50),2)}  "
              f"worst stall {fmt(max(worst),1) if worst else 'n/a'}"
              f"   ({fmt(pct(tpc,50),2) if tpc else 'n/a'} tokens/chunk)")
        if tpc and pct(tpc, 50) and pct(tpc, 50) > 1.5:
            print("           (chunks carry >1 token; ICL is frame cadence, not token cadence)")

    if est:
        print()
        print(f"  NOTE: {len(est)}/{len(ok)} runs had no server usage block; token counts")
        print("        there are chars/4 estimates. Exclude them before quoting a rate.")
    if buffered:
        print()
        print(f"  NOTE: {len(buffered)}/{len(ok)} runs delivered the whole body in one flush")
        print("        (no real streaming). Excluded from decode-rate stats; TTFT for those")
        print("        runs is effectively end-to-end latency.")

    print_reasoning(ok)
    print_drift(ok, args)
    return {
        "variant": name, "runs": len(mets), "ok": len(ok), "failed": len(failed),
        "estimated": len(est), "buffered": len(buffered),
        "pooled_decode_tps": pooled_decode, "pooled_e2e_tps": pooled_e2e,
        "decode_tps_p50": pct([m["decode_tps"] for m in dec], 50),
        "ttft_p50_ms": pct([m["ttft_ms"] for m in ok if m["ttft_ms"] is not None], 50),
        "ttft_p95_ms": pct([m["ttft_ms"] for m in ok if m["ttft_ms"] is not None], 95),
        "e2e_p50_ms": pct([m["total_ms"] for m in ok], 50),
        "output_tokens_p50": pct(toks, 50),
    }


def print_reasoning(ok: List[Dict[str, Any]]) -> None:
    reasoning = [m for m in ok if m["reasoning_chars"] or (m["reasoning_tokens"] or 0)]
    empty = [m for m in ok if m["empty_answer"]]
    if not reasoning and not empty:
        return
    print()
    print("Reasoning pass:")
    print(f"  Runs with reasoning: {len(reasoning)}/{len(ok)}")
    tok_split = [m for m in reasoning if m["reasoning_tokens"] is not None and m["output_tokens"]]
    if tok_split:
        share = [100.0 * m["reasoning_tokens"] / m["output_tokens"] for m in tok_split]
        print(f"  Reasoning share of output TOKENS: {fmt(statistics.median(share),1)}% (median)")
    elif reasoning:
        share = []
        for m in reasoning:
            tot = m["reasoning_chars"] + m["content_chars"]
            if tot:
                share.append(100.0 * m["reasoning_chars"] / tot)
        if share:
            print(f"  Reasoning share of generated CHARS: {fmt(statistics.median(share),1)}% (median)")
            print("    (server exposed no reasoning_tokens; char share over-weights prose)")
    if empty:
        trunc = [m for m in empty if m["finish_reason"] in ("length", "max_tokens")]
        print(f"  EMPTY ANSWERS: {len(empty)}/{len(ok)} runs billed tokens but returned no content")
        if trunc:
            print(f"    {len(trunc)} hit a length stop -- the reasoning pass ate the whole")
            print("    --max-tokens budget. Raise it or turn thinking off.")


def print_drift(ok: List[Dict[str, Any]], args) -> None:
    """Persistent claude-cli sessions grow context every run, so TTFT drifts up.
    That is a trend, not noise; mean and median both hide it."""
    if args.type != "claude-cli" or args.session_mode != "persistent" or len(ok) < 6:
        return
    ttft = [m["ttft_ms"] for m in sorted(ok, key=lambda m: m["run"]) if m["ttft_ms"]]
    if len(ttft) < 6:
        return
    third = len(ttft) // 3
    early, late = ttft[:third], ttft[-third:]
    e, l = statistics.median(early), statistics.median(late)
    print()
    print("Session drift (persistent mode):")
    print(f"  TTFT median first third {fmt(e,1)}ms -> last third {fmt(l,1)}ms"
          f"  ({(l/e - 1)*100:+.1f}%)" if e else "")
    print("  Growing conversation context, not model variance. Use --session-mode")
    print("  stateless to compare models.")


def print_comparison(summaries: List[Dict[str, Any]], mets: List[Dict[str, Any]]) -> None:
    """Paired deltas. Runs are interleaved round-robin against an identical prompt
    sequence, so round N of A and round N of B saw the same prompt under the same
    endpoint conditions. Compare them pairwise, not as two independent means."""
    if len(summaries) < 2:
        return
    base = summaries[0]["variant"]
    print()
    print("=" * 70)
    print(f"PAIRED COMPARISON (baseline: {base})")
    print("=" * 70)
    by_variant: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for m in mets:
        by_variant.setdefault(m["variant"], {})[m["round"]] = m

    hdr = f"{'variant':<16}{'decode tok/s':>14}{'TTFT p50':>12}{'E2E p50':>12}{'out tok p50':>13}"
    print(hdr)
    for s in summaries:
        if not s.get("ok"):
            print(f"{s['variant']:<16}{'(no data)':>14}")
            continue
        print(f"{s['variant']:<16}{fmt(s['pooled_decode_tps']):>14}"
              f"{fmt(s['ttft_p50_ms'],1):>12}{fmt(s['e2e_p50_ms'],1):>12}"
              f"{fmt(s['output_tokens_p50'],0):>13}")

    for s in summaries[1:]:
        name = s["variant"]
        pairs_ttft, pairs_rate = [], []
        for rnd, mb in by_variant.get(base, {}).items():
            mv = by_variant.get(name, {}).get(rnd)
            if not mv or not mb["ok"] or not mv["ok"]:
                continue
            if mb["ttft_ms"] and mv["ttft_ms"]:
                pairs_ttft.append(mv["ttft_ms"] - mb["ttft_ms"])
            if mb["decode_tps"] and mv["decode_tps"]:
                pairs_rate.append(mv["decode_tps"] - mb["decode_tps"])
        print()
        print(f"  {name} vs {base}  (paired, n={max(len(pairs_ttft), len(pairs_rate))})")
        if pairs_ttft:
            w = sum(1 for d in pairs_ttft if d < 0)
            l = sum(1 for d in pairs_ttft if d > 0)
            print(f"    TTFT delta:   median {statistics.median(pairs_ttft):+.1f}ms"
                  f"   [{w}/{len(pairs_ttft)} rounds faster]{_sign_verdict(w, l)}")
        if pairs_rate:
            w = sum(1 for d in pairs_rate if d > 0)
            l = sum(1 for d in pairs_rate if d < 0)
            print(f"    Decode delta: median {statistics.median(pairs_rate):+.2f} tok/s"
                  f"   [{w}/{len(pairs_rate)} rounds faster]{_sign_verdict(w, l)}")
        if pairs_ttft or pairs_rate:
            print("    Sign test is over paired rounds and tests DIRECTION only: whether a")
            print("    split this lopsided is plausible from a coin flip. It says nothing")
            print("    about the size of the win -- read the median delta for that.")


def check_switch_took_effect(mets: List[Dict[str, Any]], v: Variant) -> None:
    """The dangerous failure mode with reasoning switches is silence: an
    unsupported or misspelled toggle is accepted without error and the reasoning
    pass runs anyway."""
    if v.thinking is None and v.reasoning is None:
        return
    ok = [m for m in mets if m["ok"]]
    if not ok:
        return
    saw = any(m["reasoning_chars"] or (m["reasoning_tokens"] or 0) for m in ok)
    wanted_off = v.thinking == "off" or v.reasoning == "none"
    wanted_on = v.thinking == "on" or (v.reasoning not in (None, "none"))
    if wanted_off and saw:
        asked = "--thinking off" if v.thinking == "off" else "reasoning=none"
        print()
        print(f"WARNING [{v.name}]: {asked} was requested but responses still contain")
        print("         reasoning text. The server accepted the field and ignored it.")
        print("         These are thinking-ON numbers. Do not record them as thinking-OFF.")
    elif wanted_on and not saw:
        print()
        print(f"WARNING [{v.name}]: thinking was requested but no reasoning text came back.")
        print("         Either the model has no reasoning pass, or the server was started")
        print("         without a --reasoning-parser and folded the trace into content.")


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

CSV_COLUMNS = ["variant", "run", "round", "prompt_id", "ok", "error",
               "prompt_tokens", "output_tokens", "reasoning_tokens", "cached_tokens",
               "tokens_estimated", "buffered", "empty_answer", "finish_reason",
               "ttft_ms", "ttfa_ms", "decode_ms", "total_ms", "decode_tps", "e2e_tps",
               "chunks", "tokens_per_chunk", "icl_p50_ms", "icl_p99_ms", "icl_max_ms",
               "content_chars", "reasoning_chars", "spawn_ms", "stream"]


def environment_meta(args, variants: List[Variant]) -> Dict[str, Any]:
    return {
        "tool": "llm_bench.py", "version": VERSION,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(), "platform": platform.platform(),
        "python": sys.version.split()[0],
        "api_type": args.type, "base_url": args.base_url, "model": args.model,
        "runs": args.runs, "warmup": args.warmup, "seed": args.seed,
        "prompt_mode": args.prompt_mode, "stream": not args.nostream,
        "max_tokens": args.max_tokens, "min_tokens": args.min_tokens,
        "ignore_eos": args.ignore_eos, "temperature": args.temperature,
        "top_p": args.top_p, "top_k": args.top_k,
        "session_mode": args.session_mode if args.type == "claude-cli" else None,
        "variants": {v.name: v.overrides for v in variants},
        "argv": sys.argv[1:],
    }


def write_results(args, meta, summaries, mets) -> None:
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"meta": meta, "summary": summaries, "runs": mets}, f, indent=2)
        print(f"\nWrote {args.json}")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            w.writeheader()
            for m in mets:
                w.writerow(m)
        print(f"Wrote {args.csv}")


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def _as_bool(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def build_variant(args, v: Variant) -> Variant:
    o = v.overrides
    v.model = o.get("model", args.model)
    v.thinking = o.get("thinking", args.thinking)
    v.reasoning = o.get("reasoning", args.reasoning)
    if v.reasoning == "off":
        v.reasoning = "none"

    extra: Dict[str, Any] = {}
    effort = o.get("effort", args.effort)
    if args.type == "claude-cli":
        if effort:
            extra["effort"] = effort
        extra["session_mode"] = args.session_mode
        extra["timeout"] = args.timeout
    if v.thinking is not None:
        extra["chat_template_kwargs"] = {"enable_thinking": v.thinking == "on"}
    if v.reasoning is not None:
        extra["reasoning_effort"] = v.reasoning

    v.cfg = RunConfig(
        max_tokens=int(o.get("max_tokens", args.max_tokens)),
        min_tokens=(int(o["min_tokens"]) if "min_tokens" in o
                    else args.min_tokens),
        temperature=(float(o["temperature"]) if "temperature" in o
                     else args.temperature),
        top_p=(float(o["top_p"]) if "top_p" in o else args.top_p),
        top_k=(int(o["top_k"]) if "top_k" in o else args.top_k),
        ignore_eos=(_as_bool(o["ignore_eos"]) if "ignore_eos" in o else args.ignore_eos),
        stream=not args.nostream,
        debug=args.debug,
    )
    v.client = create_client(args.type, args.base_url, args.api_key,
                             v.model or "placeholder", extra or None,
                             insecure=args.insecure)
    return v


EPILOG = """
Examples:
  # local vLLM / llama.cpp, controlled output length, 30 runs
  %(prog)s --base-url http://localhost:8000/v1 --model qwen3 \\
           --min-tokens 512 --max-tokens 512 --runs 30

  # paired A/B of the reasoning pass -- interleaved, same prompts, same conditions
  %(prog)s --variant think_off:thinking=off --variant think_on:thinking=on --runs 40

  # two models head to head on the same server
  %(prog)s --variant a:model=qwen3-30b --variant b:model=gemma4-27b --runs 40 --json ab.json

  # Anthropic
  %(prog)s --type anthropic --model claude-sonnet-4-5 --runs 30

  # Claude Code CLI, one fresh session per run (comparable to a stateless call)
  %(prog)s --type claude-cli --model opus --session-mode stateless --runs 10

Reading the output:
  Pooled decode tok/s is the headline throughput number. Per-run P5 is the SLOW
  tail for a rate. TTFT P95 is the slow tail for a latency. A model at 150 tok/s
  behind a 2s first token loses to one at 90 tok/s with a 150ms TTFT for any
  control-plane use -- read both columns.
"""


def main() -> int:
    p = argparse.ArgumentParser(
        description="Benchmark LLM endpoints (OpenAI-compatible, Anthropic, Gemini, claude CLI)",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=EPILOG)
    p.add_argument("--type", "-t", default="openai",
                   choices=["openai", "anthropic", "gemini", "claude-cli"])
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--prompt", default="Write a detailed explanation of how transformers "
                                       "work in neural networks.")
    p.add_argument("--prompt-mode", choices=["repeated", "corpus", "shuffled"],
                   default="repeated",
                   help="repeated: one prompt every run (prefix cache stays hot -- this "
                        "measures the cached path). corpus: fixed corpus in fixed order "
                        "(uncached, reproducible). shuffled: corpus shuffled under --seed")
    p.add_argument("--seed", type=int, default=1337,
                   help="Seed for --prompt-mode shuffled (default: 1337, so A/B is repeatable)")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--min-tokens", type=int, default=None,
                   help="Floor on generated tokens (vLLM/llama.cpp extra_body). Use with "
                        "--max-tokens set equal to it to pin the workload for model A/B")
    p.add_argument("--ignore-eos", action="store_true",
                   help="vLLM only: generate exactly --max-tokens, ignoring EOS")
    p.add_argument("--runs", type=int, default=12,
                   help="Default 12: cheap enough to run without thinking, and enough "
                        "paired rounds for --variant to be callable by sign test "
                        "(10/12 is p=0.04). Raise to >=20 to unlock P95/P99 rows")
    p.add_argument("--warmup", type=int, default=1,
                   help="Default 1, which is right for an already-hot server. A freshly "
                        "started vLLM is the case where it is not: cudagraph capture and "
                        "torch.compile can make the first few requests wildly slow, and "
                        "one warmup may not absorb that. Raise it, or bench a warm server")
    p.add_argument("--temperature", type=float, default=None,
                   help="Not sent unless given, so the server/model default applies")
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--quiet", "-q", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--nostream", action="store_true")
    p.add_argument("--insecure", "-k", action="store_true")
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default=None,
                   help="claude-cli only")
    p.add_argument("--session-mode", choices=["stateless", "persistent"], default="stateless",
                   help="claude-cli only. stateless: fresh process per run (spawn cost "
                        "measured separately and excluded). persistent: one growing "
                        "conversation -- run N is turn N, context grows")
    p.add_argument("--timeout", type=float, default=600, help="claude-cli per-run timeout")
    p.add_argument("--thinking", nargs="?", const="on", choices=["on", "off"], default=None,
                   help="openai type only. Sends chat_template_kwargs={'enable_thinking':bool}")
    p.add_argument("--reasoning", choices=["off", "none", "low", "medium", "high"], default=None,
                   help="openai type only. Sets reasoning_effort. On vLLM this is not an "
                        "independent control: none maps to the same chat-template switch as "
                        "--thinking off, and the qwen/gemma parsers only distinguish none "
                        "from not-none")
    p.add_argument("--variant", action="append", default=None, metavar="NAME:k=v,k=v",
                   help="Add a config variant, run interleaved with the others against an "
                        "identical prompt sequence and compared pairwise. Keys: "
                        + ", ".join(sorted(VARIANT_KEYS)))
    p.add_argument("--json", default=None, help="Write per-run results + metadata to JSON")
    p.add_argument("--csv", default=None, help="Write per-run results to CSV")

    args = p.parse_args()

    if args.reasoning == "off":
        args.reasoning = "none"
    if (args.thinking is not None or args.reasoning is not None) and args.type != "openai":
        print(f"Error: --thinking/--reasoning are only wired for --type openai "
              f"(got --type {args.type}).")
        if args.type == "claude-cli":
            print("       Use --effort for claude-cli reasoning effort.")
        return 2
    if args.effort and args.type != "claude-cli":
        print("Error: --effort is claude-cli only. Use --reasoning for openai endpoints.")
        return 2
    if args.runs < 1:
        print("Error: --runs must be >= 1")
        return 2

    if args.base_url:
        # Strip BEFORE testing the suffix. The old order turned a trailing-slash
        # URL into .../v1/v1.
        args.base_url = args.base_url.rstrip("/")
        if args.type == "openai" and not args.base_url.endswith("/v1"):
            args.base_url += "/v1"
    elif args.type == "openai":
        args.base_url = "http://localhost:8000/v1"

    if args.api_key is None:
        if args.type == "openai":
            args.api_key = os.environ.get("OPENAI_API_KEY", "not-needed")
        elif args.type == "anthropic":
            args.api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not args.api_key:
                print("Error: ANTHROPIC_API_KEY or --api-key required")
                return 2
        elif args.type == "gemini":
            args.api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not args.api_key:
                print("Error: GOOGLE_API_KEY / GEMINI_API_KEY or --api-key required")
                return 2
    return run(args, p)


def run(args, parser) -> int:
    try:
        specs = args.variant or ["default"]
        variants = [parse_variant(s) for s in specs]
    except ValueError as e:
        print(f"Error: {e}")
        return 2
    names = [v.name for v in variants]
    if len(set(names)) != len(names):
        print("Error: variant names must be unique")
        return 2
    args._variant_names = names

    # --list-models / model auto-detect, using a throwaway client.
    if args.list_models or (args.model is None and args.type != "claude-cli"
                            and not any("model" in v.overrides for v in variants)):
        try:
            probe = create_client(args.type, args.base_url, args.api_key,
                                  args.model or "placeholder", None, insecure=args.insecure)
        except ImportError as e:
            print(f"Error: missing dependency for {args.type}: {e}")
            return 2
        try:
            models = probe.list_models()
        except Exception as e:
            models = []
            if args.list_models:
                print(f"Failed to list models: {e}")
                probe.close()
                return 1
        probe.close()
        if args.list_models:
            if not models:
                print("No models reported by this endpoint.")
            for m in models:
                print(m)
            return 0
        if models:
            args.model = models[0]
            if len(models) > 1:
                print(f"Auto-selected model: {args.model}  "
                      f"({len(models)} available -- pass --model to be explicit)")
            else:
                print(f"Auto-selected model: {args.model}")
        else:
            print("Error: --model is required (could not auto-detect)")
            return 2

    try:
        variants = [build_variant(args, v) for v in variants]
    except ImportError as e:
        print(f"Error: missing dependency for {args.type}: {e}")
        return 2
    except Exception as e:
        print(f"Error building client: {type(e).__name__}: {e}")
        return 2

    print(f"Endpoint:  {args.base_url or args.type + ' API'}")
    print(f"API type:  {args.type}")
    print(f"Model:     {args.model or '(backend default)'}")
    if args.type == "claude-cli":
        print(f"Session:   {args.session_mode}   effort: {args.effort or '(default)'}")
    print(f"Streaming: {not args.nostream}")
    print(f"Workload:  max_tokens={args.max_tokens}"
          f"{', min_tokens=' + str(args.min_tokens) if args.min_tokens else ''}"
          f"{', ignore_eos' if args.ignore_eos else ''}"
          f"   prompts={args.prompt_mode}"
          f"{' seed=' + str(args.seed) if args.prompt_mode == 'shuffled' else ''}")
    if len(variants) > 1:
        for v in variants:
            print(f"  variant {v.name}: {v.overrides or '(base config)'}")
    if args.type == "openai" and (args.thinking or args.reasoning):
        sw = []
        if args.thinking:
            sw.append(f"enable_thinking={args.thinking == 'on'}")
        if args.reasoning:
            sw.append(f"reasoning_effort={args.reasoning}")
        print(f"Reasoning switches: {', '.join(sw)}")
    if args.runs < PERCENTILE_MIN_N:
        print(f"NOTE: --runs {args.runs} is a smoke test. Use >= {PERCENTILE_MIN_N} "
              f"(50+ for hosted APIs) before making a model-selection decision.")
    print()

    rc = 0
    try:
        raws = run_benchmark(args, variants)
        mets = [compute_metrics(r) for r in raws]
        summaries = []
        for v in variants:
            vm = [m for m in mets if m["variant"] == v.name]
            summaries.append(summarize_variant(v.name, vm, args))
            check_switch_took_effect(vm, v)
        print_comparison(summaries, mets)
        write_results(args, environment_meta(args, variants), summaries, mets)
        if not any(s.get("ok") for s in summaries):
            rc = 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        rc = 130
    finally:
        for v in variants:
            try:
                if v.client:
                    v.client.close()
            except Exception:
                pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
