#!/usr/bin/env python3
"""
Unified API benchmark tool supporting OpenAI, Anthropic, and Gemini APIs.
Measures tokens/sec, TTFT, and inter-token latency.
"""

import argparse
import time
import statistics
import random
import queue
from abc import ABC, abstractmethod

# Diverse prompts to avoid cache duplication issues
BENCHMARK_PROMPTS = [
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


class APIClient(ABC):
    """Abstract base class for API clients."""

    @abstractmethod
    def chat(self, prompt: str, max_tokens: int, temperature: float, top_p: float,
             top_k: int = None, stream: bool = True, debug: bool = False) -> dict:
        """Send a chat request and return metrics."""
        pass

    @abstractmethod
    def list_models(self) -> list:
        """List available models."""
        pass


class OpenAIClient(APIClient):
    """OpenAI-compatible API client."""

    def __init__(self, base_url: str, api_key: str, model: str, extra_params: dict = None, insecure: bool = False):
        from openai import OpenAI
        import httpx
        http_client = httpx.Client(verify=not insecure) if insecure else None
        self.client = OpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
        self.model = model
        self.extra_params = extra_params or {}

    def list_models(self) -> list:
        return [m.id for m in self.client.models.list().data]

    def chat(self, prompt: str, max_tokens: int, temperature: float, top_p: float,
             top_k: int = None, stream: bool = True, debug: bool = False) -> dict:
        start_time = time.perf_counter()
        first_token_time = None
        token_times = []
        tokens = 0
        all_content = []
        # Reasoning and answer text are counted separately so a run can show how much
        # of the generation the reasoning pass consumed, and so a --thinking/--reasoning
        # switch that silently did nothing is detectable afterwards.
        reasoning_chars = 0
        content_chars = 0
        finish_reason = None
        # Time to the first token of the actual ANSWER, as opposed to TTFT which is the
        # first token of any kind. With a reasoning pass running these diverge hard: TTFT
        # stays flat while the answer waits behind the whole trace. This is the metric
        # that actually moves when thinking is toggled -- tokens/sec does not.
        first_content_time = None

        extra_body = {}
        if top_k is not None:
            extra_body["top_k"] = top_k
        for k, v in self.extra_params.items():
            if v is not None:
                extra_body[k] = v

        # Only send sampling params the caller actually asked for. Sending them
        # unconditionally overrides the server's own defaults -- for vLLM that means
        # silently discarding the model's generation_config.json (Gemma 4 ships
        # temperature=1.0/top_p=0.95/top_k=64), which changes output distribution and,
        # with speculative decoding, the draft acceptance rate. Omitted => server default.
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            stream=stream,
            **({"temperature": temperature} if temperature is not None else {}),
            **({"top_p": top_p} if top_p is not None else {}),
            **({"stream_options": {"include_usage": True}} if stream else {}),
            **({"extra_body": extra_body} if extra_body else {})
        )

        if stream:
            usage_tokens = None
            for chunk in response:
                delta = chunk.choices[0].delta if chunk.choices else None
                content_piece = None
                reasoning_piece = None
                if delta:
                    extra = getattr(delta, "model_extra", None) or {}
                    content_piece = delta.content
                    reasoning_piece = (
                        getattr(delta, "reasoning_content", None)
                        or extra.get("reasoning_content")
                        or extra.get("reasoning")
                    )
                if chunk.choices and chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
                if debug:
                    print(f"  DEBUG chunk: choices={len(chunk.choices) if chunk.choices else 0}, "
                          f"finish={chunk.choices[0].finish_reason if chunk.choices else None}, "
                          f"content={content_piece!r}, reasoning={reasoning_piece!r}")
                piece = content_piece or reasoning_piece
                if piece:
                    all_content.append(piece)
                    content_chars += len(content_piece or "")
                    reasoning_chars += len(reasoning_piece or "")
                    now = time.perf_counter()
                    if content_piece and first_content_time is None:
                        first_content_time = now
                    if first_token_time is None:
                        first_token_time = now
                    else:
                        token_times.append(now)
                    tokens += 1
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage_tokens = chunk.usage.completion_tokens
            if usage_tokens is not None:
                tokens = usage_tokens
        else:
            if response.choices:
                message = response.choices[0].message
                finish_reason = response.choices[0].finish_reason
                extra = getattr(message, "model_extra", None) or {}
                reasoning = (
                    getattr(message, "reasoning_content", None)
                    or extra.get("reasoning_content")
                    or extra.get("reasoning")
                    or ""
                )
                content = message.content or ""
                content_chars = len(content)
                reasoning_chars = len(reasoning)
                if content or reasoning:
                    first_token_time = time.perf_counter()
                    all_content.append(content)
                    # Count the tokens the server actually billed even when content is
                    # empty -- a thinking pass that ate the whole max_tokens budget costs
                    # real time, and reporting it as a zero-token failure hides that.
                    if hasattr(response, 'usage') and response.usage:
                        tokens = response.usage.completion_tokens
                    else:
                        tokens = (len(content) + len(reasoning)) // 4
                if debug:
                    print(f"  DEBUG non-stream: tokens={tokens}, finish={finish_reason}, "
                          f"reasoning_chars={reasoning_chars}, content={content[:100]!r}...")

        end_time = time.perf_counter()
        return self._compute_metrics(start_time, end_time, first_token_time, tokens, all_content, stream,
                                     reasoning_chars=reasoning_chars, content_chars=content_chars,
                                     finish_reason=finish_reason, first_content_time=first_content_time)

    def _compute_metrics(self, start_time, end_time, first_token_time, tokens, all_content, stream,
                         reasoning_chars=0, content_chars=0, finish_reason=None, first_content_time=None):
        total_time = end_time - start_time
        if stream:
            ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
            generation_time = end_time - first_token_time if first_token_time else total_time
        else:
            ttft = total_time * 1000
            generation_time = total_time

        if tokens > 0 and generation_time > 0:
            itl_ms = (generation_time * 1000) / tokens
        else:
            itl_ms = 0

        tps = tokens / generation_time if generation_time > 0 else 0

        return {
            "tokens": tokens,
            "ttft_ms": ttft,
            "itl_ms": itl_ms,
            "total_time": total_time,
            "generation_time": generation_time,
            "tokens_per_sec": tps,
            "reasoning_chars": reasoning_chars,
            "content_chars": content_chars,
            "finish_reason": finish_reason,
            "ttfct_ms": (first_content_time - start_time) * 1000 if first_content_time else None,
        }


class AnthropicClient(APIClient):
    """Anthropic API client."""

    def __init__(self, base_url: str, api_key: str, model: str, extra_params: dict = None, insecure: bool = False):
        import anthropic
        import httpx
        http_client = httpx.Client(verify=not insecure) if insecure else None
        if base_url:
            self.client = anthropic.Anthropic(base_url=base_url, api_key=api_key, http_client=http_client)
        else:
            self.client = anthropic.Anthropic(api_key=api_key, http_client=http_client)
        self.model = model
        self.extra_params = extra_params or {}

    def list_models(self) -> list:
        # Anthropic doesn't have a models endpoint, return common models
        return ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307",
                "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"]

    def chat(self, prompt: str, max_tokens: int, temperature: float, top_p: float,
             top_k: int = None, stream: bool = True, debug: bool = False) -> dict:
        start_time = time.perf_counter()
        first_token_time = None
        tokens = 0
        all_content = []

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["top_k"] = top_k

        if stream:
            last_token_time = None
            chunk_count = 0
            with self.client.messages.stream(**kwargs) as response:
                for text in response.text_stream:
                    if debug:
                        print(f"  DEBUG chunk: {text!r}")
                    if text:
                        all_content.append(text)
                        now = time.perf_counter()
                        if first_token_time is None:
                            first_token_time = now
                        last_token_time = now
                        chunk_count += 1
                # Get actual token count from final message
                final_message = response.get_final_message()
                if debug:
                    print(f"  DEBUG final_message.usage: {final_message.usage if final_message else None}")
                content_len = len("".join(all_content))
                estimated_tokens = max(1, content_len // 4)
                if final_message and final_message.usage and final_message.usage.output_tokens > 1:
                    tokens = final_message.usage.output_tokens
                else:
                    # Fallback: estimate from content length
                    tokens = estimated_tokens
            # Use last token time as end time for streaming
            if last_token_time:
                end_time = last_token_time
        else:
            response = self.client.messages.create(**kwargs)
            first_token_time = time.perf_counter()
            if response.content:
                for block in response.content:
                    if hasattr(block, 'text'):
                        all_content.append(block.text)
            if debug:
                print(f"  DEBUG response.usage: {response.usage}")
            content_len = len("".join(all_content))
            estimated_tokens = max(1, content_len // 4)
            if response.usage and response.usage.output_tokens and response.usage.output_tokens > 1:
                tokens = response.usage.output_tokens
            else:
                # Fallback: estimate from content length (~4 chars per token)
                # Server returned output_tokens=1 or 0 which is clearly wrong
                tokens = estimated_tokens
            if debug:
                print(f"  DEBUG non-stream: tokens={tokens}, content={''.join(all_content)[:100]!r}...")
            end_time = time.perf_counter()

        total_time = end_time - start_time

        if stream:
            ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
            generation_time = end_time - first_token_time if first_token_time else total_time
            # If generation_time is very small but we have tokens, server is buffering
            # (fake streaming) - use total_time instead
            if generation_time < 0.01 and tokens > 1:
                generation_time = total_time
        else:
            ttft = total_time * 1000
            generation_time = total_time

        if tokens > 0 and generation_time > 0:
            itl_ms = (generation_time * 1000) / tokens
        else:
            itl_ms = 0

        tps = tokens / generation_time if generation_time > 0 else 0

        return {
            "tokens": tokens,
            "ttft_ms": ttft,
            "itl_ms": itl_ms,
            "total_time": total_time,
            "generation_time": generation_time,
            "tokens_per_sec": tps,
        }


class GeminiClient(APIClient):
    """Google Gemini API client."""

    def __init__(self, base_url: str, api_key: str, model: str, extra_params: dict = None, insecure: bool = False):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(model)
        self.extra_params = extra_params or {}
        # Note: Gemini SDK doesn't support custom SSL settings easily

    def list_models(self) -> list:
        import google.generativeai as genai
        return [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]

    def chat(self, prompt: str, max_tokens: int, temperature: float, top_p: float,
             top_k: int = None, stream: bool = True, debug: bool = False) -> dict:
        import google.generativeai as genai

        start_time = time.perf_counter()
        first_token_time = None
        tokens = 0
        all_content = []

        gen_config = genai.GenerationConfig(
            max_output_tokens=max_tokens,
            **({"temperature": temperature} if temperature is not None else {}),
            **({"top_p": top_p} if top_p is not None else {}),
        )
        if top_k is not None:
            gen_config.top_k = top_k

        if stream:
            response = self.model.generate_content(prompt, generation_config=gen_config, stream=True)
            for chunk in response:
                if debug:
                    text = chunk.text if hasattr(chunk, 'text') else str(chunk)
                    print(f"  DEBUG chunk: {text!r}")
                if hasattr(chunk, 'text') and chunk.text:
                    all_content.append(chunk.text)
                    now = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = now
                    tokens += 1
            # Try to get actual token count
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                tokens = response.usage_metadata.candidates_token_count or tokens
        else:
            response = self.model.generate_content(prompt, generation_config=gen_config, stream=False)
            first_token_time = time.perf_counter()
            if response.text:
                all_content.append(response.text)
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                tokens = response.usage_metadata.candidates_token_count
            else:
                tokens = len(response.text) // 4 if response.text else 0
            if debug:
                content_preview = response.text[:100] if response.text else ''
                print(f"  DEBUG non-stream: tokens={tokens}, content={content_preview!r}")

        end_time = time.perf_counter()
        total_time = end_time - start_time

        if stream:
            ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
            generation_time = end_time - first_token_time if first_token_time else total_time
        else:
            ttft = total_time * 1000
            generation_time = total_time

        if tokens > 0 and generation_time > 0:
            itl_ms = (generation_time * 1000) / tokens
        else:
            itl_ms = 0

        tps = tokens / generation_time if generation_time > 0 else 0

        return {
            "tokens": tokens,
            "ttft_ms": ttft,
            "itl_ms": itl_ms,
            "total_time": total_time,
            "generation_time": generation_time,
            "tokens_per_sec": tps,
        }


class ClaudeCLIClient(APIClient):
    """Drives the actual `claude` CLI (Claude Code) directly via adapters/claude.py --
    a real --print session with full Claude Code behavior (CLAUDE.md, hooks, memory,
    tools all load normally), not an HTTP API. `model` is passed through as
    `claude --model <value>` only when explicitly set -- None means claude uses
    whatever model is currently configured as its default.

    Spawning `claude` is expensive (several seconds to load hooks and connect all
    configured MCP servers, on top of the actual model call) -- that cost is unrelated
    to model speed and would swamp TTFT/tokens-per-sec if paid on every run. So a
    single session is spawned lazily on the first chat() call and reused for every
    subsequent run (mirroring ShepherdProcess's subprocess-reuse in test_model.py):
    only run 1 pays the cold start, runs 2..N measure real per-turn latency. This
    also means later runs are additional turns in one growing conversation, not
    independent stateless completions -- same tradeoff ShepherdProcess already makes.
    Call close() when done with the client to terminate the session.
    """

    def __init__(self, base_url: str, api_key: str, model: str, extra_params: dict = None, insecure: bool = False):
        from adapters.claude import ClaudeAdapter
        self._ClaudeAdapter = ClaudeAdapter
        self.model = model if model and model != "placeholder" else None
        # Reasoning effort (low/medium/high/xhigh/max) -- never defaulted, only passed
        # through as `claude --effort <value>` when the caller explicitly set one.
        self.effort = (extra_params or {}).get("effort")
        self._adapter = None
        self._events = None

    def list_models(self) -> list:
        # No fixed model list -- claude picks its own default when --model is omitted.
        return []

    def _ensure_session(self, max_tokens: int):
        """Spawn the persistent claude session on first use. max_tokens/effort are
        session-wide (set via env var / CLI flag at spawn time), so only the first
        call's value takes effect -- every caller in this script passes the same
        --max-tokens for the whole run anyway.
        """
        if self._adapter is not None and self._adapter.alive:
            return
        events = queue.Queue()
        params = {"max_output_tokens": max_tokens} if max_tokens else {}
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

    def close(self):
        if self._adapter is not None:
            self._adapter.kill()
            self._adapter = None
            self._events = None

    def chat(self, prompt: str, max_tokens: int, temperature: float, top_p: float,
             top_k: int = None, stream: bool = True, debug: bool = False) -> dict:
        # temperature/top_p/top_k have no equivalent in `claude --print` and are ignored.
        self._ensure_session(max_tokens)

        start_time = time.perf_counter()
        self._adapter.send(prompt)

        first_token_time = None
        chunk_count = 0
        end_time = None
        result = {}
        while True:
            try:
                kind, payload, ts = self._events.get(timeout=600)
            except queue.Empty:
                break
            if kind == "delta":
                if debug:
                    print(f"  DEBUG chunk: {payload!r}")
                if first_token_time is None:
                    first_token_time = ts
                chunk_count += 1
                end_time = ts
            elif kind == "done":
                result = payload
                end_time = ts
                break
            elif kind == "exit":
                # Session died mid-run -- next call respawns a fresh one.
                self._adapter = None
                self._events = None
                end_time = ts
                break

        if end_time is None:
            end_time = time.perf_counter()

        usage = result.get("usage", {})
        # claude flags is_error=true whenever a turn ends by hitting
        # CLAUDE_CODE_MAX_OUTPUT_TOKENS -- including ordinary truncation after
        # substantial real content already streamed (chunk_count > 0). Only treat
        # a run as failed (0 tokens) when NO text ever streamed at all (e.g.
        # thinking tokens alone exhausted the budget before any reply began).
        tokens = (usage.get("output_tokens") or chunk_count) if chunk_count > 0 else 0

        total_time = end_time - start_time
        if stream:
            ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
            generation_time = end_time - first_token_time if first_token_time else total_time
        else:
            ttft = total_time * 1000
            generation_time = total_time

        if tokens > 0 and generation_time > 0:
            itl_ms = (generation_time * 1000) / tokens
        else:
            itl_ms = 0

        tps = tokens / generation_time if generation_time > 0 else 0

        return {
            "tokens": tokens,
            "ttft_ms": ttft,
            "itl_ms": itl_ms,
            "total_time": total_time,
            "generation_time": generation_time,
            "tokens_per_sec": tps,
        }


def create_client(api_type: str, base_url: str, api_key: str, model: str, extra_params: dict = None, insecure: bool = False) -> APIClient:
    """Factory function to create the appropriate client."""
    if api_type == "openai":
        return OpenAIClient(base_url, api_key, model, extra_params, insecure=insecure)
    elif api_type == "anthropic":
        return AnthropicClient(base_url, api_key, model, extra_params, insecure=insecure)
    elif api_type == "gemini":
        return GeminiClient(base_url, api_key, model, extra_params, insecure=insecure)
    elif api_type == "claude-cli":
        return ClaudeCLIClient(base_url, api_key, model, extra_params, insecure=insecure)
    else:
        raise ValueError(f"Unknown API type: {api_type}")


def run_benchmark(client: APIClient, prompt: str, max_tokens: int, runs: int, warmup: int,
                  temperature: float, top_p: float, top_k: int = None,
                  unique_prompts: bool = False, quiet: bool = False, debug: bool = False,
                  stream: bool = True) -> list:
    """Run multiple benchmark iterations."""

    # Prepare prompts
    if unique_prompts:
        prompts = BENCHMARK_PROMPTS.copy()
        random.shuffle(prompts)
        while len(prompts) < warmup + runs:
            prompts.extend(BENCHMARK_PROMPTS)
    else:
        prompts = [prompt] * (warmup + runs)

    prompt_idx = 0

    # Warmup runs
    if warmup > 0:
        print(f"Warming up ({warmup} runs)...")
        for _ in range(warmup):
            client.chat(prompts[prompt_idx], max_tokens, temperature, top_p, top_k=top_k, stream=stream, debug=debug)
            prompt_idx += 1

    # Benchmark runs
    if quiet:
        print(f"Running benchmark ({runs} runs)...", end="", flush=True)
    else:
        print(f"Running benchmark ({runs} runs)...")

    results = []
    failed_count = 0
    for i in range(runs):
        result = client.chat(prompts[prompt_idx], max_tokens, temperature, top_p, top_k=top_k, stream=stream, debug=debug)
        prompt_idx += 1
        results.append(result)
        if quiet:
            if result['tokens'] == 0:
                failed_count += 1
                print("x", end="", flush=True)
            elif (i + 1) % 10 == 0:
                print(".", end="", flush=True)
        else:
            if stream:
                print(f"  Run {i+1}: {result['tokens_per_sec']:.2f} t/s, "
                      f"TTFT: {result['ttft_ms']:.1f}ms, "
                      f"ITL: {result['itl_ms']:.2f}ms, "
                      f"tokens: {result['tokens']}")
            else:
                print(f"  Run {i+1}: {result['tokens_per_sec']:.2f} t/s, "
                      f"total: {result['total_time']*1000:.1f}ms, "
                      f"tokens: {result['tokens']}")
    if quiet:
        print(f" done ({failed_count} failed)")

    return results


def print_summary(results: list, stream: bool = True):
    """Print summary statistics."""
    successful = [r for r in results if r["tokens"] > 0]
    failed = [r for r in results if r["tokens"] == 0]

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total runs: {len(results)}")
    print(f"Successful:  {len(successful)} ({100*len(successful)/len(results):.1f}%)")
    print(f"Failed:      {len(failed)} ({100*len(failed)/len(results):.1f}%)")

    if not successful:
        print("\nNo successful runs to analyze!")
        print("=" * 60)
        return

    tps_values = [r["tokens_per_sec"] for r in successful]
    total_time_values = [r["total_time"] * 1000 for r in successful]
    token_counts = [r["tokens"] for r in successful]

    print(f"\nAvg tokens generated: {statistics.mean(token_counts):.1f}")
    print()
    print(f"Tokens/sec:")
    print(f"  Mean:   {statistics.mean(tps_values):.2f}")
    print(f"  Median: {statistics.median(tps_values):.2f}")
    if len(tps_values) > 1:
        print(f"  StdDev: {statistics.stdev(tps_values):.2f}")
        print(f"  P5:     {sorted(tps_values)[int(len(tps_values)*0.05)]:.2f}")
        print(f"  P95:    {sorted(tps_values)[int(len(tps_values)*0.95)]:.2f}")

    if stream:
        ttft_values = [r["ttft_ms"] for r in successful]
        itl_values = [r["itl_ms"] for r in successful if r["itl_ms"] > 0]

        print()
        print(f"Time to First Token (ms):")
        print(f"  Mean:   {statistics.mean(ttft_values):.1f}")
        print(f"  Median: {statistics.median(ttft_values):.1f}")
        if len(ttft_values) > 1:
            print(f"  P5:     {sorted(ttft_values)[int(len(ttft_values)*0.05)]:.1f}")
            print(f"  P95:    {sorted(ttft_values)[int(len(ttft_values)*0.95)]:.1f}")

        # Time to the first token of the ANSWER. Printed whenever a reasoning pass ran,
        # because that is when it diverges from TTFT -- the answer is stuck behind the
        # whole trace while TTFT stays flat. Tokens/sec does not move across the toggle;
        # this does, so it is the number to compare between --thinking on and off.
        ttfct_values = [r["ttfct_ms"] for r in successful if r.get("ttfct_ms")]
        if any(r.get("reasoning_chars", 0) > 0 for r in successful) and ttfct_values:
            print()
            print(f"Time to First ANSWER Token (ms):   [vs TTFT above -- gap is the reasoning pass]")
            print(f"  Mean:   {statistics.mean(ttfct_values):.1f}")
            print(f"  Median: {statistics.median(ttfct_values):.1f}")
            if len(ttfct_values) > 1:
                print(f"  P5:     {sorted(ttfct_values)[int(len(ttfct_values)*0.05)]:.1f}")
                print(f"  P95:    {sorted(ttfct_values)[int(len(ttfct_values)*0.95)]:.1f}")

        if itl_values:
            print()
            print(f"Inter-Token Latency (ms):")
            print(f"  Mean:   {statistics.mean(itl_values):.2f}")
            print(f"  Median: {statistics.median(itl_values):.2f}")
    else:
        print()
        print(f"Total Time (ms):")
        print(f"  Mean:   {statistics.mean(total_time_values):.1f}")
        print(f"  Median: {statistics.median(total_time_values):.1f}")
        if len(total_time_values) > 1:
            print(f"  P5:     {sorted(total_time_values)[int(len(total_time_values)*0.05)]:.1f}")
            print(f"  P95:    {sorted(total_time_values)[int(len(total_time_values)*0.95)]:.1f}")

    print_reasoning_summary(successful)

    print("=" * 60)


def print_reasoning_summary(successful: list):
    """Reasoning-pass breakdown. Only printed when a reasoning pass actually ran or
    when a run came back with an empty answer -- ordinary non-reasoning models keep
    the original output untouched.
    """
    reasoning_runs = [r for r in successful if r.get("reasoning_chars", 0) > 0]
    empty_answer = [r for r in successful if r.get("content_chars") == 0]
    if not reasoning_runs and not empty_answer:
        return

    print()
    print("Reasoning pass:")
    print(f"  Runs with reasoning: {len(reasoning_runs)}/{len(successful)}")

    if reasoning_runs:
        shares = []
        for r in reasoning_runs:
            total_chars = r.get("reasoning_chars", 0) + r.get("content_chars", 0)
            if total_chars:
                shares.append(100 * r["reasoning_chars"] / total_chars)
        print(f"  Mean reasoning chars: {statistics.mean([r['reasoning_chars'] for r in reasoning_runs]):.0f}")
        print(f"  Mean answer chars:    {statistics.mean([r['content_chars'] for r in reasoning_runs]):.0f}")
        if shares:
            print(f"  Reasoning share of generated text: {statistics.mean(shares):.1f}%")

    if empty_answer:
        truncated = [r for r in empty_answer if r.get("finish_reason") == "length"]
        print(f"  EMPTY ANSWERS: {len(empty_answer)}/{len(successful)} runs billed tokens but "
              f"returned no content")
        if truncated:
            print(f"    {len(truncated)} hit finish_reason=length -- the reasoning pass ate the "
                  f"whole --max-tokens budget. Raise --max-tokens or turn thinking off.")


def check_switch_took_effect(results: list, thinking: str, reasoning: str):
    """The dangerous failure mode with these switches is silence: an unsupported or
    misspelled toggle is accepted without error and the reasoning pass runs anyway.
    Compare what was asked for against what the responses actually contained.
    """
    if thinking is None and reasoning is None:
        return
    successful = [r for r in results if r["tokens"] > 0]
    if not successful:
        return

    saw_reasoning = any(r.get("reasoning_chars", 0) > 0 for r in successful)
    wanted_off = thinking == "off" or reasoning == "none"
    wanted_on = thinking == "on" or (reasoning is not None and reasoning != "none")

    if wanted_off and saw_reasoning:
        asked = "--thinking off" if thinking == "off" else "--reasoning none"
        print()
        print(f"WARNING: {asked} was requested but the responses still contain reasoning text.")
        print("         The server accepted the field and ignored it. These numbers are")
        print("         thinking-ON numbers -- do not record them as a thinking-OFF result.")
    elif wanted_on and not saw_reasoning:
        print()
        print("WARNING: thinking was requested but no reasoning text came back. Either the")
        print("         model has no reasoning pass, or the server was launched without a")
        print("         --reasoning-parser and is folding the trace into normal content.")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LLM API endpoints (OpenAI, Anthropic, Gemini)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # OpenAI-compatible server (vLLM, llama.cpp, etc.)
  %(prog)s --type openai --base-url http://localhost:8000/v1 --model llama

  # Anthropic API (direct or proxy)
  %(prog)s --type anthropic --base-url http://localhost:3456 --model claude-3-opus-20240229

  # Anthropic official API
  %(prog)s --type anthropic --model claude-3-5-sonnet-20241022

  # Google Gemini
  %(prog)s --type gemini --model gemini-1.5-pro

  # Claude Code CLI (drives the real `claude` binary, full agentic behavior)
  %(prog)s --type claude-cli --runs 3 --max-tokens 256
  %(prog)s --type claude-cli --model opus --effort high --runs 3 --max-tokens 256

  # Custom settings
  %(prog)s --type openai --base-url http://localhost:8000/v1 --model qwen --max-tokens 512 --runs 20

  # Cost of the reasoning pass on a vLLM reasoning model (A/B the same prompt)
  %(prog)s --thinking off --runs 10
  %(prog)s --thinking on  --runs 10
  %(prog)s --reasoning none --runs 10
        """,
    )
    parser.add_argument("--type", "-t", choices=["openai", "anthropic", "gemini", "claude-cli"], default="openai",
                        help="API type (default: openai)")
    parser.add_argument("--base-url", default=None,
                        help="API base URL (default: type-specific default)")
    parser.add_argument("--api-key", default=None,
                        help="API key (default: from environment or 'not-needed')")
    parser.add_argument("--model", default=None,
                        help="Model name (default: auto-detect or type-specific default)")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models and exit")
    parser.add_argument("--prompt",
                        default="Write a detailed explanation of how transformers work in neural networks.",
                        help="Prompt to use for benchmarking")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max tokens to generate (default: 4096)")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of benchmark runs (default: 5)")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Number of warmup runs (default: 1)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Temperature (default: not sent -- server/model default applies)")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Top-p (default: not sent -- server/model default applies)")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Top-k sampling (default: None)")
    parser.add_argument("--unique-prompts", action="store_true",
                        help="Use unique prompts for each run to avoid cache issues")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Quiet mode - only show summary")
    parser.add_argument("--debug", action="store_true",
                        help="Print debug info for each chunk")
    parser.add_argument("--nostream", action="store_true",
                        help="Disable streaming")
    parser.add_argument("--insecure", "-k", action="store_true",
                        help="Disable SSL certificate verification (for self-signed certs)")
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default=None,
                        help="Reasoning effort, claude-cli only (default: claude's current default, not forced)")
    parser.add_argument("--thinking", nargs="?", const="on", choices=["on", "off"], default=None,
                        help="Toggle the model's reasoning pass, openai type only. Sends "
                             "chat_template_kwargs={'enable_thinking': bool}. Bare --thinking means on. "
                             "Omitted: the field is not sent at all and the server's own default applies")
    parser.add_argument("--reasoning", choices=["off", "none", "low", "medium", "high"], default=None,
                        help="Set the reasoning_effort request field, openai type only. 'off' is an alias for "
                             "'none'. Omitted: not sent. On vLLM this is NOT a second, independent control: "
                             "reasoning_effort=none maps onto the same chat-template switch as --thinking off, "
                             "and the qwen3/gemma4 parsers only distinguish none from not-none, so low/medium/"
                             "high all behave identically to thinking-on")

    args = parser.parse_args()

    # 'off' is only a spelling convenience -- the wire value is always 'none'. Normalise
    # here so the request, the printed config line, and the post-run check all agree.
    if args.reasoning == "off":
        args.reasoning = "none"

    if (args.thinking is not None or args.reasoning is not None) and args.type != "openai":
        print(f"Error: --thinking/--reasoning are only wired for --type openai (got --type {args.type}).")
        if args.type == "claude-cli":
            print("       Use --effort for claude-cli reasoning effort.")
        return

    # Set defaults based on API type
    if args.base_url is None:
        if args.type == "openai":
            args.base_url = "http://localhost:8000/v1"
        elif args.type == "anthropic":
            args.base_url = None  # Use default Anthropic API
        elif args.type == "gemini":
            args.base_url = None  # Gemini doesn't use base_url
    elif args.type == "openai" and args.base_url and not args.base_url.endswith("/v1"):
        # OpenAI SDK expects /v1 suffix
        args.base_url = args.base_url.rstrip("/") + "/v1"

    if args.api_key is None:
        import os
        if args.type == "openai":
            args.api_key = os.environ.get("OPENAI_API_KEY", "not-needed")
        elif args.type == "anthropic":
            args.api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not args.api_key:
                print("Error: ANTHROPIC_API_KEY environment variable or --api-key required")
                return
        elif args.type == "gemini":
            args.api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not args.api_key:
                print("Error: GOOGLE_API_KEY or GEMINI_API_KEY environment variable or --api-key required")
                return

    # Create client
    extra_params = {}
    if args.effort:
        extra_params["effort"] = args.effort
    if args.thinking is not None:
        extra_params["chat_template_kwargs"] = {"enable_thinking": args.thinking == "on"}
    if args.reasoning is not None:
        extra_params["reasoning_effort"] = args.reasoning
    extra_params = extra_params or None
    try:
        initial_model = args.model if args.type == "claude-cli" else (args.model or "placeholder")
        client = create_client(args.type, args.base_url, args.api_key, initial_model, extra_params, insecure=args.insecure)
    except ImportError as e:
        print(f"Error: Missing dependency for {args.type} API: {e}")
        print(f"Install with: pip install {'openai' if args.type == 'openai' else 'anthropic' if args.type == 'anthropic' else 'google-generativeai'}")
        return

    # List models or auto-detect
    if args.list_models or args.model is None:
        try:
            models = client.list_models()
        except Exception as e:
            if args.list_models:
                print(f"Failed to list models: {e}")
                return
            models = []

        if args.list_models:
            if args.type == "claude-cli" and not models:
                print("claude-cli has no fixed model list -- pass --model <alias-or-full-name> "
                      "(e.g. opus, sonnet, haiku) or omit --model to use claude's current default.")
                return
            for m in models:
                print(m)
            return

        if models and args.model is None:
            args.model = models[0]
            print(f"Auto-selected model: {args.model}")
            # Recreate client with actual model
            client = create_client(args.type, args.base_url, args.api_key, args.model, extra_params, insecure=args.insecure)
        elif args.model is None and args.type != "claude-cli":
            print("Error: --model is required (could not auto-detect)")
            return

    print(f"Benchmarking: {args.base_url or f'{args.type} API'}")
    print(f"API Type: {args.type}")
    print(f"Model: {args.model or '(claude default)'}")
    if args.type == "claude-cli":
        print(f"Effort: {args.effort or '(claude default)'}")
    if args.type == "openai":
        switches = []
        if args.thinking is not None:
            switches.append(f"enable_thinking={args.thinking == 'on'}")
        if args.reasoning is not None:
            switches.append(f"reasoning_effort={args.reasoning}")
        print(f"Reasoning switches: {', '.join(switches) if switches else '(none sent, server default)'}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Streaming: {not args.nostream}")
    print()

    try:
        stream = not args.nostream
        results = run_benchmark(
            client,
            args.prompt,
            args.max_tokens,
            args.runs,
            args.warmup,
            args.temperature,
            args.top_p,
            top_k=args.top_k,
            unique_prompts=args.unique_prompts,
            quiet=args.quiet,
            debug=args.debug,
            stream=stream,
        )
        print_summary(results, stream=stream)
        check_switch_took_effect(results, args.thinking, args.reasoning)
    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        if hasattr(client, "close"):
            client.close()


if __name__ == "__main__":
    main()
