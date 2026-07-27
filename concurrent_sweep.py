#!/usr/bin/env python3
"""
Concurrent load sweep for vLLM servers.

Spawns N parallel OpenAI-compatible clients against the server, one client per
worker, and measures tokens/sec, TTFT, and inter-token latency at each concurrency
level. Sweeps through a sequence of concurrency values and prints a summary row
per level.

Usage:
  python3 concurrent_sweep.py
  python3 concurrent_sweep.py --steps 1,8,16,32,64,96,128 --max-tokens 2048
"""

import argparse
import asyncio
import time
import statistics
from openai import AsyncOpenAI

# 25 diverse prompts to avoid cache duplication.
PROMPTS = [
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


async def worker_request(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    results: list,
):
    """Fire a single streaming request and record metrics into results."""
    start = time.perf_counter()
    first_token = None
    tokens = 0

    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
            stream_options={"include_usage": True},
        )
        async for chunk in resp:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and (delta.content or getattr(delta, "reasoning_content", None) or delta.model_extra.get("reasoning_content") or delta.model_extra.get("reasoning")):
                    if first_token is None:
                        first_token = time.perf_counter()
            if hasattr(chunk, "usage") and chunk.usage:
                tokens = chunk.usage.completion_tokens

        end = time.perf_counter()
    except Exception as e:
        end = time.perf_counter()
        results.append({
            "tokens": 0,
            "ttft_ms": 0,
            "itl_ms": 0,
            "total_ms": (end - start) * 1000,
            "error": str(e),
        })
        return

    total_ms = (end - start) * 1000
    ttft_ms = (first_token - start) * 1000 if first_token else 0

    # Generation time: from first token to end of stream.
    # Reasoning-only models (Qwen) have no last_token, so use full window.
    gen_time = (end - first_token) if first_token else (end - start)
    itl_ms = (gen_time * 1000) / tokens if tokens > 0 and gen_time > 0 else 0
    tps = tokens / gen_time if gen_time > 0 else 0

    results.append({
        "tokens": tokens,
        "ttft_ms": ttft_ms,
        "itl_ms": itl_ms,
        "total_ms": total_ms,
        "tokens_per_sec": tps,
        "error": None,
    })


async def run_batch(client, model, concurrency, max_tokens, temperature, top_p, idx_start):
    """Run `concurrency` workers in parallel. Returns list of results dicts."""
    results = []

    async def _run(idx):
        prompt = PROMPTS[(idx_start + idx) % len(PROMPTS)]
        batch = []
        await worker_request(client, model, prompt, max_tokens, temperature, top_p, batch)
        results.extend(batch)

    await asyncio.gather(*[_run(i) for i in range(concurrency)])
    return results


async def run_sweep(
    base_url: str,
    api_key: str,
    model: str,
    steps: list[int],
    max_tokens: int,
    temperature: float,
    top_p: float,
    batches_per_level: int,
    verbose: bool,
):
    """Sweep through concurrency levels and print results."""
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    # Discover model name if not provided
    if model is None:
        try:
            models = await client.models.list()
            model = models.data[0].id
            print(f"Auto-detected model: {model}")
        except Exception as e:
            print(f"Error listing models: {e}")
            print("Pass --model explicitly.")
            return

    print(f"Server:   {base_url}")
    print(f"Model:    {model}")
    print(f"Max tok:  {max_tokens}")
    print(f"Temp:     {temperature}")
    print(f"Steps:    {steps}")
    print(f"Batches/level: {batches_per_level}")
    print()

    # Header: aggregate tok/s is the primary cliff metric (server capacity).
    # Per-request tok/s and latencies are supporting.
    hdr = (f"{'Conc':>6} {'Succ':>6} {'Fail':>5} {'%ok':>5} "
           f"{'AggToks/s':>11} {'ReqTok/s μ':>11} {'ReqTok/s med':>12} "
           f"{'TTFT(μ)':>9} {'TTFT(med)':>9} "
           f"{'ITL(μ)':>8} {'ITL(med)':>8}")
    print(hdr)
    print("-" * len(hdr))

    for level in steps:
        wall_start = time.perf_counter()
        all_batches = []
        for b in range(batches_per_level):
            batch = await run_batch(client, model, level, max_tokens, temperature, top_p, b * level)
            all_batches.append(batch)
        wall_elapsed = time.perf_counter() - wall_start

        # Flatten all batches
        batch = []
        for b in all_batches:
            batch.extend(b)

        successful = [r for r in batch if r["tokens"] > 0]
        failed = [r for r in batch if r["tokens"] == 0]
        n_ok = len(successful)
        n_fail = len(failed)
        total_reqs = n_ok + n_fail

        if not successful:
            print(f"{level:>6} {'0':>6} {n_fail:>5} {'  0%':>5} "
                  f"{'n/a':>11} {'n/a':>11} {'n/a':>12} "
                  f"{'n/a':>9} {'n/a':>9} "
                  f"{'n/a':>8} {'n/a':>8}")
            continue

        # Per-request tokens/sec (median and mean)
        tps_vals = [r["tokens_per_sec"] for r in successful]
        tps_mu = statistics.mean(tps_vals)
        tps_med = statistics.median(tps_vals)

        # TTFT and ITL (mean/median)
        ttft_vals = [r["ttft_ms"] for r in successful]
        itl_vals = [r["itl_ms"] for r in successful]
        ttft_mu = statistics.mean(ttft_vals)
        ttft_med = statistics.median(ttft_vals)
        itl_mu = statistics.mean(itl_vals)
        itl_med = statistics.median(itl_vals)

        # Aggregate throughput: total tokens generated / wall clock of this level
        total_tokens = sum(r["tokens"] for r in successful)
        agg_tps = total_tokens / wall_elapsed if wall_elapsed > 0 else 0

        print(f"{level:>6} {n_ok:>6} {n_fail:>5} {100*n_ok/total_reqs:>5.1f}% "
              f"{agg_tps:>11.1f} {tps_mu:>11.1f} {tps_med:>12.1f} "
              f"{ttft_mu:>9.1f} {ttft_med:>9.1f} "
              f"{itl_mu:>8.2f} {itl_med:>8.2f}")

        if verbose and len(tps_vals) > 1:
            print(f"  wall={wall_elapsed:.1f}s total_tok={total_tokens} agg_tok/s={agg_tps:.1f} "
                  f"stddev_tps={statistics.stdev(tps_vals):.1f}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Concurrent load sweep for vLLM")
    parser.add_argument("--base-url", default="http://localhost:8000/v1",
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument("--model", default=None, help="Model name (auto-detect if omitted)")
    parser.add_argument("--steps", default="1,2,4,8,16,32,48,64,96,128",
                        help="Comma-separated concurrency levels (default: 1,2,4,8,16,32,48,64,96,128)")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max tokens to generate (default: 2048)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--batches-per-level", type=int, default=3,
                        help="Number of independent batches to average per level (default: 3)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    steps = [int(x) for x in args.steps.split(",")]

    asyncio.run(run_sweep(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        steps=steps,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batches_per_level=args.batches_per_level,
        verbose=args.verbose,
    ))


if __name__ == "__main__":
    main()
