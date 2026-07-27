#!/usr/bin/env python3
"""Concurrent throughput benchmark for OpenAI-compatible endpoints."""

import asyncio
import argparse
import time
from dataclasses import dataclass
from openai import AsyncOpenAI


@dataclass
class RequestResult:
    prompt_tokens: int
    completion_tokens: int
    duration: float
    ttft: float
    success: bool
    error: str = ""


async def make_request(client, model, input_tokens, output_tokens, semaphore):
    prompt = "Hello world. " * (input_tokens // 3)
    
    async with semaphore:
        start = time.perf_counter()
        ttft = 0
        completion_tokens = 0
        
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=output_tokens,
                stream=True,
            )
            
            first_token = True
            async for chunk in stream:
                if first_token:
                    ttft = time.perf_counter() - start
                    first_token = False
                if chunk.choices and chunk.choices[0].delta.content:
                    completion_tokens += 1
            
            return RequestResult(input_tokens, completion_tokens, 
                                 time.perf_counter() - start, ttft, True)
        except Exception as e:
            return RequestResult(input_tokens, 0, time.perf_counter() - start, 0, False, str(e))


async def run_benchmark(base_url, model, num_concurrent, num_prompts, input_tokens, output_tokens):
    client = AsyncOpenAI(base_url=base_url, api_key="not-needed")
    semaphore = asyncio.Semaphore(num_concurrent)
    
    print(f"Benchmark: {base_url}")
    print(f"Concurrent: {num_concurrent}, Prompts: {num_prompts}")
    print(f"Input: ~{input_tokens} tokens, Output: {output_tokens} tokens\n")
    
    tasks = [make_request(client, model, input_tokens, output_tokens, semaphore) 
             for _ in range(num_prompts)]
    
    start = time.perf_counter()
    results = await asyncio.gather(*tasks)
    total_time = time.perf_counter() - start
    
    successful = [r for r in results if r.success]
    total_completion = sum(r.completion_tokens for r in successful)
    total_tokens = sum(r.prompt_tokens + r.completion_tokens for r in successful)
    avg_ttft = sum(r.ttft for r in successful) / len(successful) if successful else 0
    
    print("=" * 60)
    print(f"Total time: {total_time:.2f}s")
    print(f"Successful: {len(successful)}/{num_prompts}")
    print(f"Requests/sec: {len(successful) / total_time:.2f}")
    print(f"Aggregate output tokens/sec: {total_completion / total_time:.1f}")
    print(f"Aggregate total tokens/sec: {total_tokens / total_time:.1f}")
    print(f"Avg TTFT: {avg_ttft * 1000:.1f}ms")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="default")
    parser.add_argument("--concurrent", type=int, default=10)
    parser.add_argument("--prompts", type=int, default=100)
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=128)
    args = parser.parse_args()
    
    asyncio.run(run_benchmark(args.base_url, args.model, args.concurrent, 
                               args.prompts, args.input_tokens, args.output_tokens))

if __name__ == "__main__":
    main()
