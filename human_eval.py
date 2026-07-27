#!/bin/bash
# -*- mode: python -*-
""":"
# Auto-activate venv and run this script
VENV_PATH="$HOME/venvs/human-eval"
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
fi
exec python3 "$0" "$@"
":"""
"""
HumanEval Benchmark Runner for Shepherd

Runs the 164 HumanEval programming problems against an OpenAI-compatible API.
Uses the /v1/completions endpoint (proper completion mode, not chat mode).

Usage:
    ./human_eval.py                           # Run all 164 problems against localhost:8000
    ./human_eval.py --count 10                # Run 10 random problems
    ./human_eval.py --baseurl http://host:port  # Custom API endpoint
    ./human_eval.py --live                    # Show pass/fail in real-time
"""

import argparse
import json
import os
import sys
import time
import random
import functools
import urllib.request
import urllib.error
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add human-eval to path
HUMAN_EVAL_PATH = os.path.expanduser("~/src/human-eval")
sys.path.insert(0, HUMAN_EVAL_PATH)

try:
    from human_eval.data import read_problems, write_jsonl, stream_jsonl
    from human_eval.evaluation import evaluate_functional_correctness
    from human_eval.execution import check_correctness
    HUMAN_EVAL_AVAILABLE = True
except ImportError:
    HUMAN_EVAL_AVAILABLE = False
    print("Warning: human_eval library not found. Install from ~/src/human-eval")

print = functools.partial(print, flush=True)

DEFAULT_BASE_URL = "http://localhost:8000"
OUTPUT_DIR = "/tmp/human_eval_results"

# Stop sequences to prevent model from generating beyond the function
STOP_SEQUENCES = ["\ndef ", "\nclass ", "\nif __name__", "\n\n"]


class CompletionClient:
    """Sends prompts via OpenAI-compatible /v1/completions endpoint"""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = "default",
                 verbose: bool = False, debug: bool = False,
                 temperature: float = 0.2, max_tokens: int = 512,
                 chat_mode: bool = False):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.verbose = verbose
        self.debug = debug
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.chat_mode = chat_mode
        self.system_prompt = """Complete the Python function. Output ONLY the function body code.
- Use 4-space indentation
- No explanation, no markdown
- Just the code that goes inside the function"""

    def send_prompt(self, prompt: str) -> Tuple[str, float]:
        """Send prompt and return (completion, response_time_ms)"""
        if self.chat_mode:
            return self._send_chat(prompt)
        else:
            return self._send_completion(prompt)

    def _send_completion(self, prompt: str) -> Tuple[str, float]:
        """Send via /v1/completions endpoint"""
        start_time = time.time()

        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stop": STOP_SEQUENCES,
            "stream": True,
        }

        if self.debug:
            print(f"    [DEBUG] POST {self.base_url}/v1/completions")

        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                f"{self.base_url}/v1/completions",
                data=data,
                headers={"Content-Type": "application/json"}
            )

            content = ""
            with urllib.request.urlopen(req, timeout=300) as resp:
                for line in resp:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: "):
                        line = line[6:]
                        if line == "[DONE]":
                            break
                        try:
                            chunk = json.loads(line)
                            choices = chunk.get("choices", [])
                            if choices:
                                token = choices[0].get("text", "")
                                if token:
                                    content += token
                                    if self.verbose:
                                        print(token, end="")
                        except json.JSONDecodeError:
                            pass

            if self.verbose:
                print()

            response_time = (time.time() - start_time) * 1000

            if self.debug:
                print(f"    [DEBUG] Got {len(content)} chars in {response_time:.0f}ms")

            return content, response_time

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            print(f"\n    ERROR: {e}")
            return f"# ERROR: {e}", response_time

    def _send_chat(self, prompt: str) -> Tuple[str, float]:
        """Send via /v1/chat/completions endpoint (for reasoning models)"""
        import re
        start_time = time.time()

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Complete this Python function:\n\n{prompt}"}
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }

        if self.debug:
            print(f"    [DEBUG] POST {self.base_url}/v1/chat/completions")

        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                f"{self.base_url}/v1/chat/completions",
                data=data,
                headers={"Content-Type": "application/json"}
            )

            content = ""
            reasoning = ""
            with urllib.request.urlopen(req, timeout=300) as resp:
                for line in resp:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: "):
                        line = line[6:]
                        if line == "[DONE]":
                            break
                        try:
                            chunk = json.loads(line)
                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                # Handle both content and reasoning_content
                                token = delta.get("content", "")
                                reason_token = delta.get("reasoning_content", "")
                                if token:
                                    content += token
                                    if self.verbose:
                                        print(token, end="")
                                if reason_token:
                                    reasoning += reason_token
                        except json.JSONDecodeError:
                            pass

            if self.verbose:
                print()

            response_time = (time.time() - start_time) * 1000

            # If no content but we have reasoning, the model didn't produce output
            if not content.strip() and reasoning:
                if self.debug:
                    print(f"    [DEBUG] No content, but got {len(reasoning)} chars of reasoning")

            # Clean up chat response
            content = self._clean_chat_response(content, prompt)

            if self.debug:
                print(f"    [DEBUG] Got {len(content)} chars in {response_time:.0f}ms")

            return content, response_time

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            print(f"\n    ERROR: {e}")
            return f"# ERROR: {e}", response_time

    def _clean_chat_response(self, content: str, prompt: str) -> str:
        """Clean up chat mode response - strip markdown, fix indentation"""
        import re

        # Strip <think>...</think> blocks
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)

        # Extract code from markdown blocks
        code_match = re.search(r'```(?:python)?\s*\n(.*?)```', content, re.DOTALL)
        if code_match:
            content = code_match.group(1)

        # Strip function signature if repeated
        if content.strip().startswith('def '):
            lines = content.strip().split('\n')
            body_lines = []
            found_sig = False
            for line in lines:
                if not found_sig and line.strip().startswith('def '):
                    found_sig = True
                    continue
                if found_sig:
                    body_lines.append(line)
            if body_lines:
                content = '\n'.join(body_lines)

        # Fix indentation - add 4 spaces only to lines that have no indentation
        lines = content.strip().split('\n')
        if lines and lines[0].strip() and not lines[0].startswith(' '):
            fixed_lines = []
            for line in lines:
                if line.strip() and not line.startswith(' '):
                    # Line has content but no indentation - add 4 spaces
                    fixed_lines.append('    ' + line)
                else:
                    # Line is empty or already has indentation - keep as is
                    fixed_lines.append(line)
            content = '\n'.join(fixed_lines)
            return content

        return content.strip()


def test_completion(problem: Dict, completion: str, timeout: float = 3.0) -> Tuple[bool, str]:
    """Test a single completion against its test cases. Returns (passed, result_msg)."""
    try:
        result = check_correctness(problem, completion, timeout, 0)
        passed = result["passed"]
        if passed:
            return True, "PASS"
        else:
            err = result.get("result", "failed")
            return False, f"FAIL: {err[:50]}"
    except Exception as e:
        return False, f"ERROR: {str(e)[:50]}"


def run_human_eval(base_url: str, model: str, count: Optional[int] = None,
                   verbose: bool = False, debug: bool = False,
                   temperature: float = 0.2, max_tokens: int = 512,
                   output_file: Optional[str] = None,
                   live: bool = False, test_timeout: float = 3.0,
                   chat_mode: bool = False) -> str:
    """
    Run HumanEval benchmark and return path to results file.
    """
    if not HUMAN_EVAL_AVAILABLE:
        print("ERROR: human_eval library not available")
        sys.exit(1)

    # Load problems
    print("Loading HumanEval problems...")
    problems = read_problems()
    task_ids = list(problems.keys())

    if count and count < len(task_ids):
        task_ids = random.sample(task_ids, count)
        print(f"Randomly sampled {count} problems")

    print(f"Running {len(task_ids)} problems against {base_url}")
    print(f"Mode: {'chat (/v1/chat/completions)' if chat_mode else 'completion (/v1/completions)'}")
    print()

    # Setup output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_file is None:
        output_file = os.path.join(OUTPUT_DIR, f"samples_{timestamp}.jsonl")

    # Create client
    client = CompletionClient(
        base_url=base_url,
        model=model,
        verbose=verbose,
        debug=debug,
        temperature=temperature,
        max_tokens=max_tokens,
        chat_mode=chat_mode
    )

    # Run problems
    results = []
    total_time = 0
    passed_count = 0
    failed_count = 0

    for i, task_id in enumerate(task_ids, 1):
        problem = problems[task_id]
        prompt = problem["prompt"]
        entry_point = problem["entry_point"]

        # Show progress
        print(f"[{i}/{len(task_ids)}] {task_id}: {entry_point}")

        if debug:
            print(f"  Prompt ({len(prompt)} chars): {prompt[:100]}...")

        # Get completion - just send the prompt directly
        completion, resp_time = client.send_prompt(prompt)
        total_time += resp_time

        # Show first line of completion
        first_line = completion.split('\n')[0] if completion else "(empty)"
        status_line = f"  -> {first_line[:50]}{'...' if len(first_line) > 50 else ''}"

        # Live test if enabled
        if live and completion and not completion.startswith("# ERROR"):
            passed, test_result = test_completion(problem, completion, test_timeout)
            if passed:
                passed_count += 1
                print(f"{status_line} [{test_result}] ({resp_time:.0f}ms)")
            else:
                failed_count += 1
                print(f"{status_line} [{test_result}] ({resp_time:.0f}ms)")
        else:
            print(f"{status_line} ({resp_time:.0f}ms)")

        # Store result
        result = {
            "task_id": task_id,
            "completion": completion
        }
        results.append(result)

        # Write incrementally (in case of crash)
        write_jsonl(output_file, [result], append=(i > 1))

    # Summary
    print()
    print("=" * 60)
    print(f"Completions generated: {len(results)}")
    print(f"Output file: {output_file}")
    print(f"Total API time: {total_time/1000:.1f}s")
    print(f"Avg time per problem: {total_time/len(results):.0f}ms")
    if live:
        total_tested = passed_count + failed_count
        pct = (passed_count / total_tested * 100) if total_tested > 0 else 0
        print(f"Live results: {passed_count}/{total_tested} passed ({pct:.1f}%)")
    print("=" * 60)

    return output_file


def run_evaluation(sample_file: str, timeout: float = 3.0) -> Dict:
    """Run the HumanEval evaluation on generated samples."""
    if not HUMAN_EVAL_AVAILABLE:
        print("ERROR: human_eval library not available")
        return {}

    print()
    print("Running HumanEval evaluation...")
    print("(This executes each completion against test cases)")
    print()

    # Read samples to find which task_ids were attempted
    attempted_tasks = set()
    for sample in stream_jsonl(sample_file):
        attempted_tasks.add(sample["task_id"])

    # Load all problems
    all_problems = read_problems()

    # If not all problems were attempted, create a filtered problem file
    problem_file = None
    if len(attempted_tasks) < len(all_problems):
        print(f"Partial run detected: {len(attempted_tasks)}/{len(all_problems)} problems")

        # Create filtered problem file with only attempted problems
        filtered_problems = {k: v for k, v in all_problems.items() if k in attempted_tasks}
        problem_file = sample_file + "_problems.jsonl"

        # Write filtered problems in JSONL format
        with open(problem_file, 'w') as f:
            for task_id, problem in filtered_problems.items():
                problem_with_id = dict(problem)
                problem_with_id["task_id"] = task_id
                f.write(json.dumps(problem_with_id) + "\n")

    # Run evaluation
    try:
        if problem_file:
            results = evaluate_functional_correctness(
                sample_file,
                k=[1],
                n_workers=4,
                timeout=timeout,
                problem_file=problem_file
            )
        else:
            results = evaluate_functional_correctness(
                sample_file,
                k=[1],
                n_workers=4,
                timeout=timeout
            )
    finally:
        # Clean up temp problem file
        if problem_file and os.path.exists(problem_file):
            os.remove(problem_file)

    print()
    print("=" * 60)
    print("HUMANEVAL RESULTS")
    print("=" * 60)
    for metric, value in results.items():
        print(f"{metric}: {value*100:.2f}%")
    print("=" * 60)

    # Count passed/failed from results file
    results_file = sample_file + "_results.jsonl"
    if os.path.exists(results_file):
        passed = 0
        failed = 0
        failed_tasks = []
        for item in stream_jsonl(results_file):
            if item.get("passed", False):
                passed += 1
            else:
                failed += 1
                failed_tasks.append(item.get("task_id", "unknown"))

        print(f"\nPassed: {passed}")
        print(f"Failed: {failed}")
        print(f"Total:  {passed + failed}")

        if failed_tasks and len(failed_tasks) <= 20:
            print(f"\nFailed tasks: {', '.join(failed_tasks)}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run HumanEval benchmark against OpenAI-compatible API (completion mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    ./human_eval.py                           # Run all 164 problems
    ./human_eval.py --count 20                # Run 20 random problems
    ./human_eval.py --baseurl http://host:8000  # Custom endpoint
    ./human_eval.py --live                    # Show pass/fail in real-time
    ./human_eval.py --skip-eval               # Skip evaluation, just generate
    ./human_eval.py --eval-only /path/to/samples.jsonl  # Evaluate existing file
        """
    )

    parser.add_argument("--baseurl", default=DEFAULT_BASE_URL,
                        help=f"OpenAI API base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--model", default="default",
                        help="Model name (default: default)")
    parser.add_argument("--count", "-n", type=int,
                        help="Number of problems to run (default: all 164)")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Sampling temperature (default: 0.2)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens to generate (default: 512)")
    parser.add_argument("--timeout", type=float, default=3.0,
                        help="Timeout for each test execution in seconds (default: 3.0)")
    parser.add_argument("--output", "-o",
                        help="Output JSONL file path (default: auto-generated)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation, only generate completions")
    parser.add_argument("--eval-only",
                        help="Only run evaluation on existing samples file")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output (show full completions)")
    parser.add_argument("--debug", action="store_true",
                        help="Debug output")
    parser.add_argument("--live", action="store_true",
                        help="Test each completion immediately (shows pass/fail in real-time)")
    parser.add_argument("--chat", action="store_true",
                        help="Use chat mode (/v1/chat/completions) - use with --max-tokens 4096 for reasoning models")

    args = parser.parse_args()

    if not HUMAN_EVAL_AVAILABLE:
        print("ERROR: human_eval library not found")
        print("Make sure ~/src/human-eval exists and is set up")
        sys.exit(1)

    try:
        if args.eval_only:
            # Just run evaluation on existing file
            if not os.path.exists(args.eval_only):
                print(f"ERROR: File not found: {args.eval_only}")
                sys.exit(1)
            run_evaluation(args.eval_only, timeout=args.timeout)
        else:
            # Generate completions
            sample_file = run_human_eval(
                base_url=args.baseurl,
                model=args.model,
                count=args.count,
                verbose=args.verbose,
                debug=args.debug,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                output_file=args.output,
                live=args.live,
                test_timeout=args.timeout,
                chat_mode=args.chat
            )

            # Run evaluation unless skipped (or if --live already tested)
            if not args.skip_eval and not args.live:
                run_evaluation(sample_file, timeout=args.timeout)
            elif args.live:
                print("\n(Skipping batch evaluation - live testing was enabled)")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
