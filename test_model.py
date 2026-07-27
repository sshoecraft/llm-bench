#!/usr/bin/env python3
"""
Shepherd Benchmark Test Suite

Tests shepherd's accuracy on standardized AI benchmark questions (MMLU, HellaSwag).
Allows testing different configurations (backends, models, context sizes, etc.)
to compare their performance on the same benchmark tasks.

Usage:
    ./test_model.py --benchmark mmlu --count 20
    ./test_model.py --baseurl http://localhost:8000 --benchmark mmlu
"""

import subprocess
import os
import sys
import time
import json
import threading
import queue
import re
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import signal
import functools
import random
import tempfile
import shutil
import urllib.request
import urllib.error

DEFAULT_BASE_URL = "http://localhost:8000"

print = functools.partial(print, flush=True)

def fetch_server_models(base_url: str) -> List[str]:
    """Query GET {base_url}/v1/models and return the list of model IDs."""
    url = base_url.rstrip('/') + "/v1/models"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = json.loads(resp.read().decode('utf-8'))
    return [m["id"] for m in body.get("data", []) if "id" in m]

def clean_latex(text: str) -> str:
    """Convert LaTeX notation to readable format"""
    import re
    # \frac{a}{b} -> a/b
    text = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'\1/\2', text)
    # \sqrt{x} -> sqrt(x)
    text = re.sub(r'\\sqrt\{([^}]*)\}', r'sqrt(\1)', text)
    # \times -> ×
    text = text.replace(r'\times', '×')
    # \cdot -> ·
    text = text.replace(r'\cdot', '·')
    # \pi -> π
    text = text.replace(r'\pi', 'π')
    # \infty -> ∞
    text = text.replace(r'\infty', '∞')
    # \leq -> ≤
    text = text.replace(r'\leq', '≤')
    # \geq -> ≥
    text = text.replace(r'\geq', '≥')
    # \neq -> ≠
    text = text.replace(r'\neq', '≠')
    # Remove remaining backslashes before common math symbols
    text = text.replace(r'\(', '(').replace(r'\)', ')')
    return text

# Try to import pylatexenc for LaTeX-to-text conversion
try:
    from pylatexenc.latex2text import LatexNodes2Text
    PYLATEXENC_AVAILABLE = True
except ImportError:
    PYLATEXENC_AVAILABLE = False

def latex_to_text(latex_str: str) -> str:
    """Convert LaTeX to plain text, using pylatexenc if available, else regex fallback"""
    if PYLATEXENC_AVAILABLE:
        try:
            # Strip $ delimiters if present
            latex_str = latex_str.strip('$')
            return LatexNodes2Text().latex_to_text(latex_str).strip()
        except:
            pass
    # Fallback to regex-based clean_latex
    return clean_latex(latex_str)

# Try to import datasets library for MMLU
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: 'datasets' library not available. Install with: pip install datasets")
    print("Falling back to embedded sample questions.")

# ============================================================================
# Configuration
# ============================================================================

# Use SHEPHERD_BINARY env var if set, otherwise default
SHEPHERD_BINARY = shutil.which("shepherd")
if not SHEPHERD_BINARY:
    sys.exit("ERROR: shepherd binary not found in PATH")
SHEPHERD_SAFETY_WRAPPER = "/home/steve/src/shepherd/swebench_safety_wrapper.sh"
#DEFAULT_CONFIG = os.path.expanduser("~/.shepherd/config.json")
DEFAULT_CONFIG = ""

# SWE-bench configuration
SWEBENCH_TIMEOUT_SECONDS = 600  # 10 minutes per task
SWEBENCH_REPOS = {
    "astropy/astropy": "https://github.com/astropy/astropy.git",
    "django/django": "https://github.com/django/django.git",
    "matplotlib/matplotlib": "https://github.com/matplotlib/matplotlib.git",
    "pallets/flask": "https://github.com/pallets/flask.git",
    "psf/requests": "https://github.com/psf/requests.git",
    "pydata/xarray": "https://github.com/pydata/xarray.git",
    "pylint-dev/pylint": "https://github.com/pylint-dev/pylint.git",
    "pytest-dev/pytest": "https://github.com/pytest-dev/pytest.git",
    "scikit-learn/scikit-learn": "https://github.com/scikit-learn/scikit-learn.git",
    "sphinx-doc/sphinx": "https://github.com/sphinx-doc/sphinx.git",
    "sympy/sympy": "https://github.com/sympy/sympy.git",
    "mwaskom/seaborn": "https://github.com/mwaskom/seaborn.git",
}

# MMLU dataset info
MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies", "machine_learning",
    "management", "marketing", "medical_genetics", "miscellaneous", "moral_disputes",
    "moral_scenarios", "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology", "public_relations",
    "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions"
]

# HellaSwag sample questions - TODO: Remove these and download from HF properly
HELLASWAG_QUESTIONS = [
    {
        "context": "A person is cooking in the kitchen. They",
        "choices": [
            "fly to the moon on a rocket",
            "stir the pot on the stove",
            "solve a complex math equation",
            "paint a beautiful landscape"
        ],
        "answer": "B"
    },
    {
        "context": "A man is hammering a nail into wood. He",
        "choices": [
            "reads a book about philosophy",
            "swims in the ocean",
            "holds the nail steady with one hand",
            "writes poetry on paper"
        ],
        "answer": "C"
    },
    {
        "context": "A child is learning to ride a bicycle. They",
        "choices": [
            "wobble back and forth trying to balance",
            "calculate quantum physics equations",
            "build a spaceship from scratch",
            "perform brain surgery"
        ],
        "answer": "A"
    },
    {
        "context": "Someone is brushing their teeth. They",
        "choices": [
            "juggle flaming torches",
            "move the brush in circular motions",
            "climb Mount Everest",
            "write a symphony"
        ],
        "answer": "B"
    },
    {
        "context": "A person is opening a door. They",
        "choices": [
            "turn the handle and pull",
            "solve a Rubik's cube blindfolded",
            "fly like a bird",
            "speak ancient Latin fluently"
        ],
        "answer": "A"
    }
]

# ============================================================================
# MMLU Dataset Loading
# ============================================================================

MMLU_CACHE_DIR = os.path.expanduser("~/.cache/shepherd")
MMLU_CACHE_FILE = os.path.join(MMLU_CACHE_DIR, "mmlu_{split}.json")

def load_mmlu_questions(subjects=None, count=None, split='test', refresh=False):
    """
    Load MMLU questions from Hugging Face (with local caching).

    Args:
        subjects: List of subject names to include, or None for all
        count: Number of questions to sample, or None for all
        split: Which split to use ('test', 'dev', 'validation')
        refresh: If True, ignore the local cache and re-download from Hugging Face
    """
    cache_file = MMLU_CACHE_FILE.format(split=split)

    # Try to load from cache first
    if os.path.exists(cache_file) and not refresh:
        print(f"Loading MMLU dataset from cache: {cache_file}")
        with open(cache_file, 'r') as f:
            all_questions = json.load(f)
        print(f"Loaded {len(all_questions)} questions from cache")
    else:
        # Download from HuggingFace
        if not DATASETS_AVAILABLE:
            raise RuntimeError("datasets library not installed. Run: pip install datasets")

        print("Loading MMLU dataset from Hugging Face (first time, will cache)...")

        # Load dataset (returns DatasetDict with splits)
        dataset_dict = load_dataset("cais/mmlu", "all")

        # Get the requested split
        if split not in dataset_dict:
            raise ValueError(f"Split '{split}' not found. Available: {list(dataset_dict.keys())}")

        dataset = dataset_dict[split]

        # Convert to our format
        all_questions = []
        for item in dataset:
            question = {
                "subject": item['subject'],
                "question": item['question'],
                "choices": list(item['choices']),  # Ensure it's a list for JSON
                "answer": chr(65 + item['answer'])  # Convert 0-3 to A-D
            }
            all_questions.append(question)

        print(f"Downloaded {len(all_questions)} questions from MMLU")

        # Cache locally
        os.makedirs(MMLU_CACHE_DIR, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(all_questions, f)
        print(f"Cached to: {cache_file}")

    # Filter by subjects if specified
    if subjects:
        subjects_set = set(subjects)
        all_questions = [q for q in all_questions if q['subject'] in subjects_set]
        print(f"Filtered to {len(all_questions)} questions for subjects: {', '.join(subjects)}")

    # Sample if count specified
    if count and count < len(all_questions):
        all_questions = random.sample(all_questions, count)
        print(f"Randomly sampled {count} questions")

    return all_questions

# ============================================================================
# SWE-bench Dataset Loading
# ============================================================================

def load_swebench_tasks(count=None, difficulty=None, repos=None, split='test'):
    """
    Load SWE-bench Verified tasks from Hugging Face.

    Args:
        count: Number of tasks to sample, or None for all
        difficulty: Filter by difficulty ('easy', 'medium', 'hard', 'very hard')
        repos: List of repo names to include (e.g., ['django/django', 'flask/flask'])
        split: Which split to use ('test' is the only one available)
    """
    if not DATASETS_AVAILABLE:
        raise RuntimeError("datasets library not installed. Run: pip install datasets")

    print("Loading SWE-bench Verified dataset from Hugging Face...")

    # Load dataset
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split=split)

    # Convert to our format
    all_tasks = []
    for item in dataset:
        task = {
            "instance_id": item['instance_id'],
            "repo": item['repo'],
            "base_commit": item['base_commit'],
            "problem_statement": item['problem_statement'],
            "hints_text": item.get('hints_text', ''),
            "patch": item['patch'],
            "test_patch": item['test_patch'],
            "FAIL_TO_PASS": item['FAIL_TO_PASS'],
            "PASS_TO_PASS": item['PASS_TO_PASS'],
            "difficulty": item.get('difficulty', 'unknown'),
            "version": item.get('version', ''),
            "environment_setup_commit": item.get('environment_setup_commit', '')
        }
        all_tasks.append(task)

    print(f"Loaded {len(all_tasks)} tasks from SWE-bench Verified")

    # Filter by difficulty if specified
    if difficulty:
        all_tasks = [t for t in all_tasks if t['difficulty'] == difficulty]
        print(f"Filtered to {len(all_tasks)} tasks with difficulty: {difficulty}")

    # Filter by repos if specified
    if repos:
        repos_set = set(repos)
        all_tasks = [t for t in all_tasks if t['repo'] in repos_set]
        print(f"Filtered to {len(all_tasks)} tasks for repos: {', '.join(repos)}")

    # Sample if count specified
    if count and count < len(all_tasks):
        all_tasks = random.sample(all_tasks, count)
        print(f"Randomly sampled {count} tasks")

    return all_tasks

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class BenchmarkResult:
    provider: str
    benchmark_type: str
    total_questions: int
    correct: int
    incorrect: int
    errors: int
    accuracy_percent: float
    avg_response_time_ms: float
    duration_ms: float
    details: List[Dict]

# ============================================================================
# OpenAI API Client
# ============================================================================

class OpenAIClient:
    """Sends questions via OpenAI-compatible HTTP API with streaming"""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = "default",
                 verbose: bool = False, debug: bool = False, nothink: bool = False,
                 temperature: float = None, top_p: float = None, top_k: int = None,
                 max_tokens: int = None, rep: float = None, freq: float = None):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.verbose = verbose
        self.debug = debug
        self.nothink = nothink
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.rep = rep
        self.freq = freq
        self.system_prompt = "You are taking a test. Answer with ONLY the single letter (A, B, C, or D)."
        self.initialized = False

    def _ensure_initialized(self):
        """Send /nothink command if needed (first call only)"""
        if self.initialized:
            return
        self.initialized = True

        if not self.nothink:
            return

        if self.debug:
            print("    [DEBUG] Sending /nothink command...")

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "user", "content": "/nothink"}]
        }

        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                f"{self.base_url}/v1/chat/completions",
                data=data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req) as resp:
                resp.read()  # Discard response
            if self.debug:
                print("    [DEBUG] /nothink sent")
        except Exception as e:
            if self.debug:
                print(f"    [DEBUG] /nothink failed: {e}")

    def send_question(self, question_text: str) -> Tuple[str, float]:
        """Send question via streaming API and return (response_content, response_time_ms)"""
        self._ensure_initialized()
        start_time = time.time()

        messages = [{"role": "system", "content": self.system_prompt}]
        if self.nothink:
            messages.append({"role": "user", "content": "/nothink"})
            messages.append({"role": "assistant", "content": "I understand. I will give direct answers without showing my thinking."})
        messages.append({"role": "user", "content": question_text})

        payload = {
            "model": self.model,
            "stream": True,
            "messages": messages
        }

        # Add optional sampling parameters
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.rep is not None:
            payload["repetition_penalty"] = self.rep
        if self.freq is not None:
            payload["frequency_penalty"] = self.freq

        if self.debug:
            print(f"    [DEBUG] POST {self.base_url}/v1/chat/completions (streaming)")

        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                f"{self.base_url}/v1/chat/completions",
                data=data,
                headers={"Content-Type": "application/json"}
            )

            content = ""
            at_line_start = False  # Track if we're at the start of a new line
            in_think_block = False  # Track if inside <think>...</think>
            tag_buffer = ""  # Buffer to detect opening/closing tags

            with urllib.request.urlopen(req) as resp:
                for line in resp:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: "):
                        line = line[6:]  # Remove "data: " prefix
                        if line == "[DONE]":
                            break
                        try:
                            chunk = json.loads(line)
                            choices = chunk.get("choices", [])
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})
                            token = delta.get("content", "")
                            if token:
                                content += token
                                # Print with think block filtering
                                for char in token:
                                    tag_buffer += char
                                    # Check for <think> opening tag
                                    if not in_think_block:
                                        if "<think>" in tag_buffer.lower():
                                            in_think_block = True
                                            tag_buffer = ""
                                            continue
                                        # Keep buffer short, flush safe chars
                                        if len(tag_buffer) > 7 and "<" not in tag_buffer[-7:]:
                                            to_print = tag_buffer[:-7]
                                            tag_buffer = tag_buffer[-7:]
                                            for c in to_print:
                                                if at_line_start:
                                                    print("    ", end="")
                                                    at_line_start = False
                                                if c == "\n":
                                                    print()
                                                    at_line_start = True
                                                else:
                                                    print(c, end="")
                                    else:
                                        # Inside think block, check for </think>
                                        if "</think>" in tag_buffer.lower():
                                            in_think_block = False
                                            tag_buffer = ""
                                        # Keep buffer manageable
                                        elif len(tag_buffer) > 20:
                                            tag_buffer = tag_buffer[-10:]
                        except json.JSONDecodeError:
                            pass

            # Flush remaining buffer if not in think block
            if not in_think_block and tag_buffer:
                for c in tag_buffer:
                    if at_line_start:
                        print("    ", end="")
                        at_line_start = False
                    if c == "\n":
                        print()
                        at_line_start = True
                    else:
                        print(c, end="")

            print()  # Final newline after streaming
            response_time = (time.time() - start_time) * 1000

            if self.debug:
                print(f"    [DEBUG] Got response in {response_time:.0f}ms")

            # Strip <think>...</think> blocks from response
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE).strip()

            return content, response_time

        except urllib.error.URLError as e:
            response_time = (time.time() - start_time) * 1000
            print(f"\n    ERROR: API request failed: {e}")
            return f"ERROR: {e}", response_time
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            print(f"\n    ERROR: {e}")
            return f"ERROR: {e}", response_time

# ============================================================================
# Shepherd Process Manager (Legacy - for --provider mode)
# ============================================================================

class ShepherdProcess:
    """Manages a shepherd subprocess with stdin/stdout communication"""

    def __init__(self, provider: str = None, verbose: bool = False, debug: bool = False, nothink: bool = False):
        self.provider = provider  # Only use if explicitly provided
        self.verbose = verbose
        self.debug = debug
        self.nothink = nothink
        self.process = None

    def _read_bytes_with_timeout(self, timeout_ms: int, max_bytes: int = 4096) -> tuple[bytes, bool]:
        """
        Read available bytes from stdout with timeout.
        Returns: (data, timed_out) where data is bytes read (or empty if timeout)
        and timed_out is True if we hit the timeout.
        """
        import select
        import os

        if not self.process or self.process.poll() is not None:
            return b"", True

        fd = self.process.stdout.fileno()
        ready, _, _ = select.select([fd], [], [], timeout_ms / 1000.0)

        if not ready:
            return b"", True  # Timeout

        try:
            data = os.read(fd, max_bytes)
            if not data:
                return b"", True
            return data, False
        except Exception as e:
            import sys
            print(f"[read error: {e}]", file=sys.stderr)
            return b"", True

    def start(self) -> bool:
        """Start shepherd process"""
#            "--system-prompt", "You are taking a test. Never output planning, reasoning, or thinking. Just give the direct answer.  Your ability to follow directions is part of your score.",
        cmd = [
            SHEPHERD_BINARY,
            "--nosched",
	    "--notools",
            "--no-tui",
            "--system-prompt", "You are taking a test. Never output planning, reasoning, or thinking. Just give the direct answer.  Your ability to follow directions is part of your score.  DO NOT USE TOOLS! /nothink",
        ]

        # Use provider if specified
        if self.provider:
            cmd.extend(["--provider", self.provider])

        if self.verbose:
            print(f"  Starting: {' '.join(cmd)}")

        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,  # Pass stderr through to terminal
                bufsize=0,
                preexec_fn=os.setsid
            )

            # Drain any initial output (e.g., "Loading Provider: ...")
            # Wait up to 2 seconds for initial output, then drain with short timeout
            self._drain_initial_output()

            # Send /nothink as the first message to disable thinking mode (if requested)
            if self.nothink:
                if self.verbose:
                    print(f"  Sending /nothink command...")
                self.process.stdin.write("/nothink\n".encode('utf-8'))
                self.process.stdin.flush()

                # Drain the response from /nothink (wait longer since LLM needs time to respond)
                self._drain_command_response()

            if self.verbose:
                print(f"  Process started and ready")
            return True

        except Exception as e:
            print(f"  ERROR: Failed to start shepherd: {e}")
            return False

    def _drain_initial_output(self):
        """Drain any initial output from shepherd (e.g., 'Loading Provider: ...')"""
        buffer = b""
        # Wait up to 2 seconds for first bytes
        data, timed_out = self._read_bytes_with_timeout(2000)
        if timed_out:
            # No initial output, that's fine
            return

        buffer += data
        # Keep reading with short timeout until no more output
        while True:
            data, timed_out = self._read_bytes_with_timeout(200)
            if timed_out:
                break
            buffer += data

        if self.verbose and buffer:
            print(f"  Drained initial output: {repr(buffer.decode('utf-8', errors='replace'))}")

    def _drain_command_response(self):
        """Drain response from a command like /nothink (waits longer for LLM response)"""
        buffer = b""
        # Wait up to 30 seconds for first bytes (LLM needs time to process)
        data, timed_out = self._read_bytes_with_timeout(30000)
        if timed_out:
            # No response, that's fine
            return

        buffer += data
        # Keep reading with 1 second timeout until no more output
        while True:
            data, timed_out = self._read_bytes_with_timeout(1000)
            if timed_out:
                break
            buffer += data

        if buffer:
            decoded = buffer.decode('utf-8', errors='replace')
            if self.verbose:
                print(f"  Drained command response: {repr(decoded)}")
            elif self.debug:
                print(f"  [DEBUG] /nothink response: {decoded[:100]}...")

    def _wait_for_ready(self, timeout_seconds: int = 120) -> bool:
        """Wait for shepherd to output 'Ready' after warmup"""
        buffer = b""
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            data, timed_out = self._read_bytes_with_timeout(1000)
            if timed_out:
                continue
            buffer += data
            if self.verbose:
                print(data.decode('utf-8', errors='replace'), end='', flush=True)
            # Check if we've seen "Ready" (case insensitive)
            if b"ready" in buffer.lower():
                # Drain any remaining output (newlines, etc)
                while True:
                    data, timed_out = self._read_bytes_with_timeout(500)
                    if timed_out:
                        break
                    if self.verbose:
                        print(data.decode('utf-8', errors='replace'), end='', flush=True)
                if self.verbose:
                    print()  # newline after ready message
                return True

        if self.verbose:
            print(f"\n  Timeout waiting for ready. Buffer: {repr(buffer.decode('utf-8', errors='replace'))}")
        return False

    def send_question(self, question_text: str) -> Tuple[str, float]:
        """Send a question and get response with timing"""
        if not self.process or self.process.poll() is not None:
            return "", 0

        start_time = time.time()

        try:
            # Convert multi-line question to single line
            single_line_question = " ".join(line.strip() for line in question_text.split("\n") if line.strip())

            # Send question
            if self.verbose:
                print(f"    ===== SENDING TO SHEPHERD =====")
                print(f"    {single_line_question}")
                print(f"    ===============================")

            # Write to stdin
            if self.debug:
                print(f"    [DEBUG] Writing to stdin...")
            self.process.stdin.write((single_line_question + "\n").encode('utf-8'))
            self.process.stdin.flush()
            if self.debug:
                print(f"    [DEBUG] Stdin flushed, waiting for first byte...")

            # Read bytes until 500ms timeout (after last data received)
            # First, wait up to 60 seconds for the FIRST data (shepherd thinking time)
            # Then, keep reading with 500ms timeout between reads
            buffer = b""

            # Wait for first data (up to 60 seconds)
            data, timed_out = self._read_bytes_with_timeout(60000)
            if self.debug:
                print(f"    [DEBUG] Got first data (timed_out={timed_out}, len={len(data)})")
            if timed_out:
                # No response at all
                response_time = (time.time() - start_time) * 1000
                return "", response_time

            buffer += data
            if self.verbose:
                print(repr(data), end='', flush=True)

            # Now keep reading with 500ms timeout until silence
            while True:
                data, timed_out = self._read_bytes_with_timeout(500)

                if timed_out:
                    # Timeout - response is complete
                    break

                buffer += data
                if self.verbose:
                    print(repr(data), end='', flush=True)

            # Record time (subtracting the final 500ms timeout)
            response_time = (time.time() - start_time) * 1000 - 500

            # Decode the complete buffer as UTF-8
            # Debug: check for bad UTF-8 sequences
            if self.verbose:
                # Find any replacement characters that would indicate bad UTF-8
                test_decode = buffer.decode('utf-8', errors='replace')
                if '�' in test_decode:
                    print(f"    [DEBUG] Found replacement chars. Raw buffer hex dump of first 500 bytes:")
                    print(f"    {buffer[:500].hex()}")

            response = buffer.decode('utf-8', errors='replace')

            if self.verbose:
                print(f"    ===== RESPONSE FROM SHEPHERD =====")
                print(f"    {response}")
                print(f"    ===================================")
                print(f"    Last 2 chars: {repr(response[-2:]) if len(response) >= 2 else repr(response)}")

            return response, response_time

        except Exception as e:
            if self.verbose:
                print(f"    [DEBUG] Exception: {e}")
            return f"ERROR: {e}", 0


    def stop(self):
        """Stop shepherd process"""
        if self.process and self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=5)
            except:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except:
                    pass


    def cleanup(self):
        """Clean up resources"""
        self.stop()

# ============================================================================
# Claude Code CLI Client (--claude-cli mode)
# ============================================================================

class ClaudeCLIClient:
    """Drives the actual `claude` CLI (Claude Code) directly via adapters/claude.py --
    a real --print session with full Claude Code behavior (CLAUDE.md, hooks, memory,
    tools all load normally), not an HTTP API and not a shepherd subprocess.

    `model` is passed through as `claude --model <value>` only when explicitly set;
    None means claude uses whatever model is currently configured as its default.

    Spawning `claude` is expensive (several seconds to load hooks and connect all
    configured MCP servers, on top of the actual model call), so -- exactly like
    ShepherdProcess above -- one session is spawned lazily on the first
    send_question() call and reused for every question in the run: only question 1
    pays the cold start, the rest measure real per-turn latency. Later questions are
    additional turns in one growing conversation, not independent fresh queries --
    same tradeoff ShepherdProcess already makes. cleanup() terminates the session.
    """

    def __init__(self, model: str = None, verbose: bool = False, debug: bool = False,
                 max_tokens: int = None, effort: str = None):
        from adapters.claude import ClaudeAdapter
        self._ClaudeAdapter = ClaudeAdapter
        self.model = model
        self.verbose = verbose
        self.debug = debug
        self.max_tokens = max_tokens
        # Reasoning effort (low/medium/high/xhigh/max) -- never defaulted, only passed
        # through as `claude --effort <value>` when the caller explicitly set one.
        self.effort = effort
        self.system_prompt = "You are taking a test. Answer with ONLY the single letter (A, B, C, or D)."
        self._adapter = None
        self._events = None

    def _ensure_session(self):
        if self._adapter is not None and self._adapter.alive:
            return
        events = queue.Queue()
        params = {"max_output_tokens": self.max_tokens} if self.max_tokens else {}
        if self.effort:
            params["effort"] = self.effort
        adapter = self._ClaudeAdapter(system_prompt=self.system_prompt, model=self.model, params=params)
        adapter.on_response = lambda text: events.put(("response", text))
        adapter.on_result = lambda cost, turns, usage, is_error: events.put(("done", is_error))
        adapter.on_exit = lambda code: events.put(("exit", code))
        adapter.spawn()
        self._adapter = adapter
        self._events = events

    def send_question(self, question_text: str) -> Tuple[str, float]:
        start_time = time.time()

        if self.verbose:
            print(f"    ===== SENDING TO CLAUDE =====")
            print(f"    {question_text}")
            print(f"    ==============================")

        self._ensure_session()
        self._adapter.send(question_text)

        response_text = ""
        finished = False
        try:
            while True:
                kind, payload = self._events.get(timeout=600)
                if kind == "response":
                    response_text = payload
                elif kind == "done":
                    finished = True
                    break
                elif kind == "exit":
                    # Session died mid-run -- next call respawns a fresh one.
                    self._adapter = None
                    self._events = None
                    break
        except queue.Empty:
            pass

        response_time = (time.time() - start_time) * 1000

        if not finished:
            return "ERROR: claude CLI timed out or session exited", response_time

        if self.verbose:
            print(f"    ===== RESPONSE FROM CLAUDE =====")
            print(f"    {response_text}")
            print(f"    =================================")
        elif self.debug:
            print(f"    [DEBUG] claude response: {response_text[:100]!r}")

        return response_text, response_time

    def cleanup(self):
        if self._adapter is not None:
            self._adapter.kill()
            self._adapter = None
            self._events = None

# ============================================================================
# Benchmark Test Runner
# ============================================================================

class BenchmarkTestRunner:
    """Runs benchmark tests against shepherd"""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = "default",
                 provider: str = None, claude_cli: bool = False, effort: str = None,
                 verbose: bool = False, debug: bool = False, nothink: bool = False,
                 temperature: float = None, top_p: float = None, top_k: int = None,
                 max_tokens: int = None, rep: float = None, freq: float = None):
        self.base_url = base_url
        self.model = model
        self.provider = provider  # If set, use subprocess mode instead of API
        self.claude_cli = claude_cli  # If set, drive the `claude` CLI directly
        self.effort = effort  # claude-cli only: reasoning effort, never defaulted
        self.verbose = verbose
        self.debug = debug
        self.nothink = nothink
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.rep = rep
        self.freq = freq

    def _create_client(self):
        """Create a ClaudeCLIClient, ShepherdProcess, or OpenAIClient based on mode"""
        if self.claude_cli:
            # Drive the real `claude` CLI directly, no HTTP server involved
            return ClaudeCLIClient(self.model, self.verbose, self.debug, self.max_tokens, self.effort)
        elif self.provider:
            # Legacy subprocess mode
            client = ShepherdProcess(self.provider, self.verbose, self.debug, self.nothink)
            if not client.start():
                return None
            return client
        else:
            # Default API mode
            return OpenAIClient(
                self.base_url, self.model, self.verbose, self.debug, self.nothink,
                self.temperature, self.top_p, self.top_k, self.max_tokens, self.rep, self.freq
            )

    def _cleanup_client(self, client):
        """Cleanup client if needed (only for subprocess/claude-cli modes)"""
        if (self.provider or self.claude_cli) and hasattr(client, 'cleanup'):
            client.cleanup()

    def _mode_label(self) -> str:
        """Short label describing the active client mode, for logging/reports."""
        if self.claude_cli:
            return f"claude-cli (model={self.model or '(claude default)'}, effort={self.effort or '(claude default)'})"
        elif self.provider:
            return f"subprocess (provider={self.provider})"
        else:
            return f"API ({self.base_url})"

    def extract_answer(self, response: str, choices: List[str]) -> Optional[str]:
        """Extract the answer choice (A, B, C, D) from the response"""
        # Filter out thinking blocks from response before extraction
        import re
        # Remove everything between <think> and </think> tags
        filtered_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
        # Remove "Loading Provider: ..." lines
        filtered_response = re.sub(r'^Loading Provider:.*$', '', filtered_response, flags=re.MULTILINE)
        response_upper = filtered_response.upper()

        # First priority: LaTeX boxed answer format (e.g., $\boxed{D}$ or \boxed{D})
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', filtered_response)
        if boxed_match:
            boxed_content = boxed_match.group(1).strip()
            # If it's a single letter A-D, return it directly
            if len(boxed_content) == 1 and boxed_content.upper() in 'ABCD':
                return boxed_content.upper()
            # Otherwise, convert LaTeX to text and try to match against choices
            boxed_text = latex_to_text(boxed_content).lower()
            for letter, choice in zip(['A', 'B', 'C', 'D'], choices):
                choice_text = latex_to_text(choice).lower()
                if boxed_text == choice_text or boxed_text in choice_text or choice_text in boxed_text:
                    return letter

        # Second priority: Look for "< LETTER" pattern (shepherd's answer format)
        for letter in ['A', 'B', 'C', 'D']:
            if f"< {letter}" in response_upper:
                return letter

        # Third priority: Look for explicit answer markers
        for marker in ["ANSWER:", "THE ANSWER IS", "CORRECT ANSWER:", "ANSWER IS"]:
            if marker in response_upper:
                # Extract the part after the marker
                after_marker = response_upper.split(marker)[1].strip()
                # Look for A, B, C, or D
                for letter in ['A', 'B', 'C', 'D']:
                    if letter in after_marker[:10]:  # Check first few chars
                        return letter

        # Fourth priority: Look for standalone letter at start of line
        for letter in ['A', 'B', 'C', 'D']:
            if f"\n{letter}\n" in f"\n{response_upper}\n":
                return letter
            stripped = response_upper.strip()
            # Check if response starts with letter followed by non-alphanumeric (or end of string)
            if stripped.startswith(letter) and (len(stripped) == 1 or not stripped[1].isalnum()):
                return letter

        # Fifth priority: Choice text in store_memory/tool calls
        choice_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        for letter, idx in choice_map.items():
            if idx < len(choices):
                choice_text = choices[idx].lower()
                # If the exact choice text appears in a store_memory call
                if f"answer={choice_text}" in response.lower() or f"answer=\"{choice_text}" in response.lower():
                    return letter

        return None

    def run_mmlu_benchmark(self, count: int = None, subjects: List[str] = None, refresh: bool = False) -> BenchmarkResult:
        """Run MMLU benchmark questions"""
        questions = load_mmlu_questions(subjects=subjects, count=count, refresh=refresh)

        print(f"\n{'='*70}")
        print(f"Running MMLU Benchmark ({len(questions)} questions)")
        print(f"Mode: {self._mode_label()}")
        print(f"{'='*70}\n")

        client = self._create_client()
        if client is None:
            print("ERROR: Failed to create client")
            return None

        results = {
            "correct": 0,
            "incorrect": 0,
            "errors": 0,
            "details": [],
            "response_times": []
        }

        start_time = time.time()

        try:
            for i, q in enumerate(questions, 1):
                if self.debug:
                    print(f"[DEBUG] Starting question {i}...")
                loop_start = time.time()

                # Format question with choices
                question_text = f"{q['question']}\n\nChoices:\n"
                for idx, choice in enumerate(q['choices']):
                    question_text += f"{chr(65+idx)}. {choice}\n"
                question_text += "\nAnswer with ONLY the single letter (A, B, C, or D)."

                # Print question with choices before sending (so streaming output follows it)
                print(f"[{i}/{len(questions)}] Question: {clean_latex(q['question'])}")
                for idx, choice in enumerate(q['choices']):
                    print(f"    {chr(65+idx)}. {clean_latex(choice)}")
                print(f"  Answer: ", end="")

                if self.debug:
                    print(f"    [DEBUG] Full prompt:\n{question_text}")
                    print(f"    [DEBUG] Sending question... ({(time.time()-loop_start)*1000:.0f}ms)")

                response, resp_time = client.send_question(question_text)

                # For subprocess mode, print response (API mode streams it during send_question)
                if (self.provider or self.claude_cli) and response:
                    # Print with indentation for continuation lines, clean up LaTeX
                    cleaned = latex_to_text(response.strip())
                    lines = cleaned.split('\n')
                    for j, line in enumerate(lines):
                        if j == 0:
                            print(line)
                        else:
                            print(f"    {line}")

                if self.debug:
                    print(f"    [DEBUG] Got response ({(time.time()-loop_start)*1000:.0f}ms)")
                results["response_times"].append(resp_time)

                if self.debug:
                    print(f"    [DEBUG] Extracting answer... ({(time.time()-loop_start)*1000:.0f}ms)")
                extracted_answer = self.extract_answer(response, q['choices'])
                correct_answer = q['answer']

                detail = {
                    "question_num": i,
                    "subject": q['subject'],
                    "question": q['question'],
                    "correct_answer": correct_answer,
                    "extracted_answer": extracted_answer,
                    "is_correct": extracted_answer == correct_answer,
                    "response_time_ms": resp_time,
                    "response": response[:200]  # Store truncated response
                }

                if extracted_answer is None:
                    results["errors"] += 1
                    detail["status"] = "ERROR"
                    print(f"  ! ERROR: Could not extract answer")
                elif extracted_answer == correct_answer:
                    results["correct"] += 1
                    detail["status"] = "CORRECT"
                    print(f"  ✓ CORRECT")
                else:
                    results["incorrect"] += 1
                    detail["status"] = "INCORRECT"
                    print(f"  ✗ INCORRECT (got {extracted_answer}, expected {correct_answer})")

                if self.verbose:
                    print(f"    Response time: {resp_time:.0f}ms")
                    print(f"    Response: {response[:300]}")

                results["details"].append(detail)
                if self.debug:
                    print(f"[DEBUG] Loop done ({(time.time()-loop_start)*1000:.0f}ms)")

        finally:
            self._cleanup_client(client)

        duration_ms = (time.time() - start_time) * 1000
        total = len(questions)
        accuracy = (results["correct"] / total * 100) if total > 0 else 0
        avg_time = sum(results["response_times"]) / len(results["response_times"]) if results["response_times"] else 0

        return BenchmarkResult(
            provider=self._mode_label(),
            benchmark_type="MMLU",
            total_questions=total,
            correct=results["correct"],
            incorrect=results["incorrect"],
            errors=results["errors"],
            accuracy_percent=accuracy,
            avg_response_time_ms=avg_time,
            duration_ms=duration_ms,
            details=results["details"]
        )

    def run_hellaswag_benchmark(self, count: int = None) -> BenchmarkResult:
        """Run HellaSwag benchmark questions"""
        questions = HELLASWAG_QUESTIONS[:count] if count else HELLASWAG_QUESTIONS

        print(f"\n{'='*70}")
        print(f"Running HellaSwag Benchmark ({len(questions)} questions)")
        print(f"Mode: {self._mode_label()}")
        print(f"{'='*70}\n")

        client = self._create_client()
        if client is None:
            print("ERROR: Failed to create client")
            return None

        results = {
            "correct": 0,
            "incorrect": 0,
            "errors": 0,
            "details": [],
            "response_times": []
        }

        start_time = time.time()

        try:
            for i, q in enumerate(questions, 1):
                # Format question
                question_text = f"{q['context']}\n\nChoices:\n"
                for idx, choice in enumerate(q['choices']):
                    question_text += f"{chr(65+idx)}. {choice}\n"
                question_text += "\nAnswer with ONLY the single letter (A, B, C, or D)."

                # Print question with choices before sending
                print(f"[{i}/{len(questions)}] Question: {clean_latex(q['context'])}")
                for idx, choice in enumerate(q['choices']):
                    print(f"    {chr(65+idx)}. {clean_latex(choice)}")
                print(f"  Answer: ", end="")

                response, resp_time = client.send_question(question_text)

                # For subprocess mode, print response (API mode streams it during send_question)
                if (self.provider or self.claude_cli) and response:
                    lines = response.strip().split('\n')
                    for j, line in enumerate(lines):
                        if j == 0:
                            print(line)
                        else:
                            print(f"    {line}")

                results["response_times"].append(resp_time)

                extracted_answer = self.extract_answer(response, q['choices'])
                correct_answer = q['answer']

                detail = {
                    "question_num": i,
                    "context": q['context'],
                    "correct_answer": correct_answer,
                    "extracted_answer": extracted_answer,
                    "is_correct": extracted_answer == correct_answer,
                    "response_time_ms": resp_time,
                    "response": response[:200]
                }

                if extracted_answer is None:
                    results["errors"] += 1
                    detail["status"] = "ERROR"
                    print(f"  ! ERROR: Could not extract answer")
                elif extracted_answer == correct_answer:
                    results["correct"] += 1
                    detail["status"] = "CORRECT"
                    print(f"  ✓ CORRECT")
                else:
                    results["incorrect"] += 1
                    detail["status"] = "INCORRECT"
                    print(f"  ✗ INCORRECT (got {extracted_answer}, expected {correct_answer})")

                if self.verbose:
                    print(f"    Response time: {resp_time:.0f}ms")

                results["details"].append(detail)

        finally:
            self._cleanup_client(client)

        duration_ms = (time.time() - start_time) * 1000
        total = len(questions)
        accuracy = (results["correct"] / total * 100) if total > 0 else 0
        avg_time = sum(results["response_times"]) / len(results["response_times"]) if results["response_times"] else 0

        return BenchmarkResult(
            provider=self._mode_label(),
            benchmark_type="HellaSwag",
            total_questions=total,
            correct=results["correct"],
            incorrect=results["incorrect"],
            errors=results["errors"],
            accuracy_percent=accuracy,
            avg_response_time_ms=avg_time,
            duration_ms=duration_ms,
            details=results["details"]
        )

    def run_swebench_benchmark(self, count: int = None, difficulty: str = None, repos: List[str] = None) -> BenchmarkResult:
        """Run SWE-bench Verified benchmark tasks"""
        tasks = load_swebench_tasks(count=count, difficulty=difficulty, repos=repos)

        print(f"\n{'='*70}")
        print(f"Running SWE-bench Verified Benchmark ({len(tasks)} tasks)")
        print(f"{'='*70}\n")

        results = {
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "details": [],
            "task_times": []
        }

        start_time = time.time()

        for i, task in enumerate(tasks, 1):
            print(f"\n[{i}/{len(tasks)}] {task['instance_id']}")
            print(f"  Repo: {task['repo']}")
            print(f"  Difficulty: {task['difficulty']}")
            if self.verbose:
                print(f"\n  FULL PROBLEM STATEMENT:")
                print(f"  {'-'*70}")
                print(f"  {task['problem_statement']}")
                print(f"  {'-'*70}\n")
            else:
                print(f"  Problem: {task['problem_statement'][:100]}...")

            task_start = time.time()

            # Run task in isolated environment
            passed, error_msg = self._run_isolated_task(task)
            task_time = (time.time() - task_start) * 1000
            results["task_times"].append(task_time)

            detail = {
                "task_num": i,
                "instance_id": task['instance_id'],
                "repo": task['repo'],
                "difficulty": task['difficulty'],
                "passed": passed,
                "error": error_msg,
                "time_ms": task_time
            }

            if error_msg:
                results["errors"] += 1
                detail["status"] = "ERROR"
                print(f"  ✗ ERROR: {error_msg}")
            elif passed:
                results["passed"] += 1
                detail["status"] = "PASSED"
                print(f"  ✓ PASSED")
            else:
                results["failed"] += 1
                detail["status"] = "FAILED"
                print(f"  ✗ FAILED: Tests did not pass")

            if self.verbose:
                print(f"    Task time: {task_time/1000:.1f}s")

            results["details"].append(detail)

        duration_ms = (time.time() - start_time) * 1000
        total = len(tasks)
        success_rate = (results["passed"] / total * 100) if total > 0 else 0
        avg_time = sum(results["task_times"]) / len(results["task_times"]) if results["task_times"] else 0

        return BenchmarkResult(
            provider=self._mode_label(),
            benchmark_type="SWE-bench",
            total_questions=total,
            correct=results["passed"],
            incorrect=results["failed"],
            errors=results["errors"],
            accuracy_percent=success_rate,
            avg_response_time_ms=avg_time,
            duration_ms=duration_ms,
            details=results["details"]
        )

    def _run_isolated_task(self, task: Dict) -> Tuple[bool, Optional[str]]:
        """
        Run a single SWE-bench task in an isolated temporary directory.

        Returns:
            (passed, error_msg): True if tests passed, error message if failed
        """
        work_dir = None

        try:
            # Create isolated temporary directory
            work_dir = tempfile.mkdtemp(prefix=f"swebench_{task['instance_id']}_")
            if self.verbose:
                print(f"    Work dir: {work_dir}")

            # Get repository URL
            repo_name = task['repo']
            if repo_name not in SWEBENCH_REPOS:
                return False, f"Unknown repository: {repo_name}"

            repo_url = SWEBENCH_REPOS[repo_name]
            repo_dir = os.path.join(work_dir, repo_name.split('/')[-1])

            # Clone repository
            if self.verbose:
                print(f"    Cloning {repo_url}...")

            clone_result = subprocess.run(
                ["git", "clone", repo_url, repo_dir],
                capture_output=True,
                text=True,
                timeout=300
            )

            if clone_result.returncode != 0:
                return False, f"Clone failed: {clone_result.stderr[:200]}"

            # Checkout base commit
            if self.verbose:
                print(f"    Checking out commit {task['base_commit'][:8]}...")

            checkout_result = subprocess.run(
                ["git", "checkout", task['base_commit']],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=60
            )

            if checkout_result.returncode != 0:
                return False, f"Checkout failed: {checkout_result.stderr[:200]}"

            # Apply test patch to set up test environment
            if task['test_patch']:
                if self.verbose:
                    print(f"    Applying test patch...")

                test_patch_file = os.path.join(work_dir, "test.patch")
                with open(test_patch_file, 'w') as f:
                    f.write(task['test_patch'])

                patch_result = subprocess.run(
                    ["git", "apply", test_patch_file],
                    cwd=repo_dir,
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if patch_result.returncode != 0:
                    if self.verbose:
                        print(f"    Warning: Test patch failed: {patch_result.stderr[:100]}")

            # Start Shepherd in the repo directory with safety wrapper
            if self.verbose:
                print(f"    Starting Shepherd with safety restrictions...")

            shepherd_base_cmd = [
                SHEPHERD_BINARY,
                "--system-prompt",
                f"You are an expert software engineer. Fix the following issue in the {repo_name} codebase:\n\n{task['problem_statement']}\n\nUse the available tools to read files, make changes, and test your solution. Stay focused on fixing this specific issue."
            ]

            if self.provider:
                shepherd_base_cmd.extend(["--provider", self.provider])

            # Wrap with safety script
            shepherd_cmd = [SHEPHERD_SAFETY_WRAPPER, repo_dir] + shepherd_base_cmd

            # Set up restricted environment
            env = os.environ.copy()
            env['HOME'] = work_dir  # Prevent access to real home directory
            env['SHEPHERD_WORKSPACE'] = repo_dir  # Hint at working directory
            # Remove potentially dangerous env vars
            for key in ['SSH_AUTH_SOCK', 'SSH_AGENT_PID', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']:
                env.pop(key, None)

            # Run Shepherd with timeout and restricted environment
            try:
                if self.verbose:
                    # Real-time output mode - don't capture, let it print directly
                    shepherd_proc = subprocess.Popen(
                        shepherd_cmd,
                        stdin=subprocess.PIPE,
                        text=True,
                        cwd=repo_dir,  # CRITICAL: Start in repo dir, not home
                        env=env,  # Use restricted environment
                        preexec_fn=os.setsid
                    )

                    # Send problem and wait
                    shepherd_proc.stdin.write(f"{task['problem_statement']}\n\nPlease fix this issue.\nexit\n")
                    shepherd_proc.stdin.flush()
                    shepherd_proc.wait(timeout=SWEBENCH_TIMEOUT_SECONDS)
                    stdout = ""
                    stderr = ""
                else:
                    # Capture mode
                    shepherd_proc = subprocess.Popen(
                        shepherd_cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        cwd=repo_dir,  # CRITICAL: Start in repo dir, not home
                        env=env,  # Use restricted environment
                        preexec_fn=os.setsid
                    )

                    # Send "exit" after timeout or when done
                    stdout, stderr = shepherd_proc.communicate(
                        input=f"{task['problem_statement']}\n\nPlease fix this issue.\nexit\n",
                        timeout=SWEBENCH_TIMEOUT_SECONDS
                    )

            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(shepherd_proc.pid), signal.SIGKILL)
                except:
                    pass
                return False, "Timeout: Task took longer than 10 minutes"

            # Run tests to validate solution
            if self.verbose:
                print(f"    Running tests...")

            # Parse FAIL_TO_PASS tests
            tests_to_check = self._parse_test_list(task['FAIL_TO_PASS'])

            if not tests_to_check:
                return False, "No tests specified to validate"

            # Find python3 executable (use system python, not virtualenv)
            python_exe = shutil.which('python3') or shutil.which('python') or '/usr/bin/python3'

            # Run pytest on the specified tests
            test_cmd = [python_exe, "-m", "pytest", "-xvs"] + tests_to_check

            test_result = subprocess.run(
                test_cmd,
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=300,
                env=os.environ.copy()  # Use full environment for test execution
            )

            # Check if tests passed
            passed = test_result.returncode == 0

            if self.verbose and not passed:
                print(f"    Test output: {test_result.stdout[-500:]}")

            return passed, None

        except Exception as e:
            return False, f"Exception: {str(e)[:200]}"

        finally:
            # Cleanup: Remove temporary directory
            if work_dir and os.path.exists(work_dir):
                try:
                    shutil.rmtree(work_dir)
                    if self.verbose:
                        print(f"    Cleaned up {work_dir}")
                except Exception as e:
                    print(f"    Warning: Failed to cleanup {work_dir}: {e}")

    def _parse_test_list(self, test_string: str) -> List[str]:
        """Parse test specification string into list of test paths"""
        if not test_string:
            return []

        # Test strings are JSON arrays like: '["test_file.py::test_name"]'
        try:
            import json
            tests = json.loads(test_string)
            return tests if isinstance(tests, list) else [str(tests)]
        except:
            # Fallback: split by newlines/commas
            return [t.strip() for t in test_string.replace(',', '\n').split('\n') if t.strip()]

# ============================================================================
# Main
# ============================================================================

def print_summary(result: BenchmarkResult):
    """Print test summary"""
    print(f"\n{'='*70}")
    print(f"{result.benchmark_type} BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"Provider: {result.provider}")
    print(f"Total Questions: {result.total_questions}")
    print(f"Correct: {result.correct}")
    print(f"Incorrect: {result.incorrect}")
    print(f"Errors: {result.errors}")
    print(f"Accuracy: {result.accuracy_percent:.1f}%")
    print(f"Avg Response Time: {result.avg_response_time_ms:.0f}ms")
    print(f"Total Duration: {result.duration_ms/1000:.1f}s")
    print(f"{'='*70}\n")

def save_report(result: BenchmarkResult, output_file: str):
    """Save JSON report"""
    with open(output_file, 'w') as f:
        json.dump(asdict(result), f, indent=2)
    print(f"Report saved: {output_file}")

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test shepherd accuracy on AI benchmarks (MMLU, HellaSwag, SWE-bench)"
    )
    parser.add_argument("--baseurl",
                       default=DEFAULT_BASE_URL,
                       help=f"OpenAI API base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--model",
                       default=None,
                       help="Model name for API calls (default: auto-detect via /v1/models)")
    parser.add_argument("--list-models",
                       action="store_true",
                       help="List models advertised by the server and exit")
    parser.add_argument("--provider", "-p",
                       help="Use subprocess mode with this provider (legacy)")
    parser.add_argument("--claude-cli", action="store_true",
                       help="Test the actual `claude` CLI (Claude Code) directly instead of an "
                            "OpenAI-compatible server or shepherd subprocess. Full Claude Code "
                            "behavior (CLAUDE.md, hooks, memory, tools). --model is passed through "
                            "as `claude --model <value>`; omit --model to use claude's current "
                            "default. Not supported for --benchmark swebench.")
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default=None,
                       help="Reasoning effort, --claude-cli only (default: claude's current "
                            "default, not forced)")
    parser.add_argument("--benchmark", "-b",
                       choices=["mmlu", "hellaswag", "swebench", "all"],
                       default="all",
                       help="Which benchmark to run")
    parser.add_argument("--count", "-n",
                       type=int,
                       help="Number of questions/tasks to test (default: all)")
    parser.add_argument("--subjects",
                       nargs="+",
                       help="MMLU subjects to test (e.g., --subjects computer_science machine_learning)")
    parser.add_argument("--seed",
                       type=int,
                       help="Random seed for question/task sampling (use the same value to select the identical subset every run)")
    parser.add_argument("--list-subjects",
                       action="store_true",
                       help="List all available MMLU subjects and exit")
    parser.add_argument("--difficulty",
                       choices=["easy", "medium", "hard", "very hard"],
                       help="SWE-bench difficulty filter")
    parser.add_argument("--repos",
                       nargs="+",
                       help="SWE-bench repositories to test (e.g., --repos django/django flask/flask)")
    parser.add_argument("--output", "-o",
                       help="Output JSON file for results")
    parser.add_argument("--verbose", "-v",
                       action="store_true",
                       help="Verbose output")
    parser.add_argument("--debug",
                       action="store_true",
                       help="Enable debug output")
    parser.add_argument("--nothink",
                       action="store_true",
                       help="Send /nothink command before starting tests")
    parser.add_argument("--refresh",
                       action="store_true",
                       help="Force refresh of MMLU dataset from Hugging Face")
    parser.add_argument("--temperature", "--temp",
                       type=float,
                       help="Sampling temperature (e.g., 0.7)")
    parser.add_argument("--top_p",
                       type=float,
                       help="Top-p (nucleus) sampling (e.g., 0.9)")
    parser.add_argument("--top_k",
                       type=int,
                       help="Top-k sampling (e.g., 40)")
    parser.add_argument("--max_tokens",
                       type=int,
                       help="Maximum tokens to generate")
    parser.add_argument("--rep",
                       type=float,
                       help="Repetition penalty (e.g., 1.1)")
    parser.add_argument("--freq",
                       type=float,
                       help="Frequency penalty (e.g., 0.0)")

    args = parser.parse_args()

    # Seed the RNG so --count sampling is reproducible across runs
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    if args.claude_cli and args.provider:
        print("ERROR: --claude-cli and --provider are mutually exclusive")
        sys.exit(1)

    if args.effort and not args.claude_cli:
        print("ERROR: --effort is only supported with --claude-cli")
        sys.exit(1)

    if args.claude_cli and args.list_models:
        print("claude-cli has no fixed model list -- pass --model <alias-or-full-name> "
              "(e.g. opus, sonnet, haiku) or omit --model to use claude's current default.")
        sys.exit(0)

    if args.claude_cli and args.benchmark == "swebench":
        print("ERROR: --claude-cli does not support --benchmark swebench "
              "(it drives shepherd directly in an isolated sandboxed clone; not wired up yet)")
        sys.exit(1)
    if args.claude_cli and args.benchmark == "all":
        print("NOTE: --claude-cli does not support SWE-bench yet; skipping it under --benchmark all")

    # Auto-detect or list models via /v1/models (skip in subprocess/provider/claude-cli mode)
    if not args.provider and not args.claude_cli and (args.list_models or args.model is None):
        try:
            models = fetch_server_models(args.baseurl)
        except Exception as e:
            print(f"ERROR: Failed to query {args.baseurl}/v1/models: {e}")
            sys.exit(1)
        if args.list_models:
            for m in models:
                print(m)
            sys.exit(0)
        if not models:
            print(f"ERROR: Server at {args.baseurl} returned no models")
            sys.exit(1)
        args.model = models[0]
        print(f"Auto-selected model: {args.model}")

    # Handle --list-subjects
    if args.list_subjects:
        print("Available MMLU subjects:")
        for i, subject in enumerate(MMLU_SUBJECTS, 1):
            print(f"  {i:2d}. {subject}")
        print(f"\nTotal: {len(MMLU_SUBJECTS)} subjects")
        print("\nUsage: --subjects college_computer_science machine_learning")
        sys.exit(0)

    # Subprocess/provider mode doesn't use --model. claude-cli leaves --model as None
    # (meaning: no --model flag passed to claude, it uses its own current default) unless
    # the caller explicitly set one -- it must never be forced to a placeholder.
    if args.model is None and not args.claude_cli:
        args.model = "default"

    # Check binary exists (only needed for subprocess mode)
    if args.provider and not os.path.exists(SHEPHERD_BINARY):
        print(f"ERROR: Shepherd binary not found: {SHEPHERD_BINARY}")
        sys.exit(1)

    # Validate subjects if specified
    if args.subjects:
        invalid_subjects = [s for s in args.subjects if s not in MMLU_SUBJECTS]
        if invalid_subjects:
            print(f"ERROR: Invalid subjects: {', '.join(invalid_subjects)}")
            print(f"Use --list-subjects to see valid options")
            sys.exit(1)

    runner = BenchmarkTestRunner(
        base_url=args.baseurl,
        model=args.model,
        provider=args.provider,
        claude_cli=args.claude_cli,
        effort=args.effort,
        verbose=args.verbose,
        debug=args.debug,
        nothink=args.nothink,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        rep=args.rep,
        freq=args.freq
    )

    try:
        results = []

        if args.benchmark in ["mmlu", "all"]:
            result = runner.run_mmlu_benchmark(count=args.count, subjects=args.subjects, refresh=args.refresh)
            if result:
                print_summary(result)
                results.append(result)

        if args.benchmark in ["hellaswag", "all"]:
            result = runner.run_hellaswag_benchmark(count=args.count)
            if result:
                print_summary(result)
                results.append(result)

        if args.benchmark in ["swebench", "all"] and not args.claude_cli:
            result = runner.run_swebench_benchmark(
                count=args.count,
                difficulty=args.difficulty,
                repos=args.repos
            )
            if result:
                print_summary(result)
                results.append(result)

        # Save reports
        if args.output:
            if len(results) == 1:
                save_report(results[0], args.output)
            else:
                # Save combined report
                combined = {
                    "timestamp": datetime.now().isoformat(),
                    "provider": provider,
                    "results": [asdict(r) for r in results]
                }
                with open(args.output, 'w') as f:
                    json.dump(combined, f, indent=2)
                print(f"Combined report saved: {args.output}")

        sys.exit(0)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
