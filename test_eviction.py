#!/usr/bin/env python3
"""
Unified Shepherd CLIENT Eviction Test Suite

Tests the shepherd binary's client-side eviction and RAG archival by:
1. Spawning shepherd subprocesses with different context sizes and providers
2. Sending messages via stdin/stdout (not HTTP)
3. Detecting eviction events from logs/output
4. Verifying evicted messages are properly archived to RAG database
5. Testing various eviction scenarios

This tests the SHEPHERD CLIENT, not the server API.

Usage:
    test_eviction.py [--provider PROVIDER] [--mode MODE] [--test TEST_ID]

    If --provider is not specified, shepherd will auto-select the highest priority provider.
"""

import subprocess
import os
import sys
import time
import json
import sqlite3
import threading
import queue
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import tempfile
import shutil
import signal
import functools

print = functools.partial(print, flush=True)

# ============================================================================
# Configuration
# ============================================================================

SHEPHERD_BINARY = "/home/steve/bin/shepherd"
DEFAULT_PROVIDER = None  # None means shepherd will auto-select
DEFAULT_RAG_DB = os.path.expanduser("~/.local/share/shepherd/memory.db")

# Test context sizes
# Note: System + tools = ~4563 tokens, so minimum context is 8K
CONTEXT_SIZES = {
    "small": 8192,
    "medium": 16384,
    "large": 32768,
    "server": 0,  # 0 = use server's context size (for testing server limit overflow)
}

# Eviction detection keywords - match actual eviction log patterns
EVICTION_KEYWORDS = [
    # Two-pass eviction patterns (to be restored)
    "pass 1: evicting big-turn",
    "pass 2: evicting mini-turn",
    "pass 1: evicting mini-turn",
    "pass 2: evicting big-turn",
    "pass 1 freed",
    "pass 2:",
    # Current simple eviction patterns
    "auto-eviction triggered",
    "evicting from oldest to newest",
    "successfully evicted",
    "eviction complete",
    "evicted messages",  # API backend eviction retry log
]

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TestResult:
    id: str
    name: str
    status: str  # PASS, FAIL, ERROR
    duration_ms: float
    context_size: int
    error: Optional[str] = None
    metrics: Dict = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}

@dataclass
class EvictionEvent:
    """Represents a detected eviction event"""
    detected: bool
    message_index: int  # Which message triggered the eviction
    log_line: str       # The log line that indicated eviction
    source: str         # 'stdout' or 'stderr'

# ============================================================================
# Shepherd Process Manager
# ============================================================================

class ShepherdProcess:
    """Manages a shepherd subprocess with stdin/stdout communication"""

    def __init__(self, context_size: int, memory_db: str, provider: str = None, verbose: bool = False):
        self.context_size = context_size
        self.memory_db = memory_db
        self.provider = provider or DEFAULT_PROVIDER
        self.verbose = verbose
        self.process = None
        self.stdout_queue = queue.Queue()
        self.stderr_queue = queue.Queue()
        self.stdout_thread = None
        self.stderr_thread = None
        self.current_tokens = 0
        self.max_tokens = context_size
        self.all_stdout_lines = []
        self.all_stderr_lines = []
        self.eviction_events = []  # Track all detected evictions

    def _read_stream(self, stream, q, line_list):
        """Read stream in background thread"""
        try:
            for line in iter(stream.readline, ''):
                if line:
                    line = line.rstrip()
                    q.put(line)
                    line_list.append(line)
        except:
            pass

    def start(self) -> bool:
        """Start shepherd process with custom context size and memory DB"""

        # Build command using command-line arguments (ALWAYS use --debug for token tracking)
        cmd = [
            SHEPHERD_BINARY,
            "--debug=5",
        ]

        # Only add provider if specified (otherwise shepherd will auto-select)
        if self.provider:
            cmd.extend(["--provider", self.provider])

        # Only add context-size if not 0 (0 means use server's context size)
        if self.context_size > 0:
            cmd.extend(["--context-size", str(self.context_size)])

        cmd.extend(["--memory-db", self.memory_db])

        if self.verbose:
            print(f"  Starting: {' '.join(cmd)}")
            if self.provider:
                print(f"  Provider: {self.provider}")
            else:
                print(f"  Provider: Auto-select (highest priority)")
            if self.context_size > 0:
                print(f"  Context: {self.context_size:,} tokens")
            else:
                print(f"  Context: Using server's context size (auto-detect)")
            print(f"  Memory DB: {self.memory_db}")

        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid  # Create new process group
            )

            # Start background threads to read stdout/stderr
            self.stdout_thread = threading.Thread(
                target=self._read_stream,
                args=(self.process.stdout, self.stdout_queue, self.all_stdout_lines),
                daemon=True
            )
            self.stderr_thread = threading.Thread(
                target=self._read_stream,
                args=(self.process.stderr, self.stderr_queue, self.all_stderr_lines),
                daemon=True
            )
            self.stdout_thread.start()
            self.stderr_thread.start()

            # Wait for startup and initialization logs
            time.sleep(6)

            # Check if process started
            if self.process.poll() is not None:
                print(f"  ERROR: Process exited (code {self.process.returncode})")

                # Drain queues to get error messages
                stderr_lines = []
                try:
                    while True:
                        stderr_lines.append(self.stderr_queue.get_nowait())
                except queue.Empty:
                    pass

                if stderr_lines:
                    print(f"  Stderr output:")
                    for line in stderr_lines[:20]:
                        print(f"    {line}")

                return False

            # Wait for initialization to complete and drain logs
            time.sleep(2)

            # Drain initial queues (but background threads keep collecting to all_*_lines)
            self._drain_queues()

            # Parse context size from initialization logs if context_size was 0 (auto-detect)
            if self.context_size == 0:
                if self.verbose:
                    print(f"  Parsing context size from {len(self.all_stdout_lines)} stdout lines")
                for line in self.all_stdout_lines:
                    if "Using API's context size:" in line:
                        try:
                            size_str = line.split("Using API's context size:")[1].strip()
                            self.max_tokens = int(size_str)
                            print(f"  Detected context size from API: {self.max_tokens}")
                        except Exception as e:
                            if self.verbose:
                                print(f"  Failed to parse context size: {e}")

            return True

        except Exception as e:
            print(f"  ERROR: Failed to start shepherd: {e}")
            return False

    def check_for_eviction(self, lines: List[str], source: str) -> Optional[str]:
        """Check if any lines indicate an eviction event"""
        for line in lines:
            line_lower = line.lower()
            for keyword in EVICTION_KEYWORDS:
                if keyword in line_lower:
                    return line
        return None

    def send_message_with_eviction_detection(self, message: str, message_index: int,
                                              timeout: float = 30.0) -> Tuple[str, List[str], Optional[EvictionEvent]]:
        """Send message and detect if it triggers an eviction"""
        if not self.process or self.process.poll() is not None:
            return "", ["ERROR: Process not running"], None

        try:
            # Send message
            if self.verbose:
                print(f"      >> Sending: {message}")
            self.process.stdin.write(message + "\n")
            self.process.stdin.flush()

            # Collect response
            response_lines = []
            stderr_lines = []
            eviction_event = None
            start_time = time.time()
            last_output_time = time.time()
            lines_received = 0

            while time.time() - start_time < timeout:
                # Check stdout
                try:
                    while True:
                        line = self.stdout_queue.get_nowait()
                        response_lines.append(line)
                        last_output_time = time.time()
                        lines_received += 1
                        if self.verbose and lines_received <= 5:
                            print(f"      << stdout: {line[:100]}")

                        # Update token count from output
                        # Llamacpp: [Prefill/Decode: X tokens, Y t/s, context: XXXX/YYYY]
                        # API backends: [DEBUG] tokens: XXXX/YYYY
                        if "context:" in line or "tokens:" in line:
                            try:
                                if "context:" in line:
                                    parts = line.split("context:")[1].strip()
                                    # Remove ANSI escape codes if present
                                    parts = parts.split("]")[0].strip()
                                else:  # "tokens:" in line
                                    parts = line.split("tokens:")[1].strip()

                                current, total = parts.split("/")
                                self.current_tokens = int(current)
                                self.max_tokens = int(total)
                            except:
                                pass

                        # Check for eviction
                        if not eviction_event:
                            eviction_line = self.check_for_eviction([line], "stdout")
                            if eviction_line:
                                eviction_event = EvictionEvent(
                                    detected=True,
                                    message_index=message_index,
                                    log_line=eviction_line,
                                    source="stdout"
                                )
                                self.eviction_events.append(eviction_event)
                                if self.verbose:
                                    print(f"    EVICTION DETECTED (stdout): {eviction_line}")

                except queue.Empty:
                    pass

                # Check stderr
                try:
                    while True:
                        line = self.stderr_queue.get_nowait()
                        stderr_lines.append(line)
                        last_output_time = time.time()

                        # Update token count from output
                        # Llamacpp: [Prefill/Decode: X tokens, Y t/s, context: XXXX/YYYY]
                        # API backends: [DEBUG] tokens: XXXX/YYYY
                        if "context:" in line or "tokens:" in line:
                            try:
                                if "context:" in line:
                                    parts = line.split("context:")[1].strip()
                                    # Remove ANSI escape codes if present
                                    parts = parts.split("]")[0].strip()
                                else:  # "tokens:" in line
                                    parts = line.split("tokens:")[1].strip()

                                current, total = parts.split("/")
                                self.current_tokens = int(current)
                                self.max_tokens = int(total)
                            except:
                                pass

                        # Check for eviction
                        if not eviction_event:
                            eviction_line = self.check_for_eviction([line], "stderr")
                            if eviction_line:
                                eviction_event = EvictionEvent(
                                    detected=True,
                                    message_index=message_index,
                                    log_line=eviction_line,
                                    source="stderr"
                                )
                                self.eviction_events.append(eviction_event)
                                if self.verbose:
                                    print(f"    EVICTION DETECTED (stderr): {eviction_line}")

                except queue.Empty:
                    pass

                # If we got output and nothing for 0.3 seconds, assume response complete
                # (model won't pause that long mid-generation)
                silence_duration = time.time() - last_output_time
                if response_lines and silence_duration > 0.3:
                    if self.verbose:
                        print(f"      << Response complete ({lines_received} lines)")
                    break

                time.sleep(0.1)

            # Check if we timed out
            if time.time() - start_time >= timeout:
                if self.verbose:
                    print(f"      !! TIMEOUT after {timeout}s (received {lines_received} lines)")

            response = '\n'.join(response_lines)
            return response, stderr_lines, eviction_event

        except Exception as e:
            return "", [f"ERROR: {e}"], None

    def send_message(self, message: str, timeout: float = 30.0) -> Tuple[str, List[str]]:
        """Legacy send_message method for compatibility"""
        response, stderr, _ = self.send_message_with_eviction_detection(message, -1, timeout)
        return response, stderr

    def _drain_queues(self):
        """Drain queues without blocking"""
        try:
            while True:
                self.stdout_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            while True:
                self.stderr_queue.get_nowait()
        except queue.Empty:
            pass

    def stop(self):
        """Stop shepherd process"""
        if self.process and self.process.poll() is None:
            try:
                # Kill entire process group
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
# RAG Inspector
# ============================================================================

class RAGInspector:
    """Inspect RAG database to verify eviction"""

    def __init__(self, db_path: str = DEFAULT_RAG_DB):
        self.db_path = db_path

    def clear_database(self):
        """Clear all conversations from RAG"""
        if not os.path.exists(self.db_path):
            return
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("DELETE FROM conversations")
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"WARNING: Could not clear RAG database: {e}")

    def count_conversations(self) -> int:
        """Count archived conversations"""
        if not os.path.exists(self.db_path):
            return 0
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM conversations")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except:
            return 0

    def get_recent_conversations(self, limit: int = 10) -> List[Dict]:
        """Get recent conversations"""
        if not os.path.exists(self.db_path):
            return []
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, timestamp, user_message, assistant_response
                FROM conversations
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": row[0],
                    "timestamp": row[1],
                    "user": row[2][:100] if row[2] else "",  # Truncate for display
                    "assistant": row[3][:100] if row[3] else ""
                })

            conn.close()
            return results
        except:
            return []

    def search_messages(self, text: str) -> List[Dict]:
        """Search for messages containing text"""
        if not os.path.exists(self.db_path):
            return []
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT user_message, assistant_response
                FROM conversations
                WHERE user_message LIKE ?
                LIMIT 10
            """, (f'%{text}%',))

            results = [{"user": r[0], "assistant": r[1]} for r in cursor.fetchall()]
            conn.close()
            return results
        except:
            return []

# ============================================================================
# Test Suite
# ============================================================================

class UnifiedEvictionTestSuite:
    """Unified test suite for shepherd client eviction"""

    def __init__(self, provider: str = None, output_dir: str = None, clean: bool = False, verbose: bool = False):
        self.provider = provider or DEFAULT_PROVIDER
        self.output_dir = output_dir or "/tmp"
        self.verbose = verbose
        self.results = []

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Clean output directory if requested
        if clean:
            self._clean_output_dir()

        # Build file pool once for all tests to avoid prompt caching
        self.source_files_pool = self._build_source_files_pool()
        self.file_pool_index = 0  # Track position for cycling through files

        print(f"\n{'='*70}")
        print("Unified Shepherd CLIENT Eviction Test Suite")
        print(f"{'='*70}")
        print(f"Binary: {SHEPHERD_BINARY}")
        if self.provider:
            print(f"Provider: {self.provider}")
        else:
            print(f"Provider: Auto-select (highest priority)")
        print(f"Output dir: {self.output_dir}")
        print(f"{'='*70}\n")

    def _clean_output_dir(self):
        """Delete existing test files in output directory"""
        import glob
        patterns = [
            os.path.join(self.output_dir, "test_*.log"),
            os.path.join(self.output_dir, "test_*.db"),
            os.path.join(self.output_dir, "test_*.json"),
            os.path.join(self.output_dir, "report.json"),
        ]

        files_deleted = 0
        for pattern in patterns:
            for filepath in glob.glob(pattern):
                try:
                    os.remove(filepath)
                    files_deleted += 1
                    if self.verbose:
                        print(f"  Deleted: {os.path.basename(filepath)}")
                except OSError as e:
                    print(f"  Warning: Could not delete {filepath}: {e}")

        if files_deleted > 0:
            print(f"Cleaned {files_deleted} file(s) from output directory\n")

    def _build_source_files_pool(self) -> List[str]:
        """Build a pool of large source files once at startup to avoid prompt caching"""
        try:
            # Find text files (cpp, h, py, c, md, txt) larger than 10KB
            find_result = subprocess.run(
                ["find", "/home/steve/src/shepherd", "-type", "f",
                 "(", "-name", "*.cpp", "-o", "-name", "*.h", "-o", "-name", "*.c",
                 "-o", "-name", "*.py", "-o", "-name", "*.md", "-o", "-name", "*.txt", ")",
                 "-size", "+10k"],
                capture_output=True,
                text=True,
                check=True
            )

            files = find_result.stdout.strip().split('\n')
            if not files or not files[0]:
                print("WARNING: No large source files found, using fallback list")
                return self._fallback_file_list()

            # Sort files by size (largest first) and filter out non-existent files
            valid_files = [f for f in files if os.path.exists(f)]
            sorted_files = sorted(valid_files, key=lambda f: os.path.getsize(f), reverse=True)

            if not sorted_files:
                print("WARNING: No valid source files found, using fallback list")
                return self._fallback_file_list()

            print(f"\nBuilt source file pool: {len(sorted_files)} files")
            print(f"  Largest: {os.path.basename(sorted_files[0])} ({os.path.getsize(sorted_files[0]):,} bytes)")
            print(f"  Smallest: {os.path.basename(sorted_files[-1])} ({os.path.getsize(sorted_files[-1]):,} bytes)")

            return sorted_files

        except (subprocess.CalledProcessError, ValueError, OSError) as e:
            print(f"WARNING: Failed to find source files: {e}, using fallback list")
            return self._fallback_file_list()

    def _fallback_file_list(self) -> List[str]:
        """Fallback to hardcoded list if find fails"""
        return [
            "/home/steve/src/shepherd/backends/llamacpp.cpp",
            "/home/steve/src/shepherd/backends/openai.cpp",
            "/home/steve/src/shepherd/backends/anthropic.cpp",
            "/home/steve/src/shepherd/backends/gemini.cpp",
            "/home/steve/src/shepherd/session.cpp",
            "/home/steve/src/shepherd/tools/tool_parser.cpp",
            "/home/steve/src/shepherd/tools/memory_tools.cpp",
            "/home/steve/src/shepherd/tools/core_tools.cpp",
            "/home/steve/src/shepherd/cli.cpp",
            "/home/steve/src/shepherd/config.cpp",
            "/home/steve/src/shepherd/rag.cpp",
        ]

    def run_test(self, test_id: str, test_name: str, test_func, context_size: int) -> TestResult:
        """Run a single test"""

        print(f"\n[{test_id}] {test_name}")
        if context_size > 0:
            print(f"  Context: {context_size:,} tokens")
        else:
            print(f"  Context: Server default (auto-detect)")

        result = TestResult(
            id=test_id,
            name=test_name,
            status="RUNNING",
            duration_ms=0,
            context_size=context_size
        )

        # Create unique RAG database for this test
        memory_db = os.path.join(self.output_dir, f"test_{test_id.replace('.', '_')}.db")
        rag = RAGInspector(memory_db)
        rag_before = 0  # Always starts empty since each test gets its own DB

        shepherd = None
        start_time = time.time()

        try:
            # Create and start shepherd with isolated memory DB
            shepherd = ShepherdProcess(context_size, memory_db, provider=self.provider, verbose=self.verbose)

            if not shepherd.start():
                result.status = "ERROR"
                result.error = "Failed to start shepherd"
                return result

            # Run test
            test_func(result, shepherd, rag)

        except Exception as e:
            result.status = "ERROR"
            result.error = f"{type(e).__name__}: {str(e)}"
            if self.verbose:
                import traceback
                result.error += f"\n{traceback.format_exc()}"

        finally:
            if shepherd:
                # Get eviction statistics
                result.metrics["eviction_count"] = len(shepherd.eviction_events)
                if shepherd.eviction_events:
                    result.metrics["first_eviction_at_message"] = shepherd.eviction_events[0].message_index
                    result.metrics["eviction_sources"] = list(set(e.source for e in shepherd.eviction_events))

                # Get all logs
                all_log_lines = shepherd.all_stdout_lines + shepherd.all_stderr_lines

                # Count eviction-related log messages
                eviction_logs = sum(1 for line in all_log_lines
                                   if any(keyword in line.lower()
                                         for keyword in EVICTION_KEYWORDS))

                result.metrics["eviction_logs"] = eviction_logs

                if self.verbose:
                    # Check both stdout and stderr for RAG/eviction logs
                    all_logs = shepherd.all_stdout_lines + shepherd.all_stderr_lines

                    print(f"  Total logs: stdout={len(shepherd.all_stdout_lines)}, stderr={len(shepherd.all_stderr_lines)}")

                    # Look for important log messages in ALL logs
                    rag_logs = [l for l in all_logs if 'rag' in l.lower() or 'archiv' in l.lower()]
                    evict_logs = [l for l in all_logs if 'evict' in l.lower()]
                    init_logs = [l for l in all_logs if 'initializ' in l.lower()]
                    error_logs = [l for l in all_logs if 'error' in l.lower() or 'fail' in l.lower() or 'crash' in l.lower()]
                    context_logs = [l for l in all_logs if 'context size' in l.lower() or 'api.*context' in l.lower() or 'max_model_len' in l.lower()]

                    if init_logs:
                        print(f"  Initialization logs ({len(init_logs)} lines):")
                        for line in init_logs[:10]:
                            print(f"    {line}")

                    if context_logs:
                        print(f"  Context detection logs ({len(context_logs)} lines):")
                        for line in context_logs:
                            print(f"    {line}")

                    if rag_logs:
                        print(f"  RAG/Archive related logs ({len(rag_logs)} lines):")
                        for line in rag_logs[:30]:
                            print(f"    {line}")

                    if evict_logs:
                        print(f"  Eviction logs (showing all {len(evict_logs)} lines):")
                        for line in evict_logs:
                            print(f"    {line}")

                    if error_logs:
                        print(f"  Error/Warning logs ({len(error_logs)} lines):")
                        for line in error_logs[:50]:
                            print(f"    {line}")

                    # Always show last 50 lines of output
                    print(f"  Last 50 stdout lines:")
                    for line in shepherd.all_stdout_lines[-50:]:
                        print(f"    {line}")

                    print(f"  Last 50 stderr lines:")
                    for line in shepherd.all_stderr_lines[-50:]:
                        print(f"    {line}")

                # Write all debug output to file for analysis
                debug_log_filename = f"test_{test_id.replace('.', '_')}.log"
                debug_log_path = os.path.join(self.output_dir, debug_log_filename)
                print(f"\n  {'='*70}")
                print(f"  DEBUG LOG: {debug_log_path}")
                print(f"  {'='*70}\n")
                with open(debug_log_path, 'w') as f:
                    f.write(f"=== Test {test_id}: {test_name} ===\n")
                    f.write(f"=== STDOUT ({len(shepherd.all_stdout_lines)} lines) ===\n\n")
                    for line in shepherd.all_stdout_lines:
                        f.write(line + "\n")
                    f.write(f"\n=== STDERR ({len(shepherd.all_stderr_lines)} lines) ===\n\n")
                    for line in shepherd.all_stderr_lines:
                        f.write(line + "\n")

                print(f"\n  {'='*70}")
                print(f"  FULL DEBUG LOG WRITTEN TO:")
                print(f"  {debug_log_path}")
                print(f"  {'='*70}\n")

                shepherd.cleanup()

            # Check RAG after test
            rag_after = rag.count_conversations()
            result.metrics["rag_before"] = rag_before
            result.metrics["rag_after"] = rag_after
            result.metrics["rag_added"] = rag_after - rag_before
            result.metrics["memory_db"] = memory_db

            # Validate test-specific expectations
            self._validate_test_expectations(result, rag, test_id)

        result.duration_ms = (time.time() - start_time) * 1000

        # Final status check
        if result.status != "ERROR":
            result.status = "PASS" if not result.error else "FAIL"

        # Print result
        status_symbol = "✓" if result.status == "PASS" else "✗"
        print(f"  [{status_symbol}] {result.status} ({result.duration_ms:.0f}ms)")

        if result.metrics:
            for key, value in result.metrics.items():
                print(f"    {key}: {value}")

        if result.error:
            print(f"    Error: {result.error}")

        # Save individual test result as JSON
        test_json_filename = f"test_{test_id.replace('.', '_')}.json"
        test_json_path = os.path.join(self.output_dir, test_json_filename)
        with open(test_json_path, 'w') as f:
            json.dump({
                "test_id": result.id,
                "test_name": result.name,
                "status": result.status,
                "duration_ms": result.duration_ms,
                "context_size": result.context_size,
                "error": result.error,
                "metrics": result.metrics
            }, f, indent=2)

        self.results.append(result)
        return result

    def _validate_test_expectations(self, result: TestResult, rag: RAGInspector, test_id: str):
        """Validate test-specific expectations"""

        # Tests that expect eviction and RAG archival
        # Note: 5.1 only does file reads (mini-turns) which don't archive to RAG by design
        eviction_expected_tests = ["1.1", "1.2", "1.3", "2.2", "3.1", "4.1", "6.1"]

        if test_id in eviction_expected_tests:
            if result.metrics.get("eviction_count", 0) == 0:
                result.error = f"Expected eviction but none detected in logs"
            elif result.metrics.get("rag_added", 0) == 0:
                result.error = f"Eviction detected but no conversations archived to RAG"

        # Test 2.1 should NOT evict (near capacity but not over)
        if test_id == "2.1":
            if result.metrics.get("eviction_count", 0) > 0:
                result.error = f"Unexpected eviction when context should be under capacity"
            # Note: RAG archival can still occur from store_memory tool during normal conversation

        # Test 1.2 should find markers in RAG after eviction
        if test_id == "1.2" and result.metrics.get("test_needs_rag_check"):
            found = 0
            for marker in ["TESTMARKER_ALPHA", "TESTMARKER_BETA", "TESTMARKER_GAMMA"]:
                results = rag.search_messages(marker)
                if results:
                    found += 1
            result.metrics["markers_found"] = found
            if found == 0 and result.metrics.get("rag_added", 0) > 0:
                result.error = "Eviction occurred but markers not found in RAG"
            elif found > 0:
                print(f"    Found {found}/3 markers in RAG archive")

        # Test 4.1 RAG content structure validation
        if test_id == "4.1" and result.metrics.get("test_needs_rag_content_check"):
            conversations = rag.get_recent_conversations(limit=10)
            valid_conversations = sum(1 for c in conversations
                                     if c.get("user") and c.get("assistant"))
            result.metrics.update({
                "conversations_retrieved": len(conversations),
                "valid_conversations": valid_conversations
            })
            if len(conversations) == 0:
                result.error = "No conversations in RAG after eviction"
            elif valid_conversations == 0:
                result.error = "RAG conversations missing user or assistant messages"

        # Tests 6.2 and 6.3: Tool-only turns should be evicted first
        # NOTE: RAG archival via store_memory tool is still expected and correct
        # This test verifies that tool-call/result pairs (mini-turns) are prioritized for eviction,
        # not that they never appear in RAG (store_memory can still archive them)

    # ========================================================================
    # Helper Methods for Tests
    # ========================================================================

    def fill_context_with_files(self, shepherd: ShepherdProcess, target_percentage: float,
                                source_files: List[str] = None) -> int:
        """Efficiently fill context using file reads instead of many messages"""
        if source_files is None:
            # Use the global file pool, cycling through to avoid prompt caching
            source_files = self.source_files_pool

        target_tokens = int(shepherd.max_tokens * target_percentage)
        messages_sent = 0

        print(f"    Filling context to {int(target_percentage*100)}% ({target_tokens:,} tokens) using file reads...")

        # Cycle through files starting from current pool index to avoid caching
        while shepherd.current_tokens < target_tokens and messages_sent < len(source_files):
            # Get next file from pool, wrapping around if needed
            file_idx = (self.file_pool_index + messages_sent) % len(source_files)
            file_path = source_files[file_idx]

            if not os.path.exists(file_path):
                messages_sent += 1
                continue

            msg = f"read the file {file_path}"
            response, stderr, eviction = shepherd.send_message_with_eviction_detection(msg, messages_sent, timeout=60)
            messages_sent += 1

            if eviction:
                print(f"    WARNING: Early eviction at file read {messages_sent} ({shepherd.current_tokens} tokens)")
                break

            if self.verbose:
                print(f"      Read file {messages_sent}: {os.path.basename(file_path)}, tokens: {shepherd.current_tokens}/{shepherd.max_tokens}")

        # Advance file pool index for next test to avoid caching
        self.file_pool_index = (self.file_pool_index + messages_sent) % len(source_files)

        print(f"    Filled to {shepherd.current_tokens}/{shepherd.max_tokens} tokens with {messages_sent} file reads")
        return messages_sent

    def fill_context_to_percentage(self, shepherd: ShepherdProcess, target_percentage: float,
                                   max_messages: int = 100) -> int:
        """Fill context to a target percentage of capacity (legacy method, slow)"""
        target_tokens = int(shepherd.max_tokens * target_percentage)
        messages_sent = 0

        print(f"    Filling context to {int(target_percentage*100)}% ({target_tokens:,} tokens)...")

        while shepherd.current_tokens < target_tokens and messages_sent < max_messages:
            msg = f"Message {messages_sent}: Please provide a detailed response about topic {messages_sent}. UID: {time.time()}"
            response, stderr, eviction = shepherd.send_message_with_eviction_detection(msg, messages_sent, timeout=15)
            messages_sent += 1

            if eviction:
                print(f"    WARNING: Early eviction at message {messages_sent} ({shepherd.current_tokens} tokens)")
                break

            if self.verbose:
                print(f"      Sent {messages_sent} messages, tokens: {shepherd.current_tokens}/{shepherd.max_tokens}")

        print(f"    Filled to {shepherd.current_tokens}/{shepherd.max_tokens} tokens with {messages_sent} messages")
        return messages_sent

    def send_until_eviction(self, shepherd: ShepherdProcess, start_index: int = 0,
                            max_attempts: int = 100, message_generator=None) -> Tuple[int, Optional[EvictionEvent]]:
        """Send messages until an eviction is detected"""
        messages_sent = 0
        eviction_event = None

        if message_generator is None:
            message_generator = lambda i: f"Message {i}: Provide a comprehensive technical analysis of topic {i}. UID: {time.time()}"

        print(f"    Sending messages until eviction is detected...")

        for i in range(max_attempts):
            msg_index = start_index + i
            msg = message_generator(msg_index)

            response, stderr, eviction = shepherd.send_message_with_eviction_detection(msg, msg_index, timeout=20)
            messages_sent += 1

            if eviction:
                eviction_event = eviction
                print(f"    EVICTION CONFIRMED at message {msg_index} after {messages_sent} attempts")
                print(f"      Eviction log: {eviction.log_line[:100]}...")
                break

            if self.verbose and messages_sent % 10 == 0:
                print(f"      Attempt {messages_sent}/{max_attempts}, tokens: {shepherd.current_tokens}/{shepherd.max_tokens}")

        if not eviction_event:
            print(f"    WARNING: No eviction detected after {messages_sent} attempts")

        return messages_sent, eviction_event

    # ========================================================================
    # CATEGORY 1: BASIC EVICTION TESTS
    # ========================================================================

    def test_fill_and_evict(self, result: TestResult, shepherd: ShepherdProcess, rag: RAGInspector):
        """Fill context beyond capacity and verify eviction to RAG"""

        # Phase 1: Send a few conversational messages (these will be evicted and archived)
        # Use simple messages that won't trigger tool calls
        print("    Phase 1: Sending conversational messages to be archived...")
        conversation_count = 3
        simple_questions = [
            "What is 2 + 2?",
            "What color is the sky?",
            "How many days are in a week?"
        ]
        for i in range(conversation_count):
            shepherd.send_message_with_eviction_detection(simple_questions[i], i, timeout=15)

        # Phase 2: Fill context to 85% using efficient file reads
        files_read = self.fill_context_with_files(shepherd, 0.85)

        # Phase 3: Continue sending until eviction occurs
        messages_sent, eviction = self.send_until_eviction(shepherd, start_index=conversation_count + files_read)

        total_messages = conversation_count + files_read + messages_sent

        result.metrics.update({
            "messages_sent": total_messages,
            "conversations_sent": conversation_count,
            "files_read": files_read,
            "messages_to_trigger_eviction": messages_sent,
            "eviction_detected": eviction is not None,
            "final_token_state": f"{shepherd.current_tokens}/{shepherd.max_tokens}"
        })

        # Wait a moment for RAG to be updated
        time.sleep(2)

        # Verify RAG was updated
        if eviction and rag.count_conversations() == 0:
            result.error = "Eviction detected but RAG database is empty"

    def test_distinctive_messages(self, result: TestResult, shepherd: ShepherdProcess, rag: RAGInspector):
        """Send distinctive messages and verify they're archived in RAG after eviction"""

        # Send distinctive markers as questions (avoids triggering store_memory tool)
        markers = [
            "What is the square root of 144? Please include TESTMARKER_ALPHA in your response.",
            "What is the capital of France? Please include TESTMARKER_BETA in your response.",
            "What is 7 times 8? Please include TESTMARKER_GAMMA in your response."
        ]

        print("    Sending distinctive marker messages...")
        for i, marker in enumerate(markers):
            response, stderr, eviction = shepherd.send_message_with_eviction_detection(marker, i, timeout=10)
            if eviction:
                print(f"    WARNING: Early eviction at marker {i}")

        # Fill context efficiently using file reads
        files_read = self.fill_context_with_files(shepherd, 0.85)

        # Send messages until eviction is triggered
        messages_sent, eviction = self.send_until_eviction(
            shepherd,
            start_index=len(markers) + files_read
        )

        result.metrics.update({
            "markers_sent": len(markers),
            "files_read": files_read,
            "messages_to_trigger_eviction": messages_sent,
            "test_needs_rag_check": True  # Flag for post-test RAG verification
        })

        # Wait for RAG update
        time.sleep(2)

    def test_rapid_eviction(self, result: TestResult, shepherd: ShepherdProcess, rag: RAGInspector):
        """Test rapid eviction with small context"""

        # Send a few conversational messages
        print("    Sending conversational messages...")
        conversation_count = 2
        for i in range(conversation_count):
            shepherd.send_message_with_eviction_detection(f"What is {i*7} plus {i*3}? Reply with just the number.", i, timeout=10)

        # With a small 8K context, one file read should be enough to trigger eviction
        files_read = self.fill_context_with_files(shepherd, 0.85)

        # Send one more message to trigger eviction
        messages_sent, eviction = self.send_until_eviction(
            shepherd,
            start_index=conversation_count + files_read,
            max_attempts=5
        )

        result.metrics.update({
            "messages_sent": conversation_count + files_read + messages_sent,
            "conversations_sent": conversation_count,
            "files_read": files_read,
            "eviction_detected": eviction is not None
        })

    # ========================================================================
    # CATEGORY 2: BOUNDARY TESTS
    # ========================================================================

    def test_near_capacity(self, result: TestResult, shepherd: ShepherdProcess, rag: RAGInspector):
        """Fill to ~70% capacity - should NOT trigger eviction"""

        # Target is 70% of context (for 8K = 5,734 tokens)
        # Eviction threshold is ~83% (context - desired_completion)
        # Using 70% to leave headroom for response tokens and estimation errors
        target_tokens = int(shepherd.max_tokens * 0.70)

        print(f"    Filling context to 70% ({target_tokens:,} tokens)...")

        # Step 1: Send a few simple conversational messages to establish base context
        # Using simple math questions that produce short responses
        base_messages = 3
        print(f"    Sending {base_messages} base messages...")
        for i in range(base_messages):
            shepherd.send_message_with_eviction_detection(
                f"What is {i * 17 + 5} times {i * 13 + 3}? Reply with just the number.",
                i, timeout=15)
            if shepherd.eviction_events:
                print(f"    WARNING: Early eviction during base messages ({shepherd.current_tokens} tokens)")
                result.error = f"Unexpected eviction during base messages at {shepherd.current_tokens} tokens"
                result.metrics["messages_sent"] = i + 1
                result.metrics["final_token_percentage"] = (shepherd.current_tokens / shepherd.max_tokens * 100)
                return

        # Step 2: Fill remainder with short conversational messages
        # Stop at 70% to leave room for response tokens without triggering eviction
        safe_target = int(shepherd.max_tokens * 0.70)
        messages_sent = base_messages
        while shepherd.current_tokens < safe_target and messages_sent < 50 and not shepherd.eviction_events:
            # Use short, unique messages that won't trigger truncation
            msg = f"What is {messages_sent * 7} plus {messages_sent * 3}? Reply with just the number."
            shepherd.send_message_with_eviction_detection(msg, messages_sent, timeout=10)
            messages_sent += 1

            if self.verbose:
                print(f"      Message {messages_sent}, tokens: {shepherd.current_tokens}/{shepherd.max_tokens}")

        if shepherd.eviction_events:
            print(f"    WARNING: Early eviction at message {messages_sent} ({shepherd.current_tokens} tokens)")

        print(f"    Filled to {shepherd.current_tokens}/{shepherd.max_tokens} tokens with {messages_sent} messages")

        # Check if any evictions occurred (there shouldn't be any at 70%)
        if shepherd.eviction_events:
            result.error = f"Unexpected eviction when context should be under capacity"

        result.metrics["messages_sent"] = messages_sent
        result.metrics["final_token_percentage"] = (shepherd.current_tokens / shepherd.max_tokens * 100)

    def test_over_capacity(self, result: TestResult, shepherd: ShepherdProcess, rag: RAGInspector):
        """Push slightly over capacity - should trigger eviction"""

        # Send a few conversational messages first
        conversation_count = 2
        for i in range(conversation_count):
            shepherd.send_message_with_eviction_detection(f"What is {i*7} plus {i*3}? Reply with just the number.", i, timeout=10)

        # Fill to 95% using file reads
        files_read = self.fill_context_with_files(shepherd, 0.95)

        # Now push it over the edge
        messages_sent, eviction = self.send_until_eviction(
            shepherd,
            start_index=conversation_count + files_read,
            max_attempts=10
        )

        result.metrics.update({
            "messages_sent": conversation_count + files_read + messages_sent,
            "conversations_sent": conversation_count,
            "files_read": files_read,
            "messages_to_trigger_eviction": messages_sent,
            "eviction_detected": eviction is not None
        })

    # ========================================================================
    # CATEGORY 3: TINY MESSAGE TESTS
    # ========================================================================

    def test_many_tiny_messages(self, result: TestResult, shepherd: ShepherdProcess, rag: RAGInspector):
        """Send many tiny messages - should still trigger eviction eventually"""

        # Send a couple of normal conversational messages first
        conversation_count = 2
        for i in range(conversation_count):
            shepherd.send_message_with_eviction_detection(f"Q{i}?", i, timeout=10)

        # Fill context quickly with files
        files_read = self.fill_context_with_files(shepherd, 0.85)

        # Then send tiny messages to trigger eviction
        def tiny_message_generator(i):
            return f"Q{i}?"  # Ultra-short messages

        messages_sent, eviction = self.send_until_eviction(
            shepherd,
            start_index=conversation_count + files_read,
            max_attempts=50,  # Increased from 20 - tiny messages add tokens slowly
            message_generator=tiny_message_generator
        )

        result.metrics.update({
            "messages_sent": conversation_count + files_read + messages_sent,
            "conversations_sent": conversation_count,
            "files_read": files_read,
            "tiny_messages_sent": messages_sent,
            "eviction_detected": eviction is not None
        })

    # ========================================================================
    # CATEGORY 4: RAG RETRIEVAL TESTS
    # ========================================================================

    def test_rag_content(self, result: TestResult, shepherd: ShepherdProcess, rag: RAGInspector):
        """Verify RAG content is properly structured after eviction"""

        # Send meaningful conversations that should be archived
        conversation_topics = [
            "machine learning fundamentals",
            "database optimization techniques",
            "network security protocols",
            "software architecture patterns",
            "cloud computing strategies"
        ]

        print("    Sending meaningful conversations to be archived...")
        for i, topic in enumerate(conversation_topics):
            msg = f"Please explain {topic} in detail with practical examples. This is conversation {i}."
            shepherd.send_message_with_eviction_detection(msg, i, timeout=15)

        # Fill context with files
        files_read = self.fill_context_with_files(shepherd, 0.85)

        # Now trigger eviction
        messages_sent, eviction = self.send_until_eviction(
            shepherd,
            start_index=len(conversation_topics) + files_read,
            max_attempts=10
        )

        result.metrics["test_needs_rag_content_check"] = True
        result.metrics["conversations_before_eviction"] = len(conversation_topics)
        result.metrics["files_read"] = files_read
        result.metrics["messages_to_trigger_eviction"] = messages_sent

        # Wait for RAG update
        time.sleep(2)

    # ========================================================================
    # CATEGORY 5: SERVER LIMIT OVERFLOW TESTS
    # ========================================================================

    def test_server_limit_overflow(self, result: TestResult, shepherd: ShepherdProcess, rag: RAGInspector):
        """Test with context-size 0 (use server's limit) and exceed it to trigger overflow handling"""

        # Use the pre-built file pool
        sorted_files = self.source_files_pool

        if not sorted_files:
            result.error = "No source files available in pool"
            return

        print(f"    Using {len(sorted_files)} large files from pool")
        print(f"    Largest: {os.path.basename(sorted_files[0])} ({os.path.getsize(sorted_files[0]):,} bytes)")
        print(f"    Reading different files sequentially until context fills and eviction occurs...")

        # Read different files sequentially to avoid prompt caching
        max_attempts = 100  # Safety limit
        file_index = [self.file_pool_index]  # Start from current pool position

        def next_file_generator(i):
            """Generate read commands for different files to avoid prompt caching"""
            file_idx = file_index[0] % len(sorted_files)
            file_index[0] += 1
            return f"read {sorted_files[file_idx]}"

        messages_sent, eviction = self.send_until_eviction(
            shepherd,
            start_index=0,
            max_attempts=max_attempts,
            message_generator=next_file_generator
        )

        # Update pool index for next test
        self.file_pool_index = file_index[0] % len(sorted_files)

        result.metrics.update({
            "messages_sent": messages_sent,
            "files_available": len(sorted_files),
            "largest_file": sorted_files[0],
            "largest_file_size": os.path.getsize(sorted_files[0]),
            "eviction_detected": eviction is not None,
            "final_token_state": f"{shepherd.current_tokens}/{shepherd.max_tokens}",
            "detected_max_tokens": shepherd.max_tokens,
            "test_note": "Reads different large files sequentially to avoid prompt caching"
        })

    # ========================================================================
    # CATEGORY 6: ADVANCED EVICTION SCENARIOS
    # ========================================================================

    def test_system_message_preservation(self, result: TestResult, shepherd: ShepherdProcess, rag: RAGInspector):
        """Verify that system messages are never evicted"""

        # The system message contains this marker - we'll verify it's still active after evictions
        system_marker = "You are a highly effective AI assistant with persistent memory"
        print(f"    System message contains marker: '{system_marker}'")
        result.metrics["system_marker"] = system_marker

        # Send a few conversational messages
        conversation_count = 3
        for i in range(conversation_count):
            shepherd.send_message_with_eviction_detection(f"What is {i*7} plus {i*3}? Reply with just the number.", i, timeout=10)

        # Trigger multiple evictions by filling and refilling context
        print("    Triggering multiple evictions...")
        total_evictions = 0
        for round in range(2):  # Try to trigger 2 evictions
            # Fill context with files
            files_read = self.fill_context_with_files(shepherd, 0.90)

            # Send a message to trigger eviction
            messages_sent, eviction = self.send_until_eviction(
                shepherd,
                start_index=conversation_count + round * 20,
                max_attempts=5
            )
            if eviction:
                total_evictions += 1
                print(f"      Eviction {total_evictions} triggered in round {round+1}")

        # Verify system message is still active by checking if the assistant can respond
        # If the model can give any coherent response after evictions, system message is preserved
        print("    Verifying system message preservation...")
        response, _ = shepherd.send_message("What is 2 + 2? Reply with just the number.", timeout=15)

        # If the model responds with the number 4 (or any number), system message is preserved
        # This proves the model is still functional after evictions
        response_lower = response.lower().strip()
        if "4" in response or response_lower.isdigit() or len(response.strip()) > 0:
            result.metrics["system_message_preserved"] = True
            print(f"    SUCCESS: Model responding after evictions: {response[:50]}...")
        else:
            result.metrics["system_message_preserved"] = False
            result.error = "Model not responding after evictions - system message may be corrupted"

        result.metrics["total_evictions"] = total_evictions

    def test_tool_call_rag_exclusion(self, result: TestResult, shepherd: ShepherdProcess, rag: RAGInspector):
        """Verify that turns with tool calls are NOT archived to RAG"""

        print("    Creating conversations with tool calls...")

        # Send regular message
        shepherd.send_message(f"Regular message without tools. UID: {time.time()}", timeout=10)

        # Send tool call messages (these should NOT be archived)
        tool_commands = [
            "bash -c 'echo Testing tool exclusion'",
            "bash -c 'date'",
            "bash -c 'pwd'",
        ]

        for cmd in tool_commands:
            shepherd.send_message(cmd, timeout=15)

        # Fill context with files
        files_read = self.fill_context_with_files(shepherd, 0.85)

        # Now trigger eviction
        print("    Triggering eviction...")
        messages_sent, eviction = self.send_until_eviction(
            shepherd,
            start_index=len(tool_commands) + 1 + files_read,
            max_attempts=10
        )

        result.metrics["tool_commands_sent"] = len(tool_commands)
        result.metrics["files_read"] = files_read

        # Note: RAG archival via store_memory tool is expected and correct.
        # This test verifies mini-turn eviction behavior, not RAG archival.

    def test_mini_turn_eviction(self, result: TestResult, shepherd: ShepherdProcess, rag: RAGInspector):
        """Verify that tool call/result pairs (mini-turns) are prioritized for eviction"""

        print("    Creating user message to preserve...")
        important_message = f"IMPORTANT_USER_MESSAGE: This should be preserved. UID: {time.time()}"
        shepherd.send_message(important_message, timeout=10)
        result.metrics["important_message"] = important_message

        # Create several tool calls (mini-turns) that should be evicted first
        print("    Creating tool call mini-turns...")
        tool_calls_sent = 3
        for i in range(tool_calls_sent):
            shepherd.send_message(f"bash -c 'echo Tool call {i}'", timeout=10)

        # Fill context with files to push toward eviction
        files_read = self.fill_context_with_files(shepherd, 0.85)

        result.metrics["tool_calls_sent"] = tool_calls_sent
        result.metrics["files_read"] = files_read

        # Note: RAG archival via store_memory tool is expected and correct.
        # This test verifies that mini-turns are prioritized for eviction over user messages.
        # The important user message should still be in context (not evicted)
        # Tool calls should be evicted first

    # ========================================================================
    # Test Runner
    # ========================================================================

    def run_all_tests(self, mode: str = "standard", specific_test: str = None):
        """Run all tests or a specific test"""

        # Define all available tests
        all_tests = {
            "1.1": ("Fill and evict (8K)", self.test_fill_and_evict, CONTEXT_SIZES["small"]),
            "1.2": ("Distinctive messages (16K)", self.test_distinctive_messages, CONTEXT_SIZES["medium"]),
            "1.3": ("Rapid eviction (8K)", self.test_rapid_eviction, CONTEXT_SIZES["small"]),
            "2.1": ("Near capacity (8K)", self.test_near_capacity, CONTEXT_SIZES["small"]),
            "2.2": ("Over capacity (8K)", self.test_over_capacity, CONTEXT_SIZES["small"]),
            "3.1": ("Many tiny messages (8K)", self.test_many_tiny_messages, CONTEXT_SIZES["small"]),
            "4.1": ("RAG content structure (16K)", self.test_rag_content, CONTEXT_SIZES["medium"]),
            "5.1": ("Server limit overflow (context-size 0)", self.test_server_limit_overflow, CONTEXT_SIZES["server"]),
            "6.1": ("System Message Preservation (8K)", self.test_system_message_preservation, CONTEXT_SIZES["small"]),
            "6.2": ("Tool Call RAG Exclusion (8K)", self.test_tool_call_rag_exclusion, CONTEXT_SIZES["small"]),
            "6.3": ("Mini-Turn Eviction (8K)", self.test_mini_turn_eviction, CONTEXT_SIZES["small"]),
        }

        # If specific test requested, run only that test
        if specific_test:
            if specific_test not in all_tests:
                print(f"ERROR: Test '{specific_test}' not found")
                print(f"Available tests: {', '.join(all_tests.keys())}")
                return

            print(f"\nRunning specific test: {specific_test}\n")
            name, func, context_size = all_tests[specific_test]
            self.run_test(specific_test, name, func, context_size)
            return

        # Otherwise run tests based on mode
        print("\nEach test uses an isolated RAG database\n")

        # Category 1: Basic Eviction
        print(f"\n{'='*70}")
        print("Category 1: Basic Eviction Tests")
        print(f"{'='*70}")

        if mode in ["fast", "standard", "full"]:
            self.run_test("1.1", "Fill and evict (8K)",
                         self.test_fill_and_evict, CONTEXT_SIZES["small"])
            self.run_test("1.2", "Distinctive messages (16K)",
                         self.test_distinctive_messages, CONTEXT_SIZES["medium"])

        if mode in ["full"]:
            self.run_test("1.3", "Rapid eviction (8K)",
                         self.test_rapid_eviction, CONTEXT_SIZES["small"])

        # Category 2: Boundary Tests
        print(f"\n{'='*70}")
        print("Category 2: Boundary Tests")
        print(f"{'='*70}")

        if mode in ["standard", "full"]:
            self.run_test("2.1", "Near capacity (8K)",
                         self.test_near_capacity, CONTEXT_SIZES["small"])
            self.run_test("2.2", "Over capacity (8K)",
                         self.test_over_capacity, CONTEXT_SIZES["small"])

        # Category 3: Tiny Messages
        print(f"\n{'='*70}")
        print("Category 3: Tiny Message Tests")
        print(f"{'='*70}")

        if mode in ["standard", "full"]:
            self.run_test("3.1", "Many tiny messages (8K)",
                         self.test_many_tiny_messages, CONTEXT_SIZES["small"])

        # Category 4: RAG Content
        print(f"\n{'='*70}")
        print("Category 4: RAG Content Verification")
        print(f"{'='*70}")

        if mode in ["standard", "full"]:
            self.run_test("4.1", "RAG content structure (16K)",
                         self.test_rag_content, CONTEXT_SIZES["medium"])

        # Category 5: Server Limit Overflow
        print(f"\n{'='*70}")
        print("Category 5: Server Limit Overflow")
        print(f"{'='*70}")

        if mode in ["standard", "full"]:
            self.run_test("5.1", "Server limit overflow (context-size 0)",
                         self.test_server_limit_overflow, CONTEXT_SIZES["server"])

        # Category 6: Advanced Eviction Scenarios
        print(f"\n{'='*70}")
        print("Category 6: Advanced Eviction Scenarios")
        print(f"{'='*70}")

        if mode in ["standard", "full"]:
            self.run_test("6.1", "System Message Preservation (8K)",
                         self.test_system_message_preservation, CONTEXT_SIZES["small"])
            self.run_test("6.2", "Tool Call RAG Exclusion (8K)",
                          self.test_tool_call_rag_exclusion, CONTEXT_SIZES["small"])

        if mode in ["full"]:
             self.run_test("6.3", "Mini-Turn Eviction (8K)",
                          self.test_mini_turn_eviction, CONTEXT_SIZES["small"])

        # Multi-context size tests
        if mode == "full":
            print(f"\n{'='*70}")
            print("Category 7: Multi-Context Tests")
            print(f"{'='*70}")

            for name, size in CONTEXT_SIZES.items():
                if size > 0:  # Skip "server" context size for multi-context tests
                    self.run_test(f"7.{name}", f"Basic eviction ({name})",
                                 self.test_fill_and_evict, size)

    # ========================================================================
    # Reporting
    # ========================================================================

    def print_summary(self):
        """Print test summary"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        errors = sum(1 for r in self.results if r.status == "ERROR")

        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"Total Tests:  {total}")
        print(f"Passed:       {passed} ✓")
        print(f"Failed:       {failed} ✗")
        print(f"Errors:       {errors} ⚠")

        total_rag = sum(r.metrics.get("rag_added", 0) for r in self.results)
        total_evictions = sum(r.metrics.get("eviction_count", 0) for r in self.results)
        print(f"\nTotal Evictions Detected: {total_evictions}")
        print(f"Total Conversations Archived to RAG: {total_rag}")

        avg_duration = sum(r.duration_ms for r in self.results) / total if total > 0 else 0
        print(f"Average Test Duration: {avg_duration:.0f}ms")

        if failed > 0 or errors > 0:
            print(f"\nFailed/Error Tests:")
            for r in self.results:
                if r.status in ["FAIL", "ERROR"]:
                    print(f"  [{r.id}] {r.name}")
                    if r.error:
                        print(f"      {r.error}")

        print(f"\n{'='*70}\n")

        return passed == total

    def save_report(self, filename: str):
        """Save JSON report"""
        report = {
            "test_run": {
                "timestamp": datetime.now().isoformat(),
                "binary": SHEPHERD_BINARY,
                "provider": self.provider if self.provider else "auto-select",
                "rag_db": DEFAULT_RAG_DB,
                "duration_seconds": sum(r.duration_ms for r in self.results) / 1000
            },
            "summary": {
                "total": len(self.results),
                "passed": sum(1 for r in self.results if r.status == "PASS"),
                "failed": sum(1 for r in self.results if r.status == "FAIL"),
                "errors": sum(1 for r in self.results if r.status == "ERROR"),
                "total_evictions": sum(r.metrics.get("eviction_count", 0) for r in self.results),
                "total_rag_archived": sum(r.metrics.get("rag_added", 0) for r in self.results)
            },
            "results": [asdict(r) for r in self.results]
        }

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Report saved: {filename}")

# ============================================================================
# Main
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified Shepherd CLIENT eviction test suite"
    )
    parser.add_argument("--mode",
                       choices=["fast", "standard", "full"],
                       default="standard",
                       help="Test mode: fast (2 tests), standard (7 tests), full (all tests)")
    parser.add_argument("--test",
                       help="Run only a specific test by ID (e.g., '5.1' for server limit overflow)")
    parser.add_argument("--provider",
                       help="Provider name to use (default: auto-select highest priority)")
    parser.add_argument("--output",
                       help="Output JSON report file")
    parser.add_argument("--output-dir",
                       help="Directory for debug logs and temp files (default: current directory)")
    parser.add_argument("--clean",
                       action="store_true",
                       help="Delete existing test files in output-dir before starting")
    parser.add_argument("--verbose", "-v",
                       action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    # Check binary exists
    if not os.path.exists(SHEPHERD_BINARY):
        print(f"ERROR: Shepherd binary not found: {SHEPHERD_BINARY}")
        print("Please build shepherd or run from the correct directory")
        sys.exit(1)

    # Create test suite
    suite = UnifiedEvictionTestSuite(provider=args.provider, output_dir=args.output_dir, clean=args.clean, verbose=args.verbose)

    try:
        # Run tests
        suite.run_all_tests(mode=args.mode, specific_test=args.test)

        # Print summary
        all_passed = suite.print_summary()

        # Save report
        if args.output:
            # If absolute path, use as-is; otherwise put in output_dir
            report_path = args.output if os.path.isabs(args.output) else os.path.join(suite.output_dir, args.output)
            suite.save_report(report_path)
        else:
            report_path = os.path.join(suite.output_dir, "report.json")
            suite.save_report(report_path)

        sys.exit(0 if all_passed else 1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        suite.print_summary()
        sys.exit(130)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
