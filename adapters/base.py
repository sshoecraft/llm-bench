"""Base adapter for AI CLI backends.

Each adapter knows how to spawn a specific AI CLI tool, send messages
to it via stdin, and parse responses from its stdout.

Ported from /src/builder/adapters/base.py.
"""

import os
import signal
import subprocess
import threading
from abc import ABC, abstractmethod


class Adapter(ABC):
    """Base class for AI backend adapters."""

    def __init__(self, system_prompt: str, extra_args: list = None,
                 debug: bool = False, model: str = None, params: dict = None,
                 binary_path: str = None, endpoint: str = None,
                 api_key: str = None):
        self.system_prompt = system_prompt
        self.extra_args = extra_args or []
        self.debug = debug
        self.model = model
        self.params = params or {}
        self.binary_path = binary_path
        self.endpoint = endpoint
        self.api_key = api_key
        self.proc = None
        self.on_response = None    # callback: fn(text: str)
        self.on_delta = None       # callback: fn(text_fragment: str)
        self.on_tool_use = None    # callback: fn(name: str, desc: str)
        self.on_result = None      # callback: fn(cost: float, turns: int, usage: dict, is_error: bool)
        self.on_exit = None        # callback: fn(exit_code: int)
        self.stdout_thread = None

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Short name for this backend (e.g. 'claude', 'gemini')."""
        ...

    @abstractmethod
    def build_command(self) -> list:
        """Return the argv list to spawn the subprocess."""
        ...

    @abstractmethod
    def build_env(self) -> dict:
        """Return the environment dict for the subprocess."""
        ...

    @abstractmethod
    def send(self, text: str):
        """Send a message to the subprocess via stdin."""
        ...

    @abstractmethod
    def parse_stdout(self):
        """Read stdout and call self.on_response/on_delta/on_tool_use/on_result.

        This runs in a thread. Must loop until stdout is closed.
        """
        ...

    def spawn(self):
        """Start the subprocess."""
        argv = self.build_command()
        env = self.build_env()

        self.proc = subprocess.Popen(
            argv, env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # inherit parent's stderr
            start_new_session=True,  # own process group for clean shutdown
        )

        self.stdout_thread = threading.Thread(
            target=self._run_stdout_parser, daemon=True)
        self.stdout_thread.start()

    def cleanup(self):
        """Clean up any temporary resources. Override in subclasses."""
        pass

    def kill(self):
        """Terminate the subprocess and its entire process group."""
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    self.proc.kill()
        self.cleanup()

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _run_stdout_parser(self):
        """Wrapper that calls parse_stdout and fires on_exit when done."""
        try:
            self.parse_stdout()
        finally:
            exit_code = self.proc.wait() if self.proc else -1
            self.cleanup()
            if self.on_exit:
                self.on_exit(exit_code)
