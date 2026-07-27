"""Claude Code adapter for benchmarking against the live `claude` CLI.

Speaks the stream-json protocol over stdin/stdout, same as
/src/builder/adapters/claude.py and /src/archive/mxai/mxai/adapters/claude.py.
Unlike those, this one keeps Claude Code's OWN default system prompt intact
(CLAUDE.md, hooks, memory, tools all load normally) -- it only *appends* an
optional system message from the benchmark request, and never forces a
--model flag unless the caller explicitly asked for one.
"""

import json
import os
import shutil
import tempfile
import uuid

from .base import Adapter

CLAUDE_BIN = os.environ.get("CLAUDE_BENCH_BIN") or shutil.which("claude") or "claude"


class ClaudeAdapter(Adapter):

    backend_name = "claude"

    def __init__(self, system_prompt: str = "", extra_args: list = None,
                 debug: bool = False, model: str = None, params: dict = None,
                 binary_path: str = None):
        super().__init__(system_prompt, extra_args=extra_args, debug=debug,
                          model=model, params=params, binary_path=binary_path)
        self.session_id = str(uuid.uuid4())
        self.append_system_prompt_file = None
        # index -> content_block type ("text", "thinking", "tool_use", ...),
        # tracked so we only forward text_delta fragments, not thinking/tool deltas
        self._block_types = {}

    def build_command(self) -> list:
        cmd = [
            self.binary_path or CLAUDE_BIN,
            "--print",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--include-partial-messages",
            "--dangerously-skip-permissions",
            "--verbose",
        ]
        if self.system_prompt:
            self.append_system_prompt_file = tempfile.NamedTemporaryFile(
                mode="w", prefix="claude_bench_", suffix=".txt",
                dir="/tmp", delete=False)
            self.append_system_prompt_file.write(self.system_prompt)
            self.append_system_prompt_file.close()
            cmd.extend(["--append-system-prompt-file", self.append_system_prompt_file.name])
        # Only pass --model/--effort when the caller explicitly specified one;
        # otherwise let claude use whatever is currently configured as default.
        if self.model:
            cmd.extend(["--model", self.model])
        effort = self.params.get("effort")
        if effort:
            cmd.extend(["--effort", effort])
        cmd.extend(self.extra_args)
        return cmd

    def cleanup(self):
        f = self.append_system_prompt_file
        if f and os.path.exists(f.name):
            os.unlink(f.name)
            self.append_system_prompt_file = None

    def build_env(self) -> dict:
        env = dict(os.environ)
        env.pop("CLAUDECODE", None)
        max_tokens = self.params.get("max_output_tokens")
        if max_tokens:
            env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(max_tokens)
        return env

    def send(self, text: str):
        if not self.alive:
            return
        msg = {
            "type": "user",
            "message": {
                "role": "user",
                "content": text,
            },
            "parent_tool_use_id": None,
            "session_id": self.session_id,
        }
        line = json.dumps(msg) + "\n"
        try:
            self.proc.stdin.write(line.encode())
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def parse_stdout(self):
        collected_text = []
        debug = os.environ.get("CLAUDE_ADAPTER_DEBUG")

        for raw_line in self.proc.stdout:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type")
            if debug:
                print(f"[adapter] {raw_line[:200]}", flush=True)

            if etype == "stream_event":
                sub = event.get("event", {})
                sub_type = sub.get("type")

                if sub_type == "content_block_start":
                    index = sub.get("index")
                    block = sub.get("content_block", {})
                    self._block_types[index] = block.get("type")

                elif sub_type == "content_block_delta":
                    index = sub.get("index")
                    delta = sub.get("delta", {})
                    if self._block_types.get(index) == "text" and delta.get("type") == "text_delta":
                        frag = delta.get("text", "")
                        if frag:
                            if self.on_delta:
                                self.on_delta(frag)
                            collected_text.append(frag)

                elif sub_type == "content_block_stop":
                    self._block_types.pop(sub.get("index"), None)

            elif etype == "assistant":
                for block in event.get("message", {}).get("content", []):
                    if block.get("type") == "tool_use" and self.on_tool_use:
                        name = block.get("name", "")
                        desc = (block.get("input", {}).get("description")
                                or block.get("input", {}).get("command", "")[:80]
                                or "")
                        self.on_tool_use(name, desc)

            elif etype == "result":
                # collected_text is empty when the turn errored before any text
                # block streamed (e.g. max_tokens exhausted by thinking tokens
                # before the reply) -- event["result"] then holds the error text.
                response = "".join(collected_text) if collected_text else event.get("result", "")
                if self.on_response:
                    self.on_response(response)
                if self.on_result:
                    self.on_result(
                        event.get("total_cost_usd", 0.0),
                        event.get("num_turns", 0),
                        event.get("usage", {}),
                        bool(event.get("is_error")),
                    )
                collected_text.clear()
