#!/usr/bin/env python3
"""OpenAI-compatible HTTP shim that puts the `claude` CLI (Claude Code) behind
/v1/chat/completions and /v1/models, so api_bench.py, openai_bench.py, and
test_model.py can benchmark against it unmodified via --base-url.

Each request spawns a real `claude --print` session (adapters/claude.py) with
full Claude Code behavior -- CLAUDE.md, hooks, memory, tools all load exactly
as they would interactively. Token deltas are relayed live from claude's own
stream-json output, so streaming responses reflect real generation timing,
not a simulated/chunked replay.

Model selection:
  - If the request's "model" field is unset or equals the sentinel model id
    returned by /v1/models ("claude-code"), no --model flag is passed to
    claude -- it uses whatever model is currently configured as default.
  - Otherwise the request's "model" value is passed straight through as
    `claude --model <value>` (e.g. "opus", "sonnet", "haiku", a full model id).

Usage:
  python3 claude_shim.py --port 8099
  openai_bench.py --base-url http://localhost:8099/v1 --runs 3 --max-tokens 128
  openai_bench.py --base-url http://localhost:8099/v1 --model opus --runs 3
"""

import argparse
import json
import queue
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from adapters.claude import ClaudeAdapter

MODEL_SENTINEL = "claude-code"
REQUEST_TIMEOUT_S = 600


def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        return "".join(parts)
    return ""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print(f"[claude_shim] {self.address_string()} - {fmt % args}")

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self._send_json(200, {
                "object": "list",
                "data": [{"id": MODEL_SENTINEL, "object": "model", "owned_by": "anthropic"}],
            })
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/v1/chat/completions"):
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON body"})
            return

        messages = body.get("messages", [])
        system_text = "\n\n".join(
            extract_text(m.get("content")) for m in messages if m.get("role") == "system"
        )
        user_msgs = [m for m in messages if m.get("role") == "user"]
        user_text = extract_text(user_msgs[-1].get("content")) if user_msgs else ""

        req_model = body.get("model")
        model = req_model if req_model and req_model != MODEL_SENTINEL else None

        params = {}
        if body.get("max_tokens"):
            params["max_output_tokens"] = body["max_tokens"]

        stream = bool(body.get("stream", False))

        self._run_completion(system_text, user_text, model, params, stream)

    def _run_completion(self, system_text, user_text, model, params, stream):
        events = queue.Queue()
        result = {}

        adapter = ClaudeAdapter(system_prompt=system_text, model=model, params=params)
        adapter.on_delta = lambda frag: events.put(("delta", frag))

        def on_response(text):
            # Authoritative final text. Equals the joined deltas on success;
            # holds claude's own error message (e.g. "response exceeded the N
            # output token limit") when the turn errored before any text
            # block streamed -- in which case no "delta" events fired at all.
            result["response_text"] = text

        def on_result(cost, turns, usage, is_error):
            result["cost"] = cost
            result["turns"] = turns
            result["usage"] = usage
            result["is_error"] = is_error
            events.put(("done", None))

        def on_exit(code):
            events.put(("exit", code))

        adapter.on_response = on_response
        adapter.on_result = on_result
        adapter.on_exit = on_exit

        try:
            adapter.spawn()
            adapter.send(user_text)
        except FileNotFoundError:
            self._send_json(500, {"error": "claude CLI not found on PATH"})
            return

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        model_name = model or MODEL_SENTINEL

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def write_chunk(payload):
                data = f"data: {json.dumps(payload)}\n\n".encode()
                self.wfile.write(b"%x\r\n%b\r\n" % (len(data), data))
                self.wfile.flush()

            saw_delta = False
            done = False
            while not done:
                try:
                    kind, payload = events.get(timeout=REQUEST_TIMEOUT_S)
                except queue.Empty:
                    break
                if kind == "delta":
                    saw_delta = True
                    write_chunk({
                        "id": completion_id, "object": "chat.completion.chunk", "created": created,
                        "model": model_name,
                        "choices": [{"index": 0, "delta": {"content": payload}, "finish_reason": None}],
                    })
                elif kind in ("done", "exit"):
                    done = True

            # No text block ever streamed (e.g. claude errored out before
            # replying) -- fall back to its own error text so the caller sees
            # why, instead of a stream that silently produced nothing.
            if not saw_delta and result.get("response_text"):
                write_chunk({
                    "id": completion_id, "object": "chat.completion.chunk", "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": result["response_text"]}, "finish_reason": None}],
                })

            write_chunk({
                "id": completion_id, "object": "chat.completion.chunk", "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            })
            self.wfile.write(b"0\r\n\r\n")
            adapter.kill()
            return

        # Non-streaming: buffer all deltas, return one JSON completion.
        full_text = []
        saw_delta = False
        while True:
            try:
                kind, payload = events.get(timeout=REQUEST_TIMEOUT_S)
            except queue.Empty:
                break
            if kind == "delta":
                saw_delta = True
                full_text.append(payload)
            elif kind in ("done", "exit"):
                break
        adapter.kill()

        # Fall back to claude's own error text if no delta ever streamed.
        content = "".join(full_text) if saw_delta else result.get("response_text", "")

        usage = result.get("usage", {})
        self._send_json(200, {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "error" if result.get("is_error") else "stop",
            }],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            },
        })


def main():
    parser = argparse.ArgumentParser(description="OpenAI-compatible shim over the claude CLI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8099)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"claude_shim listening on http://{args.host}:{args.port} (proxying to `claude` CLI)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
