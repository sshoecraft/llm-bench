---
name: claude-code-custom-context-window
description: Claude Code v2.1.x custom-model context window: set CLAUDE_CODE_MAX_CONTEXT_TOKENS + DISABLE_COMPACT env vars; settings.json "models" key is ignored.
metadata:
  type: reference
tags: [claude-code, context-window, clyde, vllm, env-vars]
---

## Setting the context window for a custom/OpenAI-compatible model in Claude Code (v2.1.x)

When running Claude Code against a local server via `ANTHROPIC_BASE_URL`/`ANTHROPIC_MODEL` (e.g. the `clyde` wrapper → vLLM at 192.168.1.166:8000), Claude defaults the context window to **200000** (`mxt=200000` constant) for any model not in its built-in table. The statusline reads `context_window.context_window_size` from the JSON Claude itself emits — so a wrong window comes from Claude, not the statusline tool.

### What works
Two env vars, BOTH required:
- `CLAUDE_CODE_MAX_CONTEXT_TOKENS=<n>` — the desired window (e.g. 262144)
- `DISABLE_COMPACT=1` — the override is **gated** behind this; without it the token var is ignored.

From the bundle: `function Ati(){ if(Ge.DISABLE_COMPACT && process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS){...return parsed} return }`. `Ge.DISABLE_COMPACT` is `Oe.bool()` (so `1`/`true`). CHANGELOG confirms: "Fixed CLAUDE_CODE_MAX_CONTEXT_TOKENS to honor DISABLE_COMPACT when it is set." `DISABLE_COMPACT=1` is consistent with `autoCompactEnabled: false` already in `~/.claude/settings.json`.

### What does NOT work
- The `models` settings.json key with `maxContextTokens`/`contextWindow` is **not part of the v2.1.x schema** — `maxContextTokens` appears 0 times in the binary. Model-name key (full `org/Model` vs bare name) is irrelevant; the block is never read.

### clyde implementation
`/usr/local/bin/clyde` (python3) fetches `/v1/models`, reads the active model's `max_model_len`, and sets `CLAUDE_CODE_MAX_CONTEXT_TOKENS` + `DISABLE_COMPACT=1` before `os.execvpe("claude", ...)`. So the window auto-tracks whatever model the server serves.

### How to investigate Claude Code internals
Bundle is a single packaged executable at `~/.local/share/claude/versions/<ver>`. `grep -a -o` strings out of it, or scan bytes in python and map non-printable to `·`. Public repo at `/src/claude-code` is docs/CHANGELOG/plugins only — not the impl source.

Related: [[feedback-vllm-max-model-len-auto]]</body>
</invoke>
