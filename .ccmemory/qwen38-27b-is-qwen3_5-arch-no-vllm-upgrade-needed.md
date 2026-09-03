---
name: qwen38-27b-is-qwen3_5-arch-no-vllm-upgrade-needed
description: Qwen3.8-27B declares architectures=Qwen3_5ForConditionalGeneration, so the existing vllm 0.22.1rc1.dev466 serves it natively - do NOT upgrade vLLM ch…
metadata:
  type: project
---

## Qwen3.8-27B runs on the existing vLLM build. Do not upgrade.

Checked 2026-08-30 against `/home/steve/venvs/vllm` (vllm 0.22.1rc1.dev466+gb7f9b6ab2).

### The trap
Grepping `vllm/model_executor/models/registry.py` for "Qwen3.8" / "Qwen3_8" returns
**zero matches**, and there is no `qwen3_8*.py` under `vllm/model_executor/models/`.
That looks like "this build is too old, upgrade vLLM." **It is not.** Upgrading here
is actively harmful -- it wipes the DiffusionGemma local TP patch
(`diffusiongemma-vllm-nightly-tp-patch`) and any other local patches
(`vllm-local-patches-and-config-landmines`).

### The actual fact
`Qwen/Qwen3.8-27B`'s `config.json` declares:
```
architectures: ["Qwen3_5ForConditionalGeneration"]
model_type:    "qwen3_5"
text_config.model_type: "qwen3_5_text"
```
Qwen3.8 reuses the **Qwen3.5** architecture class. That IS registered in this build:
- `Qwen3_5ForConditionalGeneration` -- registry.py:566
- `Qwen3_5MTP` (draft model, for speculative decoding) -- registry.py:637
- impl files `qwen3_5.py`, `qwen3_5_mtp.py` under `vllm/model_executor/models/`

Confirms the hybrid stack too: `text_config.num_hidden_layers: 64`, `layer_types`
cycling 3x `linear_attention` then 1x `full_attention` = 48 linear + 16 full.
Gated DeltaNet support is present via
`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:419`
(`QwenGatedDeltaNetAttention`), used by `qwen3_5.py:47,141`.

### Parsers and spec-decode all present in this build
- tool parser `qwen3_coder` -> `Qwen3CoderToolParser`. NOTE: tool parsers moved in
  this build from `vllm/entrypoints/openai/tool_parsers/` to top-level
  `vllm/tool_parsers/` (`vllm/tool_parsers/qwen3coder_tool_parser.py:36`).
- reasoning parser `qwen3` -> `Qwen3ReasoningParser` (`vllm/reasoning/__init__.py`).
- speculative `method: "mtp"` accepted -- `vllm/config/speculative.py:48`, with
  `qwen3_5_mtp` among `MTPModelTypes`. Per-model mtp names are auto-rewritten to the
  generic `"mtp"` (speculative.py:550-554).

### Unrelated gotcha hit the same day
`orcarouter/Qwen3.8-27B-Uncensored-INT8` is `gated=auto` on HF. `hf download`
**exits 0** while pulling only LICENSE+README and printing
`Error: Access denied. This repository requires approval.` -- always verify the
downloaded byte count, not the exit code. One click on the model page clears it
(auto-approval, no human in the loop). Its real file list is 7 shards +
`model-extra-00001-of-00001.safetensors` (the BF16 MTP head) and it ships its own
`chat_template.jinja`, so do not pass `--chat-template`.
`trohrbaugh/Qwen3.8-27B-heretic-ara`, `asfgsdfg/Qwen3.8-27B-Heretic`,
`lokeshe09/Qwen3.8-27B-INT8` and `Qwen/Qwen3.8-27B` are all ungated.
