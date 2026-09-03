---
name: qwen38-27b-uncensored-model-options-4x3090
description: Qwen3.8-27B abliterated/heretic serving options on the 4x3090 box: orcarouter INT8 W8A8 is the only quantized uncensored build; heretic is BF16-only;…
metadata:
  type: project
---

## Qwen3.8-27B uncensored / abliterated options (4x3090, SM 8.6, TP4)

Surveyed 2026-08-30 when replacing the gemma-4-31B-it-INT8 service (stopped, still
`enabled`, so it returns on reboot unless disabled).

### Chosen: `orcarouter/Qwen3.8-27B-Uncensored-INT8`
- Abliterated by orthogonalizing the refusal direction out of the residual stream
  (Arditi et al. 2024), 131 residual-writing matrices edited at layer 38.
- INT8 W8A8 compressed-tensors: symmetric per-channel INT8 weights (no calibration),
  symmetric dynamic per-token INT8 activations. 400 LM linears quantized; vision
  tower, norms, lm_head, embeddings and the MTP head stay BF16.
- ~31.2 GiB / 9 shards / 1599 tensors -> ~7.8 GiB per GPU at TP4. Roomy.
- 262144 native context. Hybrid Gated DeltaNet: 48 linear-attention + 16
  full-attention layers, so only 16 layers pay KV per token -- this is why 262k is
  reachable on 24 GiB cards at all.
- Parsers: `--reasoning-parser qwen3 --tool-call-parser qwen3_coder`.
- Launch scripts in this repo: `./Qwen3.8-27B-Uncensored-INT8` and the
  `-mtp` A/B sibling.

### MTP / speculative decoding here is OPEN, do not carry the 3.6 verdict over
`gemma4-vs-qwen36-latency-tuning-4x3090` records mtp3 -11.7% / mtp2 -21.4% on
Ampere, and that is easy to over-apply. Read the *cause*: acceptance was fine
(2.33-2.37 accept_len) and the loss came from Qwen3.6-35B-A3B's MTP block being a
full 256-expert MoE layer, 1.57 GiB run per draft step. **Qwen3.8-27B is dense**, so
its draft step is a dense block -- structurally the same shape as Gemma's 840 MB
dense drafter, which measured **+21.6%** on this same box. Batch-1 decode on 4x3090
is bandwidth-bound, which is where spec decoding pays.

So: `./Qwen3.8-27B-Uncensored-INT8-mtp` exists purely to be A/B'd against the plain
script with `api_bench.py --runs 12` (taint-check against vLLM's `Running: N reqs`,
verify N=1). Expect MTP to win; it is unmeasured until it is measured. MTP cost
~0.39 GiB/GPU of KV on 3.6 (262144 -> ~210000), hence `--max-model-len 196608` in
the -mtp script; raise it if the startup KV log says there is room.

### There is NO INT8 heretic build
Every heretic-lineage Qwen3.8-27B is GGUF or plain BF16 safetensors:
- `trohrbaugh/Qwen3.8-27B-heretic-ara` -- BF16, ~55.6 GiB over 6 shards plus
  `model-auxiliary.safetensors` (849 MB = the MTP head). This is the parent that
  `0bserverx/...-Heretic-Abliterated-Uncensored-GGUF` was built on. Script:
  `./Qwen3.8-27B-heretic-ara`. ~13.9 GiB/GPU at TP4, so far tighter KV headroom --
  drop `--max-model-len` to 131072 first if startup dies allocating KV.
- `asfgsdfg/Qwen3.8-27B-Heretic` -- BF16, refusals 87/100 -> 53/100, KL 0.0081.
  Weaker abliteration than the ara lineage.
- `0bserverx` said in discussion #10 that FP16 safetensors are "added to the queue"
  but had not uploaded them as of this survey. Worth re-checking.

### Do not use the FP8 builds
`orcarouter/Qwen3.8-27B-Uncensored-FP8`, `Qwen/Qwen3.8-27B-FP8`,
`huginnfork/Qwen3.8-27B-FP8` all exist and are all wrong here -- see
`fp8-dead-end-do-not-suggest` (GA102, SM 8.6, no FP8 tensor cores). The INT8 sibling
is the same abliterated BF16 base with only the quantization differing, so nothing
is lost by taking INT8. Also skip `huginnfork/Qwen3.8-27B-NVFP4A16` (Blackwell).

### GGUF -> vLLM is not worth attempting
vLLM's `--quantization gguf` is experimental, effectively single-file, and does not
cover this architecture (VL tower + Gated DeltaNet hybrid). Reverse conversion
(gguf -> HF safetensors) exists only as community dequant scripts, and dequantizing
a lossy Q-quant back to BF16 recovers nothing the original BF16 does not already
have -- and the BF16 parents are on the hub anyway. Pull `trohrbaugh/...-heretic-ara`
directly instead.

### Config carried from the other launchers
`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` + `--max-num-batched-tokens 8192` +
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, per
`vllm-local-patches-and-config-landmines`. `--kv-cache-dtype auto` is the house
style (explicit no-op; fp8 kv-cache-dtype errors on this box). No `--chat-template`
override: the 3.6 jinja in /home/steve/models is for a different model, let the repo
supply its own. No `--enable-expert-parallel` (dense model, and it lost on 3.6 too).
