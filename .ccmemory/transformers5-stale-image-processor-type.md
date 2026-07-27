---
name: transformers5-stale-image-processor-type
description: transformers 5.x rejects old VLM repos whose preprocessor_config.json names a removed image processor class (e.g. Qwen2_5_VLImageProcessor); patch th…
metadata:
  type: project
tags: [vllm, transformers, vlm, qwen2.5-vl, awq, huggingface-cache]
---

# transformers 5.x: stale `image_processor_type` breaks VLM loading

## Symptom
`vllm serve zyoNoob/Qwen2.5-VL-3B-Instruct-AWQ` dies during processor load:

```
ValueError: Unrecognized image processor in <repo>. Should have a `image_processor_type`
key in its preprocessor_config.json of config.json, or one of the following `model_type`
keys in its config.json: ... qwen2_5_vl ...
```

Confusing because `qwen2_5_vl` **is** in the listed model_type keys.

## Root cause
transformers 5.11.0 dropped the per-model slow image-processor classes. Mapping is now
backend-keyed:

```
qwen2_5_vl -> {'torchvision': 'Qwen2VLImageProcessor', 'pil': 'Qwen2VLImageProcessorPil'}
```

There is no `Qwen2_5_VLImageProcessor` anymore. Old model repos (quantized forks of
Qwen2.5-VL especially) still write `"image_processor_type": "Qwen2_5_VLImageProcessor"`
in preprocessor_config.json.

In `transformers/models/auto/image_processing_auto.py::from_pretrained`, a **non-None but
unresolvable** `image_processor_type` poisons the fallback: the `AutoConfig` load that
would populate `config` only runs when `image_processor_type is None`. So `config` stays
None, `type(config) in IMAGE_PROCESSOR_MAPPING` is False, and it falls through to the
final "Unrecognized image processor" raise even though the model_type mapping exists.

So: a *wrong* class name is worse than *no* class name.

## Fix
Patch the cached preprocessor_config.json (HF cache root on this box is
`/home/steve/models`, not `~/.cache/huggingface`):

```
B=$(readlink -f /home/steve/models/models--<repo>/snapshots/*/preprocessor_config.json)
cp -n "$B" "$B.backup"
# set "image_processor_type": "Qwen2VLImageProcessor"
```

Editing the blob is safe — snapshot files are symlinks into `blobs/`; HF only re-fetches
if the remote etag changes.

Verify without launching vLLM (cheap, definitive):

```
AutoProcessor.from_pretrained('<repo>')  -> Qwen2_5_VLProcessor / Qwen2VLImageProcessor
```

Note `processor_class: Qwen2_5_VLProcessor` is still valid in transformers 5 — only the
*image* processor class was collapsed onto the qwen2_vl one.

## Related gotcha hit at the same time
The 4x3090 box normally already has a TP=4 server resident (gemma-4-31B-it-INT8 at
~22.8G/24.5G on every GPU). Any second `vllm serve` will get through config/weights and
then die on memory. Check `nvidia-smi` / `pgrep -af "vllm serve"` before launching a
second model — there is no room for a co-resident server at default
gpu-memory-utilization.
