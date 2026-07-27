# Force the stable V0 engine path
export VLLM_USE_V1=0

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/steve/venvs/vllm/bin/vllm serve google/diffusiongemma-26B-A4B-it \
        --trust-remote-code \
        --tensor-parallel-size 4 \
        --max-model-len 210000 \
        --kv-cache-dtype fp8_e5m2 \
        --enable-prefix-caching \
        --enable-auto-tool-choice \
        --tool-call-parser gemma4 \
        --max-num-seqs 1 \
        --gpu-memory-utilization 0.92 \
        --max-num-batched-tokens 32768 \
        --generation-config vllm \
        --hf-overrides '{"diffusion_sampler":"entropy_bound","diffusion_entropy_bound":0.1}' \
        --default-chat-template-kwargs '{"enable_thinking":true}'
