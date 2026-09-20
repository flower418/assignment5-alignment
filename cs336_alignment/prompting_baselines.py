import vllm_utils

vllm_server = vllm_utils.VLLMServer(
    model_id="allenai/OLMo-2-0425-1B",
    gpu=0,
)

