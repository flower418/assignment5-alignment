from pathlib import Path
from cs336_alignment import vllm_utils

vllm_server = vllm_utils.VLLMServer(
    model_id="allenai/OLMo-2-0425-1B",
    gpu=0,
)

template = Path("cs336_alignment/prompts/question_only.prompt").read_text()
prompt = template.format(
    question="If John has 3 apples and gets 2 more, how many apples does he have?"
)

sampling_params = {
    "temperature": 1.0,
    "max_tokens": 512,
    "n": 1,
    "seed": 0,
    "stop": ["</answer>"],
    "include_stop_str_in_output": True,
}

vllm_server.start()
completions = vllm_server.generate_completions(
    prompts=prompt,
    sampling_params=sampling_params,
    batch_size=1,
)

print(completions[0].text)
print(completions[0].finish_reason)

vllm_server.stop()