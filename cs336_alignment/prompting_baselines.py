from pathlib import Path
import json
from collections import Counter # 用于给答对的题目计数

from cs336_alignment import vllm_utils
from cs336_alignment import drgrpo_grader

# 读取 gsm8k
with open("data/gsm8k/test.jsonl") as f:
    rows = [json.loads(line) for line in f]


vllm_server = vllm_utils.VLLMServer(
    model_id="allenai/OLMo-2-0425-1B",
    gpu=0,
)

def evaluate(prompt_path, reward_fn, use_stop):
    template = Path(prompt_path).read_text() # 读取三种 prompt 模板中的一种
    prompts = [template.format(question=row["question"]) for row in rows]
    sampling_params = {
        "temperature": 1.0,
        "max_tokens": 512,
        "n": 1,
        "seed": 0,
    } 

    # r1_zero 的 prompt 需要在 </answer> 处停止
    if use_stop:
        sampling_params["stop"] = ["</answer>"],
        sampling_params["include_stop_str_in_output"] = True

    completions = vllm_server.generate_completions(
        prompts=prompts,
        sampling_params=sampling_params,
        batch_size=32,
    )

    counts = Counter()
    examples = {
        "formatted_but_wrong": [],
        "unformatted": [],
    }

    for row, completion in zip(rows, completions):
        ground_truth = row["answer"].split("####")[1].strip() # 先提取 answer，然后以 #### 分割，拆成前后两部分，对第二个部分去除空格，得到最终答案
        score = reward_fn(completion.text, ground_truth)
        if score["format_reward"] == 1 and score["answer_reward"] == 1:
            category = "correct"
        elif score["format_reward"] == 1 and score["answer_reward"] == 0:
            category = "formatted_but_wrong"
        else:
            category = "unformatted"
        counts[category] += 1

vllm_server.start()

# question_only 不需要 <think> and <answer>
evaluate(
    prompt_path="cs336_alignment/prompts/question_only.prompt",
    reward_fn=drgrpo_grader.question_only_reward_fn,
    use_stop=False,
)
# zero-shot R1
evaluate(
    prompt_path="cs336_alignment/prompts/r1_zero.prompt",
    reward_fn=drgrpo_grader.r1_zero_reward_fn,
    use_stop = True,
)
# few-shot R1
evaluate(
    prompt_path="cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt",
    reward_fn=drgrpo_grader.r1_zero_reward_fn,
    use_stop=True,
)

vllm_server.stop()