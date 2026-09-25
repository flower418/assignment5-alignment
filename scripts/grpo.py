from pathlib import Path
import random
import json

import torch
import wandb

from cs336_alignment import drgrpo_grader
from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.vllm_utils import VLLMServer
from tests.adapters import run_grpo_train_step


SEED = 0
EVAL_SEED = SEED + 1
MODEL_ID = "allenai/OLMo-2-0425-1B"

TRAIN_PATH = "data/gsm8k/train.jsonl"
VAL_PATH = "data/gsm8k/test.jsonl"
PROMPT_PATH = "cs336_alignment/prompts/r1_zero.prompt"
OUTPUT_DIR = Path("experiments/grpo")

TRAIN_EXAMPLES = 6400
EVAL_EXAMPLES = 1024
ROLLOUT_STEPS = 200

GROUP_SIZE = 8
ROLLOUT_BATCH_SIZE = 256 # 共产生 256 个 rollout，说明共有 256//8=32 个 prompt
PROMPTS_PER_STEP = ROLLOUT_BATCH_SIZE // GROUP_SIZE
GRADIENT_ACCUMULATION_STEPS = 32

LEARNING_RATE = 1e-5
MAX_GRAD_NORM = 1.0
TEMPERATURE = 1.0
MAX_TOKENS = 512

EVAL_EVERY = 10 # 每 10 个 rollout batch eval 一次
GENERATION_BATCH_SIZE = 32 # 每次向 vllm 发送 32 个 prompt，每个 prompt 生成 8 次

def load_gsm8k(path):
    rows = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            rows.append({
                "question": item["question"],
                "ground_truth": item["answer"].split("####")[1].strip()
            })
    return rows

def make_prompts(rows, template):
    return [template.format(question=row["question"]) for row in rows]

def evaluate(server, policy, rows, template, step):
    server.sync_policy_weights(policy)

    prompts = make_prompts(rows, template)
    completions = server.generate_completions(
        prompts = prompts,
        sampling_params={
            "temperature": 1.0,
            "max_tokens": MAX_TOKENS,
            "n": 1,
            "seed" :EVAL_SEED,
            "stop": ["</answer>"],
            "include_stop_str_in_output": True,       
        },
        batch_size = GENERATION_BATCH_SIZE
    )

    scores = [
        drgrpo_grader.r1_zero_reward_fn(completion.text, row["ground_truth"])
        for completion, row in zip(completions, rows)
    ]

    return {
        "eval/reward": sum(score["reward"] for score in scores) / len(scores),
        "eval/format_reward": sum(score["format_reward"] for score in scores) / len(scores),
        "eval/answer_reward": sum(score["answer_reward"] for score in scores) / len(scores),
        "eval/response_tokens": sum(len(completion.token_ids) for completion in completions) / len(completions) # 用于 eval 回答长度是否变化
    }

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    train_rows = load_gsm8k(TRAIN_PATH)
    random.shuffle(train_rows)
    train_rows = train_rows[: TRAIN_EXAMPLES]
    train_batches = [
        train_rows[i: i + PROMPTS_PER_STEP]
        for i in range(0, TRAIN_EXAMPLES, PROMPTS_PER_STEP)
    ]

    val_rows = load_gsm8k(VAL_PATH)
    random.Random(EVAL_SEED).shuffle(val_rows)
    val_rows = val_rows[: EVAL_EXAMPLES]

    prompt_template = Path(PROMPT_PATH).read_text()

    run_name = f"grpo_seed{SEED}"
    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    policy, tokenizer = get_model_and_tokenizer(MODEL_ID, device="cuda:0")
    policy.config.use_cache = False

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=0.0
    )

    wandb.init(
        project="cs336-a5-grpo",
        name=run_name,
        config={
            "model": MODEL_ID,
            "seed": SEED,
            "train_examples": TRAIN_EXAMPLES,
            "eval_examples": EVAL_EXAMPLES,
            "rollout_steps": ROLLOUT_STEPS,
            "rollout_batch_size": ROLLOUT_BATCH_SIZE,
            "group_size": GROUP_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": LEARNING_RATE,
        }
    )

    server = VLLMServer(model_id=MODEL_ID, gpu=1, seed=SEED)
    server.start()
    server.init_weight_sync(policy_device="cuda:0")

    for step in range(1, ROLLOUT_STEPS + 1):
        batch_rows = train_batches[step - 1]
        prompts = make_prompts(batch_rows, prompt_template)

        server.sync_policy_weights(policy) # 同步更新过的 policy，并在本轮用它 rollout
        completions = server.generate_completions(
            prompts=prompts,
            sampling_params={
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "n": GROUP_SIZE,
                "seed": SEED + step,
                "stop": ["</answer>"],
                "include_stop_str_in_output": True,
            },
            batch_size=GENERATION_BATCH_SIZE
        )

        responses = [completion.text for completion in completions]
        repeated_prompts = [
            prompt
            for prompt in prompts
            for _ in range(GROUP_SIZE)
        ]
        repeated_ground_truth = [
            row["ground_truth"]
            for row in batch_rows
            for _ in range(GROUP_SIZE)
        ]

        loss, metadata = run_grpo_train_step(
            model=policy,
            tokenizer=tokenizer,
            optimizer=optimizer,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            max_grad_norm=MAX_GRAD_NORM,
            reward_fn=drgrpo_grader.r1_zero_reward_fn,
            repeated_prompts=repeated_prompts,
            repeated_ground_truths=repeated_ground_truth,
            rollout_responses=responses,
            group_size=GROUP_SIZE
        )

        metrices = {
            "train/loss": float(loss),
            "train/grad_norm": float(metadata["grad_norm"]),
            "train/token_entropy": metadata["token_entropy"],
            "train/reward": metadata["mean_total"],
            "train/format": metadata["mean_format"],
            "train/response_tokens": sum(len(completion.token_ids) for completion in completions) / len(completions)
        }

        if step % EVAL_EVERY == 0:
            metrices.update(evaluate(server, policy, val_rows, prompt_template, step))

        wandb.log(metrices, step=step)
        print(f"step {step}: {metrices}")

        if step % 40 == 0:
            with (run_dir / "rollouts.jsonl").open("a") as f:
                for i in range(GROUP_SIZE):
                    f.write(json.dumps({
                        "step": step,
                        "prompt": repeated_prompts[i],
                        "ground_truth": repeated_ground_truth[i],
                        "response": responses[i],
                    }) + "\n")

    final_dir = run_dir / "final"
    policy.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)

    server.stop()
    wandb.finish()

if __name__ == "__main__":
    main()