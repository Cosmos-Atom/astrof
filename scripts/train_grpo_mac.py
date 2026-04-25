"""
ASTROF — GRPOTrainer Training Script (Mac MPS / Apple Silicon)
==============================================================
Runs on M1/M2/M3/M4 MacBook Pro using PyTorch MPS backend.
No CUDA, no bitsandbytes 4-bit — uses full fp16 precision.

Quick-start:
    pip install torch torchvision torchaudio  # MPS built-in
    pip install trl peft transformers accelerate datasets openenv-core openai astropy
    # No unsloth needed — uses standard HF TRL
    ENV_BASE_URL=https://Cosmosatom-astrof.hf.space python scripts/train_grpo_mac.py

Curriculum: easy → medium → hard
"""
import json
import os
import sys
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from novelty_detector import NoveltyDetector
from continual_learner import ContinualLearner

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME          = os.getenv("MODEL_NAME", "Qwen/Qwen3-1.7B-Instruct")  # full precision, no 4bit
ENV_BASE_URL        = os.getenv("ENV_BASE_URL", "http://localhost:7860")
HF_REPO_ID          = os.getenv("HF_REPO_ID", "")
HF_TOKEN            = os.getenv("HF_TOKEN", "")

MAX_STEPS_PER_TASK  = {"easy": 18, "medium": 44, "hard": 32, "expert": 18}
CURRICULUM          = ["easy", "medium", "hard"]
GRPO_STEPS_PER_TASK = 100
BATCH_SIZE          = 1       # MPS: keep low to avoid OOM
GRAD_ACCUM          = 16      # effective batch = 16
LR                  = 5e-5
SFT_EPOCHS          = 1
SFT_DATA            = os.path.join(os.path.dirname(__file__), "../data/sft_warmstart.jsonl")

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------
import torch

def _get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = _get_device()
print(f"[DEVICE] Using: {DEVICE}", flush=True)

# ---------------------------------------------------------------------------
# System prompts (must match inference.py exactly)
# ---------------------------------------------------------------------------
PLANNER_SYSTEM = """You are the Science Planner for a global telescope network observing exoplanets.
Score each visible target 0.0-1.0 (higher = more urgent to observe tonight).
Reply with ONLY valid JSON.
Format: {"targets": [{"target_id": "<name>", "score": <0.0-1.0>}], "too_flag": "dismiss|queue|interrupt"}"""

COORDINATOR_SYSTEM = """You are the Network Coordinator for a global telescope network.
Assign each available telescope to its best unobserved target. Never assign two telescopes the same target.
Reply with ONLY valid JSON.
Format: {"assignments": [{"telescope_id": "<id>", "target_id": "<name>"}]}
telescope_ids: mauna_kea | la_palma | siding_spring"""

EXECUTOR_SYSTEM = """You are Telescope Executor at {site}.
Choose: observe / wait / request_reassign / abort.
Reply with ONLY valid JSON.
Format: {{"action": "observe|wait|request_reassign|abort", "target_id": "<name>|null", "minutes": <int|null>, "reason": "<str|null>"}}"""

# ---------------------------------------------------------------------------
# Model loading — no unsloth, no 4-bit, pure HF transformers + PEFT
# ---------------------------------------------------------------------------
def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig, TaskType

    print(f"[INIT] Loading {MODEL_NAME} (full fp16, MPS) ...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map={"": DEVICE},
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer

# ---------------------------------------------------------------------------
# Direct model inference
# ---------------------------------------------------------------------------
def _make_direct_model_fn(model, tokenizer):
    import re

    def model_fn(system: str, user: str) -> str:
        prompt = (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    return model_fn

# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------
def _run_episode(task_id: str, model_fn) -> List[dict]:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from client import AstrofEnv
    from models import (NetworkAction, PlannerAction, CoordinatorAction,
                        ExecutorAction, TargetScore, AssignmentItem)
    import re, json as _json
    from concurrent.futures import ThreadPoolExecutor

    def parse_json(text):
        text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return _json.loads(m.group())
            except Exception:
                pass
        return None

    steps_data = []

    with AstrofEnv(base_url=ENV_BASE_URL).sync() as env:
        result = env.reset(task_id=task_id)

        for _ in range(MAX_STEPS_PER_TASK[task_id]):
            if result.done:
                break
            obs = result.observation
            p_obs, c_obs, e_obs_list = obs.planner_obs, obs.coordinator_obs, obs.executor_obs

            with ThreadPoolExecutor(max_workers=5) as pool:
                f_p  = pool.submit(model_fn, PLANNER_SYSTEM, p_obs.narrative)
                f_c  = pool.submit(model_fn, COORDINATOR_SYSTEM, c_obs.narrative)
                f_ex = [pool.submit(model_fn, EXECUTOR_SYSTEM.format(site=e.site_name), e.narrative)
                        for e in e_obs_list]

            raw_p, raw_c = f_p.result(), f_c.result()
            raw_ex = [f.result() for f in f_ex]

            p_data = parse_json(raw_p)
            c_data = parse_json(raw_c)

            planner_action = PlannerAction(
                targets=[TargetScore(target_id=str(t["target_id"]), score=float(t.get("score", 0.5)))
                         for t in (p_data or {}).get("targets", []) if "target_id" in t],
                too_flag=(p_data or {}).get("too_flag", "dismiss")
            ) if p_data else PlannerAction()

            coord_action = CoordinatorAction(
                assignments=[AssignmentItem(telescope_id=str(a["telescope_id"]), target_id=str(a["target_id"]))
                             for a in (c_data or {}).get("assignments", [])
                             if "telescope_id" in a and "target_id" in a]
            ) if c_data else CoordinatorAction()

            ex_actions = []
            for r in raw_ex:
                d = parse_json(r)
                if d and d.get("action") in ("observe", "wait", "request_reassign", "abort"):
                    ex_actions.append(ExecutorAction(action=d["action"], target_id=d.get("target_id"),
                                                     minutes=d.get("minutes"), reason=d.get("reason")))
                else:
                    ex_actions.append(ExecutorAction(action="wait"))
            while len(ex_actions) < 3:
                ex_actions.append(ExecutorAction(action="wait"))

            action = NetworkAction(planner_action=planner_action,
                                   coordinator_action=coord_action,
                                   executor_actions=ex_actions)
            result = env.step(action)
            r = result.reward or 0.0

            n_parsed = int(p_data is not None) + int(c_data is not None) + \
                       sum(1 for raw in raw_ex if parse_json(raw) is not None)
            parse_rate = n_parsed / max(2 + len(e_obs_list), 1)

            steps_data.append({"role": "planner",     "prompt": PLANNER_SYSTEM + "\n\n" + p_obs.narrative,     "completion": raw_p, "reward": r, "parse_rate": parse_rate})
            steps_data.append({"role": "coordinator", "prompt": COORDINATOR_SYSTEM + "\n\n" + c_obs.narrative, "completion": raw_c, "reward": r, "parse_rate": parse_rate})
            for e, raw_e in zip(e_obs_list, raw_ex):
                steps_data.append({"role": "executor", "prompt": EXECUTOR_SYSTEM.format(site=e.site_name) + "\n\n" + e.narrative, "completion": raw_e, "reward": r, "parse_rate": parse_rate})

    return steps_data

# ---------------------------------------------------------------------------
# SFT warm-start
# ---------------------------------------------------------------------------
def sft_warmstart(model, tokenizer):
    if not os.path.exists(SFT_DATA):
        print(f"[WARN] SFT data not found at {SFT_DATA} — skipping")
        return

    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    records = []
    with open(SFT_DATA) as f:
        for line in f:
            rec = json.loads(line)
            system = rec["messages"][0]["content"]
            user   = rec["messages"][1]["content"]
            resp   = rec["response"]
            text   = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{resp}<|im_end|>"
            records.append({"text": text})

    dataset = Dataset.from_list(records)
    print(f"[SFT] Warm-starting on {len(dataset)} examples ...", flush=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir="outputs/sft",
            num_train_epochs=SFT_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LR * 2,
            logging_steps=10,
            save_strategy="no",
            bf16=False,
            fp16=False,   # MPS uses float16 natively via model dtype
            no_cuda=True,
            use_mps_device=True,
            report_to="none",
        ),
        dataset_text_field="text",
        max_seq_length=1024,
    )
    trainer.train()
    print("[SFT] Warm-start complete.", flush=True)

# ---------------------------------------------------------------------------
# GRPO training
# ---------------------------------------------------------------------------
def train_grpo(model, tokenizer, task_id: str):
    from trl import GRPOTrainer, GRPOConfig
    from datasets import Dataset

    print(f"\n[GRPO] Starting task={task_id} steps={GRPO_STEPS_PER_TASK}", flush=True)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from client import AstrofEnv

    prompts = []
    with AstrofEnv(base_url=ENV_BASE_URL).sync() as env:
        for seed in range(20):
            result = env.reset(task_id=task_id, seed=seed)
            obs = result.observation
            prompts.append({"prompt": PLANNER_SYSTEM + "\n\n" + obs.planner_obs.narrative})

    dataset = Dataset.from_list(prompts)

    def reward_fn(completions, prompts=None, **kwargs):
        import re
        rewards = []
        for completion in completions:
            try:
                text = completion if isinstance(completion, str) else (
                    completion[0]["content"] if isinstance(completion, list) else str(completion)
                )
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                m = re.search(r"\{.*\}", text, re.DOTALL)
                if not m:
                    rewards.append(0.0)
                    continue
                data = json.loads(m.group())
                targets = data.get("targets", [])
                if isinstance(targets, list) and len(targets) > 0:
                    valid = sum(1 for t in targets if "target_id" in t and "score" in t)
                    r = 0.5 + 0.5 * (valid / max(len(targets), 1))
                else:
                    r = 0.1
                rewards.append(float(r))
            except Exception:
                rewards.append(0.0)
        return rewards

    grpo_config = GRPOConfig(
        output_dir=f"outputs/grpo_{task_id}",
        num_train_epochs=1,
        max_steps=GRPO_STEPS_PER_TASK,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        logging_steps=5,
        save_steps=50,
        bf16=False,
        fp16=False,   # MPS handles dtype via model
        no_cuda=True,
        use_mps_device=True,
        report_to="none",
        temperature=0.8,
        num_generations=2,         # lower than CUDA version to save memory
        max_completion_length=256,
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
    )
    trainer.train()
    print(f"[GRPO] task={task_id} done.", flush=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    model, tokenizer = load_model()

    sft_warmstart(model, tokenizer)

    for task_id in CURRICULUM:
        train_grpo(model, tokenizer, task_id)

    os.makedirs("outputs/final", exist_ok=True)
    model.save_pretrained("outputs/final")
    tokenizer.save_pretrained("outputs/final")
    print("[DONE] Adapter saved to outputs/final/", flush=True)

    if HF_REPO_ID and HF_TOKEN:
        model.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
        tokenizer.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
        print(f"[DONE] Pushed to {HF_REPO_ID}", flush=True)


if __name__ == "__main__":
    main()
