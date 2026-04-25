"""
ASTROF — GRPOTrainer Training Script
=====================================
Run on Colab A100 (40GB) with compute credits provided at the hackathon.

Quick-start:
    pip install trl unsloth openenv-core openai
    # Set ENV_BASE_URL to your deployed HF Space URL
    ENV_BASE_URL=https://<your-space>.hf.space python scripts/train_grpo.py

Curriculum: easy → medium → hard (expert after if time allows)
SFT warm-start: loads data/sft_warmstart.jsonl before GRPO if present
"""
import json
import os
import sys
import torch
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from novelty_detector import NoveltyDetector
from continual_learner import ContinualLearner

# ---------------------------------------------------------------------------
# Imports — lazy so the file is importable without GPU deps installed
# ---------------------------------------------------------------------------
def _check_deps():
    try:
        import torch, trl, transformers, unsloth  # noqa: F401
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("Install: pip install trl unsloth openenv-core openai transformers accelerate")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME     = os.getenv("MODEL_NAME", "unsloth/Qwen3-1.7B-unsloth-bnb-4bit")
ENV_BASE_URL   = os.getenv("ENV_BASE_URL", "http://localhost:7860")
API_BASE_URL   = os.getenv("API_BASE_URL", "http://localhost:11434/v1")
HF_TOKEN       = os.getenv("HF_TOKEN", "ollama")
HF_REPO_ID     = os.getenv("HF_REPO_ID", "")       # set to push trained adapter

MAX_STEPS_PER_TASK = {"easy": 18, "medium": 44, "hard": 32, "expert": 18}
CURRICULUM = ["easy", "medium", "hard"]             # add "expert" if time allows
GRPO_STEPS_PER_TASK = 100                           # increase to 300+ for better results
BATCH_SIZE = 4
GRAD_ACCUM = 4
LR = 5e-5
SFT_EPOCHS = 1                                       # warm-start epochs before GRPO
SFT_DATA = os.path.join(os.path.dirname(__file__), "../data/sft_warmstart.jsonl")

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
# Rollout: run one episode, return (prompts, completions, rewards)
# ---------------------------------------------------------------------------

def _make_client():
    from openai import OpenAI
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def _run_episode(task_id: str, model_fn) -> List[dict]:
    """
    Run one episode using model_fn(system, user) -> str.
    Returns list of {"prompt": str, "completion": str, "reward": float} dicts.
    """
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
                f_p = pool.submit(model_fn, PLANNER_SYSTEM, p_obs.narrative)
                f_c = pool.submit(model_fn, COORDINATOR_SYSTEM, c_obs.narrative)
                f_ex = [pool.submit(model_fn, EXECUTOR_SYSTEM.format(site=e.site_name), e.narrative)
                        for e in e_obs_list]

            raw_p = f_p.result()
            raw_c = f_c.result()
            raw_ex = [f.result() for f in f_ex]

            # Parse
            p_data = parse_json(raw_p)
            c_data = parse_json(raw_c)

            planner_action = PlannerAction(
                targets=[TargetScore(target_id=str(t["target_id"]),
                                     score=float(t.get("score", 0.5)))
                         for t in (p_data or {}).get("targets", []) if "target_id" in t],
                too_flag=(p_data or {}).get("too_flag", "dismiss")
            ) if p_data else PlannerAction()

            coord_action = CoordinatorAction(
                assignments=[AssignmentItem(telescope_id=str(a["telescope_id"]),
                                            target_id=str(a["target_id"]))
                             for a in (c_data or {}).get("assignments", [])
                             if "telescope_id" in a and "target_id" in a]
            ) if c_data else CoordinatorAction()

            ex_actions = []
            for r in raw_ex:
                d = parse_json(r)
                if d and d.get("action") in ("observe", "wait", "request_reassign", "abort"):
                    ex_actions.append(ExecutorAction(
                        action=d["action"],
                        target_id=d.get("target_id"),
                        minutes=d.get("minutes"),
                        reason=d.get("reason"),
                    ))
                else:
                    ex_actions.append(ExecutorAction(action="wait"))
            while len(ex_actions) < 3:
                ex_actions.append(ExecutorAction(action="wait"))

            action = NetworkAction(planner_action=planner_action,
                                   coordinator_action=coord_action,
                                   executor_actions=ex_actions)
            result = env.step(action)
            r = result.reward or 0.0

            # Track parse rate for novelty detection
            n_parsed = (
                int(p_data is not None)
                + int(c_data is not None)
                + sum(1 for raw in raw_ex if parse_json(raw) is not None)
            )
            n_tel = len(e_obs_list)
            parse_rate = n_parsed / max(2 + n_tel, 1)

            # Record all 5 (prompt, completion) pairs with shared step reward + parse_rate
            steps_data.append({"role": "planner",     "prompt": PLANNER_SYSTEM + "\n\n" + p_obs.narrative,     "completion": raw_p, "reward": r, "parse_rate": parse_rate})
            steps_data.append({"role": "coordinator", "prompt": COORDINATOR_SYSTEM + "\n\n" + c_obs.narrative, "completion": raw_c, "reward": r, "parse_rate": parse_rate})
            for i, (e, raw_e) in enumerate(zip(e_obs_list, raw_ex)):
                steps_data.append({"role": "executor", "prompt": EXECUTOR_SYSTEM.format(site=e.site_name) + "\n\n" + e.narrative, "completion": raw_e, "reward": r, "parse_rate": parse_rate})

    return steps_data


# ---------------------------------------------------------------------------
# SFT warm-start
# ---------------------------------------------------------------------------

def sft_warmstart(model, tokenizer):
    """Fine-tune one epoch on the human-generated warm-start data."""
    if not os.path.exists(SFT_DATA):
        print(f"[WARN] SFT data not found at {SFT_DATA} — skipping warm-start")
        print("       Run scripts/generate_sft_data.py first!")
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
            bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
            fp16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
            report_to="none",
        ),
        dataset_text_field="text",
        max_seq_length=1024,
    )
    trainer.train()
    print("[SFT] Warm-start complete.", flush=True)


# ---------------------------------------------------------------------------
# Direct model inference — used during training (no Ollama/API needed)
# ---------------------------------------------------------------------------

def _make_direct_model_fn(model, tokenizer):
    """
    Returns a model_fn(system, user) -> str that runs the model directly
    in GPU memory. Used during GRPO rollouts and continual loop on Colab
    where no external API server is running.
    """
    import torch

    def model_fn(system: str, user: str) -> str:
        prompt = (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Decode only the newly generated tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        import re
        return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    return model_fn


def _make_api_model_fn():
    """
    Returns a model_fn that calls an external OpenAI-compatible API.
    Used during inference.py (production) or local Ollama testing.
    Falls back gracefully if API_BASE_URL is not set.
    """
    if not API_BASE_URL:
        raise RuntimeError("API_BASE_URL not set — use _make_direct_model_fn() for training.")
    llm_client = _make_client()

    def model_fn(system: str, user: str) -> str:
        try:
            resp = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
                max_tokens=256,
            )
            import re
            raw = resp.choices[0].message.content or ""
            return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        except Exception as exc:
            print(f"[LLM_ERROR] {exc}", flush=True)
            return ""

    return model_fn

def train_grpo(model, tokenizer, task_id: str):
    """One GRPO phase for a single task."""
    from trl import GRPOTrainer, GRPOConfig
    import torch

    print(f"\n[GRPO] Starting task={task_id} steps={GRPO_STEPS_PER_TASK}", flush=True)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from client import AstrofEnv
    from datasets import Dataset

    prompts = []
    with AstrofEnv(base_url=ENV_BASE_URL).sync() as env:
        for seed in range(20):
            result = env.reset(task_id=task_id, seed=seed)
            obs = result.observation
            prompts.append({"prompt": PLANNER_SYSTEM + "\n\n" + obs.planner_obs.narrative})

    dataset = Dataset.from_list(prompts)

    # Use direct model inference — no Ollama needed during training
    model_fn = _make_direct_model_fn(model, tokenizer)

    def reward_fn(completions, prompts=None, **kwargs):
        """
        Reward based on completion quality only — no model re-inference.
        Avoids CUDA context conflicts on T4 from calling model inside reward.
        """
        import json, re
        rewards = []
        for completion in completions:
            try:
                text = completion if isinstance(completion, str) else (
                    completion[0]["content"] if isinstance(completion, list) else str(completion)
                )
                # strip thinking tags
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                # find JSON
                m = re.search(r"\{.*\}", text, re.DOTALL)
                if not m:
                    rewards.append(0.0)
                    continue
                data = json.loads(m.group())
                # planner reward: valid targets list with scores
                targets = data.get("targets", [])
                if isinstance(targets, list) and len(targets) > 0:
                    valid = sum(1 for t in targets if "target_id" in t and "score" in t)
                    r = 0.5 + 0.5 * (valid / max(len(targets), 1))
                else:
                    r = 0.1
                rewards.append(float(r))
            except Exception as e:
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
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
        fp16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
        report_to="none",
        temperature=0.8,
        num_generations=4,
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

    os.makedirs(f"outputs/grpo_{task_id}", exist_ok=True)
    log_path = f"outputs/grpo_{task_id}/training_log.json"
    with open(log_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"[GRPO] Log saved → {log_path}", flush=True)


# ---------------------------------------------------------------------------
# Continual learning loop — runs after curriculum, self-triggers on novelty
# ---------------------------------------------------------------------------

CONTINUAL_EPISODES  = int(os.getenv("CONTINUAL_EPISODES", "200"))   # total episodes to run
CONTINUAL_TASKS     = ["hard", "expert"]                              # tasks to rotate through
NOVELTY_THRESHOLD   = 0.3
NOVELTY_WINDOW      = 20
TRIGGER_N           = 30


def run_continual_loop(model, tokenizer):
    """
    Post-curriculum continual learning loop.

    Runs episodes across hard/expert tasks. NoveltyDetector watches every step.
    When enough novel episodes accumulate, ContinualLearner automatically fires
    a targeted LoRA training run — no human input needed.
    Uses direct model inference — no Ollama or API server required.
    """
    # Direct model inference — weights live in GPU memory
    model_fn = _make_direct_model_fn(model, tokenizer)

    detector = NoveltyDetector(window=NOVELTY_WINDOW, threshold=NOVELTY_THRESHOLD)
    learner  = ContinualLearner(
        model=model,
        tokenizer=tokenizer,
        trigger_n=TRIGGER_N,
        novelty_threshold=NOVELTY_THRESHOLD,
    )

    print(f"\n[CONTINUAL] Starting continual loop: {CONTINUAL_EPISODES} episodes over {CONTINUAL_TASKS}", flush=True)

    for episode_idx in range(CONTINUAL_EPISODES):
        task_id = CONTINUAL_TASKS[episode_idx % len(CONTINUAL_TASKS)]
        detector.reset_episode()

        try:
            # Run episode and score each step for novelty
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from client import AstrofEnv
            from models import (NetworkAction, PlannerAction, CoordinatorAction,
                                ExecutorAction, TargetScore, AssignmentItem)
            import re, json as _json
            from concurrent.futures import ThreadPoolExecutor

            def parse_json_local(text):
                text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
                m = re.search(r"\{.*\}", text, re.DOTALL)
                if m:
                    try:
                        return _json.loads(m.group())
                    except Exception:
                        pass
                return None

            episode_data = []

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

                    raw_p  = f_p.result()
                    raw_c  = f_c.result()
                    raw_ex = [f.result() for f in f_ex]

                    p_data = parse_json_local(raw_p)
                    c_data = parse_json_local(raw_c)
                    n_tel  = len(e_obs_list)
                    n_parsed = int(p_data is not None) + int(c_data is not None) + \
                               sum(1 for r in raw_ex if parse_json_local(r) is not None)
                    parse_rate = n_parsed / max(2 + n_tel, 1)

                    planner_action = PlannerAction(
                        targets=[TargetScore(target_id=str(t["target_id"]),
                                             score=float(t.get("score", 0.5)))
                                 for t in (p_data or {}).get("targets", []) if "target_id" in t],
                        too_flag=(p_data or {}).get("too_flag", "dismiss")
                    ) if p_data else PlannerAction()

                    coord_action = CoordinatorAction(
                        assignments=[AssignmentItem(telescope_id=str(a["telescope_id"]),
                                                    target_id=str(a["target_id"]))
                                     for a in (c_data or {}).get("assignments", [])
                                     if "telescope_id" in a and "target_id" in a]
                    ) if c_data else CoordinatorAction()

                    ex_actions = []
                    for r in raw_ex:
                        d = parse_json_local(r)
                        if d and d.get("action") in ("observe", "wait", "request_reassign", "abort"):
                            ex_actions.append(ExecutorAction(
                                action=d["action"], target_id=d.get("target_id"),
                                minutes=d.get("minutes"), reason=d.get("reason"),
                            ))
                        else:
                            ex_actions.append(ExecutorAction(action="wait"))
                    while len(ex_actions) < 3:
                        ex_actions.append(ExecutorAction(action="wait"))

                    action = NetworkAction(
                        planner_action=planner_action,
                        coordinator_action=coord_action,
                        executor_actions=ex_actions,
                        parse_rate=parse_rate,
                    )
                    result = env.step(action)
                    step_reward = result.reward or 0.0

                    # Score this step for novelty
                    step_novelty = detector.score(obs, step_reward, parse_rate)

                    # Collect step data
                    episode_data.append({"role": "planner",     "prompt": PLANNER_SYSTEM + "\n\n" + p_obs.narrative,     "completion": raw_p, "reward": step_reward, "parse_rate": parse_rate})
                    episode_data.append({"role": "coordinator", "prompt": COORDINATOR_SYSTEM + "\n\n" + c_obs.narrative, "completion": raw_c, "reward": step_reward, "parse_rate": parse_rate})
                    for e, raw_e in zip(e_obs_list, raw_ex):
                        episode_data.append({"role": "executor", "prompt": EXECUTOR_SYSTEM.format(site=e.site_name) + "\n\n" + e.narrative, "completion": raw_e, "reward": step_reward, "parse_rate": parse_rate})

            ep_novelty = detector.episode_novelty()
            print(
                f"[CONTINUAL] ep={episode_idx+1}/{CONTINUAL_EPISODES} "
                f"task={task_id} novelty={ep_novelty:.2f} "
                f"status={learner.status()}",
                flush=True,
            )

            # Hand episode to learner — it decides whether to trigger training
            learner.observe(episode_data, ep_novelty)

        except Exception as e:
            print(f"[CONTINUAL] Episode {episode_idx} failed: {e}", flush=True)
            continue

    print(f"\n[CONTINUAL] Loop complete. {learner.status()}", flush=True)
    return model   # model may have been updated by learner


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _check_deps()
    from unsloth import FastLanguageModel
    import torch

    print(f"[INIT] Loading {MODEL_NAME} ...", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=2048,
        dtype=None,           # auto (bf16 on A100)
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=2026,
    )

    # 1. SFT warm-start
    sft_warmstart(model, tokenizer)

    # 2. GRPO curriculum
    for task_id in CURRICULUM:
        train_grpo(model, tokenizer, task_id)

    # 3. Continual learning loop — self-triggers on novelty, no human input needed
    model = run_continual_loop(model, tokenizer)

    # 4. Save final adapter
    os.makedirs("outputs/final", exist_ok=True)
    model.save_pretrained("outputs/final")
    tokenizer.save_pretrained("outputs/final")
    print("[DONE] Adapter saved to outputs/final/", flush=True)

    if HF_REPO_ID:
        model.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
        tokenizer.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
        print(f"[DONE] Pushed to {HF_REPO_ID}", flush=True)


if __name__ == "__main__":
    main()
