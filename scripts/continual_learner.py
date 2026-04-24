"""
ContinualLearner — automatically triggers LoRA adapter training when novelty accumulates.

How it works:
  1. Receives episode data + novelty score after every episode
  2. If novelty_score > threshold, buffers the episode
  3. When buffer hits trigger_n episodes, trains a new LoRA adapter
     on novel episodes + replay sample (to prevent forgetting)
  4. Merges new adapter into the active model
  5. Clears trigger buffer, adds episodes to replay buffer

No human intervention needed. The system detects its own incompetence and self-heals.

Usage:
    learner = ContinualLearner(model, tokenizer)
    learner.observe(episode_data, novelty_score)   # call after every episode
"""
import json
import os
import random
from typing import List, Optional


ADAPTER_DIR   = os.path.join(os.path.dirname(__file__), "../outputs/adapters")
REPLAY_FILE   = os.path.join(os.path.dirname(__file__), "../data/replay_buffer.jsonl")
REPLAY_MAX    = 500    # max episodes to keep in replay buffer
TRIGGER_N     = 30     # novel episodes needed to trigger training
REPLAY_MIX    = 0.5    # fraction of training batch from replay (vs novel)
LORA_R        = 8      # smaller than initial training — targeted update
LORA_ALPHA    = 16
LORA_LR       = 2e-5   # lower LR for continual updates
LORA_STEPS    = 50     # quick targeted fine-tune


class ContinualLearner:

    def __init__(
        self,
        model=None,
        tokenizer=None,
        trigger_n: int = TRIGGER_N,
        novelty_threshold: float = 0.3,
        dry_run: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.trigger_n = trigger_n
        self.novelty_threshold = novelty_threshold
        self.dry_run = dry_run          # if True, log but don't actually train

        self._trigger_buffer: List[dict] = []    # novel episodes waiting to train
        self._adapter_count: int = 0
        self._total_triggers: int = 0

        os.makedirs(ADAPTER_DIR, exist_ok=True)
        self._load_replay_index()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def observe(self, episode_data: List[dict], novelty_score: float):
        """
        Call after every episode.
        episode_data: list of {"role", "prompt", "completion", "reward"} dicts
        novelty_score: float from NoveltyDetector.episode_novelty()
        """
        # Always add to replay buffer (regardless of novelty)
        self._append_replay(episode_data)

        if novelty_score > self.novelty_threshold:
            self._trigger_buffer.extend(episode_data)
            self._total_triggers += 1
            print(
                f"[CONTINUAL] Novel episode buffered "
                f"(score={novelty_score:.2f}, buffer={len(self._trigger_buffer)}/{self.trigger_n * 5})",
                flush=True,
            )

        # Check if we've accumulated enough novel data to train
        novel_episodes_buffered = self._total_triggers
        if novel_episodes_buffered >= self.trigger_n and self._trigger_buffer:
            self._train_adapter()
            self._trigger_buffer = []
            self._total_triggers = 0

    def status(self) -> dict:
        return {
            "adapters_trained": self._adapter_count,
            "trigger_buffer_steps": len(self._trigger_buffer),
            "novel_episodes_since_last_train": self._total_triggers,
            "trigger_threshold": self.trigger_n,
            "replay_size": self._replay_size(),
        }

    # ------------------------------------------------------------------
    # Replay buffer (disk-backed JSONL)
    # ------------------------------------------------------------------

    def _load_replay_index(self):
        self._replay_count = 0
        if os.path.exists(REPLAY_FILE):
            with open(REPLAY_FILE) as f:
                self._replay_count = sum(1 for _ in f)

    def _replay_size(self) -> int:
        if not os.path.exists(REPLAY_FILE):
            return 0
        with open(REPLAY_FILE) as f:
            return sum(1 for _ in f)

    def _append_replay(self, episode_data: List[dict]):
        with open(REPLAY_FILE, "a") as f:
            for step in episode_data:
                f.write(json.dumps(step) + "\n")

        # Trim to REPLAY_MAX episodes (each episode ~5 steps × n_steps)
        self._trim_replay()

    def _trim_replay(self):
        if not os.path.exists(REPLAY_FILE):
            return
        with open(REPLAY_FILE) as f:
            lines = f.readlines()
        if len(lines) > REPLAY_MAX * 20:    # rough: 20 steps per episode on average
            keep = lines[-(REPLAY_MAX * 20):]
            with open(REPLAY_FILE, "w") as f:
                f.writelines(keep)

    def _sample_replay(self, n: int) -> List[dict]:
        if not os.path.exists(REPLAY_FILE):
            return []
        with open(REPLAY_FILE) as f:
            lines = f.readlines()
        sample_lines = random.sample(lines, min(n, len(lines)))
        result = []
        for line in sample_lines:
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return result

    # ------------------------------------------------------------------
    # LoRA adapter training
    # ------------------------------------------------------------------

    def _train_adapter(self):
        adapter_id = f"adapter_{self._adapter_count:03d}"
        adapter_path = os.path.join(ADAPTER_DIR, adapter_id)

        n_novel = len(self._trigger_buffer)
        n_replay = int(n_novel * REPLAY_MIX)
        replay_steps = self._sample_replay(n_replay)

        training_steps = self._trigger_buffer + replay_steps
        random.shuffle(training_steps)

        print(
            f"\n[CONTINUAL] Triggering LoRA training: adapter={adapter_id} "
            f"novel_steps={n_novel} replay_steps={len(replay_steps)} "
            f"total_training_steps={len(training_steps)}",
            flush=True,
        )

        if self.dry_run:
            print(f"[CONTINUAL] DRY RUN — skipping actual training. Would save to {adapter_path}", flush=True)
            self._adapter_count += 1
            return

        if self.model is None or self.tokenizer is None:
            print("[CONTINUAL] No model loaded — cannot train. Set dry_run=True for testing.", flush=True)
            return

        try:
            self._run_lora_training(training_steps, adapter_path)
            self._merge_adapter(adapter_path)
            self._adapter_count += 1
            print(f"[CONTINUAL] Adapter {adapter_id} trained and merged. Total adapters: {self._adapter_count}", flush=True)
        except Exception as e:
            print(f"[CONTINUAL] Training failed: {e}", flush=True)

    def _run_lora_training(self, steps: List[dict], adapter_path: str):
        from trl import SFTTrainer, SFTConfig
        from peft import LoraConfig, get_peft_model
        from datasets import Dataset

        # Format into text for SFT
        records = []
        for step in steps:
            system = step.get("role", "assistant")
            prompt = step.get("prompt", "")
            completion = step.get("completion", "")
            reward = step.get("reward", 0.0)

            # Only train on steps where the model did reasonably well
            # (don't reinforce clearly wrong outputs)
            if reward < 0.05:
                continue

            text = (
                f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n{completion}<|im_end|>"
            )
            records.append({"text": text})

        if not records:
            print("[CONTINUAL] No high-quality steps to train on — skipping.", flush=True)
            return

        dataset = Dataset.from_list(records)

        # Attach a fresh small LoRA on top of current model
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "v_proj"],   # minimal target for speed
            lora_dropout=0.05,
            bias="none",
        )
        peft_model = get_peft_model(self.model, lora_config)

        trainer = SFTTrainer(
            model=peft_model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            args=SFTConfig(
                output_dir=adapter_path,
                max_steps=LORA_STEPS,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                learning_rate=LORA_LR,
                logging_steps=10,
                save_strategy="no",
                bf16=True,
                report_to="none",
            ),
            dataset_text_field="text",
            max_seq_length=512,
        )
        trainer.train()
        peft_model.save_pretrained(adapter_path)
        print(f"[CONTINUAL] Adapter saved to {adapter_path}", flush=True)

    def _merge_adapter(self, adapter_path: str):
        """Merge the new adapter weights into the base model in-place."""
        try:
            from peft import PeftModel
            merged = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = merged.merge_and_unload()
            print(f"[CONTINUAL] Adapter merged into active model.", flush=True)
        except Exception as e:
            print(f"[CONTINUAL] Merge failed (adapter still saved): {e}", flush=True)
