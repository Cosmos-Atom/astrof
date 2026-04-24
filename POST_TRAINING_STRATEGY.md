# ASTROF — Post-Training & Self-Improvement Strategy

---

## Overview

ASTROF uses a two-stage approach: GRPO-based reinforcement learning for initial training, followed by automated continual learning that allows the agent to adapt to new scientific discovery types without forgetting prior knowledge.

---

## Stage 1: Initial Training (GRPO)

**Stack:** Unsloth + TRL GRPOTrainer + Qwen3-1.7B + LoRA (r=16)

**Curriculum (easy → medium → hard → expert):**

| Task | What the agent learns |
|------|-----------------------|
| Easy | Single-telescope scheduling, priority scoring, basic JSON output format |
| Medium | Cross-telescope coordination, duplicate avoidance, sky partitioning |
| Hard | Weather-resilient reassignment, transit deadline management |
| Expert | ToO interrupt handling, in-context adaptation to new target categories |

**Reward signal (composite, 5 components):**
- Team yield: `priority_weighted_observations / max_possible` (40%)
- Planner: `high_priority_observed / available − deadline_penalty` (20%)
- Coordinator: `science_time / total + too_response_bonus + recovery_bonus − dup_penalty` (20%)
- Executor: `SNR_achieved / SNR_theoretical_max` (10%)
- Format: JSON parse success rate across all 5 roles (10%)

All rewards clamped to `(1e-4, 1 - 1e-4)`. The format reward ensures the model receives gradient signal even in early training when science yields are low.

**Warm-start:** A light SFT pass on 5,780 greedy-policy demonstrations (`data/sft_warmstart.jsonl`) runs before GRPO. This primes the model to output valid JSON across all 5 role-conditioned prompts before RL begins — preventing the cold-start problem where GRPO sees only zero-reward rollouts.

---

## Stage 2: Continual Learning (Post-Training)

After initial training, two components monitor deployed performance and automatically adapt when the model encounters unfamiliar situations.

### NoveltyDetector (`scripts/novelty_detector.py`)

Monitors every step using three independent signals:

| Signal | How it works |
|--------|--------------|
| **Performance drop** | Step reward falls >2 standard deviations below rolling 20-step baseline |
| **Format confusion** | Parse rate drops >20% below rolling baseline — model output degrades |
| **Structural novelty** | A planet category or ToO alert type appears that was never seen before |

Each signal contributes independently to a novelty score (0.0–1.0). Episode novelty = max step score across the episode. If episode novelty > 0.3, the episode is flagged for the ContinualLearner.

This design is intentional: the detector does not need to know *what* changed, only that the model is performing worse than its own baseline. It fires on both distributional shift (new phenomena) and performance regression (environment changes).

### ContinualLearner (`scripts/continual_learner.py`)

When 30 novel episodes accumulate, training triggers automatically:

```
Novel episodes (30) + Replay sample (50% mix)
         │
         ▼
  SFT on high-reward steps only (reward > 0.05)
  LoRA r=8, lr=2e-5, 50 steps
         │
         ▼
  New adapter saved → merged into active model
  Novel buffer cleared → replay buffer updated
```

**Key design choices:**

- **Replay mixing (50/50):** Each adapter trains on novel episodes mixed with random samples from `data/replay_buffer.jsonl` (capped at 500 episodes). This prevents catastrophic forgetting — the model retains prior task knowledge while incorporating new patterns.

- **Smaller LoRA for updates (r=8 vs r=16):** Initial training uses r=16 for broad capability learning. Continual updates use r=8 — smaller, targeted, faster. Full-model retraining is never required.

- **Reward threshold filtering:** Only steps with reward > 0.05 are used for training. This prevents the model from reinforcing its own confused outputs on novel inputs.

- **No human intervention required:** The full loop (detect → buffer → train → merge) runs automatically. An astronomer discovering a new phenomenon type only needs to update the environment's target catalog — the model self-adapts within the next 30 novel episodes.

### Expert Task Demonstration

The expert task (`environment.py:444`) demonstrates this capability in a controlled setting: at step 9, 2–3 planets are relabeled as `category: gravitational_wave_host` in the Planner's observation narrative. The agent must recognize the new label and prioritize those targets within 3 steps. This tests in-context generalization — the model's ability to infer the urgency of a novel category from context alone, without any retraining.

`new_category_handled` = fraction of `gravitational_wave_host` targets observed within 3 steps of first appearance (grader weight: 30% of expert score).

---

## Summary

| Capability | Mechanism | Where |
|-----------|-----------|-------|
| Initial skill acquisition | GRPO + curriculum | `scripts/train_grpo.py` |
| Format reliability | SFT warm-start + format reward | `data/sft_warmstart.jsonl`, `environment.py:431` |
| Novel category adaptation (inference-time) | In-context generalization via system prompt | `inference.py:63`, `environment.py:444` |
| Novel category adaptation (training-time) | NoveltyDetector → ContinualLearner → LoRA merge | `scripts/novelty_detector.py`, `scripts/continual_learner.py` |
| Forgetting prevention | Replay buffer mixing (50/50) | `data/replay_buffer.jsonl`, `continual_learner.py:154` |

The system is designed so that as real telescopes encounter new discovery types — gravitational waves, fast radio bursts, exotic transients — the scheduling agent adapts automatically. Old knowledge remains intact. New capability goes live within hours of enough novel episodes accumulating.
