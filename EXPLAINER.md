# ASTROF — How It Works

**Autonomous Scheduling Through Role-Oriented Federation**

---

## The Problem

Every clear night, professional observatories face a hard combinatorial problem. You have a list of exoplanet targets, each with a scientific priority score, a narrow observable window, and sometimes a hard deadline (a planet transit that happens once and won't come back). You have multiple telescopes spread across the globe. Weather is unpredictable. Cosmic emergencies arrive without warning.

A single flat agent trying to do all of this at once collapses under the complexity. It has no way to separate "which targets matter most scientifically" from "which telescope should take which target right now" from "the clouds just rolled in at La Palma, what do we do."

**ASTROF solves this by splitting the problem into a hierarchy of roles — the same way a real observatory network is actually staffed.**

---

## The 5 Agents

```
                    ┌─────────────────────┐
                    │   Science Planner   │  ← "What should we observe tonight?"
                    └──────────┬──────────┘
                               │  scored priority list
                    ┌──────────▼──────────┐
                    │  Network Coordinator│  ← "Who goes where?"
                    └──┬──────┬──────┬───┘
                       │      │      │  telescope assignments
              ┌────────▼┐  ┌──▼────┐ ┌▼────────┐
              │Executor │  │Executor│ │Executor │
              │Mauna Kea│  │La Palma│ │Siding   │
              │(Hawaii) │  │(Spain) │ │Spring   │
              └─────────┘  └───────┘ └─────────┘
                                        (Australia)
```

### Science Planner
Looks at the full sky catalog — 20 real exoplanet targets from the NASA Exoplanet Archive — and scores each one 0.0–1.0 based on scientific priority, current altitude/airmass, transit deadlines, and any active cosmic alerts. Also classifies alerts as `dismiss`, `queue`, or `interrupt`.

### Network Coordinator
Reads the Planner's priority list and the current status of all 3 telescopes (weather, failures, reassign requests) and assigns each telescope to a specific target. Never sends two telescopes to the same target. Handles failures and reallocations.

### Telescope Executors (×3)
Each executor runs locally at one site. Given its assignment and local conditions (weather, airmass, SNR estimate), it decides: `observe`, `wait`, `request_reassign`, or `abort`. If conditions are too bad, it escalates back to the Coordinator.

---

## One Time Step = One Night Slot

Each `step()` call advances the simulation by ~15 minutes of telescope time. In that slot:

1. The Planner scores all currently visible, unobserved targets
2. The Coordinator assigns telescopes to targets (using the **previous** step's priority list — explained below)
3. Each Executor either observes, waits, or escalates
4. Rewards are computed and the observation log is updated

All 5 LLM calls happen **in parallel** using `ThreadPoolExecutor(5)` — a full step completes in one round-trip latency, not five sequential ones.

---

## The One-Step Lag Design

The Planner and Coordinator cannot both run in the same parallel batch if the Coordinator needs the Planner's output. So the Coordinator reads the **previous step's** priority list. This lets all 5 calls fire simultaneously with no sequential dependency, while still giving the Coordinator valid signal — it's just one step stale, which is fine for a 15-minute scheduling slot.

```
Step N:   Planner scores targets  →  stores priority_list_N
Step N+1: Coordinator reads priority_list_N  (one step old, but still valid)
          Planner scores targets  →  stores priority_list_{N+1}
          Both run in parallel ✓
```

---

## Rewards

### Step reward (per telescope, per step)

The underlying `_TelescopeCore` (from Round 1) returns a raw reward when a telescope observes a planet:

```
raw_reward = priority_score / REWARD_SCALE   (REWARD_SCALE = 475.0)
```

Higher-priority planets yield more reward. Observing a low-altitude planet (high airmass) still works but the reward is scaled down by the environment's weather/visibility logic.

### Cross-telescope deduplication penalty

If two telescopes both observe the same planet in the same episode, the second observation gets:

```
reward = raw_reward × 0.1
```

This is the key coordination signal. The only way to avoid this penalty is for the Coordinator to assign different targets to different telescopes. There is no explicit message-passing between executors — they learn to avoid each other purely through this reward structure.

### Format reward (parse_rate)

At each step, `inference.py` counts how many of the 5 LLM calls produced valid, parseable JSON:

```
parse_rate = n_successfully_parsed / (2 + n_telescopes)
```

This is blended into the team reward at 10% weight:

```
team_reward = 0.9 × mean_science_reward + 0.1 × parse_rate
```

During GRPO training, this means the model is penalised for outputting malformed JSON even when the scheduling decision itself was good. It creates pressure to produce clean output from the very first training steps.

### Team reward (what GRPO actually sees)

```
team_reward = 0.9 × mean(per_telescope_step_rewards) + 0.1 × parse_rate
              clamped to (1e-4, 1 - 1e-4)
```

GRPO uses this scalar to credit or penalise the entire trajectory of 5 LLM calls that produced the action.

---

## Final Score (grader, end of episode)

| Task | Formula |
|------|---------|
| easy | `0.6 × (deadlines_met / 3) + 0.4 × (priority_sum / 133)` |
| medium | `priority_sum / 202 − duplicate_penalty` |
| hard | `0.5 × (priority_sum / 202) + 0.5 × (deadlines_met / 2) − duplicate_penalty` |
| expert | `0.4 × priority_yield + 0.3 × too_response_rate + 0.3 × new_category_handled` |

`duplicate_penalty = min(n_duplicates × 0.05, 0.3)`

The oracle denominators (133, 202) are the maximum priority sums achievable by the greedy policy across an entire episode — so a perfect greedy agent scores ~1.0.

---

## Why These Rewards?

**Priority sum** captures scientific value. Observing a high-priority exoplanet (high Earth Similarity Index, good atmosphere detectability, favourable stellar activity) matters more than observing a filler target.

**Deadlines met** captures time-critical science. Some planets only transit in front of their star for a narrow window. Missing that window means waiting months or years for the next one. Deadline weight is higher on easier tasks (where the main challenge is timing) and lower on harder tasks (where network coordination is the bottleneck).

**Deduplication penalty** is the multi-agent coordination signal. Without it, all three telescopes would pile onto the highest-priority target every step. With it, they must partition the sky.

**ToO response rate** captures emergency reaction. A real gamma-ray burst needs immediate telescope redirection. A model that correctly classifies `interrupt` and the Coordinator that redirects within 1–2 steps gets credit.

**New category handled** (expert only) tests in-context generalisation. At step 9, 2–3 targets are suddenly labelled `gravitational_wave_host`. The Planner has never seen this category in training. The score measures whether the agents correctly reprioritise without being retrained.

---

## How Agents Communicate — Without Explicit Messages

The agents never pass messages to each other directly. They communicate through the environment state:

| What one agent does | How another agent sees it |
|---------------------|--------------------------|
| Planner scores targets high | Coordinator's next-step observation includes that priority list |
| Coordinator assigns telescope A to planet X | Executor A sees `assignment = "planet X"` in its observation |
| Executor requests reassign | Coordinator's observation shows that site in `reassign_queue` |
| Executor observes planet X | Planet X is marked `observed=True` in all future Planner observations |
| Telescope fails | All agents see `failure_flag` in their narrative; Coordinator sees `status=FAILED` |

The reward structure does the rest. If the Coordinator keeps assigning the same target to multiple telescopes, the duplicate penalty keeps firing. GRPO sees consistently low team rewards for those assignment patterns and eventually the policy learns to distribute targets across the network.

**The coordination emerges from reward, not from the agents being explicitly told to coordinate.**

---

## What Are We Actually Achieving?

A standard single-agent RL approach would need one agent to simultaneously reason about planetary physics, network topology, local weather at three sites, transit timing, and cosmic alerts — all in a single context window. The action space would be enormous and the credit assignment nearly impossible.

ASTROF decomposes this into specialised roles with narrow action spaces:

- The Planner only outputs a scored list and a flag. Its action space is tiny.
- The Coordinator only outputs assignments. It never needs to know about airmass.
- Each Executor only decides observe/wait/reassign/abort. It never needs to know about other telescopes.

Each role can be trained independently if needed, or all 5 can be run as a single model with role-conditioned system prompts (which is what we do — one Qwen3-1.7B with 5 different prompts).

The environment is designed so that **good science emerges naturally when each role does its narrow job well** — and the reward signal is shaped to penalise any breakdown in that specialisation.

---

## Tasks as a Curriculum

| Task | What it tests | Key challenge |
|------|--------------|---------------|
| easy | Single-telescope scheduling + deadline awareness | Timing: must observe 3 deadline targets before step 5 |
| medium | Network coordination, deduplication | Partitioning 20 targets across 3 telescopes on a clear night |
| hard | Weather adaptation + deadlines at scale | Reassigning when sites cloud over, still hitting 2 hard deadlines |
| expert | In-context generalisation + emergency response | ToO interrupts mid-episode + brand-new target category injected at step 9 |

Training runs easy → medium → hard (curriculum). Expert is held out as the true generalisation test.

---

*Built on real NASA Exoplanet Archive data, astropy for altitude/airmass physics, and the OpenEnv framework.*
