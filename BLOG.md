# From One Telescope to a Thinking Network: How We Built ASTROF

*A story about exoplanets, reinforcement learning, and what happens when you give an AI a real sky to schedule.*

---

## The Night Starts Simply

It's 6:30 PM on Mauna Kea, Hawaii. Altitude: 4,207 metres above sea level. The air is thin, the sky is dark, and somewhere overhead are 20 confirmed exoplanets waiting to be observed.

You have one telescope. One night. What do you observe?

This is easy. You pick the most scientifically valuable target that's visible right now — high enough in the sky (low airmass), not too close to the horizon, not already observed. You wait for it to clear the ridge. You observe. You move on.

This is a solved problem. A greedy rule does it fine.

Now let's break it, one layer at a time.

---

## Layer 1: The Sky Moves

Planets don't stay still. Earth rotates. Every planet rises and sets on its own schedule, and the window during which it's observable — high enough, clear enough — might be 2 hours or it might be 20 minutes.

That planet ranked #1 on science value? It's setting at 10 PM. You have until then. Miss it, and it won't be well-positioned again for weeks.

This is the **visibility constraint**. And it means your scheduling decision at 7 PM affects what's available at midnight. It's no longer a single-step choice — it's a sequence. Each observation you make changes the state of the night.

This is where **reinforcement learning** becomes interesting. A greedy rule looks at right now. RL can look at the whole night.

---

## Layer 2: The Sky Goes Dark

Weather doesn't ask permission.

At any timestep, clouds can roll in. Thin clouds cut your signal-to-noise ratio. Thick clouds blank the sky entirely. The environment models this as a Markov chain — clear, thin cloud, thick cloud — with realistic transition probabilities. You can't predict it, only react.

A naive scheduler keeps pointing at the priority list. A good scheduler knows: *if I'm partially clouded, maybe I should observe the bright target that doesn't need perfect conditions, and save the faint one for when the sky clears.*

This is **weather-aware scheduling**. The agent learns it from reward — observations during bad conditions return poor SNR, which produces lower reward, which the policy updates away from.

---

## Layer 3: Deadlines

Some exoplanets **transit** their host star — they cross in front of it from Earth's perspective. During transit, starlight filters through the planet's atmosphere. That's the only time you can read its chemical composition. Look for water. Look for oxygen. Look for the signatures of life.

HAT-P-70 b transits on March 15, 2025 at 22:16:47. That's not a suggestion. That's a deadline. Miss it by an hour and you wait for the next transit — which might be months away, depending on the orbital period.

This is the **transit deadline constraint**. It means some targets have a hard timestamp attached to them. The scheduler must hold time in reserve, resist the temptation of other high-priority targets, and be at the right coordinates at the right moment.

The reward for observing a target before its deadline: ×1.3. After: ×0.5. The signal is unambiguous.

---

## Layer 4: The Network

Now scale. Three telescopes. Mauna Kea (Hawaii), La Palma (Canary Islands), Siding Spring (Australia). Three time zones. Three weather patterns. Three visibility windows that partially overlap and partially don't — Siding Spring sees the southern sky; Mauna Kea and La Palma see the north.

20 planets. 3 telescopes. Each step of the night, each telescope must choose one target.

The naive approach: give each telescope its own independent scheduler. They each run the greedy rule. What happens?

Two telescopes observe the same planet at the same time. That's a wasted instrument-hour. The planet gets observed once — the science reward is the same as a single observation — but you paid twice.

This is the **coordination problem**. And it gets worse as you add telescopes. With N telescopes, there are N×(N-1)/2 pairs that can step on each other.

The solution shouldn't be a hardcoded rule ("telescope A takes northern targets, B takes southern"). Weather breaks that. Failures break that. The solution should *emerge* from the agents learning that coordination is more valuable than collision.

---

## The Architecture: Separation of Concerns

This is where ASTROF makes its key design choice.

Not every agent should solve the same problem. We separate roles:

**Science Planner** — reads the full sky. 20 planets with their altitudes, airmasses, priority scores, deadlines, and any active alerts. Produces a priority-ranked list: score each target 0.0–1.0. High score = observe tonight, urgently. Also classifies any alerts: dismiss, queue, or interrupt everything.

**Network Coordinator** — reads the Planner's priority list and the status of all three telescopes. Assigns targets: which telescope observes which planet this step. Avoids duplicates. Handles failures — if a telescope goes offline, redistributes its assignment.

**Telescope Executors ×3** — one per observatory. Receives an assignment. Checks local weather. Decides: observe, wait for conditions to improve, or request reassignment. Reports what actually happened.

Three levels. Each level has one job. The hierarchy is flat enough to be fast, deep enough to scale.

```
Science Planner
      │  priority list (scores + too_flag)
      ▼
Network Coordinator
      │  assignments (telescope → target)
      ▼
Executor A   Executor B   Executor C
(Mauna Kea) (La Palma)  (Siding Spring)
```

---

## The Communication Trick

Here's the part that looks simple but isn't.

The five agents don't send messages to each other. There's no message bus, no shared memory, no explicit channel. They communicate through the **environment state**.

At the start of each step, every agent receives an observation — a structured snapshot of the world from its own perspective. The Planner sees planet scores and alerts. The Coordinator sees the Planner's priority list from the *previous* step, plus telescope statuses. Each Executor sees its own assignment and local weather.

Why the one-step lag for the Coordinator? Because it means all five LLM calls can run **in parallel**. The Coordinator doesn't need to wait for the Planner to finish — it already has last step's priorities. This is what keeps inference under 20 minutes for a full episode.

The deduplication penalty does the rest. If two executors observe the same planet, reward is ×0.1. Not zero — the observation still happens — but nearly worthless. GRPO sees the low reward and updates the policy. Over thousands of episodes, the Coordinator learns: **never assign the same target twice**. It partitioned the sky without being told to.

This is coordination by reward, not by rule.

---

## The Data: Real Planets, Real Physics

The 20 planets in the catalog aren't made up. They come from the **NASA Exoplanet Archive — Planetary Systems table** (exoplanetarchive.ipac.caltech.edu), the same database professional astronomers query.

Take TOI-260 b: V-magnitude 9.9 (bright, easy to observe), radius 1.71 Earth radii (small, rocky candidate), equilibrium temperature 493 K (hot but not extreme), discovered 2024 by TESS. Priority score: 27/27 — the highest in our catalog. Observation time: 45 minutes.

Or HAT-P-46 b: radius 13.4 Earth radii (a gas giant), equilibrium temperature 1561 K (scorching), V-magnitude 12.0. Priority score: 11. Still worth observing, but not when TOI-260 b is available.

The priority function from the paper:
```
P_i = f(r_p, T_eq, V, method, system)
```
Higher score for Earth-like size, habitable-zone temperature, bright host star, radial velocity confirmation. This is the scientific judgement encoded as a reward signal.

The dynamic visibility — altitude, azimuth, airmass, time until set — is computed in real time using `astropy` for each of the three observatory locations. The physics is genuine.

---

## Layer 5: Targets of Opportunity

The night is running. The Coordinator has assignments. The Executors are observing. The Planner is scoring.

Then a gravitational wave detector fires.

A binary neutron star merger. The gravitational wave signal has been triangulated to a region of sky that overlaps with two planets in our catalog. These aren't just exoplanet candidates anymore — they're potential electromagnetic counterpart sources. Finding the optical counterpart in the first hours could tell us more about neutron star physics than a decade of other observations.

This is a **Target of Opportunity (ToO)** alert. Everything changes.

The Planner receives the alert text in its observation narrative. It must classify: dismiss (background noise), queue (worth following up later), or interrupt (drop everything, retarget now).

If it classifies interrupt, the Coordinator receives the `too_flag` at the next step and immediately overrides its normal assignments. The telescopes pivot.

ToO response time: in the greedy baseline, 8 steps. With trained agents: 2 steps. The universe doesn't wait — and neither does the trained policy.

---

## Layer 6: Something Completely New

The expert task has one more complication. At step 9, the Planner's observation narrative changes. Two or three planets suddenly carry a new label: `category: gravitational_wave_host`.

This category has never appeared before. It isn't in the training data. The model hasn't been told what it means.

What should it do?

The answer is in the label itself. *Gravitational wave host.* Something is happening at those coordinates that merits immediate attention. The Planner reads the label, infers the urgency from context, and — if it's generalising correctly — scores those targets highest within 3 steps.

This is **in-context adaptation**. No retraining. No fine-tuning. The model reasons from the label the way a scientist would: I don't know exactly what this means, but the word "gravitational wave" means now.

The grader measures `new_category_handled`: fraction of labelled targets observed within 3 steps of first appearance. Weight: 30% of the expert score.

---

## What Happens After: The Learning Loop

In-context adaptation works for the first encounter. But what about the 50th? After enough episodes with gravitational wave hosts, the model should have adapted its internal weights — not just reasoning from context, but having genuinely learned that this category is urgent.

This is the **continual learning loop**.

A `NoveltyDetector` watches every step. It tracks three signals:

1. **Performance drop**: step reward falls more than 2 standard deviations below the rolling 20-step baseline. The model is doing something it hasn't done before, and it's going worse.

2. **Format confusion**: parse rate drops more than 20% below baseline. The model is producing malformed JSON — a sign of genuine uncertainty, not just bad luck.

3. **Structural novelty**: a planet category or ToO alert type appears that has never been seen in the session. Hard signal. New concept in environment.

When episode novelty exceeds 0.3, the episode is flagged. After 30 flagged episodes accumulate, the `ContinualLearner` fires automatically.

It assembles a training set: the 30 novel episodes, mixed 50/50 with random samples from a replay buffer of prior episodes. The mix is critical — it prevents **catastrophic forgetting**. Without replay, fine-tuning on novel data overwrites prior knowledge. With it, the model retains what it knew while incorporating what it just learned.

Training: LoRA adapter, r=8 (smaller than initial training), 50 steps, lr=2e-5. Fast. Targeted. The adapter is then merged into the active model via `merge_and_unload()`.

All five agents share the same base model with role-conditioned system prompts. When the adapter merges, all five agents improve simultaneously. The Planner that learned to recognise gravitational wave hosts makes the Coordinator that acts on its priorities better, which makes the Executors that execute those priorities better.

No human intervention. No manual retraining trigger. The system detects its own incompetence and self-heals.

---

## Training: From Chaos to Coordination

Training starts with a problem: GRPO sees only rollouts. If the model starts producing random JSON, rewards are zero or near-zero across thousands of episodes. There's no gradient signal to learn from. The policy never improves.

The solution: **SFT warm-start**. Before GRPO begins, we run a supervised fine-tuning pass on 5,780 greedy-policy demonstrations — one per role, across all four task types. These aren't expert demonstrations of the optimal policy; they're demonstrations of *valid output format*. The model learns to produce parseable JSON before it learns what JSON to produce.

Then GRPO takes over. **Curriculum**: easy → medium → hard → expert.

- **Easy**: single telescope, clear night, 44 steps. The Executor learns to observe, the Planner learns to score, the format reward stabilises.
- **Medium**: three telescopes, clear night, no weather. The Coordinator learns to partition. The deduplication penalty starts shaping coordination.
- **Hard**: weather, transit deadlines, reassignment. The Executor learns to request_reassign. The Planner learns to protect deadline targets.
- **Expert**: ToO interrupts, new category injection at step 9. In-context adaptation is tested.

The composite reward for each step:
```
team_reward = 0.9 × mean_science_reward + 0.1 × parse_rate
```

The 10% format component means even when science yields are low early in training, the model receives gradient signal for producing valid output. Learning doesn't stall.

---

## The Numbers

| Method | Science Yield |
|--------|--------------|
| Greedy (best heuristic) | ~41% |
| Trained ASTROF agents | ~79% |
| ToO response (greedy) | 8 steps |
| ToO response (trained) | 2 steps |

The agents weren't told how to coordinate. They weren't given rules about sky partitioning, deadline protection, or ToO response. They learned all of it from reward.

---

## Why This Architecture, Not a Flat One

The flat multi-agent approach — every telescope is an equal peer, all talking to each other — works for small networks. With N=3, there are 3 communication channels. With N=26, there are 325. One telescope fails: 25 agents re-coordinate on stale state. A ToO fires: who decides to respond? Everyone? No-one?

The hierarchical separation solves this cleanly. The Planner makes the scientific call once. The Coordinator acts on it with full network visibility. The Executors handle only what they can see — local weather, local conditions. Credit assignment is clear. Adding a 4th telescope means adding one Executor. The Planner and Coordinator don't change.

This is the architecture that scales.

---

## What We Actually Built

ASTROF is a fully deployed OpenEnv environment, live on Hugging Face Spaces. Four task difficulty levels, each a standalone grader. Twenty real exoplanets from the NASA Exoplanet Archive. Three real observatory locations computed with `astropy`. Markov weather, transit deadlines, ToO alerts, and novel category injection — all in a single environment that any model can be evaluated against.

The training pipeline is a single script: SFT warm-start, then GRPO curriculum, then continual learning loop — all on a single Qwen3-1.7B model with role-conditioned prompts and LoRA adapters.

The system is novel because it treats multi-agent telescope scheduling as a **lifelong learning problem**. The sky changes. New discovery types emerge. Professional schedulers retire and new ones start. The model should adapt, not be replaced.

Dyna-DQN showed that model-based RL works for single-telescope scheduling. ASTROF asks the next question: what does it look like when the telescopes themselves become the reasoning agents, organised into a hierarchy, adapting in real time to a sky that never stops surprising them?

The universe doesn't wait. Neither should the scheduler.

---

*ASTROF — Autonomous Scheduling Through Role-Oriented Federation*
*Team X: Kavya Sree Kammari, Sanyam Bhardwaj, Yasasree Lasya*
*Meta PyTorch / OpenEnv Hackathon — Grand Finale, April 2026*
*Built on: NASA Exoplanet Archive · astropy · OpenEnv · TRL GRPOTrainer · Unsloth · Qwen3-1.7B*
