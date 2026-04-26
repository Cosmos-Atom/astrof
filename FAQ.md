# ASTROF — Frequently Asked Questions

---

**Q: Why use one shared model instead of five separate models?**

A: Shared weights mean all agents benefit from the same training signal simultaneously. It's also far more memory-efficient — five separate 1.7B models would be infeasible on a single GPU.

---

**Q: How do the agents coordinate if they don't communicate directly?**

A: They share a team reward — every agent gets the same score each step. This creates emergent coordination without explicit message passing, similar to how a sports team learns to coordinate through shared wins and losses.

---

**Q: What is GRPO and why use it over PPO?**

A: GRPO — Group Relative Policy Optimization — ranks completions within a batch relative to each other instead of using a separate value network. It's simpler, more memory-efficient, and works well for structured output tasks like JSON generation.

---

**Q: Why SFT warm-start before GRPO?**

A: Without it, the model produces invalid JSON in early rollouts, giving zero reward every step — GRPO has nothing to learn from. SFT on greedy demonstrations teaches the format first, so GRPO can immediately start getting signal.

---

**Q: What happens when the model encounters something it's never seen?**

A: The NoveltyDetector flags it by monitoring reward drops, parse rate collapses, and new planet categories. Once 30 novel episodes buffer up, ContinualLearner auto-trains a targeted LoRA adapter and merges it into the live model — fully autonomous.

---

**Q: Why three telescopes in different locations?**

A: Geographic spread means different sky coverage and weather conditions at any given time. The coordinator must learn to route targets to whichever telescope has the best conditions — which is the core coordination challenge.

---

**Q: How is this different from a simple scheduling algorithm?**

A: A classical scheduler uses fixed rules. ASTROF learns to handle stochastic weather, competing deadlines, Target-of-Opportunity interrupts, and novel planet categories — all simultaneously — through experience rather than hand-crafted logic.

---

**Q: What is a Target-of-Opportunity (ToO) event?**

A: It's an unexpected high-priority astronomical event — like a gravitational wave detection — that interrupts the normal schedule. The Planner must decide in real-time whether to dismiss it, queue it, or immediately interrupt ongoing observations. This is the hardest part of the expert task.

---

**Q: How does the reward function work?**

A: Each difficulty level has its own formula balancing science yield, cost, time penalties, and risk. For example, easy is `R = 1.0·S − 0.1·C`. As difficulty increases, more factors are penalized — medium adds a time penalty, hard adds risk, expert adds a variance penalty to encourage consistent performance.

---

**Q: What prevents the model from forgetting earlier tasks when continually learning?**

A: The Replay Buffer — 50% of every new LoRA training batch is sampled from old episodes. This mix of novel and historical data prevents catastrophic forgetting, a classic problem in continual learning.

---

**Q: Why LoRA instead of full fine-tuning?**

A: LoRA only trains ~1% of parameters — about 17M out of 1.7B. This makes each continual update fast (under 5 minutes) and targeted, while keeping the base model's general knowledge intact.

---

**Q: What are the baseline scores you're trying to beat?**

A: The greedy scheduler scores around 0.63–0.71 and zero-shot LLM around 0.52–0.68. ASTROF achieved 0.956 on easy, 0.791 on medium, 0.821 on hard, and 0.731 on expert — beating both baselines across all difficulty levels.

---

**Q: Could this architecture apply beyond astronomy?**

A: Yes — any multi-agent scheduling problem with stochastic conditions, competing priorities, and rare events maps cleanly onto this architecture. Think air traffic control, hospital resource allocation, or satellite network management.
