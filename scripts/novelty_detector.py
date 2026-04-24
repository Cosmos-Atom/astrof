"""
NoveltyDetector — detects when the model is in unfamiliar territory.

Does NOT need to know what changed. Infers novelty from three observable symptoms:
  1. Reward drops below its own rolling baseline (performance signal)
  2. Parse rate drops — model output format is confused (confidence signal)
  3. A structurally new thing appears in the observation (hard signal)

Usage:
    detector = NoveltyDetector()
    score = detector.score(obs, step_reward, parse_rate)
    if score > detector.threshold:
        # flag this episode as novel
"""
import math
from collections import deque
from typing import Optional


class NoveltyDetector:
    """
    Scores each step 0.0–1.0 for novelty.
    Aggregates per episode via max() — if any step was novel, the episode was novel.
    """

    def __init__(self, window: int = 20, threshold: float = 0.3):
        self.window = window
        self.threshold = threshold

        # Rolling history (per-step rewards and parse rates)
        self._reward_history: deque = deque(maxlen=window)
        self._parse_history: deque = deque(maxlen=window)

        # Structural memory — things we've seen before
        self._seen_categories: set = set()
        self._seen_too_texts: set = set()

        # Per-episode accumulator
        self._episode_scores: list = []

    # ------------------------------------------------------------------
    # Per-step scoring
    # ------------------------------------------------------------------

    def score(
        self,
        obs,                  # NetworkObservation
        step_reward: float,
        parse_rate: float,
    ) -> float:
        novelty = 0.0

        # --- Signal 1: reward significantly below rolling mean ---
        if len(self._reward_history) >= 10:
            mean_r = sum(self._reward_history) / len(self._reward_history)
            variance = sum((r - mean_r) ** 2 for r in self._reward_history) / len(self._reward_history)
            std_r = math.sqrt(variance) if variance > 0 else 0.01
            if step_reward < mean_r - 2.0 * std_r:
                # How many standard deviations below?
                drop = (mean_r - step_reward) / std_r
                novelty += min(0.4, 0.1 * drop)   # caps at 0.4

        # --- Signal 2: parse rate dropped vs baseline ---
        if len(self._parse_history) >= 5:
            mean_p = sum(self._parse_history) / len(self._parse_history)
            if parse_rate < mean_p - 0.2:          # meaningfully worse than baseline
                novelty += 0.25
        elif parse_rate < 0.5:                      # cold start: absolute floor
            novelty += 0.25

        # --- Signal 3: structurally new things in observation ---
        novelty += self._structural_novelty(obs)

        novelty = min(novelty, 1.0)

        # Update histories
        self._reward_history.append(step_reward)
        self._parse_history.append(parse_rate)
        self._episode_scores.append(novelty)

        return novelty

    def _structural_novelty(self, obs) -> float:
        score = 0.0
        try:
            planner_obs = obs.planner_obs

            # New planet category appearing for the first time
            for planet in planner_obs.planets:
                if planet.category and planet.category not in self._seen_categories:
                    self._seen_categories.add(planet.category)
                    score += 0.5   # strong signal — new concept in environment

            # ToO alert text pattern we haven't seen before
            if planner_obs.too_alert:
                # Fingerprint the alert type (first 40 chars)
                fingerprint = (planner_obs.too_alert or "")[:40]
                if fingerprint not in self._seen_too_texts:
                    self._seen_too_texts.add(fingerprint)
                    score += 0.3

            # Coordinator sees a ToO interrupt flag but Planner dismissed it
            coord_obs = obs.coordinator_obs
            if planner_obs.too_alert and coord_obs.too_flag == "dismiss":
                score += 0.2   # model failed to classify the alert

        except AttributeError:
            pass   # obs structure unexpected — not a crash condition

        return min(score, 0.8)   # structural alone caps at 0.8

    # ------------------------------------------------------------------
    # Per-episode aggregation
    # ------------------------------------------------------------------

    def episode_novelty(self) -> float:
        """Call at end of episode. Returns max novelty seen during the episode."""
        if not self._episode_scores:
            return 0.0
        return max(self._episode_scores)

    def reset_episode(self):
        """Call at the start of each new episode."""
        self._episode_scores = []

    def is_novel(self) -> bool:
        return self.episode_novelty() > self.threshold

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        return {
            "reward_baseline": round(sum(self._reward_history) / max(len(self._reward_history), 1), 4),
            "parse_baseline":  round(sum(self._parse_history)  / max(len(self._parse_history),  1), 4),
            "seen_categories": sorted(self._seen_categories),
            "seen_too_types":  len(self._seen_too_texts),
            "episode_novelty": round(self.episode_novelty(), 4),
        }
