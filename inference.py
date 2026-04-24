"""
Inference Script — ASTROF Multi-Agent Telescope Scheduling Environment
======================================================================
MANDATORY

- Before submitting, ensure the following variables are defined in your
  environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the
  root directory of the project.
- Participants must use OpenAI Client for all LLM calls using above variables.

One model, 5 role-conditioned system prompts, 5 parallel LLM calls per step.
Fallback default actions applied for any unparseable response — server never crashes.
"""
import json
import os
import re
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI

from client import AstrofEnv
from models import (
    AssignmentItem,
    CoordinatorAction,
    ExecutorAction,
    NetworkAction,
    PlannerAction,
    TargetScore,
)

API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

TEMPERATURE = 0.2
MAX_TOKENS = 512
SUCCESS_SCORE_THRESHOLD = 0.4
_EPS = 1e-4

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ---------------------------------------------------------------------------
# System prompts — one per role
# ---------------------------------------------------------------------------

PLANNER_SYSTEM = textwrap.dedent("""
    You are the Science Planner for a global telescope network observing exoplanets.
    Score each visible target 0.0–1.0 (higher = more urgent to observe tonight).
    Consider: scientific priority, altitude/airmass, transit deadlines, ToO alerts.
    If a ToO alert is present, classify it: dismiss / queue / interrupt.
    If a target has a category label, factor its urgency into your score.
    Reply with ONLY valid JSON, no explanation. /no_think
    Format: {"targets": [{"target_id": "<name>", "score": <0.0-1.0>}], "too_flag": "dismiss|queue|interrupt"}
""").strip()

COORDINATOR_SYSTEM = textwrap.dedent("""
    You are the Network Coordinator for a global telescope network.
    Assign each available (non-failed) telescope to its best unobserved target.
    Never assign two telescopes to the same target.
    If too_flag=interrupt, redirect the fastest idle telescope to the ToO immediately.
    Reply with ONLY valid JSON, no explanation. /no_think
    Format: {"assignments": [{"telescope_id": "<id>", "target_id": "<name>"}]}
    telescope_ids: mauna_kea | la_palma | siding_spring
""").strip()

EXECUTOR_SYSTEM = textwrap.dedent("""
    You are Telescope Executor at {site}.
    Given your assignment and local conditions, choose one action:
      observe  — conditions good, proceed with assigned target
      wait     — clouds thin, hold for better seeing
      request_reassign — conditions too poor, escalate to Coordinator
      abort    — catastrophic hardware fault only
    Reply with ONLY valid JSON, no explanation. /no_think
    Format: {{"action": "observe|wait|request_reassign|abort", "target_id": "<name>|null", "minutes": <int|null>, "reason": "<str|null>"}}
""").strip()

# ---------------------------------------------------------------------------
# Default fallback actions
# ---------------------------------------------------------------------------

DEFAULT_PLANNER = PlannerAction(targets=[], too_flag="dismiss")
DEFAULT_COORDINATOR = CoordinatorAction(assignments=[])
DEFAULT_EXECUTOR = ExecutorAction(action="wait")

# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def _call_llm(system: str, user: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return _strip_think(resp.choices[0].message.content or "")
    except Exception as exc:
        print(f"[LLM_ERROR] {exc}", flush=True)
        return ""


def _extract_json(text: str) -> Optional[dict]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None

# ---------------------------------------------------------------------------
# Parsers — one per role
# ---------------------------------------------------------------------------


def parse_planner(raw: str) -> PlannerAction:
    data = _extract_json(raw)
    if not data:
        return DEFAULT_PLANNER
    try:
        targets = [
            TargetScore(target_id=str(t["target_id"]), score=float(t.get("score", 0.5)))
            for t in data.get("targets", [])
            if "target_id" in t
        ]
        too_flag = data.get("too_flag", "dismiss")
        if too_flag not in ("dismiss", "queue", "interrupt"):
            too_flag = "dismiss"
        return PlannerAction(targets=targets, too_flag=too_flag)
    except Exception:
        return DEFAULT_PLANNER


def parse_coordinator(raw: str) -> CoordinatorAction:
    data = _extract_json(raw)
    if not data:
        return DEFAULT_COORDINATOR
    try:
        assignments = [
            AssignmentItem(
                telescope_id=str(a["telescope_id"]),
                target_id=str(a["target_id"]),
            )
            for a in data.get("assignments", [])
            if "telescope_id" in a and "target_id" in a
        ]
        return CoordinatorAction(assignments=assignments)
    except Exception:
        return DEFAULT_COORDINATOR


def parse_executor(raw: str) -> ExecutorAction:
    data = _extract_json(raw)
    if not data:
        return DEFAULT_EXECUTOR
    try:
        action = data.get("action", "wait")
        if action not in ("observe", "wait", "request_reassign", "abort"):
            action = "wait"
        return ExecutorAction(
            action=action,
            target_id=data.get("target_id"),
            minutes=data.get("minutes"),
            reason=data.get("reason"),
        )
    except Exception:
        return DEFAULT_EXECUTOR

# ---------------------------------------------------------------------------
# Grade helper (client-side, mirrors server compute_grade)
# ---------------------------------------------------------------------------


def compute_grade(task_id: str, state) -> float:
    dup_penalty = min(getattr(state, "duplicate_count", 0) * 0.05, 0.3)
    if task_id == "easy":
        dl = min(getattr(state, "deadlines_met", 0) / 3.0, 1.0)
        pri = min(state.total_priority_observed / 133.0, 1.0)
        raw = 0.6 * dl + 0.4 * pri
    elif task_id == "medium":
        raw = min(state.total_priority_observed / 202.0, 1.0) - dup_penalty
    elif task_id == "hard":
        pri = min(state.total_priority_observed / 202.0, 1.0)
        dl = min(getattr(state, "deadlines_met", 0) / 2.0, 1.0)
        raw = 0.5 * pri + 0.5 * dl - dup_penalty
    else:  # expert
        pri = min(state.total_priority_observed / 133.0, 1.0)
        n_too = 3
        too = min(getattr(state, "too_responses", 0) / n_too, 1.0)
        nc = getattr(state, "new_category_handled", 0.0)
        raw = 0.4 * pri + 0.3 * too + 0.3 * nc - dup_penalty
    return round(max(_EPS, min(raw, 1.0 - _EPS)), 4)

# ---------------------------------------------------------------------------
# Single-task runner
# ---------------------------------------------------------------------------


def run_task(task_id: str) -> float:
    cfg = {
        "easy": {"max_steps": 18, "n_tel": 1},
        "medium": {"max_steps": 44, "n_tel": 3},
        "hard": {"max_steps": 32, "n_tel": 3},
        "expert": {"max_steps": 18, "n_tel": 3},
    }.get(task_id, {"max_steps": 18, "n_tel": 1})

    max_steps = cfg["max_steps"]
    n_tel = cfg["n_tel"]
    rewards: List[float] = []
    score = _EPS

    print(f"[START] task={task_id} model={MODEL_NAME or 'unknown'}", flush=True)

    try:
        with AstrofEnv(base_url=ENV_BASE_URL).sync() as env:
            result = env.reset(task_id=task_id)
            start_time = time.time()

            for step in range(1, max_steps + 1):
                if result.done:
                    break
                if time.time() - start_time > 240:
                    print(f"[TIMEOUT] task={task_id} at step {step} — 4-minute limit reached", flush=True)
                    break

                obs = result.observation
                p_obs = obs.planner_obs
                c_obs = obs.coordinator_obs
                e_obs_list = obs.executor_obs

                # 5 parallel LLM calls: 1 planner + 1 coordinator + n_tel executors
                futures = {}
                with ThreadPoolExecutor(max_workers=5) as pool:
                    futures["planner"] = pool.submit(
                        _call_llm, PLANNER_SYSTEM, p_obs.narrative
                    )
                    futures["coordinator"] = pool.submit(
                        _call_llm, COORDINATOR_SYSTEM, c_obs.narrative
                    )
                    for i, e_obs in enumerate(e_obs_list):
                        sys_prompt = EXECUTOR_SYSTEM.format(site=e_obs.site_name)
                        futures[f"executor_{i}"] = pool.submit(
                            _call_llm, sys_prompt, e_obs.narrative
                        )

                raw_planner = futures["planner"].result()
                raw_coordinator = futures["coordinator"].result()
                raw_executors = [futures[f"executor_{i}"].result() for i in range(len(e_obs_list))]

                # Parse — count successes for format reward logging
                planner_action = parse_planner(raw_planner)
                coordinator_action = parse_coordinator(raw_coordinator)
                executor_actions = [parse_executor(r) for r in raw_executors]

                # Pad executor_actions to 3 if fewer telescopes in easy task
                while len(executor_actions) < 3:
                    executor_actions.append(DEFAULT_EXECUTOR)

                n_parsed = (
                    int(planner_action != DEFAULT_PLANNER)
                    + int(coordinator_action != DEFAULT_COORDINATOR)
                    + sum(int(e != DEFAULT_EXECUTOR) for e in executor_actions[:n_tel])
                )
                parse_rate = n_parsed / (2 + n_tel)

                action = NetworkAction(
                    planner_action=planner_action,
                    coordinator_action=coordinator_action,
                    executor_actions=executor_actions,
                    parse_rate=parse_rate,
                )

                result = env.step(action)
                reward = result.reward or 0.0
                rewards.append(reward)

                print(
                    f"[STEP] step={step} reward={reward:.4f} "
                    f"parse_rate={parse_rate:.2f} done={str(result.done).lower()}",
                    flush=True,
                )

            try:
                final_state = env.state()
                score = compute_grade(task_id, final_state)
            except Exception as exc:
                print(f"[WARN] state() failed: {exc} — using fallback score", flush=True)
                score = _EPS

    finally:
        rewards_str = ",".join(f"{r:.4f}" for r in rewards[:10])
        print(
            f"[END] task={task_id} score={score:.4f} steps={len(rewards)} "
            f"rewards=[{rewards_str}{',...' if len(rewards) > 10 else ''}]",
            flush=True,
        )

    return score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    results = {}
    for task in ["easy", "medium", "hard", "expert"]:
        results[task] = run_task(task)

    print("\n=== ASTROF BASELINE SCORES ===", flush=True)
    for task, score in results.items():
        bar = "#" * int(score * 20)
        print(f"  {task:<8} {score:.4f}  [{bar:<20}]", flush=True)


if __name__ == "__main__":
    main()
