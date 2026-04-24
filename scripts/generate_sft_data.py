"""
Generate SFT warm-start data deterministically — no API key required.

Runs the greedy policy across many seeds and records (observation, ideal_response)
pairs for all 5 agent roles. These are used to warm-start the model before GRPO
so it outputs parseable JSON from step 1 of training.

Run: PYTHONPATH=.:server/ python scripts/generate_sft_data.py
Output: data/sft_warmstart.jsonl (~180 records, ~60 per major role)
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../server"))

from server.environment import ObservatoryNetworkEnv, TASK_CONFIGS
from models import (
    AssignmentItem, CoordinatorAction, ExecutorAction,
    NetworkAction, PlannerAction, TargetScore,
)

OUT = os.path.join(os.path.dirname(__file__), "../data/sft_warmstart.jsonl")
SITES = ["mauna_kea", "la_palma", "siding_spring"]

PLANNER_SYSTEM = """You are the Science Planner for a global telescope network observing exoplanets.
Score each visible target 0.0-1.0 (higher = more urgent to observe tonight).
Consider: scientific priority score, airmass (lower=better), transit deadlines, ToO alerts.
If a ToO alert is present, classify it: dismiss/queue/interrupt.
If category: gravitational_wave_host appears, score those targets highest immediately.
Reply with ONLY valid JSON, no explanation. /no_think
Format: {"targets": [{"target_id": "<name>", "score": <0.0-1.0>}], "too_flag": "dismiss|queue|interrupt"}"""

COORDINATOR_SYSTEM = """You are the Network Coordinator for a global telescope network.
Assign each available telescope to its best unobserved target from the priority list.
Never assign two telescopes to the same target.
If too_flag=interrupt, redirect the fastest idle telescope to the ToO target.
Reply with ONLY valid JSON, no explanation. /no_think
Format: {"assignments": [{"telescope_id": "<id>", "target_id": "<name>"}]}
telescope_ids: mauna_kea | la_palma | siding_spring"""

EXECUTOR_SYSTEM = """You are Telescope Executor at {site}.
Given your assignment and local conditions, choose one action:
  observe  — conditions good, proceed with assigned target
  wait     — clouds very thin, hold briefly
  request_reassign — conditions too poor to observe
  abort    — catastrophic hardware fault only (rare)
Reply with ONLY valid JSON, no explanation. /no_think
Format: {{"action": "observe|wait|request_reassign|abort", "target_id": "<name>|null", "minutes": <int|null>, "reason": "<str|null>"}}"""


def _record(role, system, user, response):
    return {
        "role": role,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response": response,
    }


def collect_episode(task_id: str, seed: int) -> list:
    """Run one greedy episode and collect (obs, ideal_response) for every step."""
    env = ObservatoryNetworkEnv()
    obs = env.reset(task_id=task_id, seed=seed)
    cfg = TASK_CONFIGS[task_id]
    active_sites = SITES[:cfg["n_telescopes"]]
    records = []

    for _ in range(cfg["max_steps"]):
        if obs.done:
            break

        p_obs = obs.planner_obs
        c_obs = obs.coordinator_obs
        e_obs_list = obs.executor_obs

        # --- Ideal Planner response ---
        visible_unobs = sorted(
            [p for p in p_obs.planets if p.visible and not p.observed],
            key=lambda x: (-int(x.has_deadline), -(x.category == "gravitational_wave_host"), -x.priority_score, x.airmass)
        )
        max_pri = max((p.priority_score for p in visible_unobs), default=1)
        planner_targets = [
            {"target_id": p.name, "score": round(min(1.0, p.priority_score / max(max_pri, 1)), 3)}
            for p in visible_unobs
        ]
        too_flag = "interrupt" if p_obs.too_alert and "interrupt" in (p_obs.too_alert or "") else \
                   "queue" if p_obs.too_alert and "queue" in (p_obs.too_alert or "") else \
                   "interrupt" if p_obs.too_alert else "dismiss"
        planner_resp = json.dumps({"targets": planner_targets, "too_flag": too_flag})
        records.append(_record("planner", PLANNER_SYSTEM, p_obs.narrative, planner_resp))

        # --- Ideal Coordinator response ---
        assignments_ideal = []
        used = set()
        sorted_by_priority = sorted(visible_unobs, key=lambda x: (-int(x.has_deadline), -x.priority_score))
        tel_statuses = {t.telescope_id: t.status for t in c_obs.telescope_statuses}
        for site_id in active_sites:
            if tel_statuses.get(site_id) == "failed":
                continue
            for p in sorted_by_priority:
                if p.name not in used:
                    assignments_ideal.append({"telescope_id": site_id, "target_id": p.name})
                    used.add(p.name)
                    break
        coord_resp = json.dumps({"assignments": assignments_ideal})
        records.append(_record("coordinator", COORDINATOR_SYSTEM, c_obs.narrative, coord_resp))

        # --- Ideal Executor responses ---
        assigned_map = {a["telescope_id"]: a["target_id"] for a in assignments_ideal}
        for e_obs in e_obs_list:
            site_id = e_obs.telescope_id
            assigned = assigned_map.get(site_id)
            weather = e_obs.weather
            airmass = e_obs.airmass
            if weather == "bad" or airmass > 2.5:
                ex_resp = json.dumps({"action": "request_reassign", "target_id": None, "minutes": None, "reason": "weather too poor"})
            elif weather == "partial" and airmass > 2.0:
                ex_resp = json.dumps({"action": "wait", "target_id": None, "minutes": 15, "reason": "partial cloud, waiting for clearing"})
            elif assigned:
                ex_resp = json.dumps({"action": "observe", "target_id": assigned, "minutes": None, "reason": None})
            else:
                ex_resp = json.dumps({"action": "wait", "target_id": None, "minutes": None, "reason": "no assignment"})
            sys_prompt = EXECUTOR_SYSTEM.format(site=e_obs.site_name)
            records.append(_record("executor", sys_prompt, e_obs.narrative, ex_resp))

        # --- Execute greedy step ---
        assignments_for_step = [
            AssignmentItem(telescope_id=a["telescope_id"], target_id=a["target_id"])
            for a in assignments_ideal
        ]
        ex_actions = []
        for site_id in active_sites:
            t = assigned_map.get(site_id)
            if t:
                ex_actions.append(ExecutorAction(action="observe", target_id=t))
            else:
                ex_actions.append(ExecutorAction(action="wait"))
        while len(ex_actions) < 3:
            ex_actions.append(ExecutorAction(action="wait"))

        action = NetworkAction(
            planner_action=PlannerAction(
                targets=[TargetScore(target_id=t["target_id"], score=t["score"]) for t in planner_targets[:5]],
                too_flag=too_flag,
            ),
            coordinator_action=CoordinatorAction(assignments=assignments_for_step),
            executor_actions=ex_actions,
        )
        obs = env.step(action)

    return records


def main():
    all_records = []
    tasks_seeds = [
        ("easy",   list(range(0, 20))),
        ("medium", list(range(0, 10))),
        ("hard",   list(range(0, 10))),
        ("expert", list(range(0, 10))),
    ]

    for task_id, seeds in tasks_seeds:
        task_records = []
        for seed in seeds:
            recs = collect_episode(task_id, seed)
            task_records.extend(recs)
        by_role = {}
        for r in task_records:
            by_role.setdefault(r["role"], []).append(r)
        for role, recs in by_role.items():
            print(f"  {task_id}/{role}: {len(recs)} records")
        all_records.extend(task_records)

    with open(OUT, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nTotal: {len(all_records)} records written to {OUT}")
    by_role = {}
    for r in all_records:
        by_role.setdefault(r["role"], []).append(r)
    for role, recs in sorted(by_role.items()):
        print(f"  {role}: {len(recs)}")


if __name__ == "__main__":
    main()
