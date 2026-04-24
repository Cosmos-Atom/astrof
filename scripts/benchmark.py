"""
ASTROF Baseline Benchmark
=========================
Collects the 4 data points needed for the pitch comparison table.

Usage:
    # Zero-shot LLM (needs an API key):
    HF_TOKEN=hf_xxx MODEL_NAME=Qwen/Qwen3-1.7B python scripts/benchmark.py --mode zeroshot

    # Greedy baseline (no LLM, pure heuristic):
    python scripts/benchmark.py --mode greedy

    # Random baseline (no LLM):
    python scripts/benchmark.py --mode random

    # After training — trained model:
    HF_TOKEN=hf_xxx MODEL_NAME=<your-trained-model> python scripts/benchmark.py --mode trained

Results are printed as a table AND saved to outputs/benchmark_results.json
"""
import argparse
import json
import os
import re
import sys
import time
from typing import List

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://Cosmosatom-astrof.hf.space")
HF_TOKEN     = os.getenv("HF_TOKEN", "ollama")          # Ollama ignores API key
MODEL_NAME   = os.getenv("MODEL_NAME", "qwen3:1.7b")    # ollama model tag
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:11434/v1")  # Ollama default

TASKS = ["easy", "medium", "hard", "expert"]
N_RUNS = 3   # average over N runs per task (reduces variance)

# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------

def _make_hf_model_fn(model_name: str, api_key: str, api_base: str):
    """
    OpenAI-compatible call — works with Ollama or HF Inference API.
    For Ollama (localhost), uses native /api/chat with think:false to
    disable Qwen3 extended thinking (otherwise tokens exhaust before JSON output).
    """
    is_ollama = "localhost" in api_base or "127.0.0.1" in api_base
    ollama_base = api_base.replace("/v1", "")   # http://localhost:11434

    if is_ollama:
        import requests as _req

        def model_fn(system: str, user: str) -> str:
            try:
                resp = _req.post(
                    f"{ollama_base}/api/chat",
                    json={
                        "model": model_name,
                        "think": False,   # disable extended thinking — JSON fits in <256 tokens
                        "stream": False,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user",   "content": user},
                        ],
                        "options": {"temperature": 0.2, "num_predict": 256},
                    },
                    timeout=60,
                )
                return resp.json().get("message", {}).get("content", "")
            except Exception as e:
                print(f"  [Ollama error] {e}", flush=True)
                return ""

        return model_fn

    else:
        from openai import OpenAI
        cli = OpenAI(base_url=api_base, api_key=api_key)

        def model_fn(system: str, user: str) -> str:
            try:
                resp = cli.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    max_tokens=512,
                    temperature=0.2,
                )
                msg = resp.choices[0].message
                text = msg.content or ""
                if not text.strip() and hasattr(msg, "reasoning") and msg.reasoning:
                    text = msg.reasoning or ""
                return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            except Exception as e:
                print(f"  [API error] {e}", flush=True)
                return ""

        return model_fn


def _extract_planner_names(user: str):
    """
    Parse planet names from planner narrative lines like:
      '  TOI-260 b             pri=27  alt= 45.2°  airmass=1.41'
    Returns list of names in narrative order (already sorted by priority desc).
    """
    names = []
    for line in user.splitlines():
        # Lines starting with 2 spaces and containing 'pri=' are target lines
        m = re.match(r"^\s{2}(\S.*?)\s{2,}pri=", line)
        if m:
            names.append(m.group(1).strip())
    return names


def _extract_coordinator_names(user: str):
    """
    Parse planet names from coordinator priority list lines like:
      '  TOI-260 b             score=0.900'
    Returns list of names sorted by score desc (already ordered).
    """
    names = []
    for line in user.splitlines():
        m = re.match(r"^\s{2}(\S.*?)\s{2,}score=", line)
        if m:
            names.append(m.group(1).strip())
    return names


def _make_greedy_model_fn():
    """
    Pure greedy: Planner scores by priority order in narrative,
    Coordinator assigns telescopes to top targets, Executors always observe.
    No LLM call — deterministic.
    """
    def model_fn(system: str, user: str) -> str:
        if "Science Planner" in system:
            names = _extract_planner_names(user)
            targets = [{"target_id": n, "score": round(1.0 - i * 0.05, 2)}
                       for i, n in enumerate(names[:10])]
            return json.dumps({"targets": targets, "too_flag": "dismiss"})

        elif "Network Coordinator" in system:
            tel_ids = ["mauna_kea", "la_palma", "siding_spring"]
            names = _extract_coordinator_names(user)
            assignments = [{"telescope_id": t, "target_id": n}
                           for t, n in zip(tel_ids, names) if n]
            return json.dumps({"assignments": assignments})

        else:
            return json.dumps({"action": "observe", "target_id": None,
                               "minutes": None, "reason": None})

    return model_fn


def _make_random_model_fn():
    """Random valid actions — worst-case baseline."""
    import random

    def model_fn(system: str, user: str) -> str:
        if "Science Planner" in system:
            names = _extract_planner_names(user)
            random.shuffle(names)
            targets = [{"target_id": n, "score": round(random.random(), 2)}
                       for n in names[:10]]
            return json.dumps({"targets": targets, "too_flag": "dismiss"})

        elif "Network Coordinator" in system:
            tel_ids = ["mauna_kea", "la_palma", "siding_spring"]
            names = _extract_coordinator_names(user)
            random.shuffle(names)
            assignments = [{"telescope_id": t, "target_id": n}
                           for t, n in zip(tel_ids, names) if n]
            return json.dumps({"assignments": assignments})

        else:
            action = random.choice(["observe", "observe", "observe", "wait"])
            return json.dumps({"action": action, "target_id": None,
                               "minutes": None, "reason": None})

    return model_fn


# ---------------------------------------------------------------------------
# Episode runner (inline — doesn't import train_grpo to avoid side effects)
# ---------------------------------------------------------------------------

MAX_STEPS = {"easy": 18, "medium": 44, "hard": 32, "expert": 18}

PLANNER_SYSTEM = (
    "You are the Science Planner for a global telescope network observing exoplanets.\n"
    "Score each visible target 0.0-1.0 (higher = more urgent to observe tonight).\n"
    "Consider: scientific priority, altitude/airmass, transit deadlines, ToO alerts.\n"
    "If a ToO alert is present, classify it: dismiss / queue / interrupt.\n"
    "If a target has a category label, factor its urgency into your score.\n"
    "Reply with ONLY valid JSON, no explanation. /no_think\n"
    'Format: {"targets": [{"target_id": "TOI-260 b", "score": 0.9}], "too_flag": "dismiss"}'
)
COORDINATOR_SYSTEM = (
    "You are the Network Coordinator for a global telescope network.\n"
    "Assign each available (non-failed) telescope to its best unobserved target.\n"
    "Never assign two telescopes to the same target.\n"
    "If too_flag=interrupt, redirect the fastest idle telescope to the ToO immediately.\n"
    "Reply with ONLY valid JSON, no explanation. /no_think\n"
    'Format: {"assignments": [{"telescope_id": "mauna_kea", "target_id": "TOI-260 b"}]}\n'
    "telescope_ids: mauna_kea | la_palma | siding_spring"
)
EXECUTOR_SYSTEM = (
    "You are Telescope Executor at {site}.\n"
    "Given your assignment and local conditions, choose one action:\n"
    "  observe          - conditions good, proceed with assigned target\n"
    "  wait             - clouds thin, hold for better seeing\n"
    "  request_reassign - conditions too poor, escalate to Coordinator\n"
    "  abort            - catastrophic hardware fault only\n"
    "Reply with ONLY valid JSON, no explanation. /no_think\n"
    'Format: {{"action": "observe", "target_id": null, "minutes": null, "reason": null}}'
)


def _parse_json(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


def run_episode(task_id: str, model_fn) -> dict:
    """
    Run one full episode. Returns a rich metrics dict:
    {
      "grade":           float,   # final science yield from /grade
      "parse_rate":      float,   # fraction of LLM outputs that parsed as valid JSON
      "duplicate_rate":  float,   # fraction of steps with telescope collision
      "deadlines_hit":   int,     # deadlines met before cutoff
      "deadlines_total": int,     # total deadline targets in episode
      "too_latency":     int|None,# steps until first interrupt action (expert only)
      "n_observed":      int,     # total planets observed
      "step_rewards":    [float], # per-step reward trace (for training curve plots)
      "step_parse":      [float], # per-step parse rate trace
      "steps":           int,     # total steps taken
    }
    """
    from client import AstrofEnv
    from models import (NetworkAction, PlannerAction, CoordinatorAction,
                        ExecutorAction, TargetScore, AssignmentItem)
    from concurrent.futures import ThreadPoolExecutor

    for attempt in range(3):
        try:
            return _run_episode_inner(task_id, model_fn, AstrofEnv, NetworkAction,
                                      PlannerAction, CoordinatorAction, ExecutorAction,
                                      TargetScore, AssignmentItem, ThreadPoolExecutor)
        except Exception as e:
            if attempt < 2:
                print(f"  [retry {attempt+1}/3] {e}", flush=True)
                time.sleep(12)
            else:
                raise


def _run_episode_inner(task_id, model_fn, AstrofEnv, NetworkAction, PlannerAction,
                       CoordinatorAction, ExecutorAction, TargetScore, AssignmentItem,
                       ThreadPoolExecutor) -> dict:
    import requests as _req

    step_rewards = []
    step_parse   = []
    too_latency  = None
    step_num     = 0

    with AstrofEnv(base_url=ENV_BASE_URL, message_timeout_s=120.0).sync() as env:
        result = env.reset(task_id=task_id)

        for _ in range(MAX_STEPS[task_id]):
            if result.done:
                break
            step_num += 1
            obs = result.observation
            p_obs = obs.planner_obs
            c_obs = obs.coordinator_obs
            e_obs_list = obs.executor_obs

            with ThreadPoolExecutor(max_workers=5) as pool:
                f_p  = pool.submit(model_fn, PLANNER_SYSTEM, p_obs.narrative)
                f_c  = pool.submit(model_fn, COORDINATOR_SYSTEM, c_obs.narrative)
                f_ex = [pool.submit(model_fn, EXECUTOR_SYSTEM.format(site=e.site_name), e.narrative)
                        for e in e_obs_list]

            raw_p  = f_p.result()
            raw_c  = f_c.result()
            raw_ex = [f.result() for f in f_ex]

            p_data = _parse_json(raw_p)
            c_data = _parse_json(raw_c)
            ex_data = [_parse_json(r) for r in raw_ex]

            # Parse rate: fraction of 2+n_tel outputs that parsed
            n_parsed = int(p_data is not None) + int(c_data is not None) + \
                       sum(1 for d in ex_data if d is not None)
            parse_rate = n_parsed / (2 + len(e_obs_list))
            step_parse.append(round(parse_rate, 3))

            planner_action = PlannerAction(
                targets=[TargetScore(target_id=str(t["target_id"]),
                                     score=float(t.get("score", 0.5)))
                         for t in (p_data or {}).get("targets", []) if "target_id" in t],
                too_flag=(p_data or {}).get("too_flag", "dismiss"),
            ) if p_data else PlannerAction()

            # Track ToO latency: first step where planner says interrupt
            if too_latency is None and planner_action.too_flag == "interrupt":
                too_latency = step_num

            coord_action = CoordinatorAction(
                assignments=[AssignmentItem(telescope_id=str(a["telescope_id"]),
                                            target_id=str(a["target_id"]))
                             for a in (c_data or {}).get("assignments", [])
                             if "telescope_id" in a and "target_id" in a]
            ) if c_data else CoordinatorAction()

            ex_actions = []
            for d in ex_data:
                if d and d.get("action") in ("observe", "wait", "request_reassign", "abort"):
                    minutes = d.get("minutes")
                    if isinstance(minutes, str):
                        minutes = None
                    ex_actions.append(ExecutorAction(action=d["action"],
                                                     target_id=d.get("target_id"),
                                                     minutes=minutes,
                                                     reason=d.get("reason")))
                else:
                    ex_actions.append(ExecutorAction(action="wait"))
            while len(ex_actions) < 3:
                ex_actions.append(ExecutorAction(action="wait"))

            action = NetworkAction(planner_action=planner_action,
                                   coordinator_action=coord_action,
                                   executor_actions=ex_actions)
            result = env.step(action)
            step_rewards.append(round(result.reward or 0.0, 4))

        # Final state — all env-side metrics
        final_state = env.state()
        n_steps = final_state.step

        # Duplicate rate: duplicate_count / steps (fraction of steps with collision)
        dup_rate = round(final_state.duplicate_count / max(n_steps, 1), 4)

        # Grade score from /grade endpoint
        try:
            resp = _req.post(f"{ENV_BASE_URL}/grade",
                             json=final_state.model_dump(), timeout=15)
            grade = resp.json().get("score", 0.0)
        except Exception as e:
            print(f"  [grade fallback] {e}", flush=True)
            grade = result.reward or 0.0

    return {
        "grade":           round(float(grade), 4),
        "parse_rate":      round(sum(step_parse) / max(len(step_parse), 1), 4),
        "duplicate_rate":  dup_rate,
        "deadlines_hit":   final_state.deadlines_met,
        "deadlines_total": final_state.deadlines_met_before_cutoff,
        "too_latency":     too_latency,
        "n_observed":      final_state.n_observed,
        "step_rewards":    step_rewards,
        "step_parse":      step_parse,
        "steps":           step_num,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["greedy", "random", "zeroshot", "trained"],
                        required=True, help="Which baseline to run")
    parser.add_argument("--tasks", nargs="+", default=TASKS,
                        help="Which tasks to benchmark (default: all 4)")
    parser.add_argument("--runs", type=int, default=N_RUNS,
                        help="Runs per task to average (default: 3)")
    args = parser.parse_args()

    print(f"\nASTROF Baseline Benchmark — mode={args.mode}, runs={args.runs}")
    print(f"ENV: {ENV_BASE_URL}")
    if args.mode in ("zeroshot", "trained"):
        print(f"Model: {MODEL_NAME} via {API_BASE_URL}")
    print("=" * 55)

    # Build model function
    if args.mode == "greedy":
        model_fn = _make_greedy_model_fn()
    elif args.mode == "random":
        model_fn = _make_random_model_fn()
    elif args.mode in ("zeroshot", "trained"):
        model_fn = _make_hf_model_fn(MODEL_NAME, HF_TOKEN, API_BASE_URL)
    else:
        raise ValueError(args.mode)

    results = {}
    for task in args.tasks:
        runs_data = []
        for run in range(1, args.runs + 1):
            t0 = time.time()
            metrics = run_episode(task, model_fn)
            elapsed = time.time() - t0
            metrics["elapsed_s"] = round(elapsed, 1)
            runs_data.append(metrics)

            # Print per-run summary
            dl = f"{metrics['deadlines_hit']}/{metrics['deadlines_total']}" \
                 if metrics['deadlines_total'] > 0 else "—"
            too = str(metrics['too_latency']) + "steps" if metrics['too_latency'] else "—"
            print(
                f"  {task:8s}  run {run}/{args.runs}"
                f"  grade={metrics['grade']:.4f}"
                f"  parse={metrics['parse_rate']:.0%}"
                f"  dup={metrics['duplicate_rate']:.0%}"
                f"  deadlines={dl}"
                f"  ToO={too}"
                f"  ({elapsed:.0f}s)",
                flush=True,
            )
            time.sleep(8)

        # Aggregate across runs
        def avg(key): return round(sum(r[key] for r in runs_data) / len(runs_data), 4)
        results[task] = {
            "runs": runs_data,
            "avg": {
                "grade":          avg("grade"),
                "parse_rate":     avg("parse_rate"),
                "duplicate_rate": avg("duplicate_rate"),
                "deadlines_hit":  avg("deadlines_hit"),
                "n_observed":     avg("n_observed"),
                "too_latency":    avg("too_latency") if any(r["too_latency"] for r in runs_data) else None,
            }
        }
        a = results[task]["avg"]
        print(f"  {task:8s}  AVG  grade={a['grade']:.4f}  parse={a['parse_rate']:.0%}"
              f"  dup={a['duplicate_rate']:.0%}  deadlines={a['deadlines_hit']:.1f}"
              f"  observed={a['n_observed']:.1f}\n")

    # Save full structured log
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    out_path = os.path.join(ROOT, f"outputs/benchmark_{args.mode}.json")
    log = {
        "mode":      args.mode,
        "model":     MODEL_NAME,
        "env":       ENV_BASE_URL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results":   results,
    }
    with open(out_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Full log saved to {out_path}")

    # Summary table
    print("\n" + "=" * 75)
    print(f"  {'Task':<10} {'Grade':>7} {'Parse%':>7} {'Dup%':>6} {'Deadlines':>10} {'ToO':>8}")
    print("  " + "-" * 63)
    for task, r in results.items():
        a = r["avg"]
        dl = f"{a['deadlines_hit']:.1f}" if a['deadlines_hit'] else "—"
        too = f"{a['too_latency']:.1f}st" if a['too_latency'] else "—"
        print(f"  {task:<10} {a['grade']:>7.4f} {a['parse_rate']:>7.0%}"
              f" {a['duplicate_rate']:>6.0%} {dl:>10} {too:>8}")
    print("=" * 75)
    print(f"\nMode: {args.mode.upper()}")
    if args.mode == "trained":
        print("  <- AFTER training. Compare against zeroshot for your headline result.")
    elif args.mode == "zeroshot":
        print("  <- Zero-shot LLM. No RL training.")
    elif args.mode == "greedy":
        print("  <- Greedy heuristic. No LLM.")
    elif args.mode == "random":
        print("  <- Random baseline. Lower bound.")


if __name__ == "__main__":
    main()
