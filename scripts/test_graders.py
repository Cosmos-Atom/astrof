"""
Grader unit tests — known-answer episodes for all 4 tasks.
Run: PYTHONPATH=.:server/ python scripts/test_graders.py
All asserts must pass before the hackathon.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../server"))

from server.environment import ObservatoryNetworkEnv, _EPS
from models import (
    AssignmentItem, CoordinatorAction, ExecutorAction,
    NetworkAction, PlannerAction, TargetScore,
)

SITES = ["mauna_kea", "la_palma", "siding_spring"]

def make_action(site_targets: dict, too_flag="dismiss"):
    """site_targets: {site_id: planet_name} — None means wait."""
    assignments = [
        AssignmentItem(telescope_id=s, target_id=t)
        for s, t in site_targets.items() if t
    ]
    ex_actions = []
    for s in SITES:
        t = site_targets.get(s)
        if t:
            ex_actions.append(ExecutorAction(action="observe", target_id=t))
        else:
            ex_actions.append(ExecutorAction(action="wait"))
    return NetworkAction(
        planner_action=PlannerAction(
            targets=[TargetScore(target_id=t, score=0.9) for t in site_targets.values() if t],
            too_flag=too_flag,
        ),
        coordinator_action=CoordinatorAction(assignments=assignments),
        executor_actions=ex_actions,
    )


def run_full_greedy(task_id):
    """Run a greedy episode and return final grade + state."""
    env = ObservatoryNetworkEnv()
    obs = env.reset(task_id=task_id)
    from server.environment import TASK_CONFIGS
    cfg = TASK_CONFIGS[task_id]
    sites = SITES[:cfg["n_telescopes"]]

    for _ in range(cfg["max_steps"]):
        visible_unobs = sorted(
            [p for p in obs.planner_obs.planets if p.visible and not p.observed],
            key=lambda x: (-int(x.has_deadline), -x.priority_score)
        )
        site_targets = {}
        used = set()
        for s in sites:
            for p in visible_unobs:
                if p.name not in used:
                    site_targets[s] = p.name
                    used.add(p.name)
                    break
        too_flag = "interrupt" if obs.planner_obs.too_alert else "dismiss"
        obs = env.step(make_action(site_targets, too_flag))
        if obs.done:
            break

    return env.compute_grade(), env._state


# ---------------------------------------------------------------------------
# Test: all-wait episode → minimum grade
# ---------------------------------------------------------------------------
def test_all_wait_gives_min():
    for task_id in ["easy", "medium", "hard", "expert"]:
        env = ObservatoryNetworkEnv()
        obs = env.reset(task_id=task_id)
        from server.environment import TASK_CONFIGS
        for _ in range(TASK_CONFIGS[task_id]["max_steps"]):
            obs = env.step(NetworkAction())
            if obs.done:
                break
        grade = env.compute_grade()
        assert grade == _EPS, f"{task_id}: expected _EPS for all-wait, got {grade}"
    print("PASS  test_all_wait_gives_min")


# ---------------------------------------------------------------------------
# Test: grade strictly in (_EPS, 1-_EPS)
# ---------------------------------------------------------------------------
def test_grade_bounds():
    for task_id in ["easy", "medium", "hard", "expert"]:
        grade, _ = run_full_greedy(task_id)
        assert _EPS <= grade <= 1.0 - _EPS, f"{task_id}: grade {grade} out of bounds"
    print("PASS  test_grade_bounds")


# ---------------------------------------------------------------------------
# Test: easy grader formula
# ---------------------------------------------------------------------------
def test_easy_grader_formula():
    env = ObservatoryNetworkEnv()
    obs = env.reset(task_id="easy")

    # Observe exactly 3 deadline planets in first 5 steps, then 5 more
    deadline_targets = [p.name for p in obs.planner_obs.planets if p.has_deadline][:3]
    other_targets = [p.name for p in obs.planner_obs.planets if not p.has_deadline][:5]
    all_targets = deadline_targets + other_targets

    observed_count = 0
    for _ in range(18):
        visible_unobs = [p for p in obs.planner_obs.planets if p.visible and not p.observed]
        target = None
        for t in all_targets:
            if any(p.name == t for p in visible_unobs):
                target = t
                break
        site_targets = {"mauna_kea": target} if target else {}
        obs = env.step(make_action(site_targets))
        if obs.done:
            break

    grade = env.compute_grade()
    # Grade must be > _EPS (something was observed)
    assert grade > _EPS, f"easy: grade {grade} — nothing observed"
    # Grade formula: 0.6*(dl_met/3) + 0.4*(pri/133)
    state = env._state
    expected_dl = min(state.deadlines_met / 3.0, 1.0)
    expected_pri = min(state.total_priority_observed / 133.0, 1.0)
    expected_raw = 0.6 * expected_dl + 0.4 * expected_pri
    expected = round(max(_EPS, min(expected_raw, 1.0 - _EPS)), 4)
    assert abs(grade - expected) < 1e-3, f"easy: grade={grade} expected={expected}"
    print(f"PASS  test_easy_grader_formula  (grade={grade:.4f}, dl={state.deadlines_met}, pri={state.total_priority_observed:.0f})")


# ---------------------------------------------------------------------------
# Test: duplicate observation → penalised
# ---------------------------------------------------------------------------
def test_duplicate_penalty():
    env = ObservatoryNetworkEnv()
    obs = env.reset(task_id="medium")
    first_visible = next((p.name for p in obs.planner_obs.planets if p.visible), None)
    assert first_visible, "No visible planet at reset"

    # Both mauna_kea and la_palma try to observe the same planet
    action = make_action({"mauna_kea": first_visible, "la_palma": first_visible, "siding_spring": first_visible})
    result = env.step(action)
    # Only 1 unique observation — set size should be 1
    assert env._state.n_observed == 1, f"Expected 1 unique obs, got {env._state.n_observed}"
    print(f"PASS  test_duplicate_penalty  (n_observed={env._state.n_observed})")


# ---------------------------------------------------------------------------
# Test: greedy grades are better than random
# ---------------------------------------------------------------------------
def test_greedy_beats_random():
    """Greedy on easy must beat 0.5 random baseline (easy has deadlines — random often misses them)."""
    import random as _r

    def run_random_easy(seed):
        rng = _r.Random(seed)
        env = ObservatoryNetworkEnv()
        obs = env.reset(task_id="easy")
        for _ in range(18):
            visible = [p.name for p in obs.planner_obs.planets if p.visible and not p.observed]
            target = rng.choice(visible) if visible else None
            obs = env.step(make_action({"mauna_kea": target} if target else {}))
            if obs.done:
                break
        return env.compute_grade()

    greedy, _ = run_full_greedy("easy")
    rand_avg = sum(run_random_easy(s) for s in range(5)) / 5
    assert greedy > rand_avg, f"easy: greedy {greedy:.4f} not better than avg-random {rand_avg:.4f}"
    print(f"PASS  test_greedy_beats_random  easy  greedy={greedy:.4f} vs avg-random={rand_avg:.4f}")


# ---------------------------------------------------------------------------
# Test: SUPPORTS_CONCURRENT_SESSIONS — two independent episodes
# ---------------------------------------------------------------------------
def test_concurrent_sessions():
    env1 = ObservatoryNetworkEnv()
    env2 = ObservatoryNetworkEnv()
    obs1 = env1.reset(task_id="easy", seed=1)
    obs2 = env2.reset(task_id="easy", seed=99)

    # Step env1 once with a real observation
    visible1 = [p.name for p in obs1.planner_obs.planets if p.visible]
    env1.step(make_action({"mauna_kea": visible1[0] if visible1 else None}))

    # env2 should be unaffected
    assert env2._state.n_observed == 0, "env2 contaminated by env1 step"
    assert env1._state.n_observed >= 0
    print("PASS  test_concurrent_sessions")


# ---------------------------------------------------------------------------
# Benchmark: print all greedy baseline scores
# ---------------------------------------------------------------------------
def benchmark_all():
    print("\n=== GREEDY BASELINE SCORES ===")
    for task_id in ["easy", "medium", "hard", "expert"]:
        grade, state = run_full_greedy(task_id)
        bar = "#" * int(grade * 20)
        print(f"  {task_id:<8} {grade:.4f}  [{bar:<20}]  "
              f"n_obs={state.n_observed}  pri={state.total_priority_observed:.0f}")


if __name__ == "__main__":
    test_all_wait_gives_min()
    test_grade_bounds()
    test_easy_grader_formula()
    test_duplicate_penalty()
    test_greedy_beats_random()
    test_concurrent_sessions()
    benchmark_all()
    print("\nAll tests passed.")
