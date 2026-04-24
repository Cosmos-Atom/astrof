"""
ASTROF Smoke Test
=================
Run this on Colab BEFORE the full training run to verify everything works.

    python scripts/smoke_test.py

All checks print PASS or FAIL. If everything is PASS, you're ready to train.
Takes ~2 minutes (no GPU needed, no model loaded).
"""
import json
import os
import sys
import traceback

# Make astrof/ root importable (scripts/ → ../)
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://Cosmosatom-astrof.hf.space")

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = []


def check(name, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
        results.append((name, True))
    except Exception as e:
        print(f"  {FAIL}  {name}")
        print(f"         {e}")
        traceback.print_exc()
        results.append((name, False))


# ---------------------------------------------------------------------------
print("\n[1] Import checks")
# ---------------------------------------------------------------------------

def _import_novelty():
    from novelty_detector import NoveltyDetector
    d = NoveltyDetector()
    assert hasattr(d, "score") and hasattr(d, "episode_novelty")

def _import_continual():
    from continual_learner import ContinualLearner
    c = ContinualLearner(dry_run=True)
    assert hasattr(c, "observe") and hasattr(c, "status")

def _import_models():
    from models import (NetworkAction, PlannerAction, CoordinatorAction,
                        ExecutorAction, TargetScore, AssignmentItem)

def _import_client():
    from client import AstrofEnv

check("novelty_detector imports", _import_novelty)
check("continual_learner imports", _import_continual)
check("models imports", _import_models)
check("client imports", _import_client)

# ---------------------------------------------------------------------------
print("\n[2] Data checks")
# ---------------------------------------------------------------------------

def _sft_data():
    sft_path = os.path.join(ROOT, "data", "sft_warmstart.jsonl")
    assert os.path.exists(sft_path), f"Missing: {sft_path}"
    with open(sft_path) as f:
        lines = f.readlines()
    assert len(lines) >= 100, f"Too few SFT examples: {len(lines)}"
    record = json.loads(lines[0])
    assert "role" in record and "messages" in record, f"Unexpected format: {record.keys()}"
    roles = {json.loads(l)["role"] for l in lines[:50]}
    assert "planner" in roles, f"Missing planner role. Found: {roles}"
    assert "coordinator" in roles, f"Missing coordinator role. Found: {roles}"
    assert "executor" in roles, f"Missing executor role. Found: {roles}"
    print(f"         {len(lines)} SFT examples, roles: {sorted(roles)}")

def _planet_data():
    p1 = os.path.join(ROOT, "data", "phase1_agent_observable.csv")
    p2 = os.path.join(ROOT, "data", "phase2_priority_durations.csv")
    assert os.path.exists(p1), f"Missing: {p1}"
    assert os.path.exists(p2), f"Missing: {p2}"
    with open(p1) as f:
        rows = f.readlines()
    assert len(rows) >= 2, "phase1 CSV is empty"

check("sft_warmstart.jsonl exists and has all roles", _sft_data)
check("planet CSVs exist", _planet_data)

# ---------------------------------------------------------------------------
print("\n[3] Environment connectivity")
# ---------------------------------------------------------------------------

def _env_reset():
    from client import AstrofEnv
    with AstrofEnv(base_url=ENV_BASE_URL).sync() as env:
        result = env.reset(task_id="easy")
    assert result is not None
    assert hasattr(result, "observation"), "reset() returned no observation"
    obs = result.observation
    assert hasattr(obs, "planner_obs"), "observation missing planner_obs"
    assert hasattr(obs, "coordinator_obs"), "observation missing coordinator_obs"
    assert hasattr(obs, "executor_obs"), "observation missing executor_obs"
    print(f"         connected to {ENV_BASE_URL}")
    print(f"         planner narrative: {obs.planner_obs.narrative[:80]}...")

check(f"env.reset(easy) connects to {ENV_BASE_URL}", _env_reset)

# ---------------------------------------------------------------------------
print("\n[4] Episode runner (dummy model, 3 steps)")
# ---------------------------------------------------------------------------

def _episode_runner():
    # Patch ENV_BASE_URL before importing train_grpo so it picks up the HF Space URL
    os.environ["ENV_BASE_URL"] = ENV_BASE_URL
    import train_grpo
    train_grpo.ENV_BASE_URL = ENV_BASE_URL   # patch the module-level var directly

    from train_grpo import _run_episode, MAX_STEPS_PER_TASK

    # Dummy model that returns valid JSON for each role
    def dummy_model_fn(system: str, user: str) -> str:
        if "Science Planner" in system:
            return '{"targets": [{"target_id": "TOI-260 b", "score": 0.9}, {"target_id": "WASP-2 b", "score": 0.6}], "too_flag": "dismiss"}'
        elif "Network Coordinator" in system:
            return '{"assignments": [{"telescope_id": "mauna_kea", "target_id": "TOI-260 b"}, {"telescope_id": "la_palma", "target_id": "WASP-2 b"}, {"telescope_id": "siding_spring", "target_id": "HAT-P-23 b"}]}'
        else:
            return '{"action": "observe", "target_id": null, "minutes": null, "reason": null}'

    # Temporarily cap steps to 3 for speed
    original = MAX_STEPS_PER_TASK.copy()
    MAX_STEPS_PER_TASK["easy"] = 3

    try:
        steps = _run_episode("easy", dummy_model_fn)
    finally:
        MAX_STEPS_PER_TASK.update(original)

    assert len(steps) > 0, "Episode returned no steps"
    assert "reward" in steps[0], f"Step missing reward key: {steps[0].keys()}"
    assert "parse_rate" in steps[0], f"Step missing parse_rate key: {steps[0].keys()}"
    rewards = [s["reward"] for s in steps]
    print(f"         {len(steps)} step records, rewards: {[round(r,3) for r in rewards[:5]]}")

check("_run_episode(easy, dummy_model, 3 steps)", _episode_runner)

# ---------------------------------------------------------------------------
print("\n[5] NoveltyDetector end-to-end")
# ---------------------------------------------------------------------------

def _novelty_e2e():
    from novelty_detector import NoveltyDetector

    class FakeObs:
        class FakePlannerObs:
            planets = []
            too_alert = None
        class FakeCoordObs:
            too_flag = "dismiss"
        planner_obs = FakePlannerObs()
        coordinator_obs = FakeCoordObs()

    d = NoveltyDetector(window=5, threshold=0.3)
    obs = FakeObs()

    # Warm up with normal rewards
    for _ in range(10):
        d.score(obs, step_reward=0.7, parse_rate=1.0)

    # Inject a bad step — should trigger novelty
    score = d.score(obs, step_reward=0.01, parse_rate=0.0)
    ep_novelty = d.episode_novelty()
    assert ep_novelty > 0, f"Expected novelty > 0, got {ep_novelty}"
    print(f"         bad step novelty score: {score:.3f}, episode novelty: {ep_novelty:.3f}")

check("NoveltyDetector detects bad step", _novelty_e2e)

# ---------------------------------------------------------------------------
print("\n[6] ContinualLearner dry-run")
# ---------------------------------------------------------------------------

def _continual_dry_run():
    from continual_learner import ContinualLearner

    learner = ContinualLearner(trigger_n=3, novelty_threshold=0.3, dry_run=True)

    fake_episode = [
        {"role": "planner", "prompt": "test", "completion": '{"targets":[]}', "reward": 0.8},
        {"role": "coordinator", "prompt": "test", "completion": '{"assignments":[]}', "reward": 0.8},
    ]

    # Feed 3 novel episodes — should trigger dry-run training
    for i in range(3):
        learner.observe(fake_episode, novelty_score=0.9)

    status = learner.status()
    print(f"         status: {status}")
    assert status["adapters_trained"] == 1, \
        f"Expected 1 adapter trained after 3 novel episodes, got {status['adapters_trained']}"

check("ContinualLearner dry-run triggers after 3 novel episodes", _continual_dry_run)

# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
passed = sum(1 for _, ok in results if ok)
total  = len(results)
print(f"  Results: {passed}/{total} passed")

if passed == total:
    print("  All checks passed. You're ready to train.\n")
    sys.exit(0)
else:
    failed = [name for name, ok in results if not ok]
    print(f"  Failed: {failed}")
    print("  Fix these before starting the training run.\n")
    sys.exit(1)
