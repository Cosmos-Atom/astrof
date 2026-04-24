"""
ObservatoryNetworkEnv — ASTROF multi-agent telescope scheduling environment.

5 agents bundled into a single step() call:
  Agent 1: Science Planner  — scores planets + classifies ToO alerts
  Agent 2: Network Coordinator — assigns targets to telescopes
  Agents 3-5: Telescope Executors (Mauna Kea, La Palma, Siding Spring)

Task difficulty:
  easy   — 1 telescope, 20 planets, stochastic weather, 3 transit deadlines (R1 hard carried forward)
  medium — 3 telescopes, 20 planets, clear night, no ToOs, 44 steps
  hard   — 3 telescopes, 20 planets, stochastic weather, 2 transit deadlines, 32 steps
  expert — 3 telescopes, 20 planets, dynamic weather, 3 ToO interrupts, new category at step 9, 18 steps
"""
import os
import sys
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openenv.core.env_server import Environment

from models import (
    AssignmentItem,
    CoordinatorAction,
    CoordinatorObs,
    ExecutorAction,
    ExecutorObs,
    NetworkAction,
    NetworkObservation,
    NetworkState,
    PlannerAction,
    PlannerObs,
    PlanetSnapshot,
    TargetScore,
    TelescopeStatus,
)
from server.core import (
    _TelescopeCore,
    airmass_from_altitude,
    altitude_from_ra_dec,
    load_planet_dataframe,
)
from server.sites import SITE_CONFIGS, TELESCOPE_IDS

_EPS = 1e-4

# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

TASK_CONFIGS = {
    "easy": {
        "n_telescopes": 1,
        "max_steps": 18,
        "weather_locked": False,
        "start_offset_min": 30,
        "deadline_step_cutoff": 5,
        "n_too_events": 0,
        "inject_new_category": False,
        "default_seed": 13,
        "description": "1 telescope · 20 planets · stochastic weather · 3 transit deadlines by step 5",
    },
    "medium": {
        "n_telescopes": 3,
        "max_steps": 44,
        "weather_locked": True,
        "start_offset_min": 0,
        "deadline_step_cutoff": None,
        "n_too_events": 0,
        "inject_new_category": False,
        "default_seed": 42,
        "description": "3 telescopes · 20 planets · clear night · no ToOs · no deadlines",
    },
    "hard": {
        "n_telescopes": 3,
        "max_steps": 32,
        "weather_locked": False,
        "start_offset_min": 0,
        "deadline_step_cutoff": 8,
        "n_too_events": 0,
        "inject_new_category": False,
        "default_seed": 7,
        "description": "3 telescopes · 20 planets · stochastic weather · 2 transit deadlines",
    },
    "expert": {
        "n_telescopes": 3,
        "max_steps": 18,
        "weather_locked": False,
        "start_offset_min": 30,
        "deadline_step_cutoff": 5,
        "n_too_events": 3,
        "inject_new_category": True,
        "new_category_step": 9,
        "default_seed": 99,
        "description": "3 telescopes · 20 planets · dynamic weather · 3 ToO interrupts · new category at step 9",
    },
}

# Global singleton dataframe — loaded once
_DF = None


def _get_df() -> pd.DataFrame:
    global _DF
    if _DF is None:
        _DF = load_planet_dataframe()
    return _DF


# ---------------------------------------------------------------------------
# Helper: build per-site _TelescopeCore with correct EarthLocation
# ---------------------------------------------------------------------------

def _make_core(site_id: str, df: pd.DataFrame, task_config: dict, seed: int) -> _TelescopeCore:
    site = SITE_CONFIGS[site_id]
    core_cfg = {
        "max_steps": task_config["max_steps"],
        "weather_locked": task_config["weather_locked"],
        "start_offset_min": task_config["start_offset_min"],
        "deadline_step_cutoff": task_config.get("deadline_step_cutoff"),
    }
    core = _TelescopeCore(df, core_cfg)
    # Set site location before reset so altitude calculations use the right coordinates
    core.location = site["location"]
    core.reset(seed)
    # Patch time fields AFTER reset() since reset() overwrites them with module-level constants
    start_offset = timedelta(minutes=task_config["start_offset_min"])
    core.current_time = site["sunset"] + start_offset
    core.sunrise = site["sunrise"]
    return core


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------

class ObservatoryNetworkEnv(Environment):
    """
    ASTROF: Hierarchical multi-agent telescope scheduling environment.
    Implements the OpenEnv step()/reset()/state() API with a bundled NetworkAction.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._task_id: str = "easy"
        self._task_config: dict = TASK_CONFIGS["easy"]
        self._cores: Dict[str, _TelescopeCore] = {}
        self._state = NetworkState()

        # Episode-level tracking
        self._step: int = 0
        self._observed_set: set = set()          # planet names observed by ANY telescope
        self._deadlines_met: int = 0
        self._too_responses: int = 0
        self._too_schedule: List[int] = []       # steps at which ToO alerts fire
        self._active_too: Optional[str] = None
        self._too_flag_prev: str = "dismiss"     # Planner's flag from previous step
        self._priority_list_prev: List[TargetScore] = []
        self._reassign_queue: List[str] = []     # telescope_ids requesting reassign
        self._failed_telescopes: set = set()
        self._new_category_targets: List[str] = []
        self._new_category_observed: int = 0
        self._new_category_total: int = 0

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed=None,
        episode_id=None,
        task_id: str = "easy",
        **kwargs,
    ) -> NetworkObservation:
        cfg = TASK_CONFIGS.get(task_id, TASK_CONFIGS["easy"])
        if seed is None:
            seed = cfg["default_seed"]

        self._task_id = task_id
        self._task_config = cfg
        self._step = 0
        self._observed_set = set()
        self._deadlines_met = 0
        self._too_responses = 0
        self._too_flag_prev = "dismiss"
        self._priority_list_prev = []
        self._reassign_queue = []
        self._failed_telescopes = set()
        self._new_category_targets = []
        self._new_category_observed = 0
        self._new_category_total = 0

        df = _get_df()
        n_tel = cfg["n_telescopes"]
        active_sites = TELESCOPE_IDS[:n_tel]

        self._cores = {}
        for i, site_id in enumerate(active_sites):
            core = _make_core(site_id, df, cfg, seed + i)
            self._cores[site_id] = core

        # Seed initial priority list so Coordinator has something on step 1
        first_core = next(iter(self._cores.values()))
        planet_infos = first_core.get_planet_infos()
        visible = sorted(
            [p for p in planet_infos if p["visible"]],
            key=lambda x: (-int(x.get("has_deadline", False)), -x["priority_score"])
        )
        max_pri = max((p["priority_score"] for p in visible), default=1)
        self._priority_list_prev = [
            TargetScore(
                target_id=p["name"],
                score=round(min(1.0, p["priority_score"] / max(max_pri, 1)), 3),
            )
            for p in visible
        ]

        # Schedule ToO events at random steps
        import random
        rng = random.Random(seed + 100)
        n_too = cfg.get("n_too_events", 0)
        max_s = cfg["max_steps"]
        self._too_schedule = sorted(rng.sample(range(3, max_s - 2), n_too)) if n_too > 0 else []
        self._active_too = None

        self._state = NetworkState(
            episode_id=episode_id or str(uuid.uuid4()),
            step=0,
            max_steps=cfg["max_steps"],
            n_telescopes=n_tel,
            task_id=task_id,
        )

        return self._build_obs(reward=None, done=False)

    def step(
        self,
        action: NetworkAction,
        timeout_s=None,
        **kwargs,
    ) -> NetworkObservation:
        if not self._cores:
            raise RuntimeError("Call reset() before step().")

        cfg = self._task_config
        active_sites = list(self._cores.keys())
        n_tel = len(active_sites)

        # --- 1. Run Planner ---
        planner_action = action.planner_action
        # Store priority list for Coordinator to use NEXT step (one-step lag preserved)
        new_priority_list = planner_action.targets

        # Check if Planner classified ToO correctly when one was active
        if self._active_too and planner_action.too_flag == "interrupt":
            self._too_responses += 1

        # --- 2. Run Coordinator ---
        coord_action = action.coordinator_action
        assignments: Dict[str, str] = {}
        for item in coord_action.assignments:
            if item.telescope_id in active_sites and item.telescope_id not in self._failed_telescopes:
                assignments[item.telescope_id] = item.target_id

        # Clear reassign queue (executors from previous step filled it)
        self._reassign_queue = []

        # --- 3. Run Executors ---
        # All executors call core.step() — this keeps time advancement in one place.
        # abort / request_reassign pass the wait index so core still ticks forward.
        step_rewards: Dict[str, float] = {}

        for i, site_id in enumerate(active_sites):
            core = self._cores[site_id]
            ex_action = action.executor_actions[i] if i < len(action.executor_actions) else ExecutorAction()

            if site_id in self._failed_telescopes:
                step_rewards[site_id] = _EPS
                # Still tick time so all cores stay in sync
                core.current_time += timedelta(minutes=15)
                core.step_count += 1
                continue

            if ex_action.action == "abort":
                self._failed_telescopes.add(site_id)
                step_rewards[site_id] = _EPS
                core.current_time += timedelta(minutes=15)
                core.step_count += 1
                continue

            if ex_action.action == "request_reassign":
                self._reassign_queue.append(site_id)
                # Wait this step — pass wait index to keep time and weather in sync
                if not core.done:
                    reward, _, _ = core.step(core.num_planets)
                else:
                    core.current_time += timedelta(minutes=15)
                    core.step_count += 1
                step_rewards[site_id] = _EPS
                continue

            # observe or wait — use coordinator assignment as authoritative target
            target_name = assignments.get(site_id)
            if ex_action.action == "observe" and not target_name:
                target_name = ex_action.target_id
            planet_idx = self._resolve_planet_idx(core, target_name)

            if core.done:
                core.current_time += timedelta(minutes=15)
                core.step_count += 1
                step_rewards[site_id] = _EPS
                continue

            reward, _, info = core.step(planet_idx)

            # Cross-telescope deduplication
            planet_name = info.get("planet_name")
            if planet_name and info.get("action_type") == "observe":
                if planet_name in self._observed_set:
                    reward = max(_EPS, reward * 0.1)  # duplicate penalty
                else:
                    self._observed_set.add(planet_name)
                    if planet_name in self._new_category_targets:
                        self._new_category_observed += 1

            step_rewards[site_id] = reward

        self._step += 1

        # --- 4. Inject new category for expert task ---
        if (
            cfg.get("inject_new_category")
            and self._step == cfg.get("new_category_step", 9)
            and not self._new_category_targets
        ):
            self._inject_new_category()

        # --- 5. Advance ToO schedule ---
        self._active_too = None
        if self._too_schedule and self._step >= self._too_schedule[0]:
            self._too_schedule.pop(0)
            self._active_too = "GRB_ALERT"

        # --- 6. Compute team reward ---
        total_reward = self._compute_team_reward(step_rewards, action.parse_rate)

        # --- 7. Termination ---
        done = self._step >= cfg["max_steps"] or self._all_targets_observed()

        # Update one-step-lag priority list for next Coordinator obs
        self._priority_list_prev = new_priority_list
        self._too_flag_prev = planner_action.too_flag

        # Update state
        self._update_state()

        return self._build_obs(reward=round(total_reward, 4), done=done)

    @property
    def state(self) -> NetworkState:
        return self._state

    # ------------------------------------------------------------------
    # Grader
    # ------------------------------------------------------------------

    def compute_grade(self) -> float:
        s = self._state
        if s.task_id == "easy":
            # Mirror Round 1 hard grader exactly
            deadline_score = min(s.deadlines_met / 3.0, 1.0)
            priority_score = min(s.total_priority_observed / 133.0, 1.0)
            raw = 0.6 * deadline_score + 0.4 * priority_score

        elif s.task_id == "medium":
            # Oracle: 202 = max priority sum achievable with 3-telescope greedy, clear night, seed=42
            dup_penalty = min(s.duplicate_count * 0.05, 0.3)
            raw = min(s.total_priority_observed / 202.0, 1.0) - dup_penalty

        elif s.task_id == "hard":
            priority_yield = min(s.total_priority_observed / 202.0, 1.0)
            deadline_frac = min(s.deadlines_met / 2.0, 1.0)
            dup_penalty = min(s.duplicate_count * 0.05, 0.2)
            raw = 0.5 * priority_yield + 0.5 * deadline_frac - dup_penalty

        else:  # expert
            priority_yield = min(s.total_priority_observed / 133.0, 1.0)
            too_score = min(s.too_responses / max(self._task_config.get("n_too_events", 1), 1), 1.0)
            new_cat = s.new_category_handled
            dup_penalty = min(s.duplicate_count * 0.05, 0.2)
            raw = 0.4 * priority_yield + 0.3 * too_score + 0.3 * new_cat - dup_penalty

        return round(max(_EPS, min(raw, 1.0 - _EPS)), 4)

    def get_tasks(self) -> list:
        return [
            {
                "task_id": tid,
                "description": cfg["description"],
                "max_steps": cfg["max_steps"],
            }
            for tid, cfg in TASK_CONFIGS.items()
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_planet_idx(self, core: _TelescopeCore, target_name: Optional[str]) -> int:
        """Map a planet name to its DataFrame index. Returns wait index on miss."""
        if not target_name or target_name == "wait":
            return core.num_planets  # wait action
        matches = core.planets[core.planets["pl_name"] == target_name]
        if not matches.empty:
            return int(matches.index[0])
        # Fuzzy: try case-insensitive partial match
        lower = target_name.lower()
        for idx, row in core.planets.iterrows():
            if lower in str(row["pl_name"]).lower():
                return int(idx)
        return core.num_planets  # fallback: wait

    def _compute_team_reward(self, step_rewards: Dict[str, float], parse_rate: float = 1.0) -> float:
        if not step_rewards:
            return _EPS
        mean_r = sum(step_rewards.values()) / max(len(step_rewards), 1)
        blended = 0.9 * mean_r + 0.1 * parse_rate
        return max(_EPS, min(blended, 1.0 - _EPS))

    def _all_targets_observed(self) -> bool:
        if not self._cores:
            return False
        first_core = next(iter(self._cores.values()))
        return len(self._observed_set) >= first_core.num_planets

    def _inject_new_category(self) -> None:
        """Mark 2–3 random unobserved planets as gravitational_wave_host."""
        import random
        rng = random.Random(self._task_config["default_seed"] + 999)
        first_core = next(iter(self._cores.values()))
        candidates = [
            str(row["pl_name"])
            for _, row in first_core.planets.iterrows()
            if str(row["pl_name"]) not in self._observed_set
        ]
        n = min(3, len(candidates))
        self._new_category_targets = rng.sample(candidates, n)
        self._new_category_total = n

    def _update_state(self) -> None:
        s = self._state
        s.step = self._step
        s.observed_targets = list(self._observed_set)
        s.n_observed = len(self._observed_set)
        s.n_observed_tonight = s.n_observed

        # Compute total priority from the first core's data
        if self._cores:
            first_core = next(iter(self._cores.values()))
            total_pri = sum(
                float(row["priority_score"])
                for _, row in first_core.planets.iterrows()
                if str(row["pl_name"]) in self._observed_set
            )
            s.total_priority_observed = round(total_pri, 2)

        # Deadlines met: sum across all cores
        s.deadlines_met = sum(
            getattr(c, "deadlines_met_before_cutoff", 0)
            for c in self._cores.values()
        )
        s.deadlines_met_before_cutoff = s.deadlines_met
        self._deadlines_met = s.deadlines_met

        s.too_responses = self._too_responses
        s.new_category_handled = (
            round(self._new_category_observed / self._new_category_total, 4)
            if self._new_category_total > 0
            else 0.0
        )

        # Count duplicates: any planet observed by >1 telescope in the same step
        dup_count = 0
        seen = set()
        for site_id, core in self._cores.items():
            for _, row in core.planets.iterrows():
                name = str(row["pl_name"])
                if row.get("times_observed", 0) > 1 or (name in seen and row.get("observed_tonight", False)):
                    dup_count += 1
                if row.get("observed_tonight", False):
                    seen.add(name)
        s.duplicate_count = dup_count

    def _build_obs(self, reward, done: bool) -> NetworkObservation:
        cfg = self._task_config
        active_sites = list(self._cores.keys())
        first_core = self._cores[active_sites[0]] if self._cores else None

        # --- Planner observation ---
        planets_snap: List[PlanetSnapshot] = []
        if first_core:
            planet_infos = first_core.get_planet_infos()
            for i, info in enumerate(planet_infos):
                cat = None
                if info["name"] in self._new_category_targets:
                    cat = "gravitational_wave_host"
                planets_snap.append(PlanetSnapshot(
                    name=info["name"],
                    priority_score=info["priority_score"],
                    altitude_deg=info["altitude_deg"],
                    airmass=info["airmass"],
                    visible=info["visible"],
                    observed=info["observed_tonight"],
                    has_deadline=info["has_deadline"],
                    deadline_status=info["deadline_status"],
                    category=cat,
                ))

        too_text = None
        if self._active_too:
            too_text = (
                "GRB ALERT: gamma-ray burst detected — optical afterglow window ~60 min. "
                "Classify: dismiss / queue / interrupt-now."
            )
        if self._new_category_targets and self._step >= cfg.get("new_category_step", 9):
            nc_names = ", ".join(self._new_category_targets)
            cat_note = f"NEW CATEGORY DETECTED: targets [{nc_names}] labelled 'gravitational_wave_host'. Prioritise within 3 steps."
            too_text = (too_text + " | " + cat_note) if too_text else cat_note

        failure_flag = ", ".join(self._failed_telescopes) if self._failed_telescopes else None

        planner_narrative = _build_planner_narrative(
            step=self._step,
            max_steps=cfg["max_steps"],
            planets=planets_snap,
            too_text=too_text,
            failure_flag=failure_flag,
            telescope_count=len(active_sites) - len(self._failed_telescopes),
        )

        planner_obs = PlannerObs(
            narrative=planner_narrative,
            step=self._step,
            max_steps=cfg["max_steps"],
            planets=planets_snap,
            too_alert=too_text,
            failure_flag=failure_flag,
            telescope_count=len(active_sites),
        )

        # --- Coordinator observation (uses PREVIOUS step's priority list) ---
        tel_statuses: List[TelescopeStatus] = []
        for site_id in active_sites:
            core = self._cores[site_id]
            if site_id in self._failed_telescopes:
                status = "failed"
            elif site_id in self._reassign_queue:
                status = "clouded"
            else:
                status = "idle"
            tel_statuses.append(TelescopeStatus(
                telescope_id=site_id,
                site_name=SITE_CONFIGS[site_id]["display_name"],
                status=status,
                weather=_weather_name(core.weather_state),
                airmass=round(
                    airmass_from_altitude(
                        altitude_from_ra_dec(
                            float(core.planets.iloc[0].ra),
                            float(core.planets.iloc[0].dec),
                            core.current_time,
                            core.location,
                        )
                    ), 2
                ) if not core.planets.empty else 1.0,
            ))

        coord_narrative = _build_coordinator_narrative(
            step=self._step,
            max_steps=cfg["max_steps"],
            priority_list=self._priority_list_prev,
            too_flag=self._too_flag_prev,
            tel_statuses=tel_statuses,
            reassign_queue=self._reassign_queue,
        )

        coord_obs = CoordinatorObs(
            narrative=coord_narrative,
            step=self._step,
            max_steps=cfg["max_steps"],
            priority_list=self._priority_list_prev,
            too_flag=self._too_flag_prev,
            telescope_statuses=tel_statuses,
            reassign_queue=self._reassign_queue,
        )

        # --- Executor observations ---
        executor_obs_list: List[ExecutorObs] = []
        for site_id in active_sites:
            core = self._cores[site_id]
            wx = _weather_name(core.weather_state)
            # Best visible target's airmass as proxy for site conditions
            am = 1.5
            if not core.planets.empty:
                for _, row in core.planets.iterrows():
                    alt = altitude_from_ra_dec(
                        float(row.ra), float(row.dec), core.current_time, core.location
                    )
                    if alt > 30:
                        am = round(airmass_from_altitude(alt), 2)
                        break

            snr = max(0.1, min(1.0, 1.0 / am))
            if core.weather_state == 1:
                snr *= 0.7
            elif core.weather_state == 2:
                snr *= 0.3

            ex_narrative = _build_executor_narrative(
                step=self._step,
                max_steps=cfg["max_steps"],
                site_name=SITE_CONFIGS[site_id]["display_name"],
                telescope_id=site_id,
                weather=wx,
                airmass=am,
                snr=round(snr, 2),
                failed=site_id in self._failed_telescopes,
            )

            executor_obs_list.append(ExecutorObs(
                narrative=ex_narrative,
                step=self._step,
                max_steps=cfg["max_steps"],
                site_name=SITE_CONFIGS[site_id]["display_name"],
                telescope_id=site_id,
                weather=wx,
                airmass=am,
                snr_expected=round(snr, 2),
            ))

        top_narrative = (
            f"ASTROF | Task: {self._task_id} | Step {self._step}/{cfg['max_steps']} | "
            f"Observed: {len(self._observed_set)} | "
            f"Telescopes active: {len(active_sites) - len(self._failed_telescopes)}/{len(active_sites)}"
        )

        return NetworkObservation(
            done=done,
            reward=reward,
            narrative=top_narrative,
            step=self._step,
            max_steps=cfg["max_steps"],
            task_id=self._task_id,
            planner_obs=planner_obs,
            coordinator_obs=coord_obs,
            executor_obs=executor_obs_list,
            n_telescopes=len(active_sites),
            total_priority_observed=self._state.total_priority_observed,
            n_observed=len(self._observed_set),
            deadlines_met=self._deadlines_met,
            too_responses=self._too_responses,
            new_category_handled=self._state.new_category_handled,
        )


# ---------------------------------------------------------------------------
# Narrative builders
# ---------------------------------------------------------------------------

def _weather_name(state: int) -> str:
    return {0: "clear", 1: "partial", 2: "bad"}.get(state, "clear")


def _build_planner_narrative(
    step: int,
    max_steps: int,
    planets: List[PlanetSnapshot],
    too_text: Optional[str],
    failure_flag: Optional[str],
    telescope_count: int,
) -> str:
    lines = [
        f"SCIENCE PLANNER | Step {step}/{max_steps} | Telescopes available: {telescope_count}",
        "",
    ]
    if too_text:
        lines += [f"ALERT: {too_text}", ""]
    if failure_flag:
        lines += [f"FAILURE: telescope(s) offline — {failure_flag}. Re-score with reduced capacity.", ""]

    visible = [p for p in planets if p.visible and not p.observed]
    if visible:
        lines.append("TARGETS VISIBLE AND UNOBSERVED (score each 0.0–1.0):")
        for p in sorted(visible, key=lambda x: -x.priority_score)[:10]:
            cat_str = f"  [category: {p.category}]" if p.category else ""
            dl_str = f"  DEADLINE: {p.deadline_status}" if p.has_deadline else ""
            lines.append(
                f"  {p.name:<22} pri={p.priority_score:>2}  "
                f"alt={p.altitude_deg:>5.1f}°  airmass={p.airmass:.2f}"
                f"{dl_str}{cat_str}"
            )
    else:
        lines.append("No unobserved targets currently visible.")

    lines += [
        "",
        'OUTPUT JSON: {"targets": [{"target_id": "<name>", "score": 0.0-1.0}, ...], '
        '"too_flag": "dismiss|queue|interrupt"}',
    ]
    return "\n".join(lines)


def _build_coordinator_narrative(
    step: int,
    max_steps: int,
    priority_list: List[TargetScore],
    too_flag: str,
    tel_statuses: List[TelescopeStatus],
    reassign_queue: List[str],
) -> str:
    lines = [
        f"NETWORK COORDINATOR | Step {step}/{max_steps} | too_flag={too_flag}",
        "",
        "PRIORITY LIST (from previous step's Planner):",
    ]
    for ts in sorted(priority_list, key=lambda x: -x.score)[:8]:
        lines.append(f"  {ts.target_id:<22} score={ts.score:.3f}")

    lines += ["", "TELESCOPE STATUS:"]
    for t in tel_statuses:
        reassign_note = "  [REASSIGN REQUESTED]" if t.telescope_id in reassign_queue else ""
        lines.append(
            f"  {t.site_name:<16} {t.status.upper():<10} "
            f"weather={t.weather:<8}{reassign_note}"
        )

    if too_flag == "interrupt":
        lines += ["", "ACTION REQUIRED: redirect fastest idle telescope to ToO target NOW."]

    lines += [
        "",
        'OUTPUT JSON: {"assignments": [{"telescope_id": "<id>", "target_id": "<name>"}, ...]}',
        "telescope_ids: mauna_kea | la_palma | siding_spring",
    ]
    return "\n".join(lines)


def _build_executor_narrative(
    step: int,
    max_steps: int,
    site_name: str,
    telescope_id: str,
    weather: str,
    airmass: float,
    snr: float,
    failed: bool,
) -> str:
    if failed:
        return (
            f"TELESCOPE EXECUTOR — {site_name} | Step {step}/{max_steps}\n"
            "STATUS: FAILED — hardware fault. Output abort()."
        )
    lines = [
        f"TELESCOPE EXECUTOR — {site_name} | Step {step}/{max_steps}",
        f"Weather: {weather}  |  Best airmass: {airmass:.2f}  |  Expected SNR: {snr:.0%}",
        "",
    ]
    if weather == "bad":
        lines.append("Conditions poor — consider request_reassign or wait.")
    elif weather == "partial":
        lines.append("Partial cloud — SNR reduced. Observe or request_reassign.")
    else:
        lines.append("Conditions good — observe assigned target.")

    lines += [
        "",
        'OUTPUT JSON: {"action": "observe|wait|request_reassign|abort", '
        '"target_id": "<name>|null", "minutes": <int|null>, "reason": "<str|null>"}',
    ]
    return "\n".join(lines)
