"""
Pydantic models for the ASTROF multi-agent telescope scheduling environment.
All 5 agent roles are bundled into a single NetworkAction / NetworkObservation
so the env remains OpenEnv-compliant (one step() call per timestep).
"""
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from openenv.core.env_server import Action, Observation, State


# ---------------------------------------------------------------------------
# Per-role action types
# ---------------------------------------------------------------------------

class TargetScore(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_id: str
    score: float = Field(ge=0.0, le=1.0)


class PlannerAction(BaseModel):
    """Science Planner: scored priority list + ToO classification."""
    model_config = ConfigDict(extra="forbid")
    targets: List[TargetScore] = Field(default_factory=list)
    too_flag: Literal["dismiss", "queue", "interrupt"] = "dismiss"


class AssignmentItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    telescope_id: str   # "mauna_kea" | "la_palma" | "siding_spring"
    target_id: str      # planet name string, or "wait"


class CoordinatorAction(BaseModel):
    """Network Coordinator: telescope → target assignments."""
    model_config = ConfigDict(extra="forbid")
    assignments: List[AssignmentItem] = Field(default_factory=list)


class ExecutorAction(BaseModel):
    """Telescope Executor: observe / wait / request_reassign / abort."""
    model_config = ConfigDict(extra="forbid")
    action: Literal["observe", "wait", "request_reassign", "abort"] = "wait"
    target_id: Optional[str] = None
    minutes: Optional[int] = None
    reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Bundled action (one per step() call)
# ---------------------------------------------------------------------------

class NetworkAction(Action):
    """All 5 agent actions bundled for a single step() call."""
    planner_action: PlannerAction = Field(default_factory=PlannerAction)
    coordinator_action: CoordinatorAction = Field(default_factory=CoordinatorAction)
    executor_actions: List[ExecutorAction] = Field(
        default_factory=lambda: [ExecutorAction(), ExecutorAction(), ExecutorAction()]
    )
    parse_rate: float = Field(default=1.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Per-role observation types (slices of the full NetworkObservation)
# ---------------------------------------------------------------------------

class PlanetSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    priority_score: int
    altitude_deg: float
    airmass: float
    visible: bool
    observed: bool
    has_deadline: bool
    deadline_status: str   # "no_deadline" | "before_deadline" | "past_deadline"
    category: Optional[str] = None  # injected at step 9 in expert task


class TelescopeStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")
    telescope_id: str
    site_name: str
    status: Literal["idle", "observing", "clouded", "failed"] = "idle"
    current_target: Optional[str] = None
    weather: str = "clear"
    airmass: float = 1.0


class PlannerObs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    narrative: str
    step: int
    max_steps: int
    planets: List[PlanetSnapshot]
    too_alert: Optional[str] = None      # human-readable alert text
    failure_flag: Optional[str] = None   # which telescope failed (if any)
    telescope_count: int = 3


class CoordinatorObs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    narrative: str
    step: int
    max_steps: int
    priority_list: List[TargetScore]     # from PREVIOUS step's Planner (one-step lag)
    too_flag: Literal["dismiss", "queue", "interrupt"] = "dismiss"
    telescope_statuses: List[TelescopeStatus]
    reassign_queue: List[str] = Field(default_factory=list)  # telescope_ids requesting reassign


class ExecutorObs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    narrative: str
    step: int
    max_steps: int
    site_name: str
    telescope_id: str
    assignment: Optional[str] = None     # target name assigned by Coordinator
    weather: str = "clear"
    airmass: float = 1.0
    snr_expected: float = 1.0            # fraction of theoretical max SNR


# ---------------------------------------------------------------------------
# Bundled observation
# ---------------------------------------------------------------------------

class NetworkObservation(Observation):
    """Full observation returned after every reset() and step()."""
    narrative: str                        # high-level summary for logging
    step: int
    max_steps: int
    task_id: str = "easy"

    planner_obs: PlannerObs
    coordinator_obs: CoordinatorObs
    executor_obs: List[ExecutorObs]       # one per telescope (len=3 for medium/hard/expert, len=1 for easy)

    # Aggregated metrics (for graders and logging)
    n_telescopes: int = 1
    total_priority_observed: float = 0.0
    n_observed: int = 0
    deadlines_met: int = 0
    too_responses: int = 0
    new_category_handled: float = 0.0
    parse_rate: float = 1.0              # fraction of 5 LLM calls that parsed successfully


# ---------------------------------------------------------------------------
# State (lightweight, used by /grade endpoint)
# ---------------------------------------------------------------------------

class NetworkState(State):
    task_id: str = "easy"
    step: int = 0
    max_steps: int = 18
    n_telescopes: int = 1

    # Per-site observed tracking (serialisable)
    observed_targets: List[str] = Field(default_factory=list)
    total_priority_observed: float = 0.0
    n_observed: int = 0
    deadlines_met: int = 0
    too_responses: int = 0
    new_category_handled: float = 0.0
    duplicate_count: int = 0

    # Easy-task compatibility fields (mirrors Round 1 TelescopeState)
    n_observed_tonight: int = 0
    deadlines_met_before_cutoff: int = 0
