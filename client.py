"""
Client for the ASTROF multi-agent telescope scheduling environment.
"""
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from models import NetworkAction, NetworkObservation, NetworkState


class AstrofEnv(EnvClient[NetworkAction, NetworkObservation, NetworkState]):

    def _step_payload(self, action: NetworkAction) -> dict:
        return action.model_dump()

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", payload)
        return StepResult(
            observation=NetworkObservation(**obs_data),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> NetworkState:
        return NetworkState(**payload)
