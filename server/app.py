import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

from openenv.core.env_server import create_fastapi_app
from environment import ObservatoryNetworkEnv
from models import NetworkAction, NetworkObservation, NetworkState

app = create_fastapi_app(ObservatoryNetworkEnv, NetworkAction, NetworkObservation)


@app.post("/grade")
def grade(state: NetworkState):
    env = ObservatoryNetworkEnv()
    env._state = state
    env._task_id = state.task_id
    env._task_config = __import__("environment", fromlist=["TASK_CONFIGS"]).TASK_CONFIGS.get(
        state.task_id, __import__("environment", fromlist=["TASK_CONFIGS"]).TASK_CONFIGS["easy"]
    )
    env._deadlines_met = state.deadlines_met
    env._too_responses = state.too_responses
    env._new_category_total = 1 if state.new_category_handled > 0 else 0
    env._new_category_observed = round(state.new_category_handled)
    score = env.compute_grade()
    return {"task_id": state.task_id, "score": score}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
