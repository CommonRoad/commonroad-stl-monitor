from pathlib import Path
from commonroad.common.file_reader import CommonRoadFileReader
import logging

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

relevant_scenarios = []

path_to_highD_scenarios = Path(__file__).parent.parent.parent.parent / "highD-scenarios"

for i, scenario_path in enumerate(list(path_to_highD_scenarios.glob("*.xml"))[:100]):
    if i % 10 == 0:
        _LOGGER.info(f"Processing scenario {i}")
    scenario, _ = CommonRoadFileReader(scenario_path).open()
    for vehicle in scenario.dynamic_obstacles:
        velocities = [state.velocity for state in vehicle.prediction.trajectory.state_list]
        if min(velocities) < 10:
            relevant_scenarios.append(scenario_path)
            break

print(relevant_scenarios)
