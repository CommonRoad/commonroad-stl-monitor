from enum import Enum
from pathlib import Path

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.scenario import Scenario

from crmonitor.common import World

_SCENARIOS_PATH = Path(__file__).parents[1] / "scenarios"


class _Scenarios(Enum):
    def get_path(self) -> Path: ...

    def get_commonroad_scenario(self) -> Scenario:
        scenario, _ = CommonRoadFileReader(self.get_path()).open(lanelet_assignment=True)
        return scenario

    def get_world(self) -> World:
        return World.create_from_scenario(self.get_commonroad_scenario())


class InterstateScenarios(_Scenarios):
    SAFE_DISTANCE_LANE_CHANGE = "DEU_test_safe_distance_lane_change.xml"

    def get_path(self) -> Path:
        return _SCENARIOS_PATH / "test_interstate" / self.value
