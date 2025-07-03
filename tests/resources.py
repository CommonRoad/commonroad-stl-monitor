import warnings
from enum import Enum
from pathlib import Path

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.scenario import Scenario
from crmonitor.common import World

_SCENARIOS_PATH = Path(__file__).parents[1] / "scenarios"

_WORLD_CACHE = {}


class _Scenarios(Enum):
    def get_path(self) -> Path: ...

    def get_commonroad_scenario(self) -> Scenario:
        with warnings.catch_warnings():
            # Ignore warnings about invalid scenario IDs for all of our test scenarios.
            warnings.simplefilter("ignore")
            scenario, _ = CommonRoadFileReader(self.get_path()).open(lanelet_assignment=True)
        return scenario

    def get_world(self) -> World:
        if self.value in _WORLD_CACHE:
            return _WORLD_CACHE[self.value]

        world = World.create_from_scenario(self.get_commonroad_scenario())
        _WORLD_CACHE[self.value] = world
        return world


class InterstateScenarios(_Scenarios):
    SAFE_DISTANCE_LANE_CHANGE = "DEU_test_safe_distance_lane_change.xml"
    UNNECESSARY_BRAKING = "DEU_test_unnecessary_braking.xml"
    STANDSTILL = "DEU_test_standstill.xml"
    MAX_SPEED_LIMIT = "DEU_test_max_speed_limit.xml"
    SAFE_DISTANCE = "DEU_test_safe_distance.xml"
    PRESERVE_TRAFFIC_FLOW = "DEU_test_preserve_traffic_flow.xml"
    OVERTAKING_RIGHT_CONGESTION = "DEU_test_overtaking_right_congestion.xml"
    REVERSING_AND_U_TURN = "DEU_test_reversing_and_u_turn.xml"
    EMERGENCY_LANE_THREE_LANES_WITH_SHOULDER = "DEU_test_emergency_three_lanes_with_shoulder.xml"
    CONSIDER_ENTERING_VEHICLES = "DEU_test_consider_entering_vehicles_for_lane_change.xml"

    def get_path(self) -> Path:
        return _SCENARIOS_PATH / "test_interstate" / self.value
