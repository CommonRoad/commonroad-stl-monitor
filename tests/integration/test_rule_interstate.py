import numpy as np
import pytest
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.obstacle import ObstacleType
from commonroad.scenario.state import CustomState
from crmonitor.common import (
    RoadNetwork,
    Vehicle,
    VehicleParameters,
    World,
)
from crmonitor.evaluation.evaluation import OfflineRuleEvaluator
from tests.resources import InterstateScenarios
from tests.util import parallel_lanes


class TestRuleInterstate:
    def _base_test_rule(
        self, rule_name: str, world: World, ego_id: int, exp_violation: bool
    ) -> None:
        rule_eval = OfflineRuleEvaluator.create_for_rule(rule_name, world.dt)
        rule_robustness = np.array(rule_eval.evaluate(world, ego_id))
        bool_value = rule_robustness >= 0.0

        assert exp_violation == np.all(bool_value), (
            f"expected violation {exp_violation} but got violation {np.all(bool_value)} for robustness trace {rule_robustness}."
        )

    @pytest.mark.parametrize(
        "ego_id, exp_violation",
        [
            (1000, False),
            (1001, True),
            (1002, False),
            (1003, True),
            (1004, True),
            (1005, True),
            (1006, True),
            (1007, True),
            (1008, True),
            (1009, True),
            (1010, True),
        ],
    )
    def test_safe_distance(self, ego_id: int, exp_violation: bool) -> None:
        world = InterstateScenarios.SAFE_DISTANCE.get_world()
        self._base_test_rule("R_G1", world, ego_id, exp_violation)

    @pytest.mark.parametrize(
        "ego_id, exp_violation",
        [
            (1000, True),
            (1001, True),
            (1002, False),
            (1005, True),
            (1006, True),
            (1007, True),
        ],
    )
    def test_unnecessary_braking(self, ego_id: int, exp_violation: bool) -> None:
        # one vehicle accelerates (1000)
        # one vehicle drives with constant velocity (1001)
        # one vehicle which has no leading vehicle violates acceleration constraint (1002)
        # two leading vehicle which brake only minimal (1005, 1007)
        # one vehicle following another vehicle which brakes normal (1006)
        world = InterstateScenarios.UNNECESSARY_BRAKING.get_world()
        self._base_test_rule("R_G2", world, ego_id, exp_violation)

    @pytest.mark.parametrize(
        "ego_id, exp_violation",
        [
            (1000, False),
            (1001, True),
            (1002, False),
            (1003, True),
        ],
    )
    def test_speed_limit(self, ego_id: int, exp_violation: bool) -> None:
        world = InterstateScenarios.MAX_SPEED_LIMIT.get_world()
        self._base_test_rule("R_G3", world, ego_id, exp_violation)

    @pytest.mark.parametrize(
        "ego_id, exp_violation",
        [
            (1000, True),
            (1001, True),
            (1002, True),
            (1003, False),
            (1004, True),
            (1005, False),
        ],
    )
    def test_preserve_traffic_flow(self, ego_id: int, exp_violation: bool) -> None:
        # two vehicles which preserves traffic flow (1001 ,1004)
        # two vehicles without following vehicle (1000, 1002)
        # one vehicle which does not preserve traffic flow with leading and following vehicle (1003)
        # one vehicle which drives to alone and slow on single lane -> according rule false (1005)
        world = InterstateScenarios.PRESERVE_TRAFFIC_FLOW.get_world()
        self._base_test_rule("R_G4", world, ego_id, exp_violation)

    @pytest.mark.parametrize(
        "ego_id,exp_violation",
        [
            (1000, True),
            (1001, False),
            (1002, True),
            (1003, True),
            (1004, True),
            (1005, True),
            (1006, True),
            (1007, True),
            (1008, False),
            (1009, True),
            (1010, True),
        ],
    )
    def test_standstill(self, ego_id: int, exp_violation: bool) -> None:
        # one vehicle which is in standstill with a leading vehicle in standstill(1000)
        # one vehicle which is in standstill without a leading vehicle in standstill and which is not
        # part of a congestion a leading vehicle in standstill (1001)
        # one vehicle which drives with higher velocity (1002)
        # 1008 drives slow, but neither is a congestions nor is the leading vehicle slow enough
        # six vehicles which are in a congestion and drive with slow velocity (1003, 1004, 1006, 1007, 1009,
        # 1010)
        # one vehicle which is in standstill and part of a congestion (1005)
        world = InterstateScenarios.STANDSTILL.get_world()
        self._base_test_rule("R_I1", world, ego_id, exp_violation)

    @pytest.mark.parametrize(
        "ego_id, exp_violation",
        [
            (1000, True),
            (1001, False),
            (1002, True),
            (1003, True),
            (1004, True),
            (1005, True),
            (1006, True),
            (1007, True),
            (1008, True),
        ],
    )
    def test_overtaking_right_congestion(self, ego_id: int, exp_violation: bool) -> None:
        # one vehicle which overtakes a congestion slightly faster (1000)
        # one vehicle which overtakes a congestion too fast (1001)
        # all other vehicles a part of a congestion
        world = InterstateScenarios.OVERTAKING_RIGHT_CONGESTION.get_world()
        self._base_test_rule("R_I2", world, ego_id, exp_violation)

    @pytest.mark.parametrize(
        "ego_id, exp_violation",
        [
            (1000, False),
            (1001, False),
            (1002, False),
            (1003, True),
        ],
    )
    def test_reversing_and_u_turn(self, ego_id: int, exp_violation: bool) -> None:
        # one vehicle which drives first in correct direction and then reversely (1000)
        # one vehicle which drives always reversely (1001)
        # one vehicle which drives always in correct direction (1002)
        # one vehicle which makes a u-turn (1003)
        world = InterstateScenarios.REVERSING_AND_U_TURN.get_world()
        self._base_test_rule("R_I3", world, ego_id, exp_violation)

    @pytest.mark.parametrize(
        "ego_id, exp_violation",
        [
            (1000, False),
            (1001, True),
            (1002, True),
            (1003, True),
            (1004, False),
            (1005, False),
            (1006, True),
            (1007, True),
            (1008, False),
            (1009, True),
            (1010, False),
            (1011, True),
            (1012, True),
            (1013, True),
            (1014, True),
            (1015, True),
            (1016, False),
            (1017, True),
            (1018, True),
            (1019, True),
            (1020, True),
            (1021, False),
            (1022, False),
            (1023, True),
            (1024, False),
            (1025, True),
            (1026, True),
        ],
    )
    def test_emergency_lane_broad_enough_with_shoulder(
        self, ego_id: int, exp_violation: bool
    ) -> None:
        # several vehicles which drive not leftmost (e.g., 1024, 1016)
        # several vehicles which drive not rightmost (e.g., 1008, 1004)
        # several vehicles which drive leftmost (e.g., 1023, 1006)
        # several vehicles which drive rightmost(e.g., 1002, 1018)
        # one vehicle which drives on shoulder (1021)
        world = InterstateScenarios.EMERGENCY_LANE_THREE_LANES_WITH_SHOULDER.get_world()
        self._base_test_rule("R_I4", world, ego_id, exp_violation)

    @pytest.mark.parametrize(
        "ego_id, exp_violation",
        [
            (1000, False),
            (1001, True),
            (1002, True),
        ],
    )
    def test_consider_entering_vehicles(self, ego_id: int, exp_violation: bool) -> None:
        # one vehicle driving always in the left most lane (1001)
        # one vehicle changing to rightmost main carriage way lane (1000)
        # one vehicle entering main carriage way (1002)
        world = InterstateScenarios.CONSIDER_ENTERING_VEHICLES.get_world()
        self._base_test_rule("R_I5", world, ego_id, exp_violation)

    def test_single_vehicle(self):
        dt = 0.1
        lanelet_network = LaneletNetwork()
        lanelets = parallel_lanes(1)
        lanelet_network.add_lanelet(lanelets[0])
        road_network = RoadNetwork(lanelet_network)

        ego_vehicle_param = VehicleParameters.create_for_ego_vehicle(dt=0.1)

        # ego vehicle
        cr_state_list_ego = {
            0: CustomState(position=(0, 0), orientation=0, velocity=10, time_step=0),
            1: CustomState(position=(10, 0), orientation=0, velocity=4, time_step=1),
            2: CustomState(position=(14, 0), orientation=0, velocity=10, time_step=2),
            3: CustomState(position=(24, 0), orientation=0, velocity=5, time_step=3),
            4: CustomState(position=(29, 0), orientation=0, velocity=5, time_step=4),
        }
        lanelet_assignments_ego = {0: {1}, 1: {1}, 2: {1}, 3: {1}, 4: {1}}
        ego_vehicle = Vehicle(
            id=0,
            obstacle_type=ObstacleType.CAR,
            dt=dt,
            vehicle_param=ego_vehicle_param,
            shape=Rectangle(5, 2),
            states_cr=cr_state_list_ego,
            road_network=road_network,
            lanelet_assignment=lanelet_assignments_ego,
        )

        cr_state_list_other_1 = {
            0: CustomState(position=(10, 0), orientation=0, velocity=2, time_step=0),
            1: CustomState(position=(10, 0), orientation=0, velocity=2, time_step=1),
            2: CustomState(position=(20, 0), orientation=0, velocity=2, time_step=2),
            3: CustomState(position=(30, 0), orientation=0, velocity=2, time_step=3),
        }
        lanelet_assignments_other_1 = {0: {1}, 1: {1}, 2: {1}, 3: {1}}
        other_vehicle_1 = Vehicle(
            id=1,
            obstacle_type=ObstacleType.CAR,
            vehicle_param=ego_vehicle_param,
            dt=dt,
            shape=Rectangle(5, 2),
            states_cr=cr_state_list_other_1,
            road_network=road_network,
            lanelet_assignment=lanelet_assignments_other_1,
        )

        world = World({ego_vehicle, other_vehicle_1}, road_network)

        rule_str = "A a1: (in_front_of(a0, a1))"
        rule_eval = OfflineRuleEvaluator.create_for_rule_str(rule_str, dt)
        rule_robustness = rule_eval.evaluate(world, ego_vehicle.id)
        preds = rule_eval.get_predicate_values()
        assert rule_robustness[4] == 1.0
        np.testing.assert_allclose(np.array(list(preds.values())), 1.0)

        rule_str = "E a1: (in_front_of(a0, a1))"
        rule_eval = OfflineRuleEvaluator.create_for_rule_str(rule_str, dt)
        rule_robustness = rule_eval.evaluate(world, ego_vehicle.id)
        assert rule_robustness[4] == -1.0
        preds = rule_eval.get_predicate_values()
        np.testing.assert_allclose(np.array(list(preds.values())), -1.0)
