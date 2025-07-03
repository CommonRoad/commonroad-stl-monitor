import unittest
from copy import deepcopy

import numpy as np
import pytest
from commonroad.common.solution import (
    CostFunction,
    PlanningProblemSolution,
    VehicleModel,
    VehicleType,
)
from commonroad.common.util import Interval
from commonroad.geometry.shape import Rectangle
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import (
    CustomState,
    InitialState,
    InputState,
    KSState,
    PMInputState,
    PMState,
)
from commonroad.scenario.trajectory import Trajectory
from crmonitor.common.world import World
from crmonitor.evaluation.evaluation import OfflineRuleEvaluator

from tests.resources import InterstateScenarios
from tests.util import parallel_lanes


class TestOfflineEvaluator:
    @classmethod
    def setup_class(cls) -> None:
        cls._world = InterstateScenarios.SAFE_DISTANCE_LANE_CHANGE.get_world()

    @pytest.mark.parametrize(
        "rule_str",
        [
            "A a1: (in_front_of(a0, a1) and cut_in(a0, a1))",
            "A a1: (in_front_of(a0, a1)) and single_lane(a0)",
            "E a1: (in_front_of(a0, a1) and cut_in(a0, a1))",
            "E a1: (in_front_of(a0, a1)) and single_lane(a0)",
            "single_lane(a0)",
            "single_lane(a0) and single_lane(a0)",
            "A a1: (in_front_of(a0, a1))",
        ],
    )
    def test_smoke(self, rule_str: str) -> None:
        evaluator = OfflineRuleEvaluator.create_for_rule_str(rule_str, dt=self._world.scenario.dt)

        robustness = evaluator.evaluate(self._world, ego_id=1001)
        assert isinstance(robustness, list)

    def test_reevaluation_same_states_same_robustness(self) -> None:
        ego_id = 1001

        evaluator = OfflineRuleEvaluator.create_for_rule("R_G3", self._world.scenario.dt)

        robustness0 = evaluator.evaluate(self._world, ego_id=ego_id)

        evaluator.reset()

        robustness1 = evaluator.evaluate(self._world, ego_id)
        assert np.all(np.isclose(robustness0, robustness1))

    def test_reevaluation_different_states_different_robustness(self) -> None:
        ego_id = 1001

        evaluator = OfflineRuleEvaluator.create_for_rule("R_G3", self._world.scenario.dt)

        orig_robustness = evaluator.evaluate(self._world, ego_id=ego_id)

        evaluator.reset()

        world_copy = deepcopy(self._world)
        ego_vehicle = world_copy.vehicle_by_id(ego_id)
        assert ego_vehicle is not None

        for time_step in range(ego_vehicle.start_time, ego_vehicle.end_time):
            state = ego_vehicle.get_state_at_time_step(time_step)
            state.velocity = 1000
            ego_vehicle.set_state_at_time_step(time_step, state)

        modified_robustness = evaluator.evaluate(world_copy, ego_id)
        assert not np.all(np.isclose(orig_robustness, modified_robustness))

    @pytest.mark.parametrize(
        "vehicle_model,trajectory_factory",
        [
            (
                VehicleModel.PM,
                lambda num_steps: Trajectory(
                    0, [PMInputState(i, 0.0, 0.0) for i in range(num_steps)]
                ),
            ),
            (
                VehicleModel.PM,
                lambda num_steps: Trajectory(
                    0,
                    [
                        PMState(i, np.array([2.0 + i * 0.2 * 70.0, 0.0]), 70.0, 0.0)
                        for i in range(num_steps)
                    ],
                ),
            ),
            (
                VehicleModel.KS,
                lambda num_steps: Trajectory(
                    0, [InputState(i, 0.0, 0.0) for i in range(num_steps)]
                ),
            ),
            (
                VehicleModel.KS,
                lambda num_steps: Trajectory(
                    0,
                    [
                        KSState(
                            time_step=i,
                            position=np.array([2.0 + i * 0.2 * 70.0, 0.0]),
                            steering_angle=0.0,
                            velocity=70.0,
                            orientation=0.0,
                        )
                        for i in range(num_steps)
                    ],
                ),
            ),
        ],
        ids=["PM_input", "PM_state", "KS_input", "KS_state"],
    )
    def test_solution_evaluation(self, vehicle_model: VehicleModel, trajectory_factory) -> None:
        # Create scenario and lanelet network
        lanelet_network = LaneletNetwork()
        lanelets = parallel_lanes(1, 500.0)
        for lanelet in lanelets:
            lanelet_network.add_lanelet(lanelet)

        dt = 0.2  # s
        velocity = 70.0  # m/s
        scn = Scenario(dt=dt)
        scn.add_objects(lanelet_network)
        num_time_steps = 10

        # Create planning problem
        pp = PlanningProblem(
            10,
            InitialState(0, np.array([2, 2]), 0.0, velocity, 0.0, 0.0, 0.0),
            GoalRegion([CustomState(time_step=Interval(0, 10), position=Rectangle(2, 2))]),
        )

        # Create trajectory using the factory function
        trajectory = trajectory_factory(num_time_steps)

        # Create planning problem solution
        pp_sol = PlanningProblemSolution(
            10, vehicle_model, VehicleType.BMW_320i, CostFunction.MW1, trajectory
        )

        # Create world and evaluate
        world, ego_vehicle = World.create_from_solution(scn, pp, pp_sol)
        evaluator = OfflineRuleEvaluator.create_for_rule("R_G3", dt=world.scenario.dt)
        robs = evaluator.evaluate(world, ego_vehicle.id)

        # Assert all robustness values are negative
        assert np.all(np.array(robs) < 0)


if __name__ == "__main__":
    unittest.main()
