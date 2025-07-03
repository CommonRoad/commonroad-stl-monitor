import unittest
from pathlib import Path

import pytest
import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
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
from crmonitor.predicates.predicate_factory import PredicateFactory
from crmonitor.rule.rule_node import AllNode, ExistNode, IOType, PredicateNode, RuleAstNode

from crmonitor.rule.rule_parser import RuleParser
from tests.util import parallel_lanes
from tests.resources import InterstateScenarios


RULES = [
    "A a1: (in_front_of(a0, a1) and cut_in(a0, a1))",
    "A a1: (in_front_of(a0, a1)) and single_lane(a0)",
    "E a1: (in_front_of(a0, a1) and cut_in(a0, a1))",
    "E a1: (in_front_of(a0, a1)) and single_lane(a0)",
    "single_lane(a0)",
    "single_lane(a0) and single_lane(a0)",
    "A a1: (in_front_of(a0, a1))",
]


class TestOfflineEvaluator:
    @classmethod
    def setup_class(cls) -> None:
        cls._world = World.create_from_scenario(
            InterstateScenarios.SAFE_DISTANCE_LANE_CHANGE.get_commonroad_scenario()
        )

    @pytest.mark.parametrize("rule_str", RULES)
    def test_smoke(self, rule_str: str) -> None:
        evaluator = OfflineRuleEvaluator.create_for_rule_str(rule_str, dt=self._world.scenario.dt)

        robustness = evaluator.evaluate(self._world, ego_id=1001)
        assert isinstance(robustness, list)


class TestRuleEvaluator(unittest.TestCase):
    def test_solution(self):
        lanelet_network = LaneletNetwork()
        lanelets = parallel_lanes(1, 500.0)
        for l in lanelets:
            lanelet_network.add_lanelet(l)

        dt = 0.2  # s
        velocity = 70.0  # m/s
        scn = Scenario(dt=dt)
        scn.add_objects(lanelet_network)
        num_time_steps = 10

        pp = PlanningProblem(
            10,
            InitialState(0, np.array([2, 2]), 0.0, velocity, 0.0, 0.0, 0.0),
            GoalRegion([CustomState(time_step=Interval(0, 10), position=Rectangle(2, 2))]),
        )
        pps = pp

        # Test solution as PM input trajectory
        trajectory = Trajectory(0, [PMInputState(i, 0.0, 0.0) for i in range(num_time_steps)])
        pp_sol = PlanningProblemSolution(
            10, VehicleModel.PM, VehicleType.BMW_320i, CostFunction.MW1, trajectory
        )

        world, ego_vehicle = World.create_from_solution(scn, pps, pp_sol)
        evaluator = RuleEvaluator.create_from_config(world, ego_vehicle.id, rule="R_G3")
        robs = evaluator.evaluate()
        self.assertTrue(np.all(robs < 0))

        # Test solution as PM trajectory
        trajectory = Trajectory(
            0,
            [
                PMState(i, np.array([2.0 + i * dt * velocity, 0.0]), velocity, 0.0)
                for i in range(num_time_steps)
            ],
        )
        pp_sol = PlanningProblemSolution(
            10, VehicleModel.PM, VehicleType.BMW_320i, CostFunction.MW1, trajectory
        )

        world, ego_vehicle = World.create_from_solution(scn, pps, pp_sol)
        evaluator = RuleEvaluator.create_from_config(world, ego_vehicle.id, rule="R_G3")
        robs = evaluator.evaluate()
        self.assertTrue(np.all(robs < 0))

        # Test solution as KS input trajectory
        trajectory = Trajectory(0, [InputState(i, 0.0, 0.0) for i in range(num_time_steps)])
        pp_sol = PlanningProblemSolution(
            10, VehicleModel.KS, VehicleType.BMW_320i, CostFunction.MW1, trajectory
        )

        world, ego_vehicle = World.create_from_solution(scn, pps, pp_sol)
        evaluator = RuleEvaluator.create_from_config(world, ego_vehicle.id, rule="R_G3")
        robs = evaluator.evaluate()
        self.assertTrue(np.all(robs < 0))

        # Test solution as KS trajectory
        trajectory = Trajectory(
            0,
            [
                KSState(
                    time_step=i,
                    position=np.array([2.0 + i * dt * velocity, 0.0]),
                    steering_angle=0.0,
                    velocity=velocity,
                    orientation=0.0,
                )
                for i in range(num_time_steps)
            ],
        )
        pp_sol = PlanningProblemSolution(
            10, VehicleModel.KS, VehicleType.BMW_320i, CostFunction.MW1, trajectory
        )

        world, ego_vehicle = World.create_from_solution(scn, pps, pp_sol)
        evaluator = RuleEvaluator.create_from_config(world, ego_vehicle.id, rule="R_G3")
        robs = evaluator.evaluate()
        self.assertTrue(np.all(robs < 0))


if __name__ == "__main__":
    unittest.main()
