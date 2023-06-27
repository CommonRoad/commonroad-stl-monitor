import unittest
from pathlib import Path

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.solution import (
    CostFunction,
    PlanningProblemSolution,
    Solution,
    VehicleModel,
    VehicleType,
)
from commonroad.common.util import Interval
from commonroad.geometry.shape import Rectangle
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
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
from ruamel.yaml import YAML

from crmonitor.common.world import World
from crmonitor.evaluation.evaluation import RuleEvaluator
from crmonitor.monitor.rule import (
    AllNode,
    ExistNode,
    IOType,
    PredicateNode,
    RuleNode,
    parse_rule,
)
from tests.util import parallel_lanes


class TestRuleEvaluator(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        root_path = Path(__file__).parents[1] / "crmonitor"
        config_path = root_path / "config.yaml"
        self.config = YAML().load(config_path)
        rules_path = root_path / "traffic_rules_rtamt.yaml"
        self.traffic_rule_params = YAML().load(rules_path)
        self.scenario_root_path = root_path.parent / "scenarios/test_interstate"

    def test_smoke(self):
        rules = [
            "A a1: (in_front_of__a0_a1 and cut_in__a0_a1)",
            "A a1: (in_front_of__a0_a1) and single_lane__a0",
            "E a1: (in_front_of__a0_a1 and cut_in__a0_a1)",
            "E a1: (in_front_of__a0_a1) and single_lane__a0",
            "single_lane__a0",
            "single_lane__a0 and single_lane__a0",
            "A a1: (in_front_of__a0_a1)",
        ]

        scenario, _ = CommonRoadFileReader(
            str(self.scenario_root_path / "DEU_test_safe_distance_lane_change.xml")
        ).open(True)
        for r in rules:
            rule = parse_rule(r, self.traffic_rule_params)
            ws = World.create_from_scenario(scenario)
            ego_vehicle = ws.vehicle_by_id(1001)
            evaluator = RuleEvaluator(rule, ego_vehicle, ws)
            rob = evaluator.update()
            predicates = evaluator.get_predicates()
            node_values = evaluator.ast_node_values()

        evaluator.reset(ego_vehicle, ws)

    def test_parsing(self):
        rule = parse_rule(
            "A a1: (in_front_of__a0_a1 and cut_in__a0_a1)", self.traffic_rule_params
        )
        self.assertTrue(isinstance(rule, AllNode))
        rule = parse_rule(
            "A a1: (in_front_of__a0_a1) and single_lane__a0", self.traffic_rule_params
        )
        self.assertTrue(isinstance(rule, RuleNode))
        rule = parse_rule(
            "E a1: (in_front_of__a0_a1 and cut_in__a0_a1)", self.traffic_rule_params
        )
        self.assertTrue(isinstance(rule, ExistNode))
        rule = parse_rule(
            "E a1: (in_front_of__a0_a1) and single_lane__a0", self.traffic_rule_params
        )
        self.assertTrue(isinstance(rule, RuleNode))
        rule = parse_rule("single_lane__a0", self.traffic_rule_params)
        self.assertTrue(isinstance(rule, RuleNode))
        rule = parse_rule(
            "single_lane__a0 and single_lane__a0", self.traffic_rule_params
        )
        self.assertTrue(isinstance(rule, RuleNode))

        rule = parse_rule(
            "A a1: (in_front_of__a0_a1) and single_lane_i__a0", self.traffic_rule_params
        )
        self.assertTrue(isinstance(rule, RuleNode))
        self.assertTrue(isinstance(rule.children[1], PredicateNode))
        self.assertEqual(rule.children[1].io_type, IOType.INPUT)
        self.assertEqual(
            rule.children[0].children[0].children[0].io_type, IOType.OUTPUT
        )

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
            GoalRegion(
                [CustomState(time_step=Interval(0, 10), position=Rectangle(2, 2))]
            ),
        )
        pps = pp

        # Test solution as PM input trajectory
        trajectory = Trajectory(
            0, [PMInputState(i, 0.0, 0.0) for i in range(num_time_steps)]
        )
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
        trajectory = Trajectory(
            0, [InputState(i, 0.0, 0.0) for i in range(num_time_steps)]
        )
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
