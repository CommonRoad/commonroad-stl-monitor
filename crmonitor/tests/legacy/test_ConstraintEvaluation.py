mport os
import unittest
import numpy as np

from commonroad.geometry.shape import Polygon
from commonroad.scenario.traffic_sign_interpreter import TrafficSigInterpreter
from commonroad.common.file_reader import CommonRoadFileReader

from crmonitor.predicates.legacy.predicate_collection import ConstraintType, ConstraintRepresentation, Constraint, \
    ConstraintEvaluation
from crmonitor.common.helper import *
from crmonitor.common.road_network import RoadNetwork
from crmonitor.predicates.legacy.braking_predicates import BrakingPredicateCollection

import os
import unittest

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.geometry.shape import Polygon
from commonroad.scenario.traffic_sign_interpreter import TrafficSigInterpreter

from crmonitor.common.helper import *
from crmonitor.common.road_network import RoadNetwork
from crmonitor.predicates.legacy.braking_predicates import \
    BrakingPredicateCollection
from crmonitor.predicates.legacy.predicate_collection import ConstraintType, \
    ConstraintRepresentation, Constraint, ConstraintEvaluation


class TestConstraintEvaluation(unittest.TestCase):
    def setUp(self) -> None:
        config_path = os.path.dirname(__file__) + "/../crmonitor/"
        config = load_yaml(config_path + "config.yaml")
        traffic_rules = load_yaml(config_path + "traffic_rules.yaml")
        simulation_param = create_simulation_param(config.get("simulation_param"), 0.1, 'DEU')
        other_vehicles_param = create_other_vehicles_param(config.get("other_vehicles_param"))
        ego_vehicle_param = create_other_vehicles_param(config.get("ego_vehicle_param"))
        traffic_rule_param = traffic_rules.get("traffic_rules_param")
        road_network_param = config.get("road_network_param")
        test_scenario_dir = os.path.dirname(__file__) + "/../scenarios/test_interstate/"

        scenario, planning_problem_set = CommonRoadFileReader(test_scenario_dir + "DEU_test_safe_distance.xml"). \
            open(lanelet_assignment=True)
        road_network = RoadNetwork(scenario.lanelet_network, road_network_param)
        necessary_predicates = {"keeps_safe_distance_prec__x_ego__x_o"}
        traffic_sign_interpreter = TrafficSigInterpreter(simulation_param.get("country"),
                                                         road_network.lanelet_network)
        braking_predicates = BrakingPredicateCollection(road_network, simulation_param, traffic_rule_param,
                                                        necessary_predicates, traffic_sign_interpreter)

        ego_id = 1000
        self.ego_vehicle, self.other_vehicles = \
            create_scenario_vehicles(simulation_param.get("dt"), scenario.obstacle_by_id(ego_id), ego_vehicle_param,
                                     other_vehicles_param, road_network, scenario.dynamic_obstacles)

        self.constr_eval = ConstraintEvaluation([braking_predicates])

    def test_evaluate_constraints(self):
        sol_constraints = [72.09438095238093, 72.09438095238093, 71.24038095238095, 71.24038095238095,
                           70.39038095238094, 70.39038095238094]
        time_interval = (5, 10)
        constraints_per_time_step = self.constr_eval.evaluate_constraints(self.ego_vehicle, self.other_vehicles,
                                                                          time_interval)
        constraints = []
        for constraint_list in constraints_per_time_step.values():
            constraints.append(constraint_list[0].value)
        self.assertListEqual(sol_constraints, constraints)
        self.assertEqual(len(constraints_per_time_step), 6)

    def test_unify_constraints(self):
        axis = [ConstraintType.LONGITUDINAL_CURVILINEAR_POSITION, ConstraintType.LATERAL_CURVILINEAR_POSITION]
        value = Polygon(np.array([[0, 0], [0, 1], [1, 1], [1, 0]]))
        constraint_1 = Constraint(axis, ConstraintRepresentation.INNER_BOUNDARY, value)

        axis = [ConstraintType.LONGITUDINAL_CURVILINEAR_POSITION, ConstraintType.LATERAL_CURVILINEAR_POSITION]
        value = Polygon(np.array([[0.5, 0.5], [0.5, 1.5], [1.5, 1.5], [1.5, 0.5]]))
        constraint_2 = Constraint(axis, ConstraintRepresentation.INNER_BOUNDARY, value)

        axis = [ConstraintType.LONGITUDINAL_CURVILINEAR_POSITION, ConstraintType.LATERAL_CURVILINEAR_POSITION]
        value = Polygon(np.array([[0, 0], [0, 1], [1, 1], [1, 0]]))
        constraint_3 = Constraint(axis, ConstraintRepresentation.OUTER_BOUNDARY, value)

        axis = [ConstraintType.LONGITUDINAL_CURVILINEAR_POSITION, ConstraintType.LATERAL_CURVILINEAR_POSITION]
        value = Polygon(np.array([[10, 0], [10, 1], [11, 1], [11, 0]]))
        constraint_4 = Constraint(axis, ConstraintRepresentation.OUTER_BOUNDARY, value)

        axis = [ConstraintType.LONGITUDINAL_CURVILINEAR_POSITION]
        value = -1
        constraint_5 = Constraint(axis, ConstraintRepresentation.LOWER, value)

        axis = [ConstraintType.LONGITUDINAL_CURVILINEAR_POSITION]
        value = 2
        constraint_6 = Constraint(axis, ConstraintRepresentation.LOWER, value)

        axis = [ConstraintType.LATERAL_CURVILINEAR_POSITION]
        value = 5
        constraint_7 = Constraint(axis, ConstraintRepresentation.UPPER, value)

        axis = [ConstraintType.LATERAL_CURVILINEAR_POSITION]
        value = 20
        constraint_8 = Constraint(axis, ConstraintRepresentation.UPPER, value)

        axis = [ConstraintType.LONGITUDINAL_CURVILINEAR_POSITION, ConstraintType.LATERAL_CURVILINEAR_POSITION]
        value = Polygon(np.array([[0, 0], [0, 1], [1, 1], [1, 0]]))
        constraint_9 = Constraint(axis, ConstraintRepresentation.INNER_BOUNDARY, value)

        axis = [ConstraintType.LONGITUDINAL_CURVILINEAR_POSITION, ConstraintType.LATERAL_CURVILINEAR_POSITION]
        value = Polygon(np.array([[2.0, 2.0], [2.0, 3.0], [3.0, 3.0], [3.0, 2.0]]))
        constraint_10 = Constraint(axis, ConstraintRepresentation.INNER_BOUNDARY, value)

        axis = [ConstraintType.LONGITUDINAL_CURVILINEAR_POSITION, ConstraintType.LATERAL_CURVILINEAR_POSITION]
        value = Polygon(np.array([[0, 0], [0, 1], [1, 1], [1, 0]]))
        constraint_11 = Constraint(axis, ConstraintRepresentation.OUTER_BOUNDARY, value)

        axis = [ConstraintType.LONGITUDINAL_CURVILINEAR_POSITION, ConstraintType.LATERAL_CURVILINEAR_POSITION]
        value = Polygon(np.array([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]]))
        constraint_12 = Constraint(axis, ConstraintRepresentation.OUTER_BOUNDARY, value)

        constraints_1 = [constraint_1, constraint_2, constraint_3, constraint_4, constraint_5, constraint_6,
                         constraint_7, constraint_8]
        unified_constraints_1 = self.constr_eval.unify_constraints(constraints_1)

        self.assertEqual(unified_constraints_1[0].axis[0].value, ConstraintType.LONGITUDINAL_CURVILINEAR_POSITION.value)
        self.assertEqual(unified_constraints_1[0].axis[1].value, ConstraintType.LATERAL_CURVILINEAR_POSITION.value)
        self.assertEqual(unified_constraints_1[1].axis[0].value, ConstraintType.LONGITUDINAL_CURVILINEAR_POSITION.value)
        self.assertEqual(unified_constraints_1[1].axis[1].value, ConstraintType.LATERAL_CURVILINEAR_POSITION.value)
        self.assertEqual(unified_constraints_1[2].axis[0].value, ConstraintType.LONGITUDINAL_CURVILINEAR_POSITION.value)
        self.assertEqual(unified_constraints_1[3].axis[0].value, ConstraintType.LATERAL_CURVILINEAR_POSITION.value)

        self.assertEqual(unified_constraints_1[0].constraint_representation.value,
                         ConstraintRepresentation.INNER_BOUNDARY.value)
        self.assertEqual(unified_constraints_1[1].constraint_representation.value,
                         ConstraintRepresentation.OUTER_BOUNDARY.value)
        self.assertEqual(unified_constraints_1[2].constraint_representation.value, ConstraintRepresentation.LOWER.value)
        self.assertEqual(unified_constraints_1[3].constraint_representation.value, ConstraintRepresentation.UPPER.value)

        np.testing.assert_equal(unified_constraints_1[0].value.vertices,
                                np.array([[0.5, 1.0], [1.0, 1.0], [1.0, 0.5], [0.5, 0.5], [0.5, 1.0]]))
        np.testing.assert_equal(unified_constraints_1[1].value.shapes[0].vertices,
                                np.array([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]))
        np.testing.assert_equal(unified_constraints_1[1].value.shapes[1].vertices,
                                np.array([[10, 0], [10, 1], [11, 1], [11, 0], [10, 0]]))
        self.assertEqual(unified_constraints_1[2].value, 2)
        self.assertEqual(unified_constraints_1[3].value, 5)

        constraints_1 = [constraint_9, constraint_10, constraint_11, constraint_12]
        unified_constraints_2 = self.constr_eval.unify_constraints(constraints_1)

        self.assertEqual(unified_constraints_2[0].axis[0].value, ConstraintType.LONGITUDINAL_CURVILINEAR_POSITION.value)
        self.assertEqual(unified_constraints_2[0].axis[1].value, ConstraintType.LATERAL_CURVILINEAR_POSITION.value)
        self.assertEqual(unified_constraints_2[1].axis[0].value, ConstraintType.LONGITUDINAL_CURVILINEAR_POSITION.value)
        self.assertEqual(unified_constraints_2[1].axis[1].value, ConstraintType.LATERAL_CURVILINEAR_POSITION.value)

        self.assertEqual(unified_constraints_2[0].constraint_representation.value,
                         ConstraintRepresentation.INNER_BOUNDARY.value)
        self.assertEqual(unified_constraints_2[1].constraint_representation.value,
                         ConstraintRepresentation.OUTER_BOUNDARY.value)

        self.assertIsNone(unified_constraints_2[0].value)
        np.testing.assert_equal(unified_constraints_2[1].value.vertices,
                                np.array([[0, 0], [0, 1], [0.5, 1], [0.5, 1.5], [1.5, 1.5], [1.5, 0.5], [1.0, 0.5],
                                          [1.0, 0.0], [0.0, 0.0]]))
