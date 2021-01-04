import unittest

from commonroad.common.file_reader import CommonRoadFileReader

from crmonitor.common.evaluation import \
    evaluate_rule
from crmonitor.common.helper import load_yaml, gather
from crmonitor.common.world_state import WorldState
from crmonitor.predicates.python.rule import Rule
from crmonitor.monitor.rtamt_monitor_stl import TrafficRuleMonitorForwardSTL
import logging

logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.DEBUG)


class RefactoringTests(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        config_path = "crmonitor/rtamt/config.yaml"
        self.config = load_yaml(config_path)
        self.scenario_file = "scenarios/test_interstate/DEU_test_safe_distance_lane_change.xml"
        self.rule_str = "in_front_of_i__a0_a1 and in_same_lane_i__a0_a1 " \
                        "implies " \
                        "keeps_safe_distance_prec__a1_a0"
        self.ego_id = 1005
        self.other_id = 1004

        scenario, _ = CommonRoadFileReader(self.scenario_file).open(
                lanelet_assignment=True)
        self.world_state = WorldState(scenario, self.ego_id, self.config)
        self.rule = Rule(self.rule_str, self.config.get("ego_vehicle_param"))

    def test_one_rule_one_agent_offline_stepwise(self):
        assert self.rule.num_dependent_vehicles == 1
        rob_values, pred_values = evaluate_rule(self.world_state, self.other_id, self.rule)
        pass


if __name__ == "__main__":
    unittest.main()
