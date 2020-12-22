import unittest

from commonroad.common.file_reader import CommonRoadFileReader

from crmonitor.common.evaluation import evaluate_necessary_predicates_all_agents, \
    PredicateValueCollection
from crmonitor.common.helper import load_yaml
from crmonitor.common.world_state import WorldState
from crmonitor.predicates.python.rule import Rule
from crmonitor.monitor.rtamt_monitor_stl import TrafficRuleMonitorForwardSTL


class RefactoringTests(unittest.TestCase):



    def setUp(self) -> None:
        super().setUp()
        self.config_path = "crmonitor/rtamt/config.yaml"
        self.scenario_file = "scenarios/test_interstate/DEU_test_safe_distance_lane_change.xml"
        self.rule_str = "in_front_of_i__a0_a1 and in_same_lane_i__a0_a1 " \
                   "implies " \
                   "keeps_safe_distance_prec__a0_a1"

    def test_one_rule_one_agent_offline_stepwise(self):
        ego_id = 1005
        other_id = 1004

        config = load_yaml(self.config_path)
        scenario, _ = CommonRoadFileReader(self.scenario_file).open(
            lanelet_assignment=True)
        world_state = WorldState(scenario, ego_id, config)

        rules = [Rule(self.rule_str, config.get("ego_vehicle_param"))]

        all_pred_values = PredicateValueCollection()
        for state in world_state:
            pred_values = evaluate_necessary_predicates_all_agents(rules, state)
            all_pred_values.extend(pred_values)

        monitor = TrafficRuleMonitorForwardSTL(rules[0],
                                               output_type="output-robustness")

        filtered_predicates = all_pred_values.by_ids((ego_id, other_id))
        rob_values = monitor.evaluate_monitor_offline_stepwise(
            filtered_predicates)

    def test_one_rule_all_agents_offline_stepwise(self):
        pass


if __name__ == "__main__":
    unittest.main()
