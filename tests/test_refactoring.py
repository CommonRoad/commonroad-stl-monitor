import unittest

from commonroad.common.file_reader import CommonRoadFileReader

from crmonitor.common.evaluation import evaluate_necessary_predicates_all_agents, \
    PredicateValueCollection, \
    PredicateValue
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

    # def test_one_rule_one_agent_offline_stepwise(self):
    #
    #     all_pred_values = PredicateValueCollection()
    #     for state in self.world_state:
    #         pred_values = evaluate_necessary_predicates_all_agents([self.rules], state)
    #         all_pred_values.extend(pred_values)
    #
    #     monitor = TrafficRuleMonitorForwardSTL(self.rule,
    #                                            output_type="output-robustness")
    #
    #     filtered_predicates = all_pred_values.by_ids((self.ego_id, self.other_id))
    #     rob_values = monitor.evaluate_monitor_offline_stepwise(
    #         filtered_predicates)

    def test_one_rule_all_agents_offline_stepwise(self):
        assert self.rule.num_dependent_vehicles == 1
        all_pred_values = PredicateValueCollection()
        rob_values_all = {}
        other_ids = set([v.id for v in self.world_state.other_vehicles])

        for o_id in other_ids:

            # Collect predicates
            assignment = (self.ego_id, o_id)
            pred_values = PredicateValueCollection()
            for state in self.world_state:
                for pred_assign in self.rule.predicate_assignment:
                    predicate_ids = gather(assignment,
                                           pred_assign.agent_placeholders)
                    value = PredicateValue(pred_assign.base_name, predicate_ids,
                                           state.time_step)
                    if value not in pred_values:
                        value.value = pred_assign.evaluator.evaluate_robustness(
                            state, predicate_ids)
                        all_pred_values.append(value)
                    pred_values.append(all_pred_values[value])

            # Evaluate rule
            monitor = TrafficRuleMonitorForwardSTL(self.rule,
                                                   output_type="output-robustness")
            monitor_values = {}
            for state in self.world_state:
                l = []
                for pred_assign in self.rule.predicate_assignment:
                    ids = gather(assignment, pred_assign.agent_placeholders)
                    v = pred_values.by_time_step(state.time_step).by_name(
                        pred_assign.base_name).by_ids(ids)
                    self.assertIsInstance(v, float)
                    l.append((pred_assign.full_name, v))

                monitor_values[state.time_step] = l

            rob_values = monitor.evaluate_monitor_offline_stepwise(
                monitor_values)
            rob_values_all[o_id] = rob_values
        pass


if __name__ == "__main__":
    unittest.main()
