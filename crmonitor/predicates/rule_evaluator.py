from collections import defaultdict
from functools import partial

import numpy as np

from crmonitor.common.evaluation import bool_to_norm_rob
from crmonitor.common.helper import gather
from crmonitor.common.world_state import WorldState
from crmonitor.monitor.rtamt_monitor_stl import RtamtStlMonitor
from crmonitor.predicates.rule import Rule, QuantificationType


class RuleEvaluator:

    def __init__(self, rule: Rule, output_type="standard"):
        self.use_boolean = False
        self.rule = rule
        if rule.quantification == QuantificationType.NONE:
            self.top_monitor = RtamtStlMonitor(rule, output_type)
        else:
            self.top_monitor = defaultdict(partial(RtamtStlMonitor, rule=rule, output_type=output_type))
        self.sub_rule_evaluator = {r.name: RuleEvaluator(r, output_type) for r in rule.sub_rules}

    def _evaluate_subrules(self, world_state: WorldState, quantified_vehicles):
        return {k : e.evaluate_robustness_incremental(world_state, quantified_vehicles) for k, e in self.sub_rule_evaluator.items()}

    def evaluate_robustness_incremental(self, world_state: WorldState, quantified_vehicles={}):
        if self.rule.quantification is not QuantificationType.NONE:
            quantified_placeholder = self.rule.quantified_vehicle
            other_ids = set([v.id for v in world_state.other_vehicles if v.is_valid(world_state.time_step)]).difference(quantified_vehicles.values())
            all_predicate_robustness = []
            all_rule_robustness = []
            other_vehicle = []
            while len(other_ids) > 0:
                quantified_vehicle = other_ids.pop()
                other_vehicle.append(quantified_vehicle)
                quantified_vehicles[quantified_placeholder] = quantified_vehicle
                predicate_values = self._evaluate_predicates(world_state, quantified_vehicles)
                all_predicate_robustness.append(predicate_values)
                all_rule_robustness.append(self.top_monitor[quantified_vehicle].evaluate_monitor_online(world_state.time_step, predicate_values))
            if self.rule.quantification == QuantificationType.ALL:
                idx = np.argmin(all_rule_robustness)
            else:
                idx = np.argmax(all_rule_robustness)
            # Select values of target vehicle
            rule_robustness = all_rule_robustness[idx]
            predicate_robustness = all_predicate_robustness[idx]
            other_ids_values = other_vehicle[idx]
        else:
            predicate_values = self._evaluate_predicates(world_state, quantified_vehicles)
            rule_robustness = self.top_monitor.evaluate_monitor_online(world_state.time_step, predicate_values)
        return rule_robustness

    def _evaluate_predicates(self, world_state, quantified_vehicles):
        rule_predicate_values = {}
        for pred_assign in self.rule.predicate_assignment:
            predicate_ids = gather(quantified_vehicles, pred_assign.agent_placeholders)
            if self.use_boolean:
                value = pred_assign.evaluator.evaluate_boolean(
                    world_state, predicate_ids
                )
                value = bool_to_norm_rob(value)
                world_state.predicate_values[world_state.time_step][
                    pred_assign.base_name
                ][tuple(predicate_ids)] = value
            else:
                pred_assign.evaluator.evaluate_robustness_with_cache(
                    world_state, predicate_ids
                )
            predicate_robustness = world_state.predicate_values[world_state.time_step][
                pred_assign.base_name][predicate_ids]
            rule_predicate_values[pred_assign.full_name] = predicate_robustness
        rule_predicate_values.update(self._evaluate_subrules(world_state, quantified_vehicles))
        return list(rule_predicate_values.items())