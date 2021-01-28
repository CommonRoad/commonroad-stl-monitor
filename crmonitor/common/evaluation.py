import itertools
import math
from typing import List, Tuple, Iterable

import numpy as np

from crmonitor.common.helper import gather
from crmonitor.common.vehicle import Vehicle
from crmonitor.common.world_state import WorldState
from crmonitor.monitor.rtamt_monitor_stl import TrafficRuleMonitorForwardSTL
from crmonitor.predicates.python.predicate_value import PredicateValue, \
    PredicateValueCollection
from crmonitor.predicates.python.rule import Rule


def get_valid_time_interval(vehicles: List[Vehicle]):
    start = max([v.start_time for v in vehicles])
    end = min([v.end_time for v in vehicles])
    return start, end


class RuleSetEvaluator:

    def __init__(self, rules: Iterable[Rule]) -> None:
        self.rules = tuple(rules)
        self.num_dependent_vehicles = self.rules[0].num_dependent_vehicles
        assert all(
                [self.num_dependent_vehicles == x.num_dependent_vehicles for x
                 in self.rules])
        self.predicate_assignments = set()
        for rule in rules:
            self.predicate_assignments.update(rule.predicate_assignment)
        self.predicate_values = PredicateValueCollection()
        self._last_world_state = None

    def evaluate_predicates_timestep(self, world_state: WorldState,
                                     other_ids: Tuple[int]):
        if not (self._last_world_state is world_state):
            self.predicate_values.clear()
            self._last_world_state = world_state
        ids = (world_state.ego_vehicle.id,) + other_ids
        for pred_assign in self.predicate_assignments:
            predicate_ids = gather(ids, pred_assign.agent_placeholders)
            value = PredicateValue(pred_assign.base_name, predicate_ids,
                                   world_state.time_step)
            if value not in self.predicate_values:
                value.value = pred_assign.evaluator.evaluate_robustness(
                        world_state, predicate_ids)
                self.predicate_values.append(value)

    def evaluate_rule_timestep(self, world_state: WorldState,
                               other_ids: Tuple[int],
                               monitor: TrafficRuleMonitorForwardSTL,
                               rule: Rule):
        l = []
        for pred_assign in rule.predicate_assignment:
            ids = (world_state.ego_vehicle.id,) + other_ids
            predicate_ids = gather(ids, pred_assign.agent_placeholders)
            v = self.predicate_values.by_time_step(
                    world_state.time_step).by_name(
                    pred_assign.base_name).by_ids(
                    predicate_ids).get_single_value().value
            l.append((pred_assign.full_name, v))
        rob_value = monitor.evaluate_monitor_online(world_state.time_step, l)
        return rob_value, l

    def evaluate_rule_all_timesteps(self, rule, world_state,
                                    other_ids: Tuple[int], interval=None):
        if interval is None:
            start, end = get_valid_time_interval(
                    [world_state.ego_vehicle] + [world_state.vehicle_by_id(o_id)
                                                 for o_id in other_ids])
        else:
            start, end = interval
        monitor = TrafficRuleMonitorForwardSTL(rule, output_type="standard")
        rule_predicate_values = {}
        rob_values = {}
        world_state.time_step = start
        while world_state.time_step <= end:
            self.evaluate_predicates_timestep(world_state, other_ids)
            rob_value, predicate_values = self.evaluate_rule_timestep(
                    world_state, other_ids, monitor, rule)
            rob_values[world_state.time_step] = rob_value
            rule_predicate_values[world_state.time_step] = predicate_values
            world_state.step()
        return rob_values, rule_predicate_values

    def evaluate_all_rules_all_timesteps(self, world_state,
                                         other_ids: Tuple[int], interval=None):
        rules_rob_values = []
        rules_predicate_values = []
        for rule in self.rules:
            rob_values, predicate_values = self.evaluate_rule_all_timesteps(
                    rule, world_state, other_ids, interval)
            rules_rob_values.append(rob_values)
            rules_predicate_values.append(predicate_values)
        return rules_rob_values, rules_predicate_values

    def evaluate_all_rules_all_timesteps_floating(self, world_state: WorldState):
        rule_value_dict = {}
        pred_value_dict = {}
        for rule in self.rules:
            rule_values = []
            pred_values = []
            for selected_other_ids in itertools.combinations(world_state.other_ids, rule.num_dependent_vehicles):
                rule_value, pred_value = self.evaluate_rule_all_timesteps(rule, world_state, selected_other_ids)
                rule_values.append(rule_value)
                pred_values.append(pred_value)

            for t in range(world_state.ego_vehicle.start_time, world_state.ego_vehicle.end_time):
                # Iterate over agent combinations
                t_rob_values = [r.get(t, math.inf) for r in rule_values]
                idx = np.argmin(t_rob_values)
                rule_value_dict.setdefault(rule.name, []).append(t_rob_values[idx])
                pred_value_dict.setdefault(rule.name, []).append(pred_values[idx][t])
        return rule_value_dict, pred_value_dict

