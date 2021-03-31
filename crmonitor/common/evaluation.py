import itertools
import logging
from collections import defaultdict
from functools import partial
from typing import List, Tuple, Iterable

import pandas as pd

from crmonitor.common.helper import gather
from crmonitor.common.vehicle import Vehicle
from crmonitor.common.world_state import WorldState
from crmonitor.monitor.rtamt_monitor_stl import TrafficRuleMonitorForwardSTL
from crmonitor.predicates.python.rule import Rule, QuantificationType


def flatten_nested_dict(data, path=tuple()):
    entries = []
    for key, val in data.items():
        if isinstance(val, dict):
            entries.extend(flatten_nested_dict(val, path + (key,)))
        else:
            entries.append(path + (key, val))
    return entries


def pandas_from_nested_dict(data, level_names):
    entries = flatten_nested_dict(data)
    return pd.DataFrame(entries, columns=level_names)


def get_valid_time_interval(vehicles: List[Vehicle]):
    start = max([v.start_time for v in vehicles])
    end = min([v.end_time for v in vehicles])
    return start, end


class RuleSetEvaluator:

    def __init__(self, rules: Iterable[Rule]) -> None:
        self.rules = tuple(rules)
        self.num_dependent_vehicles = self.rules[0].num_dependent_vehicles
        default_dict_factory = partial(defaultdict, dict)
        self.predicate_values = defaultdict(default_dict_factory)
        self._last_world_state = None

    def clear_cache_timesteps(self, start_time_step, end_time_step):
        for i in range(start_time_step, end_time_step + 1):
            self.predicate_values.pop(i)

    def evaluate_predicates_timestep(self, rule: Rule, world_state: WorldState,
                                     other_ids: Tuple[int]):
        if self._last_world_state is None or self._last_world_state.scenario.scenario_id != world_state.scenario.scenario_id or world_state.ego_vehicle.id != self._last_world_state.ego_vehicle.id:
            if self._last_world_state is not None:
                logging.debug("Clear predicate cache!")
            self.predicate_values.clear()
            self._last_world_state = world_state
        ids = (world_state.ego_vehicle.id,) + other_ids
        for pred_assign in rule.predicate_assignment:
            predicate_ids = gather(ids, pred_assign.agent_placeholders)
            if self.predicate_values[world_state.time_step][
                pred_assign.base_name].get(predicate_ids) is None:
                logging.debug("Evaluating predicate %s , t=%d, ids=%s",
                              pred_assign.base_name, world_state.time_step,
                              predicate_ids)
                value = pred_assign.evaluator.evaluate_robustness(world_state,
                        predicate_ids)
                self.predicate_values[world_state.time_step][
                    pred_assign.base_name][predicate_ids] = value


    def evaluate_rule_timestep(self, world_state: WorldState, other_ids: Tuple[int],
                               monitor: TrafficRuleMonitorForwardSTL, rule: Rule):
        l = {}
        self.evaluate_predicates_timestep(rule, world_state, other_ids)
        for pred_assign in rule.predicate_assignment:
            ids = (world_state.ego_vehicle.id,) + other_ids
            predicate_ids = gather(ids, pred_assign.agent_placeholders)
            v = self.predicate_values[world_state.time_step][pred_assign.base_name][
                predicate_ids]
            l[pred_assign.full_name] = v
        rob_value = monitor.evaluate_monitor_online(world_state.time_step,
                                                    list(l.items()))
        return rob_value, l


    def evaluate_rule_all_timesteps(self, rule, world_state, other_ids: Tuple[int],
                                    interval=None):
        if interval is None:
            start, end = get_valid_time_interval(
                    [world_state.ego_vehicle] + [world_state.vehicle_by_id(o_id) for
                                                 o_id in other_ids])
        else:
            start, end = interval
        monitor = TrafficRuleMonitorForwardSTL(rule, output_type="standard")
        rule_predicate_values = {}
        rob_values = {}
        world_state.time_step = start
        while world_state.time_step <= end:
            rob_value, predicate_values = self.evaluate_rule_timestep(world_state,
                    other_ids, monitor, rule)
            rob_values[world_state.time_step] = rob_value
            rule_predicate_values[world_state.time_step] = predicate_values
            world_state.step()
        return rob_values, rule_predicate_values


    def evaluate_all_rules_all_timesteps(self, world_state, other_ids: Tuple[int],
                                         interval=None):
        raise NotImplementedError


    def evaluate_all_rules_all_timesteps_floating(self, world_state: WorldState):
        rule_value_dict = {}
        pred_value_dict = {}
        for rule in self.rules:
            rule_values = {}
            pred_values = {}
            for selected_other_ids in itertools.combinations(world_state.other_ids,
                                                             rule.num_dependent_vehicles):
                rule_value, pred_value = self.evaluate_rule_all_timesteps(rule,
                                                                          world_state,
                                                                          selected_other_ids)
                rule_values[selected_other_ids] = rule_value
                pred_values[selected_other_ids] = pred_value
            rule_value_dict[rule.name] = rule_values
            pred_value_dict[rule.name] = pred_values

        df_rule = pandas_from_nested_dict(rule_value_dict,
                                          ["rule_name", "other_ids", "time_step",
                                           "rob"])
        df_pred = pandas_from_nested_dict(pred_value_dict,
                                          ["rule_name", "other_ids", "time_step",
                                           "full_name", "value"])
        df_rule_mins = []
        df_pred_mins = []
        for rule in self.rules:
            for t in range(world_state.ego_vehicle.start_time,
                           world_state.ego_vehicle.end_time + 1):
                df = df_rule[(df_rule["rule_name"] == rule.name) & (
                            df_rule["time_step"] == t)]
                if rule.quantification == QuantificationType.ALL:
                    rob_min = df.rob.min()
                else:
                    rob_min = df.rob.max()
                df = df[df["rob"] == rob_min]
                if df.empty:
                    df = pd.DataFrame.from_records([{
                                                        "rule_name": rule.name,
                                                        "time_step": t,
                                                        "other_ids": (-1,),
                                                        "rob": 1.0}])
                    pred = pd.DataFrame({"full_name": rule.predicate_names})
                    pred["rule_name"] = rule.name
                    pred["other_ids"] = (-1,)
                    pred["time_step"] = t
                    pred["value"] = 1.0 if rule.quantification == QuantificationType.ALL else -1.0
                else:
                    other_ids = df.head(1)["other_ids"].values[0]
                    pred = df_pred[(df_pred["rule_name"] == rule.name) & (
                                df_pred["time_step"] == t) & (
                                               df_pred["other_ids"] == other_ids)]
                df_rule_mins.append(df.head(1))
                df_pred_mins.append(pred)

        df_rule_mins = pd.concat(df_rule_mins)
        df_pred_mins = pd.concat(df_pred_mins)

        return df_rule_mins, df_pred_mins
