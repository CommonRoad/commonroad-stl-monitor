import copy
import itertools
import logging
from collections import defaultdict
from functools import partial
from typing import List, Tuple, Iterable, Dict

import numpy as np
import pandas as pd
from crmonitor.common.helper import gather, pandas_from_nested_dict
from crmonitor.common.vehicle import Vehicle
from crmonitor.common.world_state import WorldState
from crmonitor.monitor.rtamt_monitor_stl import TrafficRuleMonitorForwardSTL
from crmonitor.predicates.rule import Rule, QuantificationType


def get_valid_time_interval(vehicles: List[Vehicle]):
    start = max([v.start_time for v in vehicles])
    end = min([v.end_time for v in vehicles])
    return start, end


class RuleSetEvaluator:
    """
    Evaluate individual vehicles of CommonRoad scenarios
    """

    def __init__(self, rules: Iterable[Rule]) -> None:
        """
        :param rules: set of rules to be evaluated
        """
        self.rules = tuple(rules)
        self.monitors = {
            rule: defaultdict(
                partial(TrafficRuleMonitorForwardSTL, rule, output_type="standard")
            )
            for rule in rules
        }
        default_dict_factory = partial(defaultdict, dict)
        self.predicate_values = defaultdict(default_dict_factory)
        self._last_world_state = None
        self._last_time_step = -1

    def clear_cache_timesteps(self, start_time_step, end_time_step):
        """
        Clear internal cached predicates between for the given time interval
        :param start_time_step: start of the interval (inclusive)
        :param end_time_step: end of the interval (inclusive)
        :return:
        """
        for i in range(start_time_step, end_time_step + 1):
            self.predicate_values.pop(i)

    def _evaluate_predicates_timestep(
        self, rule: Rule, world_state: WorldState, other_ids: Tuple[int]
    ) -> None:
        """
        Evaluate all predicates of a rule for a given time step and a tuple
        of other vehicles. Results are written to the internal cache
        :param rule: rule containing predicates
        :param world_state: current world state
        :param other_ids: tuple of other vehicles
        """
        ids = (world_state.ego_vehicle.id,) + other_ids
        for pred_assign in rule.predicate_assignment:
            predicate_ids = gather(ids, pred_assign.agent_placeholders)
            if (
                self.predicate_values[world_state.time_step][pred_assign.base_name].get(
                    predicate_ids
                )
                is None
            ):
                logging.debug(
                    "Evaluating predicate %s , t=%d, ids=%s",
                    pred_assign.base_name,
                    world_state.time_step,
                    predicate_ids,
                )
                value = pred_assign.evaluator.evaluate_robustness(
                    world_state, predicate_ids
                )
                self.predicate_values[world_state.time_step][pred_assign.base_name][
                    predicate_ids
                ] = value

    def _evaluate_rule_timestep(
        self,
        world_state: WorldState,
        other_ids: Tuple[int],
        monitor: TrafficRuleMonitorForwardSTL,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate a rule for on time step
        :param world_state: current world state
        :param other_ids: tuple of other vehicles
        :param monitor: monitor object of the rule
        :return: Tuple of robustness value and dictionary of the predicate
        values
        """
        predicate_values = {}
        self._evaluate_predicates_timestep(monitor.rule, world_state, other_ids)
        for pred_assign in monitor.rule.predicate_assignment:
            ids = (world_state.ego_vehicle.id,) + other_ids
            predicate_ids = gather(ids, pred_assign.agent_placeholders)
            v = self.predicate_values[world_state.time_step][pred_assign.base_name][
                predicate_ids
            ]
            predicate_values[pred_assign.full_name] = v
        rob_value = monitor.evaluate_monitor_online(
            world_state.time_step, list(predicate_values.items())
        )
        return rob_value, predicate_values

    def _check_cache(self, world_state: WorldState):

        if self._last_world_state is None or (
            self._last_world_state is not world_state
            and (
                not hasattr(self._last_world_state, "scenario")
                or hasattr(self._last_world_state, "scenario")
                and (
                    self._last_world_state.scenario.scenario_id
                    != world_state.scenario.scenario_id
                    or world_state.ego_vehicle.id
                    != self._last_world_state.ego_vehicle.id
                )
            )
        ):
            # Clear predicate values and monitor states, if a different world state was given as input
            logging.debug("Clear internal cache!")
            self.predicate_values.clear()
            self._last_world_state = world_state
            self._last_time_step = -1
            for rule_mons in self.monitors.values():
                # rule_mons.clear()
                for monitor in list(rule_mons.values()):
                    monitor.reset_monitor()

        elif world_state.time_step < self._last_time_step:
            # Clear only monitor states, if time was decremented
            self._last_time_step = -1
            for rule_mons in self.monitors.values():
                # rule_mons.clear()
                for monitor in list(rule_mons.values()):
                    monitor.reset_monitor()

    def evaluate_incremental(
        self, world_state: WorldState, to_panda=True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Evaluate world state for each time step since the last evaluation.
        :param world_state: world state to evaluate
        :return: Tuple of pandas dataframes, where the first contains rule
            robustness and the second predicate robustness
        """
        self._check_cache(world_state)
        # Only evaluate if ego vehicle is present
        time_begin = max(self._last_time_step + 1, world_state.ego_vehicle.start_time)
        time_end = min(world_state.time_step, world_state.ego_vehicle.end_time)
        self._last_time_step = world_state.time_step
        # Create a flat copy of the world to iterate through time
        world_state = copy.copy(world_state)
        world_state.time_step = time_begin
        # Nested dictionary with levels: time step and rule
        rule_robustness = {}
        # Nested dictionary with levels: time step, rule, predicate name
        predicate_robustness = {}
        other_ids_values = {}
        while world_state.time_step <= time_end:
            t = world_state.time_step
            ids = [
                i
                for i in world_state.other_ids
                if world_state.vehicle_by_id(i).is_valid(t)
            ]
            rule_robustness[t] = {}
            predicate_robustness[t] = {}
            other_ids_values[t] = {}
            for rule in self.rules:
                other_ids = list(
                    itertools.combinations(ids, rule.num_dependent_vehicles)
                )
                if len(other_ids) == 0:
                    # Case where ego vehicle is the only vehicle
                    val = 1.0 if rule.quantification == QuantificationType.ALL else -1.0
                    rule_robustness[t][rule.name] = val
                    predicate_robustness[t][rule.name] = {
                        name: val for name in rule.predicate_names
                    }
                    other_ids_values[t][rule.name] = tuple()
                else:
                    rule_values = []
                    pred_values = []
                    for selected_other_ids in other_ids:
                        monitor = self.monitors[rule][selected_other_ids]
                        rule_value, pred_value = self._evaluate_rule_timestep(
                            world_state, selected_other_ids, monitor
                        )
                        rule_values.append(rule_value)
                        pred_values.append(pred_value)
                    if rule.quantification == QuantificationType.ALL:
                        idx = np.argmin(rule_values)
                    else:
                        idx = np.argmax(rule_values)
                    # Select values of target vehicle
                    rule_robustness[t][rule.name] = rule_values[idx]
                    predicate_robustness[t][rule.name] = pred_values[idx]
                    other_ids_values[t][rule.name] = other_ids[idx]
            world_state.step()

        if to_panda:
            df_rule = pandas_from_nested_dict(
                rule_robustness, ["time_step", "rule_name", "robustness"]
            )
            df_pred = pandas_from_nested_dict(
                predicate_robustness,
                ["time_step", "rule_name", "full_name", "robustness"],
            )
            df_ids = pandas_from_nested_dict(other_ids_values, ["time_step", "rule_name", "other_ids"])
            df_rule = df_rule.merge(df_ids, on=["time_step", "rule_name"])
            return df_rule, df_pred
        else:
            return rule_robustness, predicate_robustness
