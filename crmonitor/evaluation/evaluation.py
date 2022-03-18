import copy
from pathlib import Path
from typing import List, Iterable, Union, Tuple

import numpy as np

import crmonitor
from ruamel.yaml import YAML

from crmonitor.common.helper import load_yaml
from crmonitor.common.road_network import RoadNetwork
from crmonitor.common.vehicle import Vehicle
from crmonitor.common.world_state import WorldState
from crmonitor.evaluation.visitor import (MonitorCreationRuleTreeVisitor, EvaluationMonitorTreeVisitor,
                                          PredicateCollectorMonitorTreeVisitor, )
from crmonitor.monitor.rtamt_monitor_stl import OutputType
from crmonitor.predicates.rule import VisitorNode, parse_rule
import importlib.resources as pkg_resources

# def get_valid_time_interval(vehicles: List[Vehicle]):
#     start = max([v.start_time for v in vehicles])
#     end = min([v.end_time for v in vehicles])
#     return start, end

class RuleEvaluator:

    @classmethod
    def create_from_config(
        cls, world_state, ego_vehicle,
        rule: str = "R_G1",
        traffic_rules_config=None,
        use_boolean=False,
        output_type=OutputType.STANDARD,
    ):
        if traffic_rules_config is None:
            with pkg_resources.path(crmonitor, "traffic_rules_rtamt.yaml") as traffic_rules_path:
                traffic_rules_config = load_yaml(traffic_rules_path)
        rule_str_dict = traffic_rules_config["traffic_rules"]
        rule_set = parse_rule(rule_str_dict[rule], traffic_rules_config, name=rule)
        return cls(rule_set, ego_vehicle, world_state, use_boolean=use_boolean, output_type=output_type)

    def __init__(self, rule: VisitorNode, ego_vehicle: Vehicle, world: WorldState, start_time_step=0, use_boolean: bool = False,
                 output_type: OutputType = OutputType.STANDARD):
        visitor = MonitorCreationRuleTreeVisitor(world.dt, output_type)
        self._rule = rule
        self._monitor = rule.visit(visitor)
        self._last_time_step = start_time_step - 1
        self._collector_visitor = PredicateCollectorMonitorTreeVisitor()
        self._eval_visitor = EvaluationMonitorTreeVisitor(
            use_boolean=use_boolean, output_type=output_type
        )
        self._ego_vehicle = ego_vehicle
        self._world = world

    def get_predicates(self):
        predicate_values = dict(self._monitor.visit(self._collector_visitor))
        return predicate_values

    def update(self):
        """
        Advance the monitor state by one time step and return the corresponding rule evaluation value.

        :return: robustness or boolean rule value
        """
        self._last_time_step += 1
        if self._ego_vehicle.start_time > self._last_time_step or self._last_time_step > self._ego_vehicle.end_time:
            # Todo issue warning, return default
            return
        assert self._world.time_step == self._last_time_step, f"World time step not as expected! {self._world.time_step} != {self._last_time_step}"
        rule_value = self._eval_visitor.walk(self._monitor, self._world, self._ego_vehicle)
        return rule_value

    def evaluate(self) -> np.ndarray:
        """
        Evaluate the rule exhaustively until the final time step of the vehicle object is reached.

        Caution: This will change the time step of the world object!

        :return: Array of all rule values for all time steps of the vehicle's known trajectory
        """
        robustness_values = []
        for i in range(self._last_time_step + 1, self._ego_vehicle.end_time + 1):
            robustness_values.append(self.update())
            self._world.step()
        return np.array(robustness_values)

    @property
    def other_ids(self) -> Tuple[int]:
        return self._eval_visitor.other_ids[1:]

    def reset(self, ego_vehicle=None, other_vehicles=None, road_network=None):
        raise NotImplementedError



# class RuleSetEvaluator:
#     """
#     Evaluate individual vehicles of CommonRoad scenarios
#     """
#
#     @classmethod
#     def create_from_config(
#         cls,
#         rules: Union[str, Iterable[str]] = ("R_G1", "R_G2", "R_G3"),
#         traffic_rules_config=None,
#         dt=0.1,
#         use_boolean=False,
#         output_type=OutputType.STANDARD,
#     ):
#         if traffic_rules_config is None:
#             traffic_rules_config = YAML().load(
#                 Path(__file__).parent.parent / "traffic_rules_rtamt.yaml"
#             )
#         if isinstance(rules, str):
#             rules = [rules]
#         rule_str_dict = traffic_rules_config["traffic_rules"]
#         rule_set = [
#             parse_rule(rule_str_dict[r], traffic_rules_config, name=r) for r in rules
#         ]
#         return cls(rule_set, dt, use_boolean=use_boolean, output_type=output_type)
#
#     @classmethod
#     def create_from_rule_str(
#         cls,
#         rule_str: Union[str, Iterable[str], Dict[str, str]],
#         traffic_rules_config=None,
#         dt=0.1,
#         output_type=OutputType.STANDARD,
#     ):
#         if traffic_rules_config is None:
#             traffic_rules_config = YAML().load(
#                 Path(__file__).parent.parent / "traffic_rules_rtamt.yaml"
#             )
#         if isinstance(rule_str, str):
#             rule_str = {rule_str: rule_str}
#         if isinstance(rule_str, list):
#             rule_str = {r: r for r in rule_str}
#         rule_set = [
#             parse_rule(r, traffic_rules_config, name=n) for r, n in rule_str.items()
#         ]
#         return cls(rule_set, dt, output_type=output_type)
#
#     def __init__(
#         self, rules: Iterable, dt, use_boolean=False, output_type=OutputType.STANDARD
#     ):
#         """
#         :param rules: set of rules to be evaluated
#         """
#         self.rules = tuple(rules)
#         visitor = MonitorCreationRuleTreeVisitor(dt, output_type)
#         self.monitors = {rule: rule.visit(visitor) for rule in rules}
#         self._last_world_state = None
#         self._last_time_step = -1
#         self.use_boolean = use_boolean
#         self._collector_visitor = PredicateCollectorMonitorTreeVisitor()
#         self._eval_visitor = EvaluationMonitorTreeVisitor(
#             use_boolean=use_boolean, output_type=output_type
#         )
#
#     def reset_monitors(self):
#         self._last_time_step = -1
#         for monitor in self.monitors.values():
#             monitor.reset()
#
#     def _check_reset(self, world_state: WorldState):
#         if (
#             self._last_world_state is None
#             or (
#                 self._last_world_state is not world_state
#                 and (
#                     not hasattr(self._last_world_state, "scenario")
#                     or hasattr(self._last_world_state, "scenario")
#                     and (
#                         self._last_world_state.scenario.scenario_id
#                         != world_state.scenario.scenario_id
#                         or world_state.ego_vehicle.id
#                         != self._last_world_state.ego_vehicle.id
#                     )
#                 )
#             )
#             or world_state.time_step < self._last_time_step
#         ):
#             logging.debug("Clearing monitor states!")
#             self._last_world_state = world_state
#             self._last_time_step = -1
#             self.reset_monitors()
#
#     def evaluate_incremental(
#         self, world_state: WorldState, to_pandas=True
#     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
#         """
#         Evaluate world state for each time step since the last evaluation.
#         :param to_pandas: whether convert data to panda dataframe
#         :param world_state: world state to evaluate
#         :return: Tuple of pandas dataframes, where the first contains rule
#             robustness and the second predicate robustness
#         """
#         self._check_reset(world_state)
#         # Only evaluate if ego vehicle is present
#         time_begin = max(self._last_time_step + 1, world_state.ego_vehicle.start_time)
#         time_end = min(world_state.time_step, world_state.ego_vehicle.end_time)
#         self._last_time_step = world_state.time_step
#         # Create a flat copy of the world to iterate through time
#         world_state = copy.copy(world_state)
#         world_state.time_step = time_begin
#         # Nested dictionary with levels: time step and rule
#         rule_robustness = {}
#         # Nested dictionary with levels: time step, rule, predicate name
#         predicate_robustness = {}
#         other_ids_values = {}
#         while world_state.time_step <= time_end:
#             t = world_state.time_step
#             rule_robustness[t] = {}
#             predicate_robustness[t] = {}
#             other_ids_values[t] = {}
#             for rule in self.rules:
#                 rule_robustness[t][rule.name] = self._eval_visitor.walk(
#                     self.monitors[rule], world_state
#                 )
#                 predicate_robustness[t][rule.name] = dict(
#                     self.monitors[rule].visit(self._collector_visitor)
#                 )
#                 other_ids_values[t][rule.name] = self._eval_visitor.other_ids[1:]
#             world_state.step()
#
#         if to_pandas:
#             df_rule = pandas_from_nested_dict(
#                 rule_robustness, ["time_step", "rule_name", "robustness"]
#             )
#             df_pred = pandas_from_nested_dict(
#                 predicate_robustness,
#                 ["time_step", "rule_name", "full_name", "robustness"],
#             )
#             df_ids = pandas_from_nested_dict(
#                 other_ids_values, ["time_step", "rule_name", "other_ids"]
#             )
#             df_rule = df_rule.merge(df_ids, on=["time_step", "rule_name"])
#             return df_rule, df_pred
#         else:
#             return rule_robustness, predicate_robustness
