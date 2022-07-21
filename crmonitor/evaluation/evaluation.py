import importlib.resources as pkg_resources
import logging
from functools import lru_cache
from typing import Tuple, Dict

import numpy as np

import crmonitor
from crmonitor.common.helper import load_yaml
from crmonitor.common.vehicle import Vehicle
from crmonitor.common.world import World
from crmonitor.evaluation.visitor import (
    MonitorCreationRuleTreeVisitor,
    EvaluationMonitorTreeVisitor,
    PredicateCollectorMonitorTreeVisitor,
    ResetMonitorTreeVisitor,
)
from crmonitor.monitor.rtamt_monitor_stl import OutputType
from crmonitor.predicates.rule import VisitorNode, parse_rule

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def get_traffic_rule_config():
    with pkg_resources.path(crmonitor, "traffic_rules_rtamt.yaml") as traffic_rules_path:
        traffic_rules_config = load_yaml(traffic_rules_path)
    return traffic_rules_config


class RuleEvaluator:
    @classmethod
    def create_from_config(
        cls,
        world: World = None,
        ego_vehicle: Vehicle = None,
        rule: str = "R_G1",
        traffic_rules_config=None,
        use_boolean: bool = False,
        output_type: OutputType = OutputType.STANDARD,
    ):
        if traffic_rules_config is None:
            traffic_rules_config = get_traffic_rule_config()
        rule_str_dict = traffic_rules_config["traffic_rules"]
        rule_set = parse_rule(rule_str_dict[rule], traffic_rules_config, name=rule)
        return cls(
            rule_set,
            ego_vehicle,
            world,
            use_boolean=use_boolean,
            output_type=output_type,
        )

    def __init__(
        self,
        rule: VisitorNode,
        ego_vehicle: Vehicle,
        world: World,
        start_time_step=None,
        use_boolean: bool = False,
        output_type: OutputType = OutputType.STANDARD,
    ):
        visitor = MonitorCreationRuleTreeVisitor(world.dt, output_type)
        self._rule = rule
        self._monitor = rule.visit(visitor)
        self._collector_visitor = PredicateCollectorMonitorTreeVisitor()
        self._eval_visitor = EvaluationMonitorTreeVisitor(
            use_boolean=use_boolean, output_type=output_type
        )
        self._last_evaluation_time_step = -1
        self._ego_vehicle = None
        self._world = None
        if ego_vehicle is not None:
            assert world is not None
            self.reset(ego_vehicle, world, start_time_step)

    @property
    def current_time(self) -> int:
        return self._last_evaluation_time_step

    def get_predicates(self) -> Dict[str, float]:
        predicate_values = dict(self._monitor.visit(self._collector_visitor))
        return predicate_values

    def update(self):
        """
        Advance the monitor state by one time step and return the corresponding rule evaluation value.

        :return: robustness or boolean rule value
        """
        self._last_evaluation_time_step += 1
        if (
            self._ego_vehicle.start_time > self._last_evaluation_time_step
            or self._last_evaluation_time_step > self._ego_vehicle.end_time
        ):
            logger.warning("Evaluating vehicle outside its lifetime!")
            return np.inf
        rule_value = self._eval_visitor.walk(self._monitor, self._world, self._last_evaluation_time_step, self._ego_vehicle)
        return rule_value

    def evaluate(self) -> np.ndarray:
        """
        Evaluate the rule exhaustively until the final time step of the vehicle object is reached.

        Caution: This will change the time step of the world object!

        :return: Array of all rule values for all time steps of the vehicle's known trajectory
        """
        robustness_values = []
        for i in range(
            self._last_evaluation_time_step + 1, self._ego_vehicle.end_time + 1
        ):
            robustness_values.append(self.update())
        return np.array(robustness_values)

    @property
    def other_ids(self) -> Tuple[int]:
        return self._eval_visitor.other_ids[1:]

    def reset(self, ego_vehicle: Vehicle, world: World, start_time_step=None):
        self._last_evaluation_time_step = (
            start_time_step - 1
            if start_time_step is not None
            else ego_vehicle.start_time - 1
        )
        self._ego_vehicle = ego_vehicle
        self._world = world
        # Reset monitor
        reset_visitor = ResetMonitorTreeVisitor()
        self._monitor.visit(reset_visitor)
