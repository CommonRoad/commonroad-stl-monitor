"""
Module for the public evaluation interface. The classes in this module can be used to evaluate traffic rules.
"""

import logging
from abc import ABC, abstractmethod

import numpy as np
from typing_extensions import Self, override

from crmonitor.common import World
from crmonitor.common.config import (
    get_traffic_rule_from_config,
)
from crmonitor.monitor import (
    MonitorCreationRuleTreeVisitor,
    MonitorNode,
    OutputType,
    PredicateValueCollectorMonitorTreeVisitor,
    ResetMonitorTreeVisitor,
)
from crmonitor.monitor.visitors import (
    MonitorToStringVisitor,
    PredicateNameCollectionMonitorTreeVisitor,
)
from crmonitor.rule import RuleAstNode, RuleParser
from crmonitor.visualization import (
    VisualizationController,
)

from .predicate_interface import (
    PredicateEvaluationInterface,
    PredicateEvaluationInterfaceConfig,
    PredicateEvaluationMode,
)
from .visitors import (
    OfflineEvaluationMonitorTreeVisitor,
    OnlineEvaluationMonitorTreeVisitor,
)

_LOGGER = logging.getLogger(__name__)


class RuleEvaluatorInterface(ABC):
    @classmethod
    def create_for_rule(
        cls,
        rule_name: str,
        dt: float,
        output_type: OutputType = OutputType.STANDARD,
        predicate_interface_config: PredicateEvaluationInterfaceConfig = PredicateEvaluationInterfaceConfig(),
    ) -> Self:
        rule_str = get_traffic_rule_from_config(rule_name)
        if rule_str is None:
            _LOGGER.debug(
                f"Rule {rule_name} is not a known rule identifier. Interpreting it as the rule definition."
            )
            rule_str = rule_name

        return cls.create_for_rule_str(
            rule_str, dt, rule_name, output_type, predicate_interface_config
        )

    @classmethod
    def create_for_rule_str(
        cls,
        rule_str: str,
        dt: float,
        rule_name: str | None = None,
        output_type: OutputType = OutputType.STANDARD,
        predicate_interface_config: PredicateEvaluationInterfaceConfig = PredicateEvaluationInterfaceConfig(),
    ) -> Self:
        rule_node = RuleParser().parse(rule_str, name=rule_name)

        return cls(rule_node, dt, output_type, predicate_interface_config)

    def __init__(
        self,
        rule: RuleAstNode,
        dt: float,
        output_type: OutputType = OutputType.STANDARD,
        predicate_interface_config: PredicateEvaluationInterfaceConfig = PredicateEvaluationInterfaceConfig(),
    ) -> None:
        self._rule = rule
        self._predicate_interface_config = predicate_interface_config
        self._dt = dt

        monitor_creation_visitor = MonitorCreationRuleTreeVisitor()
        self._monitor = monitor_creation_visitor.visit(self._rule, self._dt, output_type)

    @property
    def monitor(self) -> MonitorNode:
        return self._monitor

    @property
    def dt(self) -> float:
        return self._dt

    @abstractmethod
    def evaluate(
        self, world: World, ego_id: int, start_time: int | None = None, end_time: int | None = None
    ) -> list[float]: ...

    def reset(self) -> None:
        """ """
        reset_visitor = ResetMonitorTreeVisitor()
        reset_visitor.reset(self.monitor)

    def visualize(self) -> None:
        """Visualize the result of the evaluation."""
        ctrl = VisualizationController()
        ctrl.visualize(self.monitor)

    def get_predicate_values(self) -> dict[str, float]:
        predicate_collector = PredicateValueCollectorMonitorTreeVisitor()
        return predicate_collector.collect_predicate_values(self.monitor)

    def get_predicate_names(self) -> list[str]:
        return PredicateNameCollectionMonitorTreeVisitor().collect_predicate_names(self.monitor)

    def get_rule_str(self) -> str:
        return MonitorToStringVisitor().to_string(self.monitor)


class OfflineRuleEvaluator(RuleEvaluatorInterface):
    def evaluate(
        self, world: World, ego_id: int, start_time: int | None = None, end_time: int | None = None
    ) -> list[float]:
        if world.dt != self.dt:
            raise ValueError()

        ego_vehicle = world.vehicle_by_id(ego_id)
        if ego_vehicle is None:
            raise ValueError()

        if start_time is None:
            start_time = ego_vehicle.start_time

        if end_time is None:
            end_time = ego_vehicle.end_time

        predicate_interface = PredicateEvaluationInterface(
            self.get_predicate_names(), self._predicate_interface_config
        )

        eval_visitor = OfflineEvaluationMonitorTreeVisitor(
            predicate_interface,
            self._predicate_interface_config.base.scale_rob,
        )
        return eval_visitor.evaluate(
            self._monitor,
            world,
            ego_vehicle,
            start_time,
            end_time,
        )


class OnlineRuleEvaluator(RuleEvaluatorInterface):
    def __init__(
        self,
        rule: RuleAstNode,
        dt: float,
        output_type: OutputType = OutputType.STANDARD,
        predicate_interface_config: PredicateEvaluationInterfaceConfig = PredicateEvaluationInterfaceConfig(),
    ) -> None:
        super().__init__(rule, dt, output_type, predicate_interface_config)
        self._eval_visitor = OnlineEvaluationMonitorTreeVisitor(
            self._predicate_interface_config.base.scale_rob,
            use_boolean=self._predicate_interface_config.mode == PredicateEvaluationMode.BOOLEAN,
            output_type=output_type,
        )
        self._last_evaluation_time_step = -1
        self._rule_value_course = []

    @property
    def last_evaluation_time_step(self) -> int:
        return self._last_evaluation_time_step

    @property
    def rule_value_course(self) -> list[float]:
        return self._rule_value_course

    def evaluate(
        self, world: World, ego_id: int, start_time: int | None = None, end_time: int | None = None
    ) -> list[float]:
        if world.dt != self.dt:
            raise ValueError()

        ego_vehicle = world.vehicle_by_id(ego_id)
        if ego_vehicle is None:
            raise ValueError()

        if start_time is None:
            start_time = self._last_evaluation_time_step + 1

        if end_time is None:
            end_time = ego_vehicle.end_time

        robustness_values = []
        for _ in range(start_time, end_time):
            robustness_values.append(self.update(world, ego_id))
        return robustness_values

    def update(self, world: World, ego_id: int) -> float:
        ego_vehicle = world.vehicle_by_id(ego_id)
        if ego_vehicle is None:
            raise ValueError()

        self._last_evaluation_time_step += 1
        if (
            ego_vehicle.start_time > self._last_evaluation_time_step
            or self._last_evaluation_time_step > ego_vehicle.end_time
        ):
            _LOGGER.warning("Evaluating vehicle %s outside its lifetime!", ego_id)
            return np.inf

        rule_value = self._eval_visitor.update(
            self._monitor,
            world,
            self._last_evaluation_time_step,
            ego_vehicle,
        )

        # TODO: Shouldn't the scaling be handled by the evaluation visitor?
        rule_value = rule_value if np.isfinite(rule_value) else np.sign(rule_value) * 1.0
        self._rule_value_course.append((self._last_evaluation_time_step, rule_value))
        return rule_value

    @override
    def reset(self) -> None:
        super().reset()
        self._last_evaluation_time_step = -1
        self._rule_value_course = []
