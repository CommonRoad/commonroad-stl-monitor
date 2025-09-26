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
    _monitor: MonitorNode

    @classmethod
    def create_for_rule(
        cls,
        rule_name: str,
        dt: float,
        output_type: OutputType = OutputType.STANDARD,
        predicate_interface_config: PredicateEvaluationInterfaceConfig = PredicateEvaluationInterfaceConfig(),
    ) -> Self:
        """Create a new rule evaluator for a pre-defined given traffic rule (e.g. R_G1, R_I2, etc.).

        :param rule_name: The name of the traffic rule which this rule evaluator will evaluate.
        :param dt: Time step size the input scenarios have. Required for the sampling frequency of RTAMT.
        :param output_type: Switch between 'normal' STL and IA-STL.
        :param predicate_interface_config: Adjust how predicates in the traffic rules are evaluated.
        """
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
        """Create a new rule evaluator for a custom traffic rule.

        :param rule_str: Custom traffic rule.
        :param dt: Time step size the input scenarios have. Required for the sampling frequency of RTAMT.
        :param rule_name: Optionally provide the name of rule, which will be used for debugging.
        :param output_type: Switch between 'normal' STL and IA-STL.
        :param predicate_interface_config: Adjust how predicates in the traffic rules are evaluated.
        """
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

    @property
    def monitor(self) -> MonitorNode:
        """The root node of the STL monitor tree."""
        return self._monitor

    @property
    def dt(self) -> float:
        return self._dt

    @abstractmethod
    def evaluate(
        self, world: World, ego_id: int, start_time: int | None = None, end_time: int | None = None
    ) -> list[float]: ...

    def reset(self) -> None:
        """Resets the evaluator so that it can be reused to evaluate other scenarios.

        This will also clear all cached predicate values, MPR GP gradients and invalidate all relevant caches.
        """
        reset_visitor = ResetMonitorTreeVisitor()
        reset_visitor.reset(self.monitor)

    def visualize(self) -> None:
        """Visualizes the result of the evaluation."""
        ctrl = VisualizationController()
        ctrl.visualize(self.monitor)

    def get_predicate_values(self) -> dict[str, float]:
        """Retrive the last value of each predicate."""
        predicate_collector = PredicateValueCollectorMonitorTreeVisitor()
        return predicate_collector.collect_predicate_values(self.monitor)

    def get_predicate_names(self) -> list[str]:
        """Retrive a list of the predicates in the traffic rule."""
        return PredicateNameCollectionMonitorTreeVisitor().collect_predicate_names(self.monitor)

    def get_rule_str(self) -> str:
        return MonitorToStringVisitor().to_string(self.monitor)


class OfflineRuleEvaluator(RuleEvaluatorInterface):
    """
    Stateless rule evaluator, which evaluates traffic rules in offline mode.
    """

    def __init__(
        self,
        rule: RuleAstNode,
        dt: float,
        output_type: OutputType = OutputType.STANDARD,
        predicate_interface_config: PredicateEvaluationInterfaceConfig = PredicateEvaluationInterfaceConfig(),
    ) -> None:
        super().__init__(rule, dt, output_type, predicate_interface_config)

        monitor_creation_visitor = MonitorCreationRuleTreeVisitor()
        self._monitor = monitor_creation_visitor.visit(
            self._rule, self._dt, output_type, online=False
        )

    @override
    def evaluate(
        self, world: World, ego_id: int, start_time: int | None = None, end_time: int | None = None
    ) -> list[float]:
        """Evaluate the traffic rule for `world` and `ego_id` in offline mode.

        :param world: The world in which `ego_id` can be found. The time step size of the world must match the time step size of the rule evaluator.
        :param ego_id: Ego vehicle for which the traffic rule is evaluated.
        :param start_time: Optionally provide a start time step, after which the rule is evaluated. If `None` is given, the start time of the ego vehicle is used.
        :param end_time: Optionally provide an end time step, until which the rule is evaluated. If `None` is given, the end time of the ego vehicle is used.

        :returns: The robustness trace.
        """
        if world.dt != self.dt:
            raise ValueError(
                f"The configured dt '{self.dt}' for this rule evaluator does not match the dt of the world '{world.dt}'"
            )

        ego_vehicle = world.vehicle_by_id(ego_id)
        if ego_vehicle is None:
            raise ValueError()

        if start_time is None:
            start_time = ego_vehicle.start_time

        if end_time is None:
            end_time = ego_vehicle.end_time

        # Create the evaluation interface for the predicates in the traffic rule.
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
    """Stateful traffic rule evaluator, which evaluates traffic rules in online mode."""

    def __init__(
        self,
        rule: RuleAstNode,
        dt: float,
        output_type: OutputType = OutputType.STANDARD,
        predicate_interface_config: PredicateEvaluationInterfaceConfig = PredicateEvaluationInterfaceConfig(),
    ) -> None:
        super().__init__(rule, dt, output_type, predicate_interface_config)

        monitor_creation_visitor = MonitorCreationRuleTreeVisitor()
        self._monitor = monitor_creation_visitor.visit(
            self._rule, self._dt, output_type, online=True
        )

        # Create the evaluation interface for the predicates in the traffic rule.
        self._predicate_evaluation_interface = PredicateEvaluationInterface(
            self.get_predicate_names(), self._predicate_interface_config
        )
        self._eval_visitor = OnlineEvaluationMonitorTreeVisitor(
            self._predicate_evaluation_interface,
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

    @override
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

        self._predicate_evaluation_interface.reset()
