from abc import ABC
from typing import TypeVar

from crmonitor.common import World
from crmonitor.evaluation.predicate_interface import PredicateEvaluationInterface
from crmonitor.monitor import MonitorVisitorInterface, OutputType, PredicateMonitorNode
from crmonitor.predicates import RobustnessScaler
from crmonitor.rule import IOType

_T = TypeVar("_T")


class BaseEvaluationMonitorTreeVisitor(MonitorVisitorInterface[_T], ABC):
    def __init__(
        self,
        predicate_evaluation_interface: PredicateEvaluationInterface,
        scale_rob: bool,
        use_boolean: bool = False,
        output_type: OutputType = OutputType.STANDARD,
    ):
        self._use_boolean = use_boolean
        self._output_type = output_type

        self._rob_scaler = RobustnessScaler(scale=scale_rob)
        self._predicate_interface = predicate_evaluation_interface

    def _do_evaluate_predicate(
        self, node: PredicateMonitorNode, world: World, time_step: int, vehicle_ids: tuple[int, ...]
    ) -> float:
        if self._should_use_boolean_predicate_evaluation(node):
            robustness_bool = self._predicate_interface.evaluate_boolean(
                node.predicate_name, world, time_step, vehicle_ids
            )
            robustness = self._rob_scaler.max if robustness_bool else self._rob_scaler.min
        else:
            robustness = self._predicate_interface.evaluate_robustness(
                node.predicate_name, world, time_step, vehicle_ids
            )
        return robustness

    def _should_use_boolean_predicate_evaluation(self, node: PredicateMonitorNode) -> bool:
        return self._use_boolean or (
            node.io_type == IOType.INPUT and self._output_type == OutputType.OUTPUT_ROBUSTNESS
        )
