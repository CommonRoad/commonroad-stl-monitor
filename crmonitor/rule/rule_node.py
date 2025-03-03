import copy
from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import Optional

from rtamt.semantics.interval.interval import Interval

from crmonitor.monitor.monitor_node import MonitorNode


class IOType(Enum):
    OUTPUT = "output"
    INPUT = "input"


class VisitorNode(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def visit(self, visitor, *ctx):
        pass


class RuleNode(VisitorNode):
    def __init__(self, children, rule_str, name):
        self.children = children
        self.name = name
        self.rule_str = rule_str

    def visit(self, visitor, *ctx):
        return visitor.visit_rule_node(self, *ctx)


class QuantNode(VisitorNode):
    def __init__(self, children, quantified_vehicle, name):
        self.children = children
        self.name = name
        self.quantified_vehicle = quantified_vehicle


class AllNode(QuantNode):
    def visit(self, visitor, *ctx):
        return visitor.visit_all_node(self, *ctx)


class ExistNode(QuantNode):
    def __init__(self, children, quantified_vehicle, name):
        self.children = children
        self.name = name
        self.quantified_vehicle = quantified_vehicle

    def visit(self, visitor, *ctx):
        return visitor.visit_exist_node(self, *ctx)


class AndsmoothNode(VisitorNode):
    def __init__(self, children, name):
        self.children = children
        self.name = name

    def visit(self, visitor, *ctx):
        return visitor.visit_andsmooth_node(self, *ctx)


class HistoricallyDurationNode(VisitorNode):
    def __init__(self, children, name: str, interval: Optional[Interval]):
        self.children = children
        self.name = name
        self.interval = interval

    def visit(self, visitor, *ctx):
        return visitor.visit_historicallyduration_node(self, *ctx)


class HistoricallyDurationSeverityNode(VisitorNode):
    def __init__(self, children, name: str, interval: Optional[Interval]):
        self.children = children
        self.name = name
        self.interval = interval

    def visit(self, visitor, *ctx):
        return visitor.visit_historicallydurationseverity_node(self, *ctx)


class SumIfPositiveNode(QuantNode):
    def visit(self, visitor, *ctx):
        return visitor.visit_sum_if_positive_node(self, *ctx)


class CompareToThresholdScaledNode(VisitorNode):
    def __init__(self, children, name, threshold: float):
        self.children = children
        self.name = name
        self.threshold = threshold

    def visit(self, visitor, *ctx):
        return visitor.visit_compare_to_threshold_scaled_node(self, *ctx)


class PredicateNode(MonitorNode, VisitorNode):
    def __init__(self, full_name, agent_placeholders, evaluator, io_type=IOType.OUTPUT):
        assert len(agent_placeholders) == evaluator.arity, (
            f"The arity of the evaluator for {full_name} should be "
            f"{len(agent_placeholders)}, but is {evaluator.arity}!"
        )
        super().__init__(full_name)
        self.agent_placeholders = tuple(agent_placeholders)
        self.evaluator = evaluator
        self.io_type = io_type
        self.latest_value = None
        self.latest_vehicle_ids = None
        self.mpr_gradient = None

    def evaluate_boolean(self, world, time_step, vehicle_ids):
        value = self.evaluator.evaluate_boolean(world, time_step, vehicle_ids)
        self.latest_value = 1.0 if value else -1.0
        self.latest_vehicle_ids = tuple(vehicle_ids)
        return value

    def evaluate_robustness(self, world, mpr_world, time_step, vehicle_ids):
        value = self.evaluator.evaluate_robustness_with_cache(
            world, mpr_world, time_step, vehicle_ids
        )
        self.latest_value = value
        self.latest_vehicle_ids = tuple(vehicle_ids)
        if mpr_world is not None:
            self.mpr_gradient = self.evaluator.gradient_mpr()
        return value

    def visit(self, visitor, *ctx):
        return visitor.visit_predicate_node(self, *ctx)

    @property
    def base_name(self):
        return self.evaluator.predicate_name

    @property
    def num_dependencies(self):
        return len(self.agent_placeholders)

    def __eq__(self, o) -> bool:
        return self.name == o.name and self.agent_placeholders == o.agent_placeholders

    def __hash__(self) -> int:
        return hash((self.name, self.agent_placeholders))

    def copy(self):
        return copy.copy(self)

    def reset(self):
        self.latest_value = None
        self.latest_vehicle_ids = None
