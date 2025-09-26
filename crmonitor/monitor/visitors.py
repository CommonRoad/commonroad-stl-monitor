import itertools
from functools import singledispatchmethod
from typing import Any, Optional

from crmonitor.rule import (
    PredicateNode,
)

from .monitor_node import (
    AllMonitorNode,
    ExistMonitorNode,
    MonitorNode,
    MonitorVisitorInterface,
    PredicateMonitorNode,
    QuantMonitorNode,
    RtamtRuleMonitorNode,
    UnaryMonitorNode,
)


class BaseValueMonitorTreeVisitor(MonitorVisitorInterface[list[tuple[str, float]]]):
    """
    Collects the values of all leaf nodes in a monitor tree. Can be subclassed to specify which values should be collected.
    """

    @singledispatchmethod
    def visit(self, node: MonitorNode, *args, **kwargs) -> list[tuple[str, float]]:
        return []

    @visit.register
    def visit_all_node(self, node: AllMonitorNode, *args, **kwargs) -> list[tuple[str, float]]:
        if node.last_selected is None:
            # Visit the prototype monitor
            val = self.visit(node.child, *args, **kwargs)
            val = [(n, v if v is not None else 1.0) for n, v in val]
        else:
            val = self.visit(node.last_selected, *args, **kwargs)
        return val

    @visit.register
    def visit_exist_node(self, node: ExistMonitorNode, *args, **kwargs) -> list[tuple[str, float]]:
        if node.last_selected is None:
            # Visit the prototype monitor
            val = self.visit(node.child, *args, **kwargs)
            val = [(n, v if v is not None else -1.0) for n, v in val]
        else:
            val = self.visit(node.last_selected, *args, **kwargs)
        return val

    @visit.register
    def visit_unary_node(self, node: UnaryMonitorNode, *args, **kwargs) -> list[tuple[str, float]]:
        return self.visit(node.child, *args, **kwargs)

    @visit.register
    def visit_rule_node(
        self, rule_node: RtamtRuleMonitorNode, *args, **kwargs
    ) -> list[tuple[str, float]]:
        r = []
        for c in rule_node.children:
            r.extend(self.visit(c))
        return r


class PredicateNameCollectionMonitorTreeVisitor(MonitorVisitorInterface[list[str]]):
    def collect_predicate_names(self, node: MonitorNode) -> list[str]:
        return self.visit(node)

    @singledispatchmethod
    def visit(self, node: MonitorNode, *args, **kwargs) -> list[str]:
        return []

    @visit.register
    def visit_unary_node(self, node: UnaryMonitorNode, *args, **kwargs) -> list[str]:
        return self.visit(node.child, *args, **kwargs)

    @visit.register
    def visit_rule_node(self, rule_node: RtamtRuleMonitorNode, *args, **kwargs) -> list[str]:
        r = []
        for c in rule_node.children:
            r.extend(self.visit(c))
        return r

    @visit.register
    def visit_predicate_node(self, node: PredicateMonitorNode, *args, **kwargs) -> list[str]:
        return [node.predicate_name]


class PredicateValueCollectorMonitorTreeVisitor(BaseValueMonitorTreeVisitor):
    def collect_predicate_values(self, root_node: MonitorNode) -> dict[str, float]:
        predicate_value_list = self.visit(root_node)
        return dict(predicate_value_list)

    @BaseValueMonitorTreeVisitor.visit.register
    def visit_predicate_node(
        self, predicate_node: PredicateMonitorNode, *args, **kwargs
    ) -> list[tuple[str, float]]:
        return [(predicate_node.name, predicate_node.last_value)]


class MPRGradientCollectorMonitorTreeVisitor(PredicateValueCollectorMonitorTreeVisitor):
    def collect_mpr_gradient(self, root_node: MonitorNode) -> dict[str, float]:
        return self.visit(root_node)

    def visit_predicate_node(self, predicate_node: PredicateNode, *ctx):
        return [(predicate_node.name, predicate_node.mpr_gradient)]


class AstNodeValueCollectorMonitorTreeVisitor(BaseValueMonitorTreeVisitor):
    @staticmethod
    def visit_rule_node(rule_node: "RtamtRuleMonitorNode", *ctx):
        return list(rule_node.monitor.ast_node_values.items())

    def visit_predicate_node(self, predicate_node: PredicateNode, *ctx):
        raise NotImplementedError()


class PredicateVisualizerMonitorTreeVisitor(MonitorVisitorInterface[Any]):
    """
    Returns list of dictionaries, each dictionary mapping vehicle ids to a possibly
    nested dict of draw-parameters
    """

    @singledispatchmethod
    def visit(self, node: MonitorNode, *args, **kwargs):
        raise NotImplementedError

    def _split_context(self, ctx):
        idx = 6
        is_effective = ctx[idx] if len(ctx) > idx else True
        return ctx[:idx], is_effective

    @visit.register
    def visit_rule_node(self, rule_node: RtamtRuleMonitorNode, *args, **kwargs):
        draw_functions_nested = [self.visit(c, *args, **kwargs) for c in rule_node.children]
        return list(itertools.chain(*draw_functions_nested))

    @visit.register
    def visit_quant_node(self, node: QuantMonitorNode, *args, **kwargs):
        ctx, is_effective_so_far = self._split_context(args)
        draw_functions_for_effective_node = []
        if node.last_selected is not None:
            draw_functions_for_effective_node = self.visit(
                node.last_selected, *ctx, True and is_effective_so_far
            )
        draw_functions_nested = [
            self.visit(monitor, *ctx, False)
            for i, monitor in node.monitors.items()
            if monitor != node.last_selected
        ]
        return list(itertools.chain(*draw_functions_nested)) + draw_functions_for_effective_node

    @visit.register
    def visit_unary_node(self, node: UnaryMonitorNode, *args, **kwargs):
        return self.visit(node.child, *args, **kwargs)

    @visit.register
    def visit_predicate_node(self, predicate_node: PredicateMonitorNode, *args, **kwargs):
        ctx, is_effective = self._split_context(args)

        (
            add_vehicle_draw_params,
            predicate_names2vehicle_ids2values,
            predicate_name2predicate_evaluator,
            world,
            time_step,
            visualization_config,
        ) = ctx

        pred_name = predicate_node.evaluator.predicate_name
        latest_vehicle_ids = predicate_node.latest_vehicle_ids

        config_entry_key = pred_name if pred_name in visualization_config else "default"
        config_obj = visualization_config.get(config_entry_key, {})
        show_non_effective_predicate_instances_for_vehicles = config_obj.get(
            "show_non_effective_predicate_instances_for_vehicles", []
        )

        if (
            not is_effective
            and latest_vehicle_ids not in show_non_effective_predicate_instances_for_vehicles
        ):
            return ()

        predicate_name2predicate_evaluator[pred_name] = predicate_node.evaluator

        return predicate_node.evaluator.visualize(
            latest_vehicle_ids,
            add_vehicle_draw_params,
            world,
            time_step,
            predicate_names2vehicle_ids2values,
        )


class ResetMonitorTreeVisitor(MonitorVisitorInterface[None]):
    """
    Visitor to reset the monitor tree.
    """

    def reset(self, node: MonitorNode) -> None:
        self.visit(node)

    @singledispatchmethod
    def visit(self, node: MonitorNode, *args, **kwargs) -> None:
        node.reset()


class MonitorToStringVisitor(MonitorVisitorInterface[str]):
    """
    Visitor to convert a monitor tree to a human readable string representation. The resulting string should be very similar to the original rule.
    """

    def to_string(self, node: MonitorNode, vehicle_ids: Optional[dict[int, int]] = None) -> str:
        """
        Serialize a monitor node tree as a string.

        :param node: The root node of the monitor tree that should be serialized. Can either be the canonical root node, or also intermediate node.
        :param vehicle_ids: Optionally provide a lookup table to resolve vehicle quantifier placeholders (e.g. a0, a1) to vehicle ids from a scenario.

        :returns: The serialized rule.
        """
        return self.visit(node, vehicle_ids)

    @singledispatchmethod
    def visit(self, node: MonitorNode, vehicle_ids: Optional[dict[int, int]] = None) -> str:
        return str(node)

    @visit.register
    def _(self, node: RtamtRuleMonitorNode, vehicle_ids: Optional[dict[int, int]] = None) -> str:
        label = node.monitor._rule
        for child in node.children:
            # Sub-Rules are represent by their placeholders (child.name) in the rule.
            # To mimic the original rule, we replace the placeholders with the rule of the sub-rules.
            child_label = self.visit(child, vehicle_ids)
            if child.name in label:
                label = label.replace(child.name, child_label)
        return label

    @visit.register
    def _(self, node: PredicateMonitorNode, vehicle_ids: Optional[dict[int, int]] = None) -> str:
        if vehicle_ids is not None:
            return node.format_with_vehicle_ids(vehicle_ids)
        else:
            return str(node)

    @visit.register
    def _(self, node: UnaryMonitorNode, vehicle_ids: Optional[dict[int, int]] = None) -> str:
        child_label = self.visit(node.child, vehicle_ids)
        return f"{str(node)} ({child_label})"


class VariableCollectionVisitor(MonitorVisitorInterface[dict[str, MonitorNode]]):
    """
    Visitor to map node names (variables in rtamt rules) to the respective nodes.
    This is usefull to lookup which node belongs to which variable when processing RTAMT ASTs.
    """

    def collect_variables(self, node: MonitorNode) -> dict[str, MonitorNode]:
        return self.visit(node, {})

    @singledispatchmethod
    def visit(self, node: MonitorNode, state: dict[str, MonitorNode]) -> dict[str, MonitorNode]:
        state[node.name] = node
        return state

    @visit.register
    def _(self, node: UnaryMonitorNode, state: dict[str, MonitorNode]) -> dict[str, MonitorNode]:
        self.visit(node.child, state)
        state[node.name] = node
        return state

    @visit.register
    def _(
        self, node: RtamtRuleMonitorNode, state: dict[str, MonitorNode]
    ) -> dict[str, MonitorNode]:
        [self.visit(child, state) for child in node.children]
        state[node.name] = node
        return state
