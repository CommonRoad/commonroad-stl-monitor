rom collections import defaultdict

import numpy as np

# from crmonitor.common.evaluation import bool_to_norm_rob
from crmonitor.common.helper import gather
from crmonitor.monitor.rtamt_monitor_stl import RtamtStlMonitor
from crmonitor.predicates.rule import RuleNode, ExistNode, PredicateNode, AllNode

from collections import defaultdict

import numpy as np

# from crmonitor.common.evaluation import bool_to_norm_rob
from crmonitor.common.helper import gather
from crmonitor.monitor.rtamt_monitor_stl import RtamtStlMonitor
from crmonitor.predicates.rule import RuleNode, ExistNode, PredicateNode, \
    AllNode


class Visitor:
    def visit(self, node):
        return node.visit(self)

    def visit_rule_node(self, rule_node: RuleNode):
        pass

    def visit_all_node(self, all_node: AllNode):
        pass

    def visit_exist_node(self, exist_node: ExistNode):
        pass

    def visit_predicate_node(self, predicate_node: PredicateNode):
        pass


class MonitorNode:
    def __init__(self, name, children=None):
        self.name = name
        self.children = children

    @classmethod
    def _copy_cls(cls, o):
        return cls(o.name, [c.copy() for c in o.children])

    def copy(self):
        return self._copy_cls(self)


class RuleMonitorNode(MonitorNode):
    def __init__(self, name, children, monitor):
        super().__init__(name, children)
        self.monitor = monitor

    def visit(self, visitor, *ctx):
        return visitor.visit_rule_node(self, *ctx)

    def evaluate_incremental(self, time, values):
        return self.monitor.evaluate_monitor_online(time, values)

    def copy(self):
        return RuleMonitorNode(
            self.name, [c.copy() for c in self.children], self.monitor.copy()
        )


class AllMonitorNode(MonitorNode):
    def __init__(self, name, children):
        assert len(children) == 1
        super().__init__(name, children)
        self.monitors = defaultdict(children[0].copy)

    def visit(self, visitor, *ctx):
        return visitor.visit_all_node(self, *ctx)


class ExistMonitorNode(MonitorNode):
    def __init__(self, name, children):
        assert len(children) == 1
        super().__init__(name, children)
        self.monitors = defaultdict(children[0].copy)

    def visit(self, visitor, *ctx):
        return visitor.visit_exist_node(self, *ctx)


class PredicateMonitorNode(MonitorNode):
    def visit(self, visitor, *ctx):
        return visitor.visit_predicate_node(self, *ctx)


class CreateEvaluatorVisitor(Visitor):
    def visit_rule_node(self, rule_node: RuleNode):
        children = [c.visit(self) for c in rule_node.children]
        monitor = RtamtStlMonitor.create_from_rule_node(rule_node)
        return RuleMonitorNode(rule_node.name, children, monitor)

    def visit_all_node(self, all_node: AllNode):
        children = [c.visit(self) for c in all_node.children]
        return AllMonitorNode(all_node.name, children)

    def visit_exist_node(self, exist_node: ExistNode):
        children = [c.visit(self) for c in exist_node.children]
        return ExistMonitorNode(exist_node.name, children)

    def visit_predicate_node(self, predicate_node: PredicateNode):
        return predicate_node


class EvaluationVisitor:
    def visit_rule_node(self, rule_node: RuleMonitorNode, *ctx):
        # Collect child_values
        child_values = {c.name: c.visit(self, *ctx) for c in rule_node.children}
        world_state = ctx[0]
        val = rule_node.evaluate_incremental(
            world_state.time_step, list(child_values.items())
        )
        return val

    def _visit_quant_node(self, node, world_state, other_ids, *ctx):
        all_ids = set(
            [
                v.id
                for v in world_state.other_vehicles
                if v.is_valid(world_state.time_step)
            ]
            + [world_state.ego_vehicle.id]
        )
        remaining_ids = tuple(all_ids.difference(other_ids))
        values = []
        for i in remaining_ids:
            ids = other_ids + (i,)
            val = node.monitors[i].visit(self, world_state, ids, *ctx)
            values.append(val)
        return values

    def visit_all_node(self, all_node: AllMonitorNode, world_state, other_ids, *ctx):
        values = self._visit_quant_node(all_node, world_state, other_ids, *ctx)
        idx = np.argmin(values)
        val = values[idx]
        return val

    def visit_exist_node(
        self, exist_node: ExistMonitorNode, world_state, other_ids, *ctx
    ):
        values = self._visit_quant_node(exist_node, world_state, other_ids, *ctx)
        idx = np.argmax(values)
        val = values[idx]
        return val

    def visit_predicate_node(
        self, predicate_node: PredicateNode, world_state, other_ids, use_boolean=False
    ):
        predicate_ids = gather(other_ids, predicate_node.agent_placeholders)
        if use_boolean:
            value = predicate_node.evaluator.evaluate_boolean(
                world_state, predicate_ids
            )
            value = 1.0 if value else -1.0
            world_state.predicate_values[world_state.time_step][
                predicate_node.base_name
            ][tuple(predicate_ids)] = value
        else:
            value = predicate_node.evaluator.evaluate_robustness_with_cache(
                world_state, predicate_ids
            )
        return value
