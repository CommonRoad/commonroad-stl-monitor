from collections import defaultdict

import numpy as np

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


class PredicateCollectorVisitor(Visitor):
    def _visit(self, node):
        r = []
        for c in node.children:
            r.extend(c.visit(self))
        return r

    def visit_rule_node(self, rule_node: RuleNode):
        return self._visit(rule_node)

    def visit_all_node(self, all_node: AllNode):
        if all_node.last_selected is None:
            val = all_node.children[0].visit(self)
            val = [(n, v if v is not None else 1.0) for n, v in val]
        else:
            val = all_node.last_selected.visit(self)
        return val

    def visit_exist_node(self, exist_node: ExistNode):
        if exist_node.last_selected is None:
            val = exist_node.children[0].visit(self)
            val = [(n, v if v is not None else -1.0) for n, v in val]
        else:
            val = exist_node.last_selected.visit(self)
        return val

    def visit_predicate_node(self, predicate_node: PredicateNode):
        return [(predicate_node.name, predicate_node.latest_value)]


class MonitorNode:
    def __init__(self, name, children=None):
        self.name = name
        self.children = children

    @classmethod
    def _copy_cls(cls, o):
        return cls(o.name, [c.copy() for c in o.children])

    def copy(self):
        return self._copy_cls(self)

    def reset(self):
        for c in self.children:
            c.reset()


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

    def reset(self):
        self.monitor.reset()


class AllMonitorNode(MonitorNode):
    def __init__(self, name, children):
        assert len(children) == 1
        super().__init__(name, children)
        self.monitors = defaultdict(children[0].copy)
        self.last_selected = None

    def visit(self, visitor, *ctx):
        return visitor.visit_all_node(self, *ctx)

    def reset(self):
        super().reset()
        self.last_selected = None
        self.monitors.clear()


class ExistMonitorNode(MonitorNode):
    def __init__(self, name, children):
        assert len(children) == 1
        super().__init__(name, children)
        self.monitors = defaultdict(children[0].copy)
        self.last_selected = None

    def visit(self, visitor, *ctx):
        return visitor.visit_exist_node(self, *ctx)

    def reset(self):
        super().reset()
        self.last_selected = None
        self.monitors.clear()


class CreateEvaluatorVisitor(Visitor):
    def __init__(self, dt):
        self.dt = dt

    def visit_rule_node(self, rule_node: RuleNode):
        children = [c.visit(self) for c in rule_node.children]
        monitor = RtamtStlMonitor.create_from_rule_node(rule_node, self.dt)
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
    def __init__(self, use_boolean=False):
        self.other_ids = tuple()
        self.use_boolean = use_boolean

    def walk(self, node: MonitorNode, world_state, *ctx):
        self.other_ids = tuple()
        return node.visit(self, world_state, (world_state.ego_vehicle.id,), *ctx)

    def visit_rule_node(self, rule_node: RuleMonitorNode, *ctx):
        world_state = ctx[0]
        # Collect child_values
        assert (
            rule_node.monitor.dt == world_state.dt
        ), f"Monitor constructed with dt={rule_node.monitor.dt} but got world state with dt={world_state.dt}!"
        child_values = {c.name: c.visit(self, *ctx) for c in rule_node.children}
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
        selected_ids = []
        for i in remaining_ids:
            ids = other_ids + (i,)
            val = node.monitors[i].visit(self, world_state, ids, *ctx)
            values.append(val)
            selected_ids.append(ids)
        return values, selected_ids

    def visit_all_node(self, all_node: AllMonitorNode, world_state, other_ids, *ctx):
        values, selected_ids = self._visit_quant_node(
            all_node, world_state, other_ids, *ctx
        )
        if len(values) > 0:
            idx = np.argmin(values)
            val = values[idx]
            self.other_ids = selected_ids[idx]
            all_node.last_selected = all_node.monitors[self.other_ids[-1]]
        else:
            val = 1.0
            self.other_ids = other_ids
            all_node.last_selected = None
        return val

    def visit_exist_node(
        self, exist_node: ExistMonitorNode, world_state, other_ids, *ctx
    ):
        values, selected_ids = self._visit_quant_node(
            exist_node, world_state, other_ids, *ctx
        )
        if len(values) > 0:
            idx = np.argmax(values)
            val = values[idx]
            self.other_ids = selected_ids[idx]
            exist_node.last_selected = exist_node.monitors[self.other_ids[-1]]
        else:
            val = -1.0
            self.other_ids = other_ids
            exist_node.last_selected = None
        return val

    def visit_predicate_node(
        self, predicate_node: PredicateNode, world_state, other_ids, *ctx
    ):
        predicate_ids = gather(other_ids, predicate_node.agent_placeholders)
        if self.use_boolean:
            value = predicate_node.evaluator.evaluate_boolean(
                world_state, predicate_ids
            )
            value = 1.0 if value else -1.0
            world_state.predicate_values[world_state.time_step][
                predicate_node.base_name
            ][tuple(predicate_ids)] = value
        else:
            value = predicate_node.evaluate_robustness(world_state, predicate_ids)
        return value
