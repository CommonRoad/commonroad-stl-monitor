from abc import abstractmethod, ABC
from typing import Union

import numpy as np

from crmonitor.common.helper import gather
from crmonitor.monitor.monitor_node import (
    MonitorNode,
    RuleMonitorNode,
    AllMonitorNode,
    ExistMonitorNode,
)
from crmonitor.monitor.rtamt_monitor_stl import RtamtStlMonitor
from crmonitor.predicates.rule import RuleNode, ExistNode, PredicateNode, AllNode


class RuleTreeVisitor(ABC):
    @abstractmethod
    def visit_rule_node(self, rule_node: Union[RuleNode, RuleMonitorNode], *ctx):
        pass

    @abstractmethod
    def visit_all_node(self, all_node: Union[AllNode, AllMonitorNode], *ctx):
        pass

    @abstractmethod
    def visit_exist_node(self, exist_node: Union[ExistNode, ExistMonitorNode], *ctx):
        pass

    @abstractmethod
    def visit_predicate_node(self, predicate_node: PredicateNode, *ctx):
        pass


class MonitorCreationRuleTreeVisitor(RuleTreeVisitor):
    def __init__(self, dt):
        self.dt = dt

    def visit_rule_node(self, rule_node: RuleNode, *ctx):
        children = [c.visit(self, *ctx) for c in rule_node.children]
        monitor = RtamtStlMonitor.create_from_rule_node(rule_node, self.dt)
        return RuleMonitorNode(rule_node.name, children, monitor)

    def visit_all_node(self, all_node: AllNode, *ctx):
        children = [c.visit(self, *ctx) for c in all_node.children]
        return AllMonitorNode(all_node.name, children)

    def visit_exist_node(self, exist_node: ExistNode, *ctx):
        children = [c.visit(self, *ctx) for c in exist_node.children]
        return ExistMonitorNode(exist_node.name, children)

    def visit_predicate_node(self, predicate_node: PredicateNode, *ctx):
        return predicate_node


class EvaluationMonitorTreeVisitor(RuleTreeVisitor):
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

    def _visit_quant_node(self, node, *ctx):
        world_state, other_ids = ctx[:2]
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
            val = node.monitors[i].visit(self, world_state, ids, *ctx[2:])
            values.append(val)
            selected_ids.append(ids)
        return values, selected_ids

    def visit_all_node(self, all_node: AllMonitorNode, *ctx):
        values, selected_ids = self._visit_quant_node(all_node, *ctx)
        other_ids = ctx[1]
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

    def visit_exist_node(self, exist_node: ExistMonitorNode, *ctx):
        values, selected_ids = self._visit_quant_node(exist_node, *ctx)
        other_ids = ctx[1]
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

    def visit_predicate_node(self, predicate_node: PredicateNode, *ctx):
        world_state, other_ids = ctx[:2]
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


class PredicateCollectorMonitorTreeVisitor(RuleTreeVisitor):
    def _visit(self, node):
        r = []
        for c in node.children:
            r.extend(c.visit(self))
        return r

    def visit_rule_node(self, rule_node: RuleMonitorNode, *ctx):
        return self._visit(rule_node)

    def visit_all_node(self, all_node: AllMonitorNode, *ctx):
        if all_node.last_selected is None:
            val = all_node.children[0].visit(self, *ctx)
            val = [(n, v if v is not None else 1.0) for n, v in val]
        else:
            val = all_node.last_selected.visit(self, *ctx)
        return val

    def visit_exist_node(self, exist_node: ExistMonitorNode, *ctx):
        if exist_node.last_selected is None:
            val = exist_node.children[0].visit(self, *ctx)
            val = [(n, v if v is not None else -1.0) for n, v in val]
        else:
            val = exist_node.last_selected.visit(self, *ctx)
        return val

    def visit_predicate_node(self, predicate_node: PredicateNode, *ctx):
        return [(predicate_node.name, predicate_node.latest_value)]
