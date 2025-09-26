from dataclasses import dataclass
from functools import singledispatchmethod

import numpy as np

from crmonitor.common.vehicle import Vehicle
from crmonitor.common.world import World
from crmonitor.monitor.monitor_node import (
    AllMonitorNode,
    ExistMonitorNode,
    MonitorNode,
    PredicateMonitorNode,
    QuantMonitorNode,
    RtamtRuleMonitorNode,
)

from ._base import BaseEvaluationMonitorTreeVisitor


@dataclass
class OnlineEvaluationMonitorTreeVisitorContext:
    world: World
    time_step: int
    captured_agents: dict[int, int]

    def capture_agent(
        self, capture_id: int, agent_id: int
    ) -> "OnlineEvaluationMonitorTreeVisitorContext":
        """
        Update the context with a newly captured vehicle during quantification.

        :param capture_id: The Id of the placeholder.
        :param other_vehicle: A tuple with the vehicle Id and its availability time interval.
        """
        new_captured_agents = self.captured_agents.copy()
        new_captured_agents[capture_id] = agent_id
        return OnlineEvaluationMonitorTreeVisitorContext(
            self.world,
            self.time_step,
            new_captured_agents,
        )


class OnlineEvaluationMonitorTreeVisitor(BaseEvaluationMonitorTreeVisitor[float]):
    def update(
        self,
        node: MonitorNode,
        world: World,
        time_step: int,
        ego_vehicle: Vehicle,
    ):
        # TODO: Default ego agent always has ID 0.
        captured_agents = {0: ego_vehicle.id}
        ctx = OnlineEvaluationMonitorTreeVisitorContext(world, time_step, captured_agents)

        return self.visit(node, ctx)

    @singledispatchmethod
    def visit(self, node: MonitorNode, ctx: OnlineEvaluationMonitorTreeVisitorContext) -> float:
        raise NotImplementedError(
            f"The monitor node '{node}' is not supported in the online evaluation!"
        )

    @visit.register
    def visit_rule(
        self, node: RtamtRuleMonitorNode, ctx: OnlineEvaluationMonitorTreeVisitorContext
    ) -> float:
        # Collect child_values
        assert node.monitor.dt == ctx.world.dt, (
            f"Monitor constructed with dt="
            f"{node.monitor.dt} but got "
            f"world state with dt={ctx.world.dt}!"
        )
        child_values = {c.name: self.visit(c, ctx) for c in node.children}
        val = node.update(ctx.time_step, list(child_values.items()))
        return val

    def _visit_quant_node(
        self, node: QuantMonitorNode, ctx: OnlineEvaluationMonitorTreeVisitorContext
    ):
        all_ids = ctx.world.vehicle_ids_for_time_step(ctx.time_step)
        remaining_ids = tuple(set(all_ids).difference(ctx.captured_agents.values()))

        values = []
        ret_selected_ids = []
        for vehicle_id in remaining_ids:
            adjusted_ctx = ctx.capture_agent(node.quantified_agent, vehicle_id)
            val = self.visit(node.monitors[vehicle_id], adjusted_ctx)
            values.append(val)
            ret_selected_ids.append(vehicle_id)
        return values, ret_selected_ids

    @visit.register
    def visit_all_node(
        self, node: AllMonitorNode, ctx: OnlineEvaluationMonitorTreeVisitorContext
    ) -> float:
        values, selected_ids = self._visit_quant_node(node, ctx)

        if len(values) > 0:
            idx = np.argmin(values)
            val = values[idx]

            pivotal_monitor = node.monitors[selected_ids[idx][-1]]
            node.last_selected = pivotal_monitor
        else:
            val = self._rob_scaler.max
            node.last_selected = None

        node.last_value = val

        return val

    @visit.register
    def visit_exist_node(
        self, node: ExistMonitorNode, ctx: OnlineEvaluationMonitorTreeVisitorContext
    ) -> float:
        values, selected_ids = self._visit_quant_node(node, ctx)

        if len(values) > 0:
            idx = np.argmax(values)
            val = values[idx]

            pivotal_monitor = node.monitors[selected_ids[idx]]
            node.last_selected = pivotal_monitor
        else:
            val = self._rob_scaler.min
            node.last_selected = None

        node.last_value = val

        return val

    @visit.register
    def visit_predicate_node(
        self, node: PredicateMonitorNode, ctx: OnlineEvaluationMonitorTreeVisitorContext
    ) -> float:
        vehicle_ids = []
        for agent_placeholder in node.agent_placeholders:
            vehicle_id = ctx.captured_agents[agent_placeholder]
            vehicle_ids.append(vehicle_id)

        value = self._do_evaluate_predicate(node, ctx.world, ctx.time_step, tuple(vehicle_ids))

        node.last_value = value

        return value
