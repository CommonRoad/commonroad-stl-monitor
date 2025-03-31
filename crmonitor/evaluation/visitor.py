import itertools
import math
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from functools import singledispatchmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from commonroad.common.util import Interval as CommonRoadInterval
from commonroad_mpr.common.observation import World as MprWorld

from crmonitor.common.helper import gather, rtamt_interval_to_commonroad_interval
from crmonitor.common.vehicle import Vehicle
from crmonitor.common.world import World
from crmonitor.monitor.monitor_node import (
    AllMonitorNode,
    BinaryMonitorNode,
    CompareToThresholdScaledMonitorNode,
    ConstantTraceMonitorNode,
    ExistMonitorNode,
    HistoricallyDurationMonitorNode,
    HistoricallyDurationSeverityMonitorNode,
    MonitorNode,
    MonitorVisitorInterface,
    PredicateMonitorNode,
    QuantMonitorNode,
    RuleMonitorNode,
    SigmoidMonitorNode,
    SumIfPositiveMonitorNode,
    UnaryMonitorNode,
)
from crmonitor.monitor.rtamt_monitor_stl import OutputType, RtamtStlMonitor
from crmonitor.predicates.base import PredicateEvaluatorConfig
from crmonitor.predicates.predicate_factory import PredicateFactory
from crmonitor.predicates.scaling import RobustnessScaler
from crmonitor.rule.rule_node import (
    AllNode,
    CompareToThresholdScaledNode,
    ExistNode,
    HistoricallyDurationNode,
    HistoricallyDurationSeverityNode,
    IOType,
    PredicateNode,
    RuleNode,
    RuleTreeVisitorInterface,
    SigmoidNode,
    SumIfPositiveNode,
    VisitorNode,
)


class MonitorCreationRuleTreeVisitor(RuleTreeVisitorInterface[MonitorNode]):
    """
    This visitor is used to transform a rule tree to a monitor tree.
    """

    def __init__(
        self,
        dt: float,
        output_type: OutputType = OutputType.STANDARD,
        predicate_evaluator_config: PredicateEvaluatorConfig = PredicateEvaluatorConfig(),
    ):
        self.dt = dt
        self.output_type = output_type
        self._predicate_factory = PredicateFactory(predicate_evaluator_config)

    @singledispatchmethod
    def visit(self, node: VisitorNode, *args, **kwargs) -> MonitorNode:
        raise NotImplementedError(
            f"Failed to create monitor for node '{node}': Transformation for this node is currently not implemented!"
        )

    @visit.register
    def _(self, node: RuleNode, *args, **kwargs) -> MonitorNode:
        children = [self.visit(child, *args, **kwargs) for child in node.children]
        monitor = RtamtStlMonitor.create_from_rule_node(node, self.dt, self.output_type)
        return RuleMonitorNode(node.name, children, monitor)

    @visit.register
    def _(self, node: AllNode, *args, **kwargs) -> MonitorNode:
        child_monitor = self.visit(node.child, *args, **kwargs)
        return AllMonitorNode(node.name, child_monitor, node.quantified_vehicle)

    @visit.register
    def _(self, node: ExistNode, *args, **kwargs) -> MonitorNode:
        child_monitor = self.visit(node.child, *args, **kwargs)
        return ExistMonitorNode(node.name, child_monitor, node.quantified_vehicle)

    @visit.register
    def _(self, node: SigmoidNode, *args, **kwargs) -> MonitorNode:
        child_monitor = self.visit(node.child, *args, **kwargs)
        return SigmoidMonitorNode(node.name, child_monitor)

    @visit.register
    def _(self, node: HistoricallyDurationNode, *args, **kwargs) -> MonitorNode:
        child_monitor = self.visit(node.child, *args, **kwargs)
        return HistoricallyDurationMonitorNode(node.name, child_monitor, node.interval)

    @visit.register
    def _(self, node: HistoricallyDurationSeverityNode, *args, **kwargs) -> MonitorNode:
        child_monitor = self.visit(node.child, *args, **kwargs)
        return HistoricallyDurationSeverityMonitorNode(node.name, child_monitor, node.interval)

    @visit.register
    def _(self, node: SumIfPositiveNode, *args, **kwargs) -> MonitorNode:
        child_monitor = self.visit(node.child, *args, **kwargs)
        return SumIfPositiveMonitorNode(node.name, child_monitor, node.quantified_vehicle)

    @visit.register
    def _(self, node: CompareToThresholdScaledNode, *args, **kwargs) -> MonitorNode:
        child_monitor = self.visit(node.child, *args, **kwargs)
        return CompareToThresholdScaledMonitorNode(node.name, child_monitor, node.threshold)

    @visit.register
    def _(self, node: PredicateNode, *args, **kwargs) -> MonitorNode:
        evaluator = self._predicate_factory.get_predicate(node.base_name)
        return PredicateMonitorNode(node.name, evaluator, node.agent_placeholders)


@dataclass
class OfflineEvaluationMonitorTreeVisitorContext:
    """
    Context for the `OfflineEvaluationMonitorTreeVisitor`. During the evaluation the context is passed down to each node.
    """

    world: World
    mpr_world: Optional[MprWorld]
    start_time_step: int
    final_time_step: int
    vehicles: Dict[int, Tuple[int, CommonRoadInterval]]
    """
    Optionally provide one other vehicle that should be considered for the evaluation of binary predicates. This field is populated during the evaluation by the quantifiers.
    """

    def with_new_vehicle(
        self, capture_id: int, other_vehicle: Tuple[int, CommonRoadInterval]
    ) -> "OfflineEvaluationMonitorTreeVisitorContext":
        """
        Update the context with a newly captured vehicle during quantification.

        :param capture_id: The Id of the placeholder.
        :param other_vehicle: A tuple with the vehicle Id and its availability time interval.
        """
        new_vehicles = self.vehicles.copy()
        new_vehicles[capture_id] = other_vehicle
        return OfflineEvaluationMonitorTreeVisitorContext(
            self.world, self.mpr_world, self.start_time_step, self.final_time_step, new_vehicles
        )

    @property
    def vehicle_ids(self) -> List[int]:
        return [params[0] for params in self.vehicles.values()]


class OfflineEvaluationMonitorTreeVisitor(MonitorVisitorInterface[List[float]]):
    def __init__(
        self, scale_rob: bool = True, use_boolean=False, output_type=OutputType.STANDARD
    ) -> None:
        self.use_boolean = use_boolean
        self.output_type = output_type

        self._rob_scaler = RobustnessScaler(scale=scale_rob)

    def walk(
        self,
        node: MonitorNode,
        world: World,
        max_time_step: int,
        ego_vehicle: Vehicle,
        mpr_world: Optional[MprWorld] = None,
    ):
        start_time_step = ego_vehicle.start_time
        vehicles = {0: (ego_vehicle.id, CommonRoadInterval(start_time_step, max_time_step))}
        ctx = OfflineEvaluationMonitorTreeVisitorContext(
            world, mpr_world, start_time_step, max_time_step, vehicles
        )
        return self.visit(node, ctx)

    @singledispatchmethod
    def visit(
        self, node: MonitorNode, ctx: OfflineEvaluationMonitorTreeVisitorContext
    ) -> List[float]:
        raise NotImplementedError

    @visit.register
    def visit_rule_node(
        self, node: RuleMonitorNode, ctx: OfflineEvaluationMonitorTreeVisitorContext
    ) -> List[float]:
        child_values = {child.name: self.visit(child, ctx) for child in node.children}

        sample_return = node.evaluate(list(child_values.items()))

        # When the rule is evaluated with IA-STL, some robustness values might be +inf.
        # This can lead to problems if the user expects scaled values.
        # Therefore, a simple clip is applied here, to make sure the robustness values
        # remain in the required robustness value interval.
        scaled_sample_return = list(
            np.clip(sample_return, self._rob_scaler.min, self._rob_scaler.max)
        )

        # Save the evaluation results
        node.values = scaled_sample_return

        return scaled_sample_return

    @visit.register
    def visit_all_node(
        self, node: AllMonitorNode, ctx: OfflineEvaluationMonitorTreeVisitorContext
    ) -> List[float]:
        # Mostly the same as visit_all_node of EvaluationMonitorTreeVisitor, except that it handles time series data (because of the offline evaluation)
        samples, selected_ids = self._visit_quant_node(node, ctx)
        robustness_values = []
        for values in samples:
            # Check if any non-nan value is present, because if not, np.nanargmin will fail.
            if len(values) > 0 and not np.all(np.isnan(values)):
                # Use np.nanargmin instead of np.argmin because the latter will select `nan` as the min.
                idx = np.nanargmin(values)
                val = values[idx]

                pivotal_monitor = node.monitors[selected_ids[idx]]
                node.last_selected = pivotal_monitor
            else:
                val = self._rob_scaler.max
                node.last_selected = None

            robustness_values.append(val)

        scaled_robustness_values = np.clip(
            robustness_values, self._rob_scaler.min, self._rob_scaler.max
        )

        node.values = scaled_robustness_values

        return list(scaled_robustness_values)

    @visit.register
    def visit_exist_node(
        self, node: ExistMonitorNode, ctx: OfflineEvaluationMonitorTreeVisitorContext
    ) -> List[float]:
        samples, selected_ids = self._visit_quant_node(node, ctx)

        robustness_values = []
        for values in samples:
            # Check if any non-nan value is present, because if not, np.nanargmax will fail.
            if len(values) > 0 and not np.all(np.isnan(values)):
                # Use np.nanargmax instead of np.argmax because the latter will select `nan` as the max.
                idx = np.nanargmax(values)
                val = values[idx]

                pivotal_monitor = node.monitors[selected_ids[idx]]
                node.last_selected = pivotal_monitor
            else:
                val = self._rob_scaler.min
                node.last_selected = None

            robustness_values.append(val)

        scaled_robustness_values = np.clip(
            robustness_values, self._rob_scaler.min, self._rob_scaler.max
        )

        node.values = scaled_robustness_values

        return list(scaled_robustness_values)

    @visit.register
    def visit_sigmoid_node(
        self, node: SigmoidMonitorNode, ctx: OfflineEvaluationMonitorTreeVisitorContext
    ) -> List[float]:
        samples = self.visit(node.child, ctx)

        scaling_param = 5
        samples_return = [
            (1 - math.exp(-scaling_param * sample)) / (1 + math.exp(-scaling_param * sample))
            for sample in samples
        ]
        node.values = samples_return

        return samples_return

    @visit.register
    def visit_historically_duration_node(
        self, node: HistoricallyDurationMonitorNode, ctx: OfflineEvaluationMonitorTreeVisitorContext
    ) -> List[float]:
        samples = self.visit(node.child, ctx)
        if node.interval is not None:
            interval = rtamt_interval_to_commonroad_interval(node.interval, ctx.world.scenario)
            begin = int(interval.start)
            end = min(ctx.final_time_step, int(interval.end))
        else:
            begin = 0
            end = ctx.final_time_step

        # Extend the samples, so that we can iterate with a static window size
        # and to make sure that the returned trace covers the interval [0, max_time_step].
        extended_samples = [self._rob_scaler.max for _ in range(end)] + samples
        samples_return = []
        for i in range(end, len(extended_samples)):
            # Iterate over the extended sample using a window of the size `(end - begin) + 1`.
            window = extended_samples[i - end : i - begin + 1]
            all_samples_are_ge_0 = all(x >= 0 for x in window)
            if all_samples_are_ge_0:
                samples_return.append(min(window))
            else:
                samples_less_0 = list(filter(lambda x: x < 0, window))
                samples_return.append(len(samples_less_0) / len(window))

        node.values = samples_return

        return samples_return

    @visit.register
    def visit_historically_duration_severity_node(
        self,
        node: HistoricallyDurationSeverityMonitorNode,
        ctx: OfflineEvaluationMonitorTreeVisitorContext,
    ) -> List[float]:
        samples = self.visit(node.child, ctx)

        if node.interval is not None:
            interval = rtamt_interval_to_commonroad_interval(node.interval, ctx.world.scenario)
            begin = int(interval.start)
            end = min(ctx.final_time_step, int(interval.end))
        else:
            begin = 0
            end = ctx.final_time_step

        # Extend the samples, so that we can iterate with a static window size
        # and to make sure that the returned trace covers the interval [0, max_time_step].
        extended_samples = [self._rob_scaler.max for _ in range(end)] + samples
        samples_return = []
        for i in range(end, len(extended_samples)):
            # Iterate over the extended sample using a window of the size `(end - begin) + 1`.
            window = extended_samples[i - end : i - begin + 1]
            all_samples_are_ge_0 = all(x >= 0 for x in window)
            if all_samples_are_ge_0:
                samples_return.append(min(window))
            else:
                samples_less_0 = list(filter(lambda x: x < 0, window))
                samples_return.append(sum(samples_less_0) / len(window))

        node.values = samples_return

        return samples_return

    @visit.register
    def visit_sum_if_positive_node(
        self, node: SumIfPositiveMonitorNode, ctx: OfflineEvaluationMonitorTreeVisitorContext
    ) -> List[float]:
        samples, _ = self._visit_quant_node(node, ctx)

        samples_return = []
        for values in samples:
            if len(values) > 0:
                val = sum([val for val in values if val > 0])
            else:
                val = float("nan")

            samples_return.append(val)

        node.values = samples_return

        return samples_return

    @visit.register
    def visit_compare_to_threshold_scaled_node(
        self,
        node: CompareToThresholdScaledMonitorNode,
        ctx: OfflineEvaluationMonitorTreeVisitorContext,
    ) -> List[float]:
        samples = self.visit(node.child, ctx)
        samples_return = [
            1 - 2 * math.exp(-sample / node.threshold * math.log(2)) for sample in samples
        ]
        node.values = samples_return
        return samples_return

    @visit.register
    def visit_predicate_node(
        self, node: PredicateMonitorNode, ctx: OfflineEvaluationMonitorTreeVisitorContext
    ) -> List[float]:
        vehicle_ids = []
        start_time = 0
        end_time = ctx.final_time_step
        for agent_placeholder in node.agent_placeholders:
            vehicle_id, vehicle_interval = ctx.vehicles[agent_placeholder]
            vehicle_ids.append(vehicle_id)
            start_time = max(vehicle_interval.start, start_time)
            end_time = min(vehicle_interval.end, end_time)

        samples = []
        for time_step in range(ctx.start_time_step, ctx.final_time_step):
            # Only evaluate the predicate if the other vehicle is available in this time frame.
            if start_time > time_step or end_time < time_step:
                samples.append(float("nan"))
                continue

            samples.append(
                node.evaluate_robustness(ctx.world, ctx.mpr_world, time_step, vehicle_ids)
            )

        node.values = samples
        return samples

    @visit.register
    def visit_constant_trace_node(
        self, node: ConstantTraceMonitorNode, ctx: OfflineEvaluationMonitorTreeVisitorContext
    ) -> List[float]:
        return node.trace

    def _visit_quant_node(
        self, node: QuantMonitorNode, ctx: OfflineEvaluationMonitorTreeVisitorContext
    ) -> Tuple[List[List[float]], List[int]]:
        """
        Performs the quantification of vehicles for quant operators.

        Those operators use predicates, which correlate the ego vehicle with all other vehicles in the scenario.
        This method performs this correlation and evaluates each sub-monitor for the permutations of ego vehicle and other vehicles.
        """
        # Track when each vehicle first appears (enters) and when it is no longer present (leaves).
        # This is necessary to define the active time intervals for each vehicle in the scenario.
        # Otherwise we run into problems, when predicates are evaluated for vehicles which are not available at the evaluated time steps.
        vehicle_start_times = {}
        vehicle_end_times = defaultdict(lambda: ctx.final_time_step)
        for time_step in range(ctx.start_time_step, ctx.final_time_step):
            all_ids = set(ctx.world.vehicle_ids_for_time_step(time_step))

            # Identify vehicles entering the scene at this timestep.
            entered_vehicle_ids = all_ids.difference(vehicle_start_times.keys())

            # Identify vehicles that have left: they were present but are now gone.
            left_vehicle_ids = (
                set(vehicle_start_times.keys())
                .difference(vehicle_end_times.keys())
                .difference(all_ids)
            )

            for entered_vehicle_id in entered_vehicle_ids:
                vehicle_start_times[entered_vehicle_id] = time_step

            # Record the last timestep vehicles were present before disappearing.
            # This analysis happens in retrospective, because we only know that a vehicle left if it left in the previous time step.
            for left_vehicle_id in left_vehicle_ids:
                vehicle_end_times[left_vehicle_id] = time_step - 1

        # Iterate over all vehicles to evaluate the quantified sub-monitors.
        # Skip vehicles already included in the current context (to avoid duplication).
        values = []
        ret_selected_ids = []
        for vehicle_id in vehicle_start_times.keys():
            if vehicle_id in ctx.vehicle_ids:
                continue

            # Define the active time interval for this vehicle.
            vehicle_interval = CommonRoadInterval(
                vehicle_start_times[vehicle_id], vehicle_end_times[vehicle_id]
            )
            # Prepare updated context that binds the current vehicle to the quantifier placeholder.
            adjusted_ctx = ctx.with_new_vehicle(
                node.quantified_vehicle, (vehicle_id, vehicle_interval)
            )
            # Retrieve and evaluate the sub-monitor corresponding to this specific vehicle set.
            # As the quantification needs to evaluate the child of the quantifier node for all other vehicles we cannot plainly evaluate the child, as this would mess up value recording.
            # Intead new monitors are implicitly created for the currently selected vehicle ids and evaluated.
            # This basically creates a new sub-monitor tree for each selection of vehicle ids.
            val = self.visit(node.monitors[vehicle_id], adjusted_ctx)
            values.append(val)
            ret_selected_ids.append(vehicle_id)

        # values is a list of lists with time step ordered samples for each predicate.
        # This transforms values into a time step ordered list of list of samples, where each list of samples contains the values for each predicate at this time step.
        return list(zip(*values)), ret_selected_ids


@dataclass
class OnlineEvaluationMonitorTreeVisitorContext:
    world: World
    mpr_world: MprWorld
    time_step: int
    other_ids: Tuple[int, ...]


class EvaluationMonitorTreeVisitor(MonitorVisitorInterface[float]):
    def __init__(self, use_boolean=False, output_type=OutputType.STANDARD):
        self.other_ids = tuple()
        self.use_boolean = use_boolean
        self.output_type = output_type
        self.all_values_all_ids = {}
        self.all_props_all_ids = {}

    def walk(self, node: MonitorNode, world, mpr_world, time_step, ego_vehicle, *ctx):
        self.other_ids = tuple()
        vehicle_ids = (ego_vehicle.vehicle_id,)
        ctx = OnlineEvaluationMonitorTreeVisitorContext(world, mpr_world, time_step, vehicle_ids)
        return self.visit(node, ctx)

    @singledispatchmethod
    def visit(self, node: MonitorNode, ctx: OnlineEvaluationMonitorTreeVisitorContext) -> float:
        raise NotImplementedError(
            f"The monitor '{node}' is not supported in the online evaluation!"
        )

    @visit.register
    def visit_rule(
        self, node: RuleMonitorNode, ctx: OnlineEvaluationMonitorTreeVisitorContext
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
        remaining_ids = tuple(set(all_ids).difference(ctx.other_ids))
        values = []
        selected_ids = []
        for i in remaining_ids:
            ids = ctx.other_ids + (i,)
            val = self.visit(node.monitors[i], ctx)
            values.append(val)
            selected_ids.append(ids)
        return values, selected_ids

    @visit.register
    def visit_all_node(
        self, node: AllMonitorNode, ctx: OnlineEvaluationMonitorTreeVisitorContext
    ) -> float:
        values, selected_ids = self._visit_quant_node(node, ctx)

        self.all_values_all_ids = {}  # reset to empty
        self.all_props_all_ids = {}  # reset to empty

        if len(values) > 0:
            idx = np.argmin(values)
            val = values[idx]
            self.other_ids = selected_ids[idx]

            pivotal_monitor = node.monitors[selected_ids[idx][-1]]
            node.last_selected = pivotal_monitor

            # Loop through all selected_ids and populate the dictionary
            for i, sid in enumerate(selected_ids):
                self.all_values_all_ids[sid[-1]] = values[i]
                if hasattr(node.monitors[sid[-1]].monitor, "_propositions"):
                    self.all_props_all_ids[sid[-1]] = node.monitors[sid[-1]].monitor._propositions
        else:
            val = 1.0
            self.other_ids = ctx.other_ids
            node.last_selected = None

        return val

    @visit.register
    def visit_exist_node(
        self, node: ExistMonitorNode, ctx: OnlineEvaluationMonitorTreeVisitorContext
    ) -> float:
        values, selected_ids = self._visit_quant_node(node, ctx)

        self.all_values_all_ids = {}  # reset to empty
        self.all_props_all_ids = {}  # reset to empty

        if len(values) > 0:
            idx = np.argmax(values)
            val = values[idx]
            self.other_ids = selected_ids[idx]

            pivotal_monitor = node.monitors[selected_ids[idx][-1]]
            node.last_selected = pivotal_monitor

            # Loop through all selected_ids and populate the dictionary
            for i, sid in enumerate(selected_ids):
                self.all_values_all_ids[sid[-1]] = values[i]
        else:
            val = -1.0
            self.other_ids = ctx.other_ids
            node.last_selected = None

            # If no values, only add other_ids if it's not empty
            if ctx.other_ids:
                self.all_values_all_ids[ctx.other_ids[-1]] = val
        return val

    @visit.register
    def visit_predicate_node(
        self, node: PredicateMonitorNode, ctx: OnlineEvaluationMonitorTreeVisitorContext
    ) -> float:
        predicate_ids = gather(ctx.other_ids, node.agent_placeholders)
        if (
            self.use_boolean
            or node.io_type == IOType.INPUT
            and self.output_type == OutputType.OUTPUT_ROBUSTNESS
        ):
            value = node.evaluate_boolean(ctx.world, ctx.time_step, predicate_ids)
            value = 1.0 if value else -1.0
        else:
            value = node.evaluate_robustness(ctx.world, ctx.mpr_world, ctx.time_step, predicate_ids)
        return value


class BaseValueMonitorTreeVisitor(MonitorVisitorInterface[List[Tuple[str, float]]], ABC):
    """
    Collects the values of all leaf nodes in a monitor tree. Can be subclassed to specify which values should be collected.
    """

    @singledispatchmethod
    def visit(self, node: MonitorNode, *args, **kwargs) -> List[Tuple[str, float]]:
        raise NotImplementedError

    @visit.register
    def visit_all_node(self, node: AllMonitorNode, *args, **kwargs) -> List[Tuple[str, float]]:
        if node.last_selected is None:
            # Visit the prototype monitor
            val = self.visit(node.child, *args, **kwargs)
            val = [(n, v if v is not None else 1.0) for n, v in val]
        else:
            val = self.visit(node.last_selected, *args, **kwargs)
        return val

    @visit.register
    def visit_exist_node(self, node: ExistMonitorNode, *args, **kwargs) -> List[Tuple[str, float]]:
        if node.last_selected is None:
            # Visit the prototype monitor
            val = self.visit(node.child, *args, **kwargs)
            val = [(n, v if v is not None else -1.0) for n, v in val]
        else:
            val = self.visit(node.last_selected, *args, **kwargs)
        return val

    @visit.register
    def visit_unary_node(self, node: UnaryMonitorNode, *args, **kwargs) -> List[Tuple[str, float]]:
        return self.visit(node, *args, **kwargs)

    @visit.register
    def visit_binary_node(
        self, node: BinaryMonitorNode, *args, **kwargs
    ) -> List[Tuple[str, float]]:
        left_values = self.visit(node.left_child, *args, **kwargs)
        right_values = self.visit(node.right_child, *args, **kwargs)

        return left_values + right_values


class PredicateCollectorMonitorTreeVisitor(BaseValueMonitorTreeVisitor):
    @BaseValueMonitorTreeVisitor.visit.register
    def visit_rule_node(self, rule_node: RuleMonitorNode, *args, **kwargs):
        r = []
        for c in rule_node.children:
            r.extend(self.visit(c))
        return r

    @BaseValueMonitorTreeVisitor.visit.register
    def visit_predicate_node(self, predicate_node: PredicateMonitorNode, *args, **kwargs):
        return [(predicate_node.name, predicate_node.latest_value)]


class MPRGradientCollectorMonitorTreeVisitor(PredicateCollectorMonitorTreeVisitor):
    def visit_predicate_node(self, predicate_node: PredicateNode, *ctx):
        return [(predicate_node.name, predicate_node.mpr_gradient)]


class AstNodeValueCollectorMonitorTreeVisitor(BaseValueMonitorTreeVisitor):
    @staticmethod
    def visit_rule_node(rule_node: "RuleMonitorNode", *ctx):
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
    def visit_rule_node(self, rule_node: RuleMonitorNode, *args, **kwargs):
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
    def visit_binary_node(self, node: BinaryMonitorNode, *args, **kwargs):
        left_draw_params = self.visit(node.left_child, *args, **kwargs)
        right_draw_params = self.visit(node.right_child, *args, **kwargs)
        return left_draw_params + right_draw_params

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

    @singledispatchmethod
    def visit(self, node: MonitorNode, *args, **kwargs) -> None:
        node.reset()

    @visit.register
    def _(self, node: QuantMonitorNode, *args, **kwargs) -> None:
        for monitor in node.monitors.values():
            self.visit(monitor, *args, **kwargs)
        node.reset()


class MonitorToStringVisitor(MonitorVisitorInterface[str]):
    """
    Visitor to convert a monitor tree to a human readable string representation. The resulting string should be very similar to the original rule.
    """

    def to_string(self, node: MonitorNode, vehicle_ids: Optional[Dict[int, int]] = None) -> str:
        """
        Serialize a monitor node tree as a string.

        :param node: The root node of the monitor tree that should be serialized. Can either be the canonical root node, or also intermediate node.
        :param vehicle_ids: Optionally provide a lookup table to resolve vehicle quantifier placeholders (e.g. a0, a1) to vehicle ids from a scenario.

        :returns: The serialized rule.
        """
        return self.visit(node, vehicle_ids)

    @singledispatchmethod
    def visit(self, node: MonitorNode, vehicle_ids: Optional[Dict[int, int]] = None) -> str:
        return str(node)

    @visit.register
    def _(self, node: RuleMonitorNode, vehicle_ids: Optional[Dict[int, int]] = None) -> str:
        label = node.monitor._rule
        for child in node.children:
            # Sub-Rules are represent by their placeholders (child.name) in the rule.
            # To mimic the original rule, we replace the placeholders with the rule of the sub-rules.
            child_label = self.visit(child, vehicle_ids)
            if child.name in label:
                label = label.replace(child.name, child_label)
        return label

    @visit.register
    def _(self, node: PredicateMonitorNode, vehicle_ids: Optional[Dict[int, int]] = None) -> str:
        if vehicle_ids is not None:
            return node.format_with_vehicle_ids(vehicle_ids)
        else:
            return str(node)

    @visit.register
    def _(self, node: UnaryMonitorNode, vehicle_ids: Optional[Dict[int, int]] = None) -> str:
        child_label = self.visit(node.child, vehicle_ids)
        return f"{str(node)} ({child_label})"


class VariableCollectionVisitor(MonitorVisitorInterface[Dict[str, MonitorNode]]):
    """
    Visitor to map node names (variables in rtamt rules) to the respective nodes.
    This is usefull to lookup which node belongs to which variable when processing RTAMT ASTs.
    """

    def collect_variables(self, node: MonitorNode) -> Dict[str, MonitorNode]:
        return self.visit(node, {})

    @singledispatchmethod
    def visit(self, node: MonitorNode, state: Dict[str, MonitorNode]) -> Dict[str, MonitorNode]:
        state[node.name] = node
        return state

    @visit.register
    def _(self, node: UnaryMonitorNode, state: Dict[str, MonitorNode]) -> Dict[str, MonitorNode]:
        self.visit(node.child, state)
        state[node.name] = node
        return state

    @visit.register
    def _(self, node: RuleMonitorNode, state: Dict[str, MonitorNode]) -> Dict[str, MonitorNode]:
        [self.visit(child, state) for child in node.children]
        state[node.name] = node
        return state
