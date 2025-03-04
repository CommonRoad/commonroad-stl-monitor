import itertools
import math
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from functools import singledispatchmethod
from typing import Dict, List, Tuple, Union

import numpy as np
from commonroad.common.util import Interval as CommonRoadInterval
from commonroad.scenario.scenario import Scenario
from commonroad_mpr.common.observation import World as MprWorld
from rtamt.semantics.interval.interval import Interval as RtamtInterval

from crmonitor.common.helper import gather
from crmonitor.common.vehicle import Vehicle
from crmonitor.common.world import World
from crmonitor.monitor.monitor_node import (
    AllMonitorNode,
    AndSmoothMonitorNode,
    CompareToThresholdScaledMonitorNode,
    ExistMonitorNode,
    HistoricallyDurationMonitorNode,
    HistoricallyDurationSeverityMonitorNode,
    MonitorNode,
    MonitorVisitorInterface,
    PredicateMonitorNode,
    QuantMonitorNode,
    RuleMonitorNode,
    SumIfPositiveMonitorNode,
)
from crmonitor.monitor.rtamt_monitor_stl import OutputType, RtamtStlMonitor
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
    SumIfPositiveNode,
    VisitorNode,
)


class MonitorCreationRuleTreeVisitor(RuleTreeVisitorInterface[MonitorNode]):
    """
    This visitor is used to transform a rule tree to a monitor tree.
    """

    def __init__(self, dt: float, output_type: OutputType = OutputType.STANDARD):
        self.dt = dt
        self.output_type = output_type
        self._predicate_factory = PredicateFactory()

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


def _rtamt_interval_to_commonroad_interval(
    interval: RtamtInterval, scenario_context: Scenario
) -> CommonRoadInterval:
    """
    Convert a rtamt interval with units to a time step based interval in the context of the scenario.

    :param interval: A rtamt interval, with optional units.
    :param scenario_context: The scenario in which this interval should be valid.

    :returns: A CommonRoad interval in time steps, which is valid in regards to the scenario context.

    :raises RuntimeError: If an invalid combination of units is used.
    """
    if len(interval.begin_unit) == 0 and len(interval.end_unit) == 0:
        normalized_begin = int(interval.begin)
        normalized_end = int(interval.end)
    elif interval.begin_unit == "s" or interval.end_unit == "s":
        normalized_begin = interval.begin / scenario_context.dt
        normalized_end = interval.end / scenario_context.dt
    else:
        raise RuntimeError(
            f"Cannot convert rtamt interval: combination of time units '{interval.begin_unit}' and '{interval.end_unit}' is not supported! Use 's' for seconds, or omit for time steps."
        )

    begin = max(0, normalized_begin)

    return CommonRoadInterval(begin, normalized_end)


@dataclass
class OfflineEvaluationMonitorTreeVisitorContext:
    """
    Context for the `OfflineEvaluationMonitorTreeVisitor`. During the evaluation the context is passed down to each node.
    """

    world: World
    mpr_world: MprWorld
    max_time_step: int
    vehicles: Dict[int, Tuple[int, CommonRoadInterval]]
    """
    Optionally provide one other vehicle that should be considered for the evaluation of binary predicates. This field is populated during the evaluation by the quantifiers.
    """

    def with_new_vehicle(
        self, capture_id: int, other_vehicle: Tuple[int, CommonRoadInterval]
    ) -> "OfflineEvaluationMonitorTreeVisitorContext":
        new_vehicles = self.vehicles.copy()
        new_vehicles[capture_id] = other_vehicle
        return OfflineEvaluationMonitorTreeVisitorContext(
            self.world, self.mpr_world, self.max_time_step, new_vehicles
        )

    @property
    def vehicle_ids(self) -> List[int]:
        return [params[0] for params in self.vehicles.values()]


class OfflineEvaluationMonitorTreeVisitor(MonitorVisitorInterface[List[float]]):
    def __init__(self, use_boolean=False, output_type=OutputType.STANDARD) -> None:
        self.use_boolean = use_boolean
        self.output_type = output_type

        # TODO: when this visitor is integrated into crmonitor directly, this option should be read from the config
        self._rob_scaler = RobustnessScaler(scale=True)

    def walk(
        self,
        node: MonitorNode,
        world: World,
        mpr_world: MprWorld,
        max_time_step: int,
        ego_vehicle: Vehicle,
    ):
        vehicles = {0: (ego_vehicle.id, CommonRoadInterval(0, max_time_step))}
        ctx = OfflineEvaluationMonitorTreeVisitorContext(world, mpr_world, max_time_step, vehicles)
        return self.visit(node, ctx)

    @singledispatchmethod
    def visit(
        self, node: MonitorNode, ctx: OfflineEvaluationMonitorTreeVisitorContext
    ) -> List[float]:
        raise NotImplementedError

    @visit.register
    def _(
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
        node.record_values(scaled_sample_return)

        return scaled_sample_return

    @visit.register
    def _(
        self, node: AllMonitorNode, ctx: OfflineEvaluationMonitorTreeVisitorContext
    ) -> List[float]:
        # Mostly the same as visit_all_node of EvaluationMonitorTreeVisitor, except that it handles time series data (because of the offline evaluation)
        samples = self._visit_quant_node(node, ctx)
        robustness_values = []
        for values in samples:
            if len(values) > 0:
                idx = np.argmin(values)
                val = values[idx]

            else:
                val = self._rob_scaler.max

            robustness_values.append(val)

        scaled_robustness_values = np.clip(
            robustness_values, self._rob_scaler.min, self._rob_scaler.max
        )

        node.record_values(scaled_robustness_values)

        return scaled_robustness_values

    @visit.register
    def _(
        self, node: ExistMonitorNode, ctx: OfflineEvaluationMonitorTreeVisitorContext
    ) -> List[float]:
        samples = self._visit_quant_node(node, ctx)

        robustness_values = []
        for values in samples:
            if len(values) > 0:
                idx = np.argmax(values)
                val = values[idx]
            else:
                val = self._rob_scaler.min

            robustness_values.append(val)

        scaled_robustness_values = np.clip(
            robustness_values, self._rob_scaler.min, self._rob_scaler.max
        )

        node.record_values(scaled_robustness_values)

        return scaled_robustness_values

    @visit.register
    def _(
        self, node: AndSmoothMonitorNode, ctx: OfflineEvaluationMonitorTreeVisitorContext
    ) -> List[float]:
        samples_left = self.visit(node.left_child, ctx)
        samples_right = self.visit(node.right_child, ctx)

        samples_return = []
        for a, b in zip(samples_left, samples_right):
            k = 2.0 * 1e-6
            x = (b - a) / k
            g = 0.5 * (x + math.sqrt(x * x + 1.0))
            smin = b - k * g
            samples_return.append(smin)

        node.record_values(samples_return)

        return samples_return

    @visit.register
    def _(
        self, node: HistoricallyDurationMonitorNode, ctx: OfflineEvaluationMonitorTreeVisitorContext
    ) -> List[float]:
        samples = self.visit(node, ctx)
        if node.interval is not None:
            interval = _rtamt_interval_to_commonroad_interval(node.interval, ctx.world.scenario)
            begin = int(interval.start)
            end = min(ctx.max_time_step, int(interval.end))
        else:
            begin = 0
            end = ctx.max_time_step

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

        node.record_values(samples_return)

        return samples_return

    @visit.register
    def _(
        self,
        node: HistoricallyDurationSeverityMonitorNode,
        ctx: OfflineEvaluationMonitorTreeVisitorContext,
    ) -> List[float]:
        samples = self.visit(node.child, ctx)

        if node.interval is not None:
            interval = _rtamt_interval_to_commonroad_interval(node.interval, ctx.world.scenario)
            begin = int(interval.start)
            end = min(ctx.max_time_step, int(interval.end))
        else:
            begin = 0
            end = ctx.max_time_step

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

        node.record_values(samples_return)

        return samples_return

    @visit.register
    def _(
        self, node: SumIfPositiveMonitorNode, ctx: OfflineEvaluationMonitorTreeVisitorContext
    ) -> List[float]:
        samples = self._visit_quant_node(node, ctx)

        samples_return = []
        for values in samples:
            if len(values) > 0:
                val = sum([val for val in values if val > 0])
            else:
                val = float("nan")

            samples_return.append(val)

        node.record_values(samples_return)

        return samples_return

    @visit.register
    def _(
        self,
        node: CompareToThresholdScaledMonitorNode,
        ctx: OfflineEvaluationMonitorTreeVisitorContext,
    ) -> List[float]:
        samples = self.visit(node.child, ctx)
        samples_return = [
            1 - 2 * math.exp(-sample / node.threshold * math.log(2)) for sample in samples
        ]
        node.record_values(samples_return)
        return samples_return

    @visit.register
    def _(
        self, node: PredicateMonitorNode, ctx: OfflineEvaluationMonitorTreeVisitorContext
    ) -> List[float]:
        vehicle_ids = []
        start_time = 0
        end_time = ctx.max_time_step
        for agent_placeholder in node.agent_placeholders:
            vehicle_id, vehicle_interval = ctx.vehicles[agent_placeholder]
            vehicle_ids.append(vehicle_id)
            start_time = max(vehicle_interval.start, start_time)
            end_time = min(vehicle_interval.end, end_time)

        samples = []
        for time_step in range(0, ctx.max_time_step):
            # Only evaluate the predicate if the other vehicle is available in this time frame.
            if start_time > time_step or end_time < time_step:
                samples.append(self._rob_scaler.max)
                continue

            samples.append(
                node.evaluate_robustness(ctx.world, ctx.mpr_world, time_step, vehicle_ids)
            )

        node.record_values(samples)
        return samples

    def _visit_quant_node(
        self, node: QuantMonitorNode, ctx: OfflineEvaluationMonitorTreeVisitorContext
    ):
        """
        This method performs the quantification for the operators all and exist.
        Those operators use predicates, which correlate the ego vehicle with all other vehicles in the scenario.
        This method performs this correlation and evaluates each sub-monitor for the permutations of ego vehicle and other vehicles.
        """
        vehicle_start_times = {}
        vehicle_end_times = defaultdict(lambda: ctx.max_time_step)
        for time_step in range(0, ctx.max_time_step):
            all_ids = set(ctx.world.vehicle_ids_for_time_step(time_step))

            entered_vehicle_ids = all_ids.difference(vehicle_start_times.keys())
            left_vehicle_ids = (
                set(vehicle_start_times.keys())
                .difference(vehicle_end_times.keys())
                .difference(all_ids)
            )

            for entered_vehicle_id in entered_vehicle_ids:
                vehicle_start_times[entered_vehicle_id] = time_step

            for left_vehicle_id in left_vehicle_ids:
                vehicle_start_times[left_vehicle_id] = time_step - 1

        values = []
        ret_selected_ids = []
        for vehicle_id in ctx.world.vehicle_ids():
            if vehicle_id in ctx.vehicle_ids:
                continue

            vehicle_interval = CommonRoadInterval(
                vehicle_start_times[vehicle_id], vehicle_end_times[vehicle_id]
            )
            other_vehicle_params = (
                vehicle_id,
                (vehicle_start_times[vehicle_id], vehicle_end_times[vehicle_id]),
            )
            adjusted_ctx = ctx.with_new_vehicle(
                node.quantified_vehicle, (vehicle_id, vehicle_interval)
            )
            val = self.visit(node.monitors[tuple(adjusted_ctx.vehicle_ids)], adjusted_ctx)
            values.append(val)
            ret_selected_ids.append(vehicle_id)

        return list(zip(*values))


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
    def _(self, node: RuleMonitorNode, ctx: OnlineEvaluationMonitorTreeVisitorContext) -> float:
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
    def _(self, node: AllMonitorNode, ctx: OnlineEvaluationMonitorTreeVisitorContext) -> float:
        values, selected_ids = self._visit_quant_node(node, ctx)

        self.all_values_all_ids = {}  # reset to empty
        self.all_props_all_ids = {}  # reset to empty

        if len(values) > 0:
            idx = np.argmin(values)
            val = values[idx]
            self.other_ids = selected_ids[idx]
            node.last_selected = node.monitors[self.other_ids[-1]]

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
    def _(self, node: ExistMonitorNode, ctx: OnlineEvaluationMonitorTreeVisitorContext) -> float:
        values, selected_ids = self._visit_quant_node(node, ctx)

        self.all_values_all_ids = {}  # reset to empty
        self.all_props_all_ids = {}  # reset to empty

        if len(values) > 0:
            idx = np.argmax(values)
            val = values[idx]
            self.other_ids = selected_ids[idx]

            node.last_selected = node.monitors[self.other_ids[-1]]

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
    def _(
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


class BaseValueMonitorTreeVisitor(MonitorVisitorInterface[None], ABC):
    def visit_all_node(self, all_node: AllMonitorNode, *ctx):
        if all_node.last_selected is None:
            # Visit the prototype monitor
            val = all_node.children[0].visit(self, *ctx)
            val = [(n, v if v is not None else 1.0) for n, v in val]
        else:
            val = all_node.last_selected.visit(self, *ctx)
        return val

    def visit_exist_node(self, exist_node: ExistMonitorNode, *ctx):
        if exist_node.last_selected is None:
            # Visit the prototype monitor
            val = exist_node.children[0].visit(self, *ctx)
            val = [(n, v if v is not None else -1.0) for n, v in val]
        else:
            val = exist_node.last_selected.visit(self, *ctx)
        return val

    def visit_andsmooth_node(self, andsmooth_node: AndSmoothMonitorNode, *ctx):
        return [c.visit(self, *ctx) for c in andsmooth_node.children]

    def visit_historicallyduration_node(
        self, historicallyduration_node: HistoricallyDurationMonitorNode, *ctx
    ):
        return [c.visit(self, *ctx) for c in historicallyduration_node.children]

    def visit_historicallydurationseverity_node(
        self, historicallydurationseverity_node: HistoricallyDurationSeverityNode, *ctx
    ):
        return [c.visit(self, *ctx) for c in historicallydurationseverity_node.children]

    def visit_compare_to_threshold_scaled_node(
        self,
        compare_to_threshold_scaled_node: Union[
            CompareToThresholdScaledNode, CompareToThresholdScaledMonitorNode
        ],
        *ctx,
    ):
        return [c.visit(self, *ctx) for c in compare_to_threshold_scaled_node.children]

    def visit_sum_if_positive_node(
        self,
        sum_if_positive_node: Union[SumIfPositiveNode, SumIfPositiveMonitorNode],
        *ctx,
    ):
        if sum_if_positive_node.last_selected is None:
            # Visit the prototype monitor
            val = sum_if_positive_node.children[0].visit(self, *ctx)
            val = [(n, v if v is not None else -1.0) for n, v in val]
        else:
            val = sum_if_positive_node.last_selected.visit(self, *ctx)
        return val


class PredicateCollectorMonitorTreeVisitor(BaseValueMonitorTreeVisitor):
    def visit_rule_node(self, rule_node: "RuleMonitorNode", *ctx):
        r = []
        for c in rule_node.children:
            r.extend(c.visit(self))
        return r

    def visit_predicate_node(self, predicate_node: PredicateNode, *ctx):
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


class PredicateVisualizerMonitorTreeVisitor(MonitorVisitorInterface[None]):
    """
    Returns list of dictionaries, each dictionary mapping vehicle ids to a possibly
    nested dict of draw-parameters
    """

    def _split_context(self, ctx):
        idx = 6
        is_effective = ctx[idx] if len(ctx) > idx else True
        return ctx[:idx], is_effective

    def visit_rule_node(self, rule_node: RuleMonitorNode, *ctx):
        draw_functions_nested = [c.visit(self, *ctx) for c in rule_node.children]
        return list(itertools.chain(*draw_functions_nested))

    def _visit_quant_node(self, node, *ctx):
        ctx, is_effective_so_far = self._split_context(ctx)
        draw_functions_for_effective_node = []
        if node.last_selected is not None:
            draw_functions_for_effective_node = node.last_selected.visit(
                self, *ctx, True and is_effective_so_far
            )
        draw_functions_nested = [
            monitor.visit(self, *ctx, False)
            for i, monitor in node.monitors.items()
            if monitor != node.last_selected
        ]
        return list(itertools.chain(*draw_functions_nested)) + draw_functions_for_effective_node

    def visit_all_node(self, all_node: AllMonitorNode, *ctx):
        return self._visit_quant_node(all_node, *ctx)

    def visit_exist_node(self, exist_node: ExistMonitorNode, *ctx):
        return self._visit_quant_node(exist_node, *ctx)

    def visit_andsmooth_node(self, andsmooth_node: AndSmoothMonitorNode, *ctx):
        print("TODO implement")
        return self._visit_quant_node(andsmooth_node, *ctx)

    def visit_historicallyduration_node(
        self, historicallyduration_node: HistoricallyDurationMonitorNode, *ctx
    ):
        return self._visit_quant_node(historicallyduration_node, *ctx)

    def visit_historicallydurationseverity_node(
        self,
        historicallydurationseverity_node: HistoricallyDurationSeverityMonitorNode,
        *ctx,
    ):
        return self._visit_quant_node(historicallydurationseverity_node, *ctx)

    def visit_sum_if_positive_node(
        self,
        sum_if_positive_node: Union[SumIfPositiveNode, SumIfPositiveMonitorNode],
        *ctx,
    ):
        return self._visit_quant_node(sum_if_positive_node, *ctx)

    def visit_compare_to_threshold_scaled_node(
        self,
        compare_to_threshold_scaled_node: Union[
            CompareToThresholdScaledNode, CompareToThresholdScaledMonitorNode
        ],
        *ctx,
    ):
        return self._visit_quant_node(compare_to_threshold_scaled_node, *ctx)

    def visit_predicate_node(self, predicate_node: PredicateNode, *ctx):
        ctx, is_effective = self._split_context(ctx)

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
    @singledispatchmethod
    def visit(self, node: MonitorNode, *args, **kwargs) -> None:
        node.reset()

    @visit.register
    def _(self, node: QuantMonitorNode, *args, **kwargs) -> None:
        for monitor in node.monitors.values():
            self.visit(monitor)
        node.reset()
