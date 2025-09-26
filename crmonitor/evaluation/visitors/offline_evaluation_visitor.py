import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from functools import singledispatchmethod

import numpy as np
from commonroad.common.util import Interval as CommonRoadInterval

from crmonitor.common.helper import rtamt_interval_to_commonroad_interval
from crmonitor.common.vehicle import Vehicle
from crmonitor.common.world import World
from crmonitor.monitor.monitor_node import (
    AllMonitorNode,
    CompareToThresholdScaledMonitorNode,
    ConstantTraceMonitorNode,
    ExistMonitorNode,
    ExistsMultipleMonitorNode,
    HistoricallyDurationMonitorNode,
    HistoricallyDurationSeverityMonitorNode,
    MonitorNode,
    PredicateMonitorNode,
    QuantMonitorNode,
    RtamtRuleMonitorNode,
    SigmoidMonitorNode,
    SumIfPositiveMonitorNode,
)
from crmonitor.monitor.rtamt_monitor_stl import OutputType
from crmonitor.rule.rule_node import IOType

from ._base import BaseEvaluationMonitorTreeVisitor

_LOGGER = logging.getLogger(__name__)


@dataclass
class OfflineEvaluationMonitorTreeVisitorContext:
    """
    Context for the `OfflineEvaluationMonitorTreeVisitor`. During the evaluation the context is passed down to each node.
    """

    world: World
    start_time_step: int
    final_time_step: int
    captured_agents: dict[int, tuple[int, CommonRoadInterval]]
    """
    Optionally provide one other vehicle that should be considered for the evaluation of binary predicates. This field is populated during the evaluation by the quantifiers.
    """

    def capture_agent(
        self, capture_id: int, agent_info: tuple[int, CommonRoadInterval]
    ) -> "OfflineEvaluationMonitorTreeVisitorContext":
        """
        Update the context with a newly captured vehicle during quantification.

        :param capture_id: The Id of the placeholder.
        :param other_vehicle: A tuple with the vehicle Id and its availability time interval.
        """
        new_captured_agents = self.captured_agents.copy()
        new_captured_agents[capture_id] = agent_info
        return OfflineEvaluationMonitorTreeVisitorContext(
            self.world,
            self.start_time_step,
            self.final_time_step,
            new_captured_agents,
        )

    @property
    def vehicle_ids(self) -> list[int]:
        return [params[0] for params in self.captured_agents.values()]


class OfflineEvaluationMonitorTreeVisitor(BaseEvaluationMonitorTreeVisitor[list[float]]):
    def evaluate(
        self,
        node: MonitorNode,
        world: World,
        ego_vehicle: Vehicle,
        start_time_step: int,
        end_time_step: int,
    ):
        vehicles = {0: (ego_vehicle.id, CommonRoadInterval(start_time_step, end_time_step))}
        ctx = OfflineEvaluationMonitorTreeVisitorContext(
            world, start_time_step, end_time_step, vehicles
        )

        return self.visit(node, ctx)

    @singledispatchmethod
    def visit(
        self, node: MonitorNode, ctx: OfflineEvaluationMonitorTreeVisitorContext
    ) -> list[float]:
        raise NotImplementedError

    @visit.register
    def visit_rule_node(
        self, node: RtamtRuleMonitorNode, ctx: OfflineEvaluationMonitorTreeVisitorContext
    ) -> list[float]:
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
    ) -> list[float]:
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
    ) -> list[float]:
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
    ) -> list[float]:
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
    ) -> list[float]:
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
                samples_return.append(-len(samples_less_0) / len(window))

        node.values = samples_return

        return samples_return

    @visit.register
    def visit_historically_duration_severity_node(
        self,
        node: HistoricallyDurationSeverityMonitorNode,
        ctx: OfflineEvaluationMonitorTreeVisitorContext,
    ) -> list[float]:
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
    ) -> list[float]:
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
    ) -> list[float]:
        samples = self.visit(node.child, ctx)
        samples_return = [
            1 - 2 * math.exp(-sample / node.threshold * math.log(2)) for sample in samples
        ]
        node.values = samples_return
        return samples_return

    @visit.register
    def visit_exists_multiple_node(
        self, node: ExistsMultipleMonitorNode, ctx: OfflineEvaluationMonitorTreeVisitorContext
    ) -> list[float]:
        samples, _ = self._visit_quant_node(node, ctx)

        samples_return = []
        for values in samples:
            values_non_nan = list(filter(lambda v: not math.isnan(v), values))
            if len(values_non_nan) >= node.threshold:
                nth_largest_value_index = np.argsort(values_non_nan)[-node.threshold]
                nth_largest_value = values_non_nan[nth_largest_value_index]
                samples_return.append(nth_largest_value)
            else:
                samples_return.append(self._rob_scaler.min)

        node.values = samples_return
        return samples_return

    @visit.register
    def visit_predicate_node(
        self, node: PredicateMonitorNode, ctx: OfflineEvaluationMonitorTreeVisitorContext
    ) -> list[float]:
        vehicle_ids = []
        start_time = 0
        end_time = ctx.final_time_step
        for agent_placeholder in node.agent_placeholders:
            vehicle_id, vehicle_interval = ctx.captured_agents[agent_placeholder]
            vehicle_ids.append(vehicle_id)
            start_time = max(vehicle_interval.start, start_time)
            end_time = min(vehicle_interval.end, end_time)

        samples = []
        for time_step in range(ctx.start_time_step, ctx.final_time_step):
            # Only evaluate the predicate if the other vehicle is available in this time frame.
            if start_time > time_step or end_time < time_step:
                samples.append(float("nan"))
                continue

            robustness = self._do_evaluate_predicate(node, ctx.world, time_step, tuple(vehicle_ids))

            samples.append(robustness)

        node.values = samples
        return samples

    @visit.register
    def visit_constant_trace_node(
        self, node: ConstantTraceMonitorNode, ctx: OfflineEvaluationMonitorTreeVisitorContext
    ) -> list[float]:
        return node.trace

    def _visit_quant_node(
        self, node: QuantMonitorNode, ctx: OfflineEvaluationMonitorTreeVisitorContext
    ) -> tuple[list[list[float]], list[int]]:
        """
        Performs the quantification of vehicles for quant operators.

        Those operators use predicates, which correlate the ego vehicle with all other vehicles in the scenario.
        This method performs this correlation and evaluates each sub-monitor for the permutations of ego vehicle and other vehicles.
        """
        # Track when each vehicle first appears (enters) and when it is no longer present (leaves).
        # This is necessary to define the active time intervals for each vehicle in the scenario.
        # Otherwise, we run into problems, when predicates are evaluated for vehicles which are not available at the evaluated time steps.
        vehicle_start_times = {}
        vehicle_end_times = defaultdict(lambda: ctx.final_time_step)
        for time_step in range(ctx.start_time_step, ctx.final_time_step + 1):
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
            adjusted_ctx = ctx.capture_agent(node.quantified_agent, (vehicle_id, vehicle_interval))
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

    def _should_use_boolean_predicate_evaluation(self, node: PredicateMonitorNode) -> bool:
        return self._use_boolean or (
            node.io_type == IOType.INPUT and self._output_type == OutputType.OUTPUT_ROBUSTNESS
        )
