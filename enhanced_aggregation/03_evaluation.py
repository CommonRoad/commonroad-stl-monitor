"""
Template script that can be used for STL monitoring with model predictive robustness and can be extended with custom aggregation logic.

To get started, you must provide the pre-trained models and put them into `/tmp/models` or change the MprConfig option to your preferred path.
You can either use your own models or download pre-trained ones from https://nextcloud.in.tum.de/index.php/s/bijGnSNZQB92GRz (see commonroad-model-predictive-robustness for more information).
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.util import Interval as CommonRoadInterval
from commonroad.scenario.scenario import Scenario
from commonroad_mpr.common.observation import World as MprWorld
from commonroad_mpr.utils.configuration_builder import ConfigurationBuilder as MprCfg
from rtamt.semantics.interval.interval import Interval as RtamtInterval

from crmonitor.common.config import get_traffic_rule_config
from crmonitor.common.vehicle import Vehicle
from crmonitor.common.world import World, get_world_config
from crmonitor.evaluation.evaluation import RuleEvaluator
from crmonitor.evaluation.visitor import MonitorCreationRuleTreeVisitor, RuleTreeVisitor
from crmonitor.evaluation.visualization import FormulaVisualizationVisitor
from crmonitor.monitor.monitor_node import (
    AllMonitorNode,
    AndsmoothMonitorNode,
    CompareToThresholdScaledMonitorNode,
    ExistMonitorNode,
    HistoricallyDurationMonitorNode,
    MonitorNode,
    RuleMonitorNode,
    SumIfPositiveMonitorNode,
)
from crmonitor.monitor.rtamt_monitor_stl import OutputType
from crmonitor.predicates.scaling import RobustnessScaler
from crmonitor.rule.rule_node import PredicateNode

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

scenario_path = "./scenarios/test_interstate/DEU_test_unnecessary_braking.xml"
use_mpr = False
# If True (default), robustness values will be normalized to the interval [-1.0, 1.0]. If False, robustness values are not normalized and may lay in the interval [-inf, +inf].
# Disable with caution when use_mpr is also enabled, as mpr with gaussian processes does not perform any normalization on its own.
scale_rob = False

# Optionally provide a Path where pre-trained models can be found. If None is specified, the models from the mpr repo are used.
model_path = Path(__file__).parent.parent.joinpath("output/models")

# Specify the traffic rule you want to evaluate. For an overview of the available traffic rules, see `traffic_rules_rtamt.yaml`.
traffic_rule = "R_G1"

# Set to `OutputType.OUTPUT_ROBUSTNESS` for IA-STL, and to `OutputType.STANDARD` for standard STL.
output_type = OutputType.OUTPUT_ROBUSTNESS

logging.basicConfig(level=logging.INFO)

# Open the scenario
# Make sure to call with lanelet_assignment=True
scenario, _ = CommonRoadFileReader(scenario_path).open(lanelet_assignment=True)

if use_mpr:
    MprCfg.build_configuration(
        config={
            "common": {
                "scenario": "interstate",
                "lane": {
                    # Increased the default parameters to work around projection limit issues in MPR
                    "lateral_projection_domain_limit": 500,
                    "extend_length": 500,
                    "large_resampling_step": 3.5,
                    "num_chankins_corner_cutting": 1,
                },
                "road_network": {
                    "interstate": {
                        "use_phantom_lane": False
                    }  # Must disable phantom lanes, because otherwise commonroad-dc segfaults...
                },
            },
            "path": {
                "path_models": "/home/beicekol/projects/work/uni/commonroad-model-predictive-robustness/output/models/"
            },  # point to the models, either the ones you have trained or the pre-trained ones.
        },
        # Path root must point to a local revision of commonroad-model-predictive-robustness.
        # This configuration, assumes that the repo is in the same directory as stl-monitor repo.
        # If this is not the case for your setup, adjust the path here accordingly.
        path_root=str(
            Path(__file__).parent.parent.parent / "commonroad-model-predictive-robustness"
        ),
        folder_config="config_files",
        default_profile="default",
    )


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
    ego_vehicle: Vehicle
    other_vehicle: Optional[Tuple[int, Tuple[int, int]]]
    """
    Optionally provide one other vehicle that should be considered for the evaluation of binary predicates. This field is populated during the evaluation by the quantifiers.
    """

    def copy_with_other_vehicles(
        self, other_vehicle: Tuple[int, Tuple[int, int]]
    ) -> "OfflineEvaluationMonitorTreeVisitorContext":
        return OfflineEvaluationMonitorTreeVisitorContext(
            self.world,
            self.mpr_world,
            self.max_time_step,
            self.ego_vehicle,
            other_vehicle,
        )


class OfflineEvaluationMonitorTreeVisitor(RuleTreeVisitor):
    def __init__(self, use_boolean=False, output_type=OutputType.STANDARD):
        self.other_ids = tuple()
        self.use_boolean = use_boolean
        self.output_type = output_type
        self.all_values_all_ids = {}
        self.all_props_all_ids = {}

        # TODO: when this visitor is integrated into crmonitor directly, this option should be read from the config
        self._rob_scaler = RobustnessScaler(scale=scale_rob)

    def walk(
        self,
        node: MonitorNode,
        world: World,
        mpr_world: MprWorld,
        max_time_step: int,
        ego_vehicle: Vehicle,
    ):
        self.other_ids = tuple()
        # TODO: The context mostly contains static objects, which could also be encoded as object attributes of the visitor.
        # The only element that must be
        ctx = OfflineEvaluationMonitorTreeVisitorContext(
            world, mpr_world, max_time_step, ego_vehicle, None
        )
        return node.visit(self, ctx)

    def _record_node_samples(
        self, node, ctx: OfflineEvaluationMonitorTreeVisitorContext, values
    ) -> None:
        vehicle_ids = [ctx.ego_vehicle.id]
        if ctx.other_vehicle:
            vehicle_ids.append(ctx.other_vehicle[0])

        if node.name not in self.all_values_all_ids:
            self.all_values_all_ids[node.name] = {}
        self.all_values_all_ids[node.name][tuple(vehicle_ids)] = values

    def visit_rule_node(
        self,
        rule_node: RuleMonitorNode,
        ctx: OfflineEvaluationMonitorTreeVisitorContext,
    ):
        world = ctx.world
        # Collect child_values
        assert rule_node.monitor.dt == world.dt, (
            f"Monitor constructed with dt="
            f"{rule_node.monitor.dt} but got "
            f"world state with dt={world.dt}!"
        )

        child_values = {c.name: c.visit(self, ctx) for c in rule_node.children}
        val = rule_node.evaluate(
            list(child_values.items())
        )  # evaluate instead of update for offline usage

        # When the rule is evaluated with IA-STL, some robustness values might be +inf.
        # This can lead to problems if the user expects scaled values.
        # Therefore, a simple clip is applied here, to make sure the robustness values
        # remain in the required robustness value interval.
        scaled_values = list(np.clip(val, self._rob_scaler.min, self._rob_scaler.max))
        self._record_node_samples(rule_node, ctx, scaled_values)
        return scaled_values

    def _visit_quant_node(self, node, ctx: OfflineEvaluationMonitorTreeVisitorContext):
        """
        This method performs the quantification for the operators all and exist.
        Those operators use predicates, which correlate the ego vehicle with all other vehicles in the scenario.
        This method performs this correlation and evaluates each sub-monitor for the permutations of ego vehicle and other vehicles.
        """
        vehicle_start_times = {}
        vehicle_end_times = defaultdict(lambda: ctx.max_time_step)
        for time_step in range(0, ctx.max_time_step):
            all_ids = set(world.vehicle_ids_for_time_step(time_step))

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
            if vehicle_id == ctx.ego_vehicle.id:
                continue

            other_vehicle_params = (
                vehicle_id,
                (vehicle_start_times[vehicle_id], vehicle_end_times[vehicle_id]),
            )
            val = node.monitors[(ctx.ego_vehicle.id, vehicle_id)].visit(
                self, ctx.copy_with_other_vehicles(other_vehicle_params)
            )
            values.append(val)
            ret_selected_ids.append(vehicle_id)

        return values, ret_selected_ids

    def visit_all_node(
        self, all_node: AllMonitorNode, ctx: OfflineEvaluationMonitorTreeVisitorContext
    ):
        # Mostly the same as visit_all_node of EvaluationMonitorTreeVisitor, except that it handles time series data (because of the offline evaluation)
        samples, selected_ids = self._visit_quant_node(all_node, ctx)
        robustness_values = []
        for time_step in range(0, ctx.max_time_step):
            values = [predicate_values[time_step] for predicate_values in samples]
            if len(values) > 0:
                idx = np.argmin(values)
                val = values[idx]
                # all_node.last_selected = all_node.monitors[selected_ids[-1]]

            else:
                val = self._rob_scaler.max
                all_node.last_selected = None

            robustness_values.append(val)

        scaled_robustness_values = np.clip(
            robustness_values, self._rob_scaler.min, self._rob_scaler.max
        )
        self.all_values_all_ids[all_node.name] = scaled_robustness_values
        return scaled_robustness_values

    def visit_exist_node(
        self,
        exist_node: ExistMonitorNode,
        ctx: OfflineEvaluationMonitorTreeVisitorContext,
    ):
        # Mostly the same as visit_exist_node of EvaluationMonitorTreeVisitor, except that it handles time series data (because of the offline evaluation)
        samples, selected_ids = self._visit_quant_node(exist_node, ctx)

        robustness_values = []
        for time_step in range(0, ctx.max_time_step):
            values = [predicate_values[time_step] for predicate_values in samples]

            if len(values) > 0:
                idx = np.argmax(values)
                val = values[idx]

                exist_node.last_selected = exist_node.monitors[selected_ids[-1]]
            else:
                val = self._rob_scaler.min
                exist_node.last_selected = None

            robustness_values.append(val)

        self.all_values_all_ids[exist_node.name] = robustness_values
        return robustness_values

    def visit_andsmooth_node(
        self,
        andsmooth_node: AndsmoothMonitorNode,
        ctx: OfflineEvaluationMonitorTreeVisitorContext,
    ):
        samples_left = andsmooth_node.children[0].visit(self, ctx)
        samples_right = andsmooth_node.children[1].visit(self, ctx)

        samples = []
        for a, b in zip(samples_left, samples_right):
            k = 2.0 * 1e-6
            x = (b - a) / k
            g = 0.5 * (x + math.sqrt(x * x + 1.0))
            smin = b - k * g
            samples.append(smin)

        self._record_node_samples(andsmooth_node, ctx, samples)
        return samples

    def visit_historicallyduration_node(
        self,
        historicallyduration_node: HistoricallyDurationMonitorNode,
        ctx: OfflineEvaluationMonitorTreeVisitorContext,
    ):
        samples = historicallyduration_node.children[0].visit(self, ctx)

        if historicallyduration_node.interval is not None:
            interval = _rtamt_interval_to_commonroad_interval(
                historicallyduration_node.interval, ctx.world.scenario
            )
            begin = int(interval.start)
            end = min(ctx.max_time_step, int(interval.end))
        else:
            begin = 0
            end = ctx.max_time_step

        # The concrete implementation of this operator closely follows the implementation of `visitTimedHistorically` from rtamt.

        # Extend the samples, so that we can iterate with a static window size
        # and to make sure that the returned trace covers the interval [0, max_time_step].
        extended_samples = [self._rob_scaler.max for _ in range(end)] + samples
        sample_return = []
        for i in range(end, len(extended_samples)):
            # Iterate over the extended sample using a window of the size `(end - begin) + 1`.
            window = extended_samples[i - end : i - begin + 1]
            all_samples_are_ge_0 = all(x >= 0 for x in window)
            if all_samples_are_ge_0:
                sample_return.append(min(window))
            else:
                samples_less_0 = list(filter(lambda x: x < 0, window))
                sample_return.append(len(samples_less_0) / len(window))

        self._record_node_samples(historicallyduration_node, ctx, values)
        return sample_return

    def visit_historicallydurationseverity_node(
        self,
        historicallydurationseverity_node: HistoricallyDurationMonitorNode,
        ctx: OfflineEvaluationMonitorTreeVisitorContext,
    ):
        samples = historicallydurationseverity_node.children[0].visit(self, ctx)

        if historicallydurationseverity_node.interval is not None:
            interval = _rtamt_interval_to_commonroad_interval(
                historicallydurationseverity_node.interval, ctx.world.scenario
            )
            begin = int(interval.start)
            end = min(ctx.max_time_step, int(interval.end))
        else:
            begin = 0
            end = ctx.max_time_step

        # The concrete implementation of this operator closely follows the implementation of `visitTimedHistorically` from rtamt.

        # Extend the samples, so that we can iterate with a static window size
        # and to make sure that the returned trace covers the interval [0, max_time_step].
        extended_samples = [self._rob_scaler.max for _ in range(end)] + samples
        sample_return = []
        for i in range(end, len(extended_samples)):
            # Iterate over the extended sample using a window of the size `(end - begin) + 1`.
            window = extended_samples[i - end : i - begin + 1]
            all_samples_are_ge_0 = all(x >= 0 for x in window)
            if all_samples_are_ge_0:
                sample_return.append(min(window))
            else:
                samples_less_0 = list(filter(lambda x: x < 0, window))
                sample_return.append(sum(samples_less_0) / len(window))

        self._record_node_samples(historicallydurationseverity_node, ctx, sample_return)
        return sample_return

    def visit_sum_if_positive_node(
        self,
        sum_if_positive_node: SumIfPositiveMonitorNode,
        ctx: OfflineEvaluationMonitorTreeVisitorContext,
    ):
        samples, selected_ids = self._visit_quant_node(sum_if_positive_node, ctx)

        samples_return = []
        for time_step in range(0, ctx.max_time_step):
            values = [predicate_values[time_step] for predicate_values in samples]

            if len(values) > 0:
                val = sum([val for val in values if val > 0])
            else:
                val = float("nan")

            samples_return.append(val)

        self.all_values_all_ids[sum_if_positive_node.name] = samples_return
        return samples_return

    def visit_compare_to_threshold_scaled_node(
        self,
        compare_to_threshold_scaled_node: CompareToThresholdScaledMonitorNode,
        ctx: OfflineEvaluationMonitorTreeVisitorContext,
    ):
        samples = compare_to_threshold_scaled_node.children[0].visit(self, *ctx)

        samples_return = [
            1 - 2 * math.exp(-sample / compare_to_threshold_scaled_node.threshold * math.log(2))
            for sample in samples
        ]
        self.all_values_all_ids[compare_to_threshold_scaled_node.name] = samples_return
        return samples_return

    def visit_predicate_node(
        self,
        predicate_node: PredicateNode,
        ctx: OfflineEvaluationMonitorTreeVisitorContext,
    ):
        vehicle_ids = [ctx.ego_vehicle.id]

        predicate_arity = len(predicate_node.agent_placeholders)
        if predicate_arity == 2:
            if ctx.other_vehicle is None:
                raise RuntimeError()
            (
                other_vehicle_id,
                (other_vehicle_start_time, other_vehicle_end_time),
            ) = ctx.other_vehicle
            vehicle_ids.append(other_vehicle_id)
        else:
            other_vehicle_start_time = 0
            other_vehicle_end_time = ctx.max_time_step

        samples = []
        for time_step in range(0, ctx.max_time_step):
            # Only evaluate the predicate if the other vehicle
            if other_vehicle_start_time > time_step or other_vehicle_end_time < time_step:
                samples.append(self._rob_scaler.max)
                continue

            samples.append(
                predicate_node.evaluate_robustness(ctx.world, ctx.mpr_world, time_step, vehicle_ids)
            )

        self._record_node_samples(predicate_node, ctx, samples)
        return samples


# ==============================
# End custom aggregation logic
# ==============================

# config used for the world creation
config = get_world_config()
# MPR must be explicitly enabled
config["use_mpr"] = use_mpr

# MPR must be explicitly enabled
rule_evaluator_config = get_traffic_rule_config()
rule_evaluator_config["traffic_rules_param"]["use_mpr"] = use_mpr
rule_evaluator_config["traffic_rules_param"]["scale_rob"] = scale_rob
rule_evaluator_config["traffic_rules_param"]["model_path"] = model_path


# Create a world state, which is a holder class for intermediate results produced by the monitoring.
# Use the convenience class method to create with default configuration from a scenario.
world = World.create_from_scenario(scenario, config=config)


# Create a rule evaluator
# Provide the vehicle to evaluate traffic rules for as ego vehicle
ego_vehicle = next(iter(world.vehicles))
rule_evaluator = RuleEvaluator.create_from_config(
    world,
    ego_vehicle.id,
    rule=traffic_rule,
    monitor_creation_visitor=MonitorCreationRuleTreeVisitor(
        dt=scenario.dt, output_type=output_type
    ),
    monitor_evaluation_visitor=OfflineEvaluationMonitorTreeVisitor(),
    output_type=output_type,
)
# Either step through time steps sequentially
robustness = rule_evaluator.evaluate_offline()
print(f"robustness is {robustness}")

# TODO: Expose the AST values over a public API, such that we no longer need to access the private attributes.
values = rule_evaluator._eval_visitor.all_values_all_ids
visualization_visitor = FormulaVisualizationVisitor(values, scale_rob)
visualization_visitor.visualize(rule_evaluator._monitor)
plt.show()
