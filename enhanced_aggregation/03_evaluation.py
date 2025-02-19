"""
Template script that can be used for STL monitoring with model predictive robustness and can be extended with custom aggregation logic.

To get started, you must provide the pre-trained models and put them into `/tmp/models` or change the MprConfig option to your preferred path.
You can either use your own models or download pre-trained ones from https://nextcloud.in.tum.de/index.php/s/bijGnSNZQB92GRz (see commonroad-model-predictive-robustness for more information).
"""

import logging
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.util import Interval as CommonRoadInterval
from commonroad.scenario.scenario import Scenario
from commonroad_mpr.utils.configuration_builder import ConfigurationBuilder as MprCfg
from rtamt.semantics.interval.interval import Interval as RtamtInterval

from crmonitor.common.config import get_traffic_rule_config
from crmonitor.common.helper import gather
from crmonitor.common.world import World, get_world_config
from crmonitor.evaluation.evaluation import RuleEvaluator
from crmonitor.evaluation.visitor import MonitorCreationRuleTreeVisitor, RuleTreeVisitor
from crmonitor.monitor.monitor_node import (
    AllMonitorNode,
    AndsmoothMonitorNode,
    ExistMonitorNode,
    HistoricallyDurationMonitorNode,
    MonitorNode,
    RuleMonitorNode,
)
from crmonitor.monitor.rtamt_monitor_stl import OutputType
from crmonitor.predicates.scaling import RobustnessScaler
from crmonitor.rule.rule_node import PredicateNode

scenario_path = "./scenarios/test_interstate/DEU_test_unnecessary_braking.xml"
use_mpr = False
# If True (default), robustness values will be normalized to the interval [-1.0, 1.0]. If False, robustness values are not normalized and may lay in the interval [-inf, +inf].
# Disable with caution when use_mpr is also enabled, as mpr with gaussian processes does not perform any normalization on its own.
scale_rob = True

# Optionally provide a Path where pre-trained models can be found. If None is specified, the models from the mpr repo are used.
model_path = None

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
            Path(__file__).parent.parent.parent
            / "commonroad-model-predictive-robustness"
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
        self, node: MonitorNode, world, mpr_world, max_time_step, ego_vehicle, *ctx
    ):
        self.other_ids = tuple()
        other_ids = [(ego_vehicle.id,)] * max_time_step
        return node.visit(self, world, mpr_world, max_time_step, other_ids, *ctx)

    def visit_rule_node(self, rule_node: RuleMonitorNode, *ctx):
        world = ctx[0]
        # Collect child_values
        assert rule_node.monitor.dt == world.dt, (
            f"Monitor constructed with dt="
            f"{rule_node.monitor.dt} but got "
            f"world state with dt={world.dt}!"
        )
        child_values = {c.name: c.visit(self, *ctx) for c in rule_node.children}
        val = rule_node.evaluate(
            list(child_values.items())
        )  # evaluate instead of update for offline usage
        scaled_values = list(np.clip(val, self._rob_scaler.min, self._rob_scaler.max))
        self.all_values_all_ids[rule_node.name] = scaled_values
        return scaled_values

    def _visit_quant_node(self, node, *ctx):
        """
        This method performs the quantification for the operators all and exist.
        Those operators use predicates, which correlate the ego vehicle with all other vehicles in the scenario.
        This method performs this correlation and evaluates each sub-monitor for the permutations of ego vehicle and other vehicles.
        """
        world, mpr_world, max_time_step, other_ids_at_time = ctx[:4]
        # Stores tuples of ego vehicle + other vehicle in a time series list indexed by the other vehicle.
        # This is passed to the evaluators below to correlate the ego vehicle with all other vehicles in the scenario.
        # TODO: Shouldn't a pair be sufficient, because other_ids_at_time always contains a single ego vehicle?
        selected_ids_by_vehicle_id = defaultdict(lambda: [()] * max_time_step)
        for time_step in range(0, max_time_step):
            other_ids = other_ids_at_time[time_step]
            all_ids = world.vehicle_ids_for_time_step(time_step)
            remaining_ids = tuple(set(all_ids).difference(other_ids))
            for remaining_id in remaining_ids:
                selected_ids_by_vehicle_id[remaining_id][time_step] = other_ids + (
                    remaining_id,
                )

        values = []
        ret_selected_ids = []
        for remaining_id, selected_ids in selected_ids_by_vehicle_id.items():
            val = node.monitors[remaining_id].visit(
                self, world, mpr_world, max_time_step, selected_ids, *ctx[2:]
            )
            values.append(val)
            ret_selected_ids.append(selected_ids)

        return values, ret_selected_ids

    def visit_all_node(self, all_node: AllMonitorNode, *ctx):
        # Mostly the same as visit_all_node of EvaluationMonitorTreeVisitor, except that it handles time series data (because of the offline evaluation)
        samples, selected_ids = self._visit_quant_node(all_node, *ctx)
        world, mpr_world, max_time_step, other_idss = ctx[:4]
        robustness_values = []
        for time_step in range(0, max_time_step):
            values = [predicate_values[time_step] for predicate_values in samples]
            if len(values) > 0:
                idx = np.argmin(values)
                val = values[idx]
                self.other_ids = selected_ids[idx]
                all_node.last_selected = all_node.monitors[self.other_ids[-1]]

            else:
                val = self._rob_scaler.max
                self.other_ids = other_idss
                all_node.last_selected = None

            robustness_values.append(val)

        self.all_values_all_ids[all_node.name] = robustness_values
        return robustness_values

    def visit_exist_node(self, exist_node: ExistMonitorNode, *ctx):
        # Mostly the same as visit_exist_node of EvaluationMonitorTreeVisitor, except that it handles time series data (because of the offline evaluation)
        samples, selected_ids = self._visit_quant_node(exist_node, *ctx)
        world, mpr_world, max_time_step, other_ids = ctx[:4]

        robustness_values = []
        for time_step in range(0, max_time_step):
            values = [predicate_values[time_step] for predicate_values in samples]

            if len(values) > 0:
                idx = np.argmax(values)
                val = values[idx]
                self.other_ids = selected_ids[idx]

                exist_node.last_selected = exist_node.monitors[self.other_ids[-1]]

            else:
                val = self._rob_scaler.min
                self.other_ids = other_ids
                exist_node.last_selected = None

            robustness_values.append(val)

        self.all_values_all_ids[exist_node.name] = robustness_values
        return robustness_values

    def visit_andsmooth_node(self, andsmooth_node: AndsmoothMonitorNode, *ctx):
        world, mpr_world, time_step, other_ids = ctx[:4]
        samples_left = andsmooth_node.children[0].visit(self, *ctx)
        samples_right = andsmooth_node.children[1].visit(self, *ctx)

        samples = []
        for a, b in zip(samples_left, samples_right):
            k = 2.0 * 1e-6
            x = (b - a) / k
            g = 0.5 * (x + math.sqrt(x * x + 1.0))
            smin = b - k * g
            samples.append(smin)

        self.all_values_all_ids[andsmooth_node.name] = samples
        return samples

    def visit_historicallyduration_node(
        self, historicallyduration_node: HistoricallyDurationMonitorNode, *ctx
    ):
        sample = historicallyduration_node.children[0].visit(self, *ctx)
        world, mpr_world, max_time_step = ctx[:3]

        if historicallyduration_node.interval is not None:
            interval = _rtamt_interval_to_commonroad_interval(
                historicallyduration_node.interval, world.scenario
            )
            begin = int(interval.start)
            end = min(max_time_step, int(interval.end))
        else:
            begin = 0
            end = max_time_step

        window_size = end - begin  # sliding average window size
        # Fill up the values before the interval, so that the returned trace is as long as the input
        sample_return = [self._rob_scaler.max] * begin
        # Computes the sliding average over the samples
        for i in range(begin, len(sample)):
            window = sample[max(begin, i - window_size) : i + 1]
            sample_return.append(sum(window) / len(window))

        self.all_values_all_ids[historicallyduration_node.name] = sample_return
        return sample_return

    def visit_historicallydurationseverity_node(
        self,
        historicallydurationseverity_node: HistoricallyDurationMonitorNode,
        *ctx,
    ):
        samples = historicallydurationseverity_node.children[0].visit(self, *ctx)
        world, _, max_time_step = ctx[:3]

        if historicallydurationseverity_node.interval is not None:
            interval = _rtamt_interval_to_commonroad_interval(
                historicallydurationseverity_node.interval, world.scenario
            )
            begin = int(interval.start)
            end = min(max_time_step, int(interval.end))
        else:
            begin = 0
            end = max_time_step

        # The concrete implementation of this operator closely follows the implementation of `visitTimedHistorically` from rtamt.

        # Extend the samples, so that we can iterate with a static window size
        # and to make sure that the returned trace covers the interval [0, max_time_step].
        extended_samples = [self._rob_scaler.max for _ in range(end)] + samples
        sample_return = []
        all_samples_are_ge_0 = all(x >= 0 for x in samples[begin : end + 1])
        for i in range(end, len(extended_samples)):
            # Iterate over the extended sample using a window of the size `(end - begin) + 1`.
            window = extended_samples[i - end : i - begin + 1]
            if all_samples_are_ge_0:
                sample_return.append(min(window))
            else:
                samples_less_0 = list(filter(lambda x: x < 0, window))
                sample_return.append(sum(samples_less_0) / len(window))

        self.all_values_all_ids[historicallydurationseverity_node.name] = sample_return
        return sample_return

    def visit_predicate_node(self, predicate_node: PredicateNode, *ctx):
        world, mpr_world, max_time_step, other_idss = ctx[:4]
        samples = []
        for time_step in range(0, max_time_step):
            other_ids = other_idss[time_step]
            predicate_ids = gather(other_ids, predicate_node.agent_placeholders)

            samples.append(
                predicate_node.evaluate_robustness(
                    world, mpr_world, time_step, predicate_ids
                )
            )

        self.all_values_all_ids[predicate_node.name] = samples
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
    rule="R_G2",
    monitor_creation_visitor=MonitorCreationRuleTreeVisitor(dt=scenario.dt),
    monitor_evaluation_visitor=OfflineEvaluationMonitorTreeVisitor(),
    output_type=OutputType.STANDARD,
)
# Either step through time steps sequentially
robustness = rule_evaluator.evaluate_offline()
print(f"robustness is {robustness}")
