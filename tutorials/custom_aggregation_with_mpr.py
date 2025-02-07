"""
Template script that can be used for STL monitoring with model predictive robustness and can be extended with custom aggregation logic.

To get started, you must provide the pre-trained models and put them into `/tmp/models` or change the MprConfig option to your preferred path.
You can either use your own models or download pre-trained ones from https://nextcloud.in.tum.de/index.php/s/bijGnSNZQB92GRz (see commonroad-model-predictive-robustness for more information).
"""

from collections import defaultdict
from pathlib import Path
import math

from commonroad.common.file_reader import CommonRoadFileReader
import numpy as np

from crmonitor.common.helper import gather
from crmonitor.common.config import get_traffic_rule_config
from crmonitor.common.world import World, get_world_config
from crmonitor.evaluation.evaluation import RuleEvaluator
from commonroad_mpr.utils.configuration_builder import ConfigurationBuilder as MprCfg
from crmonitor.evaluation.visitor import (
    MonitorCreationRuleTreeVisitor,
    RuleTreeVisitor,
)
from crmonitor.monitor.monitor_node import (
    AllMonitorNode,
    ExistMonitorNode,
    HistoricallydurationMonitorNode,
    MonitorNode,
    RuleMonitorNode,
    AndsmoothMonitorNode,
)
from crmonitor.monitor.rtamt_monitor_stl import OutputType
from crmonitor.rule.rule_node import PredicateNode

scenario_path = "./scenarios/test_interstate/DEU_test_unnecessary_braking.xml"
use_mpr = False

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


class OfflineEvaluationMonitorTreeVisitor(RuleTreeVisitor):
    def __init__(self, use_boolean=False, output_type=OutputType.STANDARD):
        self.other_ids = tuple()
        self.use_boolean = use_boolean
        self.output_type = output_type
        self.all_values_all_ids = {}
        self.all_props_all_ids = {}

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
        return val

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

        self.all_values_all_ids = {}  # reset to empty
        self.all_props_all_ids = {}  # reset to empty

        robustness_values = []
        for time_step in range(0, max_time_step):
            values = [predicate_values[time_step] for predicate_values in samples]
            if len(values) > 0:
                idx = np.argmin(values)
                val = values[idx]
                self.other_ids = selected_ids[idx]
                all_node.last_selected = all_node.monitors[self.other_ids[-1]]

                # Loop through all selected_ids and populate the dictionary
                for i, sid in enumerate(selected_ids):
                    self.all_values_all_ids[sid[-1]] = values[i]
                    if hasattr(all_node.monitors[sid[-1]].monitor, "_propositions"):
                        self.all_props_all_ids[sid[-1]] = all_node.monitors[
                            sid[-1]
                        ].monitor._propositions
            else:
                val = 1.0
                self.other_ids = other_idss
                all_node.last_selected = None

            robustness_values.append(val)

        return robustness_values

    def visit_exist_node(self, exist_node: ExistMonitorNode, *ctx):
        # Mostly the same as visit_exist_node of EvaluationMonitorTreeVisitor, except that it handles time series data (because of the offline evaluation)
        samples, selected_ids = self._visit_quant_node(exist_node, *ctx)
        world, mpr_world, max_time_step, other_ids = ctx[:4]

        self.all_values_all_ids = {}  # reset to empty
        self.all_props_all_ids = {}  # reset to empty

        robustness_values = []
        for time_step in range(0, max_time_step):
            values = [predicate_values[time_step] for predicate_values in samples]

            if len(values) > 0:
                idx = np.argmax(values)
                val = values[idx]
                self.other_ids = selected_ids[idx]

                exist_node.last_selected = exist_node.monitors[self.other_ids[-1]]

                # Loop through all selected_ids and populate the dictionary
                for i, sid in enumerate(selected_ids):
                    self.all_values_all_ids[sid[-1]] = values[i]
            else:
                val = -1.0
                self.other_ids = other_ids
                exist_node.last_selected = None

                # If no values, only add other_ids if it's not empty
                if other_ids:
                    self.all_values_all_ids[other_ids[-1]] = val

            robustness_values.append(val)

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

        return samples

    def visit_historicallyduration_node(
        self, historicallyduration_node: HistoricallydurationMonitorNode, *ctx
    ):
        child_values = historicallyduration_node.children[0].visit(self, *ctx)
        world, mpr_world, max_time_step = ctx[:3]

        # The interval can be defined with different units. Therefore, they are first normalized to time steps.
        # This currently only supports time steps and seconds as units.
        begin_unit = historicallyduration_node.interval.begin_unit
        end_unit = historicallyduration_node.interval.end_unit
        if len(begin_unit) == 0 and len(end_unit) == 0:
            normalized_begin = int(historicallyduration_node.interval.begin)
            normalized_end = int(historicallyduration_node.interval.end)
        else:
            normalized_begin = int(
                historicallyduration_node.interval.begin / world.scenario.dt
            )
            normalized_end = int(
                historicallyduration_node.interval.end / world.scenario.dt
            )
        begin = max(0, normalized_begin)
        end = min(max_time_step, normalized_end)

        spec_violations = 0
        total = 0
        for time_step in range(begin, end):
            if child_values[time_step] < 0:
                spec_violations += 1
            total += 1

        # Interpolate the robustness between -1.0 (spec_violations=total) and 1.0 (spec_violations=0)
        rob = (2.0 * ((total - spec_violations) / total)) - 1.0

        # The robustness outside the interval is filled with 1.0 and only the interval is set to the computed robustness
        samples = [1.0] * max_time_step
        for time_step in range(begin, end):
            samples[time_step] = rob

        return samples

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


# Create a world state, which is a holder class for intermediate results produced by the monitoring.
# Use the convenience class method to create with default configuration from a scenario.
world = World.create_from_scenario(scenario, config=config)


# Create a rule evaluator
# Provide the vehicle to evaluate traffic rules for as ego vehicle
ego_vehicle = next(iter(world.vehicles))
rule_evaluator = RuleEvaluator.create_from_config(
    world,
    ego_vehicle.id,
    rule="R_G3",
    monitor_creation_visitor=MonitorCreationRuleTreeVisitor(dt=scenario.dt),
    monitor_evaluation_visitor=OfflineEvaluationMonitorTreeVisitor(),
)

# Either step through time steps sequentially
robustness = rule_evaluator.evaluate_offline()
print(f"robustness is {robustness}")
