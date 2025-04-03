import csv
import logging
from collections import defaultdict
from pathlib import Path
from random import Random

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.util import Interval
from commonroad.scenario.scenario import Scenario
from commonroad_mpr.common.observation import World as WorldMPR
from commonroad_mpr.utils.configuration_builder import ConfigurationBuilder as MprCfg
from crmonitor.common.world import World
from crmonitor.predicate_grouping import (
    all_general_predicates,
    all_interstate_predicates,
    insufficient,
    changed_to_meta,
)
from crmonitor.predicates.base import PredicateEvaluatorConfig, PredicateMprConfig
from crmonitor.predicates.predicate_factory import PredicateFactory

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)


metrics_output_path = (
    Path(__file__).parent.parent / "output" / "metrics" / "mfr_and_mpr_metrics.csv"
)
metrics_output_path.parent.mkdir(exist_ok=True, parents=True)

scenarios_load_path = Path(__file__).parents[3] / "scenarios-for-semantic-aware-stl" / "highD"
iterations = 1000
models_path = Path(__file__).parent.parent / "output" / "models"
selected_predicates = all_general_predicates + all_interstate_predicates + changed_to_meta
rand_seed = 12345

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
                    "use_phantom_lane": True
                }  # Must disable phantom lanes, because otherwise commonroad-dc segfaults...
            },
        },
    },
    # Path root must point to a local revision of commonroad-model-predictive-robustness.
    # This configuration, assumes that the repo is in the same directory as stl-monitor repo.
    # If this is not the case for your setup, adjust the path here accordingly.
    path_root=str(Path(__file__).parent.parent.parent / "commonroad-model-predictive-robustness"),
    folder_config="config_files",
    default_profile="default",
)


def get_scenario_final_time_step(scenario: Scenario) -> int:
    """
    Determines the maximum time step in a scenario. This is usefull, to determine the length of a scenario.

    :param scenario: The scenario to analyze.

    :return: The final time step in the scenario, or 0 if no obstacles are in the scenario.
    """
    max_time_step = 0
    for dynamic_obstacle in scenario.dynamic_obstacles:
        if dynamic_obstacle.prediction is None:
            max_time_step = max(max_time_step, dynamic_obstacle.initial_state.time_step)
            continue

        max_time_step = max(max_time_step, dynamic_obstacle.prediction.final_time_step)

    if isinstance(max_time_step, Interval):
        return int(max_time_step.end)
    else:
        return max_time_step


scenarios = list(scenarios_load_path.glob("*.xml"))
if len(scenarios) == 0:
    raise RuntimeError(f"No scenarios were found in {scenarios_load_path}.")

predicate_evaluator_config = PredicateEvaluatorConfig(
    scale_rob=True,
    mpr=PredicateMprConfig(enabled=True, model_path=models_path),
)
predicate_factory = PredicateFactory(predicate_evaluator_config)
predicates = [
    predicate_factory.get_predicate(predicate_name) for predicate_name in selected_predicates
]
random = Random(rand_seed)

mpr_rob = defaultdict(list)  # GP predicted
mfr_rob = defaultdict(list)
for i in range(0, iterations):
    if i % 10 == 0:
        _LOGGER.info(f"Iteration {i}/{iterations}")
    scenario_path = random.choice(scenarios)

    scenario, _ = CommonRoadFileReader(scenario_path).open(lanelet_assignment=True)
    world = World.create_from_scenario(scenario)
    mpr_world = WorldMPR.create_from_scenario(scenario)

    # Search for a time step, where at least two vehicles are present.
    end_time = get_scenario_final_time_step(scenario)
    # Keep track of time steps, which were not yet considered.
    time_steps = [i for i in range(0, end_time + 1)]
    time_step = 0
    vehicle_ids_at_time_step = []
    # WorldMPR does not necessarily create a vehicle for each dynamic obstacle at it only considers vehicles which have a trajetory which covers more then one time step.
    are_vehicles_in_both_worlds = False
    are_enough_vehicles_at_time_step = False
    while (not are_enough_vehicles_at_time_step or not are_vehicles_in_both_worlds) and len(
        time_steps
    ) > 0:
        time_step = time_steps.pop(random.randint(0, len(time_steps) - 1))
        vehicle_ids_at_time_step = world.vehicle_ids_for_time_step(time_step)

        are_vehicles_in_both_worlds = all(
            mpr_world.has_vehicle(vehicle_id) for vehicle_id in vehicle_ids_at_time_step
        )
        are_enough_vehicles_at_time_step = len(vehicle_ids_at_time_step) >= 2

    if not are_enough_vehicles_at_time_step or not are_vehicles_in_both_worlds:
        _LOGGER.warning(
            f"Cannot process scenario {scenario.scenario_id}: No time step with at least two active vehicles in both worlds was found!"
        )
        continue

    ego_vehicle_id, other_vehicle_id = random.sample(vehicle_ids_at_time_step, 2)

    for predicate_evaluator in predicates:
        # TODO the evaluate_mpr_ml method has two drawbacks
        #  - by default, it performs rectification
        #  - It must recompute the features for each predicate
        #  Therefore, please implement a solution that directly makes use of the ExactGPModel.predict method.
        mpr_robustness = predicate_evaluator.evaluate_mpr_ml(
            world, mpr_world, time_step, [ego_vehicle_id, other_vehicle_id], rectification=False
        )
        if (not -1 <= mpr_robustness <= 1) or np.isnan(mpr_robustness):
            _LOGGER.warning(f"MPR: {mpr_robustness:.3f}")
            continue

        mfr_robustness = predicate_evaluator.evaluate_robustness(
            world, time_step, [ego_vehicle_id, other_vehicle_id]
        )
        # print(scenario.scenario_id, time_step, ego_vehicle_id, mpr_robustness, mfr_robustness)
        mfr_rob[predicate_evaluator.predicate_name].append(mfr_robustness)
        mpr_rob[predicate_evaluator.predicate_name].append(mpr_robustness)


metrics = []
for predicate_name in selected_predicates:
    y_true = np.array([rob > 0 for rob in mfr_rob[predicate_name]])
    y_pred = np.array([rob > 0 for rob in mpr_rob[predicate_name]])

    TP = np.logical_and(y_true, y_pred).sum()
    FP = np.logical_and(~y_true, y_pred).sum()
    FN = np.logical_and(y_true, ~y_pred).sum()
    TN = np.logical_and(~y_true, ~y_pred).sum()

    eps = 1e-9
    precision = (TP + eps) / (TP + FP + eps)
    recall = (TP + eps) / (TP + FN + eps)
    f1_score = 2 * precision * recall / (precision + recall + eps)
    mpr_variance = np.var(mpr_rob[predicate_name])
    mpr_span = np.max(mpr_rob[predicate_name]) - np.min(mpr_rob[predicate_name])

    mfr_variance = np.var(mfr_rob[predicate_name])
    mfr_span = np.max(mfr_rob[predicate_name]) - np.min(mfr_rob[predicate_name])

    metrics.append(
        {
            "predicate": predicate_name,
            "precision": precision,
            "recall": recall,
            "f1": f1_score,
            "mpr_var": mpr_variance,
            "mpr_span": mpr_span,
            "mfr_var": mfr_variance,
            "mfr_span": mfr_span,
        }
    )

with open(metrics_output_path, "w") as f:
    writer = csv.DictWriter(
        f,
        [
            "predicate",
            "precision",
            "recall",
            "f1",
            "mpr_var",
            "mpr_span",
            "mfr_var",
            "mfr_span",
        ],
    )
    writer.writeheader()
    writer.writerows(metrics)
