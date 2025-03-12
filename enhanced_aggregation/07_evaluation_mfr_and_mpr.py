from collections import defaultdict
import logging
from random import Random
from pathlib import Path
import csv

import numpy as np
from commonroad_mpr.utils.configuration_builder import ConfigurationBuilder as MprCfg
from commonroad_mpr.common.observation import World as WorldMPR
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad_mpr.utils.configuration_builder import ConfigurationBuilder as MprCfg
from crmonitor.common.world import World
from crmonitor.predicates.base import PredicateMprConfig, PredicateEvaluatorConfig
from crmonitor.predicates.predicate_factory import PredicateFactory

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)


metrics_output_path = (
    Path(__file__).parent.parent / "output" / "metrics" / "mfr_and_mpr_metrics.csv"
)
metrics_output_path.parent.mkdir(exist_ok=True, parents=True)

scenarios_load_path = Path(__file__).parent.parent.parent.parent / "highD-scenarios"
iterations = 1000
models_path = None
selected_predicates = ["in_front_of", "in_same_lane"]
rand_seed = 3478134569078

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
    },
    # Path root must point to a local revision of commonroad-model-predictive-robustness.
    # This configuration, assumes that the repo is in the same directory as stl-monitor repo.
    # If this is not the case for your setup, adjust the path here accordingly.
    path_root=str(Path(__file__).parent.parent.parent / "commonroad-model-predictive-robustness"),
    folder_config="config_files",
    default_profile="default",
)


scenarios = list(scenarios_load_path.glob("*.xml"))
predicate_evaluator_config = PredicateEvaluatorConfig(
    scale_rob=True,
    mpr=PredicateMprConfig(enabled=True, model_path=models_path),
)
predicate_factory = PredicateFactory(predicate_evaluator_config)
predicates = [
    predicate_factory.get_predicate(predicate_name) for predicate_name in selected_predicates
]
random = Random(rand_seed)

mpr_rob = defaultdict(list)
mpr_predicted_rob = defaultdict(list)
mfr_rob = defaultdict(list)
for i in range(0, iterations):
    scenario_path = scenarios.pop(random.randint(0, len(scenarios)))

    scenario, _ = CommonRoadFileReader(scenario_path).open(lanelet_assignment=True)
    world = World.create_from_scenario(scenario)
    mpr_world = WorldMPR.create_from_scenario(scenario)

    vehicles = list(world.vehicles)
    if len(vehicles) < 2:
        _LOGGER.warning(
            f"Cannot process scenario {scenario.scenario_id}: Only one vehicle in scenario!"
        )
        continue
    ego_vehicle, other_vehicle = random.sample(vehicles, 2)
    end_time = min(ego_vehicle.end_time, other_vehicle.end_time)
    start_time = max(ego_vehicle.start_time, other_vehicle.start_time)

    time_step = random.randint(start_time, end_time)

    for predicate_evaluator in predicates:
        predicted_robustness = predicate_evaluator.evaluate_mpr_ml(
            world, mpr_world, time_step, [ego_vehicle.id, other_vehicle.id]
        )
        robustness = predicate_evaluator.evaluate_mpr(
            world, mpr_world, time_step, [ego_vehicle.id, other_vehicle.id]
        )["robustness"]

        mpr_rob[predicate_evaluator.predicate_name].append(robustness)
        mpr_predicted_rob[predicate_evaluator.predicate_name].append(predicted_robustness)

    for predicate_evaluator in predicates:
        robustness = predicate_evaluator.evaluate_robustness(
            world, time_step, [ego_vehicle.id, other_vehicle.id]
        )

        mfr_rob[predicate_evaluator.predicate_name].append(robustness)


metrics = []
for predicate_name in selected_predicates:
    y_true = np.array([rob > 0 for rob in mpr_rob[predicate_name]])
    y_pred = np.array([rob > 0 for rob in mpr_predicted_rob[predicate_name]])

    TP = np.logical_and(y_true, y_pred).sum()
    FP = np.logical_and(~y_true, y_pred).sum()
    FN = np.logical_and(y_true, ~y_pred).sum()
    TN = np.logical_and(~y_true, ~y_pred).sum()

    eps = 1e-9
    precision = (TP + eps) / (TP + FP + eps)
    recall = (TP + eps) / (TP + FN + eps)
    f1_score = 2 * precision * recall / (precision + recall + eps)
    mpr_gp_variance = np.var(mpr_predicted_rob[predicate_name])
    mpr_gp_span = max(mpr_predicted_rob[predicate_name]) - min(mpr_predicted_rob[predicate_name])

    mpr_variance = np.var(mpr_rob[predicate_name])
    mpr_span = max(mpr_rob[predicate_name]) - min(mpr_rob[predicate_name])

    mfr_variance = np.var(mfr_rob[predicate_name])
    mfr_span = max(mfr_rob[predicate_name]) - min(mfr_rob[predicate_name])

    metrics.append(
        {
            "predicate": predicate_name,
            "precision": precision,
            "recall": recall,
            "f1": f1_score,
            "mpr_var": mpr_variance,
            "mpr_span": mpr_span,
            "mpr_gp_var": mpr_gp_variance,
            "mpr_gp_span": mpr_gp_span,
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
            "mpr_gp_var",
            "mpr_gp_span",
            "mfr_var",
            "mfr_span",
        ],
    )
    writer.writeheader()
    writer.writerows(metrics)
