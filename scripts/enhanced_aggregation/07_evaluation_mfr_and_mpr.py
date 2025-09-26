import csv
import logging
from collections import defaultdict
from pathlib import Path
from random import Random

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.util import Interval
from commonroad.scenario.scenario import Scenario
from crmonitor.common.world import World
from crmonitor.evaluation.predicate_interface import PredicateEvaluationInterfaceConfig
from crmonitor.predicates import (
    ALL_GENERAL_PREDICATE_NAMES,
    ALL_INTERSTATE_PREDICATE_NAMES,
    CHANGED_TO_META_PREDICATE_NAMES,
)
from crmonitor.predicates.base import PredicateConfig
from crmonitor.mpr import MprGpPredicateEvaluatorConfig, mpr_gp_predicate_evaluator
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
selected_predicates = (
    ALL_GENERAL_PREDICATE_NAMES + ALL_INTERSTATE_PREDICATE_NAMES + CHANGED_TO_META_PREDICATE_NAMES
)
rand_seed = 12345


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

predicate_evaluator_config = PredicateConfig(
    scale_rob=True,
)
predicate_factory = PredicateFactory(predicate_evaluator_config)
predicates = [
    predicate_factory.get_predicate(predicate_name) for predicate_name in selected_predicates
]

mpr_gp_evaluator = mpr_gp_predicate_evaluator.MprGpPredicateEvaluator(
    predicates, config=MprGpPredicateEvaluatorConfig(model_path=models_path)
)

random = Random(rand_seed)

mpr_rob = defaultdict(list)  # GP predicted
mfr_rob = defaultdict(list)
for i in range(0, iterations):
    if i % 10 == 0:
        _LOGGER.info(f"Iteration {i}/{iterations}")
    scenario_path = random.choice(scenarios)

    scenario, _ = CommonRoadFileReader(scenario_path).open(lanelet_assignment=True)
    world = World.create_from_scenario(scenario)

    # Search for a time step, where at least two vehicles are present.
    end_time = get_scenario_final_time_step(scenario)
    # Keep track of time steps, which were not yet considered.
    time_steps = [i for i in range(0, end_time + 1)]
    time_step = 0
    vehicle_ids_at_time_step = []
    are_enough_vehicles_at_time_step = False
    while (not are_enough_vehicles_at_time_step) and len(time_steps) > 0:
        time_step = time_steps.pop(random.randint(0, len(time_steps) - 1))
        vehicle_ids_at_time_step = world.vehicle_ids_for_time_step(time_step)

        are_enough_vehicles_at_time_step = len(vehicle_ids_at_time_step) >= 2

    if not are_enough_vehicles_at_time_step:
        _LOGGER.warning(
            f"Cannot process scenario {scenario.scenario_id}: No time step with at least two active vehicles in both worlds was found!"
        )
        continue

    ego_vehicle_id, other_vehicle_id = random.sample(vehicle_ids_at_time_step, 2)
    vehicle_ids = (ego_vehicle_id, other_vehicle_id)

    mpr_gp_robustness_values = mpr_gp_evaluator.evaluate(world, time_step, vehicle_ids)

    for predicate_evaluator in predicates:
        mpr_robustness = mpr_gp_robustness_values[predicate_evaluator.predicate_name].robustness
        if (not -1 <= mpr_robustness <= 1) or np.isnan(mpr_robustness):
            _LOGGER.warning(f"MPR: {mpr_robustness:.3f}")
            continue

        mfr_robustness = predicate_evaluator.evaluate_robustness(world, time_step, vehicle_ids)
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
