import itertools
import logging
import multiprocessing as mp
import random
from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.scenario import Scenario
from crmonitor.common.world import World
from crmonitor.evaluation.evaluation import OfflineRuleEvaluator
from crmonitor.evaluation.predicate_interface import (
    PredicateEvaluationInterfaceConfig,
    PredicateEvaluationMode,
)
from crmonitor.monitor.rtamt_monitor_stl import OutputType
from crmonitor.mpr import (
    MprGpPredicateEvaluator,
    MprGpPredicateEvaluatorConfig,
)
from crmonitor.mpr.mpr_predicate_evaluator import (
    MprPredicateEvaluatorConfig,
    RobustnessNormalizationProvider,
)
from crmonitor.mpr.prediction.state_sampling import (
    EndStateOptions,
    FutureStateSamplerConfig,
    SwitchableEndStateOptions,
    VelocityMode,
)
from crmonitor.predicates.base import (
    PredicateConfig,
)

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)

input_scenarios = (
    Path(__file__).parents[3] / "scenarios-for-semantic-aware-stl" / "highD_downsample"
)
output_file = Path(__file__).parent.parent / "output" / "ablation_study_results_no_gps.csv"


num_vehicles_per_scenarios = 5
output_type = OutputType.OUTPUT_ROBUSTNESS
model_path = Path(__file__).parent.parent / "output" / "models"
normalization_file = Path(__file__).parent.parent / "output" / "normalization.csv"
snapshot_frequency = 10
mpr_only_on_violation = False
enable_gps = False  # Enable/Disable the MPR evaluation with GPs


normalization_provider = RobustnessNormalizationProvider.from_csv(normalization_file)


def process_scenario_with_rule(
    scenario: Scenario, ego_vehicle_id: int, rule: str, use_mpr: bool
) -> dict:
    world = World.create_from_scenario(scenario)
    # Create a rule evaluator
    # Provide the vehicle to evaluate traffic rules for as ego vehicle
    evaluation_mode = (
        (PredicateEvaluationMode.MPR_GP if enable_gps else PredicateEvaluationMode.MPR)
        if use_mpr
        else PredicateEvaluationMode.MFR
    )
    predicate_interface_config = PredicateEvaluationInterfaceConfig(
        mode=evaluation_mode,
        base=PredicateConfig(scale_rob=True),
        mpr=MprPredicateEvaluatorConfig(
            sampler_config=FutureStateSamplerConfig(
                SwitchableEndStateOptions(
                    modes={VelocityMode.HIGH_VELOCITY_MODE: EndStateOptions(number=100)}
                )
            )
        ),
        mpr_gp=MprGpPredicateEvaluatorConfig(model_path=model_path),
    )

    # Create a rule evaluator
    # Provide the vehicle to evaluate traffic rules for as ego vehicle
    ego_vehicle = world.vehicle_by_id(ego_vehicle_id)
    assert ego_vehicle is not None
    _LOGGER.info(
        f"Evaluating rule {rule} with {'MPR' if use_mpr else 'MFR'} for vehicle {ego_vehicle.id} in scenario {scenario.scenario_id} from time step {ego_vehicle.start_time} to {ego_vehicle.end_time}"
    )
    rule_evaluator = OfflineRuleEvaluator.create_for_rule(
        rule,
        world,
        ego_vehicle.id,
        output_type=output_type,
        predicate_interface_config=predicate_interface_config,
    )
    # Either step through time steps sequentially
    robustness = rule_evaluator.evaluate()

    return {
        "scenario_id": str(scenario.scenario_id),
        "rule": rule,
        "mpr": use_mpr,
        "vehicle_id": ego_vehicle.id,
        "start_time_step": ego_vehicle.start_time,
        "end_time_step": ego_vehicle.end_time,
        "violation": any(value < 0.0 for value in robustness),
        "robustness": ", ".join(map(str, robustness)),
    }


def process_scenario(scenario: Scenario, ego_vehicle_id: int, rule: str) -> list:
    mfr_result = process_scenario_with_rule(scenario, ego_vehicle_id, rule, use_mpr=False)
    results = [mfr_result]
    if not mpr_only_on_violation or mfr_result["violation"] is True:
        mpr_result = process_scenario_with_rule(scenario, ego_vehicle_id, rule, use_mpr=True)
        results.append(mpr_result)

    return results


rules = [
    "R_G1",
    "R_G2",
    "R_G3",
    "R_G4",
    "R_I1",
    "R_I2",
    "R_I3",
    "R_I4",
    "R_I5",
]

scenarios_paths = np.random.choice(list(input_scenarios.glob("*.xml")), 200, replace=False)
scenarios = list(
    map(
        lambda scenario_path: CommonRoadFileReader(scenario_path).open(lanelet_assignment=True)[0],
        scenarios_paths,
    )
)

ego_vehicles_per_scenario = {}
for scenario in scenarios:
    vehicle_ids = list(map(lambda v: v.obstacle_id, scenario.dynamic_obstacles))
    if len(vehicle_ids) < num_vehicles_per_scenarios:
        continue
    ego_vehicles_per_scenario[scenario.scenario_id] = random.sample(
        vehicle_ids, k=num_vehicles_per_scenarios
    )


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    results_since_last_snapshot = 0
    results = []
    tasks = {}
    with ProcessPoolExecutor(max_workers=70, mp_context=ctx) as executor:
        for scenario, rule in itertools.product(scenarios, rules):
            for ego_vehicle_id in ego_vehicles_per_scenario[scenario.scenario_id]:
                task = executor.submit(process_scenario, scenario, ego_vehicle_id, rule)
                tasks[task] = (scenario.scenario_id, ego_vehicle_id, rule)

        for finished_future in futures.as_completed(tasks.keys()):
            exception = finished_future.exception()
            if exception is not None:
                task_arguments = tasks[finished_future]
                _LOGGER.warning(f"Exception {exception} occurred while processing {task_arguments}")
                continue

            result = finished_future.result()
            results.extend(finished_future.result())
            results_since_last_snapshot += 1
            del tasks[finished_future]

            if results_since_last_snapshot >= snapshot_frequency:
                _LOGGER.info(f"Writing snapshot to {output_file}")
                results_df = pd.DataFrame(results)
                results_df.to_csv(output_file, index=False)
                results_since_last_snapshot = 0

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
