import logging
import multiprocessing.connection
import traceback
from pathlib import Path
from typing import List, Tuple

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.util import Interval
from commonroad.scenario.scenario import Scenario

from commonroad_mpr.common import World as MprWorld
from commonroad_mpr.learning import DataGenerator
from commonroad_mpr.learning.feature_variable import FeatureExtrator
from commonroad_mpr.utils.configuration_builder import ConfigurationBuilder as MprCfg
from commonroad_mpr.utils.configuration_builder import ScenarioType
from crmonitor.common.world import World
from crmonitor.predicates.predicate_factory import PredicateFactory

from crmonitor.predicate_grouping import all_general_predicates, all_interstate_predicates

# Use 'all_general_predicates' to generate learning data for all predicates that are used for general traffic rules.
# Alternatively, supply a list of specific predicates you want to evaluate.
predicate_names = all_general_predicates + all_interstate_predicates
scenarios_load_path = Path(__file__).parent.parent.parent.parent / "highD-scenarios"

output_path = Path(__file__).parent.parent / "output" / "learning_data" / "learning_data.csv"
# Optional: Limit the number of scenarios that are processed e.g. for faster prototyping
scenario_limit = 5000

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)


# Although the DataGenerator does not require config files, internal mpr methods (might) do.
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


def _get_scenario_final_time_step(scenario: Scenario) -> int:
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


class PredicateEvaluationWrapper:
    """
    Wrapper around the crmonitor predicate evaluators that mimicks the return of the default MPR evaluator, to ensure the learning data can be used for GP regression.
    """

    def __init__(self, predicate_names: List[str]) -> None:
        self._predicate_names = predicate_names
        predicate_factory = PredicateFactory()
        self._evaluators = {
            predicate_name: predicate_factory.get_predicate(predicate_name)
            for predicate_name in self._predicate_names
        }

    def evaluate(
        self, world: World, world_mpr: MprWorld, vehicle_ids: List[int], time_step: int
    ) -> dict:
        robustness = {}
        for predicate_name, evaluator in self._evaluators.items():
            robustness[predicate_name] = evaluator.evaluate_mpr(
                world, world_mpr, time_step, vehicle_ids
            )

        return robustness


class CustomDataGenerator(DataGenerator):
    """
    Data generator, which uses crmonitor predicates with model-predictive evaluation. This is needed, because the crmonitor predicates require two worlds: one for mpr and one for crmonitor.
    """

    def _process_vehicles_patched(
        self,
        vehicle_ids: Tuple[int, ...],
        time_step: int,
        world_mpr: MprWorld,
        world: World,
    ) -> Tuple[dict, dict, dict]:
        _LOGGER.info(
            f"Processing the vehicles {vehicle_ids} in scenario {world.scenario.scenario_id} at time step {time_step}"
        )

        # This is different from the base `_process_vehicles`, as we use our wrapper instead of the evaluator from mpr.
        mpr = PredicateEvaluationWrapper(self._predicate_names)
        predicates_dict = mpr.evaluate(
            world=world,
            world_mpr=world_mpr,
            vehicle_ids=vehicle_ids,
            time_step=time_step,
        )
        features_dict = FeatureExtrator.all_feature_variables(
            world_state=world_mpr, vehicle_ids=vehicle_ids, time_step=time_step
        )
        # Not all of these features will be used for training the GPs for each of the predicates.
        # The relevant features can still be selected later on.

        dict_entry_id = {
            "scenario_id": str(world.scenario.scenario_id),
            "time_step": time_step,
            "ego_id": vehicle_ids[0],
            "other_id": vehicle_ids[1],
        }
        return dict_entry_id, features_dict, predicates_dict

    def _process_scenario(self, scenario_path: Path, result_pipe: multiprocessing.connection.Connection) -> None:
        try:
            scenario, _ = CommonRoadFileReader(scenario_path).open(lanelet_assignment=True)
            world_mpr = MprWorld.create_from_scenario(scenario)  # everything still cartesian
            # Create an additional world for crmonitor predicates
            world = World.create_from_scenario(scenario)  # everything still cartesian

            end_time = _get_scenario_final_time_step(scenario) - self._state_sampling_ts - 1

            for time_step in np.linspace(1, end_time, self._time_steps_per_scenario, dtype=int):
                for vehicle_ids in self._vehicle_ids_iter(scenario, time_step):
                    data_entry = self._process_vehicles_patched(vehicle_ids, time_step, world_mpr, world)
                    result_pipe.send(data_entry)

                    data_entry = self._process_vehicles_patched(
                        tuple(reversed(vehicle_ids)), time_step, world_mpr, world
                    )
                    result_pipe.send(data_entry)

        except Exception as exp:
            _LOGGER.debug(traceback.format_exc())
            raise RuntimeError(f"Failed to process scenario {scenario_path.stem}: {exp}") from exp
        finally:
            # Signal to the main process, that the scenario was fully processed.
            result_pipe.send(None)


data_generator = CustomDataGenerator(
    predicate_names=all_general_predicates+all_interstate_predicates,
    scenarios_path=scenarios_load_path,
    dt=0.04,
    output_path=output_path,
    vehicle_pair_steps=20,
    state_sampling_time_horizon=1.5,
    time_steps_per_scenario=5,
    scenario_type=ScenarioType.INTERSTATE,
    snapshot_frequency=50,
)


_LOGGER.info(
    "Started processing scenarios from %s to generate learning data for predicates %s",
    scenarios_load_path,
    ", ".join(predicate_names),
)

_LOGGER.info(f"Number of CPUs: {multiprocessing.cpu_count()}")

data_generator.generate_data(workers=70, limit=scenario_limit)  # multiprocessing.cpu_count()
_LOGGER.info("Finished processing scenarios; writing output to %s", output_path)
data_generator.save_data(output_path)
