import logging
import multiprocessing
import traceback
from pathlib import Path
from typing import List, Tuple

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad_mpr.common import World as MprWorld
from commonroad_mpr.learning import DataGenerator
from commonroad_mpr.learning.feature_variable import FeatureExtrator
from commonroad_mpr.utils.configuration_builder import ConfigurationBuilder as MprCfg

from crmonitor.common.world import World
from crmonitor.predicates.predicate_factory import PredicateFactory

# List of all predicates that are needed for
all_general_predicates = [
    "in_front_of",
    "in_same_lane",
    "cut_in",
    "keeps_safe_distance_prec",
    "brakes_abruptly",
    "rel_brakes_abruptly",
    "precedes",
    "keeps_lane_speed_limit",
    "keeps_type_speed_limit",
    "keeps_fov_speed_limit",
    "keeps_lane_speed_limit_star",
    "slow_leading_vehicle",
    "preserves_traffic_flow",
]
# Use 'all_general_predicates' to generate learning data for all predicates that are used for general traffic rules.
# Alternativly, supply specific predicates you want to evaluate.
predicate_names = all_general_predicates
scenarios_load_path = (
    Path(__file__).parent.parent.parent / "scenarios-for-semantic-aware-stl" / "highD"
)
output_path = (
    Path(__file__).parent.parent / "output" / "learning_data" / "learning_data.csv"
)
# Optional: Limit the number of scenarios that are processed e.g. for faster prototyping
scenario_limit = 4

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
    path_root=str(
        Path(__file__).parent.parent.parent / "commonroad-model-predictive-robustness"
    ),
    folder_config="config_files",
    default_profile="default",
)


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
        _LOGGER.debug(
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

        dict_entry_id = {
            "scenario_id": str(world.scenario.scenario_id),
            "time_step": time_step,
            "ego_id": vehicle_ids[0],
            "other_id": vehicle_ids[1],
        }
        return (dict_entry_id, features_dict, predicates_dict)

    def _process_scenario(self, scenario_path: Path) -> List[Tuple[dict, dict, dict]]:
        try:
            scenario, _ = CommonRoadFileReader(scenario_path).open(
                lanelet_assignment=True
            )
            world_mpr = MprWorld.create_from_scenario(scenario)
            # Create an additional world for crmonitor predicates
            world = World.create_from_scenario(scenario)
            data_entries = []

            for time_step in self._time_step_iteration:
                for vehicle_ids in self._vehicle_ids_iter(scenario, time_step):
                    data_entry = self._process_vehicles_patched(
                        vehicle_ids, time_step, world_mpr, world
                    )
                    data_entries.append(data_entry)

                    data_entry = self._process_vehicles_patched(
                        tuple(reversed(vehicle_ids)), time_step, world_mpr, world
                    )
                    data_entries.append(data_entry)

            return data_entries
        except Exception as e:
            _LOGGER.debug(traceback.format_exc())
            raise RuntimeError(
                f"Failed to process scenario {scenario_path.stem}: {e}"
            ) from e


data_generator = CustomDataGenerator(
    predicate_names,
    scenarios_load_path,
    scenario_duration=50,
    dt=0.2,
    time_steps_per_scenario=1,
)


_LOGGER.info(
    "Started processing scenarios from %s to generate learning data for predicates %s",
    scenarios_load_path,
    ", ".join(predicate_names),
)
data_generator.generate_data(workers=multiprocessing.cpu_count(), limit=scenario_limit)
_LOGGER.info("Finished processing scenarios; writing output to %s", output_path)
data_generator.save_data(output_path)
